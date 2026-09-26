"""Step 2: honest validation.  Usage (from repo root):
    PYTHONPATH=src python -m ber.run_validate --data_dir dataset                      # random entity split
    PYTHONPATH=src python -m ber.run_validate --data_dir dataset --holdout_country India   # leave-one-country-out
    PYTHONPATH=src python -m ber.run_validate --data_dir dataset --max_s1 3000        # quick dry run

Builds candidates + features separately for the train part and the validation part, trains LightGBM, tunes the two
decision thresholds on the validation part with the exact challenge metric, and writes artifacts/config.json,
which run_predict.py reads."""
import argparse
import json
import os
import time
import warnings
import numpy as np
import lightgbm as lgb
from .io_utils import load_split
from .split import make_split, subsample
from .pipeline import build_pair_table
from .features import feature_cols
from .decode import tune_thresholds, predict_map
from .metric import macro_f05

warnings.filterwarnings("ignore", message=".*eval_set.*")   # newer LightGBM versions warn about eval_set naming

LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--art_dir", default="artifacts")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--holdout_country", default=None)
    ap.add_argument("--max_s1", type=int, default=None, help="dry-run: subsample this many S1 entities")
    ap.add_argument("--max_bucket", type=int, default=1000, help="drop blocking keys shared by more than this many records; raise if blocking recall is too low")
    ap.add_argument("--no_same_country", action="store_true", help="block across countries (use if EDA shows cross-country matches)")
    ap.add_argument("--no_one_to_one", action="store_true", help="allow a target record to match several S1 entities")
    ap.add_argument("--tag", default="run", help="name used for the saved config file")
    args = ap.parse_args()
    os.makedirs(args.art_dir, exist_ok=True)
    t0 = time.time()
    one_to_one = not args.no_one_to_one
    block_kw = dict(same_country=not args.no_same_country, max_bucket=args.max_bucket)

    s1, s2, s3, gt = load_split(args.data_dir, "train")
    s1, s2, s3, gt = subsample(s1, s2, s3, gt, args.max_s1)
    tr, va = make_split(s1, s2, s3, gt, args.val_frac, holdout_country=args.holdout_country)

    rec_tr, df_tr, _ = build_pair_table(tr["s1"], tr["s2"], tr["s3"], tr["gt"], tag="train-part", **block_kw)
    rec_va, df_va, _ = build_pair_table(va["s1"], va["s2"], va["s3"], va["gt"], tag="val-part", **block_kw)

    feats = feature_cols(df_tr)
    print(f"\n[model] training LightGBM on {len(df_tr):,} pairs ({df_tr['label'].mean():.3%} positive), {len(feats)} features")
    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(df_tr[feats], df_tr["label"], eval_set=[(df_va[feats], df_va["label"])], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    best_iter = int(clf.best_iteration_ or LGB_PARAMS["n_estimators"])
    df_va["p"] = clf.predict_proba(df_va[feats])[:, 1]
    imp = sorted(zip(clf.feature_importances_, feats), reverse=True)[:12]
    print("[model] best_iter:", best_iter, "| top features:", [f for _, f in imp])

    thr2, thr3, fast = tune_thresholds(df_va, rec_va, va["gt"], one_to_one)
    pred = predict_map(df_va, rec_va, thr2, thr3, one_to_one)
    s1_ids = va["s1"]["entity_id"].tolist()
    score, per = macro_f05(pred, va["gt"], s1_ids)
    empty = macro_f05({}, va["gt"], s1_ids)[0]
    print(f"\n=== VALIDATION macro F0.5 = {score:.4f}  (thr S2={thr2}, S3={thr3}; fast-eval check {fast:.4f})")
    print(f"    baseline 'predict nothing' = {empty:.4f}  (= singleton rate)")
    ck = va["s1"]["country"].str.strip().str.lower().to_numpy()
    for c in np.unique(ck):
        print(f"    country={c}: F0.5 = {per[ck == c].mean():.4f}  (n={int((ck == c).sum())})")
    assert abs(score - fast) < 1e-6, "fast and reference metrics disagree -> bug"

    cfg = dict(best_iter=best_iter, thr2=thr2, thr3=thr3, val_f05=score, features=feats, one_to_one=one_to_one,
               block=block_kw, holdout_country=args.holdout_country, max_s1=args.max_s1, lgb=LGB_PARAMS)
    with open(os.path.join(args.art_dir, f"config_{args.tag}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.art_dir, "config.json"), "w") as f:      # the file run_predict.py reads
        json.dump(cfg, f, indent=2)
    out = df_va.assign(a_id=rec_va["entity_id"].to_numpy()[df_va["a"]], b_id=rec_va["entity_id"].to_numpy()[df_va["b"]])
    out[["a_id", "b_id", "src", "label", "p"]].to_csv(os.path.join(args.art_dir, "val_scored_pairs.tsv.gz"), sep="\t", index=False)
    print(f"[done] {time.time() - t0:.0f}s | saved artifacts/config.json and val_scored_pairs.tsv.gz (for error analysis)")


if __name__ == "__main__":
    main()
