"""Step 3: final training on ALL training data + inference on the test set + writing the two output files.
    PYTHONPATH=src python -m ber.run_predict --data_dir dataset --out_dir output
Requires artifacts/config.json from run_validate.py (best iteration, thresholds, blocking settings)."""
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .io_utils import load_split, write_id_list_tsv
from .pipeline import build_pair_table
from .features import feature_cols
from .decode import predict_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--art_dir", default="artifacts")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--config", default=None, help="path to a config json (default: <art_dir>/config.json)")
    args = ap.parse_args()
    cfg = json.load(open(args.config or os.path.join(args.art_dir, "config.json")))
    block_kw, one_to_one = cfg["block"], cfg["one_to_one"]

    # ---- final model: train on every labelled training pair ----
    s1, s2, s3, gt = load_split(args.data_dir, "train")
    _, df_tr, _ = build_pair_table(s1, s2, s3, gt, tag="full-train", **block_kw)
    feats = cfg["features"]
    params = {**cfg["lgb"], "n_estimators": max(50, int(cfg["best_iter"] * 1.15))}   # slightly more data -> slightly more trees
    clf = lgb.LGBMClassifier(**params).fit(df_tr[feats], df_tr["label"])
    os.makedirs(args.art_dir, exist_ok=True)
    clf.booster_.save_model(os.path.join(args.art_dir, "final_model.txt"))

    # ---- inference on test ----
    t1, t2, t3, _ = load_split(args.data_dir, "test")
    rec, df_te, codes = build_pair_table(t1, t2, t3, None, tag="test", **block_kw)
    df_te["p"] = clf.predict_proba(df_te[feats])[:, 1] if len(df_te) else []
    match = predict_map(df_te, rec, cfg["thr2"], cfg["thr3"], one_to_one)

    ids = rec["entity_id"].to_numpy()
    cand = {}
    for a, b in zip(df_te["a"].to_numpy(), df_te["b"].to_numpy()):
        cand.setdefault(ids[a], []).append(ids[b])
    s1_ids = t1["entity_id"].tolist()      # exactly one row per test S1 entity, in file order
    write_id_list_tsv(os.path.join(args.out_dir, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")
    write_id_list_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")

    # ---- sanity report (watch France: its rates should look plausible next to the other countries) ----
    ck = t1["country"].str.strip().str.lower()
    has = pd.Series([len(match.get(i, [])) > 0 for i in s1_ids], index=t1.index)
    nm = pd.Series([len(match.get(i, [])) for i in s1_ids], index=t1.index)
    nc = pd.Series([len(cand.get(i, [])) for i in s1_ids], index=t1.index)
    rep = pd.DataFrame({"country": ck, "predicted_match_rate": has, "avg_matches": nm, "avg_candidates": nc}).groupby("country").mean()
    print("\n[test] per-country sanity (train singleton rate is printed by run_eda; compare 1-match_rate to it):\n", rep.round(3))
    print(f"[test] wrote {args.out_dir}/matching_results.tsv and candidate_pairs.tsv ({len(s1_ids):,} rows each)")
    assert all(m in set(cand.get(s, [])) for s, ms in list(match.items())[:2000] for m in ms), "match not in candidates"

    v = os.path.join("utils", "validate_submission.py")
    if os.path.exists(v):
        print("\n[validator]")
        subprocess.run([sys.executable, v, "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data_dir, "test")])
    else:
        print("[validator] utils/validate_submission.py not found - copy it from student_resource/utils and run it before uploading")


if __name__ == "__main__":
    main()
