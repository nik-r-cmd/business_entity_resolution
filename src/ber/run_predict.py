"""Step 3: final training + inference on the test set + writing the two output files."""
import argparse
import gc
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
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_cand_override", type=int, default=None)
    ap.add_argument("--train_max_s1", type=int, default=None)
    ap.add_argument("--use_rescue", action="store_true", help="load the embedding model and force rescue on ALL entities, matching a rescue-trained config's training conditions")
    args = ap.parse_args()
    cfg = json.load(open(args.config or os.path.join(args.art_dir, "config.json")))
    block_kw, one_to_one = dict(cfg["block"]), cfg["one_to_one"]
    if args.max_cand_override is not None:
        block_kw["max_cand"] = args.max_cand_override

    embed_model = None
    rescue_kw = None
    if args.use_rescue:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[predict] loading embedding model on {device} for rescue (forced on ALL entities, matching training)")
        embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)
        rescue_kw = dict(weak_threshold=999, top_k=10, sim_min=0.5)

    s1, s2, s3, gt = load_split(args.data_dir, "train")
    if args.train_max_s1 is not None:
        from .split import subsample
        s1, s2, s3, gt = subsample(s1, s2, s3, gt, args.train_max_s1)
        print(f"[full-train] subsampled to {args.train_max_s1:,} S1 entities for memory safety")
    _, df_tr, _ = build_pair_table(s1, s2, s3, gt, tag="full-train", embed_model=embed_model,
                                    rescue_kw=rescue_kw, use_quality_filter=False, **block_kw)
    feats = cfg["features"]
    params = {**cfg["lgb"], "n_estimators": max(50, int(cfg["best_iter"] * 1.15))}
    clf = lgb.LGBMClassifier(**params).fit(df_tr[feats], df_tr["label"])
    os.makedirs(args.art_dir, exist_ok=True)
    clf.booster_.save_model(os.path.join(args.art_dir, "final_model.txt"))

    del df_tr, s1, s2, s3, gt
    gc.collect()
    print("[full-train] released training data from memory before processing test set")

    t1, t2, t3, _ = load_split(args.data_dir, "test")
    rec, df_te, codes = build_pair_table(t1, t2, t3, None, tag="test", embed_model=None,
                                          rescue_kw=None, **block_kw)
    df_te["p"] = clf.predict_proba(df_te[feats])[:, 1] if len(df_te) else []
    match = predict_map(df_te, rec, cfg["thr2"], cfg["thr3"], one_to_one)

    ids = rec["entity_id"].to_numpy()
    cand = {}
    for a, b in zip(df_te["a"].to_numpy(), df_te["b"].to_numpy()):
        cand.setdefault(ids[a], []).append(ids[b])
    s1_ids = t1["entity_id"].tolist()
    write_id_list_tsv(os.path.join(args.out_dir, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")
    write_id_list_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")

    ck = t1["country"].str.strip().str.lower()
    has = pd.Series([len(match.get(i, [])) > 0 for i in s1_ids], index=t1.index)
    nm = pd.Series([len(match.get(i, [])) for i in s1_ids], index=t1.index)
    nc = pd.Series([len(cand.get(i, [])) for i in s1_ids], index=t1.index)
    rep = pd.DataFrame({"country": ck, "predicted_match_rate": has, "avg_matches": nm, "avg_candidates": nc}).groupby("country").mean()
    print("\n[test] per-country sanity:\n", rep.round(3))
    print(f"[test] wrote {args.out_dir}/matching_results.tsv and candidate_pairs.tsv ({len(s1_ids):,} rows each)")

    v = os.path.join("utils", "validate_submission.py")
    if os.path.exists(v):
        print("\n[validator]")
        subprocess.run([sys.executable, v, "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data_dir, "test")])
    else:
        print("[validator] utils/validate_submission.py not found")


if __name__ == "__main__":
    main()
