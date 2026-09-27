"""Skip training (reuse a saved model) and run ONLY the test-side blocking/features/inference/output step.
Use this to recover after a crash during the test step, without redoing the expensive training step.
    python -m ber.run_predict_test_only --config artifacts/config.json --model artifacts/final_model.txt --out_dir output
"""
import argparse
import json
import os
import subprocess
import sys
import pandas as pd
import lightgbm as lgb
from .io_utils import load_split, write_id_list_tsv
from .pipeline import build_pair_table
from .decode import predict_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--art_dir", default="artifacts")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="path to saved LightGBM model (final_model.txt)")
    ap.add_argument("--max_cand_override", type=int, default=3, help="lower than before for memory safety")
    args = ap.parse_args()
    cfg = json.load(open(args.config or os.path.join(args.art_dir, "config.json")))
    block_kw, one_to_one = dict(cfg["block"]), cfg["one_to_one"]
    block_kw["max_cand"] = args.max_cand_override
    feats = cfg["features"]

    booster = lgb.Booster(model_file=args.model or os.path.join(args.art_dir, "final_model.txt"))
    print(f"[test-only] loaded saved model from {args.model}")

    t1, t2, t3, _ = load_split(args.data_dir, "test")
    rec, df_te, codes = build_pair_table(t1, t2, t3, None, tag="test", embed_model=None, rescue_kw=None, **block_kw)
    df_te["p"] = booster.predict(df_te[feats]) if len(df_te) else []
    match = predict_map(df_te, rec, cfg["thr2"], cfg["thr3"], one_to_one)

    ids = rec["entity_id"].to_numpy()
    cand = {}
    for a, b in zip(df_te["a"].to_numpy(), df_te["b"].to_numpy()):
        cand.setdefault(ids[a], []).append(ids[b])
    s1_ids = t1["entity_id"].tolist()
    write_id_list_tsv(os.path.join(args.out_dir, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")
    write_id_list_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")
    print(f"[test-only] wrote {args.out_dir}/matching_results.tsv and candidate_pairs.tsv ({len(s1_ids):,} rows each)")

    v = os.path.join("utils", "validate_submission.py")
    if os.path.exists(v):
        print("\n[validator]")
        subprocess.run([sys.executable, v, "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data_dir, "test")])


if __name__ == "__main__":
    main()
