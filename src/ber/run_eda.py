"""Step 1: understand the data and the ground-truth structure.  Usage:
    PYTHONPATH=src python -m ber.run_eda --data_dir dataset
Prints the numbers that decide the design (one-to-one? cross-country matches? singleton rate? orphans?)."""
import argparse
import pandas as pd
from .io_utils import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    args = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_colwidth", 70)

    s1, s2, s3, gt = load_split(args.data_dir, "train")
    t1, t2, t3, _ = load_split(args.data_dir, "test")
    print("== sizes ==")
    print(f"train: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} GT rows={len(gt):,}")
    print(f"test : S1={len(t1):,} S2={len(t2):,} S3={len(t3):,}")
    print("\n== country counts ==")
    for name, df in [("train S1", s1), ("train S2", s2), ("train S3", s3), ("test S1", t1), ("test S2", t2), ("test S3", t3)]:
        print(f"{name}: {df['country'].value_counts().to_dict()}")

    print("\n== ground-truth structure ==")
    n = pd.Series({k: len(v) for k, v in gt.items()})
    print(f"singleton rate (S1 with no match): {(n == 0).mean():.3f}")
    print("matches per S1 entity:", n.value_counts().sort_index().head(10).to_dict())
    flat = [t for v in gt.values() for t in v]
    fs = pd.Series(flat)
    print(f"target ids claimed by more than one S1 entity: {fs.duplicated().sum()}   (0 => one-to-one decoding is safe)")
    print(f"share of S2 records that match some S1: {fs.str.startswith('S2-').sum() / len(s2):.3f}   "
          f"S3: {fs.str.startswith('S3-').sum() / len(s3):.3f}   (rest are orphans / distractors)")
    ctry = pd.concat([s1, s2, s3]).set_index("entity_id")["country"]
    cross = sum(ctry[a] != ctry[b] for a, v in gt.items() for b in v)
    print(f"matches whose country label differs from the S1 record: {cross}   (0 => same-country blocking is safe)")
    both = sum(any(t.startswith('S2-') for t in v) and any(t.startswith('S3-') for t in v) for v in gt.values())
    print(f"S1 entities matching BOTH an S2 and an S3 record: {both:,}")
    print("\n== empty fields (train) ==")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"{name}: empty name={(df['business_name'] == '').mean():.3f} empty address={(df['business_address'] == '').mean():.3f}")
    print("\n== samples ==")
    print(s1.sample(5, random_state=0).to_string())
    print(s2.sample(5, random_state=0).to_string())
    print(s3.sample(5, random_state=0).to_string())
    one = next((k for k, v in gt.items() if len(v) >= 2), None)
    if one:
        print("\n== one matched group (S1 row, then its matches) ==")
        allr = pd.concat([s1, s2, s3]).set_index("entity_id")
        print(allr.loc[[one] + sorted(gt[one])].to_string())


if __name__ == "__main__":
    main()
