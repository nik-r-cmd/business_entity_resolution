"""Reading / writing the challenge TSV files.

Gotchas handled here:
  * every file is TAB separated (addresses and ID lists contain commas)
  * keep_default_na=False so empty strings stay "" (no NaN surprises)
  * QUOTE_NONE so stray double-quotes inside business names cannot swallow rows
"""
import csv
import os
import pandas as pd


def read_tsv(path):
    """Read a challenge TSV with all columns as plain strings."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def parse_gt(gt_df):
    """Ground-truth dataframe -> dict {source1_id: set(matched ids)} (empty set = singleton)."""
    out = {}
    for sid, ids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        out[sid] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def load_split(data_dir, split):
    """Load one split ('train' or 'test'). Returns (s1, s2, s3, gt_or_None)."""
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    gt_path = os.path.join(d, f"{split}_ground_truth.tsv")
    gt = parse_gt(read_tsv(gt_path)) if os.path.exists(gt_path) else None
    return s1, s2, s3, gt


def write_id_list_tsv(path, s1_ids, mapping, second_col):
    """Write 'source1_entity_id <TAB> id,id,id' with exactly one row per S1 id (in the given order).
    Empty list -> empty second column. IDs sorted and de-duplicated."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{second_col}\n")
        for sid in s1_ids:
            ids = sorted(set(mapping.get(sid, [])))
            f.write(f"{sid}\t{','.join(ids)}\n")
