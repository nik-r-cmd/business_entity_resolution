"""Shared glue: raw source tables -> (records, candidate pairs, feature table [+ labels])."""
import numpy as np
from .records import build_records
from .blocking import generate_candidates, truth_codes, blocking_report
from .features import build_features, Views
from .embed_rescue import rescue_candidates

BLOCK_DEFAULTS = dict(same_country=True, max_bucket=1000)


def build_pair_table(s1, s2, s3, gt=None, tag="", embed_model=None, rescue_kw=None, **block_kw):
    """Run normalisation -> inverted-index blocking [-> optional embedding rescue for weak-candidate entities]
    -> pair features. embed_model: a loaded SentenceTransformer, or None to skip (default). rescue_kw: dict of
    extra kwargs for rescue_candidates. If gt is given, adds a 0/1 `label` column and prints the blocking
    recall ceiling. Returns (rec, df, cand_codes)."""
    kw = {**BLOCK_DEFAULTS, **block_kw}
    print(f"[{tag}] S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    rec, id2row = build_records(s1, s2, s3)
    codes = generate_candidates(rec, **kw)
    if embed_model is not None:
        new_codes = rescue_candidates(rec, codes, embed_model, **(rescue_kw or {}))
        if len(new_codes):
            codes = np.union1d(codes, new_codes)
    df = build_features(rec, Views(rec), codes)
    if gt is not None:
        tc = truth_codes(gt, id2row, len(rec))
        df["label"] = np.isin(codes, tc).astype(np.int8)
        blocking_report(rec, codes, tc, tag=" " + tag)
    return rec, df, codes
