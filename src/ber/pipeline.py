"""Shared glue: raw source tables -> (records, candidate pairs, feature table [+ labels])."""
import resource
import numpy as np
from .records import build_records
from .blocking import generate_candidates, get_weak_entities, truth_codes, blocking_report
from .features import build_features, Views
from .embed_rescue import rescue_candidates

BLOCK_DEFAULTS = dict(same_country=True)


def _mem(label):
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"    [mem] {gb:.2f} GB peak so far ({label})")


def build_pair_table(s1, s2, s3, gt=None, tag="", embed_model=None, rescue_kw=None, use_quality_filter=True,
                      **block_kw):
    """Run normalisation -> inverted-index blocking [-> optional embedding rescue for weak-candidate entities]
    -> pair features. embed_model: a loaded SentenceTransformer, or None to skip (default). rescue_kw: dict
    of extra kwargs (top_k, sim_min, quality_threshold if use_quality_filter=True, else weak_threshold).
    use_quality_filter: select the rescue set via candidate QUALITY (usually 5-15% of entities) instead of
    raw candidate count (which selects ~everyone). Prefer True. Returns (rec, df, cand_codes)."""
    kw = {**BLOCK_DEFAULTS, **block_kw}
    print(f"[{tag}] S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    rec, id2row = build_records(s1, s2, s3)
    _mem(f"{tag}: records built")
    codes = generate_candidates(rec, **kw)
    _mem(f"{tag}: blocking done")
    if embed_model is not None:
        rk = dict(rescue_kw or {})
        if use_quality_filter:
            qt = rk.pop("quality_threshold", 3.0)
            weak = get_weak_entities(rec, quality_threshold=qt)
            new_codes = rescue_candidates(rec, codes, embed_model, weak_override=weak, **rk)
        else:
            new_codes = rescue_candidates(rec, codes, embed_model, **rk)
        if len(new_codes):
            codes = np.union1d(codes, new_codes)
        _mem(f"{tag}: rescue done")
    df = build_features(rec, Views(rec), codes)
    _mem(f"{tag}: features done")
    if gt is not None:
        tc = truth_codes(gt, id2row, len(rec))
        df["label"] = np.isin(codes, tc).astype(np.int8)
        blocking_report(rec, codes, tc, tag=" " + tag)
    return rec, df, codes
