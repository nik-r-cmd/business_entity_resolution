"""Validation splits that mimic the test situation (entity-level, so no leakage between S1 and its matches)."""
import numpy as np


def make_split(s1, s2, s3, gt, val_frac=0.2, seed=42, holdout_country=None):
    """Split S1 entities into train/val. Matched S2/S3 records follow their S1 entity; unmatched (orphan) records are
    assigned randomly (or by country when holdout_country is set). With holdout_country the validation set contains
    ONLY that country and training contains none of it -> closest proxy for the unseen country (France) in the test set."""
    rng = np.random.RandomState(seed)
    ck = s1["country"].str.strip().str.lower().to_numpy()
    if holdout_country:
        is_val = ck == holdout_country.strip().lower()
    else:
        is_val = np.zeros(len(s1), bool)
        for c in np.unique(ck):
            idx = np.flatnonzero(ck == c)
            is_val[rng.choice(idx, int(round(len(idx) * val_frac)), replace=False)] = True
    val_ids = set(s1["entity_id"][is_val])
    owner = {t: sid for sid, ts in gt.items() for t in ts}

    def side_val(df):
        out = np.zeros(len(df), bool)
        for i, (eid, c) in enumerate(zip(df["entity_id"], df["country"])):
            if eid in owner:
                out[i] = owner[eid] in val_ids
            elif holdout_country:
                out[i] = c.strip().lower() == holdout_country.strip().lower()
            else:
                out[i] = rng.rand() < val_frac
        return out

    v2, v3 = side_val(s2), side_val(s3)   # computed once (orphan assignment is random)
    tr = {"s1": s1[~is_val].reset_index(drop=True), "s2": s2[~v2].reset_index(drop=True),
          "s3": s3[~v3].reset_index(drop=True), "gt": {k: v for k, v in gt.items() if k not in val_ids}}
    va = {"s1": s1[is_val].reset_index(drop=True), "s2": s2[v2].reset_index(drop=True),
          "s3": s3[v3].reset_index(drop=True), "gt": {k: v for k, v in gt.items() if k in val_ids}}
    return tr, va


def subsample(s1, s2, s3, gt, max_s1, seed=0):
    """Quick dry-run helper: keep max_s1 random S1 entities + their matches + the same fraction of orphan records."""
    if max_s1 is None or max_s1 >= len(s1):
        return s1, s2, s3, gt
    rng = np.random.RandomState(seed)
    frac = max_s1 / len(s1)
    keep = set(rng.choice(s1["entity_id"].to_numpy(), max_s1, replace=False))
    owner = {t: sid for sid, ts in gt.items() for t in ts}

    def sel(df):
        return df[[(owner[e] in keep) if e in owner else (rng.rand() < frac) for e in df["entity_id"]]].reset_index(drop=True)

    s1n = s1[s1["entity_id"].isin(keep)].reset_index(drop=True)
    return s1n, sel(s2), sel(s3), {k: v for k, v in gt.items() if k in keep}
