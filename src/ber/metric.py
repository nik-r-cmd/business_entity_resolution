"""Exact challenge metric: F0.5 per Source-1 entity, macro-averaged (singletons included)."""
import numpy as np


def f05(pred, truth):
    """F0.5 for one entity. pred/truth are sets of ids.
    empty/empty -> 1.0 (correct singleton); empty vs non-empty in either direction -> 0.0."""
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, truth_map, s1_ids):
    """Mean F0.5 over ALL s1_ids. Returns (score, per-entity numpy array)."""
    scores = np.array([f05(set(pred_map.get(s, ())), truth_map.get(s, set())) for s in s1_ids])
    return float(scores.mean()), scores


if __name__ == "__main__":
    # Example from the problem statement: predicted [S2-00047,S2-00193,S3-00812], truth [S2-00047,S3-00812] -> 0.714
    v = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(v - 0.7142857) < 1e-4, v
    print("metric OK:", round(v, 4))
