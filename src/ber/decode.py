"""Turn pair probabilities into final match lists, and tune the decision thresholds on validation data.

Decoding rules (precision-oriented because the metric is F0.5):
  1. one_to_one: every S2/S3 record is assigned to AT MOST ONE S1 entity (its highest-probability candidate).
     A Source-1 entity can still collect many S2/S3 records. Only valid if the training ground truth never assigns
     one S2/S3 id to two S1 entities -> src/ber/run_eda.py prints this; set --no_one_to_one otherwise.
  2. keep the pair only if p >= threshold (separate thresholds for S2 and S3 targets).
  3. S1 entities left with nothing are predicted as singletons (empty list).
"""
import numpy as np


def best_per_target(df):
    """Keep only the highest-probability S1 candidate for every target record."""
    return df.sort_values("p", ascending=False).drop_duplicates("b")


def per_entity_scores(best, thr2, thr3, pos, truth_cnt):
    """Vectorised F0.5 per S1 entity for a threshold pair. `best` needs columns a, b, src, p, label.
    pos: array mapping rec row -> S1 position; truth_cnt: number of true matches per S1 position."""
    n1 = len(truth_cnt)
    thr = np.where(best["src"].to_numpy() == 0, thr2, thr3)
    keep = best["p"].to_numpy() >= thr
    ap = pos[best["a"].to_numpy()]
    lab = best["label"].to_numpy() == 1
    pred = np.bincount(ap[keep], minlength=n1).astype(float)
    tp = np.bincount(ap[keep & lab], minlength=n1).astype(float)
    P = np.divide(tp, pred, out=np.zeros(n1), where=pred > 0)
    R = np.divide(tp, truth_cnt, out=np.zeros(n1), where=truth_cnt > 0)
    den = 0.25 * P + R
    f = np.divide(1.25 * P * R, den, out=np.zeros(n1), where=den > 0)
    f[(pred == 0) & (truth_cnt == 0)] = 1.0
    return f


def tune_thresholds(df, rec, gt, one_to_one=True, grid=None):
    """Grid-search (thr_S2, thr_S3) maximising the macro F0.5 on validation pairs. Returns (thr2, thr3, best_score)."""
    grid = np.round(np.arange(0.30, 0.981, 0.02), 2) if grid is None else grid
    s1_rows = np.flatnonzero(rec["source"].to_numpy() == "S1")
    pos = np.full(len(rec), -1, np.int64)
    pos[s1_rows] = np.arange(len(s1_rows))
    ids = rec["entity_id"].to_numpy()[s1_rows]
    truth_cnt = np.array([len(gt.get(i, ())) for i in ids], float)
    best = best_per_target(df) if one_to_one else df
    top = (-1.0, 0.5, 0.5)
    for t2 in grid:
        for t3 in grid:
            s = per_entity_scores(best, t2, t3, pos, truth_cnt).mean()
            if s >= top[0]:          # ties -> keep the HIGHER threshold (safer for a precision-weighted metric)
                top = (s, t2, t3)
    return float(top[1]), float(top[2]), float(top[0])


def predict_map(df, rec, thr2, thr3, one_to_one=True):
    """Final {s1_entity_id: [matched ids]} from scored pairs (df needs a, b, src, p)."""
    best = best_per_target(df) if one_to_one else df
    thr = np.where(best["src"].to_numpy() == 0, thr2, thr3)
    kept = best[best["p"].to_numpy() >= thr]
    ids = rec["entity_id"].to_numpy()
    out = {}
    for a, b in zip(kept["a"].to_numpy(), kept["b"].to_numpy()):
        out.setdefault(ids[a], []).append(ids[b])
    return out
