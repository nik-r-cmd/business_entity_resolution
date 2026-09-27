"""Candidate generation (blocking) — SCALABLE version for multi-million-row sources.

Why not TF-IDF cosine kNN over the whole corpus: with ~2M S1 records and ~5M S2/S3 records per side,
a pairwise similarity search is O(N*M) and does not fit in memory or time on CPU.

Why not plain "union of anything sharing one key": character 4-grams (ng4) generate dozens of keys per record,
so a naive union explodes to 1000+ candidates per S1 entity — infeasible once the feature step scores every pair.

Approach used here: WEIGHTED KEY VOTING + TOP-K.
  1. Build several blocking keys per record: distinctive NAME tokens/prefix/4-grams, distinctive ADDRESS tokens
     (added after error analysis showed many India records are transliterated into a different script — the name
     shares zero characters across sources, but the address stays in Latin script and matches almost verbatim),
     exact postal code, house/plot numbers.
  2. For each S1 record, accumulate a weighted vote for every S2/S3 record sharing at least one key (weight depends
     on key type — an exact postal-code match counts far more than one shared 4-gram).
  3. Keep only the top `max_cand` records by vote score. This bounds candidates-per-entity to a FIXED number
     regardless of how many keys a name happens to generate, which is what keeps this tractable at 5M+ rows.
Recall depends on the keys, the weights, and max_cand — if blocking recall is too low, raise max_cand or add keys.
"""
import time
from collections import defaultdict
import numpy as np

STOP_NAME = {"the", "and", "of", "co", "inc", "ltd", "llc", "corp", "company", "limited", "private", "pvt",
             "group", "services", "service", "international", "national", "enterprises", "trading", "traders"}
STOP_ADDR = {"no", "floor", "flr", "unit", "near", "opp", "opposite", "plot", "door", "house", "shop", "office",
             "ground", "road", "rd", "street", "st", "colony", "layout", "sector", "nagar", "east", "west",
             "north", "south", "new", "old", "main", "cross"}

KEY_WEIGHT = {"tok": 3.0, "pre4": 2.0, "ng4": 0.4, "pin": 8.0, "apre6": 1.0, "atok": 2.5, "num": 1.5}


def _name_tokens(name_core):
    return [t for t in name_core.split() if t not in STOP_NAME and len(t) >= 3]


def _addr_tokens(addr_core):
    return [t for t in addr_core.split() if t not in STOP_ADDR and len(t) >= 4 and not t.isdigit()]


def _keys_for_row(name_core, addr_core, postal, nums):
    """Yield (key, weight) for one record. Country is applied separately (keys are grouped per-country)."""
    for t in _name_tokens(name_core):
        yield ("tok", t), KEY_WEIGHT["tok"]
    compact = name_core.replace(" ", "")
    if len(compact) >= 4:
        yield ("pre4", compact[:4]), KEY_WEIGHT["pre4"]
        for i in range(0, len(compact) - 3):
            yield ("ng4", compact[i:i + 4]), KEY_WEIGHT["ng4"]
    for p in postal:
        yield ("pin", p), KEY_WEIGHT["pin"]
    astr = addr_core.replace(" ", "")
    if len(astr) >= 6:
        yield ("apre6", astr[:6]), KEY_WEIGHT["apre6"]
    for t in _addr_tokens(addr_core):
        yield ("atok", t), KEY_WEIGHT["atok"]
    for n in nums:
        yield ("num", n), KEY_WEIGHT["num"]


def build_index(rec, mask, max_bucket):
    """Inverted index: key -> list of row indices, restricted to rows where mask is True. Oversized buckets dropped."""
    idx = defaultdict(list)
    names = rec["name_core"].to_numpy()
    addrs = rec["addr_core"].to_numpy()
    posts = rec["postal"].to_numpy()
    nums = rec["nums"].to_numpy()
    for i in np.flatnonzero(mask):
        seen = set()
        for (k, _w) in _keys_for_row(names[i], addrs[i], posts[i], nums[i]):
            if k not in seen:
                seen.add(k)
                idx[k].append(i)
    dropped = sum(1 for v in idx.values() if len(v) > max_bucket)
    if dropped:
        print(f"  [blocking] dropped {dropped:,} oversized keys (> {max_bucket} records each)")
    return {k: v for k, v in idx.items() if len(v) <= max_bucket}


def generate_candidates(rec, same_country=True, max_bucket=1000, max_cand=50):
    """Return int64 array of encoded pairs (a_row * N + b_row), a = S1 row, b = S2/S3 row.
    max_bucket: drop keys this common. max_cand: keep only the top-scoring candidates per S1 entity — the main
    lever that keeps this tractable at multi-million-row scale.
    Also records, per S1 row, its single best vote score (across both target sources) into module-level
    `_last_best_score` — used by get_weak_entities() to find entities blocking is genuinely struggling on
    (raw candidate COUNT is not a useful signal here, since every entity gets filled up to max_cand regardless)."""
    N = len(rec)
    src = rec["source"].to_numpy()
    ck = rec["country_key"].to_numpy() if same_country else np.zeros(N, dtype=object)
    names = rec["name_core"].to_numpy()
    addrs = rec["addr_core"].to_numpy()
    posts = rec["postal"].to_numpy()
    nums_col = rec["nums"].to_numpy()
    codes = []
    best_score = {}
    t0 = time.time()
    for g in np.unique(ck):
        gmask = ck == g
        a_idx = np.flatnonzero(gmask & (src == "S1"))
        if len(a_idx) == 0:
            continue
        for tgt in ("S2", "S3"):
            b_mask = gmask & (src == tgt)
            if not b_mask.any():
                continue
            index = build_index(rec, b_mask, max_bucket)
            for i in a_idx:
                votes = defaultdict(float)
                for (k, w) in _keys_for_row(names[i], addrs[i], posts[i], nums_col[i]):
                    for j in index.get(k, ()):
                        votes[j] += w
                if not votes:
                    continue
                if len(votes) > max_cand:
                    top = sorted(votes.items(), key=lambda kv: -kv[1])[:max_cand]
                else:
                    top = votes.items()
                codes.extend(i * N + j for j, _ in top)
                best_score[i] = max(best_score.get(i, 0.0), max(votes.values()))
    codes = np.array(sorted(set(codes)), dtype=np.int64)
    print(f"  [blocking] {len(codes):,} candidate pairs in {time.time() - t0:.1f}s "
          f"(candidates per S1 = {len(codes) / max(1, (src == 'S1').sum()):.1f})")
    global _last_best_score
    _last_best_score = best_score
    return codes


_last_best_score = {}


def get_weak_entities(rec, quality_threshold=3.0):
    """S1 rows whose best candidate only matched via weak keys (e.g. only a few ng4 hits, never a token/address-
    token/postal match) — a real signal for likely-transliterated or otherwise hard names. Unlike raw candidate
    COUNT (always ~max_cand regardless of quality), this reflects how CONFIDENT blocking's best guess was.
    Must be called right after generate_candidates() on the same `rec` — reads its cached best-score map."""
    src = rec["source"].to_numpy()
    a_idx = np.flatnonzero(src == "S1")
    return np.array([i for i in a_idx if _last_best_score.get(i, 0.0) < quality_threshold], dtype=np.int64)


def truth_codes(gt, id2row, N):
    out = []
    for sid, ts in gt.items():
        if sid not in id2row:
            continue
        for t in ts:
            if t in id2row:
                out.append(id2row[sid] * N + id2row[t])
    return np.array(sorted(set(out)), dtype=np.int64)


def blocking_report(rec, cand_codes, true_codes, tag=""):
    N = len(rec)
    hit = np.isin(true_codes, cand_codes)
    b = true_codes % N
    a = true_codes // N
    tgt = rec["source"].to_numpy()[b]
    cty = rec["country_key"].to_numpy()[a]
    print(f"  [blocking{tag}] pair recall = {hit.mean():.4f}  ({hit.sum():,}/{len(hit):,})")
    for s in ("S2", "S3"):
        mm = tgt == s
        if mm.any():
            print(f"      recall {s}: {hit[mm].mean():.4f}")
    for c in np.unique(cty):
        mm = cty == c
        print(f"      recall country={c}: {hit[mm].mean():.4f}")
    return float(hit.mean())
