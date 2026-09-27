"""Embedding-based RESCUE blocking for the entities token-based blocking is failing on — mainly transliterated
names (a different script entirely, e.g. Kannada/Hindi/Punjabi vs Latin) that share zero characters with their
match, so no token/n-gram/edit-distance key can ever find them. A multilingual sentence encoder embeds the two
scripts close together (many are trained on parallel/translation corpora), which token overlap cannot do.

Cost control: we do NOT re-embed the whole corpus. We only embed:
  - S1 records flagged as weak (via blocking.get_weak_entities — a QUALITY signal, not raw count; typically
    only 5-15% of entities, not everyone)
  - the S2/S3 pool of their own country
Both the anchor and target sides are processed in chunks so the similarity matrix never spikes memory.
"""
import time
import numpy as np


def _embed(texts, model, batch_size=256):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)


def rescue_candidates(rec, existing_codes, model, weak_threshold=5, top_k=10, same_country=True,
                       model_batch=256, sim_min=0.5, weak_override=None):
    """Find extra candidates via embedding nearest-neighbour for S1 entities with few/no existing candidates.
    weak_override: if given (e.g. from blocking.get_weak_entities), use this set of S1 row indices directly
    instead of the count-based `weak_threshold` filter. Prefer this — raw candidate count is nearly useless
    here since blocking fills every entity up to max_cand regardless of match quality."""
    N = len(rec)
    src = rec["source"].to_numpy()

    if weak_override is not None:
        weak = np.asarray(weak_override)
        print(f"  [rescue] {len(weak):,} S1 entities flagged by QUALITY filter — running embedding rescue on these")
    else:
        a_idx_all = np.flatnonzero(src == "S1")
        cand_a = existing_codes // N
        counts = np.bincount(cand_a, minlength=N)
        weak = a_idx_all[counts[a_idx_all] < weak_threshold]
        print(f"  [rescue] {len(weak):,} / {len(a_idx_all):,} S1 entities have < {weak_threshold} candidates — running embedding rescue on these")
    if len(weak) == 0:
        return np.array([], dtype=np.int64)

    ck = rec["country_key"].to_numpy() if same_country else np.zeros(N, dtype=object)
    names = rec["business_name"].to_numpy()
    new_codes = []
    t0 = time.time()
    max_cells = 5e7
    for g in np.unique(ck[weak]):
        weak_g = weak[ck[weak] == g]
        for tgt in ("S2", "S3"):
            b_idx = np.flatnonzero((ck == g) & (src == tgt))
            if len(b_idx) == 0 or len(weak_g) == 0:
                continue
            b_emb = _embed(names[b_idx].tolist(), model, model_batch)
            a_chunk_size = max(1, int(max_cells // max(1, len(b_idx))))
            for s in range(0, len(weak_g), a_chunk_size):
                a_slice = weak_g[s:s + a_chunk_size]
                a_emb = _embed(names[a_slice].tolist(), model, model_batch)
                sims = a_emb @ b_emb.T
                k = min(top_k, sims.shape[1])
                top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                top_val = np.take_along_axis(sims, top_idx, axis=1)
                for i, ai in enumerate(a_slice):
                    for j, v in zip(top_idx[i], top_val[i]):
                        if v >= sim_min:
                            new_codes.append(ai * N + b_idx[j])
    new_codes = np.array(sorted(set(new_codes)), dtype=np.int64)
    print(f"  [rescue] added {len(new_codes):,} new candidate pairs in {time.time() - t0:.1f}s")
    return new_codes
