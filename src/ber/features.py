"""Pair features for the classifier. Everything here is COUNTRY-AGNOSTIC (no country feature) so the model can
transfer to France, which never appears in training. Missing address information -> NaN (LightGBM handles NaN)."""
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer


class Views:
    """TF-IDF vectorizers fit ONLY on the rows that appear in the candidate pairs (`rows`), not the whole corpus —
    at multi-million-row scale, fitting/transforming everything would be wasted work. Call `.fit(rows)` once per
    build_pair_table call, then `.cos_for_pairs(name, a, b)` scores arbitrary pairs of those rows."""

    def __init__(self, rec):
        self.rec = rec
        self.vecs = {}
        self.mats = {}
        self.rows = None

    def fit(self, rows):
        self.rows = np.asarray(rows)
        sub = self.rec.iloc[self.rows]
        kw_char = dict(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, min_df=2, max_df=0.5, dtype=np.float32)
        specs = {
            "name_char": (kw_char, sub["name_core"]),
            "name_word": (dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                              sublinear_tf=True, dtype=np.float32), sub["name_core"]),
            "full_char": (kw_char, sub["name_addr"]),
            "addr_char": (kw_char, sub["addr_core"]),
        }
        for name, (kw, col) in specs.items():
            v = TfidfVectorizer(**kw).fit(col)
            self.vecs[name] = v
            self.mats[name] = v.transform(col)
        return self

    def cos_for_pairs(self, name, a, b, chunk=200_000):
        """Cosine similarity for pairs given as ORIGINAL rec row indices a[i], b[i] (looked up into self.rows).
        Chunked: fancy-indexing a sparse matrix with repeated row indices materialises a full copy of those rows,
        so scoring millions of pairs at once can spike memory. Chunking bounds peak memory."""
        pos = {r: i for i, r in enumerate(self.rows)}
        ai_full = np.fromiter((pos[x] for x in a), dtype=np.int64, count=len(a))
        bi_full = np.fromiter((pos[x] for x in b), dtype=np.int64, count=len(b))
        M = self.mats[name]
        out = np.empty(len(a), np.float32)
        for s in range(0, len(a), chunk):
            e = s + chunk
            out[s:e] = np.asarray(M[ai_full[s:e]].multiply(M[bi_full[s:e]]).sum(axis=1)).ravel()
        return out


def _jacc(x, y):
    if not x or not y:
        return np.nan
    return len(x & y) / len(x | y)


def string_features(rec, a, b):
    """Fuzzy-string / token / postal-code features for every pair. Python loop over pairs (rapidfuzz is fast)."""
    nc, nn = rec["name_core"].to_numpy(), rec["name_norm"].to_numpy()
    acr, ad = rec["name_acr"].to_numpy(), rec["addr_core"].to_numpy()
    post, nums = rec["postal"].to_numpy(), rec["nums"].to_numpy()
    n = len(a)
    names = ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_lev", "nfull_tset", "n_len_ratio",
             "n_tok_jacc", "n_first_tok", "n_compact_eq", "n_acr", "n_len_a", "n_len_b",
             "a_ratio", "a_tsort", "a_tset", "a_partial", "a_tok_jacc", "a_len_ratio", "a_missing",
             "pc_eq", "pc_both", "num_jacc", "num_common"]
    F = {k: np.full(n, np.nan, np.float32) for k in names}
    for t in range(n):
        i, j = a[t], b[t]
        x, y = nc[i], nc[j]
        if x and y:
            F["n_ratio"][t] = fuzz.ratio(x, y) / 100
            F["n_tsort"][t] = fuzz.token_sort_ratio(x, y) / 100
            F["n_tset"][t] = fuzz.token_set_ratio(x, y) / 100
            F["n_partial"][t] = fuzz.partial_ratio(x, y) / 100
            F["n_jw"][t] = JaroWinkler.similarity(x, y)
            F["n_lev"][t] = Levenshtein.normalized_similarity(x, y)
            F["n_len_ratio"][t] = min(len(x), len(y)) / max(len(x), len(y))
            xs, ys = set(x.split()), set(y.split())
            F["n_tok_jacc"][t] = _jacc(xs, ys)
            F["n_first_tok"][t] = float(x.split()[0] == y.split()[0])
            xc, yc = x.replace(" ", ""), y.replace(" ", "")
            F["n_compact_eq"][t] = float(xc == yc)
            F["n_acr"][t] = float((len(xc) >= 2 and acr[j] == xc) or (len(yc) >= 2 and acr[i] == yc))
        F["n_len_a"][t], F["n_len_b"][t] = len(x), len(y)
        if nn[i] and nn[j]:
            F["nfull_tset"][t] = fuzz.token_set_ratio(nn[i], nn[j]) / 100
        p, q = ad[i], ad[j]
        F["a_missing"][t] = float(not p or not q)
        if p and q:
            F["a_ratio"][t] = fuzz.ratio(p, q) / 100
            F["a_tsort"][t] = fuzz.token_sort_ratio(p, q) / 100
            F["a_tset"][t] = fuzz.token_set_ratio(p, q) / 100
            F["a_partial"][t] = fuzz.partial_ratio(p, q) / 100
            F["a_tok_jacc"][t] = _jacc(set(p.split()), set(q.split()))
            F["a_len_ratio"][t] = min(len(p), len(q)) / max(len(p), len(q))
        pi, pj = post[i], post[j]
        F["pc_both"][t] = float(bool(pi) and bool(pj))
        if pi and pj:
            F["pc_eq"][t] = float(len(pi & pj) > 0)
        ni, nj = nums[i], nums[j]
        F["num_jacc"][t] = _jacc(ni, nj)
        F["num_common"][t] = len(ni & nj)
    return pd.DataFrame(F)


def add_rank_features(df):
    """Competition features: how does this pair compare with the OTHER candidates of the same S1 record (s1_*)
    and of the same target record (t_*)? Mutual best matches are very strong evidence in entity resolution.
    NOTE: raw candidate-list SIZES are deliberately NOT used as features - they depend on pool size and country mix,
    which differ between training and test (France), so the model would learn a spurious shortcut."""
    df["base"] = (df["name_char"] + df["name_word"] + df["full_char"]) / 3.0
    for key, pre in (("a", "s1"), ("b", "t")):
        g = df.groupby(key)["base"]
        df[f"{pre}_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
        top = g.transform("max")
        d = df[[key, "base"]].sort_values([key, "base"], ascending=[True, False])
        d["_r"] = d.groupby(key).cumcount()
        second = d[d["_r"] == 1].set_index(key)["base"]
        second = df[key].map(second).fillna(0.0)
        df[f"{pre}_gap"] = np.where(df[f"{pre}_rank"] == 1, df["base"] - second, df["base"] - top).astype(np.float32)
    return df


def build_features(rec, views, cand_codes):
    """Candidate codes -> feature DataFrame with columns a, b, src (0=S2,1=S3) and all features.
    `views` is fit HERE, only on the rows touched by cand_codes (see Views.fit) — not the whole corpus."""
    t0 = time.time()
    N = len(rec)
    a, b = cand_codes // N, cand_codes % N
    df = pd.DataFrame({"a": a, "b": b})
    df["src"] = (rec["source"].to_numpy()[b] == "S3").astype(np.int8)
    if len(a):
        views.fit(np.unique(np.concatenate([a, b])))
        for name in views.mats:
            df[name] = views.cos_for_pairs(name, a, b)
    else:
        for name in ("name_char", "name_word", "full_char", "addr_char"):
            df[name] = pd.Series(dtype=np.float32)
    df = pd.concat([df, string_features(rec, a, b)], axis=1)
    df = add_rank_features(df)
    print(f"  [features] {len(df):,} pairs x {df.shape[1]} cols in {time.time() - t0:.1f}s")
    return df


NON_FEATURE_COLS = {"a", "b", "label", "p"}


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
