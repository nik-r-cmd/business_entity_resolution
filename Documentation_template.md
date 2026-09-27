# Methodology Document — confused perceptrons

## 1. Methodology Used
Entity resolution via a three-stage pipeline: (1) scalable inverted-index blocking to generate candidate
pairs, (2) supervised pairwise binary classification (LightGBM) to score each candidate, (3) one-to-one
decoding with per-target-source thresholds tuned to directly maximize the macro F0.5 metric.

The core design constraint is scale (~2.2M Source-1 records, ~5M each of Source-2/Source-3): naive pairwise
comparison (O(N×M)) or even naive TF-IDF nearest-neighbour blocking over the full corpus is computationally
infeasible, so blocking uses a weighted inverted-index over cheap-to-compute keys instead.

## 2. Candidate Generation / Blocking Strategy
For every Source-1 record, we build several **blocking keys** from its normalised name and address:
- Distinctive name tokens (stop-words removed)
- A name prefix and character 4-grams of the compacted name (catches typos/reorderings)
- Exact postal/PIN/ZIP code
- Distinctive address tokens (added after error analysis showed many India records are **transliterated
  into a different script** between sources — e.g. a business name in Latin script on one source and
  Devanagari/Kannada/Punjabi on another, sharing zero characters — while the address text typically stays in
  Latin script and matches almost verbatim)
- House/plot/door numbers

Each key type has a weight (an exact postal-code match counts far more than a single shared 4-gram). For
each Source-1 record we accumulate a weighted vote for every same-country Source-2/Source-3 record sharing
at least one key, and keep only the top-K by vote score. This bounds candidates-per-entity to a fixed number
regardless of how many keys a name happens to generate — the property that keeps this tractable at scale.
Keys shared by an excessive number of records are dropped as too generic before voting.

**Embedding-based rescue pass:** to specifically address the transliteration case above, we additionally run
a multilingual sentence encoder ([MODEL NAME/LICENSE — fill in]) over records whose best token-based match
relied only on weak signals, and add nearest-neighbour candidates by embedding cosine similarity within the
same country. This measurably improved recall and downstream F0.5 on India in our validation.

**Recall ceiling / reduction ratio:** [FILL IN — final blocking recall numbers from your best run's log,
train-part and val-part/test, overall and per-country]

## 3. Model Architecture and Feature Engineering
**Model:** LightGBM binary classifier (gradient-boosted decision trees) predicting match probability per
candidate pair. Trained on labelled candidate pairs from the training ground truth; early-stopped on a
held-out validation split.

**Features (~35-36):**
- Fuzzy string similarity on normalised name/address: Levenshtein ratio, Jaro-Winkler, token-sort/token-set
  ratio, partial ratio (rapidfuzz)
- TF-IDF cosine similarity on name/address (character n-gram and word n-gram vectorisers, fit only on the
  records touched by candidate pairs — not the full corpus, for memory efficiency)
- Exact/structured overlap: postal code match, house/plot number Jaccard, token Jaccard
- Acronym detection (e.g. "IBM" vs "International Business Machines")
- **Rank/gap features**: for each candidate, its rank among all other candidates of the same Source-1 entity
  (and vice versa for the target record), and the score gap to the runner-up/best. These were consistently
  among the most important features across every experiment.
- Multilingual embedding cosine similarity (name), when the rescue pass is used

**Decoding:** each Source-2/Source-3 record is assigned to at most one Source-1 entity (its highest-
probability candidate) — validated as a safe assumption from the training ground truth (zero target IDs
claimed by more than one Source-1 entity). Per-target-source probability thresholds are grid-searched on
held-out validation data to directly maximise macro F0.5. Country is intentionally never used as a model
feature, since the test set contains France, absent from training entirely.

## 4. Other Relevant Information
- **No external data lookups, APIs, or geocoding services used** at any stage.
- **Validation strategy:** entity-level held-out split (80/20), plus a leave-one-country-out check (e.g.
  train on US, validate on India and vice versa) as a proxy for generalising to the unseen France country.
- **Compute constraints:** the full training set (2.2M Source-1 entities) proved too memory-intensive to
  process end-to-end in a single pass in our environment; the final model was trained on a representative
  subsample of [FILL IN — e.g. 500,000] Source-1 entities from the full training set, with inference always
  run on the complete test set as required.
- **Embedding model licence:** [FILL IN — confirm MIT/Apache-2.0 and parameter count of whichever multilingual
  sentence-transformer model was used, per the challenge's model-license rule]
- **Final validation F0.5:** [FILL IN]
- **Final leaderboard score:** [FILL IN]
