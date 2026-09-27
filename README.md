# Business Entity Resolution: Amazon ML Challenge 2026

**Pipeline:** normalise → block (weighted inverted-index key voting + embedding-based rescue for transliterated/hard-to-match names) → pairwise features → LightGBM classifier → one-to-one decoding with tuned thresholds.

**Metric:** macro F0.5, with singletons included.

**Data policy:** No external data lookups. Country is never used as a model feature (the test set includes France, which is unseen in training).

---

## Repo Layout

```text
business_entity_resolution/
├── src/
│   └── ber/
│       ├── io_utils.py
│       ├── metric.py
│       ├── text_norm.py
│       ├── records.py
│       ├── blocking.py
│       ├── embed_rescue.py
│       ├── features.py
│       ├── decode.py
│       ├── split.py
│       └── pipeline.py
│
├── run_eda.py
├── run_validate.py
├── run_predict.py
│
├── tests/
│   └── make_synthetic.py
│       # Tiny fake dataset to smoke-test the environment
│
├── experiments/
│   └── LOG.md
│       # One row per experiment
│
└── submissions/
    └── SUBMISSIONS.md
        # One row per leaderboard upload
```

Place the challenge `dataset/` folder containing `train/` and `test/`, along with `utils/validate_submission.py`, at the repository root before running.

Both are git-ignored.

---

## Reproduce End-to-End

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers
```

`sentence-transformers` is only required when using `--use_rescue`.

Set the Python path:

```bash
export PYTHONPATH=src
```

### 2. Look at the data

```bash
python -m ber.run_eda --data_dir dataset
```

### 3. Validate and tune thresholds

Run validation on a subsample first. The full 2.2M-row dataset is too slow and memory-heavy for rapid iteration.

```bash
python -m ber.run_validate \
    --data_dir dataset \
    --max_s1 300000 \
    --max_cand 20 \
    --tag med300k
```

### 4. Train and predict

Train on a subsample of the full training set and predict on the **full test set**:

```bash
python -m ber.run_predict \
    --data_dir dataset \
    --config artifacts/config_med300k.json \
    --use_rescue \
    --max_cand_override 5 \
    --train_max_s1 500000 \
    --out_dir output
```

### 5. Validate the submission files

Before submitting, validate the output format:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Key Flags

### `--max_cand`

Number of candidates retained per S1 entity per target source.

This controls the trade-off between blocking recall and computational cost:

* **Higher values**: more candidates, potentially higher recall
* **Lower values**: faster execution and lower memory usage

Example:

```text
--max_cand 20
```

### `--max_bucket`

Drops blocking keys shared by more than the specified number of records.

Very common keys are usually too generic to provide useful candidate matches.

Example:

```text
--max_bucket <N>
```

### `--use_rescue`

Adds an embedding-based candidate-rescue pass using a multilingual sentence encoder.

This is intended for entities whose best token-based match relies only on weak signals and helps recover difficult matches, including transliterated names.

For example, the same business name may appear in completely different scripts across two sources.

Example:

```text
--use_rescue
```

### `--train_max_s1`

Used by `run_predict` to limit the number of S1 training records used for model fitting.

This is primarily a memory-safety control when working at the full ~2.2M-row scale.

Example:

```text
--train_max_s1 500000
```

### `--max_cand_override`

Used by `run_predict` to override the candidate width from the validation configuration.

This is also primarily intended for memory and runtime control during full-scale prediction.

Example:

```text
--max_cand_override 5
```

---

## Results

> **TODO:** Fill in the final validated F0.5, per-country breakdown, and actual leaderboard score once available.

| Run                           |     Scale | Overall F0.5 | India F0.5 | US F0.5 | Notes                              |
| ----------------------------- | --------: | -----------: | ---------: | ------: | ---------------------------------- |
| med300k (no rescue)           |      300K |       0.9789 |     0.9618 |  0.9904 | Baseline                           |
| rescue_med300k                |      300K |       0.9810 |     0.9657 |  0.9912 | + embedding rescue on all entities |
| **[final leaderboard score]** | Full test |            — |          — |       — | **[TO FILL]**                      |

---

## Experiment Tracking

See [`experiments/LOG.md`](experiments/LOG.md) for the full experiment history.

Each experiment records the configuration, scale, validation metrics, and relevant observations.

---

## Submission Tracking

See [`submissions/SUBMISSIONS.md`](submissions/SUBMISSIONS.md) for the leaderboard submission history.

---

## Notes

* The pipeline does **not** use external data lookups.
* `country` is deliberately excluded as a model feature.
* The test set contains **France**, which is unseen in the training data.
* The embedding rescue stage is optional and can be enabled with `--use_rescue`.
* Full-scale runs are substantially more memory-intensive than the validation runs, so `--train_max_s1` and `--max_cand_override` can be used to control resource usage.
