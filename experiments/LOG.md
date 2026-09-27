# Experiment log

| id | scale | change | overall F0.5 | India F0.5 | US F0.5 | blocking recall (India) | notes |
|----|-------|--------|-------------|-----------|---------|------------------------|-------|
| e01 | 30K | baseline blocking (TF-IDF kNN, before address-token key) | 0.9917 | ~0.978 | ~0.997 | — | |
| e02 | 30K | + address-token blocking key (fixes transliterated-name recall) | 0.9931 | ~0.987 | ~0.997 | — | major India recall jump |
| e03 | 30K | switched to inverted-index weighted-vote blocking (scale fix), max_cand=20 | 0.9917 | 0.9838 | 0.9967 | 0.9644 | |
| e04 | 30K | + embedding similarity as a classifier FEATURE (not blocking) | 0.9914 | 0.9835 | 0.9967 | 0.9644 | no aggregate improvement — targets wrong stage |
| e05 | 30K | + embedding RESCUE blocking, forced on ALL entities | **0.9926** | **0.9864** | 0.9968 | ~0.98 | real improvement, targets actual bottleneck |
| e06 | 300K | med300k, no rescue | 0.9789 | 0.9618 | 0.9904 | — | |
| e07 | 300K | rescue_med300k, forced on all entities | **0.9810** | **0.9657** | 0.9912 | 0.9424-0.9787 | best validated config, used for final predict |
| e08 | full/subsample | final run_predict, train_max_s1=500000, max_cand_override=5 | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | actual submission run |

## Key findings
- Blocking recall (not classifier quality) was the real bottleneck for India throughout — string/edit-distance
  features cannot help entities blocking never surfaces as candidates in the first place.
- India's main failure mode: names transliterated into a different script (Kannada/Hindi/Punjabi/etc.) between
  sources, sharing zero characters — no token/n-gram key can ever catch this; addresses, staying in Latin
  script, were the more reliable blocking signal (added as the `atok` key).
- Embedding rescue as a FEATURE (scoring already-found candidates) did not move the aggregate score;
  embedding rescue as BLOCKING (finding new candidates for weak entities) did — the lesson being to fix
  the stage that's actually the bottleneck, not just add signal downstream of it.
- Quality-based weak-entity selection (candidate confidence) under-triggered at threshold=3.0 (~0.04% of
  entities) — most of the benefit came from running rescue on all entities, not a filtered subset; this
  is a known cost/quality trade-off not fully resolved before the deadline.
- France: zero training examples in every configuration — structural blind spot, not addressed by any of
  the above; diagnostic check (see below) showed blocking mechanics work fine on France records regardless.
- France-specific fix: added "R" -> "rue" address abbreviation (French street abbreviation, previously unmapped).
