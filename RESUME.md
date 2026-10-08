# Resume Point: Tests 1–2 (baseline + repetition), 8 models, 6 languages

_Last updated: 2026-10-08 09:30 PKT_

## Where things stand: tests 1–2 complete ✅

All 8 models × 6 languages × 2 methods are complete. The run produced **280,816 generations** with **0 errors** and **0 prompts skipped as too long**.

| Language | Jobs per model (items × scenarios × 2 methods) |
|---|---|
| pa | 5,992 |
| ur | 6,040 |
| ps | 5,096 |
| ar | 6,024 |
| fa | 5,994 |
| sd | 5,956 |

| What | Where |
|---|---|
| Raw generations (72 JSONL files, 213 MB) | Kaggle dataset `fiznasir/pr-results-checkpoint` (flat `<model>__<lang>[.shardIofN].jsonl`) and private HF dataset `fizzah12pew/prompt-repetition-results` (`raw/<model>/`) |
| Result tables | `results/tests_1_2/` (`accuracy_table.csv`, `mcnemar_results.csv`, `wins_summary.csv`) |
| Per-generation scores | `scored.csv` (76 MB). Not in git; regenerate with `python score_results.py --results <dir with raw/>` |

### Run history
- **Kaggle versions 3–4:** Qwen ×3 and Mistral. Version 4 skipped the gated models because no `HF_TOKEN` was attached.
- **Version 5 (6.96 h):**
  - Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B, Gemma-2 2B.
  - Pulled the repo before each model, so models started after the NLLB push included the new fa/sd items.
- **Version 8 (1.4 h):** fa/sd NLLB top-up for Llama 1B, Qwen ×3 and Mistral. Version 7 started with a stale input and was stopped.
- **Translations:** NLLB GPU job `fiznasir/pr-translate-fa-sd-nllb-gpu`, versions 1 and 2, merged as described below.

## Next steps

- **Tests 3–4 (cross-lingual):** add `"cross_lingual_t1", "cross_lingual_t2"` to `METHODS` in the notebook.
  - Completed baseline/repetition rows are reused.
  - Start from the editor so the `HF_TOKEN` secret is attached, with `pr-results-checkpoint` (latest) as input.
- **Native-speaker review** of system prompts, retrieval strings and the fa/sd NLLB files.

## How the pieces fit

- **`pr_runner.py`:**
  - Runs one model per call.
  - Appends to `raw/<model>/<lang>[.shardIofN].jsonl` with fsync per chunk.
  - Mirrors to HF if `HF_RESULTS_REPO`/`HF_TOKEN` are set; this needs a write token, or every model crashes at startup.
  - Skips anything already done. Jobs store `distractors` and `anchors` (MiddleMatch question names).
- **`run_experiments.ipynb`:**
  - Loops over `MODELS` on Kaggle T4×2.
  - Small models run as one shard per GPU. 7–8B models and Gemma (fp32) use tensor parallelism.
  - Restores from `/kaggle/input` and the HF mirror, and stops itself at 11.3 h.
  - **Start it from the Kaggle editor (Save & Run All).** `kaggle kernels push` drops UI-attached secrets.
- **`score_results.py`:**
  - Re-scores raw responses with the current `experiment_runner.is_correct`.
  - Rebuilds MiddleMatch anchors from the loaders for old rows.
  - Prefers successful rows on duplicate keys and computes accuracy over paired items only.
  - McNemar: chi-square, no continuity correction, p < 0.1, as in the paper. It also reports the **exact (binomial) McNemar test** as a robustness check (`p_exact`, `significant_exact`, `wins_exact`, `losses_exact`).
- **Scoring:**
  - MCQ and math answers are extracted from the reply.
  - Retrieval uses **exact name match after Perso-Arabic spelling normalisation**, not fuzzy matching. A restated MiddleMatch question is removed before matching.
- **Kaggle gotchas:**
  - Uninstall `torchaudio` after installing vLLM 0.30.0 (the notebook does this).
  - On Windows, set `PYTHONIOENCODING=utf-8` for `kaggle` downloads (charmap error).
  - Run `kaggle datasets version` from a short working directory with a relative `-p` (path-length error).
  - A dataset version replaces **all** files, so always upload the full set.
  - Gemma-2 runs in fp32 on T4 at about 2 prompts/s, roughly 4× slower than the others; run it last.

## Decisions on record

- Balochi was dropped: 2 of its 3 MCQ benchmarks were missing.
- Conditions follow arXiv:2512.14982 (A.3/A.4 prompt formats, McNemar p < 0.1), except for the 8 chosen models and the repo's 4 tests.
- Translation engines (must be disclosed in the paper):
  - pa/ur via Google Translate (existing).
  - ar via Google Translate (gtx endpoint).
  - fa/sd via **NLLB-200 1.3B**, merged row by row from two passes using automatic QA flags.
    - MCQ files use the sentence-level pass, falling back to the whole-line pass.
    - MGSM uses the whole-line pass (formal register, fewer unit/decimal conversions), falling back to the sentence-level pass.
    - Rows defective in both passes were excluded: fa 4 (CSQA 1, MGSM 3) and sd 8 (ARC 3, OBQA 1, MGSM 4).
    - The source pass for each row is in the `NLLB Pass` column. Final counts: fa 1932 items, sd 1922.
- Retrieval scoring is exact normalised name match. Fuzzy ≥ 0.80 was dropped because distinct names such as صفیہ بیگم / روبینہ بیگم score above it.
- System prompts and native retrieval strings were machine-checked only; they still need a native-speaker review.
- Also disclose:
  - Gemma's system prompt is merged into the user turn.
  - Llama 3.x chat templates insert a "Today Date".

## Results: repetition vs baseline (8 models × 6 languages)

Significant wins / losses (McNemar, p < 0.1; the exact test is in brackets):
- **All tasks:** 134 / 21 out of 480 tests [118 / 15].
- **Paper tasks only** (ARC, OpenBookQA, GSM8K, NameIndex, MiddleMatch): 71 / 17 out of 336 [65 / 11].

| Model | Wins / losses (60 tests) | Mean accuracy, baseline → repetition |
|---|---|---|
| Llama 3.1 8B | 26 / 1 | 37.4% → 41.3% |
| Qwen 2.5 3B | 23 / 1 | 26.4% → 29.0% |
| Qwen 2.5 7B | 21 / 0 | 32.8% → 36.0% |
| Gemma-2 2B | 14 / 2 | 25.5% → 27.2% |
| Llama 3.2 3B | 14 / 2 | 29.4% → 31.1% |
| Llama 3.2 1B | 14 / 7 | 19.6% → 20.2% |
| Qwen 2.5 1.5B | 11 / 2 | 22.1% → 22.9% |
| Mistral 7B v0.3 | 11 / 6 | 21.5% → 21.8% |

By language (80 tests each):

| ar | fa | ps | ur | pa | sd |
|---|---|---|---|---|---|
| 38 / 4 | 34 / 2 | 17 / 3 | 17 / 2 | 15 / 6 | 13 / 4 |

By task:

| Task | Wins / losses | Tests |
|---|---|---|
| ScriptMixed | 36 / 0 | 48 |
| OpenBookQA | 31 / 2 | 96 |
| CommonSenseQA | 27 / 4 | 96 |
| ARC | 20 / 5 | 96 |
| GSM8K | 8 / 2 | 48 |
| NameIndex | 5 / 1 | 48 |
| MiddleMatch | 7 / 7 | 48 |

Mean accuracy is item-weighted over all languages, tasks and scenarios (from `accuracy_table.csv`).
