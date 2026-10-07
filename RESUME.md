# Resume Point: Tests 1–2 (baseline + repetition), 8 models, 6 languages

_Last updated: 2026-10-07 23:45 PKT_

## Where things stand

| Item | State | Where the output is |
|---|---|---|
| Qwen 1.5B / 3B / 7B, Mistral 7B | ✅ **Done** on every item available so far (26,132 generations each, 0 errors) | Kaggle dataset `fiznasir/pr-results-checkpoint` (flat `<model>__<lang>[.shardIofN].jsonl`) |
| Llama 3.2 1B / 3B, Llama 3.1 8B, Gemma-2 2B | ⏳ **Not run yet.** Version 4 skipped them because no `HF_TOKEN` secret was attached. To be run as **version 5** | Kaggle output of version 5 + HF mirror `<hf-user>/prompt-repetition-results` |
| Persian (fa) + Sindhi (sd) NLLB translations of ARC / OpenBookQA / CommonSenseQA / MGSM | ⏳ **Re-translating.** GPU job `fiznasir/pr-translate-fa-sd-nllb-gpu` **version 2** | Kaggle output: `out/*_NLLB.xlsx` |

Checkpoint note: the Google-translated Sindhi GSM8K rows were **removed** from the checkpoint, because Sindhi switches to NLLB and the item ids are the same. Run with `SKIP = ["sd:GSM8K"]` until the NLLB files are in the repo.

NLLB version 1 failed QA:
- NLLB-200 drops sentences when given multi-sentence input, often the question itself.
- It also loops on some inputs.
- `translate_benchmarks.py` now translates sentence by sentence, retries looping outputs and folds Persian ي/ك to ی/ک (cache prefix `nllb2-`).
- The CPU job `fiznasir/pr-translate-fa-sd-nllb` runs the old code. It is obsolete; ignore or stop it.

What each language currently covers:
- **pa, ur, ps, ar:** all 7 tasks.
- **fa, sd:** NameIndex, MiddleMatch, ScriptMixed, until the NLLB files are added. Then Persian and Sindhi also get ARC, OpenBookQA, CommonSenseQA and MGSM, all from NLLB.

The current branch is **`kaggle-tests-1-2`**, the one the notebook clones. Do not delete it: PR #2 uses it as its head branch.

## Next steps (in order)

1. **Start version 5 from the Kaggle editor, not with `kaggle kernels push`.** Secrets attached in the UI are dropped by API pushes.
   - In cell 2, set `MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "gemma2-2b"]` and `SKIP = ["sd:GSM8K"]`.
   - Input: `pr-results-checkpoint` (latest version). Secret `HF_TOKEN` must be a **write** token and ticked. GPU T4 ×2, internet on.
   - Save Version → Save & Run All.
   - The log must show `HF mirror: <user>/prompt-repetition-results`, and its `git log` line must show the pushed commit.
   - Estimate: 4.5–7.5 h. Gemma runs last because it runs in fp32 and is slow if vLLM falls back to transformers.
2. **When NLLB version 2 passes QA:** copy `out/*_Persian_NLLB.xlsx` → `datasets/persian/` and `out/*_Sindhi_NLLB.xlsx` → `datasets/sindhi/`. Then run `python -c "import persian_datasets_loader as p, sindhi_datasets_loader as s; p.build_dataset(); s.build_dataset()"`, commit and push.
3. **When version 5 finishes:** download its output. Check that the 4 gated models are complete. Flatten `raw/<model>/<file>` to `<model>__<file>` and upload all files as a new checkpoint version. A dataset version replaces every file, so include all 8 models.
4. **Version 6 top-up:**
   - Set `MODELS` to all 8 and `SKIP = []`.
   - Only the new fa/sd NLLB items run: about 9.1–9.6k prompts per model, about 2.5–3 h in total.
5. **Final analysis:** `python score_results.py --results <dir with raw/>`. Add exact-McNemar p-values as a robustness column: 9 of 67 significant results in the 4-model data depend on the chi-square variant.

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
  - If `llama3.2-1b` (the smoke-test model) lacks a licence, the whole run aborts. Other gated failures only show in the status dict.
- **`score_results.py`:**
  - Re-scores raw responses with the current `experiment_runner.is_correct`.
  - Rebuilds MiddleMatch anchors from the loaders for old rows.
  - Prefers successful rows on duplicate keys and computes accuracy over paired items only.
  - McNemar: chi-square, no continuity correction, p < 0.1, as in the paper.
- **Scoring:**
  - MCQ and math answers are extracted from the reply.
  - Retrieval uses **exact name match after Perso-Arabic spelling normalisation**, not fuzzy matching. A restated MiddleMatch question is removed before matching.
- **Kaggle gotchas:**
  - Uninstall `torchaudio` after installing vLLM 0.30.0 (the notebook does this).
  - On Windows, set `PYTHONIOENCODING=utf-8` for `kaggle` downloads (charmap error).
  - Run `kaggle datasets version` from a short working directory with a relative `-p` (path-length error).

## Decisions on record

- Balochi was dropped: 2 of its 3 MCQ benchmarks were missing.
- Conditions follow arXiv:2512.14982 (A.3/A.4 prompt formats, McNemar p < 0.1), except for the 8 chosen models and the repo's 4 tests. Tests 1–2 (baseline, repetition) run first. Tests 3–4 can be added later with `--methods ... cross_lingual_t1 cross_lingual_t2`, reusing the finished rows.
- Translation engines (must be disclosed in the paper):
  - pa/ur via Google Translate (existing).
  - ar via Google Translate (gtx endpoint).
  - fa/sd via **NLLB-200 1.3B**, sentence-level.
- Retrieval scoring is exact normalised name match. Fuzzy ≥ 0.80 was dropped because distinct names such as صفیہ بیگم / روبینہ بیگم score above it.
- System prompts and native retrieval strings were machine-checked only; they still need a native-speaker review.
- Also disclose:
  - Gemma's system prompt is merged into the user turn.
  - Llama 3.x chat templates insert a "Today Date".

## Results so far (4 models, fixed scorer)

Significant wins / losses for repetition over baseline (McNemar, p < 0.1):
- **All tasks:** 56 / 7 out of 188 tests.
- **Paper tasks only:** 27 / 5 out of 132.

| Model | All tasks |
|---|---|
| Qwen-1.5B | 13 / 1 |
| Qwen-3B | 19 / 3 |
| Qwen-7B | 15 / 2 |
| Mistral-7B | 10 / 4 |

These per-model counts are from the old scorer; the fixed scorer changes 8 outcomes, 7 of them MiddleMatch. The largest gains are on options-first MCQ and ScriptMixed. NameIndex accuracy is about 1–5% for both methods.
