# Resume Point: Tests 1–2 (baseline + repetition), 8 models, 6 languages

_Last updated: 2026-10-07 18:25 PKT_

## Where things stand

| Item | State | Where the output is |
|---|---|---|
| Qwen 1.5B / 3B / 7B | ✅ **Done**: 78,396 generations, 0 errors (Kaggle version 3) | `results/qwen_v3/raw_jsonl.zip` + `analysis/`; Kaggle dataset `fiznasir/pr-results-checkpoint` |
| Llama 3.2 1B / 3B, Gemma-2 2B, Mistral 7B, Llama 3.1 8B | ⏳ **Running**: Kaggle notebook `fiznasir/prompt-repetition-tests-1-2` **version 4**, started 18:16 PKT, ETA ~21:00–23:00 | Kaggle output of version 4; live mirror in HF dataset `<hf-user>/prompt-repetition-results` (`raw/<model>/`) |
| Persian (fa) + Sindhi (sd) translations of ARC / OpenBookQA / CommonSenseQA / MGSM | ⏳ **Running**: Kaggle CPU job `fiznasir/pr-translate-fa-sd-nllb` (NLLB-200 1.3B), started 16:47 PKT, ETA ~19:00–21:00 | Kaggle output of that job: `out/*_NLLB.xlsx` |

What each language currently covers:
- **pa, ur, ps, ar:** all 7 tasks.
- **fa:** NameIndex, MiddleMatch and ScriptMixed only, until the NLLB files are added.
- **sd:** retrieval tasks plus MGSM (Google), until the NLLB files are added. Once they are, all Sindhi tasks switch to NLLB, keeping one engine per language.

The current branch is **`kaggle-tests-1-2`**, the one the notebook clones.

## Next steps (in order)

1. **Check that both Kaggle jobs finished.** Use `kaggle kernels status fiznasir/<slug>`, or go to kaggle.com → *View Active Events*.

2. **Add the NLLB translations:**
   ```bash
   kaggle kernels output fiznasir/pr-translate-fa-sd-nllb -p /tmp/nllb
   cp /tmp/nllb/out/*_Persian_NLLB.xlsx datasets/persian/
   cp /tmp/nllb/out/*_Sindhi_NLLB.xlsx  datasets/sindhi/
   python -c "import persian_datasets_loader as p, sindhi_datasets_loader as s; p.build_dataset(); s.build_dataset()"
   ```
   - Check the quality summary in the job log: empty translations, identical options, the MGSM number check, and Sindhi letters present.
   - Then commit and push to `kaggle-tests-1-2`.
   - `translated_benchmarks.py` uses the `_NLLB` files automatically for any language that has them.

3. **Refresh the results checkpoint with version 4's output:**
   ```bash
   kaggle kernels output fiznasir/prompt-repetition-tests-1-2 -p /tmp/v4
   # flatten raw/<model>/<file> -> <model>__<file>, then:
   kaggle datasets version -p <flat dir> -m "add v4 models"   # dataset fiznasir/pr-results-checkpoint
   ```

4. **Top-up run (version 5) for the new fa/sd items:**
   - Push the notebook again with `dataset_sources: ["fiznasir/pr-results-checkpoint"]`.
   - All 8 models run, and only the new fa/sd jobs execute (about 3k per model, roughly 1.5–2 h total).
   - The HF mirror also restores progress if the checkpoint is stale.

5. **Final analysis:** `python score_results.py --results <dir with raw/>`. The notebook's last cell also does this. It writes `accuracy_table.csv`, `mcnemar_results.csv` and `wins_summary.csv`.

## How the pieces fit

- **`pr_runner.py`:** one model per call. It appends to `raw/<model>/<lang>[.shardIofN].jsonl` with fsync per chunk, mirrors to HF if `HF_RESULTS_REPO`/`HF_TOKEN` are set, and skips anything already done. It never repeats work.
- **`run_experiments.ipynb`:** loops over the 8 models on Kaggle T4×2. Small models run as one shard per GPU; 7–8B models and Gemma (fp32) use tensor parallelism across both GPUs. It restores earlier results from `/kaggle/input` and the HF mirror, and stops itself at 11.3 h.
- **`score_results.py`:** re-scores raw responses and runs McNemar (no continuity correction, p < 0.1, as in the paper).
- **Kaggle API pushes** use `kernel-metadata.json` with `machine_shape: NvidiaTeslaT4`, internet on, private. The `HF_TOKEN` secret is attached in the Kaggle UI (the API can't set secrets).
- **Kaggle gotcha:** after installing vLLM 0.30.0, uninstall `torchaudio` (the notebook does this), or transformers fails with a CUDA-version mismatch.

## Decisions on record

- Balochi was dropped: 2 of its 3 MCQ benchmarks were missing.
- Conditions follow arXiv:2512.14982 (A.3/A.4 prompt formats, McNemar p < 0.1), except for the 8 chosen models and the repo's 4 tests. Tests 1–2 (baseline, repetition) run first; tests 3–4 can be added later with `--methods ... cross_lingual_t1 cross_lingual_t2`, reusing the finished rows.
- Translation engines: pa/ur via Google Translate (existing); ar via Google Translate (gtx endpoint, before Google rate-limited this IP); fa/sd via **NLLB-200 1.3B**. This must be disclosed in the paper.
- System prompts and native retrieval strings were machine-checked only; they still need a native-speaker review.

## Early Qwen results (tests 1–2)

Significant wins for repetition over baseline (McNemar, p < 0.1):
- **All tasks:** 47 wins / 6 losses out of 141 tests.
- **Paper tasks only:** 23 / 6 out of 99.
- The largest gains are on options-first MCQ and ScriptMixed.
- NameIndex accuracy is about 1–5% for both methods. This is real, not a scoring bug: the correct name appears anywhere in the reply only 3.4% of the time.
