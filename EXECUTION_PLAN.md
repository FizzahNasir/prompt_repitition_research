# Execution Plan: Multilingual RTL Prompt Repetition Benchmark
**Research:** Replication of arXiv:2512.14982 for RTL / Low-resource Languages

---

## 1. Status Legend
- [x] Done
- [~] In Progress
- [ ] Not Started
- [!] Blocked

---

## 2. API Key Availability

| Provider | Env Var | Status | Models Available |
|----------|---------|--------|-----------------|
| OpenAI | `OPENAI_API_KEY` | ✅ Available | gpt-4o, gpt-4o-mini |
| Google | `GEMINI_API_KEY` | ✅ Available | gemini-2.0-flash, gemini-2.0-flash-lite |
| Anthropic | `ANTHROPIC_API_KEY` | ❌ Not Set | Claude 3.7 Sonnet unavailable |
| DeepSeek | `DEEPSEEK_API_KEY` | ❌ Not Set | DeepSeek-V3 unavailable |
| TogetherAI | `TOGETHER_API_KEY` | ❌ Not Set | Qwen, Llama, Aya unavailable |

**Original paper models (arXiv:2512.14982):**
- GPT-4o ✅ Available
- GPT-4o-mini ✅ Available
- Gemini 2.0 Flash ✅ Available
- Claude 3.7 Sonnet ❌ Key missing
- DeepSeek V3 ❌ Key missing

---

## 3. Language Status

| Language | Code | Datasets | Loader | System Prompt | Scenario |
|----------|------|----------|--------|---------------|----------|
| Punjabi (Shahmukhi) | `pa` | ✅ 1,553 items | ✅ Complete | ✅ `SYSTEM_PROMPT_PA` | ✅ Ready |
| Urdu | `ur` | ✅ 6,825 items | ✅ Complete | ✅ `SYSTEM_PROMPT_UR` | ✅ Ready |
| Pashto | `ps` | ✅ 1,320 items (CSV) | ✅ `pashto_datasets_loader.py` | ✅ `SYSTEM_PROMPT_PS` | ✅ Ready |
| Balochi | `bal` | ⚠️ 300 items (Excel only) | ❌ Not created | ✅ `SYSTEM_PROMPT_BAL` | ⚠️ Partial |
| Arabic | `ar` | ⚠️ Available externally | ❌ Not integrated | ❌ Not created | ⚠️ Pending |
| Persian | `fa` | ⚠️ Available externally | ❌ Not integrated | ❌ Not created | ⚠️ Pending |
| Sindhi | `sd` | ⚠️ Available externally | ❌ Not integrated | ❌ Not created | ⚠️ Pending |

---

## 4. Execution Phases

### Phase 1: Dry Run Validation
```powershell
python run_punjabi.py --language pa --dry-run
python run_punjabi.py --language ur --dry-run
python run_punjabi.py --language ps --dry-run
```
Verify prompt building, no API calls.

### Phase 2: Core Languages — First Model (gpt-4o-mini)
```powershell
# Punjabi (1,553 items) — ~30 min
python run_punjabi.py --language pa --models gpt-4o-mini

# Pashto (1,320 items) — ~25 min
python run_punjabi.py --language ps --models gpt-4o-mini
```

### Phase 3: Urdu (6,825 items — use subsets)
```powershell
# Subset for faster iteration
python run_punjabi.py --language ur --models gpt-4o-mini --tasks GSM8K ARC
python run_punjabi.py --language ur --models gpt-4o-mini --tasks OpenBookQA CommonSenseQA
python run_punjabi.py --language ur --models gpt-4o-mini --tasks NameIndex MiddleMatch ScriptMixed
```

### Phase 4: Pashto Full
```powershell
python run_punjabi.py --language ps --models gpt-4o-mini
```

### Phase 5: Multi-Model Comparison
```powershell
# Run top 3 available models
python run_punjabi.py --language pa --models gpt-4o gpt-4o-mini gemini/gemini-2.0-flash
python run_punjabi.py --language ps --models gpt-4o gpt-4o-mini gemini/gemini-2.0-flash
```

### Phase 6: Analysis (each language/model combo)
```powershell
# Auto-runs after experiment, or manually:
python run_punjabi.py --analysis-only --results results_pa_20260927_183000.csv
# Or:
python analysis.py --csv results_pa_*.csv --out results/
python analysis.py --json results_pa_*.json --out results/
```

---

## 5. Execution Commands (Ready to Run)

### Single-language dry run:
```powershell
# Punjabi
python run_punjabi.py --language pa --dry-run

# Urdu  
python run_punjabi.py --language ur --dry-run

# Pashto
python run_punjabi.py --language ps --dry-run
```

### Real experiment run:
```powershell
# Fastest path — one model, one language
python run_punjabi.py --language pa --models gpt-4o-mini

# Full run — multiple models
python run_punjabi.py --language pa --models gpt-4o gpt-4o-mini gemini/gemini-2.0-flash

# Pashto
python run_punjabi.py --language ps --models gpt-4o-mini

# Urdu (subset for time efficiency)
python run_punjabi.py --language ur --models gpt-4o-mini --tasks GSM8K ARC
```

### Analysis only:
```powershell
python run_punjabi.py --analysis-only --results results_pa_YYYYMMDD_HHMMSS.csv
python analysis.py --json results_ps_YYYYMMDD_HHMMSS.json --out results/
```

---

## 6. Output Artifacts

| File | Description |
|------|-------------|
| `results_<lang>_<timestamp>.json` | Resume-safe checkpoint (saved after each item) |
| `results_<lang>_<timestamp>.csv` | Flat CSV for analysis |
| `accuracy_table.csv` | Per (model × task × scenario × method) accuracy |
| `mcnemar_results.csv` | Paired McNemar test results (p < 0.1) |
| `figure1.png` | Grouped bar chart (replicates Figure 1 of paper) |
| `win_loss_report.json` | Win/loss summary vs baseline |

---

## 7. Guardrails (Active)

| Guardrail | Description |
|-----------|-------------|
| Dataset Integrity | Protects `images/` and `datasets/` from deletion |
| RTL Script Guard | Ensures Shahmukhi script (never Gurmukhi) for Punjabi |
| API Token Guard | Prevents `.env`/tokens from being committed |
| Temp Guard | Enforces temperature=0, max_tokens=100 (non-reasoning) |
| McNemar Guard | Ensures p < 0.1, correction=False for significance |
| Progress Tracker | Logs experiment completion after each step |

---

## 8. Next Steps (External Datasets)

For Arabic, Persian (Farsi), and Sindhi, the following external datasets are available:

**Arabic:**
- `ArabicSense` — Commonsense reasoning benchmark: https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense
- `Arabic_Reasoning_QA` — Math/reasoning QA: https://huggingface.co/datasets/MohammedNasser/ARabic_Reasoning_QA
- `ArabicMMLU` — MMLU-style 14.5K MCQs: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
- `AlGhafa Arabic LLM Benchmark` — Multi-task: https://huggingface.co/datasets/OALL/AlGhafa-Arabic-LLM-Benchmark-Native
- `tawkeed-arabic-benchmark` — 970 questions: https://huggingface.co/datasets/tawkeed-sa/tawkeed-arabic-benchmark
- `juletxara/mgsm` — Multilingual GSM8K (includes Arabic): https://huggingface.co/datasets/juletxara/mgsm

**Persian/Farsi:**
- `ParsiAI/FarsTail` — Persian NLI dataset: https://huggingface.co/datasets/ParsiAI/FarsTail
- `persiannlp/parsinlu_reading_comprehension` — Reading comprehension: https://huggingface.co/datasets/persiannlp/parsinlu_reading_comprehension
- `safora/persian-scientific-qa` — Scientific QA: https://huggingface.co/datasets/safora/persian-scientific-qa
- `SLPL/naab` — Large Farsi corpus: https://huggingface.co/datasets/SLPL/naab

**Sindhi:**
- `SdQuAD` — 15,000 QA pairs: https://aclanthology.org/2026.resourceful-4.6/
- `Sindhi-Reasoning-Instruct` — Instruction following dataset: https://aclanthology.org/ (Pireh Soomro)
- `SiNFluD` — Figurative language benchmark: https://arxiv.org/abs/2605.01323
- Existing Sindhi resources: https://github.com/strickvl/awesome-balochi-nlp (includes Sindhi references)

**Balochi:**
- `Salman95s/Balochi-Multilingual-dataset` — Tens of thousands of records: https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset
- `balochiml` organization — Tokenizer + NER: https://huggingface.co/balochiml
- `awesome-balochi-nlp` — Resource repository: https://github.com/strickvl/awesome-balochi-nlp
- Kaggle: Balochi sentences dataset: https://www.kaggle.com/datasets/junaidjameel67/balochi-language-sentences-dataset

These datasets can be integrated to extend the benchmark to Arabic, Persian, Sindhi, and Balochi.
