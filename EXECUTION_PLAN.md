# Execution Plan: Multilingual Prompt Repetition Benchmark
**Research:** Replication of arXiv:2512.14982 for RTL / Low-resource Pakistani Languages

---

## 1. API Key Availability

| Provider | Key Name | Status | Notes |
|----------|----------|--------|-------|
| OpenAI | `OPENAI_API_KEY` | ✅ Available | Supports gpt-4o, gpt-4o-mini |
| Google | `GEMINI_API_KEY` | ✅ Available | Supports gemini-2.0-flash, gemini-2.0-flash-lite |
| Anthropic | `ANTHROPIC_API_KEY` | ❌ Not Set | Claude 3.7 Sonnet not available |
| DeepSeek | `DEEPSEEK_API_KEY` | ❌ Not Set | DeepSeek-V3 not available |
| TogetherAI | `TOGETHER_API_KEY` | ❌ Not Set | Qwen, Llama, Aya not available |
| Groq | `GROQ_API_KEY` | ❌ Not Set | Not available |

### Models We Can Run (with available keys):
1. **gpt-4o** (OpenAI)
2. **gpt-4o-mini** (OpenAI)
3. **gemini/gemini-2.0-flash** (Google)
4. **gemini/gemini-2.0-flash-lite** (Google)

### Models From Original Paper (not available):
- claude-sonnet-4-6 (Anthropic) — key not set
- deepseek/deepseek-chat (DeepSeek) — key not set

---

## 2. Execution Phases

### Phase 1: Dry Run Validation
**Goal:** Verify all loaders, prompts, and evaluation logic work without API calls.

```powershell
# Punjabi (Shahmukhi)
python run_punjabi.py --language pa --dry-run

# Urdu
python run_punjabi.py --language ur --dry-run

# Pashto
python run_punjabi.py --language ps --dry-run
```

**Expected output:** Prompt sizes printed for each method, no API calls made.

---

### Phase 2: Punjabi Full Run (Available Models)
**Goal:** Execute full benchmark for Punjabi Shahmukhi with available models.

```powershell
# Single model (fastest)
python run_punjabi.py --language pa --models gpt-4o-mini

# Multiple models
python run_punjabi.py --language pa --models gpt-4o gpt-4o-mini gemini/gemini-2.0-flash
```

**Scope:** 1,553 items × 5 methods × N scenarios × N models

**Estimated time:**
- gpt-4o-mini: ~2 hours (1,553 × 5 × 250 calls at 4.5s/call = ~30 min, but with multiple scenarios ~2h)
- gpt-4o: ~3 hours
- gemini-2.0-flash: ~3 hours

**Output files:**
- `results_pa_YYYYMMDD_HHMMSS.json` (resume-safe checkpoint)
- `results_pa_YYYYMMDD_HHMMSS.csv` (flat CSV for analysis)
- `accuracy_table.csv`
- `mcnemar_results.csv`
- `figure1.png`

---

### Phase 3: Urdu Full Run
```powershell
python run_punjabi.py --language ur --models gpt-4o-mini gemini/gemini-2.0-flash
```

**Scope:** 6,825 items × 5 methods × N scenarios × N models
**Estimated time:** ~5-6 hours per model (can use --tasks to subset)

---

### Phase 4: Pashto Full Run
```powershell
python run_punjabi.py --language ps --models gpt-4o-mini
```

**Scope:** 1,320 items × 5 methods × N scenarios × N models
**Estimated time:** ~1-2 hours per model

---

### Phase 5: Balochi (Pending)
**Status:** ⏳ Balochi datasets not yet created
- Need `balochi_datasets_loader.py`
- Need Balochi translated datasets in `datasets/balochi/`

---

## 3. Resumable Execution

All experiments are resume-safe:
```powershell
# Resume from checkpoint
python run_punjabi.py --language pa --models gpt-4o-mini --resume results_pa_20260612_114823.json
```

---

## 4. Analysis Commands

```powershell
# Analysis only (from existing results)
python run_punjabi.py --analysis-only --results results_pa_20260612_114823.csv

# Or directly
python analysis.py --json results_pa_20260612_114823.json --out results/
python analysis.py --csv results_pa_20260612_114823.csv --out results/
```

---

## 5. Execution Guardrails

The following hooks are active (`.agents/hooks.json`):

1. **Dataset Integrity Guard** — Prevents deletion of `images/` and `datasets/` files
2. **RTL Script Guard** — Ensures Shahmukhi script for Punjabi, never Gurmukhi
3. **API Token Guard** — Prevents `.env` or tokens from being committed
4. **Experiment Progress Tracker** — Auto-checks results after each command
5. **Python Validator** — Validates syntax/imports after code changes
6. **McNemar Test Guard** — Ensures p < 0.1, correction=False for significance
7. **Temp Guard** — Enforces temperature=0, max_tokens=100 (non-reasoning regime)

---

## 6. Results Analysis (Jupyter Notebook)

A `results_analysis.ipynb` notebook is provided for interactive analysis with cells for:
- Loading results (JSON/CSV)
- Computing accuracy per (model × task × scenario × method)
- McNemar statistical significance tests
- Figure 1 generation (grouped bar charts)
- Summary statistics and win/loss counts

---

## 7. Overleaf Integration

- **Project ID:** `6aaeccbdf27c3c07a1095482`
- **MCP server:** `@mjyoo2/overleaf-mcp`
- After experiments complete, push findings (accuracy tables, McNemar results, Figure 1) to Overleaf via MCP tools

---

## 8. Execution Order (Recommended)

```
1. Dry run validation (all languages)
2. Punjabi full run (gpt-4o-mini first, fastest)
3. Punjabi analysis + results notebook
4. Urdu full run (subset with --tasks GSM8K ARC first)
5. Urdu analysis
6. Pashto full run
7. Pashto analysis
8. Multi-model comparison (gpt-4o, gemini-2.0-flash)
9. Final analysis across all languages
10. Push findings to Overleaf
```
