# Multilingual RTL Prompt Repetition Benchmark — Implementation Plan
**Research context:** Replicating & extending arXiv:2512.14982 (*Leviathan et al., Dec 2025, Google Research: "Prompt Repetition Improves Non-Reasoning LLMs"*) to Right-to-Left (RTL) low-resource Pakistani languages:
1. **Punjabi (Shahmukhi script)** (`pa`)
2. **Urdu** (`ur`)
3. **Pashto** (`ps`)
4. **Balochi** (`bal`)

**Hypothesis:** Prompt repetition yields larger accuracy gains for low-resource RTL languages than for English due to severe BPE subword fragmentation (3–5 tokens/word), greater causal attention dilution, and lower pre-training representation.

---

## 1. System Architecture

```
                                  Datasets
  ┌───────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
  │  Punjabi (Shahmukhi)  │         Urdu          │        Pashto         │        Balochi        │
  │     1,553 items       │      6,825 items      │      (Roadmap)        │      (Roadmap)        │
  │ (Renamed clean xlsx)  │      (Clean xlsx)     │                       │                       │
  └───────────┬───────────┴───────────┬───────────┴───────────┬───────────┴───────────┬───────────┘
              │                       │                       │                       │
              ▼                       ▼                       ▼                       ▼
    punjabi_datasets_loader.py   urdu_datasets_loader.py   pashto_loader.py        balochi_loader.py
              └───────────────────────┼───────────────────────┴───────────────────────┘
                                      ▼
                        Unified Dataset Schema (dict)
                                      ▼
                             experiment_runner.py
                  ├─ 5 Repetition Methods (Single, Double, Verbose, Triple, Native)
                  ├─ Non-Reasoning Enforcement (temp=0, max_tokens=100)
                  ├─ Native System Prompts & Bridge Phrases
                  └─ Dual Sink: results_LANG_TIMESTAMP.json + .csv
                                      ▼
                                 analysis.py
                  ├─ Accuracy tables by (model, task, scenario, method)
                  ├─ Paired McNemar tests (p < 0.1, correction=False)
                  └─ Figure 1 Bar Charts (figure1_LANG.png)
```

---

## 2. Component Status Matrix

| Component | Status | Role & Details |
|-----------|--------|----------------|
| `CONTEXT_MEMORY.md` | ✅ Complete | Master context documentation & linguistic/theoretical reference |
| `AGENTS.md` | ✅ Complete | Persistent workspace rules enforcing RTL Shahmukhi & non-reasoning constraints |
| `.agents/hooks.json` | ✅ Complete | Antigravity lifecycle hooks for dataset protection & script validation |
| `.agents/mcp_config.json` | ✅ Complete | GitHub Model Context Protocol server configuration |
| `punjabi_datasets_loader.py` | ✅ Complete | Loads all 7 Punjabi tasks (1,553 items), supports cleaned filenames |
| `urdu_datasets_loader.py` | ✅ Complete | Loads all 7 Urdu tasks (6,825 items), supports merged & clean filenames |
| `generate_retrieval_tasks.py` | ✅ Active | Generates NameIndex, MiddleMatch, ScriptMixed for pa & ur; extending to ps & bal |
| `experiment_runner.py` | ✅ Complete | API calling, prompt transformations, latency/token logging, CSV append |
| `analysis.py` | ✅ Complete | Groupby `(model, task, scenario, method)`, McNemar testing, Figure 1 plots |
| `run_punjabi.py` | ✅ Complete | Unified CLI entrypoint supporting `--language {pa, ur}`, `--models`, `--dry-run` |

---

## 3. The 7 Benchmark Tasks

1. **MGSM / GSM8K** (Math word problems, 250 items): Scenario `4_Question_Only`. Strict integer extraction.
2. **ARC-Challenge** (Science MCQ, 300 items): Scenarios `2_Question_First`, `3_Options_First`. Choice extraction (A–D).
3. **OpenBookQA** (Fact retrieval MCQ, 495+ items): Scenarios `2_Question_First`, `3_Options_First`. Choice extraction (A–D).
4. **CommonSenseQA** (Commonsense reasoning, 289 items): Scenarios `2_Question_First`, `3_Options_First`. Choice extraction (A–E).
5. **NameIndex** (50-name list retrieval, 100 items): Scenario `10_Data_First_Retrieval`. Fuzzy match ($\ge 0.80$).
6. **MiddleMatch** (40-name list triplet retrieval, 100 items): Scenario `10_Data_First_Retrieval`. Fuzzy match ($\ge 0.80$).
7. **ScriptMixed** (Code-switching English + RTL, 20 items): Scenario `8_English_to_RTL`. Fuzzy match ($\ge 0.80$).

---

## 4. Execution Phases

### Phase 1: Punjabi & Urdu Full Runs
- Punjabi: `python run_punjabi.py --language pa --models gpt-4o-mini` (1,553 items)
- Urdu: `python run_punjabi.py --language ur --models gpt-4o-mini` (6,825 items)

### Phase 2: Pashto & Balochi Dataset Construction
- Extend `generate_retrieval_tasks.py` with Pashto & Balochi native name lists and query templates.
- Translate MGSM, ARC, OpenBookQA, and CommonSenseQA to Pashto (`ps`) and Balochi (`bal`).
- Build `pashto_datasets_loader.py` and `balochi_datasets_loader.py`.

### Phase 3: Multi-Model Evaluation
- Run top models: `gpt-4o-mini`, `gpt-4o`, `gemini/gemini-2.0-flash`, `deepseek/deepseek-chat`, `claude-3-7-sonnet`.
- Generate comparative McNemar statistical matrices and Figure 1 multi-panel plots.
