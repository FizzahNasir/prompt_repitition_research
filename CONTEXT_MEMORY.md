# Multilingual Prompt Repetition Research: Full Context Memory
**Paper Reference:** arXiv:2512.14982 (*Leviathan, Kalman, Matias — Google Research, Dec 2025*)  
**Project Scope:** Replicating & extending prompt repetition across four Right-to-Left (RTL) low-resource Pakistani languages:
1. **Punjabi (Shahmukhi script)** (`pa`)
2. **Urdu** (`ur`)
3. **Pashto** (`ps`)
4. **Balochi** (`bal`)

---

## 1. Theoretical Foundation (arXiv:2512.14982)

### The Core Problem: Causal Attention Asymmetry
- Modern autoregressive LLMs are trained with a **causal attention mask**: token $t_i$ can attend only to preceding tokens $t_1, \dots, t_{i-1}$, never to future tokens $t_{i+1}, \dots, t_N$.
- This imposes directional bias:
  - In a `Options-First` prompt (`<OPTIONS> <QUESTION>`), the options cannot attend to the question they are answering.
  - In a `Question-First` prompt (`<QUESTION> <OPTIONS>`), the question cannot attend to the candidate options.
- The order of information profoundly alters predictive accuracy even when the semantic content is identical.

### The Mechanism: Prompt Repetition
- Instead of feeding query $Q$, the model receives repeated forms:
  - **Simple Double:** `$Q\n\n$Q`
  - **Verbose Double:** `$Q\n\n[Bridge Phrase]\n\n$Q`
  - **Triple:** `$Q\n\n[Bridge 1]\n\n$Q\n\n[Bridge 2]\n\n$Q`
- In repeated prompts, **every token in the second instance can attend to every token in the first instance**. This effectively synthesizes full bidirectional self-attention across all prompt tokens without architectural modifications.

### Key Empirical Invariants:
1. **Zero Decoding Overhead:** Repetition occurs entirely in the parallelized pre-fill (KV-cache generation) phase. Output token length and autoregressive generation latency remain unchanged.
2. **Non-Reasoning Regime:** Tested under non-reasoning settings (temperature = 0, system prompt forbidding step-by-step thinking, `max_tokens = 100`). Prevents reasoning models from spending generation tokens repeating the user's prompt.
3. **Statistical Significance:** Measured using paired **McNemar tests** ($p < 0.1$, `correction=False`). Leviathan et al. demonstrated 47 wins out of 70 test conditions with 0 losses on leading models (Gemini 2.0 Flash, GPT-4o, Claude 3.7 Sonnet, DeepSeek V3).

---

## 2. Low-Resource RTL Pakistani Languages Extension

While the original paper focused primarily on English and high-resource benchmarks, low-resource RTL languages suffer disproportionately from causal attention limitations:

### Language Profiles:
| Language | Script / Orthography | Direction | Speakers | Tokenizer Vulnerability |
|----------|----------------------|-----------|----------|-------------------------|
| **Punjabi (Shahmukhi)** (`pa`) | Perso-Arabic (Nastaliq / Naskh) | RTL | ~80M (Pakistan) | Severe over-segmentation; almost all NLP benchmarks use Indian Gurmukhi (LTR), making this dataset novel. |
| **Urdu** (`ur`) | Perso-Arabic (Nastaliq) | RTL | ~230M | 3–5 tokens per word in standard BPE tokenizers (cl100k_base / o200k_base). |
| **Pashto** (`ps`) | Extended Perso-Arabic (ټ، څ، ځ، ډ، ړ، ږ، ښ، ګ، ڼ، ې، ۍ، ئ) | RTL | ~50M | Heavy character fragmentation; high attention drift across long contexts. |
| **Balochi** (`bal`) | Perso-Arabic (Balochi standard) | RTL | ~10M | Very low web representation; subword tokenization splits words into single bytes. |

### Why Repetition Yields Larger Gains in RTL / Low-Resource Settings:
1. **Tokenizer Bloat:** Because words are split into subword fragments or bytes, prompt length (in tokens) is 2.5× to 4× longer than equivalent English queries.
2. **Attention Dilution:** Causal attention over long fragmented token sequences suffers from higher dispersion before reaching the operative question token.
3. **First Shahmukhi Reasoning Benchmark:** As documented in `important.pdf`, virtually all existing Punjabi benchmarks use Gurmukhi script (LTR). This project provides the first native Shahmukhi reasoning & retrieval benchmark suite.

---

## 3. The 7 Benchmark Tasks

| Task | Domain | Items | Scenarios | Evaluation Metric |
|------|--------|-------|-----------|-------------------|
| **MGSM / GSM8K** | Grade-School Math | 250 | `4_Question_Only` | Numeric integer extraction (exact match) |
| **ARC-Challenge** | Science MCQ (4 choices) | 300 | `2_Question_First`, `3_Options_First` | Choice letter extraction (A–D, Alif–Dal) |
| **OpenBookQA** | Science Fact MCQ (4 choices) | 495+ | `2_Question_First`, `3_Options_First` | Choice letter extraction (A–D, Alif–Dal) |
| **CommonSenseQA**| Everyday Reasoning (5 choices) | 289 | `2_Question_First`, `3_Options_First` | Choice letter extraction (A–E, Alif–Hey) |
| **NameIndex** | Long-context Retrieval (50 names) | 100 | `10_Data_First_Retrieval` | Fuzzy string match ($\ge 0.80$ similarity) |
| **MiddleMatch** | Contextual Retrieval (40 names) | 100 | `10_Data_First_Retrieval` | Fuzzy string match ($\ge 0.80$ similarity) |
| **ScriptMixed** | Code-Switching (English + RTL) | 20 | `8_English_to_RTL` | Fuzzy string match ($\ge 0.80$ similarity) |

---

## 4. Current Dataset Status & Folder Inventory

### Datasets Directory Structure:
```
datasets/
├── english/                                  ← Source English benchmarks
│   ├── arc_challenge.csv
│   ├── commonsenseqa.csv
│   ├── mgsm_english_250.csv
│   ├── mgsm_english_fewshot_8.csv
│   └── openbookqa.csv
├── punjabi/                                  ← Renamed & clean Shahmukhi files
│   ├── MGSM_Punjabi.xlsx                     (250 items)
│   ├── ARC_Punjabi.xlsx                      (300 items)
│   ├── OpenBookQA_Punjabi.xlsx               (495 items)
│   ├── CommonSenseQA_Punjabi.xlsx            (289 items)
│   ├── NameIndex_Punjabi.xlsx                (100 items)
│   ├── MiddleMatch_Punjabi.xlsx              (100 items)
│   └── ScriptMixed_Punjabi.xlsx              (20 items)
├── urdu/                                     ← Urdu translated & generated datasets
│   ├── MGSM_Urdu_GoogleTranslate.xlsx        (250 items)
│   ├── ARC_Urdu_GoogleTranslate.xlsx         (300 items)
│   ├── OpenBookQA_Urdu_merged.xlsx           (5,767 items)
│   ├── CommonSenseQA_Urdu_GoogleTranslate.xlsx (289 items)
│   ├── NameIndex_Urdu.xlsx                   (100 items)
│   ├── MiddleMatch_Urdu.xlsx                 (100 items)
│   └── ScriptMixed_Urdu.xlsx                 (20 items)
├── pashto/                                   ← (Roadmap) To be translated & generated
└── balochi/                                  ← (Roadmap) To be translated & generated
```

### Dataset Provenance & Protected Assets:
- `images/`: Contains 9 WhatsApp photographic scans of the native Lahori Punjabi handwritten translation notebook (GSM8K and ARC items). **Crucial provenance — do NOT delete.**
- `punjabi_datasets_loader.py` & `urdu_datasets_loader.py`: Unified loaders normalizing all sheets to `{id, language, task, question, options, shuffled_options, correct_answer, shuffled_correct_answer, long_data, query}` schema.

---

## 5. Repetition Prompts & Native Bridge Phrases

| Language | System Prompt (Non-Reasoning) | Bridge Phrase (Double/Verbose) | Secondary Bridge (Triple) | Native Repeat Directive |
|----------|-------------------------------|--------------------------------|---------------------------|-------------------------|
| **Punjabi (`pa`)** | `براہ راست جواب دیو۔ اپنی سوچ دی وضاحت نہ کرو۔ قدم بہ قدم نہ سوچو۔` | `میں دوبارہ آکھدا ہاں:` | `اک ہور واری:` | `مہربانی کرکے اوپر دوبارہ پڑھو تے جواب دیو:` |
| **Urdu (`ur`)** | `براہ راست جواب دیں۔ اپنی سوچ کی وضاحت نہ کریں۔ مرحلہ وار نہ سوچیں۔` | `میں دوبارہ کہتا ہوں:` | `ایک بار پھر:` | `براہ کرم اوپر والا سوال دوبارہ پڑھیں اور جواب دیں:` |
| **Pashto (`ps`)** | `مستقیم ځواب ورکړئ. خپل فکر مه تشریح کوئ. ګام په ګام مه فکر کوئ.` | `زه بیا وایم:` | `یو ځل بیا:` | `مهرباني وکړئ پورته پوښتنه بیا ولولئ او ځواب ورکړئ:` |
| **Balochi (`bal`)** | `تچک ءَ پسو بہ دئے. وتی ھیال ءَ مَہ درشان کن. گام پہ گام مَہ جیڑ.` | `من پدا گشاں:` | `یک رندے پدا:` | `مھربانی بہ کن بُرزی جست ءَ پدا بہ وان ءُ پسو بہ دئے:` |

---

## 6. Execution & Analysis Pipeline
- **Runner (`run_punjabi.py` / `experiment_runner.py`)**:
  - Supports `--language {pa,ur,ps,bal}`, `--models`, `--tasks`, `--scenarios`, `--dry-run`, `--resume`.
  - Records per-call metrics: `timestamp, model, task, scenario, method, item_id, language, correct_answer, response, is_correct, prompt_tokens, output_tokens, latency_ms`.
- **Analysis (`analysis.py`)**:
  - Accuracy grouped by `(model, task, scenario, method)`.
  - McNemar 2×2 paired statistical tests.
  - Figure 1 generation (`figure1.png`).
