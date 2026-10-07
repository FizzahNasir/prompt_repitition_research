# Multilingual Prompt Repetition Research: Full Context Memory
**Paper Reference:** arXiv:2512.14982 (*Leviathan, Kalman, Matias — Google Research, Dec 2025*)  
**Project Scope:** Replicating & extending prompt repetition across 7 Right-to-Left (RTL) languages:
1. **Punjabi (Shahmukhi script)** (`pa`)
2. **Urdu** (`ur`)
3. **Pashto** (`ps`)
4. **Balochi** (`bal`)
5. **Arabic** (`ar`)
6. **Persian/Farsi** (`fa`)
7. **Sindhi** (`sd`)

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
  - **Simple Repetition:** `$Q\n\n$Q`
  - **Cross-Lingual T1:** `$Q\n\n[English Bridge]\n\n$Q` (English always used as reference)
  - **Cross-Lingual T2:** `$Q\n\n[Native Bridge]\n\n$Q` (target language bridge phrase)
- In repeated prompts, **every token in the second instance can attend to every token in the first instance**. This effectively synthesizes full bidirectional self-attention across all prompt tokens without architectural modifications.

### Key Empirical Invariants:
1. **Zero Decoding Overhead:** Repetition occurs entirely in the parallelized pre-fill (KV-cache generation) phase. Output token length and autoregressive generation latency remain unchanged.
2. **Non-Reasoning Regime:** Tested under non-reasoning settings (temperature = 0, system prompt forbidding step-by-step thinking, `max_tokens = 100`). Prevents reasoning models from spending generation tokens repeating the user's prompt.
3. **Statistical Significance:** Measured using paired **McNemar tests** ($p < 0.1$, `correction=False`).

---

## 2. Low-Resource RTL Languages Extension

While the original paper focused primarily on English and high-resource benchmarks, low-resource RTL languages suffer disproportionately from causal attention limitations.

### Language Profiles:
| Language | Script / Orthography | Direction | Speakers | Tokenizer Vulnerability |
|----------|----------------------|-----------|----------|-------------------------|
| **Punjabi (Shahmukhi)** (`pa`) | Perso-Arabic (Nastaliq / Naskh) | RTL | ~80M (Pakistan) | Severe over-segmentation; almost all NLP benchmarks use Indian Gurmukhi (LTR), making this dataset novel. |
| **Urdu** (`ur`) | Perso-Arabic (Nastaliq) | RTL | ~230M | 3–5 tokens per word in standard BPE tokenizers (cl100k_base / o200k_base). |
| **Pashto** (`ps`) | Extended Perso-Arabic (ټ، څ، ځ، ډ، ړ، ږ، ښ، ګ، ڼ، ې، ۍ، ئ) | RTL | ~50M | Heavy character fragmentation; high attention drift across long contexts. |
| **Balochi** (`bal`) | Perso-Arabic (Balochi standard) | RTL | ~10M | Very low web representation; subword tokenization splits words into single bytes. |
| **Arabic** (`ar`) | Modern Standard Arabic | RTL | ~400M (L2) | Standard tokenization; good tokenizer coverage but causal attention still applies. |
| **Persian** (`fa`) | Farsi (extended Perso-Arabic) | RTL | ~80M | Moderate tokenization; benefits from cross-lingual transfer. |
| **Sindhi** (`sd`) | Sindhi (Perso-Arabic) | RTL | ~25M | Low web representation; limited benchmark coverage. |

---

## 3. The 7 Benchmark Tasks

| Task | Domain | Items | Scenarios | Evaluation Metric |
|------|--------|-------|-----------|-------------------|
| **GSM8K / MGSM** | Grade-School Math | 250 | `4_Question_Only` | Numeric integer extraction (exact match) |
| **ARC-Challenge** | Science MCQ (4 choices) | 300 | `2_Question_First`, `3_Options_First` | Choice letter extraction (A–D, Alif–Dal) |
| **OpenBookQA** | Science Fact MCQ (4 choices) | 495 | `2_Question_First`, `3_Options_First` | Choice letter extraction (A–D, Alif–Dal) |
| **CommonSenseQA** | Everyday Reasoning (5 choices) | 289 | `2_Question_First`, `3_Options_First` | Choice letter extraction (A–E, Alif–Hey) |
| **NameIndex** | Long-context Retrieval (50 names) | 100 | `10_Data_First_Retrieval` | Exact name match (normalized spelling) |
| **MiddleMatch** | Contextual Retrieval (40 names) | 100 | `10_Data_First_Retrieval` | Exact name match (normalized spelling) |
| **ScriptMixed** | Code-Switching (English + RTL) | 20 | `8_English_to_RTL` | Exact name match (normalized spelling) |

---

## 4. The 4 Prompt Methods

| Method | Description | Prompt Format |
|--------|-------------|---------------|
| **baseline** | Single instance of the query (no repetition) | `$Q` |
| **repetition** | Same query repeated twice (in-language) | `$Q\n\n$QUERY` |
| **cross_lingual_t1** | Target language query + English bridge phrase + repeated query | `$Q\n\n[EN: "Let me repeat that:"]\n\n$QUERY` |
| **cross_lingual_t2** | Target language query + native bridge phrase + repeated query | `$Q\n\n[Native: "لنكرر ذلك:"]\n\n$QUERY` |

### Bridge Phrases by Language
| Language | Code | Native Bridge Phrase | English Reference (T1) |
|----------|------|---------------------|----------------------|
| Arabic | `ar` | `لنكرر ذلك:` | `Let me repeat that:` |
| Urdu | `ur` | `میں دوبارہ کہتا ہوں:` | `Let me repeat that:` |
| Punjabi | `pa` | `میں دوبارہ آکھدا ہاں:` | `Let me repeat that:` |
| Pashto | `ps` | `زه بیا وایم:` | `Let me repeat that:` |
| Balochi | `bal` | `من پدا گشاں:` | `Let me repeat that:` |
| Persian | `fa` | `بیا دوباره می‌گویم:` | `Let me repeat that:` |
| Sindhi | `sd` | `مه ٻيهر کيا آں:` | `Let me repeat that:` |

---

## 5. Dataset Status & Folder Inventory

### Datasets Directory Structure:
```
datasets/
├── english/                                  ← Source English benchmarks
│   ├── arc_challenge.csv
│   ├── commonsenseqa.csv
│   ├── mgsm_english_250.csv
│   ├── mgsm_english_fewshot_8.csv
│   └── openbookqa.csv
├── punjabi/                                  ← Shahmukhi translations
│   ├── MGSM_Punjabi.xlsx                     (250 items)
│   ├── ARC_Punjabi.xlsx                      (300 items)
│   ├── OpenBookQA_Punjabi.xlsx               (495 items)
│   ├── CommonSenseQA_Punjabi.xlsx            (289 items)
│   ├── NameIndex_Punjabi.xlsx                (100 items)
│   ├── MiddleMatch_Punjabi.xlsx              (100 items)
│   └── ScriptMixed_Punjabi.xlsx              (20 items)
├── urdu/
│   ├── MGSM_Urdu_GoogleTranslate.xlsx        (250 items)
│   ├── ARC_Urdu_GoogleTranslate.xlsx         (300 items)
│   ├── OpenBookQA_Urdu_merged.xlsx           (5,767 items)
│   ├── CommonSenseQA_Urdu_GoogleTranslate.xlsx (289 items)
│   ├── NameIndex_Urdu.xlsx                   (100 items)
│   ├── MiddleMatch_Urdu.xlsx                 (100 items)
│   └── ScriptMixed_Urdu.xlsx                 (20 items)
├── pashto/
│   ├── MGSM_Pashto.csv                       (250 items, translated)
│   ├── ARC_Pashto.csv                        (892 items, translated)
│   ├── OpenBookQA_Pashto.csv                 (513 items, translated)
│   ├── CommonSenseQA_Pashto.csv              (561 items, translated)
│   ├── NameIndex_Pashto.xlsx                 (100 items, generated)
│   ├── MiddleMatch_Pashto.xlsx               (100 items, generated)
│   └── ScriptMixed_Pashto.xlsx               (20 items, generated)
├── balochi/
│   ├── GSM8K_Balochi.csv                     (108 items, translated)
│   ├── ARC_Balochi.csv                       (58 items, translated)
│   ├── NameIndex_Balochi.xlsx                (100 items, generated)
│   ├── MiddleMatch_Balochi.xlsx              (100 items, generated)
│   └── ScriptMixed_Balochi.xlsx              (20 items, generated)
├── arabic/
│   ├── GSM8K_Arabic.csv                      (250 items, native)
│   ├── ARC_Arabic.csv                        (300 items, native)
│   ├── OpenBookQA_Arabic.csv                 (445 items, native)
│   ├── CommonSenseQA_Arabic.csv              (245 items, native)
│   ├── ArabicMMLU_archive.csv                (14,455 items, native)
│   ├── NameIndex_Arabic.xlsx                 (100 items, generated)
│   ├── MiddleMatch_Arabic.xlsx               (100 items, generated)
│   └── ScriptMixed_Arabic.xlsx               (20 items, generated)
├── persian/
│   ├── FarsTail_train.csv                    (7,266 items, native)
│   ├── FarsTail_test.csv                     (1,564 items, native)
│   ├── FarsTail_val.csv                      (1,537 items, native)
│   ├── PersianSciQA_full.csv                 (31,837 items, native)
│   ├── PersianSciQA_high_rel.csv             (300 items, native)
│   ├── PersianMMLU.csv                       (14,042 items, translated)
│   ├── NameIndex_Persian.xlsx                (100 items, generated)
│   ├── MiddleMatch_Persian.xlsx              (100 items, generated)
│   └── ScriptMixed_Persian.xlsx              (20 items, generated)
└── sindhi/
    ├── SindhiFactualQA_train.csv             (99 items, native)
    ├── SindhiNER_train.csv                   (10,000 items, native)
    ├── SindhiNER_validation.csv              (4,503 items, native)
    ├── SindhiNER_test.csv                    (5,628 items, native)
    ├── SindhiOpenLexicon.csv                 (223,342 words, native)
    ├── SindhiPretrain_sample.csv             (1,000 items, native)
    ├── NameIndex_Sindhi.xlsx                 (100 items, generated)
    ├── MiddleMatch_Sindhi.xlsx               (100 items, generated)
    └── ScriptMixed_Sindhi.xlsx               (20 items, generated)
```

### Translation Status Legend:
| Status | Meaning |
|--------|---------|
| ✅ Native | Original content created by native speakers or sourced from native materials |
| ⚠️ Translated | Content translated from another language to target language |
| ✅ Generated | Retrieval tasks created programmatically from curated name pools |
| ❌ BLOCKED | Blocked — requires API quota reset or manual acquisition |

### Blocked Items:
| Dataset | Reason | Status |
|---------|--------|--------|
| Balochi OpenBookQA | Translation blocked (OpenAI credits + Gemini quota exhausted) | ❌ BLOCKED |
| Balochi CommonSenseQA | Translation blocked (same as above) | ❌ BLOCKED |
| SdQuAD (15K items) | Manual acquisition required from ACL Anthology authors | ❌ BLOCKED |
| SiNFluD | Manual acquisition required from arXiv/GitHub authors | ❌ BLOCKED |

---

## 6. System Prompts (Non-Reasoning)

All system prompts instruct the model to answer directly without chain-of-thought reasoning:

| Language | Code | System Prompt |
|----------|------|---------------|
| Punjabi | `pa` | براہ راست جواب دیو۔ اپنی سوچ دی وضاحت نہ کرو۔ قدم بہ قدم نہ سوچو۔ |
| Urdu | `ur` | براہ راست جواب دیں۔ اپنی سوچ کی وضاحت نہ کریں۔ قدم بقدم مت سوچیں۔ |
| Pashto | `ps` | مستقیم ځواب ورکړئ. خپل فکر مه تشریح کوئ. گام په گام مه فکر کوئ. |
| Balochi | `bal` | تچک ءَ پسو بہ دئے. وتی ھیال ءَ مَہ درشان کن. گام پہ گام مَہ جیڑ. |
| Arabic | `ar` | أجب بإيجابية مباشرة. لا تشرح تفكيرك. لا تتخطى خطوة بخطوة. |
| Persian | `fa` | فقط به طور مستقیم پاسخ بده. فكر خود را توضيح نده. قدم به قدم فكر نكن. |
| Sindhi | `sd` | مستقیم جواب ڏيو. اپنی سوچ کی وضاحت نه کنو. قدم سان قدم تارڪ نه کنو. |

---

## 7. Execution & Analysis Pipeline
- **Runner (`run_punjabi.py` / `experiment_runner.py`)**:
  - Supports `--language {pa,ur,ps,bal,ar,fa,sd}`, `--models`, `--dry-run`, `--resume`.
  - Records per-call metrics: `timestamp, model, task, scenario, method, item_id, language, correct_answer, response, is_correct, prompt_tokens, output_tokens, latency_ms`.
- **Analysis (`analysis.py`)**:
  - Accuracy grouped by `(model, task, scenario, method)`.
  - McNemar 2×2 paired statistical tests.
  - Figure 1 generation (`figure1.png`).

---

## 8. Dataset Sources (All Verified)

1. **ArabicMMLU**: https://huggingface.co/datasets/MBZUAI/ArabicMMLU (ACL 2024, cc-by-nc-4.0)
2. **FarsTail**: https://huggingface.co/datasets/azarijafari/FarsTail (arXiv:2009.08820, Apache 2.0)
3. **PersianSciQA**: https://huggingface.co/datasets/safora/persian-scientific-qa (RANLP 2025)
4. **PersianMMLU (ammlu)**: https://huggingface.co/datasets/Hennara/ammlu (AceGPT, CC BY-NC 4.0)
5. **SindhiFactualQA**: https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA
6. **SindhiNER**: https://huggingface.co/datasets/mirfan899/sindhi-ner
7. **SindhiOpenLexicon**: https://huggingface.co/datasets/SindhiLanguageorg/Sindhi-Open-Lexicon (SindhiLanguage.org, Amar Fayaz Buriro)
8. **Sindhi Sentiment**: https://huggingface.co/datasets/Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k (2026)
9. **Balochi Multilingual**: https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset (CC-BY-SA 4.0)