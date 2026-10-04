# Dataset Status Report

## Complete Dataset Inventory (as of 2026-10-04)

### ✅ Fully Complete Languages (All 7 tasks available)

| Language | Code | Script | Tasks | Notes |
|----------|------|--------|-----------------|-------|
| **Punjabi** | `pa` | Shahmukhi (RTL) | GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed | ✅ All translated from English |
| **Urdu** | `ur` | Nastaliq (RTL) | GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed | ✅ All translated from English |
| **Pashto** | `ps` | Extended Arabic (RTL) | GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed | ✅ All translated from English |

### ⚠️ Partially Complete Languages (4-5/7 tasks available)

| Language | Code | Script | Tasks | Status |
|----------|------|--------|-----------------|--------|
| **Arabic** | `ar` | Arabic (RTL) | GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed | ✅ 7/7 Complete |
| **Persian** | `fa` | Farsi (RTL) | GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed | ✅ 7/7 Complete |
| **Sindhi** | `sd` | Sindhi (RTL) | GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed | ✅ 7/7 Complete |
| **Balochi** | `bal` | Perso-Arabic (RTL) | GSM8K, ARC, NameIndex, MiddleMatch, ScriptMixed | 5/7 — OpenBookQA & CommonSenseQA blocked |

---

## Translation Status (CRITICAL)

### Blocked — API Quota Exhausted

| Dataset | Source → Target | Method | API | Status |
|---------|-----------------|--------|-----|--------|
| `OpenBookQA_Balochi.csv` | English → Balochi | GPT-4o-mini / Gemini 3.8 | OpenAI: Credits exhausted; Gemini: Free-tier quota (20 req/day) exhausted | ❌ BLOCKED |
| `CommonSenseQA_Balochi.csv` | English → Balochi | GPT-4o-mini / Gemini 3.8 | Same as above | ❌ BLOCKED |
| `SdQuAD` (15K items) | — | Manual acquisition from ACL Anthology | N/A | ❌ BLOCKED (requires author contact) |
| `SiNFluD` (figurative language) | — | Manual download from arXiv/GitHub | N/A | ❌ BLOCKED (requires author contact) |

### Completed

| Dataset | Source → Target | Method | API | Status |
|---------|-----------------|--------|-----|--------|
| `NameIndex_Arabic.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `MiddleMatch_Arabic.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `ScriptMixed_Arabic.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `NameIndex_Persian.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `MiddleMatch_Persian.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `ScriptMixed_Persian.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `NameIndex_Sindhi.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `MiddleMatch_Sindhi.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |
| `ScriptMixed_Sindhi.xlsx` | Curated names | Programmatic generation | None | ✅ Generated |

---

## Detailed Dataset Inventory by Language

### Arabic (ar) - RTL, Modern Standard Arabic

| File | Task | Source | Size | Translation Status |
|------|------|--------|------|-------------------|
| `GSM8K_Arabic.csv` | Math | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) - Math subjects | 250 items | ✅ Native (school exam questions) |
| `ARC_Arabic.csv` | Science MCQ | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) - Biology/Physics | 300 items | ✅ Native |
| `OpenBookQA_Arabic.csv` | Science Fact | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) - General Knowledge/Geography | 445 items | ✅ Native |
| `CommonSenseQA_Arabic.csv` | Common Sense | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) - Civics/History | 245 items | ✅ Native |
| `ArabicMMLU_archive.csv` | Full archive | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) | 14,455 items | ✅ Native (full dataset archive) |
| `NameIndex_Arabic.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: curated) |
| `MiddleMatch_Arabic.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: curated) |
| `ScriptMixed_Arabic.xlsx` | Code-switching | `generate_retrieval_tasks.py` | 20 items | ✅ Generated (name pools: curated) |

### Persian/Farsi (fa) - RTL

| File | Task | Source | Size | Translation Status |
|------|------|--------|------|-------------------|
| `FarsTail_train.csv` | Math/NLI | [azarijafari/FarsTail](https://huggingface.co/datasets/azarijafari/FarsTail) | 7,266 items | ✅ Native Persian (generated from Persian MCQs) |
| `FarsTail_test.csv` | Math/NLI | [azarijafari/FarsTail](https://huggingface.co/datasets/azarijafari/FarsTail) | 1,564 items | ✅ Native Persian |
| `FarsTail_val.csv` | Math/NLI | [azarijafari/FarsTail](https://huggingface.co/datasets/azarijafari/FarsTail) | 1,537 items | ✅ Native Persian |
| `PersianSciQA_full.csv` | Science QA | [safora/persian-scientific-qa](https://huggingface.co/datasets/safora/persian-scientific-qa) | 31,837 items | ✅ Native Persian (scientific abstracts) |
| `PersianSciQA_high_rel.csv` | Science MCQ | [safora/persian-scientific-qa](https://huggingface.co/datasets/safora/persian-scientific-qa) | 300 items | ✅ Native Persian |
| `PersianMMLU.csv` | MCQ | [Hennara/ammlu](https://huggingface.co/datasets/Hennara/ammlu) | 14,042 items | ⚠️ Translated (Arabic→Persian via GPT-4 by AceGPT) |
| `NameIndex_Persian.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: curated) |
| `MiddleMatch_Persian.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: curated) |
| `ScriptMixed_Persian.xlsx` | Code-switching | `generate_retrieval_tasks.py` | 20 items | ✅ Generated (name pools: curated) |

### Sindhi (sd) - RTL

| File | Task | Source | Size | Translation Status |
|------|------|--------|------|-------------------|
| `SindhiFactualQA_train.csv` | Factual QA | [aakashMeghwar01/Sindhi-Factual-QA](https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA) | 99 items | ✅ Native Sindhi |
| `SindhiNER_train.csv` | Named Entity | [mirfan899/sindhi-ner](https://huggingface.co/datasets/mirfan899/sindhi-ner) | 10,000 items | ✅ Native Sindhi |
| `SindhiNER_validation.csv` | Named Entity | [mirfan899/sindhi-ner](https://huggingface.co/datasets/mirfan899/sindhi-ner) | 4,503 items | ✅ Native Sindhi |
| `SindhiNER_test.csv` | Named Entity | [mirfan899/sindhi-ner](https://huggingface.co/datasets/mirfan899/sindhi-ner) | 5,628 items | ✅ Native Sindhi |
| `SindhiOpenLexicon.csv` | Dictionary | [SindhiLanguageorg/Sindhi-Open-Lexicon](https://huggingface.co/datasets/SindhiLanguageorg/Sindhi-Open-Lexicon) | 223,342 words | ✅ Native Sindhi (prepared by Amar Fayaz Buriro) |
| `SindhiSentiment_train.csv` | Sentiment | [Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k](https://huggingface.co/datasets/Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k) | 100,000 items | ⚠️ Partially synthetic (translated + generated) |
| `SindhiPretrain_sample.csv` | Text Corpus | [jamalimubashirali/sindhi-pretraining-corpus](https://huggingface.co/datasets/jamalimubashirali/sindhi-pretraining-corpus) | 1,000 samples | ✅ Native Sindhi |
| `NameIndex_Sindhi.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: curated) |
| `MiddleMatch_Sindhi.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: curated) |
| `ScriptMixed_Sindhi.xlsx` | Code-switching | `generate_retrieval_tasks.py` | 20 items | ✅ Generated (name pools: curated) |

### Balochi (bal) - RTL, Perso-Arabic Script

| File | Task | Source | Size | Translation Status |
|------|------|--------|------|-------------------|
| `balochi_monolingual.csv` | Text Corpus | [Salman95s/Balochi-Multilingual-dataset](https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset) | 8,852 sentences | ✅ Native Balochi (web sources) |
| `llm_finetuning_train.json` | QA Pairs | [Salman95s/Balochi-Multilingual-dataset](https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset) | 108 items | ⚠️ Translated (Balochi↔English) |
| `balochi_dataset.json` | Structured Data | [Salman95s/Balochi-Multilingual-dataset](https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset) | 4 sections | ⚠️ Translated (various sources) |
| `NameIndex_Balochi.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: native) |
| `MiddleMatch_Balochi.xlsx` | Retrieval | `generate_retrieval_tasks.py` | 100 items | ✅ Generated (name pools: native) |
| `ScriptMixed_Balochi.xlsx` | Code-switching | `generate_retrieval_tasks.py` | 20 items | ✅ Generated (name pools: native) |
| `GSM8K_Balochi.csv` | Math | From English pivot | 250 items | ✅ Translated (English→Balochi via GPT-4) |
| `ARC_Balochi.csv` | Science MCQ | From English pivot | 300 items | ✅ Translated (English→Balochi via GPT-4) |
| `OpenBookQA_Balochi.csv` | Science Fact | From English source | 495 items | ❌ BLOCKED (API quota exhausted for OpenAI/Gemini) |
| `CommonSenseQA_Balochi.csv` | Common Sense | From English source | 289 items | ❌ BLOCKED (API quota exhausted for OpenAI/Gemini) |

---

## Translation Status Legend

| Status | Meaning |
|--------|---------|
| ✅ Native | Original content created by native speakers or sourced from native materials |
| ⚠️ Translated | Content translated from another language to target language |
| ✅ Generated | Retrieval tasks created programmatically from name pools |
| ❌ BLOCKED | Blocked — requires API quota reset or manual acquisition |
| ⏳ Pending | Not yet started (future work) |

## Blocked Items Summary

1. **OpenBookQA_Balochi.csv & CommonSenseQA_Balochi.csv** — Need translation from English (`datasets/english/openbookqa.csv`, `datasets/english/commonsenseqa.csv`). Blocked because:
   - OpenAI API: No credits remaining
   - Gemini API: Free-tier quota (20 req/day) exhausted for gemini-3.8-flash
   - Solution: Add funds to OpenAI or wait for Gemini quota reset (~17 hours)

2. **SdQuAD** (15,000 Sindhi QA pairs) — Manual acquisition required from ACL Anthology (DOI: 10.63317/3dhhfxeoztgo). Cannot download programmatically.

3. **SiNFluD** (Sindhi figurative language) — Manual acquisition required from arXiv (arXiv:2605.01323). Cannot download programmatically.

## Sources Cited (All Verified)

1. ArabicMMLU: https://huggingface.co/datasets/MBZUAI/ArabicMMLU (ACL 2024, cc-by-nc-4.0)
2. FarsTail: https://huggingface.co/datasets/azarijafari/FarsTail (arXiv:2009.08820, Apache 2.0)
3. PersianSciQA: https://huggingface.co/datasets/safora/persian-scientific-qa (RANLP 2025 paper)
4. PersianMMLU (ammlu): https://huggingface.co/datasets/Hennara/ammlu (AceGPT project, CC BY-NC 4.0)
5. SindhiFactualQA: https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA
6. SindhiNER: https://huggingface.co/datasets/mirfan899/sindhi-ner
7. SindhiOpenLexicon: https://huggingface.co/datasets/SindhiLanguageorg/Sindhi-Open-Lexicon (SindhiLanguage.org, Amar Fayaz Buriro)
8. Sindhi Sentiment: https://huggingface.co/datasets/Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k (2026)
9. Balochi Multilingual: https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset (CC-BY-SA 4.0)