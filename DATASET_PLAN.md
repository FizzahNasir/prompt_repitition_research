# Multilingual RTL Dataset Plan: Punjabi, Urdu, Pashto, Balochi, Arabic, Persian, Sindhi

**Goal:** Produce complete 7-task benchmarks for 7 low-resource Right-to-Left (RTL) languages.  
**Reference:** arXiv:2512.14982 replication and low-resource RTL extension.  
**Updated:** 2026-10-03

---

## 1. Status Overview

### Languages with Complete Datasets (✅ Ready)

| Language | Script / Orthography | Task | Items | Current File Path | Status |
|----------|----------------------|------|-------|-------------------|--------|
| **Punjabi** | Shahmukhi (RTL) | MGSM | 250 | `datasets/punjabi/MGSM_Punjabi.xlsx` | ✅ Done (Renamed & clean) |
| **Punjabi** | Shahmukhi (RTL) | ARC | 300 | `datasets/punjabi/ARC_Punjabi.xlsx` | ✅ Done (Renamed & clean) |
| **Punjabi** | Shahmukhi (RTL) | OpenBookQA | 495 | `datasets/punjabi/OpenBookQA_Punjabi.xlsx` | ✅ Done (Renamed & clean) |
| **Punjabi** | Shahmukhi (RTL) | CommonSenseQA | 288 | `datasets/punjabi/CommonSenseQA_Punjabi.xlsx` | ✅ Done (Renamed & clean) |
| **Punjabi** | Shahmukhi (RTL) | NameIndex | 100 | `datasets/punjabi/NameIndex_Punjabi.xlsx` | ✅ Done |
| **Punjabi** | Shahmukhi (RTL) | MiddleMatch | 100 | `datasets/punjabi/MiddleMatch_Punjabi.xlsx` | ✅ Done |
| **Punjabi** | Shahmukhi (RTL) | ScriptMixed | 20 | `datasets/punjabi/ScriptMixed_Punjabi.xlsx` | ✅ Done |
| **Urdu** | Nastaliq (RTL) | MGSM | 250 | `datasets/urdu/MGSM_Urdu_GoogleTranslate.xlsx` | ✅ Done |
| **Urdu** | Nastaliq (RTL) | ARC | 300 | `datasets/urdu/ARC_Urdu_GoogleTranslate.xlsx` | ✅ Done |

### Languages Requiring Dataset Acquisition (⚠️ Planned)

| Language | Script / Orthography | Task | Items | Current File Path | Status |
|----------|----------------------|------|-------|-------------------|--------|
| **Arabic** | Arabic (RTL) | All 7 Tasks | ~1,550 | `datasets/arabic/` | ⚠️ Acquisition ready |
| **Persian** | Farsi (RTL) | All 7 Tasks | ~1,550 | `datasets/persian/` | ⚠️ Acquisition ready |
| **Sindhi** | Sindhi (RTL) | All 7 Tasks | ~1,550 | `datasets/sindhi/` | ⚠️ Manual acquisition |
| **Pashto** | Extended Arabic (RTL) | 7 Tasks | ~1,550 | `datasets/pashto/` | ⏳ Planned (Phase 2A) |
| **Balochi** | Perso-Arabic (RTL) | 7 Tasks | ~1,550 | `datasets/balochi/` | ⏳ Planned (Phase 2B) |

---

## 2. Dataset Provenance & Sources

### Arabic Datasets (`datasets/arabic/`)

1. **ArabicMMLU** — https://huggingface.co/datasets/MBZUAI/ArabicMMLU
   - 14,575 Arabic MCQs from school exams (Modern Standard Arabic)
   - Native speakers from Jordan, Egypt, Lebanon, UAE, KSA
   - License: cc-by-nc-4.0 (Creative Commons Attribution-NonCommercial 4.0)
   - GitHub: https://github.com/mbzuai-nlp/ArabicMMLU
   - Paper (ACL 2024): https://arxiv.org/abs/2402.13xxx

2. **CommonsenseQA Arabic** (if available) — https://huggingface.co/datasets/arb_mlm/commonsense_qa_arabic

### Persian/Farsi Datasets (`datasets/persian/`)

1. **PersianMMLU (ammlu)** — https://huggingface.co/datasets/Hennara/ammlu
   - 1,361 Arabic MMLU questions translated to Persian by GPT-4
   - Source: AceGPT project (Cohere for AI)
   - License: Creative Commons Attribution-NonCommercial 4.0

2. **FarsTail** — https://huggingface.co/datasets/ParsiAI/FarsTail
   - 10,367 Persian NLI samples (7,266 train, 1,537 val, 1,564 test)
   - Paper: https://arxiv.org/abs/2009.08820
   - License: Apache 2.0

3. **PersianSciQA** — https://huggingface.co/datasets/safora/persian-scientific-qa
   - 39,809 Persian scientific QA pairs from RANLP 2025
   - Paper: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf

4. **Persian QA Dataset** — https://huggingface.co/datasets/mshojaei77/Persian_QA
   - 5,900 Persian QA pairs generated via GPT-4o

### Sindhi Datasets (`datasets/sindhi/`)

1. **SdQuAD** — ACL Anthology: https://aclanthology.org/2026.resourceful-4.6/
   - DOI: 10.63317/3dhhfxeoztgo
   - 15,000 QA pairs from news, history, science, geography, business, tourism
   - Published at LREC 2026 RESOURCEFUL workshop
   - Status: Contact paper authors for access (Wazir Ali, Muhammad Rafay Shaikh)

2. **SiNFluD** — https://arxiv.org/abs/2605.01323
   - 4,451 Sindhi figurative language instances (literal/non-literal)
   - Categories: idioms, similes, proverbs, metaphors
   - License: CC BY 4.0
   - GitHub: https://github.com/Sindhi-NLP/sindhi-NLP-dataset

3. **Sindhi Open Lexicon** — https://sindhilanguage.org/dataset/
   - 223,342 Sindhi word entries
   - Prepared by: Amar Fayaz Buriro (امر فياض ڻرو)

---

## 3. Punjabi & Urdu Dataset Verification

Both loaders are verified and operating:
- `punjabi_datasets_loader.py` loads **1,553 items** with fallback matching for renamed clean files.
- `urdu_datasets_loader.py` loads **6,825 items** seamlessly.

---

## 4. Pashto, Balochi, Arabic, Persian, Sindhi Dataset Construction Plan

### Phase 2A: Retrieval Task Generation (`generate_retrieval_tasks.py`)

- Curate native name pools for each language:
  - **Pashto 50-name pool** (e.g., زرغونه, ملالۍ, خوشحال, بټو, شاه زمان, تورپیکۍ)
  - **Balochi 50-name pool** (e.g., بالاچ, چاکر, ھانی, شے مرید, بیبگر, مہرلب)
  - **Arabic 50-name pool**: From ArabicMMLU demographic data
  - **Persian 50-name pool**: From Persian literature / FarsTail annotations
  - **Sindhi 50-name pool**: From SdQuAD author names / Sindhi Open Lexicon

### Phase 2B: Dataset Acquisition (Arabic, Persian, Sindhi)

**Arabic & Persian**: Use `dataset_acquisition.py` to download from HuggingFace
```bash
python dataset_acquisition.py --arabic --persian
```

**Sindhi**: Manual acquisition required
1. Contact SdQuAD authors (ACL Anthology LREC 2026)
2. Clone https://github.com/Sindhi-NLP/sindhi-NLP-dataset for SiNFluD
3. Download from https://sindhilanguage.org/dataset/ for Sindhi Open Lexicon

### Phase 2C: Question Translation (Pashto, Balochi)

- Machine translation pipeline with native speaker spot-checking.
- Output clean schemas matching `MGSM_{Language}.xlsx`, `ARC_{Language}.xlsx`, etc.

---

## 5. Dataset Loaders

All loaders follow the unified schema:
```python
{
    "id": str,               # unique identifier
    "language": str,         # ISO code (pa, ur, ps, bal, ar, fa, sd)
    "task": str,             # GSM8K, ARC, OpenBookQA, CommonSenseQA, NameIndex, MiddleMatch, ScriptMixed
    "question": str,         # question text (for MCQ/math tasks)
    "options": dict,         # {"A": ..., "B": ..., ...} (for MCQ tasks)
    "shuffled_options": dict, # shuffled version with original correct mapping
    "correct_answer": str,   # original correct answer letter
    "shuffled_correct_answer": str,  # correct answer in shuffled options
    "long_data": str,        # context/data (for retrieval tasks)
    "query": str,            # query formulation (for retrieval tasks)
}
```

| Loader | Language | Status |
|--------|----------|--------|
| `punjabi_datasets_loader.py` | pa | ✅ Complete |
| `urdu_datasets_loader.py` | ur | ✅ Complete |
| `pashto_datasets_loader.py` | ps | ✅ Complete |
| `balochi_datasets_loader.py` | bal | ⏳ Placeholder needed |
| `arabic_datasets_loader.py` | ar | ✅ Created |
| `persian_datasets_loader.py` | fa | ✅ Created |
| `sindhi_datasets_loader.py` | sd | ✅ Created |
