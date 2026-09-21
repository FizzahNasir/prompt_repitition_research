# Multilingual RTL Dataset Plan: Punjabi, Urdu, Pashto, Balochi

**Goal:** Produce complete 7-task benchmarks for 4 low-resource Right-to-Left (RTL) Pakistani languages.  
**Reference:** arXiv:2512.14982 replication and low-resource RTL extension.  
**Updated:** 2026-09-21  

---

## 1. Status Overview

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
| **Urdu** | Nastaliq (RTL) | OpenBookQA | 5,767 | `datasets/urdu/OpenBookQA_Urdu_merged.xlsx` | ✅ Done |
| **Urdu** | Nastaliq (RTL) | CommonSenseQA | 288 | `datasets/urdu/CommonSenseQA_Urdu_GoogleTranslate.xlsx` | ✅ Done |
| **Urdu** | Nastaliq (RTL) | NameIndex | 100 | `datasets/urdu/NameIndex_Urdu.xlsx` | ✅ Done |
| **Urdu** | Nastaliq (RTL) | MiddleMatch | 100 | `datasets/urdu/MiddleMatch_Urdu.xlsx` | ✅ Done |
| **Urdu** | Nastaliq (RTL) | ScriptMixed | 20 | `datasets/urdu/ScriptMixed_Urdu.xlsx` | ✅ Done |
| **Pashto** | Extended Arabic (RTL) | 7 Tasks | ~1,550 | `datasets/pashto/` | ⏳ Planned (Phase 2A) |
| **Balochi** | Perso-Arabic (RTL) | 7 Tasks | ~1,550 | `datasets/balochi/` | ⏳ Planned (Phase 2B) |

---

## 2. Punjabi & Urdu Dataset Verification

Both loaders are verified and operating:
- `punjabi_datasets_loader.py` loads **1,553 items** with fallback matching for renamed clean files.
- `urdu_datasets_loader.py` loads **6,825 items** seamlessly.

---

## 3. Pashto & Balochi Dataset Construction Plan

### Phase 2A: Retrieval Task Generation (`generate_retrieval_tasks.py`)
- Curate native name pools:
  - **Pashto 50-name pool** (e.g., زرغونه, ملالۍ, خوشحال, بټو, شاه زمان, تورپیکۍ)
  - **Balochi 50-name pool** (e.g., بالاچ, چاکر, ھانی, شے مرید, بیبگر, مہرلب)
- Query templates:
  - Pashto NameIndex: `په لیست کې {N}م نوم څه دی؟`
  - Pashto MiddleMatch: `په لیست کې د {A} او {B} ترمنځ کوم نوم دی؟`
  - Balochi NameIndex: `تہر بند ءَ {N}می نام چے اِنت؟`
  - Balochi MiddleMatch: `تہر بند ءَ {A} ءُ {B} ءِ نیام ءَ چے نام اِنت؟`

### Phase 2B: Question Translation (MGSM, ARC, OBQA, CSQA)
- Machine translation pipeline with native speaker spot-checking.
- Output clean schemas matching `MGSM_{Language}.xlsx`, `ARC_{Language}.xlsx`, etc.
