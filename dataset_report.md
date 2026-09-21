# Multilingual RTL Dataset Report
**Date:** 2026-09-21  
**Languages Covered:** Punjabi Shahmukhi (`pa`), Urdu (`ur`), Pashto (`ps`), Balochi (`bal`)  
**Research Reference:** arXiv:2512.14982 (Leviathan et al., Dec 2025)

---

## 1. Directory Structure

```
datasets/
├── english/                                  ← Original English HuggingFace sources
│   ├── mgsm_english_250.csv
│   ├── mgsm_english_fewshot_8.csv
│   ├── arc_challenge.csv
│   ├── openbookqa.csv
│   └── commonsenseqa.csv
│
├── punjabi/                                  ← Punjabi Shahmukhi (Clean, renamed files)
│   ├── MGSM_Punjabi.xlsx                     (250 items)
│   ├── ARC_Punjabi.xlsx                      (300 items)
│   ├── OpenBookQA_Punjabi.xlsx               (495 items)
│   ├── CommonSenseQA_Punjabi.xlsx            (288 items)
│   ├── NameIndex_Punjabi.xlsx                (100 items)
│   ├── MiddleMatch_Punjabi.xlsx              (100 items)
│   └── ScriptMixed_Punjabi.xlsx              (20 items)
│
├── urdu/                                     ← Urdu datasets
│   ├── MGSM_Urdu_GoogleTranslate.xlsx        (250 items)
│   ├── ARC_Urdu_GoogleTranslate.xlsx         (300 items)
│   ├── OpenBookQA_Urdu_merged.xlsx           (5,767 items)
│   ├── CommonSenseQA_Urdu_GoogleTranslate.xlsx (288 items)
│   ├── NameIndex_Urdu.xlsx                   (100 items)
│   ├── MiddleMatch_Urdu.xlsx                 (100 items)
│   ├── ScriptMixed_Urdu.xlsx                 (20 items)
│   ├── openbookqa_urdu_train.csv             (4,957 rows)
│   ├── openbookqa_urdu_validation.csv        (500 rows)
│   └── openbookqa_urdu_test.csv              (500 rows)
│
├── pashto/                                   ← (Roadmap) Pashto datasets
└── balochi/                                  ← (Roadmap) Balochi datasets
```

---

## 2. Benchmark Summary

| Task | Type | Punjabi (Shahmukhi) | Urdu | Pashto (Target) | Balochi (Target) |
|------|------|--------------------:|-----:|----------------:|-----------------:|
| **MGSM** | Math (open-ended) | 250 | 250 | 250 | 250 |
| **ARC-Challenge** | Science MCQ (4 choices) | 300 | 300 | 300 | 300 |
| **OpenBookQA** | Science Fact MCQ (4 choices) | 495 | 5,767 | 500 | 500 |
| **CommonSenseQA**| Everyday Reasoning (5 choices)| 288 | 288 | 300 | 300 |
| **NameIndex** | Long-context Retrieval | 100 | 100 | 100 | 100 |
| **MiddleMatch** | Contextual Triplet Retrieval | 100 | 100 | 100 | 100 |
| **ScriptMixed** | Code-Switching (English + RTL) | 20 | 20 | 20 | 20 |
| **Total Usable** | | **1,553** | **6,825** | **~1,570** | **~1,570** |

---

## 3. Provenance & Preservation Note
- The folder `images/` contains 9 camera photographs of the original handwritten Lahori Punjabi translation notebook pages for MGSM and ARC items. These images serve as physical provenance for the benchmark and must be preserved.
