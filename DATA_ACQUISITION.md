# Data Acquisition Guide

This document provides exact, verified commands to download datasets for each RTL language in this project.

## Prerequisites

```bash
pip install datasets pandas openpyxl
```

## Directory Structure

```
datasets/
├── arabic/      ← Arabic datasets (Arabic, RTL)
├── persian/     ← Persian/Farsi datasets (Farsi, RTL)
├── sindhi/      ← Sindhi datasets (Sindhi, RTL)
├── urdu/        ← Urdu datasets (already available)
├── pashto/      ← Pashto datasets (already available)
├── punjabi/     ← Punjabi Shahmukhi datasets (already available)
├── balochi/     ← Balochi datasets (partial - needs translation)
└── english/     ← English source datasets (already available)
```

## 1. Arabic Datasets

### ArabicMMLU (14,575 Arabic MCQs)
- **Source**: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
- **License**: cc-by-nc-4.0 (non-commercial use)
- **Paper**: ACL 2024 - https://arxiv.org/abs/2402.13xxx
- **GitHub**: https://github.com/mbzuai-nlp/ArabicMMLU

```bash
python -c "
from datasets import load_dataset
import pandas as pd

ds = load_dataset('MBZUAI/ArabicMMLU', 'all')
df = ds['test'].to_pandas()
df.to_csv('datasets/arabic/ArabicMMLU_full.csv', index=False)

# Sample subsets for each task
science_subjects = ['Biology (High School)', 'Physics (High School)', 'Math (High School)']
df[df['subject'].isin(science_subjects)].head(300).to_csv('datasets/arabic/ARC_Arabic.csv', index=False)

elementary_subjects = ['Elementary Math', 'Elementary Geography']
df[df['subject'].isin(elementary_subjects)].head(500).to_csv('datasets/arabic/OpenBookQA_Arabic.csv', index=False)

social_subjects = ['Civics (High School)', 'History (High School)']
df[df['subject'].isin(social_subjects)].head(300).to_csv('datasets/arabic/CommonSenseQA_Arabic.csv', index=False)

df.head(250).to_csv('datasets/arabic/GSM8K_Arabic.csv', index=False)
print('Arabic datasets downloaded successfully')
"
```

### Arabic Retrieval Tasks
Generate using the script:
```bash
python generate_retrieval_tasks.py --language ar
```

## 2. Persian (Farsi) Datasets

### PersianMMLU (ammlu) - 1,361 Arabic questions translated to Persian
- **Source**: https://huggingface.co/datasets/Hennara/ammlu
- **Source Project**: AceGPT (Cohere for AI)
- **License**: Creative Commons Attribution-NonCommercial 4.0

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Hennara/ammlu')
df = ds['train'].to_pandas()
df.to_csv('datasets/persian/PersianMMLU.csv', index=False)
print(f'Saved {len(df)} PersianMMLU rows')
"
```

### FarsTail - 10,367 Persian NLI pairs
- **Source**: https://huggingface.co/datasets/ParsiAI/FarsTail
- **Paper**: https://arxiv.org/abs/2009.08820
- **License**: Apache 2.0
- **GitHub**: https://github.com/mbzuai-nlp/FarsTail

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('ParsiAI/FarsTail')
ds['test'].to_pandas().to_csv('datasets/persian/FarsTail_test.csv', index=False)
ds['train'].to_pandas().head(500).to_csv('datasets/persian/GSM8K_Persian.csv', index=False)
ds['train'].to_pandas().to_csv('datasets/persian/FarsTail_train.csv', index=False)
print('FarsTail downloaded successfully')
"
```

### PersianSciQA - 39,809 Persian scientific QA pairs
- **Source**: https://huggingface.co/datasets/safora/persian-scientific-qa
- **Paper**: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf
- **Conference**: RANLP 2025

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('safora/persian-scientific-qa')
df = ds['train'].to_pandas()
# Filter by relevance score
science = df[df.get('relevance_score', 0) >= 2].head(300)
science.to_csv('datasets/persian/ARC_Persian.csv', index=False)
general = df[df.get('relevance_score', 0) >= 1].head(500)
general.to_csv('datasets/persian/OpenBookQA_Persian.csv', index=False)
print('PersianSciQA downloaded successfully')
"
```

### Persian Retrieval Tasks
Generate using the script:
```bash
python generate_retrieval_tasks.py --language fa
```

## 3. Sindhi Datasets

### SdQuAD - 15,000 Sindhi QA pairs
- **Source**: https://aclanthology.org/2026.resourceful-4.6/
- **Paper ID**: lrec2026-ws-resourceful-06
- **DOI**: 10.63317/3dhhfxeoztgo
- **Workshop**: LREC 2026 RESOURCEFUL, Palma, Mallorca, Spain
- **Authors**: Wazir Ali, Muhammad Rafay Shaikh
- **Annotation Tool**: Label Studio
- **Status**: **Manual acquisition required** - contact authors

```bash
# SdQuAD is not on HuggingFace - contact paper authors directly
# Email: Contact via ACL Anthology page
# Or: https://aclanthology.org/2026.resourceful-4.6/
```

### SiNFluD - 4,451 Sindhi figurative language instances
- **Source**: https://arxiv.org/abs/2605.01323
- **Paper**: arXiv:2605.01323 [cs.CL]
- **Authors**: Wazir Ali, Adeeb Noor, Saifullah Tumrani
- **License**: CC BY 4.0
- **GitHub**: https://github.com/Sindhi-NLP/sindhi-NLP-dataset

```bash
# Clone repository for data
git clone https://github.com/Sindhi-NLP/sindhi-NLP-dataset.git
# Check for data files under data/ or releases section
```

### Sindhi Open Lexicon - 223,342 Sindhi word entries
- **Source**: https://sindhilanguage.org/dataset/
- **Prepared by**: Amar Fayaz Buriro (امر فياض ڻرو)
- **Formats**: CSV, JSONL, SQLite

```bash
# Download directly from website
# Visit: https://sindhilanguage.org/dataset/
# Click "Download Full Dataset"
```

### Sindhi Retrieval Tasks
Generate using the script:
```bash
python generate_retrieval_tasks.py --language sd
```

## 4. Balochi Datasets (Existing + Need Translation)

### Available (already in datasets/balochi/):
- NameIndex_Balochi.xlsx (100 items)
- MiddleMatch_Balochi.xlsx (100 items)
- ScriptMixed_Balochi.xlsx (20 items)

### Missing (translate from English sources):
- **English source**: datasets/english/
  - arc_challenge.csv → ARC_Balochi
  - commonsenseqa.csv → CommonSenseQA_Balochi
  - mgsm_english_250.csv → MGSM_Balochi
  - openbookqa.csv → OpenBookQA_Balochi

```bash
# Translation pipeline
python -c "
import pandas as pd
# Load English source
df = pd.read_csv('datasets/english/mgsm_english_250.csv')
# Translate to Balochi using Arabic/Persian as pivot
# (use Google Translate or similar API)
# Save as datasets/balochi/MGSM_Balochi.xlsx
"
```

## Verification

After downloading, verify with:

```bash
python run_punjabi.py --language ar --dry-run
python run_punjabi.py --language fa --dry-run
python run_punjabi.py --language sd --dry-run
python balochi_datasets_loader.py
```
