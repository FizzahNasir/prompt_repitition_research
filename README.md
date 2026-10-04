# Multilingual Prompt Repetition Research for RTL Languages

This repository replicates and extends [arXiv:2512.14982](https://arxiv.org/abs/2512.14982) (*Leviathan et al., Dec 2025, Google Research: "Prompt Repetition Improves Non-Reasoning LLMs"*) applied to **seven Right-to-Left (RTL) languages** using Perso-Arabic orthographies:

- **Punjabi** (Shahmukhi script) - `pa`
- **Urdu** - `ur`
- **Pashto** - `ps`
- **Balochi** - `bal`
- **Arabic** (Modern Standard) - `ar`
- **Persian/Farsi** - `fa`
- **Sindhi** - `sd`

## Research Context

Prompt repetition has been shown to improve performance in non-reasoning LLMs. This project evaluates whether similar gains hold for RTL languages with Perso-Arabic orthographies, which are underrepresented in mainstream LLM benchmarks.

## Key Parameters

- **Sampling temperature**: 0 (deterministic decoding)
- **System prompts**: Strict non-reasoning (e.g., "براہ راست جواب دیو۔ اپنی سوچ دی وضاحت نہ کرو۔")
- **max_tokens**: 100 (prevents chain-of-thought generation)
- **Statistical significance**: McNemar test ($p < 0.1$, `correction=False`)
- **Retrieval evaluation**: Fuzzy substring matching with threshold $\ge 0.80$

## Script Integrity

- Punjabi in this project uses **Shahmukhi** (Perso-Arabic, RTL) script only.
- Indian Gurmukhi script (LTR) is **not** used.
- Pashto, Balochi, Arabic, Persian, and Sindhi all use their respective Perso-Arabic rtl orthographies.

## Dataset Structure

```
datasets/
├── english/             # English benchmarks (MGSM, ARC, OpenBookQA, CommonSenseQA)
├── urdu/                # Urdu translations (Google Translate)
├── punjabi/             # Punjabi (Shahmukhi) translations (Google Translate)
├── pashto/              # Pashto translations (machine-translated)
├── balochi/             # Balochi - native + translated datasets
├── arabic/              # Arabic (native from ArabicMMLU + generated retrieval tasks)
├── persian/             # Persian (native from FarsTail/PersianMMLU + generated retrieval tasks)
└── sindhi/              # Sindhi (native from SindhiNER/OpenLexicon + generated retrieval tasks)
```

### Dataset Summary by Language

| Language | Code | GSM8K | ARC | OpenBookQA | CommonSenseQA | NameIndex | MiddleMatch | ScriptMixed | Total |
|----------|------|-------|-----|------------|---------------|-----------|-------------|-------------|-------|
| Punjabi  | pa   | 250   | 300 | 495        | 288           | 100       | 100         | 20          | 7/7 ✅ |
| Urdu     | ur   | 250   | 300 | 5767       | 288           | 100       | 100         | 20          | 7/7 ✅ |
| Pashto   | ps   | 250   | 892 | 513        | 561           | 100       | 100         | 20          | 7/7 ✅ |
| Arabic   | ar   | 250   | 300 | 445        | 245           | 100       | 100         | 20          | 7/7 ✅ |
| Persian  | fa   | 1564  | 300 | 500        | 300           | 100       | 100         | 20          | 7/7 ✅ |
| Sindhi   | sd   | 99    | 99  | 99         | 300           | 100       | 100         | 20          | 7/7 ✅ |
| Balochi  | bal  | 108   | 58  | BLOCKED    | BLOCKED       | 100       | 100         | 20          | 5/7 ⚠️ |

### Translation Status

- **Native**: Content created in the target language by native speakers/source materials
- **Translated (trans.)**: Machine-translated from English or Arabic
- **Generated (gen.)**: Programmatically generated from curated name pools

## Dataset Sources

### Arabic (ar)
- **ArabicMMLU**: [MBZUAI/ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) — 14,455 Arabic MCQs from school exams (ACL 2024, cc-by-nc-4.0)

### Persian/Farsi (fa)
- **FarsTail**: [azarijafari/FarsTail](https://huggingface.co/datasets/azarijafari/FarsTail) — 9,367 Persian entailment/contradiction pairs (arXiv:2009.08820, Apache 2.0)
- **PersianMMLU (ammlu)**: [Hennara/ammlu](https://huggingface.co/datasets/Hennara/ammlu) — 14,042 items (AceGPT, CC BY-NC 4.0)
- **PersianSciQA**: [safora/persian-scientific-qa](https://huggingface.co/datasets/safora/persian-scientific-qa) — 31,837 scientific QA pairs (RANLP 2025)

### Sindhi (sd)
- **SindhiFactualQA**: [aakashMeghwar01/Sindhi-Factual-QA](https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA) — 99 factual QA pairs
- **SindhiNER**: [mirfan899/sindhi-ner](https://huggingface.co/datasets/mirfan899/sindhi-ner) — 20,132 token-level NER annotations
- **SindhiOpenLexicon**: [SindhiLanguageorg/Sindhi-Open-Lexicon](https://huggingface.co/datasets/SindhiLanguageorg/Sindhi-Open-Lexicon) — 223,342 word dictionary (Amar Fayaz Buriro)
- **Sindhi Sentiment**: [Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k](https://huggingface.co/datasets/Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k) — 100,000 sentiment labels

### Balochi (bal)
- **Balochi Multilingual**: [Salman95s/Balochi-Multilingual-dataset](https://huggingface.co/datasets/Salman95s/Balochi-Multilingual-dataset) — 8,852 monolingual sentences + 108 Q&A pairs (CC-BY-SA 4.0)

## Files

- `experiment_runner.py` - Main experiment execution engine
- `analysis.py` / `analysis.ipynb` - Analysis and visualization
- `run_punjabi.py` - CLI entry point supporting `--language {pa|ur|ps|bal|ar|fa|sd}`
- `punjabi_datasets_loader.py` - Punjabi dataset loader
- `urdu_datasets_loader.py` - Urdu dataset loader
- `pashto_datasets_loader.py` - Pashto dataset loader
- `balochi_datasets_loader.py` - Balochi dataset loader
- `arabic_datasets_loader.py` - Arabic dataset loader
- `persian_datasets_loader.py` - Persian dataset loader
- `sindhi_datasets_loader.py` - Sindhi dataset loader
- `generate_retrieval_tasks.py` - Retrieval task generation (NameIndex, MiddleMatch, ScriptMixed)
- `dataset_acquisition.py` - Dataset download scripts for all 7 languages
- `DATASET_STATUS.md` - Complete dataset inventory with translation status
- `dataset_status_table.tex` - LaTeX table for paper inclusion
- `CONTEXT_MEMORY.md` - Unified schema and project context

## Blocked Items (Pending API Quota / Manual Acquisition)

| Dataset | Reason | Status |
|---------|--------|--------|
| Balochi OpenBookQA | Translation blocked (OpenAI credits + Gemini quota exhausted) | ⏳ Resume when API quota resets |
| Balochi CommonSenseQA | Same as above | ⏳ Resume when API quota resets |
| SdQuAD (15K items) | Manual acquisition required from ACL Anthology authors | ❌ Blocked |
| SiNFluD | Manual acquisition required from arXiv/GitHub authors | ❌ Blocked |

## Data Integrity

The `images/` directory contains 9 photographic evidence images of native handwritten Lahori Punjabi notebook translations. These should not be deleted or modified.

## Setup

1. Install dependencies:
```bash
pip install openai pandas openpyxl scipy litellm
```

2. Configure your API keys in environment variables (see `.env.example` for required keys).

3. Run experiments:
```bash
python run_punjabi.py --language ar --models gpt-4o-mini
```

## Citation

If you use this work, please cite the original paper:
> Leviathan et al. "Prompt Repetition Improves Non-Reasoning LLMs", arXiv:2512.14982, Dec 2025.
