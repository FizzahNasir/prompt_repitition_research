# Multilingual Prompt Repetition Research for RTL Pakistani Languages

This repository replicates and extends [arXiv:2512.14982](https://arxiv.org/abs/2512.14982) (*Leviathan et al., Dec 2025, Google Research: "Prompt Repetition Improves Non-Reasoning LLMs"*) applied to four Right-to-Left (RTL) low-resource Pakistani languages:

- **Punjabi** (Shahmukhi script) - `pa`
- **Urdu** - `ur`
- **Pashto** - `ps`
- **Balochi** - `bal`

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
- Pashto and Balochi use their respective Perso-Arabic RTL orthographies.

## Dataset Structure

```
datasets/
├── english/          # English benchmarks (MGSM, ARC, OpenBookQA, etc.)
├── urdu/             # Urdu translations
├── punjabi/          # Punjabi (Shahmukhi) translations
├── pashto/           # Pashto translations
└── balochi/          # Balochi translations
```

Punjabi datasets are named without `_GoogleTranslate` suffix (e.g., `MGSM_Punjabi.xlsx`).

## Files

- `experiment_runner.py` - Main experiment execution
- `analysis.py` / `analysis.ipynb` - Analysis and visualization
- `punjabi_datasets_loader.py` - Punjabi dataset loader
- `urdu_datasets_loader.py` - Urdu dataset loader
- `generate_retrieval_tasks.py` - Retrieval task generation
- `Prompt_Repetition_Dataset_Catalog.xlsx` - Dataset catalog
- `dataset_report.md` - Dataset documentation
- `PLAN.md`, `DATASET_PLAN.md`, `RUN_PLAN.md` - Project planning

## Data Integrity

The `images/` directory contains 9 photographic evidence images of native handwritten Lahori Punjabi notebook translations. These should not be deleted or modified.

## Setup

1. Install dependencies:
```bash
pip install openai pandas openpyxl scipy
```

2. Configure your API keys in environment variables (see `.gitignore` for excluded patterns).

## Citation

If you use this work, please cite the original paper:
> Leviathan et al. "Prompt Repetition Improves Non-Reasoning LLMs", arXiv:2512.14982, Dec 2025.