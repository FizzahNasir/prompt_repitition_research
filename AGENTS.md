# Workspace Rules & Research Context: Multilingual Prompt Repetition (RTL)

This repository contains the replication and extension of **arXiv:2512.14982** (*Leviathan et al., Dec 2025, Google Research: "Prompt Repetition Improves Non-Reasoning LLMs"*) applied to four Right-to-Left (RTL) low-resource Pakistani languages:
- **Punjabi (Shahmukhi script)** (`pa`)
- **Urdu** (`ur`)
- **Pashto** (`ps`)
- **Balochi** (`bal`)

## Core Directives & Memory

1. **Script Integrity (CRITICAL):**
   - Punjabi in this project is strictly **Shahmukhi** (Perso-Arabic script, Right-to-Left).
   - Under NO circumstances should Indian Gurmukhi script (Left-to-Right) be introduced, as the research targets RTL and Pakistani linguistic context.
   - Pashto and Balochi also use their respective Perso-Arabic RTL orthographies.

2. **Theoretical Regime:**
   - Prompt repetition evaluates non-reasoning LLMs.
   - Sampling temperature = 0.
   - Strict non-reasoning system prompts (e.g. `براہ راست جواب دیو۔ اپنی سوچ دی وضاحت نہ کرو۔`).
   - `max_tokens = 100` to prevent chain-of-thought generation.

3. **Data Integrity & Provenance:**
   - Do NOT delete or modify files in `images/`. These 9 images are photographic evidence of the native handwritten Lahori Punjabi notebook translations.
   - Punjabi datasets in `datasets/punjabi/` are named without `_GoogleTranslate` (`MGSM_Punjabi.xlsx`, `ARC_Punjabi.xlsx`, etc.).
   - All dataset loaders must yield items matching the unified schema defined in `CONTEXT_MEMORY.md`.

4. **Evaluation:**
   - Statistical significance is assessed with the **McNemar test** ($p < 0.1$, `correction=False`).
   - Retrieval evaluation uses fuzzy substring matching with threshold $\ge 0.80$.
