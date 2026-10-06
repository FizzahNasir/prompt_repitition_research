# Colab Setup Guide for RTL Prompt Repetition Experiments

## Step-by-Step Colab Setup

### 1. Create a new Colab notebook (use GPU)
- Go to https://colab.research.google.com/
- Create new notebook
- Runtime → Change runtime type → Hardware accelerator: **T4 GPU** or **L4 GPU**

### 2. Install dependencies
Run this as the first cell:
```python
# Install required packages
!pip install -q transformers torch pandas openpyxl scipy rapidfuzz

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 3. Clone the repository
```python
# Clone repo from GitHub
!git clone https://github.com/FizzahNasir/prompt_repitition_research.git
%cd prompt_repitition_research
```

### 4. Install additional dependencies
```python
# Install local inference dependencies
!pip install -q sentencepiece  # required for some tokenizers
```

### 5. Run experiments

#### Option A: Run a single language with specific model (dry run first)
```python
# Dry run to verify prompts build correctly
!python run_all_languages.py --dry-run --language ar --model qwen2.5-3b
```

#### Option B: Run actual experiment on Arabic
```python
!python run_all_languages.py --language ar --model qwen2.5-3b --methods baseline repetition
```

#### Option C: Run with smaller model for faster iteration
```python
!python colab_run.py --language pa --model qwen2.5-1.5b --methods baseline repetition
```

#### Option D: Run all languages (sequential)
```python
# This will take a long time on a single GPU
# Better to run language-by-language
!python run_all_languages.py --language ar --model qwen2.5-3b
!python run_all_languages.py --language ur --model qwen2.5-3b
!python run_all_languages.py --language fa --model qwen2.5-3b
# etc.
```

### 6. Check results
```python
# List generated result files
!ls results_*.csv results_*.json
```

### 7. Run analysis (separate notebook or script)
```python
!python run_punjabi.py --analysis-only --results results_ar_llama3.2-3b_*.csv
```

## Available Commands

### Models:
- `qwen2.5-1.5b` - Qwen2.5 1.5B Instruct (fastest, open)
- `qwen2.5-3b` - Qwen2.5 3B Instruct (recommended starting point, open)
- `qwen2.5-7b` - Qwen2.5 7B Instruct (best quality, open)
- `mistral-7b-v0.3` - Mistral 7B Instruct v0.3 (open)
- `gemma2-2b` - Gemma 2 2B IT (open)
- `llama3.2-1b` - Llama 3.2 1B Instruct (requires HF login)
- `llama3.2-3b` - Llama 3.2 3B Instruct (requires HF login)
- `llama3.1-8b` - Llama 3.1 8B Instruct (requires HF login)

**Note:** Qwen, Mistral, and Gemma models are open-access (no HuggingFace login required). Llama models are gated and need authentication.

### Methods:
- `baseline` - Single query (no repetition)
- `repetition` - Query repeated twice
- `cross_lingual_t1` - Query + English bridge + Query
- `cross_lingual_t2` - Query + native bridge + Query

### Languages:
- `pa` - Punjabi (Shahmukhi)
- `ur` - Urdu
- `ps` - Pashto
- `bal` - Balochi (5/7 tasks - missing OpenBookQA & CommonSenseQA)
- `ar` - Arabic
- `fa` - Persian
- `sd` - Sindhi

## Expected Runtime Estimate (Colab T4 GPU)

| Model | 100 MCQ items | 250 Math items | 100 Retrieval items |
|-------|---------------|----------------|---------------------|
| Qwen2.5 1.5B | ~5 min | ~15 min | ~5 min |
| Qwen2.5 3B | ~10 min | ~30 min | ~10 min |
| Qwen2.5 7B | ~15 min | ~45 min | ~15 min |
| Mistral 7B | ~15 min | ~45 min | ~15 min |
| Gemma 2 2B | ~8 min | ~25 min | ~8 min |

Total for all 7 languages with baseline + repetition methods:
- 3B model: ~2-3 hours
- 7B model: ~3-4 hours

## Notes:
- All models are non-reasoning (instruction-tuned, not reasoning models)
- Temperature = 0 (deterministic)
- max_tokens = 100 (prevents chain-of-thought)
- Results are saved as both JSON (resume-safe) and CSV (for analysis)
- The script will cache models after first load, so subsequent languages run faster
