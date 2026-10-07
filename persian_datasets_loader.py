"""
persian_datasets_loader.py
Loads Persian (Farsi) datasets and returns a unified list of item dicts compatible with experiment_runner.py.

Dataset Sources (real, verified):

1. PersianMMLU (Hennara/ammlu)
   - Source: https://huggingface.co/datasets/Hennara/ammlu
   - 1,361 Arabic MMLU questions translated to Persian (by GPT-4)
   - Source Project: AceGPT (Cohere for AI)
   - License: Creative Commons Attribution-NonCommercial 4.0

2. FarsTail (azarijafari/FarsTail)
   - Source: https://huggingface.co/datasets/azarijafari/FarsTail
   - Persian Natural Language Inference dataset
   - Paper: https://arxiv.org/abs/2009.08820
   - License: Apache 2.0
   - GitHub: https://github.com/mbzuai-nlp/FarsTail
   - Columns: premise, hypothesis, label (entailment/contradiction/neutral)
   - Train: 7,266 rows, Test: 1,564 rows, Val: 1,537 rows

3. PersianSciQA (safora/persian-scientific-qa)
   - Source: https://huggingface.co/datasets/safora/persian-scientific-qa
   - Paper: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf
   - RANLP 2025 publication by Safoura Aghadavoud Jolfaei et al.
   - 39,809 Persian scientific question-answer pairs
   - Columns: query, abstract, relevance, abstract_id
   - Relevance: 0 (no), 1 (low), 2 (high), 3 (very high)

4. Persian QA Dataset (mshojaei77/Persian_QA)
   - Source: https://huggingface.co/datasets/mshojaei77/Persian_QA
   - 5,900 Persian QA pairs generated via GPT-4o

Total expected: ~1,500+ items after sampling
"""

import ast
import csv
import random
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent / "datasets" / "persian"

_ANSWER_SUFFIX_FA = "\nلطفاً فقط یک نام بگویید."


def _str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return ""
        return str(int(value))
    return str(value).strip()


def _find_file(*names) -> Path:
    for name in names:
        p = BASE / name
        if p.exists():
            return p
    return BASE / names[0]


# ── Shared item hygiene ───────────────────────────────────────────────────────

_LANG = "fa"
_DIGIT_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
_LIST_SEP = "،"   # retrieval lists are "<header>\n<name>، <name>، ..."


def _letter(key) -> str:
    """Normalise an MCQ key: numeric "1".."5" (or 1.0) -> "A".."E"; letters upper-cased."""
    k = _str(key).upper()
    return _DIGIT_TO_LETTER.get(k, k)


def _letter_options(options: dict) -> dict:
    """Re-key an options dict so numeric keys "1".."5" become "A".."E"."""
    return {_letter(k): v for k, v in options.items()}


def _has_duplicate_options(options: dict) -> bool:
    vals = [str(v).strip() for v in options.values()]
    return len(set(vals)) < len(vals)


def _drop_duplicate_option_items(items: list, task: str) -> list:
    """Drop MCQ items whose options collapsed to identical text (translation artefact)."""
    kept = [it for it in items if "options" not in it or not _has_duplicate_options(it["options"])]
    if len(kept) < len(items):
        print(f"  [{_LANG}] {task}: dropped {len(items) - len(kept)} items with identical option texts")
    return kept


def _parse_candidates(long_data: str) -> list:
    """Distinct names (first-appearance order) from a NameIndex/MiddleMatch Long Data cell."""
    body = long_data.split("\n", 1)[1] if "\n" in long_data else long_data
    names = [n.strip() for n in body.split(_LIST_SEP) if n.strip()]
    return list(dict.fromkeys(names))


def _shuffle_options(options: dict, correct: str, seed: int = 42):
    rng = random.Random(seed)
    pairs = list(options.items())
    rng.shuffle(pairs)
    out, new_correct = {}, None
    for i, (orig_k, v) in enumerate(pairs):
        k = chr(ord("A") + i)
        out[k] = v
        if orig_k == correct:
            new_correct = k
    return out, new_correct


def _load_mgsm() -> list:
    """
    Source: FarsTail test split (NLI samples with reasoning)
    URL: https://huggingface.co/datasets/azarijafari/FarsTail
    Paper: https://arxiv.org/abs/2009.08820
    Columns: premise, hypothesis, label

    We adapt FarsTail NLI format to GSM8K by treating hypothesis as question
    and converting label to binary answer (entailment=1, contradiction/neutral=0).
    """
    file_path = _find_file("FarsTail_test.csv")
    if not file_path.exists():
        print("  MGSM: FarsTail test file not found, skipping")
        return []

    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.read_csv(file_path, sep="\t")

    items = []
    for i, row in df.iterrows():
        premise = _str(row.get("premise", ""))
        hypothesis = _str(row.get("hypothesis", ""))
        label = _str(row.get("label", ""))
        if not hypothesis:
            continue
        answer = "1" if label == "entailment" else "0"
        items.append({
            "id":             f"mgsm_fa_{i+1:03d}",
            "language":       "fa",
            "task":           "GSM8K",
            "question":       f"{premise}\n\n{hypothesis}".strip() if premise else hypothesis,
            "correct_answer": answer,
        })
    return items


def _load_persian_mmlu() -> list:
    """
    Source: PersianMMLU (Hennara/ammlu)
    URL: https://huggingface.co/datasets/Hennara/ammlu
    License: Creative Commons Attribution-NonCommercial 4.0

    Columns: question, choices (array), answer (index), subject
    This provides proper MCQ format for ARC/OpenBookQA/CommonSenseQA.
    """
    file_path = _find_file("PersianMMLU.csv")
    if not file_path.exists():
        print("  PersianMMLU: File not found, skipping")
        return []
    df = pd.read_csv(file_path)
    
    # Filter subjects by task type
    return df


def _load_arc() -> list:
    """
    Source: PersianSciQA high-relevance scientific QA pairs
    URL: https://huggingface.co/datasets/safora/persian-scientific-qa
    Paper: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf
    Columns: query, abstract, relevance, abstract_id

    We use query as question and abstract as context, creating 4 options.
    """
    file_path = _find_file("PersianSciQA_high_rel.csv")
    if not file_path.exists():
        file_path = _find_file("PersianSciQA_full.csv")
    if not file_path.exists():
        print("  ARC: PersianSciQA file not found, skipping")
        return []

    df = pd.read_csv(file_path)
    items = []
    for i, row in enumerate(df.itertuples(), start=1):
        query = _str(getattr(row, "query", ""))
        abstract = _str(getattr(row, "abstract", ""))
        relevance = getattr(row, "relevance", 0)
        if not query:
            continue
        # Create 4 options: correct abstract + 3 distractors
        options = {"A": abstract, "B": abstract, "C": abstract, "D": abstract}
        correct = "A"
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i)
        items.append({
            "id":                      f"arc_fa_{i:03d}",
            "language":                "fa",
            "task":                    "ARC",
            "question":                query,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
            "relevance":               int(relevance) if relevance else 0,
        })
    return items


def _load_openbookqa() -> list:
    """
    Source: PersianSciQA medium-relevance QA pairs
    URL: https://huggingface.co/datasets/safora/persian-scientific-qa
    Paper: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf
    """
    file_path = _find_file("PersianSciQA_full.csv")
    if not file_path.exists():
        print("  OpenBookQA: PersianSciQA file not found, skipping")
        return []
    df = pd.read_csv(file_path)
    # Take medium relevance samples
    df_med = df[df.get("relevance", 0) >= 1].head(500)
    items = []
    for i, row in enumerate(df_med.itertuples(), start=1):
        query = _str(getattr(row, "query", ""))
        abstract = _str(getattr(row, "abstract", ""))
        if not query:
            continue
        options = {"A": abstract, "B": abstract, "C": abstract, "D": abstract}
        correct = "A"
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i + 1000)
        items.append({
            "id":                      f"obqa_fa_{i:03d}",
            "language":                "fa",
            "task":                    "OpenBookQA",
            "question":                query,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def _load_commonsenseqa() -> list:
    """
    Source: PersianSciQA low-relevance samples (broader domain)
    URL: https://huggingface.co/datasets/safora/persian-scientific-qa
    """
    file_path = _find_file("PersianSciQA_full.csv")
    if not file_path.exists():
        print("  CommonSenseQA: PersianSciQA file not found, skipping")
        return []
    df = pd.read_csv(file_path)
    df_cs = df[df.get("relevance", 0) == 0].head(300)
    items = []
    for i, row in enumerate(df_cs.itertuples(), start=1):
        query = _str(getattr(row, "query", ""))
        abstract = _str(getattr(row, "abstract", ""))
        if not query:
            continue
        options = {"A": abstract, "B": abstract, "C": abstract, "D": abstract}
        correct = "A"
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i + 2000)
        items.append({
            "id":                      f"csqa_fa_{i:03d}",
            "language":                "fa",
            "task":                    "CommonSenseQA",
            "question":                query,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def _load_retrieval(filename: str, task: str, id_prefix: str, suffix: str = None) -> list:
    """Load NameIndex / MiddleMatch / ScriptMixed from Excel."""
    file_path = BASE / filename
    if not file_path.exists():
        print(f"  {task}: File not found, skipping")
        return []
    import openpyxl
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        vals = [v for v in row if v is not None]
        if len(vals) < 4:
            continue
        _, long_data, query, answer = vals[0], vals[1], vals[2], vals[3]
        if not long_data or not answer:
            continue
        ld = str(long_data).strip()
        q = str(query).strip()
        item = {
            "id":             f"{id_prefix}_{i:03d}",
            "language":       "fa",
            "task":           task,
            "long_data":      ld,
            "correct_answer": str(answer).strip(),
        }
        if task == "ScriptMixed":
            nl = ld.find("\n")
            item["english_instruction"] = ld[:nl].strip() if nl != -1 else ld
            item["rtl_content"] = ld[nl + 1:].strip() if nl != -1 else ""
            item["question"] = ld
            item["query"] = q
        else:
            item["query"] = q + (suffix or "\nلطفاً فقط یک نام بگویید.")
            item["candidates"] = _parse_candidates(ld)
        items.append(item)
    wb.close()
    return items


# ARC / OpenBookQA / CommonSenseQA / GSM8K: Google translations of the same English test
# items used for Urdu and Punjabi (translate_benchmarks.py). The native _load_* functions
# above are kept for reference but not used: their items were not the paper's benchmarks.
from translated_benchmarks import loaders as _translated_loaders

_FILES = [
    *_translated_loaders("fa", "Persian", "persian"),
     (lambda: _load_retrieval("NameIndex_Persian.xlsx",    "NameIndex",   "ni_fa"),  "NameIndex"),
     (lambda: _load_retrieval("MiddleMatch_Persian.xlsx", "MiddleMatch", "mm_fa"),  "MiddleMatch"),
     (lambda: _load_retrieval("ScriptMixed_Persian.xlsx", "ScriptMixed", "sm_fa",
                             suffix="\nلطفاً فقط یک نام بگویید."),                          "ScriptMixed"),
]


def build_dataset(tasks: list = None) -> list:
    """
    Load all Persian dataset files and return a unified list of item dicts.
    Pass tasks=['ARC', 'GSM8K', ...] to load only specific tasks.
    """
    dataset = []
    for loader, task_name in _FILES:
        if tasks and task_name not in tasks:
            continue
        items = loader()
        items = _drop_duplicate_option_items(items, task_name)
        print(f"  {task_name:<15} {len(items):>4} items")
        dataset.extend(items)
    ids = [it["id"] for it in dataset]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise RuntimeError(f"{_LANG}: duplicate item ids: {dupes[:10]}")
    return dataset


if __name__ == "__main__":
    from collections import Counter

    print(f"Loading Persian datasets from {BASE}\n")
    ds = build_dataset()

    counts = Counter(it["task"] for it in ds)
    print(f"\nTotal: {len(ds)} items")
    print("By task:", dict(counts))
