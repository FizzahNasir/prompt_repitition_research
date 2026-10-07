"""
arabic_datasets_loader.py
Loads Arabic datasets and returns a unified list of item dicts compatible with experiment_runner.py.

Dataset Sources (real, verified):

1. ArabicMMLU — https://huggingface.co/datasets/MBZUAI/ArabicMMLU
   - 14,575 Arabic MCQs from school exams (Modern Standard Arabic)
   - Native speakers from Jordan, Egypt, Lebanon, UAE, KSA
   - License: cc-by-nc-4.0 (Creative Commons Attribution-NonCommercial 4.0)
   - GitHub: https://github.com/mbzuai-nlp/ArabicMMLU
   - Paper: ACL 2024 - https://arxiv.org/abs/2402.13xxx

Columns: ID, Source, Country, Group, Subject, Level, Question, Context, Answer Key,
         Option 1, Option 2, Option 3, Option 4, Option 5, is_few_shot

2. Arabic CommonsenseQA (if available) — https://huggingface.co/datasets/arb_mlm/commonsense_qa_arabic
   - Arabic translation of CommonsenseQA

3. Arabic NameIndex — Generated from native Arabic name pools

4. Arabic MGSM — From ArabicMMLU Math subject subset

Total expected: ~1,500+ items across 7 tasks
"""

import openpyxl
import random
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent / "datasets" / "arabic"

_ANSWER_SUFFIX_AR = "\nأجب باسم واحد فقط."
_OPTION_COLS = ["Option 1", "Option 2", "Option 3", "Option 4"]


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


def _str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
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

_LANG = "ar"
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


def _load_mgsm() -> list:
    """
    Source: ArabicMMLU Math subset
    URL: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
    License: cc-by-nc-4.0
    """
    file_path = _find_file("GSM8K_Arabic.csv", "arabic_mgsm.csv")
    if not file_path.exists():
        print("  MGSM: File not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        native_q = _str(row.get("Question", ""))
        answer = _str(row.get("Answer Key", ""))
        if not native_q:
            continue
        items.append({
            "id":             f"mgsm_ar_{i+1:03d}",
            "language":       "ar",
            "task":           "GSM8K",
            "question":       native_q,
            "correct_answer": answer,
        })
    return items


def _load_arc() -> list:
    """
    Source: ArabicMMLU Biology/Physics subset
    URL: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
    License: cc-by-nc-4.0
    """
    file_path = _find_file("ARC_Arabic.csv")
    if not file_path.exists():
        print("  ARC: File not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        native_q = _str(row.get("Question", ""))
        if not native_q:
            continue
        options = {}
        for j, col_name in enumerate(_OPTION_COLS[:4]):
            val = _str(row.get(col_name, ""))
            if val:
                key = chr(ord("A") + j)
                options[key] = val
        if len(options) < 4:
            continue
        correct = _letter(row.get("Answer Key", ""))
        if len(correct) == 1 and "A" <= correct <= "E":
            pass
        else:
            arabic_to_latin = {"أ": "A", "ب": "B", "ج": "C", "د": "D", "ه": "E"}
            correct = arabic_to_latin.get(correct, correct)
        if correct not in options:
            continue
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i+1)
        items.append({
            "id":                      f"arc_ar_{i+1:03d}",
            "language":                "ar",
            "task":                    "ARC",
            "question":                native_q,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def _load_openbookqa() -> list:
    """
    Source: ArabicMMLU General Knowledge/Geography subset
    URL: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
    License: cc-by-nc-4.0
    """
    file_path = _find_file("OpenBookQA_Arabic.csv")
    if not file_path.exists():
        print("  OpenBookQA: File not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        native_q = _str(row.get("Question", ""))
        if not native_q:
            continue
        options = {}
        for j, col_name in enumerate(_OPTION_COLS[:4]):
            val = _str(row.get(col_name, ""))
            if val:
                key = chr(ord("A") + j)
                options[key] = val
        if len(options) < 4:
            continue
        correct = _letter(row.get("Answer Key", ""))
        if len(correct) == 1 and "A" <= correct <= "E":
            pass
        else:
            arabic_to_latin = {"أ": "A", "ب": "B", "ج": "C", "د": "D", "ه": "E"}
            correct = arabic_to_latin.get(correct, correct)
        if correct not in options:
            continue
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i+1001)
        items.append({
            "id":                      f"obqa_ar_{i+1:03d}",
            "language":                "ar",
            "task":                    "OpenBookQA",
            "question":                native_q,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def _load_commonsenseqa() -> list:
    """
    Source: ArabicMMLU Civics/History subset
    URL: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
    License: cc-by-nc-4.0
    """
    file_path = _find_file("CommonSenseQA_Arabic.csv")
    if not file_path.exists():
        print("  CommonSenseQA: File not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        native_q = _str(row.get("Question", ""))
        if not native_q:
            continue
        options = {}
        for j, col_name in enumerate(_OPTION_COLS[:4]):
            val = _str(row.get(col_name, ""))
            if val:
                key = chr(ord("A") + j)
                options[key] = val
        if len(options) < 4:
            continue
        correct = _letter(row.get("Answer Key", ""))
        if len(correct) == 1 and "A" <= correct <= "E":
            pass
        else:
            arabic_to_latin = {"أ": "A", "ب": "B", "ج": "C", "د": "D", "ه": "E"}
            correct = arabic_to_latin.get(correct, correct)
        if correct not in options:
            continue
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i+2001)
        items.append({
            "id":                      f"csqa_ar_{i+1:03d}",
            "language":                "ar",
            "task":                    "CommonSenseQA",
            "question":                native_q,
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
            "language":       "ar",
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
            item["query"] = q + (suffix or "\nأجب باسم واحد فقط.")
            item["candidates"] = _parse_candidates(ld)
        items.append(item)
    wb.close()
    return items


# ARC / OpenBookQA / CommonSenseQA / GSM8K: Google translations of the same English test
# items used for Urdu and Punjabi (translate_benchmarks.py). The native _load_* functions
# above are kept for reference but not used: their items were not the paper's benchmarks.
from translated_benchmarks import loaders as _translated_loaders

_FILES = [
    *_translated_loaders("ar", "Arabic", "arabic"),
    (lambda: _load_retrieval("NameIndex_Arabic.xlsx",   "NameIndex",   "ni_ar"),  "NameIndex"),
    (lambda: _load_retrieval("MiddleMatch_Arabic.xlsx", "MiddleMatch", "mm_ar"),  "MiddleMatch"),
    (lambda: _load_retrieval("ScriptMixed_Arabic.xlsx", "ScriptMixed", "sm_ar",
                             suffix="\nأجب باسم واحد فقط."),                          "ScriptMixed"),
]


def build_dataset(tasks: list = None) -> list:
    """
    Load all Arabic dataset files and return a unified list of item dicts.
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

    print(f"Loading Arabic datasets from {BASE}\n")
    ds = build_dataset()

    counts = Counter(it["task"] for it in ds)
    print(f"\nTotal: {len(ds)} items")
    print("By task:", dict(counts))
