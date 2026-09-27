"""
pashto_datasets_loader.py
Loads all 7 Pashto dataset files from datasets/pashto/ and returns a unified
list of item dicts compatible with experiment_runner.py.

Files consumed (datasets/pashto/):
  mgsm_pashto_translated.csv                 250 items (GSM8K math)
  arc_pashto.csv                              ~892 items (ARC-Challenge)
  openbookqa_pashto.csv                       ~513 items (OpenBookQA)
  COMMONSENSE PASHTO - commonsenseqa_pashto.csv ~561 items (CommonSenseQA)
  NameIndex_Pashto.xlsx                       100 items
  MiddleMatch_Pashto.xlsx                     100 items
  ScriptMixed_Pashto.xlsx                      20 items

Total: ~2,446 items (varies by exact row counts)
"""

import ast
import csv
import random
from pathlib import Path

import openpyxl

BASE = Path(__file__).parent / "datasets" / "pashto"

_ANSWER_SUFFIX_PS = "\nد یوه نوم څخه ځواب ورکړه."

_ANSWER_MAP = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
}


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
        return str(int(value))
    return str(value).strip()


def _parse_dict_str(value: str) -> dict:
    """Parse a dict-like string (from CSV) into a Python dict."""
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Handle JSON-style strings
        import json
        try:
            parsed = json.loads(value.replace("'", '"'))
            if "text" in parsed and "label" in parsed:
                return dict(zip(parsed["label"], parsed["text"]))
            return parsed
        except Exception:
            return {}


def _parse_choices(value: str) -> dict:
    """Parse choices from various formats into {A: text, B: text, ...}."""
    if isinstance(value, dict):
        return value
    try:
        parsed = _parse_dict_str(value)
        if "text" in parsed and "label" in parsed:
            labels = parsed["label"] if isinstance(parsed["label"], list) else list(parsed["label"])
            texts = parsed["text"] if isinstance(parsed["text"], list) else list(parsed["text"])
            return dict(zip(labels, texts))
        return parsed
    except Exception:
        return {}


def _load_mgsm() -> list:
    """Load MGSM Pashto translated math dataset from CSV."""
    items = []
    file_path = BASE / "mgsm_pashto_translated.csv"
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            question = row.get("pashto_question", "") or row.get("question", "")
            answer = row.get("answer_number", "")
            if not question or not answer:
                continue
            items.append({
                "id":             f"mgsm_ps_{i:03d}",
                "language":       "ps",
                "task":           "GSM8K",
                "question":       _str(question),
                "correct_answer": _str(answer),
            })
    return items


def _load_arc() -> list:
    """Load ARC Pashto dataset from CSV."""
    items = []
    file_path = BASE / "arc_pashto.csv"
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            question = row.get("pashto_question", "") or row.get("question", "")
            choices_str = row.get("pashto_choices", "") or row.get("choices", "")
            answer = _str(row.get("answerKey", "")).upper()
            if not question or not answer:
                continue
            options = _parse_choices(choices_str)
            if not options or answer not in options:
                continue
            sh_opts, sh_correct = _shuffle_options(options, answer, seed=i)
            items.append({
                "id":                      f"arc_ps_{i:03d}",
                "language":                "ps",
                "task":                    "ARC",
                "question":                _str(question),
                "options":                 options,
                "shuffled_options":        sh_opts,
                "correct_answer":          answer,
                "shuffled_correct_answer": sh_correct,
            })
    return items


def _load_openbookqa() -> list:
    """Load OpenBookQA Pashto dataset from CSV."""
    items = []
    file_path = BASE / "openbookqa_pashto.csv"
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            question = row.get("pashto_question", "") or row.get("question_stem", "")
            choices_str = row.get("pashto_choices", "") or row.get("choices", "")
            answer = _str(row.get("answerKey", "")).upper()
            if not question or not answer:
                continue
            options = _parse_choices(choices_str)
            if not options or answer not in options:
                continue
            sh_opts, sh_correct = _shuffle_options(options, answer, seed=i)
            items.append({
                "id":                      f"obqa_ps_{i:04d}",
                "language":                "ps",
                "task":                    "OpenBookQA",
                "question":                _str(question),
                "options":                 options,
                "shuffled_options":        sh_opts,
                "correct_answer":          answer,
                "shuffled_correct_answer": sh_correct,
            })
    return items


def _load_commonsenseqa() -> list:
    """Load CommonSenseQA Pashto dataset from CSV."""
    items = []
    file_paths = [
        BASE / "COMMONSENSE PASHTO - commonsenseqa_pashto.csv",
        BASE / "commonsenseqa_pashto.csv",
    ]
    file_path = None
    for fp in file_paths:
        if fp.exists():
            file_path = fp
            break
    if file_path is None:
        print(f"  WARNING: No CommonSenseQA Pashto file found in {BASE}")
        return items

    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            question = row.get("pashto_question", "") or row.get("question", "")
            choices_str = row.get("pashto_choices", "") or row.get("choices", "")
            answer = _str(row.get("answerKey", "")).upper()
            if not question or not answer:
                continue
            options = _parse_choices(choices_str)
            if not options or answer not in options:
                continue
            sh_opts, sh_correct = _shuffle_options(options, answer, seed=i)
            items.append({
                "id":                      f"csqa_ps_{i:03d}",
                "language":                "ps",
                "task":                    "CommonSenseQA",
                "question":                _str(question),
                "options":                 options,
                "shuffled_options":        sh_opts,
                "correct_answer":          answer,
                "shuffled_correct_answer": sh_correct,
            })
    return items


def _load_retrieval(filename: str, task: str, id_prefix: str) -> list:
    """Load NameIndex / MiddleMatch / ScriptMixed from Excel."""
    file_path = BASE / filename
    if not file_path.exists():
        print(f"  WARNING: {filename} not found")
        return []
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        # Expected cols: # | Long Data | Query | Correct Answer | Language | Task
        vals = [v for v in row if v is not None]
        if len(vals) < 4:
            continue
        _, long_data, query, answer = vals[0], vals[1], vals[2], vals[3]
        if not long_data or not answer:
            continue
        ld = str(long_data).strip()
        q  = str(query).strip()
        item = {
            "id":             f"{id_prefix}_{i:03d}",
            "language":       "ps",
            "task":           task,
            "long_data":      ld,
            "correct_answer": str(answer).strip(),
        }
        if task == "ScriptMixed":
            nl = ld.find("\n")
            item["english_instruction"] = ld[:nl].strip() if nl != -1 else ld
            item["rtl_content"]         = ld[nl + 1:].strip() if nl != -1 else ""
            item["question"]            = ld
            item["query"]               = q
        else:
            item["query"] = q + _ANSWER_SUFFIX_PS
        items.append(item)
    wb.close()
    return items


# ── Public API ────────────────────────────────────────────────────────────────

_FILES = [
    (_load_mgsm,          "GSM8K"),
    (_load_arc,           "ARC"),
    (_load_openbookqa,    "OpenBookQA"),
    (_load_commonsenseqa, "CommonSenseQA"),
    (lambda: _load_retrieval("NameIndex_Pashto.xlsx",   "NameIndex",   "ni_ps"),   "NameIndex"),
    (lambda: _load_retrieval("MiddleMatch_Pashto.xlsx", "MiddleMatch", "mm_ps"),   "MiddleMatch"),
    (lambda: _load_retrieval("ScriptMixed_Pashto.xlsx", "ScriptMixed", "sm_ps"),   "ScriptMixed"),
]


def build_dataset(tasks: list = None) -> list:
    """
    Load all 7 Pashto dataset files and return a unified list of item dicts.
    Pass tasks=['ARC', 'GSM8K', ...] to load only specific tasks.
    """
    dataset = []
    for loader, task_name in _FILES:
        if tasks and task_name not in tasks:
            continue
        items = loader()
        print(f"  {task_name:<15} {len(items):>4} items")
        dataset.extend(items)
    return dataset


if __name__ == "__main__":
    from collections import Counter

    print(f"Loading Pashto datasets from {BASE}\n")
    ds = build_dataset()

    counts = Counter(it["task"] for it in ds)
    print(f"\nTotal: {len(ds)} items")
    print("By task:", dict(counts))

    seen = set()
    print("\n-- Spot checks ------------------------------------------")
    for item in ds:
        key = item["task"]
        if key in seen:
            continue
        seen.add(key)
        print(f"\n  id={item['id']}  task={item['task']}")
        if "question" in item:
            print(f"  question : {item['question'][:80]}")
        if "long_data" in item and "question" not in item:
            lines = item["long_data"].splitlines()
            print(f"  long_data: {lines[0]} … ({len(lines)} lines)")
        if "query" in item:
            print(f"  query    : {item['query'][:80]}")
        if "options" in item:
            print(f"  options  : {item['options']}")
        if "english_instruction" in item:
            print(f"  en_instr : {item['english_instruction'][:60]}")
        print(f"  correct  : {item['correct_answer']}")
