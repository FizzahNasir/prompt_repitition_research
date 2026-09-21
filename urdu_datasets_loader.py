"""
urdu_datasets_loader.py
Loads all 7 Urdu translated datasets from datasets/urdu/
and returns a unified list of item dicts compatible with experiment_runner.py.

Files consumed (datasets/urdu/):
  MGSM_Urdu_GoogleTranslate.xlsx            250 items
  ARC_Urdu_GoogleTranslate.xlsx             300 items
  OpenBookQA_Urdu_merged.xlsx             5,767 items
  CommonSenseQA_Urdu_GoogleTranslate.xlsx   289 usable items
  NameIndex_Urdu.xlsx                       100 items
  MiddleMatch_Urdu.xlsx                     100 items
  ScriptMixed_Urdu.xlsx                      20 items

Total: ~6,826 items
"""

import random
from pathlib import Path

import openpyxl

BASE = Path(__file__).parent / "datasets" / "urdu"

_ARABIC_TO_LATIN = {
    "الف": "A", "ب": "B", "ج": "C", "د": "D", "ہ": "E",
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
}

_ANSWER_SUFFIX_UR = "\nصرف ایک نام سے جواب دیں۔"


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


def _find_file(*names) -> Path:
    for name in names:
        p = BASE / name
        if p.exists():
            return p
    return BASE / names[0]


# ── File loaders ──────────────────────────────────────────────────────────────

def _load_mgsm() -> list:
    # Cols: # | English Question | Google Translate Urdu | Answer | Equation | Status
    file_path = _find_file("MGSM_Urdu.xlsx", "MGSM_Urdu_GoogleTranslate.xlsx")
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        _, _en, native_q, answer, *_ = row
        if not native_q:
            continue
        items.append({
            "id":             f"mgsm_ur_{i:03d}",
            "language":       "ur",
            "task":           "GSM8K",
            "question":       str(native_q).strip(),
            "correct_answer": _str(answer),
        })
    wb.close()
    return items


def _load_arc() -> list:
    # Cols: # | English Question | Urdu Question | Eng A-D | Urdu A-D | Answer | Status
    file_path = _find_file("ARC_Urdu.xlsx", "ARC_Urdu_GoogleTranslate.xlsx")
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        _, _en, native_q, _ea, _eb, _ec, _ed, opt_a, opt_b, opt_c, opt_d, answer, *_ = row
        if not native_q:
            continue
        options = {"A": _str(opt_a), "B": _str(opt_b), "C": _str(opt_c), "D": _str(opt_d)}
        correct = str(answer).strip().upper()
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i)
        items.append({
            "id":                      f"arc_ur_{i:03d}",
            "language":                "ur",
            "task":                    "ARC",
            "question":                str(native_q).strip(),
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    wb.close()
    return items


def _load_openbookqa() -> list:
    # Cols: # | Split | ID | English Question | Urdu Question | Eng A-D | Urdu A-D | Answer
    # Note: no Status column; compact row numbering (no gaps from skipped rows)
    file_path = _find_file("OpenBookQA_Urdu_merged.xlsx", "OpenBookQA_Urdu.xlsx")
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        _, _split, _id, _en, native_q, _ea, _eb, _ec, _ed, opt_a, opt_b, opt_c, opt_d, answer = row[:14]
        if not native_q:
            continue
        options = {"A": _str(opt_a), "B": _str(opt_b), "C": _str(opt_c), "D": _str(opt_d)}
        correct = str(answer).strip().upper()
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i)
        items.append({
            "id":                      f"obqa_ur_{i:04d}",
            "language":                "ur",
            "task":                    "OpenBookQA",
            "question":                str(native_q).strip(),
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    wb.close()
    return items


def _load_commonsenseqa() -> list:
    # Cols: # | English Question | Urdu Question | Eng A-E | Urdu A-E | Answer | Status
    file_path = _find_file("CommonSenseQA_Urdu.xlsx", "CommonSenseQA_Urdu_GoogleTranslate.xlsx")
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        _, _en, native_q, _ea, _eb, _ec, _ed, _ee, opt_a, opt_b, opt_c, opt_d, opt_e, answer, *_ = row
        if not native_q:
            continue
        options = {
            "A": _str(opt_a), "B": _str(opt_b), "C": _str(opt_c),
            "D": _str(opt_d), "E": _str(opt_e),
        }
        correct = _ARABIC_TO_LATIN.get(str(answer).strip(), str(answer).strip().upper())
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i)
        items.append({
            "id":                      f"csqa_ur_{i:03d}",
            "language":                "ur",
            "task":                    "CommonSenseQA",
            "question":                str(native_q).strip(),
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    wb.close()
    return items


def _load_retrieval(filename: str, task: str, id_prefix: str) -> list:
    # Cols: # | Long Data | Query | Correct Answer | Language | Task
    wb = openpyxl.load_workbook(BASE / filename, read_only=True, data_only=True)
    ws = wb.active
    items = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
        _, long_data, query, answer, *_ = row
        if not long_data or not answer:
            continue
        ld = str(long_data).strip()
        q  = str(query).strip()
        item = {
            "id":             f"{id_prefix}_{i:03d}",
            "language":       "ur",
            "task":           task,
            "long_data":      ld,
            "correct_answer": str(answer).strip(),
        }
        if task == "ScriptMixed":
            # Long Data is a self-contained English-only prompt; split for 8_English_to_RTL:
            # first line = English header ("Here's a list of names:")
            # rest = RTL name list + embedded question
            nl = ld.find("\n")
            item["english_instruction"] = ld[:nl].strip() if nl != -1 else ld
            item["rtl_content"]         = ld[nl + 1:].strip() if nl != -1 else ""
            item["question"]            = ld          # fallback for 4_Question_Only
            item["query"]               = q           # no suffix (prompt has "Reply with one name only.")
        else:
            item["query"] = q + _ANSWER_SUFFIX_UR
        items.append(item)
    wb.close()
    return items


# ── Public API ────────────────────────────────────────────────────────────────

_FILES = [
    (_load_mgsm,          "GSM8K"),
    (_load_arc,           "ARC"),
    (_load_openbookqa,    "OpenBookQA"),
    (_load_commonsenseqa, "CommonSenseQA"),
    (lambda: _load_retrieval("NameIndex_Urdu.xlsx",   "NameIndex",   "ni_ur"),   "NameIndex"),
    (lambda: _load_retrieval("MiddleMatch_Urdu.xlsx", "MiddleMatch", "mm_ur"),   "MiddleMatch"),
    (lambda: _load_retrieval("ScriptMixed_Urdu.xlsx", "ScriptMixed", "sm_ur"),   "ScriptMixed"),
]


def build_dataset(tasks: list = None) -> list:
    """
    Load all 7 Urdu dataset files and return a unified list of item dicts.
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

    print(f"Loading Urdu datasets from {BASE}\n")
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
