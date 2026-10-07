"""
translated_benchmarks.py — Loaders for the Google-translated benchmark files
made by translate_benchmarks.py (Arabic, Persian, Sindhi).

They translate the same English test items as the Urdu/Punjabi files
(ARC-Challenge 300, OpenBookQA test 500, CommonSenseQA 300, MGSM 250), so all
languages share parallel items. Columns are read by name:
    MCQ:  English Question | <Lang> Question | Eng A.. | <Lang> A.. | Answer | Status | Source ID ...
    MGSM: English Question | Google Translate <Lang> | Answer | ...
"""

import random
from pathlib import Path

import pandas as pd

DATASETS = Path(__file__).resolve().parent / "datasets"

_TASK_FILES = {
    "ARC":           ("ARC",           "arc"),
    "OpenBookQA":    ("OpenBookQA",    "obqa"),
    "CommonSenseQA": ("CommonSenseQA", "csqa"),
    "GSM8K":         ("MGSM",          "mgsm"),
}
_DIGIT_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _shuffle_options(options: dict, correct: str, seed: int):
    rng = random.Random(seed)
    pairs = list(options.items())
    rng.shuffle(pairs)
    out, new_correct = {}, None
    for i, (k, v) in enumerate(pairs):
        nk = chr(ord("A") + i)
        out[nk] = v
        if k == correct:
            new_correct = nk
    return out, new_correct


def load(lang: str, lang_name: str, task: str, folder: str) -> list:
    """Items for one translated benchmark, e.g. load("fa", "Persian", "ARC", "persian")."""
    stem, prefix = _TASK_FILES[task]
    path = DATASETS / folder / f"{stem}_{lang_name}_GoogleTranslate.xlsx"
    if not path.exists():
        print(f"  [{lang}] {task}: {path.name} not found, skipping")
        return []
    df = pd.read_excel(path)
    items = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        if task == "GSM8K":
            q = _s(row.get(f"Google Translate {lang_name}"))
            if not q:
                continue
            items.append({"id": f"{prefix}_{lang}_{i:03d}", "language": lang, "task": task,
                          "question": q, "correct_answer": _s(row.get("Answer"))})
            continue
        # Keep the item set parallel with Urdu/Punjabi (their files lack 12 CommonSenseQA items)
        if _s(row.get("In PaUr Set")) == "No":
            continue
        q = _s(row.get(f"{lang_name} Question"))
        options = {k: _s(row.get(f"{lang_name} {k}")) for k in "ABCDE"
                   if _s(row.get(f"{lang_name} {k}"))}
        correct = _DIGIT_TO_LETTER.get(_s(row.get("Answer")).upper(), _s(row.get("Answer")).upper())
        if not q or correct not in options:
            continue
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i)
        items.append({
            "id":                      f"{prefix}_{lang}_{i:03d}",
            "source_id":               _s(row.get("Source ID")),
            "language":                lang,
            "task":                    task,
            "question":                q,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def loaders(lang: str, lang_name: str, folder: str) -> list:
    """(loader, task) pairs for the 4 translated benchmarks, in _FILES format."""
    return [(lambda t=t: load(lang, lang_name, t, folder), t)
            for t in ("GSM8K", "ARC", "OpenBookQA", "CommonSenseQA")]
