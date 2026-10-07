"""
sindhi_datasets_loader.py
Loads Sindhi datasets and returns a unified list of item dicts compatible with experiment_runner.py.

Dataset Sources (real, verified, downloaded from HuggingFace):

1. SindhiFactualQA — https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA
   - 99 Sindhi factual question-answer pairs
   - Columns: instruction (question), output (answer), text (full context)
   - Questions cover: geography, history, culture, cities, rivers of Sindh

2. SindhiNER — https://huggingface.co/datasets/mirfan899/sindhi-ner
   - Sindhi Named Entity Recognition dataset
   - Columns: id, tokens, ner_tags
   - Train: 18,009 rows, Validation: 4,503, Test: 5,628
   - NER tags: 0=padding, 3=location, 7=person, 8=organization
   - Can extract person/location names for retrieval tasks

3. Sindhi Sentiment Analysis — https://huggingface.co/datasets/Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k
   - 100,000 Sindhi sentences with sentiment labels
   - Columns: Text, Label (positive/negative/neutral)

4. Sindhi Open Lexicon — https://huggingface.co/datasets/SindhiLanguageorg/Sindhi-Open-Lexicon
   - 223,342 Sindhi word entries
   - Columns: lexical_id, entry_id, word, part_of_speech, domain, definition, source_dictionary
   - Sources: Jama Turshe, Official Terms, Mewaram Dictionary, PanLex, etc.
   - License: Research use with attribution

5. SdQuAD — ACL Anthology: https://aclanthology.org/2026.resourceful-4.6/
   - DOI: 10.63317/3dhhfxeoztgo
   - 15,000 QA pairs from news, history, science, geography, business, tourism
   - Status: Paper accepted but data not yet on HuggingFace - requires manual acquisition
   - Authors: Wazir Ali, Muhammad Rafay Shaikh (contact via ACL Anthology)

6. SiNFluD — https://arxiv.org/abs/2605.01323
   - arXiv:2605.01323 [cs.CL]
   - 4,451 Sindhi figurative language instances
   - GitHub: https://github.com/Sindhi-NLP/sindhi-NLP-dataset
   - License: CC BY 4.0 (requires manual download)

7. Sindhi Pretraining Corpus — https://huggingface.co/datasets/jamalimubashirali/sindhi-pretraining-corpus
   - Large Sindhi text corpus from multiple sources
   - Columns: text, source

Total expected: ~500+ items from available datasets
"""

import ast
import json
import random
from pathlib import Path

import openpyxl
import pandas as pd

BASE = Path(__file__).parent / "datasets" / "sindhi"

_ANSWER_SUFFIX_SD = "\nرڳو هڪ نالو لکي جواب ڏيو."


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

_LANG = "sd"
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


def _extract_ner_names(df, ner_column="ner_tags", token_column="tokens", n_samples=100):
    """Extract named entities (persons, locations, organizations) from NER data."""
    names = set()
    tag_to_type = {3: "location", 7: "person", 8: "organization"}
    for i, row in df.iterrows():
        if i >= n_samples:
            break
        tokens_str = _str(row.get(token_column, ""))
        ner_str = _str(row.get(ner_column, ""))
        if not tokens_str or not ner_str:
            continue
        try:
            tokens = ast.literal_eval(tokens_str) if tokens_str.startswith("[") else tokens_str.split()
            ner_tags = ast.literal_eval(ner_str) if ner_str.startswith("[") else []
            if isinstance(ner_tags, list) and len(ner_tags) == len(tokens):
                for j, tag in enumerate(ner_tags):
                    if tag in tag_to_type:
                        names.add(tokens[j])
        except Exception:
            continue
    return list(names)


def _load_mgsm() -> list:
    """
    Source: SindhiFactualQA (factual Q&A pairs)
    URL: https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA

    Columns: instruction (question), output (answer)
    """
    file_path = _find_file("SindhiFactualQA_train.csv")
    if not file_path.exists():
        print("  MGSM: SindhiFactualQA not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        question = _str(row.get("instruction", ""))
        answer = _str(row.get("output", ""))
        if not question or not answer:
            continue
        items.append({
            "id":             f"mgsm_sd_{i+1:03d}",
            "language":       "sd",
            "task":           "GSM8K",
            "question":       question,
            "correct_answer": answer,
        })
    return items


def _load_arc() -> list:
    """
    Source: SindhiFactualQA (factual Q&A pairs adapted to MCQ)
    URL: https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA
    """
    file_path = _find_file("SindhiFactualQA_train.csv")
    if not file_path.exists():
        print("  ARC: SindhiFactualQA not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        question = _str(row.get("instruction", ""))
        answer = _str(row.get("output", ""))
        if not question:
            continue
        options = {"A": answer, "B": answer, "C": answer, "D": answer}
        correct = "A"
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i+1)
        items.append({
            "id":                      f"arc_sd_{i+1:03d}",
            "language":                "sd",
            "task":                    "ARC",
            "question":                question,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def _load_openbookqa() -> list:
    """
    Source: SindhiFactualQA (general knowledge questions)
    URL: https://huggingface.co/datasets/aakashMeghwar01/Sindhi-Factual-QA
    """
    file_path = _find_file("SindhiFactualQA_train.csv")
    if not file_path.exists():
        print("  OpenBookQA: SindhiFactualQA not found, skipping")
        return []
    df = pd.read_csv(file_path)
    items = []
    for i, row in df.iterrows():
        question = _str(row.get("instruction", ""))
        answer = _str(row.get("output", ""))
        if not question:
            continue
        options = {"A": answer, "B": answer, "C": answer, "D": answer}
        correct = "A"
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i+1001)
        items.append({
            "id":                      f"obqa_sd_{i+1:03d}",
            "language":                "sd",
            "task":                    "OpenBookQA",
            "question":                question,
            "options":                 options,
            "shuffled_options":        sh_opts,
            "correct_answer":          correct,
            "shuffled_correct_answer": sh_correct,
        })
    return items


def _load_commonsenseqa() -> list:
    """
    Source: Sindhi Sentiment Analysis (adapted to common sense questions)
    URL: https://huggingface.co/datasets/Mahnoornaz/Sindhi-sentiment-analysis-dataset-100k
    """
    file_path = _find_file("SindhiSentiment_train.csv")
    if not file_path.exists():
        print("  CommonSenseQA: SindhiSentiment not found, skipping")
        return []
    df = pd.read_csv(file_path, nrows=300)
    items = []
    for i, row in df.iterrows():
        text = _str(row.get("Text", ""))
        label = _str(row.get("Label", ""))
        if not text:
            continue
        question = f"اهـ کــن تـحريـك دا طرز کي چئـن کـِ رهـائش ڪيو؟"
        options = {"A": label, "B": label, "C": label, "D": label}
        correct = "A"
        sh_opts, sh_correct = _shuffle_options(options, correct, seed=i+2001)
        items.append({
            "id":                      f"csqa_sd_{i+1:03d}",
            "language":                "sd",
            "task":                    "CommonSenseQA",
            "question":                question,
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
            "language":       "sd",
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
            item["query"] = q + (suffix or _ANSWER_SUFFIX_SD)
            item["candidates"] = _parse_candidates(ld)
        items.append(item)
    wb.close()
    return items


# ARC / OpenBookQA / CommonSenseQA / GSM8K: Google translations of the same English test
# items used for Urdu and Punjabi (translate_benchmarks.py). The native _load_* functions
# above are kept for reference but not used: their items were not the paper's benchmarks.
from translated_benchmarks import loaders as _translated_loaders

_FILES = [
    *_translated_loaders("sd", "Sindhi", "sindhi"),
    (lambda: _load_retrieval("NameIndex_Sindhi.xlsx",   "NameIndex",   "ni_sd"),  "NameIndex"),
    (lambda: _load_retrieval("MiddleMatch_Sindhi.xlsx", "MiddleMatch", "mm_sd"),  "MiddleMatch"),
    (lambda: _load_retrieval("ScriptMixed_Sindhi.xlsx", "ScriptMixed", "sm_sd",
                             suffix="\nرڳو هڪ نالو لکي جواب ڏيو."),                          "ScriptMixed"),
]


def build_dataset(tasks: list = None) -> list:
    """
    Load all Sindhi dataset files and return a unified list of item dicts.
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

    print(f"Loading Sindhi datasets from {BASE}\n")
    ds = build_dataset()

    counts = Counter(it["task"] for it in ds)
    print(f"\nTotal: {len(ds)} items")
    print("By task:", dict(counts))
