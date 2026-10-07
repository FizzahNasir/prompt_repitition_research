"""
Multilingual Prompt Repetition Benchmark
Extension of arXiv:2512.14982 for RTL / Low-resource Languages
(Arabic, Balochi, Pashto, Persian, Punjabi-Shahmukhi, Sindhi, Urdu)

Implements:
  4 prompt methods  : baseline, repetition, cross_lingual_t1 (English bridge), cross_lingual_t2 (native bridge)
  10 scenario templates
  Task-specific evaluation : MCQ, Math, Retrieval
  Resume-safe JSON output   (crash -> resume from last saved item)
  Win/loss analysis table

Install: pip install litellm openai anthropic google-generativeai
API keys: set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
"""

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

try:
    import litellm
    litellm.set_verbose = False
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("litellm not installed. Run: pip install litellm")


# ── Models ─────────────────────────────────────────────────────────────────────
# Local LLM models for CPU/GPU inference (transformers)
# All are non-reasoning instruction-tuned models per arXiv:2512.14982
# Using open-access models (no gating) for Colab compatibility

LOCAL_MODELS = [
    "qwen2.5-3b",
    "qwen2.5-7b",
    "mistral-7b-v0.3",
    "gemma2-2b",
]

# Model resolution for HuggingFace transformers
MODEL_MAP = {
    "llama3.2-1b":      "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b":      "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b":      "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-1.5b":     "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":       "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":       "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b-v0.3":  "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-2b":        "google/gemma-2-2b-it",
}

try:
    import torch
    USE_CUDA = torch.cuda.is_available()
except ImportError:  # prompt building / scoring / dry runs don't need torch
    torch = None
    USE_CUDA = False


# ── 1. System Prompts & Repeat Phrases ──────────────────────────────────────────

SYSTEM_PROMPT_PA = "براہ راست جواب دیو۔ اپنی سوچ دی وضاحت نہ کرو۔ قدم بہ قدم نہ سوچو۔"
SYSTEM_PROMPT_UR = "براہ راست جواب دیں۔ اپنی سوچ کی وضاحت نہ کریں۔ قدم بقدم مت سوچیں۔"
SYSTEM_PROMPT_PS = "مستقیم ځواب ورکړئ. خپل فکر مه تشریح کوئ. ګام په ګام مه فکر کوئ."
SYSTEM_PROMPT_BAL = "تچک ءَ پسو بہ دئے. وتی ھیال ءَ مَہ درشان کن. گام پہ گام مَہ جیڑ."
SYSTEM_PROMPT_AR = "أجب مباشرةً. لا تشرح طريقة تفكيرك. لا تفكر خطوة بخطوة."
SYSTEM_PROMPT_FA = "فقط به طور مستقیم پاسخ بده. فکر خود را توضیح نده. قدم به قدم فکر نکن."
SYSTEM_PROMPT_SD = "سڌو جواب ڏيو. پنهنجي سوچ جي وضاحت نه ڪريو. قدم بقدم نه سوچيو."

# Second bridge phrase used only in the triple (×3) method
TRIPLE_SECOND_PHRASES = {
    "en":  "One more time:",
    "ar":  "مرة أخرى:",
    "ur":  "ایک اور بار:",
    "pa":  "اک ہور واری:",
    "sd":  "هڪ ٻي ڀيرو:",
    "ps":  "یوه بله ځل:",
    "bal": "یک بار دیگر:",
    "fa":  "یک بار دیگر:",
}


# ── 2. The 4 Prompt Methods ────────────────────────────────────────────────────
# 
# Based on arXiv:2512.14982 extension for RTL languages:
#   Method 1 (baseline):     Single instance of the query
#   Method 2 (repetition):   Query repeated twice (in-language)
#   Method 3 (cross_lingual_t1): Query in target language + English bridge + Query repeated
#   Method 4 (cross_lingual_t2): Query in target language + Target-language bridge + Query repeated
#
# English is ALWAYS the reference language for cross-lingual T1.
# Target language is used for T2 bridges.

METHODS = ["baseline", "repetition", "cross_lingual_t1", "cross_lingual_t2"]

ENGLISH_REPEAT_PHRASE = "Let me repeat that:"
NATIVE_REPEAT_PHRASES = {
    "ar":  "لنكرر ذلك:",
    "ur":  "میں دوبارہ کہتا ہوں:",
    "pa":  "میں دوبارہ آکھدا ہاں:",
    "ps":  "زه بیا وایم:",
    "bal": "من پدا گشاں:",
    "fa":  "بیا دوباره می‌گویم:",
    "sd":  "مه ٻيهر کيا آں:",
}

def build_all_methods(base_prompt: str, language: str = "en") -> dict:
    """Build all 4 prompt variants from a single base prompt."""
    native_phrase = NATIVE_REPEAT_PHRASES.get(language, NATIVE_REPEAT_PHRASES["ar"])
    return {
        "baseline":           base_prompt,
        # Paper A.4 puts the copies on consecutive lines: <QUERY>\n<QUERY>
        "repetition":         f"{base_prompt}\n{base_prompt}",
        "cross_lingual_t1":   f"{base_prompt}\n{ENGLISH_REPEAT_PHRASE}\n{base_prompt}",
        "cross_lingual_t2":   f"{base_prompt}\n{native_phrase}\n{base_prompt}",
    }


# ── 3. The 10 Scenario Templates ──────────────────────────────────────────────
# Each entry lists required item fields and a format string.
# Items missing any required field are silently skipped for that scenario.
#
# Scenarios 2, 3, 4 and 10 follow arXiv:2512.14982 Appendix A.3/A.4: no section labels,
# options as "A. text", and the paper's English answer-format line, kept verbatim in every
# language so answer extraction is comparable.

MCQ_FORMAT_LINE  = "Reply with one letter ({letters}) in the format: The answer is <ANSWER>."
MATH_FORMAT_LINE = "Reply with only the final number in the format: The answer is <ANSWER>."

SCENARIOS = {
    "1_Instruction_First": {
        "fields": ["instruction", "question", "options"],
        "template": (
            "Instruction: {instruction}\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Reply with just the correct option letter."
        ),
    },
    "2_Question_First": {
        "fields": ["question", "options"],
        "template": "{question}\n{options}\n" + MCQ_FORMAT_LINE,
    },
    "3_Options_First": {
        "fields": ["question", "options"],
        "template": "{options}\n{question}\n" + MCQ_FORMAT_LINE,
    },
    "4_Question_Only": {
        "fields": ["question"],
        "template": "{question}\n" + MATH_FORMAT_LINE,
    },
    "5_Shuffled_Options": {
        "fields": ["question", "shuffled_options"],
        "template": (
            "Question: {question}\n\n"
            "Options:\n{shuffled_options}\n\n"
            "Reply with just the correct option letter."
        ),
    },
    "6_PreInstruction_Context": {
        "fields": ["context", "question", "options"],
        "template": (
            "Heads-up: Read the following context carefully.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Reply with just the correct option letter."
        ),
    },
    "7_Reading_Comprehension": {
        "fields": ["context", "question", "options"],
        "template": (
            "Passage:\n{context}\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Reply with just the correct option letter."
        ),
    },
    "8_English_to_RTL": {
        "fields": ["english_instruction", "rtl_content"],
        "template": "{english_instruction}\n\n{rtl_content}",
    },
    "9_RTL_to_English": {
        "fields": ["rtl_instruction", "english_content"],
        "template": "{rtl_instruction}\n\n{english_content}",
    },
    "10_Data_First_Retrieval": {
        "fields": ["long_data", "query"],
        "template": "{long_data}\n{query}",
    },
}


def _format_options(opts) -> str:
    if isinstance(opts, dict):
        return "\n".join(f"{k}. {v}" for k, v in opts.items())
    return str(opts)


def build_base_prompt(scenario_name: str, item: dict):
    """Format item into a base prompt for the given scenario. Returns None if fields missing."""
    s = SCENARIOS[scenario_name]
    ctx = dict(item)

    # Letters for the answer-format line, e.g. 'A', 'B', 'C', 'D'
    opts = item.get("shuffled_options" if scenario_name == "5_Shuffled_Options" else "options")
    if isinstance(opts, dict):
        ctx["letters"] = ", ".join(f"'{k}'" for k in opts)

    # Pre-format option dicts into labeled strings
    for key in ("options", "shuffled_options"):
        if key in ctx and isinstance(ctx[key], dict):
            ctx[key] = _format_options(ctx[key])

    # Skip items with missing or None required fields
    if not all(ctx.get(f) is not None for f in s["fields"]):
        return None

    try:
        return s["template"].format(**ctx)
    except KeyError:
        return None


# ── 4. Task-Specific Answer Evaluation ───────────────────────────────────────

MCQ_TASKS  = {"ARC", "OpenBookQA", "MMLU", "MMLU_Pro", "CommonSenseQA"}
MATH_TASKS = {"GSM8K", "MATH", "TokenizationControl"}
RETR_TASKS = {"NameIndex", "MiddleMatch", "ScriptMixed"}

_ANSWER_IS = re.compile(r"answer\s+is", re.IGNORECASE)

# Arabic-Indic and Extended (Persian/Urdu) digits, Arabic decimal/thousands separators
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"


def _extract_mcq(text: str):
    """Letter from 'The answer is X' (paper format); else a leading letter; else the only
    standalone letter in the reply. Echoes of the prompt or option list are not answers."""
    t = text.strip()
    m = re.findall(r"(?i:answer\s+is)\s*[:：]?\s*[\(\[<\*\"']*\s*([A-J])(?![A-Za-z])", t)
    if m:
        return m[-1]
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if ("Reply with one letter" in t
            or (len(lines) > 1 and all(re.match(r"\s*[A-J][\.\)]", ln) for ln in lines[:2]))):
        return None
    m = re.match(r"[\(\[<\*\s]*([A-J])(?![A-Za-z])", t)
    if m:
        return m.group(1)
    letters = set(re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", t))
    return letters.pop() if len(letters) == 1 else None


def _extract_number(text: str):
    """Number after 'The answer is' (paper format); else the last number in the reply."""
    t = text.translate(_DIGITS)
    nums = (re.findall(r"(?i:answer\s+is)\D{0,3}?(" + _NUM + ")", t, re.ASCII)
            or re.findall(_NUM, t, re.ASCII))
    return nums[-1].replace(",", "") if nums else None


def _normalize_name(s: str) -> str:
    """Fold spelling variants of Perso-Arabic letters, drop diacritics/ZWNJ/punctuation."""
    s = str(s)
    s = re.sub(r"[ً-ٰٟۖ-ۭ‌‍‏‎ـ]", "", s)
    for src, dst in (("ي", "ی"), ("ى", "ی"), ("ې", "ی"), ("ے", "ی"), ("ك", "ک"), ("ڪ", "ک"),
                     ("ة", "ه"), ("ۃ", "ه"), ("ہ", "ه"), ("ۀ", "ه"), ("ھ", "ه"),
                     ("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا")):
        s = s.replace(src, dst)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.lower().split())


_LIST_ITEM = re.compile(r"\s*(?:\d+|[٠-٩۰-۹]+)[\.\)]|\s*[-•*]\s")


def _answer_span(response: str) -> str:
    """Text after the last 'answer is', else the first non-empty line. A lead-in line
    ending in ':' or '?' ("The 25th name is:") also takes the next line, or the whole
    list if one follows, so the distractor check still rejects echoed lists."""
    parts = _ANSWER_IS.split(response)
    if len(parts) > 1:
        return parts[-1]
    lines = [ln for ln in response.splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) > 1 and lines[0].rstrip().rstrip("\"'*").endswith((":", "：", "?", "؟")):
        span = lines[:2]
        if _LIST_ITEM.match(lines[1]):
            span += [ln for ln in lines[2:] if _LIST_ITEM.match(ln)]
        return "\n".join(span)
    return lines[0]


def is_correct(task: str, response: str, answer: str, distractors=None, anchors=None) -> bool:
    """Score one response.

    MCQ:       extracted letter == answer letter.
    Math:      extracted number == answer (numeric compare, any digit script).
    Retrieval: exact (normalized) name match in the answer span, as in the paper. If
               `distractors` (other names from the list) are given, the span must not
               also contain one of them, so echoing the whole list doesn't count.
               `anchors` are the names in the question (MiddleMatch's "between X and
               Y"); a restated question is removed first, so an anchor that is also the
               gold answer only counts when it is stated on its own.
    """
    response = response or ""
    if task in MCQ_TASKS:
        return _extract_mcq(response) == str(answer).strip().upper()
    if task in MATH_TASKS:
        got = _extract_number(response)
        try:
            return got is not None and abs(float(got) - float(str(answer).translate(_DIGITS).replace(",", ""))) < 1e-6
        except ValueError:
            return False
    span = f" {_normalize_name(_answer_span(response))} "
    names = [n for n in (_normalize_name(a) for a in anchors or []) if n]
    while len(names) >= 2 and all(f" {n} " in span for n in names):
        for n in names:
            span = span.replace(f" {n} ", " ", 1)
    target = _normalize_name(answer)
    if not target or f" {target} " not in span:
        return False
    for d in distractors or []:
        d = _normalize_name(d)
        if d and d != target and d not in target and f" {d} " in span:
            return False
    return True


# ── 5. Local Model Call ─────────────────────────────────────────────────────

_model_cache = {}

def _load_model(model_key: str):
    """Load a local HuggingFace model with tokenizer (cached)."""
    if model_key not in _model_cache:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        hf_name = MODEL_MAP[model_key]
        print(f"  Loading model: {hf_name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.float16 if USE_CUDA else torch.float32,
            device_map="auto" if USE_CUDA else "cpu",
        )
        _model_cache[model_key] = (tokenizer, model)
    return _model_cache[model_key]


def call_model(
    model: str,
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 100,
    retries: int = 3,
) -> dict:
    """
    Call a local model and return a dict:
      {response, prompt_tokens, output_tokens, latency_ms}
    
    Uses HuggingFace transformers for local inference.
    max_tokens=100 is this project's choice (the paper gives no limit); it prevents CoT responses.
    temperature=0 (deterministic) per paper specification.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer, model_obj = _load_model(model)
    hf_name = MODEL_MAP[model]
    
    # Build chat template
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        # Fallback for tokenizers without chat template
        formatted = ""
        if system_prompt:
            formatted += f"System: {system_prompt}\n\n"
        formatted += f"User: {prompt}\n\nAssistant:"
        inputs = tokenizer(formatted, return_tensors="pt")
    
    input_ids = inputs.input_ids
    if USE_CUDA:
        input_ids = input_ids.cuda()
    
    prompt_tokens = input_ids.shape[1]
    
    t0 = time.perf_counter()
    
    with torch.no_grad():
        outputs = model_obj.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    latency_ms = (time.perf_counter() - t0) * 1000
    
    # Decode response (skip the input prompt)
    response_ids = outputs[0][prompt_tokens:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    output_tokens = len(response_ids)
    
    return {
        "response":       response,
        "prompt_tokens":  prompt_tokens,
        "output_tokens":  output_tokens,
        "latency_ms":     round(latency_ms, 1),
    }


# ── 6. CSV sink ───────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "model", "task", "scenario", "method", "item_id",
    "language", "correct_answer", "response", "is_correct",
    "prompt_tokens", "output_tokens", "latency_ms", "prompt_chars",
]


def _append_csv(csv_file: str, row: dict) -> None:
    write_header = not Path(csv_file).exists()
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── 7. Core Experiment Loop ───────────────────────────────────────────────────

def run_experiment(
    models: list,
    dataset: list,
    output_file: str = None,
    csv_file: str = "results.csv",
    scenarios: list = None,
    system_prompt: str = None,
    methods: list = None,
    verbose: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Run prompt methods x selected scenarios x dataset items for each model.
    Saves JSON after every item (resume-safe) and appends to CSV after every call.

    Args:
        models:        List of litellm model strings.
        dataset:       List of item dicts.
        output_file:   Resume-safe JSON path. Auto-named if None.
        csv_file:      Flat CSV path for analysis. Appended to, never overwritten.
        scenarios:     Subset of SCENARIOS keys to run. Defaults to all.
        system_prompt: Optional system message sent with every call.
        methods:       Subset of METHODS to run. Defaults to all.
        verbose:       Print per-item progress.
        dry_run:       Skip API calls; print prompt sizes only.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())
    if methods is None:
        methods = METHODS
    if output_file is None:
        output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Resume from existing file if present
    results: dict = {}
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from: {output_file}")

    compatible_count = sum(
        1 for sc in scenarios for it in dataset if build_base_prompt(sc, it)
    )
    print(f"\n{'='*65}")
    print(f"Benchmark: {len(models)} model(s) x {len(scenarios)} scenario(s)")
    print(f"           ~{compatible_count} compatible (model x scenario x item) x {len(methods)} methods")
    print(f"Methods:   {methods}")
    print(f"JSON:      {output_file}")
    print(f"CSV:       {csv_file}")
    if system_prompt:
        print(f"SysPrompt: {system_prompt[:60]}{chr(8230) if len(system_prompt) > 60 else ''}")
    if dry_run:
        print("MODE:      DRY RUN - no API calls, prompt sizes only")
    print(f"{'='*65}\n")

    for model in models:
        results.setdefault(model, {})
        print(f"Model: {model}")

        for sc in scenarios:
            results[model].setdefault(sc, [])
            done_ids = {r["item_id"] for r in results[model][sc]}
            from run_punjabi import _TASK_SCENARIOS
            todo = [
                it for it in dataset
                if it.get("id") not in done_ids
                and sc in _TASK_SCENARIOS.get(it.get("task", ""), [sc])
                and build_base_prompt(sc, it) is not None
            ]
            if not todo:
                continue

            print(f"  Scenario: {sc}  ({len(todo)} items)")

            for item in todo:
                iid     = item.get("id", "?")
                lang    = item.get("language", "en")
                task    = item.get("task", "MCQ")
                answer  = (
                    item.get("shuffled_correct_answer", item.get("correct_answer"))
                    if sc == "5_Shuffled_Options"
                    else item.get("correct_answer")
                )

                base    = build_base_prompt(sc, item)
                prompts = build_all_methods(base, lang)

                if verbose:
                    print(f"\n    [{iid}]  lang={lang}  task={task}  answer={answer}")

                row = {
                    "item_id":        iid,
                    "language":       lang,
                    "task":           task,
                    "scenario":       sc,
                    "correct_answer": answer,
                }

                for method in methods:
                    prompt = prompts[method]
                    if dry_run:
                        print(f"      [{method:<12s}] {len(prompt):>5} chars")
                        row[method] = {"response": "[dry-run]", "correct": None}
                        continue
                    try:
                        result = call_model(model, prompt, system_prompt=system_prompt)
                        resp   = result["response"]
                        ok     = is_correct(task, resp, answer)
                        row[method] = {
                            "response":      resp[:120],
                            "correct":       ok,
                            "prompt_tokens": result["prompt_tokens"],
                            "output_tokens": result["output_tokens"],
                            "latency_ms":    result["latency_ms"],
                        }
                        if verbose:
                            mark = "+" if ok else "-"
                            tok  = result["output_tokens"] or "?"
                            ms   = result["latency_ms"]
                            print(f"      [{method:<12s}] [{mark}] {tok:>3}tok {ms:>7.0f}ms  '{resp[:45]}'")
                        if csv_file:
                            _append_csv(csv_file, {
                                "timestamp":     datetime.now().isoformat(timespec="seconds"),
                                "model":         model,
                                "task":          task,
                                "scenario":      sc,
                                "method":        method,
                                "item_id":       iid,
                                "language":      lang,
                                "correct_answer": answer,
                                "response":      resp[:120],
                                "is_correct":    int(ok),
                                "prompt_tokens": result["prompt_tokens"],
                                "output_tokens": result["output_tokens"],
                                "latency_ms":    result["latency_ms"],
                                "prompt_chars":  len(prompt),
                            })
                    except Exception as e:
                        row[method] = {"response": None, "correct": False, "error": str(e)}
                        if verbose:
                            print(f"      [{method:<12s}] [ERR] {e}")
                    # Local inference - no rate limiting needed

                results[model][sc].append(row)
                # Write after every item — safe to interrupt and resume
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to: {output_file}")
    return results


# ── 8. Win / Loss Analysis ────────────────────────────────────────────────────

def compute_accuracy(results: dict) -> dict:
    """Return per-model, per-scenario, per-method accuracy and delta vs baseline."""
    report: dict = {}
    for model, scenarios in results.items():
        report[model] = {}
        for sc, items in scenarios.items():
            if not items:
                continue
            stats = {m: {"c": 0, "n": 0} for m in METHODS}
            for row in items:
                for m in METHODS:
                    if m in row and row[m].get("correct") is not None:
                        stats[m]["n"] += 1
                        if row[m]["correct"]:
                            stats[m]["c"] += 1
            base_acc = stats["baseline"]["c"] / stats["baseline"]["n"] if stats["baseline"]["n"] else 0
            report[model][sc] = {}
            for m in METHODS:
                n = stats[m]["n"]
                c = stats[m]["c"]
                acc = c / n if n else 0
                report[model][sc][m] = {
                    "accuracy": round(acc, 4),
                    "correct":  c,
                    "n":        n,
                    "delta":    round(acc - base_acc, 4),
                    "win":      acc > base_acc,
                }
    return report


def print_table(report: dict):
    """Print a formatted win/loss table to stdout."""
    COL = 10
    COLS = len(METHODS)
    W = 30 + COL * COLS

    print("\n" + "=" * W)
    print("WIN / LOSS TABLE    (+ = beats baseline)")
    print("=" * W)

    hdr = f"{'Scenario':<28}  " + "  ".join(f"{m[:7]:^{COL}}" for m in METHODS)

    for model, scenarios in report.items():
        print(f"\nModel: {model}")
        print(hdr)
        print("-" * W)
        wins = {m: 0 for m in METHODS[1:]}

        for sc, methods in scenarios.items():
            if not methods:
                continue
            row = f"{sc[:27]:<28}  "
            for m in METHODS:
                if m not in methods:
                    row += f"{'—':^{COL}}  "
                    continue
                acc  = methods[m]["accuracy"]
                mark = "+" if methods[m]["win"] else " "
                row += f"{acc*100:>6.1f}%{mark} "
                if m != "baseline" and methods[m]["win"]:
                    wins[m] += 1
            print(row)

        print("-" * W)
        total = len(scenarios)
        win_row = f"{'Wins vs baseline':<28}  {'—':^{COL}}  "
        win_row += "  ".join(f"{wins[m]:>4}/{total}   " for m in METHODS[1:])
        print(win_row)

    print("=" * W)


# ── 9. Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from sample_data import DEMO_DATASET
    except ImportError:
        print("ERROR: sample_data.py not found in this directory.")
        raise SystemExit(1)

    print(f"Dataset loaded: {len(DEMO_DATASET)} items")

    # Quick demo: one model, three high-signal scenarios, dry-run=True
    # Change dry_run=False and set your API keys to run for real
    results = run_experiment(
        models    = ["gpt-4o-mini"],
        dataset   = DEMO_DATASET,
        scenarios = ["2_Question_First", "3_Options_First", "10_Data_First_Retrieval"],
        verbose   = True,
        dry_run   = True,   # <-- flip to False for real API calls
    )

    report = compute_accuracy(results)
    print_table(report)

    with open("win_loss_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Win/loss report saved to: win_loss_report.json")
