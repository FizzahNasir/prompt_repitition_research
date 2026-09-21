"""
Multilingual Prompt Repetition Benchmark
Extension of arXiv:2512.14982 for RTL / Low-resource Languages
(Urdu, Punjabi-Shahmukhi, Sindhi, Balochi, Pashto, Arabic)

Implements:
  5 prompt methods  : baseline, repetition, verbose, triple, padding
  10 scenario templates
  Task-specific evaluation : MCQ, Math, Retrieval
  Resume-safe JSON output   (crash → resume from last saved item)
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

try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


# ── Models ────────────────────────────────────────────────────────────────────

TIER1_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "deepseek/deepseek-chat",
]

# Seconds to sleep between API calls within one item.
# Gemini free tier: 15 RPM → need ≥4s. Set to 4.5s to stay safely under the limit.
# For paid-tier OpenAI/Gemini this can be reduced to 0.4s.
INTER_CALL_SLEEP = 4.5

TIER2_MODELS = [
    "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    "together_ai/CohereForAI/aya-expanse-32b",
    "together_ai/meta-llama/Llama-3.1-70B-Instruct-Turbo",
    "together_ai/inceptionai/jais-adapted-13b-chat",
]


# ── 1. Repeat Phrases (for Verbose & Triple methods) ─────────────────────────

# System prompt for Punjabi experiments (instructs model: answer directly, no CoT)
SYSTEM_PROMPT_PA = "براہ راست جواب دیو۔ اپنی سوچ دی وضاحت نہ کرو۔ قدم بہ قدم نہ سوچو۔"

# System prompt for Urdu experiments (same intent, Urdu phrasing)
SYSTEM_PROMPT_UR = "براہ راست جواب دیں۔ اپنی سوچ کی وضاحت نہ کریں۔ قدم بقدم مت سوچیں۔"

# First repeat-bridge phrase used in verbose and triple methods
REPEAT_PHRASES = {
    "en":  "Please re-read the above and answer:",
    "ar":  "يرجى إعادة قراءة ما سبق والإجابة:",
    "ur":  "براہ کرم اوپر دوبارہ پڑھیں اور جواب دیں:",
    "pa":  "میں دوبارہ آکھدا ہاں:",                         # Shahmukhi Punjabi
    "sd":  "مهرباني ڪري مٿيون ٻيهر پڙهو ۽ جواب ڏيو:",      # Sindhi
    "ps":  "مهرباني وکړئ پورته بیا ولولئ او ځواب ورکړئ:",   # Pashto
    "bal": "مهربانی کن بالا دوباره بخوان و جواب بده:",       # Balochi
}

# Second bridge phrase used only in the triple (×3) method
TRIPLE_SECOND_PHRASES = {
    "en":  "One more time:",
    "ar":  "مرة أخرى:",
    "ur":  "ایک اور بار:",
    "pa":  "اک ہور واری:",
    "sd":  "هڪ ٻي ڀيرو:",
    "ps":  "یوه بله ځل:",
    "bal": "یک بار دیگر:",
}


# ── 2. The 5 Prompt Methods ───────────────────────────────────────────────────

METHODS = ["baseline", "repetition", "verbose", "triple", "padding"]

def build_all_methods(base_prompt: str, language: str = "en") -> dict:
    """Build all 5 prompt variants from a single base prompt."""
    phrase1 = REPEAT_PHRASES.get(language, REPEAT_PHRASES["en"])
    phrase2 = TRIPLE_SECOND_PHRASES.get(language, TRIPLE_SECOND_PHRASES["en"])
    padding = "." * len(base_prompt)
    return {
        "baseline":   base_prompt,
        "repetition": f"{base_prompt}\n\n{base_prompt}",
        "verbose":    f"{base_prompt}\n\n{phrase1}\n\n{base_prompt}",
        "triple":     f"{base_prompt}\n\n{phrase1}\n\n{base_prompt}\n\n{phrase2}\n\n{base_prompt}",
        "padding":    f"{base_prompt}\n\n{padding}",
    }


# ── 3. The 10 Scenario Templates ──────────────────────────────────────────────
# Each entry lists required item fields and a format string.
# Items missing any required field are silently skipped for that scenario.

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
        "template": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Reply with just the correct option letter."
        ),
    },
    "3_Options_First": {
        "fields": ["question", "options"],
        "template": (
            "Options:\n{options}\n\n"
            "Question: {question}\n\n"
            "Reply with just the correct option letter."
        ),
    },
    "4_Question_Only": {
        "fields": ["question"],
        "template": "{question}",
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
        "template": "Data:\n{long_data}\n\nQuery: {query}",
    },
}


def _format_options(opts) -> str:
    if isinstance(opts, dict):
        return "\n".join(f"{k}) {v}" for k, v in opts.items())
    return str(opts)


def build_base_prompt(scenario_name: str, item: dict):
    """Format item into a base prompt for the given scenario. Returns None if fields missing."""
    s = SCENARIOS[scenario_name]
    ctx = dict(item)

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

FUZZY_THRESHOLD = 80  # rapidfuzz score 0–100; matches plan spec (≥ 0.8)


def _fuzzy_match(answer: str, response: str) -> bool:
    """Return True if answer appears in response, allowing for Shahmukhi spelling variation."""
    ans = answer.strip()
    if ans.lower() in response.lower():
        return True  # fast path: exact substring
    if _RAPIDFUZZ_AVAILABLE:
        # partial_ratio: best alignment of the shorter string within the longer
        return _rfuzz.partial_ratio(ans, response) >= FUZZY_THRESHOLD
    # difflib fallback: compare answer against each whitespace token in response
    import difflib
    ans_l = ans.lower()
    for token in response.split():
        if difflib.SequenceMatcher(None, ans_l, token.lower()).ratio() >= FUZZY_THRESHOLD / 100:
            return True
    return False


def _extract_mcq(text: str):
    m = re.search(r'\b([A-Ja-j])\b', text.strip())
    return m.group(1).upper() if m else None


def _extract_number(text: str):
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return nums[-1].replace(",", "") if nums else None


def is_correct(task: str, response: str, answer: str) -> bool:
    if task in MCQ_TASKS:
        return _extract_mcq(response) == str(answer).upper()
    if task in MATH_TASKS:
        got = _extract_number(response)
        return got == str(answer).replace(",", "") if got else False
    # Retrieval: correct name should appear somewhere in the response
    return str(answer).strip().lower() in response.lower()


# ── 5. API Call with Exponential Retry ───────────────────────────────────────

def call_model(
    model: str,
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 100,
    retries: int = 3,
) -> dict:
    """
    Call the model and return a dict:
      {response, prompt_tokens, output_tokens, latency_ms}
    max_tokens=100 matches the original paper (prevents CoT responses).
    """
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm not installed. Run: pip install litellm")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            r = litellm.completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            usage = r.usage or {}
            return {
                "response":      r.choices[0].message.content.strip(),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "latency_ms":    round(latency_ms, 1),
            }
        except Exception as e:
            if attempt == retries - 1:
                raise
            # Rate-limit errors: respect provider hint (Gemini free tier asks ~31s)
            err_str = str(e).lower()
            if "ratelimit" in err_str or "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                wait = 35 * (2 ** attempt)   # 35s, 70s
            else:
                wait = 2 ** attempt          # 1s, 2s for transient errors
            print(f"      Retry {attempt + 1}/{retries}: {e} — waiting {wait}s")
            time.sleep(wait)


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
    verbose: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Run all 5 prompt methods × selected scenarios × dataset items for each model.
    Saves JSON after every item (resume-safe) and appends to CSV after every call.

    Args:
        models:        List of litellm model strings.
        dataset:       List of item dicts (from punjabi_loader.build_dataset or sample_data).
        output_file:   Resume-safe JSON path. Auto-named if None.
        csv_file:      Flat CSV path for analysis. Appended to, never overwritten.
        scenarios:     Subset of SCENARIOS keys to run. Defaults to all.
        system_prompt: Optional system message sent with every call.
        verbose:       Print per-item progress.
        dry_run:       Skip API calls; print prompt sizes only.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())
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
    print(f"           ~{compatible_count} compatible (model x scenario x item) x {len(METHODS)} methods")
    print(f"JSON:      {output_file}")
    print(f"CSV:       {csv_file}")
    if system_prompt:
        print(f"SysPrompt: {system_prompt[:60]}{'…' if len(system_prompt) > 60 else ''}")
    if dry_run:
        print("MODE:      DRY RUN — no API calls, prompt sizes only")
    print(f"{'='*65}\n")

    for model in models:
        results.setdefault(model, {})
        print(f"Model: {model}")

        for sc in scenarios:
            results[model].setdefault(sc, [])
            done_ids = {r["item_id"] for r in results[model][sc]}
            todo = [
                it for it in dataset
                if it.get("id") not in done_ids and build_base_prompt(sc, it) is not None
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

                for method in METHODS:
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
                    time.sleep(INTER_CALL_SLEEP)

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
