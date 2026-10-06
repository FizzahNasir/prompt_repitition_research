"""
run_all_languages.py — Batch runner for all 7 RTL languages

Runs baseline + repetition methods across all 7 languages in sequence.
Each language runs separately with its own dataset, system prompt, and results file.

Usage:
    python run_all_languages.py --dry-run --models gpt-4o-mini
    python run_all_languages.py --models gemma-2b-it
    python run_all_languages.py --language ar fa sd --models gpt-4o-mini
"""

import argparse
import sys
from pathlib import Path

_LANGUAGES = {
    "pa":  ("Punjabi (Shahmukhi)",    "punjabi_datasets_loader",  "SYSTEM_PROMPT_PA"),
    "ur":  ("Urdu",                   "urdu_datasets_loader",     "SYSTEM_PROMPT_UR"),
    "ps":  ("Pashto",                 "pashto_datasets_loader",   "SYSTEM_PROMPT_PS"),
    "bal": ("Balochi",                "balochi_datasets_loader",    "SYSTEM_PROMPT_BAL"),
    "ar":  ("Arabic",                 "arabic_datasets_loader",     "SYSTEM_PROMPT_AR"),
    "fa":  ("Persian (Farsi)",        "persian_datasets_loader",    "SYSTEM_PROMPT_FA"),
    "sd":  ("Sindhi",                 "sindhi_datasets_loader",     "SYSTEM_PROMPT_SD"),
}


def main():
    p = argparse.ArgumentParser(
        description="Batch runner for all 7 RTL prompt repetition languages"
    )
    p.add_argument(
        "--language", nargs="+", choices=list(_LANGUAGES.keys()) + ["all"],
        default=["all"],
        help="Language(s) to run (default: all)",
    )
    p.add_argument(
        "--models", nargs="+", default=["qwen2.5-3b"],
        help="Local model key(s) to use (default: qwen2.5-3b). "
             "Open models: qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b, "
             "mistral-7b-v0.3, gemma2-2b, llama3.2-1b, llama3.2-3b, llama3.1-8b "
             "(Llama models require HF login)",
    )
    p.add_argument(
        "--methods", nargs="+",
        default=["baseline", "repetition"],
        choices=["baseline", "repetition", "cross_lingual_t1", "cross_lingual_t2"],
        help="Prompt methods to evaluate (default: baseline repetition)",
    )
    p.add_argument("--tasks", nargs="+", default=None,
        help="Filter to specific task(s)")
    p.add_argument("--scenarios", nargs="+", default=None,
        help="Override scenarios")
    p.add_argument("--dry-run", action="store_true",
        help="No API calls — verify prompts build correctly")
    p.add_argument("--out", default=".",
        help="Output directory for results")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    langs = list(_LANGUAGES.keys()) if "all" in args.language else args.language

    from experiment_runner import run_experiment

    for lang in langs:
        label, loader_mod, prompt_attr = _LANGUAGES[lang]
        print(f"\n{'='*65}")
        print(f"Running experiments for: {label} ({lang})")
        print(f"{'='*65}\n")

        loader = __import__(loader_mod)
        build_dataset = loader.build_dataset
        system_prompt = getattr(sys.modules['experiment_runner'], prompt_attr)

        dataset = build_dataset()
        print(f"  Total loaded: {len(dataset)} items")

        if args.tasks:
            dataset = [it for it in dataset if it.get("task") in set(args.tasks)]
            print(f"  After --tasks {args.tasks}: {len(dataset)} items")

        from run_punjabi import _select_scenarios
        scenarios = _select_scenarios(dataset, override=args.scenarios)
        if not scenarios:
            print(f"  ERROR: no scenarios could be determined")
            continue

        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = f"{args.out}/results_{lang}_{stamp}.json"
        output_csv = f"{args.out}/results_{lang}_{stamp}.csv"

        run_experiment(
            models=args.models,
            dataset=dataset,
            output_file=output_json,
            csv_file=output_csv,
            scenarios=scenarios,
            system_prompt=system_prompt,
            methods=args.methods,
            verbose=True,
            dry_run=args.dry_run,
        )

    print("\nAll languages complete!")


if __name__ == "__main__":
    main()
