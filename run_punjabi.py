"""
run_punjabi.py — Phase 5: CLI entry point for the Punjabi / Urdu Benchmark

Replication of arXiv:2512.14982 for RTL / low-resource languages.

Usage examples:
    python run_punjabi.py --dry-run
    python run_punjabi.py --language ur --dry-run
    python run_punjabi.py --models gpt-4o-mini
    python run_punjabi.py --language ur --models gpt-4o-mini --tasks GSM8K ARC
    python run_punjabi.py --models gpt-4o gpt-4o-mini --tasks GSM8K ARC
    python run_punjabi.py --models gpt-4o-mini --resume results_20260522_120000.json
    python run_punjabi.py --analysis-only --results results_20260522_120000.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


# ── Task → scenario mapping ───────────────────────────────────────────────────

_TASK_SCENARIOS = {
    "ARC":           ["2_Question_First", "3_Options_First"],
    "CommonSenseQA": ["2_Question_First", "3_Options_First"],
    "OpenBookQA":    ["2_Question_First", "3_Options_First"],
    "GSM8K":         ["4_Question_Only"],
    "NameIndex":     ["10_Data_First_Retrieval"],
    "MiddleMatch":   ["10_Data_First_Retrieval"],
    "ScriptMixed":   ["8_English_to_RTL", "4_Question_Only"],
}

_ALL_TASKS = sorted(_TASK_SCENARIOS.keys())

# Stable display order
_SCENARIO_ORDER = [
    "2_Question_First",
    "3_Options_First",
    "4_Question_Only",
    "8_English_to_RTL",
    "10_Data_First_Retrieval",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_scenarios(dataset: list, override: list = None) -> list:
    if override:
        return override
    needed = set()
    for item in dataset:
        for sc in _TASK_SCENARIOS.get(item.get("task", ""), []):
            needed.add(sc)
    return [sc for sc in _SCENARIO_ORDER if sc in needed]


def _run_analysis(results_path: str, out_dir: str, no_plot: bool) -> None:
    from analysis import (
        load_csv, load_json,
        compute_accuracy, compute_mcnemar,
        print_accuracy_table, print_mcnemar_table,
        print_summary, plot_figure1,
    )

    p = Path(results_path)
    if not p.exists():
        sys.exit(f"ERROR: results file not found: {results_path}")

    if p.suffix.lower() == ".json":
        print(f"Loading JSON: {results_path}")
        df = load_json(results_path)
    else:
        print(f"Loading CSV: {results_path}")
        df = load_csv(results_path)

    print(
        f"  {len(df):,} rows | {df['model'].nunique()} model(s) | "
        f"{df['scenario'].nunique()} scenario(s) | {df['item_id'].nunique()} items"
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = compute_accuracy(df)
    print_accuracy_table(acc)
    acc_path = out / "accuracy_table.csv"
    acc.to_csv(acc_path, index=False)
    print(f"Accuracy table saved → {acc_path}")

    mc = compute_mcnemar(df)
    print_mcnemar_table(mc)
    mc_path = out / "mcnemar_results.csv"
    mc.to_csv(mc_path, index=False)
    print(f"McNemar results saved → {mc_path}")

    print_summary(acc, mc)

    if not no_plot:
        try:
            fig_path = str(out / "figure1.png")
            plot_figure1(acc, out_path=fig_path)
        except ImportError as e:
            print(f"[Figure 1] Skipped — missing package: {e}")
            print("  Install with: pip install matplotlib")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="RTL/Low-Resource Prompt Repetition Benchmark (arXiv:2512.14982 replication)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--language", default="pa", choices=["pa", "ur"],
        help="Language to benchmark: pa=Punjabi (default), ur=Urdu",
    )
    p.add_argument(
        "--models", nargs="+", default=["gpt-4o-mini"],
        metavar="MODEL",
        help="litellm model string(s) to run  (default: gpt-4o-mini)",
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        metavar="TASK",
        choices=_ALL_TASKS,
        help=f"Filter dataset to specific task(s). Choices: {_ALL_TASKS}",
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        metavar="SCENARIO",
        help="Override scenario selection (default: auto-derived from task types)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="No API calls — verify prompts build correctly and print sizes",
    )
    p.add_argument(
        "--resume", metavar="FILE",
        help="Resume from a previous JSON output file",
    )
    p.add_argument(
        "--analysis-only", action="store_true",
        help="Skip experiment; run analysis on --results file",
    )
    p.add_argument(
        "--results", metavar="FILE",
        help="CSV or JSON results file (required with --analysis-only)",
    )
    p.add_argument(
        "--out", default=".",
        help="Output directory for analysis artefacts  (default: current dir)",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip Figure 1 generation",
    )
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()

    # ── analysis-only shortcut ────────────────────────────────────────────────
    if args.analysis_only:
        if not args.results:
            sys.exit("ERROR: --analysis-only requires --results <file>")
        _run_analysis(args.results, args.out, args.no_plot)
        return

    # ── select loader + system prompt by language ─────────────────────────────
    if args.language == "ur":
        from urdu_datasets_loader import build_dataset
        from experiment_runner import SYSTEM_PROMPT_UR as system_prompt
        lang_label = "Urdu"
    else:
        from punjabi_datasets_loader import build_dataset
        from experiment_runner import SYSTEM_PROMPT_PA as system_prompt
        lang_label = "Punjabi (Shahmukhi)"

    # ── load + filter dataset ─────────────────────────────────────────────────
    print(f"Loading {lang_label} dataset …")
    dataset = build_dataset()
    print(f"  Total loaded: {len(dataset)} items")

    if args.tasks:
        allowed = set(args.tasks)
        dataset = [it for it in dataset if it.get("task") in allowed]
        print(f"  After --tasks {args.tasks}: {len(dataset)} items")
        if not dataset:
            sys.exit("ERROR: no items match the requested tasks.")

    # ── determine scenarios ───────────────────────────────────────────────────
    scenarios = _select_scenarios(dataset, override=args.scenarios)
    if not scenarios:
        sys.exit("ERROR: no scenarios could be determined for the selected tasks.")
    print(f"  Scenarios selected: {scenarios}")

    # ── resolve output file paths ─────────────────────────────────────────────
    if args.resume:
        output_json = args.resume
        output_csv = str(Path(args.resume).with_suffix(".csv"))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = f"results_{args.language}_{stamp}.json"
        output_csv  = f"results_{args.language}_{stamp}.csv"

    # ── run experiment ────────────────────────────────────────────────────────
    from experiment_runner import run_experiment

    run_experiment(
        models        = args.models,
        dataset       = dataset,
        output_file   = output_json,
        csv_file      = output_csv,
        scenarios     = scenarios,
        system_prompt = system_prompt,
        verbose       = True,
        dry_run       = args.dry_run,
    )

    # ── post-run analysis (real runs only) ────────────────────────────────────
    if not args.dry_run:
        print("\nRunning post-experiment analysis …")
        _run_analysis(output_csv, args.out, args.no_plot)


if __name__ == "__main__":
    main()
