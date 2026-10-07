"""
colab_run.py — Ready-to-run script for Google Colab GPU inference

Usage on Colab:
  1. Upload this repo to Colab (or clone from GitHub)
  2. Set Runtime → GPU (T4 or L4)
  3. Run: !pip install transformers torch pandas openpyxl scipy rapidfuzz
  4. Run: !python colab_run.py --language ar --model llama3.2-3b

This script uses local HuggingFace transformers models on GPU.
All 8 models are non-reasoning instruction-tuned models from arXiv:2512.14982.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description="Run experiments on Google Colab with GPU"
    )
    p.add_argument(
        "--language", choices=["pa", "ur", "ps", "ar", "fa", "sd", "all"],
        default="all",
        help="Language to run (default: all)",
    )
    p.add_argument(
        "--model", choices=[
            "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
            "mistral-7b-v0.3", "gemma2-2b",
            "llama3.2-1b", "llama3.2-3b", "llama3.1-8b"
        ],
        default="qwen2.5-3b",
        help="Model to use (default: qwen2.5-3b). Llama models require HuggingFace login.",
    )
    p.add_argument(
        "--methods", nargs="+",
        default=["baseline", "repetition"],
        choices=["baseline", "repetition", "cross_lingual_t1", "cross_lingual_t2"],
        help="Methods to evaluate (default: baseline repetition)",
    )
    p.add_argument("--tasks", nargs="+", default=None,
        help="Filter to specific task(s)")
    p.add_argument("--dry-run", action="store_true",
        help="No model calls — verify prompts build correctly")

    args = p.parse_args()

    # Verify GPU
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from experiment_runner import run_experiment, LOCAL_MODELS

    # Language selection
    all_langs = ["pa", "ur", "ps", "ar", "fa", "sd"]
    langs = all_langs if args.language == "all" else [args.language]

    from run_punjabi import _select_scenarios

    for lang in langs:
        loaders = {
            "pa":  ("Punjabi (Shahmukhi)",    "punjabi_datasets_loader",  "SYSTEM_PROMPT_PA"),
            "ur":  ("Urdu",                   "urdu_datasets_loader",     "SYSTEM_PROMPT_UR"),
            "ps":  ("Pashto",                 "pashto_datasets_loader",   "SYSTEM_PROMPT_PS"),
            "ar":  ("Arabic",                 "arabic_datasets_loader",     "SYSTEM_PROMPT_AR"),
            "fa":  ("Persian (Farsi)",        "persian_datasets_loader",    "SYSTEM_PROMPT_FA"),
            "sd":  ("Sindhi",                 "sindhi_datasets_loader",     "SYSTEM_PROMPT_SD"),
        }
        label, loader_mod, prompt_attr = loaders[lang]
        
        print(f"\n{'='*65}")
        print(f"Running: {label} ({lang}) | Model: {args.model}")
        print(f"{'='*65}\n")

        loader = __import__(loader_mod)
        system_prompt = getattr(sys.modules['experiment_runner'], prompt_attr)

        dataset = loader.build_dataset()
        print(f"  Total loaded: {len(dataset)} items")

        if args.tasks:
            dataset = [it for it in dataset if it.get("task") in set(args.tasks)]
            print(f"  After --tasks {args.tasks}: {len(dataset)} items")

        scenarios = _select_scenarios(dataset)
        if not scenarios:
            print("  ERROR: no scenarios could be determined")
            continue

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = f"results_{lang}_{args.model}_{stamp}.json"
        output_csv = f"results_{lang}_{args.model}_{stamp}.csv"

        run_experiment(
            models=[args.model],
            dataset=dataset,
            output_file=output_json,
            csv_file=output_csv,
            scenarios=scenarios,
            system_prompt=system_prompt,
            methods=args.methods,
            verbose=True,
            dry_run=args.dry_run,
        )

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
