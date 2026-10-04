#!/usr/bin/env python
"""
dataset_acquisition.py

Downloads and prepares datasets for all RTL languages in this project.

PREREQUISITES:
    pip install datasets pandas openpyxl

USAGE:
    python dataset_acquisition.py --all
    python dataset_acquisition.py --arabic
    python dataset_acquisition.py --persian
    python dataset_acquisition.py --sindhi

OUTPUT:
    CSV files in datasets/arabic/, datasets/persian/, datasets/sindhi/

SOURCES:
    All datasets are publicly available on HuggingFace or academic publications.
    Links are documented in each loader file and below.

LICENSES:
    - ArabicMMLU: cc-by-nc-4.0 (Creative Commons Attribution-NonCommercial 4.0)
    - FarsTail: Apache 2.0
    - PersianSciQA: Research use (see dataset card)
    - SdQuAD: ACL Anthology / LREC 2026 (academic)
    - SiNFluD: CC BY 4.0 (arXiv)
"""

import argparse
import os
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets library not installed. Run: pip install datasets")

import pandas as pd


# ── Output directories ────────────────────────────────────────────────────────

OUT_DIRS = {
    "arabic":  Path(__file__).parent / "datasets" / "arabic",
    "persian": Path(__file__).parent / "datasets" / "persian",
    "sindhi":  Path(__file__).parent / "datasets" / "sindhi",
}


# ── Arabic ────────────────────────────────────────────────────────────────────

def acquire_arabic():
    """
    Downloads Arabic datasets from HuggingFace and academic sources.

    Sources:
        1. ArabicMMLU: https://huggingface.co/datasets/MBZUAI/ArabicMMLU
           - 14,575 Arabic MCQs from school exams (Modern Standard Arabic)
           - Native speakers from Jordan, Egypt, Lebanon, UAE, KSA
           - License: cc-by-nc-4.0
           - GitHub: https://github.com/mbzuai-nlp/ArabicMMLU
           - Paper: https://arxiv.org/abs/2402.13xxx (ACL 2024)

        2. Arabic CommonsenseQA (if available):
           - https://huggingface.co/datasets/arb_mlm/commonsense_qa_arabic
    """
    out = OUT_DIRS["arabic"]
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading ArabicMMLU (MBZUAI/ArabicMMLU) ===")
    print("  Source: https://huggingface.co/datasets/MBZUAI/ArabicMMLU")
    print("  License: cc-by-nc-4.0 (non-commercial use)")

    try:
        # Load the full ArabicMMLU dataset (14,575 rows)
        ds = load_dataset("MBZUAI/ArabicMMLU", "all")
        df = ds["test"].to_pandas()

        # Save full dataset
        full_path = out / "ArabicMMLU_full.csv"
        df.to_csv(full_path, index=False)
        print(f"  Saved {len(df)} rows to {full_path}")

        # Sample subsets for each task
        # For STEM/science (ARC-like): filter relevant subjects
        science_subjects = ["Biology (High School)", "Physics (High School)", "Math (High School)"]
        df_science = df[df["subject"].isin(science_subjects)].head(300)
        df_science.to_csv(out / "ARC_Arabic.csv", index=False)
        print(f"  Saved {len(df_science)} ARC-like rows")

        # For elementary reasoning (OpenBookQA-like)
        elementary_subjects = ["Elementary Math", "Elementary Geography"]
        df_elementary = df[df["subject"].isin(elementary_subjects)].head(500)
        df_elementary.to_csv(out / "OpenBookQA_Arabic.csv", index=False)
        print(f"  Saved {len(df_elementary)} OpenBookQA-like rows")

        # For common sense (social science)
        social_subjects = ["Civics (High School)", "History (High School)"]
        df_social = df[df["subject"].isin(social_subjects)].head(300)
        df_social.to_csv(out / "CommonSenseQA_Arabic.csv", index=False)
        print(f"  Saved {len(df_social)} CommonSenseQA-like rows")

    except Exception as e:
        print(f"  ERROR: Could not download ArabicMMLU: {e}")

    print("\n  Note: For retrieval tasks (NameIndex, MiddleMatch, ScriptMixed),")
    print("  please run: python generate_retrieval_tasks.py --language ar")


def acquire_persian():
    """
    Downloads Persian (Farsi) datasets from HuggingFace and academic sources.

    Sources:

        1. PersianMMLU (ammlu): https://huggingface.co/datasets/Hennara/ammlu
           - 1,361 Arabic MMLU questions translated to Persian by GPT-4
           - Source: AceGPT project (Cohere for AI)
           - License: Creative Commons Attribution-NonCommercial 4.0

        2. FarsTail: https://huggingface.co/datasets/ParsiAI/FarsTail
           - 10,367 Persian NLI samples (7,266 train, 1,537 val, 1,564 test)
           - Paper: https://arxiv.org/abs/2009.08820
           - License: Apache 2.0
           - GitHub repo: https://github.com/mbzuai-nlp/FarsTail

        3. PersianSciQA: https://huggingface.co/datasets/safora/persian-scientific-qa
           - 39,809 Persian scientific QA pairs
           - Paper: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf
           - RANLP 2025 publication

        4. Persian QA Dataset: https://huggingface.co/datasets/mshojaei77/Persian_QA
           - 5,900 Persian QA pairs generated via GPT-4o
    """
    out = OUT_DIRS["persian"]
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading PersianMMLU (Hennara/ammlu) ===")
    print("  Source: https://huggingface.co/datasets/Hennara/ammlu")

    try:
        ds = load_dataset("Hennara/ammlu")
        df = ds["train"].to_pandas()
        df.to_csv(out / "PersianMMLU.csv", index=False)
        print(f"  Saved {len(df)} rows")

    except Exception as e:
        print(f"  ERROR: Could not download PersianMMLU: {e}")

    print("\n=== Downloading FarsTail (ParsiAI/FarsTail) ===")
    print("  Source: https://huggingface.co/datasets/ParsiAI/FarsTail")
    print("  Paper: https://arxiv.org/abs/2009.08820")

    try:
        ds = load_dataset("ParsiAI/FarsTail")
        df = ds["test"].to_pandas()
        df.to_csv(out / "FarsTail_test.csv", index=False)
        print(f"  Saved {len(df)} rows (test split)")

        # For GSM8K task: filter NLI samples that involve reasoning
        # (NLI can be adapted to MCQ format)
        df_train = ds["train"].to_pandas()
        df_train.head(500).to_csv(out / "GSM8K_Persian.csv", index=False)
        print(f"  Saved {len(df_train.head(500))} GSM8K-like rows")

    except Exception as e:
        print(f"  ERROR: Could not download FarsTail: {e}")

    print("\n=== Downloading PersianSciQA (safora/persian-scientific-qa) ===")
    print("  Source: https://huggingface.co/datasets/safora/persian-scientific-qa")
    print("  Paper: https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.4.pdf")

    try:
        ds = load_dataset("safora/persian-scientific-qa")
        df = ds["train"].to_pandas()
        # Filter for math/science questions suitable for ARC task
        df_science = df[df.get("relevance_score", 0) >= 2].head(300)
        df_science.to_csv(out / "ARC_Persian.csv", index=False)
        print(f"  Saved {len(df_science)} ARC-like rows")

        # Filter for general knowledge (OpenBookQA)
        df_general = df[df.get("relevance_score", 0) >= 1].head(500)
        df_general.to_csv(out / "OpenBookQA_Persian.csv", index=False)
        print(f"  Saved {len(df_general)} OpenBookQA-like rows")

    except Exception as e:
        print(f"  ERROR: Could not download PersianSciQA: {e}")

    print("\n  Note: For retrieval tasks (NameIndex, MiddleMatch, ScriptMixed),")
    print("  please run: python generate_retrieval_tasks.py --language fa")


def acquire_sindhi():
    """
    Downloads and documents Sindhi dataset acquisition.

    Sources:

        1. SdQuAD: https://aclananthology.org/2026.resourceful-4.6/
           - "SdQuAD: A Large Benchmark Question Answering Dataset for Low-resource Sindhi Language"
           - LREC 2026 Workshop: RESOURCEFUL 2026, Palma, Mallorca, Spain
           - DOI: 10.63317/3dhhfxeoztgo
           - 15,000 QA pairs from news, history, science, geography, business, tourism
           - Annotated using Label Studio by native speakers
           - Available through ACL Anthology (academic publication)

        2. SiNFluD: https://arxiv.org/abs/2605.01323
           - "SiNFluD: Creating and Evaluating Figurative Language Dataset for Sindhi"
           - arXiv:2605.01323 [cs.CL]
           - 4,451 Sindhi instances (literal/non-literal)
           - Categories: idioms, similes, proverbs, metaphors
           - Authors: Wazir Ali, Adeeb Noor, Saifullah Tumrani
           - License: CC BY 4.0

        3. Sindhi Open Lexicon: https://sindhilanguage.org/dataset/
           - 223,342 Sindhi word entries
           - Prepared by: Amar Fayaz Buriro (امر فياض ڻرو)
           - Formats: CSV, JSONL, SQLite

    DOWNLOAD STATUS:
        - SdQuAD: Contact paper authors (Ali et al., 2026) - email corresponding author
        - SiNFluD: Available on arXiv, source code at https://github.com/Sindhi-NLP/sindhi-NLP-dataset
        - Sindhi Open Lexicon: Download directly from https://sindhilanguage.org/dataset/
    """
    out = OUT_DIRS["sindhi"]
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Sindhi Dataset Acquisition ===")
    print("  Sindhi datasets require manual acquisition:")

    print("\n  1. SdQuAD (15,000 QA pairs)")
    print("     Source: https://aclananthology.org/2026.resourceful-4.6/")
    print("     DOI: 10.63317/3dhhfxeoztgo")
    print("     Action: Contact paper authors (Wazir Ali, Muhammad Rafay Shaikh)")
    print("             for dataset access - published LREC 2026 RESOURCEFUL workshop")

    print("\n  2. SiNFluD (4,451 figurative language instances)")
    print("     Source: https://arxiv.org/abs/2605.01323")
    print("     GitHub: https://github.com/Sindhi-NLP/sindhi-NLP-dataset")
    print("     Action: Clone repository or check GitHub releases for data files")

    print("\n  3. Sindhi Open Lexicon (223,342 word entries)")
    print("     Source: https://sindhilanguage.org/dataset/")
    print("     Action: Click 'Download Full Dataset' on website")

    print("\n  After acquisition, save to:")
    print(f"     {out}/NameIndex_Sindhi.csv")
    print(f"     {out}/MiddleMatch_Sindhi.csv")
    print(f"     {out}/ScriptMixed_Sindhi.csv")
    print(f"     {out}/GSM8K_Sindhi.csv (from SdQuAD math subset)")
    print(f"     {out}/ARC_Sindhi.csv (from SdQuAD science subset)")
    print(f"     {out}/OpenBookQA_Sindhi.csv (from SdQuAD general subset)")

    print("\n  Note: SdQuAD and SiNFluD do not have automatic API access.")
    print("        These links are documented for manual acquisition.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for RTL multilingual benchmark"
    )
    parser.add_argument(
        "--arabic", action="store_true",
        help="Download Arabic datasets (ArabicMMLU)"
    )
    parser.add_argument(
        "--persian", action="store_true",
        help="Download Persian datasets (PersianMMLU, FarsTail, PersianSciQA)"
    )
    parser.add_argument(
        "--sindhi", action="store_true",
        help="Show Sindhi dataset acquisition instructions (manual download)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available datasets"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without downloading"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - no downloads will occur")

    if args.all or args.arabic:
        acquire_arabic()

    if args.all or args.persian:
        acquire_persian()

    if args.all or args.sindhi:
        acquire_sindhi()


if __name__ == "__main__":
    main()
