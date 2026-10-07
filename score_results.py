"""
score_results.py — Score raw pr_runner outputs and run the paper's statistics.

Re-scores every stored response with the current experiment_runner.is_correct,
so scoring fixes never require re-running the models. Writes, into --out:
    scored.csv            one row per generation (analysis.py schema + language)
    accuracy_table.csv    per language × model × task × scenario × method
    mcnemar_results.csv   each method vs baseline (McNemar, no correction, p < 0.1)
    wins_summary.csv      significant wins / losses per language × model × method

Usage:
    python score_results.py --results /kaggle/working/results
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis
import experiment_runner as er
from pr_runner import load_dataset, question_names, read_jsonl

PAPER_TASKS = {"ARC", "OpenBookQA", "GSM8K", "NameIndex", "MiddleMatch"}


def load_raw(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((results_dir / "raw").glob("*/*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        sys.exit(f"No results found under {results_dir / 'raw'}")
    df = pd.DataFrame(rows)
    # A key can be in two files (e.g. ar.jsonl and ar.shard0of2.jsonl): a successful
    # row wins over an errored or skipped one, as in pr_runner.merge_rows.
    failed = df["response"].isna()
    if "error" in df:
        failed |= df["error"].notna()
    df = (df.assign(_failed=failed).sort_values("_failed", kind="stable")
            .drop_duplicates(["model", "key"], keep="first").drop(columns="_failed"))
    return df


def add_anchors(df: pd.DataFrame) -> list:
    """MiddleMatch question names per row; rebuilt from the loaders for rows written
    before pr_runner stored them."""
    stored = df["anchors"] if "anchors" in df else [None] * len(df)
    rows = list(zip(df["task"], df["language"], df["item_id"], stored))
    names = {}
    for lang in {l for t, l, _, a in rows if t == "MiddleMatch" and not isinstance(a, list)}:
        for item in load_dataset(lang):
            if item.get("task") == "MiddleMatch":
                names[(lang, item.get("id"))] = question_names(item)
    return [a if isinstance(a, list) else names.get((l, i)) if t == "MiddleMatch" else None
            for t, l, i, a in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="Directory containing raw/<model>/<lang>.jsonl")
    p.add_argument("--out", default=None, help="Output directory (default: <results>/analysis)")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results_dir = Path(args.results)
    out = Path(args.out) if args.out else results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load_raw(results_dir)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    n_skip = int(df["skipped"].notna().sum()) if "skipped" in df else 0
    ok = df["response"].notna()
    if "error" in df:
        ok &= df["error"].isna()
    df = df[ok].copy()
    distractors = df["distractors"] if "distractors" in df else [None] * len(df)
    anchors = add_anchors(df)
    df["is_correct"] = [int(bool(er.is_correct(t, r, a, d if isinstance(d, list) else None, n)))
                        for t, r, a, d, n in zip(df["task"], df["response"], df["correct_answer"],
                                                 distractors, anchors)]
    df = df.drop(columns=["distractors", "anchors"], errors="ignore")
    print(f"{len(df):,} scored generations | {n_err} errored | {n_skip} skipped (prompt too long)")

    df.to_csv(out / "scored.csv", index=False, encoding="utf-8")

    # Accuracy only over items every method answered, so a method whose longer prompt was
    # skipped or errored isn't scored on a different item set (McNemar pairs anyway).
    item = ["model", "language", "task", "scenario", "item_id"]
    paired = (df.groupby(item)["method"].transform("nunique")
              == df.groupby(["model", "language"])["method"].transform("nunique"))
    if (~paired).any():
        print(f"{int((~paired).sum()):,} generations without a counterpart in every method "
              f"left out of the accuracy table")
    df = df[paired]

    acc_parts, mc_parts = [], []
    for lang, ldf in df.groupby("language"):
        acc = analysis.compute_accuracy(ldf)
        acc.insert(0, "language", lang)
        acc_parts.append(acc)
        if ldf["method"].nunique() > 1:
            mc = analysis.compute_mcnemar(ldf)
            mc = mc[mc["n_pairs"] > 0]
            mc.insert(0, "language", lang)
            mc_parts.append(mc)

    acc = pd.concat(acc_parts, ignore_index=True)
    acc.to_csv(out / "accuracy_table.csv", index=False)
    print(f"\nAccuracy table -> {out / 'accuracy_table.csv'}")

    if mc_parts:
        mc = pd.concat(mc_parts, ignore_index=True)
        mc.to_csv(out / "mcnemar_results.csv", index=False)
        mc["win"] = mc["significant"] & (mc["direction"] == "method_better")
        mc["loss"] = mc["significant"] & (mc["direction"] == "baseline_better")
        wins = (mc.groupby(["language", "model", "method"])
                  .agg(tests=("win", "size"), wins=("win", "sum"), losses=("loss", "sum"))
                  .reset_index())
        wins.to_csv(out / "wins_summary.csv", index=False)
        print(f"McNemar (p < {analysis.ALPHA}, no correction) -> {out / 'mcnemar_results.csv'}")
        print("\nSignificant wins / losses vs baseline:")
        print(wins.to_string(index=False))
        total = wins.groupby("method")[["tests", "wins", "losses"]].sum()
        print("\nOverall, all tasks:")
        print(total.to_string())
        # The paper's benchmarks only (CommonSenseQA and ScriptMixed are this project's additions)
        paper = mc[mc["task"].isin(PAPER_TASKS)]
        paper_tot = paper.groupby("method").agg(tests=("win", "size"), wins=("win", "sum"),
                                                losses=("loss", "sum"))
        print(f"\nPaper tasks only {sorted(PAPER_TASKS)} (cf. paper: 47 wins / 70 tests, 0 losses):")
        print(paper_tot.to_string())


if __name__ == "__main__":
    main()
