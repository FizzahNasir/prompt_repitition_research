"""
analysis.py — Phase 4: Statistical analysis + Figure 1

Reads results.csv (or JSON) produced by experiment_runner.py and outputs:
  1. Accuracy table  : per (model × scenario × method), Δ vs baseline
  2. McNemar test    : p-values for each method vs baseline, per (model × scenario)
  3. Figure 1        : grouped-bar chart (scenarios × methods), one subplot per model
  4. Saved files     : accuracy_table.csv, mcnemar_results.csv, figure1.png

Usage:
    python analysis.py                          # reads results.csv
    python analysis.py --csv my_results.csv
    python analysis.py --json results.json
    python analysis.py --csv r.csv --out figs/  # custom output directory
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib — use non-interactive backend so it works without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS = ["baseline", "repetition", "verbose", "triple", "padding"]

METHOD_COLORS = {
    "baseline":   "#4C72B0",
    "repetition": "#DD8452",
    "verbose":    "#55A868",
    "triple":     "#C44E52",
    "padding":    "#8172B2",
}

# Compact display labels for scenarios (strip numeric prefix)
def _short(sc: str) -> str:
    parts = sc.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1].replace("_", " ")
    return sc.replace("_", " ")


# ── 1. Data Ingestion ─────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load flat CSV written by experiment_runner._append_csv."""
    df = pd.read_csv(path, dtype={"is_correct": float})
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)
    required = {"model", "scenario", "method", "item_id", "is_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def load_json(path: str) -> pd.DataFrame:
    """Convert the nested JSON (from run_experiment) to the flat CSV schema."""
    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for model, scenarios in results.items():
        for sc, items in scenarios.items():
            for item in items:
                for method in METHODS:
                    if method not in item:
                        continue
                    m = item[method]
                    rows.append({
                        "model":         model,
                        "scenario":      sc,
                        "method":        method,
                        "item_id":       item.get("item_id", item.get("id", "?")),
                        "task":          item.get("task", ""),
                        "language":      item.get("language", ""),
                        "correct_answer": item.get("correct_answer", ""),
                        "response":      m.get("response", ""),
                        "is_correct":    int(bool(m.get("correct", False))),
                        "prompt_tokens": m.get("prompt_tokens"),
                        "output_tokens": m.get("output_tokens"),
                        "latency_ms":    m.get("latency_ms"),
                    })
    return pd.DataFrame(rows)


# ── 2. Accuracy Table ─────────────────────────────────────────────────────────

def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        model, task, scenario, method, n, correct, accuracy, delta, win
    where delta = accuracy - baseline_accuracy for that (model, task, scenario).
    """
    if "task" not in df.columns:
        df = df.copy()
        df["task"] = ""

    grp = (
        df.groupby(["model", "task", "scenario", "method"], sort=False)["is_correct"]
        .agg(n="count", correct="sum")
        .reset_index()
    )
    grp["accuracy"] = grp["correct"] / grp["n"]

    # Merge baseline accuracy onto every row
    base = grp[grp["method"] == "baseline"][["model", "task", "scenario", "accuracy"]]
    base = base.rename(columns={"accuracy": "baseline_acc"})
    grp = grp.merge(base, on=["model", "task", "scenario"], how="left")
    grp["delta"] = grp["accuracy"] - grp["baseline_acc"]
    grp["win"]   = grp["delta"] > 0
    grp.drop(columns="baseline_acc", inplace=True)

    # Enforce method order
    grp["method"] = pd.Categorical(grp["method"], categories=METHODS, ordered=True)
    grp = grp.sort_values(["model", "task", "scenario", "method"]).reset_index(drop=True)
    return grp


def print_accuracy_table(acc: pd.DataFrame) -> None:
    """Print a human-readable win/loss table to stdout."""
    COL = 12
    M_COLS = len(METHODS)
    W = 32 + COL * M_COLS

    print("\n" + "=" * W)
    print("ACCURACY TABLE   (D = method - baseline,  + = win)")
    print("=" * W)

    hdr = f"{'Task: Scenario':<30}  " + "  ".join(f"{m[:9]:^{COL}}" for m in METHODS)

    for model, mdf in acc.groupby("model", sort=False):
        print(f"\nModel: {model}")
        print(hdr)
        print("-" * W)
        wins = {m: 0 for m in METHODS[1:]}
        total_sc = 0

        for (task, sc), sdf in mdf.groupby(["task", "scenario"], sort=False):
            label = f"{task}: {_short(sc)}"
            row_str = f"{label[:29]:<30}  "
            method_map = dict(zip(sdf["method"], sdf.itertuples()))
            for m in METHODS:
                if m not in method_map:
                    row_str += f"{'—':^{COL}}  "
                    continue
                r = method_map[m]
                acc_pct = r.accuracy * 100
                if m == "baseline":
                    row_str += f"{acc_pct:>7.1f}%     "
                else:
                    sign = "+" if r.win else " "
                    d    = abs(r.delta) * 100
                    row_str += f"{acc_pct:>7.1f}%{sign}{d:>4.1f} "
                    if r.win:
                        wins[m] += 1
            total_sc += 1
            print(row_str)

        print("-" * W)
        win_row = f"{'Wins vs baseline':<30}  {'—':^{COL}}  "
        win_row += "  ".join(f"{wins[m]:>3}/{total_sc}{'':>6}" for m in METHODS[1:])
        print(win_row)

    print("=" * W + "\n")


# ── 3. McNemar Test ───────────────────────────────────────────────────────────

def _mcnemar_p(b: int, c: int) -> float:
    """
    McNemar's test with continuity correction (Edwards 1948).
    b = baseline correct & method wrong
    c = baseline wrong & method correct
    Returns p-value (two-sided). Returns nan when b+c < 1.
    """
    n = b + c
    if n == 0:
        return float("nan")
    if n < 25:
        # Exact binomial: P(X <= min(b,c)) * 2, X ~ Bin(n, 0.5)
        from scipy.stats import binom
        p = 2 * binom.cdf(min(b, c), n, 0.5)
        return min(p, 1.0)
    # Chi-squared with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2
    return float(chi2.sf(chi2_stat, df=1))


def compute_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, task, scenario, method≠baseline) compute:
        b, c, n_discordant, p_value, significant (α=0.05)
    where b = baseline wins, c = method wins on discordant pairs.
    """
    if "task" not in df.columns:
        df = df.copy()
        df["task"] = ""

    # Pivot so each item has one column per method
    pivot = (
        df.pivot_table(
            index=["model", "task", "scenario", "item_id"],
            columns="method",
            values="is_correct",
            aggfunc="first",
        )
        .reset_index()
    )
    # Ensure all method columns present
    for m in METHODS:
        if m not in pivot.columns:
            pivot[m] = np.nan

    rows = []
    for (model, task, sc), gdf in pivot.groupby(["model", "task", "scenario"]):
        base = gdf["baseline"]
        for m in METHODS[1:]:
            col = gdf[m]
            valid = base.notna() & col.notna()
            b_arr = base[valid].astype(int)
            c_arr = col[valid].astype(int)
            b = int(((b_arr == 1) & (c_arr == 0)).sum())  # baseline correct, method wrong
            c = int(((b_arr == 0) & (c_arr == 1)).sum())  # baseline wrong, method correct
            p = _mcnemar_p(b, c)
            rows.append({
                "model":         model,
                "task":          task,
                "scenario":      sc,
                "method":        m,
                "n_pairs":       int(valid.sum()),
                "b_base_wins":   b,
                "c_method_wins": c,
                "n_discordant":  b + c,
                "p_value":       round(p, 4) if not math.isnan(p) else float("nan"),
                "significant":   (not math.isnan(p)) and (p < 0.05),
                "direction":     "method_better" if c > b else ("baseline_better" if b > c else "tie"),
            })

    return pd.DataFrame(rows)


def print_mcnemar_table(mc: pd.DataFrame) -> None:
    """Print McNemar results, highlighting significant differences."""
    print("\n" + "=" * 88)
    print("McNEMAR TEST   (p < 0.05 => *,  p < 0.01 => **,  p < 0.001 => ***)")
    print("=" * 88)

    task_col = "task" in mc.columns

    for model, mdf in mc.groupby("model", sort=False):
        print(f"\nModel: {model}")
        print(f"  {'Task: Scenario':<35}  {'Method':<12}  {'b':>4}  {'c':>4}  {'p-val':>8}  sig  direction")
        print("  " + "-" * 80)
        for _, row in mdf.iterrows():
            p = row["p_value"]
            if math.isnan(p):
                stars = "n/a"
            elif p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "** "
            elif p < 0.05:
                stars = "*  "
            else:
                stars = "   "
            p_str = f"{p:.4f}" if not math.isnan(p) else "  nan"
            task = row["task"] if task_col else ""
            label = f"{task}: {_short(row['scenario'])}" if task else _short(row["scenario"])
            print(
                f"  {label[:34]:<35}  "
                f"{row['method']:<12}  "
                f"{row['b_base_wins']:>4}  {row['c_method_wins']:>4}  "
                f"{p_str:>8}  {stars}  {row['direction']}"
            )

    print("=" * 88 + "\n")


# ── 4. Figure 1 ───────────────────────────────────────────────────────────────

def plot_figure1(acc: pd.DataFrame, out_path: str = "figure1.png") -> None:
    """
    Grouped bar chart replicating Figure 1 of arXiv:2512.14982.

    Layout  : one subplot row per model
    X-axis  : (task, scenario) pairs — keeps task types separate
    Y-axis  : accuracy (%)
    Bars    : 5 methods, colour-coded
    Annotations: Δ printed above each non-baseline bar (green + / red −)
    """
    models = acc["model"].unique().tolist()
    n_models = len(models)

    task_col = "task" in acc.columns
    if task_col:
        task_scs = (
            acc[["task", "scenario"]]
            .drop_duplicates()
            .sort_values(["task", "scenario"])
            .itertuples(index=False, name=None)
        )
        task_scs = list(task_scs)
    else:
        task_scs = [("", sc) for sc in acc["scenario"].unique().tolist()]

    n_sc = len(task_scs)
    if n_sc == 0 or n_models == 0:
        print("[Figure 1] No data to plot.")
        return

    bar_w   = 0.14
    offsets = np.linspace(-(len(METHODS) - 1) / 2, (len(METHODS) - 1) / 2, len(METHODS)) * bar_w
    x       = np.arange(n_sc)

    fig_w = max(12, n_sc * 1.8)
    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(fig_w, 4.5 * n_models),
        squeeze=False,
    )
    fig.suptitle(
        "Prompt Repetition Benchmark — Punjabi (Shahmukhi)\n"
        "Accuracy by Task × Scenario & Method",
        fontsize=13, fontweight="bold", y=1.01,
    )

    sc_labels = [
        f"{task}\n{_short(sc)}" if task else _short(sc)
        for task, sc in task_scs
    ]

    for row_idx, model in enumerate(models):
        ax  = axes[row_idx, 0]
        mdf = acc[acc["model"] == model]

        for m_idx, method in enumerate(METHODS):
            vals = []
            for task, sc in task_scs:
                mask = (mdf["scenario"] == sc) & (mdf["method"] == method)
                if task_col and task:
                    mask &= mdf["task"] == task
                sub = mdf[mask]
                vals.append(sub["accuracy"].values[0] * 100 if len(sub) else 0.0)

            bars = ax.bar(
                x + offsets[m_idx],
                vals,
                width=bar_w * 0.92,
                color=METHOD_COLORS[method],
                label=method,
                zorder=2,
            )

            # Print Δ above non-baseline bars
            if method != "baseline":
                base_vals = []
                for task, sc in task_scs:
                    mask = (mdf["scenario"] == sc) & (mdf["method"] == "baseline")
                    if task_col and task:
                        mask &= mdf["task"] == task
                    sub = mdf[mask]
                    base_vals.append(sub["accuracy"].values[0] * 100 if len(sub) else 0.0)
                for bar, v, bv in zip(bars, vals, base_vals):
                    delta = v - bv
                    if abs(delta) >= 0.5:
                        color = "#1a7a1a" if delta > 0 else "#c0392b"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f"{delta:+.0f}",
                            ha="center", va="bottom",
                            fontsize=5.5, color=color, fontweight="bold",
                        )

        ax.set_title(f"Model: {model}", fontsize=10, loc="left", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(sc_labels, rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Single legend below all subplots
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], label=m)
        for m in METHODS
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=len(METHODS), fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True, framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved → {out_path}")


# ── 5. Summary Stats ──────────────────────────────────────────────────────────

def print_summary(acc: pd.DataFrame, mc: pd.DataFrame) -> None:
    """Print a high-level summary: wins, significant wins, best method."""
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    for model in acc["model"].unique():
        mdf = acc[(acc["model"] == model) & (acc["method"] != "baseline")]
        mc_m = mc[mc["model"] == model]

        # Count distinct (task, scenario) groups for this model
        total_pairs = len(acc[(acc["model"] == model) & (acc["method"] == "baseline")])

        print(f"\n  Model: {model}   ({total_pairs} scenario×item combos)")
        for method in METHODS[1:]:
            sub  = mdf[mdf["method"] == method]
            wins = int(sub["win"].sum())
            sig  = int(mc_m[(mc_m["method"] == method) & mc_m["significant"]].shape[0])
            avg_delta = sub["delta"].mean() * 100
            print(
                f"    {method:<12}: {wins:>2}/{total_pairs} wins, "
                f"{sig:>2} significant,  avg Δ = {avg_delta:+.1f}%"
            )

    best = (
        acc[acc["method"] != "baseline"]
        .groupby(["model", "method"], observed=True)["delta"]
        .mean()
        .reset_index()
    )
    if not best.empty:
        idx = best.groupby("model")["delta"].idxmax()
        print("\n  Best method per model (avg Δ accuracy):")
        for _, row in best.loc[idx].iterrows():
            print(f"    {row['model']}: {row['method']}  (avg Δ = {row['delta']*100:+.1f}%)")

    print("-" * 60 + "\n")


# ── 6. CLI Entry Point ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prompt Repetition — Phase 4 Analysis")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv",  default="results.csv",  help="Flat CSV from experiment_runner")
    src.add_argument("--json", default=None,            help="Nested JSON from experiment_runner")
    p.add_argument("--out",    default=".",             help="Output directory for saved files")
    p.add_argument("--no-plot", action="store_true",    help="Skip Figure 1 (matplotlib not required)")
    return p.parse_args()


def main():
    import sys, io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.json:
        src = args.json
        if not Path(src).exists():
            sys.exit(f"ERROR: {src} not found.")
        print(f"Loading JSON: {src}")
        df = load_json(src)
    else:
        src = args.csv
        if not Path(src).exists():
            sys.exit(
                f"ERROR: {src} not found.\n"
                "Run experiment_runner.py first, or pass --json <file>."
            )
        print(f"Loading CSV: {src}")
        df = load_csv(src)

    print(f"  {len(df):,} rows  |  "
          f"{df['model'].nunique()} model(s)  |  "
          f"{df['scenario'].nunique()} scenario(s)  |  "
          f"{df['item_id'].nunique()} unique items")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    acc = compute_accuracy(df)
    print_accuracy_table(acc)

    acc_path = out / "accuracy_table.csv"
    acc.to_csv(acc_path, index=False)
    print(f"Accuracy table saved → {acc_path}")

    # ── McNemar ───────────────────────────────────────────────────────────────
    mc = compute_mcnemar(df)
    print_mcnemar_table(mc)

    mc_path = out / "mcnemar_results.csv"
    mc.to_csv(mc_path, index=False)
    print(f"McNemar results saved → {mc_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(acc, mc)

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            fig_path = str(out / "figure1.png")
            plot_figure1(acc, out_path=fig_path)
        except ImportError as e:
            print(f"[Figure 1] Skipped — missing package: {e}")
            print("  Install with: pip install matplotlib")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
