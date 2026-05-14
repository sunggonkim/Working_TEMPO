#!/usr/bin/env python3
"""
eval/ablation/plot_ablation.py — TEMPO Ablation Study Figure Generator
=======================================================================
Reads results/ablation/{mode}/nccl_bw_rank0.csv for all available modes
and produces:
  - results/figures/fig_ablation.pdf  (publication figure)
  - results/figures/fig_ablation.png
  - results/figures/ablation_summary.csv

Figure type: Grouped bar chart
  X-axis: ablation mode (baseline → core → core+P1 → core+P1+P2 → core+P1+P2+P3)
  Y-axis: Mean AllReduce bandwidth at checkpoint steps (GB/s, algo BW)

Two bar groups per mode: at-ckpt-step and non-ckpt-step, to show interference
isolation effect.

Usage:
    python eval/ablation/plot_ablation.py \
        --results-dir results/ablation \
        --output-dir  results/figures
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

ABLATION_MODES = [
    "baseline",
    "core",
    "core_p1",
    "core_p1_p2",
    "core_p1_p2_p3",
]

MODE_LABELS = {
    "baseline":      "Baseline\n(no TEMPO)",
    "core":          "Core\n(phase-gate)",
    "core_p1":       "Core\n+P1 doorbell",
    "core_p1_p2":    "Core\n+P1+P2 NVLink",
    "core_p1_p2_p3": "Core\n+P1+P2+P3\n(full V6)",
}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def partition(rows):
    ckpt = [float(r["algbw_GBs"]) for r in rows if int(r.get("is_ckpt_step", 0))]
    non  = [float(r["algbw_GBs"]) for r in rows if not int(r.get("is_ckpt_step", 0))]
    return ckpt, non


def safe_stats(vals):
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/ablation")
    p.add_argument("--output-dir",  default="results/figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect ────────────────────────────────────────────────────────────────
    data = {}
    available = []
    for mode in ABLATION_MODES:
        csv_path = results_dir / mode / "nccl_bw_rank0.csv"
        if not csv_path.exists():
            print(f"[skip] {mode}: {csv_path} not found", file=sys.stderr)
            continue
        rows = load_csv(csv_path)
        ckpt, non = partition(rows)
        data[mode] = {
            "ckpt_mean": safe_stats(ckpt)[0],
            "ckpt_std":  safe_stats(ckpt)[1],
            "ckpt_n":    len(ckpt),
            "non_mean":  safe_stats(non)[0],
            "non_std":   safe_stats(non)[1],
            "non_n":     len(non),
        }
        available.append(mode)

    if not available:
        print("No ablation CSVs found. Run the ablation job first.", file=sys.stderr)
        return

    # Baseline reference for improvement computation
    b_ref = data.get("baseline", {}).get("ckpt_mean", None)

    # ── Print text summary ─────────────────────────────────────────────────────
    print("\n=== TEMPO Ablation Study: AllReduce BW at Checkpoint Steps ===")
    print(f"{'Mode':<22}  {'@ckpt (GB/s)':>14}  {'@non-ckpt (GB/s)':>17}  {'vs baseline':>12}")
    print("-" * 72)
    for mode in available:
        d = data[mode]
        improvement = ""
        if b_ref and mode != "baseline":
            improvement = f"{(d['ckpt_mean'] - b_ref) / b_ref * 100:+.1f}%"
        print(f"{mode:<22}  {d['ckpt_mean']:>12.3f}    {d['non_mean']:>15.3f}    {improvement:>12}")

    # ── Write summary CSV ──────────────────────────────────────────────────────
    summary_path = output_dir / "ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ablation_mode",
            "ckpt_algbw_mean_GBs", "ckpt_algbw_std_GBs", "ckpt_n",
            "non_ckpt_algbw_mean_GBs", "non_ckpt_algbw_std_GBs", "non_n",
            "ckpt_vs_baseline_pct",
        ])
        w.writeheader()
        for mode in available:
            d = data[mode]
            vs = ((d["ckpt_mean"] - b_ref) / b_ref * 100) if (b_ref and mode != "baseline") else 0.0
            w.writerow({
                "ablation_mode":          mode,
                "ckpt_algbw_mean_GBs":    round(d["ckpt_mean"], 4),
                "ckpt_algbw_std_GBs":     round(d["ckpt_std"],  4),
                "ckpt_n":                 d["ckpt_n"],
                "non_ckpt_algbw_mean_GBs": round(d["non_mean"], 4),
                "non_ckpt_algbw_std_GBs": round(d["non_std"],  4),
                "non_n":                  d["non_n"],
                "ckpt_vs_baseline_pct":   round(vs, 2),
            })
    print(f"\nSummary CSV → {summary_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(9, 5))

        x       = np.arange(len(available))
        width   = 0.35
        labels  = [MODE_LABELS.get(m, m) for m in available]

        ckpt_means = [data[m]["ckpt_mean"] for m in available]
        ckpt_stds  = [data[m]["ckpt_std"]  for m in available]
        non_means  = [data[m]["non_mean"]  for m in available]
        non_stds   = [data[m]["non_std"]   for m in available]

        bars_ckpt = ax.bar(x - width / 2, ckpt_means, width,
                           yerr=ckpt_stds, capsize=4,
                           color="#E05252", label="At checkpoint step",
                           error_kw={"elinewidth": 1.2})
        bars_non  = ax.bar(x + width / 2, non_means, width,
                           yerr=non_stds, capsize=4,
                           color="#4477AA", label="Non-checkpoint step",
                           error_kw={"elinewidth": 1.2})

        # Annotate improvement over baseline on ckpt bars
        if b_ref:
            for i, (m, bm) in enumerate(zip(available, ckpt_means)):
                if m == "baseline":
                    continue
                vs = (bm - b_ref) / b_ref * 100
                ax.text(x[i] - width / 2, bm + max(ckpt_stds) * 0.15,
                        f"{vs:+.0f}%", ha="center", va="bottom",
                        fontsize=8, color="#C04040", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("AllReduce algbw (GB/s)", fontsize=12)
        ax.set_title("TEMPO Ablation Study: Incremental Component Contribution\n"
                     "Perlmutter A100 · 2 nodes · GPT-1B FSDP · ckpt_every=20",
                     fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.35)
        fig.tight_layout()

        for ext in ("pdf", "png"):
            out = output_dir / f"fig_ablation.{ext}"
            fig.savefig(out, dpi=150 if ext == "png" else None)
            print(f"Figure → {out}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available; skipping plot (summary CSV written).")


if __name__ == "__main__":
    main()
