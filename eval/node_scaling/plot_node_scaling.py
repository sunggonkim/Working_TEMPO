#!/usr/bin/env python3
"""
eval/node_scaling/plot_node_scaling.py — Node Scaling Figure Generator
=======================================================================
Reads results/node_scaling/scaling_N{N}_{baseline,tempo}.csv for all
available N values and produces:
  - results/figures/fig_node_scaling.pdf  (publication figure)
  - results/figures/fig_node_scaling.png

X-axis: number of nodes (2, 4, 8, 16, 32)
Y-axis: mean AllReduce latency at checkpoint steps (ms)

Two lines: Baseline (red) vs TEMPO (blue).
Error bars: ±1 std across ckpt-step measurements.

Usage:
    python eval/node_scaling/plot_node_scaling.py \
        --results-dir results/node_scaling \
        --output-dir  results/figures
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def load_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def extract_ckpt_latencies(rows):
    """Return list of AllReduce latencies at checkpoint steps."""
    return [float(r["allreduce_ms"]) for r in rows if int(r.get("is_ckpt_step", 0))]


def extract_non_ckpt_latencies(rows):
    return [float(r["allreduce_ms"]) for r in rows if not int(r.get("is_ckpt_step", 0))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/node_scaling")
    p.add_argument("--output-dir",  default="results/figures")
    p.add_argument("--nodes",       nargs="+", type=int, default=[2, 4, 8, 16, 32])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect data ──────────────────────────────────────────────────────────
    data = {}   # N → {baseline: {mean, std, n}, tempo: {mean, std, n}}
    available = []

    for N in args.nodes:
        b_path = results_dir / f"scaling_N{N}_baseline.csv"
        t_path = results_dir / f"scaling_N{N}_tempo.csv"
        if not b_path.exists() or not t_path.exists():
            print(f"[skip] N={N}: missing CSV(s)", file=sys.stderr)
            continue

        b_rows = load_csv(b_path)
        t_rows = load_csv(t_path)

        b_lats = extract_ckpt_latencies(b_rows)
        t_lats = extract_ckpt_latencies(t_rows)
        b_non  = extract_non_ckpt_latencies(b_rows)
        t_non  = extract_non_ckpt_latencies(t_rows)

        if not b_lats or not t_lats:
            print(f"[skip] N={N}: no ckpt-step rows found", file=sys.stderr)
            continue

        data[N] = {
            "baseline": {
                "ckpt_mean": statistics.mean(b_lats),
                "ckpt_std":  statistics.stdev(b_lats) if len(b_lats) > 1 else 0.0,
                "ckpt_n":    len(b_lats),
                "non_mean":  statistics.mean(b_non) if b_non else 0.0,
            },
            "tempo": {
                "ckpt_mean": statistics.mean(t_lats),
                "ckpt_std":  statistics.stdev(t_lats) if len(t_lats) > 1 else 0.0,
                "ckpt_n":    len(t_lats),
                "non_mean":  statistics.mean(t_non) if t_non else 0.0,
            },
        }
        available.append(N)

    if not available:
        print("No data found. Run scaling jobs first.", file=sys.stderr)
        return

    available.sort()

    # ── Print text summary ────────────────────────────────────────────────────
    print("\n=== Node Scaling: AllReduce Latency at Checkpoint Steps ===")
    print(f"{'N':>5}  {'Baseline (ms)':>16}  {'TEMPO (ms)':>14}  {'Reduction':>10}")
    print("-" * 56)
    for N in available:
        b = data[N]["baseline"]["ckpt_mean"]
        t = data[N]["tempo"]["ckpt_mean"]
        r = (t - b) / b * 100
        print(f"{N:>5}  {b:>14.3f}    {t:>12.3f}    {r:>+9.1f}%")

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = output_dir / "node_scaling_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "n_nodes",
            "baseline_ckpt_mean_ms", "baseline_ckpt_std_ms", "baseline_ckpt_n",
            "baseline_non_ckpt_mean_ms",
            "tempo_ckpt_mean_ms",    "tempo_ckpt_std_ms",    "tempo_ckpt_n",
            "tempo_non_ckpt_mean_ms",
            "reduction_pct",
        ])
        w.writeheader()
        for N in available:
            b = data[N]["baseline"]
            t = data[N]["tempo"]
            r = (t["ckpt_mean"] - b["ckpt_mean"]) / b["ckpt_mean"] * 100
            w.writerow({
                "n_nodes":                   N,
                "baseline_ckpt_mean_ms":     round(b["ckpt_mean"], 4),
                "baseline_ckpt_std_ms":      round(b["ckpt_std"],  4),
                "baseline_ckpt_n":           b["ckpt_n"],
                "baseline_non_ckpt_mean_ms": round(b["non_mean"],  4),
                "tempo_ckpt_mean_ms":        round(t["ckpt_mean"], 4),
                "tempo_ckpt_std_ms":         round(t["ckpt_std"],  4),
                "tempo_ckpt_n":              t["ckpt_n"],
                "tempo_non_ckpt_mean_ms":    round(t["non_mean"],  4),
                "reduction_pct":             round(r, 2),
            })
    print(f"\nSummary CSV → {summary_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(7, 4))

        ns = available
        b_means = [data[N]["baseline"]["ckpt_mean"] for N in ns]
        b_stds  = [data[N]["baseline"]["ckpt_std"]  for N in ns]
        t_means = [data[N]["tempo"]["ckpt_mean"]    for N in ns]
        t_stds  = [data[N]["tempo"]["ckpt_std"]     for N in ns]

        ax.errorbar(ns, b_means, yerr=b_stds, fmt="o-",  color="#E05252",
                    linewidth=2, markersize=7, capsize=4, label="Baseline (greedy flush)")
        ax.errorbar(ns, t_means, yerr=t_stds, fmt="s--", color="#4477AA",
                    linewidth=2, markersize=7, capsize=4, label="TEMPO (paced flush)")

        # annotate reduction % at each point
        for N, bm, tm in zip(ns, b_means, t_means):
            r = (tm - bm) / bm * 100
            ax.annotate(f"{r:+.0f}%", xy=(N, tm), xytext=(4, -14),
                        textcoords="offset points", fontsize=8, color="#4477AA")

        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Number of nodes", fontsize=12)
        ax.set_ylabel("AllReduce latency at ckpt steps (ms)", fontsize=12)
        ax.set_title("TEMPO: AllReduce Latency Scaling\n"
                     "Perlmutter A100 · GPT-1B FSDP · ckpt_every=20",
                     fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()

        for ext in ("pdf", "png"):
            out = output_dir / f"fig_node_scaling.{ext}"
            fig.savefig(out, dpi=150 if ext == "png" else None)
            print(f"Figure → {out}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available; skipping plot (summary CSV written).")


if __name__ == "__main__":
    main()
