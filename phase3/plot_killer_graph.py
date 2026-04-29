#!/usr/bin/env python3
"""
phase3/plot_killer_graph.py — Publication-Quality "Killer Graph" for TEMPO Paper

Generates the central figure of the paper:
  X-axis : Training step
  Y-axis : NCCL All-Reduce algorithmic bandwidth (GB/s)
  Lines  : (1) Baseline — interference-free
           (2) Contention — greedy checkpoint flush (40% BW drop)
           (3) TEMPO — paced flush (BW stays flat near baseline)

Input : CSV files produced by train_llm_profiling.py / train_with_tempo.py
        Columns: step, latency_ms, algbw_GBs, busbw_GBs, timestamp

Output: results/figures/killer_graph.pdf  (camera-ready)
        results/figures/killer_graph.png  (high-res raster, 300 DPI)

Usage:
  python phase3/plot_killer_graph.py \\
      --baseline-csv   results/baseline/nccl_bw_rank0.csv \\
      --contention-csv results/contention/nccl_bw_rank0.csv \\
      --tempo-csv      results/tempo/nccl_bw_rank0.csv \\
      --output-dir     results/figures

  # Demo mode (no real data required — generates synthetic data):
  python phase3/plot_killer_graph.py --demo
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Use non-interactive Matplotlib backend (works without a display on compute nodes)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Plot TEMPO Killer Graph")
    p.add_argument("--baseline-csv",   type=str, default=None)
    p.add_argument("--contention-csv", type=str, default=None)
    p.add_argument("--tempo-csv",      type=str, default=None)
    p.add_argument("--output-dir",     type=str, default="results/figures")
    p.add_argument("--title",          type=str,
                   default="TEMPO: Pacing Eliminates PCIe Contention on Perlmutter")
    p.add_argument("--ckpt-every",     type=int, default=50)
    p.add_argument("--bw-metric",      choices=["algbw_GBs", "busbw_GBs"],
                   default="algbw_GBs")
    p.add_argument("--smooth",         type=int, default=5)
    p.add_argument("--demo",           action="store_true")
    p.add_argument("--no-tempo",       action="store_true")
    return p.parse_args()


# ============================================================================
# CSV loading
# ============================================================================

def load_csv(path: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    steps, bw = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            bw.append(float(row[metric]))
    return np.array(steps), np.array(bw)


# ============================================================================
# Synthetic demo data
# ============================================================================

def generate_demo_data(
    num_steps:  int = 400,
    ckpt_every: int = 50,
    seed:       int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng   = np.random.default_rng(seed)
    steps = np.arange(num_steps)

    baseline = 24.2 + rng.normal(0, 0.25, num_steps)
    baseline = np.clip(baseline, 20, 28)

    contention = baseline.copy() + rng.normal(0, 0.15, num_steps)
    flush_duration = 30

    for ckpt_step in range(0, num_steps, ckpt_every):
        flush_end = min(ckpt_step + flush_duration, num_steps)
        for i in range(ckpt_step, flush_end):
            severity = 1.0 - 0.45 * np.exp(-(i - ckpt_step) / 5.0)
            contention[i] *= severity
            contention[i] += rng.normal(0, 0.5)

    contention = np.clip(contention, 8, 28)

    tempo = 23.8 + rng.normal(0, 0.30, num_steps)
    for ckpt_step in range(0, num_steps, ckpt_every):
        if ckpt_step < num_steps:
            tempo[ckpt_step] -= rng.uniform(0.2, 0.5)
    tempo = np.clip(tempo, 20, 28)

    return steps, baseline, contention, tempo


# ============================================================================
# Rolling average helper
# ============================================================================

def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


# ============================================================================
# Plot styling — IEEE two-column format
# ============================================================================

STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.4,
    "lines.antialiased": True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}

COLOR = {
    "baseline":   "#1f77b4",
    "contention": "#d62728",
    "tempo":      "#2ca02c",
    "shade":      "#d62728",
}


# ============================================================================
# Main plotting function
# ============================================================================

def plot_killer_graph(
    steps_bl: np.ndarray, bw_bl: np.ndarray,
    steps_ct: np.ndarray, bw_ct: np.ndarray,
    steps_tp: Optional[np.ndarray], bw_tp: Optional[np.ndarray],
    ckpt_every:  int,
    smooth_w:    int,
    title:       str,
    output_dir:  Path,
    no_tempo:    bool,
) -> None:

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 3.2))

        bw_bl_s = rolling_mean(bw_bl, smooth_w)
        bw_ct_s = rolling_mean(bw_ct, smooth_w)

        flush_dur = 30
        first_zone = True
        for ckpt_step in range(0, int(steps_ct.max()) + 1, ckpt_every):
            zone_end = min(ckpt_step + flush_dur, int(steps_ct.max()))
            label = "Checkpoint flush active" if first_zone else None
            ax.axvspan(ckpt_step, zone_end,
                       alpha=0.08, color=COLOR["shade"],
                       label=label, zorder=1)
            first_zone = False

        ax.plot(steps_bl, bw_bl, color=COLOR["baseline"],   alpha=0.18, lw=0.6, zorder=2)
        ax.plot(steps_ct, bw_ct, color=COLOR["contention"], alpha=0.18, lw=0.6, zorder=2)

        ax.plot(steps_bl, bw_bl_s, color=COLOR["baseline"],   lw=2.0, zorder=3,
                label=f"Baseline  (no I/O,  avg {bw_bl.mean():.1f} GB/s)")
        ax.plot(steps_ct, bw_ct_s, color=COLOR["contention"], lw=2.0, zorder=3,
                label=f"Contention (greedy flush, avg {bw_ct.mean():.1f} GB/s)")

        if not no_tempo and steps_tp is not None and bw_tp is not None:
            bw_tp_s = rolling_mean(bw_tp, smooth_w)
            ax.plot(steps_tp, bw_tp,   color=COLOR["tempo"], alpha=0.18, lw=0.6, zorder=2)
            ax.plot(steps_tp, bw_tp_s, color=COLOR["tempo"], lw=2.0, zorder=3,
                    label=f"TEMPO      (paced flush,  avg {bw_tp.mean():.1f} GB/s)")

        common_len = min(len(bw_bl_s), len(bw_ct_s))
        if common_len > 0:
            worst_idx    = int(np.argmin(bw_ct_s[:common_len] - bw_bl_s[:common_len]))
            worst_step   = steps_ct[worst_idx]
            worst_bl_bw  = bw_bl_s[worst_idx]
            worst_ct_bw  = bw_ct_s[worst_idx]
            degradation  = (worst_bl_bw - worst_ct_bw) / worst_bl_bw * 100

            ax.annotate(
                f"−{degradation:.0f}% BW",
                xy     = (worst_step, worst_ct_bw),
                xytext = (worst_step + max(5, len(steps_ct) * 0.06),
                          worst_ct_bw + (worst_bl_bw - worst_ct_bw) * 0.35),
                fontsize   = 8,
                fontweight = "bold",
                color      = COLOR["contention"],
                arrowprops = dict(arrowstyle="-|>",
                                  color=COLOR["contention"],
                                  lw=1.2,
                                  connectionstyle="arc3,rad=0.15"),
                zorder = 5,
            )

        x_max = max(steps_bl.max() if len(steps_bl) else 0,
                    steps_ct.max() if len(steps_ct) else 0)
        for s in range(0, int(x_max) + 1, ckpt_every):
            ax.axvline(s, color="gray", lw=0.5, ls="--", alpha=0.4, zorder=1)

        ax.set_xlabel("Training Step", labelpad=4)
        ax.set_ylabel("NCCL All-Reduce\nAlgorithmic Bandwidth (GB/s)", labelpad=4)
        ax.set_title(title, pad=6, fontweight="bold")

        y_min = min(bw_ct.min() * 0.88, 0)
        y_max = max(bw_bl.max() * 1.10, 30)
        ax.set_ylim(max(0, y_min), y_max)
        ax.set_xlim(left=0)

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        handles, labels = ax.get_legend_handles_labels()
        flush_patch = mpatches.Patch(
            facecolor=COLOR["shade"], alpha=0.2, label="Checkpoint flush window"
        )
        ax.legend(handles=[flush_patch] + handles,
                  labels=["Checkpoint flush window"] + labels,
                  loc="lower left", ncol=1,
                  handlelength=1.5, columnspacing=1.0)

        info = ("NERSC Perlmutter  |  4 nodes × 4× A100  |  "
                "4× Slingshot 11 NIC/node  |  Lustre $PSCRATCH")
        ax.text(0.99, 0.02, info,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=6.5, color="gray",
                style="italic")

        fig.tight_layout(pad=0.5)

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "killer_graph.pdf"
        png_path = output_dir / "killer_graph.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

        print(f"[plot] Saved PDF: {pdf_path}")
        print(f"[plot] Saved PNG: {png_path}")

        print("\n  Bandwidth Summary")
        print(f"  {'Scenario':<22} {'Mean (GB/s)':>12}  {'Min (GB/s)':>10}  {'Max (GB/s)':>10}")
        print(f"  {'-'*58}")
        print(f"  {'Baseline':<22} {bw_bl.mean():12.2f}  {bw_bl.min():10.2f}  {bw_bl.max():10.2f}")
        print(f"  {'Contention':<22} {bw_ct.mean():12.2f}  {bw_ct.min():10.2f}  {bw_ct.max():10.2f}")
        if not no_tempo and bw_tp is not None:
            print(f"  {'TEMPO':<22} {bw_tp.mean():12.2f}  {bw_tp.min():10.2f}  {bw_tp.max():10.2f}")
            improvement = (bw_tp.mean() - bw_ct.mean()) / bw_ct.mean() * 100
            print(f"\n  TEMPO improves over Contention by +{improvement:.1f}%")


# ============================================================================
# Entry point
# ============================================================================

def main():
    args = parse_args()
    metric     = args.bw_metric
    output_dir = Path(args.output_dir)

    if args.demo:
        print("[plot] Demo mode — generating synthetic data")
        steps_bl, bw_bl, bw_ct, bw_tp = generate_demo_data(
            num_steps=400, ckpt_every=args.ckpt_every
        )
        steps_ct = steps_bl.copy()
        steps_tp = steps_bl.copy()
    else:
        missing = []
        if not args.baseline_csv:
            missing.append("--baseline-csv")
        if not args.contention_csv:
            missing.append("--contention-csv")
        if missing:
            print(f"ERROR: {', '.join(missing)} required (or use --demo).",
                  file=sys.stderr)
            sys.exit(1)

        print(f"[plot] Loading baseline   : {args.baseline_csv}")
        steps_bl, bw_bl = load_csv(args.baseline_csv,   metric)

        print(f"[plot] Loading contention : {args.contention_csv}")
        steps_ct, bw_ct = load_csv(args.contention_csv, metric)

        steps_tp = bw_tp = None
        if args.tempo_csv and not args.no_tempo:
            print(f"[plot] Loading TEMPO      : {args.tempo_csv}")
            steps_tp, bw_tp = load_csv(args.tempo_csv, metric)

    plot_killer_graph(
        steps_bl   = steps_bl,
        bw_bl      = bw_bl,
        steps_ct   = steps_ct,
        bw_ct      = bw_ct,
        steps_tp   = steps_tp if not args.no_tempo else None,
        bw_tp      = bw_tp    if not args.no_tempo else None,
        ckpt_every = args.ckpt_every,
        smooth_w   = args.smooth,
        title      = args.title,
        output_dir = output_dir,
        no_tempo   = args.no_tempo,
    )


if __name__ == "__main__":
    main()
