#!/usr/bin/env python3
"""
Phase 0: Killer Graph Generator — ITL vs KV Eviction I/O
=========================================================
MISSION: Produce the single graph that proves the hypothesis:
   "The exact moment KV eviction I/O spikes → ITL spikes simultaneously"

This is OSDI Figure 1 / Motivation.

Usage:
    python plot_inference_interference.py                          # real data
    python plot_inference_interference.py --demo                   # synthetic demo
    python plot_inference_interference.py --itl itl_profile.csv \\
                                          --io  io_profile.csv  \\
                                          --out results/figures/fig1_interference.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed on compute node
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(os.getenv("PHASE0_OUTPUT", "results/phase0"))
DEFAULT_FIGURES    = Path("results/figures")

DEFAULT_ITL_CSV = DEFAULT_OUTPUT_DIR / "itl_profile.csv"
DEFAULT_IO_CSV  = DEFAULT_OUTPUT_DIR / "io_profile.csv"
DEFAULT_OUT_PDF = DEFAULT_FIGURES   / "fig1_itl_vs_kv_eviction.pdf"
DEFAULT_OUT_PNG = DEFAULT_FIGURES   / "fig1_itl_vs_kv_eviction.png"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLOR_ITL_P50   = "#1a6bb5"   # blue — median latency
COLOR_ITL_P99   = "#c0392b"   # red  — tail latency (the villain)
COLOR_IO        = "#e67e22"   # orange — NVMe write BW
COLOR_SPIKE_BG  = "#fdebd0"   # light orange shaded region during eviction bursts
COLOR_SPIKE_LN  = "#e67e22"

SPIKE_ALPHA     = 0.25
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK  = 10
FONT_SIZE_ANNOT = 9

# ---------------------------------------------------------------------------
# Synthetic demo data (used when --demo flag is set)
# ---------------------------------------------------------------------------
def make_demo_data(duration_s: float = 60.0, dt_s: float = 0.05) -> tuple:
    """
    Generate synthetic ITL and I/O traces that demonstrate the correlation.
    Three eviction bursts are injected at t=15, 30, 45 seconds.
    """
    rng = np.random.default_rng(42)
    t = np.arange(0, duration_s, dt_s)
    n = len(t)

    # ----- I/O trace (NVMe write MB/s) -----
    # Baseline: ~2 MB/s background noise
    io_bw = rng.exponential(scale=2.0, size=n)

    # Inject three eviction bursts
    burst_centers = [15.0, 30.0, 45.0]
    burst_widths  = [3.0,  4.0,  2.5]
    burst_peaks   = [620.0, 850.0, 780.0]

    for center, width, peak in zip(burst_centers, burst_widths, burst_peaks):
        mask = np.abs(t - center) < width
        io_bw[mask] += peak * np.exp(-((t[mask] - center) ** 2) / (2 * (width / 3) ** 2))

    io_df = pd.DataFrame({
        "time_s":          t,
        "nvme_write_mbps": np.clip(io_bw, 0, None),
    })

    # ----- ITL token records -----
    # ~20 tokens/s per request × 64 concurrent → many rows
    token_times = np.sort(rng.uniform(0, duration_s, size=15000))
    # Baseline ITL ~50ms with small noise
    itl_base = rng.exponential(scale=50.0, size=len(token_times))

    # Spike tokens that coincide with eviction bursts (with slight delay — causal lag)
    LAG = 0.2   # 200ms response lag
    for center, width, peak in zip(burst_centers, burst_widths, burst_peaks):
        spike_start = center + LAG
        spike_end   = center + width + LAG
        in_burst = (token_times >= spike_start) & (token_times <= spike_end)
        spike_multiplier = 1.0 + (peak / 100.0) * rng.uniform(0.5, 1.5, size=in_burst.sum())
        itl_base[in_burst] *= spike_multiplier

    itl_df = pd.DataFrame({
        "timestamp_ns": (token_times * 1e9).astype(np.int64),
        "request_id":   [f"req_{i % 64:04d}" for i in range(len(token_times))],
        "token_idx":    np.tile(np.arange(256), len(token_times) // 256 + 1)[: len(token_times)],
        "itl_ms":       np.clip(itl_base, 1.0, None),
    })

    return itl_df, io_df, burst_centers, burst_widths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_itl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_s"] = (df["timestamp_ns"] - df["timestamp_ns"].min()) / 1e9
    return df


def load_io(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_s"] = (df["timestamp_ns"] - df["timestamp_ns"].min()) / 1e9
    return df


# ---------------------------------------------------------------------------
# Synchronize timestamps between the two traces
# ---------------------------------------------------------------------------
def synchronize(itl_df: pd.DataFrame, io_df: pd.DataFrame,
                itl_marker: float = None, io_marker: float = None) -> tuple:
    """
    Align the two time series to a common t=0 origin.
    If marker files exist, use them; otherwise use the first timestamp of each.
    """
    if itl_marker is not None and io_marker is not None:
        # Shift both to wall-clock origin (markers are in seconds since epoch)
        itl_t0_ns = int(itl_marker * 1e9)
        io_t0_ns  = int(io_marker  * 1e9)
        # Common origin = minimum of the two
        t0 = min(itl_t0_ns, io_t0_ns)
        itl_df["time_s"] = (itl_df["timestamp_ns"] - t0) / 1e9
        io_df["time_s"]  = (io_df["timestamp_ns"]  - t0) / 1e9
    # else: already relative from load_itl/load_io; acceptable if experiment ran <1h
    return itl_df, io_df


# ---------------------------------------------------------------------------
# Aggregate ITL into rolling percentiles at fixed bin width
# ---------------------------------------------------------------------------
def aggregate_itl(itl_df: pd.DataFrame, bin_s: float = 0.5) -> pd.DataFrame:
    """
    Bin tokens into time windows and compute P50/P99/P999 per bin.
    Returns DataFrame indexed by bin_center_s.
    """
    t_max = itl_df["time_s"].max()
    bins = np.arange(0, t_max + bin_s, bin_s)
    itl_df = itl_df.copy()
    itl_df["bin"] = pd.cut(itl_df["time_s"], bins=bins, labels=bins[:-1] + bin_s / 2)
    itl_df["bin"] = itl_df["bin"].astype(float)

    agg = itl_df.groupby("bin")["itl_ms"].agg(
        p50=lambda x: np.percentile(x, 50),
        p95=lambda x: np.percentile(x, 95),
        p99=lambda x: np.percentile(x, 99),
        p999=lambda x: np.percentile(x, 99.9),
        count="count",
    ).reset_index()
    agg.rename(columns={"bin": "time_s"}, inplace=True)
    return agg


# ---------------------------------------------------------------------------
# Detect eviction bursts in the I/O trace
# ---------------------------------------------------------------------------
def detect_eviction_bursts(io_df: pd.DataFrame,
                            threshold_factor: float = 5.0,
                            min_gap_s: float = 1.0) -> list:
    """
    Returns list of (start_s, end_s) intervals where NVMe write BW
    exceeds threshold_factor × median.
    """
    median_bw = io_df["nvme_write_mbps"].median()
    if median_bw <= 0:
        return []
    threshold = median_bw * threshold_factor
    above = io_df["nvme_write_mbps"] > threshold

    bursts = []
    in_burst = False
    burst_start = 0.0
    prev_end = -99.0

    for _, row in io_df.iterrows():
        if above[row.name] and not in_burst:
            if row["time_s"] - prev_end > min_gap_s:
                in_burst = True
                burst_start = row["time_s"]
        elif not above[row.name] and in_burst:
            in_burst = False
            prev_end = row["time_s"]
            bursts.append((burst_start, row["time_s"]))

    if in_burst:
        bursts.append((burst_start, io_df["time_s"].max()))

    return bursts


# ---------------------------------------------------------------------------
# The Killer Graph
# ---------------------------------------------------------------------------
def plot_killer_graph(
    itl_agg: pd.DataFrame,
    io_df: pd.DataFrame,
    bursts: list,
    out_pdf: Path,
    out_png: Path,
    args: argparse.Namespace,
) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 4.5))

    # ---- Secondary axis (I/O BW) ----
    ax2 = ax1.twinx()

    # Plot NVMe write BW as shaded area
    ax2.fill_between(
        io_df["time_s"],
        io_df["nvme_write_mbps"],
        alpha=0.35,
        color=COLOR_IO,
        label="NVMe Write BW (KV Eviction I/O)",
    )
    ax2.plot(
        io_df["time_s"],
        io_df["nvme_write_mbps"],
        color=COLOR_IO,
        linewidth=0.8,
        alpha=0.6,
    )
    ax2.set_ylabel("NVMe Write Throughput (MB/s)\n[KV Cache Eviction I/O]",
                   color=COLOR_IO, fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis="y", labelcolor=COLOR_IO, labelsize=FONT_SIZE_TICK)
    ax2.set_ylim(bottom=0)

    # ---- Primary axis (ITL) ----
    ax1.plot(
        itl_agg["time_s"],
        itl_agg["p50"],
        color=COLOR_ITL_P50,
        linewidth=1.5,
        label="ITL P50",
    )
    ax1.plot(
        itl_agg["time_s"],
        itl_agg["p99"],
        color=COLOR_ITL_P99,
        linewidth=2.0,
        linestyle="-",
        label="ITL P99",
    )
    ax1.fill_between(
        itl_agg["time_s"],
        itl_agg["p50"],
        itl_agg["p99"],
        alpha=0.15,
        color=COLOR_ITL_P99,
    )

    ax1.set_xlabel("Time (seconds)", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Inter-Token Latency — ITL (ms)", color=COLOR_ITL_P50, fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis="y", labelcolor=COLOR_ITL_P50, labelsize=FONT_SIZE_TICK)
    ax1.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid(axis="x", which="minor", alpha=0.2, linestyle=":")
    ax1.grid(axis="x", which="major", alpha=0.35)

    # ---- Annotate eviction bursts ----
    for i, (start, end) in enumerate(bursts):
        ax1.axvspan(start, end, color=COLOR_SPIKE_BG, alpha=0.5, zorder=0)
        ax1.axvline(start, color=COLOR_SPIKE_LN, linewidth=1.2, linestyle="--", alpha=0.7)
        # Find peak P99 in this interval
        mask = (itl_agg["time_s"] >= start) & (itl_agg["time_s"] <= end)
        if mask.any():
            peak_itl = itl_agg.loc[mask, "p99"].max()
            peak_t   = itl_agg.loc[mask & (itl_agg["p99"] == peak_itl), "time_s"].values[0]
            ax1.annotate(
                f"Eviction\nburst #{i+1}\nP99={peak_itl:.0f}ms",
                xy=(peak_t, peak_itl),
                xytext=(peak_t + 0.5, peak_itl * 1.05),
                fontsize=FONT_SIZE_ANNOT,
                color=COLOR_ITL_P99,
                arrowprops=dict(arrowstyle="->", color=COLOR_ITL_P99, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_ITL_P99, alpha=0.8),
            )

    # ---- Title and legend ----
    mode_tag = " [SYNTHETIC DEMO]" if args.demo else " [NERSC Perlmutter — A100×4]"
    ax1.set_title(
        f"TEMPO Phase 0: KV Cache Eviction I/O → ITL Spike Correlation{mode_tag}\n"
        r"$\bf{Hypothesis:}$ PCIe contention from KV offload degrades Decode throughput",
        fontsize=FONT_SIZE_TITLE,
        pad=10,
    )

    # Combine legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    eviction_patch = mpatches.Patch(color=COLOR_SPIKE_BG, label="Eviction burst window", alpha=0.8)
    ax1.legend(
        h1 + h2 + [eviction_patch],
        l1 + l2 + ["Eviction burst window"],
        loc="upper left",
        fontsize=FONT_SIZE_ANNOT,
        framealpha=0.9,
    )

    # ---- Stats box ----
    max_p99  = itl_agg["p99"].max()
    base_p99 = itl_agg.loc[~itl_agg["time_s"].apply(
        lambda t: any(s <= t <= e for s, e in bursts)), "p99"].median() if bursts else itl_agg["p99"].median()
    baseline_p99 = base_p99 if not np.isnan(base_p99) else itl_agg["p99"].median()
    amplification = max_p99 / baseline_p99 if baseline_p99 > 0 else 0.0
    max_io   = io_df["nvme_write_mbps"].max()

    stats_text = (
        f"Baseline P99 ITL:  {baseline_p99:.0f} ms\n"
        f"Peak P99 ITL:      {max_p99:.0f} ms\n"
        f"Spike amplification: {amplification:.1f}×\n"
        f"Peak eviction I/O: {max_io:.0f} MB/s"
    )
    ax1.text(
        0.99, 0.97,
        stats_text,
        transform=ax1.transAxes,
        fontsize=FONT_SIZE_ANNOT,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()

    # ---- Save ----
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved PDF → {out_pdf}")
    print(f"[PLOT] Saved PNG → {out_png}")

    # Print summary to stdout (useful in SLURM logs)
    print(f"\n[PLOT] === KILLER GRAPH STATISTICS ===")
    print(f"[PLOT]   Eviction bursts detected : {len(bursts)}")
    print(f"[PLOT]   Baseline P99 ITL         : {baseline_p99:.1f} ms")
    print(f"[PLOT]   Peak P99 ITL             : {max_p99:.1f} ms")
    print(f"[PLOT]   Spike amplification       : {amplification:.1f}×")
    print(f"[PLOT]   Peak NVMe write BW        : {max_io:.0f} MB/s")
    if amplification >= 3.0:
        print(f"[PLOT]   >>> HYPOTHESIS CONFIRMED: {amplification:.1f}× ITL spike during eviction <<<")
    else:
        print(f"[PLOT]   WARNING: amplification only {amplification:.1f}×; try lower gpu_util or higher concurrency")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: Generate ITL vs KV-eviction killer graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--itl",      type=Path, default=DEFAULT_ITL_CSV, help="ITL CSV from workload_injector.py")
    parser.add_argument("--io",       type=Path, default=DEFAULT_IO_CSV,  help="I/O CSV from hardware_monitor.sh")
    parser.add_argument("--out-pdf",  type=Path, default=DEFAULT_OUT_PDF, help="Output PDF path")
    parser.add_argument("--out-png",  type=Path, default=DEFAULT_OUT_PNG, help="Output PNG path")
    parser.add_argument("--bin-s",    type=float, default=0.5,            help="ITL aggregation bin width (seconds)")
    parser.add_argument("--spike-threshold", type=float, default=5.0,
                        help="I/O spike detection: N× median triggers burst annotation")
    parser.add_argument("--demo",     action="store_true",
                        help="Generate synthetic demo data (no CSV files needed)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        print("[PLOT] --demo mode: generating synthetic data to illustrate expected result")
        itl_df, io_df, burst_centers, burst_widths = make_demo_data()
        # Convert to aggregated form
        itl_agg = aggregate_itl(itl_df, bin_s=args.bin_s)
        bursts = [(c - w / 2, c + w / 2) for c, w in zip(burst_centers, burst_widths)]
    else:
        # Load real data
        if not args.itl.exists():
            print(f"[FATAL] ITL CSV not found: {args.itl}", file=sys.stderr)
            print(f"        Run workload_injector.py first, or use --demo", file=sys.stderr)
            sys.exit(1)
        if not args.io.exists():
            print(f"[FATAL] I/O CSV not found: {args.io}", file=sys.stderr)
            print(f"        Run hardware_monitor.sh first, or use --demo", file=sys.stderr)
            sys.exit(1)

        print(f"[PLOT] Loading ITL data from: {args.itl}")
        print(f"[PLOT] Loading I/O  data from: {args.io}")

        itl_df = load_itl(args.itl)
        io_df  = load_io(args.io)

        # Attempt timestamp synchronization via marker files
        itl_marker_path = args.itl.parent / "experiment_start.marker"
        io_marker_path  = args.io.parent  / "monitor_start.marker"
        itl_marker = float(itl_marker_path.read_text()) if itl_marker_path.exists() else None
        io_marker  = float(io_marker_path.read_text())  if io_marker_path.exists()  else None

        if itl_marker and io_marker:
            print(f"[PLOT] Synchronizing timestamps via marker files")
            itl_df, io_df = synchronize(itl_df, io_df, itl_marker, io_marker)
        else:
            print(f"[PLOT] WARN: marker files not found — using relative time (may be misaligned)")

        # Clip to common time range
        t_start = max(itl_df["time_s"].min(), io_df["time_s"].min())
        t_end   = min(itl_df["time_s"].max(), io_df["time_s"].max())
        itl_df = itl_df[(itl_df["time_s"] >= t_start) & (itl_df["time_s"] <= t_end)]
        io_df  = io_df[ (io_df["time_s"]  >= t_start) & (io_df["time_s"]  <= t_end)]

        print(f"[PLOT] Common time window: {t_start:.1f}s → {t_end:.1f}s "
              f"({t_end - t_start:.1f}s total)")
        print(f"[PLOT] ITL records: {len(itl_df):,}  |  I/O samples: {len(io_df):,}")

        itl_agg = aggregate_itl(itl_df, bin_s=args.bin_s)
        bursts  = detect_eviction_bursts(io_df, threshold_factor=args.spike_threshold)
        print(f"[PLOT] Detected {len(bursts)} eviction burst(s): {bursts}")

    plot_killer_graph(itl_agg, io_df, bursts, args.out_pdf, args.out_png, args)


if __name__ == "__main__":
    main()
