#!/usr/bin/env python3
"""Render SHA-bound tables and SVGs for the terminal C4 negative result."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


NEGATIVE_SCHEMA = "tempo-pd-c4-negative-conclusion-v1"
PHASE_SCHEMA = "tempo-pd-c4-phase-screen-analysis-v2"
ARMS = ("local", "remote", "predictor", "tempo")
ARM_COLORS = {
    "local": "#4c78a8",
    "remote": "#f58518",
    "predictor": "#72b7b2",
    "tempo": "#e45756",
}
PHASE_LABELS = {
    "c0_cool": "C0 cool",
    "c1_decoder_hot": "C1 D-hot",
    "c2_remote_hot": "C2 remote-hot",
    "c2_kv_remote_hot": "C2 KV-hot",
    "c3_both_hot": "C3 both-hot",
    "recovery": "recovery",
}
PHASE_ORDER = tuple(PHASE_LABELS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"{name} is missing")
    _require(
        len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        f"{name} SHA must be canonical",
    )
    _require(_sha256(path) == expected_sha256, f"{name} digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _format_value(value: float, *, percent: bool) -> str:
    if percent:
        return f"{value * 100:.1f}%"
    if value >= 1000:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _render_panel(
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    categories: tuple[str, ...],
    series: tuple[tuple[str, str, tuple[float, ...]], ...],
    percent: bool,
) -> list[str]:
    left, right, top, bottom = 64, 16, 40, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for _, _, row in series for value in row]
    _require(bool(values) and all(value >= 0 for value in values),
             f"invalid values for panel {title}")
    maximum = max(values) or 1.0
    ceiling = maximum * 1.18
    lines = [
        f'<g transform="translate({x0},{y0})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        'rx="8" fill="#ffffff" stroke="#d9dde3"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
        f'class="panel-title">{escape(title)}</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        y = top + plot_height * (1 - fraction)
        label = _format_value(ceiling * fraction, percent=percent)
        lines.extend((
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
            f'y2="{y:.2f}" stroke="#e8ebef"/>',
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="tick">{escape(label)}</text>',
        ))
    group_width = plot_width / len(categories)
    bar_gap = 3
    bars_width = group_width * 0.72
    bar_width = min(44.0, (bars_width - bar_gap * (len(series) - 1)) / len(series))
    occupied = bar_width * len(series) + bar_gap * (len(series) - 1)
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        start = center - occupied / 2
        for series_index, (_, color, row) in enumerate(series):
            value = row[category_index]
            if value == 0:
                continue
            bar_height = plot_height * value / ceiling
            x = start + series_index * (bar_width + bar_gap)
            y = top + plot_height - bar_height
            lines.extend((
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" rx="2"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{max(top + 10, y - 4):.2f}" '
                f'text-anchor="middle" class="value">'
                f'{escape(_format_value(value, percent=percent))}</text>',
            ))
        lines.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 20}" '
            f'text-anchor="middle" class="category">{escape(category)}</text>')
    legend_x = left
    legend_y = height - 12
    for name, color, _ in series:
        lines.extend((
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" '
            f'fill="{color}"/>',
            f'<text x="{legend_x + 14}" y="{legend_y}" class="legend">'
            f'{escape(name)}</text>',
        ))
        legend_x += 30 + len(name) * 7
    lines.append("</g>")
    return lines


def _render_svg(
    *, title: str,
    panels: tuple[dict[str, Any], ...],
    output: Path,
) -> None:
    _require(len(panels) == 4, "report SVG requires four panels")
    width, height = 1240, 820
    panel_width, panel_height = 590, 350
    origins = ((20, 75), (630, 75), (20, 445), (630, 445))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: DejaVu Sans, Arial, sans-serif; fill: #20242a; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".panel-title { font-size: 15px; font-weight: 700; }",
        ".tick,.category,.legend { font-size: 11px; }",
        ".value { font-size: 9px; font-weight: 600; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f5f7fa"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">'
        f'{escape(title)}</text>',
    ]
    for origin, panel in zip(origins, panels, strict=True):
        lines.extend(_render_panel(
            x0=origin[0], y0=origin[1], width=panel_width,
            height=panel_height, **panel))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_table(negative: dict[str, Any]) -> list[str]:
    lines = [
        "| Candidate | Mechanism | Fixed median gain | Predictor median gain | "
        "Goodput gain | Paired wins | TPOT p99 regression | Worst regression |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in negative["candidates"]:
        lines.append(
            f"| {row['label']} | `{row['mechanism']}` | "
            f"{row['e2e_median_gain_vs_fixed'] * 100:+.2f}% | "
            f"{row['e2e_median_gain_vs_predictor'] * 100:+.2f}% | "
            f"{row['goodput_gain_vs_fixed'] * 100:+.2f}% | "
            f"{row['paired_win_fraction_vs_fixed'] * 100:.2f}% | "
            f"{row['tpot_p99_regression_vs_fixed'] * 100:+.2f}% | "
            f"{row['worst_paired_e2e_regression_ms']:+.1f} ms |")
    return lines


def _phase_table(phase: dict[str, Any]) -> list[str]:
    rows = phase["tempo_vs_strongest_fixed"]["by_phase"]
    lines = [
        "| Workload | Fixed E2E median | TEMPO E2E median | Paired wins | "
        "Fixed/TEMPO TPOT p99 | Fixed/TEMPO goodput |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in PHASE_ORDER:
        row = rows[key]
        baseline = row["baseline"]
        candidate = row["candidate"]
        lines.append(
            f"| {PHASE_LABELS[key]} | {baseline['e2e_median_ms']:.1f} ms | "
            f"{candidate['e2e_median_ms']:.1f} ms | "
            f"{row['paired_win_fraction'] * 100:.1f}% | "
            f"{baseline['tpot_p99_ms']:.1f}/{candidate['tpot_p99_ms']:.1f} ms | "
            f"{baseline['goodput_fraction'] * 100:.1f}%/"
            f"{candidate['goodput_fraction'] * 100:.1f}% |")
    return lines


def _write_markdown(
    *,
    negative: dict[str, Any],
    negative_path: Path,
    phase: dict[str, Any],
    phase_path: Path,
    output: Path,
) -> None:
    stop = negative["stop_rule"]
    lines = [
        "# TEMPO C4 frozen negative conclusion",
        "",
        "All three candidates passed the live correctness/data-plane checks, "
        "but none jointly passed the original 10% median and tail bundle. "
        "All three diagnostic phase-oracle policies also failed the full gate. "
        "The preregistered stop condition is therefore satisfied without "
        "weakening a threshold.",
        "",
        "## Candidate summary",
        "",
        *_candidate_table(negative),
        "",
        "## Candidate C workload groups",
        "",
        *_phase_table(phase),
        "",
        "## Verdict",
        "",
        f"- Independent mechanisms: `{stop['independent_candidate_mechanisms_exact']}`",
        f"- Median+tail joint passes: `{stop['median_and_tail_joint_pass_count']}`",
        f"- Full phase-oracle passes: "
        f"`{stop['diagnostic_phase_oracle_full_gate_pass_count']}`",
        f"- Reproducible negative conclusion allowed: "
        f"`{stop['reproducible_negative_conclusion_allowed']}`",
        "- Scope: dynamic contention admission/routing on the frozen four-node "
        "C4 workload with unchanged vLLM/LMCache P/D data plane.",
        "- Not claimed: universal LMCache inferiority, a physical switch "
        "bottleneck, or impossibility of production-scale orchestration.",
        "",
        "## Bound inputs",
        "",
        f"- Negative analysis: `{negative_path}` "
        f"(`{_sha256(negative_path)}`)",
        f"- Candidate C phase analysis: `{phase_path}` "
        f"(`{_sha256(phase_path)}`)",
        "- Plots: `candidate_c_pooled_metrics.svg`, "
        "`candidate_c_phase_metrics.svg`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(
    *,
    negative_path: Path,
    negative_sha256: str,
    phase_path: Path,
    phase_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    negative_path = negative_path.resolve()
    phase_path = phase_path.resolve()
    negative = _load_bound(
        negative_path, negative_sha256, name="negative analysis")
    phase = _load_bound(phase_path, phase_sha256, name="phase analysis")
    _require(
        negative.get("schema") == NEGATIVE_SCHEMA
        and negative.get("stop_rule", {}).get(
            "reproducible_negative_conclusion_allowed") is True,
        "negative verdict differs",
    )
    _require(
        phase.get("schema") == PHASE_SCHEMA
        and phase.get("live_screen_correctness_pass") is True
        and phase.get("performance_claim_allowed") is False,
        "phase verdict differs",
    )
    candidate_c = negative["candidates"][2]
    _require(
        candidate_c["label"] == "C"
        and Path(candidate_c["analysis"]).resolve() == phase_path
        and candidate_c["analysis_sha256"] == phase_sha256,
        "Candidate C phase binding differs",
    )
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), "refusing to overwrite report directory")
    output_dir.mkdir(parents=True)

    pooled = phase["pooled_arm_metrics"]
    pooled_categories = tuple(arm.upper() for arm in ARMS)
    pooled_panels = []
    for title, metric, percent in (
        ("Pooled E2E median (ms)", "e2e_median_ms", False),
        ("Pooled TTFT median (ms)", "ttft_median_ms", False),
        ("Pooled TPOT p99 (ms)", "tpot_p99_ms", False),
        ("Pooled request goodput", "goodput_fraction", True),
    ):
        pooled_panels.append({
            "title": title,
            "categories": pooled_categories,
            "series": tuple(
                (arm.upper(), ARM_COLORS[arm], tuple(
                    pooled[item][metric] if item == arm else 0.0
                    for item in ARMS))
                for arm in ARMS),
            "percent": percent,
        })
    pooled_svg = output_dir / "candidate_c_pooled_metrics.svg"
    _render_svg(
        title="Candidate C: pooled four-arm metrics",
        panels=tuple(pooled_panels), output=pooled_svg)

    phase_rows = phase["tempo_vs_strongest_fixed"]["by_phase"]
    phase_categories = tuple(PHASE_LABELS[key] for key in PHASE_ORDER)
    phase_panels = []
    for title, metric, percent in (
        ("E2E median by workload", "e2e_median_ms", False),
        ("TTFT median by workload", "ttft_median_ms", False),
        ("TPOT p99 by workload", "tpot_p99_ms", False),
        ("Request goodput by workload", "goodput_fraction", True),
    ):
        phase_panels.append({
            "title": title,
            "categories": phase_categories,
            "series": (
                ("strongest fixed (remote)", ARM_COLORS["remote"], tuple(
                    phase_rows[key]["baseline"][metric] for key in PHASE_ORDER)),
                ("TEMPO", ARM_COLORS["tempo"], tuple(
                    phase_rows[key]["candidate"][metric] for key in PHASE_ORDER)),
            ),
            "percent": percent,
        })
    phase_svg = output_dir / "candidate_c_phase_metrics.svg"
    _render_svg(
        title="Candidate C: strongest-fixed versus TEMPO by workload",
        panels=tuple(phase_panels), output=phase_svg)

    report = output_dir / "negative_conclusion_report.md"
    _write_markdown(
        negative=negative, negative_path=negative_path,
        phase=phase, phase_path=phase_path, output=report)
    outputs = [report, pooled_svg, phase_svg]
    preview_converter = shutil.which("magick") or shutil.which("convert")
    if preview_converter is not None:
        for svg_path in (pooled_svg, phase_svg):
            png_path = svg_path.with_suffix(".png")
            subprocess.run(
                [preview_converter, str(svg_path), str(png_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            outputs.append(png_path)
    renderer = Path(__file__).resolve()
    manifest = {
        "schema": "tempo-pd-c4-negative-report-manifest-v1",
        "renderer": str(renderer),
        "renderer_sha256": _sha256(renderer),
        "negative_analysis": str(negative_path),
        "negative_analysis_sha256": negative_sha256,
        "phase_analysis": str(phase_path),
        "phase_analysis_sha256": phase_sha256,
        "preview_converter": preview_converter,
        "outputs": {
            path.name: _sha256(path)
            for path in outputs
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-analysis", type=Path, required=True)
    parser.add_argument("--negative-analysis-sha256", required=True)
    parser.add_argument("--phase-analysis", type=Path, required=True)
    parser.add_argument("--phase-analysis-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render(
        negative_path=args.negative_analysis,
        negative_sha256=args.negative_analysis_sha256,
        phase_path=args.phase_analysis,
        phase_sha256=args.phase_analysis_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "output_dir": str(args.output_dir.resolve()),
        "outputs": manifest["outputs"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
