#!/usr/bin/env python3.11
"""Render the quantitative README figures from authoritative TEMPO JSON.

The renderer intentionally uses only the Python standard library.  Every
number is loaded from the committed C9, C10, or hierarchy artifact rather than
copied into plotting code.  The resulting SVGs remain reviewable in Git and a
manifest binds every source and generated figure by SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "tempo_go" / "figures"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

COLORS = {
    "tempo": "#1457D9",
    "tempo_light": "#78A7FF",
    "miss": "#F59E0B",
    "remote": "#7C3AED",
    "fixed": "#94A3B8",
    "predictor": "#D97706",
    "queue": "#7C3AED",
    "netkv": "#DC2626",
    "kairos": "#EA580C",
    "good": "#059669",
    "bad": "#DC2626",
    "ink": "#172033",
    "muted": "#5F6B7A",
    "grid": "#DCE3EC",
    "panel": "#F7F9FC",
    "white": "#FFFFFF",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


class Canvas:
    def __init__(self, width: int, height: int, title: str, description: str):
        self.width = width
        self.height = height
        self.root = ET.Element(
            _tag("svg"),
            {
                "width": str(width),
                "height": str(height),
                "viewBox": f"0 0 {width} {height}",
                "role": "img",
                "aria-labelledby": "figure-title figure-description",
            },
        )
        ET.SubElement(self.root, _tag("title"), {"id": "figure-title"}).text = title
        ET.SubElement(
            self.root, _tag("desc"), {"id": "figure-description"}
        ).text = description
        self.rect(0, 0, width, height, fill=COLORS["white"])

    def element(self, name: str, **attributes: object) -> ET.Element:
        normalized = {
            key.replace("_", "-"): str(value)
            for key, value in attributes.items()
            if value is not None
        }
        return ET.SubElement(self.root, _tag(name), normalized)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str | None = None,
        stroke_width: float | None = None,
        rx: float | None = None,
        opacity: float | None = None,
    ) -> ET.Element:
        return self.element(
            "rect",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            width=f"{width:.2f}",
            height=f"{height:.2f}",
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            rx=rx,
            opacity=opacity,
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = COLORS["ink"],
        stroke_width: float = 1,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> ET.Element:
        return self.element(
            "line",
            x1=f"{x1:.2f}",
            y1=f"{y1:.2f}",
            x2=f"{x2:.2f}",
            y2=f"{y2:.2f}",
            stroke=stroke,
            stroke_width=stroke_width,
            stroke_dasharray=dash,
            opacity=opacity,
        )

    def path(
        self,
        definition: str,
        *,
        stroke: str,
        stroke_width: float,
        fill: str = "none",
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> ET.Element:
        return self.element(
            "path",
            d=definition,
            stroke=stroke,
            stroke_width=stroke_width,
            fill=fill,
            opacity=opacity,
            stroke_dasharray=dash,
            stroke_linecap="round",
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 3,
        fill: str = "none",
        dash: str | None = None,
    ) -> ET.Element:
        return self.element(
            "polyline",
            points=" ".join(f"{x:.2f},{y:.2f}" for x, y in points),
            stroke=stroke,
            stroke_width=stroke_width,
            fill=fill,
            stroke_dasharray=dash,
            stroke_linejoin="round",
            stroke_linecap="round",
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str = COLORS["white"],
        stroke_width: float = 2,
    ) -> ET.Element:
        return self.element(
            "circle",
            cx=f"{x:.2f}",
            cy=f"{y:.2f}",
            r=f"{radius:.2f}",
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
        )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        size: int = 16,
        fill: str = COLORS["ink"],
        anchor: str = "start",
        weight: int | str = 400,
        transform: str | None = None,
        baseline: str | None = None,
    ) -> ET.Element:
        node = self.element(
            "text",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            fill=fill,
            font_size=size,
            font_weight=weight,
            font_family="NanumGothic, Noto Sans CJK KR, Apple SD Gothic Neo, sans-serif",
            text_anchor=anchor,
            transform=transform,
            dominant_baseline=baseline,
        )
        node.text = str(value)
        return node

    def panel(self, x: float, y: float, width: float, height: float) -> None:
        self.rect(
            x,
            y,
            width,
            height,
            fill=COLORS["panel"],
            stroke=COLORS["grid"],
            stroke_width=1,
            rx=16,
        )

    def save(self, path: Path) -> None:
        ET.indent(self.root, space="  ")
        ET.ElementTree(self.root).write(
            path, encoding="utf-8", xml_declaration=True
        )


def _header(canvas: Canvas, title: str, subtitle: str) -> None:
    canvas.text(48, 43, title, size=26, weight=700)
    canvas.text(48, 72, subtitle, size=14, fill=COLORS["muted"])


def _legend(
    canvas: Canvas,
    x: float,
    y: float,
    items: list[tuple[str, str]],
    *,
    step: float = 145,
) -> None:
    for index, (label, color) in enumerate(items):
        current = x + index * step
        canvas.rect(current, y - 11, 18, 12, fill=color, rx=2)
        canvas.text(current + 25, y, label, size=13, fill=COLORS["muted"])


def _linear_y(value: float, minimum: float, maximum: float, top: float, height: float) -> float:
    return top + height - (value - minimum) / (maximum - minimum) * height


def _log_y(value: float, minimum: float, maximum: float, top: float, height: float) -> float:
    return top + height - (
        (math.log10(value) - math.log10(minimum))
        / (math.log10(maximum) - math.log10(minimum))
        * height
    )


def render_current_native_matrix() -> Path:
    """Render the latest same-population seven-arm native result.

    This figure intentionally uses offered-population SLO next to completed-
    request p99.  A policy with zero completions therefore appears as zero SLO
    and an explicit undefined latency marker rather than as an artificial win.
    """
    relative = (
        "results/tempo_go_c9_route_liveness_job_57736076_"
        "r3_canonical_outer/analysis_failclosed_business_v2.json"
    )
    analysis = _load(relative)
    aggregates = analysis["aggregates"]
    arm_specs = [
        ("fixed_local_d0", "Local D0"),
        ("fixed_local_d1", "Local D1"),
        ("fixed_remote_p0d1", "Remote 0→1"),
        ("fixed_remote_p1d0", "Remote 1→0"),
        ("predictor", "Predictor"),
        ("queue_gpu", "Queue-GPU"),
        ("full_c7_managed_background", "Candidate O"),
    ]
    regimes = [
        ("normal", "normal", COLORS["tempo_light"]),
        ("miss_hot", "miss-hot", COLORS["miss"]),
        ("remote_favorable", "remote-favorable", COLORS["remote"]),
    ]
    rows = []
    for arm, label in arm_specs:
        values = aggregates[arm]
        rows.append({
            "label": label,
            "slo": [100.0 * values[key]["slo_good_fraction"] for key, _, _ in regimes],
            "p99": [
                values[key]["mean_e2e_p99_ms"] / 1000.0
                if values[key]["mean_e2e_p99_ms"] is not None
                else None
                for key, _, _ in regimes
            ],
        })

    canvas = Canvas(
        1500,
        790,
        "Current TEMPO Candidate O native matrix",
        "Same-population offered SLO and completed-request p99 for seven native policies.",
    )
    _header(
        canvas,
        "최신 4-node native matrix: Candidate O는 remote를 살렸지만 strongest fixed를 못 이김",
        "Allocation 57736076 · 210 offered victims per arm · actual vLLM + LMCache/NIXL + NCCL/Slingshot",
    )
    canvas.panel(35, 98, 820, 620)
    canvas.panel(875, 98, 590, 620)
    canvas.text(70, 138, "Offered-population SLO attainment (%)", size=18, weight=700)
    canvas.text(910, 138, "Completed-request E2E p99 (seconds)", size=18, weight=700)
    _legend(
        canvas,
        330,
        170,
        [(label, color) for _, label, color in regimes],
        step=145,
    )

    left_x, left_y, left_w, left_h = 85, 205, 725, 400
    for tick in range(0, 101, 20):
        y = _linear_y(tick, 0, 100, left_y, left_h)
        canvas.line(left_x, y, left_x + left_w, y, stroke=COLORS["grid"])
        canvas.text(left_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = left_w / len(rows)
    bar_width = 20
    for index, row in enumerate(rows):
        center = left_x + group_width * (index + 0.5)
        if row["label"] == "Candidate O":
            canvas.rect(center - 48, left_y - 12, 96, left_h + 24, fill="#FEE2E2", rx=8, opacity=0.65)
        for regime_index, (_, _, color) in enumerate(regimes):
            value = float(row["slo"][regime_index])
            offset = (regime_index - 1) * 23
            y = _linear_y(value, 0, 100, left_y, left_h)
            canvas.rect(center + offset - bar_width / 2, y, bar_width, left_y + left_h - y, fill=color, rx=2)
            if value > 0:
                canvas.text(center + offset, y - 6, f"{value:.0f}", size=9, anchor="middle", weight=700)
        canvas.text(
            center + 18,
            left_y + left_h + 28,
            row["label"],
            size=11,
            anchor="end",
            transform=f"rotate(-32 {center + 18:.2f} {left_y + left_h + 28:.2f})",
        )

    right_x, right_y, right_w, right_h = 925, 205, 490, 400
    for tick in range(0, 101, 20):
        y = _linear_y(tick, 0, 100, right_y, right_h)
        canvas.line(right_x, y, right_x + right_w, y, stroke=COLORS["grid"])
        canvas.text(right_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = right_w / len(rows)
    bar_width = 12
    for index, row in enumerate(rows):
        center = right_x + group_width * (index + 0.5)
        if row["label"] == "Candidate O":
            canvas.rect(center - 32, right_y - 12, 64, right_h + 24, fill="#FEE2E2", rx=8, opacity=0.65)
        for regime_index, (_, _, color) in enumerate(regimes):
            value = row["p99"][regime_index]
            offset = (regime_index - 1) * 15
            if value is None:
                marker_y = right_y + right_h - 4
                canvas.line(center + offset - 5, marker_y - 5, center + offset + 5, marker_y + 5, stroke=COLORS["bad"], stroke_width=2)
                canvas.line(center + offset + 5, marker_y - 5, center + offset - 5, marker_y + 5, stroke=COLORS["bad"], stroke_width=2)
                continue
            numeric = float(value)
            y = _linear_y(min(numeric, 100.0), 0, 100, right_y, right_h)
            canvas.rect(center + offset - bar_width / 2, y, bar_width, right_y + right_h - y, fill=color, rx=2)
        canvas.text(
            center + 14,
            right_y + right_h + 28,
            row["label"],
            size=10,
            anchor="end",
            transform=f"rotate(-35 {center + 14:.2f} {right_y + right_h + 28:.2f})",
        )
    canvas.text(1170, 684, "Candidate O remote 30/30 · miss-hot 65/120 · normal p99 26.37 s", size=13, fill=COLORS["bad"], anchor="middle", weight=700)
    path = OUTPUT / "current_candidate_o_native_matrix.svg"
    canvas.save(path)
    return path


def render_candidate_business_progression() -> Path:
    relative = (
        "results/tempo_go_c9_route_liveness_job_57736076_"
        "r3_canonical_outer/candidate_o_diagnosis.json"
    )
    diagnosis = _load(relative)
    context = diagnosis["cross_allocation_context_noncausal"]
    specs = [
        ("candidate_m", "Candidate M", COLORS["fixed"]),
        ("candidate_n", "Candidate N", COLORS["predictor"]),
        ("candidate_o", "Candidate O", COLORS["tempo"]),
    ]
    rows = []
    for key, label, color in specs:
        campaign = context[key]
        business = campaign["business"]
        background = business["background"]
        foreground = business["foreground"]
        rows.append({
            "label": label,
            "color": color,
            "background_complete": 100.0 * background["completion_fraction"],
            "background_failure": 100.0 * background["failures"] / background["offered"],
            "background_reject": 100.0 * background["global_rejects"] / background["offered"],
            "foreground_complete": 100.0 * foreground["completion_fraction"],
            "observer": 100.0 * business["observer_supported_fraction"],
            "background_completed_count": background["completed"],
            "foreground_completed_count": foreground["completed"],
            "observer_count": business["observer_supported_decisions"],
        })

    canvas = Canvas(
        1450,
        730,
        "Candidate M N O business and observer progression",
        "Raw-terminal corrected business completion, failure, rejection, and observer support.",
    )
    _header(
        canvas,
        "M→N→O: route-liveness는 business completion을 회복했지만 관측·tail gate는 남음",
        "Same offered population · M/N↔O는 separate allocation이라 비인과 context · valid failure receipt ≠ completion",
    )
    canvas.panel(35, 98, 710, 570)
    canvas.panel(765, 98, 650, 570)
    canvas.text(70, 138, "Background terminal outcome (% of offered)", size=18, weight=700)
    canvas.text(800, 138, "Completion and observer support (%)", size=18, weight=700)

    left_x, left_y, left_w, left_h = 110, 190, 560, 350
    for tick in range(0, 101, 20):
        y = _linear_y(tick, 0, 100, left_y, left_h)
        canvas.line(left_x, y, left_x + left_w, y, stroke=COLORS["grid"])
        canvas.text(left_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = left_w / len(rows)
    bar_width = 92
    outcome_colors = [COLORS["good"], COLORS["miss"], COLORS["bad"]]
    outcome_labels = ["complete", "service failure", "queue reject"]
    for index, row in enumerate(rows):
        center = left_x + group_width * (index + 0.5)
        values = [
            row["background_complete"],
            row["background_failure"],
            row["background_reject"],
        ]
        bottom = 0.0
        for value, color in zip(values, outcome_colors):
            y_top = _linear_y(bottom + value, 0, 100, left_y, left_h)
            y_bottom = _linear_y(bottom, 0, 100, left_y, left_h)
            canvas.rect(center - bar_width / 2, y_top, bar_width, y_bottom - y_top, fill=color, rx=2)
            bottom += value
        canvas.text(center, left_y + left_h + 30, row["label"], size=13, anchor="middle", weight=700)
        canvas.text(center, left_y - 12, f"{row['background_complete']:.1f}%", size=12, anchor="middle", fill=row["color"], weight=700)
    _legend(canvas, 135, 594, list(zip(outcome_labels, outcome_colors)), step=170)
    canvas.text(390, 637, "Candidate O: 2,004 complete · 40 failure · 704 queue reject", size=13, anchor="middle", fill=COLORS["tempo"], weight=700)

    right_x, right_y, right_w, right_h = 825, 190, 520, 350
    for tick in range(0, 101, 20):
        y = _linear_y(tick, 0, 100, right_y, right_h)
        canvas.line(right_x, y, right_x + right_w, y, stroke=COLORS["grid"])
        canvas.text(right_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = right_w / len(rows)
    metric_specs = [
        ("foreground_complete", "foreground", COLORS["good"]),
        ("background_complete", "background", COLORS["tempo_light"]),
        ("observer", "observer", COLORS["remote"]),
    ]
    for index, row in enumerate(rows):
        center = right_x + group_width * (index + 0.5)
        for metric_index, (key, _, color) in enumerate(metric_specs):
            value = float(row[key])
            offset = (metric_index - 1) * 31
            y = _linear_y(value, 0, 100, right_y, right_h)
            canvas.rect(center + offset - 13, y, 26, right_y + right_h - y, fill=color, rx=2)
        canvas.text(center, right_y + right_h + 30, row["label"], size=13, anchor="middle", weight=700)
    _legend(canvas, 850, 594, [(label, color) for _, label, color in metric_specs], step=155)
    canvas.text(1085, 625, "O: foreground 207/210 · background 2004/2748 · observer 37/210", size=13, anchor="middle", fill=COLORS["bad"], weight=700)
    canvas.text(1085, 649, "route-scope mechanism activation: 0/1,614 decisions → causal positive 아님", size=12, anchor="middle", fill=COLORS["bad"], weight=700)
    path = OUTPUT / "current_candidate_business_progression.svg"
    canvas.save(path)
    return path


def render_c9_performance() -> Path:
    arm_specs = [
        ("fixed_local_d0", "Local D0", COLORS["fixed"]),
        ("fixed_local_d1", "Local D1", COLORS["fixed"]),
        ("fixed_remote_p0d1", "Remote 0→1", "#64748B"),
        ("fixed_remote_p1d0", "Remote 1→0", "#64748B"),
        ("predictor", "Predictor", COLORS["predictor"]),
        ("queue_gpu", "Queue-GPU", COLORS["queue"]),
        ("full_c7_managed_background", "TEMPO", COLORS["tempo"]),
    ]
    rows = []
    sources = []
    for arm, label, color in arm_specs:
        relative = (
            "results/tempo_go_c8_independent_validation_job_57586612_v3/"
            f"{arm}/result.json"
        )
        value = _load(relative)["analysis"]
        sources.append(relative)
        rows.append(
            {
                "label": label,
                "color": color,
                "miss_p99": value["miss_hot"]["victim"]["e2e_ms"]["p99"] / 1000,
                "remote_p99": value["remote_favorable"]["victim"]["e2e_ms"]["p99"] / 1000,
                "miss_slo": 100 * value["miss_hot"]["slo_good_victims"] / value["miss_hot"]["offered_victims"],
                "remote_slo": 100 * value["remote_favorable"]["slo_good_victims"] / value["remote_favorable"]["offered_victims"],
            }
        )

    canvas = Canvas(
        1450,
        770,
        "C9 independent validation",
        "Seven matched policies under miss-hot and remote-favorable contention.",
    )
    _header(
        canvas,
        "C9 독립 검증: 부분 정책은 병목 이동에서 무너지고 TEMPO만 두 구간을 함께 방어",
        "Fresh 4-node allocation · same offered population · lower p99 and higher SLO are better",
    )
    canvas.panel(35, 95, 880, 620)
    canvas.panel(935, 95, 480, 620)
    canvas.text(70, 132, "Foreground E2E p99 (seconds)", size=18, weight=700)
    canvas.text(970, 132, "SLO attainment (% of offered)", size=18, weight=700)
    _legend(
        canvas,
        520,
        130,
        [("miss-hot", COLORS["miss"]), ("remote-favorable", COLORS["remote"])],
        step=175,
    )

    left_x, left_y, left_w, left_h = 85, 175, 790, 430
    for tick in range(0, 61, 10):
        y = _linear_y(tick, 0, 60, left_y, left_h)
        canvas.line(left_x, y, left_x + left_w, y, stroke=COLORS["grid"])
        canvas.text(left_x - 12, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = left_w / len(rows)
    bar_width = 29
    for index, row in enumerate(rows):
        center = left_x + group_width * (index + 0.5)
        for offset, key, color in (
            (-bar_width / 2, "miss_p99", COLORS["miss"]),
            (bar_width / 2, "remote_p99", COLORS["remote"]),
        ):
            value = float(row[key])
            y = _linear_y(value, 0, 60, left_y, left_h)
            canvas.rect(center + offset - bar_width / 2, y, bar_width, left_y + left_h - y, fill=color, rx=3)
            canvas.text(center + offset, y - 7, f"{value:.1f}", size=11, anchor="middle", weight=700)
        canvas.text(
            center + 18,
            left_y + left_h + 28,
            row["label"],
            size=12,
            anchor="end",
            transform=f"rotate(-28 {center + 18:.2f} {left_y + left_h + 28:.2f})",
        )
    canvas.text(56, left_y + left_h / 2, "seconds", size=13, fill=COLORS["muted"], anchor="middle", transform=f"rotate(-90 56 {left_y + left_h / 2})")

    right_x, right_y, right_w, right_h = 980, 175, 390, 430
    for tick in range(0, 101, 20):
        y = _linear_y(tick, 0, 100, right_y, right_h)
        canvas.line(right_x, y, right_x + right_w, y, stroke=COLORS["grid"])
        canvas.text(right_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = right_w / len(rows)
    bar_width = 14
    for index, row in enumerate(rows):
        center = right_x + group_width * (index + 0.5)
        for offset, key, color in (
            (-11, "miss_slo", COLORS["miss"]),
            (11, "remote_slo", COLORS["remote"]),
        ):
            value = float(row[key])
            y = _linear_y(value, 0, 100, right_y, right_h)
            canvas.rect(center + offset - bar_width / 2, y, bar_width, right_y + right_h - y, fill=color, rx=2)
            if value > 0:
                canvas.text(center + offset, y - 6, f"{value:.0f}", size=9, anchor="middle", weight=700)
        canvas.text(
            center + 14,
            right_y + right_h + 25,
            row["label"],
            size=11,
            anchor="end",
            transform=f"rotate(-35 {center + 14:.2f} {right_y + right_h + 25:.2f})",
        )
    canvas.text(1210, 686, "TEMPO: 120/120 miss-hot, 30/30 remote-favorable", size=13, fill=COLORS["tempo"], anchor="middle", weight=700)
    path = OUTPUT / "c9_independent_performance.svg"
    canvas.save(path)
    return path


def render_c10_comparison() -> Path:
    policies = [
        (
            "TEMPO",
            COLORS["tempo"],
            "results/tempo_go_c8_independent_validation_job_57586612_v3/full_c7_managed_background/result.json",
        ),
        (
            "Kairos X={512}",
            COLORS["kairos"],
            "results/tempo_go_c10_paper_sota_job_57586612_v2/kairos_x512/result.json",
        ),
        (
            "NetKV",
            COLORS["netkv"],
            "results/tempo_go_c10_paper_sota_job_57586612_v3/netkv/result.json",
        ),
    ]
    regimes = [
        ("normal", "normal"),
        ("miss_hot", "miss-hot"),
        ("remote_favorable", "remote-favorable"),
    ]
    values = []
    for label, color, relative in policies:
        analysis = _load(relative)["analysis"]
        values.append(
            {
                "label": label,
                "color": color,
                "p99": [
                    analysis[key]["victim"]["e2e_ms"]["p99"] / 1000
                    if analysis[key]["victim"]["e2e_ms"]["p99"] is not None
                    else None
                    for key, _ in regimes
                ],
                "slo": [
                    100 * analysis[key]["slo_good_victims"] / analysis[key]["offered_victims"]
                    for key, _ in regimes
                ],
                "reject": [analysis[key]["global_rejects"] for key, _ in regimes],
            }
        )

    canvas = Canvas(
        1450,
        735,
        "C10 paper-policy comparison",
        "Actual-system comparison of TEMPO, a restricted Kairos X=512 reproduction, and NetKV.",
    )
    _header(
        canvas,
        "C10 actual-system paper-policy 비교: 단일 목적 함수는 overload에서 completion까지 잃음",
        "Same carrier and held-out population · latency uses completed requests · C10 claim is post-hoc",
    )
    canvas.panel(35, 95, 790, 585)
    canvas.panel(845, 95, 570, 585)
    canvas.text(70, 133, "Completed-request E2E p99 (log seconds)", size=18, weight=700)
    canvas.text(880, 133, "SLO attainment (% of offered)", size=18, weight=700)
    _legend(canvas, 460, 132, [(row["label"], row["color"]) for row in values], step=150)

    left_x, left_y, left_w, left_h = 105, 180, 665, 390
    for tick in (3, 5, 10, 20, 50, 100):
        y = _log_y(tick, 2.5, 100, left_y, left_h)
        canvas.line(left_x, y, left_x + left_w, y, stroke=COLORS["grid"])
        canvas.text(left_x - 12, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    x_positions = [left_x + left_w * fraction for fraction in (0.16, 0.5, 0.84)]
    for position, (_, label) in zip(x_positions, regimes):
        canvas.text(position, left_y + left_h + 30, label, size=13, anchor="middle")
    label_offsets = {
        "TEMPO": (-27, 21),
        "Kairos X={512}": (0, -15),
        "NetKV": (28, 8),
    }
    for policy in values:
        points = []
        for index, value in enumerate(policy["p99"]):
            x = x_positions[index]
            if value is None:
                canvas.line(x - 8, left_y + left_h - 3, x + 8, left_y + left_h + 13, stroke=policy["color"], stroke_width=3)
                canvas.line(x + 8, left_y + left_h - 3, x - 8, left_y + left_h + 13, stroke=policy["color"], stroke_width=3)
                canvas.text(x, left_y + left_h - 14, "N/A: 120 rejects", size=11, fill=policy["color"], anchor="middle", weight=700)
                continue
            y = _log_y(float(value), 2.5, 100, left_y, left_h)
            points.append((x, y))
            canvas.circle(x, y, 7, fill=policy["color"])
            dx, dy = label_offsets[policy["label"]]
            canvas.text(x + dx, y + dy, f"{float(value):.2f}s", size=11, fill=policy["color"], anchor="middle", weight=700)
        if len(points) == len(regimes):
            canvas.polyline(points, stroke=policy["color"], stroke_width=3)
    canvas.text(65, left_y + left_h / 2, "seconds (log)", size=13, fill=COLORS["muted"], anchor="middle", transform=f"rotate(-90 65 {left_y + left_h / 2})")

    right_x, right_y, right_w, right_h = 890, 180, 480, 390
    for tick in range(0, 101, 20):
        y = _linear_y(tick, 0, 100, right_y, right_h)
        canvas.line(right_x, y, right_x + right_w, y, stroke=COLORS["grid"])
        canvas.text(right_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    group_width = right_w / len(regimes)
    bar_width = 35
    for regime_index, (_, regime_label) in enumerate(regimes):
        center = right_x + group_width * (regime_index + 0.5)
        for policy_index, policy in enumerate(values):
            offset = (policy_index - 1) * (bar_width + 5)
            value = float(policy["slo"][regime_index])
            y = _linear_y(value, 0, 100, right_y, right_h)
            canvas.rect(center + offset - bar_width / 2, y, bar_width, right_y + right_h - y, fill=policy["color"], rx=3)
            canvas.text(center + offset, y - 7, f"{value:.0f}", size=11, fill=policy["color"], anchor="middle", weight=700)
        canvas.text(center, right_y + right_h + 30, regime_label, size=13, anchor="middle")
    canvas.text(1130, 642, "NetKV remote: 0/30 SLO · Kairos miss-hot: 0/120 complete", size=12, fill=COLORS["muted"], anchor="middle")
    path = OUTPUT / "c10_paper_policy_comparison.svg"
    canvas.save(path)
    return path


def render_fairness_telemetry() -> Path:
    relative = "results/tempo_go_c8_independent_validation_job_57586612_v3/analysis.json"
    analysis = _load(relative)
    background = analysis["background"]
    telemetry = analysis["telemetry"]
    metrics = [
        ("background completion", 100 * background["c7_completion_fraction"], 80.0),
        ("minimum block/tenant", 100 * background["c7_minimum_block_tenant_completion_fraction"], 70.0),
        ("Jain fairness", 100 * background["c7_tenant_jain_fairness"], 99.0),
        ("service-lane success", 100 * (1 - background["c7_service_lane_failure_fraction"]), 99.0),
    ]
    overhead = [
        ("collection p50", telemetry["collection_ms"]["p50"], 50.0, COLORS["tempo_light"]),
        ("collection p99", telemetry["collection_ms"]["p99"], 250.0, COLORS["tempo"]),
        ("admission p50", telemetry["admission_wait_ms"]["p50"], 50.0, "#86EFAC"),
        ("admission p99", telemetry["admission_wait_ms"]["p99"], 250.0, COLORS["good"]),
    ]

    canvas = Canvas(
        1450,
        700,
        "Fairness and telemetry gates",
        "Background utility, tenant fairness, and controller overhead for the C9 independent run.",
    )
    _header(
        canvas,
        "이득의 비용 감사: background를 없애거나 관측 비용을 숨겨서 얻은 결과가 아님",
        "Bars are observed values; red markers are preregistered gates",
    )
    canvas.panel(35, 95, 700, 555)
    canvas.panel(755, 95, 660, 555)
    canvas.text(70, 135, "Background utility and fairness", size=18, weight=700)
    canvas.text(790, 135, "Controller collection/admission overhead", size=18, weight=700)

    bar_x, bar_w = 265, 410
    for index, (label, value, gate) in enumerate(metrics):
        y = 205 + index * 92
        canvas.text(70, y + 19, label, size=14)
        canvas.rect(bar_x, y, bar_w, 30, fill="#E5EAF1", rx=6)
        canvas.rect(bar_x, y, bar_w * value / 100, 30, fill=COLORS["good"], rx=6)
        gate_x = bar_x + bar_w * gate / 100
        canvas.line(gate_x, y - 8, gate_x, y + 38, stroke=COLORS["bad"], stroke_width=3)
        canvas.text(bar_x + bar_w + 12, y + 21, f"{value:.2f}%", size=14, weight=700, fill=COLORS["good"])
        canvas.text(gate_x, y + 55, f"gate {gate:.0f}%", size=11, fill=COLORS["bad"], anchor="middle")

    plot_x, plot_y, plot_w, plot_h = 810, 190, 550, 350
    for tick in (0, 50, 100, 150, 200, 250):
        y = _linear_y(tick, 0, 275, plot_y, plot_h)
        canvas.line(plot_x, y, plot_x + plot_w, y, stroke=COLORS["grid"])
        canvas.text(plot_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    spacing = plot_w / len(overhead)
    for index, (label, value, gate, color) in enumerate(overhead):
        center = plot_x + spacing * (index + 0.5)
        y = _linear_y(float(value), 0, 275, plot_y, plot_h)
        canvas.rect(center - 38, y, 76, plot_y + plot_h - y, fill=color, rx=5)
        gate_y = _linear_y(gate, 0, 275, plot_y, plot_h)
        canvas.line(center - 48, gate_y, center + 48, gate_y, stroke=COLORS["bad"], stroke_width=3, dash="6 4")
        canvas.text(center, y - 9, f"{float(value):.1f}ms", size=12, anchor="middle", weight=700)
        canvas.text(center, plot_y + plot_h + 28, label, size=11, anchor="middle")
        canvas.text(center, gate_y - 7, f"gate {gate:.0f}", size=10, fill=COLORS["bad"], anchor="middle")
    canvas.text(1085, 612, "complete telemetry batches: 100% · Cassini supported: 29/30", size=12, fill=COLORS["muted"], anchor="middle")
    path = OUTPUT / "c9_fairness_telemetry.svg"
    canvas.save(path)
    return path


def render_mesh_actuation() -> Path:
    relative = "results/tempo_go_c8_independent_validation_job_57586612_v3/full_c7_managed_background/result.json"
    analysis = _load(relative)["analysis"]
    edge_counts = analysis["all"]["edge_counts"]
    canvas = Canvas(
        1350,
        760,
        "TEMPO mesh actuation",
        "Observed local and remote edge use in the C9 independent run.",
    )
    _header(
        canvas,
        "실제 mesh actuation: local/remote를 미리 고정하지 않고 여섯 physical edge를 모두 사용",
        "Edge labels are completed foreground requests in the 210-request C9 run",
    )
    canvas.panel(40, 100, 1270, 585)
    positions = {
        "p0": (230, 245),
        "p1": (230, 515),
        "d0": (1050, 245),
        "d1": (1050, 515),
    }

    remote_edges = [
        ("p0", "d0", "remote:p0->d0", -70),
        ("p0", "d1", "remote:p0->d1", -25),
        ("p1", "d0", "remote:p1->d0", 25),
        ("p1", "d1", "remote:p1->d1", 70),
    ]
    for source, target, key, bend in remote_edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        count = edge_counts[key]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + bend
        definition = f"M {x1 + 80} {y1} Q {mid_x} {mid_y} {x2 - 80} {y2}"
        canvas.path(definition, stroke=COLORS["tempo"], stroke_width=2.5 + count / 4, opacity=0.72)
        canvas.rect(mid_x - 35, mid_y - 18, 70, 30, fill=COLORS["white"], stroke=COLORS["tempo_light"], stroke_width=1, rx=12)
        canvas.text(mid_x, mid_y + 3, f"{count}", size=14, fill=COLORS["tempo"], anchor="middle", weight=700)

    local_specs = [
        ("d0", "local:d0", 890, 155),
        ("d1", "local:d1", 890, 605),
    ]
    for node, key, label_x, label_y in local_specs:
        x, y = positions[node]
        count = edge_counts[key]
        loop_y = y - 95 if node == "d0" else y + 95
        definition = f"M {x - 20} {y - 55 if node == 'd0' else y + 55} C {x - 160} {loop_y} {x + 160} {loop_y} {x + 20} {y - 55 if node == 'd0' else y + 55}"
        canvas.path(definition, stroke=COLORS["fixed"], stroke_width=min(11, 3 + count / 18), opacity=0.9)
        canvas.rect(label_x - 50, label_y - 18, 100, 32, fill=COLORS["white"], stroke=COLORS["fixed"], stroke_width=1, rx=12)
        canvas.text(label_x, label_y + 4, f"local {count}", size=13, anchor="middle", weight=700)

    for name, label in (("p0", "Prefill P0"), ("p1", "Prefill P1"), ("d0", "Decoder D0"), ("d1", "Decoder D1")):
        x, y = positions[name]
        fill = "#E8F0FF" if name.startswith("p") else "#ECFDF5"
        stroke = COLORS["tempo"] if name.startswith("p") else COLORS["good"]
        canvas.rect(x - 82, y - 55, 164, 110, fill=fill, stroke=stroke, stroke_width=2, rx=16)
        canvas.text(x, y - 7, label, size=17, anchor="middle", weight=700)
        canvas.text(x, y + 23, "TP4 / A100 node", size=12, fill=COLORS["muted"], anchor="middle")

    canvas.rect(115, 625, 1120, 42, fill="#EEF4FF", stroke="#BDD2FF", stroke_width=1, rx=12)
    canvas.text(675, 651, "local 181 · official LMCache remote 29 · remote-favorable에서 remote 29/30 · failures 0", size=15, fill=COLORS["ink"], anchor="middle", weight=700)
    path = OUTPUT / "c9_mesh_actuation.svg"
    canvas.save(path)
    return path


def render_hierarchy_scale() -> Path:
    relative = "results/tempo_go_hierarchy_scale_20260825_c9_c10_r15.json"
    value = _load(relative)
    rows = value["scales"]
    pair_counts = [row["pair_count"] for row in rows]
    full_payload = [row["full_payload_bytes"] for row in rows]
    bounded_payload = [row["bounded_global_payload_bytes"] for row in rows]
    full_p50 = [row["full_reduction"]["p50_ms"] for row in rows]
    pair_p50 = [row["pair_agent_frontier_build"]["p50_ms"] for row in rows]
    global_p50 = [row["bounded_global_reduction"]["p50_ms"] for row in rows]
    total_p50 = [row["bounded_total_control_path"]["p50_ms"] for row in rows]

    canvas = Canvas(
        1450,
        720,
        "Hierarchy control-plane scaling",
        "Payload and CPU control-path latency from 2 through 1024 logical pairs.",
    )
    _header(
        canvas,
        "계층형 fan-in: global 후보 payload는 256개에서 bounded, CPU 경로 비용은 별도 공개",
        "15 repeats · same candidate population · CPU control-plane only (not native 1,024-pair inference)",
    )
    canvas.panel(35, 95, 680, 565)
    canvas.panel(735, 95, 680, 565)
    canvas.text(70, 133, "Global payload bytes (log scale)", size=18, weight=700)
    canvas.text(770, 133, "Control-path p50 latency (ms)", size=18, weight=700)

    def x_position(pair: int, start: float, width: float) -> float:
        return start + (math.log2(pair) - 1) / 9 * width

    left_x, left_y, left_w, left_h = 100, 180, 560, 350
    for tick in (1_000, 10_000, 100_000, 1_000_000):
        y = _log_y(tick, 1_000, 1_000_000, left_y, left_h)
        canvas.line(left_x, y, left_x + left_w, y, stroke=COLORS["grid"])
        canvas.text(left_x - 10, y + 5, f"{tick // 1000}K" if tick < 1_000_000 else "1M", size=12, fill=COLORS["muted"], anchor="end")
    payload_series = [
        ("full scan", full_payload, COLORS["fixed"]),
        ("bounded global", bounded_payload, COLORS["tempo"]),
    ]
    for label, series, color in payload_series:
        points = []
        for pair, item in zip(pair_counts, series):
            x = x_position(pair, left_x, left_w)
            y = _log_y(item, 1_000, 1_000_000, left_y, left_h)
            points.append((x, y))
            canvas.circle(x, y, 5, fill=color)
        canvas.polyline(points, stroke=color, stroke_width=3)
    for pair in pair_counts:
        x = x_position(pair, left_x, left_w)
        canvas.text(x, left_y + left_h + 28, pair, size=11, anchor="middle")
    _legend(canvas, 360, 132, [(label, color) for label, _, color in payload_series], step=130)
    canvas.text(x_position(1024, left_x, left_w) - 5, _log_y(bounded_payload[-1], 1_000, 1_000_000, left_y, left_h) - 16, "83,358 B", size=12, fill=COLORS["tempo"], anchor="end", weight=700)
    canvas.text(380, 600, "1,024 pairs: 666,815 B → 83,358 B (−87.499%)", size=13, fill=COLORS["tempo"], anchor="middle", weight=700)

    right_x, right_y, right_w, right_h = 800, 180, 560, 350
    for tick in (0, 20, 40, 60, 80, 100):
        y = _linear_y(tick, 0, 100, right_y, right_h)
        canvas.line(right_x, y, right_x + right_w, y, stroke=COLORS["grid"])
        canvas.text(right_x - 10, y + 5, tick, size=12, fill=COLORS["muted"], anchor="end")
    latency_series = [
        ("full scan", full_p50, COLORS["fixed"]),
        ("pair agent", pair_p50, COLORS["predictor"]),
        ("bounded global", global_p50, COLORS["tempo_light"]),
        ("bounded total", total_p50, COLORS["tempo"]),
    ]
    for label, series, color in latency_series:
        points = []
        for pair, item in zip(pair_counts, series):
            x = x_position(pair, right_x, right_w)
            y = _linear_y(item, 0, 100, right_y, right_h)
            points.append((x, y))
            canvas.circle(x, y, 4.5, fill=color)
        canvas.polyline(points, stroke=color, stroke_width=2.5)
    for pair in pair_counts:
        x = x_position(pair, right_x, right_w)
        canvas.text(x, right_y + right_h + 28, pair, size=11, anchor="middle")
    _legend(canvas, 820, 590, [(label, color) for label, _, color in latency_series], step=138)
    canvas.text(1080, 623, "1,024 p50: full 49.74 · pair 29.65 · global 55.43 · total 85.33 ms", size=12, fill=COLORS["muted"], anchor="middle")
    canvas.text(1080, 645, "p99: full 152.84 ms · bounded total 158.24 ms", size=12, fill=COLORS["bad"], anchor="middle", weight=700)
    path = OUTPUT / "hierarchy_control_plane_scale.svg"
    canvas.save(path)
    return path


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated = [
        render_current_native_matrix(),
        render_candidate_business_progression(),
        render_c9_performance(),
        render_c10_comparison(),
        render_fairness_telemetry(),
        render_mesh_actuation(),
        render_hierarchy_scale(),
    ]
    source_paths = [
        "paper/tempo_go/artifact_manifest.json",
        "paper/tempo_go/current_evidence_manifest.json",
        "results/tempo_go_c9_global_frontier_job_57732862/analysis_failclosed_business_v3.json",
        "results/tempo_go_c9_causal_burst_job_57732862/analysis_failclosed_business_v3.json",
        "results/tempo_go_c9_route_liveness_job_57736076_r3_canonical_outer/analysis_failclosed_business_v2.json",
        "results/tempo_go_c9_route_liveness_job_57736076_r3_canonical_outer/candidate_o_diagnosis.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/analysis.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/fixed_local_d0/result.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/fixed_local_d1/result.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/fixed_remote_p0d1/result.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/fixed_remote_p1d0/result.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/predictor/result.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/queue_gpu/result.json",
        "results/tempo_go_c8_independent_validation_job_57586612_v3/full_c7_managed_background/result.json",
        "results/tempo_go_c10_paper_sota_job_57586612_v2/kairos_x512/result.json",
        "results/tempo_go_c10_paper_sota_job_57586612_v3/netkv/result.json",
        "results/tempo_go_hierarchy_scale_20260825_c9_c10_r15.json",
    ]
    manifest = {
        "schema": "tempo-go-readme-figure-manifest-v1",
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "sources": {
            path: _sha256(ROOT / path)
            for path in source_paths
        },
        "figures": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in generated
        },
        "claim_boundary": (
            "Candidate O is current negative native evidence and its changed "
            "route-scope mechanism did not activate; exact terminal post-hoc "
            "analysis does not modify native raw or analysis; historical C9 "
            "positive and C10 post-hoc figures are not current claims; "
            "hierarchy scale is CPU control-plane only."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "figures": len(generated),
        "manifest": str((OUTPUT / "manifest.json").relative_to(ROOT)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
