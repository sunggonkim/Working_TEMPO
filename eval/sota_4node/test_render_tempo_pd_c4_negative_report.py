from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from eval.sota_4node import render_tempo_pd_c4_negative_report as report


def _panel(title: str):
    return {
        "title": title,
        "categories": ("A", "B"),
        "series": (
            ("fixed", "#000000", (1.0, 2.0)),
            ("tempo", "#ffffff", (2.0, 1.0)),
        ),
        "percent": False,
    }


def test_svg_report_is_well_formed_and_contains_all_panels(tmp_path):
    output = tmp_path / "report.svg"
    report._render_svg(
        title="fixture",
        panels=tuple(_panel(f"panel-{index}") for index in range(4)),
        output=output,
    )

    root = ET.parse(output).getroot()
    assert root.tag.endswith("svg")
    text = output.read_text(encoding="utf-8")
    assert "fixture" in text
    assert all(f"panel-{index}" in text for index in range(4))


def test_svg_report_requires_exact_panel_inventory(tmp_path):
    with pytest.raises(ValueError, match="four panels"):
        report._render_svg(
            title="fixture",
            panels=(_panel("one"),),
            output=tmp_path / "report.svg",
        )
