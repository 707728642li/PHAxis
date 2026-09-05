from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from phaxis.io import sha256_file


def _module(project_root: Path):
    path = project_root / "scripts" / "phaxis" / "render_distal_axis_profile_figure.py"
    spec = importlib.util.spec_from_file_location("_profile_figure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_figure_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    module = _module(project_root)
    panel_titles = tuple(panel[1] for panel in module.PROFILE_PANELS)
    panel_ylabels = tuple(panel[-1] for panel in module.PROFILE_PANELS)
    assert "Conditional projected length" in panel_titles
    assert "Conditional projected length (µm)" in panel_ylabels
    assert "Conditional elongation" not in panel_titles
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    rows = []
    fields = (
        "cohort_role",
        "condition_code",
        "bin_index",
        "bin_start_um",
        "bin_end_um",
        "mean_attached_identity_count",
        "mean_attached_identity_count_ci95_low",
        "mean_attached_identity_count_ci95_high",
        "median_of_source_unit_conditional_median_length_um",
        "median_of_source_unit_conditional_median_length_um_ci95_low",
        "median_of_source_unit_conditional_median_length_um_ci95_high",
        "endpoint_complete_support_fraction",
        "endpoint_complete_support_fraction_ci95_low",
        "endpoint_complete_support_fraction_ci95_high",
    )
    conditions = ("RHD6_EV_22C", "RHD6_EV_30C", "RHD6_OE_22C", "RHD6_OE_30C")
    for condition_index, condition in enumerate(conditions):
        for bin_index in range(5):
            value = 1 + condition_index + bin_index
            rows.append(
                {
                    "cohort_role": "primary_SHA_disjoint",
                    "condition_code": condition,
                    "bin_index": bin_index,
                    "bin_start_um": bin_index * 1000,
                    "bin_end_um": (bin_index + 1) * 1000,
                    "mean_attached_identity_count": value,
                    "mean_attached_identity_count_ci95_low": value - 0.2,
                    "mean_attached_identity_count_ci95_high": value + 0.2,
                    "median_of_source_unit_conditional_median_length_um": 100 + value,
                    "median_of_source_unit_conditional_median_length_um_ci95_low": 95 + value,
                    "median_of_source_unit_conditional_median_length_um_ci95_high": 105 + value,
                    "endpoint_complete_support_fraction": 0.5,
                    "endpoint_complete_support_fraction_ci95_low": 0.4,
                    "endpoint_complete_support_fraction_ci95_high": 0.6,
                }
            )
    table = analysis / "distal_axis_profile_group_summaries.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "PHAxis-distal-axis-profile-analysis-1.0.0",
        "status": "completed_exploratory_source_unit_profile_summaries",
        "output_table_sha256": sha256_file(table),
        "hypothesis_tests_performed": False,
        "blind_images_used": 0,
    }
    (analysis / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = module.render(
        analysis_root=analysis,
        output=tmp_path / "figure",
        cohort_role="primary_SHA_disjoint",
        provisional_watermark=True,
    )
    assert result["status"] == "rendered_and_raster_verified"
    assert result["provisional_watermark"] is True
    assert (tmp_path / "figure" / "PHAxis_distal_axis_profiles.pdf").is_file()
    assert (tmp_path / "figure" / "PHAxis_distal_axis_profiles_600dpi.png").is_file()
