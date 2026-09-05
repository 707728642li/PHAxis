"""Render a publication-format PHAxis distal-axis profile figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


CONDITIONS = (
    ("RHD6_EV_22C", "RHD6-EV, 22 °C", "#0072B2", "o"),
    ("RHD6_EV_30C", "RHD6-EV, 30 °C", "#56B4E9", "s"),
    ("RHD6_OE_22C", "RHD6-OE, 22 °C", "#D55E00", "^"),
    ("RHD6_OE_30C", "RHD6-OE, 30 °C", "#E69F00", "D"),
)
PROFILE_PANELS = (
    (
        "a",
        "Root-hair abundance",
        "mean_attached_identity_count",
        "mean_attached_identity_count_ci95_low",
        "mean_attached_identity_count_ci95_high",
        "Hair count per 1-mm bin",
    ),
    (
        "b",
        "Conditional projected length",
        "median_of_source_unit_conditional_median_length_um",
        "median_of_source_unit_conditional_median_length_um_ci95_low",
        "median_of_source_unit_conditional_median_length_um_ci95_high",
        "Conditional projected length (µm)",
    ),
    (
        "c",
        "Length observability",
        "endpoint_complete_support_fraction",
        "endpoint_complete_support_fraction_ci95_low",
        "endpoint_complete_support_fraction_ci95_high",
        "Complete-length support (%)",
    ),
)


def _optional(row: dict[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    return float(value) if value else float("nan")


def _atomic_save_figure(figure: plt.Figure, destination: Path, **kwargs: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary_name, **kwargs)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _force_rgb_raster(path: Path, *, format_name: str, dpi: int) -> None:
    """Publish an opaque RGB raster while preserving atomic replacement."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.rgb.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        with Image.open(path) as source:
            rgb = Image.new("RGB", source.size, "white")
            if source.mode == "RGBA":
                rgb.paste(source.convert("RGB"), mask=source.getchannel("A"))
            else:
                rgb.paste(source.convert("RGB"))
            save_options: dict[str, Any] = {"dpi": (dpi, dpi)}
            if format_name == "TIFF":
                save_options["compression"] = "tiff_lzw"
            elif format_name == "PNG":
                save_options["compress_level"] = 6
            rgb.save(temporary_name, format=format_name, **save_options)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def render(
    *,
    analysis_root: Path,
    output: Path,
    cohort_role: str,
    provisional_watermark: bool,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    summary_path = analysis_root / "summary.json"
    table_path = analysis_root / "distal_axis_profile_group_summaries.csv"
    summary = read_json(summary_path)
    if (
        summary.get("schema_version") != "PHAxis-distal-axis-profile-analysis-1.0.0"
        or summary.get("status")
        != "completed_exploratory_source_unit_profile_summaries"
        or summary.get("blind_images_used") != 0
        or summary.get("hypothesis_tests_performed") is not False
    ):
        raise RuntimeError("invalid or blind-tainted profile analysis")
    if sha256_file(table_path) != summary.get("output_table_sha256"):
        raise RuntimeError("profile-analysis table hash mismatch")
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["cohort_role"] == cohort_role]
    if not rows:
        raise RuntimeError(f"no rows for cohort role: {cohort_role}")
    if {row["condition_code"] for row in rows} != {item[0] for item in CONDITIONS}:
        raise RuntimeError("profile figure requires the locked four D15 conditions")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), constrained_layout=False)
    figure.patch.set_facecolor("white")
    for axis, (letter, title, value_field, low_field, high_field, ylabel) in zip(
        axes, PROFILE_PANELS, strict=True
    ):
        for condition, label, color, marker in CONDITIONS:
            selected = sorted(
                (row for row in rows if row["condition_code"] == condition),
                key=lambda row: int(row["bin_index"]),
            )
            x = np.asarray(
                [
                    (float(row["bin_start_um"]) + float(row["bin_end_um"]))
                    / 2000.0
                    for row in selected
                ]
            )
            values = np.asarray([_optional(row, value_field) for row in selected])
            low = np.asarray([_optional(row, low_field) for row in selected])
            high = np.asarray([_optional(row, high_field) for row in selected])
            if letter == "c":
                values, low, high = values * 100.0, low * 100.0, high * 100.0
            finite = np.isfinite(values)
            axis.plot(
                x[finite],
                values[finite],
                color=color,
                marker=marker,
                markersize=3.6,
                markeredgewidth=0.5,
                linewidth=1.15,
                label=label,
                zorder=3,
            )
            error_finite = finite & np.isfinite(low) & np.isfinite(high)
            if error_finite.any():
                axis.fill_between(
                    x[error_finite],
                    low[error_finite],
                    high[error_finite],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    zorder=2,
                )
        axis.set_title(title, pad=5)
        axis.set_xlabel("Distance from distal point (mm)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.3, 5.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xticks(np.arange(0.5, 5.0, 1.0))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.0,
            1.09,
            letter,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.32),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )
    footer = (
        "Source-image/root units; shaded 95% source-unit bootstrap intervals. "
        "Length is conditional on one-to-one endpoint-complete centreline support."
    )
    figure.text(0.995, 0.012, footer, ha="right", va="bottom", fontsize=6.2, color="#444444")
    if provisional_watermark:
        figure.text(
            0.5,
            0.53,
            "PROVISIONAL — SOFTWARE-CONTRACT QA ONLY",
            ha="center",
            va="center",
            fontsize=13,
            color="#7A0019",
            alpha=0.18,
            rotation=18,
            fontweight="bold",
        )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.34, wspace=0.39)
    output.mkdir(parents=True, exist_ok=True)
    base = output / "PHAxis_distal_axis_profiles"
    paths = {
        "pdf": base.with_suffix(".pdf"),
        "png_600dpi": output / "PHAxis_distal_axis_profiles_600dpi.png",
        "tiff_300dpi": output / "PHAxis_distal_axis_profiles_300dpi.tiff",
    }
    _atomic_save_figure(
        figure,
        paths["pdf"],
        format="pdf",
        facecolor="white",
        bbox_inches=None,
        metadata={"Title": "PHAxis distal-axis root-hair profiles"},
    )
    _atomic_save_figure(
        figure,
        paths["png_600dpi"],
        format="png",
        dpi=600,
        facecolor="white",
        bbox_inches=None,
    )
    _atomic_save_figure(
        figure,
        paths["tiff_300dpi"],
        format="tiff",
        dpi=300,
        facecolor="white",
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    _force_rgb_raster(paths["png_600dpi"], format_name="PNG", dpi=600)
    _force_rgb_raster(paths["tiff_300dpi"], format_name="TIFF", dpi=300)
    raster_audit: dict[str, Any] = {}
    for name in ("png_600dpi", "tiff_300dpi"):
        with Image.open(paths[name]) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise RuntimeError(f"unexpected figure colour mode: {image.mode}")
            raster_audit[name] = {
                "mode": image.mode,
                "width_px": image.width,
                "height_px": image.height,
                "dpi": [float(value) for value in image.info.get("dpi", ())],
            }
    receipt: dict[str, Any] = {
        "schema_version": "PHAxis-distal-axis-profile-figure-1.0.0",
        "status": "rendered_and_raster_verified",
        "cohort_role": cohort_role,
        "provisional_watermark": provisional_watermark,
        "analysis_summary_sha256": sha256_file(summary_path),
        "analysis_table_sha256": sha256_file(table_path),
        "files_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "raster_audit": raster_audit,
        "figure_width_in": 7.05,
        "figure_width_mm": 179.07,
        "white_background": True,
        "hypothesis_tests_displayed": False,
        "individual_hairs_treated_as_independent_replicates": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    receipt["figure_identity_sha256"] = sha256_json(receipt)
    atomic_write_json(output / "summary.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-role", default="primary_SHA_disjoint")
    parser.add_argument("--provisional-watermark", action="store_true")
    args = parser.parse_args()
    result = render(
        analysis_root=args.analysis_root.resolve(),
        output=args.output.resolve(),
        cohort_role=args.cohort_role,
        provisional_watermark=bool(args.provisional_watermark),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
