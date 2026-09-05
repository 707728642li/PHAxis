from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts/phaxis"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import render_paper_first_figure5 as renderer  # noqa: E402
from phaxis.io import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from phaxis.narrative_decision import build_narrative_decision  # noqa: E402


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _synthetic_source(root: Path) -> tuple[Path, str]:
    package = root / "synthetic_source"
    package.mkdir(parents=True)

    base_values = {
        renderer.ENDPOINT_ORDER[0]: (5.0, 6.0, 8.0, 10.0),
        renderer.ENDPOINT_ORDER[1]: (95.0, 105.0, 118.0, 132.0),
        renderer.ENDPOINT_ORDER[2]: (260.0, 245.0, 220.0, 205.0),
        renderer.ENDPOINT_ORDER[3]: (51.0, 50.0, 54.0, 53.0),
        renderer.ENDPOINT_ORDER[4]: (5100.0, 5000.0, 5400.0, 5300.0),
    }
    rows: list[dict] = []
    for group_index, group in enumerate(renderer.GROUP_ORDER):
        for replicate in range(8):
            source_unit = f"D15-{group_index}-{replicate}"
            image_sha = _hash(f"image-{source_unit}")
            for endpoint in renderer.ENDPOINT_ORDER:
                if endpoint == renderer.ENDPOINT_ORDER[1] and replicate >= 6:
                    continue
                if endpoint == renderer.ENDPOINT_ORDER[2] and replicate >= 5:
                    continue
                value = base_values[endpoint][group_index] + (replicate - 3.5) * (
                    1.0 if endpoint == renderer.ENDPOINT_ORDER[0] else 0.3
                )
                if endpoint == renderer.ENDPOINT_ORDER[0]:
                    value = float(round(value))
                rows.append(
                    {
                        "source_unit": source_unit,
                        "cohort": renderer.COHORT_ORDER[0],
                        "condition_code": group,
                        "formal_eligible": True,
                        "endpoint_key": endpoint,
                        "value": value,
                        "unit": renderer.ENDPOINT_UNITS[endpoint],
                        "source_image_sha256": image_sha,
                    }
                )
    points = pd.DataFrame(rows)
    points_path = package / renderer.RESOURCE_FILES["phenotype_points"]
    _write_csv(points_path, points)

    effect_rows: list[dict] = []
    for endpoint_index, endpoint in enumerate(renderer.ENDPOINT_ORDER):
        for effect_index, effect in enumerate(renderer.EFFECT_ORDER):
            clean_estimate = 0.82 + endpoint_index * 0.08 + effect_index * 0.05
            for cohort_index, cohort in enumerate(renderer.COHORT_ORDER):
                estimate = clean_estimate + cohort_index * 0.02
                effect_rows.append(
                    {
                        "endpoint_key": endpoint,
                        "effect_key": effect,
                        "cohort": cohort,
                        "estimate": estimate,
                        "ci_low": estimate - 0.08,
                        "ci_high": estimate + 0.08,
                        "endpoint_n": 32 if cohort_index == 0 else 36,
                        "effect_scale": "ratio",
                        "raw_effect_estimate": (endpoint_index + 1) * (effect_index + 1) * 0.5,
                        "raw_effect_ci_low": (endpoint_index + 1) * (effect_index + 1) * 0.5 - 0.2,
                        "raw_effect_ci_high": (endpoint_index + 1) * (effect_index + 1) * 0.5 + 0.2,
                        "raw_effect_unit": renderer.ENDPOINT_UNITS[endpoint],
                        "raw_effect_estimand": "synthetic_factorial_contrast",
                        "raw_effect_interval_method": "synthetic_interval",
                        "raw_effect_bootstrap_replicates": 0,
                        "raw_effect_bootstrap_seed": np.nan,
                        "standardized_effect": 0.2 * (endpoint_index + effect_index + 1),
                        "standardized_ci_low": 0.2 * (endpoint_index + effect_index + 1) - 0.1,
                        "standardized_ci_high": 0.2 * (endpoint_index + effect_index + 1) + 0.1,
                    }
                )
    effects = pd.DataFrame(effect_rows)
    effects_path = package / renderer.RESOURCE_FILES["phenotype_effects"]
    _write_csv(effects_path, effects)

    support = pd.DataFrame(
        [
            {
                "condition_code": group,
                "support_fraction": (30 + index) / 40,
                "supported_hairs": 30 + index,
                "identity_hairs": 40,
                "source_units": 8,
            }
            for index, group in enumerate(renderer.GROUP_ORDER)
        ]
    )
    support_path = package / renderer.RESOURCE_FILES["assurance_support"]
    _write_csv(support_path, support)

    profile_rows: list[dict] = []
    for metric in renderer.PROFILE_ORDER:
        for group_index, group in enumerate(renderer.GROUP_ORDER):
            for bin_index in range(5):
                if metric == "length_support_fraction":
                    estimate = 0.70 + 0.02 * group_index + 0.01 * bin_index
                    half_width = 0.05
                elif metric == "conditional_median_length_um":
                    estimate = 90.0 + 4.0 * group_index + 3.0 * bin_index
                    half_width = 4.0
                else:
                    estimate = 2.0 + group_index + 0.6 * bin_index
                    half_width = 0.3
                profile_rows.append(
                    {
                        "cohort": renderer.COHORT_ORDER[0],
                        "condition_code": group,
                        "bin_start_mm": float(bin_index),
                        "bin_end_mm": float(bin_index + 1),
                        "metric_key": metric,
                        "estimate": estimate,
                        "ci_low": estimate - half_width,
                        "ci_high": estimate + half_width,
                        "eligible_n": 8,
                        "length_supported_n": 6,
                        "bootstrap_repetitions": 10000,
                        "unit_of_analysis": "source image/root",
                    }
                )
    profiles_path = package / renderer.RESOURCE_FILES["axial_profiles"]
    _write_csv(profiles_path, pd.DataFrame(profile_rows))

    decision = build_narrative_decision(
        effects.to_dict("records"),
        source_sha256={
            "primary_tests": _hash("synthetic-primary"),
            "sensitivity_tests": _hash("synthetic-sensitivity"),
        },
    )
    decision_path = package / renderer.RESOURCE_FILES["narrative_decision"]
    atomic_write_json(decision_path, decision)

    resources = {
        role: {"path": filename, "sha256": sha256_file(package / filename)}
        for role, filename in renderer.RESOURCE_FILES.items()
    }
    manifest = {
        "schema_version": renderer.SOURCE_SCHEMA,
        "status": renderer.SOURCE_STATUS,
        "scope": "Figure_5_only_no_Fig1_6_suite_dependency",
        "resources": resources,
        "endpoint_order": list(renderer.ENDPOINT_ORDER),
        "effect_order": list(renderer.EFFECT_ORDER),
        "cohort_order": list(renderer.COHORT_ORDER),
        "condition_order": list(renderer.GROUP_ORDER),
        "narrative_decision_identity_sha256": decision[
            "narrative_decision_identity_sha256"
        ],
        "model_contract_proposal_identity_sha256": _hash("synthetic-proposal"),
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
        "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
        "hair_identity_count_expert_id": renderer.STAGEB_EXPERT_ID,
        "profile_hypothesis_tests_performed": False,
        "root_cap_region_statistics_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    manifest["source_package_identity_sha256"] = sha256_json(manifest)
    manifest_path = package / "source_package.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path, sha256_file(manifest_path)


def test_final_renderer_materializes_locked_600dpi_four_format_bundle(tmp_path: Path) -> None:
    manifest, digest = _synthetic_source(tmp_path)
    output = tmp_path / "rendered_figure5"
    receipt = renderer.render_paper_first_figure5(
        source_package=manifest,
        expected_source_package_sha256=digest,
        output=output,
    )

    assert receipt["status"] == renderer.RENDER_STATUS
    assert receipt["sentinel_order"] == ["N/H08", "L/H11", "F/H07", "W/R07", "A/R01"]
    assert receipt["clean_full_sensitivity_shown"] is True
    assert receipt["raster_dpi"] == 600
    assert receipt["gpu_program_started"] is False
    assert receipt["blind_images_used"] == 0
    assert set(path.name for path in output.iterdir()) == {
        f"{renderer.FIGURE_STEM}.png",
        f"{renderer.FIGURE_STEM}.tiff",
        f"{renderer.FIGURE_STEM}.pdf",
        f"{renderer.FIGURE_STEM}.svg",
        "Figure5_caption.md",
        "source_lock.json",
        "visual_qa.json",
        "receipt.json",
        "SHA256SUMS.txt",
    }
    expected_pixels = (
        int(renderer.WIDTH_MM / 25.4 * renderer.RASTER_DPI),
        int(renderer.HEIGHT_MM / 25.4 * renderer.RASTER_DPI),
    )
    with Image.open(output / f"{renderer.FIGURE_STEM}.png") as image:
        assert image.size == expected_pixels
        assert all(abs(value - 600) < 0.1 for value in image.info["dpi"])
    with Image.open(output / f"{renderer.FIGURE_STEM}.tiff") as image:
        assert image.size == expected_pixels
        assert image.mode == "RGB"
        assert all(abs(value - 600) < 0.1 for value in image.info["dpi"])
    caption = (output / "Figure5_caption.md").read_text(encoding="utf-8")
    for token in ("N→L→F→W→A", "H11/L", "H07/F", "Clean-cohort", "Full283"):
        assert token in caption
    qa = json.loads((output / "visual_qa.json").read_text(encoding="utf-8"))
    unsigned_qa = deepcopy(qa)
    identity = unsigned_qa.pop("visual_qa_identity_sha256")
    assert sha256_json(unsigned_qa) == identity
    assert qa["status"] == "pass_submission_size_readability"
    assert qa["format"]["edge_ink_by_side_outer_2px"] == {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0,
    }

    with pytest.raises(renderer.Figure5RenderError, match="overwrite"):
        renderer.render_paper_first_figure5(
            source_package=manifest,
            expected_source_package_sha256=digest,
            output=output,
        )


def test_explicit_sha_and_semantic_support_checks_fail_closed(tmp_path: Path) -> None:
    manifest, digest = _synthetic_source(tmp_path)
    with pytest.raises(renderer.Figure5RenderError, match="explicit lock"):
        renderer.render_paper_first_figure5(
            source_package=manifest,
            expected_source_package_sha256="f" * 64,
            output=tmp_path / "wrong_hash_output",
        )

    support_path = manifest.parent / renderer.RESOURCE_FILES["assurance_support"]
    support = pd.read_csv(support_path)
    support.loc[0, "supported_hairs"] = int(support.loc[0, "supported_hairs"]) - 1
    _write_csv(support_path, support)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["resources"]["assurance_support"]["sha256"] = sha256_file(support_path)
    payload.pop("source_package_identity_sha256")
    payload["source_package_identity_sha256"] = sha256_json(payload)
    atomic_write_json(manifest, payload)
    digest = sha256_file(manifest)
    with pytest.raises(renderer.Figure5RenderError, match="counts/fractions do not close"):
        renderer.render_paper_first_figure5(
            source_package=manifest,
            expected_source_package_sha256=digest,
            output=tmp_path / "semantic_error_output",
        )


@pytest.mark.parametrize(
    "marker",
    ("legacy_443cv", "provisional", "fixture", "blind", "final-validation"),
)
def test_forbidden_lineage_and_evaluation_paths_are_refused(
    tmp_path: Path, marker: str
) -> None:
    manifest, _digest = _synthetic_source(tmp_path / "valid_inputs")
    copied = tmp_path / marker / "source_package"
    copied.parent.mkdir(parents=True)
    shutil.copytree(manifest.parent, copied)
    copied_manifest = copied / "source_package.json"
    with pytest.raises(renderer.Figure5RenderError, match="path refused"):
        renderer.render_paper_first_figure5(
            source_package=copied_manifest,
            expected_source_package_sha256=sha256_file(copied_manifest),
            output=tmp_path / "unused_output",
        )


def test_forbidden_output_path_is_refused_before_render(tmp_path: Path) -> None:
    manifest, digest = _synthetic_source(tmp_path)
    with pytest.raises(renderer.Figure5RenderError, match="path refused"):
        renderer.render_paper_first_figure5(
            source_package=manifest,
            expected_source_package_sha256=digest,
            output=tmp_path / "provisional_figure5",
        )
