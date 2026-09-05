from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from phaxis.io import sha256_file, sha256_json
from tests.phaxis.test_manuscript_figure_suite import _resources


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/phaxis/build_paper_first_figures_1_3_4.py"
SPEC = importlib.util.spec_from_file_location("paper_first_fig134", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _seal(payload: dict, field: str) -> dict:
    result = deepcopy(payload)
    result.pop(field, None)
    result[field] = sha256_json(result)
    return result


def _fast_bundle(figure, base_path: Path, *, width_mm: float, height_mm: float, **_) -> dict:
    files = {}
    hashes = {}
    for kind, suffix in (("pdf", ".pdf"), ("png", ".png"), ("tiff", ".tiff")):
        path = base_path.with_suffix(suffix)
        path.write_bytes(f"{kind}:{base_path.name}".encode())
        files[kind] = str(path.resolve())
        hashes[kind] = sha256_file(path)
    plt.close(figure)
    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "files": files,
        "sha256": hashes,
        "png_pixels": [1, 1],
        "png_dpi": [600.0, 600.0],
        "tiff_pixels": [1, 1],
        "tiff_dpi": [300.0, 300.0],
        "tiff_mode": "RGB",
        "edge_ink_pixels_outer_2px": 0,
        "edge_ink_by_side_outer_2px": {"top": 0, "bottom": 0, "left": 0, "right": 0},
    }


def _current_fixture(root: Path) -> tuple[builder.Inputs, dict[str, str]]:
    input_root = root / "current_inputs"
    input_root.mkdir(parents=True)
    resources = _resources(input_root)

    # The current sealed QC-development authority has 37 visible bars and
    # seven trusted-metadata calibrations (not the legacy 38/6 fixture split).
    assurance_metrics = pd.read_csv(resources["assurance_metrics"])
    assurance_metrics["scale_visible_truth_n"] = 37
    assurance_metrics["scale_trusted_metadata_n"] = 7
    scale_coverage = assurance_metrics["metric_key"].eq("scale_detection_coverage")
    assurance_metrics.loc[scale_coverage, ["value", "ci_low", "ci_high"]] = 1.0
    assurance_metrics.loc[scale_coverage, ["n", "instances"]] = 37
    scale_conditional = assurance_metrics["metric_key"].isin(
        ["scale_geometry_endpoint_error_um", "scale_relative_error_percent"]
    )
    assurance_metrics.loc[scale_conditional, ["n", "instances"]] = 37
    assurance_metrics.to_csv(
        resources["assurance_metrics"], index=False, lineterminator="\n"
    )

    proposal = PROJECT_ROOT / "outputs/phaxis_paperfirst_public327_authority_run1/proposal.json"
    trait_contract = PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json"
    story = PROJECT_ROOT / "outputs/phaxis_trait_story_map_20260830T093441Z/trait_story_map.json"
    story_receipt = PROJECT_ROOT / "outputs/phaxis_trait_story_map_20260830T093441Z/HASH_RECEIPT.json"

    role_tasks = {
        "representative": builder.FIGURE1_TASK_ID,
        "low_contrast": builder.FIGURE4_ANCHORS["low_contrast"],
        "curved_dense": builder.FIGURE4_ANCHORS["curved_dense"],
        "continuity": "RHSCU-0d193f9385dd74c3",
        "fail_closed": "RHSCU-bc9223e70e962f9b",
    }
    selection = pd.read_csv(resources["overlay_selection"])
    selection["task_id"] = selection["case_role"].map(role_tasks)
    selection["prediction_sha256"] = selection["task_id"].map(
        lambda task_id: sha256_json({"current_prediction": task_id})
    )
    selection["raw_source_image_sha256"] = selection["task_id"].map(
        lambda task_id: sha256_json({"raw": task_id})
    )
    selection["full_cohort_review_overlay_path"] = selection.apply(
        lambda row: (
            "full283_review_overlays/test/test/"
            + ("formal/" if bool(row["formal_statistics_eligible"]) else "review_only/")
            + str(row["task_id"])
            + ".phaxis_overlay.png"
        ),
        axis=1,
    )
    selection.to_csv(resources["overlay_selection"], index=False, lineterminator="\n")

    case_tasks = selection["task_id"].astype(str).tolist()
    filler_tasks = [f"RHSCU-test-{index:04d}" for index in range(283 - len(case_tasks))]
    all_tasks = case_tasks + filler_tasks
    selection_by_task = selection.set_index("task_id")
    trait_rows = []
    fusion_records = []
    for task_id in all_tasks:
        is_case = task_id in selection_by_task.index
        formal = not (
            is_case
            and str(selection_by_task.loc[task_id, "case_role"]) == "fail_closed"
        )
        raw_sha = (
            str(selection_by_task.loc[task_id, "raw_source_image_sha256"])
            if is_case
            else sha256_json({"raw": task_id})
        )
        prediction_sha = (
            str(selection_by_task.loc[task_id, "prediction_sha256"])
            if is_case
            else sha256_json({"current_prediction": task_id})
        )
        trait_rows.append(
            {
                "task_id": task_id,
                "source_image_sha256": raw_sha,
                "formal_statistics_eligible": formal,
                "exclusion_reason": "" if formal else "review_only_fail_closed",
                "visible_root_axis_length_um": 6000.0 if formal else None,
                "hair_count": 10 if formal else None,
                "hair_length_measurement_hair_count": 7 if formal else None,
                "hair_length_measurement_fraction": 0.7 if formal else None,
                "distal_window_1_4mm_eligible": formal,
            }
        )
        fusion_records.append(
            {
                "task_id": task_id,
                "prediction_sha256": prediction_sha,
                "hair_identity_count_expert": builder.EXPECTED_STAGEB_EXPERT_ID,
                "hybrid_identity_count": 8,
                "phaxis_identity_count": 10,
                "matched_endpoint_complete_lengths": 7,
            }
        )
    full_traits = input_root / "full_traits.csv"
    pd.DataFrame(trait_rows).to_csv(full_traits, index=False, lineterminator="\n")

    fusion_payload = {
        "schema_version": "PHAxis-fusion-run-1.1",
        "status": "completed",
        "images": 283,
        "model_bundle_id": builder.EXPECTED_MODEL_BUNDLE_ID,
        "root_expert": builder.EXPECTED_ROOT_EXPERT_ID,
        "hair_identity_count_expert": builder.EXPECTED_STAGEB_EXPERT_ID,
        "records": fusion_records,
        "model_contract_proposal_sha256": builder.EXPECTED_PROPOSAL_FILE_SHA256,
        "model_contract_proposal_identity_sha256": builder.EXPECTED_PROPOSAL_IDENTITY_SHA256,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
    }
    fusion_payload = _seal(fusion_payload, "summary_identity_sha256")
    fusion_summary = _write_json(input_root / "fusion_summary.json", fusion_payload)

    traits_payload = {
        "schema_version": "PHAxis-trait-export-1.0",
        "status": "completed",
        "tasks": 283,
        "traits_sha256": sha256_file(full_traits),
        "model_bundle_id": builder.EXPECTED_MODEL_BUNDLE_ID,
        "root_expert_id": builder.EXPECTED_ROOT_EXPERT_ID,
        "hair_identity_count_expert": builder.EXPECTED_STAGEB_EXPERT_ID,
        "model_contract_proposal_sha256": builder.EXPECTED_PROPOSAL_FILE_SHA256,
        "model_contract_proposal_identity_sha256": builder.EXPECTED_PROPOSAL_IDENTITY_SHA256,
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    traits_payload = _seal(traits_payload, "export_identity_sha256")
    traits_summary = _write_json(input_root / "traits_summary.json", traits_payload)

    geometry_payload = json.loads(resources["figure1_geometry"].read_text(encoding="utf-8"))
    representative = selection_by_task.loc[builder.FIGURE1_TASK_ID]
    geometry_payload.update(
        {
            "schema_version": "PHAxis-figure1-display-geometry-1.0",
            "status": "completed_final_prediction_bound_display_geometry",
            "task_id": builder.FIGURE1_TASK_ID,
            "prediction_sha256": str(representative["prediction_sha256"]),
            "fusion_summary_sha256": sha256_file(fusion_summary),
            "fusion_summary_identity_sha256": fusion_payload["summary_identity_sha256"],
            "canonical_annotations_read": False,
            "condition_metadata_read": False,
            "root_cap_region_output": False,
            "blind_images_used": 0,
        }
    )
    geometry_payload = _seal(
        geometry_payload, "figure1_display_geometry_identity_sha256"
    )
    figure1_geometry = _write_json(
        input_root / "figure1_geometry_current.json", geometry_payload
    )
    figure1_receipt_payload = {
        "schema_version": "PHAxis-figure1-geometry-materialization-1.0",
        "status": "completed_from_preselected_case_and_final_prediction",
        "task_id": builder.FIGURE1_TASK_ID,
        "prediction_sha256": str(representative["prediction_sha256"]),
        "source_image_sha256": sha256_file(resources["figure1_image"]),
        "figure1_geometry_sha256": sha256_file(figure1_geometry),
        "figure1_display_geometry_identity_sha256": geometry_payload[
            "figure1_display_geometry_identity_sha256"
        ],
        "fusion_summary_sha256": sha256_file(fusion_summary),
        "fusion_summary_identity_sha256": fusion_payload["summary_identity_sha256"],
        "canonical_annotations_read": False,
        "condition_metadata_read": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    figure1_receipt_payload = _seal(
        figure1_receipt_payload,
        "figure1_geometry_materialization_identity_sha256",
    )
    figure1_receipt = _write_json(
        input_root / "figure1_receipt.json", figure1_receipt_payload
    )

    formal_case_tasks = [
        str(row["task_id"])
        for row in selection.to_dict("records")
        if bool(row["formal_statistics_eligible"])
    ]
    topology = input_root / "assurance_topology.csv"
    pd.DataFrame(
        [
            {
                "source_unit": task_id,
                "axis_in_root_coverage_fraction": 0.995,
                "axis_single_component_coverage_fraction": 0.990,
                "longest_unsupported_axis_gap_um": 8.0,
                "axis_containment_fraction": 0.995,
                "identity_hair_n": 10,
            }
            for task_id in formal_case_tasks
        ]
    ).to_csv(topology, index=False, lineterminator="\n")

    proposal_payload = json.loads(proposal.read_text(encoding="utf-8"))
    stageb = proposal_payload["promotion"]["stageb_binding"]
    assurance_payload = {
        "schema_version": "PHAxis-measurement-assurance-receipt-1.0",
        "status": "completed_locked_qc_development_assurance",
        "scope": "QC-development measurement assurance; non-independent",
        "source_table_sha256": {
            "metrics": sha256_file(resources["assurance_metrics"]),
            "pairs": sha256_file(resources["assurance_pairs"]),
            "support": sha256_file(resources["assurance_support"]),
            "topology": sha256_file(topology),
        },
        "source_authority_sha256": {
            "application_fusion_summary": sha256_file(fusion_summary),
        },
        "source_authority_identity_sha256": {
            "application_fusion_summary": fusion_payload["summary_identity_sha256"],
        },
        "shared_stageb_authority": {
            "expert_id": stageb["expert_id"],
            "checkpoint_sha256": stageb["checkpoint_sha256"],
            "selected_score_threshold": stageb["selected_score_threshold"],
            "candidate_bundle_identity_sha256": stageb[
                "candidate_bundle_identity_sha256"
            ],
            "selected_model_metadata_identity_sha256": stageb[
                "selected_model_metadata_identity_sha256"
            ],
            "selection_receipt_identity_sha256": stageb[
                "selection_receipt_identity_sha256"
            ],
        },
        "independent_accuracy_claim_allowed": False,
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    assurance_payload = _seal(
        assurance_payload, "measurement_assurance_identity_sha256"
    )
    assurance_receipt = _write_json(
        input_root / "assurance_receipt.json", assurance_payload
    )

    overlay_payload = {
        "schema_version": "PHAxis-manuscript-overlay-selection-receipt-1.2",
        "status": "completed_locked_preselected_gallery_and_exact_cohort_review_export",
        "selection_csv_sha256": sha256_file(resources["overlay_selection"]),
        "source_authority_sha256": {
            "fusion_summary": sha256_file(fusion_summary),
            "full_traits": sha256_file(full_traits),
        },
        "fusion_summary_identity_sha256": fusion_payload["summary_identity_sha256"],
        "images": 5,
        "exact_cohort_review_images": 283,
        "inset_contract": {
            "locked_anchor_task_ids": builder.FIGURE4_ANCHORS,
        },
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    overlay_payload = _seal(overlay_payload, "overlay_selection_identity_sha256")
    overlay_receipt = _write_json(input_root / "overlay_receipt.json", overlay_payload)

    inputs = builder.Inputs(
        model_contract_proposal=proposal,
        fusion_summary=fusion_summary,
        traits_summary=traits_summary,
        full_traits=full_traits,
        trait_contract=trait_contract,
        trait_story_map=story,
        trait_story_receipt=story_receipt,
        figure1_receipt=figure1_receipt,
        figure1_image=resources["figure1_image"],
        figure1_geometry=figure1_geometry,
        assurance_receipt=assurance_receipt,
        assurance_metrics=resources["assurance_metrics"],
        assurance_pairs=resources["assurance_pairs"],
        assurance_support=resources["assurance_support"],
        assurance_topology=topology,
        overlay_receipt=overlay_receipt,
        overlay_selection=resources["overlay_selection"],
    )
    expected = {
        role: sha256_file(getattr(inputs, role)) for role in builder.INPUT_ROLES
    }
    return inputs, expected


def test_current_lineage_focused_render_is_create_only(tmp_path: Path, monkeypatch) -> None:
    inputs, expected = _current_fixture(tmp_path)
    monkeypatch.setattr(builder._figures, "save_figure_bundle", _fast_bundle)
    output = tmp_path / "paper_fig134_result"
    receipt = builder.build_paper_first_figures(
        inputs=inputs,
        expected_sha256=expected,
        output=output,
    )
    assert receipt["status"] == builder.STATUS
    assert receipt["model_bundle_id"] == builder.EXPECTED_MODEL_BUNDLE_ID
    assert receipt["old_443cv_hair_results_used"] is False
    assert receipt["gpu_program_started"] is False
    assert receipt["blind_images_used"] == 0
    assert receipt["figure_count"] == 3
    assert set(receipt["figure4_locked_anchor_task_ids"].values()) == set(
        builder.FIGURE4_ANCHORS.values()
    )
    for stem in builder.FIGURE_STEMS.values():
        for suffix in (".pdf", ".png", ".tiff"):
            assert (output / f"{stem}{suffix}").is_file()
    audit = pd.read_csv(output / "source_data/Figure4_case_audit.csv")
    assert set(audit.loc[audit["case_role"].isin(builder.FIGURE4_ANCHORS), "task_id"]) == set(
        builder.FIGURE4_ANCHORS.values()
    )
    assert "(f) Raw-image-to-atlas outputs" in (
        output / "FIGURE_LEGENDS.md"
    ).read_text(encoding="utf-8")
    with pytest.raises(builder.PaperFirstFigureError, match="overwrite"):
        builder.build_paper_first_figures(
            inputs=inputs,
            expected_sha256=expected,
            output=output,
        )


def test_current_six_panel_figure1_passes_real_publication_bundle(tmp_path: Path) -> None:
    inputs, expected = _current_fixture(tmp_path)
    prepared = builder.prepare_inputs(inputs, expected)
    figure = builder._figure1_current(prepared.resources, prepared.story)
    bundle = builder._figures.save_figure_bundle(
        figure,
        tmp_path / "figure1_real_smoke",
        width_mm=178.0,
        height_mm=165.0,
        check_edge_ink=True,
    )
    assert bundle["png_dpi"][0] == pytest.approx(600.0, abs=0.2)
    assert bundle["tiff_mode"] == "RGB"
    assert bundle["edge_ink_pixels_outer_2px"] == 0


def test_hash_failure_leaves_no_published_output(tmp_path: Path) -> None:
    inputs, expected = _current_fixture(tmp_path)
    expected["overlay_selection"] = "0" * 64
    output = tmp_path / "hash_failure_output"
    with pytest.raises(builder.PaperFirstFigureError, match="file SHA-256 mismatch"):
        builder.build_paper_first_figures(
            inputs=inputs,
            expected_sha256=expected,
            output=output,
        )
    assert not output.exists()


def test_locked_low_contrast_anchor_cannot_be_substituted_even_when_rehashed(
    tmp_path: Path,
) -> None:
    inputs, expected = _current_fixture(tmp_path)
    selection = pd.read_csv(inputs.overlay_selection)
    selection.loc[
        selection["case_role"].eq("low_contrast"), "task_id"
    ] = builder.FIGURE1_TASK_ID
    selection.to_csv(inputs.overlay_selection, index=False, lineterminator="\n")
    overlay = json.loads(inputs.overlay_receipt.read_text(encoding="utf-8"))
    overlay["selection_csv_sha256"] = sha256_file(inputs.overlay_selection)
    overlay = _seal(overlay, "overlay_selection_identity_sha256")
    _write_json(inputs.overlay_receipt, overlay)
    expected["overlay_selection"] = sha256_file(inputs.overlay_selection)
    expected["overlay_receipt"] = sha256_file(inputs.overlay_receipt)
    output = tmp_path / "anchor_substitution_output"
    with pytest.raises(builder.PaperFirstFigureError, match="anchor changed"):
        builder.build_paper_first_figures(
            inputs=inputs,
            expected_sha256=expected,
            output=output,
        )
    assert not output.exists()


def test_entry_point_has_no_gpu_or_run_discovery_route() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "import torch" not in source
    assert "cuda" not in source
    assert "nvidia-smi" not in source
    assert ".glob(" not in source
    assert "rglob(" in source  # only hashes files already created in staging
    assert builder.EXPECTED_CHECKPOINT_SHA256 == tuple(
        json.loads(
            (
                PROJECT_ROOT
                / "outputs/phaxis_paperfirst_public327_authority_run1/proposal.json"
            ).read_text(encoding="utf-8")
        )["promotion"]["stageb_binding"]["checkpoint_sha256"]
    )
