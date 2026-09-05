from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phaxis.hair_attachment_assurance import (
    FORMAL_ATTACHMENT_TOLERANCE_UM,
    FORMAL_MATCHER_CONFIG,
    HAIR_ATTACHMENT_COORDINATE_SPACE,
    HAIR_ATTACHMENT_INPUT_SCHEMA,
    HAIR_POLYLINE_ORIENTATION,
    PROXY_TOLERANCES_UM,
    SELECTED_PROXY_TOLERANCE_UM,
    build_from_input_contract as build_hair_attachment_from_input_contract,
)
from phaxis.hair_stageb.candidate_bundle import (
    CANDIDATE_STATUS,
    FORMAL_TRAIN399_SEEDS,
    TRAIN399_CHECKPOINT_POLICY,
    validate_train399_detection_model_metadata,
)
from phaxis.hair_stageb.serialization import make_detection_payload
from phaxis.io import sha256_json
from phaxis.root_continuity_assurance import (
    ROOT_CONTINUITY_COORDINATE_SPACE,
    ROOT_CONTINUITY_INPUT_SCHEMA,
    ROOT_CONTINUITY_PREDICTION_DEFINITION,
    ROOT_CONTINUITY_REFERENCE_DEFINITION,
    build_from_input_contract as build_root_continuity_from_input_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_producer():
    path = PROJECT_ROOT / "scripts/phaxis/build_measurement_assurance_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "phaxis_measurement_assurance_component_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


producer = _load_producer()


def _hash(role: str) -> str:
    return sha256_json({"role": role})


def test_application_axis_support_separates_union_from_one_component() -> None:
    axis = np.column_stack((np.arange(10, dtype=np.float64), np.full(10, 2.0)))
    fragmented = np.zeros((6, 12), dtype=bool)
    fragmented[2, 0:4] = True
    fragmented[2, 6:10] = True
    metrics = producer._axis_root_support_metrics(
        fragmented, axis, um_per_px=2.0
    )
    assert metrics["axis_in_root_coverage_fraction"] == pytest.approx(0.8)
    assert metrics["axis_single_component_coverage_fraction"] == pytest.approx(0.4)
    assert metrics["longest_unsupported_axis_gap_um"] == pytest.approx(12.0)
    assert metrics["root_mask_component_count"] == 2

    continuous = np.zeros_like(fragmented)
    continuous[2, 0:10] = True
    metrics = producer._axis_root_support_metrics(
        continuous, axis, um_per_px=2.0
    )
    assert metrics["axis_in_root_coverage_fraction"] == 1.0
    assert metrics["axis_single_component_coverage_fraction"] == 1.0
    assert metrics["longest_unsupported_axis_gap_um"] == 0.0


def _selected_model_metadata() -> dict:
    checkpoint_sha256 = [_hash(f"checkpoint-{index}") for index in range(5)]
    model = {
        "checkpoint_policy": TRAIN399_CHECKPOINT_POLICY,
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "ensemble_members": 5,
        "deployment_role": CANDIDATE_STATUS,
        "training_images": 399,
        "validation_images": 44,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
        "seeds": list(FORMAL_TRAIN399_SEEDS),
        "member_ids": [f"seed_{seed}" for seed in FORMAL_TRAIN399_SEEDS],
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": [
            _hash(f"model-state-{index}") for index in range(5)
        ],
        "training_task_ids_sha256": _hash("training-task-ids"),
        "split_manifest_sha256": _hash("split-manifest"),
        "training_lock_identity_sha256": _hash("training-lock"),
        "candidate_bundle_identity_sha256": _hash("candidate-bundle"),
        "operating_point_selection_contract_sha256": _hash(
            "operating-point-contract"
        ),
        "operating_point_status": "selected_on_locked_QCdevelopment44",
        "selected_score_threshold": 0.225,
        "selection_receipt_sha256": _hash("selection-receipt-file"),
        "selection_receipt_identity_sha256": _hash(
            "selection-receipt-identity"
        ),
        "candidate_pool_identity_sha256": _hash("candidate-pool"),
    }
    model["selected_model_metadata_identity_sha256"] = sha256_json(model)
    model["precision_mode"] = "fp32_locked"
    validate_train399_detection_model_metadata(model)
    return model


def _authority_fixture() -> tuple[list[str], dict, dict]:
    task_ids = [f"QC-{index:03d}" for index in range(44)]
    model = _selected_model_metadata()
    evaluation_files = [
        {"task_id": task_id, "sha256": _hash(f"eval-wrapper-{task_id}")}
        for task_id in task_ids
    ]
    evaluation = {
        "training_contract": {
            "training_images": 399,
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "checkpoint_sha256": list(model["checkpoint_sha256"]),
            "candidate_bundle_identity_sha256": model[
                "candidate_bundle_identity_sha256"
            ],
            "selected_model_metadata_identity_sha256": model[
                "selected_model_metadata_identity_sha256"
            ],
            "selection_receipt_identity_sha256": model[
                "selection_receipt_identity_sha256"
            ],
        },
        "prediction_input_locks": {
            "stageb_detection_files": evaluation_files,
            "stageb_detection_set_identity_sha256": sha256_json(evaluation_files),
        },
    }
    production_records = [
        {
            "task_id": task_id,
            "detections": 1,
            "detection_file_sha256": _hash(f"production-file-{task_id}"),
            "detection_identity_sha256": _hash(
                f"production-logical-{task_id}"
            ),
        }
        for task_id in task_ids
    ]
    stageb = {
        "images": 44,
        "detections": 44,
        "checkpoint_sha256": list(model["checkpoint_sha256"]),
        "detection_model_metadata": model,
        "score_threshold": model["selected_score_threshold"],
        "records": production_records,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
    }
    return task_ids, evaluation, stageb


def _production_detection() -> dict:
    model = _selected_model_metadata()
    prediction = {
        "base": np.asarray([[5.0, 6.0], [15.0, 12.0]], dtype=np.float32),
        "tip": np.asarray([[2.0, 3.0], [10.0, 4.0]], dtype=np.float32),
        "score": np.asarray([0.9, 0.8], dtype=np.float32),
        "length_um": np.asarray([12.0, 18.0], dtype=np.float32),
        "working_shape": [32, 64],
        "source_shape": [64, 128],
        "source_to_working_scale": 0.5,
        "source_to_working_scale_xy": [0.5, 0.5],
        "realized_um_per_px_xy": [2.0, 2.0],
        "tip_snapped": np.asarray([True, False]),
        "length_semantics": (
            "regressed_polyline_arc_length_um_diagnostic_only"
        ),
    }
    payload = make_detection_payload(
        task_id="QC-000",
        source_image_sha256=_hash("source-image"),
        source_um_per_px=1.0,
        prediction=prediction,
        precision_mode="fp32_locked",
        model_metadata=model,
        score_threshold=0.225,
    )
    payload.pop("detection_identity_sha256")
    payload.update(
        {
            "model_contract_proposal_sha256": _hash("proposal-file"),
            "model_contract_proposal_identity_sha256": _hash(
                "proposal-identity"
            ),
            "model_bundle_id": "PHAxis-model-bundle-SYNTHETIC",
            "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
        }
    )
    payload["detection_identity_sha256"] = sha256_json(payload)
    return payload


def _fused_prediction(stageb: dict) -> dict:
    hairs = deepcopy(producer._stageb_hairs(stageb))
    for index, hair in enumerate(hairs):
        hair.update(
            {
                "root_axis_projection_xy": [float(index), 0.0],
                "root_attachment_valid": True,
                "complete_length_measurement_eligible": index == 0,
            }
        )
    model = stageb["model"]
    return {
        "identity_hair_variant": "phaxis_stage_b_train399_five_seed_identity",
        "identity_hairs": hairs,
        "count_hair_variant": "phaxis_stage_b_train399_five_seed_count",
        "count_hairs": deepcopy(hairs),
        "blind_images_used": 0,
        "phaxis": {
            "hair_identity_count_expert": model["expert_id"],
            "hair_identity_count_checkpoint_policy": TRAIN399_CHECKPOINT_POLICY,
            "hair_identity_count_candidate_bundle_identity_sha256": model[
                "candidate_bundle_identity_sha256"
            ],
            "stageb_detection_identity_sha256": stageb[
                "detection_identity_sha256"
            ],
            "formal_stageb_identity_count": stageb["n"],
            "expert_boundary": {
                "phaxis_stage_b_train399": ["hair_identity", "hair_count"]
            },
        },
    }


def _hair_attachment_receipt() -> dict:
    record = {
        "pair_type": "hair_attachment",
        "source_unit": "hair-one",
        "source_image_sha256": _hash("hair-image"),
        "coordinate_space": HAIR_ATTACHMENT_COORDINATE_SPACE,
        "polyline_orientation": HAIR_POLYLINE_ORIENTATION,
        "annotation_artifact_sha256": _hash("hair-annotation"),
        "prediction_artifact_sha256": _hash("stageb-detection"),
        "predicted_polylines_xy_um": [[[15.0, 0.0], [100.0, 0.0]]],
        "annotated_polylines_xy_um": [[[0.0, 0.0], [100.0, 0.0]]],
    }
    contract = producer._seal_portable_input_contract(
        {
            "schema_version": HAIR_ATTACHMENT_INPUT_SCHEMA,
            "source_units": ["hair-one"],
            "annotation_authority_sha256": _hash("annotation-authority"),
            "prediction_authority_identity_sha256": _hash(
                "hair-prediction-authority"
            ),
            "metric_config": {
                "proxy_tolerances_um": list(PROXY_TOLERANCES_UM),
                "selected_proxy_tolerance_um": SELECTED_PROXY_TOLERANCE_UM,
                "formal_attachment_tolerance_um": (
                    FORMAL_ATTACHMENT_TOLERANCE_UM
                ),
                "formal_matcher": dict(FORMAL_MATCHER_CONFIG),
            },
            "records": [record],
            "independent_accuracy_claim_allowed": False,
            "blind_images_used": 0,
        }
    )
    return build_hair_attachment_from_input_contract(contract)


def test_eval_only_and_production_file_sets_may_differ_when_authority_matches() -> None:
    task_ids, evaluation, stageb = _authority_fixture()
    result = producer._validate_stageb_authority(evaluation, stageb, task_ids)
    assert (
        result["evaluation_detection_ordered_file_set_identity_sha256"]
        != result["production_detection_ordered_file_set_identity_sha256"]
    )
    assert result["shared_model_authority"][
        "selected_model_metadata_identity_sha256"
    ] == stageb["detection_model_metadata"][
        "selected_model_metadata_identity_sha256"
    ]


def test_eval_production_shared_model_identity_drift_fails_closed() -> None:
    task_ids, evaluation, stageb = _authority_fixture()
    evaluation["training_contract"][
        "selected_model_metadata_identity_sha256"
    ] = _hash("different-selected-model")
    with pytest.raises(
        producer.MeasurementAssuranceError,
        match="selected_model_metadata_identity_sha256 drift",
    ):
        producer._validate_stageb_authority(evaluation, stageb, task_ids)


def test_authoritative_stageb_conversion_matches_fused_identity_order() -> None:
    stageb = _production_detection()
    fused = _fused_prediction(stageb)
    physical = producer._validate_fused_stageb_identity(fused, stageb)
    assert len(physical) == 2
    np.testing.assert_allclose(
        physical[0],
        np.asarray([[10.0, 12.0], [4.0, 6.0]]),
        rtol=0.0,
        atol=1e-12,
    )
    assert [hair["source_instance_id"] for hair in fused["identity_hairs"]] == [
        "PHSB-0001",
        "PHSB-0002",
    ]


def test_fused_identity_hair_geometry_drift_fails_closed() -> None:
    stageb = _production_detection()
    fused = _fused_prediction(stageb)
    fused["identity_hairs"][0]["points_xy"][0][0] += 1.0
    fused["count_hairs"] = deepcopy(fused["identity_hairs"])
    with pytest.raises(
        producer.MeasurementAssuranceError,
        match=r"hair 0 points_xy drift",
    ):
        producer._validate_fused_stageb_identity(fused, stageb)


def test_all_skeleton_components_keep_union_coverage_from_hiding_fracture() -> None:
    mask = np.zeros((5, 111), dtype=bool)
    mask[2, :51] = True
    mask[2, 60:] = True
    components = producer._skeleton_components_xy_um(mask, um_per_px=1.0)
    assert len(components) == 2
    for component in components:
        assert np.max(
            np.abs(np.diff(np.asarray(component, dtype=float), axis=0))
        ) <= 1.0
    record = {
        "pair_type": "primary_root_continuity",
        "source_unit": "fractured",
        "source_image_sha256": _hash("fractured-image"),
        "coordinate_space": ROOT_CONTINUITY_COORDINATE_SPACE,
        "reference_axis_definition": ROOT_CONTINUITY_REFERENCE_DEFINITION,
        "prediction_axis_definition": ROOT_CONTINUITY_PREDICTION_DEFINITION,
        "reference_axis_artifact_sha256": _hash("reference-axis"),
        "prediction_axis_artifact_sha256": _hash("prediction-mask"),
        "reference_axis_xy_um": [[0.0, 2.0], [110.0, 2.0]],
        "predicted_axis_components_xy_um": components,
    }
    contract = producer._seal_portable_input_contract(
        {
            "schema_version": ROOT_CONTINUITY_INPUT_SCHEMA,
            "source_units": ["fractured"],
            "reference_authority_sha256": _hash("reference-authority"),
            "prediction_authority_identity_sha256": _hash(
                "prediction-authority"
            ),
            "metric_config": {
                "support_tolerance_um": 5.0,
                "sampling_step_um": 2.0,
            },
            "records": [record],
            "independent_accuracy_claim_allowed": False,
            "blind_images_used": 0,
        }
    )
    receipt = build_root_continuity_from_input_contract(contract)
    row = receipt["per_image"][0]
    assert row["union_reference_axis_fully_supported"] is True
    assert row["maximum_single_component_coverage"] < 1.0
    assert row["union_coverage_hides_fragmentation"] is True
    assert row["break_free"] is False


def test_hair_portable_contract_keeps_formal_match_separate_from_base_proxy() -> None:
    receipt = _hair_attachment_receipt()
    assert receipt["metric_contract"][
        "threshold_selection_used_as_formal_accuracy"
    ] is False
    formal = receipt["summary"]["formal_matched_attachment_accuracy"]
    assert formal["formal_biological_presence"]["tp"] == 1
    assert formal["attachment_qualified_identity"]["tp"] == 1
    assert set(
        receipt["summary"]["attachment_proxy_threshold_selection"]
    ) == {"5", "10", "20"}


def test_production_evaluator_biological_presence_drift_fails_closed() -> None:
    receipt = _hair_attachment_receipt()
    formal = receipt["summary"]["formal_matched_attachment_accuracy"][
        "formal_biological_presence"
    ]
    evaluation = {
        "per_image": [
            {
                "task_id": "hair-one",
                "stageb_train399": {
                    "n_pred": formal["n_pred"],
                    "n_gt": formal["n_gt"],
                    "biological_presence_tp": {"20.0": formal["tp"]},
                },
            }
        ],
        "overall": {
            "stageb_train399": {
                "predicted_hairs": formal["n_pred"],
                "ground_truth_hairs": formal["n_gt"],
                "tolerant_biological_presence": {"20": dict(formal)},
            }
        },
    }
    locks, identity = producer._crosscheck_hair_biological_presence(
        evaluation, receipt, ["hair-one"]
    )
    assert identity == sha256_json(locks)
    assert locks[0]["biological_presence_tp_20um"] == 1

    evaluation["per_image"][0]["stageb_train399"][
        "biological_presence_tp"
    ]["20.0"] = 0
    with pytest.raises(
        producer.MeasurementAssuranceError,
        match="counts differ from evaluator Stage-B@20um",
    ):
        producer._crosscheck_hair_biological_presence(
            evaluation, receipt, ["hair-one"]
        )
    evaluation["per_image"][0]["stageb_train399"][
        "biological_presence_tp"
    ]["20.0"] = formal["tp"]
    evaluation["overall"]["stageb_train399"][
        "tolerant_biological_presence"
    ]["20"]["tp"] = 0
    with pytest.raises(
        producer.MeasurementAssuranceError,
        match="overall biological-presence counts differ",
    ):
        producer._crosscheck_hair_biological_presence(
            evaluation, receipt, ["hair-one"]
        )
