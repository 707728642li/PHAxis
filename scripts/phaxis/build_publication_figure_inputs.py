#!/usr/bin/env python3
"""Compile the PHAxis manuscript figure inputs from named sealed evidence.

This command is the only production route to ``figure_inputs.json``.  It does
not discover a newest output, accept already-normalized plotting numbers, or
read a blind dataset.  Every plotted cell is either recomputed from a named
source-unit table/receipt or copied after byte-hash and sealed-identity checks.

The complete directory is assembled in a sibling staging directory and moved
into place atomically.  ``final`` mode is deliberately unreachable until the
historical-development, measurement-assurance, overlay-selection, profile-
analysis, and two-mode direct-runtime receipts are all final and mutually
bound to the eight core PHAxis receipts.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.contracts import ContractError  # noqa: E402
from phaxis.biological_analysis import (  # noqa: E402
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    raw_median_bootstrap_seed,
)
from phaxis.hair_stageb.evaluation_inference import (  # noqa: E402
    EVALUATION_ARTIFACT_ROLE,
    EVALUATION_DETECTION_SCHEMA,
    EVALUATION_RUN_SCHEMA,
)
from phaxis.public_identity import validate_proposal_public_identity  # noqa: E402
from phaxis.publication_evidence import (  # noqa: E402
    FIGURE_SOURCE_INPUT_ROLES,
    WT_SECONDARY_RESOURCE_ROLES,
    WT_SECONDARY_TABLE_FILENAMES,
    supplementary_figure_contract,
    validate_wt_secondary_analysis_binding,
    validate_wt_secondary_evidence,
)
from phaxis.multitrait_atlas import (  # noqa: E402
    MultitraitAtlasError,
    build_multitrait_atlas,
)
from phaxis import hair_attachment_assurance as _hair_attachment  # noqa: E402
from phaxis import root_continuity_assurance as _root_continuity  # noqa: E402
from phaxis.evaluation_metrics import precision_recall_f1  # noqa: E402
from phaxis.narrative_decision import (  # noqa: E402
    build_narrative_decision,
    validate_narrative_decision,
)
from phaxis.publication_authority import (  # noqa: E402
    PublicationAuthorityError,
    build_qcdev_assignment,
    derive_overlay_audit,
)


INPUT_SCHEMA_VERSION = "PHAxis-manuscript-figure-inputs-2.0"
ASSEMBLER_SCHEMA_VERSION = "PHAxis-publication-figure-input-assembly-1.0"
HISTORICAL_RECEIPT_SCHEMA = "PHAxis-historical-OOF443-development-receipt-1.0"
ASSURANCE_RECEIPT_SCHEMA = "PHAxis-measurement-assurance-receipt-1.0"
OVERLAY_RECEIPT_SCHEMA = "PHAxis-manuscript-overlay-selection-receipt-1.2"
OVERLAY_RECEIPT_STATUS = (
    "completed_locked_preselected_gallery_and_exact_cohort_review_export"
)
OVERLAY_REVIEW_SCHEMA = "PHAxis-exact-cohort-review-overlay-export-1.0"
OVERLAY_REVIEW_STATUS = "completed_exact_cohort_final_fusion_review_export"
OVERLAY_REVIEW_PENDING_STATUS = "pending_manual_visual_review"
OVERLAY_CASE_SELECTION_BASIS = "preselected_morphology_acquisition_challenge_roles"
OVERLAY_INSET_ROLES = ("low_contrast", "curved_dense")
OVERLAY_LOCKED_ANCHOR_TASK_IDS = {
    "low_contrast": "RHSCU-aa5b6e37df15821f",
    "curved_dense": "RHSCU-bbf649822174e0a2",
}
PROFILE_ANALYSIS_SCHEMA = "PHAxis-distal-axis-profile-analysis-1.0.0"
FIGURE_INPUT_STAGING_PREFIX = ".figure-inputs-"
SCALE_ABSENCE_SPECIFICITY_STATUS = (
    "not_estimable_no_absent_or_untrusted_scale_cases"
)
SCALE_FAIL_CLOSED_EVIDENCE_BASIS = "software_contract_and_unit_tests"
ROOT_CONTINUITY_FORMAL_METRIC_KEYS = (
    "root_continuity_maximum_single_component_coverage_mean",
    "root_continuity_maximum_single_component_coverage_median",
    "root_continuity_best_component_gap_median_um",
    "root_continuity_break_free_rate",
    "root_continuity_visible_axis_extent_mae_um",
)
HAIR_ATTACHMENT_FORMAL_METRIC_KEYS = (
    "hair_attachment_qualified_precision_20um",
    "hair_attachment_qualified_recall_20um",
    "hair_attachment_qualified_f1_20um",
    "hair_attachment_error_median_um",
    "hair_attachment_error_p95_um",
)
ROOT_CONTINUITY_DIAGNOSTIC_METRIC_KEY = (
    "root_continuity_reference_axis_coverage_mean"
)

CORE_ROLES = (
    "train399_evaluation",
    "root_exact283",
    "stageb",
    "fusion",
    "traits",
    "cohorts",
    "analysis",
    "profiles",
)
RESOURCE_ROLES = (
    "trait_contract",
    "figure1_image",
    "figure1_geometry",
    "development_per_image",
    "development_tolerance",
    "development_threshold",
    "development_strata",
    "assurance_metrics",
    "assurance_pairs",
    "assurance_support",
    "qcdev_assignment",
    "overlay_selection",
    "overlay_audit",
    "phenotype_points",
    "phenotype_effects",
    "narrative_decision",
    "multitrait_atlas",
    "axial_profiles",
    "cohort_flow",
    "workflow_stages",
    "runtime_summary",
    "runtime_per_image",
    *WT_SECONDARY_RESOURCE_ROLES,
)
QC_COMPARATOR_MAP = {
    "stageb_train399": "stageb_train399",
    "hybrid_max": "legacy_hybrid_endpoint_complete_identity_layer",
}
# Backward-compatible module name; formal QCdev evidence is intentionally the
# non-deployable wrapper schema, never a production Stage-B detection schema.
STAGEB_DETECTION_SCHEMA = EVALUATION_DETECTION_SCHEMA
LEGACY_HYBRID_COMPARATOR_SCHEMA = (
    "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
)
LEGACY_HYBRID_IDENTITY_VARIANT = "hybrid_verified_increment"
LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256 = (
    "ede309b8a828aec35be64d9f8afbc2ac9bf92b5a9e1b1b262d5acf603a746f36"
)
HISTORICAL_COMPARATOR = "historical_family_isolated_oof443"
GROUP_ORDER = ("RHD6_EV_22C", "RHD6_EV_30C", "RHD6_OE_22C", "RHD6_OE_30C")
PRIMARY_ENDPOINTS = (
    "local_hair_count_1_4mm",
    "local_median_hair_length_um_1_4mm",
    "first_hair_ge40um_distance_from_distal_point_um",
    "median_root_width_um",
    "visible_root_axis_length_um",
)
PRIMARY_ENDPOINT_COMPONENTS = {
    PRIMARY_ENDPOINTS[0]: "count_rate",
    PRIMARY_ENDPOINTS[1]: "continuous",
    PRIMARY_ENDPOINTS[2]: "continuous",
    PRIMARY_ENDPOINTS[3]: "continuous",
    PRIMARY_ENDPOINTS[4]: "continuous",
}
ENDPOINT_UNITS = {
    PRIMARY_ENDPOINTS[0]: "count",
    PRIMARY_ENDPOINTS[1]: "um",
    PRIMARY_ENDPOINTS[2]: "um",
    PRIMARY_ENDPOINTS[3]: "um",
    PRIMARY_ENDPOINTS[4]: "um",
}
EFFECT_MAP = {
    "construct_OE_minus_EV": "OE_vs_EV",
    "temperature_30C_minus_22C": "30C_vs_22C",
    "construct_by_temperature_interaction": "interaction",
}
EFFECT_SOURCE_ORDER = tuple(EFFECT_MAP)
EFFECT_ORDER = tuple(EFFECT_MAP[source_key] for source_key in EFFECT_SOURCE_ORDER)
PHENOTYPE_EFFECT_COHORT_ORDER = ("primary_clean261", "sensitivity_full283")
H11_ENDPOINT = "local_median_hair_length_um_1_4mm"
H11_RAW_BOOTSTRAP_REPLICATES = 5000
H11_RAW_BOOTSTRAP_BASE_SEED = 20260823
CASE_ROLES = ("representative", "low_contrast", "curved_dense", "continuity", "fail_closed")
IDENTITY_FIELDS = {
    "model_contract_proposal": "model_contract_identity_sha256",
    "train399_selection": "selection_receipt_identity_sha256",
    "historical_development": "historical_development_identity_sha256",
    "measurement_assurance": "measurement_assurance_identity_sha256",
    "overlay_index": "overlay_selection_identity_sha256",
    "profile_analysis": "analysis_identity_sha256",
    "runtime_latency": "summary_identity_sha256",
    "runtime_production": "summary_identity_sha256",
    "runtime_latency_comparison": "comparison_identity_sha256",
    "runtime_production_comparison": "comparison_identity_sha256",
    "baseline_runtime_latency": "summary_identity_sha256",
    "baseline_runtime_production": "summary_identity_sha256",
}
FINAL_PATH_MARKERS = ("provisional", "blocked_pending", "not_for_submission")


class FigureInputAssemblyError(RuntimeError):
    """A source is not sufficient to construct publication-safe inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureInputAssemblyError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value[key]
            yield from _walk(value[key], path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            yield path, item
            yield from _walk(item, path)


def _resolve_file(value: str | Path, role: str, *, final: bool) -> Path:
    path = Path(value).resolve()
    _require(path.is_file(), f"{role}: missing file: {path}")
    _require(not path.is_symlink(), f"{role}: symlink inputs are forbidden")
    lowered = str(path).casefold()
    _require("blind" not in lowered, f"{role}: blind-labelled path refused")
    if final:
        _require(
            not any(marker in lowered for marker in FINAL_PATH_MARKERS),
            f"{role}: provisional/non-final path refused in final mode",
        )
    return path


def _read_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except Exception as error:
        raise FigureInputAssemblyError(f"{role}: invalid UTF-8 JSON object") from error
    _require(isinstance(payload, dict), f"{role}: JSON root must be an object")
    return payload


def _read_table(path: Path, role: str, columns: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise FigureInputAssemblyError(f"{role}: unreadable CSV") from error
    missing = [column for column in columns if column not in frame.columns]
    _require(not missing, f"{role}: missing columns {missing}")
    _require(len(frame) > 0, f"{role}: empty table")
    return frame


def _guard_red_lines(role: str, payload: Mapping[str, Any]) -> None:
    blind_fields = 0
    for path, value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if leaf == "blind_images_used":
            blind_fields += 1
            _require(value == 0, f"{role}: {path} must be 0")
        if leaf in {"root_cap_region_output", "root_cap_region_statistics_included"}:
            _require(value is False, f"{role}: {path} must be false")
        if leaf == "condition_metadata_used_for_routing":
            _require(value is False, f"{role}: condition metadata routed inference")
    _require(blind_fields > 0, f"{role}: explicit blind_images_used guard missing")


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    identity = payload.get(field)
    _require(_is_sha256(identity), f"{role}: invalid or missing {field}")
    unsigned = deepcopy(dict(payload))
    unsigned.pop(field, None)
    _require(sha256_json(unsigned) == identity, f"{role}: sealed identity mismatch")
    return str(identity)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def _finite_number(value: Any, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FigureInputAssemblyError(f"{role}: value is not numeric") from error
    _require(math.isfinite(result), f"{role}: non-finite value")
    return result


def _integer_number(value: Any, role: str) -> int:
    result = _finite_number(value, role)
    _require(result.is_integer(), f"{role}: integer required")
    return int(result)


def _prf(tp: int, predicted: int, truth: int) -> tuple[float, float, float]:
    precision = tp / predicted if predicted else (1.0 if truth == 0 else 0.0)
    recall = tp / truth if truth else (1.0 if predicted == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _ccc(observed: np.ndarray, predicted: np.ndarray) -> float:
    first = np.asarray(observed, dtype=np.float64)
    second = np.asarray(predicted, dtype=np.float64)
    _require(
        first.shape == second.shape
        and first.ndim == 1
        and len(first) >= 2
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second)),
        "CCC sufficient statistics are invalid",
    )
    covariance = float(np.cov(first, second, ddof=1)[0, 1])
    denominator = float(
        np.var(first, ddof=1)
        + np.var(second, ddof=1)
        + (np.mean(first) - np.mean(second)) ** 2
    )
    _require(denominator > 0.0, "CCC denominator is zero")
    return float(2.0 * covariance / denominator)


def _group_seed(base_seed: int, *labels: str) -> int:
    return int(
        np.random.SeedSequence(
            [base_seed, *[int.from_bytes(label.encode("utf-8")[:8].ljust(8, b"\0"), "big") for label in labels]]
        ).generate_state(1)[0]
    )


def _bootstrap_prf(
    sufficient: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    _require(len(sufficient) >= 2, "bootstrap: fewer than two source units")
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = sufficient[generator.integers(0, len(sufficient), len(sufficient))]
        values[index] = _prf(
            int(sample[:, 0].sum()),
            int(sample[:, 1].sum()),
            int(sample[:, 2].sum()),
        )[2]
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _copy_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _require(sha256_file(source) == sha256_file(destination), f"copy verification failed: {source}")


def _validate_train399_prediction_inputs(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock both QC44 prediction authorities and their ordered task identity.

    The evaluator hashes every concrete prediction JSON.  The assembler does
    not need those large directories, but it must refuse a receipt whose two
    ordered lock lists, logical set identities, per-image order, or explicitly
    declared legacy-comparator semantics differ.
    """

    per_image = evaluation.get("per_image")
    _require(
        isinstance(per_image, list) and len(per_image) == 44,
        "train399 evaluation per-image scope is not QC44",
    )
    task_order = [str(row.get("task_id")) for row in per_image]
    _require(
        len(set(task_order)) == 44 and all(task_order),
        "train399 evaluation task order is not 44 unique IDs",
    )
    locks = evaluation.get("prediction_input_locks")
    _require(isinstance(locks, Mapping), "train399 prediction input locks missing")
    evaluation_authority = evaluation.get("evaluation_inference_authority")
    _require(
        isinstance(evaluation_authority, Mapping)
        and evaluation_authority.get("schema_version") == EVALUATION_RUN_SCHEMA
        and evaluation_authority.get("artifact_role") == EVALUATION_ARTIFACT_ROLE
        and evaluation_authority.get("evaluation_detection_schema_version")
        == EVALUATION_DETECTION_SCHEMA,
        "train399 evaluation-only inference authority/schema missing",
    )
    for field in (
        "evaluation_inference_summary_sha256",
        "evaluation_inference_summary_identity_sha256",
        "evaluation_gate_identity_sha256",
        "evaluation_detection_set_identity_sha256",
    ):
        _require(
            _is_sha256(evaluation_authority.get(field)),
            f"train399 evaluation-only authority {field} is invalid",
        )
    _require(
        evaluation_authority.get("model_contract_proposal_required_for_artifact")
        is False
        and evaluation_authority.get("model_contract_proposal_present") is False
        and evaluation_authority.get("production_consumption_allowed") is False
        and evaluation_authority.get("fusion_consumption_allowed") is False
        and evaluation_authority.get("traits_consumption_allowed") is False
        and evaluation_authority.get("canonical_annotations_read_during_inference")
        is False
        and evaluation_authority.get("condition_metadata_used_for_routing") is False
        and evaluation_authority.get("independent_accuracy_claim_allowed") is False
        and evaluation_authority.get("blind_images_used") == 0,
        "train399 evaluation-only artifacts are circular, deployable, or tainted",
    )
    source_hashes = evaluation.get("inputs_sha256")
    training_contract = evaluation.get("training_contract")
    _require(
        isinstance(source_hashes, Mapping)
        and source_hashes.get("evaluation_inference_summary")
        == evaluation_authority["evaluation_inference_summary_sha256"]
        and isinstance(training_contract, Mapping)
        and training_contract.get("evaluation_gate_identity_sha256")
        == evaluation_authority["evaluation_gate_identity_sha256"]
        and training_contract.get("evaluation_inference_summary_identity_sha256")
        == evaluation_authority[
            "evaluation_inference_summary_identity_sha256"
        ],
        "train399 evaluator does not hash-bind evaluation-only inference",
    )
    definitions = (
        (
            "stageb_detection_files",
            "stageb_detection_set_identity_sha256",
            evaluation_authority["evaluation_detection_schema_version"],
        ),
        (
            "hybrid_prediction_files",
            "hybrid_prediction_set_identity_sha256",
            LEGACY_HYBRID_COMPARATOR_SCHEMA,
        ),
    )
    normalized: dict[str, Any] = {"task_order": task_order}
    for list_field, identity_field, schema in definitions:
        records = locks.get(list_field)
        _require(
            isinstance(records, list) and len(records) == 44,
            f"train399 {list_field} is not a 44-file lock",
        )
        _require(
            all(
                isinstance(record, Mapping)
                and set(record) == {"task_id", "sha256"}
                and _is_sha256(record.get("sha256"))
                for record in records
            ),
            f"train399 {list_field} contains malformed file locks",
        )
        locked_order = [str(record["task_id"]) for record in records]
        _require(
            locked_order == task_order,
            f"train399 {list_field} order differs from evaluator per-image order",
        )
        declared_identity = locks.get(identity_field)
        _require(
            _is_sha256(declared_identity)
            and declared_identity == sha256_json(records),
            f"train399 {identity_field} does not seal the ordered file locks",
        )
        normalized[list_field] = [dict(record) for record in records]
        normalized[identity_field] = str(declared_identity)
        normalized[f"{list_field}_schema_version"] = schema

    _require(
        evaluation_authority["evaluation_detection_set_identity_sha256"]
        == normalized["stageb_detection_set_identity_sha256"],
        "train399 evaluation-only detection set differs from evaluator locks",
    )
    normalized["stageb_evaluation_inference_authority"] = deepcopy(
        dict(evaluation_authority)
    )

    comparator = evaluation.get("comparator_contract", {}).get("hybrid_max")
    _require(isinstance(comparator, Mapping), "legacy Hybrid comparator contract missing")
    _require(
        comparator.get("evidence_role") == "locked_legacy_development_comparator"
        and comparator.get("schema_version") == LEGACY_HYBRID_COMPARATOR_SCHEMA
        and comparator.get("identity_hair_variant") == LEGACY_HYBRID_IDENTITY_VARIANT
        and comparator.get("count_hair_variant") == LEGACY_HYBRID_IDENTITY_VARIANT
        and comparator.get("endpoint_complete_identity_layer") is True
        and comparator.get("phaxis_payload_allowed") is False
        and comparator.get("stageb_identity_source_allowed") is False
        and comparator.get("prediction_set_identity_sha256")
        == normalized["hybrid_prediction_set_identity_sha256"]
        and comparator.get("expected_prediction_set_identity_sha256")
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
        and normalized["hybrid_prediction_set_identity_sha256"]
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256,
        "legacy Hybrid comparator schema/variant/source semantics changed",
    )
    normalized["legacy_hybrid_comparator_contract"] = dict(comparator)
    return normalized


def _validate_core(
    *,
    paths: Mapping[str, Path],
    payloads: Mapping[str, dict[str, Any]],
    proposal_path: Path,
    proposal: Mapping[str, Any],
    selection_path: Path,
    selection: Mapping[str, Any],
    split_manifest_path: Path,
    final: bool,
) -> tuple[dict[str, str], str, str]:
    for role, payload in payloads.items():
        _guard_red_lines(role, payload)
    _guard_red_lines("model_contract_proposal", proposal)
    _guard_red_lines("train399_selection", selection)
    proposal_identity = _sealed(
        proposal, "model_contract_identity_sha256", "model_contract_proposal"
    )
    try:
        derived_public_identity = validate_proposal_public_identity(proposal)
    except ContractError as error:
        raise FigureInputAssemblyError(
            "model_contract_proposal: canonical public identity is invalid"
        ) from error
    selection_identity = _sealed(
        selection, "selection_receipt_identity_sha256", "train399_selection"
    )
    evaluation = payloads["train399_evaluation"]
    _require(
        evaluation.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
        "train399 evaluation: unsupported schema",
    )
    _require(evaluation.get("status") == "completed", "train399 evaluation incomplete")
    training = evaluation.get("training_contract")
    _require(isinstance(training, Mapping), "train399 evaluation training contract missing")
    _require(training.get("training_images") == 399, "train399 scope changed")
    _require(training.get("validation_images") == 44, "QC-development scope changed")
    _require(
        training.get("validation_labels_used_for_gradient_or_early_stopping") is False,
        "QC-development labels entered optimization",
    )
    _require(
        training.get("selection_receipt_identity_sha256") == selection_identity,
        "evaluation/selection logical identity mismatch",
    )
    hierarchy = evaluation.get("metric_hierarchy")
    _require(
        isinstance(hierarchy, Mapping)
        and hierarchy.get("primary")
        == "one-to-one tolerant biological-hair presence; bidirectional partial centreline coverage without endpoint gates"
        and hierarchy.get("primary_minimum_truth_coverage") == 0.25
        and hierarchy.get("primary_minimum_prediction_coverage") == 0.25
        and hierarchy.get("primary_minimum_direction_cosine") == 0.0,
        "train399 evaluation does not use the locked tolerant biological-presence primary metric",
    )
    inputs = evaluation.get("inputs_sha256")
    _require(isinstance(inputs, Mapping), "evaluation input hashes missing")
    _require(
        inputs.get("selection_receipt") == sha256_file(selection_path),
        "evaluation does not bind named selection receipt",
    )
    _require(
        inputs.get("split_manifest") == sha256_file(split_manifest_path),
        "evaluation does not bind named split manifest",
    )
    _require(
        selection.get("images") == 44
        and selection.get("independent_accuracy_claim_allowed") is False,
        "selection is not locked QC-development44 evidence",
    )

    expected = {
        "root_exact283": "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "stageb": "PHAxis-StageB-inference-run-1.1",
        "fusion": "PHAxis-fusion-run-1.1",
        "traits": "PHAxis-trait-export-1.0",
        "cohorts": "PHAxis-biological-cohorts-1.0",
        "analysis": "PHAxis-exploratory-biological-analysis-1.0",
        "profiles": "PHAxis-distal-axis-profile-export-1.0.0",
    }
    for role, schema in expected.items():
        _require(payloads[role].get("schema_version") == schema, f"{role}: unsupported schema")
    if final:
        _require(
            proposal.get("formal_release_status") == "passed_proposal_not_official",
            "final assembly requires a passed, unapplied model-contract proposal",
        )
        promotion = proposal.get("promotion")
        _require(
            isinstance(promotion, Mapping)
            and promotion.get("schema_version") == "PHAxis-model-contract-promotion-1.0"
            and promotion.get("status") == "validated_proposal_not_applied"
            and promotion.get("official_apply_performed") is False,
            "model-contract proposal promotion guard changed",
        )
        _require(payloads["fusion"].get("images") == 283, "fusion is not exact283")
        _require(payloads["traits"].get("tasks") == 283, "traits are not exact283")
        public_model_bundle_id = derived_public_identity["model_bundle_id"]
        public_root_provider_role = derived_public_identity["root_provider_role"]
        public_root_expert_id = derived_public_identity["root_expert_id"]
        proposal_root = proposal.get("root_expert")
        root_receipt = payloads["root_exact283"]
        _require(
            isinstance(public_model_bundle_id, str)
            and public_model_bundle_id.startswith("PHAXIS-V1.0.0-STRICT-TRAIN399-")
            and public_root_provider_role == "PHAxis-portable-root-provider"
            and isinstance(public_root_expert_id, str)
            and public_root_expert_id.startswith("PHAxis-root-provider-"),
            "proposal-derived public model/root identity is invalid",
        )
        root_field_by_role = {
            "stageb": "root_expert_id",
            "fusion": "root_expert",
            "traits": "root_expert_id",
            "cohorts": "root_expert_id",
            "analysis": "root_expert_id",
            "profiles": "root_expert_id",
        }
        for role, root_field in root_field_by_role.items():
            _require(
                payloads[role].get("model_bundle_id") == public_model_bundle_id
                and payloads[role].get(root_field) == public_root_expert_id,
                f"{role}: public model/root identity differs from proposal",
            )
        hair_identity_expert_id = payloads["stageb"].get(
            "detection_model_metadata", {}
        ).get("expert_id")
        _require(
            isinstance(hair_identity_expert_id, str)
            and bool(hair_identity_expert_id)
            and payloads["fusion"].get("hair_identity_count_expert")
            == hair_identity_expert_id
            and payloads["traits"].get("hair_identity_count_expert")
            == hair_identity_expert_id,
            "Stage-B/fusion/traits hair-identity expert differs",
        )
        _require(
            isinstance(proposal_root, Mapping)
            and proposal_root.get("bundle_identity_sha256")
            == root_receipt.get("bundle_identity_sha256")
            and proposal_root.get("pipeline_identity_sha256")
            == root_receipt.get("pipeline_identity_sha256")
            and proposal_root.get("fresh_exact283_audit_identity_sha256")
            == root_receipt.get("audit_identity_sha256"),
            "model-contract proposal does not bind the named root-provider bundle/audit receipt",
        )
        counts = payloads["cohorts"].get("counts", {})
        _require(
            counts.get("biological_full") == 283 and counts.get("biological_clean") == 261,
            "cohort scope is not full283/clean261",
        )
        _require(
            payloads["profiles"].get("tasks") == 261
            and payloads["profiles"].get("locked_1_4mm_trait_crosscheck_mismatches") == 0,
            "profiles are not cross-checked clean261",
        )
    source_hashes = {role: sha256_file(paths[role]) for role in CORE_ROLES}
    return source_hashes, sha256_file(proposal_path), proposal_identity


def _prediction_map(fusion: Mapping[str, Any], *, final: bool) -> dict[str, str]:
    records = fusion.get("records")
    _require(isinstance(records, list), "fusion prediction records missing")
    if final:
        _require(len(records) == 283, "fusion prediction map is not exact283")
    result: dict[str, str] = {}
    for record in records:
        _require(isinstance(record, Mapping), "fusion prediction record malformed")
        task_id = record.get("task_id")
        digest = record.get("prediction_sha256")
        _require(isinstance(task_id, str) and task_id not in result, "duplicate fusion task")
        _require(_is_sha256(digest), f"{task_id}: invalid prediction SHA")
        result[task_id] = str(digest)
    return result


def _build_qc_development(
    evaluation: Mapping[str, Any],
    selection: Mapping[str, Any],
    split_manifest: pd.DataFrame,
    prediction_inputs: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = evaluation.get("per_image")
    _require(isinstance(rows, list) and len(rows) == 44, "evaluation per-image rows are not QC44")
    family_lookup = {
        str(row.task_id): str(row.family_key)
        for row in split_manifest.itertuples(index=False)
        if str(row.split) == "val"
    }
    _require(len(family_lookup) == 44, "split manifest does not contain exactly 44 validation families")
    input_specs = {
        "stageb_train399": {
            "files": prediction_inputs["stageb_detection_files"],
            "set_identity": prediction_inputs[
                "stageb_detection_set_identity_sha256"
            ],
            "schema": prediction_inputs[
                "stageb_detection_files_schema_version"
            ],
            "variant": "stageb_train399",
            "evidence_role": prediction_inputs[
                "stageb_evaluation_inference_authority"
            ]["artifact_role"],
        },
        "hybrid_max": {
            "files": prediction_inputs["hybrid_prediction_files"],
            "set_identity": prediction_inputs[
                "hybrid_prediction_set_identity_sha256"
            ],
            "schema": prediction_inputs["legacy_hybrid_comparator_contract"][
                "schema_version"
            ],
            "variant": prediction_inputs["legacy_hybrid_comparator_contract"][
                "identity_hair_variant"
            ],
            "evidence_role": prediction_inputs[
                "legacy_hybrid_comparator_contract"
            ]["evidence_role"],
        },
    }
    lock_maps = {
        source_name: {
            str(record["task_id"]): str(record["sha256"])
            for record in spec["files"]
        }
        for source_name, spec in input_specs.items()
    }

    def biological_tp(record: Mapping[str, Any], tolerance_um: int) -> int:
        values = record.get("biological_presence_tp")
        _require(isinstance(values, Mapping), "biological-presence sufficient statistics missing")
        for key in (str(tolerance_um), f"{tolerance_um}.0", tolerance_um):
            if key in values:
                return int(values[key])
        raise FigureInputAssemblyError(
            f"biological-presence TP missing at {tolerance_um} um"
        )

    output_rows: list[dict[str, Any]] = []
    sufficient: dict[str, dict[int, np.ndarray]] = {
        source_name: {} for source_name in QC_COMPARATOR_MAP
    }
    for source_order, row in enumerate(rows):
        task_id = str(row.get("task_id"))
        _require(task_id in family_lookup, f"{task_id}: absent from locked val split")
        for source_name, comparator in QC_COMPARATOR_MAP.items():
            record = row.get(source_name)
            _require(isinstance(record, Mapping), f"{task_id}: missing {source_name} metrics")
            n_pred = int(record["n_pred"])
            n_gt = int(record["n_gt"])
            tp_by_tolerance = {
                tolerance_um: biological_tp(record, tolerance_um)
                for tolerance_um in (5, 10, 20)
            }
            _require(
                all(0 <= value <= min(n_pred, n_gt) for value in tp_by_tolerance.values()),
                f"{task_id}/{source_name}: impossible biological-presence TP",
            )
            spec = input_specs[source_name]
            output_rows.append(
                {
                    "source_unit": task_id,
                    "source_unit_order": source_order,
                    "family_key": family_lookup[task_id],
                    "comparator": comparator,
                    "gt_count": n_gt,
                    "predicted_count": n_pred,
                    "biological_presence_tp_5um": tp_by_tolerance[5],
                    "biological_presence_tp_10um": tp_by_tolerance[10],
                    "biological_presence_tp_20um": tp_by_tolerance[20],
                    "prediction_input_sha256": lock_maps[source_name][task_id],
                    "prediction_input_set_identity_sha256": spec["set_identity"],
                    "prediction_input_schema_version": spec["schema"],
                    "identity_hair_variant": spec["variant"],
                    "evidence_role": spec["evidence_role"],
                }
            )
    per_image = pd.DataFrame(output_rows).sort_values(
        ["source_unit_order", "comparator"], kind="stable"
    )

    for source_name in QC_COMPARATOR_MAP:
        for tolerance_um in (5, 10, 20):
            sufficient[source_name][tolerance_um] = np.asarray(
                [
                    [
                        biological_tp(row[source_name], tolerance_um),
                        int(row[source_name]["n_pred"]),
                        int(row[source_name]["n_gt"]),
                    ]
                    for row in rows
                ],
                dtype=np.int64,
            )

    bootstrap = evaluation.get("paired_bootstrap_95ci")
    _require(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("method") == "paired image-level nonparametric bootstrap",
        "evaluation uncertainty is not paired image-level bootstrap",
    )
    repetitions = int(bootstrap.get("repetitions", 0))
    seed = int(bootstrap.get("seed", 0))
    _require(
        repetitions == 10000 and seed == 20260828,
        "evaluation bootstrap must be the locked 10000-replicate seed-20260828 analysis",
    )
    generator = np.random.default_rng(seed)
    sampled_indices = generator.integers(0, len(rows), size=(repetitions, len(rows)))
    sampled_f1: dict[str, dict[int, np.ndarray]] = {
        source_name: {} for source_name in QC_COMPARATOR_MAP
    }
    for source_name in QC_COMPARATOR_MAP:
        for tolerance_um in (5, 10, 20):
            statistics = sufficient[source_name][tolerance_um]
            values = np.empty(repetitions, dtype=np.float64)
            for index, sample_index in enumerate(sampled_indices):
                sample = statistics[sample_index]
                values[index] = _prf(
                    int(sample[:, 0].sum()),
                    int(sample[:, 1].sum()),
                    int(sample[:, 2].sum()),
                )[2]
            sampled_f1[source_name][tolerance_um] = values

    paired_delta = {
        tolerance_um: sampled_f1["stageb_train399"][tolerance_um]
        - sampled_f1["hybrid_max"][tolerance_um]
        for tolerance_um in (5, 10, 20)
    }
    paired_delta_point = {
        tolerance_um: _prf(
            int(sufficient["stageb_train399"][tolerance_um][:, 0].sum()),
            int(sufficient["stageb_train399"][tolerance_um][:, 1].sum()),
            int(sufficient["stageb_train399"][tolerance_um][:, 2].sum()),
        )[2]
        - _prf(
            int(sufficient["hybrid_max"][tolerance_um][:, 0].sum()),
            int(sufficient["hybrid_max"][tolerance_um][:, 1].sum()),
            int(sufficient["hybrid_max"][tolerance_um][:, 2].sum()),
        )[2]
        for tolerance_um in (5, 10, 20)
    }
    tolerance_rows: list[dict[str, Any]] = []
    for source_name, comparator in QC_COMPARATOR_MAP.items():
        for tolerance_um in (5, 10, 20):
            statistics = sufficient[source_name][tolerance_um]
            precision, recall, f1 = _prf(
                int(statistics[:, 0].sum()),
                int(statistics[:, 1].sum()),
                int(statistics[:, 2].sum()),
            )
            ci_low, ci_high = np.quantile(
                sampled_f1[source_name][tolerance_um], (0.025, 0.975)
            )
            delta_low, delta_high = np.quantile(
                paired_delta[tolerance_um], (0.025, 0.975)
            )
            declared = evaluation["overall"][source_name]["tolerant_biological_presence"][str(tolerance_um)]
            _require(
                all(
                    math.isclose(observed, float(declared[key]), rel_tol=0.0, abs_tol=1e-12)
                    for observed, key in ((precision, "precision"), (recall, "recall"), (f1, "f1"))
                ),
                f"{source_name}@{tolerance_um}: pooled metric differs from evaluation receipt",
            )
            tolerance_rows.append(
                {
                    "comparator": comparator,
                    "tolerance_um": tolerance_um,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "paired_delta_stageb_minus_legacy_f1": paired_delta_point[
                        tolerance_um
                    ],
                    "paired_delta_ci_low": float(delta_low),
                    "paired_delta_ci_high": float(delta_high),
                    "ci_method": "image-level nonparametric bootstrap",
                    "bootstrap_repetitions": repetitions,
                    "primary_metric": "one_to_one_tolerant_biological_hair_presence",
                    "minimum_truth_coverage": 0.25,
                    "minimum_prediction_coverage": 0.25,
                    "minimum_direction_cosine": 0.0,
                    "endpoint_gate_used": False,
                }
            )

    declared_experts = bootstrap.get("experts")
    declared_delta = bootstrap.get("delta_stageb_train399_minus_hybrid")
    _require(
        isinstance(declared_experts, Mapping) and isinstance(declared_delta, Mapping),
        "evaluation paired bootstrap endpoint maps missing",
    )
    for source_name in QC_COMPARATOR_MAP:
        declared_interval = declared_experts.get(source_name, {}).get(
            "biological_presence_f1_20um"
        )
        observed_interval = np.quantile(
            sampled_f1[source_name][20], (0.025, 0.975)
        )
        _require(
            isinstance(declared_interval, Mapping)
            and math.isclose(
                float(declared_interval.get("lower_2_5")),
                float(observed_interval[0]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(declared_interval.get("upper_97_5")),
                float(observed_interval[1]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{source_name}: primary biological-presence CI differs from evaluator",
        )
    declared_delta_interval = declared_delta.get("biological_presence_f1_20um")
    observed_delta_interval = np.quantile(paired_delta[20], (0.025, 0.975))
    _require(
        isinstance(declared_delta_interval, Mapping)
        and math.isclose(
            float(declared_delta_interval.get("lower_2_5")),
            float(observed_delta_interval[0]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(declared_delta_interval.get("upper_97_5")),
            float(observed_delta_interval[1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "primary paired biological-presence delta CI differs from evaluator",
    )

    threshold_metrics = selection.get("threshold_metrics")
    _require(isinstance(threshold_metrics, list) and threshold_metrics, "selection threshold grid missing")
    selected = selection.get("selected")
    _require(isinstance(selected, Mapping), "selected operating point missing")
    selected_threshold = float(selected["threshold"])
    threshold_rows = []
    for row in threshold_metrics:
        threshold_rows.append(
            {
                "threshold": float(row["threshold"]),
                "f1_20um": float(
                    row["tolerant_biological_presence_20um"]["f1"]
                ),
                "attachment_proxy_f1_20um": float(
                    row["identity_attachment_proxy_20um"]["f1"]
                ),
                "count_mae": float(row["count_mae"]),
                "selected": math.isclose(
                    float(row["threshold"]), selected_threshold, rel_tol=0.0, abs_tol=1e-12
                ),
                "selection_metric": "tolerant_biological_presence_f1_20um",
                "straight_base_to_tip_presence_proxy_used": True,
                "distal_endpoint_or_length_used_as_selection_gate": False,
            }
        )
    _require(sum(bool(row["selected"]) for row in threshold_rows) == 1, "selection grid is ambiguous")
    return (
        per_image.reset_index(drop=True),
        pd.DataFrame(tolerance_rows),
        pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True),
        {
            "method": "paired image-level nonparametric bootstrap",
            "repetitions": repetitions,
            "seed": seed,
            "primary_delta_metric": "biological_presence_f1_20um",
            "primary_delta_stageb_train399_minus_legacy": paired_delta_point[20],
            "primary_delta_ci95": {
                "lower_2_5": float(observed_delta_interval[0]),
                "upper_97_5": float(observed_delta_interval[1]),
            },
        },
    )


def _historical_group_rows(
    per_image: pd.DataFrame,
    receipt: Mapping[str, Any],
) -> pd.DataFrame:
    required = (
        "source_unit",
        "family_key",
        "fold",
        "quality_band",
        "density_band",
        "annotation_mode",
        "n_pred",
        "n_gt",
        "biological_presence_tp_20um",
    )
    missing = [column for column in required if column not in per_image.columns]
    _require(not missing, f"historical sufficient statistics missing {missing}")
    _require(per_image["source_unit"].nunique() == len(per_image) == 443, "historical OOF table is not 443 unique images")
    _require(per_image["family_key"].astype(str).str.len().gt(0).all(), "historical family key missing")
    _require(
        per_image.groupby("family_key")["fold"].nunique().eq(1).all()
        and per_image["fold"].nunique() == 5,
        "historical OOF family/fold isolation is not proven",
    )
    numeric = per_image[["n_pred", "n_gt", "biological_presence_tp_20um"]].apply(pd.to_numeric, errors="coerce")
    _require(np.isfinite(numeric.to_numpy()).all(), "historical sufficient statistics are non-finite")
    _require((numeric >= 0).all().all(), "historical sufficient statistics are negative")
    _require((numeric["biological_presence_tp_20um"] <= numeric[["n_pred", "n_gt"]].min(axis=1)).all(), "historical true positives are impossible")

    metric_contract = receipt.get("metric_contract")
    _require(
        isinstance(metric_contract, Mapping)
        and metric_contract.get("primary_metric")
        == "one_to_one_tolerant_biological_hair_presence"
        and metric_contract.get("tolerance_um") == 20.0
        and metric_contract.get("minimum_truth_coverage") == 0.25
        and metric_contract.get("minimum_prediction_coverage") == 0.25
        and metric_contract.get("minimum_direction_cosine") == 0.0
        and metric_contract.get("endpoint_gate_used") is False,
        "historical development receipt does not lock the biological-presence metric",
    )

    uncertainty = receipt.get("uncertainty")
    _require(isinstance(uncertainty, Mapping), "historical receipt uncertainty contract missing")
    _require(
        uncertainty.get("method") == "image-level nonparametric bootstrap",
        "historical CI must use image-level bootstrap",
    )
    repetitions = int(uncertainty.get("repetitions", 0))
    seed = int(uncertainty.get("seed", 0))
    _require(repetitions >= 200, "historical bootstrap has too few repetitions")
    groups = (
        ("quality", "Q2_25_50", "quality_band", "Q2_25_50", "quality_Q2_25_50"),
        ("quality", "Q3_50_75", "quality_band", "Q3_50_75", "quality_Q3_50_75"),
        ("quality", "Q4_75_100", "quality_band", "Q4_75_100", "quality_Q4_75_100"),
        ("density", "sparse_1_49", "density_band", "sparse_1_49", "density_sparse_1_49"),
        ("density", "medium_50_99", "density_band", "medium_50_99", "density_medium_50_99"),
        ("density", "dense_100_199", "density_band", "dense_100_199", "density_dense_100_199"),
        ("density", "very_dense_ge200", "density_band", "very_dense_ge200", "density_very_dense_ge200"),
        ("annotation", "fully_manual", "annotation_mode", "fully_manual", "annotation_fully_manual"),
        ("annotation", "model_assisted_refined", "annotation_mode", "model_assisted_refined", "annotation_model_assisted_refined"),
    )
    output: list[dict[str, Any]] = []
    for dimension, stratum, field, value, source_key in groups:
        selected = per_image[per_image[field].astype(str) == value].copy()
        _require(len(selected) >= 2, f"historical stratum is too small: {source_key}")
        sufficient = selected[["biological_presence_tp_20um", "n_pred", "n_gt"]].to_numpy(dtype=np.int64)
        precision, recall, f1 = _prf(
            int(sufficient[:, 0].sum()), int(sufficient[:, 1].sum()), int(sufficient[:, 2].sum())
        )
        ci_low, ci_high = _bootstrap_prf(
            sufficient,
            repetitions=repetitions,
            seed=_group_seed(seed, dimension, stratum),
        )
        count_bias = float(
            (pd.to_numeric(selected["n_pred"]) - pd.to_numeric(selected["n_gt"])).mean()
        )
        output.append(
            {
                "dimension": dimension,
                "stratum": stratum,
                "comparator": HISTORICAL_COMPARATOR,
                "f1_20um": f1,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_images": len(selected),
                "count_bias": count_bias,
                "precision": precision,
                "recall": recall,
                "primary_metric": "one_to_one_tolerant_biological_hair_presence",
                "ci_method": "image-level nonparametric bootstrap",
                "bootstrap_repetitions": repetitions,
            }
        )
    return pd.DataFrame(output)


def _require_exact_json(observed: Any, expected: Any, message: str) -> None:
    try:
        identical = sha256_json(observed) == sha256_json(expected)
    except (TypeError, ValueError) as error:
        raise FigureInputAssemblyError(f"{message}: non-canonical JSON") from error
    _require(identical, message)


def _validate_nested_seal(
    payload: Any, *, identity_field: str, role: str
) -> dict[str, Any]:
    _require(isinstance(payload, Mapping), f"{role}: embedded receipt missing")
    normalized = deepcopy(dict(payload))
    identity = normalized.pop(identity_field, None)
    _require(
        _is_sha256(identity) and sha256_json(normalized) == identity,
        f"{role}: {identity_field} does not seal the complete embedded receipt",
    )
    return deepcopy(dict(payload))


def _validate_assurance_source_rows(
    payload: Mapping[str, Any], *, role: str
) -> list[dict[str, Any]]:
    rows = payload.get("per_image")
    _require(isinstance(rows, list) and len(rows) == 44, f"{role}: per-image denominator is not QC-development44")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(rows):
        _require(isinstance(value, Mapping), f"{role}: malformed per-image row {index}")
        row = deepcopy(dict(value))
        unsigned = deepcopy(row)
        identity = unsigned.pop("row_identity_sha256", None)
        _require(
            _is_sha256(identity) and sha256_json(unsigned) == identity,
            f"{role}: per-image row identity drift at index {index}",
        )
        _require(
            isinstance(row.get("source_unit"), str)
            and bool(row["source_unit"])
            and _is_sha256(row.get("source_image_sha256")),
            f"{role}: source-unit/source-image identity missing at index {index}",
        )
        normalized.append(row)
    source_units = [str(row["source_unit"]) for row in normalized]
    image_hashes = [str(row["source_image_sha256"]) for row in normalized]
    _require(
        len(set(source_units)) == len(set(image_hashes)) == len(normalized),
        f"{role}: source-image bootstrap units are duplicated",
    )
    source_set = [
        {
            "source_unit": row["source_unit"],
            "source_image_sha256": row["source_image_sha256"],
        }
        for row in normalized
    ]
    _require(
        _integer_number(payload.get("source_unit_total"), f"{role} source-unit total")
        == len(normalized)
        and payload.get("per_image_set_identity_sha256") == sha256_json(normalized)
        and payload.get("source_unit_set_identity_sha256") == sha256_json(source_set),
        f"{role}: per-image/source-unit set identity drift",
    )
    return normalized


def _validate_assurance_bootstrap(payload: Mapping[str, Any], *, role: str) -> None:
    bootstrap = payload.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), f"{role}: bootstrap contract missing")
    _require(
        bootstrap.get("method") == _root_continuity.BOOTSTRAP_METHOD
        and bootstrap.get("unit") == "source_image"
        and _integer_number(bootstrap.get("repetitions"), f"{role} bootstrap repetitions")
        == _root_continuity.BOOTSTRAP_REPETITIONS
        and _integer_number(bootstrap.get("seed"), f"{role} bootstrap seed")
        == _root_continuity.BOOTSTRAP_SEED
        and bootstrap.get("interval")
        == "two-sided 95% percentile (2.5%, 97.5%)"
        and bootstrap.get("sufficient_statistics_location")
        == "per_image[*].bootstrap_sufficient_statistics",
        f"{role}: source-image bootstrap contract drift",
    )


def _validate_root_continuity_assurance(
    embedded: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    role = "root_continuity_assurance"
    payload = _validate_nested_seal(
        embedded,
        identity_field="root_continuity_assurance_identity_sha256",
        role=role,
    )
    _require(
        payload.get("schema_version")
        == _root_continuity.ROOT_CONTINUITY_ASSURANCE_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("scope")
        == "QC-development primary-root continuity assurance; non-independent"
        and payload.get("evidence_role")
        == _root_continuity.ROOT_CONTINUITY_EVIDENCE_ROLE
        and payload.get("independent_accuracy_claim_allowed") is False
        and payload.get("blind_images_used") == 0
        and payload.get("provider_equivalence_used_as_accuracy") is False
        and payload.get("reference_axis_definition")
        == _root_continuity.ROOT_CONTINUITY_REFERENCE_DEFINITION
        and payload.get("prediction_axis_definition")
        == _root_continuity.ROOT_CONTINUITY_PREDICTION_DEFINITION
        and payload.get("coordinate_space")
        == _root_continuity.ROOT_CONTINUITY_COORDINATE_SPACE,
        f"{role}: schema, role, or geometry semantics drift",
    )
    expected_contract = {
        "support_tolerance_um": 5.0,
        "maximum_reference_sampling_interval_um": 2.0,
        "reference_axis_coverage": (
            "union arc-length-weighted fraction of reference-axis intervals whose "
            "midpoint is within support_tolerance_um of at least one predicted "
            "connected axis component"
        ),
        "longest_unsupported_gap_um": (
            "longest contiguous run unsupported by the union of connected components"
        ),
        "maximum_single_component_coverage": (
            "largest reference-axis coverage achieved by any one connected predicted axis component"
        ),
        "longest_unsupported_gap_um_on_best_component": (
            "longest gap on the component with maximum coverage; ties use smaller gap then stable component order"
        ),
        "spanning_component_count": (
            "number of individual connected predicted axis components supporting every reference interval"
        ),
        "break_free": (
            "at least one single connected predicted axis component supports every "
            "reference interval; union support from multiple fragments is insufficient"
        ),
        "union_coverage_hides_fragmentation": (
            "union reference support is complete but no single connected component spans the reference axis"
        ),
        "visible_axis_extent_error": (
            "absolute and signed error of the predicted proximal-to-distal span after "
            "projection onto the ordered reference axis; internal gaps are scored separately"
        ),
    }
    _require_exact_json(
        payload.get("metric_contract"),
        expected_contract,
        f"{role}: metric contract drift",
    )
    _validate_assurance_bootstrap(payload, role=role)
    rows = _validate_assurance_source_rows(payload, role=role)
    for row in rows:
        source_unit = str(row["source_unit"])
        sufficient = row.get("bootstrap_sufficient_statistics")
        expected_sufficient = {
            "reference_axis_coverage": row.get("reference_axis_coverage"),
            "longest_unsupported_gap_um": row.get("longest_unsupported_gap_um"),
            "maximum_single_component_coverage": row.get(
                "maximum_single_component_coverage"
            ),
            "longest_unsupported_gap_um_on_best_component": row.get(
                "longest_unsupported_gap_um_on_best_component"
            ),
            "break_free": row.get("break_free"),
            "visible_axis_extent_error_um_abs": row.get(
                "visible_axis_extent_error_um_abs"
            ),
        }
        _require_exact_json(
            sufficient,
            expected_sufficient,
            f"{role}/{source_unit}: bootstrap sufficient statistics drift",
        )
        coverage = _finite_number(
            row.get("maximum_single_component_coverage"),
            f"{role}/{source_unit} single-component coverage",
        )
        union_coverage = _finite_number(
            row.get("reference_axis_coverage"),
            f"{role}/{source_unit} union coverage",
        )
        gap = _finite_number(
            row.get("longest_unsupported_gap_um_on_best_component"),
            f"{role}/{source_unit} best-component gap",
        )
        extent = _finite_number(
            row.get("visible_axis_extent_error_um_abs"),
            f"{role}/{source_unit} extent error",
        )
        components = row.get("connected_component_support")
        _require(
            0.0 <= coverage <= union_coverage <= 1.0
            and gap >= 0.0
            and extent >= 0.0
            and isinstance(row.get("break_free"), bool)
            and isinstance(components, list)
            and _integer_number(
                row.get("prediction_connected_component_count"),
                f"{role}/{source_unit} component count",
            )
            == len(components),
            f"{role}/{source_unit}: invalid continuity sufficient statistics",
        )
        if components:
            for component_index, component in enumerate(components):
                _require(
                    isinstance(component, Mapping)
                    and _integer_number(
                        component.get("component_index"),
                        f"{role}/{source_unit} component index",
                    )
                    == component_index,
                    f"{role}/{source_unit}: component ordering drift",
                )
            best = min(
                range(len(components)),
                key=lambda index: (
                    -_finite_number(
                        components[index].get("reference_axis_coverage"),
                        f"{role}/{source_unit} component coverage",
                    ),
                    _finite_number(
                        components[index].get("longest_unsupported_gap_um"),
                        f"{role}/{source_unit} component gap",
                    ),
                    index,
                ),
            )
            spanning = sum(
                bool(component.get("spans_reference_axis"))
                for component in components
            )
            _require(
                row.get("best_component_index") == best
                and math.isclose(
                    coverage,
                    float(components[best]["reference_axis_coverage"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and math.isclose(
                    gap,
                    float(components[best]["longest_unsupported_gap_um"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and _integer_number(
                    row.get("spanning_component_count"),
                    f"{role}/{source_unit} spanning count",
                )
                == spanning
                and row.get("break_free") is (spanning > 0),
                f"{role}/{source_unit}: best-component/break-free semantics drift",
            )
        else:
            _require(
                row.get("best_component_index") is None
                and coverage == 0.0
                and _integer_number(
                    row.get("spanning_component_count"),
                    f"{role}/{source_unit} spanning count",
                )
                == 0
                and row.get("break_free") is False,
                f"{role}/{source_unit}: empty-component semantics drift",
            )

    try:
        bootstrap = _root_continuity._root_bootstrap(rows)
    except Exception as error:
        raise FigureInputAssemblyError(
            f"{role}: bootstrap sufficient statistics cannot be recomputed"
        ) from error
    coverage = np.asarray([row["reference_axis_coverage"] for row in rows], dtype=float)
    gaps = np.asarray([row["longest_unsupported_gap_um"] for row in rows], dtype=float)
    single = np.asarray(
        [row["maximum_single_component_coverage"] for row in rows], dtype=float
    )
    best_gaps = np.asarray(
        [row["longest_unsupported_gap_um_on_best_component"] for row in rows],
        dtype=float,
    )
    signed_extent = np.asarray(
        [row["visible_axis_extent_error_um_signed"] for row in rows], dtype=float
    )
    absolute_extent = np.asarray(
        [row["visible_axis_extent_error_um_abs"] for row in rows], dtype=float
    )
    expected_summary = {
        "images": len(rows),
        "break_free_images": int(sum(bool(row["break_free"]) for row in rows)),
        "break_free_image_rate": float(np.mean([bool(row["break_free"]) for row in rows])),
        "images_with_spanning_component": int(
            sum(int(row["spanning_component_count"]) > 0 for row in rows)
        ),
        "spanning_component_count_total": int(
            sum(int(row["spanning_component_count"]) for row in rows)
        ),
        "union_fully_supported_images": int(
            sum(bool(row["union_reference_axis_fully_supported"]) for row in rows)
        ),
        "union_coverage_hides_fragmentation_images": int(
            sum(bool(row["union_coverage_hides_fragmentation"]) for row in rows)
        ),
        "union_coverage_hides_fragmentation_rate": float(
            np.mean([bool(row["union_coverage_hides_fragmentation"]) for row in rows])
        ),
        "reference_axis_coverage_mean": float(np.mean(coverage)),
        "reference_axis_coverage_median": float(np.median(coverage)),
        "reference_axis_coverage_min": float(np.min(coverage)),
        "longest_unsupported_gap_um_median": float(np.median(gaps)),
        "longest_unsupported_gap_um_p95": float(np.quantile(gaps, 0.95)),
        "longest_unsupported_gap_um_max": float(np.max(gaps)),
        "maximum_single_component_coverage_mean": float(np.mean(single)),
        "maximum_single_component_coverage_median": float(np.median(single)),
        "maximum_single_component_coverage_min": float(np.min(single)),
        "longest_unsupported_gap_um_on_best_component_median": float(
            np.median(best_gaps)
        ),
        "longest_unsupported_gap_um_on_best_component_p95": float(
            np.quantile(best_gaps, 0.95)
        ),
        "longest_unsupported_gap_um_on_best_component_max": float(
            np.max(best_gaps)
        ),
        "visible_axis_extent_error_um_mae": float(np.mean(absolute_extent)),
        "visible_axis_extent_error_um_median_abs": float(np.median(absolute_extent)),
        "visible_axis_extent_error_um_p95_abs": float(np.quantile(absolute_extent, 0.95)),
        "visible_axis_extent_error_um_bias": float(np.mean(signed_extent)),
        "bootstrap_95ci": bootstrap,
    }
    _require_exact_json(
        payload.get("summary"),
        expected_summary,
        f"{role}: summary/bootstrap recomputation mismatch",
    )
    for field in (
        "source_unit_set_identity_sha256",
        "input_geometry_set_identity_sha256",
        "reference_authority_sha256",
        "prediction_authority_identity_sha256",
        "implementation_sha256",
    ):
        _require(_is_sha256(payload.get(field)), f"{role}: invalid {field}")
    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping), f"{role}: provenance missing")
    for field in (
        "source_unit_set_identity_sha256",
        "input_geometry_set_identity_sha256",
        "reference_authority_sha256",
        "prediction_authority_identity_sha256",
        "implementation_sha256",
    ):
        _require(
            provenance.get(field) == payload.get(field),
            f"{role}: provenance identity mismatch for {field}",
        )
    _require(
        payload.get("implementation_sha256")
        == sha256_file(Path(_root_continuity.__file__).resolve())
        and provenance.get("canonical_annotations_read_during_inference") is False
        and provenance.get("canonical_annotations_read_during_scoring") is True
        and provenance.get("val_labels_used_for_training") is False
        and provenance.get("blind_images_used") == 0,
        f"{role}: implementation or scoring provenance drift",
    )
    key_map = {
        "root_continuity_reference_axis_coverage_mean": "reference_axis_coverage_mean",
        "root_continuity_maximum_single_component_coverage_mean": "maximum_single_component_coverage_mean",
        "root_continuity_maximum_single_component_coverage_median": "maximum_single_component_coverage_median",
        "root_continuity_best_component_gap_median_um": "longest_unsupported_gap_um_on_best_component_median",
        "root_continuity_break_free_rate": "break_free_image_rate",
        "root_continuity_visible_axis_extent_mae_um": "visible_axis_extent_error_um_mae",
    }
    expected_metrics = {
        key: {
            "value": bootstrap[bootstrap_key]["point_estimate"],
            "ci_low": bootstrap[bootstrap_key]["ci_low_2_5"],
            "ci_high": bootstrap[bootstrap_key]["ci_high_97_5"],
            "n": len(rows),
            "bootstrap_key": bootstrap_key,
        }
        for key, bootstrap_key in key_map.items()
    }
    return payload, expected_metrics


def _validate_hair_attachment_assurance(
    embedded: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    role = "hair_attachment_assurance"
    payload = _validate_nested_seal(
        embedded,
        identity_field="hair_attachment_assurance_identity_sha256",
        role=role,
    )
    _require(
        payload.get("schema_version")
        == _hair_attachment.HAIR_ATTACHMENT_ASSURANCE_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("scope")
        == "QC-development root-hair attachment assurance; non-independent"
        and payload.get("evidence_role")
        == _hair_attachment.HAIR_ATTACHMENT_EVIDENCE_ROLE
        and payload.get("independent_accuracy_claim_allowed") is False
        and payload.get("blind_images_used") == 0
        and payload.get("val_labels_used_for_training") is False
        and payload.get("provider_equivalence_used_as_accuracy") is False
        and payload.get("coordinate_space")
        == _hair_attachment.HAIR_ATTACHMENT_COORDINATE_SPACE
        and payload.get("polyline_orientation")
        == _hair_attachment.HAIR_POLYLINE_ORIENTATION,
        f"{role}: schema, role, or polyline semantics drift",
    )
    expected_contract = {
        "manual_annotation_semantics": "one centreline polyline per visible biological root hair; no width truth",
        "attachment_proxy_threshold_selection": {
            "role": "development-only base-proxy tolerance sensitivity; not formal matched attachment accuracy",
            "tolerances_um": list(_hair_attachment.PROXY_TOLERANCES_UM),
            "selected_tolerance_um": _hair_attachment.SELECTED_PROXY_TOLERANCE_UM,
            "matching": "maximum-cardinality one-to-one Hungarian matching on attachment/base distance",
        },
        "formal_biological_identity_matcher": dict(
            _hair_attachment.FORMAL_MATCHER_CONFIG
        ),
        "formal_attachment_tolerance_um": _hair_attachment.FORMAL_ATTACHMENT_TOLERANCE_UM,
        "formal_position_error_denominator": (
            "all one-to-one identities returned by the formal tolerant biological-presence matcher; no base-only rematching"
        ),
        "threshold_selection_used_as_formal_accuracy": False,
    }
    _require_exact_json(
        payload.get("metric_contract"),
        expected_contract,
        f"{role}: metric contract drift or base proxy promoted to formal accuracy",
    )
    _validate_assurance_bootstrap(payload, role=role)
    bootstrap_contract = payload["bootstrap"]
    _require(
        bootstrap_contract.get("cluster_rule")
        == (
            "resample source images with replacement and carry all formal "
            "attachment matches/errors from each sampled image; individual hairs "
            "are never bootstrap units"
        ),
        f"{role}: individual hairs cannot replace source images as bootstrap units",
    )
    rows = _validate_assurance_source_rows(payload, role=role)
    proxy_summary: dict[str, Any] = {}
    formal_errors: list[float] = []
    formal_presence_totals = [0, 0, 0]
    qualified_totals = [0, 0, 0]
    for tolerance in _hair_attachment.PROXY_TOLERANCES_UM:
        key = str(int(tolerance))
        pooled = [0, 0, 0]
        proxy_errors: list[float] = []
        for row in rows:
            source_unit = str(row["source_unit"])
            proxy = row.get("attachment_proxy_threshold_selection")
            _require(isinstance(proxy, Mapping) and key in proxy, f"{role}/{source_unit}: proxy diagnostic missing")
            cell = proxy[key]
            _require(isinstance(cell, Mapping), f"{role}/{source_unit}: proxy diagnostic malformed")
            matches = cell.get("matched_pairs")
            _require(isinstance(matches, list), f"{role}/{source_unit}: proxy matches malformed")
            expected_prf = precision_recall_f1(
                len(matches), int(row["predicted_hairs"]), int(row["annotated_hairs"])
            )
            _require_exact_json(
                {field: cell.get(field) for field in expected_prf},
                expected_prf,
                f"{role}/{source_unit}: proxy diagnostic denominator drift",
            )
            errors = [
                _finite_number(match.get("attachment_error_um"), f"{role}/{source_unit} proxy error")
                for match in matches
            ]
            _require(
                all(0.0 <= error <= tolerance for error in errors)
                and math.isclose(float(cell.get("tolerance_um")), tolerance),
                f"{role}/{source_unit}: proxy tolerance/error drift",
            )
            _require_exact_json(
                cell.get("position_error"),
                _hair_attachment._error_summary(errors),
                f"{role}/{source_unit}: proxy error summary drift",
            )
            pooled[0] += len(matches)
            pooled[1] += int(row["predicted_hairs"])
            pooled[2] += int(row["annotated_hairs"])
            proxy_errors.extend(errors)
        proxy_summary[key] = {
            **precision_recall_f1(*pooled),
            "tolerance_um": tolerance,
            "position_error": _hair_attachment._error_summary(proxy_errors),
        }

    for row in rows:
        source_unit = str(row["source_unit"])
        predicted = _integer_number(row.get("predicted_hairs"), f"{role}/{source_unit} predicted hairs")
        annotated = _integer_number(row.get("annotated_hairs"), f"{role}/{source_unit} annotated hairs")
        formal = row.get("formal_matched_attachment_accuracy")
        _require(predicted >= 0 and annotated >= 0 and isinstance(formal, Mapping), f"{role}/{source_unit}: invalid formal denominator")
        presence = formal.get("formal_biological_presence")
        qualified = formal.get("attachment_qualified_identity")
        matches = formal.get("formal_identity_matches")
        _require(
            isinstance(presence, Mapping)
            and isinstance(qualified, Mapping)
            and isinstance(matches, list)
            and math.isclose(
                float(formal.get("attachment_tolerance_um")),
                _hair_attachment.FORMAL_ATTACHMENT_TOLERANCE_UM,
            ),
            f"{role}/{source_unit}: formal matched-identity structure drift",
        )
        errors = [
            _finite_number(match.get("attachment_error_um"), f"{role}/{source_unit} formal attachment error")
            for match in matches
        ]
        within = [
            error <= _hair_attachment.FORMAL_ATTACHMENT_TOLERANCE_UM
            for error in errors
        ]
        for match, expected_within in zip(matches, within, strict=True):
            _require(
                match.get("attachment_within_formal_tolerance") is expected_within,
                f"{role}/{source_unit}: formal attachment tolerance flag drift",
            )
        expected_presence = precision_recall_f1(len(matches), predicted, annotated)
        expected_qualified = precision_recall_f1(sum(within), predicted, annotated)
        _require_exact_json(
            {field: presence.get(field) for field in expected_presence},
            expected_presence,
            f"{role}/{source_unit}: formal biological-identity denominator drift",
        )
        _require(
            math.isclose(
                float(presence.get("curve_tolerance")),
                float(_hair_attachment.FORMAL_MATCHER_CONFIG["curve_tolerance_um"]),
            )
            and math.isclose(
                float(presence.get("minimum_truth_coverage")),
                float(_hair_attachment.FORMAL_MATCHER_CONFIG["minimum_truth_coverage"]),
            )
            and math.isclose(
                float(presence.get("minimum_prediction_coverage")),
                float(
                    _hair_attachment.FORMAL_MATCHER_CONFIG[
                        "minimum_prediction_coverage"
                    ]
                ),
            )
            and math.isclose(
                float(presence.get("minimum_direction_cosine")),
                float(_hair_attachment.FORMAL_MATCHER_CONFIG["minimum_direction_cosine"]),
            ),
            f"{role}/{source_unit}: formal biological-identity matcher drift",
        )
        _require_exact_json(
            qualified,
            expected_qualified,
            f"{role}/{source_unit}: attachment-qualified denominator drift",
        )
        _require_exact_json(
            formal.get("attachment_position_error_on_all_formal_identity_matches"),
            _hair_attachment._error_summary(errors),
            f"{role}/{source_unit}: formal position-error denominator drift",
        )
        expected_sufficient = {
            "predicted_hairs": predicted,
            "annotated_hairs": annotated,
            "formal_attachment_qualified_true_positive": sum(within),
            "formal_attachment_errors_um": errors,
        }
        _require_exact_json(
            row.get("bootstrap_sufficient_statistics"),
            expected_sufficient,
            f"{role}/{source_unit}: bootstrap sufficient statistics drift",
        )
        formal_presence_totals[0] += len(matches)
        formal_presence_totals[1] += predicted
        formal_presence_totals[2] += annotated
        qualified_totals[0] += sum(within)
        qualified_totals[1] += predicted
        qualified_totals[2] += annotated
        formal_errors.extend(errors)

    try:
        bootstrap = _hair_attachment._hair_bootstrap(rows)
    except Exception as error:
        raise FigureInputAssemblyError(
            f"{role}: bootstrap sufficient statistics cannot be recomputed"
        ) from error
    expected_summary = {
        "images": len(rows),
        "predicted_hairs": qualified_totals[1],
        "annotated_hairs": qualified_totals[2],
        "attachment_proxy_threshold_selection": proxy_summary,
        "formal_matched_attachment_accuracy": {
            "formal_biological_presence": precision_recall_f1(*formal_presence_totals),
            "attachment_tolerance_um": _hair_attachment.FORMAL_ATTACHMENT_TOLERANCE_UM,
            "attachment_qualified_identity": precision_recall_f1(*qualified_totals),
            "attachment_position_error_on_all_formal_identity_matches": _hair_attachment._error_summary(formal_errors),
            "bootstrap_95ci": bootstrap,
        },
    }
    _require_exact_json(
        payload.get("summary"),
        expected_summary,
        f"{role}: formal summary/bootstrap recomputation mismatch",
    )
    for field in (
        "source_unit_set_identity_sha256",
        "input_geometry_set_identity_sha256",
        "annotation_authority_sha256",
        "prediction_authority_identity_sha256",
        "implementation_sha256",
    ):
        _require(_is_sha256(payload.get(field)), f"{role}: invalid {field}")
    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping), f"{role}: provenance missing")
    for field in (
        "source_unit_set_identity_sha256",
        "input_geometry_set_identity_sha256",
        "annotation_authority_sha256",
        "prediction_authority_identity_sha256",
        "implementation_sha256",
    ):
        _require(
            provenance.get(field) == payload.get(field),
            f"{role}: provenance identity mismatch for {field}",
        )
    _require(
        payload.get("implementation_sha256")
        == sha256_file(Path(_hair_attachment.__file__).resolve())
        and provenance.get("canonical_annotations_read_during_inference") is False
        and provenance.get("canonical_annotations_read_during_scoring") is True
        and provenance.get("val_labels_used_for_training") is False
        and provenance.get("blind_images_used") == 0,
        f"{role}: implementation or scoring provenance drift",
    )
    formal_summary = expected_summary["formal_matched_attachment_accuracy"]
    qualified = formal_summary["attachment_qualified_identity"]
    errors = formal_summary["attachment_position_error_on_all_formal_identity_matches"]
    key_map = {
        "hair_attachment_qualified_precision_20um": (
            "formal_attachment_precision",
            qualified["precision"],
        ),
        "hair_attachment_qualified_recall_20um": (
            "formal_attachment_recall",
            qualified["recall"],
        ),
        "hair_attachment_qualified_f1_20um": (
            "formal_attachment_f1",
            qualified["f1"],
        ),
        "hair_attachment_error_median_um": (
            "formal_attachment_error_median_um",
            errors["median_um"],
        ),
        "hair_attachment_error_p95_um": (
            "formal_attachment_error_p95_um",
            errors["p95_um"],
        ),
    }
    expected_metrics = {
        key: {
            "value": value,
            "ci_low": bootstrap[bootstrap_key]["ci_low_2_5"],
            "ci_high": bootstrap[bootstrap_key]["ci_high_97_5"],
            "n": len(rows),
            "formal_match_n": int(errors["n"]),
            "predicted_hairs": int(qualified["n_pred"]),
            "annotated_hairs": int(qualified["n_gt"]),
            "bootstrap_key": bootstrap_key,
        }
        for key, (bootstrap_key, value) in key_map.items()
    }
    return payload, expected_metrics


def _validate_component_assurance_audit_metadata(
    measurement_receipt: Mapping[str, Any],
    *,
    root_continuity: Mapping[str, Any],
    hair_attachment: Mapping[str, Any],
    receipt_path: Path | None,
) -> None:
    components = measurement_receipt.get("component_receipts")
    _require(
        isinstance(components, Mapping)
        and set(components) == {"root_continuity", "hair_attachment"},
        "measurement assurance component-receipt audit metadata is incomplete",
    )
    contracts = (
        (
            "root_continuity",
            root_continuity,
            "root_continuity_assurance_identity_sha256",
            _root_continuity.build_from_input_contract,
        ),
        (
            "hair_attachment",
            hair_attachment,
            "hair_attachment_assurance_identity_sha256",
            _hair_attachment.build_from_input_contract,
        ),
    )
    authority_map = measurement_receipt.get("source_authority_identity_sha256")
    _require(
        isinstance(authority_map, Mapping),
        "measurement assurance source-authority identity map missing",
    )
    for component, embedded, identity_field, rebuild in contracts:
        record = components.get(component)
        _require(isinstance(record, Mapping), f"{component}: component audit metadata missing")
        audit_copy = record.get("audit_copy")
        input_copy = record.get("input_contract_audit_copy")
        _require(
            isinstance(audit_copy, str)
            and Path(audit_copy).name == audit_copy
            and isinstance(input_copy, str)
            and Path(input_copy).name == input_copy
            and record.get("identity_field") == identity_field
            and record.get("identity_sha256") == embedded.get(identity_field)
            and record.get("input_contract_identity_sha256")
            == embedded.get("input_contract_identity_sha256")
            and _is_sha256(record.get("audit_copy_sha256"))
            and _is_sha256(record.get("input_contract_audit_copy_sha256")),
            f"{component}: component audit identity/SHA metadata drift",
        )
        authority_key = f"{component}_assurance"
        _require(
            authority_map.get(authority_key) == embedded.get(identity_field),
            f"{component}: embedded receipt and authority-map identity differ",
        )
        if receipt_path is None:
            continue
        base = Path(receipt_path).resolve().parent
        audit_path = (base / audit_copy).resolve()
        input_path = (base / input_copy).resolve()
        _require(
            audit_path.parent == base
            and input_path.parent == base
            and audit_path.is_file()
            and input_path.is_file()
            and not audit_path.is_symlink()
            and not input_path.is_symlink()
            and sha256_file(audit_path) == record.get("audit_copy_sha256")
            and sha256_file(input_path)
            == record.get("input_contract_audit_copy_sha256"),
            f"{component}: audit copy is missing, escaped, symlinked, or hash-drifted",
        )
        _require_exact_json(
            _read_object(audit_path, f"{component} audit copy"),
            embedded,
            f"{component}: embedded receipt differs from its audit copy",
        )
        input_contract = _read_object(
            input_path, f"{component} input-contract audit copy"
        )
        unsigned_input = deepcopy(input_contract)
        input_identity = unsigned_input.pop("input_contract_identity_sha256", None)
        _require(
            input_identity == record.get("input_contract_identity_sha256")
            and _is_sha256(input_identity)
            and sha256_json(unsigned_input) == input_identity,
            f"{component}: input-contract audit-copy identity drift",
        )
        try:
            recomputed = rebuild(input_contract)
        except Exception as error:
            raise FigureInputAssemblyError(
                f"{component}: input-contract geometry recomputation failed"
            ) from error
        _require_exact_json(
            recomputed,
            embedded,
            f"{component}: embedded receipt differs from full geometry recomputation",
        )


def _validate_component_assurance_metric_rows(
    metrics: pd.DataFrame,
    *,
    expected_root: Mapping[str, Mapping[str, Any]],
    expected_hair: Mapping[str, Mapping[str, Any]],
    root_receipt: Mapping[str, Any],
    hair_receipt: Mapping[str, Any],
    measurement_receipt: Mapping[str, Any],
) -> None:
    root_specs = {
        "root_continuity_reference_axis_coverage_mean": (
            "Mean union reference-axis coverage",
            "fraction",
            "union support diagnostic across every sealed final-mask skeleton component; not a single-component continuity claim",
        ),
        "root_continuity_maximum_single_component_coverage_mean": (
            "Mean maximum single-component root coverage",
            "fraction",
            "mean per-image coverage from the best one connected final-mask skeleton component",
        ),
        "root_continuity_maximum_single_component_coverage_median": (
            "Median maximum single-component root coverage",
            "fraction",
            "median per-image coverage from the best one connected final-mask skeleton component",
        ),
        "root_continuity_best_component_gap_median_um": (
            "Median longest gap on the best root component",
            "um",
            "median longest unsupported reference-axis gap on the maximum-coverage single connected component",
        ),
        "root_continuity_break_free_rate": (
            "Break-free root image rate",
            "fraction",
            "fraction of source images with at least one single connected component spanning every reference interval",
        ),
        "root_continuity_visible_axis_extent_mae_um": (
            "Visible root-axis extent MAE",
            "um",
            "mean absolute proximal-to-distal projected extent error; internal gaps are scored separately",
        ),
    }
    hair_specs = {
        "hair_attachment_qualified_precision_20um": (
            "Attachment-qualified precision @20 µm",
            "fraction",
            "pooled precision whose true positives are formal biological-presence identities with base error <=20 µm",
        ),
        "hair_attachment_qualified_recall_20um": (
            "Attachment-qualified recall @20 µm",
            "fraction",
            "pooled recall whose true positives are formal biological-presence identities with base error <=20 µm",
        ),
        "hair_attachment_qualified_f1_20um": (
            "Attachment-qualified F1 @20 µm",
            "fraction",
            "pooled F1 from the explicit predicted/annotated denominators and attachment-qualified formal identities",
        ),
        "hair_attachment_error_median_um": (
            "Median base error on formal hair identities",
            "um",
            "median attachment/base error over all formal biological-presence matches; no base-only rematching",
        ),
        "hair_attachment_error_p95_um": (
            "P95 base error on formal hair identities",
            "um",
            "95th-percentile attachment/base error over all formal biological-presence matches; no base-only rematching",
        ),
    }
    expected_keys = set(root_specs) | set(hair_specs)
    selected = metrics[metrics["metric_key"].astype(str).isin(expected_keys)].copy()
    _require(
        len(selected) == len(expected_keys)
        and set(selected["metric_key"].astype(str)) == expected_keys,
        "root-continuity/hair-attachment metric rows are missing or duplicated",
    )
    root_n = int(root_receipt["source_unit_total"])
    hair_n = int(hair_receipt["source_unit_total"])
    hair_summary = hair_receipt["summary"]["formal_matched_attachment_accuracy"]
    qualified = hair_summary["attachment_qualified_identity"]
    formal_matches = hair_summary[
        "attachment_position_error_on_all_formal_identity_matches"
    ]["n"]
    expected_instances = {
        **{key: root_n for key in root_specs},
        "hair_attachment_qualified_precision_20um": int(qualified["n_pred"]),
        "hair_attachment_qualified_recall_20um": int(qualified["n_gt"]),
        "hair_attachment_qualified_f1_20um": int(
            qualified["n_pred"] + qualified["n_gt"]
        ),
        "hair_attachment_error_median_um": int(formal_matches),
        "hair_attachment_error_p95_um": int(formal_matches),
    }
    expected_by_key = {**expected_root, **expected_hair}
    for record in selected.to_dict("records"):
        key = str(record["metric_key"])
        expected = expected_by_key[key]
        label, unit, definition = (root_specs | hair_specs)[key]
        expected_domain = (
            "root_continuity" if key in root_specs else "hair_attachment"
        )
        expected_n = root_n if key in root_specs else hair_n
        _require(
            str(record.get("domain")) == expected_domain
            and str(record.get("label")) == label
            and str(record.get("unit")) == unit
            and str(record.get("definition")) == definition
            and str(record.get("evidence_role"))
            == "annotated_qc_development_non_independent"
            and str(record.get("ci_method"))
            == "image/source-unit nonparametric bootstrap"
            and _integer_number(record.get("bootstrap_repetitions"), f"{key} bootstrap repetitions")
            == 10_000
            and _integer_number(record.get("bootstrap_seed"), f"{key} bootstrap seed")
            == 20_260_828
            and _integer_number(record.get("n"), f"{key} source-image denominator")
            == expected_n
            and _integer_number(record.get("instances"), f"{key} instance denominator")
            == expected_instances[key],
            f"{key}: metric row semantics/denominators drift",
        )
        for column in ("value", "ci_low", "ci_high"):
            observed = _finite_number(record.get(column), f"{key}.{column}")
            target = expected[column]
            _require(
                target is not None
                and math.isclose(
                    observed, float(target), rel_tol=0.0, abs_tol=1e-12
                ),
                f"{key}: metric {column} differs from embedded source-image sufficient statistics",
            )
    counts = measurement_receipt.get("counts")
    _require(isinstance(counts, Mapping), "component assurance receipt counts missing")
    root_summary = root_receipt["summary"]
    formal_presence = hair_summary["formal_biological_presence"]
    _require(
        _integer_number(counts.get("root_continuity_source_units"), "root continuity count")
        == root_n
        and _integer_number(counts.get("root_continuity_break_free_images"), "root break-free count")
        == int(root_summary["break_free_images"])
        and _integer_number(
            counts.get("root_continuity_union_coverage_hides_fragmentation_images"),
            "root union-fragmentation diagnostic count",
        )
        == int(root_summary["union_coverage_hides_fragmentation_images"])
        and _integer_number(counts.get("hair_attachment_source_units"), "hair attachment source count")
        == hair_n
        and _integer_number(counts.get("hair_attachment_predicted_hairs"), "hair predicted denominator")
        == int(qualified["n_pred"])
        and _integer_number(counts.get("hair_attachment_annotated_hairs"), "hair annotated denominator")
        == int(qualified["n_gt"])
        and _integer_number(counts.get("hair_attachment_formal_identity_matches"), "hair formal match denominator")
        == int(formal_presence["tp"])
        and _integer_number(
            counts.get("hair_attachment_qualified_true_positives_20um"),
            "hair attachment-qualified TP count",
        )
        == int(qualified["tp"]),
        "component assurance receipt counts do not close embedded denominators",
    )
    crosscheck = measurement_receipt.get(
        "qcdev_stageb_biological_presence_20um_crosscheck_locks"
    )
    authority_map = measurement_receipt.get("source_authority_identity_sha256")
    hair_rows = hair_receipt["per_image"]
    _require(
        isinstance(crosscheck, list)
        and len(crosscheck) == len(hair_rows) == 44
        and isinstance(authority_map, Mapping)
        and authority_map.get(
            "qcdev_stageb_biological_presence_20um_crosscheck"
        )
        == sha256_json(crosscheck)
        and _integer_number(
            counts.get("hair_attachment_evaluator_crosschecked_source_units"),
            "hair evaluator crosscheck source-unit count",
        )
        == 44,
        "hair production/evaluator biological-presence crosscheck identity drift",
    )
    for lock, hair_row in zip(crosscheck, hair_rows, strict=True):
        formal = hair_row["formal_matched_attachment_accuracy"][
            "formal_biological_presence"
        ]
        _require(
            isinstance(lock, Mapping)
            and lock.get("task_id") == hair_row.get("source_unit")
            and _integer_number(lock.get("n_pred"), "hair crosscheck predicted count")
            == int(formal["n_pred"])
            and _integer_number(lock.get("n_gt"), "hair crosscheck annotated count")
            == int(formal["n_gt"])
            and _integer_number(
                lock.get("biological_presence_tp_20um"),
                "hair crosscheck biological-presence TP",
            )
            == int(formal["tp"])
            and lock.get("hair_attachment_row_identity_sha256")
            == hair_row.get("row_identity_sha256"),
            f"{hair_row.get('source_unit')}: hair production/evaluator identity crosscheck drift",
        )


def _normalize_assurance(
    receipt: Mapping[str, Any],
    metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    support: pd.DataFrame,
    topology: pd.DataFrame,
    *,
    receipt_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require(receipt.get("schema_version") == ASSURANCE_RECEIPT_SCHEMA, "measurement assurance schema changed")
    _require(
        receipt.get("status") == "completed_locked_qc_development_assurance"
        and receipt.get("scope") == "QC-development measurement assurance; non-independent"
        and receipt.get("independent_accuracy_claim_allowed") is False,
        "measurement assurance is not explicit non-independent QC-development evidence",
    )
    _guard_red_lines("measurement_assurance", receipt)
    root_continuity, expected_root_metrics = _validate_root_continuity_assurance(
        receipt.get("root_continuity_assurance")
    )
    hair_attachment, expected_hair_metrics = _validate_hair_attachment_assurance(
        receipt.get("hair_attachment_assurance")
    )
    _require(
        root_continuity["source_unit_set_identity_sha256"]
        == hair_attachment["source_unit_set_identity_sha256"],
        "root-continuity and hair-attachment source-image denominators differ",
    )
    _validate_component_assurance_audit_metadata(
        receipt,
        root_continuity=root_continuity,
        hair_attachment=hair_attachment,
        receipt_path=receipt_path,
    )
    role_map = receipt.get("metric_evidence_role_by_key")
    _require(isinstance(role_map, Mapping), "measurement assurance evidence-role map missing")
    metric_keys = set(metrics["metric_key"].astype(str))
    _require(metric_keys == set(role_map), "measurement assurance role map does not cover metrics exactly")
    normalized = metrics.copy()
    normalized["evidence_role"] = normalized["metric_key"].map(role_map)
    allowed = {
        "annotated_qc_development_non_independent",
        "application_observability_non_accuracy",
        "exact_portable_provider_equivalence",
    }
    _require(set(normalized["evidence_role"]).issubset(allowed), "measurement assurance evidence role is invalid")
    required_metrics = {
        "root_dice",
        "root_boundary_f1",
        "root_hd95_um",
        "distal_median_error_um",
        "distal_pck",
        "scale_detection_coverage",
        "scale_geometry_endpoint_error_um",
        "scale_relative_error_percent",
        "conditional_length_mae_um",
        "conditional_length_ccc",
        "matched_endpoint_error_um",
        "matched_trajectory_continuity",
        "endpoint_complete_support_fraction",
        "root_trait_agreement",
        "axis_containment_median",
        "axis_containment_min",
        "unsupported_attachment_n",
        "provider_exact_fraction",
        ROOT_CONTINUITY_DIAGNOSTIC_METRIC_KEY,
        *ROOT_CONTINUITY_FORMAL_METRIC_KEYS,
        *HAIR_ATTACHMENT_FORMAL_METRIC_KEYS,
    }
    _require(required_metrics.issubset(metric_keys), "measurement assurance metric set incomplete")
    _require(
        not any(
            "proxy" in key.casefold() or "base_only" in key.casefold()
            for key in metric_keys
        ),
        "development-only attachment base proxy entered the formal assurance table",
    )
    _require(
        set(pairs["pair_type"]) == {"scale", "conditional_length", "root_trait"},
        "assurance agreement pairs incomplete",
    )
    _require(set(support["condition_code"]) == set(GROUP_ORDER), "assurance group support incomplete")
    _require(
        len(topology) == topology["source_unit"].nunique() == 261,
        "application topology assurance is not exact clean261",
    )
    values = normalized.set_index("metric_key")["value"]

    _validate_component_assurance_metric_rows(
        normalized,
        expected_root=expected_root_metrics,
        expected_hair=expected_hair_metrics,
        root_receipt=root_continuity,
        hair_receipt=hair_attachment,
        measurement_receipt=receipt,
    )
    normalized["publication_metric_role"] = "other_assurance"
    normalized.loc[
        normalized["metric_key"] == ROOT_CONTINUITY_DIAGNOSTIC_METRIC_KEY,
        "publication_metric_role",
    ] = "diagnostic_only_union_coverage"
    normalized.loc[
        normalized["metric_key"].isin(
            [
                *ROOT_CONTINUITY_FORMAL_METRIC_KEYS,
                *HAIR_ATTACHMENT_FORMAL_METRIC_KEYS,
            ]
        ),
        "publication_metric_role",
    ] = "formal_measurement_assurance"

    def check_cell(key: str, observed: float, tolerance: float = 1e-10) -> None:
        _require(
            key in values
            and math.isfinite(float(values.loc[key]))
            and math.isclose(
                float(values.loc[key]), float(observed), rel_tol=tolerance, abs_tol=tolerance
            ),
            f"measurement assurance metric {key} differs from sufficient statistics",
        )

    scale_applicability = receipt.get("scale_applicability")
    scale_counts = receipt.get("counts")
    _require(
        isinstance(scale_applicability, Mapping)
        and isinstance(scale_counts, Mapping),
        "scale applicability/count receipt is missing",
    )
    visible_scale_n = _integer_number(
        scale_applicability.get("visible_annotated_scale_bar_cases"),
        "visible annotated scale-bar count",
    )
    trusted_metadata_n = _integer_number(
        scale_applicability.get(
            "trusted_metadata_without_visible_bar_cases"
        ),
        "trusted-metadata scale count",
    )
    absence_test_n = _integer_number(
        scale_applicability.get("absent_or_untrusted_scale_truth_cases"),
        "absent/untrusted scale-test count",
    )
    qcdevelopment_n = _integer_number(
        scale_applicability.get("qcdevelopment_images"),
        "scale applicability QC-development count",
    )
    _require(
        qcdevelopment_n == 44
        and visible_scale_n == 37
        and trusted_metadata_n == 7
        and absence_test_n == 0
        and visible_scale_n + trusted_metadata_n + absence_test_n
        == qcdevelopment_n
        and scale_applicability.get("absence_specificity_status")
        == SCALE_ABSENCE_SPECIFICITY_STATUS
        and scale_applicability.get("fail_closed_evidence_basis")
        == SCALE_FAIL_CLOSED_EVIDENCE_BASIS
        and scale_applicability.get("empirical_absence_specificity_claimed")
        is False,
        "scale applicability must close as 37 visible + 7 trusted metadata + 0 absence-test cases",
    )
    _require(
        _integer_number(
            scale_counts.get("qcdevelopment_images"),
            "receipt QC-development count",
        )
        == qcdevelopment_n
        and _integer_number(
            scale_counts.get("visible_scale_bars"),
            "receipt visible scale bars",
        )
        == visible_scale_n
        and _integer_number(
            scale_counts.get("trusted_metadata_without_visible_bar_cases"),
            "receipt trusted-metadata scale cases",
        )
        == trusted_metadata_n
        and _integer_number(
            scale_counts.get("absent_or_untrusted_scale_truth_cases"),
            "receipt absent/untrusted scale cases",
        )
        == absence_test_n,
        "scale applicability/count receipt denominators disagree",
    )

    scale_rows = normalized[
        normalized["metric_key"].astype(str).isin(
            {
                "scale_detection_coverage",
                "scale_geometry_endpoint_error_um",
                "scale_relative_error_percent",
            }
        )
    ].copy()
    _require(
        len(scale_rows) == 3
        and set(scale_rows["metric_key"].astype(str))
        == {
            "scale_detection_coverage",
            "scale_geometry_endpoint_error_um",
            "scale_relative_error_percent",
        }
        and set(scale_rows["evidence_role"].astype(str))
        == {"annotated_qc_development_non_independent"},
        "three empirical scale-assurance metrics are not unique annotated development evidence",
    )
    for row in scale_rows.to_dict("records"):
        _require(
            str(row.get("ci_method"))
            == "image/source-unit nonparametric bootstrap"
            and _integer_number(
                row.get("bootstrap_repetitions"),
                f"{row['metric_key']} bootstrap repetitions",
            )
            == 10000
            and _integer_number(
                row.get("bootstrap_seed"),
                f"{row['metric_key']} bootstrap seed",
            )
            == 20260828
            and math.isfinite(float(row["ci_low"]))
            and math.isfinite(float(row["ci_high"]))
            and float(row["ci_low"]) <= float(row["ci_high"]),
            f"{row['metric_key']}: scale metric lacks source-image bootstrap CI",
        )

    coverage_row = scale_rows[
        scale_rows["metric_key"] == "scale_detection_coverage"
    ].iloc[0]
    detected_scale_n = _integer_number(
        coverage_row["instances"], "scale detected count"
    )
    _require(
        _integer_number(coverage_row["n"], "scale coverage denominator")
        == visible_scale_n
        and detected_scale_n
        == _integer_number(
            scale_counts.get("detected_scale_bars"),
            "receipt detected scales",
        )
        and 0 <= detected_scale_n <= visible_scale_n
        and math.isclose(
            float(coverage_row["value"]),
            detected_scale_n / visible_scale_n,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "scale coverage denominator/count/receipt crosscheck failed",
    )
    scale_pairs = pairs[pairs["pair_type"].astype(str) == "scale"].copy()
    required_scale_pair_columns = {
        "pair_id",
        "relative_error_percent",
        "scale_line_endpoint_error_um",
        "source_image_sha256",
    }
    _require(
        required_scale_pair_columns.issubset(scale_pairs.columns)
        and len(scale_pairs)
        == scale_pairs["source_unit"].nunique()
        == scale_pairs["pair_id"].nunique()
        == scale_pairs["source_image_sha256"].nunique()
        == detected_scale_n,
        "scale pair rows do not match the detected visible-bar denominator",
    )
    observed_scale = pd.to_numeric(scale_pairs["observed"], errors="coerce").to_numpy(float)
    predicted_scale = pd.to_numeric(scale_pairs["predicted"], errors="coerce").to_numpy(float)
    stored_relative_error = pd.to_numeric(
        scale_pairs["relative_error_percent"], errors="coerce"
    ).to_numpy(float)
    localization_error = pd.to_numeric(
        scale_pairs["scale_line_endpoint_error_um"], errors="coerce"
    ).to_numpy(float)
    recomputed_relative_error = (
        np.abs(predicted_scale - observed_scale) / observed_scale * 100.0
    )
    _require(
        detected_scale_n >= 2
        and np.isfinite(observed_scale).all()
        and np.isfinite(predicted_scale).all()
        and np.isfinite(stored_relative_error).all()
        and np.isfinite(localization_error).all()
        and bool((observed_scale > 0).all())
        and set(scale_pairs["unit"].astype(str)) == {"um_per_px"}
        and np.allclose(
            stored_relative_error,
            recomputed_relative_error,
            rtol=0.0,
            atol=1e-12,
        ),
        "scale pair sufficient statistics are invalid",
    )
    calibration_row = scale_rows[
        scale_rows["metric_key"] == "scale_relative_error_percent"
    ].iloc[0]
    localization_row = scale_rows[
        scale_rows["metric_key"] == "scale_geometry_endpoint_error_um"
    ].iloc[0]
    _require(
        _integer_number(calibration_row["n"], "scale calibration denominator")
        == _integer_number(
            calibration_row["instances"], "scale calibration instances"
        )
        == _integer_number(
            scale_counts.get("scale_calibration_pairs"),
            "receipt scale calibration pairs",
        )
        == detected_scale_n
        and _integer_number(
            localization_row["n"], "scale localization denominator"
        )
        == _integer_number(
            localization_row["instances"], "scale localization instances"
        )
        == _integer_number(
            scale_counts.get("scale_localization_pairs"),
            "receipt scale localization pairs",
        )
        == detected_scale_n,
        "scale localization/calibration metric denominators disagree with pairs/receipt",
    )
    check_cell(
        "scale_relative_error_percent",
        float(np.median(recomputed_relative_error)),
    )
    check_cell(
        "scale_geometry_endpoint_error_um",
        float(np.median(localization_error)),
    )
    normalized["scale_visible_truth_n"] = visible_scale_n
    normalized["scale_trusted_metadata_n"] = trusted_metadata_n
    normalized["scale_absence_test_n"] = absence_test_n
    normalized["scale_absence_specificity_status"] = (
        SCALE_ABSENCE_SPECIFICITY_STATUS
    )
    normalized["scale_fail_closed_evidence_basis"] = (
        SCALE_FAIL_CLOSED_EVIDENCE_BASIS
    )

    length_pairs = pairs[pairs["pair_type"].astype(str) == "conditional_length"]
    _require(
        len(length_pairs) >= 2
        and {
            "endpoint_error_um",
            "trajectory_continuity",
        }.issubset(length_pairs.columns),
        "conditional-length pair sufficient statistics are incomplete",
    )
    observed_length = pd.to_numeric(length_pairs["observed"]).to_numpy(float)
    predicted_length = pd.to_numeric(length_pairs["predicted"]).to_numpy(float)
    check_cell(
        "conditional_length_mae_um",
        float(np.mean(np.abs(predicted_length - observed_length))),
    )
    check_cell(
        "conditional_length_bias_um",
        float(np.mean(predicted_length - observed_length)),
    )
    check_cell("conditional_length_ccc", _ccc(observed_length, predicted_length))
    check_cell(
        "matched_endpoint_error_um",
        float(np.median(pd.to_numeric(length_pairs["endpoint_error_um"]))),
    )
    check_cell(
        "matched_trajectory_continuity",
        float(np.mean(pd.to_numeric(length_pairs["trajectory_continuity"]))),
    )
    check_cell(
        "endpoint_complete_support_fraction",
        float(support["supported_hairs"].sum() / support["identity_hairs"].sum()),
    )
    containment = pd.to_numeric(topology["axis_containment_fraction"]).to_numpy(float)
    axis_in_root = pd.to_numeric(
        topology["axis_in_root_coverage_fraction"]
    ).to_numpy(float)
    single_component = pd.to_numeric(
        topology["axis_single_component_coverage_fraction"]
    ).to_numpy(float)
    unsupported_gap = pd.to_numeric(
        topology["longest_unsupported_axis_gap_um"]
    ).to_numpy(float)
    _require(
        np.all(np.isfinite(axis_in_root))
        and np.all(np.isfinite(single_component))
        and np.all(np.isfinite(unsupported_gap))
        and np.allclose(containment, axis_in_root, rtol=0.0, atol=1e-12)
        and np.all(single_component <= axis_in_root + 1e-12)
        and np.all(single_component >= 0.0)
        and np.all(axis_in_root <= 1.0)
        and np.all(unsupported_gap >= 0.0),
        "application topology union/single-component/gap contract changed",
    )
    check_cell("axis_containment_median", float(np.median(containment)))
    check_cell("axis_containment_min", float(np.min(containment)))
    check_cell(
        "unsupported_attachment_n",
        float(pd.to_numeric(topology["unsupported_attachment_n"]).sum()),
    )
    return normalized, pairs.copy(), support.copy()


def _derive_phenotype_points(clean_traits: pd.DataFrame) -> pd.DataFrame:
    selected = clean_traits[
        (clean_traits["experiment_key"].astype(str) == "D15_8d")
        & (clean_traits["study_role"].astype(str) == "rhd6_factorial_8d_primary")
        & clean_traits["condition_code"].astype(str).isin(GROUP_ORDER)
        & _bool_series(clean_traits["formal_statistics_eligible"])
    ]
    rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        for endpoint in PRIMARY_ENDPOINTS:
            value = pd.to_numeric(pd.Series([record.get(endpoint)]), errors="coerce").iloc[0]
            if not math.isfinite(float(value)):
                continue
            rows.append(
                {
                    "source_unit": str(record["task_id"]),
                    "cohort": "primary_clean261",
                    "condition_code": str(record["condition_code"]),
                    "formal_eligible": True,
                    "endpoint_key": endpoint,
                    "value": float(value),
                    "unit": ENDPOINT_UNITS[endpoint],
                    "source_image_sha256": str(record["source_image_sha256"]),
                }
            )
    result = pd.DataFrame(rows)
    _require(set(result["endpoint_key"]) == set(PRIMARY_ENDPOINTS), "one or more plant-facing endpoints have no observations")
    return result.sort_values(["endpoint_key", "condition_code", "source_unit"]).reset_index(drop=True)


def _derive_phenotype_effects(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_source_cells = {
        (endpoint, effect)
        for endpoint in PRIMARY_ENDPOINTS
        for effect in EFFECT_SOURCE_ORDER
    }
    for frame, expected_cohort in zip(
        (primary, sensitivity), PHENOTYPE_EFFECT_COHORT_ORDER, strict=True
    ):
        selected = frame[
            frame["endpoint"].astype(str).isin(PRIMARY_ENDPOINTS)
            & frame["effect"].astype(str).isin(EFFECT_MAP)
        ]
        observed_source_cells = [
            (str(endpoint), str(effect))
            for endpoint, effect in zip(
                selected["endpoint"], selected["effect"], strict=True
            )
        ]
        _require(
            len(observed_source_cells) == len(set(observed_source_cells)),
            f"{expected_cohort}: fixed 15-effect family contains duplicate cells",
        )
        _require(
            set(observed_source_cells) == expected_source_cells,
            f"{expected_cohort}: fixed 15-effect family is incomplete or contains unexpected cells",
        )
        _require(set(selected["cohort"].astype(str)) == {expected_cohort}, f"{expected_cohort}: cohort label drift")
        for record in selected.to_dict("records"):
            endpoint = str(record["endpoint"])
            _require(
                str(record.get("model_component")) == PRIMARY_ENDPOINT_COMPONENTS[endpoint],
                f"{expected_cohort}/{endpoint}: model component is outside the fixed conditional phenotype family",
            )
            _require(record.get("causal_treatment_claim_allowed") in {False, "False", "false", 0, "0"}, "causal effect claim entered exploratory table")
            raw_estimate = _finite_number(
                record.get("raw_effect_estimate"), "raw phenotype effect"
            )
            raw_low = _finite_number(
                record.get("raw_effect_ci95_low"), "raw phenotype effect CI"
            )
            raw_high = _finite_number(
                record.get("raw_effect_ci95_high"), "raw phenotype effect CI"
            )
            _require(
                raw_low <= raw_high,
                f"{expected_cohort}/{endpoint}: raw-effect interval is reversed",
            )
            raw_estimand = str(record.get("raw_effect_estimand"))
            raw_interval_method = str(record.get("raw_effect_interval_method"))
            raw_replicates = int(record.get("raw_effect_bootstrap_replicates"))
            raw_seed_value = record.get("raw_effect_bootstrap_seed")
            raw_seed = None if pd.isna(raw_seed_value) else int(raw_seed_value)
            if endpoint == H11_ENDPOINT:
                _require(
                    raw_estimand == RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                    and raw_interval_method == RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                    and raw_replicates == H11_RAW_BOOTSTRAP_REPLICATES
                    and raw_seed
                    == raw_median_bootstrap_seed(
                        seed=H11_RAW_BOOTSTRAP_BASE_SEED,
                        field=H11_ENDPOINT,
                        component="continuous",
                    ),
                    f"{expected_cohort}/{endpoint}: H11 raw-median companion drift",
                )
            else:
                _require(
                    raw_estimand == RAW_EFFECT_OLS_MEAN_CONTRAST
                    and raw_interval_method == RAW_EFFECT_HC3_INTERVAL
                    and raw_replicates == 0
                    and raw_seed is None,
                    f"{expected_cohort}/{endpoint}: raw-mean companion drift",
                )
            rows.append(
                {
                    "cohort": expected_cohort,
                    "endpoint_key": endpoint,
                    "effect_key": EFFECT_MAP[str(record["effect"])],
                    "estimate": _finite_number(record["estimate"], "phenotype effect"),
                    "ci_low": _finite_number(record["ci95_low"], "phenotype effect CI"),
                    "ci_high": _finite_number(record["ci95_high"], "phenotype effect CI"),
                    "endpoint_n": int(record["n"]),
                    "effect_scale": str(record["effect_scale"]),
                    "raw_effect_estimate": raw_estimate,
                    "raw_effect_ci_low": raw_low,
                    "raw_effect_ci_high": raw_high,
                    "raw_effect_unit": ENDPOINT_UNITS[endpoint],
                    "raw_effect_estimand": raw_estimand,
                    "raw_effect_interval_method": raw_interval_method,
                    "raw_effect_bootstrap_replicates": raw_replicates,
                    "raw_effect_bootstrap_seed": raw_seed,
                    "standardized_effect": _finite_number(
                        record.get("standardized_effect"),
                        "standardized phenotype effect",
                    ),
                    "standardized_ci_low": _finite_number(
                        record.get("standardized_ci95_low"),
                        "standardized phenotype effect CI",
                    ),
                    "standardized_ci_high": _finite_number(
                        record.get("standardized_ci95_high"),
                        "standardized phenotype effect CI",
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["raw_effect_bootstrap_seed"] = pd.Series(
        [
            None if pd.isna(value) else int(value)
            for value in result["raw_effect_bootstrap_seed"]
        ],
        dtype=object,
    )
    index_columns = ["endpoint_key", "effect_key", "cohort"]
    _require(
        not result.duplicated(index_columns).any(),
        "fixed phenotype-effect family contains duplicate normalized cells",
    )
    expected_index = pd.MultiIndex.from_product(
        [PRIMARY_ENDPOINTS, EFFECT_ORDER, PHENOTYPE_EFFECT_COHORT_ORDER],
        names=index_columns,
    )
    observed_index = pd.MultiIndex.from_frame(result[index_columns])
    _require(
        set(observed_index) == set(expected_index),
        "fixed phenotype-effect family is missing an ordered endpoint/effect/cohort cell",
    )
    return result.set_index(index_columns).reindex(expected_index).reset_index()


def _derive_profiles(table: pd.DataFrame) -> pd.DataFrame:
    selected = table[
        (table["cohort"].astype(str) == "primary_clean261")
        & (table["cohort_role"].astype(str) == "primary_SHA_disjoint")
        & table["condition_code"].astype(str).isin(GROUP_ORDER)
    ]
    _require(len(selected) == 20, "primary clean axial profile is not four conditions x five bins")
    metric_columns = (
        (
            "identity_abundance",
            "mean_attached_identity_count",
            "mean_attached_identity_count_ci95_low",
            "mean_attached_identity_count_ci95_high",
        ),
        (
            "conditional_median_length_um",
            "median_of_source_unit_conditional_median_length_um",
            "median_of_source_unit_conditional_median_length_um_ci95_low",
            "median_of_source_unit_conditional_median_length_um_ci95_high",
        ),
        (
            "length_support_fraction",
            "endpoint_complete_support_fraction",
            "endpoint_complete_support_fraction_ci95_low",
            "endpoint_complete_support_fraction_ci95_high",
        ),
    )
    rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        for metric_key, estimate, low, high in metric_columns:
            rows.append(
                {
                    "cohort": "primary_clean261",
                    "condition_code": str(record["condition_code"]),
                    "bin_start_mm": float(record["bin_start_um"]) / 1000.0,
                    "bin_end_mm": float(record["bin_end_um"]) / 1000.0,
                    "metric_key": metric_key,
                    "estimate": record.get(estimate),
                    "ci_low": record.get(low),
                    "ci_high": record.get(high),
                    "eligible_n": int(record["eligible_source_units"]),
                    "length_supported_n": int(record["length_measurable_source_units"]),
                    "bootstrap_repetitions": int(record["bootstrap_replicates_requested"]),
                    "unit_of_analysis": str(record["unit_of_analysis"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric_key", "condition_code", "bin_start_mm"]).reset_index(drop=True)


def _validate_wt_secondary_resources(
    *,
    analysis_summary: Mapping[str, Any],
    named_paths: Mapping[str, Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read and hash-close the independent WT secondary evidence family."""

    columns = {
        "wt_within_experiment_contrasts": (
            "cohort", "cohort_role", "endpoint", "experiment_key",
            "developmental_day", "developmental_day_status", "effect_scale",
            "log_effect_30C_over_22C", "log_effect_standard_error",
            "sampling_variance", "estimate_30C_over_22C", "ci95_low",
            "ci95_high", "p_value_model", "p_value_model_BH_FDR",
            "reject_model_BH_FDR_0p05", "multiplicity_family",
            "analysis_status", "not_estimable_reason", "meta_eligible",
            "meta_exclusion_reason", "inference_status",
        ),
        "wt_within_day_meta_analysis": (
            "cohort", "cohort_role", "endpoint", "developmental_day",
            "k_eligible_experiments", "eligible_experiments", "model",
            "effect_scale", "log_effect_30C_over_22C",
            "log_effect_standard_error_hartung_knapp",
            "estimate_30C_over_22C", "ci95_low", "ci95_high",
            "p_value_hartung_knapp", "p_value_hartung_knapp_BH_FDR",
            "reject_hartung_knapp_BH_FDR_0p05", "multiplicity_family",
            "analysis_status", "not_estimable_reason",
            "cross_day_pooling_performed", "unknown_day_contrasts_included",
            "inference_status",
        ),
        "wt_temperature_qc_flow": (
            "cohort", "cohort_role", "experiment_key", "developmental_day",
            "developmental_day_status", "endpoint", "base_gate_pass",
            "endpoint_gate_pass", "model_status", "not_estimable_reason",
            "phenotype_outlier_filter_applied",
        ),
    }
    frames = {
        role: _read_table(named_paths[role], role, columns[role])
        for role in WT_SECONDARY_RESOURCE_ROLES
    }
    try:
        evidence = validate_wt_secondary_evidence(
            contrasts=frames["wt_within_experiment_contrasts"].to_dict(
                "records"
            ),
            meta=frames["wt_within_day_meta_analysis"].to_dict("records"),
            flow=frames["wt_temperature_qc_flow"].to_dict("records"),
        )
        binding = validate_wt_secondary_analysis_binding(
            analysis_summary=analysis_summary,
            evidence_summary=evidence,
            table_sha256={
                role: sha256_file(named_paths[role])
                for role in WT_SECONDARY_RESOURCE_ROLES
            },
        )
    except ValueError as error:
        raise FigureInputAssemblyError(
            f"WT secondary evidence validation failed: {error}"
        ) from error
    return frames, binding


def _receipt_identity(payload: Mapping[str, Any], role: str) -> str:
    field = IDENTITY_FIELDS[role]
    return _sealed(payload, field, role)


def _identity_for_core(role: str, payload: Mapping[str, Any], file_sha: str) -> str:
    fields = {
        "root_exact283": "audit_identity_sha256",
        "stageb": "summary_identity_sha256",
        "fusion": "summary_identity_sha256",
        "traits": "export_identity_sha256",
        "cohorts": "cohort_build_identity_sha256",
        "analysis": "analysis_identity_sha256",
        "profiles": "export_identity_sha256",
    }
    field = fields.get(role)
    identity = payload.get(field) if field else file_sha
    _require(_is_sha256(identity), f"{role}: output identity missing")
    return str(identity)


def _derive_cohort_flow(
    evaluation: Mapping[str, Any], cohorts: Mapping[str, Any]
) -> pd.DataFrame:
    counts = cohorts["counts"]
    full = int(counts["biological_full"])
    clean = int(counts["biological_clean"])
    overlap = int(counts["human_curated_overlap"])
    formal = int(cohorts["sensitivity_full_formal_statistics_eligible"])
    review = full - formal
    training = evaluation["training_contract"]
    rows = [
        ("human443", "HumanCurated", int(training["training_images"]) + int(training["validation_images"]), "", "development"),
        ("train399", "Train", int(training["training_images"]), "human443", "training"),
        ("qcdevelopment44", "QC-development", int(training["validation_images"]), "human443", "selection"),
        ("bio_full", "Application", full, "", "application"),
        ("overlap", "SHA overlap", overlap, "bio_full", "sensitivity"),
        ("bio_clean", "Clean primary", clean, "bio_full", "primary"),
        ("formal", "Formal", formal, "bio_full", "formal"),
        ("review_only", "Review-only", review, "bio_full", "review"),
    ]
    _require(full == clean + overlap, "cohort counts do not close")
    _require(full == formal + review, "formal/review counts do not close")
    return pd.DataFrame(rows, columns=("node_id", "label", "count", "parent_id", "role"))


def _derive_workflow(
    payloads: Mapping[str, Mapping[str, Any]], source_hashes: Mapping[str, str]
) -> pd.DataFrame:
    names = {
        "train399_evaluation": "Train399 QC-development evaluation",
        "root_exact283": "Fresh exact283 root provider",
        "stageb": "Stage-B identities",
        "fusion": "Identity/length fusion",
        "traits": "Canonical trait export",
        "cohorts": "Clean261/full283 cohorts",
        "analysis": "Exploratory biological analysis",
        "profiles": "Distal-axis profiles",
    }
    return pd.DataFrame(
        [
            {
                "stage_order": index,
                "stage_name": names[role],
                "receipt_role": role,
                "output_identity_sha256": _identity_for_core(
                    role, payloads[role], source_hashes[role]
                ),
            }
            for index, role in enumerate(CORE_ROLES, 1)
        ]
    )


def _validate_overlay(
    *,
    receipt: Mapping[str, Any],
    selection_path: Path,
    selection: pd.DataFrame,
    prediction_sha: Mapping[str, str],
    full_traits: pd.DataFrame,
) -> None:
    _require(receipt.get("schema_version") == OVERLAY_RECEIPT_SCHEMA, "overlay receipt schema changed")
    _require(
        receipt.get("status") == OVERLAY_RECEIPT_STATUS
        and receipt.get("case_plan_columns") == ["case_role", "task_id"]
        and receipt.get("case_selection_basis") == OVERLAY_CASE_SELECTION_BASIS
        and receipt.get("random_or_representative_performance_sample") is False
        and receipt.get("experimental_condition_metadata_used_for_rendering") is False
        and receipt.get("experimental_condition_metadata_used_for_evidence_assembly") is False,
        "overlay evidence contract is not a preselected non-performance acquisition-challenge gallery",
    )
    review = receipt.get("full_cohort_review_export")
    review_source_authority = receipt.get("source_authority_sha256")
    _require(
        isinstance(review, Mapping)
        and review.get("schema_version") == OVERLAY_REVIEW_SCHEMA
        and review.get("status") == OVERLAY_REVIEW_STATUS
        and review.get("expected_task_count") == 283
        and review.get("images") == 283
        and review.get("index_rows") == 283
        and review.get("checklist_rows") == 283
        and review.get("review_root") == "full283_review_overlays"
        and review.get("index_csv") == "full283_review_index.csv"
        and review.get("checklist_csv") == "full283_review_checklist.csv"
        and review.get("readme_cn") == "README_CN.md"
        and review.get("summary_json") == "full283_review_summary.json"
        and review.get("review_status_on_export")
        == OVERLAY_REVIEW_PENDING_STATUS
        and review.get("organization_fields")
        == [
            "experiment_key",
            "condition_code",
            "formal_statistics_eligible",
        ]
        and review.get("experimental_condition_metadata_used_for_prediction")
        is False
        and review.get("experimental_condition_metadata_used_for_rendering")
        is False
        and review.get(
            "experimental_condition_metadata_used_for_output_organization"
        )
        is True
        and review.get("create_only") is True
        and review.get("canonical_annotations_read") is False
        and review.get("root_cap_region_statistics_included") is False
        and review.get("blind_images_used") == 0
        and receipt.get("exact_cohort_review_images") == 283
        and receipt.get(
            "paper_overlay_bytes_reused_from_full_cohort_review_export"
        )
        is True
        and receipt.get(
            "paper_overlay_sha256_matches_full_cohort_review_export"
        )
        is True
        and isinstance(review_source_authority, Mapping)
        and _is_sha256(review_source_authority.get("application_manifest"))
        and _is_sha256(review_source_authority.get("full_traits"))
        and _is_sha256(review_source_authority.get("fusion_summary"))
        and _is_sha256(review_source_authority.get("overlay_builder_source"))
        and _is_sha256(review_source_authority.get("renderer_source"))
        and all(
            _is_sha256(review.get(field))
            for field in (
                "index_csv_sha256",
                "checklist_csv_sha256",
                "readme_cn_sha256",
                "summary_json_sha256",
                "review_export_identity_sha256",
                "ordered_task_set_identity_sha256",
                "overlay_png_set_identity_sha256",
            )
        ),
        "overlay evidence lacks the exact283 final-fusion review export contract",
    )
    inset_contract = receipt.get("inset_contract")
    _require(
        isinstance(inset_contract, Mapping)
        and inset_contract.get("roles") == list(OVERLAY_INSET_ROLES)
        and inset_contract.get("locked_anchor_task_ids")
        == OVERLAY_LOCKED_ANCHOR_TASK_IDS
        and inset_contract.get("source_and_overlay_use_identical_crop_coordinates")
        is True
        and inset_contract.get("whole_image_context_retained") is True
        and inset_contract.get("performance_based_crop_selection") is False,
        "overlay evidence lacks the deterministic two-anchor inset contract",
    )
    _guard_red_lines("overlay_index", receipt)
    _require(receipt.get("selection_csv_sha256") == sha256_file(selection_path), "overlay receipt does not bind selection CSV")
    _require(set(selection["case_role"].astype(str)) == set(CASE_ROLES), "overlay case roles incomplete")
    _require(selection["case_role"].value_counts().eq(1).all(), "overlay case roles are not unique")
    traits_source = dict(
        zip(full_traits["task_id"].astype(str), full_traits["source_image_sha256"].astype(str), strict=True)
    )
    for row in selection.to_dict("records"):
        task_id = str(row["task_id"])
        _require(task_id in prediction_sha and task_id in traits_source, f"{task_id}: overlay task not in final prediction cohort")
        _require(row["prediction_sha256"] == prediction_sha[task_id], f"{task_id}: overlay prediction SHA mismatch")
        _require(
            row["raw_source_image_sha256"] == traits_source[task_id],
            f"{task_id}: overlay raw-source image SHA mismatch",
        )
        _require(
            row["overlay_sha256"]
            == row["full_cohort_review_overlay_sha256"]
            and str(
                row["overlay_bytes_reused_from_full_cohort_review_export"]
            ).strip().casefold()
            in {"true", "1", "yes"}
            and str(row["full_cohort_review_overlay_path"]).startswith(
                "full283_review_overlays/"
            ),
            f"{task_id}: paper overlay is not byte-identical to its exact283 review PNG",
        )
        _require(
            str(row["case_selection_basis"]) == OVERLAY_CASE_SELECTION_BASIS,
            f"{task_id}: overlay case-selection basis changed",
        )
        for field in (
            "random_or_representative_performance_sample",
            "experimental_condition_metadata_used_for_rendering",
            "experimental_condition_metadata_used_for_evidence_assembly",
        ):
            _require(
                str(row[field]).strip().casefold() in {"false", "0", "no"},
                f"{task_id}: overlay contract requires {field}=false",
            )
        inset_required = str(row["inset_required"]).strip().casefold() in {
            "true",
            "1",
            "yes",
        }
        _require(
            inset_required == (str(row["case_role"]) in OVERLAY_INSET_ROLES),
            f"{task_id}: deterministic inset role changed",
        )
        if str(row["case_role"]) in OVERLAY_LOCKED_ANCHOR_TASK_IDS:
            _require(
                task_id
                == OVERLAY_LOCKED_ANCHOR_TASK_IDS[str(row["case_role"])],
                f"{task_id}: prelocked Fig.4 anchor task ID changed",
            )
        if inset_required:
            coordinates = [
                float(row[field])
                for field in ("inset_x0", "inset_y0", "inset_x1", "inset_y1")
            ]
            _require(
                all(math.isfinite(value) for value in coordinates)
                and coordinates[0] < coordinates[2]
                and coordinates[1] < coordinates[3]
                and _is_sha256(row["inset_geometry_sha256"])
                and str(row["inset_rule"]) != "not_applicable",
                f"{task_id}: deterministic inset geometry is incomplete",
            )
        else:
            _require(
                str(row["inset_rule"]) == "not_applicable",
                f"{task_id}: non-anchor case carries an inset rule",
            )


def _validate_runtime_bundle(
    *,
    current_latency_path: Path,
    current_latency: Mapping[str, Any],
    current_per_image_path: Path,
    current_production_path: Path,
    current_production: Mapping[str, Any],
    baseline_latency_path: Path,
    baseline_latency: Mapping[str, Any],
    baseline_per_image_path: Path,
    baseline_production_path: Path,
    baseline_production: Mapping[str, Any],
    latency_comparison: Mapping[str, Any],
    production_comparison: Mapping[str, Any],
    final: bool,
) -> dict[str, Any]:
    """Validate the two-mode benchmark contract and return a plotting summary.

    The workflow owns the precise benchmark schemas.  This compiler enforces
    the cross-system scientific semantics: two genuine modes, exact same 283
    source/image locks and hardware, direct I/O-inclusive scope, and no
    component-only number masquerading as a speedup.
    """

    summaries = (
        ("runtime_latency", current_latency),
        ("runtime_production", current_production),
        ("baseline_runtime_latency", baseline_latency),
        ("baseline_runtime_production", baseline_production),
    )
    comparisons = (
        ("runtime_latency_comparison", latency_comparison),
        ("runtime_production_comparison", production_comparison),
    )
    for role, payload in (*summaries, *comparisons):
        _receipt_identity(payload, role)
        _guard_red_lines(role, payload)
    latency_schema = "PHAxis-full-workflow-sequential-latency-benchmark-1.0"
    production_schema = "PHAxis-full-workflow-production-batch-benchmark-1.0"
    comparison_schema = "PHAxis-full-workflow-benchmark-comparison-1.0"
    scope = "raw_image_to_final_traits_and_profiles_direct"
    latency_modes = {
        "sequential_persistent_full283",
        "sequential_cold_cli_full283",
    }
    for label, latency, production in (
        ("PHAxis", current_latency, current_production),
        ("frozen_v1", baseline_latency, baseline_production),
    ):
        _require(latency.get("schema_version") == latency_schema, f"{label}: latency schema changed")
        _require(production.get("schema_version") == production_schema, f"{label}: production schema changed")
        _require(
            latency.get("status") == "completed_direct_full283"
            and latency.get("benchmark_mode") in latency_modes
            and latency.get("measurement_scope") == scope
            and latency.get("images") == 283
            and latency.get("fresh_direct_run") is True
            and latency.get("resume_or_cache_used") is False,
            f"{label}: sequential full283 latency benchmark is not a permitted direct mode",
        )
        if latency.get("benchmark_mode") == "sequential_persistent_full283":
            _require(
                latency.get("startup_included_in_per_image_wall") is False,
                f"{label}: persistent latency incorrectly includes per-image startup",
            )
        else:
            _require(
                latency.get("startup_included_in_per_image_wall") is True,
                f"{label}: cold-CLI latency excludes required per-image startup",
            )
        _require(
            production.get("status") == "completed_direct_full283"
            and production.get("benchmark_mode") == "production_batch_full283"
            and production.get("measurement_scope") == scope
            and production.get("images") == 283
            and production.get("fresh_direct_run") is True
            and production.get("resume_or_cache_used") is False
            and production.get("per_image_latency_reported") is False,
            f"{label}: production batch full283 benchmark is not direct",
        )
        for payload in (latency, production):
            _require(
                payload.get("includes_io") is True
                and payload.get("includes_preprocess") is True
                and payload.get("includes_stitching_fusion_traits_profiles") is True,
                f"{label}: benchmark excludes part of the workflow",
            )
        for field in (
            "source_manifest_sha256",
            "source_image_lock_identity_sha256",
            "hardware_identity_sha256",
            "model_contract_proposal_sha256",
            "model_contract_proposal_identity_sha256",
        ):
            _require(latency.get(field) == production.get(field), f"{label}: A/B modes differ in {field}")
    _require(
        current_latency.get("benchmark_mode") == baseline_latency.get("benchmark_mode"),
        "PHAxis and frozen-v1 latency measurements use different latency modes",
    )
    _require(
        current_latency.get("per_image_csv_sha256") == sha256_file(current_per_image_path),
        "PHAxis latency summary does not bind its per-image table",
    )
    _require(
        baseline_latency.get("per_image_csv_sha256") == sha256_file(baseline_per_image_path),
        "baseline latency summary does not bind its per-image table",
    )
    for label, comparison, candidate_path, baseline_path, mode, speedup_field in (
        (
            "latency",
            latency_comparison,
            current_latency_path,
            baseline_latency_path,
            str(current_latency.get("benchmark_mode")),
            "median_latency_speedup_frozen_v1_over_phaxis",
        ),
        (
            "production",
            production_comparison,
            current_production_path,
            baseline_production_path,
            "production_batch_full283",
            "batch_wall_speedup_frozen_v1_over_phaxis",
        ),
    ):
        _require(comparison.get("schema_version") == comparison_schema, f"{label}: comparison schema changed")
        comparable = comparison.get("comparable") is True
        if final:
            _require(comparable, f"final figures require comparable frozen-v1 {label} benchmark")
        _require(
            comparison.get("phaxis_summary_sha256") == sha256_file(candidate_path)
            and comparison.get("baseline_summary_sha256") == sha256_file(baseline_path),
            f"{label}: comparison does not bind named benchmark summaries",
        )
        if comparable:
            _require(
                comparison.get("status") == "comparable_direct_full283"
                and comparison.get("benchmark_mode") == mode
                and comparison.get("same_283_source_manifest_hardware_and_io_scope") is True
                and _finite_number(comparison.get(speedup_field), f"{label} speedup") > 0,
                f"{label}: like-for-like speedup is incomplete",
            )
        else:
            _require(speedup_field not in comparison, f"{label}: non-comparable receipt reports speedup")

    return {
        "schema_version": "PHAxis-manuscript-two-mode-runtime-input-1.0",
        "status": "completed_two_mode_direct_full283" if final else "provisional_two_mode_runtime",
        "measurement_scope": scope,
        "latency_mode": current_latency.get("benchmark_mode"),
        "sequential_latency_full283": deepcopy(dict(current_latency)),
        "production_batch_full283": deepcopy(dict(current_production)),
        "baseline_sequential_latency_full283": deepcopy(dict(baseline_latency)),
        "baseline_production_batch_full283": deepcopy(dict(baseline_production)),
        "latency_comparison": deepcopy(dict(latency_comparison)),
        "production_comparison": deepcopy(dict(production_comparison)),
        "per_image_csv_sha256": sha256_file(current_per_image_path),
        "baseline_per_image_csv_sha256": sha256_file(baseline_per_image_path),
        "batch_latency_is_never_derived_per_image": True,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }


def _make_staging_directory(destination: Path) -> Path:
    """Create a private staging directory without repeating a long output name."""

    return Path(
        tempfile.mkdtemp(
            prefix=FIGURE_INPUT_STAGING_PREFIX,
            dir=destination.parent,
        )
    )


def build_publication_figure_inputs(
    *,
    mode: str,
    output: str | Path,
    model_contract_proposal: str | Path,
    train399_candidate: str | Path,
    train399_selection: str | Path,
    dataset_manifest: str | Path,
    split_manifest: str | Path,
    trait_contract: str | Path,
    image_traits_schema: str | Path,
    figure1_image: str | Path,
    figure1_geometry: str | Path,
    historical_development_receipt: str | Path,
    historical_oof_per_image: str | Path,
    measurement_assurance_receipt: str | Path,
    assurance_metrics: str | Path,
    assurance_pairs: str | Path,
    assurance_support: str | Path,
    assurance_topology: str | Path,
    overlay_index_receipt: str | Path,
    overlay_selection: str | Path,
    clean_traits: str | Path,
    full_traits: str | Path,
    full_image_traits: str | Path,
    analysis_primary_table: str | Path,
    analysis_sensitivity_table: str | Path,
    profile_analysis_summary: str | Path,
    profile_analysis_table: str | Path,
    sensitivity_profiles_summary: str | Path,
    runtime_latency_summary: str | Path,
    runtime_per_image: str | Path,
    runtime_production_summary: str | Path,
    runtime_latency_comparison: str | Path,
    runtime_production_comparison: str | Path,
    baseline_runtime_latency_summary: str | Path,
    baseline_runtime_per_image: str | Path,
    baseline_runtime_production_summary: str | Path,
    benchmark_same_hardware: str | Path,
    benchmark_artifact_inventory: str | Path,
    training_receipts: Mapping[int, str | Path],
    receipt_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    _require(mode in {"final", "provisional"}, "mode must be final or provisional")
    final = mode == "final"
    destination = Path(output).resolve()
    _require(not destination.exists(), f"refusing to overwrite output: {destination}")
    _require(set(receipt_paths) == set(CORE_ROLES), "core receipt roles are incomplete")
    expected_training_seeds = set(range(2026082801, 2026082806))
    _require(
        set(training_receipts) == expected_training_seeds,
        "exact five fixed-seed training completion receipts are required",
    )

    core_paths = {
        role: _resolve_file(receipt_paths[role], role, final=final) for role in CORE_ROLES
    }
    core_payloads = {role: _read_object(path, role) for role, path in core_paths.items()}
    named_paths = {
        "model_contract_proposal": _resolve_file(model_contract_proposal, "model_contract_proposal", final=final),
        "train399_candidate": _resolve_file(train399_candidate, "train399_candidate", final=final),
        "train399_selection": _resolve_file(train399_selection, "train399_selection", final=final),
        "dataset_manifest": _resolve_file(dataset_manifest, "dataset_manifest", final=final),
        "split_manifest": _resolve_file(split_manifest, "split_manifest", final=final),
        "trait_contract": _resolve_file(trait_contract, "trait_contract", final=final),
        "image_traits_schema": _resolve_file(image_traits_schema, "image_traits_schema", final=final),
        "figure1_image": _resolve_file(figure1_image, "figure1_image", final=final),
        "figure1_geometry": _resolve_file(figure1_geometry, "figure1_geometry", final=final),
        "historical_development": _resolve_file(historical_development_receipt, "historical_development", final=final),
        "historical_oof_per_image": _resolve_file(historical_oof_per_image, "historical_oof_per_image", final=final),
        "measurement_assurance": _resolve_file(measurement_assurance_receipt, "measurement_assurance", final=final),
        "assurance_metrics": _resolve_file(assurance_metrics, "assurance_metrics", final=final),
        "assurance_pairs": _resolve_file(assurance_pairs, "assurance_pairs", final=final),
        "assurance_support": _resolve_file(assurance_support, "assurance_support", final=final),
        "assurance_topology": _resolve_file(assurance_topology, "assurance_topology", final=final),
        "overlay_index": _resolve_file(overlay_index_receipt, "overlay_index", final=final),
        "overlay_selection": _resolve_file(overlay_selection, "overlay_selection", final=final),
        "clean_traits": _resolve_file(clean_traits, "clean_traits", final=final),
        "full_traits": _resolve_file(full_traits, "full_traits", final=final),
        "full_image_traits": _resolve_file(full_image_traits, "full_image_traits", final=final),
        "analysis_primary_table": _resolve_file(analysis_primary_table, "analysis_primary_table", final=final),
        "analysis_sensitivity_table": _resolve_file(analysis_sensitivity_table, "analysis_sensitivity_table", final=final),
        "profile_analysis": _resolve_file(profile_analysis_summary, "profile_analysis", final=final),
        "profile_analysis_table": _resolve_file(profile_analysis_table, "profile_analysis_table", final=final),
        "sensitivity_profiles_summary": _resolve_file(sensitivity_profiles_summary, "sensitivity_profiles_summary", final=final),
        "runtime_latency": _resolve_file(runtime_latency_summary, "runtime_latency", final=final),
        "runtime_per_image": _resolve_file(runtime_per_image, "runtime_per_image", final=final),
        "runtime_production": _resolve_file(runtime_production_summary, "runtime_production", final=final),
        "runtime_latency_comparison": _resolve_file(runtime_latency_comparison, "runtime_latency_comparison", final=final),
        "runtime_production_comparison": _resolve_file(runtime_production_comparison, "runtime_production_comparison", final=final),
        "baseline_runtime_latency": _resolve_file(baseline_runtime_latency_summary, "baseline_runtime_latency", final=final),
        "baseline_runtime_per_image": _resolve_file(baseline_runtime_per_image, "baseline_runtime_per_image", final=final),
        "baseline_runtime_production": _resolve_file(baseline_runtime_production_summary, "baseline_runtime_production", final=final),
        "benchmark_same_hardware": _resolve_file(benchmark_same_hardware, "benchmark_same_hardware", final=final),
        "benchmark_artifact_inventory": _resolve_file(benchmark_artifact_inventory, "benchmark_artifact_inventory", final=final),
    }
    _require(
        named_paths["analysis_primary_table"].parent
        == named_paths["analysis_sensitivity_table"].parent,
        "analysis primary/sensitivity tables do not share one sealed table root",
    )
    analysis_table_root = named_paths["analysis_primary_table"].parent
    for role in WT_SECONDARY_RESOURCE_ROLES:
        named_paths[role] = _resolve_file(
            analysis_table_root / WT_SECONDARY_TABLE_FILENAMES[role],
            role,
            final=final,
        )
    for seed in sorted(expected_training_seeds):
        role = f"training_receipt_seed_{seed}"
        named_paths[role] = _resolve_file(
            training_receipts[seed], role, final=final
        )
    proposal = _read_object(named_paths["model_contract_proposal"], "model_contract_proposal")
    selection = _read_object(named_paths["train399_selection"], "train399_selection")
    prediction_inputs = _validate_train399_prediction_inputs(
        core_payloads["train399_evaluation"]
    )
    source_hashes, proposal_sha, proposal_identity = _validate_core(
        paths=core_paths,
        payloads=core_payloads,
        proposal_path=named_paths["model_contract_proposal"],
        proposal=proposal,
        selection_path=named_paths["train399_selection"],
        selection=selection,
        split_manifest_path=named_paths["split_manifest"],
        final=final,
    )
    prediction_sha = _prediction_map(core_payloads["fusion"], final=final)

    split = _read_table(named_paths["split_manifest"], "split_manifest", ("task_id", "split", "family_key"))
    qc_per_image, qc_tolerance, qc_threshold, qc_uncertainty = _build_qc_development(
        core_payloads["train399_evaluation"], selection, split, prediction_inputs
    )

    historical_receipt = _read_object(named_paths["historical_development"], "historical_development")
    _require(historical_receipt.get("schema_version") == HISTORICAL_RECEIPT_SCHEMA, "historical development schema changed")
    _require(
        historical_receipt.get("status") == "completed_locked_historical_oof443_development"
        and historical_receipt.get("scope") == "family-isolated OOF443 development evidence; non-independent"
        and historical_receipt.get("independent_accuracy_claim_allowed") is False,
        "historical development scope is invalid",
    )
    _guard_red_lines("historical_development", historical_receipt)
    historical_identity = _receipt_identity(historical_receipt, "historical_development")
    historical_sources = historical_receipt.get("source_table_sha256")
    _require(isinstance(historical_sources, Mapping), "historical source hashes missing")
    _require(
        historical_sources.get("per_image_sufficient_statistics")
        == sha256_file(named_paths["historical_oof_per_image"]),
        "historical receipt source hash mismatch",
    )
    _require(
        historical_receipt.get("source_authority_sha256", {}).get("trusted_local_oof_pickle")
        and historical_receipt.get("family_key_overlap_across_folds") == 0,
        "historical receipt does not bind the OOF prediction authority/family isolation",
    )
    historical_per_image = _read_table(
        named_paths["historical_oof_per_image"],
        "historical_oof_per_image",
        ("source_unit", "family_key", "fold", "quality_band", "density_band", "annotation_mode", "n_pred", "n_gt", "biological_presence_tp_20um"),
    )
    historical_strata = _historical_group_rows(historical_per_image, historical_receipt)

    assurance_receipt = _read_object(named_paths["measurement_assurance"], "measurement_assurance")
    assurance_identity = _receipt_identity(assurance_receipt, "measurement_assurance")
    assurance_sources = assurance_receipt.get("source_table_sha256")
    _require(isinstance(assurance_sources, Mapping), "measurement assurance source hashes missing")
    for role in (
        "assurance_metrics",
        "assurance_pairs",
        "assurance_support",
        "assurance_topology",
    ):
        short = role.removeprefix("assurance_")
        _require(assurance_sources.get(short) == sha256_file(named_paths[role]), f"measurement assurance {short} hash mismatch")
    assurance_authorities = assurance_receipt.get("source_authority_sha256")
    assurance_identities = assurance_receipt.get(
        "source_authority_identity_sha256"
    )
    _require(
        isinstance(assurance_authorities, Mapping)
        and assurance_authorities.get("train399_evaluation")
        == source_hashes["train399_evaluation"]
        and assurance_authorities.get("cohorts_receipt")
        == source_hashes["cohorts"]
        and assurance_authorities.get("root_exact283_receipt")
        == source_hashes["root_exact283"]
        and assurance_authorities.get("application_fusion_summary")
        == source_hashes["fusion"]
        and assurance_authorities.get("clean_traits")
        == sha256_file(named_paths["clean_traits"]),
        "measurement assurance does not bind the named evaluator/cohort/provider/trait authorities",
    )
    _require(
        isinstance(assurance_identities, Mapping)
        and assurance_identities.get("stageb_detection_ordered_file_set")
        == prediction_inputs["stageb_detection_set_identity_sha256"]
        and _is_sha256(
            assurance_identities.get(
                "qcdev_fusion_prediction_ordered_file_set"
            )
        )
        and assurance_identities.get("application_fusion_summary")
        == _identity_for_core(
            "fusion", core_payloads["fusion"], source_hashes["fusion"]
        )
        and isinstance(
            assurance_receipt.get("application_prediction_file_locks"), list
        )
        and len(assurance_receipt["application_prediction_file_locks"]) == 261
        and assurance_identities.get(
            "application_formal_prediction_ordered_file_set"
        )
        == sha256_json(
            assurance_receipt["application_prediction_file_locks"]
        ),
        "measurement assurance prediction-set provenance differs from evaluator/fusion authorities",
    )
    assurance_contract = assurance_receipt.get("measurement_contract")
    _require(
        isinstance(assurance_contract, Mapping)
        and assurance_contract.get("root_boundary_tolerance_um") == 5.0
        and assurance_contract.get("distal_pck_threshold_um") == 25.0
        and assurance_contract.get("conditional_length_base_match_tolerance_um")
        == 20.0
        and assurance_contract.get("matched_trajectory_tolerance_um") == 20.0
        and assurance_contract.get("matched_trajectory_resample_step_um") == 2.0
        and assurance_contract.get("axis_containment_sampling")
        == "nearest integer pixel on sealed ordered axis versus sealed final root mask"
        and assurance_contract.get("application_axis_single_component_policy")
        == "8-connected root-mask component supporting the largest number of sealed ordered-axis samples; ties use the smaller component label"
        and assurance_contract.get(
            "application_longest_unsupported_axis_gap_definition"
        )
        == "longest contiguous ordered-axis arc-length run whose segment endpoints are not both supported by the winning root-mask component"
        and assurance_contract.get("fig4_case_audit_schema")
        == "PHAxis-Fig4-case-audit-2.0"
        and assurance_contract.get("fig4_profile_0_5mm_eligibility")
        == "formal_statistics_eligible and visible_root_axis_length_um at least 5000"
        and assurance_contract.get("root_trait_count") == 19
        and assurance_contract.get("root_cap_region_output") is False,
        "measurement assurance geometry contract changed",
    )
    _require(
        assurance_contract.get("scale_coverage_denominator")
        == "visible_annotated_scale_bar_cases"
        and assurance_contract.get("scale_localization_denominator")
        == "detected_visible_scale_bars"
        and assurance_contract.get("scale_calibration_denominator")
        == "detected_visible_scale_bars"
        and assurance_contract.get("scale_absence_specificity_status")
        == SCALE_ABSENCE_SPECIFICITY_STATUS
        and assurance_contract.get("scale_fail_closed_evidence_basis")
        == SCALE_FAIL_CLOSED_EVIDENCE_BASIS,
        "measurement assurance scale applicability contract changed",
    )
    metrics_source = _read_table(named_paths["assurance_metrics"], "assurance_metrics", ("domain", "metric_key", "label", "value", "ci_low", "ci_high", "unit", "n", "instances", "ci_method", "bootstrap_repetitions", "bootstrap_seed"))
    pairs_source = _read_table(named_paths["assurance_pairs"], "assurance_pairs", ("pair_type", "source_unit", "pair_id", "observed", "predicted", "unit", "relative_error_percent", "scale_line_endpoint_error_um", "source_image_sha256", "endpoint_error_um", "trajectory_continuity"))
    support_source = _read_table(named_paths["assurance_support"], "assurance_support", ("condition_code", "support_fraction", "supported_hairs", "identity_hairs", "source_units"))
    topology_source = _read_table(
        named_paths["assurance_topology"],
        "assurance_topology",
        (
            "source_unit", "axis_containment_fraction",
            "axis_in_root_coverage_fraction",
            "axis_single_component_coverage_fraction",
            "longest_unsupported_axis_gap_um", "root_mask_component_count",
            "axis_support_component_label", "unsupported_attachment_n",
            "identity_hair_n",
        ),
    )
    assurance_metrics_frame, assurance_pairs_frame, assurance_support_frame = _normalize_assurance(
        assurance_receipt,
        metrics_source,
        pairs_source,
        support_source,
        topology_source,
        receipt_path=named_paths["measurement_assurance"],
    )
    try:
        qcdev_assignment = build_qcdev_assignment(
            assurance_receipt,
            receipt_path=named_paths["measurement_assurance"],
        )
    except PublicationAuthorityError as error:
        raise FigureInputAssemblyError(str(error)) from error

    traits_columns = (
        "task_id", "source_image_sha256", "experiment_key", "condition_code",
        "study_role", "formal_statistics_eligible", "measurement_tier",
        "automatic_measurement_fail_closed", "exclusion_reason", "hair_count",
        "hair_length_measurement_hair_count", "hair_length_measurement_fraction",
        "attachment_axis_valid_fraction", "distal_window_1_4mm_eligible",
        *PRIMARY_ENDPOINTS,
    )
    clean = _read_table(named_paths["clean_traits"], "clean_traits", traits_columns)
    full = _read_table(named_paths["full_traits"], "full_traits", traits_columns)
    full_image = _read_table(
        named_paths["full_image_traits"],
        "full_image_traits",
        (
            "task_id",
            "source_image_sha256",
            "model_bundle_id",
            "root_expert_id",
            "formal_statistics_eligible",
        ),
    )
    cohort_outputs = core_payloads["cohorts"].get("output_sha256", {})
    _require(
        cohort_outputs.get("primary_clean261", {}).get("traits") == sha256_file(named_paths["clean_traits"])
        and cohort_outputs.get("sensitivity_full283", {}).get("traits") == sha256_file(named_paths["full_traits"]),
        "cohort receipt does not bind named clean/full trait tables",
    )
    _require(
        core_payloads["traits"].get("image_traits_sha256")
        == sha256_file(named_paths["full_image_traits"]),
        "trait-export receipt does not bind the named canonical image_traits table",
    )
    if final:
        _require(clean["task_id"].nunique() == len(clean) == 261, "clean traits are not clean261")
        _require(full["task_id"].nunique() == len(full) == 283, "full traits are not full283")
        _require(
            full_image["task_id"].nunique() == len(full_image) == 283,
            "canonical image_traits are not exact283",
        )
        _require(set(clean["task_id"]).issubset(set(full["task_id"])), "clean261 is not a subset of full283")

    geometry = _read_object(named_paths["figure1_geometry"], "figure1_geometry")
    _guard_red_lines("figure1_geometry", geometry)
    figure1_task = str(geometry.get("task_id"))
    _require(geometry.get("source_image_sha256") == sha256_file(named_paths["figure1_image"]), "Figure 1 geometry/image SHA mismatch")
    _require(geometry.get("prediction_sha256") == prediction_sha.get(figure1_task), "Figure 1 geometry is not bound to selected final prediction")
    source_lookup = dict(zip(full["task_id"].astype(str), full["source_image_sha256"].astype(str), strict=True))
    _require(source_lookup.get(figure1_task) == geometry.get("source_image_sha256"), "Figure 1 source image is not in full283 traits")
    trait_payload = _read_object(named_paths["trait_contract"], "trait_contract")
    _guard_red_lines("trait_contract", trait_payload)
    _require(
        trait_payload.get("schema_version") == "PHAxis-trait-contract-1.0.0"
        and trait_payload.get("counts", {}).get("nonredundant_biological_numeric_fields") == 32
        and trait_payload.get("counts", {}).get("primary_root_fields") == 19
        and trait_payload.get("counts", {}).get("root_hair_fields") == 13
        and trait_payload.get("counts", {}).get("root_cap_region_fields") == 0,
        "trait contract is not the locked 32-trait ontology",
    )
    canonical_trait_fields = [
        str(record.get("field"))
        for family in ("primary_root_traits", "root_hair_traits")
        for record in trait_payload.get(family, [])
        if isinstance(record, Mapping)
    ]
    _require(
        len(canonical_trait_fields) == len(set(canonical_trait_fields)) == 32
        and set(canonical_trait_fields).issubset(full_image.columns),
        "canonical image_traits table does not expose all 32 locked trait fields",
    )
    if final:
        public_identity = {
            "model_bundle_id": proposal.get("model_bundle_id"),
            "root_expert_id": proposal.get("root_expert", {}).get("expert_id"),
        }
        _require(
            set(full_image["model_bundle_id"].astype(str))
            == {str(public_identity["model_bundle_id"])}
            and set(full_image["root_expert_id"].astype(str))
            == {str(public_identity["root_expert_id"])},
            "canonical image_traits public model identities differ from proposal",
        )

    overlay_receipt = _read_object(named_paths["overlay_index"], "overlay_index")
    overlay_identity = _receipt_identity(overlay_receipt, "overlay_index")
    overlay_frame = _read_table(
        named_paths["overlay_selection"], "overlay_selection",
        (
            "task_id", "prediction_sha256", "raw_source_image_sha256", "case_id",
            "case_role", "source_path", "source_sha256", "overlay_path",
            "overlay_sha256", "case_selection_basis",
            "random_or_representative_performance_sample",
            "experimental_condition_metadata_used_for_rendering",
            "experimental_condition_metadata_used_for_evidence_assembly",
            "inset_required", "inset_rule", "inset_x0", "inset_y0",
            "inset_x1", "inset_y1", "inset_geometry_sha256",
        ),
    )
    _validate_overlay(
        receipt=overlay_receipt,
        selection_path=named_paths["overlay_selection"],
        selection=overlay_frame,
        prediction_sha=prediction_sha,
        full_traits=full,
    )
    try:
        overlay_audit = derive_overlay_audit(
            overlay_frame,
            full,
            topology_source,
            case_roles=CASE_ROLES,
        )
    except PublicationAuthorityError as error:
        raise FigureInputAssemblyError(str(error)) from error

    analysis_columns = (
        "cohort", "endpoint", "effect", "n", "estimate", "ci95_low",
        "ci95_high", "effect_scale", "causal_treatment_claim_allowed",
        "raw_effect_estimate", "raw_effect_ci95_low", "raw_effect_ci95_high",
        "raw_effect_estimand", "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates", "raw_effect_bootstrap_seed",
        "standardized_effect", "standardized_ci95_low", "standardized_ci95_high",
    )
    primary_analysis = _read_table(
        named_paths["analysis_primary_table"],
        "analysis_primary_table",
        analysis_columns,
    )
    sensitivity_analysis = _read_table(
        named_paths["analysis_sensitivity_table"],
        "analysis_sensitivity_table",
        analysis_columns,
    )
    analysis_hashes = core_payloads["analysis"].get("output_table_sha256", {})
    _require(
        analysis_hashes.get("primary_tests") == sha256_file(named_paths["analysis_primary_table"])
        and analysis_hashes.get("sensitivity_tests") == sha256_file(named_paths["analysis_sensitivity_table"]),
        "analysis receipt does not bind named primary/sensitivity tables",
    )
    wt_secondary_frames, wt_secondary_evidence = _validate_wt_secondary_resources(
        analysis_summary=core_payloads["analysis"],
        named_paths=named_paths,
    )
    phenotype_points = _derive_phenotype_points(clean)
    phenotype_effects = _derive_phenotype_effects(primary_analysis, sensitivity_analysis)
    try:
        narrative_decision = build_narrative_decision(
            phenotype_effects.to_dict("records"),
            source_sha256={
                "analysis_primary_table": sha256_file(named_paths["analysis_primary_table"]),
                "analysis_sensitivity_table": sha256_file(named_paths["analysis_sensitivity_table"]),
                "phenotype_effects": sha256_json(phenotype_effects.to_dict("records")),
            },
        )
        validate_narrative_decision(narrative_decision)
    except ValueError as error:
        raise FigureInputAssemblyError(str(error)) from error
    try:
        multitrait_atlas = build_multitrait_atlas(
            trait_contract=trait_payload,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary_analysis,
            sensitivity_analysis=sensitivity_analysis,
            source_sha256={
                "trait_contract": sha256_file(named_paths["trait_contract"]),
                "clean_traits": sha256_file(named_paths["clean_traits"]),
                "full_traits": sha256_file(named_paths["full_traits"]),
                "canonical_image_traits": sha256_file(
                    named_paths["full_image_traits"]
                ),
                "analysis_primary_table": sha256_file(
                    named_paths["analysis_primary_table"]
                ),
                "analysis_sensitivity_table": sha256_file(
                    named_paths["analysis_sensitivity_table"]
                ),
            },
        )
    except MultitraitAtlasError as error:
        raise FigureInputAssemblyError(
            f"multitrait atlas derivation failed: {error}"
        ) from error

    profile_analysis_receipt = _read_object(named_paths["profile_analysis"], "profile_analysis")
    sensitivity_profiles_receipt = _read_object(
        named_paths["sensitivity_profiles_summary"],
        "sensitivity_profiles_summary",
    )
    _require(profile_analysis_receipt.get("schema_version") == PROFILE_ANALYSIS_SCHEMA, "profile analysis schema changed")
    profile_analysis_identity = _receipt_identity(profile_analysis_receipt, "profile_analysis")
    sensitivity_profile_identity = _sealed(
        sensitivity_profiles_receipt,
        "export_identity_sha256",
        "sensitivity_profiles_summary",
    )
    _guard_red_lines("profile_analysis", profile_analysis_receipt)
    _guard_red_lines("sensitivity_profiles_summary", sensitivity_profiles_receipt)
    profile_public_identity = {
        "model_contract_proposal_sha256": proposal_sha,
        "model_contract_proposal_identity_sha256": proposal_identity,
        "model_bundle_id": proposal.get("model_bundle_id"),
        "root_expert_id": proposal.get("root_expert", {}).get("expert_id"),
    }
    primary_profile_identity = _identity_for_core(
        "profiles", core_payloads["profiles"], source_hashes["profiles"]
    )
    cohort_receipt = core_payloads["cohorts"]
    cohort_hashes = cohort_receipt.get("output_sha256")
    primary_profile_receipt = core_payloads["profiles"]
    primary_binding = primary_profile_receipt.get("cohort_binding")
    sensitivity_binding = sensitivity_profiles_receipt.get("cohort_binding")
    _require(
        isinstance(cohort_hashes, Mapping)
        and isinstance(primary_binding, Mapping)
        and isinstance(sensitivity_binding, Mapping)
        and primary_profile_receipt.get("tasks") == 261
        and sensitivity_profiles_receipt.get("tasks") == 283
        and primary_profile_receipt.get("rows") == 261 * 5
        and sensitivity_profiles_receipt.get("rows") == 283 * 5
        and primary_binding.get("schema_version")
        == "PHAxis-distal-axis-profile-cohort-binding-1.0.0"
        and sensitivity_binding.get("schema_version")
        == "PHAxis-distal-axis-profile-cohort-binding-1.0.0"
        and primary_binding.get("cohort_name") == "primary_clean261"
        and primary_binding.get("cohort_role") == "primary_SHA_disjoint"
        and primary_binding.get("cohort_tasks") == 261
        and sensitivity_binding.get("cohort_name") == "sensitivity_full283"
        and sensitivity_binding.get("cohort_role")
        == "overlap_contaminated_sensitivity"
        and sensitivity_binding.get("cohort_tasks") == 283
        and primary_binding.get("cohort_build_summary_sha256")
        == sensitivity_binding.get("cohort_build_summary_sha256")
        == sha256_file(core_paths["cohorts"])
        and primary_binding.get("cohort_build_identity_sha256")
        == sensitivity_binding.get("cohort_build_identity_sha256")
        == cohort_receipt.get("cohort_build_identity_sha256")
        and primary_binding.get("cohort_membership_csv_sha256")
        == sensitivity_binding.get("cohort_membership_csv_sha256")
        == cohort_hashes.get("cohort_membership")
        and primary_binding.get("cohort_task_membership_sha256")
        != sensitivity_binding.get("cohort_task_membership_sha256")
        and primary_profile_receipt.get("traits_csv_sha256")
        == cohort_hashes.get("primary_clean261", {}).get("traits")
        and primary_profile_receipt.get("hair_instances_csv_sha256")
        == cohort_hashes.get("primary_clean261", {}).get("hair_instances")
        and sensitivity_profiles_receipt.get("traits_csv_sha256")
        == cohort_hashes.get("sensitivity_full283", {}).get("traits")
        and sensitivity_profiles_receipt.get("hair_instances_csv_sha256")
        == cohort_hashes.get("sensitivity_full283", {}).get("hair_instances")
        and primary_profile_receipt.get("profiles_csv_sha256")
        != sensitivity_profiles_receipt.get("profiles_csv_sha256")
        and primary_profile_identity != sensitivity_profile_identity,
        "profile analysis clean261/full283 cohort provenance is not distinct and closed",
    )
    _require(
        profile_analysis_receipt.get("status") == "completed_exploratory_source_unit_profile_summaries"
        and profile_analysis_receipt.get("output_table_sha256") == sha256_file(named_paths["profile_analysis_table"])
        and profile_analysis_receipt.get("primary_profile_summary_sha256") == sha256_file(core_paths["profiles"])
        and profile_analysis_receipt.get("primary_profile_identity_sha256")
        == primary_profile_identity
        and profile_analysis_receipt.get("sensitivity_profile_summary_sha256")
        == sha256_file(named_paths["sensitivity_profiles_summary"])
        and profile_analysis_receipt.get("sensitivity_profile_identity_sha256")
        == sensitivity_profile_identity
        and all(
            profile_analysis_receipt.get(field) == value
            for field, value in profile_public_identity.items()
        )
        and all(
            core_payloads["profiles"].get(field) == value
            for field, value in profile_public_identity.items()
        )
        and all(
            sensitivity_profiles_receipt.get(field) == value
            for field, value in profile_public_identity.items()
        )
        and sensitivity_profiles_receipt.get("schema_version")
        == "PHAxis-distal-axis-profile-export-1.0.0"
        and sensitivity_profiles_receipt.get("status") == "completed",
        "profile analysis does not bind distinct core clean261/full283 profile exports",
    )
    profile_table = _read_table(
        named_paths["profile_analysis_table"], "profile_analysis_table",
        ("cohort", "cohort_role", "condition_code", "bin_start_um", "bin_end_um", "eligible_source_units", "length_measurable_source_units", "bootstrap_replicates_requested", "unit_of_analysis", "mean_attached_identity_count", "mean_attached_identity_count_ci95_low", "mean_attached_identity_count_ci95_high", "endpoint_complete_support_fraction", "endpoint_complete_support_fraction_ci95_low", "endpoint_complete_support_fraction_ci95_high", "median_of_source_unit_conditional_median_length_um", "median_of_source_unit_conditional_median_length_um_ci95_low", "median_of_source_unit_conditional_median_length_um_ci95_high"),
    )
    axial_profiles = _derive_profiles(profile_table)

    runtime_latency_payload = _read_object(named_paths["runtime_latency"], "runtime_latency")
    runtime_production_payload = _read_object(named_paths["runtime_production"], "runtime_production")
    runtime_latency_comparison_payload = _read_object(named_paths["runtime_latency_comparison"], "runtime_latency_comparison")
    runtime_production_comparison_payload = _read_object(named_paths["runtime_production_comparison"], "runtime_production_comparison")
    baseline_latency_payload = _read_object(named_paths["baseline_runtime_latency"], "baseline_runtime_latency")
    baseline_production_payload = _read_object(named_paths["baseline_runtime_production"], "baseline_runtime_production")
    runtime_frame = _read_table(named_paths["runtime_per_image"], "runtime_per_image", ("source_unit", "wall_seconds", "megapixels", "io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"))
    baseline_frame = _read_table(named_paths["baseline_runtime_per_image"], "baseline_runtime_per_image", ("source_unit", "wall_seconds", "megapixels", "io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"))
    if final:
        _require(runtime_frame["source_unit"].nunique() == len(runtime_frame) == 283, "PHAxis latency table is not full283")
        _require(baseline_frame["source_unit"].nunique() == len(baseline_frame) == 283, "v1 latency table is not full283")
        _require(set(runtime_frame["source_unit"]) == set(baseline_frame["source_unit"]), "runtime systems used different source units")
    normalized_runtime = _validate_runtime_bundle(
        current_latency_path=named_paths["runtime_latency"],
        current_latency=runtime_latency_payload,
        current_per_image_path=named_paths["runtime_per_image"],
        current_production_path=named_paths["runtime_production"],
        current_production=runtime_production_payload,
        baseline_latency_path=named_paths["baseline_runtime_latency"],
        baseline_latency=baseline_latency_payload,
        baseline_per_image_path=named_paths["baseline_runtime_per_image"],
        baseline_production_path=named_paths["baseline_runtime_production"],
        baseline_production=baseline_production_payload,
        latency_comparison=runtime_latency_comparison_payload,
        production_comparison=runtime_production_comparison_payload,
        final=final,
    )

    cohort_flow = _derive_cohort_flow(core_payloads["train399_evaluation"], core_payloads["cohorts"])
    workflow = _derive_workflow(core_payloads, source_hashes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _make_staging_directory(destination)
    try:
        resource_root = staging / "resources"
        source_root = staging / "source_inputs"
        provenance_root = staging / "provenance"
        resource_root.mkdir(parents=True)
        source_root.mkdir(parents=True)
        provenance_root.mkdir(parents=True)

        resource_paths: dict[str, Path] = {}
        for role, source in (
            ("trait_contract", named_paths["trait_contract"]),
            ("figure1_image", named_paths["figure1_image"]),
            ("figure1_geometry", named_paths["figure1_geometry"]),
        ):
            target = resource_root / f"{role}{source.suffix.lower()}"
            _copy_bytes(source, target)
            resource_paths[role] = target
        for role, frame in (
            ("development_per_image", qc_per_image),
            ("development_tolerance", qc_tolerance),
            ("development_threshold", qc_threshold),
            ("development_strata", historical_strata),
            ("assurance_metrics", assurance_metrics_frame),
            ("assurance_pairs", assurance_pairs_frame),
            ("assurance_support", assurance_support_frame),
            ("overlay_audit", overlay_audit),
            ("phenotype_points", phenotype_points),
            ("phenotype_effects", phenotype_effects),
            ("axial_profiles", axial_profiles),
            ("cohort_flow", cohort_flow),
            ("workflow_stages", workflow),
            ("runtime_per_image", runtime_frame),
        ):
            target = resource_root / f"{role}.csv"
            _write_csv(target, frame)
            resource_paths[role] = target
        for role in WT_SECONDARY_RESOURCE_ROLES:
            target = resource_root / f"{role}.csv"
            _copy_bytes(named_paths[role], target)
            resource_paths[role] = target

        multitrait_target = resource_root / "multitrait_atlas.json"
        atomic_write_json(multitrait_target, multitrait_atlas)
        resource_paths["multitrait_atlas"] = multitrait_target

        assignment_target = resource_root / "qcdev_assignment.json"
        atomic_write_json(assignment_target, qcdev_assignment)
        resource_paths["qcdev_assignment"] = assignment_target

        decision_target = resource_root / "narrative_decision.json"
        atomic_write_json(decision_target, narrative_decision)
        resource_paths["narrative_decision"] = decision_target

        # Copy review cases into the bundle and rewrite only their paths; all
        # numeric/display/semantic cells remain byte-derived from the sealed
        # selection index.
        overlay_normalized = overlay_frame.copy()
        for index, row in overlay_normalized.iterrows():
            for prefix in ("source", "overlay"):
                raw = Path(str(row[f"{prefix}_path"]))
                if not raw.is_absolute():
                    raw = named_paths["overlay_selection"].parent / raw
                source = _resolve_file(raw, f"overlay_{prefix}_{row['case_id']}", final=final)
                _require(sha256_file(source) == row[f"{prefix}_sha256"], f"{row['case_id']}: {prefix} file hash mismatch")
                suffix = source.suffix.lower() or ".png"
                target = resource_root / "overlay_cases" / f"{row['case_id']}_{prefix}{suffix}"
                _copy_bytes(source, target)
                overlay_normalized.at[index, f"{prefix}_path"] = str(target.relative_to(resource_root)).replace("\\", "/")
        overlay_target = resource_root / "overlay_selection.csv"
        _write_csv(overlay_target, overlay_normalized)
        resource_paths["overlay_selection"] = overlay_target

        runtime_target = resource_root / "runtime_summary.json"
        atomic_write_json(runtime_target, normalized_runtime)
        resource_paths["runtime_summary"] = runtime_target
        _require(set(resource_paths) == set(RESOURCE_ROLES), "internal resource route is incomplete")

        source_records: dict[str, dict[str, str]] = {}
        _require(
            set(FIGURE_SOURCE_INPUT_ROLES) <= set(named_paths),
            "internal supplementary source-input route is incomplete",
        )
        for role in FIGURE_SOURCE_INPUT_ROLES:
            source = named_paths[role]
            target = source_root / f"{role}{source.suffix.lower()}"
            _copy_bytes(source, target)
            source_records[role] = {
                "path": str(target.relative_to(staging)).replace("\\", "/"),
                "sha256": sha256_file(target),
            }

        provenance_payloads = {
            "historical_development": (historical_receipt, historical_identity),
            "measurement_assurance": (assurance_receipt, assurance_identity),
            "overlay_index": (overlay_receipt, overlay_identity),
            "profile_analysis": (profile_analysis_receipt, profile_analysis_identity),
            "runtime_latency": (runtime_latency_payload, runtime_latency_payload["summary_identity_sha256"]),
            "runtime_production": (runtime_production_payload, runtime_production_payload["summary_identity_sha256"]),
            "runtime_latency_comparison": (runtime_latency_comparison_payload, runtime_latency_comparison_payload["comparison_identity_sha256"]),
            "runtime_production_comparison": (runtime_production_comparison_payload, runtime_production_comparison_payload["comparison_identity_sha256"]),
            "baseline_runtime_latency": (baseline_latency_payload, baseline_latency_payload["summary_identity_sha256"]),
            "baseline_runtime_production": (baseline_production_payload, baseline_production_payload["summary_identity_sha256"]),
        }
        provenance_records: dict[str, dict[str, str]] = {}
        for role, (_payload, identity) in provenance_payloads.items():
            source = named_paths[role]
            target = provenance_root / f"{role}.json"
            _copy_bytes(source, target)
            provenance_records[role] = {
                "path": str(target.relative_to(staging)).replace("\\", "/"),
                "sha256": sha256_file(target),
                "identity_field": IDENTITY_FIELDS[role],
                "identity_sha256": str(identity),
            }

        resources = {
            role: {
                "path": str(path.relative_to(staging)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for role, path in resource_paths.items()
        }
        lineage = {
            "trait_contract": ["trait_contract", "traits", "full_image_traits"],
            "figure1_image": ["figure1_image", "full_traits", "fusion"],
            "figure1_geometry": ["figure1_geometry", "full_traits", "fusion"],
            "development_per_image": ["train399_evaluation", "split_manifest"],
            "development_tolerance": ["train399_evaluation"],
            "development_threshold": ["train399_selection"],
            "development_strata": ["historical_development", "historical_oof_per_image"],
            "assurance_metrics": ["measurement_assurance", "assurance_metrics", "assurance_topology"],
            "assurance_pairs": ["measurement_assurance", "assurance_pairs"],
            "assurance_support": ["measurement_assurance", "assurance_support"],
            "qcdev_assignment": [
                "measurement_assurance",
                "hair_attachment_assurance_input",
                "train399_evaluation",
            ],
            "overlay_selection": ["overlay_index", "fusion", "full_traits"],
            "overlay_audit": [
                "overlay_index",
                "full_traits",
                "assurance_topology",
            ],
            "phenotype_points": ["cohorts", "clean_traits"],
            "phenotype_effects": ["analysis", "analysis_primary_table", "analysis_sensitivity_table"],
            "narrative_decision": [
                "analysis",
                "analysis_primary_table",
                "analysis_sensitivity_table",
                "phenotype_effects",
            ],
            "multitrait_atlas": [
                "trait_contract",
                "cohorts",
                "analysis",
                "clean_traits",
                "full_traits",
                "analysis_primary_table",
                "analysis_sensitivity_table",
            ],
            "axial_profiles": [
                "profiles",
                "profile_analysis",
                "profile_analysis_table",
                "sensitivity_profiles_summary",
            ],
            "cohort_flow": ["train399_evaluation", "cohorts"],
            "workflow_stages": list(CORE_ROLES),
            "runtime_summary": ["runtime_latency", "runtime_production", "runtime_latency_comparison", "runtime_production_comparison", "baseline_runtime_latency", "baseline_runtime_production"],
            "runtime_per_image": ["runtime_latency"],
            "wt_within_experiment_contrasts": [
                "analysis",
                "wt_within_experiment_contrasts",
            ],
            "wt_within_day_meta_analysis": [
                "analysis",
                "wt_within_experiment_contrasts",
                "wt_within_day_meta_analysis",
            ],
            "wt_temperature_qc_flow": [
                "analysis",
                "wt_within_experiment_contrasts",
                "wt_temperature_qc_flow",
            ],
        }
        supplementary_contract = supplementary_figure_contract()
        supplementary_contract["contract_identity_sha256"] = sha256_json(
            supplementary_contract
        )
        manifest: dict[str, Any] = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "assembler_schema_version": ASSEMBLER_SCHEMA_VERSION,
            "status": "final" if final else "provisional",
            "source_summary_sha256": source_hashes,
            "model_contract_proposal_sha256": proposal_sha,
            "model_contract_proposal_identity_sha256": proposal_identity,
            "model_contract_public_identity": {
                "model_bundle_id": proposal.get("model_bundle_id"),
                "root_expert_id": proposal.get("root_expert", {}).get(
                    "expert_id"
                ),
                "root_provider_role": proposal.get("root_expert", {}).get(
                    "provider_role"
                ),
            },
            "model_bundle_id": proposal.get("model_bundle_id"),
            "root_expert_id": proposal.get("root_expert", {}).get(
                "expert_id"
            ),
            "hair_identity_expert_id": core_payloads["stageb"].get(
                "detection_model_metadata", {}
            ).get("expert_id"),
            "train399_selection_sha256": sha256_file(named_paths["train399_selection"]),
            "train399_selection_identity_sha256": selection["selection_receipt_identity_sha256"],
            "train399_prediction_input_provenance": {
                "task_order_identity_sha256": sha256_json(
                    prediction_inputs["task_order"]
                ),
                "stageb_train399": {
                    "schema_version": prediction_inputs[
                        "stageb_detection_files_schema_version"
                    ],
                    "artifact_role": prediction_inputs[
                        "stageb_evaluation_inference_authority"
                    ]["artifact_role"],
                    "evaluation_inference_summary_sha256": prediction_inputs[
                        "stageb_evaluation_inference_authority"
                    ]["evaluation_inference_summary_sha256"],
                    "evaluation_inference_summary_identity_sha256": prediction_inputs[
                        "stageb_evaluation_inference_authority"
                    ]["evaluation_inference_summary_identity_sha256"],
                    "evaluation_gate_identity_sha256": prediction_inputs[
                        "stageb_evaluation_inference_authority"
                    ]["evaluation_gate_identity_sha256"],
                    "production_consumption_allowed": False,
                    "fusion_consumption_allowed": False,
                    "traits_consumption_allowed": False,
                    "ordered_file_set_identity_sha256": prediction_inputs[
                        "stageb_detection_set_identity_sha256"
                    ],
                },
                "legacy_hybrid_endpoint_complete_identity_layer": {
                    **prediction_inputs["legacy_hybrid_comparator_contract"],
                    "ordered_file_set_identity_sha256": prediction_inputs[
                        "hybrid_prediction_set_identity_sha256"
                    ],
                },
            },
            "resources": resources,
            "resource_lineage": lineage,
            "source_inputs": source_records,
            "provenance_receipts": provenance_records,
            "supplementary_figure_contract": supplementary_contract,
            "uncertainty_contracts": {
                "qcdevelopment44": qc_uncertainty,
                "historical_oof443": historical_receipt["uncertainty"],
                "axial_profiles": {
                    "method": "source-unit percentile bootstrap",
                    "repetitions": 10000,
                },
                "wt_temperature_secondary": {
                    "within_experiment_estimand": "30C_over_22C_ratio",
                    "same_day_meta_model": (
                        "random_effects_REML_with_Hartung_Knapp_interval"
                    ),
                    "multiplicity": "BH_FDR_within_clean_or_full_WT_family",
                    "minimum_experiments_per_day_meta_analysis": 3,
                    "unknown_day_is_descriptive_only": True,
                },
            },
            "wt_secondary_evidence": wt_secondary_evidence,
            "independent_accuracy_claim_allowed": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
            "narrative_decision_identity_sha256": narrative_decision[
                "narrative_decision_identity_sha256"
            ],
            "narrative_branch_id": narrative_decision["branch_id"],
            "qcdev_assignment_identity_sha256": qcdev_assignment[
                "assignment_identity_sha256"
            ],
        }
        manifest["figure_input_assembly_identity_sha256"] = sha256_json(manifest)
        manifest_path = staging / "figure_inputs.json"
        atomic_write_json(manifest_path, manifest)
        assembly_summary = {
            "schema_version": ASSEMBLER_SCHEMA_VERSION,
            "status": "completed_final" if final else "completed_provisional",
            "figure_inputs_sha256": sha256_file(manifest_path),
            "figure_input_assembly_identity_sha256": manifest["figure_input_assembly_identity_sha256"],
            "resource_sha256": {role: resources[role]["sha256"] for role in RESOURCE_ROLES},
            "source_summary_sha256": source_hashes,
            "model_contract_public_identity": manifest[
                "model_contract_public_identity"
            ],
            "model_bundle_id": manifest["model_bundle_id"],
            "root_expert_id": manifest["root_expert_id"],
            "hair_identity_expert_id": manifest[
                "hair_identity_expert_id"
            ],
            "measurement_assurance_receipt_sha256": provenance_records["measurement_assurance"]["sha256"],
            "measurement_assurance_identity_sha256": assurance_identity,
            "supplementary_figure_contract_identity_sha256": supplementary_contract[
                "contract_identity_sha256"
            ],
            "train399_prediction_input_provenance": manifest[
                "train399_prediction_input_provenance"
            ],
            "wt_secondary_evidence": wt_secondary_evidence,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
            "narrative_decision_identity_sha256": narrative_decision[
                "narrative_decision_identity_sha256"
            ],
            "narrative_branch_id": narrative_decision["branch_id"],
            "qcdev_assignment_identity_sha256": qcdev_assignment[
                "assignment_identity_sha256"
            ],
        }
        atomic_write_json(staging / "assembly_summary.json", assembly_summary)
        _require(not destination.exists(), f"output appeared during assembly: {destination}")
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("final", "provisional"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-contract-proposal", required=True)
    parser.add_argument("--train399-candidate", required=True)
    parser.add_argument("--train399-selection", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--trait-contract", required=True)
    parser.add_argument("--image-traits-schema", required=True)
    parser.add_argument("--figure1-image", required=True)
    parser.add_argument("--figure1-geometry", required=True)
    parser.add_argument("--historical-development-receipt", required=True)
    parser.add_argument("--historical-oof-per-image", required=True)
    parser.add_argument("--measurement-assurance-receipt", required=True)
    parser.add_argument("--assurance-metrics", required=True)
    parser.add_argument("--assurance-pairs", required=True)
    parser.add_argument("--assurance-support", required=True)
    parser.add_argument("--assurance-topology", required=True)
    parser.add_argument("--overlay-index-receipt", required=True)
    parser.add_argument("--overlay-selection", required=True)
    parser.add_argument("--clean-traits", required=True)
    parser.add_argument("--full-traits", required=True)
    parser.add_argument("--full-image-traits", required=True)
    parser.add_argument("--analysis-primary-table", required=True)
    parser.add_argument("--analysis-sensitivity-table", required=True)
    parser.add_argument("--profile-analysis-summary", required=True)
    parser.add_argument("--profile-analysis-table", required=True)
    parser.add_argument("--sensitivity-profiles-summary", required=True)
    parser.add_argument("--runtime-latency-summary", required=True)
    parser.add_argument("--runtime-per-image", required=True)
    parser.add_argument("--runtime-production-summary", required=True)
    parser.add_argument("--runtime-latency-comparison", required=True)
    parser.add_argument("--runtime-production-comparison", required=True)
    parser.add_argument("--baseline-runtime-latency-summary", required=True)
    parser.add_argument("--baseline-runtime-per-image", required=True)
    parser.add_argument("--baseline-runtime-production-summary", required=True)
    parser.add_argument("--benchmark-same-hardware", required=True)
    parser.add_argument("--benchmark-artifact-inventory", required=True)
    parser.add_argument(
        "--training-receipt",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="exact fixed-seed completion receipt; repeat five times",
    )
    for role in CORE_ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}", dest=role, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt_paths = {role: getattr(args, role) for role in CORE_ROLES}
    training_receipts: dict[int, str] = {}
    for raw in args.training_receipt:
        seed_text, separator, path = raw.partition("=")
        _require(bool(separator) and bool(path), "--training-receipt must be SEED=PATH")
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise FigureInputAssemblyError("training receipt seed must be an integer") from error
        _require(seed not in training_receipts, f"duplicate training receipt seed: {seed}")
        training_receipts[seed] = path
    manifest = build_publication_figure_inputs(
        mode=args.mode,
        output=args.output,
        model_contract_proposal=args.model_contract_proposal,
        train399_candidate=args.train399_candidate,
        train399_selection=args.train399_selection,
        dataset_manifest=args.dataset_manifest,
        split_manifest=args.split_manifest,
        trait_contract=args.trait_contract,
        image_traits_schema=args.image_traits_schema,
        figure1_image=args.figure1_image,
        figure1_geometry=args.figure1_geometry,
        historical_development_receipt=args.historical_development_receipt,
        historical_oof_per_image=args.historical_oof_per_image,
        measurement_assurance_receipt=args.measurement_assurance_receipt,
        assurance_metrics=args.assurance_metrics,
        assurance_pairs=args.assurance_pairs,
        assurance_support=args.assurance_support,
        assurance_topology=args.assurance_topology,
        overlay_index_receipt=args.overlay_index_receipt,
        overlay_selection=args.overlay_selection,
        clean_traits=args.clean_traits,
        full_traits=args.full_traits,
        full_image_traits=args.full_image_traits,
        analysis_primary_table=args.analysis_primary_table,
        analysis_sensitivity_table=args.analysis_sensitivity_table,
        profile_analysis_summary=args.profile_analysis_summary,
        profile_analysis_table=args.profile_analysis_table,
        sensitivity_profiles_summary=args.sensitivity_profiles_summary,
        runtime_latency_summary=args.runtime_latency_summary,
        runtime_per_image=args.runtime_per_image,
        runtime_production_summary=args.runtime_production_summary,
        runtime_latency_comparison=args.runtime_latency_comparison,
        runtime_production_comparison=args.runtime_production_comparison,
        baseline_runtime_latency_summary=args.baseline_runtime_latency_summary,
        baseline_runtime_per_image=args.baseline_runtime_per_image,
        baseline_runtime_production_summary=args.baseline_runtime_production_summary,
        benchmark_same_hardware=args.benchmark_same_hardware,
        benchmark_artifact_inventory=args.benchmark_artifact_inventory,
        training_receipts=training_receipts,
        receipt_paths=receipt_paths,
    )
    print(manifest["figure_input_assembly_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
