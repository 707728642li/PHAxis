"""Structured evidence authorities materialized at PHAxis stage 36."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .evaluation_metrics import (
    biological_hair_presence_matcher_contract,
    match_biological_hair_presence,
    precision_recall_f1,
)
from .io import read_json, sha256_file, sha256_json


class PublicationAuthorityError(ValueError):
    """A structured publication artifact cannot be derived safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationAuthorityError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    _require(normalized in {"true", "1", "yes", "false", "0", "no"}, f"{label} is not boolean")
    return normalized in {"true", "1", "yes"}


def build_qcdev_assignment(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    """Recompute per-instance assignment from the sealed assurance input."""

    components = receipt.get("component_receipts")
    _require(isinstance(components, Mapping), "measurement assurance component receipts missing")
    component = components.get("hair_attachment")
    _require(isinstance(component, Mapping), "hair-attachment component receipt missing")
    relative = component.get("input_contract_audit_copy")
    _require(isinstance(relative, str) and relative, "hair-attachment geometry input path missing")
    input_path = (receipt_path.parent / relative).resolve()
    _require(input_path.is_file() and not input_path.is_symlink(), "hair-attachment geometry input missing")
    _require("blind" not in str(input_path).casefold(), "blind-labelled assignment input refused")
    _require(sha256_file(input_path) == component.get("input_contract_audit_copy_sha256"), "hair-attachment geometry input SHA mismatch")
    source = read_json(input_path)
    _require(source.get("schema_version") == "PHAxis-hair-attachment-assurance-input-1.0", "hair-attachment input schema changed")
    _require(source.get("blind_images_used") == 0, "blind images entered QC-development assignment")
    unsigned_source = deepcopy(source)
    input_identity = unsigned_source.pop("input_contract_identity_sha256", None)
    _require(
        input_identity == component.get("input_contract_identity_sha256")
        and _is_sha256(input_identity)
        and sha256_json(unsigned_source) == input_identity,
        "hair-attachment geometry input identity mismatch",
    )
    contract = biological_hair_presence_matcher_contract()
    declared = source.get("metric_config", {}).get("formal_matcher")
    _require(isinstance(declared, Mapping), "formal matcher declaration missing")
    expected_matcher = {
        key: contract[key]
        for key in (
            "curve_tolerance_um",
            "minimum_truth_coverage",
            "minimum_prediction_coverage",
            "minimum_direction_cosine",
            "proximal_arc_fraction",
            "resample_points",
        )
    }
    _require(
        dict(declared) == expected_matcher,
        "QC-development matcher differs from evaluator",
    )
    locks = receipt.get("qcdev_stageb_biological_presence_20um_crosscheck_locks")
    _require(isinstance(locks, list), "stage7 TP closure locks missing")
    lock_by_task = {str(row.get("task_id")): row for row in locks if isinstance(row, Mapping)}
    records = source.get("records")
    _require(isinstance(records, list) and records, "hair-attachment geometry records missing")
    assignments: list[dict[str, Any]] = []
    for record in records:
        _require(isinstance(record, Mapping), "hair-attachment geometry record invalid")
        task_id = str(record.get("source_unit"))
        _require(task_id in lock_by_task, f"{task_id}: no stage7 TP closure lock")
        predicted = record.get("predicted_polylines_xy_um")
        annotated = record.get("annotated_polylines_xy_um")
        _require(isinstance(predicted, list) and isinstance(annotated, list), f"{task_id}: assignment geometry missing")
        metrics, matches = match_biological_hair_presence(
            predicted,
            annotated,
            units_per_coordinate=1.0,
            tolerance=float(contract["curve_tolerance_um"]),
            minimum_truth_coverage=float(contract["minimum_truth_coverage"]),
            minimum_prediction_coverage=float(contract["minimum_prediction_coverage"]),
            minimum_direction_cosine=float(contract["minimum_direction_cosine"]),
            proximal_arc_fraction=float(contract["proximal_arc_fraction"]),
            resample_points=int(contract["resample_points"]),
        )
        lock = lock_by_task[task_id]
        _require(
            int(metrics["n_pred"]) == int(lock.get("n_pred", -1))
            and int(metrics["n_gt"]) == int(lock.get("n_gt", -1))
            and int(metrics["tp"]) == int(lock.get("biological_presence_tp_20um", -1)),
            f"{task_id}: recomputed assignment does not close to stage7",
        )
        matched_pred = {int(match["predicted_index"]) for match in matches}
        matched_gt = {int(match["annotated_index"]) for match in matches}
        assignments.append(
            {
                "source_unit": task_id,
                "source_image_sha256": str(record.get("source_image_sha256")),
                "prediction_artifact_sha256": str(record.get("prediction_artifact_sha256")),
                "annotation_artifact_sha256": str(record.get("annotation_artifact_sha256")),
                "coordinate_space": "physical_um_xy",
                "predicted_polylines_xy_um": predicted,
                "annotated_polylines_xy_um": annotated,
                "matches": matches,
                "unmatched_prediction_indices": sorted(set(range(len(predicted))) - matched_pred),
                "unmatched_truth_indices": sorted(set(range(len(annotated))) - matched_gt),
                "metrics": metrics,
            }
        )
    _require(set(lock_by_task) == {row["source_unit"] for row in assignments}, "assignment task set differs from stage7")
    pooled_counts = {
        "tp": sum(int(row["metrics"]["tp"]) for row in assignments),
        "n_pred": sum(int(row["metrics"]["n_pred"]) for row in assignments),
        "n_gt": sum(int(row["metrics"]["n_gt"]) for row in assignments),
    }
    declared_pooled = receipt.get("hair_attachment_assurance", {}).get("summary", {}).get("formal_matched_attachment_accuracy", {}).get("formal_biological_presence")
    _require(
        isinstance(declared_pooled, Mapping)
        and all(int(declared_pooled.get(key, -1)) == value for key, value in pooled_counts.items()),
        "recomputed assignment does not close to pooled stage7 authority",
    )
    display = sorted(
        assignments,
        key=lambda row: (
            not bool(row["matches"]),
            not bool(row["unmatched_prediction_indices"] or row["unmatched_truth_indices"]),
            -len(row["matches"]),
            row["source_unit"],
        ),
    )[0]
    payload: dict[str, Any] = {
        "schema_version": "PHAxis-qcdev-instance-assignment-1.0",
        "status": "completed_recomputed_from_sealed_geometry",
        "evidence_role": "selected_qc_development_non_independent",
        "matcher_contract": contract,
        "matcher_contract_sha256": sha256_json(contract),
        "source_input_sha256": sha256_file(input_path),
        "source_input_identity_sha256": input_identity,
        "stage7_lock_set_identity_sha256": sha256_json(locks),
        "display_source_unit": display["source_unit"],
        "pooled": precision_recall_f1(pooled_counts["tp"], pooled_counts["n_pred"], pooled_counts["n_gt"]),
        "assignments": assignments,
        "blind_images_used": 0,
        "independent_accuracy_claim_allowed": False,
    }
    payload["assignment_identity_sha256"] = sha256_json(payload)
    return payload


def derive_overlay_audit(
    selection: pd.DataFrame,
    full_traits: pd.DataFrame,
    topology: pd.DataFrame,
    *,
    case_roles: tuple[str, ...],
) -> pd.DataFrame:
    """Derive the Fig. 4 audit-2.0 card from sealed traits and topology."""

    trait_by_task = {str(row["task_id"]): row for row in full_traits.to_dict("records")}
    topology_by_task = {str(row["source_unit"]): row for row in topology.to_dict("records")}
    rows: list[dict[str, Any]] = []
    for case in selection.to_dict("records"):
        task_id = str(case["task_id"])
        _require(task_id in trait_by_task, f"{task_id}: Fig.4 trait row missing")
        trait = trait_by_task[task_id]
        formal = _bool(trait.get("formal_statistics_eligible"), f"{task_id}.formal_statistics_eligible")
        case_formal = _bool(case.get("formal_statistics_eligible", formal), f"{task_id}.overlay formal state")
        _require(formal == case_formal, f"{task_id}: overlay/trait formal state differs")
        topology_row = topology_by_task.get(task_id)
        if formal:
            _require(topology_row is not None, f"{task_id}: formal Fig.4 case lacks topology audit")
            axis_in_root = float(topology_row["axis_in_root_coverage_fraction"])
            single_component = float(
                topology_row["axis_single_component_coverage_fraction"]
            )
            longest_gap_um = float(
                topology_row["longest_unsupported_axis_gap_um"]
            )
            _require(
                math.isfinite(axis_in_root)
                and math.isfinite(single_component)
                and math.isfinite(longest_gap_um)
                and 0.0 <= single_component <= axis_in_root <= 1.0
                and longest_gap_um >= 0.0,
                f"{task_id}: invalid union/single-component axis support",
            )
            _require(
                math.isclose(
                    axis_in_root,
                    float(topology_row["axis_containment_fraction"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{task_id}: axis-in-root/legacy containment drift",
            )
            identity_count: int | None = int(topology_row["identity_hair_n"])
            _require(
                identity_count == int(trait["hair_count"]) and identity_count >= 0,
                f"{task_id}: topology/trait formal identity count differs",
            )
            support_count: int | None = int(
                trait["hair_length_measurement_hair_count"]
            )
            _require(
                0 <= support_count <= identity_count,
                f"{task_id}: impossible endpoint-complete support count",
            )
            support: float | None = (
                support_count / identity_count if identity_count else None
            )
            reported_support = trait.get("hair_length_measurement_fraction")
            if support is None:
                _require(
                    reported_support is None or pd.isna(reported_support),
                    f"{task_id}: zero identity denominator requires null endpoint support",
                )
            else:
                _require(
                    reported_support is not None
                    and not pd.isna(reported_support)
                    and math.isclose(
                        support,
                        float(reported_support),
                        rel_tol=0.0,
                        abs_tol=5e-7,
                    ),
                    f"{task_id}: endpoint support fraction differs from count denominator",
                )
            root_length_um = float(trait["visible_root_axis_length_um"])
            _require(
                math.isfinite(root_length_um) and root_length_um >= 0.0,
                f"{task_id}: invalid visible-axis extent",
            )
            window_eligible = _bool(
                trait["distal_window_1_4mm_eligible"],
                f"{task_id}.distal_window_1_4mm_eligible",
            )
            _require(
                window_eligible == (root_length_um >= 4000.0),
                f"{task_id}: locked [1,4) mm eligibility drift",
            )
            profile_eligible = root_length_um >= 5000.0
            window_reason = (
                "eligible_visible_axis_reaches_4mm"
                if window_eligible
                else "visible_root_axis_shorter_than_4mm"
            )
            profile_reason = (
                "eligible_visible_axis_reaches_5mm"
                if profile_eligible
                else "visible_root_axis_shorter_than_5mm"
            )
            reason = "formal_statistics_eligible"
        else:
            _require(topology_row is None, f"{task_id}: review-only Fig.4 case entered formal topology")
            axis_in_root = None
            single_component = None
            longest_gap_um = None
            support = None
            support_count = None
            identity_count = None
            reason = str(trait.get("exclusion_reason") or "review_only_fail_closed")
            window_eligible = False
            profile_eligible = False
            window_reason = f"formal_statistics_ineligible:{reason}"
            profile_reason = f"formal_statistics_ineligible:{reason}"
        rows.append(
            {
                "schema_version": "PHAxis-Fig4-case-audit-2.0",
                "case_id": str(case["case_id"]),
                "case_role": str(case["case_role"]),
                "task_id": task_id,
                "source_image_sha256": str(trait["source_image_sha256"]),
                "prediction_sha256": str(case["prediction_sha256"]),
                "formal_state": "formal" if formal else "review_only",
                "axis_in_root_coverage_fraction": axis_in_root,
                "axis_single_component_coverage_fraction": single_component,
                "longest_unsupported_axis_gap_um": longest_gap_um,
                "formal_identity_count": identity_count,
                "endpoint_complete_support_count": support_count,
                "endpoint_complete_support_fraction": support,
                "distal_window_1_4mm_eligible": window_eligible,
                "distal_window_1_4mm_reason": window_reason,
                "profile_0_5mm_eligible": profile_eligible,
                "profile_0_5mm_reason": profile_reason,
                "downstream_eligible": formal,
                "downstream_reason": reason,
                "condition_metadata_used": False,
            }
        )
    result = pd.DataFrame(rows)
    _require(list(result["case_role"]) == list(case_roles), "Fig.4 audit role order changed")
    return result


__all__ = ["PublicationAuthorityError", "build_qcdev_assignment", "derive_overlay_audit"]
