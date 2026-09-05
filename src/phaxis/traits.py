"""Professional PHAxis trait export with explicit cross-expert semantics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    PRODUCT_VERSION,
    PUBLIC_HAIR_LENGTH_EXPERT_ID,
    PUBLIC_HAIR_LENGTH_SEMANTICS,
)
from .contracts import ContractError, validate_hybrid_prediction
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .public_identity import MODEL_BUNDLE_PREFIX, ROOT_EXPERT_PREFIX


ROOT_TRAIT_FIELDS = (
    "visible_root_axis_length_um",
    "root_axis_chord_um",
    "root_centerline_chord_tortuosity",
    "root_straightness",
    "root_projected_area_um2",
    "root_projected_area_um2_per_root_mm",
    "median_root_width_um",
    "root_width_p10_um",
    "root_width_q25_um",
    "root_width_q75_um",
    "root_width_p90_um",
    "root_width_cv",
    "root_width_tip_third_median_um",
    "root_width_middle_third_median_um",
    "root_width_shootward_third_median_um",
    "root_width_shootward_to_tip_ratio",
    "root_width_axial_slope_um_per_mm",
    "root_centerline_curvature_median_rad_per_mm",
    "root_centerline_curvature_p95_rad_per_mm",
)

HAIR_TRAIT_FIELDS = (
    "hair_count",
    "mean_hair_length_um",
    "median_hair_length_um",
    "total_hair_length_um",
    "hair_density_per_mm_visible_root",
    "first_hair_distance_from_distal_point_um",
    "first_hair_ge40um_distance_from_distal_point_um",
    "local_hair_count_1_4mm",
    "local_hair_density_per_mm_1_4mm",
    "local_mean_hair_length_um_1_4mm",
    "local_median_hair_length_um_1_4mm",
    "local_total_hair_length_um_per_root_mm_1_4mm",
    "visible_hair_attachment_span_um_descriptive_right_censored",
)

COMMON_FIELDS = (
    "task_id",
    "source_image_sha256",
    "experiment_key",
    "condition_code",
    "study_role",
    "developmental_day",
    "genotype_or_construct",
    "temperature_c",
    "qc_disposition",
    "formal_statistics_eligible",
    "automatic_formal_phenotype_eligible",
    "root_axis_source",
    "root_global_width_source",
    "root_global_width_reference_applied",
)

IMAGE_TRAIT_FIELDS = (
    "schema_version",
    "product_version",
    "model_bundle_id",
    "root_expert_id",
    "hair_identity_count_expert_id",
    "hair_length_expert_id",
    "task_id",
    "source_image_sha256",
    "prediction_sha256",
    "root_lock_sha256",
    "stageb_detection_identity_sha256",
    "measurement_tier",
    "formal_statistics_eligible",
    "automatic_measurement_fail_closed",
    "exclusion_reason",
    "scale_status",
    "physical_units_valid",
    "um_per_px",
    "scale_value_um",
    "scale_length_px",
    "root_cap_region_output",
    "root_cap_area_used",
    "root_cap_point_x_px",
    "root_cap_point_y_px",
    "root_cap_point_border_distance_px",
    "root_cap_point_border_visible",
    "shootward_endpoint_x_px",
    "shootward_endpoint_y_px",
    "shootward_endpoint_border_distance_px",
    "shootward_endpoint_border_visible",
    "root_orientation_deg",
    *ROOT_TRAIT_FIELDS,
    *HAIR_TRAIT_FIELDS,
    "hair_length_measurement_hair_count",
    "hair_length_measurement_fraction",
    "hair_length_semantics",
    "total_hair_length_is_partial",
    "distal_window_1_4mm_eligible",
    "attachment_axis_valid_fraction",
    "whole_hair_zone_confirmatory_allowed",
    "root_cap_point_to_axis_bridge_um",
    "axis_endpoint_count",
    "axis_reference_radius_px",
    "width_samples",
    "width_end_exclusion_um",
    "root_axis_source",
    "root_global_width_source",
    "root_continuity_status",
    "root_continuity_applied",
    "condition_metadata_used_for_routing",
    "canonical_annotations_read_during_inference",
    "blind_images_used",
)


def _hair_identity_expert(prediction: Mapping[str, Any]) -> str:
    phaxis = prediction.get("phaxis")
    expert = phaxis.get("hair_identity_count_expert") if isinstance(phaxis, Mapping) else None
    if not isinstance(expert, str) or not expert.strip():
        raise ContractError(
            f"{prediction.get('task_id', '<unknown>')}: hair expert identity is absent"
        )
    return expert.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise RuntimeError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            writer.writerow(
                {
                    field: "" if raw.get(field) is None else raw.get(field, "")
                    for field in fields
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _polyline_length_um(hair: Mapping[str, Any], *, um_per_px: float) -> float:
    points = np.asarray(hair.get("points_xy"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ContractError("invalid endpoint-complete hair polyline")
    if not np.all(np.isfinite(points)):
        raise ContractError("non-finite endpoint-complete hair polyline")
    length_um = float(
        np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * um_per_px
    )
    if not np.isfinite(length_um) or length_um <= 0.0:
        raise ContractError("endpoint-complete hair polyline must have positive length")
    return length_um


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _measured_total_or_null(
    values: Sequence[float], *, identity_count: int
) -> float | None:
    """Separate a biological zero from an unobserved conditional total.

    No accepted identities is a measured zero.  Accepted identities with no
    endpoint-complete match have no length observation and must remain null.
    Once at least one match exists, the sum is a measured (possibly partial)
    total whose support is reported separately.
    """

    if identity_count < 0 or len(values) > identity_count:
        raise ContractError("endpoint-complete length support exceeds identity count")
    if identity_count == 0:
        return 0.0
    if not values:
        return None
    return float(np.sum(values))


def _scale_contract_state(
    prediction: Mapping[str, Any], *, metadata_um_per_px: float
) -> tuple[bool, str | None]:
    """Return whether physical units are jointly trusted and, if not, why.

    A prediction-side scale flag is not sufficient for publication: the
    calibration embedded in the prediction must also agree with the sealed
    analysis metadata used for trait materialization.  This helper is shared
    by eligibility and canonical-row publication so the two cannot diverge.
    """

    if not np.isfinite(metadata_um_per_px) or metadata_um_per_px <= 0.0:
        raise ContractError("analysis metadata um_per_px must be finite and positive")
    scale = prediction.get("scale")
    if not isinstance(scale, Mapping) or scale.get("fail_closed") is not False:
        return False, "scale_fail_closed"
    predicted_raw = scale.get("predicted_um_per_px", scale.get("um_per_px"))
    try:
        predicted_um_per_px = float(predicted_raw)
    except (TypeError, ValueError):
        return False, "scale_fail_closed"
    if not np.isfinite(predicted_um_per_px) or predicted_um_per_px <= 0.0:
        return False, "scale_fail_closed"
    if not np.isclose(
        predicted_um_per_px,
        metadata_um_per_px,
        rtol=1e-8,
        atol=1e-10,
    ):
        return False, "scale_metadata_mismatch"
    return True, None


def _common(
    prediction: Mapping[str, Any], metadata: Mapping[str, str], *, formal: bool
) -> dict[str, Any]:
    return {
        "task_id": prediction["task_id"],
        "source_image_sha256": prediction["source_image_sha256"],
        "experiment_key": metadata.get("experiment_key", ""),
        "condition_code": metadata.get("condition_code", ""),
        "study_role": metadata.get("study_role", ""),
        "developmental_day": metadata.get("developmental_day", ""),
        "genotype_or_construct": metadata.get("genotype_or_construct", ""),
        "temperature_c": metadata.get("temperature_c", ""),
        "qc_disposition": metadata.get("qc_disposition", ""),
        "formal_statistics_eligible": formal,
        "automatic_formal_phenotype_eligible": bool(
            prediction.get("formal_phenotype_eligible")
        ),
        "root_axis_source": prediction.get("root_axis_source", ""),
        "root_global_width_source": prediction.get(
            "root_global_width_source", "selected_root_geometry"
        ),
        "root_global_width_reference_applied": bool(
            prediction.get("root_global_width_reference_applied", False)
        ),
    }


def _statistics(prediction: Mapping[str, Any], *, formal: bool) -> Mapping[str, Any]:
    preferred = "detailed_root_statistics" if formal else "detailed_root_statistics_review_only"
    statistics = prediction.get(preferred)
    if not isinstance(statistics, Mapping):
        statistics = prediction.get("detailed_root_statistics")
    if not isinstance(statistics, Mapping):
        statistics = prediction.get("detailed_root_statistics_review_only")
    if not isinstance(statistics, Mapping):
        raise ContractError(f"{prediction['task_id']}: root statistics are absent")
    return statistics


def _one_prediction(
    prediction: Mapping[str, Any], metadata: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if prediction.get("blind_images_used") != 0 or prediction.get("phaxis", {}).get(
        "blind_images_used"
    ) != 0:
        raise ContractError(f"{prediction['task_id']}: blind-tainted prediction")
    if prediction.get("root_cap_region_output") is not False:
        raise ContractError(f"{prediction['task_id']}: root-cap region output is forbidden")
    try:
        source_um_per_px = float(metadata["um_per_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(
            f"{prediction['task_id']}: analysis metadata um_per_px is absent or invalid"
        ) from error
    physical_units_valid, _scale_failure_reason = _scale_contract_state(
        prediction,
        metadata_um_per_px=source_um_per_px,
    )
    formal = bool(
        physical_units_valid and prediction.get("formal_phenotype_eligible")
    )
    common = _common(prediction, metadata, formal=formal)
    hair_identity_expert = _hair_identity_expert(prediction)
    statistics = _statistics(prediction, formal=formal)
    root_row = {**common, **dict(statistics)}
    for field in ROOT_TRAIT_FIELDS:
        if field not in statistics:
            raise ContractError(f"{prediction['task_id']}: missing root trait {field}")

    identities = list(prediction.get("identity_hairs", ()))
    if len(identities) != int(prediction.get("phaxis", {}).get("formal_stageb_identity_count", -1)):
        raise ContractError(f"{prediction['task_id']}: Stage-B identity count drift")
    identity_by_id: dict[str, Mapping[str, Any]] = {}
    for hair in identities:
        identity_id = str(hair.get("source_instance_id", ""))
        if not identity_id or identity_id in identity_by_id:
            raise ContractError(f"{prediction['task_id']}: invalid Stage-B identity ID")
        identity_by_id[identity_id] = hair
    length_by_identity: dict[str, tuple[Mapping[str, Any], float]] = {}
    for curve in prediction.get("length_hairs", ()):
        identity_id = str(curve.get("identity_source_instance_id", ""))
        if identity_id not in identity_by_id or identity_id in length_by_identity:
            raise ContractError(f"{prediction['task_id']}: invalid length/identity association")
        length_by_identity[identity_id] = (
            curve,
            _polyline_length_um(curve, um_per_px=source_um_per_px),
        )
    expected_matches = int(
        prediction.get("phaxis", {})
        .get("length_identity_association", {})
        .get("matched_length_identities", -1)
    )
    if expected_matches != len(length_by_identity):
        raise ContractError(f"{prediction['task_id']}: matched length count drift")

    valid_axis: list[float] = []
    length_values: list[float] = []
    local_length_values: list[float] = []
    first_ge40_candidates: list[float] = []
    local_count = 0
    hair_rows: list[dict[str, Any]] = []
    root_length_um = float(statistics["visible_root_axis_length_um"])
    window_eligible = bool(formal and root_length_um >= 4000.0)
    for index, hair in enumerate(identities, start=1):
        identity_id = str(hair["source_instance_id"])
        attachment_valid = bool(hair.get("root_attachment_valid", False))
        axis_um = float(hair.get("root_axis_distance_from_tip_um", float("nan")))
        if attachment_valid and np.isfinite(axis_um):
            valid_axis.append(axis_um)
        in_window = bool(
            window_eligible and attachment_valid and 1000.0 <= axis_um < 4000.0
        )
        if in_window:
            local_count += 1
        length_record = length_by_identity.get(identity_id)
        length_um = length_record[1] if length_record is not None else None
        if length_um is not None:
            length_values.append(length_um)
            if in_window:
                local_length_values.append(length_um)
            if attachment_valid and length_um >= 40.0:
                first_ge40_candidates.append(axis_um)
        hair_rows.append(
            {
                **common,
                "hair_id": identity_id,
                "hair_index": index,
                "identity_expert": hair_identity_expert,
                "stageb_score": hair.get("stageb_score"),
                "stageb_predicted_length_um_diagnostic_only": hair.get(
                    "stageb_predicted_length_um"
                ),
                "attachment_distance_from_distal_point_um": (
                    axis_um if attachment_valid and np.isfinite(axis_um) else None
                ),
                "attachment_boundary_error_um": hair.get(
                    "root_boundary_attachment_error_um"
                ),
                "attachment_valid_within_40um": attachment_valid,
                "in_preregistered_distal_window_1_4mm": in_window,
                "complete_length_measurement_eligible": length_record is not None,
                "length_expert": (
                    PUBLIC_HAIR_LENGTH_EXPERT_ID
                    if length_record is not None
                    else None
                ),
                "length_um": length_um,
                "length_identity_base_match_error_um": hair.get(
                    "length_identity_base_match_error_um"
                ),
                "identity_points_xy_json": json.dumps(
                    hair["points_xy"], separators=(",", ":")
                ),
                "length_points_xy_json": (
                    json.dumps(length_record[0]["points_xy"], separators=(",", ":"))
                    if length_record is not None
                    else None
                ),
            }
        )
    count = len(identities)
    valid_axis_array = np.asarray(valid_axis, dtype=np.float64)
    measured_local_total = (
        _measured_total_or_null(local_length_values, identity_count=local_count)
        if window_eligible
        else None
    )
    traits = {
        **common,
        "hair_count": count,
        "hair_length_measurement_hair_count": len(length_values),
        "hair_length_measurement_fraction": (
            len(length_values) / count if count else None
        ),
        "hair_length_semantics": PUBLIC_HAIR_LENGTH_SEMANTICS,
        "total_hair_length_is_partial": len(length_values) != count,
        "mean_hair_length_um": _mean(length_values),
        "median_hair_length_um": _median(length_values),
        "total_hair_length_um": _measured_total_or_null(
            length_values, identity_count=count
        ),
        "hair_density_per_mm_visible_root": (
            count / (root_length_um / 1000.0) if root_length_um > 0 else None
        ),
        "first_hair_distance_from_distal_point_um": (
            float(valid_axis_array.min()) if valid_axis_array.size else None
        ),
        "first_hair_ge40um_distance_from_distal_point_um": (
            float(min(first_ge40_candidates)) if first_ge40_candidates else None
        ),
        "distal_window_1_4mm_eligible": window_eligible,
        "local_hair_count_1_4mm": local_count if window_eligible else None,
        "local_hair_density_per_mm_1_4mm": (
            local_count / 3.0 if window_eligible else None
        ),
        "local_mean_hair_length_um_1_4mm": (
            _mean(local_length_values) if window_eligible else None
        ),
        "local_median_hair_length_um_1_4mm": (
            _median(local_length_values) if window_eligible else None
        ),
        "local_total_hair_length_um_per_root_mm_1_4mm": (
            measured_local_total / 3.0 if measured_local_total is not None else None
        ),
        "visible_hair_attachment_span_um_descriptive_right_censored": (
            float(np.ptp(valid_axis_array)) if valid_axis_array.size >= 2 else 0.0
            if valid_axis_array.size == 1
            else None
        ),
        "attachment_axis_valid_fraction": (
            len(valid_axis) / count if count else None
        ),
        "whole_hair_zone_confirmatory_allowed": False,
        "visible_root_axis_length_um": statistics["visible_root_axis_length_um"],
        "median_root_width_um": statistics["median_root_width_um"],
    }
    return traits, root_row, hair_rows


def _canonical_image_traits(
    prediction: Mapping[str, Any],
    traits: Mapping[str, Any],
    statistics: Mapping[str, Any],
    *,
    prediction_sha256: str,
    um_per_px: float,
) -> dict[str, Any]:
    phaxis = prediction.get("phaxis")
    if not isinstance(phaxis, Mapping):
        raise ContractError("PHAxis prediction provenance is absent")
    proposal_bound = all(
        isinstance(phaxis.get(field), str) and len(str(phaxis[field])) == 64
        for field in (
            "model_contract_proposal_sha256",
            "model_contract_proposal_identity_sha256",
        )
    )
    model_bundle_id = phaxis.get("model_bundle_id")
    root_expert_id = phaxis.get("root_expert")
    if not proposal_bound or (
        not isinstance(model_bundle_id, str)
        or not model_bundle_id.startswith(MODEL_BUNDLE_PREFIX)
        or not isinstance(root_expert_id, str)
        or not root_expert_id.startswith(ROOT_EXPERT_PREFIX)
    ):
        raise ContractError(
            "proposal-bound prediction lacks the proposal-owned public model/root IDs"
        )
    scale_value = prediction.get("scale")
    scale = scale_value if isinstance(scale_value, Mapping) else {}
    physical_units_valid, scale_failure_reason = _scale_contract_state(
        prediction,
        metadata_um_per_px=um_per_px,
    )
    point = prediction.get("root_cap_point_xy", (None, None))
    fail_closed = bool(
        not physical_units_valid
        or prediction.get(
            "automatic_measurement_fail_closed",
            not bool(prediction.get("formal_phenotype_eligible")),
        )
    )
    row: dict[str, Any] = {
        "schema_version": "PHAxis-image-traits-1.0.0",
        "product_version": PRODUCT_VERSION,
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_expert_id,
        "hair_identity_count_expert_id": _hair_identity_expert(prediction),
        "hair_length_expert_id": PUBLIC_HAIR_LENGTH_EXPERT_ID,
        "task_id": prediction["task_id"],
        "source_image_sha256": prediction["source_image_sha256"],
        "prediction_sha256": prediction_sha256,
        "root_lock_sha256": prediction["phaxis"]["root_lock_sha256"],
        "stageb_detection_identity_sha256": prediction["phaxis"][
            "stageb_detection_identity_sha256"
        ],
        "measurement_tier": (
            "automatic_unreviewed"
            if traits["formal_statistics_eligible"]
            else "review_only"
        ),
        "formal_statistics_eligible": traits["formal_statistics_eligible"],
        "automatic_measurement_fail_closed": fail_closed,
        "exclusion_reason": (
            prediction.get("automatic_measurement_fail_reason")
            or scale_failure_reason
        ),
        "scale_status": "detected" if physical_units_valid else "fail_closed",
        "physical_units_valid": physical_units_valid,
        "um_per_px": um_per_px if physical_units_valid else None,
        "scale_value_um": scale.get("predicted_value_um") if physical_units_valid else None,
        "scale_length_px": scale.get("predicted_length_px") if physical_units_valid else None,
        "root_cap_region_output": False,
        "root_cap_area_used": False,
        "root_cap_point_x_px": point[0],
        "root_cap_point_y_px": point[1],
        "root_continuity_status": prediction.get("root_continuity_status"),
        "root_continuity_applied": bool(prediction.get("root_continuity_applied", False)),
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read_during_inference": False,
        "blind_images_used": 0,
    }
    for field in ROOT_TRAIT_FIELDS:
        row[field] = statistics.get(field) if physical_units_valid else None
    for field in HAIR_TRAIT_FIELDS:
        row[field] = traits.get(field) if physical_units_valid or field == "hair_count" else None
    for field in (
        "root_cap_point_border_distance_px",
        "root_cap_point_border_visible",
        "shootward_endpoint_x_px",
        "shootward_endpoint_y_px",
        "shootward_endpoint_border_distance_px",
        "shootward_endpoint_border_visible",
        "root_orientation_deg",
        "root_cap_point_to_axis_bridge_um",
        "axis_endpoint_count",
        "axis_reference_radius_px",
        "width_samples",
        "width_end_exclusion_um",
    ):
        row[field] = (
            None
            if not physical_units_valid
            and field
            in {
                "root_cap_point_to_axis_bridge_um",
                "width_end_exclusion_um",
            }
            else statistics.get(field)
        )
    for field in (
        "hair_length_measurement_hair_count",
        "hair_length_measurement_fraction",
        "hair_length_semantics",
        "total_hair_length_is_partial",
        "distal_window_1_4mm_eligible",
        "attachment_axis_valid_fraction",
        "whole_hair_zone_confirmatory_allowed",
        "root_axis_source",
        "root_global_width_source",
    ):
        row[field] = traits.get(field)
    if set(row) != set(IMAGE_TRAIT_FIELDS) or len(IMAGE_TRAIT_FIELDS) != len(set(IMAGE_TRAIT_FIELDS)):
        missing = sorted(set(IMAGE_TRAIT_FIELDS) - set(row))
        extra = sorted(set(row) - set(IMAGE_TRAIT_FIELDS))
        raise ContractError(f"canonical image-trait field drift; missing={missing}, extra={extra}")
    return row


def export_traits(
    *,
    prediction_root: str | Path,
    metadata_csv: str | Path,
    output: str | Path,
    model_contract_proposal: Mapping[str, str],
    model_contract_public_identity: Mapping[str, str],
) -> dict[str, Any]:
    prediction_root = Path(prediction_root).resolve()
    artifact_root = prediction_root.parent
    metadata_csv = Path(metadata_csv).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    metadata_rows = _read_csv(metadata_csv)
    metadata = {row["task_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ContractError("duplicate task_id in analysis metadata")
    prediction_paths = sorted(prediction_root.glob("*.json"))
    if {path.stem for path in prediction_paths} != set(metadata):
        raise ContractError("prediction/metadata task sets differ")
    trait_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    hair_rows: list[dict[str, Any]] = []
    image_trait_rows: list[dict[str, Any]] = []
    prediction_sha256: dict[str, str] = {}
    timing_records: list[dict[str, Any]] = []
    proposal_fields = dict(model_contract_proposal)
    if set(proposal_fields) != {
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
    }:
        raise ContractError("trait export received an invalid model-contract binding")
    try:
        valid_proposal = all(
            len(value) == 64 and int(value, 16) >= 0
            for value in proposal_fields.values()
        )
    except (TypeError, ValueError):
        valid_proposal = False
    if not valid_proposal:
        raise ContractError("trait export model-contract hashes are invalid")
    public_identity = dict(model_contract_public_identity)
    if set(public_identity) != {"model_bundle_id", "root_expert_id"}:
        raise ContractError(
            "proposal-bound trait export requires the exact public model/root identity"
        )
    if public_identity and (
        not isinstance(public_identity.get("model_bundle_id"), str)
        or not public_identity["model_bundle_id"].startswith(MODEL_BUNDLE_PREFIX)
        or not isinstance(public_identity.get("root_expert_id"), str)
        or not public_identity["root_expert_id"].startswith(ROOT_EXPERT_PREFIX)
    ):
        raise ContractError("trait export public model/root identity is invalid")
    for path in prediction_paths:
        task_started = time.perf_counter()
        prediction = read_json(path)
        validate_hybrid_prediction(prediction, artifact_root=artifact_root)
        task_id = str(prediction["task_id"])
        phaxis = prediction.get("phaxis")
        if not isinstance(phaxis, Mapping):
            raise ContractError(f"{task_id}: fused PHAxis provenance is absent")
        for field, expected in proposal_fields.items():
            if phaxis.get(field) != expected:
                raise ContractError(f"{task_id}: fused model-contract binding mismatch")
        if (
            phaxis.get("model_bundle_id") != public_identity["model_bundle_id"]
            or phaxis.get("root_expert") != public_identity["root_expert_id"]
        ):
            raise ContractError(f"{task_id}: fused public model/root identity mismatch")
        row = metadata[task_id]
        if prediction["source_image_sha256"].casefold() != row["image_sha256"].casefold():
            raise ContractError(f"{task_id}: prediction/metadata image hash mismatch")
        traits, roots, hairs = _one_prediction(prediction, row)
        observed_prediction_sha256 = sha256_file(path)
        trait_rows.append(traits)
        root_rows.append(roots)
        hair_rows.extend(hairs)
        image_trait_rows.append(
            _canonical_image_traits(
                prediction,
                traits,
                roots,
                prediction_sha256=observed_prediction_sha256,
                um_per_px=float(row["um_per_px"]),
            )
        )
        prediction_sha256[task_id] = observed_prediction_sha256
        timing_records.append(
            {
                "task_id": task_id,
                "wall_seconds_including_prediction_io": time.perf_counter()
                - task_started,
                "timing_trace_kind": "direct_per_source_nonoverlapping",
            }
        )

    trait_fields = (
        *COMMON_FIELDS,
        "hair_length_measurement_hair_count",
        "hair_length_measurement_fraction",
        "hair_length_semantics",
        "total_hair_length_is_partial",
        *HAIR_TRAIT_FIELDS,
        "attachment_axis_valid_fraction",
        "distal_window_1_4mm_eligible",
        "whole_hair_zone_confirmatory_allowed",
        "visible_root_axis_length_um",
        "median_root_width_um",
    )
    root_extra = sorted({key for row in root_rows for key in row} - set(COMMON_FIELDS))
    hair_extra = (
        "hair_id",
        "hair_index",
        "identity_expert",
        "stageb_score",
        "stageb_predicted_length_um_diagnostic_only",
        "attachment_distance_from_distal_point_um",
        "attachment_boundary_error_um",
        "attachment_valid_within_40um",
        "in_preregistered_distal_window_1_4mm",
        "complete_length_measurement_eligible",
        "length_expert",
        "length_um",
        "length_identity_base_match_error_um",
        "identity_points_xy_json",
        "length_points_xy_json",
    )
    trait_path = output / "traits.csv"
    root_path = output / "detailed_root_statistics.csv"
    hair_path = output / "hair_instances.csv"
    image_trait_path = output / "image_traits.csv"
    _atomic_csv(trait_path, trait_rows, trait_fields)
    _atomic_csv(root_path, root_rows, (*COMMON_FIELDS, *root_extra))
    _atomic_csv(
        hair_path,
        hair_rows,
        (*COMMON_FIELDS, *hair_extra),
        allow_empty=True,
    )
    _atomic_csv(image_trait_path, image_trait_rows, IMAGE_TRAIT_FIELDS)
    formal_count = sum(bool(row["formal_statistics_eligible"]) for row in trait_rows)
    hair_expert_ids = {
        str(row["hair_identity_count_expert_id"]) for row in image_trait_rows
    }
    if len(hair_expert_ids) != 1:
        raise ContractError(
            f"trait export mixes hair experts across images: {sorted(hair_expert_ids)}"
        )
    hair_expert_id = next(iter(hair_expert_ids))
    model_bundle_ids = {str(row["model_bundle_id"]) for row in image_trait_rows}
    root_expert_ids = {str(row["root_expert_id"]) for row in image_trait_rows}
    if len(model_bundle_ids) != 1 or len(root_expert_ids) != 1:
        raise ContractError(
            "trait export mixes public model/root identities across images"
        )
    if public_identity and (
        model_bundle_ids != {public_identity["model_bundle_id"]}
        or root_expert_ids != {public_identity["root_expert_id"]}
    ):
        raise ContractError("trait export public identity differs from model contract")
    summary: dict[str, Any] = {
        "schema_version": "PHAxis-trait-export-1.0",
        "status": "completed",
        "tasks": len(trait_rows),
        "formal_statistics_eligible": formal_count,
        "review_only": len(trait_rows) - formal_count,
        "hair_identities": len(hair_rows),
        "endpoint_complete_length_identities": sum(
            bool(row["complete_length_measurement_eligible"]) for row in hair_rows
        ),
        "nonduplicate_biological_numeric_traits": 32,
        "root_trait_fields": list(ROOT_TRAIT_FIELDS),
        "hair_trait_fields": list(HAIR_TRAIT_FIELDS),
        "traits_sha256": sha256_file(trait_path),
        "image_traits_sha256": sha256_file(image_trait_path),
        "detailed_root_statistics_sha256": sha256_file(root_path),
        "hair_instances_sha256": sha256_file(hair_path),
        "analysis_metadata_sha256": sha256_file(metadata_csv),
        "prediction_sha256": prediction_sha256,
        "per_source_timing_records": timing_records,
        "per_source_timing_semantics": (
            "direct prediction_input_io_validation_and_trait_compute; aggregate CSV "
            "publication is recorded only in the enclosing batch stage wall"
        ),
        "root_cap_region_statistics_included": False,
        "hair_identity_count_expert": hair_expert_id,
        "model_bundle_id": next(iter(model_bundle_ids)),
        "root_expert_id": next(iter(root_expert_ids)),
        "hair_length_expert": PUBLIC_HAIR_LENGTH_EXPERT_ID,
        "stageb_predicted_length_used_as_formal_trait": False,
        "whole_hair_zone_confirmatory_traits_allowed": False,
        "condition_metadata_used_for_model_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
    }
    summary["export_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "summary.json", summary)
    return summary
