"""Deterministic primary-root continuity measurement assurance.

The reference is an ordered axis recomputed from the canonical vector root
polygon and annotated distal point.  Predictions are *connected* primary-axis
components skeletonized from the sealed final fused root foreground delivered
to trait extraction.  The final foreground may already contain PHAxis's own
formal continuity repair; the evaluator itself may not interpolate, bridge,
or complete a gap.  Keeping components separate is essential: two fragments
may jointly cover the reference within tolerance, but they cannot satisfy the
formal single-component ``break_free`` criterion.

All geometry supplied to this module is already in physical micrometre XY
coordinates.  The intended producer wiring point is the QC-development
scoring adapter, immediately after it has loaded (1) the canonical
vector-derived reference axis and (2) the sealed prediction mask.  That
adapter must skeletonize every connected component of the final fused
predicted-root foreground without evaluator-side gap completion, convert both
geometries with the image's locked scale, and bind the source artifacts by
SHA-256 in every record.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import atomic_write_json, read_json, sha256_file, sha256_json


ROOT_CONTINUITY_INPUT_SCHEMA = "PHAxis-primary-root-continuity-assurance-input-1.0"
ROOT_CONTINUITY_ASSURANCE_SCHEMA = "PHAxis-primary-root-continuity-assurance-1.0"
ROOT_CONTINUITY_EVIDENCE_ROLE = "annotated_qc_development_non_independent"
ROOT_CONTINUITY_REFERENCE_DEFINITION = (
    "ordered visible primary-root axis recomputed from the canonical "
    "vector-derived root polygon and annotated distal/root-cap point"
)
ROOT_CONTINUITY_PREDICTION_DEFINITION = (
    "connected primary-axis components skeletonized from the sealed final fused "
    "predicted root foreground delivered to trait extraction; no evaluator-side "
    "interpolation, bridging, or gap completion"
)
ROOT_CONTINUITY_COORDINATE_SPACE = "physical_um_xy"
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_828
BOOTSTRAP_METHOD = "source-image nonparametric percentile bootstrap"


class RootContinuityAssuranceError(RuntimeError):
    """A root-continuity geometry, denominator, or identity contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RootContinuityAssuranceError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_positive(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RootContinuityAssuranceError(f"{name} must be finite and positive") from error
    _require(math.isfinite(numeric) and numeric > 0.0, f"{name} must be finite and positive")
    return numeric


def _polyline(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(
        array.ndim == 2
        and array.shape[1] == 2
        and len(array) >= 2
        and np.all(np.isfinite(array)),
        f"{name} must be a finite N x 2 polyline with at least two points",
    )
    _require(
        float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum()) > 0.0,
        f"{name} has zero arc length",
    )
    return array


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _reference_interval_midpoints(
    points: np.ndarray, maximum_step_um: float
) -> tuple[np.ndarray, float, float]:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    points = points[keep]
    intervals = max(1, int(math.ceil(total / maximum_step_um)))
    interval_um = total / intervals
    locations = (np.arange(intervals, dtype=np.float64) + 0.5) * interval_um
    sampled = np.column_stack(
        (
            np.interp(locations, cumulative, points[:, 0]),
            np.interp(locations, cumulative, points[:, 1]),
        )
    )
    return sampled, interval_um, total


def _segments(components: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for component in components:
        delta = np.diff(component, axis=0)
        keep = np.linalg.norm(delta, axis=1) > 1e-12
        if np.any(keep):
            starts.append(component[:-1][keep])
            ends.append(component[1:][keep])
    if not starts:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.concatenate(starts), np.concatenate(ends)


def _distance_to_segments(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    if len(starts) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    result = np.full(len(points), np.inf, dtype=np.float64)
    vectors = ends - starts
    squared = np.sum(vectors * vectors, axis=1)
    # Chunking bounds temporary memory while retaining exact point-to-segment
    # distances (rather than an implementation-dependent raster approximation).
    for first in range(0, len(points), 2048):
        chunk = points[first : first + 2048]
        relative = chunk[:, None, :] - starts[None, :, :]
        projection = np.clip(
            np.sum(relative * vectors[None, :, :], axis=2)
            / squared[None, :],
            0.0,
            1.0,
        )
        nearest = starts[None, :, :] + projection[:, :, None] * vectors[None, :, :]
        result[first : first + len(chunk)] = np.sqrt(
            np.min(np.sum((chunk[:, None, :] - nearest) ** 2, axis=2), axis=1)
        )
    return result


def _project_to_reference(
    points: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    starts = reference[:-1]
    ends = reference[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 1e-12
    starts, vectors, lengths = starts[keep], vectors[keep], lengths[keep]
    squared = lengths * lengths
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))[:-1]
    relative = points[:, None, :] - starts[None, :, :]
    fraction = np.clip(
        np.sum(relative * vectors[None, :, :], axis=2) / squared[None, :],
        0.0,
        1.0,
    )
    nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    selected = np.argmin(distances, axis=1)
    return (
        distances[np.arange(len(points)), selected],
        cumulative[selected] + fraction[np.arange(len(points)), selected] * lengths[selected],
    )


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask.tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _first_difference(observed: Any, expected: Any, path: str = "$") -> str | None:
    if type(observed) is not type(expected):
        return f"{path}: type {type(observed).__name__} != {type(expected).__name__}"
    if isinstance(observed, Mapping):
        observed_keys, expected_keys = set(observed), set(expected)
        if observed_keys != expected_keys:
            return f"{path}: keys missing={sorted(expected_keys-observed_keys)} extra={sorted(observed_keys-expected_keys)}"
        for key in sorted(observed):
            difference = _first_difference(observed[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(observed, list):
        if len(observed) != len(expected):
            return f"{path}: length {len(observed)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if observed == expected else f"{path}: {observed!r} != {expected!r}"


def _normalise_records(
    records: Sequence[Mapping[str, Any]], source_units: Sequence[str]
) -> list[dict[str, Any]]:
    units = [str(value) for value in source_units]
    _require(units and len(units) == len(set(units)) and all(units), "source-unit denominator is empty or duplicated")
    by_unit: dict[str, dict[str, Any]] = {}
    for raw in records:
        _require(isinstance(raw, Mapping), "root-continuity record is malformed")
        source_unit = str(raw.get("source_unit", ""))
        _require(source_unit in units and source_unit not in by_unit, f"unknown or duplicate source unit: {source_unit}")
        _require(raw.get("pair_type") == "primary_root_continuity", f"{source_unit}: pair_type drift")
        _require(raw.get("coordinate_space") == ROOT_CONTINUITY_COORDINATE_SPACE, f"{source_unit}: coordinate space drift")
        _require(raw.get("reference_axis_definition") == ROOT_CONTINUITY_REFERENCE_DEFINITION, f"{source_unit}: reference-axis definition drift")
        _require(raw.get("prediction_axis_definition") == ROOT_CONTINUITY_PREDICTION_DEFINITION, f"{source_unit}: prediction-axis definition drift")
        for field in (
            "source_image_sha256",
            "reference_axis_artifact_sha256",
            "prediction_axis_artifact_sha256",
        ):
            _require(_is_sha256(raw.get(field)), f"{source_unit}: invalid {field}")
        reference = _polyline(raw.get("reference_axis_xy_um"), f"{source_unit}.reference_axis_xy_um")
        components_raw = raw.get("predicted_axis_components_xy_um")
        _require(isinstance(components_raw, list), f"{source_unit}: predicted components must be a list")
        components = [
            _polyline(component, f"{source_unit}.predicted_axis_components_xy_um[{index}]")
            for index, component in enumerate(components_raw)
        ]
        normalized = {
            "pair_type": "primary_root_continuity",
            "source_unit": source_unit,
            "source_image_sha256": str(raw["source_image_sha256"]),
            "coordinate_space": ROOT_CONTINUITY_COORDINATE_SPACE,
            "reference_axis_definition": ROOT_CONTINUITY_REFERENCE_DEFINITION,
            "prediction_axis_definition": ROOT_CONTINUITY_PREDICTION_DEFINITION,
            "reference_axis_artifact_sha256": str(raw["reference_axis_artifact_sha256"]),
            "prediction_axis_artifact_sha256": str(raw["prediction_axis_artifact_sha256"]),
            "reference_axis_xy_um": reference.tolist(),
            "predicted_axis_components_xy_um": [component.tolist() for component in components],
        }
        normalized["input_geometry_identity_sha256"] = sha256_json(normalized)
        by_unit[source_unit] = normalized
    _require(set(by_unit) == set(units), "root-continuity source-unit denominator drift")
    normalized = [by_unit[source_unit] for source_unit in units]
    image_hashes = [row["source_image_sha256"] for row in normalized]
    _require(
        len(image_hashes) == len(set(image_hashes)),
        "root-continuity bootstrap requires exactly one record per source image",
    )
    return normalized


def _measure_record(
    record: Mapping[str, Any], *, support_tolerance_um: float, sampling_step_um: float
) -> dict[str, Any]:
    reference = np.asarray(record["reference_axis_xy_um"], dtype=np.float64)
    components = [
        np.asarray(component, dtype=np.float64)
        for component in record["predicted_axis_components_xy_um"]
    ]
    midpoints, interval_um, reference_length_um = _reference_interval_midpoints(
        reference, sampling_step_um
    )
    component_support: list[np.ndarray] = []
    component_rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        starts, ends = _segments([component])
        supported_by_component = (
            _distance_to_segments(midpoints, starts, ends) <= support_tolerance_um
        )
        component_support.append(supported_by_component)
        component_rows.append(
            {
                "component_index": component_index,
                "reference_axis_coverage": float(np.mean(supported_by_component)),
                "longest_unsupported_gap_um": float(
                    _longest_true_run(~supported_by_component) * interval_um
                ),
                "spans_reference_axis": bool(np.all(supported_by_component)),
            }
        )
    if component_support:
        support_matrix = np.column_stack(component_support)
        union_supported = np.any(support_matrix, axis=1)
        best_component_index = min(
            range(len(component_rows)),
            key=lambda index: (
                -float(component_rows[index]["reference_axis_coverage"]),
                float(component_rows[index]["longest_unsupported_gap_um"]),
                index,
            ),
        )
        maximum_single_component_coverage = float(
            component_rows[best_component_index]["reference_axis_coverage"]
        )
        longest_gap_best_component_um = float(
            component_rows[best_component_index]["longest_unsupported_gap_um"]
        )
        spanning_component_count = int(
            sum(bool(row["spans_reference_axis"]) for row in component_rows)
        )
    else:
        union_supported = np.zeros(len(midpoints), dtype=bool)
        best_component_index = None
        maximum_single_component_coverage = 0.0
        longest_gap_best_component_um = reference_length_um
        spanning_component_count = 0
    coverage = float(np.mean(union_supported))
    longest_gap_um = float(_longest_true_run(~union_supported) * interval_um)
    union_fully_supported = bool(np.all(union_supported))
    union_hides_fragmentation = bool(
        union_fully_supported and spanning_component_count == 0
    )

    prediction_points = (
        np.concatenate(components, axis=0)
        if components
        else np.empty((0, 2), dtype=np.float64)
    )
    if len(prediction_points):
        projection_distance, position_um = _project_to_reference(prediction_points, reference)
        accepted = projection_distance <= support_tolerance_um
        predicted_extent_um = (
            float(np.max(position_um[accepted]) - np.min(position_um[accepted]))
            if np.any(accepted)
            else 0.0
        )
    else:
        predicted_extent_um = 0.0
    signed_extent_error = predicted_extent_um - reference_length_um
    measured: dict[str, Any] = {
        "source_unit": record["source_unit"],
        "source_image_sha256": record["source_image_sha256"],
        "reference_axis_artifact_sha256": record["reference_axis_artifact_sha256"],
        "prediction_axis_artifact_sha256": record["prediction_axis_artifact_sha256"],
        "input_geometry_identity_sha256": record["input_geometry_identity_sha256"],
        "reference_axis_length_um": reference_length_um,
        "prediction_connected_component_count": len(components),
        "reference_sampling_intervals": len(midpoints),
        "realized_sampling_interval_um": interval_um,
        "reference_axis_coverage": coverage,
        "unsupported_axis_length_um": float(
            np.count_nonzero(~union_supported) * interval_um
        ),
        "longest_unsupported_gap_um": longest_gap_um,
        "maximum_single_component_coverage": maximum_single_component_coverage,
        "best_component_index": best_component_index,
        "longest_unsupported_gap_um_on_best_component": (
            longest_gap_best_component_um
        ),
        "spanning_component_count": spanning_component_count,
        "union_reference_axis_fully_supported": union_fully_supported,
        "union_coverage_hides_fragmentation": union_hides_fragmentation,
        "connected_component_support": component_rows,
        "break_free": spanning_component_count > 0,
        "predicted_visible_axis_extent_um": predicted_extent_um,
        "visible_axis_extent_error_um_signed": signed_extent_error,
        "visible_axis_extent_error_um_abs": abs(signed_extent_error),
        "bootstrap_sufficient_statistics": {
            "reference_axis_coverage": coverage,
            "longest_unsupported_gap_um": longest_gap_um,
            "maximum_single_component_coverage": maximum_single_component_coverage,
            "longest_unsupported_gap_um_on_best_component": (
                longest_gap_best_component_um
            ),
            "break_free": spanning_component_count > 0,
            "visible_axis_extent_error_um_abs": abs(signed_extent_error),
        },
    }
    measured["row_identity_sha256"] = sha256_json(measured)
    return measured


def _bootstrap_interval(point: float, estimates: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(estimates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    _require(len(finite) == BOOTSTRAP_REPETITIONS, "root bootstrap produced undefined replicates")
    low, high = np.quantile(finite, (0.025, 0.975))
    return {
        "point_estimate": float(point),
        "ci_low_2_5": float(low),
        "ci_high_97_5": float(high),
        "estimable_replicates": int(len(finite)),
    }


def _root_bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sufficient = [row["bootstrap_sufficient_statistics"] for row in rows]
    coverage = np.asarray(
        [row["reference_axis_coverage"] for row in sufficient], dtype=np.float64
    )
    gap = np.asarray(
        [row["longest_unsupported_gap_um"] for row in sufficient], dtype=np.float64
    )
    single_component_coverage = np.asarray(
        [row["maximum_single_component_coverage"] for row in sufficient],
        dtype=np.float64,
    )
    best_component_gap = np.asarray(
        [row["longest_unsupported_gap_um_on_best_component"] for row in sufficient],
        dtype=np.float64,
    )
    break_free = np.asarray(
        [row["break_free"] for row in sufficient], dtype=np.float64
    )
    extent = np.asarray(
        [row["visible_axis_extent_error_um_abs"] for row in sufficient],
        dtype=np.float64,
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(rows), size=(BOOTSTRAP_REPETITIONS, len(rows))
    )
    sampled_coverage = coverage[indices]
    sampled_gap = gap[indices]
    sampled_single_component_coverage = single_component_coverage[indices]
    sampled_best_component_gap = best_component_gap[indices]
    sampled_break_free = break_free[indices]
    sampled_extent = extent[indices]
    return {
        "reference_axis_coverage_mean": _bootstrap_interval(
            float(np.mean(coverage)), np.mean(sampled_coverage, axis=1)
        ),
        "reference_axis_coverage_median": _bootstrap_interval(
            float(np.median(coverage)), np.median(sampled_coverage, axis=1)
        ),
        "longest_unsupported_gap_um_median": _bootstrap_interval(
            float(np.median(gap)), np.median(sampled_gap, axis=1)
        ),
        "maximum_single_component_coverage_mean": _bootstrap_interval(
            float(np.mean(single_component_coverage)),
            np.mean(sampled_single_component_coverage, axis=1),
        ),
        "maximum_single_component_coverage_median": _bootstrap_interval(
            float(np.median(single_component_coverage)),
            np.median(sampled_single_component_coverage, axis=1),
        ),
        "longest_unsupported_gap_um_on_best_component_median": _bootstrap_interval(
            float(np.median(best_component_gap)),
            np.median(sampled_best_component_gap, axis=1),
        ),
        "break_free_image_rate": _bootstrap_interval(
            float(np.mean(break_free)), np.mean(sampled_break_free, axis=1)
        ),
        "visible_axis_extent_error_um_mae": _bootstrap_interval(
            float(np.mean(extent)), np.mean(sampled_extent, axis=1)
        ),
    }


def build_root_continuity_assurance(
    *,
    records: Sequence[Mapping[str, Any]],
    source_units: Sequence[str],
    reference_authority_sha256: str,
    prediction_authority_identity_sha256: str,
    support_tolerance_um: float = 5.0,
    sampling_step_um: float = 2.0,
    input_contract_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute and seal continuity metrics from physical-axis geometries."""

    _require(_is_sha256(reference_authority_sha256), "reference authority SHA-256 missing")
    _require(_is_sha256(prediction_authority_identity_sha256), "prediction authority identity missing")
    _require(
        input_contract_identity_sha256 is None
        or _is_sha256(input_contract_identity_sha256),
        "input-contract identity is invalid",
    )
    tolerance = _finite_positive(support_tolerance_um, "support_tolerance_um")
    step = _finite_positive(sampling_step_um, "sampling_step_um")
    _require(step <= tolerance, "sampling_step_um must not exceed support_tolerance_um")
    normalized = _normalise_records(records, source_units)
    rows = [
        _measure_record(record, support_tolerance_um=tolerance, sampling_step_um=step)
        for record in normalized
    ]
    coverage = np.asarray([row["reference_axis_coverage"] for row in rows], dtype=np.float64)
    gaps = np.asarray([row["longest_unsupported_gap_um"] for row in rows], dtype=np.float64)
    single_component_coverage = np.asarray(
        [row["maximum_single_component_coverage"] for row in rows],
        dtype=np.float64,
    )
    best_component_gaps = np.asarray(
        [row["longest_unsupported_gap_um_on_best_component"] for row in rows],
        dtype=np.float64,
    )
    signed_extent = np.asarray([row["visible_axis_extent_error_um_signed"] for row in rows], dtype=np.float64)
    absolute_extent = np.abs(signed_extent)
    bootstrap_intervals = _root_bootstrap(rows)
    source_unit_rows = [
        {"source_unit": row["source_unit"], "source_image_sha256": row["source_image_sha256"]}
        for row in normalized
    ]
    source_unit_set_identity = sha256_json(source_unit_rows)
    input_geometry_set_identity = sha256_json(normalized)
    implementation_identity = sha256_file(Path(__file__))
    payload: dict[str, Any] = {
        "schema_version": ROOT_CONTINUITY_ASSURANCE_SCHEMA,
        "status": "completed",
        "scope": "QC-development primary-root continuity assurance; non-independent",
        "evidence_role": ROOT_CONTINUITY_EVIDENCE_ROLE,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
        "provider_equivalence_used_as_accuracy": False,
        "reference_axis_definition": ROOT_CONTINUITY_REFERENCE_DEFINITION,
        "prediction_axis_definition": ROOT_CONTINUITY_PREDICTION_DEFINITION,
        "coordinate_space": ROOT_CONTINUITY_COORDINATE_SPACE,
        "metric_contract": {
            "support_tolerance_um": tolerance,
            "maximum_reference_sampling_interval_um": step,
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
        },
        "source_unit_total": len(rows),
        "source_unit_set_identity_sha256": source_unit_set_identity,
        "input_geometry_set_identity_sha256": input_geometry_set_identity,
        "input_contract_identity_sha256": input_contract_identity_sha256,
        "reference_authority_sha256": reference_authority_sha256,
        "prediction_authority_identity_sha256": prediction_authority_identity_sha256,
        "implementation_sha256": implementation_identity,
        "provenance": {
            "producer_wiring_point": (
                "scripts/phaxis/build_measurement_assurance_evidence.py, inside the "
                "locked QCdevelopment44 prediction loop immediately after canonical "
                "and sealed final fused predicted root masks plus the distal point have "
                "been verified; skeletonize connected final-mask components without "
                "evaluator-side interpolation, bridging, or gap completion"
            ),
            "reference_authority_sha256": reference_authority_sha256,
            "prediction_authority_identity_sha256": prediction_authority_identity_sha256,
            "source_unit_set_identity_sha256": source_unit_set_identity,
            "input_geometry_set_identity_sha256": input_geometry_set_identity,
            "input_contract_identity_sha256": input_contract_identity_sha256,
            "implementation_sha256": implementation_identity,
            "canonical_annotations_read_during_inference": False,
            "canonical_annotations_read_during_scoring": True,
            "val_labels_used_for_training": False,
            "blind_images_used": 0,
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "unit": "source_image",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "interval": "two-sided 95% percentile (2.5%, 97.5%)",
            "sufficient_statistics_location": "per_image[*].bootstrap_sufficient_statistics",
        },
        "summary": {
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
                np.mean(
                    [bool(row["union_coverage_hides_fragmentation"]) for row in rows]
                )
            ),
            "reference_axis_coverage_mean": float(np.mean(coverage)),
            "reference_axis_coverage_median": float(np.median(coverage)),
            "reference_axis_coverage_min": float(np.min(coverage)),
            "longest_unsupported_gap_um_median": float(np.median(gaps)),
            "longest_unsupported_gap_um_p95": float(np.quantile(gaps, 0.95)),
            "longest_unsupported_gap_um_max": float(np.max(gaps)),
            "maximum_single_component_coverage_mean": float(
                np.mean(single_component_coverage)
            ),
            "maximum_single_component_coverage_median": float(
                np.median(single_component_coverage)
            ),
            "maximum_single_component_coverage_min": float(
                np.min(single_component_coverage)
            ),
            "longest_unsupported_gap_um_on_best_component_median": float(
                np.median(best_component_gaps)
            ),
            "longest_unsupported_gap_um_on_best_component_p95": float(
                np.quantile(best_component_gaps, 0.95)
            ),
            "longest_unsupported_gap_um_on_best_component_max": float(
                np.max(best_component_gaps)
            ),
            "visible_axis_extent_error_um_mae": float(np.mean(absolute_extent)),
            "visible_axis_extent_error_um_median_abs": float(np.median(absolute_extent)),
            "visible_axis_extent_error_um_p95_abs": float(np.quantile(absolute_extent, 0.95)),
            "visible_axis_extent_error_um_bias": float(np.mean(signed_extent)),
            "bootstrap_95ci": bootstrap_intervals,
        },
        "per_image": rows,
        "per_image_set_identity_sha256": sha256_json(rows),
    }
    payload["root_continuity_assurance_identity_sha256"] = sha256_json(payload)
    return payload


def validate_root_continuity_assurance(
    payload: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    source_units: Sequence[str],
    reference_authority_sha256: str,
    prediction_authority_identity_sha256: str,
) -> dict[str, Any]:
    """Fail closed by recomputing every metric, denominator, and identity."""

    _require(payload.get("schema_version") == ROOT_CONTINUITY_ASSURANCE_SCHEMA, "root-continuity assurance schema drift")
    _require(payload.get("evidence_role") == ROOT_CONTINUITY_EVIDENCE_ROLE, "wrong root-continuity evidence role")
    _require(payload.get("independent_accuracy_claim_allowed") is False, "development assurance was mislabelled independent")
    _require(payload.get("blind_images_used") == 0, "blind images entered root-continuity assurance")
    _require(payload.get("provider_equivalence_used_as_accuracy") is False, "provider equivalence cannot satisfy continuity accuracy")
    config = payload.get("metric_contract")
    _require(isinstance(config, Mapping), "root-continuity metric contract missing")
    expected = build_root_continuity_assurance(
        records=records,
        source_units=source_units,
        reference_authority_sha256=reference_authority_sha256,
        prediction_authority_identity_sha256=prediction_authority_identity_sha256,
        support_tolerance_um=config.get("support_tolerance_um"),
        sampling_step_um=config.get("maximum_reference_sampling_interval_um"),
        input_contract_identity_sha256=payload.get("input_contract_identity_sha256"),
    )
    difference = _first_difference(dict(payload), expected)
    _require(difference is None, f"root-continuity values, denominator, or identity drift ({difference})")
    return deepcopy(expected)


def build_from_input_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a portable JSON input contract and build its sealed receipt."""

    _require(payload.get("schema_version") == ROOT_CONTINUITY_INPUT_SCHEMA, "wrong root-continuity input schema")
    _require(payload.get("blind_images_used") == 0, "root-continuity input is blind-tainted")
    _require(payload.get("independent_accuracy_claim_allowed") is False, "root-continuity development input was mislabelled independent")
    unsigned = deepcopy(dict(payload))
    observed_identity = unsigned.pop("input_contract_identity_sha256", None)
    _require(_is_sha256(observed_identity) and sha256_json(unsigned) == observed_identity, "root-continuity input identity mismatch")
    config = payload.get("metric_config")
    _require(isinstance(config, Mapping), "root-continuity metric_config missing")
    _require(set(config) == {"support_tolerance_um", "sampling_step_um"}, "root-continuity metric_config fields drift")
    result = build_root_continuity_assurance(
        records=payload.get("records", ()),
        source_units=payload.get("source_units", ()),
        reference_authority_sha256=str(payload.get("reference_authority_sha256", "")),
        prediction_authority_identity_sha256=str(payload.get("prediction_authority_identity_sha256", "")),
        support_tolerance_um=config["support_tolerance_um"],
        sampling_step_um=config["sampling_step_um"],
        input_contract_identity_sha256=str(observed_identity),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PHAxis primary-root continuity assurance")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = build_from_input_contract(read_json(args.input))
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], "images": result["source_unit_total"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "ROOT_CONTINUITY_ASSURANCE_SCHEMA",
    "ROOT_CONTINUITY_COORDINATE_SPACE",
    "ROOT_CONTINUITY_EVIDENCE_ROLE",
    "ROOT_CONTINUITY_INPUT_SCHEMA",
    "ROOT_CONTINUITY_PREDICTION_DEFINITION",
    "ROOT_CONTINUITY_REFERENCE_DEFINITION",
    "RootContinuityAssuranceError",
    "build_from_input_contract",
    "build_root_continuity_assurance",
    "validate_root_continuity_assurance",
]


if __name__ == "__main__":
    raise SystemExit(main())
