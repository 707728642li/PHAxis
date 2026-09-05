"""Root-hair attachment/base measurement assurance on existing annotations.

Two estimands are intentionally kept separate:

``attachment_proxy_threshold_selection``
    Base-only Hungarian matching at 5, 10, and 20 micrometres.  This is a
    development tolerance-sensitivity/operating-point diagnostic.  It must
    not be relabelled as the formal matched attachment accuracy.

``formal_matched_attachment_accuracy``
    Attachment error evaluated on the *same one-to-one biological hair
    identities* established by the locked tolerant-centreline matcher
    (20 micrometres, bidirectional coverage >= 0.25, proximal cosine >= 0,
    32 arc-length samples).  No second base-only rematching is allowed.  The
    position-error denominator is therefore explicit and auditable.

Manual hair annotations remain one attachment-to-visible-end centreline per
hair; this module never treats them as dense-width masks.  Its intended
producer wiring point is the formal QC-development evaluator immediately
after it has materialized prediction polylines and canonical, attachment-first
manual polylines in physical micrometre coordinates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_metrics import (
    biological_hair_presence_matcher_contract,
    match_biological_hair_presence,
    match_points,
    precision_recall_f1,
)
from .io import atomic_write_json, read_json, sha256_file, sha256_json


HAIR_ATTACHMENT_INPUT_SCHEMA = "PHAxis-hair-attachment-assurance-input-1.0"
HAIR_ATTACHMENT_ASSURANCE_SCHEMA = "PHAxis-hair-attachment-assurance-1.0"
HAIR_ATTACHMENT_EVIDENCE_ROLE = "annotated_qc_development_non_independent"
HAIR_ATTACHMENT_COORDINATE_SPACE = "physical_um_xy"
HAIR_POLYLINE_ORIENTATION = "attachment_to_visible_distal_endpoint"
PROXY_TOLERANCES_UM = (5.0, 10.0, 20.0)
SELECTED_PROXY_TOLERANCE_UM = 20.0
FORMAL_ATTACHMENT_TOLERANCE_UM = 20.0
_PRIMARY_MATCHER_CONTRACT = biological_hair_presence_matcher_contract()
FORMAL_MATCHER_CONFIG = {
    key: _PRIMARY_MATCHER_CONTRACT[key]
    for key in (
        "curve_tolerance_um",
        "minimum_truth_coverage",
        "minimum_prediction_coverage",
        "minimum_direction_cosine",
        "proximal_arc_fraction",
        "resample_points",
    )
}
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_828
BOOTSTRAP_METHOD = "source-image nonparametric percentile bootstrap"


class HairAttachmentAssuranceError(RuntimeError):
    """A hair-attachment geometry, matching, denominator, or identity failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HairAttachmentAssuranceError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _error_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "mean_um": None,
            "median_um": None,
            "p95_um": None,
            "max_um": None,
        }
    _require(np.all(np.isfinite(array)) and np.all(array >= 0.0), "attachment error vector is invalid")
    return {
        "n": int(len(array)),
        "mean_um": float(np.mean(array)),
        "median_um": float(np.median(array)),
        "p95_um": float(np.quantile(array, 0.95)),
        "max_um": float(np.max(array)),
    }


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
        _require(isinstance(raw, Mapping), "hair-attachment record is malformed")
        source_unit = str(raw.get("source_unit", ""))
        _require(source_unit in units and source_unit not in by_unit, f"unknown or duplicate source unit: {source_unit}")
        _require(raw.get("pair_type") == "hair_attachment", f"{source_unit}: pair_type drift")
        _require(raw.get("coordinate_space") == HAIR_ATTACHMENT_COORDINATE_SPACE, f"{source_unit}: coordinate space drift")
        _require(raw.get("polyline_orientation") == HAIR_POLYLINE_ORIENTATION, f"{source_unit}: polyline orientation drift")
        for field in (
            "source_image_sha256",
            "annotation_artifact_sha256",
            "prediction_artifact_sha256",
        ):
            _require(_is_sha256(raw.get(field)), f"{source_unit}: invalid {field}")
        predicted_raw = raw.get("predicted_polylines_xy_um")
        annotated_raw = raw.get("annotated_polylines_xy_um")
        _require(isinstance(predicted_raw, list) and isinstance(annotated_raw, list), f"{source_unit}: polyline collections must be lists")
        predicted = [
            _polyline(value, f"{source_unit}.predicted_polylines_xy_um[{index}]")
            for index, value in enumerate(predicted_raw)
        ]
        annotated = [
            _polyline(value, f"{source_unit}.annotated_polylines_xy_um[{index}]")
            for index, value in enumerate(annotated_raw)
        ]
        normalized = {
            "pair_type": "hair_attachment",
            "source_unit": source_unit,
            "source_image_sha256": str(raw["source_image_sha256"]),
            "coordinate_space": HAIR_ATTACHMENT_COORDINATE_SPACE,
            "polyline_orientation": HAIR_POLYLINE_ORIENTATION,
            "annotation_artifact_sha256": str(raw["annotation_artifact_sha256"]),
            "prediction_artifact_sha256": str(raw["prediction_artifact_sha256"]),
            "predicted_polylines_xy_um": [value.tolist() for value in predicted],
            "annotated_polylines_xy_um": [value.tolist() for value in annotated],
        }
        normalized["input_geometry_identity_sha256"] = sha256_json(normalized)
        by_unit[source_unit] = normalized
    _require(set(by_unit) == set(units), "hair-attachment source-unit denominator drift")
    normalized = [by_unit[source_unit] for source_unit in units]
    image_hashes = [row["source_image_sha256"] for row in normalized]
    _require(
        len(image_hashes) == len(set(image_hashes)),
        "hair-attachment bootstrap requires exactly one record per source image",
    )
    return normalized


def _measure_record(record: Mapping[str, Any]) -> dict[str, Any]:
    predicted = [np.asarray(value, dtype=np.float64) for value in record["predicted_polylines_xy_um"]]
    annotated = [np.asarray(value, dtype=np.float64) for value in record["annotated_polylines_xy_um"]]
    predicted_base = np.asarray([value[0] for value in predicted], dtype=np.float64).reshape((-1, 2))
    annotated_base = np.asarray([value[0] for value in annotated], dtype=np.float64).reshape((-1, 2))

    proxy: dict[str, Any] = {}
    for tolerance in PROXY_TOLERANCES_UM:
        predicted_index, annotated_index, errors = match_points(
            predicted_base, annotated_base, tolerance
        )
        metrics = precision_recall_f1(len(predicted_index), len(predicted), len(annotated))
        proxy[str(int(tolerance))] = {
            **metrics,
            "tolerance_um": tolerance,
            "position_error": _error_summary(errors.tolist()),
            "matched_pairs": [
                {
                    "predicted_index": int(prediction),
                    "annotated_index": int(annotation),
                    "attachment_error_um": float(error),
                }
                for prediction, annotation, error in zip(
                    predicted_index, annotated_index, errors, strict=True
                )
            ],
        }

    presence, matches = match_biological_hair_presence(
        predicted,
        annotated,
        1.0,
        FORMAL_MATCHER_CONFIG["curve_tolerance_um"],
        minimum_truth_coverage=FORMAL_MATCHER_CONFIG["minimum_truth_coverage"],
        minimum_prediction_coverage=FORMAL_MATCHER_CONFIG["minimum_prediction_coverage"],
        minimum_direction_cosine=FORMAL_MATCHER_CONFIG["minimum_direction_cosine"],
        proximal_arc_fraction=FORMAL_MATCHER_CONFIG["proximal_arc_fraction"],
        resample_points=FORMAL_MATCHER_CONFIG["resample_points"],
    )
    formal_matches: list[dict[str, Any]] = []
    for match in matches:
        predicted_index = int(match["predicted_index"])
        annotated_index = int(match["annotated_index"])
        attachment_error = float(
            np.linalg.norm(predicted[predicted_index][0] - annotated[annotated_index][0])
        )
        formal_matches.append(
            {
                **match,
                "attachment_error_um": attachment_error,
                "attachment_within_formal_tolerance": attachment_error
                <= FORMAL_ATTACHMENT_TOLERANCE_UM,
            }
        )
    formal_errors = [float(match["attachment_error_um"]) for match in formal_matches]
    attachment_true_positive = sum(
        bool(match["attachment_within_formal_tolerance"])
        for match in formal_matches
    )
    measured: dict[str, Any] = {
        "source_unit": record["source_unit"],
        "source_image_sha256": record["source_image_sha256"],
        "annotation_artifact_sha256": record["annotation_artifact_sha256"],
        "prediction_artifact_sha256": record["prediction_artifact_sha256"],
        "input_geometry_identity_sha256": record["input_geometry_identity_sha256"],
        "predicted_hairs": len(predicted),
        "annotated_hairs": len(annotated),
        "attachment_proxy_threshold_selection": proxy,
        "formal_matched_attachment_accuracy": {
            "formal_biological_presence": dict(presence),
            "attachment_tolerance_um": FORMAL_ATTACHMENT_TOLERANCE_UM,
            "attachment_qualified_identity": precision_recall_f1(
                attachment_true_positive, len(predicted), len(annotated)
            ),
            "attachment_position_error_on_all_formal_identity_matches": _error_summary(formal_errors),
            "formal_identity_matches": formal_matches,
        },
        "bootstrap_sufficient_statistics": {
            "predicted_hairs": len(predicted),
            "annotated_hairs": len(annotated),
            "formal_attachment_qualified_true_positive": int(
                attachment_true_positive
            ),
            "formal_attachment_errors_um": formal_errors,
        },
    }
    measured["row_identity_sha256"] = sha256_json(measured)
    return measured


def _pool_prf(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, float | int]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        selected.append(value)
    return precision_recall_f1(
        sum(int(value["tp"]) for value in selected),
        sum(int(value["n_pred"]) for value in selected),
        sum(int(value["n_gt"]) for value in selected),
    )


def _bootstrap_interval(
    point: float | None, estimates: np.ndarray
) -> dict[str, float | int | None]:
    finite = np.asarray(estimates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "point_estimate": point,
            "ci_low_2_5": None,
            "ci_high_97_5": None,
            "estimable_replicates": 0,
        }
    low, high = np.quantile(finite, (0.025, 0.975))
    return {
        "point_estimate": point,
        "ci_low_2_5": float(low),
        "ci_high_97_5": float(high),
        "estimable_replicates": int(len(finite)),
    }


def _hair_bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sufficient = [row["bootstrap_sufficient_statistics"] for row in rows]
    true_positive = np.asarray(
        [row["formal_attachment_qualified_true_positive"] for row in sufficient],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [row["predicted_hairs"] for row in sufficient], dtype=np.int64
    )
    annotated = np.asarray(
        [row["annotated_hairs"] for row in sufficient], dtype=np.int64
    )
    error_arrays = [
        np.asarray(row["formal_attachment_errors_um"], dtype=np.float64)
        for row in sufficient
    ]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(rows), size=(BOOTSTRAP_REPETITIONS, len(rows))
    )
    sampled_tp = np.sum(true_positive[indices], axis=1)
    sampled_predicted = np.sum(predicted[indices], axis=1)
    sampled_annotated = np.sum(annotated[indices], axis=1)
    precision = np.divide(
        sampled_tp,
        sampled_predicted,
        out=np.zeros(BOOTSTRAP_REPETITIONS, dtype=np.float64),
        where=sampled_predicted > 0,
    )
    recall = np.divide(
        sampled_tp,
        sampled_annotated,
        out=np.zeros(BOOTSTRAP_REPETITIONS, dtype=np.float64),
        where=sampled_annotated > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(BOOTSTRAP_REPETITIONS, dtype=np.float64),
        where=(precision + recall) > 0.0,
    )
    error_median = np.full(BOOTSTRAP_REPETITIONS, np.nan, dtype=np.float64)
    error_p95 = np.full(BOOTSTRAP_REPETITIONS, np.nan, dtype=np.float64)
    for replicate, sampled_images in enumerate(indices):
        blocks = [
            error_arrays[index]
            for index in sampled_images
            if len(error_arrays[index])
        ]
        if not blocks:
            continue
        values = np.concatenate(blocks)
        error_median[replicate] = float(np.median(values))
        error_p95[replicate] = float(np.quantile(values, 0.95))
    point = precision_recall_f1(
        int(np.sum(true_positive)), int(np.sum(predicted)), int(np.sum(annotated))
    )
    nonempty_errors = [values for values in error_arrays if len(values)]
    all_errors = (
        np.concatenate(nonempty_errors)
        if nonempty_errors
        else np.empty(0, dtype=np.float64)
    )
    point_median = float(np.median(all_errors)) if len(all_errors) else None
    point_p95 = float(np.quantile(all_errors, 0.95)) if len(all_errors) else None
    return {
        "formal_attachment_precision": _bootstrap_interval(
            float(point["precision"]), precision
        ),
        "formal_attachment_recall": _bootstrap_interval(
            float(point["recall"]), recall
        ),
        "formal_attachment_f1": _bootstrap_interval(float(point["f1"]), f1),
        "formal_attachment_error_median_um": _bootstrap_interval(
            point_median, error_median
        ),
        "formal_attachment_error_p95_um": _bootstrap_interval(
            point_p95, error_p95
        ),
    }


def build_hair_attachment_assurance(
    *,
    records: Sequence[Mapping[str, Any]],
    source_units: Sequence[str],
    annotation_authority_sha256: str,
    prediction_authority_identity_sha256: str,
    input_contract_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic receipt for proxy and formally matched attachment metrics."""

    _require(_is_sha256(annotation_authority_sha256), "annotation authority SHA-256 missing")
    _require(_is_sha256(prediction_authority_identity_sha256), "prediction authority identity missing")
    _require(
        input_contract_identity_sha256 is None
        or _is_sha256(input_contract_identity_sha256),
        "input-contract identity is invalid",
    )
    normalized = _normalise_records(records, source_units)
    rows = [_measure_record(record) for record in normalized]

    proxy_summary: dict[str, Any] = {}
    for tolerance in PROXY_TOLERANCES_UM:
        key = str(int(tolerance))
        errors = [
            float(match["attachment_error_um"])
            for row in rows
            for match in row["attachment_proxy_threshold_selection"][key]["matched_pairs"]
        ]
        proxy_summary[key] = {
            **_pool_prf(rows, ("attachment_proxy_threshold_selection", key)),
            "tolerance_um": tolerance,
            "position_error": _error_summary(errors),
        }

    formal_errors = [
        float(match["attachment_error_um"])
        for row in rows
        for match in row["formal_matched_attachment_accuracy"]["formal_identity_matches"]
    ]
    bootstrap_intervals = _hair_bootstrap(rows)
    source_unit_rows = [
        {"source_unit": row["source_unit"], "source_image_sha256": row["source_image_sha256"]}
        for row in normalized
    ]
    source_unit_set_identity = sha256_json(source_unit_rows)
    input_geometry_set_identity = sha256_json(normalized)
    implementation_identity = sha256_file(Path(__file__))
    payload: dict[str, Any] = {
        "schema_version": HAIR_ATTACHMENT_ASSURANCE_SCHEMA,
        "status": "completed",
        "scope": "QC-development root-hair attachment assurance; non-independent",
        "evidence_role": HAIR_ATTACHMENT_EVIDENCE_ROLE,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
        "val_labels_used_for_training": False,
        "provider_equivalence_used_as_accuracy": False,
        "coordinate_space": HAIR_ATTACHMENT_COORDINATE_SPACE,
        "polyline_orientation": HAIR_POLYLINE_ORIENTATION,
        "metric_contract": {
            "manual_annotation_semantics": "one centreline polyline per visible biological root hair; no width truth",
            "attachment_proxy_threshold_selection": {
                "role": "development-only base-proxy tolerance sensitivity; not formal matched attachment accuracy",
                "tolerances_um": list(PROXY_TOLERANCES_UM),
                "selected_tolerance_um": SELECTED_PROXY_TOLERANCE_UM,
                "matching": "maximum-cardinality one-to-one Hungarian matching on attachment/base distance",
            },
            "formal_biological_identity_matcher": dict(FORMAL_MATCHER_CONFIG),
            "formal_attachment_tolerance_um": FORMAL_ATTACHMENT_TOLERANCE_UM,
            "formal_position_error_denominator": (
                "all one-to-one identities returned by the formal tolerant biological-presence matcher; no base-only rematching"
            ),
            "threshold_selection_used_as_formal_accuracy": False,
        },
        "source_unit_total": len(rows),
        "source_unit_set_identity_sha256": source_unit_set_identity,
        "input_geometry_set_identity_sha256": input_geometry_set_identity,
        "input_contract_identity_sha256": input_contract_identity_sha256,
        "annotation_authority_sha256": annotation_authority_sha256,
        "prediction_authority_identity_sha256": prediction_authority_identity_sha256,
        "implementation_sha256": implementation_identity,
        "provenance": {
            "producer_wiring_point": (
                "scripts/phaxis/evaluate_stageb_train399_qcdev44.py, after each "
                "selected-model prediction and canonical attachment-first centreline "
                "set have been converted to physical micrometre coordinates"
            ),
            "annotation_authority_sha256": annotation_authority_sha256,
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
            "cluster_rule": (
                "resample source images with replacement and carry all formal "
                "attachment matches/errors from each sampled image; individual hairs "
                "are never bootstrap units"
            ),
            "sufficient_statistics_location": "per_image[*].bootstrap_sufficient_statistics",
        },
        "summary": {
            "images": len(rows),
            "predicted_hairs": sum(int(row["predicted_hairs"]) for row in rows),
            "annotated_hairs": sum(int(row["annotated_hairs"]) for row in rows),
            "attachment_proxy_threshold_selection": proxy_summary,
            "formal_matched_attachment_accuracy": {
                "formal_biological_presence": _pool_prf(
                    rows,
                    ("formal_matched_attachment_accuracy", "formal_biological_presence"),
                ),
                "attachment_tolerance_um": FORMAL_ATTACHMENT_TOLERANCE_UM,
                "attachment_qualified_identity": _pool_prf(
                    rows,
                    ("formal_matched_attachment_accuracy", "attachment_qualified_identity"),
                ),
                "attachment_position_error_on_all_formal_identity_matches": _error_summary(formal_errors),
                "bootstrap_95ci": bootstrap_intervals,
            },
        },
        "per_image": rows,
        "per_image_set_identity_sha256": sha256_json(rows),
    }
    payload["hair_attachment_assurance_identity_sha256"] = sha256_json(payload)
    return payload


def validate_hair_attachment_assurance(
    payload: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    source_units: Sequence[str],
    annotation_authority_sha256: str,
    prediction_authority_identity_sha256: str,
) -> dict[str, Any]:
    """Fail closed by recomputing matches, errors, denominators, and hashes."""

    _require(payload.get("schema_version") == HAIR_ATTACHMENT_ASSURANCE_SCHEMA, "hair-attachment assurance schema drift")
    _require(payload.get("evidence_role") == HAIR_ATTACHMENT_EVIDENCE_ROLE, "wrong hair-attachment evidence role")
    _require(payload.get("independent_accuracy_claim_allowed") is False, "development assurance was mislabelled independent")
    _require(payload.get("blind_images_used") == 0, "blind images entered hair-attachment assurance")
    metric_contract = payload.get("metric_contract")
    _require(isinstance(metric_contract, Mapping), "hair-attachment metric contract missing")
    _require(metric_contract.get("threshold_selection_used_as_formal_accuracy") is False, "attachment-proxy selection masquerades as formal accuracy")
    expected = build_hair_attachment_assurance(
        records=records,
        source_units=source_units,
        annotation_authority_sha256=annotation_authority_sha256,
        prediction_authority_identity_sha256=prediction_authority_identity_sha256,
        input_contract_identity_sha256=payload.get("input_contract_identity_sha256"),
    )
    difference = _first_difference(dict(payload), expected)
    _require(difference is None, f"hair-attachment values, denominator, or identity drift ({difference})")
    return deepcopy(expected)


def build_from_input_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a portable JSON input contract and build its sealed receipt."""

    _require(payload.get("schema_version") == HAIR_ATTACHMENT_INPUT_SCHEMA, "wrong hair-attachment input schema")
    _require(payload.get("blind_images_used") == 0, "hair-attachment input is blind-tainted")
    _require(payload.get("independent_accuracy_claim_allowed") is False, "hair-attachment development input was mislabelled independent")
    unsigned = deepcopy(dict(payload))
    observed_identity = unsigned.pop("input_contract_identity_sha256", None)
    _require(_is_sha256(observed_identity) and sha256_json(unsigned) == observed_identity, "hair-attachment input identity mismatch")
    config = payload.get("metric_config")
    expected_config = {
        "proxy_tolerances_um": list(PROXY_TOLERANCES_UM),
        "selected_proxy_tolerance_um": SELECTED_PROXY_TOLERANCE_UM,
        "formal_attachment_tolerance_um": FORMAL_ATTACHMENT_TOLERANCE_UM,
        "formal_matcher": dict(FORMAL_MATCHER_CONFIG),
    }
    _require(config == expected_config, "hair-attachment metric_config differs from the locked formal contract")
    result = build_hair_attachment_assurance(
        records=payload.get("records", ()),
        source_units=payload.get("source_units", ()),
        annotation_authority_sha256=str(payload.get("annotation_authority_sha256", "")),
        prediction_authority_identity_sha256=str(payload.get("prediction_authority_identity_sha256", "")),
        input_contract_identity_sha256=str(observed_identity),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PHAxis hair attachment/base assurance")
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
    "FORMAL_ATTACHMENT_TOLERANCE_UM",
    "FORMAL_MATCHER_CONFIG",
    "HAIR_ATTACHMENT_ASSURANCE_SCHEMA",
    "HAIR_ATTACHMENT_COORDINATE_SPACE",
    "HAIR_ATTACHMENT_EVIDENCE_ROLE",
    "HAIR_ATTACHMENT_INPUT_SCHEMA",
    "HAIR_POLYLINE_ORIENTATION",
    "PROXY_TOLERANCES_UM",
    "SELECTED_PROXY_TOLERANCE_UM",
    "HairAttachmentAssuranceError",
    "build_from_input_contract",
    "build_hair_attachment_assurance",
    "validate_hair_attachment_assurance",
]


if __name__ == "__main__":
    raise SystemExit(main())
