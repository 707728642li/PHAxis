"""Evaluate a strict train399-only Stage-B ensemble on locked QC-development44.

The primary endpoint is one-to-one tolerant biological-hair presence: a
prediction must correspond to a meaningful part of the annotated single-trunk
centreline, but distal endpoints and complete-line coincidence are not hard
detection gates.  Attachment identity and strict whole-line correspondence are
reported separately.  Ground truth is loaded directly from the hash-verified
canonical dataset vectors and oriented by the same root-polygon endpoint rule
used to construct train399 targets; historical OOF predictions or label
carriers are not an authority for this evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.evaluation_metrics import (  # noqa: E402
    biological_hair_presence_matcher_contract,
    evaluate_image,
    match_biological_hair_presence,
    presence_match_strict,
    prf,
)
from phaxis.hair_stageb.evaluation_inference import (  # noqa: E402
    EVALUATION_ARTIFACT_ROLE,
    EVALUATION_DETECTION_SCHEMA,
    EVALUATION_RUN_SCHEMA,
    build_evaluation_gate_binding,
    validate_evaluation_detection_payload,
    validate_evaluation_inference_summary,
)
from phaxis.hair_stageb.candidate_bundle import read_candidate_manifest  # noqa: E402
from phaxis.hair_stageb.canonical_ground_truth import (  # noqa: E402
    load_canonical_qcdev_ground_truth,
)
from phaxis.hair_stageb.selection import (  # noqa: E402
    read_selection_receipt,
    validate_selected_operating_point_binding,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


TOLERANCES = (5.0, 10.0, 20.0)
COUNT_BINS = (
    ("negative_0", 0, 0),
    ("sparse_1_49", 1, 49),
    ("medium_50_99", 50, 99),
    ("dense_100_199", 100, 199),
    ("very_dense_ge200", 200, None),
)

LEGACY_HYBRID_COMPARATOR_SCHEMA = (
    "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
)
LEGACY_HYBRID_IDENTITY_VARIANT = "hybrid_verified_increment"
LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256 = (
    "ede309b8a828aec35be64d9f8afbc2ac9bf92b5a9e1b1b262d5acf603a746f36"
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _csv_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["task_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"duplicate task_id in {path}")
    return indexed


def _locked_ids(path: Path) -> list[str]:
    values = [value.strip() for value in path.read_text(encoding="utf-8").splitlines()]
    task_ids = [value for value in values if value]
    if len(task_ids) != 44 or len(set(task_ids)) != 44:
        raise RuntimeError("locked QC-development set must contain 44 unique task IDs")
    return task_ids


def _load_formal_model_gate(
    *,
    candidate_manifest_path: Path,
    selected_model_metadata_path: Path,
    selection_receipt_path: Path,
) -> tuple[dict, dict, dict]:
    """Load and cross-bind the three immutable train399 deployment receipts."""

    candidate_manifest = read_candidate_manifest(candidate_manifest_path)
    selected_model_metadata = read_json(selected_model_metadata_path)
    selection_receipt = read_selection_receipt(selection_receipt_path)
    validate_selected_operating_point_binding(
        candidate_manifest=candidate_manifest,
        selected_model_metadata=selected_model_metadata,
        selection_receipt=selection_receipt,
        selection_receipt_file_sha256=sha256_file(selection_receipt_path),
    )
    return candidate_manifest, selected_model_metadata, selection_receipt


def _load_evaluation_inference_authority(
    *,
    summary_path: Path,
    detections_dir: Path,
    candidate_manifest_path: Path,
    selected_model_metadata_path: Path,
    selection_receipt_path: Path,
) -> tuple[dict, dict, dict[str, dict]]:
    """Validate the exact44 non-production run that breaks the proposal cycle."""

    summary = read_json(summary_path)
    embedded_gate = summary.get("evaluation_gate_binding")
    checkpoint_locks = (
        embedded_gate.get("checkpoint_locks")
        if isinstance(embedded_gate, dict)
        else None
    )
    if not isinstance(checkpoint_locks, list):
        raise RuntimeError("evaluation inference summary has no checkpoint locks")
    (
        _candidate_manifest,
        selected_model_metadata,
        selection_receipt,
        expected_gate,
    ) = build_evaluation_gate_binding(
        candidate_manifest_path=candidate_manifest_path,
        selected_model_metadata_path=selected_model_metadata_path,
        selection_receipt_path=selection_receipt_path,
        checkpoint_locks=checkpoint_locks,
    )
    validate_evaluation_inference_summary(
        summary, expected_evaluation_gate=expected_gate
    )
    records = summary["records"]
    expected_names = [f"{record['task_id']}.json" for record in records]
    observed_names = sorted(path.name for path in detections_dir.glob("*.json"))
    if sorted(expected_names) != observed_names:
        raise RuntimeError(
            "evaluation-only detection directory differs from the sealed exact44 run"
        )
    source_locks = {
        row["task_id"]: row for row in selection_receipt["task_image_locks"]
    }
    for record in records:
        task_id = record["task_id"]
        detection_path = detections_dir / f"{task_id}.json"
        if sha256_file(detection_path) != record["evaluation_detection_file_sha256"]:
            raise RuntimeError(f"{task_id}: evaluation-only detection file hash drift")
        payload = read_json(detection_path)
        validate_evaluation_detection_payload(
            payload,
            expected_task_id=task_id,
            expected_image_sha256=source_locks[task_id]["source_image_sha256"],
            expected_model_metadata=selected_model_metadata,
            expected_evaluation_gate=expected_gate,
        )
        if payload.get("evaluation_detection_identity_sha256") != record.get(
            "evaluation_detection_identity_sha256"
        ):
            raise RuntimeError(f"{task_id}: evaluation-only detection identity drift")
    return summary, expected_gate, {record["task_id"]: record for record in records}


def _stageb_prediction(
    payload: dict, metadata: dict[str, str]
) -> dict[str, np.ndarray]:
    """Return Stage-B geometry in physical micrometre coordinates.

    The resize requested at 2 um/px cannot generally be represented by integer
    image dimensions.  New train399 payloads therefore carry the actually
    realized x/y resize factors.  Converting through source pixels prevents the
    old 0.5--1 um far-edge registration drift from re-entering evaluation.
    """

    if payload.get("blind_images_used") != 0:
        raise RuntimeError(f"{payload.get('task_id')}: blind_images_used must be zero")
    model = payload.get("model", {})
    if model.get("checkpoint_policy") != "five_seed_train399_last_epoch_60":
        raise RuntimeError(
            f"{payload.get('task_id')}: not a strict train399-only five-seed payload"
        )
    if int(model.get("training_images", -1)) != 399:
        raise RuntimeError(f"{payload.get('task_id')}: training_images is not 399")
    coordinate = payload.get("coordinate_space", {})
    if not np.isclose(float(coordinate.get("working_um_per_px", -1)), 2.0):
        raise RuntimeError(f"{payload.get('task_id')}: unexpected working scale")
    working_shape = tuple(int(value) for value in coordinate.get("working_shape", ()))
    if len(working_shape) != 2 or min(working_shape) <= 0:
        raise RuntimeError(f"{payload.get('task_id')}: invalid working shape")
    source_shape = (int(metadata["image_height"]), int(metadata["image_width"]))
    expected_scale_xy = np.asarray(
        [working_shape[1] / source_shape[1], working_shape[0] / source_shape[0]],
        dtype=np.float64,
    )
    scale_xy = np.asarray(
        coordinate.get("source_to_working_scale_xy", ()), dtype=np.float64
    )
    if scale_xy.shape != (2,) or not np.allclose(
        scale_xy, expected_scale_xy, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            f"{payload.get('task_id')}: realized source/working scale mismatch"
        )
    source_um_per_px = float(metadata["source_um_per_px"])
    if not np.isclose(
        float(coordinate.get("source_um_per_px", np.nan)),
        source_um_per_px,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{payload.get('task_id')}: payload physical scale differs from canonical metadata"
        )

    def working_to_um(values) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        return points / scale_xy[None, :] * source_um_per_px

    detections = payload.get("detections", [])
    if int(payload.get("n", -1)) != len(detections):
        raise RuntimeError(f"{payload.get('task_id')}: detection count mismatch")
    return {
        "base": working_to_um(
            [item["base_xy_working"] for item in detections]
        ),
        "tip": working_to_um(
            [item["tip_xy_working"] for item in detections]
        ),
        "score": np.asarray(
            [item["score"] for item in detections], dtype=np.float64
        ),
        "length_um": np.asarray(
            [item["predicted_length_um"] for item in detections], dtype=np.float64
        ),
    }


def _hybrid_prediction(payload: dict, source_um_per_px: float) -> dict[str, np.ndarray]:
    """Return the Hybrid-Max source-coordinate curves in micrometres."""

    polylines = []
    for hair in payload["identity_hairs"]:
        polyline = np.asarray(hair["points_xy"], dtype=np.float64)
        if (
            polyline.ndim != 2
            or polyline.shape[1:] != (2,)
            or len(polyline) < 2
            or not np.all(np.isfinite(polyline))
        ):
            raise RuntimeError(
                f"{payload.get('task_id')}: invalid legacy Hybrid identity curve"
            )
        polylines.append(polyline * source_um_per_px)
    if polylines:
        base = np.asarray([polyline[0] for polyline in polylines], dtype=np.float64)
        tip = np.asarray([polyline[-1] for polyline in polylines], dtype=np.float64)
        lengths = np.asarray(
            [
                np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum()
                for polyline in polylines
            ],
            dtype=np.float64,
        )
    else:
        base = np.empty((0, 2), dtype=np.float64)
        tip = np.empty((0, 2), dtype=np.float64)
        lengths = np.empty((0,), dtype=np.float64)
    return {
        "base": base,
        "tip": tip,
        "polys": polylines,
        "score": np.ones(len(base), dtype=np.float64),
        "length_um": lengths,
    }


def _validate_legacy_hybrid_comparator(
    payload: dict, *, task_id: str, expected_image_sha256: str | None
) -> None:
    """Reject PHAxis/Stage-B fusion payloads masquerading as the legacy comparator."""

    if (
        payload.get("schema_version") != LEGACY_HYBRID_COMPARATOR_SCHEMA
        or payload.get("task_id") != task_id
        or payload.get("source_image_sha256") != expected_image_sha256
        or payload.get("blind_images_used") != 0
        or payload.get("canonical_annotations_read_during_inference") is not False
        or payload.get("identity_hair_variant") != LEGACY_HYBRID_IDENTITY_VARIANT
        or payload.get("count_hair_variant") != LEGACY_HYBRID_IDENTITY_VARIANT
        or "phaxis" in payload
    ):
        raise RuntimeError(f"{task_id}: legacy Hybrid comparator identity is invalid")
    hairs = payload.get("identity_hairs")
    if not isinstance(hairs, list):
        raise RuntimeError(f"{task_id}: legacy Hybrid identity_hairs must be a list")
    prohibited_sources = {"phaxis_stage_b_train399", "rhaxiscc_stage_b"}
    if any(hair.get("source") in prohibited_sources for hair in hairs):
        raise RuntimeError(
            f"{task_id}: Stage-B identities cannot enter the legacy Hybrid comparator"
        )


def _evaluate_prediction(prediction: dict, ground_truth: dict) -> dict:
    """Evaluate one expert with all metric roles kept explicitly separate."""

    evaluated = evaluate_image(prediction, ground_truth, 1.0, TOLERANCES)
    predicted_polylines = prediction.get("polys")
    if predicted_polylines is None:
        predicted_polylines = [
            np.stack((base, tip))
            for base, tip in zip(
                prediction["base"], prediction["tip"], strict=True
            )
        ]
    matcher = biological_hair_presence_matcher_contract()
    return {
        "n_pred": int(evaluated["n_pred"]),
        "n_gt": int(evaluated["n_gt"]),
        "base_tp": {
            tolerance: int(evaluated["tol"][tolerance]["tp"])
            for tolerance in TOLERANCES
        },
        "biological_presence_tp": {
            tolerance: int(
                match_biological_hair_presence(
                    predicted_polylines,
                    ground_truth["polys"],
                    1.0,
                    tolerance,
                    minimum_truth_coverage=matcher["minimum_truth_coverage"],
                    minimum_prediction_coverage=matcher[
                        "minimum_prediction_coverage"
                    ],
                    minimum_direction_cosine=matcher[
                        "minimum_direction_cosine"
                    ],
                    proximal_arc_fraction=matcher["proximal_arc_fraction"],
                    resample_points=matcher["resample_points"],
                )[0]["tp"]
            )
            for tolerance in TOLERANCES
        },
        "strict_tp": {
            tolerance: int(
                presence_match_strict(
                    prediction["base"],
                    prediction["tip"],
                    ground_truth["polys"],
                    ground_truth["base"],
                    ground_truth["tip"],
                    1.0,
                    tolerance,
                )
            )
            for tolerance in TOLERANCES
        },
    }


def _pool(rows: list[dict], expert: str, *, prf) -> dict:
    predicted = int(sum(row[expert]["n_pred"] for row in rows))
    truth = int(sum(row[expert]["n_gt"] for row in rows))
    base = {
        str(int(tolerance)): prf(
            int(sum(row[expert]["base_tp"][tolerance] for row in rows)),
            predicted,
            truth,
        )
        for tolerance in TOLERANCES
    }
    biological_presence = {
        str(int(tolerance)): prf(
            int(
                sum(
                    row[expert]["biological_presence_tp"][tolerance]
                    for row in rows
                )
            ),
            predicted,
            truth,
        )
        for tolerance in TOLERANCES
    }
    strict = {
        str(int(tolerance)): prf(
            int(sum(row[expert]["strict_tp"][tolerance] for row in rows)),
            predicted,
            truth,
        )
        for tolerance in TOLERANCES
    }
    pred_counts = np.asarray([row[expert]["n_pred"] for row in rows], dtype=float)
    gt_counts = np.asarray([row[expert]["n_gt"] for row in rows], dtype=float)
    error = pred_counts - gt_counts
    if len(rows) >= 2 and np.std(pred_counts) > 0 and np.std(gt_counts) > 0:
        pearson = float(np.corrcoef(pred_counts, gt_counts)[0, 1])
        covariance = float(np.cov(pred_counts, gt_counts, ddof=1)[0, 1])
        ccc = float(
            2.0
            * covariance
            / (
                np.var(pred_counts, ddof=1)
                + np.var(gt_counts, ddof=1)
                + (np.mean(pred_counts) - np.mean(gt_counts)) ** 2
            )
        )
    else:
        pearson = float("nan")
        ccc = float("nan")
    return {
        "images": len(rows),
        "predicted_hairs": predicted,
        "ground_truth_hairs": truth,
        "tolerant_biological_presence": biological_presence,
        "identity_attachment_proxy": base,
        "strict_whole_line_correspondence": strict,
        "count": {
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "pearson_r": pearson,
            "ccc": ccc,
        },
    }


def _bootstrap(rows: list[dict], *, prf, repetitions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = ("stageb_train399", "hybrid_max")
    metrics = {
        name: {
            "biological_presence_f1_20um": [],
            "identity_f1_20um": [],
            "count_mae": [],
            "count_ccc": [],
        }
        for name in names
    }
    delta = {
        "biological_presence_f1_20um": [],
        "identity_f1_20um": [],
        "count_mae": [],
        "count_ccc": [],
    }
    for _ in range(repetitions):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        pooled = {name: _pool(sampled, name, prf=prf) for name in names}
        for name in names:
            metrics[name]["biological_presence_f1_20um"].append(
                pooled[name]["tolerant_biological_presence"]["20"]["f1"]
            )
            metrics[name]["identity_f1_20um"].append(
                pooled[name]["identity_attachment_proxy"]["20"]["f1"]
            )
            metrics[name]["count_mae"].append(pooled[name]["count"]["mae"])
            metrics[name]["count_ccc"].append(pooled[name]["count"]["ccc"])
        for key in delta:
            delta[key].append(metrics["stageb_train399"][key][-1] - metrics["hybrid_max"][key][-1])

    def interval(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return {
            "lower_2_5": float(np.quantile(finite, 0.025)),
            "upper_97_5": float(np.quantile(finite, 0.975)),
        }

    return {
        "method": "paired image-level nonparametric bootstrap",
        "repetitions": repetitions,
        "seed": seed,
        "experts": {
            name: {key: interval(values) for key, values in by_metric.items()}
            for name, by_metric in metrics.items()
        },
        "delta_stageb_train399_minus_hybrid": {
            key: interval(values) for key, values in delta.items()
        },
    }


def _density_label(count: int) -> str:
    for label, lower, upper in COUNT_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    raise AssertionError(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument(
        "--evaluation-inference-summary",
        type=Path,
        required=True,
        help=(
            "sealed exact-QCdev44 evaluation-only inference summary; production "
            "Stage-B detections are not accepted as pre-proposal evidence"
        ),
    )
    parser.add_argument("--hybrid-predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--locked-val-ids", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--selected-model-metadata", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap_repetitions < 200:
        raise ValueError("bootstrap-repetitions must be at least 200")

    (
        candidate_manifest,
        selected_model_metadata,
        selection_receipt,
    ) = _load_formal_model_gate(
        candidate_manifest_path=args.candidate_manifest,
        selected_model_metadata_path=args.selected_model_metadata,
        selection_receipt_path=args.selection_receipt,
    )
    (
        evaluation_inference_summary,
        evaluation_gate,
        evaluation_inference_records,
    ) = _load_evaluation_inference_authority(
        summary_path=args.evaluation_inference_summary,
        detections_dir=args.detections,
        candidate_manifest_path=args.candidate_manifest,
        selected_model_metadata_path=args.selected_model_metadata,
        selection_receipt_path=args.selection_receipt,
    )

    metadata = _csv_index(args.dataset_manifest)
    task_ids = _locked_ids(args.locked_val_ids)
    canonical_gt, canonical_provenance = load_canonical_qcdev_ground_truth(
        dataset_root=args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        split_manifest=args.split_manifest,
        expected_task_ids=task_ids,
    )
    if task_ids != list(canonical_gt):
        raise RuntimeError("locked task IDs differ from canonical QC-development44")
    training_lock = candidate_manifest["identity_payload"]["training_lock"]
    for provenance_field, lock_field in (
        ("dataset_manifest_sha256", "dataset_manifest_sha256"),
        ("split_manifest_sha256", "split_manifest_sha256"),
        ("integrity_manifest_sha256", "integrity_manifest_sha256"),
    ):
        if canonical_provenance[provenance_field] != training_lock[lock_field]:
            raise RuntimeError(
                f"canonical evaluation truth differs from candidate training lock: "
                f"{provenance_field}"
            )
    if (
        canonical_provenance["canonical_ground_truth_lock_identity_sha256"]
        != selection_receipt["canonical_ground_truth_lock_identity_sha256"]
    ):
        raise RuntimeError(
            "canonical evaluation geometry/scale differs from the selection receipt"
        )
    if [row["task_id"] for row in selection_receipt["task_image_locks"]] != task_ids:
        raise RuntimeError("evaluation task order differs from the selection receipt")

    rows = []
    detection_file_locks = []
    hybrid_file_locks = []
    checkpoint_sets = set()
    for task_id in task_ids:
        detection_path = args.detections / f"{task_id}.json"
        payload = read_json(detection_path)
        if payload.get("task_id") != task_id:
            raise RuntimeError(f"{task_id}: detection task identity mismatch")
        expected_image_sha = metadata[task_id].get("image_sha256") or metadata[task_id].get("source_image_sha256")
        if expected_image_sha and payload.get("source_image_sha256") != expected_image_sha:
            raise RuntimeError(f"{task_id}: source image hash mismatch")
        sealed_record = evaluation_inference_records.get(task_id)
        if not isinstance(sealed_record, dict):
            raise RuntimeError(f"{task_id}: absent from evaluation inference summary")
        if sha256_file(detection_path) != sealed_record.get(
            "evaluation_detection_file_sha256"
        ):
            raise RuntimeError(f"{task_id}: evaluation-only detection file hash drift")
        core_payload = validate_evaluation_detection_payload(
            payload,
            expected_task_id=task_id,
            expected_image_sha256=expected_image_sha,
            expected_model_metadata=selected_model_metadata,
            expected_evaluation_gate=evaluation_gate,
        )
        if payload.get("evaluation_detection_identity_sha256") != sealed_record.get(
            "evaluation_detection_identity_sha256"
        ):
            raise RuntimeError(f"{task_id}: evaluation detection identity drift")
        detection_file_locks.append(
            {"task_id": task_id, "sha256": sha256_file(detection_path)}
        )
        checkpoint_sets.add(tuple(core_payload["model"].get("checkpoint_sha256", [])))
        predictions = {
            "stageb_train399": _stageb_prediction(core_payload, metadata[task_id]),
            "hybrid_max": None,
        }
        hybrid_path = args.hybrid_predictions / f"{task_id}.json"
        hybrid_payload = read_json(hybrid_path)
        _validate_legacy_hybrid_comparator(
            hybrid_payload,
            task_id=task_id,
            expected_image_sha256=expected_image_sha,
        )
        hybrid_file_locks.append(
            {"task_id": task_id, "sha256": sha256_file(hybrid_path)}
        )
        predictions["hybrid_max"] = _hybrid_prediction(
            hybrid_payload,
            float(metadata[task_id]["source_um_per_px"]),
        )
        canonical = canonical_gt[task_id]
        gt = {
            "base": np.asarray(canonical["base"], dtype=np.float64),
            "tip": np.asarray(canonical["tip"], dtype=np.float64),
            "polys": [np.asarray(polyline, dtype=np.float64) for polyline in canonical["polys"]],
            "length_um": np.asarray(canonical["length_um"], dtype=np.float64),
        }
        row = {
            "task_id": task_id,
            "quality_band": metadata[task_id]["acquisition_quality_band"],
            "fully_manual": bool(canonical["fully_manual"]),
            "canonical_annotation_sha256": canonical[
                "canonical_annotation_sha256"
            ],
            "canonical_physical_geometry_identity_sha256": canonical[
                "physical_geometry_identity_sha256"
            ],
            "density": _density_label(len(gt["base"])),
        }
        for expert, prediction in predictions.items():
            row[expert] = _evaluate_prediction(prediction, gt)
        rows.append(row)
    if len(checkpoint_sets) != 1 or len(next(iter(checkpoint_sets), ())) != 5:
        raise RuntimeError("all detections must carry one identical five-checkpoint hash set")
    if (
        sha256_json(detection_file_locks)
        != evaluation_inference_summary[
            "evaluation_detection_set_identity_sha256"
        ]
    ):
        raise RuntimeError(
            "evaluation input files differ from the sealed evaluation-only run"
        )
    hybrid_prediction_set_identity = sha256_json(hybrid_file_locks)
    if (
        hybrid_prediction_set_identity
        != LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
    ):
        raise RuntimeError(
            "legacy Hybrid comparator differs from the locked QC-development44 set"
        )

    overall = {
        expert: _pool(rows, expert, prf=prf)
        for expert in ("stageb_train399", "hybrid_max")
    }
    groups = {}
    for label in ("Q2_25_50", "Q3_50_75", "Q4_75_100"):
        selected = [row for row in rows if row["quality_band"] == label]
        groups[f"quality_{label}"] = {
            expert: _pool(selected, expert, prf=prf) for expert in overall
        }
    for label, _lower, _upper in COUNT_BINS:
        selected = [row for row in rows if row["density"] == label]
        if selected:
            groups[f"density_{label}"] = {
                expert: _pool(selected, expert, prf=prf) for expert in overall
            }
    for label, value in (("fully_manual", True), ("model_assisted_refined", False)):
        selected = [row for row in rows if row["fully_manual"] is value]
        if selected:
            groups[f"annotation_{label}"] = {
                expert: _pool(selected, expert, prf=prf) for expert in overall
            }

    stageb = overall["stageb_train399"]
    hybrid = overall["hybrid_max"]
    primary_matcher = biological_hair_presence_matcher_contract()
    payload = {
        "schema_version": "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
        "status": "completed",
        "scope": "locked overlay-visible QC-development44; not independent accuracy",
        "metric_hierarchy": {
            "primary": (
                "one-to-one tolerant biological-hair presence; bidirectional "
                "partial centreline coverage without endpoint gates"
            ),
            "primary_matcher_contract": primary_matcher,
            "primary_matcher_contract_sha256": sha256_json(primary_matcher),
            "primary_minimum_truth_coverage": primary_matcher[
                "minimum_truth_coverage"
            ],
            "primary_minimum_prediction_coverage": primary_matcher[
                "minimum_prediction_coverage"
            ],
            "primary_minimum_direction_cosine": primary_matcher[
                "minimum_direction_cosine"
            ],
            "secondary": (
                "attachment/base identity; strict whole-line correspondence; "
                "distal endpoint and length geometry"
            ),
            "primary_tolerance_um": primary_matcher["curve_tolerance_um"],
            "coordinate_evaluation": (
                "physical_um_after_per_axis_realized_resize_correction"
            ),
        },
        "training_contract": {
            "training_images": 399,
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "ensemble": "five fixed-seed members trained on the same 399 images",
            "checkpoint_sha256": list(next(iter(checkpoint_sets))),
            "candidate_bundle_identity_sha256": candidate_manifest[
                "candidate_bundle_identity_sha256"
            ],
            "selected_model_metadata_identity_sha256": selected_model_metadata[
                "selected_model_metadata_identity_sha256"
            ],
            "selection_receipt_identity_sha256": selection_receipt[
                "selection_receipt_identity_sha256"
            ],
            "evaluation_gate_identity_sha256": evaluation_gate[
                "evaluation_gate_identity_sha256"
            ],
            "evaluation_inference_summary_identity_sha256": (
                evaluation_inference_summary[
                    "evaluation_inference_summary_identity_sha256"
                ]
            ),
        },
        "evaluation_inference_authority": {
            "schema_version": EVALUATION_RUN_SCHEMA,
            "artifact_role": EVALUATION_ARTIFACT_ROLE,
            "evaluation_detection_schema_version": EVALUATION_DETECTION_SCHEMA,
            "evaluation_inference_summary_sha256": sha256_file(
                args.evaluation_inference_summary
            ),
            "evaluation_inference_summary_identity_sha256": (
                evaluation_inference_summary[
                    "evaluation_inference_summary_identity_sha256"
                ]
            ),
            "evaluation_gate_identity_sha256": evaluation_gate[
                "evaluation_gate_identity_sha256"
            ],
            "evaluation_detection_set_identity_sha256": (
                evaluation_inference_summary[
                    "evaluation_detection_set_identity_sha256"
                ]
            ),
            "model_contract_proposal_required_for_artifact": False,
            "model_contract_proposal_present": False,
            "production_consumption_allowed": False,
            "fusion_consumption_allowed": False,
            "traits_consumption_allowed": False,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "independent_accuracy_claim_allowed": False,
            "blind_images_used": 0,
        },
        "comparator_contract": {
            "hybrid_max": {
                "evidence_role": "locked_legacy_development_comparator",
                "schema_version": LEGACY_HYBRID_COMPARATOR_SCHEMA,
                "identity_hair_variant": LEGACY_HYBRID_IDENTITY_VARIANT,
                "count_hair_variant": LEGACY_HYBRID_IDENTITY_VARIANT,
                "endpoint_complete_identity_layer": True,
                "phaxis_payload_allowed": False,
                "stageb_identity_source_allowed": False,
                "prediction_set_identity_sha256": sha256_json(
                    hybrid_file_locks
                ),
                "expected_prediction_set_identity_sha256": (
                    LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
                ),
            }
        },
        "overall": overall,
        "delta_stageb_train399_minus_hybrid": {
            "biological_presence_f1_20um": stageb[
                "tolerant_biological_presence"
            ]["20"]["f1"]
            - hybrid["tolerant_biological_presence"]["20"]["f1"],
            "identity_f1_20um": stageb["identity_attachment_proxy"]["20"]["f1"]
            - hybrid["identity_attachment_proxy"]["20"]["f1"],
            "strict_f1_20um": stageb["strict_whole_line_correspondence"]["20"]["f1"]
            - hybrid["strict_whole_line_correspondence"]["20"]["f1"],
            "count_mae": stageb["count"]["mae"] - hybrid["count"]["mae"],
            "count_bias": stageb["count"]["bias"] - hybrid["count"]["bias"],
            "count_ccc": stageb["count"]["ccc"] - hybrid["count"]["ccc"],
        },
        "paired_bootstrap_95ci": _bootstrap(
            rows,
            prf=prf,
            repetitions=args.bootstrap_repetitions,
            seed=20260828,
        ),
        "strata": groups,
        "per_image": rows,
        "prediction_input_locks": {
            "stageb_detection_files": detection_file_locks,
            "stageb_detection_set_identity_sha256": sha256_json(
                detection_file_locks
            ),
            "hybrid_prediction_files": hybrid_file_locks,
            "hybrid_prediction_set_identity_sha256": sha256_json(
                hybrid_file_locks
            ),
        },
        "inputs_sha256": {
            "dataset_manifest": sha256_file(args.dataset_manifest),
            "split_manifest": sha256_file(args.split_manifest),
            "integrity_manifest": canonical_provenance[
                "integrity_manifest_sha256"
            ],
            "canonical_ground_truth_lock_identity": canonical_provenance[
                "canonical_ground_truth_lock_identity_sha256"
            ],
            "locked_val_ids": sha256_file(args.locked_val_ids),
            "candidate_manifest": sha256_file(args.candidate_manifest),
            "selected_model_metadata": sha256_file(args.selected_model_metadata),
            "selection_receipt": sha256_file(args.selection_receipt),
            "evaluation_inference_summary": sha256_file(
                args.evaluation_inference_summary
            ),
        },
        "canonical_ground_truth_provenance": canonical_provenance,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    atomic_write_json(args.output, _json_safe(payload))
    print(
        f"completed: biological-presence F1@20 StageB-train399="
        f"{stageb['tolerant_biological_presence']['20']['f1']:.6f}, "
        f"Hybrid={hybrid['tolerant_biological_presence']['20']['f1']:.6f}; "
        f"count MAE={stageb['count']['mae']:.4f}"
    )


if __name__ == "__main__":
    main()
