#!/usr/bin/env python3
"""Materialize Figure 1 image/geometry from a pre-result case lock.

The case identity is supplied by ``build_release_case_prelocks.py`` and cannot
be changed here.  This post-result producer verifies the final fused prediction
and its source image, copies the immutable raw image bytes, and derives only
display geometry (root boundary, ordered axis, distal point, hair identities,
and endpoint-complete length curves).  No annotation or experimental condition
is read.
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
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.contracts import validate_hybrid_prediction  # noqa: E402
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.phenotypes import load_axis_geometry  # noqa: E402


CASE_SCHEMA = "PHAxis-figure1-case-selection-1.0"
CASE_STATUS = "locked_before_model_result_consumption"
SCHEMA_VERSION = "PHAxis-figure1-geometry-materialization-1.0"
STATUS = "completed_from_preselected_case_and_final_prediction"


class Figure1GeometryError(RuntimeError):
    """Figure 1 geometry cannot be bound to the preselected final prediction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Figure1GeometryError(message)


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(isinstance(observed, str) and observed == sha256_json(unsigned), f"{role}: identity mismatch")
    return observed


def _decimate(points: Any, *, maximum: int) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float64)
    _require(array.ndim == 2 and array.shape[1] == 2 and len(array) >= 2, "invalid display polyline")
    _require(np.all(np.isfinite(array)), "display polyline contains non-finite coordinates")
    if len(array) > maximum:
        indices = np.linspace(0, len(array) - 1, maximum, dtype=np.int64)
        array = array[np.unique(indices)]
    return [[float(x), float(y)] for x, y in array]


def _root_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _hierarchy = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    _require(bool(contours), "final root mask has no external contour")
    contour = max(contours, key=cv2.contourArea)
    _require(cv2.contourArea(contour) > 0.0, "final root contour has zero area")
    perimeter = float(cv2.arcLength(contour, True))
    approximation = cv2.approxPolyDP(contour, max(0.5, perimeter * 0.00025), True)
    points = approximation.reshape(-1, 2).astype(np.float64)
    if len(points) > 1200:
        indices = np.linspace(0, len(points) - 1, 1200, dtype=np.int64)
        points = points[np.unique(indices)]
    _require(len(points) >= 3, "final root contour has fewer than three vertices")
    return [[float(x), float(y)] for x, y in points]


def _scale_bar(um_per_px: float, width: int) -> tuple[float, float]:
    for micrometres in (500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0):
        pixels = micrometres / um_per_px
        if pixels <= width * 0.30:
            return micrometres, pixels
    raise Figure1GeometryError("preselected image is too narrow for a physical scale bar")


def _image_size(path: Path) -> tuple[int, int]:
    """Read source geometry without decoding a potentially giant TIFF."""

    if path.suffix.casefold() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as opened:
            _require(bool(opened.series), "Figure 1 TIFF has no image series")
            series = opened.series[0]
            axes = str(series.axes)
            shape = tuple(int(value) for value in series.shape)
        _require(
            axes.count("X") == 1 and axes.count("Y") == 1 and len(axes) == len(shape),
            "Figure 1 TIFF does not expose unambiguous X/Y geometry",
        )
        return shape[axes.index("X")], shape[axes.index("Y")]
    with Image.open(path) as opened:
        return tuple(int(value) for value in opened.size)


def materialize_figure1_geometry(
    *,
    case_selection: str | Path,
    application_manifest: str | Path,
    fusion_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    case_path = Path(case_selection).resolve()
    manifest_path = Path(application_manifest).resolve()
    fusion = Path(fusion_root).resolve()
    destination = Path(output).resolve()
    for role, path in (("case selection", case_path), ("application manifest", manifest_path)):
        _require(path.is_file() and not path.is_symlink(), f"{role} is absent or symlinked")
        _require("blind" not in str(path).casefold(), f"{role} has a blind-labelled path")
    _require(fusion.is_dir() and not fusion.is_symlink(), "fusion root is absent or symlinked")
    _require("blind" not in str(fusion).casefold(), "fusion root has a blind-labelled path")
    _require(not destination.exists(), f"refusing to overwrite {destination}")

    case = read_json(case_path)
    case_identity = _sealed(case, "figure1_case_selection_identity_sha256", "Figure 1 case selection")
    _require(
        case.get("schema_version") == CASE_SCHEMA
        and case.get("status") == CASE_STATUS
        and case.get("selected_before_model_result_consumption") is True
        and case.get("selected_by_prediction_or_trait_outcome") is False
        and case.get("classic_challenge_panel_task") is False
        and case.get("condition_metadata_read") is False
        and case.get("canonical_annotations_read") is False
        and case.get("blind_images_used") == 0,
        "Figure 1 case selection is not the result-independent authority",
    )
    task_id = str(case.get("task_id") or "")
    _require(bool(task_id) and Path(task_id).name == task_id, "Figure 1 task identity is invalid")

    import csv

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    by_task = {str(row.get("task_id") or row.get("image_id") or ""): row for row in rows}
    _require(len(rows) == len(by_task) == 283 and task_id in by_task, "application manifest is not exact283 or omits Figure 1")
    row = by_task[task_id]
    raw_value = row.get("image_path") or row.get("input_path")
    _require(bool(raw_value), "Figure 1 source-image path is absent")
    supplied = Path(str(raw_value))
    source = supplied if supplied.is_absolute() else manifest_path.parent / supplied
    source = source.resolve()
    source_sha = str(row.get("image_sha256") or row.get("source_image_sha256") or "").casefold()
    _require(source.is_file() and not source.is_symlink(), "Figure 1 source image is absent or symlinked")
    _require("blind" not in str(source).casefold(), "Figure 1 source has a blind-labelled path")
    _require(
        sha256_file(source) == source_sha == str(case.get("source_image_sha256", "")).casefold()
        and source.stat().st_size == case.get("source_image_bytes"),
        "Figure 1 source differs from the pre-result case lock",
    )

    summary_path = fusion / "fusion_summary.json"
    _require(summary_path.is_file(), "final fusion summary is absent")
    summary = read_json(summary_path)
    summary_identity = _sealed(summary, "summary_identity_sha256", "final fusion summary")
    _require(
        summary.get("schema_version") == "PHAxis-fusion-run-1.1"
        and summary.get("status") == "completed"
        and summary.get("images") == 283
        and summary.get("condition_metadata_used_for_routing") is False
        and summary.get("canonical_annotations_read") is False
        and summary.get("root_cap_region_output") is False
        and summary.get("blind_images_used") == 0,
        "final fusion summary violates the release contract",
    )
    records = {
        str(record.get("task_id")): record
        for record in summary.get("records", ())
        if isinstance(record, Mapping)
    }
    _require(len(records) == 283 and task_id in records, "final fusion record set is not exact283")
    prediction_path = fusion / "predictions" / f"{task_id}.json"
    _require(prediction_path.is_file(), "preselected final prediction is absent")
    prediction_sha = sha256_file(prediction_path)
    _require(prediction_sha == records[task_id].get("prediction_sha256"), "final prediction/fusion summary hash mismatch")
    prediction = read_json(prediction_path)
    validate_hybrid_prediction(prediction, artifact_root=fusion)
    _require(
        prediction.get("task_id") == task_id
        and str(prediction.get("source_image_sha256", "")).casefold() == source_sha
        and prediction.get("condition_metadata_used_for_routing") is False
        and prediction.get("canonical_annotations_read_during_inference") is False
        and prediction.get("root_cap_region_output") is False
        and prediction.get("blind_images_used") == 0,
        "preselected prediction violates the final inference contract",
    )

    mask_path = fusion / str(prediction["root_mask_relpath"])
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _require(mask is not None, "cannot decode final root mask")
    width, height = _image_size(source)
    _require(mask.shape == (height, width), "final root mask/source geometry mismatch")
    axis_path = fusion / str(prediction["root_axis_geometry_relpath"])
    axis = load_axis_geometry(axis_path, expected_image_sha256=source_sha)
    axis_xy = _decimate(axis["path_xy"], maximum=900)
    distal = np.asarray(prediction.get("root_cap_point_xy"), dtype=np.float64)
    _require(distal.shape == (2,) and np.all(np.isfinite(distal)), "final distal point is invalid")

    length_by_identity: dict[str, Mapping[str, Any]] = {}
    for hair in prediction.get("length_hairs", ()):
        _require(isinstance(hair, Mapping), "final length-hair row is invalid")
        identity_id = str(hair.get("identity_source_instance_id") or "")
        _require(bool(identity_id) and identity_id not in length_by_identity, "final length-to-identity map is invalid")
        length_by_identity[identity_id] = hair
    hair_identities: list[dict[str, Any]] = []
    for hair in prediction.get("identity_hairs", ()):
        _require(isinstance(hair, Mapping), "final identity-hair row is invalid")
        identity_id = str(hair.get("source_instance_id") or "")
        identity_xy = _decimate(hair.get("points_xy"), maximum=16)
        length = length_by_identity.pop(identity_id, None)
        hair_identities.append(
            {
                "source_instance_id": identity_id,
                "attachment_xy": identity_xy[0],
                "identity_xy": identity_xy,
                "length_curve_xy": None
                if length is None
                else _decimate(length.get("points_xy"), maximum=240),
            }
        )
    _require(not length_by_identity, "a final length curve is not associated with an identity hair")
    try:
        um_per_px = float(row.get("um_per_px") or row.get("source_um_per_px"))
    except (TypeError, ValueError) as error:
        raise Figure1GeometryError("Figure 1 physical calibration is absent") from error
    _require(math.isfinite(um_per_px) and um_per_px > 0.0, "Figure 1 physical calibration is invalid")
    scale_um, scale_px = _scale_bar(um_per_px, width)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        source_name = "figure1_source_image" + (".tif" if source.suffix.casefold() in {".tif", ".tiff"} else source.suffix.casefold())
        source_target = staging / source_name
        shutil.copyfile(source, source_target)
        _require(sha256_file(source_target) == source_sha, "copied Figure 1 source-image hash changed")
        geometry: dict[str, Any] = {
            "schema_version": "PHAxis-figure1-display-geometry-1.0",
            "status": "completed_final_prediction_bound_display_geometry",
            "task_id": task_id,
            "source_image_sha256": source_sha,
            "prediction_sha256": prediction_sha,
            "display": {"kind": "linear_global", "lower": 0.0, "upper": 255.0},
            "scale_bar": {"pixels": scale_px, "micrometres": scale_um},
            "root_polygon_xy": _root_polygon(mask),
            "axis_xy": axis_xy,
            "distal_point_xy": [float(distal[0]), float(distal[1])],
            "hair_identities": hair_identities,
            "case_selection_sha256": sha256_file(case_path),
            "case_selection_identity_sha256": case_identity,
            "fusion_summary_sha256": sha256_file(summary_path),
            "fusion_summary_identity_sha256": summary_identity,
            "geometry_is_display_only": True,
            "canonical_annotations_read": False,
            "condition_metadata_read": False,
            "root_cap_region_output": False,
            "blind_images_used": 0,
        }
        geometry["figure1_display_geometry_identity_sha256"] = sha256_json(geometry)
        geometry_path = staging / "figure1_geometry.json"
        atomic_write_json(geometry_path, geometry)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "task_id": task_id,
            "case_selection_sha256": sha256_file(case_path),
            "case_selection_identity_sha256": case_identity,
            "application_manifest_sha256": sha256_file(manifest_path),
            "fusion_summary_sha256": sha256_file(summary_path),
            "fusion_summary_identity_sha256": summary_identity,
            "prediction_sha256": prediction_sha,
            "source_image_file": source_name,
            "source_image_sha256": source_sha,
            "figure1_geometry_sha256": sha256_file(geometry_path),
            "figure1_display_geometry_identity_sha256": geometry[
                "figure1_display_geometry_identity_sha256"
            ],
            "root_polygon_vertices": len(geometry["root_polygon_xy"]),
            "axis_display_points": len(axis_xy),
            "identity_hairs": len(hair_identities),
            "endpoint_complete_length_curves": sum(
                item["length_curve_xy"] is not None for item in hair_identities
            ),
            "case_selection_changed_after_results": False,
            "canonical_annotations_read": False,
            "condition_metadata_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["figure1_geometry_materialization_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return deepcopy(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-selection", type=Path, required=True)
    parser.add_argument("--application-manifest", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = materialize_figure1_geometry(
        case_selection=args.case_selection,
        application_manifest=args.application_manifest,
        fusion_root=args.fusion_root,
        output=args.output,
    )
    print(receipt["figure1_geometry_materialization_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
