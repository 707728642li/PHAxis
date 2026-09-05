#!/usr/bin/env python3
"""Render and seal exact-cohort review overlays plus five locked figure cases.

The only human-authored input is a five-row plan containing ``case_role`` and
``task_id``.  The command never accepts plotted images or hashes from that
plan.  It first verifies and renders every member of the exact application
cohort without consulting experimental-condition metadata.  Only after all
pixels have been rendered does it use ``experiment_key``, ``condition_code``
and ``formal_statistics_eligible`` to organise the review-only output tree.

The five paper cases then reuse the exact PNG bytes from that cohort export.
Thus there is one overlay authority per task rather than an independently
rendered paper gallery that could silently diverge from the user-review view.

The historical filename is retained only as a command-line compatibility entry
point.  The current receipt makes no claim about blindness at case-selection
time.  It records only the code-verifiable facts that experimental-condition
metadata does not enter prediction, overlay pixels, or morphology-evidence
values before output organisation.  The compatibility field
``experimental_condition_metadata_used_for_evidence_assembly=false`` is
explicitly scoped to those pixels and morphology evidence cards; only after
pixels are fixed may metadata organise directories and assign formal/review
labels.  The five cases form a preselected morphology/acquisition-challenge
gallery rather than a random or representative performance sample.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.contracts import validate_hybrid_prediction  # noqa: E402
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.rendering import render_display_background, render_prediction_overlay  # noqa: E402


SCHEMA_VERSION = "PHAxis-manuscript-overlay-selection-receipt-1.2"
RECEIPT_STATUS = (
    "completed_locked_preselected_gallery_and_exact_cohort_review_export"
)
REVIEW_SCHEMA_VERSION = "PHAxis-exact-cohort-review-overlay-export-1.0"
REVIEW_STATUS = "completed_exact_cohort_final_fusion_review_export"
DEFAULT_EXPECTED_TASK_COUNT = 283
REVIEW_ROOT_NAME = "full283_review_overlays"
REVIEW_INDEX_NAME = "full283_review_index.csv"
REVIEW_CHECKLIST_NAME = "full283_review_checklist.csv"
REVIEW_SUMMARY_NAME = "full283_review_summary.json"
REVIEW_README_NAME = "README_CN.md"
REVIEW_PENDING_STATUS = "pending_manual_visual_review"
CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE = (
    "overlay_pixels_and_morphology_evidence_cards_before_output_organization"
)
CASE_SELECTION_BASIS = "preselected_morphology_acquisition_challenge_roles"
CASE_ROLES = ("representative", "low_contrast", "curved_dense", "continuity", "fail_closed")
COLOURS = {
    # Exact RGB equivalents of the BGR tuples in src/phaxis/rendering.py.
    "root_boundary_colour": "#19AADC",
    "axis_colour": "#E6E6E6",
    "distal_colour": "#DC3CFF",
    "length_curve_colour": "#73F55A",
    "identity_vector_colour": "#FFCD14",
    "hair_base_colour": "#FFFF00",
    "visible_endpoint_colour": "#FF6919",
}
SELECTION_RULES = {
    "representative": "preselected representative eligible source unit",
    "low_contrast": "preselected low-contrast upper-root acquisition challenge",
    "curved_dense": "preselected curved dense-hair acquisition challenge",
    "continuity": "preselected root-continuity acquisition challenge",
    "fail_closed": "preselected review-only fail-closed source unit",
}
INSET_RULES = {
    "low_contrast": "shootward_axis_last_35_percent_with_image_relative_margin",
    "curved_dense": "maximum_chord_deviation_axis_window_with_image_relative_margin",
}
LOCKED_ANCHOR_TASK_IDS = {
    "low_contrast": "RHSCU-aa5b6e37df15821f",
    "curved_dense": "RHSCU-bbf649822174e0a2",
}
WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class OverlayEvidenceError(RuntimeError):
    """An overlay cannot be traced to the locked image and prediction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OverlayEvidenceError(message)


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    _require(isinstance(observed, str) and observed == sha256_json(unsigned), f"{role} identity mismatch")
    return observed


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), f"empty CSV: {path}")
    return rows


def _index(rows: list[dict[str, str]], role: str) -> dict[str, dict[str, str]]:
    task_ids = [_safe_task_id(row.get("task_id")) for row in rows]
    result = dict(zip(task_ids, rows, strict=True))
    _require(
        len(result) == len(rows)
        and len({task_id.casefold() for task_id in task_ids}) == len(task_ids),
        f"{role}: duplicate/case-colliding task_id",
    )
    return result


def _strict_bool(value: Any, *, role: str) -> bool:
    normalized = str(value).strip().casefold()
    _require(
        normalized in {"true", "1", "yes", "false", "0", "no"},
        f"{role} is not an explicit Boolean",
    )
    return normalized in {"true", "1", "yes"}


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value).strip().casefold()))


def _is_windows_reserved_component(value: str) -> bool:
    # Windows applies device-name reservation before the first extension, so
    # both ``CON`` and ``CON.txt`` are forbidden path components.
    stem = value.rstrip(" .").split(".", 1)[0].upper()
    return stem in WINDOWS_RESERVED_STEMS


def _safe_task_id(value: Any) -> str:
    task_id = str(value).strip()
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id))
        and task_id not in {".", ".."}
        and task_id == task_id.rstrip(" .")
        and not _is_windows_reserved_component(task_id),
        f"unsafe task_id for review path: {task_id!r}",
    )
    return task_id


def _safe_slug(value: Any, *, role: str) -> str:
    original = str(value).strip()
    _require(bool(original), f"empty {role}")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", original).strip("._")
    _require(
        bool(slug)
        and len(slug) <= 128
        and slug not in {".", ".."}
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug))
        and not _is_windows_reserved_component(slug),
        f"unsafe {role} for review path: {original!r}",
    )
    return slug


def _exact_nonnegative_int(value: Any, *, role: str) -> int:
    text = str(value).strip()
    _require(bool(text), f"{role} is missing")
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise OverlayEvidenceError(f"{role} is not numeric") from error
    _require(np.isfinite(number) and number >= 0 and number.is_integer(), f"{role} is not a non-negative integer")
    return int(number)


def _relative_inside(root: Path, path: Path, *, role: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise OverlayEvidenceError(f"{role} escapes its declared root") from error
    _require(not path.is_symlink(), f"{role} is symlinked")
    for parent in (path, *path.parents):
        if parent == root:
            break
        _require(not parent.is_symlink(), f"{role} traverses a symlink")
    return relative.as_posix()


def _locked_fusion_artifact(
    fusion_root: Path, relative_value: Any, *, role: str
) -> Path:
    relative = Path(str(relative_value))
    _require(
        bool(str(relative_value))
        and not relative.is_absolute()
        and ".." not in relative.parts,
        f"{role} has an unsafe relative path",
    )
    path = fusion_root / relative
    _require(path.is_file(), f"{role} is missing")
    _relative_inside(fusion_root, path, role=role)
    return path


def _scale_bar(um_per_px: float, width: int) -> tuple[float, float]:
    for micrometres in (500.0, 200.0, 100.0, 50.0):
        pixels = micrometres / um_per_px
        if pixels <= width * 0.30:
            return micrometres, pixels
    raise OverlayEvidenceError("image is too narrow for a positive physical scale bar")


def _expanded_bbox(
    points_xy: np.ndarray,
    *,
    image_shape: tuple[int, ...],
    minimum_fraction: float = 0.28,
) -> tuple[int, int, int, int]:
    """Return a deterministic, clamped xyxy crop around selected axis points."""

    height, width = int(image_shape[0]), int(image_shape[1])
    _require(height >= 8 and width >= 8, "inset source image is too small")
    points = np.asarray(points_xy, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 2
        and len(points) >= 2
        and np.all(np.isfinite(points)),
        "inset axis window is invalid",
    )
    margin = max(8.0, 0.08 * float(min(height, width)))
    x0 = float(points[:, 0].min() - margin)
    y0 = float(points[:, 1].min() - margin)
    x1 = float(points[:, 0].max() + margin)
    y1 = float(points[:, 1].max() + margin)
    minimum_width = min(float(width), minimum_fraction * float(width))
    minimum_height = min(float(height), minimum_fraction * float(height))
    centre_x, centre_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if x1 - x0 < minimum_width:
        x0, x1 = centre_x - minimum_width / 2.0, centre_x + minimum_width / 2.0
    if y1 - y0 < minimum_height:
        y0, y1 = centre_y - minimum_height / 2.0, centre_y + minimum_height / 2.0
    x0, x1 = max(0.0, x0), min(float(width), x1)
    y0, y1 = max(0.0, y0), min(float(height), y1)
    if x1 - x0 < minimum_width:
        if x0 <= 0.0:
            x1 = min(float(width), minimum_width)
        else:
            x0 = max(0.0, float(width) - minimum_width)
    if y1 - y0 < minimum_height:
        if y0 <= 0.0:
            y1 = min(float(height), minimum_height)
        else:
            y0 = max(0.0, float(height) - minimum_height)
    bbox = (
        int(np.floor(x0)),
        int(np.floor(y0)),
        int(np.ceil(x1)),
        int(np.ceil(y1)),
    )
    _require(
        0 <= bbox[0] < bbox[2] <= width
        and 0 <= bbox[1] < bbox[3] <= height,
        "deterministic inset crop is outside the source image",
    )
    return bbox


def _deterministic_inset(
    *,
    role: str,
    prediction: Mapping[str, Any],
    fusion_root: Path,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Derive the two prelocked Fig. 4 insets from sealed axis geometry."""

    if role not in INSET_RULES:
        return {
            "inset_required": False,
            "inset_rule": "not_applicable",
            "inset_x0": None,
            "inset_y0": None,
            "inset_x1": None,
            "inset_y1": None,
            "inset_geometry_sha256": None,
        }
    axis_path = fusion_root / str(prediction["root_axis_geometry_relpath"])
    _require(
        axis_path.is_file()
        and sha256_file(axis_path) == prediction["root_axis_geometry_sha256"],
        f"{prediction.get('task_id')}: root-axis geometry hash mismatch",
    )
    with np.load(axis_path, allow_pickle=False) as geometry:
        _require(
            "path_xy" in geometry.files
            and "distance_from_tip_px" in geometry.files
            and "source_image_sha256" in geometry.files,
            f"{prediction.get('task_id')}: inset axis fields missing",
        )
        axis_xy = np.asarray(geometry["path_xy"], dtype=np.float64)
        distance_px = np.asarray(
            geometry["distance_from_tip_px"], dtype=np.float64
        )
        source_sha = str(np.asarray(geometry["source_image_sha256"]).item())
    _require(
        axis_xy.ndim == 2
        and axis_xy.shape[1] == 2
        and len(axis_xy) >= 3
        and distance_px.shape == (len(axis_xy),)
        and np.all(np.isfinite(axis_xy))
        and np.all(np.isfinite(distance_px))
        and np.all(np.diff(distance_px) >= -1e-9)
        and source_sha == prediction.get("source_image_sha256"),
        f"{prediction.get('task_id')}: inset axis geometry is invalid",
    )
    total_distance = float(distance_px[-1] - distance_px[0])
    _require(total_distance > 0.0, f"{prediction.get('task_id')}: zero-length inset axis")
    normalized = (distance_px - distance_px[0]) / total_distance
    if role == "low_contrast":
        selected = axis_xy[normalized >= 0.65]
    else:
        chord = axis_xy[-1] - axis_xy[0]
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm > 1e-9:
            deviations = np.abs(
                chord[0] * (axis_xy[:, 1] - axis_xy[0, 1])
                - chord[1] * (axis_xy[:, 0] - axis_xy[0, 0])
            ) / chord_norm
        else:
            deviations = np.zeros(len(axis_xy), dtype=np.float64)
        interior = (normalized >= 0.10) & (normalized <= 0.90)
        _require(interior.any(), f"{prediction.get('task_id')}: curved inset has no interior axis")
        candidate_indices = np.flatnonzero(interior)
        centre_index = int(
            candidate_indices[int(np.argmax(deviations[candidate_indices]))]
        )
        centre_distance = normalized[centre_index]
        selected = axis_xy[np.abs(normalized - centre_distance) <= 0.12]
    if len(selected) < 2:
        selected = axis_xy
    x0, y0, x1, y1 = _expanded_bbox(
        selected, image_shape=image_shape
    )
    geometry_contract = {
        "task_id": prediction.get("task_id"),
        "source_image_sha256": prediction.get("source_image_sha256"),
        "root_axis_geometry_sha256": prediction["root_axis_geometry_sha256"],
        "inset_rule": INSET_RULES[role],
        "bbox_xyxy": [x0, y0, x1, y1],
    }
    return {
        "inset_required": True,
        "inset_rule": INSET_RULES[role],
        "inset_x0": x0,
        "inset_y0": y0,
        "inset_x1": x1,
        "inset_y1": y1,
        "inset_geometry_sha256": sha256_json(geometry_contract),
    }


def _atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] | None = None,
) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path.name}")
    fieldnames = fields or tuple(rows[0])
    _require(
        len(fieldnames) == len(set(fieldnames)),
        f"duplicate CSV fields: {path.name}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", prefix=f".{path.name}.",
        suffix=".partial", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _review_readme(*, expected_task_count: int) -> str:
    return f"""# PHAxis 最终融合预测逐图审阅目录

本目录由 formal Stage27 从同一批最终融合预测一次性、确定性导出，共 {expected_task_count} 张。实验条件不进入模型预测、阈值路由、overlay 像素或形态证据卡；PNG 像素固定后，才使用 `experiment_key/condition_code/formal_statistics_eligible` 组织目录并添加 `formal|review_only` 行政标签。

## 目录与表格

- `{REVIEW_ROOT_NAME}/<experiment>/<condition>/<formal|review_only>/`：逐图 PNG。
- `{REVIEW_INDEX_NAME}`：任务、分类、identity 数、endpoint-complete 长度数、原图/预测/PNG 哈希、尺寸和字节数。
- `{REVIEW_CHECKLIST_NAME}`：逐图人工审阅清单；初始状态均为 `{REVIEW_PENDING_STATUS}`，请填写连续性、根冠点、根毛覆盖、可测长度和备注。
- `{REVIEW_SUMMARY_NAME}`：机器可读的完整性、来源和哈希封印。

## 颜色与测量语义

- 根外边界：`{COLOURS['root_boundary_colour']}`；有序主根轴：`{COLOURS['axis_colour']}`；根冠/远端点：`{COLOURS['distal_colour']}`。
- 绿色 `{COLOURS['length_curve_colour']}` 是已与 identity 一对一匹配的 endpoint-complete 中心线，可用于条件性长度统计。
- 琥珀色 `{COLOURS['identity_vector_colour']}` 是 Stage-B 判定的根毛 identity/presence 向量；它计入根毛数量，但没有完整可见终点时不得虚构长度。
- 黄色 `{COLOURS['hair_base_colour']}` 为附着点。橙色 `{COLOURS['visible_endpoint_colour']}` 位于琥珀 identity 上时只表示 Stage-B 向量末端，不证明 endpoint-complete；只有绿色 matched curve 的末端及其完整曲线支持 endpoint-complete 长度。

## 审阅边界

清单用于结果质量审阅和论文图例确认，不得把本批审阅意见反馈为本次模型的训练标签、阈值选择或条件路由依据；如发现问题，应单独记录为后续版本工作项。`review_only` 图可诊断但不进入正式生物学统计。原始人工标注、blind/final-validation 数据和根冠区域统计均未被本阶段读取或使用。
"""


def build_overlay_evidence(
    *,
    case_plan: str | Path,
    application_manifest: str | Path,
    full_traits: str | Path,
    fusion_root: str | Path,
    output: str | Path,
    expected_task_count: int = DEFAULT_EXPECTED_TASK_COUNT,
) -> dict[str, Any]:
    _require(
        isinstance(expected_task_count, int)
        and not isinstance(expected_task_count, bool)
        and expected_task_count >= len(CASE_ROLES),
        "expected_task_count must be an integer at least five",
    )
    unresolved_inputs = (
        ("case plan", Path(case_plan)),
        ("application manifest", Path(application_manifest)),
        ("full traits", Path(full_traits)),
    )
    for role, path in unresolved_inputs:
        _require(not path.is_symlink(), f"{role} is symlinked")
    unresolved_fusion = Path(fusion_root)
    _require(not unresolved_fusion.is_symlink(), "fusion root is symlinked")
    plan_path = unresolved_inputs[0][1].resolve()
    manifest_path = unresolved_inputs[1][1].resolve()
    traits_path = unresolved_inputs[2][1].resolve()
    fusion = unresolved_fusion.resolve()
    destination = Path(output).resolve()
    for role, path in (("case plan", plan_path), ("application manifest", manifest_path), ("full traits", traits_path)):
        _require(path.is_file() and not path.is_symlink(), f"{role} missing or symlinked")
        _require("blind" not in str(path).casefold(), f"{role} has a blind-labelled path")
    _require(fusion.is_dir() and not fusion.is_symlink(), "fusion root missing or symlinked")
    _require("blind" not in str(fusion).casefold(), "fusion root has a blind-labelled path")
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    _require(destination.parent.is_dir(), "output parent does not exist")

    plan_rows = _rows(plan_path)
    _require(len(plan_rows) == 5, "case plan must contain exactly five rows")
    _require(set(plan_rows[0]) == {"case_role", "task_id"}, "case plan may contain only case_role and task_id")
    _require({row["case_role"] for row in plan_rows} == set(CASE_ROLES), "case roles are incomplete")
    plan_task_ids = [_safe_task_id(row["task_id"]) for row in plan_rows]
    _require(
        len({task_id.casefold() for task_id in plan_task_ids}) == 5,
        "case task IDs must be unique on Windows",
    )
    plan_by_role = {
        row["case_role"]: _safe_task_id(row["task_id"]) for row in plan_rows
    }
    _require(
        all(
            plan_by_role.get(role) == task_id
            for role, task_id in LOCKED_ANCHOR_TASK_IDS.items()
        ),
        "the two prelocked Fig.4 anchor task IDs changed",
    )

    manifest = _index(_rows(manifest_path), "application manifest")
    traits = _index(_rows(traits_path), "full traits")
    _require(
        len(manifest) == len(traits) == expected_task_count
        and set(manifest) == set(traits),
        "application manifest/exact-cohort traits differ",
    )
    fusion_summary_path = fusion / "fusion_summary.json"
    _require(
        fusion_summary_path.is_file() and not fusion_summary_path.is_symlink(),
        "fusion summary missing or symlinked",
    )
    fusion_summary = read_json(fusion_summary_path)
    _require(fusion_summary.get("schema_version") == "PHAxis-fusion-run-1.1" and fusion_summary.get("status") == "completed", "fusion summary schema/status changed")
    _sealed(fusion_summary, "summary_identity_sha256", "fusion summary")
    _require(
        fusion_summary.get("images") == expected_task_count
        and fusion_summary.get("condition_metadata_used_for_routing") is False
        and fusion_summary.get("canonical_annotations_read") is False
        and fusion_summary.get("root_cap_region_output") is False
        and fusion_summary.get("blind_images_used") == 0,
        "fusion summary violates final inference guards",
    )
    fusion_record_rows = fusion_summary.get("records")
    _require(
        isinstance(fusion_record_rows, list)
        and len(fusion_record_rows) == expected_task_count
        and all(isinstance(row, Mapping) for row in fusion_record_rows),
        "fusion records must be an exact list with expected_task_count rows",
    )
    fusion_task_ids = [
        _safe_task_id(row.get("task_id")) for row in fusion_record_rows
    ]
    fusion_records = dict(zip(fusion_task_ids, fusion_record_rows, strict=True))
    _require(
        len(fusion_records) == expected_task_count
        and len({task_id.casefold() for task_id in fusion_task_ids})
        == expected_task_count
        and set(fusion_records) == set(manifest),
        "fusion records contain duplicate/case-colliding tasks or differ from the exact cohort",
    )
    _require(
        all(_safe_task_id(task_id) == task_id for task_id in fusion_records),
        "fusion contains an unsafe task_id",
    )

    builder_source_path = Path(__file__).resolve()
    renderer_source_path = PROJECT_ROOT / "src/phaxis/rendering.py"
    _require(renderer_source_path.is_file(), "renderer source is missing")
    source_authority = {
        "case_plan": sha256_file(plan_path),
        "application_manifest": sha256_file(manifest_path),
        "full_traits": sha256_file(traits_path),
        "fusion_summary": sha256_file(fusion_summary_path),
        "overlay_builder_source": sha256_file(builder_source_path),
        "renderer_source": sha256_file(renderer_source_path),
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    output_rows: list[dict[str, Any]] = []
    try:
        # Render to a metadata-independent flat staging area.  No experiment,
        # condition or formal-eligibility field is accessed in this phase.
        flat_render_root = staging / ".exact_cohort_render_stage"
        flat_render_root.mkdir()
        render_records: dict[str, dict[str, Any]] = {}
        for task_id in sorted(manifest):
            manifest_row = manifest[task_id]
            trait_row = traits[task_id]
            raw_path_value = manifest_row.get("image_path") or manifest_row.get(
                "input_path"
            )
            _require(bool(raw_path_value), f"{task_id}: image path missing")
            _require(
                "blind" not in str(raw_path_value).casefold(),
                f"{task_id}: raw image has a blind-labelled path",
            )
            raw_path = Path(str(raw_path_value))
            if not raw_path.is_absolute():
                raw_path = manifest_path.parent / raw_path
            _require(not raw_path.is_symlink(), f"{task_id}: raw image is symlinked")
            raw_path = raw_path.resolve()
            _require(
                raw_path.is_file() and not raw_path.is_symlink(),
                f"{task_id}: raw image missing or symlinked",
            )
            raw_sha = str(
                manifest_row.get("image_sha256")
                or manifest_row.get("source_image_sha256")
                or ""
            ).casefold()
            _require(_is_sha256(raw_sha), f"{task_id}: invalid raw image SHA-256")
            _require(
                sha256_file(raw_path) == raw_sha,
                f"{task_id}: raw image hash mismatch",
            )
            _require(
                str(trait_row.get("source_image_sha256", "")).casefold()
                == raw_sha,
                f"{task_id}: traits/raw image lock mismatch",
            )

            prediction_path = fusion / "predictions" / f"{task_id}.json"
            _require(
                prediction_path.is_file() and not prediction_path.is_symlink(),
                f"{task_id}: fused prediction missing or symlinked",
            )
            _relative_inside(fusion, prediction_path, role=f"{task_id}: prediction")
            prediction_sha = sha256_file(prediction_path)
            _require(
                prediction_sha == fusion_records[task_id].get("prediction_sha256"),
                f"{task_id}: prediction hash mismatch",
            )
            prediction = read_json(prediction_path)
            _require(
                prediction.get("task_id") == task_id,
                f"{task_id}: prediction task lock mismatch",
            )
            for relative_field, digest_field in (
                ("root_mask_relpath", "root_mask_sha256"),
                ("root_axis_geometry_relpath", "root_axis_geometry_sha256"),
            ):
                artifact = _locked_fusion_artifact(
                    fusion,
                    prediction.get(relative_field),
                    role=f"{task_id}: {relative_field}",
                )
                _require(
                    sha256_file(artifact)
                    == str(prediction.get(digest_field, "")).casefold(),
                    f"{task_id}: {relative_field} hash mismatch",
                )
            validate_hybrid_prediction(prediction, artifact_root=fusion)
            _require(
                prediction.get("source_image_sha256") == raw_sha,
                f"{task_id}: prediction/raw image lock mismatch",
            )
            _require(
                prediction.get("condition_metadata_used_for_routing") is False
                and prediction.get("canonical_annotations_read_during_inference")
                is False
                and prediction.get("root_cap_region_output") is False
                and prediction.get("blind_images_used") == 0,
                f"{task_id}: prediction violates final inference guards",
            )
            identities = prediction.get("identity_hairs")
            lengths = prediction.get("length_hairs")
            _require(
                isinstance(identities, list) and isinstance(lengths, list),
                f"{task_id}: prediction hair arrays are invalid",
            )
            identity_count = len(identities)
            length_count = len(lengths)
            _require(
                _exact_nonnegative_int(
                    trait_row.get("hair_count"),
                    role=f"{task_id}: trait hair_count",
                )
                == identity_count,
                f"{task_id}: trait/prediction identity count mismatch",
            )
            _require(
                _exact_nonnegative_int(
                    trait_row.get("hair_length_measurement_hair_count"),
                    role=f"{task_id}: trait endpoint-complete count",
                )
                == length_count,
                f"{task_id}: trait/prediction endpoint-complete count mismatch",
            )
            _require(
                _exact_nonnegative_int(
                    fusion_records[task_id].get("phaxis_identity_count"),
                    role=f"{task_id}: fusion identity count",
                )
                == identity_count,
                f"{task_id}: fusion/prediction identity count mismatch",
            )
            _require(
                _exact_nonnegative_int(
                    fusion_records[task_id].get(
                        "matched_endpoint_complete_lengths"
                    ),
                    role=f"{task_id}: fusion endpoint-complete count",
                )
                == length_count,
                f"{task_id}: fusion/prediction endpoint-complete count mismatch",
            )

            image = tifffile.imread(raw_path)
            overlay = render_prediction_overlay(
                image,
                prediction,
                artifact_root=fusion,
                include_text=False,
                display_gamma=1.0,
            )
            _require(
                overlay.ndim == 3
                and overlay.shape[2] == 3
                and overlay.dtype == np.uint8,
                f"{task_id}: renderer returned an invalid PNG canvas",
            )
            flat_target = flat_render_root / f"{task_id}.png"
            _require(
                cv2.imwrite(
                    str(flat_target),
                    overlay,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6],
                ),
                f"{task_id}: review overlay write failed",
            )
            render_records[task_id] = {
                "task_id": task_id,
                "raw_path": raw_path,
                "raw_source_image_sha256": raw_sha,
                "prediction_sha256": prediction_sha,
                "identity_hair_count": identity_count,
                "endpoint_complete_length_count": length_count,
                "flat_target": flat_target,
                "width_px": int(overlay.shape[1]),
                "height_px": int(overlay.shape[0]),
            }
        _require(
            len(render_records) == expected_task_count,
            "metadata-independent render phase is not exact-cohort complete",
        )

        # Organise only after rendering has completed.  These three metadata
        # fields can change paths/labels but can never change overlay pixels.
        review_root = staging / REVIEW_ROOT_NAME
        review_root.mkdir()
        experiment_slugs: dict[str, str] = {}
        condition_slugs: dict[str, str] = {}
        review_rows: list[dict[str, Any]] = []
        review_by_task: dict[str, dict[str, Any]] = {}
        for task_id in sorted(render_records):
            trait_row = traits[task_id]
            experiment_key = str(trait_row.get("experiment_key", "")).strip()
            condition_code = str(trait_row.get("condition_code", "")).strip()
            experiment_slug = _safe_slug(experiment_key, role="experiment_key")
            condition_slug = _safe_slug(condition_code, role="condition_code")
            previous_experiment = experiment_slugs.setdefault(
                experiment_slug.casefold(), experiment_key
            )
            previous_condition = condition_slugs.setdefault(
                condition_slug.casefold(), condition_code
            )
            _require(
                previous_experiment == experiment_key,
                "experiment slug collision",
            )
            _require(previous_condition == condition_code, "condition slug collision")
            eligible = _strict_bool(
                trait_row.get("formal_statistics_eligible"),
                role=f"{task_id}: formal_statistics_eligible",
            )
            review_partition = "formal" if eligible else "review_only"
            target = (
                review_root
                / experiment_slug
                / condition_slug
                / review_partition
                / f"{task_id}.phaxis_overlay.png"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            relative_from_review_root = _relative_inside(
                review_root, target, role=f"{task_id}: review overlay target"
            )
            os.replace(render_records[task_id]["flat_target"], target)
            _require(
                target.is_file() and not target.is_symlink(),
                f"{task_id}: organised review overlay missing",
            )
            row = {
                "task_id": task_id,
                "experiment_key": experiment_key,
                "experiment_slug": experiment_slug,
                "condition_code": condition_code,
                "condition_slug": condition_slug,
                "formal_statistics_eligible": eligible,
                "review_partition": review_partition,
                "review_status": REVIEW_PENDING_STATUS,
                "identity_hair_count": render_records[task_id][
                    "identity_hair_count"
                ],
                "endpoint_complete_length_count": render_records[task_id][
                    "endpoint_complete_length_count"
                ],
                "raw_source_image_sha256": render_records[task_id][
                    "raw_source_image_sha256"
                ],
                "prediction_sha256": render_records[task_id][
                    "prediction_sha256"
                ],
                "output_png_relative_path": (
                    f"{REVIEW_ROOT_NAME}/{relative_from_review_root}"
                ),
                "output_png_sha256": sha256_file(target),
                "output_png_bytes": target.stat().st_size,
                "width_px": render_records[task_id]["width_px"],
                "height_px": render_records[task_id]["height_px"],
            }
            review_rows.append(row)
            review_by_task[task_id] = row
        _require(not any(flat_render_root.iterdir()), "flat render staging is not empty")
        flat_render_root.rmdir()

        review_index_fields = tuple(review_rows[0])
        review_index_path = staging / REVIEW_INDEX_NAME
        _atomic_csv(
            review_index_path,
            review_rows,
            fields=review_index_fields,
        )
        checklist_fields = (
            "task_id",
            "experiment_key",
            "condition_code",
            "review_partition",
            "output_png_relative_path",
            "review_status",
            "root_continuity_ok",
            "distal_point_ok",
            "hair_identity_coverage_ok",
            "endpoint_complete_lengths_ok",
            "notes",
            "reviewer",
            "reviewed_at",
        )
        checklist_rows = [
            {
                "task_id": row["task_id"],
                "experiment_key": row["experiment_key"],
                "condition_code": row["condition_code"],
                "review_partition": row["review_partition"],
                "output_png_relative_path": row["output_png_relative_path"],
                "review_status": REVIEW_PENDING_STATUS,
                "root_continuity_ok": "",
                "distal_point_ok": "",
                "hair_identity_coverage_ok": "",
                "endpoint_complete_lengths_ok": "",
                "notes": "",
                "reviewer": "",
                "reviewed_at": "",
            }
            for row in review_rows
        ]
        review_checklist_path = staging / REVIEW_CHECKLIST_NAME
        _atomic_csv(
            review_checklist_path,
            checklist_rows,
            fields=checklist_fields,
        )
        _require(
            len(_rows(review_index_path))
            == len(_rows(review_checklist_path))
            == expected_task_count,
            "review index/checklist row count changed after serialization",
        )
        review_readme_path = staging / REVIEW_README_NAME
        _atomic_text(
            review_readme_path,
            _review_readme(expected_task_count=expected_task_count),
        )

        png_locks = [
            {
                "task_id": row["task_id"],
                "relative_path": row["output_png_relative_path"],
                "sha256": row["output_png_sha256"],
                "bytes": row["output_png_bytes"],
                "width_px": row["width_px"],
                "height_px": row["height_px"],
            }
            for row in review_rows
        ]
        observed_pngs = sorted(review_root.rglob("*.png"))
        _require(
            len(observed_pngs) == expected_task_count
            and not any(path.is_symlink() for path in observed_pngs),
            "review PNG tree is not an exact, symlink-free cohort",
        )
        expected_relative_paths = {
            str(row["output_png_relative_path"]) for row in review_rows
        }
        observed_relative_paths = {
            path.relative_to(staging).as_posix() for path in observed_pngs
        }
        _require(
            observed_relative_paths == expected_relative_paths,
            "review PNG tree has missing or extra files",
        )
        category_counts: list[dict[str, Any]] = []
        category_keys = sorted(
            {
                (
                    str(row["experiment_key"]),
                    str(row["condition_code"]),
                    str(row["review_partition"]),
                )
                for row in review_rows
            }
        )
        for experiment_key, condition_code, partition in category_keys:
            category_counts.append(
                {
                    "experiment_key": experiment_key,
                    "condition_code": condition_code,
                    "review_partition": partition,
                    "images": sum(
                        row["experiment_key"] == experiment_key
                        and row["condition_code"] == condition_code
                        and row["review_partition"] == partition
                        for row in review_rows
                    ),
                }
            )
        review_summary: dict[str, Any] = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "status": REVIEW_STATUS,
            "expected_task_count": expected_task_count,
            "images": expected_task_count,
            "index_rows": expected_task_count,
            "checklist_rows": expected_task_count,
            "review_root": REVIEW_ROOT_NAME,
            "index_csv": REVIEW_INDEX_NAME,
            "index_csv_sha256": sha256_file(review_index_path),
            "checklist_csv": REVIEW_CHECKLIST_NAME,
            "checklist_csv_sha256": sha256_file(review_checklist_path),
            "readme_cn": REVIEW_README_NAME,
            "readme_cn_sha256": sha256_file(review_readme_path),
            "ordered_task_set_identity_sha256": sha256_json(
                sorted(render_records)
            ),
            "overlay_png_set_identity_sha256": sha256_json(png_locks),
            "category_counts": category_counts,
            "source_authority_sha256": source_authority,
            "fusion_summary_identity_sha256": fusion_summary[
                "summary_identity_sha256"
            ],
            "renderer_contract": {
                "function": "phaxis.rendering.render_prediction_overlay",
                "include_text": False,
                "display_gamma": 1.0,
                "png_compression": 6,
                "renderer_source_sha256": source_authority["renderer_source"],
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "tifffile_version": tifffile.__version__,
            },
            "review_status_on_export": REVIEW_PENDING_STATUS,
            "organization_fields": [
                "experiment_key",
                "condition_code",
                "formal_statistics_eligible",
            ],
            "experimental_condition_metadata_used_for_prediction": False,
            "experimental_condition_metadata_used_for_rendering": False,
            "experimental_condition_metadata_used_for_evidence_assembly": False,
            "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
            ),
            "experimental_condition_metadata_used_for_output_organization": True,
            "create_only": True,
            "deterministic_task_order": "lexicographic_task_id",
            "canonical_annotations_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        review_summary["review_export_identity_sha256"] = sha256_json(
            review_summary
        )
        review_summary_path = staging / REVIEW_SUMMARY_NAME
        atomic_write_json(review_summary_path, review_summary)

        for plan in sorted(plan_rows, key=lambda row: CASE_ROLES.index(row["case_role"])):
            role = plan["case_role"]
            task_id = _safe_task_id(plan["task_id"])
            _require(task_id in manifest, f"{task_id}: case is not in exact cohort")
            manifest_row = manifest[task_id]
            trait_row = traits[task_id]
            raw_path = Path(render_records[task_id]["raw_path"])
            raw_sha = str(render_records[task_id]["raw_source_image_sha256"])
            _require(
                "blind" not in str(raw_path).casefold()
                and raw_path.is_file()
                and not raw_path.is_symlink()
                and sha256_file(raw_path) == raw_sha,
                f"{task_id}: raw image hash/path mismatch",
            )
            _require(str(trait_row.get("source_image_sha256")) == raw_sha, f"{task_id}: traits/raw image lock mismatch")
            prediction_path = fusion / "predictions" / f"{task_id}.json"
            _require(
                prediction_path.is_file() and not prediction_path.is_symlink(),
                f"{task_id}: fused prediction missing",
            )
            prediction_sha = sha256_file(prediction_path)
            _require(prediction_sha == fusion_records[task_id].get("prediction_sha256"), f"{task_id}: prediction hash mismatch")
            prediction = read_json(prediction_path)
            validate_hybrid_prediction(prediction, artifact_root=fusion)
            _require(prediction.get("source_image_sha256") == raw_sha, f"{task_id}: prediction/raw image lock mismatch")
            eligible = _strict_bool(
                trait_row.get("formal_statistics_eligible"),
                role=f"{task_id}: formal_statistics_eligible",
            )
            _require((role == "fail_closed") == (not eligible), f"{task_id}: fail-closed/formal eligibility role mismatch")

            image = tifffile.imread(raw_path)
            source_display = render_display_background(image, gamma=1.0)
            case_id = f"{role}__{task_id}"
            case_root = staging / "cases"
            case_root.mkdir(parents=True, exist_ok=True)
            source_target = case_root / f"{case_id}__source.png"
            overlay_target = case_root / f"{case_id}__overlay.png"
            _require(cv2.imwrite(str(source_target), source_display, [cv2.IMWRITE_PNG_COMPRESSION, 6]), f"{task_id}: source display write failed")
            full_review_row = review_by_task[task_id]
            full_review_overlay = staging / str(
                full_review_row["output_png_relative_path"]
            )
            _require(
                full_review_overlay.is_file()
                and sha256_file(full_review_overlay)
                == full_review_row["output_png_sha256"],
                f"{task_id}: full-cohort review overlay lock mismatch",
            )
            shutil.copyfile(full_review_overlay, overlay_target)
            _require(
                sha256_file(overlay_target)
                == full_review_row["output_png_sha256"],
                f"{task_id}: paper overlay differs from full-cohort review PNG",
            )
            scale = float(manifest_row.get("um_per_px") or prediction.get("scale", {}).get("predicted_um_per_px") or np.nan)
            _require(np.isfinite(scale) and scale > 0.0, f"{task_id}: no valid physical calibration")
            scale_um, scale_px = _scale_bar(scale, int(source_display.shape[1]))
            inset = _deterministic_inset(
                role=role,
                prediction=prediction,
                fusion_root=fusion,
                image_shape=source_display.shape,
            )
            output_rows.append(
                {
                    "task_id": task_id,
                    "prediction_sha256": prediction_sha,
                    "raw_source_image_sha256": raw_sha,
                    "case_id": case_id,
                    "case_role": role,
                    "source_path": str(source_target.relative_to(staging)).replace("\\", "/"),
                    "source_sha256": sha256_file(source_target),
                    "overlay_path": str(overlay_target.relative_to(staging)).replace("\\", "/"),
                    "overlay_sha256": sha256_file(overlay_target),
                    "full_cohort_review_overlay_path": full_review_row[
                        "output_png_relative_path"
                    ],
                    "full_cohort_review_overlay_sha256": full_review_row[
                        "output_png_sha256"
                    ],
                    "overlay_bytes_reused_from_full_cohort_review_export": True,
                    "scale_bar_um": scale_um,
                    "scale_bar_px": scale_px,
                    "display_lower": 0.0,
                    "display_upper": 255.0,
                    "selection_rule": SELECTION_RULES[role],
                    "case_selection_basis": CASE_SELECTION_BASIS,
                    "random_or_representative_performance_sample": False,
                    "experimental_condition_metadata_used_for_rendering": False,
                    "experimental_condition_metadata_used_for_evidence_assembly": False,
                    "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                        CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
                    ),
                    "formal_statistics_eligible": eligible,
                    **inset,
                    **COLOURS,
                }
            )
        selection_path = staging / "overlay_selection.csv"
        _atomic_csv(selection_path, output_rows)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": RECEIPT_STATUS,
            "selection_csv_sha256": sha256_file(selection_path),
            "source_authority_sha256": source_authority,
            "fusion_summary_identity_sha256": fusion_summary["summary_identity_sha256"],
            "case_roles": list(CASE_ROLES),
            "images": 5,
            "exact_cohort_review_images": expected_task_count,
            "case_plan_columns": ["case_role", "task_id"],
            "case_selection_basis": CASE_SELECTION_BASIS,
            "random_or_representative_performance_sample": False,
            "experimental_condition_metadata_used_for_rendering": False,
            "experimental_condition_metadata_used_for_evidence_assembly": False,
            "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
            ),
            "experimental_condition_metadata_used_for_output_organization": True,
            "source_and_overlay_rerendered_by_builder": True,
            "paper_overlay_bytes_reused_from_full_cohort_review_export": True,
            "paper_overlay_sha256_matches_full_cohort_review_export": True,
            "full_cohort_review_export": {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "status": REVIEW_STATUS,
                "expected_task_count": expected_task_count,
                "images": expected_task_count,
                "index_rows": expected_task_count,
                "checklist_rows": expected_task_count,
                "review_root": REVIEW_ROOT_NAME,
                "index_csv": REVIEW_INDEX_NAME,
                "index_csv_sha256": sha256_file(review_index_path),
                "checklist_csv": REVIEW_CHECKLIST_NAME,
                "checklist_csv_sha256": sha256_file(review_checklist_path),
                "readme_cn": REVIEW_README_NAME,
                "readme_cn_sha256": sha256_file(review_readme_path),
                "summary_json": REVIEW_SUMMARY_NAME,
                "summary_json_sha256": sha256_file(review_summary_path),
                "review_export_identity_sha256": review_summary[
                    "review_export_identity_sha256"
                ],
                "ordered_task_set_identity_sha256": review_summary[
                    "ordered_task_set_identity_sha256"
                ],
                "overlay_png_set_identity_sha256": review_summary[
                    "overlay_png_set_identity_sha256"
                ],
                "review_status_on_export": REVIEW_PENDING_STATUS,
                "organization_fields": review_summary["organization_fields"],
                "experimental_condition_metadata_used_for_prediction": False,
                "experimental_condition_metadata_used_for_rendering": False,
                "experimental_condition_metadata_used_for_evidence_assembly": False,
                "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                    CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
                ),
                "experimental_condition_metadata_used_for_output_organization": True,
                "create_only": True,
                "canonical_annotations_read": False,
                "root_cap_region_statistics_included": False,
                "blind_images_used": 0,
            },
            "display_contract": {
                "global_source_transform": "linear inverted 0.2-99.8 percentile",
                "gamma": 1.0,
                "source_and_overlay_share_identical_background_bytes": True,
            },
            "inset_contract": {
                "roles": list(INSET_RULES),
                "locked_anchor_task_ids": LOCKED_ANCHOR_TASK_IDS,
                "rules": INSET_RULES,
                "coordinates": "source_pixel_xyxy_left_top_inclusive_right_bottom_exclusive",
                "source_and_overlay_use_identical_crop_coordinates": True,
                "whole_image_context_retained": True,
                "performance_based_crop_selection": False,
            },
            "canonical_annotations_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["overlay_selection_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "overlay_selection_receipt.json", receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", required=True)
    parser.add_argument("--application-manifest", required=True)
    parser.add_argument("--full-traits", required=True)
    parser.add_argument("--fusion-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=DEFAULT_EXPECTED_TASK_COUNT,
        help="exact cohort size (formal Stage27 is locked to 283)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = build_overlay_evidence(
        case_plan=args.case_plan,
        application_manifest=args.application_manifest,
        full_traits=args.full_traits,
        fusion_root=args.fusion_root,
        output=args.output,
        expected_task_count=args.expected_task_count,
    )
    print(receipt["overlay_selection_identity_sha256"])


if __name__ == "__main__":
    main()
