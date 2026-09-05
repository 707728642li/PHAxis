"""Reference hashes and CPU-only equivalence gates for the frozen root chain."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json


REFERENCE_SCHEMA = "PHAxis-root-provider-reference283-1.0"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def ndarray_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _root_shape(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    roots = [
        shape
        for shape in payload.get("shapes", [])
        if shape.get("label") == "root" and shape.get("shape_type") == "polygon"
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one V20 root polygon, found {len(roots)}")
    shape = roots[0]
    return {
        "label": "root",
        "shape_type": "polygon",
        "points": shape["points"],
        "geometry_role": shape.get("geometry_role"),
        "source_root_continuation_status": shape.get(
            "source_root_continuation_status"
        ),
    }


def _attachment_rescue_record(prediction: Mapping[str, Any]) -> dict[str, Any] | None:
    continuity = (
        prediction.get("fusion_audit", {})
        .get("geometry_gate", {})
        .get("root_continuity", {})
    )
    metrics = continuity.get("metrics", {})
    if not continuity.get("applied") or metrics.get("strict_extension_gate_passed"):
        return None
    attachment = bool(metrics.get("attachment_rescue_gate_passed"))
    strong = bool(metrics.get("strong_attachment_rescue_gate_passed"))
    if not (attachment or strong):
        return None
    return {
        "task_id": prediction["task_id"],
        "source_image_sha256": prediction["source_image_sha256"],
        "root_mask_sha256": prediction["root_mask_sha256"],
        "status": continuity["status"],
        "attachment_supported_extension_count": int(
            metrics["attachment_supported_extension_count"]
        ),
        "attachment_rescue_gate_passed": attachment,
        "strong_attachment_rescue_gate_passed": strong,
    }


def build_reference_registry(
    *,
    v1_root: str | Path,
    v20_root: str | Path,
    final_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    v1 = Path(v1_root).resolve()
    v20 = Path(v20_root).resolve()
    final = Path(final_root).resolve()
    v20_rows = _rows(v20 / "package_selection_manifest.csv")
    v20_by_id = {row["task_id"]: row for row in v20_rows}
    prediction_paths = sorted((final / "predictions").glob("*.json"))
    if len(prediction_paths) != 283 or len(v20_by_id) != 283:
        raise RuntimeError("root-provider reference requires exactly 283 tasks")
    records: list[dict[str, Any]] = []
    attachment_rescue: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        prediction = read_json(prediction_path)
        task_id = str(prediction["task_id"])
        if prediction.get("blind_images_used") != 0:
            raise RuntimeError(f"blind-tainted final prediction: {task_id}")
        v1_path = v1 / task_id / "masks.npz"
        with np.load(v1_path, allow_pickle=False) as archive:
            root_mask = np.asarray(archive["root_mask"])
            v1_mask_hash = ndarray_sha256(root_mask)
            v1_mask_shape = list(root_mask.shape)
            v1_mask_pixels = int(np.count_nonzero(root_mask))
        v20_row = v20_by_id[task_id]
        prefill_path = Path(v20_row["preannotation_path"]).resolve()
        if sha256_file(prefill_path) != v20_row["preannotation_sha256"]:
            raise RuntimeError(f"V20 prefill hash mismatch: {task_id}")
        prefill = read_json(prefill_path)
        if prefill.get("source_image_sha256") != prediction["source_image_sha256"]:
            raise RuntimeError(f"V20/final source identity mismatch: {task_id}")
        root_shape = _root_shape(prefill)
        final_mask_path = (final / prediction["root_mask_relpath"]).resolve()
        final_axis_path = (final / prediction["root_axis_geometry_relpath"]).resolve()
        if sha256_file(final_mask_path) != prediction["root_mask_sha256"]:
            raise RuntimeError(f"final root-mask hash mismatch: {task_id}")
        if sha256_file(final_axis_path) != prediction["root_axis_geometry_sha256"]:
            raise RuntimeError(f"final root-axis hash mismatch: {task_id}")
        rescue = _attachment_rescue_record(prediction)
        if rescue is not None:
            attachment_rescue.append(rescue)
        records.append(
            {
                "task_id": task_id,
                "source_image_sha256": prediction["source_image_sha256"],
                "v12_strip_root_mask_array_sha256": v1_mask_hash,
                "v12_strip_root_mask_shape_yx": v1_mask_shape,
                "v12_strip_root_mask_foreground_pixels": v1_mask_pixels,
                "v20_prefill_sha256": v20_row["preannotation_sha256"],
                "v20_root_polygon_sha256": sha256_json(root_shape),
                "v20_root_polygon_points": len(root_shape["points"]),
                "v20_source_root_continuation_status": root_shape[
                    "source_root_continuation_status"
                ],
                "final_root_mask_sha256": prediction["root_mask_sha256"],
                "final_root_axis_geometry_sha256": prediction[
                    "root_axis_geometry_sha256"
                ],
                "final_root_continuity_status": prediction[
                    "root_continuity_status"
                ],
                "final_root_source": prediction["root_source"],
                "prepared_radius_fallback": task_id
                in {"RHSCU-b6fd7e83d179c1d7", "RHSCU-bc9223e70e962f9b"},
                "attachment_supported_extension_rescue": rescue is not None,
            }
        )
    records.sort(key=lambda row: row["task_id"])
    attachment_rescue.sort(key=lambda row: row["task_id"])
    if len(attachment_rescue) != 7:
        raise RuntimeError(
            f"expected seven attachment-supported root rescues, found {len(attachment_rescue)}"
        )
    prepared_fallback = [
        row["task_id"] for row in records if row["prepared_radius_fallback"]
    ]
    if prepared_fallback != ["RHSCU-b6fd7e83d179c1d7", "RHSCU-bc9223e70e962f9b"]:
        raise RuntimeError("prepared-radius fallback identity drift")
    identity_payload = {
        "schema_version": REFERENCE_SCHEMA,
        "images": 283,
        "records": records,
        "prepared_radius_fallback_task_ids": prepared_fallback,
        "attachment_supported_extension_rescue": attachment_rescue,
    }
    payload = {
        **identity_payload,
        "status": "locked_cached_chain_reference",
        "reference_identity_sha256": sha256_json(identity_payload),
        "gates": {
            "v12_strip_root_mask_reference_coverage": "283/283",
            "v20_root_polygon_reference_coverage": "283/283",
            "final_hybrid_root_mask_reference_coverage": "283/283",
            "prepared_radii_fallback_count": 2,
            "attachment_supported_extension_rescue_count": 7,
        },
        "claim_boundary": {
            "cached_artifact_chain_verified": True,
            "fresh_portable_raw_image_rerun_completed": False,
            "fresh_283_exact_reproduction_claim_allowed": False,
            "reason": (
                "A fresh raw-image GPU rerun from the materialized portable bundle is "
                "still required; this registry locks the exact targets and audits the "
                "already completed frozen chain without reopening labels."
            ),
        },
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    atomic_write_json(output, payload)
    return payload


def verify_reference_registry(
    registry: str | Path,
    *,
    v1_root: str | Path,
    v20_root: str | Path,
    final_root: str | Path,
) -> dict[str, Any]:
    expected = read_json(registry)
    temporary = Path(registry).with_suffix(".verify.tmp.json")
    try:
        actual = build_reference_registry(
            v1_root=v1_root,
            v20_root=v20_root,
            final_root=final_root,
            output=temporary,
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if actual["reference_identity_sha256"] != expected.get(
        "reference_identity_sha256"
    ):
        raise RuntimeError("root-provider reference identity mismatch")
    return {
        "schema_version": "PHAxis-root-provider-reference283-verification-1.0",
        "status": "pass",
        "images": 283,
        "v12_strip_masks_exact": 283,
        "v20_root_polygons_exact": 283,
        "final_hybrid_root_masks_exact": 283,
        "prepared_radius_fallback_count": 2,
        "attachment_supported_extension_rescue_count": 7,
        "reference_identity_sha256": expected["reference_identity_sha256"],
        "fresh_portable_raw_image_rerun_completed": False,
        "blind_images_used": 0,
    }


def audit_fresh_reference(
    *,
    reference_registry: str | Path,
    fresh_v1_root: str | Path,
    fresh_v20_root: str | Path,
    fresh_final_root: str | Path,
    pipeline_state: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Compare a provenance-attested raw-image rerun with all 283 locked targets."""

    reference = read_json(reference_registry)
    state = read_json(pipeline_state)
    required_stages = {
        "prepare",
        "v1_shards",
        "merge_v1",
        "materialize_v1",
        "v20_shards",
        "merge_v20",
        "q8_shards",
        "merge_q8",
        "hybrid",
    }
    state_stages = set(state.get("completed_stages", ()))
    evidence = state.get("stage_evidence", {})

    def evidence_matches(stage: str, expected_root: str | Path) -> bool:
        record = evidence.get(stage, {})
        root = Path(expected_root).resolve()
        if Path(str(record.get("output", "__missing__"))).resolve() != root:
            return False
        files = record.get("files", {})
        if not isinstance(files, Mapping) or not files:
            return False
        return all(
            (root / relative).is_file()
            and sha256_file(root / relative) == digest
            for relative, digest in files.items()
        )

    evidence_gate = all(
        (
            evidence_matches("merge_v1", fresh_v1_root),
            evidence_matches("merge_v20", fresh_v20_root),
            evidence_matches("hybrid", fresh_final_root),
        )
    )
    state_gate = (
        state.get("raw_image_provider") is True
        and required_stages.issubset(state_stages)
        and evidence_gate
        and isinstance(state.get("bundle_identity_sha256"), str)
        and len(state["bundle_identity_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in state["bundle_identity_sha256"]
        )
        and state.get("canonical_annotations_read") is False
        and state.get("blind_images_used") == 0
    )
    temporary = Path(output).resolve().with_suffix(".fresh-registry.partial.json")
    try:
        fresh = build_reference_registry(
            v1_root=fresh_v1_root,
            v20_root=fresh_v20_root,
            final_root=fresh_final_root,
            output=temporary,
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    reference_by_id = {row["task_id"]: row for row in reference["records"]}
    fresh_by_id = {row["task_id"]: row for row in fresh["records"]}
    if set(reference_by_id) != set(fresh_by_id) or len(reference_by_id) != 283:
        raise RuntimeError("fresh/reference task identities are not exactly 283-aligned")

    field_by_layer = {
        "v12_strip_root_mask": "v12_strip_root_mask_array_sha256",
        "v20_root_polygon": "v20_root_polygon_sha256",
        "final_hybrid_root_mask": "final_root_mask_sha256",
    }
    layers: dict[str, Any] = {}
    for layer, field in field_by_layer.items():
        mismatches = [
            task_id
            for task_id in sorted(reference_by_id)
            if fresh_by_id[task_id][field] != reference_by_id[task_id][field]
        ]
        layers[layer] = {
            "exact": 283 - len(mismatches),
            "expected": 283,
            "mismatch_count": len(mismatches),
            "mismatch_task_ids": mismatches,
            "gate_pass": not mismatches,
        }
    source_mismatches = [
        task_id
        for task_id in sorted(reference_by_id)
        if fresh_by_id[task_id]["source_image_sha256"]
        != reference_by_id[task_id]["source_image_sha256"]
    ]
    prepared_reference = reference["prepared_radius_fallback_task_ids"]
    prepared_fresh = fresh["prepared_radius_fallback_task_ids"]
    rescue_reference = [
        row["task_id"] for row in reference["attachment_supported_extension_rescue"]
    ]
    rescue_fresh = [
        row["task_id"] for row in fresh["attachment_supported_extension_rescue"]
    ]
    all_exact = (
        state_gate
        and not source_mismatches
        and all(layer["gate_pass"] for layer in layers.values())
        and prepared_fresh == prepared_reference
        and rescue_fresh == rescue_reference
    )
    identity_payload = {
        "schema_version": "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "reference_identity_sha256": reference["reference_identity_sha256"],
        "fresh_reference_identity_sha256": fresh["reference_identity_sha256"],
        "bundle_identity_sha256": state.get("bundle_identity_sha256"),
        "pipeline_identity_sha256": state.get("pipeline_identity_sha256"),
        "layers": layers,
        "source_image_mismatch_task_ids": source_mismatches,
        "prepared_radius_fallback_task_ids": prepared_fresh,
        "attachment_supported_extension_rescue_task_ids": rescue_fresh,
        "pipeline_raw_image_provenance_gate": state_gate,
        "pipeline_stage_evidence_gate": evidence_gate,
    }
    payload = {
        **identity_payload,
        "status": "pass_exact_283" if all_exact else "fail_not_exact",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_identity_sha256": sha256_json(identity_payload),
        "fresh_portable_raw_image_rerun_completed": True,
        "fresh_283_exact_reproduction_claim_allowed": all_exact,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    atomic_write_json(output, payload)
    return payload


__all__ = [
    "audit_fresh_reference",
    "build_reference_registry",
    "ndarray_sha256",
    "verify_reference_registry",
]
