#!/usr/bin/env python3
"""Materialize result-independent release case selections from exact283 inputs.

This producer is intentionally small and CPU-only.  The task identities and
their acquisition-challenge roles are declared in source code, so neither a
model output nor experimental-condition metadata can influence the selection.
It emits the two authorities needed later in the release DAG:

* the five-row overlay case plan consumed by the overlay renderer; and
* the single Figure 1 case receipt consumed by post-result geometry
  materialization.

Only the canonical application manifest is read.  Selected source-image bytes
are rehashed, but image pixels are not decoded or inspected.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-release-case-prelocks-1.0"
STATUS = "completed_result_independent_exact283_case_prelocks"
FIGURE1_SCHEMA = "PHAxis-figure1-case-selection-1.0"
FIGURE1_STATUS = "locked_before_model_result_consumption"
CASE_SELECTION_BASIS = "preregistered_fixed_task_ids_and_acquisition_challenge_roles"

# These identities were fixed by the pre-existing biological/acquisition case
# contract.  The two user-designated classic challenges remain challenge-panel
# cases; neither is reused as the routine Figure 1 anatomy example.
CASE_TASKS: tuple[tuple[str, str], ...] = (
    ("representative", "RHSCU-aed576d543e90377"),
    ("low_contrast", "RHSCU-aa5b6e37df15821f"),
    ("curved_dense", "RHSCU-bbf649822174e0a2"),
    ("continuity", "RHSCU-0d193f9385dd74c3"),
    ("fail_closed", "RHSCU-bc9223e70e962f9b"),
)
FIGURE1_TASK_ID = "RHSCU-aed576d543e90377"


class CasePrelockError(RuntimeError):
    """The exact283 manifest cannot satisfy the preregistered case contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CasePrelockError(message)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    _require(bool(rows), "application manifest is empty")
    return rows


def _resolve_image(manifest: Path, row: Mapping[str, str], task_id: str) -> tuple[Path, str]:
    value = row.get("image_path") or row.get("input_path")
    digest = str(row.get("image_sha256") or row.get("source_image_sha256") or "").casefold()
    _require(bool(value), f"{task_id}: source-image path is absent")
    _require(len(digest) == 64 and all(character in "0123456789abcdef" for character in digest), f"{task_id}: source-image SHA-256 is invalid")
    supplied = Path(str(value))
    image = supplied if supplied.is_absolute() else manifest.parent / supplied
    image = image.resolve()
    _require(image.is_file() and not image.is_symlink(), f"{task_id}: source image is absent or symlinked")
    _require("blind" not in str(image).casefold(), f"{task_id}: blind-labelled path is forbidden")
    _require(sha256_file(image) == digest, f"{task_id}: source-image SHA-256 mismatch")
    return image, digest


def _write_case_plan(path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_role", "task_id"), lineterminator="\n")
        writer.writeheader()
        for role, task_id in CASE_TASKS:
            writer.writerow({"case_role": role, "task_id": task_id})
        handle.flush()
        os.fsync(handle.fileno())


def build_case_prelocks(*, application_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    manifest = Path(application_manifest).resolve()
    destination = Path(output).resolve()
    _require(manifest.is_file() and not manifest.is_symlink(), "application manifest is absent or symlinked")
    _require("blind" not in str(manifest).casefold(), "blind-labelled application manifest is forbidden")
    _require(not destination.exists(), f"refusing to overwrite {destination}")

    rows = _rows(manifest)
    by_task = {str(row.get("task_id") or row.get("image_id") or ""): row for row in rows}
    _require(len(rows) == len(by_task) == 283 and all(by_task), "application manifest is not unique exact283")
    selected_ids = [task_id for _role, task_id in CASE_TASKS]
    _require(len(set(selected_ids)) == len(selected_ids) == 5, "source-declared case IDs are not five unique tasks")
    _require(set(selected_ids).issubset(by_task), "application manifest omits a preregistered release case")

    source_locks: list[dict[str, Any]] = []
    resolved: dict[str, tuple[Path, str]] = {}
    for role, task_id in CASE_TASKS:
        image, digest = _resolve_image(manifest, by_task[task_id], task_id)
        resolved[task_id] = (image, digest)
        source_locks.append(
            {
                "case_role": role,
                "task_id": task_id,
                "source_image_sha256": digest,
                "source_image_bytes": image.stat().st_size,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        case_plan = staging / "overlay_case_plan.csv"
        _write_case_plan(case_plan)
        figure_image, figure_sha = resolved[FIGURE1_TASK_ID]
        figure1: dict[str, Any] = {
            "schema_version": FIGURE1_SCHEMA,
            "status": FIGURE1_STATUS,
            "task_id": FIGURE1_TASK_ID,
            "case_role": "representative",
            "source_image_path": str(figure_image),
            "source_image_sha256": figure_sha,
            "source_image_bytes": figure_image.stat().st_size,
            "selection_basis": CASE_SELECTION_BASIS,
            "selected_before_model_result_consumption": True,
            "selected_by_prediction_or_trait_outcome": False,
            "classic_challenge_panel_task": False,
            "condition_metadata_read": False,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
        }
        figure1["figure1_case_selection_identity_sha256"] = sha256_json(figure1)
        figure_path = staging / "figure1_case_selection.json"
        atomic_write_json(figure_path, figure1)

        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "application_manifest_sha256": sha256_file(manifest),
            "application_tasks": len(rows),
            "overlay_case_plan_sha256": sha256_file(case_plan),
            "figure1_case_selection_sha256": sha256_file(figure_path),
            "figure1_case_selection_identity_sha256": figure1[
                "figure1_case_selection_identity_sha256"
            ],
            "case_selection_basis": CASE_SELECTION_BASIS,
            "source_locks": source_locks,
            "source_lock_set_identity_sha256": sha256_json(source_locks),
            "selection_code_authority": str(Path(__file__).resolve()),
            "selection_code_sha256": sha256_file(Path(__file__).resolve()),
            "model_outputs_read": False,
            "trait_outputs_read": False,
            "condition_metadata_read": False,
            "canonical_annotations_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["case_prelock_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return deepcopy(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_case_prelocks(
        application_manifest=args.application_manifest,
        output=args.output,
    )
    print(receipt["case_prelock_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
