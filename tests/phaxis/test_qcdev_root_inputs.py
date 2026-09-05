from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from phaxis.io import sha256_file, sha256_json
from scripts.phaxis.build_qcdev44_root_provider_inputs import (
    QCdevRootInputError,
    build_suite,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "canonical443"
    images = dataset / "images" / "all"
    images.mkdir(parents=True)
    ids = [f"RHAUD-{index:03d}" for index in range(1, 45)]
    inference_rows: list[dict[str, str]] = []
    dataset_rows: list[dict[str, str]] = []
    for index, task in enumerate(ids, start=1):
        image = images / f"{task}.ome.tif"
        image.write_bytes(f"immutable-image-{index}".encode("ascii"))
        width = 10 + index
        height = 20 + index
        megapixels = width * height / 1e6
        scale = 1.5 + index / 1000
        inference_rows.append(
            {
                "task_id": task,
                "image_path": str(image),
                "image_sha256": sha256_file(image),
                "um_per_px": repr(scale),
                "source_megapixels": repr(megapixels),
            }
        )
        dataset_rows.append(
            {
                "task_id": task,
                "split": "val",
                "image_relpath": image.relative_to(dataset).as_posix(),
                "image_sha256": sha256_file(image),
                "image_width": str(width),
                "image_height": str(height),
                "source_um_per_px": repr(scale),
                "source_megapixels": repr(megapixels),
                # Annotation-looking columns demonstrate that the producer
                # neither resolves nor opens them.
                "canonical_annotation_relpath": f"must-not-read/{task}.json",
            }
        )
    manifest = tmp_path / "qcdev44.csv"
    _write_csv(manifest, tuple(inference_rows[0]), inference_rows)
    dataset_manifest = dataset / "manifests" / "dataset_manifest.csv"
    _write_csv(dataset_manifest, tuple(dataset_rows[0]), dataset_rows)
    locked = tmp_path / "locked_ids.txt"
    locked.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return manifest, dataset, dataset_manifest, locked


def test_qcdev_root_suite_is_exact44_label_free_and_deployment_loadable(
    tmp_path: Path,
) -> None:
    manifest, dataset, dataset_manifest, locked = _fixture(tmp_path)
    output = tmp_path / "suite"
    result = build_suite(
        manifest=manifest,
        dataset_root=dataset,
        dataset_manifest=dataset_manifest,
        locked_val_ids=locked,
        output=output,
    )
    assert result["status"] == "completed_locked_exact44_label_free_source_contract"
    assert result["tasks"] == 44
    assert result["labels_or_annotation_files_read"] is False
    assert result["locked_members_posthoc_filtered"] is False
    assert result["acquisition_gate_can_remove_locked_member"] is False
    assert result["summary_identity_sha256"] == sha256_json(
        {key: value for key, value in result.items() if key != "summary_identity_sha256"}
    )
    assert sha256_file(output / "summary.json")

    with (output / "deployment_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        deployment_rows = list(csv.DictReader(handle))
    assert len(deployment_rows) == 44
    assert [row["task_id"] for row in deployment_rows] == result["task_ids"]
    for row in deployment_rows:
        image = dataset / row["image_relpath"]
        assert image.is_file()
        assert sha256_file(image) == row["image_sha256"]

    # Keep this source-release test portable: validate the exact deployment
    # lock contract from its bytes instead of importing the private predecessor
    # namespace that consumes the model-asset bundle at runtime.
    deployment_lock = json.loads(
        (output / "deployment_manifest_lock.json").read_text(encoding="utf-8")
    )
    manifest_sha = sha256_file(output / "deployment_manifest.csv")
    source_lock = [
        {
            "task_id": row["task_id"],
            "image_sha256": row["image_sha256"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "um_per_px": float(row["um_per_px"]),
        }
        for row in deployment_rows
    ]
    source_identity = sha256_json(source_lock)
    projection = [
        {
            "task_id": row["task_id"],
            "image_relpath": row["image_relpath"],
            "image_sha256": row["image_sha256"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "um_per_px": float(row["um_per_px"]),
        }
        for row in deployment_rows
    ]
    expected_deployment_identity = sha256_json(
        {
            "schema_version": "RHAxis-NextGen-deployment-identity-1.0",
            "manifest_sha256": manifest_sha,
            "source_qc_lock_identity_sha256": source_identity,
            "samples": projection,
        }
    )
    assert deployment_lock["samples"] == 44
    assert deployment_lock["manifest"] == "deployment_manifest.csv"
    assert deployment_lock["manifest_sha256"] == manifest_sha
    assert deployment_lock["source_qc_lock_identity_sha256"] == source_identity
    assert (
        deployment_lock["deployment_identity_sha256"]
        == expected_deployment_identity
    )
    assert deployment_lock["canonical_annotations_read"] is False
    assert deployment_lock["condition_used_for_model_routing"] is False
    assert deployment_lock["blind_images_used"] == 0

    with (output / "root_input_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == 44
    assert {row["analysis_mode"] for row in raw} == {"sparse_instance"}
    assert all(row["image_id"].startswith("RHAUD-") for row in raw)

    gate = json.loads((output / "acquisition_gate.json").read_text(encoding="utf-8"))
    assert gate["manual_phenotype_truth_used"] is False
    assert gate["calibration"]["method"] == (
        "locked_exact44_no_posthoc_acquisition_exclusion"
    )


def test_qcdev_root_suite_is_create_only_and_rejects_locked_order_drift(
    tmp_path: Path,
) -> None:
    manifest, dataset, dataset_manifest, locked = _fixture(tmp_path)
    output = tmp_path / "suite"
    build_suite(
        manifest=manifest,
        dataset_root=dataset,
        dataset_manifest=dataset_manifest,
        locked_val_ids=locked,
        output=output,
    )
    before = (output / "summary.json").read_bytes()
    with pytest.raises(QCdevRootInputError, match="refusing to overwrite"):
        build_suite(
            manifest=manifest,
            dataset_root=dataset,
            dataset_manifest=dataset_manifest,
            locked_val_ids=locked,
            output=output,
        )
    assert (output / "summary.json").read_bytes() == before

    drifted = tmp_path / "drifted_ids.txt"
    values = locked.read_text(encoding="utf-8").splitlines()
    values[0], values[1] = values[1], values[0]
    drifted.write_text("\n".join(values) + "\n", encoding="utf-8")
    with pytest.raises(QCdevRootInputError, match="order differs"):
        build_suite(
            manifest=manifest,
            dataset_root=dataset,
            dataset_manifest=dataset_manifest,
            locked_val_ids=drifted,
            output=tmp_path / "drifted-suite",
        )
