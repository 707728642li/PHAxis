from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts/phaxis"
sys.path.insert(0, str(SCRIPT_ROOT))

import handover_manifest_producers as producers  # noqa: E402


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(
    path: Path, payload: Mapping[str, Any], identity_field: str | None = None
) -> Path:
    value = dict(payload)
    if identity_field:
        value[identity_field] = _canonical_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _relative(project: Path, path: Path) -> str:
    return path.relative_to(project).as_posix()


def _attestation(project: Path) -> Path:
    scope = {field: True for field in producers.SCOPE_FIELDS}
    payload = {
        "schema_version": producers.ATTESTATION_SCHEMA,
        "status": "approved_for_formal_handover",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "authority_name": "synthetic release owner",
        "approval_reference": "TEST-ONLY-NOT-A-REAL-RELEASE",
        "scope_attestation": scope,
        "authorized_materialisation_roles": list(producers.MATERIALISATION_ROLES),
        "license_basis_by_materialisation_role": {
            role: "synthetic-test-owner-authorization"
            for role in producers.MATERIALISATION_ROLES
        },
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "historical_or_provisional_backfill_used": False,
    }
    return _json(
        project / "inputs/release_attestation.json",
        payload,
        "attestation_identity_sha256",
    )


def _dataset_inputs(project: Path) -> dict[str, Path]:
    source = project / "inputs/manual.ome.tif"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic-manual-image")
    raw = project / "inputs/raw.json"
    raw.write_text('{"synthetic": true}\n', encoding="utf-8")
    canonical_root = project / "canonical443"
    canonical = canonical_root / "annotations/rhaxis_canonical/canonical.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text('{"canonical": true}\n', encoding="utf-8")
    accepted = {f"RHAUD-{index:03d}" for index in range(1, 444)}
    image_manifest = _csv(
        project / "inputs/tiff_staging_manifest.csv",
        [
            {
                "task_id": task,
                "staged_tiff_path": _relative(project, source),
                "source_image_sha256": _hash(source),
                "staged_tiff_sha256": _hash(source),
                "bytes": source.stat().st_size,
                "status": "copied_verified",
            }
            for task in sorted(producers.EXACT_MANUAL_TASKS)
        ],
    )
    decisions = _csv(
        canonical_root / "manifests/filter_decisions_all500.csv",
        [
            {
                "task_id": task,
                "dataset_decision": "accepted_core" if task in accepted else "excluded_hard",
                "decision_reasons": "synthetic fixture decision",
                "review_notes": "preserved note",
                "returned_annotation_path": _relative(project, raw),
            }
            for task in sorted(producers.EXACT_MANUAL_TASKS)
        ],
    )
    canonical_manifest = _csv(
        canonical_root / "manifests/dataset_manifest.csv",
        [
            {
                "task_id": task,
                "dataset_version": producers.DATASET_ID_CANONICAL443,
                "image_relpath": f"images/all/{task}.ome.tif",
                "raw_annotation_relpath": f"annotations/raw_return/{task}.json",
                "canonical_annotation_relpath": "annotations/rhaxis_canonical/canonical.json",
                "image_sha256": _hash(source),
                "raw_annotation_sha256": _hash(raw),
            }
            for task in sorted(accepted)
        ],
    )
    integrity = _csv(
        canonical_root / "manifests/integrity_sha256.csv",
        [
            {
                "task_id": task,
                "role": "canonical_annotation",
                "relative_path": "annotations/rhaxis_canonical/canonical.json",
                "sha256": _hash(canonical),
                "size_bytes": canonical.stat().st_size,
            }
            for task in sorted(accepted)
        ],
    )
    _csv(
        canonical_root / "manifests/split_manifest.csv",
        [
            {"task_id": task, "split": "val" if index < 44 else "train"}
            for index, task in enumerate(sorted(accepted))
        ],
    )
    _json(
        canonical_root / "build_summary.json",
        {
            "schema_version": "RHAxis-standard-dataset-build-summary-1.0",
            "returned_tasks": 500,
            "accepted_core_tasks": 443,
            "train_tasks": 399,
            "val_tasks": 44,
            "verification_status": "passed",
            "blind_images_used": 0,
        },
    )
    _json(
        canonical_root / "verification_report.json",
        {
            "schema_version": "RHAxis-standard-dataset-verification-1.0",
            "status": "passed",
            "tasks": 443,
            "train": 399,
            "val": 44,
            "blind_images_used": 0,
        },
    )
    _json(
        canonical_root / "provenance.json",
        {
            "schema_version": "RHAxis-standard-dataset-provenance-1.0",
            "dataset_version": producers.DATASET_ID_CANONICAL443,
            "raw_annotations_modified": False,
            "canonical_geometry_modified": False,
            "blind_images_used": 0,
        },
    )
    for name in ("DATASET_CARD.md", "LICENSE_DATA.md", "README_CN.md"):
        (canonical_root / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    _json(
        canonical_root / "label_schema.json",
        {"schema_version": "RHAxis-label-schema-1.0"},
    )
    notes = project / "inputs/ALL500_DATA_NOTES_CN.md"
    notes.write_text("synthetic exact500 notes\n", encoding="utf-8")
    return {
        "manual_image_manifest": image_manifest,
        "all500_decisions": decisions,
        "canonical_dataset_root": canonical_root,
        "canonical_dataset_manifest": canonical_manifest,
        "canonical_integrity_manifest": integrity,
        "all500_notes": notes,
    }


def test_dataset_producer_exact_counts_check_execute_and_determinism(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _dataset_inputs(project)
    attestation = _attestation(project)
    first = project / "out/dataset-a.csv"
    check = producers.build_dataset_manifest(
        project_root=project,
        **inputs,
        release_attestation=attestation,
        output=first,
        execute=False,
    )
    assert check["status"] == "passed_check_only_not_written"
    assert not first.exists()
    produced = producers.build_dataset_manifest(
        project_root=project,
        **inputs,
        release_attestation=attestation,
        output=first,
        execute=True,
    )
    second = project / "out/dataset-b.csv"
    producers.build_dataset_manifest(
        project_root=project,
        **inputs,
        release_attestation=attestation,
        output=second,
        execute=True,
    )
    assert produced["rows"] == 500 + 500 + 443 + len(producers.DATASET_SUPPORT) + 1
    assert first.read_bytes() == second.read_bytes()
    rows = list(csv.DictReader(first.open(encoding="utf-8")))
    assert Counter(row["annotation_kind"] for row in rows)["manual500_source_image"] == 500
    with pytest.raises(producers.ProducerError, match="overwrite"):
        producers.build_dataset_manifest(
            project_root=project,
            **inputs,
            release_attestation=attestation,
            output=first,
            execute=True,
        )


def _image_inputs(project: Path) -> tuple[Path, Path, Path]:
    image_root = project / "images283"
    rows = []
    for index in range(283):
        image = image_root / f"condition-{index % 2}/bio-{index:03d}.ome.tif"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"bio-{index:03d}".encode("ascii"))
        rows.append(
            {
                "task_id": f"BIO-{index:03d}",
                "image_relpath": image.relative_to(image_root).as_posix(),
                "image_sha256": _hash(image),
                "image_bytes": image.stat().st_size,
                "temperature_c": "22" if index % 2 else "30",
                "genotype_or_construct": "RHD6-OE" if index % 3 else "WT",
                "condition_code": "fixture",
                "study_role": "synthetic",
                "qc_disposition": "eligible",
            }
        )
    manifest = _csv(project / "inputs/deployment_manifest.csv", rows)
    lock = _json(
        project / "inputs/deployment_manifest_lock.json",
        {
            "schema_version": "RHAxis-NextGen-deployment-manifest-lock-1.0",
            "status": "locked_before_phenotype_inference",
            "samples": 283,
            "manifest": manifest.name,
            "manifest_sha256": _hash(manifest),
            "deployment_identity_sha256": "a" * 64,
            "canonical_annotations_read": False,
            "phenotype_model_predictions_used": False,
            "condition_used_for_model_routing": False,
            "blind_images_used": 0,
        },
    )
    return image_root, manifest, lock


def test_image_and_model_source_producers_are_exact_and_fail_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    attestation = _attestation(project)
    image_root, deployment, lock = _image_inputs(project)
    image_output = project / "out/image_manifest.csv"
    report = producers.build_image_manifest(
        project_root=project,
        deployment_manifest=deployment,
        deployment_lock=lock,
        image_root=image_root,
        release_attestation=attestation,
        output=image_output,
        execute=True,
    )
    assert report["rows"] == 283
    image_output_2 = project / "out/image_manifest-2.csv"
    producers.build_image_manifest(
        project_root=project,
        deployment_manifest=deployment,
        deployment_lock=lock,
        image_root=image_root,
        release_attestation=attestation,
        output=image_output_2,
        execute=True,
    )
    assert image_output.read_bytes() == image_output_2.read_bytes()
    tampered = image_root / "condition-0/bio-000.ome.tif"
    tampered.write_bytes(b"tampered")
    with pytest.raises(producers.ProducerError, match="SHA-256 mismatch"):
        producers.build_image_manifest(
            project_root=project,
            deployment_manifest=deployment,
            deployment_lock=lock,
            image_root=image_root,
            release_attestation=attestation,
            output=project / "out/image_manifest-tamper.csv",
            execute=False,
        )

    release_root = project / "formal-source"
    runtime = release_root / "src/phaxis/runtime.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("print('fixture')\n", encoding="utf-8")
    gate = _json(
        release_root / "FORMAL_RELEASE_GATE_RECEIPT.json",
        {
            "schema_version": "PHAxis-source-release-gate-1.0",
            "status": "passed",
            "formal_release_allowed": True,
        },
    )
    supply_chain = []
    license_inventory = {
        "schema_version": "PHAxis-third-party-license-inventory-1.0",
        "status": "complete_declared_direct_dependency_inventory",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "dependencies": [{"name": "numpy"}],
    }
    license_inventory["inventory_identity_sha256"] = _canonical_hash(
        license_inventory
    )
    for relative, content in (
        ("NOTICE", "PHAxis 1.0.0\nApache-2.0\n"),
        (
            "THIRD_PARTY_NOTICES.md",
            "# PHAxis 1.0.0 third-party notices\n",
        ),
        (
            "THIRD_PARTY_LICENSES.json",
            json.dumps(license_inventory, sort_keys=True) + "\n",
        ),
        (
            "SBOM.cdx.json",
            json.dumps(
                {
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.6",
                    "metadata": {
                        "component": {
                            "type": "application",
                            "name": "phaxis",
                            "version": "1.0.0",
                        }
                    },
                    "components": [{"type": "library", "name": "numpy"}],
                },
                sort_keys=True,
            )
            + "\n",
        ),
    ):
        path = release_root / relative
        path.write_text(content, encoding="utf-8", newline="\n")
        supply_chain.append(path)
    files = [
        {
            "path": path.relative_to(release_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _hash(path),
            "origin": "synthetic-project-source",
        }
        for path in sorted(
            (gate, runtime, *supply_chain),
            key=lambda item: item.relative_to(release_root).as_posix(),
        )
    ]
    source_manifest = _json(
        release_root / "SOURCE_MANIFEST.json",
        {
            "schema_version": "PHAxis-source-release-manifest-2.0",
            "distribution": "phaxis",
            "version": "1.0.0",
            "release_mode": "formal",
            "source_policy": "explicit_path_bounded_allowlist",
            "files": files,
            "tree_identity_sha256": _canonical_hash(files),
        },
    )
    source_output = project / "out/model_source.csv"
    result = producers.build_model_source_manifest(
        project_root=project,
        source_release_root=release_root,
        source_release_manifest=source_manifest,
        release_attestation=attestation,
        output=source_output,
        execute=True,
    )
    assert result["source_files"] == 7
    assert result["manifest_control_files"] == 1
    source_rows = list(csv.DictReader(source_output.open(encoding="utf-8")))
    manifest_rows = [
        row
        for row in source_rows
        if row["package_path"]
        == "model/source_release/SOURCE_MANIFEST.json"
    ]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["sha256"] == _hash(source_manifest)
    source_package_paths = {row["package_path"] for row in source_rows}
    assert {
        "model/source_release/NOTICE",
        "model/source_release/THIRD_PARTY_NOTICES.md",
        "model/source_release/THIRD_PARTY_LICENSES.json",
        "model/source_release/SBOM.cdx.json",
    }.issubset(source_package_paths)
    source_output_2 = project / "out/model_source-2.csv"
    producers.build_model_source_manifest(
        project_root=project,
        source_release_root=release_root,
        source_release_manifest=source_manifest,
        release_attestation=attestation,
        output=source_output_2,
        execute=True,
    )
    assert source_output.read_bytes() == source_output_2.read_bytes()

    notice_path = release_root / "NOTICE"
    valid_notice = notice_path.read_text(encoding="utf-8")
    notice_path.write_text(
        "RHPheno legacy workspace notice\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    notice_record = next(
        row for row in manifest_payload["files"] if row["path"] == "NOTICE"
    )
    notice_record["bytes"] = notice_path.stat().st_size
    notice_record["sha256"] = _hash(notice_path)
    manifest_payload["tree_identity_sha256"] = _canonical_hash(
        manifest_payload["files"]
    )
    _json(source_manifest, manifest_payload)
    with pytest.raises(producers.ProducerError, match="legacy/non-PHAxis notice"):
        producers.build_model_source_manifest(
            project_root=project,
            source_release_root=release_root,
            source_release_manifest=source_manifest,
            release_attestation=attestation,
            output=project / "out/source-legacy-notice.csv",
            execute=False,
        )
    notice_path.write_text(valid_notice, encoding="utf-8", newline="\n")
    notice_record["bytes"] = notice_path.stat().st_size
    notice_record["sha256"] = _hash(notice_path)
    manifest_payload["tree_identity_sha256"] = _canonical_hash(
        manifest_payload["files"]
    )
    _json(source_manifest, manifest_payload)

    (release_root / "unlisted.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(producers.ProducerError, match="differs"):
        producers.build_model_source_manifest(
            project_root=project,
            source_release_root=release_root,
            source_release_manifest=source_manifest,
            release_attestation=attestation,
            output=project / "out/source-tamper.csv",
            execute=False,
        )


def _model_asset_inputs(project: Path) -> dict[str, Any]:
    checkpoints: list[Path] = []
    members: list[dict[str, Any]] = []
    for index, seed in enumerate(producers.FORMAL_TRAIN399_SEEDS):
        checkpoint = project / "inputs/checkpoints" / f"member-{index}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"strict-train399-member-{index}".encode("ascii"))
        checkpoints.append(checkpoint)
        members.append(
            {
                "member_index": index,
                "seed": seed,
                "checkpoint_sha256": _hash(checkpoint),
            }
        )
    candidate_identity_payload = {"members": members}
    candidate = _json(
        project / "inputs/candidate.json",
        {
            "schema_version": "PHAxis-StageB-train399-candidate-bundle-1.0",
            "status": "candidate_gate_passed_not_promoted",
            "identity_payload": candidate_identity_payload,
            "candidate_bundle_identity_sha256": _canonical_hash(
                candidate_identity_payload
            ),
            "blind_images_used": 0,
        },
        "candidate_manifest_identity_sha256",
    )

    bundle_root = project / "inputs/root-provider"
    root_asset = bundle_root / "weights/root.bin"
    root_asset.parent.mkdir(parents=True, exist_ok=True)
    root_asset.write_bytes(b"portable-root-provider")
    root_files = [
        {
            "path": "weights/root.bin",
            "sha256": _hash(root_asset),
            "bytes": root_asset.stat().st_size,
            "roles": ["root-provider"],
        }
    ]
    root_identity_payload = {
        "schema_version": "PHAxis-root-provider-model-bundle-1.0",
        "bundle_id": "synthetic-root-provider",
        "files": root_files,
        "contracts": {},
    }
    root_identity = _canonical_hash(root_identity_payload)
    root_manifest = _json(
        bundle_root / "root_provider_bundle.json",
        {
            **root_identity_payload,
            "status": "materialized_unverified",
            "files_count": 1,
            "bytes": root_asset.stat().st_size,
            "bundle_identity_sha256": root_identity,
            "portable_execution_contract": {
                "implicit_download": False,
                "blind_images_used": 0,
            },
        },
    )
    root_verification = _json(
        project / "inputs/root_provider_bundle_verification.json",
        {
            "schema_version": "PHAxis-root-provider-model-bundle-verification-1.0",
            "status": "pass",
            "bundle_id": "synthetic-root-provider",
            "bundle_identity_sha256": root_identity,
            "files_verified": 1,
            "bytes_verified": root_asset.stat().st_size,
            "exact_file_closure_required": True,
            "exact_file_closure_passed": True,
            "unlisted_file_count": 0,
            "missing_closure_file_count": 0,
            "blind_images_used": 0,
        },
    )
    proposal = _json(
        project / "inputs/model-contract-proposal.json",
        {
            "schema_version": "PHAxis-model-contract-1.0.0",
            "formal_release_status": "passed_proposal_not_official",
            "blind_images_used": 0,
        },
        "model_contract_identity_sha256",
    )
    proposal_payload = json.loads(proposal.read_text(encoding="utf-8"))
    applied = _json(
        project / "inputs/applied-model-contract.json",
        {
            "schema_version": "PHAxis-model-contract-1.0.0",
            "product": "PHAxis",
            "product_version": "1.0.0",
            "model_bundle_id": "PHAXIS-SYNTHETIC-STRICT-TRAIN399",
            "formal_release_status": "passed",
            "promotion": {
                "status": "applied_formal_release",
                "official_apply_performed": True,
                "proposal_file_sha256": _hash(proposal),
                "proposal_identity_sha256": proposal_payload[
                    "model_contract_identity_sha256"
                ],
            },
            "red_lines": {
                "blind_images_used": 0,
                "root_cap_region_statistics_included": False,
                "formal_train399_only_stageb_weights_available": True,
            },
            "hair_identity_count_expert": {
                "expert_id": "PHAxis-StageB-train399-five-seed",
                "checkpoint_sha256_in_member_order": [
                    member["checkpoint_sha256"] for member in members
                ]
            },
            "root_expert": {
                "expert_id": "PHAxis-root-provider-synthetic",
                "bundle_identity_sha256": root_identity,
            },
        },
        "model_contract_identity_sha256",
    )
    release_example = project / "inputs/release-example"
    example_image = release_example / "inputs/sample_source_image.ome.tif"
    example_image.parent.mkdir(parents=True, exist_ok=True)
    example_image.write_bytes(b"real-nonblind-release-example")
    for name in (
        "root_input_manifest.csv",
        "stageb_input_manifest.csv",
        "traits_metadata.csv",
    ):
        (release_example / "inputs" / name).write_text(
            "task_id,image_sha256,um_per_px\nBIO-001," + _hash(example_image) + ",2.0\n",
            encoding="utf-8",
        )
    root_authorities: dict[str, Path] = {}
    for field, suffix in (
        ("acquisition_gate", ".json"),
        ("deployment_metadata", ".csv"),
        ("canonical_manifest", ".csv"),
        ("deployment_manifest", ".csv"),
        ("deployment_lock", ".json"),
        ("reference_registry", ".json"),
    ):
        path = release_example / "inputs" / f"{field}{suffix}"
        if suffix == ".csv":
            path.write_text("task_id,value\nBIO-001,portable\n", encoding="utf-8")
        else:
            path.write_text(
                json.dumps({"field": field, "blind_images_used": 0}) + "\n",
                encoding="utf-8",
            )
        root_authorities[field] = path
    selected = _json(
        project / "inputs/selected_model_metadata.json",
        {"checkpoint_policy": "five_seed_train399_last_epoch_60", "blind_images_used": 0},
    )
    selection = _json(
        project / "inputs/selection_receipt.json",
        {"status": "selected", "blind_images_used": 0},
    )
    profile = _json(
        project / "inputs/profile_contract.json",
        {
            "root_cap_region_output": False,
            "canonical_annotations_read": False,
            "stageb_two_point_vector_used_as_length": False,
            "blind_images_used": 0,
        },
    )
    def locked(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": _hash(path)}

    workflow = _json(
        release_example / "release_example_manifest.json",
        {
            "schema_version": "PHAxis-analysis-workflow-manifest-1.0",
            "model_contract_proposal": locked(proposal),
            "root_provider": {
                "project": ".",
                "bundle": {
                    "path": _relative(project, bundle_root),
                    "registry_sha256": _hash(root_manifest),
                    "bundle_identity_sha256": root_identity,
                },
                "input_manifest": locked(
                    release_example / "inputs/root_input_manifest.csv"
                ),
                "image_root": "inputs",
                **{field: locked(path) for field, path in root_authorities.items()},
                "v1_physical_gpus": [0],
                "q8_physical_gpus": [0],
            },
            "stageb": {
                "input_manifest": locked(
                    release_example / "inputs/stageb_input_manifest.csv"
                ),
                "image_root": "inputs",
                "checkpoints": [locked(path) for path in checkpoints],
                "candidate_manifest": locked(candidate),
                "selected_model_metadata": locked(selected),
                "selection_receipt": locked(selection),
                "physical_gpu": 0,
                "internal_device": "cuda:0",
            },
            "traits": {
                "metadata_csv": locked(
                    release_example / "inputs/traits_metadata.csv"
                )
            },
            "distal_axis_profiles": {"contract_json": locked(profile)},
            "benchmark_contract": {
                "ordered_raw_source_manifest": locked(
                    release_example / "inputs/stageb_input_manifest.csv"
                )
            },
            "guards": {
                "blind_images_used": 0,
                "canonical_annotations_read": False,
                "condition_metadata_used_for_routing": False,
                "root_cap_region_output": False,
            },
            "release_example": {
                "input_kind": "real_nonblind_release_example",
                "release_authorized": True,
                "tasks": 1,
                "task_id": "BIO-001",
                "source_image_relpath": "inputs/sample_source_image.ome.tif",
                "source_image_sha256": _hash(example_image),
                "blind_images_used": 0,
            },
        },
        "manifest_identity_sha256",
    )
    _json(
        release_example / "receipt.json",
        {
            "schema_version": "PHAxis-clean-install-sample-input-suite-1.0",
            "status": "completed_real_nonblind_release_example_manifest",
            "input_kind": "real_nonblind_release_example",
            "release_authorized": True,
            "development_or_synthetic_smoke": False,
            "tasks": 1,
            "release_example_manifest_sha256": _hash(workflow),
            "release_example_manifest_identity_sha256": json.loads(
                workflow.read_text(encoding="utf-8")
            )["manifest_identity_sha256"],
            "blind_images_used": 0,
        },
        "sample_input_suite_identity_sha256",
    )
    return {
        "applied": applied,
        "candidate": candidate,
        "checkpoints": checkpoints,
        "bundle_root": bundle_root,
        "root_manifest": root_manifest,
        "root_verification": root_verification,
        "release_example": release_example,
        "proposal": proposal,
    }


def test_model_asset_producer_seals_exact_five_members_and_is_deterministic(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    attestation = _attestation(project)
    inputs = _model_asset_inputs(project)

    def build(suffix: str, *, execute: bool) -> dict[str, Any]:
        return producers.build_model_asset_manifest(
            project_root=project,
            applied_model_contract=inputs["applied"],
            candidate_manifest=inputs["candidate"],
            checkpoint_paths=inputs["checkpoints"],
            root_provider_bundle_root=inputs["bundle_root"],
            root_provider_bundle_manifest=inputs["root_manifest"],
            root_provider_verification_receipt=inputs["root_verification"],
            release_example_root=inputs["release_example"],
            portable_capsule_output=project / f"out/{suffix}/portable_capsule",
            bundle_manifest_output=project / f"out/{suffix}/MODEL_BUNDLE_MANIFEST.json",
            release_attestation=attestation,
            output=project / f"out/{suffix}/model_asset_manifest.csv",
            execute=execute,
        )

    check = build("check", execute=False)
    repeat_check = build("check", execute=False)
    assert check["status"] == "passed_check_only_not_written"
    assert check == repeat_check
    assert not (project / "out/check/model_asset_manifest.csv").exists()
    first = build("first", execute=True)
    second = build("second", execute=True)
    assert first["model_bundle_manifest_sha256"] == second[
        "model_bundle_manifest_sha256"
    ]
    assert (project / "out/first/MODEL_BUNDLE_MANIFEST.json").read_bytes() == (
        project / "out/second/MODEL_BUNDLE_MANIFEST.json"
    ).read_bytes()
    with (project / "out/first/model_asset_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        first_rows = list(csv.DictReader(handle))
    assert Counter(row["asset_role"] for row in first_rows) == {
        "stageb_checkpoint": 5,
        "root_provider_asset": 1,
        "root_provider_authority": 2,
        "runtime_authority": 12,
        "release_example_input": 6,
        "release_example_provenance": 1,
        "model_bundle_manifest": 1,
    }
    capsule = project / "out/first/portable_capsule"
    portable_manifest_path = (
        capsule / "model/examples/clean_install/release_example_manifest.json"
    )
    portable = json.loads(portable_manifest_path.read_text(encoding="utf-8"))
    assert portable["root_provider"]["project"] == "."
    assert "python_executable" not in portable["root_provider"]
    assert "reference_registry" not in portable["root_provider"]
    assert portable["release_example"]["portable_capsule_finalized"] is True
    assert portable_manifest_path.read_bytes() == (
        project
        / "out/second/portable_capsule/model/examples/clean_install/release_example_manifest.json"
    ).read_bytes()

    # Simulate a hard stop after the directory target was atomically published
    # but before its two sibling files (and therefore before the stage receipt).
    build("interrupted", execute=True)
    interrupted_root = project / "out/interrupted"
    capsule_hashes = {
        path.relative_to(interrupted_root / "portable_capsule").as_posix(): _hash(path)
        for path in (interrupted_root / "portable_capsule").rglob("*")
        if path.is_file()
    }
    (interrupted_root / "MODEL_BUNDLE_MANIFEST.json").unlink()
    (interrupted_root / "model_asset_manifest.csv").unlink()
    recovered = build("interrupted", execute=True)
    assert recovered["partial_publish_recovery_used"] is True
    assert recovered["recovered_existing_outputs"] == [
        "out/interrupted/portable_capsule"
    ]
    assert recovered["outputs_created_this_invocation"] == [
        "out/interrupted/MODEL_BUNDLE_MANIFEST.json",
        "out/interrupted/model_asset_manifest.csv",
    ]
    assert capsule_hashes == {
        path.relative_to(interrupted_root / "portable_capsule").as_posix(): _hash(path)
        for path in (interrupted_root / "portable_capsule").rglob("*")
        if path.is_file()
    }
    fully_recovered = build("interrupted", execute=True)
    assert fully_recovered["partial_publish_recovery_used"] is True
    assert fully_recovered["outputs_created_this_invocation"] == []

    build("tampered-prefix", execute=True)
    tampered_root = project / "out/tampered-prefix"
    (tampered_root / "MODEL_BUNDLE_MANIFEST.json").unlink()
    (tampered_root / "model_asset_manifest.csv").unlink()
    tampered_member = (
        tampered_root
        / "portable_capsule/model/examples/clean_install/receipt.json"
    )
    tampered_member.write_bytes(tampered_member.read_bytes() + b"tamper")
    with pytest.raises(producers.ProducerError, match="member differs"):
        build("tampered-prefix", execute=True)
    assert not (tampered_root / "MODEL_BUNDLE_MANIFEST.json").exists()
    assert not (tampered_root / "model_asset_manifest.csv").exists()
    for section, field in (
        ("root_provider", "acquisition_gate"),
        ("root_provider", "deployment_manifest"),
        ("stageb", "candidate_manifest"),
        ("traits", "metadata_csv"),
        ("distal_axis_profiles", "contract_json"),
    ):
        path = (portable_manifest_path.parent / portable[section][field]["path"]).resolve()
        assert path.is_file()
        path.relative_to(capsule.resolve())
    bundle = json.loads(
        (project / "out/first/MODEL_BUNDLE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    identity = bundle.pop("model_bundle_manifest_identity_sha256")
    assert identity == _canonical_hash(bundle)
    assert bundle["schema_version"] == "PHAxis-model-bundle-release-manifest-1.0"
    assert bundle["status"] == "completed_final_immutable_bundle"
    with pytest.raises(producers.ProducerError, match="differs"):
        producers.build_model_asset_manifest(
            project_root=project,
            applied_model_contract=inputs["applied"],
            candidate_manifest=inputs["candidate"],
            checkpoint_paths=list(reversed(inputs["checkpoints"])),
            root_provider_bundle_root=inputs["bundle_root"],
            root_provider_bundle_manifest=inputs["root_manifest"],
            root_provider_verification_receipt=inputs["root_verification"],
            release_example_root=inputs["release_example"],
            portable_capsule_output=project / "out/bad/portable_capsule",
            bundle_manifest_output=project / "out/bad/MODEL_BUNDLE_MANIFEST.json",
            release_attestation=attestation,
            output=project / "out/bad/model_asset_manifest.csv",
            execute=False,
        )


def _benchmark_inputs(project: Path) -> tuple[Path, Path]:
    hardware = "a" * 64
    receipt = project / "inputs/benchmark/same-hardware.json"
    latency_paths: list[Path] = []
    for name in ("phaxis", "v1"):
        latency = project / f"inputs/benchmark/{name}-per-image.csv"
        latency.parent.mkdir(parents=True, exist_ok=True)
        latency.write_text("task_id,latency_s\nBIO-001,0.01\n", encoding="utf-8")
        latency_paths.append(latency)

    summaries: dict[str, Path] = {}
    for implementation in ("phaxis", "v1"):
        summaries[f"{implementation}_production_summary"] = _json(
            project / f"inputs/benchmark/{implementation}-production.json",
            {
                "schema_version": "PHAxis-full-workflow-production-batch-benchmark-1.0",
                "status": "completed_direct_full283",
                "benchmark_mode": "production_batch_full283",
                "images": 283,
                "hardware_identity_sha256": hardware,
                "blind_images_used": 0,
                "rootcap_region_metric": False,
            },
            "summary_identity_sha256",
        )
    for implementation, latency in zip(("phaxis", "v1"), latency_paths, strict=True):
        summaries[f"{implementation}_sequential_summary"] = _json(
            project / f"inputs/benchmark/{implementation}-sequential.json",
            {
                "schema_version": "PHAxis-full-workflow-sequential-latency-benchmark-1.0",
                "status": "completed_direct_full283",
                "benchmark_mode": "sequential_persistent_full283",
                "images": 283,
                "hardware_identity_sha256": hardware,
                "per_image_csv_sha256": _hash(latency),
                "blind_images_used": 0,
                "rootcap_region_metric": False,
            },
            "summary_identity_sha256",
        )
    comparisons: dict[str, Path] = {}
    for mode in ("production", "sequential"):
        comparisons[f"{mode}_comparison_receipt"] = _json(
            project / f"inputs/benchmark/{mode}-comparison.json",
            {
                "schema_version": "PHAxis-full-workflow-benchmark-comparison-1.0",
                "status": "comparable_direct_full283",
                "comparable": True,
                "same_283_source_manifest_hardware_and_io_scope": True,
                "phaxis_summary_sha256": _hash(
                    summaries[f"phaxis_{mode}_summary"]
                ),
                "baseline_summary_sha256": _hash(summaries[f"v1_{mode}_summary"]),
                "blind_images_used": 0,
                "rootcap_region_metric": False,
            },
            "comparison_identity_sha256",
        )
    telemetry = _json(
        project / "inputs/benchmark/gpu-telemetry.json", {"gpu": "synthetic"}
    )
    preflight = _json(
        project / "inputs/benchmark/hardware-preflight.json",
        {"hardware_identity_sha256": hardware},
    )
    receipt_run_roles = (
        ("phaxis_production", "phaxis_production_summary"),
        ("phaxis_sequential", "phaxis_sequential_summary"),
        ("frozen_v1_production", "v1_production_summary"),
        ("frozen_v1_sequential", "v1_sequential_summary"),
    )
    receipt = _json(
        receipt,
        {
            "schema_version": "PHAxis-same-hardware-benchmark-receipt-1.0",
            "status": "passed",
            "product": "PHAxis",
            "product_version": "1.0.0",
            "images": 283,
            "hardware_identity_sha256": hardware,
            "runs": [
                {
                    "role": receipt_role,
                    "mode": json.loads(summaries[summary_role].read_text(encoding="utf-8"))[
                        "benchmark_mode"
                    ],
                    "summary_sha256": _hash(summaries[summary_role]),
                    "summary_identity_sha256": json.loads(
                        summaries[summary_role].read_text(encoding="utf-8")
                    )["summary_identity_sha256"],
                    "hardware_identity_sha256": hardware,
                    "fresh_direct_run": True,
                    "resume_or_cache_used": False,
                    "full_workflow_io_included": True,
                }
                for receipt_role, summary_role in receipt_run_roles
            ],
            "comparisons": {
                mode: {
                    "comparison_sha256": _hash(
                        comparisons[f"{mode}_comparison_receipt"]
                    ),
                    "comparison_identity_sha256": json.loads(
                        comparisons[f"{mode}_comparison_receipt"].read_text(
                            encoding="utf-8"
                        )
                    )["comparison_identity_sha256"],
                }
                for mode in ("production", "sequential")
            },
            "same_ordered_exact283_sources": True,
            "same_hardware_uuid_and_driver": True,
            "same_io_and_full_workflow_scope": True,
            "fresh_no_cache": True,
            "historical_98_47_min_component_receipt_used": False,
            "forward_only_runtime_used": False,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "receipt_identity_sha256",
    )
    artifacts: list[tuple[str, Path]] = [
        ("same_hardware_receipt", receipt),
        *summaries.items(),
        *comparisons.items(),
        *(("per_image_latency_csv", path) for path in latency_paths),
        ("gpu_telemetry", telemetry),
        ("hardware_preflight", preflight),
    ]
    rows = [
        {
            "source_path": _relative(project, path),
            "package_path": f"model/benchmark/{index:02d}-{path.name}",
            "sha256": _hash(path),
            "bytes": path.stat().st_size,
            "provenance": "synthetic explicit benchmark inventory",
            "notes": "synthetic test evidence",
            "release_authorized": "true",
            "artifact_role": role,
        }
        for index, (role, path) in enumerate(artifacts)
    ]
    return receipt, _csv(project / "inputs/benchmark/inventory.csv", rows)


def test_benchmark_manifest_requires_complete_explicit_evidence_inventory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    attestation = _attestation(project)
    receipt, inventory = _benchmark_inputs(project)
    output = project / "out/benchmark_manifest.csv"
    check = producers.build_benchmark_manifest(
        project_root=project,
        same_hardware_receipt=receipt,
        artifact_inventory=inventory,
        release_attestation=attestation,
        output=output,
        execute=False,
    )
    assert check["status"] == "passed_check_only_not_written"
    assert not output.exists()
    built = producers.build_benchmark_manifest(
        project_root=project,
        same_hardware_receipt=receipt,
        artifact_inventory=inventory,
        release_attestation=attestation,
        output=output,
        execute=True,
    )
    assert built["artifacts"] == 11
    assert "inputs/benchmark/" in output.read_text(encoding="utf-8")
    deterministic_output = project / "out/benchmark_manifest-2.csv"
    producers.build_benchmark_manifest(
        project_root=project,
        same_hardware_receipt=receipt,
        artifact_inventory=inventory,
        release_attestation=attestation,
        output=deterministic_output,
        execute=True,
    )
    assert output.read_bytes() == deterministic_output.read_bytes()
    with inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    incomplete = _csv(
        project / "inputs/benchmark/inventory-no-preflight.csv",
        [row for row in rows if row["artifact_role"] != "hardware_preflight"],
    )
    with pytest.raises(producers.ProducerError, match="hardware preflight"):
        producers.build_benchmark_manifest(
            project_root=project,
            same_hardware_receipt=receipt,
            artifact_inventory=incomplete,
            release_attestation=attestation,
            output=project / "out/incomplete.csv",
            execute=False,
        )
    bad_hash_rows = [dict(row) for row in rows]
    bad_hash_rows[0]["sha256"] = "0" * 64
    bad_hash_inventory = _csv(
        project / "inputs/benchmark/inventory-bad-hash.csv", bad_hash_rows
    )
    with pytest.raises(producers.ProducerError, match="SHA-256 mismatch"):
        producers.build_benchmark_manifest(
            project_root=project,
            same_hardware_receipt=receipt,
            artifact_inventory=bad_hash_inventory,
            release_attestation=attestation,
            output=project / "out/bad-hash.csv",
            execute=False,
        )


def test_final_contract_assembler_seals_16_bindings_and_publicly_inspects(
    tmp_path: Path,
) -> None:
    legacy = runpy.run_path(str(PROJECT_ROOT / "tests/phaxis/test_handover_package.py"))
    project, old_contract_path = legacy["_fixture"](tmp_path / "legacy")
    old_contract = json.loads(old_contract_path.read_text(encoding="utf-8"))
    bindings = {
        role: project / record["path"]
        for role, record in old_contract["bindings"].items()
    }
    asset_manifest = Path(bindings["model_asset_manifest"])
    with asset_manifest.open(encoding="utf-8", newline="") as handle:
        asset_rows = list(csv.DictReader(handle))
    member = 0
    checkpoint_paths: list[Path] = []
    for row in asset_rows:
        row["member_index"] = ""
        row["seed"] = ""
        if row["asset_role"] == "stageb_checkpoint":
            row["member_index"] = str(member)
            row["seed"] = str(producers.FORMAL_TRAIN399_SEEDS[member])
            checkpoint_paths.append(project / row["source_path"])
            member += 1
    _csv(asset_manifest, asset_rows)
    attestation = _attestation(project)

    def assemble(name: str, *, execute: bool) -> dict[str, Any]:
        return producers.assemble_handover_build_contract(
            project_root=project,
            bindings=bindings,
            checkpoint_paths=checkpoint_paths,
            release_attestation=attestation,
            output=project / f"inputs/{name}.json",
            execute=execute,
        )

    check = assemble("contract-check", execute=False)
    assert check["bindings"] == 16
    assert check["status"] == "passed_check_only_not_written"
    assert not (project / "inputs/contract-check.json").exists()
    first = assemble("contract-a", execute=True)
    second = assemble("contract-b", execute=True)
    assert first["contract_identity_sha256"] == second["contract_identity_sha256"]
    assert (project / "inputs/contract-a.json").read_bytes() == (
        project / "inputs/contract-b.json"
    ).read_bytes()
    contract = json.loads(
        (project / "inputs/contract-a.json").read_text(encoding="utf-8")
    )
    assert len(contract["bindings"]) == 16
    assert set(contract["bindings"]) == set(producers.handover.REQUIRED_BINDINGS)
    assert [row["seed"] for row in contract["train399_checkpoints"]] == list(
        producers.FORMAL_TRAIN399_SEEDS
    )
    assert contract["license_attestation"]["blind_images_used"] == 0
    with pytest.raises(producers.ProducerError, match="order differs"):
        producers.assemble_handover_build_contract(
            project_root=project,
            bindings=bindings,
            checkpoint_paths=list(reversed(checkpoint_paths)),
            release_attestation=attestation,
            output=project / "inputs/contract-bad.json",
            execute=False,
        )
