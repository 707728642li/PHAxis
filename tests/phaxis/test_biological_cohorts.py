from __future__ import annotations

import csv
import json
from pathlib import Path

from phaxis.biology import build_biological_cohorts, recompute_biological_overlap
from phaxis.io import atomic_write_json, sha256_file, sha256_json
from phaxis.public_identity import (
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)


def _proposal(path: Path) -> tuple[Path, dict[str, str]]:
    checkpoints = [sha256_json({"member": index}) for index in range(5)]
    identities = {
        "candidate_bundle_identity_sha256": sha256_json("candidate"),
        "selection_receipt_identity_sha256": sha256_json("selection"),
        "selected_model_metadata_identity_sha256": sha256_json("metadata"),
    }
    root_audit = sha256_json("root-audit")
    root_pipeline = sha256_json("root-pipeline")
    root_bundle = sha256_json("root-bundle")
    stageb_binding = {
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "checkpoint_sha256": checkpoints,
        "selected_score_threshold": 0.225,
        **identities,
    }
    public = derive_public_identity(
        stageb_binding,
        root_bundle_identity_sha256=root_bundle,
    )
    payload = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "model_bundle_id": public["model_bundle_id"],
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "formal_release_status": "passed_proposal_not_official",
        "root_expert": {
            "provider_role": public["root_provider_role"],
            "expert_id": public["root_expert_id"],
            "fresh_exact283_audit_identity_sha256": root_audit,
            "bundle_identity_sha256": root_bundle,
            "pipeline_identity_sha256": root_pipeline,
            "root_bundle_authority": {
                "bundle_identity_sha256": root_bundle,
                "pipeline_identity_sha256": root_pipeline,
            },
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public["root_expert_id"],
            "hair_identity_and_count": stageb_binding["expert_id"],
        },
        "red_lines": {
            "blind_images_used": 0,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_statistics_included": False,
        },
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "validated_proposal_not_applied",
            "official_apply_performed": False,
            "formal_gate_source_sha256": {
                "train399_candidate": sha256_json("candidate-file"),
                "train399_selection": sha256_json("selection-file"),
                "train399_evaluation": sha256_json("evaluation-file"),
                "root_exact283": sha256_json("root-file"),
            },
            "formal_gate_identity_sha256": {
                **identities,
                "root_exact283_audit_identity_sha256": root_audit,
            },
            "stageb_binding": stageb_binding,
        },
    }
    payload["model_contract_identity_sha256"] = sha256_json(payload)
    atomic_write_json(path, payload)
    return path, {
        "model_contract_proposal_sha256": sha256_file(path),
        "model_contract_proposal_identity_sha256": payload[
            "model_contract_identity_sha256"
        ],
        "model_bundle_id": payload["model_bundle_id"],
        "root_expert_id": payload["root_expert"]["expert_id"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_overlap(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    human_root = tmp_path / "human"
    human_images = human_root / "images"
    bio_images = tmp_path / "biology"
    human_images.mkdir(parents=True)
    bio_images.mkdir(parents=True)
    payloads = {
        "human_a": b"shared-image-bytes",
        "human_b": b"human-only-bytes",
        "bio_a": b"shared-image-bytes",
        "bio_b": b"biology-only-bytes",
    }
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        root = human_images if name.startswith("human") else bio_images
        path = root / f"{name}.tif"
        path.write_bytes(payload)
        paths[name] = path
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    human_manifest = human_root / "dataset_manifest.csv"
    _write_csv(
        human_manifest,
        [
            {
                "task_id": "RHAUD-A",
                "split": "train",
                "image_relpath": "images/human_a.tif",
                "image_sha256": hashes["human_a"],
                "family_key": "family-a",
            },
            {
                "task_id": "RHAUD-B",
                "split": "val",
                "image_relpath": "images/human_b.tif",
                "image_sha256": hashes["human_b"],
                "family_key": "family-b",
            },
        ],
    )
    biological_manifest = tmp_path / "biological_manifest.csv"
    _write_csv(
        biological_manifest,
        [
            {
                "task_id": "BIO-A",
                "image_path": str(paths["bio_a"]),
                "image_sha256": hashes["bio_a"],
            },
            {
                "task_id": "BIO-B",
                "image_path": str(paths["bio_b"]),
                "image_sha256": hashes["bio_b"],
            },
        ],
    )
    output = tmp_path / "overlap_output"
    summary = recompute_biological_overlap(
        human_dataset_root=human_root,
        human_manifest=human_manifest,
        biological_manifest=biological_manifest,
        output=output,
        workers=2,
        expected_human_images=2,
        expected_biological_images=2,
        expected_overlap_images=1,
    )
    assert summary["overlap_biological_images"] == 1
    assert summary["clean_biological_images"] == 1
    assert (output / "overlap_biological_task_ids.txt").read_text().strip() == "BIO-A"
    return output, {"BIO-A": hashes["bio_a"], "BIO-B": hashes["bio_b"]}


def test_recomputed_overlap_uses_bytes_not_legacy_ids(tmp_path: Path) -> None:
    overlap, _hashes = _fixture_overlap(tmp_path)
    summary = (overlap / "summary.json").read_text(encoding="utf-8")
    assert '"legacy_rhaxiscc_overlap_file_used": false' in summary
    assert '"canonical_annotations_read": false' in summary


def test_build_clean_primary_and_full_sensitivity_cohorts(tmp_path: Path) -> None:
    overlap, hashes = _fixture_overlap(tmp_path)
    proposal_path, proposal_binding = _proposal(tmp_path / "proposal.json")
    trait_export = tmp_path / "trait_export"
    trait_export.mkdir()
    traits = [
        {
            "task_id": task_id,
            "source_image_sha256": image_sha,
            "formal_statistics_eligible": "True",
        }
        for task_id, image_sha in hashes.items()
    ]
    roots = [
        {"task_id": row["task_id"], "source_image_sha256": row["source_image_sha256"]}
        for row in traits
    ]
    image_traits = [dict(row) for row in roots]
    hairs = [
        {"task_id": task_id, "hair_id": f"{task_id}-H0001"}
        for task_id in hashes
    ]
    tables = {
        "traits": traits,
        "detailed_root_statistics": roots,
        "image_traits": image_traits,
        "hair_instances": hairs,
    }
    for name, rows in tables.items():
        _write_csv(trait_export / f"{name}.csv", rows)
    atomic_write_json(
        trait_export / "summary.json",
        {
            "status": "completed",
            "tasks": 2,
            "traits_sha256": sha256_file(trait_export / "traits.csv"),
            "detailed_root_statistics_sha256": sha256_file(
                trait_export / "detailed_root_statistics.csv"
            ),
            "image_traits_sha256": sha256_file(trait_export / "image_traits.csv"),
            "hair_instances_sha256": sha256_file(
                trait_export / "hair_instances.csv"
            ),
            "canonical_annotations_read": False,
            "root_cap_region_statistics_included": False,
            "whole_hair_zone_confirmatory_traits_allowed": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
    )
    metadata = tmp_path / "metadata.csv"
    _write_csv(
        metadata,
        [
            {
                "task_id": "BIO-A",
                "image_sha256": hashes["BIO-A"],
                "study_role": "rhd6_factorial_8d_primary",
                "experiment_key": "D15_8d",
                "condition_code": "RHD6_EV_22C",
                "genotype_or_construct": "RHD6-EV",
                "temperature_c": "22",
                "developmental_day": "8",
                "qc_disposition": "eligible",
            },
            {
                "task_id": "BIO-B",
                "image_sha256": hashes["BIO-B"],
                "study_role": "rhd6_factorial_8d_primary",
                "experiment_key": "D15_8d",
                "condition_code": "RHD6_OE_30C",
                "genotype_or_construct": "RHD6-OE",
                "temperature_c": "30",
                "developmental_day": "8",
                "qc_disposition": "eligible",
            },
        ],
    )
    design = tmp_path / "design.csv"
    _write_csv(
        design,
        [
            {
                "biological_unit_id": task_id,
                "destination_sha256": image_sha,
                "batch_id": "batch-authoritative",
                "batch_folder": "batch_1",
                "source_group": f"source-{task_id}",
            }
            for task_id, image_sha in hashes.items()
        ],
    )
    contract = tmp_path / "contract.json"
    atomic_write_json(
        contract,
        {
            "schema_version": "PHAxis-biological-analysis-contract-1.0",
            "expected_cohort_counts": {
                "human_curated443": 2,
                "biological_full": 2,
                "human_curated_overlap": 1,
                "biological_clean": 1,
            },
            "primary_model_scope": {
                "study_role": "rhd6_factorial_8d_primary",
                "experiment_key": "D15_8d",
            },
        },
    )
    output = tmp_path / "cohorts"
    summary = build_biological_cohorts(
        trait_export=trait_export,
        analysis_metadata=metadata,
        design_manifest=design,
        overlap_audit=overlap,
        analysis_contract=contract,
        model_contract_proposal=proposal_path,
        output=output,
    )
    assert summary["counts"]["biological_clean"] == 1
    assert summary["counts"]["biological_full"] == 2
    assert (output / "primary_clean1/traits.csv").is_file()
    assert (output / "sensitivity_full2/traits.csv").is_file()
    assert summary["design_identifiability"]["biological_plate_id_available"] is False
    assert summary["biological_effect_models_fitted"] is False
    assert all(summary[field] == value for field, value in proposal_binding.items())
    lock = json.loads(
        (output / "analysis_contract_lock.json").read_text(encoding="utf-8")
    )
    assert all(lock[field] == value for field, value in proposal_binding.items())
