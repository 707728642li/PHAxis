from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from phaxis.evaluation_metrics import biological_hair_presence_matcher_contract
from phaxis.hair_stageb.candidate_bundle import (
    PREREGISTERED_SCORE_THRESHOLDS,
    operating_point_selection_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/phaxis/handover_package_common.py"
SPEC = importlib.util.spec_from_file_location("handover_package_common_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
handover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handover)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _selection_prf(true_positive: int, predicted: int, ground_truth: int) -> dict:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / ground_truth if ground_truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _biological_selection_rows() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for threshold in PREREGISTERED_SCORE_THRESHOLDS:
        if threshold < 0.225:
            predicted, biological_tp = 2, 1
        elif threshold == 0.225:
            predicted, biological_tp = 1, 1
        else:
            predicted, biological_tp = 0, 0
        per_image = [
            {
                "task_id": f"qc-{index:02d}",
                "predicted": predicted,
                "ground_truth": 1,
                "biological_presence_true_positive_20um": biological_tp,
                "attachment_proxy_true_positive_20um": 0,
                "biological_presence_matched_pairs": (
                    [
                        {
                            "predicted_index_after_threshold": 0,
                            "annotated_index": 0,
                            "prediction_coverage": 0.50,
                            "truth_coverage": 0.75,
                            "proximal_direction_cosine": 0.50,
                        }
                    ]
                    if biological_tp
                    else []
                ),
                "count_error": predicted - 1,
            }
            for index in range(44)
        ]
        predicted_total = 44 * predicted
        rows.append(
            {
                "threshold": float(threshold),
                "tolerant_biological_presence_20um": _selection_prf(
                    44 * biological_tp, predicted_total, 44
                ),
                "identity_attachment_proxy_20um": _selection_prf(
                    0, predicted_total, 44
                ),
                "count_mae": float(abs(predicted - 1)),
                "count_bias": float(predicted - 1),
                "per_image": per_image,
            }
        )
    selected = next(row for row in rows if row["threshold"] == 0.225)
    return rows, selected


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict[str, Any], identity: str | None = None) -> Path:
    if identity:
        payload = dict(payload)
        payload[identity] = _canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _asset_row(
    source: Path, destination: str, *, project: Path, **extra: Any
) -> dict[str, Any]:
    return {
        "source_path": source.relative_to(project).as_posix(),
        "package_path": destination,
        "sha256": _file_hash(source),
        "bytes": source.stat().st_size,
        "provenance": "test-authority",
        "notes": "fixture note",
        "release_authorized": "true",
        **extra,
    }


def _catalog_fixture_text(contract: dict[str, Any]) -> str:
    unit_cells = {
        "um": "µm (`um`)",
        "um2": "µm² (`um2`)",
        "um2_per_mm": "µm²/mm (`um2_per_mm`)",
        "um_per_mm": "µm/mm (`um_per_mm`)",
        "rad_per_mm": "rad/mm (`rad_per_mm`)",
        "count_per_mm": "count/mm",
        "count": "count",
        "ratio": "ratio",
    }
    lines = [
        "# PHAxis catalogue fixture",
        "",
        "The single authoritative, human-readable catalogue is copied at build time; "
        "it is never an independently edited second catalogue.",
        "The root-cap representation is exactly one distal/root-cap point; there is "
        "no root-cap region.",
        "H06、H07、H13 use `[1,4) mm` axial semantics; endpoint-complete lengths and "
        "right-censored support remain explicit.",
        "",
        "| ID | 中文名称 / English name | 字段 / Field | 单位 / Unit | Fixture semantics |",
        "|---|---|---|---|---|",
    ]
    for row in [
        *contract["primary_root_traits"],
        *contract["root_hair_traits"],
    ]:
        lines.append(
            f"| {row['id']} | {row['display_name_cn']}<br>{row['display_name_en']} | "
            f"`{row['field']}` | {unit_cells[row['unit']]} | contract fixture |"
        )
    return "\n".join(lines) + "\n"


def test_handover_binds_exact_runtime_biological_matcher_contract() -> None:
    assert (
        handover.BIOLOGICAL_PRESENCE_MATCHER_CONTRACT
        == biological_hair_presence_matcher_contract()
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    inputs = project / "inputs"
    inputs.mkdir(parents=True)

    manual_image = inputs / "manual-image.tif"
    manual_image.write_bytes(b"manual-image")
    raw_return = inputs / "raw-return.json"
    raw_return.write_text('{"manual": true}\n', encoding="utf-8")
    canonical = inputs / "canonical.json"
    canonical.write_text('{"canonical": true}\n', encoding="utf-8")
    dataset_rows: list[dict[str, Any]] = []
    for index in range(500):
        task_id = f"RHAUD-{index + 1:03d}"
        dataset_rows.extend(
            (
                _asset_row(
                    manual_image,
                    f"data/human_annotated500/images/{task_id}.tif",
                    project=project,
                    task_id=task_id,
                    dataset_id="RHAxis-Arabidopsis-HumanAnnotated500",
                    annotation_kind="manual500_source_image",
                ),
                _asset_row(
                    raw_return,
                    f"data/human_annotated500/annotations/raw_return/{task_id}.json",
                    project=project,
                    task_id=task_id,
                    dataset_id="RHAxis-Arabidopsis-HumanAnnotated500",
                    annotation_kind="manual500_raw_return_json",
                ),
            )
        )
        if index < 443:
            dataset_rows.append(
                _asset_row(
                    canonical,
                    f"data/human_annotated500/annotations/rhaxis_canonical/{task_id}.json",
                    project=project,
                    task_id=task_id,
                    dataset_id="RHAxis-Arabidopsis-HumanCurated443-v1.0",
                    annotation_kind="canonical443_vector_json",
                )
            )
    support = inputs / "dataset-support.txt"
    support.write_text("fixture dataset support\n", encoding="utf-8")
    for destination in sorted(handover.REQUIRED_DATASET_SUPPORT_PATHS):
        dataset_rows.append(
            _asset_row(
                support,
                destination,
                project=project,
                task_id="",
                dataset_id="RHAxis-Arabidopsis-HumanAnnotated500",
                annotation_kind="dataset_support",
            )
        )
    dataset_manifest = _manifest(
        inputs / "dataset_manifest.csv",
        dataset_rows,
    )

    images: list[dict[str, Any]] = []
    for index in range(283):
        image = inputs / "images" / f"bio-{index:03d}.tif"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(f"bio-{index:03d}".encode("ascii"))
        images.append(
            _asset_row(
                image,
                f"data/biological283/images/bio-{index:03d}.tif",
                project=project,
                task_id=f"BIO-{index:03d}",
                temperature_c="22" if index % 2 else "30",
                genotype_or_construct="RHD6-OE" if index % 3 else "EV",
            )
        )
    image_manifest = _manifest(inputs / "image_manifest.csv", images)

    runtime = inputs / "runtime.py"
    runtime.write_text("print('PHAxis runtime')\n", encoding="utf-8")
    canonical_trait_contract = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    phenotype_catalog = project / "docs/phaxis/TRAIT_CONTRACT_CN.md"
    phenotype_catalog.parent.mkdir(parents=True, exist_ok=True)
    repository_catalog = PROJECT_ROOT / "docs/phaxis/TRAIT_CONTRACT_CN.md"
    if repository_catalog.is_file():
        phenotype_catalog.write_bytes(repository_catalog.read_bytes())
    else:
        phenotype_catalog.write_text(
            _catalog_fixture_text(canonical_trait_contract), encoding="utf-8"
        )
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_hashes: list[str] = []
    for index in range(5):
        checkpoint = inputs / f"train399-fold-{index}.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode("ascii"))
        checkpoint_hashes.append(_file_hash(checkpoint))
        checkpoint_rows.append(
            _asset_row(
                checkpoint,
                f"model/assets/stageb/member-{index}.pt",
                project=project,
                asset_role="stageb_checkpoint",
            )
        )
    root_asset = inputs / "root-provider.asset"
    root_asset.write_bytes(b"root-provider")
    checkpoint_rows.append(
        _asset_row(
            root_asset,
            "model/assets/root_provider/root-provider.asset",
            project=project,
            asset_role="root_provider_asset",
        )
    )
    bundle_manifest_asset = inputs / "MODEL_BUNDLE_MANIFEST.json"
    bundle_manifest_asset.write_text('{"bundle": "fixture"}\n', encoding="utf-8")
    checkpoint_rows.append(
        _asset_row(
            bundle_manifest_asset,
            "model/assets/MODEL_BUNDLE_MANIFEST.json",
            project=project,
            asset_role="model_bundle_manifest",
        )
    )
    model_asset_manifest = _manifest(
        inputs / "model_asset_manifest.csv", checkpoint_rows
    )
    benchmark_table = inputs / "benchmark.csv"
    benchmark_table.write_text("mode,seconds\na,1\nb,2\n", encoding="utf-8")
    benchmark_manifest = _manifest(
        inputs / "benchmark_manifest.csv",
        [_asset_row(benchmark_table, "model/benchmark/benchmark.csv", project=project)],
    )

    candidate_payload = {
        "members": [{"member_index": index} for index in range(5)],
        "operating_point_selection_contract": operating_point_selection_contract(),
    }
    candidate = _json(
        inputs / "candidate.json",
        {
            "schema_version": "PHAxis-StageB-train399-candidate-bundle-1.0",
            "status": "candidate_gate_passed_not_promoted",
            "identity_payload": candidate_payload,
            "candidate_bundle_identity_sha256": _canonical_hash(candidate_payload),
            "blind_images_used": 0,
        },
        "candidate_manifest_identity_sha256",
    )
    candidate_identity = json.loads(candidate.read_text(encoding="utf-8"))[
        "candidate_bundle_identity_sha256"
    ]
    matcher = biological_hair_presence_matcher_contract()
    threshold_metrics, selected = _biological_selection_rows()
    selection = _json(
        inputs / "selection.json",
        {
            "schema_version": "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3",
            "status": "completed",
            "images": 44,
            "candidate_bundle_identity_sha256": candidate_identity,
            "independent_accuracy_claim_allowed": False,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "straight_base_to_tip_presence_proxy_evaluated_during_selection": True,
            "distal_endpoint_error_used_as_selection_gate": False,
            "complete_line_overlap_used_as_selection_gate": False,
            "length_error_used_as_selection_gate": False,
            "manual_hair_width_assumed": False,
            "primary_matcher_contract": matcher,
            "primary_matcher_contract_sha256": _canonical_hash(matcher),
            "selection_contract": operating_point_selection_contract(),
            "threshold_metrics": threshold_metrics,
            "selected": selected,
            "blind_images_used": 0,
        },
        "selection_receipt_identity_sha256",
    )
    selection_identity = json.loads(selection.read_text(encoding="utf-8"))[
        "selection_receipt_identity_sha256"
    ]
    selected_metadata_identity = "9" * 64
    evaluation_payload = {
        "schema_version": "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
        "status": "completed",
        "training_contract": {
            "training_images": 399,
            "validation_images": 44,
            "candidate_bundle_identity_sha256": candidate_identity,
            "selection_receipt_identity_sha256": selection_identity,
            "selected_model_metadata_identity_sha256": selected_metadata_identity,
        },
        "blind_images_used": 0,
    }
    evaluation = _json(inputs / "evaluation.json", evaluation_payload)
    root_identity_payload = {
        "schema_version": "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "reference_identity_sha256": "1" * 64,
        "fresh_reference_identity_sha256": "2" * 64,
        "bundle_identity_sha256": "3" * 64,
        "pipeline_identity_sha256": "4" * 64,
        "layers": {
            name: {"exact": 283, "mismatch_count": 0}
            for name in ("root_mask", "ordered_axis", "distal_point")
        },
        "source_image_mismatch_task_ids": [],
        "prepared_radius_fallback_task_ids": [],
        "attachment_supported_extension_rescue_task_ids": [],
        "pipeline_raw_image_provenance_gate": True,
        "pipeline_stage_evidence_gate": True,
    }
    exact = _json(
        inputs / "exact283.json",
        {
            **root_identity_payload,
            "status": "pass_exact_283",
            "fresh_portable_raw_image_rerun_completed": True,
            "blind_images_used": 0,
            "audit_identity_sha256": _canonical_hash(root_identity_payload),
        },
    )
    fusion = _json(
        inputs / "fusion.json",
        {"schema_version": "PHAxis-fusion-run-1.1", "status": "completed", "images": 283},
        "summary_identity_sha256",
    )
    traits = _json(
        inputs / "traits.json",
        {"schema_version": "PHAxis-trait-export-1.0", "status": "completed", "tasks": 283},
        "export_identity_sha256",
    )
    hardware = "a" * 64
    benchmark = _json(
        inputs / "same_hardware.json",
        {
            "schema_version": "PHAxis-same-hardware-benchmark-receipt-1.0",
            "status": "passed",
            "images": 283,
            "hardware_identity_sha256": hardware,
            "runs": [
                {"mode": "stageb", "hardware_identity_sha256": hardware},
                {"mode": "full", "hardware_identity_sha256": hardware},
            ],
            "blind_images_used": 0,
        },
        "receipt_identity_sha256",
    )
    source_files = [
        {
            "path": "src/phaxis/runtime.py",
            "bytes": runtime.stat().st_size,
            "sha256": _file_hash(runtime),
            "origin": "project:inputs/runtime.py",
        }
    ]
    source_release = _json(
        inputs / "SOURCE_MANIFEST.json",
        {
            "schema_version": "PHAxis-source-release-manifest-2.0",
            "distribution": "phaxis",
            "version": "1.0.0",
            "release_mode": "formal",
            "source_policy": "explicit_path_bounded_allowlist",
            "tree_identity_sha256": _canonical_hash(source_files),
            "files": source_files,
        },
    )
    model_source_manifest = _manifest(
        inputs / "model_source_manifest.csv",
        [
            _asset_row(
                source_release,
                "model/source_release/SOURCE_MANIFEST.json",
                project=project,
            ),
            _asset_row(
                runtime,
                "model/source_release/src/phaxis/runtime.py",
                project=project,
            ),
        ],
    )
    model_bundle_id = "PHAXIS-V1.0.0-STRICT-TRAIN399-" + "A" * 20
    root_expert_id = "PHAxis-root-provider-" + "B" * 20
    hair_expert_id = "PHAxis-StageB-train399-five-seed"
    proposal_file_sha256 = "5" * 64
    proposal_identity_sha256 = "6" * 64
    clean = _json(
        inputs / "clean_install.json",
        {
            "schema_version": "PHAxis-clean-install-verification-1.0",
            "status": "completed_final_clean_install",
            "example_output_identity_sha256": "8" * 64,
            "model_contract_proposal_sha256": proposal_file_sha256,
            "model_contract_proposal_identity_sha256": proposal_identity_sha256,
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
            "root_bundle_identity_sha256": root_identity_payload[
                "bundle_identity_sha256"
            ],
            "hair_identity_count_expert": hair_expert_id,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
            "source_release_manifest_sha256": _file_hash(source_release),
        },
        "clean_install_receipt_identity_sha256",
    )
    trait_contract = _json(
        inputs / "trait_contract.json",
        canonical_trait_contract,
    )

    bindings = {
        role: {"path": path.relative_to(project).as_posix(), "sha256": _file_hash(path)}
        for role, path in {
            "train399_candidate_manifest": candidate,
            "train399_selection_receipt": selection,
            "train399_evaluation_receipt": evaluation,
            "fresh_exact283_receipt": exact,
            "final_fusion_receipt": fusion,
            "final_traits_receipt": traits,
            "same_hardware_benchmark_receipt": benchmark,
            "source_release_manifest": source_release,
            "clean_install_receipt": clean,
            "dataset_manifest": dataset_manifest,
            "image_manifest": image_manifest,
            "model_source_manifest": model_source_manifest,
            "model_asset_manifest": model_asset_manifest,
            "benchmark_manifest": benchmark_manifest,
            "trait_contract": trait_contract,
        }.items()
    }
    model_contract = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "model_bundle_id": model_bundle_id,
        "formal_release_status": "passed",
        "promotion": {
            "status": "applied_formal_release",
            "official_apply_performed": True,
            "proposal_file_sha256": proposal_file_sha256,
            "proposal_identity_sha256": proposal_identity_sha256,
            "formal_gate_source_sha256": {
                "train399_candidate": bindings["train399_candidate_manifest"]["sha256"],
                "train399_selection": bindings["train399_selection_receipt"]["sha256"],
                "train399_evaluation": bindings["train399_evaluation_receipt"]["sha256"],
                "root_exact283": bindings["fresh_exact283_receipt"]["sha256"],
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": candidate_identity,
                "selection_receipt_identity_sha256": selection_identity,
                "selected_model_metadata_identity_sha256": selected_metadata_identity,
                "root_exact283_audit_identity_sha256": _canonical_hash(root_identity_payload),
            },
            "final_receipt_source_sha256": {
                "fusion": bindings["final_fusion_receipt"]["sha256"],
                "traits": bindings["final_traits_receipt"]["sha256"],
            },
        },
        "red_lines": {
            "blind_images_used": 0,
            "formal_train399_only_stageb_weights_available": True,
        },
        "hair_identity_count_expert": {
            "checkpoint_sha256_in_member_order": checkpoint_hashes,
            "expert_id": hair_expert_id,
        },
        "root_expert": {
            "expert_id": root_expert_id,
            "bundle_identity_sha256": root_identity_payload[
                "bundle_identity_sha256"
            ],
        },
        "development_evidence": {
            "qcdev44": {
                "source": {
                    "evaluation_sha256": bindings["train399_evaluation_receipt"]["sha256"],
                    "evaluation_content_identity_sha256": _canonical_hash(evaluation_payload),
                }
            }
        },
    }
    model_path = _json(
        inputs / "model_contract.json",
        model_contract,
        "model_contract_identity_sha256",
    )
    bindings["applied_model_contract"] = {
        "path": model_path.relative_to(project).as_posix(),
        "sha256": _file_hash(model_path),
    }
    contract = {
        "schema_version": handover.CONTRACT_SCHEMA,
        "product": "PHAxis",
        "product_version": "1.0.0",
        "scope_attestation": {
            "all_legally_deliverable_manual_annotations_included": True,
            "training_and_validation_annotations_may_be_mixed": True,
            "annotation_notes_provenance_hashes_preserved": True,
            "biological283_includes_temperature_and_rhd6_design": True,
            "image_assembly_excluded": True,
            "blind_and_final_partitions_excluded": True,
            "frozen_v1_untouched": True,
        },
        "bindings": bindings,
        "train399_checkpoints": [
            {
                "member_index": index,
                "seed": 2026082801 + index,
                "sha256": value,
            }
            for index, value in enumerate(checkpoint_hashes)
        ],
    }
    contract_path = _json(
        inputs / "handover_contract.json", contract, "contract_identity_sha256"
    )
    return project, contract_path


def test_formal_handover_build_is_deterministic_and_self_verifying(
    tmp_path: Path,
) -> None:
    project, contract = _fixture(tmp_path)
    first = project / "release-a"
    second = project / "release-b"
    first_manifest = handover.build_handover_package(
        project_root=project, contract_path=contract, output=first
    )
    second_manifest = handover.build_handover_package(
        project_root=project, contract_path=contract, output=second
    )
    assert first_manifest == second_manifest
    assert (first / "PACKAGE_MANIFEST.json").read_bytes() == (
        second / "PACKAGE_MANIFEST.json"
    ).read_bytes()
    assert handover.verify_handover_package(first)["status"] == "passed"
    assert (first / "model/source_release/SOURCE_MANIFEST.json").read_bytes() == (
        project / "inputs/SOURCE_MANIFEST.json"
    ).read_bytes()
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONDONTWRITEBYTECODE", None)
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "verify_package.py", str(first)],
            cwd=first,
            env=clean_env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
    assert not list(first.rglob("__pycache__"))
    assert len(list((first / "data/biological283/images").glob("*.tif"))) == 283
    capabilities = (first / "PHENOTYPE_CAPABILITIES_CN.md").read_text(
        encoding="utf-8"
    )
    assert (first / "PHENOTYPE_CAPABILITIES_CN.md").read_bytes() == (
        project / "docs/phaxis/TRAIT_CONTRACT_CN.md"
    ).read_bytes()
    assert all(f"| R{index:02d} |" in capabilities for index in range(1, 20))
    assert all(f"| H{index:02d} |" in capabilities for index in range(1, 14))
    assert "single authoritative, human-readable" in capabilities
    assert "The root-cap representation is exactly one distal/root-cap point" in capabilities
    assert "H06、H07、H13" in capabilities
    assert "`[1,4) mm`" in capabilities
    assert "right-censored" in capabilities
    assert "endpoint-complete" in capabilities


def test_missing_same_hardware_binding_fails_closed_without_output(
    tmp_path: Path,
) -> None:
    project, contract_path = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.pop("contract_identity_sha256")
    contract["bindings"].pop("same_hardware_benchmark_receipt")
    contract["contract_identity_sha256"] = _canonical_hash(contract)
    _json(contract_path, contract)
    output = project / "must-not-exist"
    with pytest.raises(handover.HandoverError, match="required binding is absent"):
        handover.build_handover_package(
            project_root=project, contract_path=contract_path, output=output
        )
    assert not output.exists()


def test_tamper_and_image_assembly_component_are_rejected(tmp_path: Path) -> None:
    project, contract_path = _fixture(tmp_path)
    output = project / "release"
    handover.build_handover_package(
        project_root=project, contract_path=contract_path, output=output
    )
    with (output / "README_CN.md").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(handover.HandoverError, match="hash/size mismatch"):
        handover.verify_handover_package(output)

    project2, contract2 = _fixture(tmp_path / "second")
    payload = json.loads(contract2.read_text(encoding="utf-8"))
    source_manifest_path = project2 / payload["bindings"]["model_source_manifest"]["path"]
    rows = list(csv.DictReader(source_manifest_path.open(encoding="utf-8")))
    rows[0]["package_path"] = "model/source_release/scripts/stitch_images.py"
    _manifest(source_manifest_path, rows)
    payload.pop("contract_identity_sha256")
    payload["bindings"]["model_source_manifest"]["sha256"] = _file_hash(
        source_manifest_path
    )
    payload["contract_identity_sha256"] = _canonical_hash(payload)
    _json(contract2, payload)
    with pytest.raises(handover.HandoverError, match="image-assembly component"):
        handover.inspect_handover_contract(project2, contract2)


def test_path_copy_and_assembly_boundary_guards(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "asset.bin"
    source.write_bytes(b"immutable-asset")
    with pytest.raises(handover.HandoverError, match="project-relative"):
        handover._resolve_project_file(project, str(source.resolve()))
    with pytest.raises(handover.HandoverError, match="unsafe package_path"):
        handover._safe_relative("data/human_annotated500/file.json:stream")

    hardlink = project / "asset-hardlink.bin"
    os.link(source, hardlink)
    copied = project / "copied" / "asset.bin"
    handover._copy_exact(
        hardlink,
        copied,
        expected_sha256=_file_hash(source),
        expected_bytes=source.stat().st_size,
    )
    assert copied.read_bytes() == source.read_bytes()
    assert not os.path.samefile(source, copied)
    wrong = project / "copied" / "wrong.bin"
    with pytest.raises(handover.HandoverError, match="source changed"):
        handover._copy_exact(
            source,
            wrong,
            expected_sha256="0" * 64,
            expected_bytes=source.stat().st_size,
        )
    assert not wrong.exists()

    assert not handover._is_forbidden_assembly_component(
        "model/source_release/docs/IMAGE_ASSEMBLY_EXCLUDED.md"
    )
    assert handover._is_forbidden_assembly_component(
        "model/source_release/scripts/stitch_images.py"
    )
