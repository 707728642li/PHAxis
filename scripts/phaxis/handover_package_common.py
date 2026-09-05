#!/usr/bin/env python
"""Deterministic, fail-closed PHAxis 1.0.0 reuse handover packaging.

The module intentionally has no discovery logic.  Every byte admitted to a
formal handover tree must be named by an input contract or one of its sealed
materialisation manifests.  It never imports PHAxis runtime code and never
runs inference, training, or an image-assembly program.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping


CONTRACT_SCHEMA = "PHAxis-reuse-handover-build-contract-1.0"
PACKAGE_SCHEMA = "PHAxis-reuse-handover-package-manifest-1.0"
RECEIPT_SCHEMA = "PHAxis-reuse-handover-build-receipt-1.0"
PRODUCT = "PHAxis"
VERSION = "1.0.0"
BIOLOGICAL_PRESENCE_MATCHER_CONTRACT = {
    "schema_version": "PHAxis-biological-hair-presence-matcher-1.0",
    "target": "one_manual_single_trunk_centreline_per_visible_root_hair",
    "coordinate_space": "physical_um_xy",
    "curve_tolerance_um": 20.0,
    "minimum_truth_coverage": 0.25,
    "minimum_prediction_coverage": 0.25,
    "minimum_direction_cosine": 0.0,
    "proximal_arc_fraction": 0.25,
    "resample_points": 32,
    "assignment": (
        "per_source_image_maximum_cardinality_one_to_one_Hungarian_then_"
        "minimum_supported_curve_cost"
    ),
    "coverage": "bidirectional_arc_length_resampled_point_support",
    "stageB_predicted_geometry_proxy": "straight_base_to_tip",
    "manual_hair_width_assumed": False,
    "distal_endpoint_is_identity_gate": False,
    "complete_centreline_overlap_is_identity_gate": False,
    "length_error_is_identity_gate": False,
    "image_intensity_or_colour_is_matcher_input": False,
}
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
RECEIPT_NAME = "BUILD_RECEIPT.json"
CATALOG_PROJECT_PATH = "docs/phaxis/TRAIT_CONTRACT_CN.md"
CATALOG_HANDOVER_PATH = "PHENOTYPE_CAPABILITIES_CN.md"

REQUIRED_BINDINGS = (
    "applied_model_contract",
    "train399_candidate_manifest",
    "train399_selection_receipt",
    "train399_evaluation_receipt",
    "fresh_exact283_receipt",
    "final_fusion_receipt",
    "final_traits_receipt",
    "same_hardware_benchmark_receipt",
    "source_release_manifest",
    "clean_install_receipt",
    "dataset_manifest",
    "image_manifest",
    "model_source_manifest",
    "model_asset_manifest",
    "benchmark_manifest",
    "trait_contract",
)
MATERIALISATION_BINDINGS = (
    "dataset_manifest",
    "image_manifest",
    "model_source_manifest",
    "model_asset_manifest",
    "benchmark_manifest",
)
CSV_FIELDS = {
    "source_path",
    "package_path",
    "sha256",
    "bytes",
    "provenance",
    "notes",
    "release_authorized",
}
DATASET_FIELDS = {"task_id", "dataset_id", "annotation_kind"}
MODEL_ASSET_FIELDS = {"asset_role"}
DATASET_COMPONENT_COUNTS = {
    "manual500_source_image": 500,
    "manual500_raw_return_json": 500,
    "canonical443_vector_json": 443,
}
REQUIRED_DATASET_SUPPORT_PATHS = {
    "data/human_annotated500/DATASET_CARD.md",
    "data/human_annotated500/LICENSE_DATA.md",
    "data/human_annotated500/README_CN.md",
    "data/human_annotated500/build_summary.json",
    "data/human_annotated500/label_schema.json",
    "data/human_annotated500/manifests/dataset_manifest.csv",
    "data/human_annotated500/manifests/filter_decisions_all500.csv",
    "data/human_annotated500/manifests/split_manifest.csv",
    "data/human_annotated500/notes/ALL500_DATA_NOTES_CN.md",
    "data/human_annotated500/provenance.json",
    "data/human_annotated500/verification_report.json",
}
CHECKPOINTS = 5
FORMAL_TRAIN399_SEEDS = (
    2026082801,
    2026082802,
    2026082803,
    2026082804,
    2026082805,
)
SHA256_LEN = 64
FORBIDDEN_ASSEMBLY_TOKENS = (
    "stitch",
    "mosaic",
    "tile_assembly",
    "image_assembly",
    "拼接",
)
EXECUTABLE_OR_CODE_SUFFIXES = {".bat", ".cmd", ".exe", ".ps1", ".py", ".sh"}


class HandoverError(RuntimeError):
    """A formal handover contract or package failed closed."""


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _biological_presence_matcher_contract_ok(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and dict(payload) == BIOLOGICAL_PRESENCE_MATCHER_CONTRACT
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_LEN and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HandoverError(f"cannot read JSON authority {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HandoverError(f"JSON authority is not an object: {path}")
    return payload


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoverError("package_path is absent")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise HandoverError(f"unsafe package_path: {value}")
    # The package is built on Windows but must also extract safely on POSIX.
    # Reject alternate-data-stream syntax, control characters, and names whose
    # Windows normalization could alias a second manifest row.
    for part in path.parts:
        if (
            not part
            or ":" in part
            or any(ord(character) < 32 for character in part)
            or part.endswith((" ", "."))
        ):
            raise HandoverError(f"unsafe package_path: {value}")
    return path.as_posix()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_project_file(
    project_root: Path, value: Any, *, require_relative: bool = True
) -> Path:
    if not isinstance(value, str) or not value:
        raise HandoverError("source path is absent")
    supplied = Path(value)
    if require_relative and supplied.is_absolute():
        raise HandoverError(f"source path must be project-relative: {value}")
    if ".." in supplied.parts:
        raise HandoverError(f"source path contains parent traversal: {value}")
    lexical = supplied if supplied.is_absolute() else project_root / supplied
    try:
        lexical_relative = lexical.relative_to(project_root)
    except ValueError as error:
        raise HandoverError(f"source escapes project root: {value}") from error
    cursor = project_root
    for part in lexical_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise HandoverError(f"source traverses a symlink: {value}")
    candidate = lexical.resolve()
    if not _inside(candidate, project_root):
        raise HandoverError(f"source escapes project root: {value}")
    if not candidate.is_file():
        raise HandoverError(f"source is absent, non-file, or symlink: {value}")
    return candidate


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not CSV_FIELDS.issubset(reader.fieldnames):
                missing = sorted(CSV_FIELDS - set(reader.fieldnames or ()))
                raise HandoverError(
                    f"materialisation manifest lacks required columns {missing}: {path}"
                )
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise HandoverError(f"cannot read materialisation manifest {path}: {error}") from error


def _binding_path(
    contract: Mapping[str, Any], role: str, project_root: Path
) -> tuple[Path, Mapping[str, Any]]:
    bindings = contract.get("bindings")
    record = bindings.get(role) if isinstance(bindings, Mapping) else None
    if not isinstance(record, Mapping):
        raise HandoverError(f"required binding is absent: {role}")
    path = _resolve_project_file(project_root, record.get("path"))
    expected = record.get("sha256")
    if not _is_sha256(expected) or _sha256_file(path) != expected:
        raise HandoverError(f"binding SHA-256 mismatch: {role}")
    return path, record


def _identity_ok(payload: Mapping[str, Any], field: str) -> bool:
    claimed = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return _is_sha256(claimed) and _sha256_json(unsigned) == claimed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoverError(message)


def _is_forbidden_assembly_component(relative: str) -> bool:
    path = PurePosixPath(relative.casefold())
    return path.suffix in EXECUTABLE_OR_CODE_SUFFIXES and any(
        token in path.stem for token in FORBIDDEN_ASSEMBLY_TOKENS
    )


def _semantic_gate(
    contract: Mapping[str, Any], authorities: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    checks: list[str] = []
    applied = authorities["applied_model_contract"]
    _require(applied.get("schema_version") == "PHAxis-model-contract-1.0.0", "applied model contract schema is invalid")
    _require(_identity_ok(applied, "model_contract_identity_sha256"), "applied model contract logical identity is invalid")
    _require(applied.get("product") == PRODUCT and applied.get("product_version") == VERSION, "applied model identity is not PHAxis 1.0.0")
    _require(applied.get("formal_release_status") == "passed", "model contract is not formally released")
    promotion = applied.get("promotion")
    _require(isinstance(promotion, Mapping), "applied model promotion authority is absent")
    _require(promotion.get("status") == "applied_formal_release" and promotion.get("official_apply_performed") is True, "model contract proposal was not formally applied")
    red_lines = applied.get("red_lines")
    _require(isinstance(red_lines, Mapping) and red_lines.get("blind_images_used") == 0, "model contract used blind images")
    _require(red_lines.get("formal_train399_only_stageb_weights_available") is True, "strict train399 weights are not formal")
    expert = applied.get("hair_identity_count_expert")
    expected_checkpoints = (
        expert.get("checkpoint_sha256_in_member_order")
        if isinstance(expert, Mapping)
        else None
    )
    checkpoint_records = contract.get("train399_checkpoints")
    _require(isinstance(checkpoint_records, list) and len(checkpoint_records) == CHECKPOINTS, "exactly five train399 checkpoints are required")
    _require(
        all(isinstance(record, Mapping) for record in checkpoint_records)
        and [record.get("member_index") for record in checkpoint_records]
        == list(range(CHECKPOINTS))
        and tuple(record.get("seed") for record in checkpoint_records)
        == FORMAL_TRAIN399_SEEDS,
        "train399 checkpoint member/seed order is invalid",
    )
    checkpoint_hashes = [record.get("sha256") for record in checkpoint_records if isinstance(record, Mapping)]
    _require(len(checkpoint_hashes) == CHECKPOINTS and len(set(checkpoint_hashes)) == CHECKPOINTS and all(_is_sha256(value) for value in checkpoint_hashes), "five checkpoint hashes must be distinct SHA-256 values")
    _require(checkpoint_hashes == expected_checkpoints, "checkpoint order/hashes differ from applied model contract")
    formal_sources = promotion.get("formal_gate_source_sha256")
    _require(
        isinstance(formal_sources, Mapping)
        and formal_sources.get("train399_candidate") == contract["bindings"]["train399_candidate_manifest"]["sha256"]
        and formal_sources.get("train399_selection") == contract["bindings"]["train399_selection_receipt"]["sha256"]
        and formal_sources.get("train399_evaluation") == contract["bindings"]["train399_evaluation_receipt"]["sha256"]
        and formal_sources.get("root_exact283") == contract["bindings"]["fresh_exact283_receipt"]["sha256"],
        "applied contract does not bind the named train399/exact283 Gate inputs",
    )

    candidate = authorities["train399_candidate_manifest"]
    _require(
        candidate.get("schema_version")
        == "PHAxis-StageB-train399-candidate-bundle-1.0"
        and candidate.get("status") == "candidate_gate_passed_not_promoted",
        "train399 candidate manifest schema/status is invalid",
    )
    _require(
        _identity_ok(candidate, "candidate_manifest_identity_sha256"),
        "train399 candidate manifest identity is invalid",
    )
    candidate_identity = candidate.get("candidate_bundle_identity_sha256")
    candidate_payload = candidate.get("identity_payload")
    _require(
        _is_sha256(candidate_identity)
        and isinstance(candidate_payload, Mapping)
        and _sha256_json(candidate_payload) == candidate_identity,
        "train399 candidate bundle identity is invalid",
    )
    formal_identities = promotion.get("formal_gate_identity_sha256")
    _require(
        isinstance(formal_identities, Mapping)
        and formal_identities.get("candidate_bundle_identity_sha256")
        == candidate_identity,
        "applied contract candidate logical identity mismatch",
    )

    selection = authorities["train399_selection_receipt"]
    _require(
        selection.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3"
        and selection.get("status") == "completed"
        and selection.get("images") == 44
        and selection.get(
            "straight_base_to_tip_presence_proxy_evaluated_during_selection"
        )
        is True
        and selection.get("distal_endpoint_error_used_as_selection_gate") is False
        and selection.get("complete_line_overlap_used_as_selection_gate") is False
        and selection.get("length_error_used_as_selection_gate") is False
        and selection.get("manual_hair_width_assumed") is False,
        "train399 selection receipt schema/status is invalid",
    )
    matcher = selection.get("primary_matcher_contract")
    _require(
        _biological_presence_matcher_contract_ok(matcher)
        and selection.get("primary_matcher_contract_sha256")
        == _sha256_json(matcher),
        "train399 selection receipt primary matcher is invalid",
    )
    selection_contract = selection.get("selection_contract")
    candidate_selection_contract = candidate_payload.get(
        "operating_point_selection_contract"
    )
    _require(
        isinstance(selection_contract, Mapping)
        and selection_contract == candidate_selection_contract
        and selection_contract.get("primary_selection_metric")
        == "one_to_one_tolerant_biological_presence_F1_at_20um"
        and selection_contract.get("primary_matcher_contract") == matcher
        and selection_contract.get("primary_matcher_contract_sha256")
        == _sha256_json(matcher)
        and selection_contract.get("tie_break_order")
        == [
            "maximum_primary_biological_presence_F1_at_20um",
            "minimum_per_image_count_MAE",
            "minimum_absolute_count_bias",
            "higher_score_threshold",
        ],
        "train399 candidate/selection biological-presence protocol is invalid",
    )
    threshold_metrics = selection.get("threshold_metrics")
    _require(
        isinstance(threshold_metrics, list)
        and len(threshold_metrics) == 10
        and selection.get("selected") in threshold_metrics
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("tolerant_biological_presence_20um"), Mapping)
            and isinstance(row.get("identity_attachment_proxy_20um"), Mapping)
            and isinstance(row.get("count_mae"), (int, float))
            and isinstance(row.get("count_bias"), (int, float))
            and isinstance(row.get("per_image"), list)
            and len(row["per_image"]) == 44
            for row in threshold_metrics
        ),
        "train399 selection receipt lacks complete biological/count sufficient statistics",
    )
    _require(
        _identity_ok(selection, "selection_receipt_identity_sha256"),
        "train399 selection receipt identity is invalid",
    )
    _require(
        selection.get("candidate_bundle_identity_sha256") == candidate_identity
        and formal_identities.get("selection_receipt_identity_sha256")
        == selection.get("selection_receipt_identity_sha256"),
        "candidate/selection/applied identities are not closed",
    )

    evaluation = authorities["train399_evaluation_receipt"]
    training_contract = evaluation.get("training_contract")
    development_evidence = applied.get("development_evidence")
    qcdev44 = (
        development_evidence.get("qcdev44")
        if isinstance(development_evidence, Mapping)
        else None
    )
    evidence_source = qcdev44.get("source") if isinstance(qcdev44, Mapping) else None
    _require(
        evaluation.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
        and evaluation.get("status") == "completed"
        and isinstance(training_contract, Mapping)
        and training_contract.get("training_images") == 399
        and training_contract.get("validation_images") == 44,
        "train399 evaluation receipt schema/status is invalid",
    )
    _require(
        training_contract.get("candidate_bundle_identity_sha256")
        == candidate_identity
        and training_contract.get("selection_receipt_identity_sha256")
        == selection.get("selection_receipt_identity_sha256")
        and formal_identities.get("selected_model_metadata_identity_sha256")
        == training_contract.get("selected_model_metadata_identity_sha256"),
        "candidate/selection/evaluation identities are not closed",
    )
    _require(
        isinstance(evidence_source, Mapping)
        and evidence_source.get("evaluation_sha256")
        == contract["bindings"]["train399_evaluation_receipt"]["sha256"]
        and evidence_source.get("evaluation_content_identity_sha256")
        == _sha256_json(evaluation),
        "applied contract development evidence does not bind evaluation content",
    )
    checks.extend(("applied_model_contract", "five_train399_checkpoints"))

    exact = authorities["fresh_exact283_receipt"]
    _require(
        exact.get("schema_version")
        == "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "fresh root-provider receipt schema is invalid",
    )
    _require(exact.get("status") == "pass_exact_283", "fresh root-provider receipt did not pass exact283")
    _require(exact.get("fresh_portable_raw_image_rerun_completed") is True, "fresh exact283 rerun is absent")
    _require(exact.get("blind_images_used") == 0, "exact283 receipt used blind images")
    layers = exact.get("layers")
    _require(isinstance(layers, Mapping) and len(layers) >= 3 and all(isinstance(row, Mapping) and row.get("exact") == 283 and row.get("mismatch_count") == 0 for row in layers.values()), "fresh exact283 layers are incomplete")
    root_identity_payload = {
        field: exact.get(field)
        for field in (
            "schema_version",
            "reference_identity_sha256",
            "fresh_reference_identity_sha256",
            "bundle_identity_sha256",
            "pipeline_identity_sha256",
            "layers",
            "source_image_mismatch_task_ids",
            "prepared_radius_fallback_task_ids",
            "attachment_supported_extension_rescue_task_ids",
            "pipeline_raw_image_provenance_gate",
            "pipeline_stage_evidence_gate",
        )
    }
    _require(
        _is_sha256(exact.get("audit_identity_sha256"))
        and _sha256_json(root_identity_payload) == exact.get("audit_identity_sha256")
        and formal_identities.get("root_exact283_audit_identity_sha256")
        == exact.get("audit_identity_sha256"),
        "fresh exact283 logical identity is invalid",
    )
    checks.append("fresh_exact283")

    fusion = authorities["final_fusion_receipt"]
    traits = authorities["final_traits_receipt"]
    _require(fusion.get("schema_version") == "PHAxis-fusion-run-1.1" and fusion.get("status") == "completed" and fusion.get("images") == 283, "final fusion is not completed exact283")
    _require(_identity_ok(fusion, "summary_identity_sha256"), "final fusion identity is invalid")
    _require(traits.get("schema_version") == "PHAxis-trait-export-1.0" and traits.get("status") == "completed" and traits.get("tasks") == 283, "final traits are not completed exact283")
    _require(_identity_ok(traits, "export_identity_sha256"), "final traits identity is invalid")
    final_sources = promotion.get("final_receipt_source_sha256")
    _require(isinstance(final_sources, Mapping), "applied contract lacks final receipt source hashes")
    bindings = contract["bindings"]
    _require(final_sources.get("fusion") == bindings["final_fusion_receipt"]["sha256"] and final_sources.get("traits") == bindings["final_traits_receipt"]["sha256"], "final fusion/traits do not bind to applied contract")
    checks.append("final_fusion_traits")

    benchmark = authorities["same_hardware_benchmark_receipt"]
    _require(benchmark.get("schema_version") == "PHAxis-same-hardware-benchmark-receipt-1.0", "same-hardware benchmark schema is invalid")
    _require(benchmark.get("status") == "passed" and benchmark.get("images") == 283, "same-hardware benchmark is not passed exact283")
    _require(_identity_ok(benchmark, "receipt_identity_sha256"), "same-hardware benchmark receipt identity is invalid")
    hardware = benchmark.get("hardware_identity_sha256")
    runs = benchmark.get("runs")
    _require(_is_sha256(hardware) and isinstance(runs, list) and len(runs) >= 2 and all(isinstance(run, Mapping) and run.get("hardware_identity_sha256") == hardware for run in runs), "benchmark modes were not measured on the same hardware")
    _require(benchmark.get("blind_images_used") == 0, "benchmark used blind images")
    checks.append("same_hardware_benchmark")

    source = authorities["source_release_manifest"]
    source_files = source.get("files")
    _require(source.get("schema_version") == "PHAxis-source-release-manifest-2.0" and source.get("distribution") == "phaxis" and source.get("version") == VERSION and source.get("release_mode") == "formal" and source.get("source_policy") == "explicit_path_bounded_allowlist", "source release manifest is not formal PHAxis 1.0.0 authority")
    _require(
        isinstance(source_files, list)
        and source.get("tree_identity_sha256") == _sha256_json(source_files),
        "source release manifest tree identity is invalid",
    )
    clean = authorities["clean_install_receipt"]
    _require(clean.get("schema_version") == "PHAxis-clean-install-verification-1.0" and clean.get("status") == "completed_final_clean_install", "clean-install receipt is invalid")
    _require(_identity_ok(clean, "clean_install_receipt_identity_sha256"), "clean-install receipt identity is invalid")
    _require(clean.get("source_release_manifest_sha256") == bindings["source_release_manifest"]["sha256"], "clean-install receipt does not bind source release")
    root_expert = applied.get("root_expert")
    _require(
        clean.get("model_contract_proposal_sha256")
        == promotion.get("proposal_file_sha256")
        and clean.get("model_contract_proposal_identity_sha256")
        == promotion.get("proposal_identity_sha256"),
        "clean-install verification belongs to a different model-contract proposal",
    )
    _require(
        clean.get("model_bundle_id") == applied.get("model_bundle_id")
        and isinstance(root_expert, Mapping)
        and clean.get("root_expert_id") == root_expert.get("expert_id")
        and clean.get("root_bundle_identity_sha256")
        == root_expert.get("bundle_identity_sha256")
        and clean.get("hair_identity_count_expert") == expert.get("expert_id"),
        "clean-install public/model-bundle identity differs from applied contract",
    )
    _require(
        clean.get("blind_images_used") == 0
        and clean.get("root_cap_region_statistics_included") is False
        and _is_sha256(clean.get("example_output_identity_sha256")),
        "clean-install verification violates final evidence boundaries",
    )
    checks.append("source_release_clean_install")

    trait = authorities["trait_contract"]
    counts = trait.get("counts")
    invariants = trait.get("invariants")
    _require(trait.get("schema_version") == "PHAxis-trait-contract-1.0.0", "trait contract schema is invalid")
    _require(isinstance(counts, Mapping) and counts.get("nonredundant_biological_numeric_fields") == 32 and counts.get("primary_root_fields") == 19 and counts.get("root_hair_fields") == 13 and counts.get("root_cap_region_fields") == 0, "trait contract is not the canonical 19+13 contract")
    _require(isinstance(invariants, Mapping) and invariants.get("root_cap_region_output") is False and invariants.get("blind_images_used") == 0, "trait red lines are invalid")
    roots = trait.get("primary_root_traits")
    hairs = trait.get("root_hair_traits")
    _require(
        isinstance(roots, list)
        and isinstance(hairs, list)
        and [row.get("id") for row in roots if isinstance(row, Mapping)] == [f"R{index:02d}" for index in range(1, 20)]
        and [row.get("id") for row in hairs if isinstance(row, Mapping)] == [f"H{index:02d}" for index in range(1, 14)],
        "trait contract does not enumerate canonical R01--R19 and H01--H13",
    )
    catalog = trait.get("plant_facing_catalog")
    _require(
        isinstance(catalog, Mapping)
        and catalog.get("path") == "docs/phaxis/TRAIT_CONTRACT_CN.md"
        and catalog.get("languages") == ["zh-CN", "en"]
        and catalog.get("handover_copy") == CATALOG_HANDOVER_PATH
        and catalog.get("handover_copy_semantics")
        == "byte-identical copy of the catalog path at handover build time; never an independently maintained catalog",
        "trait contract does not bind the unique bilingual plant-facing catalogue",
    )
    all_traits = [*roots, *hairs]
    _require(
        len({row.get("field") for row in all_traits if isinstance(row, Mapping)})
        == 32
        and all(
            isinstance(row, Mapping)
            and all(
                isinstance(row.get(field), str) and bool(row[field].strip())
                for field in (
                    "id",
                    "field",
                    "display_name_cn",
                    "display_name_en",
                    "unit",
                )
            )
            for row in all_traits
        ),
        "trait contract bilingual names, fields or units are incomplete or duplicated",
    )
    checks.append("trait_contract_32")
    checks.append("trait_catalog_bilingual")

    attestation = contract.get("scope_attestation")
    _require(isinstance(attestation, Mapping), "scope attestation is absent")
    for field in (
        "all_legally_deliverable_manual_annotations_included",
        "training_and_validation_annotations_may_be_mixed",
        "annotation_notes_provenance_hashes_preserved",
        "biological283_includes_temperature_and_rhd6_design",
        "image_assembly_excluded",
        "blind_and_final_partitions_excluded",
        "frozen_v1_untouched",
    ):
        _require(attestation.get(field) is True, f"scope attestation is false or absent: {field}")
    checks.append("scope_attestation")
    return tuple(checks)


def _validate_materialisation_rows(
    *, role: str, rows: list[dict[str, str]], project_root: Path
) -> list[tuple[Path, str, dict[str, str]]]:
    _require(bool(rows), f"{role} is empty")
    accepted: list[tuple[Path, str, dict[str, str]]] = []
    destinations: set[str] = set()
    task_ids: set[str] = set()
    model_asset_roles: list[str] = []
    dataset_tasks: dict[str, set[str]] = {
        component: set() for component in DATASET_COMPONENT_COUNTS
    }
    for index, row in enumerate(rows, 2):
        destination = _safe_relative(row.get("package_path"))
        destination_key = destination.casefold()
        _require(destination_key not in destinations, f"duplicate/case-colliding package path in {role}: {destination}")
        destinations.add(destination_key)
        expected_prefix = {
            "dataset_manifest": "data/human_annotated500/",
            "image_manifest": "data/biological283/images/",
            "model_source_manifest": "model/source_release/",
            "model_asset_manifest": "model/assets/",
            "benchmark_manifest": "model/benchmark/",
        }[role]
        _require(destination.startswith(expected_prefix), f"{role} row {index} has wrong package boundary")
        _require(str(row.get("release_authorized", "")).strip().lower() == "true", f"{role} row {index} is not release-authorized")
        _require(bool(str(row.get("provenance", "")).strip()), f"{role} row {index} lacks provenance")
        source = _resolve_project_file(project_root, row.get("source_path"))
        expected_hash = row.get("sha256")
        _require(_is_sha256(expected_hash) and _sha256_file(source) == expected_hash, f"{role} row {index} SHA-256 mismatch")
        try:
            expected_bytes = int(row.get("bytes", ""))
        except (TypeError, ValueError) as error:
            raise HandoverError(f"{role} row {index} has invalid byte size") from error
        _require(source.stat().st_size == expected_bytes, f"{role} row {index} byte-size mismatch")
        if role == "dataset_manifest":
            _require(
                DATASET_FIELDS.issubset(row)
                and bool(row.get("annotation_kind"))
                and bool(row.get("dataset_id")),
                f"dataset row {index} lacks annotation identity",
            )
            component = str(row.get("annotation_kind", "")).strip()
            if component in dataset_tasks:
                task = str(row.get("task_id", "")).strip()
                _require(
                    bool(task) and task not in dataset_tasks[component],
                    f"dataset row {index} has absent/duplicate task_id for {component}",
                )
                dataset_tasks[component].add(task)
        if role == "image_manifest":
            task = row.get("task_id", "")
            _require(bool(task) and task not in task_ids, f"image row {index} has absent/duplicate task_id")
            task_ids.add(task)
            _require(bool(row.get("temperature_c")) and bool(row.get("genotype_or_construct")), f"image row {index} lacks temperature/RHD6-design metadata")
        if role == "model_asset_manifest":
            _require(
                MODEL_ASSET_FIELDS.issubset(row)
                and bool(str(row.get("asset_role", "")).strip()),
                f"model asset row {index} lacks asset_role",
            )
            model_asset_roles.append(str(row["asset_role"]).strip())
        if role == "model_source_manifest":
            _require(
                not _is_forbidden_assembly_component(destination),
                f"image-assembly component is forbidden: {destination}",
            )
        accepted.append((source, destination, row))
    if role == "image_manifest":
        _require(len(accepted) == 283, "biological image manifest is not exact283")
        temperatures = {str(row[2].get("temperature_c", "")).strip() for row in accepted}
        genotypes = {str(row[2].get("genotype_or_construct", "")).strip() for row in accepted}
        _require(
            len(temperatures) >= 2
            and any("rhd6" in genotype.casefold() for genotype in genotypes),
            "biological image manifest does not cover temperature and RHD6 design",
        )
    if role == "dataset_manifest":
        for component, expected in DATASET_COMPONENT_COUNTS.items():
            _require(
                len(dataset_tasks[component]) == expected,
                f"dataset manifest {component} is not exact{expected}",
            )
        _require(
            dataset_tasks["manual500_source_image"]
            == dataset_tasks["manual500_raw_return_json"],
            "manual500 source-image/raw-return task identities differ",
        )
        _require(
            dataset_tasks["manual500_source_image"]
            == {f"RHAUD-{index:03d}" for index in range(1, 501)},
            "manual500 task identities are not exact RHAUD-001--500",
        )
        _require(
            dataset_tasks["canonical443_vector_json"].issubset(
                dataset_tasks["manual500_raw_return_json"]
            ),
            "canonical443 is not a subset of the manual500 return identities",
        )
        _require(
            REQUIRED_DATASET_SUPPORT_PATHS.issubset(
                {row[1] for row in accepted}
            ),
            "dataset manifest lacks required notes/schema/provenance/split support files",
        )
    if role == "model_asset_manifest":
        _require(
            model_asset_roles.count("stageb_checkpoint") == CHECKPOINTS,
            "model assets do not contain exactly five Stage-B checkpoints",
        )
        _require(
            model_asset_roles.count("model_bundle_manifest") == 1
            and model_asset_roles.count("root_provider_asset") >= 1,
            "model assets do not contain the model-bundle manifest/root-provider payload",
        )
    return accepted


def _validate_source_release_closure(
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    source_manifest_bytes: int,
    rows: list[tuple[Path, str, dict[str, str]]],
) -> None:
    records = source_manifest.get("files")
    _require(isinstance(records, list), "source release file inventory is absent")
    expected: dict[str, tuple[str, int]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"source release row {index} is invalid")
        relative = _safe_relative(record.get("path"))
        digest = record.get("sha256")
        size = record.get("bytes")
        _require(
            relative not in expected
            and _is_sha256(digest)
            and isinstance(size, int)
            and size >= 0,
            f"source release row {index} has invalid identity",
        )
        expected[relative] = (digest, size)
    _require(
        _is_sha256(source_manifest_sha256)
        and isinstance(source_manifest_bytes, int)
        and source_manifest_bytes > 0,
        "named source-release manifest identity is invalid",
    )
    expected["SOURCE_MANIFEST.json"] = (
        source_manifest_sha256,
        source_manifest_bytes,
    )
    observed: dict[str, tuple[str, int]] = {}
    prefix = "model/source_release/"
    for _source, destination, row in rows:
        relative = destination.removeprefix(prefix)
        observed[relative] = (row["sha256"], int(row["bytes"]))
    _require(
        observed == expected,
        "model source materialisation is not the exact formal source-release tree",
    )


def inspect_handover_contract(project_root: Path, contract_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    supplied_contract = Path(contract_path)
    if supplied_contract.is_absolute():
        try:
            supplied_contract = supplied_contract.relative_to(project_root)
        except ValueError as error:
            raise HandoverError("build contract must be inside project root") from error
    contract_path = _resolve_project_file(project_root, supplied_contract.as_posix())
    contract = _read_json(contract_path)
    _require(contract.get("schema_version") == CONTRACT_SCHEMA, "unsupported handover build-contract schema")
    _require(contract.get("product") == PRODUCT and contract.get("product_version") == VERSION, "handover contract is not PHAxis 1.0.0")
    unsigned = dict(contract)
    identity = unsigned.pop("contract_identity_sha256", None)
    _require(_is_sha256(identity) and _sha256_json(unsigned) == identity, "handover contract identity is invalid")
    authorities: dict[str, dict[str, Any]] = {}
    binding_paths: dict[str, Path] = {}
    for role in REQUIRED_BINDINGS:
        path, _ = _binding_path(contract, role, project_root)
        binding_paths[role] = path
        if role not in MATERIALISATION_BINDINGS:
            authorities[role] = _read_json(path)
    checks = _semantic_gate(contract, authorities)
    materialisations: dict[str, list[tuple[Path, str, dict[str, str]]]] = {}
    for role in MATERIALISATION_BINDINGS:
        materialisations[role] = _validate_materialisation_rows(
            role=role,
            rows=_read_rows(binding_paths[role]),
            project_root=project_root,
        )
    _validate_source_release_closure(
        source_manifest=authorities["source_release_manifest"],
        source_manifest_sha256=contract["bindings"]["source_release_manifest"][
            "sha256"
        ],
        source_manifest_bytes=binding_paths[
            "source_release_manifest"
        ].stat().st_size,
        rows=materialisations["model_source_manifest"],
    )
    checkpoint_rows = {row[2]["sha256"]: row[1] for row in materialisations["model_asset_manifest"]}
    for record in contract["train399_checkpoints"]:
        _require(record["sha256"] in checkpoint_rows, "train399 checkpoint is absent from model asset manifest")
    return {
        "status": "passed",
        "contract": contract,
        "contract_path": contract_path,
        "binding_paths": binding_paths,
        "authorities": authorities,
        "materialisations": materialisations,
        "checks": list(checks) + [f"materialisation_{role}" for role in MATERIALISATION_BINDINGS],
    }


def _copy_exact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    _require(_is_sha256(expected_sha256), f"copy expected SHA-256 is invalid: {destination}")
    _require(
        isinstance(expected_bytes, int) and expected_bytes >= 0,
        f"copy expected byte size is invalid: {destination}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        digest = hashlib.sha256()
        copied_bytes = 0
        with source.open("rb") as reader:
            for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                handle.write(chunk)
                digest.update(chunk)
                copied_bytes += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if copied_bytes != expected_bytes or digest.hexdigest() != expected_sha256:
            raise HandoverError(
                f"source changed after inspection or copy verification failed: {destination}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _catalog_unit_cell(unit: str) -> str:
    return {
        "um": "µm (`um`)",
        "um2": "µm² (`um2`)",
        "um2_per_mm": "µm²/mm (`um2_per_mm`)",
        "um_per_mm": "µm/mm (`um_per_mm`)",
        "rad_per_mm": "rad/mm (`rad_per_mm`)",
        "count_per_mm": "count/mm",
        "count": "count",
        "ratio": "ratio",
    }.get(unit, unit)


def _validate_catalog_text(text: str, trait: Mapping[str, Any]) -> None:
    rows = [*trait["primary_root_traits"], *trait["root_hair_traits"]]
    compact = " ".join(text.split())
    _require(len(rows) == 32, "trait lists do not contain 19+13 rows")
    for row in rows:
        marker = (
            f"| {row['id']} | {row['display_name_cn']}<br>{row['display_name_en']} | "
            f"`{row['field']}` | {_catalog_unit_cell(row['unit'])} |"
        )
        _require(
            text.count(marker) == 1,
            f"bilingual catalogue row differs from the trait contract: {row['id']}",
        )
    _require(
        "single authoritative, human-readable" in compact
        and "The root-cap representation is exactly one distal/root-cap point" in text
        and "no root-cap region" in text
        and "H06、H07、H13" in text
        and "`[1,4) mm`" in text
        and "never an independently edited second catalogue" in compact,
        "plant-facing catalogue lost its authority, root-cap, axial-window or copy boundary",
    )


def _readme() -> str:
    return """# PHAxis 1.0.0 复用交接包

本目录是哈希锁定的 PHAxis 1.0.0 复用交付物。顶层 `data/human_annotated500/` 保存依法可交付且任务身份闭合的 500 张人工标注源图、500 份当前原始人工回传，以及其中 443 例 HumanCurated443 canonical vector；训练/验证标注允许混合，但逐文件备注、provenance、授权状态和 SHA-256 均保留。`data/biological283/images/` 保存温度、RHD6 等生物学设计的 283 张源图。`model/` 保存已通过 clean-install 的最终工作流源码、使用材料、正式模型资产和同硬件 benchmark；不包含显微图片拼接/组装工具。

正式包只有在 applied model contract、五个严格 train399 checkpoint、fresh exact283、最终 fusion/traits、same-hardware benchmark、正式 source release/clean-install 以及数据/图像清单全部交叉绑定时才能构建。任一项缺失或哈希漂移都会拒绝构建。

使用前运行 `python verify_package.py <包目录>`。随后按 `model/source_release/README.md` 安装并按其中工作流说明准备新图；输入应是已经完成采集拼接的完整图像。32 项能力与测量边界见中英双语 `PHENOTYPE_CAPABILITIES_CN.md`；它在构建时从项目权威路径 `docs/phaxis/TRAIT_CONTRACT_CN.md` 逐字节复制，不是另一套表型定义。
"""


def _verifier_launcher() -> str:
    return """#!/usr/bin/env python
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "model/workflow"))
from handover_package_common import HandoverError, verify_handover_package
try:
    print(verify_handover_package(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent))
except HandoverError as error:
    print(f"PHAxis handover verification failed: {error}")
    raise SystemExit(2)
"""


def _package_records(root: Path, roles: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "role": roles.get(relative, "generated_control")})
    return records


def _verify_packaged_authority_closure(
    *, root: Path, records: list[Mapping[str, Any]], receipt: Mapping[str, Any]
) -> list[str]:
    contract_matches = [
        record for record in records if record.get("role") == "build_contract"
    ]
    _require(len(contract_matches) == 1, "packaged build contract is absent or duplicated")
    contract = _read_json(root / _safe_relative(contract_matches[0].get("path")))
    _require(
        contract.get("schema_version") == CONTRACT_SCHEMA
        and contract.get("product") == PRODUCT
        and contract.get("product_version") == VERSION,
        "packaged build contract identity is invalid",
    )
    unsigned = dict(contract)
    contract_identity = unsigned.pop("contract_identity_sha256", None)
    _require(
        _is_sha256(contract_identity)
        and _sha256_json(unsigned) == contract_identity
        and receipt.get("contract_identity_sha256") == contract_identity,
        "packaged build contract logical identity mismatch",
    )
    bindings = contract.get("bindings")
    _require(
        isinstance(bindings, Mapping)
        and set(bindings) == set(REQUIRED_BINDINGS),
        "packaged build contract binding set is incomplete or expanded",
    )
    _require(
        receipt.get("binding_sha256")
        == {role: bindings[role]["sha256"] for role in sorted(bindings)},
        "packaged build contract/receipt binding map differs",
    )
    _require(
        receipt.get("checkpoint_sha256_in_member_order")
        == [
            record.get("sha256")
            for record in contract.get("train399_checkpoints", ())
            if isinstance(record, Mapping)
        ],
        "packaged build receipt checkpoint order differs from contract",
    )

    authority_records: dict[str, Mapping[str, Any]] = {}
    for role in REQUIRED_BINDINGS:
        matches = [
            record
            for record in records
            if record.get("role") == f"authority_{role}"
        ]
        _require(
            len(matches) == 1,
            f"packaged authority is absent or duplicated: {role}",
        )
        _require(
            matches[0].get("sha256") == bindings[role]["sha256"],
            f"packaged authority hash differs from its contract binding: {role}",
        )
        authority_records[role] = matches[0]
    authorities = {
        role: _read_json(root / _safe_relative(authority_records[role]["path"]))
        for role in REQUIRED_BINDINGS
        if role not in MATERIALISATION_BINDINGS
    }
    checks = list(_semantic_gate(contract, authorities))

    materialisations: dict[
        str, list[tuple[Path, str, dict[str, str]]]
    ] = {}
    for role in MATERIALISATION_BINDINGS:
        manifest_path = root / _safe_relative(authority_records[role]["path"])
        rows = _read_rows(manifest_path)
        package_rows: list[dict[str, str]] = []
        for row in rows:
            local = dict(row)
            local["source_path"] = _safe_relative(row.get("package_path"))
            package_rows.append(local)
        materialisations[role] = _validate_materialisation_rows(
            role=role,
            rows=package_rows,
            project_root=root,
        )
        declared = {
            _safe_relative(row.get("package_path")) for row in rows
        }
        packaged = {
            str(record.get("path"))
            for record in records
            if record.get("role") == role
        }
        _require(
            declared == packaged,
            f"packaged {role} payload closure differs from its authority",
        )
        checks.append(f"materialisation_{role}")
    _validate_source_release_closure(
        source_manifest=authorities["source_release_manifest"],
        source_manifest_sha256=str(
            authority_records["source_release_manifest"].get("sha256")
        ),
        source_manifest_bytes=int(
            authority_records["source_release_manifest"].get("bytes", -1)
        ),
        rows=materialisations["model_source_manifest"],
    )
    checkpoint_rows = {
        row[2]["sha256"] for row in materialisations["model_asset_manifest"]
    }
    _require(
        all(
            isinstance(record, Mapping)
            and record.get("sha256") in checkpoint_rows
            for record in contract.get("train399_checkpoints", ())
        ),
        "packaged train399 checkpoint is absent from model assets",
    )
    return checks


def build_handover_package(*, project_root: Path, contract_path: Path, output: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    output = output.resolve()
    _require(not output.exists(), "output must be absent; existing packages are never overwritten")
    _require(_inside(output, project_root), "output must stay inside the project root")
    inspection = inspect_handover_contract(project_root, contract_path)
    contract = inspection["contract"]
    trait = inspection["authorities"]["trait_contract"]
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    roles: dict[str, str] = {}
    try:
        for role, rows in inspection["materialisations"].items():
            for source, relative, row in rows:
                _copy_exact(
                    source,
                    staging / relative,
                    expected_sha256=row["sha256"],
                    expected_bytes=int(row["bytes"]),
                )
                roles[relative] = role
        for role, source in inspection["binding_paths"].items():
            relative = f"model/authorities/{role}{source.suffix.lower()}"
            binding = contract["bindings"][role]
            _copy_exact(
                source,
                staging / relative,
                expected_sha256=binding["sha256"],
                expected_bytes=source.stat().st_size,
            )
            roles[relative] = f"authority_{role}"
        contract_relative = "model/authorities/handover_build_contract.json"
        _copy_exact(
            inspection["contract_path"],
            staging / contract_relative,
            expected_sha256=_sha256_file(inspection["contract_path"]),
            expected_bytes=inspection["contract_path"].stat().st_size,
        )
        roles[contract_relative] = "build_contract"
        runtime_source = Path(__file__).resolve()
        _copy_exact(
            runtime_source,
            staging / "model/workflow/handover_package_common.py",
            expected_sha256=_sha256_file(runtime_source),
            expected_bytes=runtime_source.stat().st_size,
        )
        roles["model/workflow/handover_package_common.py"] = "generated_verifier_runtime"
        _write(staging / "README_CN.md", _readme().encode("utf-8"))
        catalog_source = _resolve_project_file(project_root, CATALOG_PROJECT_PATH)
        catalog_bytes = catalog_source.read_bytes()
        try:
            catalog_text = catalog_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HandoverError(
                "plant-facing phenotype catalogue is not UTF-8"
            ) from error
        _validate_catalog_text(catalog_text, trait)
        _copy_exact(
            catalog_source,
            staging / CATALOG_HANDOVER_PATH,
            expected_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            expected_bytes=len(catalog_bytes),
        )
        roles[CATALOG_HANDOVER_PATH] = "phenotype_catalog_byte_copy"
        _write(staging / "verify_package.py", _verifier_launcher().encode("utf-8"))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "passed",
            "product": PRODUCT,
            "product_version": VERSION,
            "contract_identity_sha256": contract["contract_identity_sha256"],
            "checks": inspection["checks"],
            "binding_sha256": {role: record["sha256"] for role, record in sorted(contract["bindings"].items())},
            "checkpoint_sha256_in_member_order": [row["sha256"] for row in contract["train399_checkpoints"]],
            "plant_facing_catalog": {
                "source_path": CATALOG_PROJECT_PATH,
                "package_path": CATALOG_HANDOVER_PATH,
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "bytes": len(catalog_bytes),
                "copy_semantics": "byte_identical_at_handover_build_time",
            },
        }
        receipt["receipt_identity_sha256"] = _sha256_json(receipt)
        _write(staging / RECEIPT_NAME, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        records = _package_records(staging, roles)
        manifest = {"schema_version": PACKAGE_SCHEMA, "product": PRODUCT, "product_version": VERSION, "files": records, "tree_identity_sha256": _sha256_json(records)}
        _write(staging / MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        verify_handover_package(staging)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_handover_package(root: Path) -> dict[str, Any]:
    supplied_root = Path(root)
    _require(not supplied_root.is_symlink(), "package root may not be a symlink")
    root = supplied_root.resolve()
    _require(root.is_dir(), f"package directory is absent: {root}")
    _require((root / "data").is_dir() and (root / "model").is_dir(), "top-level data/ and model/ are required")
    allowed_top = {"data", "model", "README_CN.md", CATALOG_HANDOVER_PATH, "verify_package.py", RECEIPT_NAME, MANIFEST_NAME}
    _require(all(path.name in allowed_top for path in root.iterdir()), "unexpected top-level package entry")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"package may not contain symlinks: {path}")
    manifest = _read_json(root / MANIFEST_NAME)
    _require(manifest.get("schema_version") == PACKAGE_SCHEMA and manifest.get("product") == PRODUCT and manifest.get("product_version") == VERSION, "package manifest identity is invalid")
    records = manifest.get("files")
    _require(isinstance(records, list), "package manifest files are invalid")
    paths = [row.get("path") for row in records if isinstance(row, Mapping)]
    _require(paths == sorted(paths) and len(paths) == len(set(paths)), "package manifest paths are unsorted or duplicated")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / MANIFEST_NAME
    )
    _require(paths == actual, "package file closure differs from manifest")
    for record in records:
        path = root / _safe_relative(record["path"])
        _require(path.stat().st_size == record.get("bytes") and _sha256_file(path) == record.get("sha256"), f"package file hash/size mismatch: {record['path']}")
    _require(manifest.get("tree_identity_sha256") == _sha256_json(records), "package tree identity mismatch")
    receipt = _read_json(root / RECEIPT_NAME)
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA and receipt.get("status") == "passed", "build receipt did not pass")
    _require(_identity_ok(receipt, "receipt_identity_sha256"), "build receipt identity is invalid")
    binding_sha = receipt.get("binding_sha256")
    _require(isinstance(binding_sha, Mapping) and set(binding_sha) == set(REQUIRED_BINDINGS), "build receipt binding set is incomplete")
    for role, expected in binding_sha.items():
        matches = [record for record in records if record.get("role") == f"authority_{role}"]
        _require(len(matches) == 1 and matches[0].get("sha256") == expected, f"packaged authority does not match build receipt: {role}")
    semantic_checks = _verify_packaged_authority_closure(
        root=root, records=records, receipt=receipt
    )
    _require(
        receipt.get("checks") == semantic_checks,
        "packaged semantic check set differs from build receipt",
    )
    _require(sum(record.get("role") == "image_manifest" for record in records) == 283, "packaged biological image set is not exact283")
    catalog_path = root / CATALOG_HANDOVER_PATH
    catalog_receipt = receipt.get("plant_facing_catalog")
    catalog_records = [
        record for record in records if record.get("role") == "phenotype_catalog_byte_copy"
    ]
    _require(
        catalog_path.is_file()
        and isinstance(catalog_receipt, Mapping)
        and catalog_receipt.get("source_path") == CATALOG_PROJECT_PATH
        and catalog_receipt.get("package_path") == CATALOG_HANDOVER_PATH
        and catalog_receipt.get("copy_semantics")
        == "byte_identical_at_handover_build_time"
        and len(catalog_records) == 1
        and catalog_records[0].get("path") == CATALOG_HANDOVER_PATH
        and catalog_records[0].get("sha256") == catalog_receipt.get("sha256")
        and catalog_records[0].get("bytes") == catalog_receipt.get("bytes")
        and _sha256_file(catalog_path) == catalog_receipt.get("sha256"),
        "handover phenotype catalogue does not match its build-time source receipt",
    )
    catalog_text = catalog_path.read_text(encoding="utf-8")
    trait_records = [
        record for record in records if record.get("role") == "authority_trait_contract"
    ]
    _require(
        len(trait_records) == 1,
        "packaged trait-contract authority is absent or duplicated",
    )
    packaged_trait = _read_json(root / _safe_relative(trait_records[0]["path"]))
    _validate_catalog_text(catalog_text, packaged_trait)
    model_paths = [value for value in paths if value.startswith("model/source_release/")]
    _require(
        not any(_is_forbidden_assembly_component(value) for value in model_paths),
        "model contains forbidden image-assembly component",
    )
    return {"status": "passed", "product": PRODUCT, "product_version": VERSION, "files": len(records), "tree_identity_sha256": manifest["tree_identity_sha256"]}


__all__ = [
    "CONTRACT_SCHEMA",
    "HandoverError",
    "build_handover_package",
    "inspect_handover_contract",
    "verify_handover_package",
]
