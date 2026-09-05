#!/usr/bin/env python
"""Build the unique, hash-closed PHAxis manuscript/release evidence graph.

This command performs no discovery.  Every authoritative receipt must be named
on the command line.  It never reads images, predictions, checkpoints, or model
code and is therefore suitable for a CPU-only release audit.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_ROOT), str(PROJECT_ROOT / "src")]

from phaxis.publication_evidence import (
    FIGURE_SOURCE_INPUT_ROLES,
    SUPPLEMENTARY_FIGURE_STEMS,
    WT_SECONDARY_RESOURCE_ROLES,
    supplementary_figure_contract,
    figure_suite_identity_preimage,
    validate_wt_secondary_analysis_binding,
    validate_wt_secondary_evidence,
)
from phaxis.contracts import ContractError
from phaxis.hair_stageb.evaluation_inference import (
    EVALUATION_ARTIFACT_ROLE,
    EVALUATION_DETECTION_SCHEMA,
    EVALUATION_RUN_SCHEMA,
)
from phaxis.public_identity import validate_proposal_public_identity
from phaxis.multitrait_atlas import (
    MultitraitAtlasError,
    validate_multitrait_atlas_structure,
)
from phaxis.narrative_decision import (
    NarrativeDecisionError,
    validate_narrative_decision,
)
from phaxis.publication_titles import title_contract
from phaxis.supplementary_tables import (  # noqa: E402
    BUNDLE_RECEIPT as SUPPLEMENTARY_TABLE_RECEIPT,
    TABLE_STEMS as SUPPLEMENTARY_TABLE_STEMS,
    SupplementaryTableError,
    validate_supplementary_table_data_bundle,
)

from source_release_common import (
    LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256,
    _candidate_gate,
    _evaluation_gate,
    _root_provider_gate,
    _selection_gate,
)


SCHEMA_VERSION = "PHAxis-manuscript-release-evidence-graph-1.1"
ROLE_ORDER = (
    "model_contract_proposal",
    "train399_candidate",
    "train399_selection",
    "train399_evaluation",
    "root_exact283",
    "stageb",
    "fusion",
    "traits",
    "cohorts",
    "analysis",
    "profiles",
    "figure_inputs",
    "figures",
)
SUMMARY_ROLES = ("stageb", "fusion", "traits", "cohorts", "analysis", "profiles", "figure_inputs", "figures")
EXPECTED_SCHEMAS = {
    "stageb": "PHAxis-StageB-inference-run-1.1",
    "fusion": "PHAxis-fusion-run-1.1",
    "traits": "PHAxis-trait-export-1.0",
    "cohorts": "PHAxis-biological-cohorts-1.0",
    "analysis": "PHAxis-exploratory-biological-analysis-1.0",
    "profiles": "PHAxis-distal-axis-profile-export-1.0.0",
    "figure_inputs": "PHAxis-manuscript-figure-inputs-2.0",
    "figures": "PHAxis-publication-figure-suite-1.0",
}
IDENTITY_FIELDS = {
    "model_contract_proposal": "model_contract_identity_sha256",
    "train399_candidate": "candidate_manifest_identity_sha256",
    "train399_selection": "selection_receipt_identity_sha256",
    "root_exact283": "audit_identity_sha256",
    "stageb": "summary_identity_sha256",
    "fusion": "summary_identity_sha256",
    "traits": "export_identity_sha256",
    "cohorts": "cohort_build_identity_sha256",
    "analysis": "analysis_identity_sha256",
    "profiles": "export_identity_sha256",
    "figure_inputs": "figure_input_assembly_identity_sha256",
    "figures": "figure_suite_identity_sha256",
}
FORBIDDEN_FINAL_MARKERS = (
    "provisional",
    "development_only",
    "development-only",
    "development evidence only",
    "development_evidence_only",
    "blocked_pending",
    "not_for_submission",
    "not for submission",
)
FIGURE_RESOURCE_ROLES = (
    "trait_contract",
    "figure1_image",
    "figure1_geometry",
    "development_per_image",
    "development_tolerance",
    "development_threshold",
    "development_strata",
    "assurance_metrics",
    "assurance_pairs",
    "assurance_support",
    "qcdev_assignment",
    "overlay_selection",
    "overlay_audit",
    "phenotype_points",
    "phenotype_effects",
    "narrative_decision",
    "multitrait_atlas",
    "axial_profiles",
    "cohort_flow",
    "workflow_stages",
    "runtime_summary",
    "runtime_per_image",
    *WT_SECONDARY_RESOURCE_ROLES,
)
FIGURE_PROVENANCE_ROLES = (
    "historical_development",
    "measurement_assurance",
    "overlay_index",
    "profile_analysis",
    "runtime_latency",
    "runtime_production",
    "runtime_latency_comparison",
    "runtime_production_comparison",
    "baseline_runtime_latency",
    "baseline_runtime_production",
)


class EvidenceManifestError(RuntimeError):
    """An evidence receipt or cross-stage binding is not release-closed."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceManifestError(message)


def _read_json_object(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"{role}: receipt does not exist: {resolved}")
    _require(not resolved.is_symlink(), f"{role}: symlink receipts are forbidden")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceManifestError(f"{role}: invalid UTF-8 JSON receipt") from error
    _require(isinstance(payload, dict), f"{role}: receipt must be one JSON object")
    return resolved, payload


def _sealed_identity(payload: Mapping[str, Any], field: str, role: str) -> str:
    identity = payload.get(field)
    _require(_is_sha256(identity), f"{role}: missing or invalid {field}")
    unsigned = deepcopy(dict(payload))
    unsigned.pop(field, None)
    _require(
        sha256_json(unsigned) == identity,
        f"{role}: {field} does not seal the complete receipt",
    )
    return str(identity)


def _walk(payload: Any, prefix: str = ""):
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, payload[key]
            yield from _walk(payload[key], path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]"
            yield path, value
            yield from _walk(value, path)


def _guard_final_summary(role: str, payload: Mapping[str, Any]) -> None:
    _require(
        payload.get("schema_version") == EXPECTED_SCHEMAS[role],
        f"{role}: unsupported schema_version",
    )
    _require(payload.get("blind_images_used") == 0, f"{role}: blind_images_used must be 0")
    root_fields = 0
    for path, value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if leaf in {"root_cap_region_output", "root_cap_region_statistics_included"}:
            root_fields += 1
            _require(value is False, f"{role}: {path} must be false")
        if leaf == "blind_images_used":
            _require(value == 0, f"{role}: {path} must be 0")
        if isinstance(value, str):
            lowered = value.casefold()
            _require(
                not any(marker in lowered for marker in FORBIDDEN_FINAL_MARKERS),
                f"{role}: development/provisional marker at {path}",
            )
    _require(root_fields > 0, f"{role}: no explicit root-cap-region false guard")


def _validate_formal_gate_receipts(
    paths: Mapping[str, Path], payloads: Mapping[str, dict[str, Any]]
) -> None:
    checks: list[dict[str, Any]] = []
    candidate = _candidate_gate(paths["train399_candidate"], checks)
    selection = _selection_gate(paths["train399_selection"], candidate, checks)
    _evaluation_gate(
        paths["train399_evaluation"],
        paths["train399_candidate"],
        paths["train399_selection"],
        candidate,
        selection,
        checks,
    )
    _root_provider_gate(paths["root_exact283"], checks)
    failures = [row["code"] for row in checks if not row["passed"]]
    _require(not failures, "formal Gate receipt checks failed: " + ", ".join(failures))
    _require(candidate == payloads["train399_candidate"], "candidate receipt changed while read")
    _require(selection == payloads["train399_selection"], "selection receipt changed while read")


def _validate_evaluator12(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
        and payload.get("status") == "completed"
        and payload.get("independent_accuracy_claim_allowed") is False,
        "train399_evaluation: evaluator1.2 status/scope changed",
    )
    hierarchy = payload.get("metric_hierarchy")
    _require(
        isinstance(hierarchy, Mapping)
        and hierarchy.get("primary")
        == "one-to-one tolerant biological-hair presence; bidirectional partial centreline coverage without endpoint gates"
        and hierarchy.get("primary_minimum_truth_coverage") == 0.25
        and hierarchy.get("primary_minimum_prediction_coverage") == 0.25
        and hierarchy.get("primary_minimum_direction_cosine") == 0.0,
        "train399_evaluation: tolerant biological-presence hierarchy changed",
    )
    bootstrap = payload.get("paired_bootstrap_95ci")
    _require(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("method")
        == "paired image-level nonparametric bootstrap"
        and bootstrap.get("repetitions") == 10000
        and bootstrap.get("seed") == 20260828
        and isinstance(
            bootstrap.get("delta_stageb_train399_minus_hybrid", {}).get(
                "biological_presence_f1_20um"
            ),
            Mapping,
        ),
        "train399_evaluation: primary paired image-level bootstrap changed",
    )
    rows = payload.get("per_image")
    _require(
        isinstance(rows, list)
        and len(rows) == 44
        and len({str(row.get("task_id")) for row in rows}) == 44,
        "train399_evaluation: per-image QC44 scope changed",
    )
    task_order = [str(row["task_id"]) for row in rows]
    for row in rows:
        for expert in ("stageb_train399", "hybrid_max"):
            record = row.get(expert)
            presence = (
                record.get("biological_presence_tp")
                if isinstance(record, Mapping)
                else None
            )
            _require(
                isinstance(presence, Mapping)
                and {str(key) for key in presence} == {"5.0", "10.0", "20.0"},
                f"train399_evaluation: {expert} per-image biological TP missing",
            )
    locks = payload.get("prediction_input_locks")
    _require(isinstance(locks, Mapping), "train399_evaluation: prediction locks missing")
    for list_field, identity_field in (
        ("stageb_detection_files", "stageb_detection_set_identity_sha256"),
        ("hybrid_prediction_files", "hybrid_prediction_set_identity_sha256"),
    ):
        records = locks.get(list_field)
        _require(
            isinstance(records, list)
            and len(records) == 44
            and [str(record.get("task_id")) for record in records] == task_order
            and all(
                isinstance(record, Mapping)
                and set(record) == {"task_id", "sha256"}
                and _is_sha256(record.get("sha256"))
                for record in records
            )
            and locks.get(identity_field) == sha256_json(records),
            f"train399_evaluation: {list_field} ordered lock changed",
        )
    evaluation_authority = payload.get("evaluation_inference_authority")
    _require(
        isinstance(evaluation_authority, Mapping)
        and evaluation_authority.get("schema_version") == EVALUATION_RUN_SCHEMA
        and evaluation_authority.get("artifact_role") == EVALUATION_ARTIFACT_ROLE
        and evaluation_authority.get("evaluation_detection_schema_version")
        == EVALUATION_DETECTION_SCHEMA,
        "train399_evaluation: evaluation-only inference authority/schema missing",
    )
    for field in (
        "evaluation_inference_summary_sha256",
        "evaluation_inference_summary_identity_sha256",
        "evaluation_gate_identity_sha256",
        "evaluation_detection_set_identity_sha256",
    ):
        _require(
            _is_sha256(evaluation_authority.get(field)),
            f"train399_evaluation: evaluation-only {field} invalid",
        )
    _require(
        evaluation_authority.get("evaluation_detection_set_identity_sha256")
        == locks["stageb_detection_set_identity_sha256"]
        and evaluation_authority.get(
            "model_contract_proposal_required_for_artifact"
        )
        is False
        and evaluation_authority.get("model_contract_proposal_present") is False
        and evaluation_authority.get("production_consumption_allowed") is False
        and evaluation_authority.get("fusion_consumption_allowed") is False
        and evaluation_authority.get("traits_consumption_allowed") is False
        and evaluation_authority.get(
            "canonical_annotations_read_during_inference"
        )
        is False
        and evaluation_authority.get("condition_metadata_used_for_routing") is False
        and evaluation_authority.get("independent_accuracy_claim_allowed") is False
        and evaluation_authority.get("blind_images_used") == 0,
        "train399_evaluation: evaluation-only authority is circular/deployable/tainted",
    )
    inputs = payload.get("inputs_sha256")
    training_contract = payload.get("training_contract")
    _require(
        isinstance(inputs, Mapping)
        and inputs.get("evaluation_inference_summary")
        == evaluation_authority["evaluation_inference_summary_sha256"]
        and isinstance(training_contract, Mapping)
        and training_contract.get("evaluation_gate_identity_sha256")
        == evaluation_authority["evaluation_gate_identity_sha256"]
        and training_contract.get(
            "evaluation_inference_summary_identity_sha256"
        )
        == evaluation_authority[
            "evaluation_inference_summary_identity_sha256"
        ],
        "train399_evaluation: evaluation-only summary/gate binding changed",
    )
    comparator = payload.get("comparator_contract", {}).get("hybrid_max")
    hybrid_identity = locks["hybrid_prediction_set_identity_sha256"]
    _require(
        isinstance(comparator, Mapping)
        and comparator.get("evidence_role")
        == "locked_legacy_development_comparator"
        and comparator.get("schema_version")
        == "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
        and comparator.get("identity_hair_variant") == "hybrid_verified_increment"
        and comparator.get("count_hair_variant") == "hybrid_verified_increment"
        and comparator.get("endpoint_complete_identity_layer") is True
        and comparator.get("phaxis_payload_allowed") is False
        and comparator.get("stageb_identity_source_allowed") is False
        and comparator.get("prediction_set_identity_sha256") == hybrid_identity
        and comparator.get("expected_prediction_set_identity_sha256")
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
        and hybrid_identity
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256,
        "train399_evaluation: legacy comparator contract/set identity changed",
    )
    return {
        "task_order_identity_sha256": sha256_json(task_order),
        "stageb_detection_set_identity_sha256": locks[
            "stageb_detection_set_identity_sha256"
        ],
        "stageb_detection_schema_version": evaluation_authority[
            "evaluation_detection_schema_version"
        ],
        "stageb_detection_artifact_role": evaluation_authority["artifact_role"],
        "stageb_evaluation_inference_summary_sha256": evaluation_authority[
            "evaluation_inference_summary_sha256"
        ],
        "stageb_evaluation_inference_summary_identity_sha256": (
            evaluation_authority[
                "evaluation_inference_summary_identity_sha256"
            ]
        ),
        "stageb_evaluation_gate_identity_sha256": evaluation_authority[
            "evaluation_gate_identity_sha256"
        ],
        "stageb_production_consumption_allowed": False,
        "stageb_fusion_consumption_allowed": False,
        "stageb_traits_consumption_allowed": False,
        "legacy_hybrid_prediction_set_identity_sha256": hybrid_identity,
        "legacy_comparator_schema_version": comparator["schema_version"],
        "legacy_comparator_identity_hair_variant": comparator[
            "identity_hair_variant"
        ],
    }


def _validate_model_contract_proposal(payload: Mapping[str, Any]) -> tuple[str, str]:
    _require(
        payload.get("schema_version") == "PHAxis-model-contract-1.0.0",
        "model_contract_proposal: unsupported contract schema",
    )
    _require(
        payload.get("formal_release_status") == "passed_proposal_not_official",
        "model_contract_proposal: not a passed, unapplied contract candidate",
    )
    promotion = payload.get("promotion")
    _require(isinstance(promotion, Mapping), "model_contract_proposal: promotion block missing")
    _require(
        promotion.get("schema_version") == "PHAxis-model-contract-promotion-1.0"
        and promotion.get("status") == "validated_proposal_not_applied"
        and promotion.get("official_apply_performed") is False,
        "model_contract_proposal: proposal status/guard changed",
    )
    identity = _sealed_identity(payload, "model_contract_identity_sha256", "model_contract_proposal")
    try:
        public_identity = validate_proposal_public_identity(payload)
    except ContractError as error:
        raise EvidenceManifestError(
            "model_contract_proposal: canonical public identity is invalid"
        ) from error
    return identity, str(payload.get("formal_release_status"))


def _require_proposal_binding(
    role: str,
    payload: Mapping[str, Any],
    *,
    proposal_sha256: str,
    proposal_identity_sha256: str,
) -> None:
    _require(
        payload.get("model_contract_proposal_sha256") == proposal_sha256,
        f"{role}: model-contract proposal file SHA mismatch",
    )
    _require(
        payload.get("model_contract_proposal_identity_sha256")
        == proposal_identity_sha256,
        f"{role}: model-contract proposal logical identity mismatch",
    )


def _resolve_figure_input_file(
    manifest_path: Path, record: Mapping[str, Any], role: str
) -> Path:
    raw = record.get("path")
    digest = record.get("sha256")
    _require(isinstance(raw, str) and raw, f"figure_inputs/{role}: path missing")
    _require(_is_sha256(digest), f"figure_inputs/{role}: SHA-256 missing")
    relative = Path(raw)
    _require(not relative.is_absolute(), f"figure_inputs/{role}: absolute paths forbidden")
    resolved = (manifest_path.parent / relative).resolve()
    try:
        resolved.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise EvidenceManifestError(
            f"figure_inputs/{role}: path escapes assembled evidence root"
        ) from error
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"figure_inputs/{role}: file missing/non-regular",
    )
    _require(
        "blind" not in relative.as_posix().casefold(),
        f"figure_inputs/{role}: blind path refused",
    )
    _require(
        sha256_file(resolved) == digest,
        f"figure_inputs/{role}: source file SHA mismatch",
    )
    return resolved


def _read_csv_records(path: Path, role: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(reader.fieldnames is not None, f"{role}: CSV header missing")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise EvidenceManifestError(f"{role}: invalid UTF-8 CSV") from error


def _validate_figure_input_assembly(
    *,
    path: Path,
    payload: Mapping[str, Any],
    proposal: Mapping[str, Any],
    proposal_sha256: str,
    proposal_identity_sha256: str,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == "PHAxis-manuscript-figure-inputs-2.0"
        and payload.get("assembler_schema_version")
        == "PHAxis-publication-figure-input-assembly-1.0"
        and payload.get("status") == "final",
        "figure_inputs: not a final production assembly",
    )
    assembly_identity = _sealed_identity(
        payload, "figure_input_assembly_identity_sha256", "figure_inputs"
    )
    expected_supplementary_contract = supplementary_figure_contract()
    expected_supplementary_contract["contract_identity_sha256"] = sha256_json(
        expected_supplementary_contract
    )
    _require(
        payload.get("supplementary_figure_contract")
        == expected_supplementary_contract,
        "figure_inputs: ordered supplementary S1--S9 contract changed",
    )
    _require_proposal_binding(
        "figure_inputs",
        payload,
        proposal_sha256=proposal_sha256,
        proposal_identity_sha256=proposal_identity_sha256,
    )
    proposal_root = proposal.get("root_expert")
    expected_public = {
        "model_bundle_id": proposal.get("model_bundle_id"),
        "root_expert_id": (
            proposal_root.get("expert_id")
            if isinstance(proposal_root, Mapping)
            else None
        ),
        "root_provider_role": (
            proposal_root.get("provider_role")
            if isinstance(proposal_root, Mapping)
            else None
        ),
    }
    _require(
        payload.get("model_contract_public_identity") == expected_public,
        "figure_inputs: proposal-owned public model identity mismatch",
    )
    stageb_expert = json.loads(
        paths["stageb"].read_text(encoding="utf-8")
    ).get("detection_model_metadata", {}).get("expert_id")
    _require(
        payload.get("model_bundle_id") == expected_public["model_bundle_id"]
        and payload.get("root_expert_id") == expected_public["root_expert_id"]
        and payload.get("hair_identity_expert_id") == stageb_expert,
        "figure_inputs: top-level public model identity mismatch",
    )
    expected_sources = {
        "train399_evaluation": sha256_file(paths["train399_evaluation"]),
        "root_exact283": sha256_file(paths["root_exact283"]),
        **{
            role: sha256_file(paths[role])
            for role in ("stageb", "fusion", "traits", "cohorts", "analysis", "profiles")
        },
    }
    _require(
        payload.get("source_summary_sha256") == expected_sources,
        "figure_inputs: exact eight-source receipt closure mismatch",
    )
    resources = payload.get("resources")
    source_inputs = payload.get("source_inputs")
    provenance = payload.get("provenance_receipts")
    lineage = payload.get("resource_lineage")
    _require(
        isinstance(resources, Mapping)
        and set(resources) == set(FIGURE_RESOURCE_ROLES),
        "figure_inputs: exact resource route is incomplete",
    )
    _require(
        isinstance(source_inputs, Mapping)
        and set(source_inputs) == set(FIGURE_SOURCE_INPUT_ROLES),
        "figure_inputs: source-input route is incomplete",
    )
    _require(
        isinstance(provenance, Mapping)
        and set(provenance) == set(FIGURE_PROVENANCE_ROLES),
        "figure_inputs: provenance receipt route is incomplete",
    )
    _require(
        isinstance(lineage, Mapping)
        and set(lineage) == set(FIGURE_RESOURCE_ROLES)
        and all(isinstance(value, list) and value for value in lineage.values()),
        "figure_inputs: resource lineage is incomplete",
    )
    allowed_lineage_roles = (
        set(expected_sources)
        | set(FIGURE_RESOURCE_ROLES)
        | set(FIGURE_SOURCE_INPUT_ROLES)
        | set(FIGURE_PROVENANCE_ROLES)
        | {"train399_selection"}
    )
    _require(
        all(set(value).issubset(allowed_lineage_roles) for value in lineage.values()),
        "figure_inputs: resource lineage references an unknown authority",
    )
    selection_payload = json.loads(
        paths["train399_selection"].read_text(encoding="utf-8")
    )
    _require(
        payload.get("train399_selection_sha256")
        == sha256_file(paths["train399_selection"])
        and payload.get("train399_selection_identity_sha256")
        == selection_payload.get("selection_receipt_identity_sha256"),
        "figure_inputs: train399 selection authority mismatch",
    )
    resource_paths: dict[str, Path] = {}
    for role, record in resources.items():
        _require(isinstance(record, Mapping), f"figure_inputs/{role}: resource malformed")
        resource_paths[role] = _resolve_figure_input_file(
            path, record, f"resource:{role}"
        )
    _, narrative_payload = _read_json_object(
        resource_paths["narrative_decision"],
        "figure_inputs/narrative_decision",
    )
    try:
        narrative_decision = validate_narrative_decision(narrative_payload)
    except NarrativeDecisionError as error:
        raise EvidenceManifestError(
            f"figure_inputs: narrative decision is invalid: {error}"
        ) from error
    narrative_identity = narrative_decision[
        "narrative_decision_identity_sha256"
    ]
    _require(
        payload.get("narrative_decision_identity_sha256") == narrative_identity
        and payload.get("narrative_branch_id") == narrative_decision["branch_id"],
        "figure_inputs: narrative decision binding differs",
    )
    _, assignment = _read_json_object(
        resource_paths["qcdev_assignment"],
        "figure_inputs/qcdev_assignment",
    )
    assignment_identity = assignment.get("assignment_identity_sha256")
    unsigned_assignment = deepcopy(assignment)
    unsigned_assignment.pop("assignment_identity_sha256", None)
    _require(
        _is_sha256(assignment_identity)
        and sha256_json(unsigned_assignment) == assignment_identity
        and payload.get("qcdev_assignment_identity_sha256")
        == assignment_identity,
        "figure_inputs: QC-development assignment binding differs",
    )
    source_paths: dict[str, Path] = {}
    for role, record in source_inputs.items():
        _require(isinstance(record, Mapping), f"figure_inputs/{role}: source malformed")
        source_paths[role] = _resolve_figure_input_file(path, record, f"source:{role}")
    for role in WT_SECONDARY_RESOURCE_ROLES:
        _require(
            sha256_file(resource_paths[role]) == sha256_file(source_paths[role]),
            f"figure_inputs: WT resource/source bytes differ for {role}",
        )
    try:
        wt_secondary_evidence = validate_wt_secondary_evidence(
            contrasts=_read_csv_records(
                resource_paths["wt_within_experiment_contrasts"],
                "figure_inputs/WT contrasts",
            ),
            meta=_read_csv_records(
                resource_paths["wt_within_day_meta_analysis"],
                "figure_inputs/WT same-day meta-analysis",
            ),
            flow=_read_csv_records(
                resource_paths["wt_temperature_qc_flow"],
                "figure_inputs/WT model-QC flow",
            ),
        )
        wt_secondary_binding = validate_wt_secondary_analysis_binding(
            analysis_summary=json.loads(
                paths["analysis"].read_text(encoding="utf-8")
            ),
            evidence_summary=wt_secondary_evidence,
            table_sha256={
                role: sha256_file(source_paths[role])
                for role in WT_SECONDARY_RESOURCE_ROLES
            },
        )
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceManifestError(
            f"figure_inputs: WT secondary evidence validation failed: {error}"
        ) from error
    _require(
        payload.get("wt_secondary_evidence") == wt_secondary_binding,
        "figure_inputs: WT secondary evidence binding differs",
    )
    provenance_payloads: dict[str, dict[str, Any]] = {}
    for role, record in provenance.items():
        _require(isinstance(record, Mapping), f"figure_inputs/{role}: provenance malformed")
        receipt_path = _resolve_figure_input_file(path, record, f"provenance:{role}")
        _, receipt = _read_json_object(receipt_path, f"figure_inputs/{role}")
        identity_field = record.get("identity_field")
        _require(
            isinstance(identity_field, str)
            and receipt.get(identity_field) == record.get("identity_sha256"),
            f"figure_inputs/{role}: provenance logical identity differs",
        )
        _sealed_identity(receipt, identity_field, f"figure_inputs/{role}")
        provenance_payloads[role] = receipt

    assurance = provenance_payloads["measurement_assurance"]
    _require(
        assurance.get("schema_version")
        == "PHAxis-measurement-assurance-receipt-1.0"
        and assurance.get("status")
        == "completed_locked_qc_development_assurance"
        and assurance.get("scope")
        == "QC-development measurement assurance; non-independent"
        and assurance.get("independent_accuracy_claim_allowed") is False,
        "figure_inputs: measurement assurance scope changed",
    )
    table_hashes = assurance.get("source_table_sha256")
    _require(isinstance(table_hashes, Mapping), "figure_inputs: assurance table hashes missing")
    for short, role in (
        ("metrics", "assurance_metrics"),
        ("pairs", "assurance_pairs"),
        ("support", "assurance_support"),
        ("topology", "assurance_topology"),
    ):
        _require(
            table_hashes.get(short) == sha256_file(source_paths[role]),
            f"figure_inputs: assurance {short} source is not receipt-bound",
        )
    _require(
        sha256_file(source_paths["full_image_traits"])
        == json.loads(paths["traits"].read_text(encoding="utf-8")).get(
            "image_traits_sha256"
        ),
        "figure_inputs: canonical full image-trait table is not trait-receipt-bound",
    )
    _, trait_contract = _read_json_object(
        resource_paths["trait_contract"], "figure_inputs/trait_contract"
    )
    _, multitrait_atlas = _read_json_object(
        resource_paths["multitrait_atlas"], "figure_inputs/multitrait_atlas"
    )
    atlas_source_hashes = {
        "trait_contract": sha256_file(resource_paths["trait_contract"]),
        "canonical_image_traits": sha256_file(
            source_paths["full_image_traits"]
        ),
        **{
            role: sha256_file(source_paths[role])
            for role in (
                "clean_traits",
                "full_traits",
                "analysis_primary_table",
                "analysis_sensitivity_table",
            )
        },
    }
    try:
        validate_multitrait_atlas_structure(
            multitrait_atlas,
            trait_contract=trait_contract,
            expected_source_sha256=atlas_source_hashes,
        )
    except MultitraitAtlasError as error:
        raise EvidenceManifestError(
            f"figure_inputs: multitrait atlas validation failed: {error}"
        ) from error
    return {
        "assembly_identity_sha256": assembly_identity,
        "source_summary_sha256": expected_sources,
        "resource_sha256": {
            role: str(resources[role]["sha256"]) for role in FIGURE_RESOURCE_ROLES
        },
        "source_input_sha256": {
            role: str(source_inputs[role]["sha256"])
            for role in FIGURE_SOURCE_INPUT_ROLES
        },
        "provenance_identity_sha256": {
            role: str(provenance[role]["identity_sha256"])
            for role in FIGURE_PROVENANCE_ROLES
        },
        "multitrait_atlas_identity_sha256": str(
            multitrait_atlas["atlas_identity_sha256"]
        ),
        "wt_secondary_evidence": wt_secondary_binding,
        "supplementary_figure_contract_identity_sha256": str(
            expected_supplementary_contract["contract_identity_sha256"]
        ),
        "narrative_decision_identity_sha256": str(narrative_identity),
        "narrative_branch_id": str(narrative_decision["branch_id"]),
        "publication_title_contract": title_contract(narrative_decision),
        "qcdev_assignment_identity_sha256": str(assignment_identity),
    }


def _validate_proposal_gate_binding(
    proposal: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    payloads: Mapping[str, dict[str, Any]],
    stageb_binding: Mapping[str, Any],
) -> None:
    promotion = proposal["promotion"]
    expected_files = {
        role: sha256_file(paths[role])
        for role in (
            "train399_candidate",
            "train399_selection",
            "train399_evaluation",
            "root_exact283",
        )
    }
    _require(
        promotion.get("formal_gate_source_sha256") == expected_files,
        "model_contract_proposal: formal Gate source-file SHA binding mismatch",
    )
    expected_identities = {
        "candidate_bundle_identity_sha256": payloads["train399_candidate"].get(
            "candidate_bundle_identity_sha256"
        ),
        "selection_receipt_identity_sha256": payloads["train399_selection"].get(
            "selection_receipt_identity_sha256"
        ),
        "selected_model_metadata_identity_sha256": payloads["train399_evaluation"]
        .get("training_contract", {})
        .get("selected_model_metadata_identity_sha256"),
        "root_exact283_audit_identity_sha256": payloads["root_exact283"].get(
            "audit_identity_sha256"
        ),
    }
    _require(
        promotion.get("formal_gate_identity_sha256") == expected_identities,
        "model_contract_proposal: formal Gate logical-identity binding mismatch",
    )
    _require(
        promotion.get("stageb_binding") == stageb_binding,
        "model_contract_proposal: expert/checkpoint/threshold binding mismatch",
    )
    proposal_root = proposal.get("root_expert")
    root_receipt = payloads["root_exact283"]
    _require(
        isinstance(proposal_root, Mapping)
        and proposal_root.get("bundle_identity_sha256")
        == root_receipt.get("bundle_identity_sha256")
        and proposal_root.get("pipeline_identity_sha256")
        == root_receipt.get("pipeline_identity_sha256")
        and proposal_root.get("fresh_exact283_audit_identity_sha256")
        == root_receipt.get("audit_identity_sha256"),
        "model_contract_proposal: root-provider bundle/pipeline/audit differs from named exact283 receipt",
    )


def _selected_stageb_binding(payloads: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = payloads["train399_candidate"]
    selection = payloads["train399_selection"]
    evaluation = payloads["train399_evaluation"]
    stageb = payloads["stageb"]
    pending = candidate.get("detection_model_metadata")
    selected = stageb.get("detection_model_metadata")
    contract = evaluation.get("training_contract")
    chosen = selection.get("selected")
    _require(isinstance(pending, Mapping), "candidate: detection_model_metadata missing")
    _require(isinstance(selected, Mapping), "stageb: detection_model_metadata missing")
    _require(isinstance(contract, Mapping), "evaluation: training_contract missing")
    _require(isinstance(chosen, Mapping), "selection: selected operating point missing")

    expert = pending.get("expert_id")
    checkpoints = pending.get("checkpoint_sha256")
    threshold = chosen.get("threshold")
    _require(isinstance(expert, str) and bool(expert), "candidate: expert_id missing")
    _require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 5
        and len(set(checkpoints)) == 5
        and all(_is_sha256(value) for value in checkpoints),
        "candidate: checkpoints are not five distinct SHA-256 values",
    )
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold)),
        "selection: selected threshold is not finite",
    )
    expected = {
        "expert_id": expert,
        "checkpoint_sha256": checkpoints,
        "candidate_bundle_identity_sha256": candidate["candidate_bundle_identity_sha256"],
        "selection_receipt_identity_sha256": selection["selection_receipt_identity_sha256"],
    }
    for field, value in expected.items():
        _require(selected.get(field) == value, f"stageb: selected metadata mismatch: {field}")
    _require(
        selected.get("deployment_role") == "candidate_gate_passed_not_promoted"
        and selected.get("deployment_role") == pending.get("deployment_role"),
        "stageb: selected metadata must retain the non-promoting candidate role",
    )
    _require(
        selected.get("operating_point_status") == "selected_on_locked_QCdevelopment44",
        "stageb: operating point is not the locked QCdevelopment44 selection",
    )
    _require(
        math.isclose(
            float(selected.get("selected_score_threshold", float("nan"))),
            float(threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "stageb: selected threshold differs from selection receipt",
    )
    metadata_identity = selected.get("selected_model_metadata_identity_sha256")
    _require(_is_sha256(metadata_identity), "stageb: selected metadata identity missing")
    unsigned = deepcopy(dict(selected))
    unsigned.pop("precision_mode", None)
    unsigned.pop("selected_model_metadata_identity_sha256", None)
    _require(
        sha256_json(unsigned) == metadata_identity,
        "stageb: selected metadata logical identity is invalid",
    )
    _require(
        contract.get("selected_model_metadata_identity_sha256") == metadata_identity,
        "evaluation: selected metadata identity differs from StageB",
    )
    _require(contract.get("checkpoint_sha256") == checkpoints, "evaluation: checkpoint mismatch")
    _require(stageb.get("checkpoint_sha256") == checkpoints, "stageb: checkpoint mismatch")
    _require(
        math.isclose(float(stageb.get("score_threshold", float("nan"))), float(threshold), rel_tol=0.0, abs_tol=1e-12),
        "stageb: top-level threshold mismatch",
    )
    return {
        "expert_id": expert,
        "checkpoint_sha256": list(checkpoints),
        "selected_score_threshold": float(threshold),
        "candidate_bundle_identity_sha256": candidate["candidate_bundle_identity_sha256"],
        "selection_receipt_identity_sha256": selection["selection_receipt_identity_sha256"],
        "selected_model_metadata_identity_sha256": metadata_identity,
    }


def _prediction_map(fusion: Mapping[str, Any]) -> dict[str, str]:
    records = fusion.get("records")
    _require(isinstance(records, list) and len(records) == 283, "fusion: records must be exact283")
    result: dict[str, str] = {}
    for record in records:
        _require(isinstance(record, Mapping), "fusion: invalid record")
        task = record.get("task_id")
        digest = record.get("prediction_sha256")
        _require(isinstance(task, str) and task and task not in result, "fusion: duplicate/empty task_id")
        _require(_is_sha256(digest), f"fusion: invalid prediction SHA for {task}")
        result[task] = str(digest)
    return result


def _validate_chain(
    paths: Mapping[str, Path],
    payloads: Mapping[str, dict[str, Any]],
    binding: Mapping[str, Any],
    figure_assembly: Mapping[str, Any],
    evaluator_prediction_provenance: Mapping[str, Any],
) -> None:
    stageb, fusion, traits = (payloads[name] for name in ("stageb", "fusion", "traits"))
    cohorts, analysis, profiles, figure_inputs, figures = (
        payloads[name]
        for name in ("cohorts", "analysis", "profiles", "figure_inputs", "figures")
    )
    _require(stageb.get("status") == "completed" and stageb.get("images") == 283, "stageb: not completed exact283")
    _require(fusion.get("status") == "completed" and fusion.get("images") == 283, "fusion: not completed exact283")
    _require(traits.get("status") == "completed" and traits.get("tasks") == 283, "traits: not completed exact283")
    _require(
        fusion.get("source_stageb_summary_sha256") == sha256_file(paths["stageb"]),
        "fusion: source StageB summary SHA mismatch",
    )
    _require(
        fusion.get("hair_identity_count_expert") == binding["expert_id"],
        "fusion: expert differs from train399 selection",
    )
    predictions = _prediction_map(fusion)
    _require(traits.get("prediction_sha256") == predictions, "traits: prediction SHA map differs from fusion")
    _require(
        traits.get("hair_identity_count_expert") == binding["expert_id"],
        "traits: expert differs from train399 selection",
    )

    counts = cohorts.get("counts")
    directories = cohorts.get("cohort_directories")
    outputs = cohorts.get("output_sha256")
    inputs = cohorts.get("input_sha256")
    _require(isinstance(counts, Mapping), "cohorts: counts missing")
    _require(counts.get("biological_full") == 283, "cohorts: biological_full must be 283")
    _require(counts.get("biological_clean") == 261, "cohorts: biological_clean must be 261")
    _require(
        directories == {"primary": "primary_clean261", "sensitivity": "sensitivity_full283"},
        "cohorts: clean261/full283 directory identities changed",
    )
    _require(isinstance(inputs, Mapping) and inputs.get("trait_export_summary") == sha256_file(paths["traits"]), "cohorts: trait summary SHA mismatch")
    _require(isinstance(outputs, Mapping), "cohorts: output SHA map missing")
    primary = outputs.get("primary_clean261")
    _require(isinstance(primary, Mapping), "cohorts: primary_clean261 output hashes missing")

    _require(
        analysis.get("status") == "completed_exploratory_clean_primary_full_sensitivity",
        "analysis: final clean/full analysis status missing",
    )
    _require(analysis.get("primary_cohort") == "primary_clean261", "analysis: primary cohort is not clean261")
    _require(analysis.get("sensitivity_cohort") == "sensitivity_full283", "analysis: sensitivity cohort is not full283")
    _require(analysis.get("cohort_build_summary_sha256") == sha256_file(paths["cohorts"]), "analysis: cohort summary SHA mismatch")

    _require(profiles.get("status") == "completed" and profiles.get("tasks") == 261, "profiles: primary export must be completed clean261")
    _require(profiles.get("locked_1_4mm_trait_crosscheck_tasks") == 261, "profiles: clean261 crosscheck task count changed")
    _require(profiles.get("locked_1_4mm_trait_crosscheck_mismatches") == 0, "profiles: trait crosscheck mismatch")
    _require(profiles.get("traits_csv_sha256") == primary.get("traits"), "profiles: clean261 traits SHA mismatch")
    _require(profiles.get("hair_instances_csv_sha256") == primary.get("hair_instances"), "profiles: clean261 hair table SHA mismatch")

    _require(figures.get("status") == "final_sealed_strict_train399_only", "figures: not final strict train399-only")
    _require(figures.get("formal_train399_only_gate_passed") is True, "figures: formal train399 Gate not passed")
    _require(figures.get("deployment_figures_generated") is True, "figures: deployment figures absent")
    _require(figures.get("deployment_figures_provisional") is False, "figures: provisional deployment figures")
    _require(figures.get("submission_use_allowed") is True, "figures: submission use not allowed")
    _require(
        figures.get("figure_input_manifest_sha256")
        == sha256_file(paths["figure_inputs"])
        and figures.get("figure_input_assembly_identity_sha256")
        == figure_assembly["assembly_identity_sha256"],
        "figures: production figure-input assembly binding mismatch",
    )
    _require(
        figures.get("multitrait_atlas_identity_sha256")
        == figure_assembly.get("multitrait_atlas_identity_sha256"),
        "figures: multitrait atlas identity differs from figure-input evidence",
    )
    _require(
        figures.get("figure_resource_sha256")
        == figure_assembly.get("resource_sha256")
        and figures.get("wt_secondary_evidence")
        == figure_assembly.get("wt_secondary_evidence")
        and figures.get("claim_contract", {}).get(
            "wt_secondary_alters_D15_fixed_effect_family"
        )
        is False
        and figures.get("claim_contract", {}).get(
            "wt_cross_day_pooling_performed"
        )
        is False
        and figures.get("claim_contract", {}).get(
            "wt_unknown_day_meta_analysis_performed"
        )
        is False,
        "figures: WT secondary evidence/independence binding differs",
    )
    _require(
        figures.get("narrative_decision_identity_sha256")
        == figure_assembly.get("narrative_decision_identity_sha256")
        and figures.get("narrative_branch_id")
        == figure_assembly.get("narrative_branch_id")
        and figures.get("title_contract")
        == figure_assembly.get("publication_title_contract")
        and figures.get("claim_contract", {}).get(
            "narrative_decision_identity_sha256"
        )
        == figure_assembly.get("narrative_decision_identity_sha256")
        and figures.get("claim_contract", {}).get(
            "profiles_select_or_veto_narrative_branch"
        )
        is False,
        "figures: narrative decision/title authority differs from stage36",
    )
    supplementary = figures.get("supplementary_figure_bundle_sha256")
    supplementary_identity = figures.get(
        "supplementary_figure_bundle_identity_sha256"
    )
    supplementary_records = figures.get("supplementary_figures")
    expected_supplementary_contract = supplementary_figure_contract()
    expected_supplementary_contract["contract_identity_sha256"] = sha256_json(
        expected_supplementary_contract
    )
    _require(
        isinstance(supplementary, Mapping)
        and list(supplementary) == list(SUPPLEMENTARY_FIGURE_STEMS)
        and all(
            isinstance(supplementary.get(stem), Mapping)
            for stem in SUPPLEMENTARY_FIGURE_STEMS
        )
        and supplementary_identity == sha256_json(supplementary)
        and isinstance(supplementary_records, Mapping)
        and list(supplementary_records) == list(SUPPLEMENTARY_FIGURE_STEMS)
        and all(
            isinstance(record, Mapping)
            and record.get("number") == f"S{index}"
            and record.get("status") == "final"
            and record.get("title")
            == expected_supplementary_contract["figures"][index - 1]["title"]
            and record.get("resource_roles")
            == expected_supplementary_contract["figures"][index - 1][
                "resource_roles"
            ]
            and record.get("receipt_roles")
            == expected_supplementary_contract["figures"][index - 1][
                "receipt_roles"
            ]
            and record.get("receipt_file_sha256")
            == {
                role: sha256_file(paths[role])
                for role in expected_supplementary_contract["figures"][
                    index - 1
                ]["receipt_roles"]
            }
            and record.get("source_data_sha256")
            == supplementary[stem].get("source_data")
            for index, (stem, record) in enumerate(
                supplementary_records.items(), start=1
            )
        )
        and figures.get("supplementary_figure_contract")
        == expected_supplementary_contract
        and figures.get("supplementary_figure_contract_identity_sha256")
        == figure_assembly.get(
            "supplementary_figure_contract_identity_sha256"
        )
        and figures.get("claim_contract", {}).get("main_figure_count") == 6
        and figures.get("claim_contract", {}).get("supplementary_figure_count")
        == 9,
        "figures: ordered supplementary S1--S9 receipts are not hash-closed",
    )
    table_receipt_relative = figures.get("supplementary_table_bundle_receipt")
    _require(
        isinstance(table_receipt_relative, str)
        and Path(table_receipt_relative).name == SUPPLEMENTARY_TABLE_RECEIPT,
        "figures: supplementary table/data receipt path is invalid",
    )
    table_receipt_path = (paths["figures"].parent / table_receipt_relative).resolve()
    _require(
        table_receipt_path.is_relative_to(paths["figures"].parent.resolve()),
        "figures: supplementary table/data receipt escapes figure suite",
    )
    try:
        table_bundle = validate_supplementary_table_data_bundle(
            table_receipt_path, require_final=True
        )
    except SupplementaryTableError as error:
        raise EvidenceManifestError(
            f"figures: supplementary Table/Data S1--S10 validation failed: {error}"
        ) from error
    expected_table_sources = {
        role: str(figure_inputs["source_inputs"][role]["sha256"])
        for role in FIGURE_SOURCE_INPUT_ROLES
    }
    authority_candidates: dict[str, str] = {
        **{
            f"source/{role}": str(record["sha256"])
            for role, record in figure_inputs["source_inputs"].items()
        },
        **{
            f"resource/{role}": str(record["sha256"])
            for role, record in figure_inputs["resources"].items()
        },
        **{
            f"receipt/{role}": sha256_file(paths[role])
            for role in (
                "train399_evaluation",
                "root_exact283",
                "stageb",
                "fusion",
                "traits",
                "cohorts",
                "analysis",
                "profiles",
            )
        },
        "proposal/model_contract_proposal": sha256_file(
            paths["model_contract_proposal"]
        ),
    }
    table_authorities = table_bundle["source_authority_sha256"]
    _require(
        set(table_authorities) <= set(authority_candidates),
        "figures: supplementary Table/Data names an unrouteable authority",
    )
    expected_table_authorities = {
        role: authority_candidates[role] for role in table_authorities
    }
    _require(
        figures.get("supplementary_tables") == table_bundle["items"]
        and list(figures["supplementary_tables"]) == list(SUPPLEMENTARY_TABLE_STEMS)
        and figures.get("supplementary_table_bundle_receipt_sha256")
        == table_bundle["receipt_sha256"]
        and figures.get("supplementary_table_bundle_identity_sha256")
        == table_bundle["bundle_identity_sha256"]
        and figures.get("supplementary_table_bundle_sha256")
        == table_bundle["bundle_file_sha256"]
        and figures.get("supplementary_table_source_input_sha256")
        == expected_table_sources
        and figures.get("supplementary_table_source_authority_sha256")
        == expected_table_authorities
        and table_authorities == expected_table_authorities
        and figures.get("supplementary_table_source_authority_identity")
        == table_bundle["source_authority_identity"]
        and table_bundle.get("figure_input_manifest_sha256")
        == sha256_file(paths["figure_inputs"])
        and table_bundle.get("figure_input_assembly_identity_sha256")
        == figure_assembly["assembly_identity_sha256"]
        and table_bundle.get("model_contract_proposal_identity_sha256")
        == figures.get("model_contract_proposal_identity_sha256")
        and figures.get("claim_contract", {}).get(
            "supplementary_table_data_resource_count"
        )
        == 10,
        "figures: supplementary Table/Data S1--S10 hash/denominator closure differs",
    )
    _require(
        figures.get("train399_prediction_input_provenance")
        == figure_inputs.get("train399_prediction_input_provenance")
        and figure_inputs.get("train399_prediction_input_provenance", {}).get(
            "task_order_identity_sha256"
        )
        == evaluator_prediction_provenance["task_order_identity_sha256"]
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["ordered_file_set_identity_sha256"]
        == evaluator_prediction_provenance[
            "stageb_detection_set_identity_sha256"
        ]
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["schema_version"]
        == evaluator_prediction_provenance["stageb_detection_schema_version"]
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["artifact_role"]
        == evaluator_prediction_provenance["stageb_detection_artifact_role"]
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["production_consumption_allowed"]
        is False
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["fusion_consumption_allowed"]
        is False
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["traits_consumption_allowed"]
        is False
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["evaluation_inference_summary_sha256"]
        == evaluator_prediction_provenance[
            "stageb_evaluation_inference_summary_sha256"
        ]
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["evaluation_inference_summary_identity_sha256"]
        == evaluator_prediction_provenance[
            "stageb_evaluation_inference_summary_identity_sha256"
        ]
        and figure_inputs["train399_prediction_input_provenance"][
            "stageb_train399"
        ]["evaluation_gate_identity_sha256"]
        == evaluator_prediction_provenance[
            "stageb_evaluation_gate_identity_sha256"
        ]
        and figure_inputs["train399_prediction_input_provenance"][
            "legacy_hybrid_endpoint_complete_identity_layer"
        ]["ordered_file_set_identity_sha256"]
        == evaluator_prediction_provenance[
            "legacy_hybrid_prediction_set_identity_sha256"
        ],
        "figures: QC44 prediction authority differs from evaluator/assembly",
    )
    _require(
        figures.get("model_contract_public_identity")
        == figure_inputs.get("model_contract_public_identity"),
        "figures: public model identities differ from figure input assembly",
    )
    public_identity = figures["model_contract_public_identity"]
    _require(
        figures.get("model_bundle_id") == public_identity.get("model_bundle_id")
        and figures.get("root_expert_id") == public_identity.get("root_expert_id")
        and figures.get("hair_identity_expert_id") == binding["expert_id"],
        "figures: top-level public expert/model identities changed",
    )
    source_bindings = figures.get("source_summary_sha256")
    _require(isinstance(source_bindings, Mapping), "figures: source_summary_sha256 map missing")
    expected_sources = {
        "train399_evaluation": sha256_file(paths["train399_evaluation"]),
        "root_exact283": sha256_file(paths["root_exact283"]),
        **{role: sha256_file(paths[role]) for role in ("stageb", "fusion", "traits", "cohorts", "analysis", "profiles")},
    }
    _require(dict(source_bindings) == expected_sources, "figures: upstream summary SHA closure mismatch")
    bundles = figures.get("figure_bundle_sha256")
    _require(isinstance(bundles, Mapping) and bool(bundles), "figures: figure bundle hashes missing")
    declared = figures.get("figures")
    _require(isinstance(declared, Mapping) and set(declared) == set(bundles), "figures: figure identities disagree")
    for path, value in _walk(bundles):
        if not isinstance(value, (Mapping, list)):
            _require(_is_sha256(value), f"figures: invalid bundle SHA at {path}")
    expected_figure_identity = sha256_json(
        figure_suite_identity_preimage(
            status="final",
            figure_hashes=bundles,
            source_hashes=source_bindings,
            figure_input_assembly_identity_sha256=figure_assembly[
                "assembly_identity_sha256"
            ],
            model_contract_proposal_identity_sha256=figures[
                "model_contract_proposal_identity_sha256"
            ],
            model_contract_public_identity=figures[
                "model_contract_public_identity"
            ],
            train399_prediction_input_provenance=figures[
                "train399_prediction_input_provenance"
            ],
            supplementary_table_bundle_identity_sha256=figures[
                "supplementary_table_bundle_identity_sha256"
            ],
            supplementary_table_bundle_receipt_sha256=figures[
                "supplementary_table_bundle_receipt_sha256"
            ],
        )
    )
    _require(
        figures.get("figure_suite_identity_sha256") == expected_figure_identity,
        "figures: figure suite logical identity mismatch",
    )


def _declared_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if _is_sha256(value) and ("sha256" in path or "identity" in leaf):
            result[path] = str(value)
    return result


def _figure_table_identities(payloads: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    keywords = ("figure", "table", "traits", "instances", "profile", "output_sha256", "legend")
    result: dict[str, dict[str, str]] = {}
    for role in (
        "traits",
        "cohorts",
        "analysis",
        "profiles",
        "figure_inputs",
        "figures",
    ):
        selected = {
            path: value
            for path, value in _declared_hashes(payloads[role]).items()
            if any(keyword in path.casefold() for keyword in keywords)
        }
        _require(selected, f"{role}: no declared figure/table SHA identities")
        result[role] = selected
    return result


def build_manuscript_evidence_manifest(
    *,
    model_contract_proposal: str | Path,
    train399_candidate: str | Path,
    train399_selection: str | Path,
    train399_evaluation: str | Path,
    root_exact283: str | Path,
    stageb_summary: str | Path,
    fusion_summary: str | Path,
    traits_summary: str | Path,
    cohorts_summary: str | Path,
    analysis_summary: str | Path,
    profiles_summary: str | Path,
    figure_inputs: str | Path,
    figures_summary: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate all evidence and atomically create one deterministic manifest."""

    destination = Path(output).resolve()
    _require(not destination.exists(), f"output already exists: {destination}")
    arguments = {
        "model_contract_proposal": model_contract_proposal,
        "train399_candidate": train399_candidate,
        "train399_selection": train399_selection,
        "train399_evaluation": train399_evaluation,
        "root_exact283": root_exact283,
        "stageb": stageb_summary,
        "fusion": fusion_summary,
        "traits": traits_summary,
        "cohorts": cohorts_summary,
        "analysis": analysis_summary,
        "profiles": profiles_summary,
        "figure_inputs": figure_inputs,
        "figures": figures_summary,
    }
    paths: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        paths[role], payloads[role] = _read_json_object(arguments[role], role)
    _require(len(set(paths.values())) == len(paths), "each evidence role must name a distinct file")
    _validate_formal_gate_receipts(paths, payloads)
    evaluator_prediction_provenance = _validate_evaluator12(
        payloads["train399_evaluation"]
    )
    proposal_identity, _proposal_status = _validate_model_contract_proposal(
        payloads["model_contract_proposal"]
    )
    derived_public_identity = validate_proposal_public_identity(
        payloads["model_contract_proposal"]
    )
    proposal_sha = sha256_file(paths["model_contract_proposal"])
    for role in SUMMARY_ROLES:
        _guard_final_summary(role, payloads[role])
        _require_proposal_binding(
            role,
            payloads[role],
            proposal_sha256=proposal_sha,
            proposal_identity_sha256=proposal_identity,
        )
    for role in ("stageb", "fusion", "traits", "cohorts", "analysis", "profiles"):
        _sealed_identity(payloads[role], IDENTITY_FIELDS[role], role)
    for role, root_field in {
        "stageb": "root_expert_id",
        "fusion": "root_expert",
        "traits": "root_expert_id",
        "cohorts": "root_expert_id",
        "analysis": "root_expert_id",
        "profiles": "root_expert_id",
    }.items():
        _require(
            payloads[role].get("model_bundle_id")
            == derived_public_identity["model_bundle_id"]
            and payloads[role].get(root_field)
            == derived_public_identity["root_expert_id"],
            f"{role}: public model/root identity differs from proposal",
        )
    binding = _selected_stageb_binding(payloads)
    _validate_proposal_gate_binding(
        payloads["model_contract_proposal"],
        paths=paths,
        payloads=payloads,
        stageb_binding=binding,
    )
    figure_assembly = _validate_figure_input_assembly(
        path=paths["figure_inputs"],
        payload=payloads["figure_inputs"],
        proposal=payloads["model_contract_proposal"],
        proposal_sha256=proposal_sha,
        proposal_identity_sha256=proposal_identity,
        paths=paths,
    )
    _validate_chain(
        paths,
        payloads,
        binding,
        figure_assembly,
        evaluator_prediction_provenance,
    )

    artifacts: dict[str, Any] = {}
    for role in ROLE_ORDER:
        identity_field = IDENTITY_FIELDS.get(role)
        identities = _declared_hashes(payloads[role])
        artifact: dict[str, Any] = {
            "source_file_sha256": sha256_file(paths[role]),
            "schema_version": payloads[role].get("schema_version"),
            "status": payloads[role].get("status"),
            "declared_sha256_identities": identities,
        }
        if identity_field:
            artifact["primary_identity_field"] = identity_field
            artifact["primary_identity_sha256"] = payloads[role][identity_field]
        artifacts[role] = artifact
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_formal_evidence_graph",
        "formal_release_evidence_closed": True,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "cohort_identities": {"full": "full283", "primary": "clean261"},
        "model_contract_proposal_sha256": proposal_sha,
        "model_contract_proposal_identity_sha256": proposal_identity,
        "model_contract_public_identity": {
            "model_bundle_id": derived_public_identity["model_bundle_id"],
            "root_expert_id": derived_public_identity["root_expert_id"],
            "root_provider_role": derived_public_identity["root_provider_role"],
        },
        "model_bundle_id": derived_public_identity["model_bundle_id"],
        "root_expert_id": derived_public_identity["root_expert_id"],
        "hair_identity_expert_id": binding["expert_id"],
        "stageb_binding": binding,
        "train399_prediction_input_provenance": evaluator_prediction_provenance,
        "figure_input_assembly": figure_assembly,
        "supplementary_table_data": {
            "ordered_item_count": 10,
            "bundle_receipt_sha256": payloads["figures"][
                "supplementary_table_bundle_receipt_sha256"
            ],
            "bundle_identity_sha256": payloads["figures"][
                "supplementary_table_bundle_identity_sha256"
            ],
            "source_authority_sha256": payloads["figures"][
                "supplementary_table_source_authority_sha256"
            ],
            "ordered_item_identity_sha256": {
                stem: payloads["figures"]["supplementary_tables"][stem][
                    "item_identity_sha256"
                ]
                for stem in SUPPLEMENTARY_TABLE_STEMS
            },
        },
        "artifacts": artifacts,
        "figure_table_identities": _figure_table_identities(payloads),
    }
    manifest["manifest_identity_sha256"] = sha256_json(manifest)

    # Repeat immediately before publication to close the validation-time race.
    _require(not destination.exists(), f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-contract-proposal", required=True)
    parser.add_argument("--train399-candidate", required=True)
    parser.add_argument("--train399-selection", required=True)
    parser.add_argument("--train399-evaluation", required=True)
    parser.add_argument("--root-exact283", required=True)
    parser.add_argument("--figure-inputs", required=True)
    for name in ("stageb", "fusion", "traits", "cohorts", "analysis", "profiles", "figures"):
        parser.add_argument(f"--{name}-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manuscript_evidence_manifest(
        model_contract_proposal=args.model_contract_proposal,
        train399_candidate=args.train399_candidate,
        train399_selection=args.train399_selection,
        train399_evaluation=args.train399_evaluation,
        root_exact283=args.root_exact283,
        stageb_summary=args.stageb_summary,
        fusion_summary=args.fusion_summary,
        traits_summary=args.traits_summary,
        cohorts_summary=args.cohorts_summary,
        analysis_summary=args.analysis_summary,
        profiles_summary=args.profiles_summary,
        figure_inputs=args.figure_inputs,
        figures_summary=args.figures_summary,
        output=args.output,
    )
    print(manifest["manifest_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
