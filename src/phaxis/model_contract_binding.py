"""Fail-closed PHAxis model-contract authority binding.

This module deliberately contains no training, image, torch, or CUDA imports.
Promotion uses :func:`read_model_contract_proposal`, which accepts only the
immutable, unapplied proposal.  Runtime consumers use
:func:`read_model_contract_authority`, which additionally accepts the official
contract produced by the atomic apply operation while retaining the original
proposal receipt as the downstream provenance authority.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ContractError
from .io import read_json, sha256_file, sha256_json
from .public_identity import (
    MODEL_BUNDLE_PREFIX,
    ROOT_EXPERT_PREFIX,
    ROOT_PROVIDER_ROLE,
    validate_proposal_public_identity,
)


MODEL_CONTRACT_PROPOSAL_SCHEMA = "PHAxis-model-contract-1.0.0"
MODEL_CONTRACT_PROMOTION_SCHEMA = "PHAxis-model-contract-promotion-1.0"
PENDING_CANDIDATE_ROLE = "candidate_gate_passed_not_promoted"
UNAPPLIED_PROPOSAL_LIFECYCLE = "unapplied_proposal"
APPLIED_OFFICIAL_LIFECYCLE = "applied_official"
RUN_SCOPED_AUTHORITY_PIN_SCHEMA = (
    "PHAxis-run-scoped-model-contract-authority-pin-1.0"
)
RUN_SCOPED_AUTHORITY_PIN_STATUS = (
    "sealed_unapplied_proposal_for_production"
)
RUN_SCOPED_AUTHORITY_PIN_LIFECYCLE = "run_scoped_unapplied_proposal_pin"

_APPLY_ONLY_PROMOTION_FIELDS = (
    "proposal_file_sha256",
    "proposal_identity_sha256",
    "expected_source_model_contract_sha256",
    "final_receipt_source_sha256",
    "final_receipt_identity_sha256",
    "final_receipt_public_identity",
)
_FINAL_RECEIPT_ROLES = ("stageb", "fusion", "traits", "evidence")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sealed_identity(payload: Mapping[str, Any], field: str, *, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    if not _is_sha256(observed):
        raise ContractError(f"{role}: {field} is absent or invalid")
    expected = sha256_json(unsigned)
    if str(observed).casefold() != expected.casefold():
        raise ContractError(f"{role}: {field} does not seal the complete JSON object")
    return expected


@dataclass(frozen=True)
class ModelContractProposalBinding:
    """Validated proposal receipt plus the currently read authority identity.

    ``file_sha256`` and ``identity_sha256`` always name the original unapplied
    proposal.  For an applied official contract, ``authority_*`` names the
    official file that was actually read.  This distinction prevents runtime
    outputs from silently changing their proposal receipt after promotion.
    """

    path: Path
    file_sha256: str
    identity_sha256: str
    stageb_binding: Mapping[str, Any]
    model_bundle_id: str
    root_expert_id: str
    root_provider_role: str
    root_bundle_identity_sha256: str
    root_pipeline_identity_sha256: str
    root_audit_identity_sha256: str
    authority_file_sha256: str
    authority_identity_sha256: str
    authority_lifecycle: str

    def receipt_fields(self) -> dict[str, str]:
        return {
            "model_contract_proposal_sha256": self.file_sha256,
            "model_contract_proposal_identity_sha256": self.identity_sha256,
        }

    def public_identity_fields(self) -> dict[str, str]:
        """Return proposal-owned public IDs for final predictions and tables."""

        return {
            "model_bundle_id": self.model_bundle_id,
            "root_expert_id": self.root_expert_id,
        }

    def output_identity_fields(self) -> dict[str, str]:
        """Return the complete proposal and public identity carried downstream."""

        return {**self.receipt_fields(), **self.public_identity_fields()}


def _pretty_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the exact deterministic JSON representation used by promotion."""

    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validate_model_contract_content(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate contract content shared by proposal and official lifecycles."""

    if payload.get("schema_version") != MODEL_CONTRACT_PROPOSAL_SCHEMA:
        raise ContractError("unsupported PHAxis model-contract proposal schema")
    if payload.get("product") != "PHAxis" or payload.get("product_version") != "1.0.0":
        raise ContractError("model-contract proposal product/version mismatch")
    promotion = payload.get("promotion")
    if not isinstance(promotion, Mapping):
        raise ContractError("model-contract proposal promotion block is absent")
    if promotion.get("schema_version") != MODEL_CONTRACT_PROMOTION_SCHEMA:
        raise ContractError("model-contract proposal promotion schema changed")
    red_lines = payload.get("red_lines")
    if not isinstance(red_lines, Mapping):
        raise ContractError("model-contract proposal red_lines block is absent")
    expected_guards = {
        "blind_images_used": 0,
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
    }
    for field, expected in expected_guards.items():
        if red_lines.get(field) != expected:
            raise ContractError(f"model-contract proposal red line changed: {field}")
    stageb_binding = promotion.get("stageb_binding")
    if not isinstance(stageb_binding, Mapping):
        raise ContractError("model-contract proposal Stage-B binding is absent")
    checkpoints = stageb_binding.get("checkpoint_sha256")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 5
        or len(set(checkpoints)) != 5
        or not all(_is_sha256(value) for value in checkpoints)
    ):
        raise ContractError("proposal Stage-B checkpoints are not five distinct SHA-256 values")
    threshold = stageb_binding.get("selected_score_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
    ):
        raise ContractError("proposal Stage-B score threshold is not finite")
    for field in (
        "expert_id",
        "candidate_bundle_identity_sha256",
        "selection_receipt_identity_sha256",
        "selected_model_metadata_identity_sha256",
    ):
        value = stageb_binding.get(field)
        if field == "expert_id":
            valid = isinstance(value, str) and bool(value.strip())
        else:
            valid = _is_sha256(value)
        if not valid:
            raise ContractError(f"proposal Stage-B binding field is invalid: {field}")
    formal_sources = promotion.get("formal_gate_source_sha256")
    if not isinstance(formal_sources, Mapping) or any(
        not _is_sha256(formal_sources.get(role))
        for role in (
            "train399_candidate",
            "train399_selection",
            "train399_evaluation",
            "root_exact283",
        )
    ):
        raise ContractError("proposal formal Gate source-file hashes are incomplete")
    formal_identities = promotion.get("formal_gate_identity_sha256")
    if not isinstance(formal_identities, Mapping) or any(
        not _is_sha256(formal_identities.get(field))
        for field in (
            "candidate_bundle_identity_sha256",
            "selection_receipt_identity_sha256",
            "selected_model_metadata_identity_sha256",
            "root_exact283_audit_identity_sha256",
        )
    ):
        raise ContractError("proposal formal Gate logical identities are incomplete")
    for field in (
        "candidate_bundle_identity_sha256",
        "selection_receipt_identity_sha256",
        "selected_model_metadata_identity_sha256",
    ):
        if formal_identities.get(field) != stageb_binding.get(field):
            raise ContractError(f"proposal Stage-B/formal Gate authority mismatch: {field}")
    checkpoint_authority = promotion.get("checkpoint_file_sha256_in_member_order")
    if checkpoint_authority is not None and checkpoint_authority != checkpoints:
        raise ContractError("proposal Stage-B checkpoint authority/order mismatch")

    derived_public_identity = validate_proposal_public_identity(payload)
    model_bundle_id = payload.get("model_bundle_id")
    root_expert = payload.get("root_expert")
    root_expert_id = root_expert.get("expert_id") if isinstance(root_expert, Mapping) else None
    root_provider_role = (
        root_expert.get("provider_role") if isinstance(root_expert, Mapping) else None
    )
    root_bundle_identity = (
        root_expert.get("bundle_identity_sha256")
        if isinstance(root_expert, Mapping)
        else None
    )
    root_pipeline_identity = (
        root_expert.get("pipeline_identity_sha256")
        if isinstance(root_expert, Mapping)
        else None
    )
    root_audit_identity = formal_identities.get("root_exact283_audit_identity_sha256")
    expert_boundary = payload.get("expert_boundary")
    if (
        not isinstance(model_bundle_id, str)
        or not model_bundle_id.startswith(MODEL_BUNDLE_PREFIX)
        or not isinstance(root_expert_id, str)
        or not root_expert_id.startswith(ROOT_EXPERT_PREFIX)
        or root_provider_role != ROOT_PROVIDER_ROLE
        or model_bundle_id != derived_public_identity["model_bundle_id"]
        or root_expert_id != derived_public_identity["root_expert_id"]
        or not _is_sha256(root_bundle_identity)
        or not _is_sha256(root_pipeline_identity)
        or not _is_sha256(root_audit_identity)
        or not isinstance(expert_boundary, Mapping)
        or expert_boundary.get("hair_identity_and_count")
        != stageb_binding.get("expert_id")
    ):
        raise ContractError("model-contract proposal public model/root identity is invalid")
    return {
        "promotion": promotion,
        "stageb_binding": stageb_binding,
        "checkpoints": checkpoints,
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_expert_id,
        "root_provider_role": root_provider_role,
        "root_bundle_identity_sha256": root_bundle_identity,
        "root_pipeline_identity_sha256": root_pipeline_identity,
        "root_audit_identity_sha256": root_audit_identity,
    }


def _binding(
    *,
    path: Path,
    authority_file_sha256: str,
    authority_identity_sha256: str,
    authority_lifecycle: str,
    proposal_file_sha256: str,
    proposal_identity_sha256: str,
    validated: Mapping[str, Any],
) -> ModelContractProposalBinding:
    return ModelContractProposalBinding(
        path=path,
        file_sha256=proposal_file_sha256,
        identity_sha256=proposal_identity_sha256,
        stageb_binding=deepcopy(dict(validated["stageb_binding"])),
        model_bundle_id=str(validated["model_bundle_id"]),
        root_expert_id=str(validated["root_expert_id"]),
        root_provider_role=str(validated["root_provider_role"]),
        root_bundle_identity_sha256=str(validated["root_bundle_identity_sha256"]),
        root_pipeline_identity_sha256=str(validated["root_pipeline_identity_sha256"]),
        root_audit_identity_sha256=str(validated["root_audit_identity_sha256"]),
        authority_file_sha256=authority_file_sha256,
        authority_identity_sha256=authority_identity_sha256,
        authority_lifecycle=authority_lifecycle,
    )


def build_run_scoped_authority_pin(
    proposal: str | Path,
    *,
    pin_path: str | Path,
    run_id: str,
    release_manifest_sha256: str,
    release_plan_identity_sha256: str,
) -> dict[str, Any]:
    """Return a sealed run-local pointer to an unapplied proposal.

    The pin is deliberately not an official model contract.  It lets
    production readers share one immutable proposal while the official
    compare-and-swap target remains untouched until the final release Gate.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ContractError("run-scoped authority pin requires a non-empty run_id")
    for value, field in (
        (release_manifest_sha256, "release_manifest_sha256"),
        (release_plan_identity_sha256, "release_plan_identity_sha256"),
    ):
        if not _is_sha256(value):
            raise ContractError(f"run-scoped authority pin {field} is invalid")
    destination = Path(pin_path).resolve()
    binding = read_model_contract_proposal(proposal)
    relative = os.path.relpath(binding.path, destination.parent).replace("\\", "/")
    payload: dict[str, Any] = {
        "schema_version": RUN_SCOPED_AUTHORITY_PIN_SCHEMA,
        "status": RUN_SCOPED_AUTHORITY_PIN_STATUS,
        "artifact_role": (
            "run_scoped_production_authority_pin_not_official_contract"
        ),
        "run_id": run_id,
        "release_manifest_sha256": release_manifest_sha256,
        "release_plan_identity_sha256": release_plan_identity_sha256,
        "proposal_path": relative,
        "proposal_file_sha256": binding.file_sha256,
        "proposal_identity_sha256": binding.identity_sha256,
        "model_bundle_id": binding.model_bundle_id,
        "root_expert_id": binding.root_expert_id,
        "root_provider_role": binding.root_provider_role,
        "root_bundle_identity_sha256": binding.root_bundle_identity_sha256,
        "official_apply_performed": False,
        "official_model_contract_modified": False,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    payload["authority_pin_identity_sha256"] = sha256_json(payload)
    return payload


def _read_run_scoped_authority_pin(
    pin_path: Path,
    payload: Mapping[str, Any],
    *,
    authority_file_sha256: str,
) -> ModelContractProposalBinding:
    if payload.get("status") != RUN_SCOPED_AUTHORITY_PIN_STATUS:
        raise ContractError("run-scoped authority pin status changed")
    expected_guards = {
        "artifact_role": "run_scoped_production_authority_pin_not_official_contract",
        "official_apply_performed": False,
        "official_model_contract_modified": False,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    for field, expected in expected_guards.items():
        if payload.get(field) != expected:
            raise ContractError(f"run-scoped authority pin guard changed: {field}")
    for field in (
        "release_manifest_sha256",
        "release_plan_identity_sha256",
        "proposal_file_sha256",
        "proposal_identity_sha256",
        "root_bundle_identity_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise ContractError(f"run-scoped authority pin {field} is invalid")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ContractError("run-scoped authority pin run_id is invalid")
    pin_identity = _sealed_identity(
        payload,
        "authority_pin_identity_sha256",
        role="run-scoped authority pin",
    )
    proposal_value = payload.get("proposal_path")
    if not isinstance(proposal_value, str) or not proposal_value.strip():
        raise ContractError("run-scoped authority pin proposal_path is invalid")
    proposal_path = Path(proposal_value)
    if not proposal_path.is_absolute():
        proposal_path = pin_path.parent / proposal_path
    proposal_binding = read_model_contract_proposal(
        proposal_path.resolve(),
        expected_file_sha256=str(payload["proposal_file_sha256"]),
    )
    expected = {
        "proposal_identity_sha256": proposal_binding.identity_sha256,
        "model_bundle_id": proposal_binding.model_bundle_id,
        "root_expert_id": proposal_binding.root_expert_id,
        "root_provider_role": proposal_binding.root_provider_role,
        "root_bundle_identity_sha256": (
            proposal_binding.root_bundle_identity_sha256
        ),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ContractError(
                f"run-scoped authority pin/proposal mismatch: {field}"
            )
    return ModelContractProposalBinding(
        path=proposal_binding.path,
        file_sha256=proposal_binding.file_sha256,
        identity_sha256=proposal_binding.identity_sha256,
        stageb_binding=proposal_binding.stageb_binding,
        model_bundle_id=proposal_binding.model_bundle_id,
        root_expert_id=proposal_binding.root_expert_id,
        root_provider_role=proposal_binding.root_provider_role,
        root_bundle_identity_sha256=(
            proposal_binding.root_bundle_identity_sha256
        ),
        root_pipeline_identity_sha256=(
            proposal_binding.root_pipeline_identity_sha256
        ),
        root_audit_identity_sha256=proposal_binding.root_audit_identity_sha256,
        authority_file_sha256=authority_file_sha256,
        authority_identity_sha256=pin_identity,
        authority_lifecycle=RUN_SCOPED_AUTHORITY_PIN_LIFECYCLE,
    )


def read_model_contract_proposal(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> ModelContractProposalBinding:
    """Validate an unapplied, sealed proposal and return its immutable binding."""

    proposal_path = Path(path).resolve()
    file_sha256 = sha256_file(proposal_path)
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256.casefold():
        raise ContractError("model-contract proposal file SHA mismatch")
    payload = read_json(proposal_path)
    validated = _validate_model_contract_content(payload)
    if payload.get("formal_release_status") != "passed_proposal_not_official":
        raise ContractError("model-contract proposal is not the passed, unapplied candidate")
    promotion = validated["promotion"]
    if (
        promotion.get("status") != "validated_proposal_not_applied"
        or promotion.get("official_apply_performed") is not False
    ):
        raise ContractError("model-contract proposal promotion guard changed")
    identity = _sealed_identity(
        payload,
        "model_contract_identity_sha256",
        role="model-contract proposal",
    )
    return _binding(
        path=proposal_path,
        authority_file_sha256=file_sha256,
        authority_identity_sha256=identity,
        authority_lifecycle=UNAPPLIED_PROPOSAL_LIFECYCLE,
        proposal_file_sha256=file_sha256,
        proposal_identity_sha256=identity,
        validated=validated,
    )


def _reconstruct_unapplied_proposal(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reverse only the fields changed by the atomic promotion operation."""

    proposal = deepcopy(dict(payload))
    proposal.pop("model_contract_identity_sha256", None)
    proposal["formal_release_status"] = "passed_proposal_not_official"
    hair = proposal.get("hair_identity_count_expert")
    if not isinstance(hair, dict):
        raise ContractError("applied model contract has no Stage-B hair authority")
    hair["current_checkpoint_role"] = "formal_train399_only_deployment_candidate"
    hair["strict_train399_only_retraining_gate"] = "passed_proposal_not_official"
    promotion = proposal.get("promotion")
    if not isinstance(promotion, dict):
        raise ContractError("applied model contract promotion block is absent")
    promotion["status"] = "validated_proposal_not_applied"
    promotion["official_apply_performed"] = False
    for field in _APPLY_ONLY_PROMOTION_FIELDS:
        promotion.pop(field, None)
    proposal["model_contract_identity_sha256"] = sha256_json(proposal)
    return proposal


def _validate_applied_official(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
) -> tuple[str, str]:
    """Validate applied-only authority and recover the original proposal receipt."""

    promotion = validated["promotion"]
    if (
        payload.get("formal_release_status") != "passed"
        or promotion.get("status") != "applied_formal_release"
        or promotion.get("official_apply_performed") is not True
    ):
        raise ContractError("model-contract authority lifecycle is neither proposal nor official")

    proposal_file_sha256 = promotion.get("proposal_file_sha256")
    proposal_identity_sha256 = promotion.get("proposal_identity_sha256")
    source_identity = promotion.get("source_model_contract_sha256")
    expected_source_identity = promotion.get("expected_source_model_contract_sha256")
    if (
        not _is_sha256(proposal_file_sha256)
        or not _is_sha256(proposal_identity_sha256)
        or not _is_sha256(source_identity)
        or expected_source_identity != source_identity
    ):
        raise ContractError("applied model contract proposal/source receipt is invalid")

    source_receipts = promotion.get("final_receipt_source_sha256")
    logical_receipts = promotion.get("final_receipt_identity_sha256")
    if (
        not isinstance(source_receipts, Mapping)
        or set(source_receipts) != set(_FINAL_RECEIPT_ROLES)
        or any(not _is_sha256(source_receipts.get(role)) for role in _FINAL_RECEIPT_ROLES)
        or not isinstance(logical_receipts, Mapping)
        or set(logical_receipts) != set(_FINAL_RECEIPT_ROLES)
        or any(not _is_sha256(logical_receipts.get(role)) for role in _FINAL_RECEIPT_ROLES)
    ):
        raise ContractError("applied model contract final receipt authority is incomplete")
    public_receipts = promotion.get("final_receipt_public_identity")
    expected_public = {
        "model_bundle_id": validated["model_bundle_id"],
        "root_expert_id": validated["root_expert_id"],
    }
    if (
        not isinstance(public_receipts, Mapping)
        or set(public_receipts) != {"fusion", "traits"}
        or any(public_receipts.get(role) != expected_public for role in ("fusion", "traits"))
    ):
        raise ContractError("applied model contract final public identity is invalid")

    checkpoints = validated["checkpoints"]
    if promotion.get("checkpoint_file_sha256_in_member_order") != checkpoints:
        raise ContractError("applied model contract Stage-B checkpoint authority changed")
    hair = payload.get("hair_identity_count_expert")
    if (
        not isinstance(hair, Mapping)
        or hair.get("current_checkpoint_role") != "formal_train399_only_deployment"
        or hair.get("strict_train399_only_retraining_gate") != "passed"
        or hair.get("deployment_ensemble_used_qcdev44_labels_in_some_members") is not False
        or hair.get("expert_id") != validated["stageb_binding"].get("expert_id")
        or hair.get("checkpoint_sha256_in_member_order") != checkpoints
        or not isinstance(hair.get("score_threshold"), (int, float))
        or isinstance(hair.get("score_threshold"), bool)
        or not math.isclose(
            float(hair["score_threshold"]),
            float(validated["stageb_binding"]["selected_score_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ContractError("applied model contract Stage-B deployment authority is invalid")
    root_expert = payload.get("root_expert")
    root_authority = (
        root_expert.get("root_bundle_authority")
        if isinstance(root_expert, Mapping)
        else None
    )
    if (
        not isinstance(root_authority, Mapping)
        or root_authority.get("binding")
        != "transitively_sealed_by_fresh_exact283_pipeline_identity"
        or root_authority.get("bundle_identity_sha256")
        != validated["root_bundle_identity_sha256"]
        or root_authority.get("pipeline_identity_sha256")
        != validated["root_pipeline_identity_sha256"]
    ):
        raise ContractError("applied model contract root-provider authority is invalid")

    reconstructed = _reconstruct_unapplied_proposal(payload)
    if reconstructed["model_contract_identity_sha256"] != proposal_identity_sha256:
        raise ContractError("applied model contract does not preserve proposal logical identity")
    if _pretty_json_sha256(reconstructed) != proposal_file_sha256:
        raise ContractError("applied model contract does not preserve proposal file receipt")
    return str(proposal_file_sha256), str(proposal_identity_sha256)


def read_model_contract_authority(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> ModelContractProposalBinding:
    """Read an unapplied proposal or its atomically applied official contract.

    The expected file hash always binds the authority file at ``path``.  An
    official authority returns the original proposal receipt embedded by the
    apply operation, after reconstructing and hashing that proposal exactly.
    """

    authority_path = Path(path).resolve()
    authority_file_sha256 = sha256_file(authority_path)
    if (
        expected_file_sha256 is not None
        and authority_file_sha256 != expected_file_sha256.casefold()
    ):
        raise ContractError("model-contract authority file SHA mismatch")
    payload = read_json(authority_path)
    if payload.get("schema_version") == RUN_SCOPED_AUTHORITY_PIN_SCHEMA:
        return _read_run_scoped_authority_pin(
            authority_path,
            payload,
            authority_file_sha256=authority_file_sha256,
        )
    validated = _validate_model_contract_content(payload)
    authority_identity_sha256 = _sealed_identity(
        payload,
        "model_contract_identity_sha256",
        role="model-contract authority",
    )
    promotion = validated["promotion"]
    if (
        payload.get("formal_release_status") == "passed_proposal_not_official"
        and promotion.get("status") == "validated_proposal_not_applied"
        and promotion.get("official_apply_performed") is False
    ):
        return _binding(
            path=authority_path,
            authority_file_sha256=authority_file_sha256,
            authority_identity_sha256=authority_identity_sha256,
            authority_lifecycle=UNAPPLIED_PROPOSAL_LIFECYCLE,
            proposal_file_sha256=authority_file_sha256,
            proposal_identity_sha256=authority_identity_sha256,
            validated=validated,
        )
    proposal_file_sha256, proposal_identity_sha256 = _validate_applied_official(
        payload,
        validated,
    )
    return _binding(
        path=authority_path,
        authority_file_sha256=authority_file_sha256,
        authority_identity_sha256=authority_identity_sha256,
        authority_lifecycle=APPLIED_OFFICIAL_LIFECYCLE,
        proposal_file_sha256=proposal_file_sha256,
        proposal_identity_sha256=proposal_identity_sha256,
        validated=validated,
    )


def validate_stageb_proposal_binding(
    binding: ModelContractProposalBinding,
    *,
    candidate_manifest_path: str | Path,
    candidate_manifest: Mapping[str, Any],
    selected_model_metadata_path: str | Path,
    selected_model_metadata: Mapping[str, Any],
    selection_receipt_path: str | Path,
    selection_receipt: Mapping[str, Any],
    checkpoints: Sequence[str | Path],
) -> None:
    """Require the Gate trio and checkpoint order to equal the proposal.

    The selected metadata stays a candidate.  This function never mutates any
    payload and explicitly rejects a prematurely promoted deployment role.
    """

    proposal = read_json(binding.path)
    promotion = proposal["promotion"]
    source_hashes = promotion.get("formal_gate_source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ContractError("proposal formal Gate file-hash binding is absent")
    if source_hashes.get("train399_candidate") != sha256_file(candidate_manifest_path):
        raise ContractError("proposal/candidate manifest file SHA mismatch")
    if source_hashes.get("train399_selection") != sha256_file(selection_receipt_path):
        raise ContractError("proposal/selection receipt file SHA mismatch")
    if read_json(candidate_manifest_path) != dict(candidate_manifest):
        raise ContractError("candidate manifest changed during Gate validation")
    if read_json(selected_model_metadata_path) != dict(selected_model_metadata):
        raise ContractError("selected metadata changed during Gate validation")
    if read_json(selection_receipt_path) != dict(selection_receipt):
        raise ContractError("selection receipt changed during Gate validation")

    pending = candidate_manifest.get("detection_model_metadata")
    if not isinstance(pending, Mapping):
        raise ContractError("candidate detection_model_metadata is absent")
    if pending.get("deployment_role") != PENDING_CANDIDATE_ROLE:
        raise ContractError("candidate metadata no longer has the non-promoted role")
    if selected_model_metadata.get("deployment_role") != PENDING_CANDIDATE_ROLE:
        raise ContractError("selected metadata must remain candidate_gate_passed_not_promoted")
    selected_unsigned = deepcopy(dict(selected_model_metadata))
    selected_identity = selected_unsigned.pop(
        "selected_model_metadata_identity_sha256", None
    )
    # The package detection contract deliberately treats runtime-only
    # precision_mode as outside the selected operating-point identity.
    selected_unsigned.pop("precision_mode", None)
    if not _is_sha256(selected_identity) or sha256_json(
        selected_unsigned
    ) != selected_identity:
        raise ContractError("selected Stage-B metadata identity is invalid")
    receipt_identity = _sealed_identity(
        selection_receipt,
        "selection_receipt_identity_sha256",
        role="selection receipt",
    )
    chosen = selection_receipt.get("selected")
    if not isinstance(chosen, Mapping):
        raise ContractError("selection receipt has no selected operating point")
    threshold = chosen.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ContractError("selection receipt threshold is invalid")
    checkpoint_sha256 = [sha256_file(path) for path in checkpoints]
    candidate_identity = candidate_manifest.get("candidate_bundle_identity_sha256")
    expected = {
        "expert_id": pending.get("expert_id"),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_score_threshold": float(threshold),
        "candidate_bundle_identity_sha256": candidate_identity,
        "selection_receipt_identity_sha256": receipt_identity,
        "selected_model_metadata_identity_sha256": selected_identity,
    }
    if dict(binding.stageb_binding) != expected:
        raise ContractError("model-contract proposal/Gate Stage-B binding mismatch")
    formal_identities = promotion["formal_gate_identity_sha256"]
    for field in (
        "candidate_bundle_identity_sha256",
        "selection_receipt_identity_sha256",
        "selected_model_metadata_identity_sha256",
    ):
        if formal_identities.get(field) != expected[field]:
            raise ContractError(f"proposal formal Gate logical identity mismatch: {field}")
    selected_expected = {
        "expert_id": expected["expert_id"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "selected_score_threshold": expected["selected_score_threshold"],
        "candidate_bundle_identity_sha256": expected[
            "candidate_bundle_identity_sha256"
        ],
        "selection_receipt_identity_sha256": expected[
            "selection_receipt_identity_sha256"
        ],
    }
    for field, value in selected_expected.items():
        observed = selected_model_metadata.get(field)
        if field == "selected_score_threshold":
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), float(value), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ContractError(f"selected metadata proposal mismatch: {field}")
        elif observed != value:
            raise ContractError(f"selected metadata proposal mismatch: {field}")
    if promotion.get("checkpoint_file_sha256_in_member_order") != checkpoint_sha256:
        raise ContractError("proposal checkpoint file order differs from workflow order")
    if pending.get("checkpoint_sha256") != checkpoint_sha256:
        raise ContractError("candidate checkpoint order differs from workflow order")


def require_receipt_binding(
    payload: Mapping[str, Any],
    binding: ModelContractProposalBinding,
    *,
    role: str,
) -> None:
    """Check that an output receipt carries the exact proposal identities."""

    for field, expected in binding.receipt_fields().items():
        if payload.get(field) != expected:
            raise ContractError(f"{role}: {field} mismatch")


def require_output_identity(
    payload: Mapping[str, Any],
    binding: ModelContractProposalBinding,
    *,
    role: str,
    root_field: str = "root_expert_id",
) -> None:
    """Require both immutable proposal hashes and proposal-owned public IDs."""

    require_receipt_binding(payload, binding, role=role)
    if (
        payload.get("model_bundle_id") != binding.model_bundle_id
        or payload.get(root_field) != binding.root_expert_id
    ):
        raise ContractError(f"{role}: public model/root identity mismatch")


__all__ = [
    "APPLIED_OFFICIAL_LIFECYCLE",
    "MODEL_CONTRACT_PROPOSAL_SCHEMA",
    "MODEL_CONTRACT_PROMOTION_SCHEMA",
    "ModelContractProposalBinding",
    "PENDING_CANDIDATE_ROLE",
    "RUN_SCOPED_AUTHORITY_PIN_LIFECYCLE",
    "RUN_SCOPED_AUTHORITY_PIN_SCHEMA",
    "RUN_SCOPED_AUTHORITY_PIN_STATUS",
    "UNAPPLIED_PROPOSAL_LIFECYCLE",
    "build_run_scoped_authority_pin",
    "read_model_contract_authority",
    "read_model_contract_proposal",
    "require_receipt_binding",
    "require_output_identity",
    "validate_stageb_proposal_binding",
]
