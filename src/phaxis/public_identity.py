"""Canonical PHAxis 1.0.0 public model-identity derivation.

The functions in this module are deliberately pure and contain no filesystem,
image, torch, or CUDA access.  Proposal creation and every proposal consumer
share this single derivation so a self-sealed JSON object cannot invent a
different user-visible model or root-expert identity.
"""

from __future__ import annotations

from typing import Any, Mapping

from .errors import ContractError
from .io import sha256_json


PUBLIC_SYSTEM_IDENTITY_SCHEMA = "PHAxis-public-system-identity-1.0"
MODEL_BUNDLE_PREFIX = "PHAXIS-V1.0.0-STRICT-TRAIN399-"
ROOT_EXPERT_PREFIX = "PHAxis-root-provider-"
ROOT_PROVIDER_ROLE = "PHAxis-portable-root-provider"
PUBLIC_SYSTEM_DERIVATION = (
    "Stage-B selection identities plus stable root-provider bundle identity"
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.casefold()


def public_system_identity_preimage(
    stageb_binding: Mapping[str, Any],
    *,
    root_bundle_identity_sha256: str,
) -> dict[str, str]:
    """Return the complete canonical preimage for the public system identity."""

    sources = {
        "stageb_candidate_bundle_identity_sha256": stageb_binding.get(
            "candidate_bundle_identity_sha256"
        ),
        "stageb_selection_receipt_identity_sha256": stageb_binding.get(
            "selection_receipt_identity_sha256"
        ),
        "stageb_selected_model_metadata_identity_sha256": stageb_binding.get(
            "selected_model_metadata_identity_sha256"
        ),
        "root_provider_bundle_identity_sha256": root_bundle_identity_sha256,
    }
    for field, value in sources.items():
        if not _is_sha256(value):
            raise ContractError(f"public identity source is invalid: {field}")
    return {
        "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
        **sources,
    }


def derive_public_identity(
    stageb_binding: Mapping[str, Any],
    *,
    root_bundle_identity_sha256: str,
) -> dict[str, str]:
    """Derive the stable PHAxis bundle and root-expert IDs."""

    preimage = public_system_identity_preimage(
        stageb_binding,
        root_bundle_identity_sha256=root_bundle_identity_sha256,
    )
    system_identity = sha256_json(preimage)
    return {
        "public_system_identity_sha256": system_identity,
        "model_bundle_id": MODEL_BUNDLE_PREFIX + system_identity[:20].upper(),
        "root_expert_id": (
            ROOT_EXPERT_PREFIX + root_bundle_identity_sha256[:20].upper()
        ),
        "root_provider_role": ROOT_PROVIDER_ROLE,
    }


def validate_proposal_public_identity(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Recompute and validate every proposal-owned public identity field."""

    promotion = payload.get("promotion")
    root_expert = payload.get("root_expert")
    public_system = payload.get("public_system_identity")
    expert_boundary = payload.get("expert_boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (promotion, root_expert, public_system, expert_boundary)
    ):
        raise ContractError("proposal public-identity structure is incomplete")
    stageb_binding = promotion.get("stageb_binding")
    formal_identities = promotion.get("formal_gate_identity_sha256")
    root_authority = root_expert.get("root_bundle_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (stageb_binding, formal_identities, root_authority)
    ):
        raise ContractError("proposal public-identity authority is incomplete")
    root_audit = formal_identities.get("root_exact283_audit_identity_sha256")
    root_pipeline = root_expert.get("pipeline_identity_sha256")
    root_bundle = root_expert.get("bundle_identity_sha256")
    derived = derive_public_identity(
        stageb_binding,
        root_bundle_identity_sha256=root_bundle,
    )
    if (
        root_expert.get("fresh_exact283_audit_identity_sha256") != root_audit
        or root_authority.get("pipeline_identity_sha256") != root_pipeline
        or root_authority.get("bundle_identity_sha256") != root_bundle
        or public_system.get("schema_version") != PUBLIC_SYSTEM_IDENTITY_SCHEMA
        or public_system.get("identity_sha256")
        != derived["public_system_identity_sha256"]
        or public_system.get("derivation") != PUBLIC_SYSTEM_DERIVATION
        or payload.get("model_bundle_id") != derived["model_bundle_id"]
        or root_expert.get("expert_id") != derived["root_expert_id"]
        or root_expert.get("provider_role") != derived["root_provider_role"]
        or expert_boundary.get("root_point_scale_continuity_statistics")
        != derived["root_expert_id"]
    ):
        raise ContractError("proposal public model/root identity derivation mismatch")
    return derived


__all__ = [
    "MODEL_BUNDLE_PREFIX",
    "PUBLIC_SYSTEM_DERIVATION",
    "PUBLIC_SYSTEM_IDENTITY_SCHEMA",
    "ROOT_EXPERT_PREFIX",
    "ROOT_PROVIDER_ROLE",
    "derive_public_identity",
    "public_system_identity_preimage",
    "validate_proposal_public_identity",
]
