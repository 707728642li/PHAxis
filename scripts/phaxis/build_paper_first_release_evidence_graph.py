#!/usr/bin/env python
"""Build the current-lineage paper-first evidence graph for PHAxis 1.0.0.

This CPU-only adapter closes the model-promotion evidence graph without
reintroducing the retired speed Figure 6.  It consumes only explicitly named,
self-sealed current-train399/exact283 receipts and never reads images, model
weights, predictions, annotations, or condition metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_ROOT), str(PROJECT_ROOT / "src")]

from build_manuscript_evidence_manifest import (  # noqa: E402
    SCHEMA_VERSION,
    EvidenceManifestError,
    _guard_final_summary,
    _require_proposal_binding,
    _sealed_identity,
    _validate_formal_gate_receipts,
    _validate_model_contract_proposal,
    sha256_file,
    sha256_json,
)


STATUS = "passed_formal_evidence_graph"
MODEL_ID = "PHAXIS-V1.0.0-STRICT-TRAIN399-D8C44505008FFD539011"
ROOT_ID = "PHAxis-root-provider-3272360075B066394EED"
HAIR_ID = "PHAxis-StageB-train399-five-seed"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class PaperFirstEvidenceError(RuntimeError):
    """A named authority was absent, stale, or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperFirstEvidenceError(message)


def _read(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{role}: invalid file")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PaperFirstEvidenceError(f"{role}: invalid UTF-8 JSON") from error
    _require(isinstance(payload, dict), f"{role}: JSON root is not an object")
    return resolved, payload


def _all_declared_sha(payload: Any) -> dict[str, str]:
    rows: dict[str, str] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                path = f"{prefix}.{key}" if prefix else str(key)
                item = value[key]
                if isinstance(item, str) and SHA_RE.fullmatch(item):
                    rows[path] = item
                visit(item, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(payload, "")
    return rows


def _artifact(path: Path, payload: Mapping[str, Any], identity_field: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_file_sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "declared_sha256_identities": _all_declared_sha(payload),
    }
    if identity_field is not None:
        record["primary_identity_field"] = identity_field
        record["primary_identity_sha256"] = payload[identity_field]
    return record


def _proposal_bound(
    role: str,
    payload: Mapping[str, Any],
    *,
    proposal_sha: str,
    proposal_identity: str,
) -> None:
    try:
        _require_proposal_binding(
            role,
            payload,
            proposal_sha256=proposal_sha,
            proposal_identity_sha256=proposal_identity,
        )
    except EvidenceManifestError as error:
        raise PaperFirstEvidenceError(str(error)) from error


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    try:
        return _sealed_identity(payload, field, role)
    except EvidenceManifestError as error:
        raise PaperFirstEvidenceError(str(error)) from error


def build_graph(*, output: str | Path, **named: str | Path) -> dict[str, Any]:
    destination = Path(output).resolve()
    _require(not destination.exists(), f"output already exists: {destination}")
    loaded = {role: _read(value, role) for role, value in named.items()}
    paths = {role: pair[0] for role, pair in loaded.items()}
    payloads = {role: pair[1] for role, pair in loaded.items()}
    _require(len(set(paths.values())) == len(paths), "all inputs must be distinct files")

    gate_roles = (
        "train399_candidate",
        "train399_selection",
        "train399_evaluation",
        "root_exact283",
    )
    try:
        _validate_formal_gate_receipts(
            {role: paths[role] for role in gate_roles},
            {role: payloads[role] for role in gate_roles},
        )
        proposal_identity, _ = _validate_model_contract_proposal(payloads["model_contract_proposal"])
    except EvidenceManifestError as error:
        raise PaperFirstEvidenceError(str(error)) from error
    proposal_sha = sha256_file(paths["model_contract_proposal"])
    proposal = payloads["model_contract_proposal"]
    _require(proposal.get("model_bundle_id") == MODEL_ID, "proposal model identity changed")
    _require(proposal.get("root_expert", {}).get("expert_id") == ROOT_ID, "proposal root identity changed")
    stageb_binding = proposal.get("promotion", {}).get("stageb_binding")
    _require(isinstance(stageb_binding, Mapping), "proposal StageB binding is absent")
    _require(stageb_binding.get("expert_id") == HAIR_ID, "proposal hair expert changed")

    stageb = payloads["stageb"]
    _require(
        stageb.get("schema_version") == "PHAxis-StageB-inference-run-1.1"
        and stageb.get("status") == "completed"
        and stageb.get("blind_images_used") == 0,
        "stageb: final exact283 hair-only summary is invalid",
    )
    for role in ("fusion", "traits"):
        try:
            _guard_final_summary(role, payloads[role])
        except EvidenceManifestError as error:
            raise PaperFirstEvidenceError(str(error)) from error
    for role in ("stageb", "fusion", "traits"):
        _proposal_bound(
            role,
            payloads[role],
            proposal_sha=proposal_sha,
            proposal_identity=proposal_identity,
        )
    _sealed(payloads["stageb"], "summary_identity_sha256", "stageb")
    _sealed(payloads["fusion"], "summary_identity_sha256", "fusion")
    _sealed(payloads["traits"], "export_identity_sha256", "traits")
    _require(payloads["stageb"].get("images") == 283, "stageb scope is not exact283")
    _require(payloads["fusion"].get("images") == 283, "fusion scope is not exact283")
    _require(payloads["traits"].get("tasks") == 283, "trait scope is not exact283")

    scientific_specs = {
        "cohorts": ("cohort_build_identity_sha256", "PHAxis-biological-cohorts-1.0"),
        "analysis": ("analysis_identity_sha256", "PHAxis-exploratory-biological-analysis-1.0"),
        "profiles": ("cohort_profile_bundle_identity_sha256", "PHAxis-distal-axis-cohort-profile-bundle-1.0.0"),
    }
    for role, (identity_field, schema) in scientific_specs.items():
        payload = payloads[role]
        _require(payload.get("schema_version") == schema, f"{role}: schema changed")
        _require(payload.get("blind_images_used") == 0, f"{role}: blind-tainted")
        _require(payload.get("model_bundle_id") == MODEL_ID, f"{role}: model identity changed")
        _require(payload.get("root_expert_id") == ROOT_ID, f"{role}: root identity changed")
        _proposal_bound(
            role,
            payload,
            proposal_sha=proposal_sha,
            proposal_identity=proposal_identity,
        )
        _sealed(payload, identity_field, role)

    sealed_specs = {
        "focused": "receipt_identity_sha256",
        "manuscript": "receipt_identity_sha256",
        "figure134": "receipt_identity_sha256",
        "figure2": "receipt_identity_sha256",
        "figure5": "render_identity_sha256",
    }
    for role, field in sealed_specs.items():
        payload = payloads[role]
        _require(payload.get("blind_images_used") == 0, f"{role}: blind-tainted")
        _sealed(payload, field, role)
        if "model_bundle_id" in payload:
            _require(payload.get("model_bundle_id") == MODEL_ID, f"{role}: model identity changed")
        if "root_expert_id" in payload:
            _require(payload.get("root_expert_id") == ROOT_ID, f"{role}: root identity changed")

    for role in ("focused", "figure134"):
        _proposal_bound(
            role,
            payloads[role],
            proposal_sha=proposal_sha,
            proposal_identity=proposal_identity,
        )
    _require(
        payloads["figure5"].get("model_contract_proposal_identity_sha256")
        == proposal_identity,
        "figure5: model-contract proposal identity mismatch",
    )
    audit = payloads["consistency_audit"]
    _require(
        audit.get("status") == "passed_exact_73_scientific_slot_backfill"
        and audit.get("scientific_slot_count") == 73
        and audit.get("science_slots_remaining") == 0
        and audit.get("blind_images_used") == 0
        and audit.get("root_cap_region_statistics_included") is False,
        "manuscript consistency audit is not the exact 73-slot pass",
    )
    _require(
        payloads["manuscript"].get("focused_receipt_identity_sha256")
        == payloads["focused"].get("receipt_identity_sha256"),
        "manuscript/focused receipt identity differs",
    )
    _require(
        payloads["figure5"].get("source_package_identity_sha256")
        == payloads["focused"].get("fig5_source_package_identity_sha256"),
        "Figure5/focused source package identity differs",
    )
    _require(
        payloads["figure134"].get("figure_count") == 3
        and payloads["figure2"].get("visual_qa_status") == "PASS"
        and payloads["figure5"].get("status")
        == "completed_current_train399_exact283_figure5_final",
        "paper-first figure receipt set is incomplete",
    )

    identity_fields = {
        "model_contract_proposal": "model_contract_identity_sha256",
        "train399_candidate": "candidate_manifest_identity_sha256",
        "train399_selection": "selection_receipt_identity_sha256",
        "train399_evaluation": None,
        "root_exact283": "audit_identity_sha256",
        "stageb": "summary_identity_sha256",
        "fusion": "summary_identity_sha256",
        "traits": "export_identity_sha256",
        "cohorts": "cohort_build_identity_sha256",
        "analysis": "analysis_identity_sha256",
        "profiles": "cohort_profile_bundle_identity_sha256",
    }
    artifacts = {
        role: _artifact(paths[role], payloads[role], field)
        for role, field in identity_fields.items()
    }
    artifacts["figure_inputs"] = _artifact(
        paths["focused"], payloads["focused"], "receipt_identity_sha256"
    )
    figure_declared: dict[str, str] = {
        "model_contract_proposal_sha256": proposal_sha,
        "model_contract_proposal_identity_sha256": proposal_identity,
    }
    for role in ("figure134", "figure2", "figure5", "manuscript"):
        for key, value in _all_declared_sha(payloads[role]).items():
            figure_declared[f"{role}.{key}"] = value
    artifacts["figures"] = {
        "source_file_sha256": sha256_file(paths["figure134"]),
        "schema_version": "PHAxis-paper-first-main-figure-suite-1.0",
        "status": "completed_figures_1_to_5_current_lineage",
        "declared_sha256_identities": figure_declared,
        "component_source_file_sha256": {
            role: sha256_file(paths[role])
            for role in ("figure134", "figure2", "figure5", "manuscript")
        },
    }

    graph: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "graph_profile": "paper_first_science_complete_five_main_figures",
        "formal_release_evidence_closed": True,
        "software_publication_authority_closed": False,
        "software_publication_authority_deferred_reason": (
            "author_verified_release_metadata_and_public_coordinates_required"
        ),
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
        "model_contract_proposal_sha256": proposal_sha,
        "model_contract_proposal_identity_sha256": proposal_identity,
        "model_bundle_id": MODEL_ID,
        "root_expert_id": ROOT_ID,
        "hair_identity_expert_id": HAIR_ID,
        "stageb_binding": deepcopy(dict(stageb_binding)),
        "cohort_identities": {"full": "full283", "primary": "clean261", "formal": "formal238", "D15_primary": 47},
        "paper_first_components": {
            role: {
                "source_file_sha256": sha256_file(paths[role]),
                "declared_sha256_identities": _all_declared_sha(payloads[role]),
            }
            for role in ("focused", "manuscript", "consistency_audit", "figure134", "figure2", "figure5")
        },
        "artifacts": artifacts,
    }
    graph["manifest_identity_sha256"] = sha256_json(graph)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(graph, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _require(not destination.exists(), f"output appeared during build: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in (
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
        "focused",
        "manuscript",
        "consistency_audit",
        "figure134",
        "figure2",
        "figure5",
    ):
        parser.add_argument("--" + role.replace("_", "-"), required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    named = {
        key: value
        for key, value in vars(args).items()
        if key != "output"
    }
    try:
        graph = build_graph(output=args.output, **named)
    except PaperFirstEvidenceError as error:
        print(f"PHAxis paper-first evidence graph blocked: {error}", file=sys.stderr)
        return 2
    print(graph["manifest_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
