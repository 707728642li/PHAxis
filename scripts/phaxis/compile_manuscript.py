#!/usr/bin/env python
"""Compile a PHAxis manuscript only from a sealed evidence graph and values.

The accepted values schema is the machine-derived, cell-provenanced
``PHAxis-manuscript-values-1.2``.  Hand-authored ``value/source_role`` files
from the earlier scaffolding schema are deliberately rejected.

This command performs no evidence discovery and never reads images,
annotations, predictions, checkpoints, or GPU state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.manuscript_values import (  # noqa: E402
    EVIDENCE_ARTIFACT_ROLES,
    EVIDENCE_GRAPH_SCHEMA,
    ManuscriptValuesError,
    VALUES_SCHEMA as DERIVED_VALUES_SCHEMA,
    build_token_source_contract as _shared_token_source_contract,
    validate_values_payload,
)
from phaxis.manuscript_contract import (  # noqa: E402
    ABSTRACT_WORD_LIMIT,
    ManuscriptTextContractError,
    require_abstract_within_limit,
)


VALUES_SCHEMA = DERIVED_VALUES_SCHEMA
TOKEN_CONTRACT_SCHEMA = "PHAxis-manuscript-token-source-contract-1.0"
RECEIPT_SCHEMA = "PHAxis-manuscript-compile-receipt-1.2"
TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
RESIDUAL_PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}\r\n]+\}\}")
TITLE_SLOT_PATTERN = re.compile(
    r"⟦RESULT SLOT → publication_title_contract\.(figures|tables)\.([1-9])⟧"
)
FORBIDDEN_VALUE_MARKERS = ("todo", "tbd", "provisional")
FORBIDDEN_LITERAL_VALUES = frozenset(
    {"null", "nan", "+nan", "-nan", "infinity", "+infinity", "-infinity"}
)
EXPECTED_EVIDENCE_ROLES = EVIDENCE_ARTIFACT_ROLES
AUTHOR_METADATA_TOKENS = frozenset(
    {
        "FINAL_ACKNOWLEDGMENTS",
        "FINAL_FUNDING_STATEMENT",
        "FINAL_AUTHOR_CONTRIBUTIONS",
        "FINAL_COMPETING_INTERESTS",
        "FINAL_ETHICS_STATEMENT",
    }
)

# Ordered most-specific-first.  These rules machine-enforce the source registry
# stated in the manuscript while resolving overlaps such as FINAL_ROOT_LENGTH_
# (biological analysis) versus other FINAL_ROOT_ assurance metrics.
TOKEN_FAMILY_RULES = (
    (
        "author_submission_metadata",
        tuple(sorted(AUTHOR_METADATA_TOKENS)),
        "author_submission_metadata",
        (),
        True,
    ),
    (
        "train399_selection",
        ("FINAL_STAGEB_",),
        "train399_selection_evaluation",
        ("train399_candidate", "train399_selection", "train399_evaluation", "stageb"),
        False,
    ),
    (
        "root_continuity_assurance",
        ("FINAL_ROOT_CONTINUITY_",),
        "measurement_assurance",
        ("figures",),
        False,
    ),
    (
        "hair_attachment_assurance",
        ("FINAL_HAIR_ATTACHMENT_",),
        "measurement_assurance",
        ("figures",),
        False,
    ),
    (
        "train399_evaluation",
        ("FINAL_HAIR_", "FINAL_QCDEV_"),
        "train399_selection_evaluation",
        ("train399_candidate", "train399_selection", "train399_evaluation", "stageb"),
        False,
    ),
    (
        "legacy_same_matcher_comparator",
        ("LOCKED_LEGACY_HYBRID_IDENTITY_",),
        "legacy_same_matcher_comparator",
        ("figures",),
        False,
    ),
    (
        "historical_oof443",
        ("HISTORICAL_OOF_",),
        "historical_oof443",
        ("figures",),
        False,
    ),
    (
        "biological_root_effects",
        ("FINAL_ROOT_WIDTH_", "FINAL_ROOT_LENGTH_"),
        "biological_analysis",
        ("cohorts", "analysis"),
        False,
    ),
    (
        "root_provider_equivalence",
        ("FINAL_ROOT_PROVIDER_",),
        "root_distal_scale_assurance",
        ("root_exact283",),
        False,
    ),
    (
        "root_distal_scale_assurance",
        ("FINAL_ROOT_", "FINAL_DISTAL_", "FINAL_SCALE_", "FINAL_AXIS_"),
        "root_distal_scale_assurance",
        ("root_exact283", "figures"),
        False,
    ),
    (
        "fusion_and_trait_export",
        (
            "FINAL_MATCHED_",
            "FINAL_ENDPOINT_COMPLETE_",
            "FINAL_TOTAL_HAIR_",
            "FINAL_FORMAL_IMAGE_",
            "FINAL_REVIEW_ONLY_IMAGE_",
            "FINAL_TRAIT_COVERAGE_",
            "FINAL_UNSUPPORTED_ATTACHMENT_",
        ),
        "fusion_trait_export",
        ("fusion", "traits"),
        False,
    ),
    (
        "distal_axis_profiles",
        ("FINAL_D15_AXIAL_", "FINAL_PROFILE_"),
        "distal_axis_profiles",
        ("profiles",),
        False,
    ),
    (
        "biological_analysis",
        (
            "FINAL_D15_",
            "FINAL_ABUNDANCE_",
            "FINAL_LENGTH_",
            "FINAL_FIRST_HAIR_",
            "FINAL_CLEAN_FULL_",
            "FINAL_DISCUSSION_BIOLOGICAL_",
            "FINAL_MULTITRAIT_",
        ),
        "biological_analysis",
        ("cohorts", "analysis"),
        False,
    ),
    (
        "observed_workflow_benchmark",
        ("FINAL_E2E_", "FINAL_RUNTIME_", "FINAL_BENCHMARK_"),
        "workflow_benchmark_record",
        ("figures",),
        False,
    ),
    (
        "model_bundle_release",
        ("FINAL_MODEL_BUNDLE_", "PHAXIS_MODEL_"),
        "model_bundle_release_record",
        ("train399_candidate", "root_exact283", "stageb"),
        False,
    ),
    (
        "clean_install_verification",
        ("FINAL_CLEAN_INSTALL_",),
        "clean_install_verification",
        (),
        False,
    ),
    (
        "manuscript_data_release",
        ("PHAXIS_MANUSCRIPT_DATA_",),
        "data_release_registry",
        ("traits", "cohorts", "analysis", "profiles", "figures"),
        False,
    ),
    (
        "software_release",
        (
            "PHAXIS_RELEASE_",
            "PHAXIS_GIT_",
            "PHAXIS_REPOSITORY_",
            "PHAXIS_SOFTWARE_",
        ),
        "software_release_registry",
        (),
        False,
    ),
    (
        "biological_and_training_data_records",
        ("FINAL_BIOLOGICAL_", "HUMANCURATED443_"),
        "data_acquisition_registry",
        ("cohorts",),
        False,
    ),
)


class ManuscriptCompileError(RuntimeError):
    """A manuscript input or output violates the closed compilation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptCompileError(message)


def _canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManuscriptCompileError("payload is not canonical finite JSON") from error


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_bytes(path: Path, role: str) -> bytes:
    _require(not path.is_symlink(), f"{role} may not be a symlink")
    resolved = path.resolve()
    _require(resolved.is_file(), f"{role} does not exist: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise ManuscriptCompileError(f"cannot read {role}: {resolved}") from error


def _read_json_object(path: Path, role: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes(path, role)
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ManuscriptCompileError(
            f"{role} must be strict UTF-8 JSON without duplicate keys or NaN"
        ) from error
    _require(isinstance(payload, dict), f"{role} must contain one JSON object")
    return raw, payload


def _token_rule(token: str) -> tuple[str, str, tuple[str, ...]]:
    for family, patterns, source_role, evidence_roles, exact in TOKEN_FAMILY_RULES:
        matched = token in patterns if exact else token.startswith(patterns)
        if matched:
            return family, source_role, evidence_roles
    raise ManuscriptCompileError(f"no token-family/source-role contract for {token}")


def build_token_source_contract(master_text: str) -> dict[str, Any]:
    """Return the deterministic, sealed token family/source-role contract."""
    try:
        return _shared_token_source_contract(master_text)
    except ManuscriptValuesError as error:
        raise ManuscriptCompileError(str(error)) from error


def _validate_evidence_graph(payload: Mapping[str, Any]) -> str:
    _require(
        payload.get("schema_version") == EVIDENCE_GRAPH_SCHEMA,
        "unsupported evidence graph schema",
    )
    _require(
        payload.get("status") == "passed_formal_evidence_graph",
        "evidence graph is not final and closed",
    )
    _require(
        payload.get("formal_release_evidence_closed") is True,
        "formal release evidence is not closed",
    )
    _require(payload.get("blind_images_used") == 0, "evidence graph is blind-tainted")
    _require(
        payload.get("root_cap_region_statistics_included") is False,
        "evidence graph permits root-cap-region statistics",
    )
    identity = payload.get("manifest_identity_sha256")
    _require(_is_sha256(identity), "invalid evidence graph identity")
    unsigned = deepcopy(dict(payload))
    unsigned.pop("manifest_identity_sha256", None)
    _require(
        sha256_json(unsigned) == identity,
        "evidence graph identity does not seal the complete graph",
    )
    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, Mapping), "evidence graph artifacts are missing")
    missing = sorted(set(EXPECTED_EVIDENCE_ROLES) - set(artifacts))
    extra = sorted(set(artifacts) - set(EXPECTED_EVIDENCE_ROLES))
    _require(not missing and not extra, f"evidence graph role mismatch; missing={missing}, extra={extra}")
    for role in EXPECTED_EVIDENCE_ROLES:
        artifact = artifacts[role]
        _require(isinstance(artifact, Mapping), f"evidence graph role is invalid: {role}")
        _require(
            _is_sha256(artifact.get("source_file_sha256")),
            f"evidence graph role lacks a source file SHA-256: {role}",
        )
    return str(identity)


def _render_value(token: str, value: Any, *, author_metadata: bool) -> str:
    _require(value is not None, f"{token}: null value is forbidden")
    _require(not isinstance(value, (dict, list)), f"{token}: value must be scalar")
    if isinstance(value, float):
        _require(math.isfinite(value), f"{token}: NaN or infinity is forbidden")
    if isinstance(value, str):
        rendered = value
        _require(bool(rendered.strip()), f"{token}: empty value is forbidden")
        _require(
            rendered.strip().casefold() not in FORBIDDEN_LITERAL_VALUES,
            f"{token}: null/NaN/infinity text is forbidden",
        )
    elif isinstance(value, (int, float, bool)):
        rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
    else:
        raise ManuscriptCompileError(f"{token}: unsupported value type")
    _require(
        not any(marker in rendered.casefold() for marker in FORBIDDEN_VALUE_MARKERS),
        f"{token}: TODO/TBD/provisional value is forbidden",
    )
    _require(
        RESIDUAL_PLACEHOLDER_PATTERN.search(rendered) is None,
        f"{token}: residual manuscript token is forbidden",
    )
    if author_metadata:
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"{token}: author metadata must be a non-empty string",
        )
    return rendered


def _validate_values(
    payload: Mapping[str, Any],
    *,
    master_raw: bytes,
    evidence_graph_raw: bytes,
    evidence_identity: str,
    contract: Mapping[str, Any],
) -> dict[str, str]:
    try:
        return validate_values_payload(
            payload,
            master_raw=master_raw,
            evidence_graph_raw=evidence_graph_raw,
            evidence_graph_identity_sha256=evidence_identity,
            token_contract=contract,
        )
    except ManuscriptValuesError as error:
        raise ManuscriptCompileError(str(error)) from error


def _publish_without_overwrite(path: Path, temporary: Path) -> None:
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ManuscriptCompileError(f"refusing to overwrite existing output: {path}") from error
    except OSError as error:
        raise ManuscriptCompileError(f"atomic no-overwrite publication failed: {path}") from error
    temporary.unlink()


def _prepare_temporary(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def compile_manuscript(
    *,
    master: str | Path,
    evidence_graph: str | Path,
    values: str | Path,
    output: str | Path,
    receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, compile, and atomically publish manuscript plus receipt."""
    master_path = Path(master).resolve()
    evidence_path = Path(evidence_graph).resolve()
    values_path = Path(values).resolve()
    output_path = Path(output).resolve()
    receipt_path = (
        Path(receipt).resolve()
        if receipt is not None
        else output_path.with_name(f"{output_path.name}.receipt.json")
    )
    _require(output_path != receipt_path, "output and receipt paths must differ")
    _require(not output_path.exists(), f"refusing to overwrite existing output: {output_path}")
    _require(not receipt_path.exists(), f"refusing to overwrite existing receipt: {receipt_path}")

    master_raw = _read_bytes(master_path, "master manuscript")
    try:
        master_text = master_raw.decode("utf-8")
    except UnicodeError as error:
        raise ManuscriptCompileError("master manuscript must be UTF-8") from error
    evidence_raw, graph = _read_json_object(evidence_path, "evidence graph")
    values_raw, values_payload = _read_json_object(values_path, "manuscript values")
    evidence_identity = _validate_evidence_graph(graph)
    contract = build_token_source_contract(master_text)
    rendered = _validate_values(
        values_payload,
        master_raw=master_raw,
        evidence_graph_raw=evidence_raw,
        evidence_identity=evidence_identity,
        contract=contract,
    )

    compiled = master_text
    for token in sorted(rendered):
        compiled = compiled.replace(f"{{{{{token}}}}}", rendered[token])
    title_contract = values_payload["publication_title_contract"]
    expected_title_slots = {
        (kind, number)
        for kind, maximum in (("figures", 6), ("tables", 3))
        for number in map(str, range(1, maximum + 1))
    }
    observed_title_slots = set(TITLE_SLOT_PATTERN.findall(compiled))
    _require(
        observed_title_slots == expected_title_slots,
        "master publication-title result slots are incomplete or unexpected",
    )
    compiled = TITLE_SLOT_PATTERN.sub(
        lambda match: str(title_contract[match.group(1)][match.group(2)]),
        compiled,
    )
    residual = sorted(set(RESIDUAL_PLACEHOLDER_PATTERN.findall(compiled)))
    _require(not residual, f"compiled manuscript retains tokens: {residual}")
    try:
        abstract_words = require_abstract_within_limit(compiled)
    except ManuscriptTextContractError as error:
        raise ManuscriptCompileError(str(error)) from error
    output_raw = compiled.encode("utf-8")
    source_role_counts = Counter(
        row["source_role"] for row in contract["tokens"].values()
    )
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "completed_strict_final_manuscript_compilation",
        "master_sha256": sha256_bytes(master_raw),
        "values_sha256": sha256_bytes(values_raw),
        "values_identity_sha256": values_payload["values_identity_sha256"],
        "evidence_graph_sha256": sha256_bytes(evidence_raw),
        "evidence_graph_identity_sha256": evidence_identity,
        "figure_input_assembly_identity_sha256": values_payload[
            "figure_input_assembly_identity_sha256"
        ],
        "model_contract_proposal_sha256": values_payload[
            "model_contract_proposal_sha256"
        ],
        "model_contract_proposal_identity_sha256": values_payload[
            "model_contract_proposal_identity_sha256"
        ],
        "model_bundle_id": values_payload["model_bundle_id"],
        "root_expert_id": values_payload["root_expert_id"],
        "root_bundle_identity_sha256": values_payload[
            "root_bundle_identity_sha256"
        ],
        "hair_identity_count_expert": values_payload[
            "hair_identity_count_expert"
        ],
        "human_metadata_identity_sha256": values_payload[
            "human_metadata_identity_sha256"
        ],
        "model_bundle_manifest_identity_sha256": values_payload[
            "model_bundle_manifest_identity_sha256"
        ],
        "clean_install_receipt_identity_sha256": values_payload[
            "clean_install_receipt_identity_sha256"
        ],
        "source_release_tree_identity_sha256": values_payload[
            "source_release_tree_identity_sha256"
        ],
        "software_release_cross_binding_identity_sha256": values_payload[
            "software_release_cross_binding_identity_sha256"
        ],
        "narrative_decision_identity_sha256": values_payload[
            "narrative_decision_identity_sha256"
        ],
        "narrative_branch_id": values_payload["narrative_branch_id"],
        "publication_title_contract_identity_sha256": title_contract[
            "title_contract_identity_sha256"
        ],
        "publication_title_slot_count": len(observed_title_slots),
        "token_contract_identity_sha256": contract["contract_identity_sha256"],
        "output_sha256": sha256_bytes(output_raw),
        "token_count": len(rendered),
        "source_role_token_counts": dict(sorted(source_role_counts.items())),
        "unresolved_token_count": 0,
        "author_metadata_complete": True,
        "abstract_word_count": abstract_words,
        "abstract_word_limit": ABSTRACT_WORD_LIMIT,
        "abstract_word_limit_passed": True,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    result["receipt_identity_sha256"] = sha256_json(result)
    receipt_raw = (
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    # Prepare both complete files before either becomes visible.  Hard-link
    # publication is atomic and fails if the destination appears concurrently.
    output_temporary = _prepare_temporary(output_path, output_raw)
    try:
        receipt_temporary = _prepare_temporary(receipt_path, receipt_raw)
    except BaseException:
        output_temporary.unlink(missing_ok=True)
        raise
    published: list[Path] = []
    try:
        _require(
            not output_path.exists() and not receipt_path.exists(),
            "output or receipt appeared during compilation",
        )
        _publish_without_overwrite(output_path, output_temporary)
        published.append(output_path)
        _publish_without_overwrite(receipt_path, receipt_temporary)
        published.append(receipt_path)
    except BaseException:
        output_temporary.unlink(missing_ok=True)
        receipt_temporary.unlink(missing_ok=True)
        for published_path in reversed(published):
            published_path.unlink(missing_ok=True)
        raise
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--evidence-graph", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = compile_manuscript(
        master=args.master,
        evidence_graph=args.evidence_graph,
        values=args.values,
        output=args.output,
        receipt=args.receipt,
    )
    print(receipt["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
