"""Hash-closed primitives for deriving PHAxis manuscript values.

The production assembler lives in :mod:`scripts.phaxis.build_manuscript_values`.
This module owns the reusable, installed-package-safe validation primitives so
that a values file cannot be reduced to an unverified ``value/source_role``
lookup table.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence
import uuid

from . import _toml_compat as tomllib
from .contracts import ContractError
from .public_identity import (
    MODEL_BUNDLE_PREFIX,
    ROOT_EXPERT_PREFIX,
    ROOT_PROVIDER_ROLE,
    validate_proposal_public_identity,
)
from .root_trait_assurance import ROOT_TRAIT_ASSURANCE_TOKENS
from .publication_evidence import (
    FIGURE_SOURCE_INPUT_ROLES,
    WT_SECONDARY_RESOURCE_ROLES,
    validate_wt_secondary_analysis_binding,
    validate_wt_secondary_evidence,
)
from .narrative_decision import validate_narrative_decision
from .publication_titles import title_contract


VALUES_SCHEMA = "PHAxis-manuscript-values-1.2"
VALUES_BUILDER_SCHEMA = "PHAxis-manuscript-values-builder-1.1"
HUMAN_METADATA_SCHEMA = "PHAxis-manuscript-human-metadata-1.0"
HUMAN_METADATA_REPORT_SCHEMA = "PHAxis-manuscript-human-metadata-report-1.0"
EVIDENCE_GRAPH_SCHEMA = "PHAxis-manuscript-release-evidence-graph-1.1"
FIGURE_INPUT_SCHEMA = "PHAxis-manuscript-figure-inputs-2.0"
FIGURE_ASSEMBLER_SCHEMA = "PHAxis-publication-figure-input-assembly-1.0"
MODEL_BUNDLE_MANIFEST_SCHEMA = "PHAxis-model-bundle-release-manifest-1.0"
CLEAN_INSTALL_RECEIPT_SCHEMA = "PHAxis-clean-install-verification-1.0"
TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
RESIDUAL_TOKEN_PATTERN = re.compile(r"\{\{[^{}\r\n]+\}\}")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DOI_PATTERN = re.compile(r"^(?:https://doi\.org/)?10\.[0-9]{4,9}/\S+$", re.I)
URL_PATTERN = re.compile(r"^https://[^\s]+$", re.I)

BIOLOGICAL_ACQUISITION_TOKENS = frozenset(
    {
        "FINAL_BIOLOGICAL_ACCESSION",
        "FINAL_BIOLOGICAL_CONSTRUCT_CONTROL_IDENTITY_AND_SOURCE",
        "FINAL_BIOLOGICAL_GROWTH_MEDIUM",
        "FINAL_BIOLOGICAL_PHOTOPERIOD",
        "FINAL_BIOLOGICAL_GROWTH_TIMELINE",
        "FINAL_BIOLOGICAL_TEMPERATURE_EXPOSURE_ONSET",
        "FINAL_BIOLOGICAL_TEMPERATURE_EXPOSURE_DURATION",
        "FINAL_BIOLOGICAL_PLATE_BLOCK_AND_PLANT_UNIT",
        "FINAL_BIOLOGICAL_REPLICATION_AND_RANDOMIZATION",
        "FINAL_BIOLOGICAL_IMAGING_DEVICE",
        "FINAL_BIOLOGICAL_IMAGING_OBJECTIVE",
        "FINAL_BIOLOGICAL_NATIVE_PIXEL_SAMPLING",
        "FINAL_BIOLOGICAL_FIELD_SAMPLING_AND_STITCHING",
        "FINAL_BIOLOGICAL_PHYSICAL_CALIBRATION",
        "FINAL_BIOLOGICAL_EXCLUSION_RULES",
    }
)
HUMAN_METADATA_TOKENS = frozenset(
    {
        "FINAL_ACKNOWLEDGMENTS",
        "FINAL_FUNDING_STATEMENT",
        "FINAL_AUTHOR_CONTRIBUTIONS",
        "FINAL_COMPETING_INTERESTS",
        "FINAL_ETHICS_STATEMENT",
        "FINAL_BIOLOGICAL_IMAGE_AVAILABILITY_STATEMENT",
        "HUMANCURATED443_DATA_URL",
        "HUMANCURATED443_DATA_DOI",
        "HUMANCURATED443_LICENSE",
        "PHAXIS_MANUSCRIPT_DATA_URL",
        "PHAXIS_MANUSCRIPT_DATA_DOI",
        "PHAXIS_REPOSITORY_URL",
        "PHAXIS_RELEASE_TAG",
        "PHAXIS_RELEASE_DOI",
        "PHAXIS_SOFTWARE_LICENSE",
        "PHAXIS_MODEL_BUNDLE_URL",
        "PHAXIS_MODEL_BUNDLE_MANIFEST_URL",
        "PHAXIS_MODEL_LICENSE",
        *BIOLOGICAL_ACQUISITION_TOKENS,
    }
)
SOFTWARE_RELEASE_TOKENS = frozenset(
    {
        "PHAXIS_REPOSITORY_URL",
        "PHAXIS_RELEASE_TAG",
        "PHAXIS_RELEASE_DOI",
        "PHAXIS_SOFTWARE_LICENSE",
    }
)
URL_TOKENS = frozenset(token for token in HUMAN_METADATA_TOKENS if token.endswith("_URL"))
DOI_TOKENS = frozenset(token for token in HUMAN_METADATA_TOKENS if token.endswith("_DOI"))

EVIDENCE_ARTIFACT_ROLES = (
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

HISTORICAL_TOKEN_PREFIXES = (
    "HISTORICAL_OOF_",
    "LOCKED_LEGACY_HYBRID_IDENTITY_",
)
FORBIDDEN_TEXT_MARKERS = (
    "todo",
    "tbd",
    "provisional",
    "blocked_pending",
    "not_for_submission",
)
DEFERRED_HUMAN_METADATA_MARKERS = ("deferred",)
FORBIDDEN_LITERAL_VALUES = frozenset(
    {"null", "nan", "+nan", "-nan", "infinity", "+infinity", "-infinity"}
)
LEGACY_DEPLOYMENT_MARKERS = (
    "443cv",
    "rhaxiscc-stageb-5fold",
    "rhaxiscc_stage_b_5fold",
    "five family-grouped folds spanning all humancurated443",
)
INTERNAL_PROVIDER_ABI_MARKERS = ("phaxis-v1.0-frozen",)
STALE_PUBLIC_VERSION_PATTERN = re.compile(r"\bphaxis\s+1\.0(?!\.0)\b", re.I)

CORE_FIGURE_SOURCE_ROLES = (
    "train399_evaluation",
    "root_exact283",
    "stageb",
    "fusion",
    "traits",
    "cohorts",
    "analysis",
    "profiles",
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

SAME_HARDWARE_RUNTIME_TOKENS = (
    "FINAL_BENCHMARK_LATENCY_MODE_LABEL",
    "FINAL_E2E_FROZEN_V1_BATCH_TOTAL_MIN",
    "FINAL_E2E_BATCH_SPEEDUP_FROZEN_V1_OVER_PHAXIS",
    "FINAL_E2E_FROZEN_V1_MEDIAN_IMAGE_S",
    "FINAL_E2E_MEDIAN_LATENCY_SPEEDUP_FROZEN_V1_OVER_PHAXIS",
)

ROOT_CONTINUITY_ASSURANCE_TOKENS = (
    "FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEAN",
    "FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEDIAN",
    "FINAL_ROOT_CONTINUITY_LONGEST_UNSUPPORTED_GAP_UM_ON_BEST_COMPONENT_MEDIAN",
    "FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE",
    "FINAL_ROOT_CONTINUITY_VISIBLE_AXIS_EXTENT_MAE_UM",
    "FINAL_ROOT_CONTINUITY_VALIDATION_N",
    "FINAL_ROOT_CONTINUITY_METRICS_CI",
)

HAIR_ATTACHMENT_ASSURANCE_TOKENS = (
    "FINAL_HAIR_ATTACHMENT_QUALIFIED_PRECISION_AT_20UM",
    "FINAL_HAIR_ATTACHMENT_QUALIFIED_RECALL_AT_20UM",
    "FINAL_HAIR_ATTACHMENT_QUALIFIED_F1_AT_20UM",
    "FINAL_HAIR_ATTACHMENT_FORMAL_MATCHED_ERROR_MEDIAN_UM",
    "FINAL_HAIR_ATTACHMENT_FORMAL_MATCHED_ERROR_P95_UM",
    "FINAL_HAIR_ATTACHMENT_VALIDATION_N",
    "FINAL_HAIR_ATTACHMENT_PREDICTED_N",
    "FINAL_HAIR_ATTACHMENT_ANNOTATED_N",
    "FINAL_HAIR_ATTACHMENT_QUALIFIED_TP_N",
    "FINAL_HAIR_ATTACHMENT_FORMAL_MATCH_N",
    "FINAL_HAIR_ATTACHMENT_METRICS_CI",
)

# Ordered most-specific-first.  This is the same semantic registry consumed by
# the compiler; the values builder adds cell-level provenance beneath it.
TOKEN_FAMILY_RULES = (
    (
        "author_submission_metadata",
        (
            "FINAL_ACKNOWLEDGMENTS",
            "FINAL_FUNDING_STATEMENT",
            "FINAL_AUTHOR_CONTRIBUTIONS",
            "FINAL_COMPETING_INTERESTS",
            "FINAL_ETHICS_STATEMENT",
        ),
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
        "root_derived_trait_assurance",
        ROOT_TRAIT_ASSURANCE_TOKENS,
        "root_derived_trait_assurance",
        ("figures",),
        True,
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
        "same_hardware_frozen_v1_benchmark",
        SAME_HARDWARE_RUNTIME_TOKENS,
        "workflow_benchmark_record",
        ("figures",),
        True,
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
        ("PHAXIS_RELEASE_", "PHAXIS_GIT_", "PHAXIS_REPOSITORY_", "PHAXIS_SOFTWARE_"),
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


class ManuscriptValuesError(RuntimeError):
    """A manuscript-values input or derivation violates the closed contract."""


class HumanMetadataError(ManuscriptValuesError):
    """External author/release metadata are missing, extra, or invalid."""

    def __init__(
        self,
        message: str,
        *,
        missing: Sequence[str] = (),
        extra: Sequence[str] = (),
        invalid: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing = tuple(sorted(missing))
        self.extra = tuple(sorted(extra))
        self.invalid = dict(sorted((invalid or {}).items()))


@dataclass(frozen=True)
class JsonSource:
    """One strict JSON source plus its raw-file identity."""

    role: str
    path: Path
    raw: bytes
    payload: dict[str, Any]
    file_sha256: str
    logical_identity_sha256: str | None = None


@dataclass(frozen=True)
class FileSource:
    """One hash-verified non-JSON resource or source table."""

    role: str
    path: Path
    file_sha256: str
    container_identity_sha256: str


@dataclass(frozen=True)
class BuildContext:
    """All mutually bound inputs needed for deterministic value derivation."""

    master_path: Path
    master_raw: bytes
    master_text: str
    evidence_graph: JsonSource
    evidence_artifacts: Mapping[str, JsonSource]
    figure_inputs: JsonSource
    figure_assembly_summary: JsonSource
    model_contract_proposal: JsonSource
    human_metadata: JsonSource
    human_values: Mapping[str, str]
    model_bundle_manifest: JsonSource
    clean_install_receipt: JsonSource
    source_release_manifest: JsonSource
    source_release_metadata: JsonSource
    software_release_cross_binding: Mapping[str, Any]
    narrative_decision: Mapping[str, Any]
    model_bundle_id: str
    root_expert_id: str
    root_bundle_identity_sha256: str
    hair_identity_count_expert: str
    resources: Mapping[str, FileSource]
    source_inputs: Mapping[str, FileSource]
    provenance_receipts: Mapping[str, JsonSource]


def validate_wt_secondary_source_inputs(context: BuildContext) -> dict[str, Any]:
    """Validate the hash-closed WT secondary family without creating tokens.

    These tables are carried by the values receipt through its source-file
    registry and by Table S9 as typed supplementary blocks.  They are not a
    ``FINAL_WT_*`` manuscript-token family and cannot alter the D15 branch or
    fixed 15-effect analysis.
    """

    require(
        not any(
            token.startswith("FINAL_WT")
            for token in TOKEN_PATTERN.findall(context.master_text)
        ),
        "WT secondary evidence must not create FINAL_WT manuscript tokens",
    )
    rows: dict[str, list[dict[str, str]]] = {}
    table_sha256: dict[str, str] = {}
    for role in WT_SECONDARY_RESOURCE_ROLES:
        require(
            role in context.source_inputs and role in context.resources,
            f"WT secondary source/resource role missing: {role}",
        )
        source = context.source_inputs[role]
        resource = context.resources[role]
        require(
            source.file_sha256 == resource.file_sha256
            and sha256_file(source.path) == source.file_sha256
            and sha256_file(resource.path) == resource.file_sha256,
            f"WT secondary source/resource hash mismatch: {role}",
        )
        try:
            with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                require(
                    reader.fieldnames is not None
                    and len(reader.fieldnames) == len(set(reader.fieldnames)),
                    f"WT secondary CSV header is missing or duplicated: {role}",
                )
                rows[role] = [dict(record) for record in reader]
        except (OSError, UnicodeError, csv.Error) as error:
            raise ManuscriptValuesError(
                f"WT secondary CSV cannot be read: {role}"
            ) from error
        table_sha256[role] = source.file_sha256
    analysis = context.evidence_artifacts.get("analysis")
    require(analysis is not None, "WT secondary analysis receipt is missing")
    require(
        analysis.payload.get("D15_fixed_effect_rows") == 15
        and analysis.payload.get(
            "D15_fixed_effect_family_changed_by_WT_secondary"
        )
        is False,
        "WT secondary evidence changed the D15 15-effect family",
    )
    try:
        evidence = validate_wt_secondary_evidence(
            contrasts=rows["wt_within_experiment_contrasts"],
            meta=rows["wt_within_day_meta_analysis"],
            flow=rows["wt_temperature_qc_flow"],
        )
        binding = validate_wt_secondary_analysis_binding(
            analysis_summary=analysis.payload,
            evidence_summary=evidence,
            table_sha256=table_sha256,
        )
    except ValueError as error:
        raise ManuscriptValuesError(
            f"WT secondary values-source validation failed: {error}"
        ) from error
    return {
        **binding,
        "typed_blocks": {
            "wt_gate_flow": "wt_temperature_qc_flow",
            "wt_experiment_contrasts": "wt_within_experiment_contrasts",
            "wt_same_day_meta": "wt_within_day_meta_analysis",
        },
        "D15_fixed_effect_rows": 15,
        "D15_narrative_branch_changed": False,
        "FINAL_WT_tokens_created": False,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptValuesError(message)


def token_rule(token: str) -> tuple[str, str, tuple[str, ...]]:
    for family, patterns, source_role, evidence_roles, exact in TOKEN_FAMILY_RULES:
        matched = token in patterns if exact else token.startswith(patterns)
        if matched:
            return family, source_role, tuple(evidence_roles)
    raise ManuscriptValuesError(f"no token-family/source-role contract for {token}")


def build_token_source_contract(master_text: str) -> dict[str, Any]:
    tokens = sorted(set(TOKEN_PATTERN.findall(master_text)))
    require(tokens, "master manuscript contains no machine-fill tokens")
    rows: dict[str, Any] = {}
    for token in tokens:
        family, source_role, evidence_roles = token_rule(token)
        rows[token] = {
            "family": family,
            "source_role": source_role,
            "required_evidence_roles": list(evidence_roles),
        }
    contract: dict[str, Any] = {
        "schema_version": "PHAxis-manuscript-token-source-contract-1.0",
        "tokens": rows,
    }
    contract["contract_identity_sha256"] = sha256_json(contract)
    return contract


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManuscriptValuesError("payload is not canonical finite JSON") from error


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_bytes(path: str | Path, role: str) -> tuple[Path, bytes]:
    resolved = Path(path).resolve()
    require(resolved.is_file(), f"{role}: missing file: {resolved}")
    require(not resolved.is_symlink(), f"{role}: symlink inputs are forbidden")
    require("blind" not in str(resolved).casefold(), f"{role}: blind-labelled path refused")
    try:
        return resolved, resolved.read_bytes()
    except OSError as error:
        raise ManuscriptValuesError(f"{role}: cannot read file: {resolved}") from error


def read_json_object(path: str | Path, role: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved, raw = read_bytes(path, role)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ManuscriptValuesError(
            f"{role}: strict UTF-8 JSON object without duplicate keys or NaN required"
        ) from error
    require(isinstance(payload, dict), f"{role}: JSON root must be an object")
    return resolved, raw, payload


def _json_source(
    path: str | Path,
    role: str,
    *,
    identity_field: str | None = None,
) -> JsonSource:
    resolved, raw, payload = read_json_object(path, role)
    identity = (
        validate_sealed_identity(payload, identity_field, role)
        if identity_field is not None
        else None
    )
    return JsonSource(
        role=role,
        path=resolved,
        raw=raw,
        payload=payload,
        file_sha256=sha256_bytes(raw),
        logical_identity_sha256=identity,
    )


def walk(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            path = f"{prefix}.{key}" if prefix else str(key)
            value = payload[key]
            yield path, value
            yield from walk(value, path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]"
            yield path, value
            yield from walk(value, path)


def validate_sealed_identity(
    payload: Mapping[str, Any], identity_field: str, role: str
) -> str:
    identity = payload.get(identity_field)
    require(is_sha256(identity), f"{role}: missing or invalid {identity_field}")
    unsigned = deepcopy(dict(payload))
    unsigned.pop(identity_field, None)
    require(
        sha256_json(unsigned) == identity,
        f"{role}: {identity_field} does not seal the complete payload",
    )
    return str(identity)


def guard_final_payload(
    role: str,
    payload: Mapping[str, Any],
    *,
    historical_authority: bool = False,
) -> None:
    """Enforce blind/root-cap/finality guards recursively.

    ``historical_authority`` allows a source to describe an explicitly labelled
    development/comparator role.  It never relaxes blind, root-cap, placeholder,
    or provisional guards and never permits that source to satisfy ``FINAL_*``.
    """

    saw_blind_guard = False
    for path, value in walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if leaf == "blind_images_used" or leaf.endswith("blind_images_used"):
            saw_blind_guard = True
            require(value == 0, f"{role}: {path} must be 0")
        if leaf in {
            "root_cap_region_output",
            "root_cap_region_statistics_included",
            "rootcap_region_metric",
        }:
            require(value is False, f"{role}: {path} must be false")
        if leaf in {
            "canonical_annotations_read",
            "canonical_annotations_read_during_inference",
            "condition_metadata_used_for_routing",
            "condition_metadata_used_for_model_routing",
        }:
            require(value is False, f"{role}: forbidden route/read flag at {path}")
        if isinstance(value, float):
            require(math.isfinite(value), f"{role}: non-finite number at {path}")
        if isinstance(value, str):
            lowered = value.casefold()
            require(
                not any(marker in lowered for marker in FORBIDDEN_TEXT_MARKERS),
                f"{role}: non-final marker at {path}",
            )
            if not historical_authority:
                require(
                    not any(marker in lowered for marker in LEGACY_DEPLOYMENT_MARKERS),
                    f"{role}: legacy 443CV deployment marker at {path}",
                )
    require(saw_blind_guard, f"{role}: explicit blind_images_used=0 guard missing")


def _validate_scalar_text(token: str, value: Any) -> str:
    if not isinstance(value, str):
        raise HumanMetadataError(
            f"{token}: external metadata must be a non-empty string",
            invalid={token: "not_a_string"},
        )
    rendered = value.strip()
    if not rendered:
        raise HumanMetadataError(
            f"{token}: external metadata is empty",
            invalid={token: "empty"},
        )
    lowered = rendered.casefold()
    if lowered in FORBIDDEN_LITERAL_VALUES or any(
        marker in lowered
        for marker in (*FORBIDDEN_TEXT_MARKERS, *DEFERRED_HUMAN_METADATA_MARKERS)
    ):
        raise HumanMetadataError(
            f"{token}: placeholder/non-final external metadata is forbidden",
            invalid={token: "placeholder_or_nonfinal"},
        )
    if RESIDUAL_TOKEN_PATTERN.search(rendered):
        raise HumanMetadataError(
            f"{token}: residual manuscript token is forbidden",
            invalid={token: "residual_token"},
        )
    return rendered


def validate_human_metadata(payload: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    expected_top = {
        "schema_version",
        "status",
        "blind_images_used",
        "root_cap_region_statistics_included",
        "values",
        "human_metadata_identity_sha256",
    }
    require(set(payload) == expected_top, "human metadata top-level schema mismatch")
    require(payload.get("schema_version") == HUMAN_METADATA_SCHEMA, "human metadata schema changed")
    require(
        payload.get("status") == "complete_author_verified_external_metadata",
        "human metadata are not author-verified and complete",
    )
    require(payload.get("blind_images_used") == 0, "human metadata blind guard changed")
    require(
        payload.get("root_cap_region_statistics_included") is False,
        "human metadata root-cap guard changed",
    )
    identity = validate_sealed_identity(payload, "human_metadata_identity_sha256", "human metadata")
    entries = payload.get("values")
    require(isinstance(entries, Mapping), "human metadata values must be one object")
    missing = sorted(HUMAN_METADATA_TOKENS - set(entries))
    extra = sorted(set(entries) - HUMAN_METADATA_TOKENS)
    if missing or extra:
        raise HumanMetadataError(
            f"human metadata key mismatch; missing={missing}, extra={extra}",
            missing=missing,
            extra=extra,
        )
    rendered: dict[str, str] = {}
    invalid: dict[str, str] = {}
    for token in sorted(HUMAN_METADATA_TOKENS):
        try:
            value = _validate_scalar_text(token, entries[token])
            if token in URL_TOKENS and URL_PATTERN.fullmatch(value) is None:
                raise HumanMetadataError(
                    f"{token}: an absolute https URL is required",
                    invalid={token: "invalid_https_url"},
                )
            if token in DOI_TOKENS and DOI_PATTERN.fullmatch(value) is None:
                raise HumanMetadataError(
                    f"{token}: a DOI or https://doi.org DOI is required",
                    invalid={token: "invalid_doi"},
                )
            if token == "PHAXIS_RELEASE_TAG" and value != "v1.0.0":
                raise HumanMetadataError(
                    f"{token}: final public release tag must be v1.0.0",
                    invalid={token: "invalid_public_version_tag"},
                )
            rendered[token] = value
        except HumanMetadataError as error:
            invalid.update(error.invalid or {token: str(error)})
    if invalid:
        raise HumanMetadataError(
            f"invalid human metadata fields: {sorted(invalid)}",
            invalid=invalid,
        )
    return rendered, identity


def _resolve_manifest_file(
    manifest_path: Path,
    record: Mapping[str, Any],
    role: str,
    *,
    container_identity_sha256: str,
) -> FileSource:
    raw_path = record.get("path")
    expected = record.get("sha256")
    require(isinstance(raw_path, str) and raw_path, f"{role}: manifest path missing")
    require(is_sha256(expected), f"{role}: manifest SHA-256 missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    path, raw = read_bytes(candidate, role)
    observed = sha256_bytes(raw)
    require(observed == expected, f"{role}: source file SHA-256 drift")
    return FileSource(
        role=role,
        path=path,
        file_sha256=observed,
        container_identity_sha256=container_identity_sha256,
    )


def _validate_evidence_graph(
    evidence_graph: JsonSource,
    artifact_paths: Mapping[str, str | Path],
) -> dict[str, JsonSource]:
    graph = evidence_graph.payload
    require(graph.get("schema_version") == EVIDENCE_GRAPH_SCHEMA, "evidence graph schema changed")
    require(
        graph.get("status") == "passed_formal_evidence_graph"
        and graph.get("formal_release_evidence_closed") is True,
        "evidence graph is not final and closed",
    )
    guard_final_payload("evidence graph", graph)
    artifacts = graph.get("artifacts")
    require(isinstance(artifacts, Mapping), "evidence graph artifacts missing")
    require(
        set(artifacts) == set(EVIDENCE_ARTIFACT_ROLES),
        "evidence graph artifact roles are not exact",
    )
    require(
        set(artifact_paths) == set(EVIDENCE_ARTIFACT_ROLES),
        "named evidence artifact paths are not exact",
    )
    loaded: dict[str, JsonSource] = {}
    for role in EVIDENCE_ARTIFACT_ROLES:
        record = artifacts[role]
        require(isinstance(record, Mapping), f"{role}: evidence artifact record malformed")
        source = _json_source(artifact_paths[role], f"evidence artifact {role}")
        require(
            source.file_sha256 == record.get("source_file_sha256"),
            f"{role}: evidence source file SHA-256 mismatch",
        )
        identity_field = record.get("primary_identity_field")
        if identity_field is not None:
            require(isinstance(identity_field, str) and identity_field, f"{role}: invalid identity field")
            identity = validate_sealed_identity(source.payload, identity_field, role)
            require(
                identity == record.get("primary_identity_sha256"),
                f"{role}: evidence logical identity mismatch",
            )
            source = JsonSource(
                role=source.role,
                path=source.path,
                raw=source.raw,
                payload=source.payload,
                file_sha256=source.file_sha256,
                logical_identity_sha256=identity,
            )
        guard_final_payload(role, source.payload)
        loaded[role] = source
    return loaded


def _load_source_release_authority(
    source_release_manifest: str | Path,
    *,
    human_values: Mapping[str, str],
) -> tuple[JsonSource, JsonSource, dict[str, Any]]:
    """Cross-bind public manuscript coordinates to one formal source tree."""

    manifest = _json_source(source_release_manifest, "source release manifest")
    payload = manifest.payload
    require(
        payload.get("schema_version") == "PHAxis-source-release-manifest-2.0"
        and payload.get("distribution") == "phaxis"
        and payload.get("version") == "1.0.0"
        and payload.get("release_mode") == "formal",
        "source release manifest is not the formal PHAxis 1.0.0 authority",
    )
    require(
        payload.get("source_policy") == "explicit_path_bounded_allowlist",
        "source release manifest policy changed",
    )
    files = payload.get("files")
    require(isinstance(files, list) and files, "source release manifest files missing")
    require(
        payload.get("tree_identity_sha256") == sha256_json(files),
        "source release tree identity mismatch",
    )
    records: dict[str, Mapping[str, Any]] = {}
    for record in files:
        require(isinstance(record, Mapping), "source release file record malformed")
        relative = record.get("path")
        require(
            isinstance(relative, str) and relative and relative not in records,
            "source release paths invalid",
        )
        require(is_sha256(record.get("sha256")), f"source release file SHA missing: {relative}")
        records[relative] = record
    for required in ("RELEASE_HUMAN_METADATA.json", "LICENSE", "pyproject.toml", "CITATION.cff"):
        require(required in records, f"source release authority is missing {required}")

    release_root = manifest.path.parent.resolve()

    def release_file(relative: str) -> Path:
        pure = PurePosixPath(relative)
        require(
            not pure.is_absolute()
            and bool(pure.parts)
            and all(part not in {"", ".", ".."} for part in pure.parts),
            f"source release path is unsafe: {relative}",
        )
        unresolved = release_root.joinpath(*pure.parts)
        require(not unresolved.is_symlink(), f"source release file may not be a symlink: {relative}")
        candidate = unresolved.resolve()
        require(
            candidate.parent == release_root or release_root in candidate.parents,
            f"source release path escapes its root: {relative}",
        )
        require(
            candidate.is_file(),
            f"source release file is missing: {relative}",
        )
        require(
            sha256_file(candidate) == records[relative].get("sha256"),
            f"source release file SHA differs from manifest: {relative}",
        )
        declared_bytes = records[relative].get("bytes")
        if declared_bytes is not None:
            require(
                isinstance(declared_bytes, int)
                and declared_bytes >= 0
                and candidate.stat().st_size == declared_bytes,
                f"source release file size differs from manifest: {relative}",
            )
        return candidate

    required_paths = {
        relative: release_file(relative)
        for relative in (
            "RELEASE_HUMAN_METADATA.json",
            "LICENSE",
            "pyproject.toml",
            "CITATION.cff",
        )
    }

    metadata = _json_source(
        required_paths["RELEASE_HUMAN_METADATA.json"],
        "source release human metadata",
        identity_field="metadata_identity_sha256",
    )
    require(
        metadata.file_sha256 == records["RELEASE_HUMAN_METADATA.json"].get("sha256"),
        "source release metadata hash differs from source manifest",
    )
    release = metadata.payload
    require(
        release.get("schema_version") == "PHAxis-release-human-metadata-1.3"
        and release.get("status") == "author_verified_release_authority"
        and release.get("product") == "PHAxis"
        and release.get("product_version") == "1.0.0"
        and release.get("distribution") == "phaxis",
        "source release human metadata is not a final PHAxis 1.0.0 authority",
    )
    project_urls = release.get("project_urls")
    coordinates = release.get("release_coordinates")
    rights = release.get("rights")
    require(
        isinstance(project_urls, Mapping)
        and isinstance(coordinates, Mapping)
        and isinstance(rights, Mapping),
        "source release public coordinate blocks are missing",
    )
    binding: dict[str, Any] = {
        "repository_url": project_urls.get("Repository"),
        "release_tag": coordinates.get("github_release_tag"),
        "version": payload.get("version"),
        "release_doi": coordinates.get("release_doi"),
        "software_license": rights.get("source_license_spdx"),
        "source_release_tree_identity_sha256": payload.get("tree_identity_sha256"),
        "source_release_manifest_sha256": manifest.file_sha256,
        "release_metadata_identity_sha256": metadata.logical_identity_sha256,
        "release_metadata_sha256": metadata.file_sha256,
        "license_file_sha256": rights.get("license_file_sha256"),
        "pyproject_sha256": records["pyproject.toml"].get("sha256"),
        "citation_cff_sha256": records["CITATION.cff"].get("sha256"),
    }
    require(
        binding["repository_url"] == coordinates.get("github_repository_url")
        == human_values.get("PHAXIS_REPOSITORY_URL"),
        "repository URL differs across manuscript and source release",
    )
    require(
        binding["release_tag"] == f"v{binding['version']}"
        == human_values.get("PHAXIS_RELEASE_TAG"),
        "release tag/version differs across manuscript and source release",
    )
    require(
        binding["release_doi"] == human_values.get("PHAXIS_RELEASE_DOI"),
        "release DOI differs across manuscript and source release",
    )
    require(
        binding["software_license"] == human_values.get("PHAXIS_SOFTWARE_LICENSE"),
        "software license differs across manuscript and source release",
    )
    require(
        rights.get("source_release_authorized") is True
        and rights.get("license_file_sha256") == records["LICENSE"].get("sha256"),
        "source release license authority is not hash-closed",
    )
    try:
        pyproject = tomllib.loads(
            required_paths["pyproject.toml"].read_text(encoding="utf-8")
        )
        project = pyproject["project"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ManuscriptValuesError("source release pyproject.toml is invalid") from error
    require(
        isinstance(project, Mapping)
        and project.get("name") == "phaxis"
        and project.get("version") == binding["version"]
        and project.get("license") == binding["software_license"]
        and isinstance(project.get("urls"), Mapping)
        and project["urls"].get("Repository") == binding["repository_url"],
        "pyproject public coordinates differ from release authority",
    )

    try:
        citation = required_paths["CITATION.cff"].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ManuscriptValuesError("source release CITATION.cff is unreadable") from error

    def citation_scalar(key: str) -> str | None:
        match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", citation, flags=re.MULTILINE)
        if match is None:
            return None
        raw = match.group(1)
        if raw.startswith('"'):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ManuscriptValuesError(
                    f"source release CITATION.cff has an invalid {key} scalar"
                ) from error
            return value if isinstance(value, str) else None
        return raw

    require(
        citation_scalar("version") == binding["version"]
        and citation_scalar("license") == binding["software_license"]
        and citation_scalar("repository-code") == binding["repository_url"]
        and citation_scalar("doi") == binding["release_doi"],
        "CITATION.cff public coordinates differ from release authority",
    )
    binding["cross_binding_identity_sha256"] = sha256_json(binding)
    manifest = JsonSource(
        role=manifest.role,
        path=manifest.path,
        raw=manifest.raw,
        payload=manifest.payload,
        file_sha256=manifest.file_sha256,
        logical_identity_sha256=str(payload["tree_identity_sha256"]),
    )
    return manifest, metadata, binding


def load_build_context(
    *,
    master: str | Path,
    evidence_graph: str | Path,
    evidence_artifacts: Mapping[str, str | Path],
    figure_inputs: str | Path,
    figure_assembly_summary: str | Path,
    model_contract_proposal: str | Path,
    human_metadata: str | Path,
    model_bundle_manifest: str | Path,
    clean_install_receipt: str | Path,
    source_release_manifest: str | Path,
) -> BuildContext:
    """Load and mutually validate every source used by the values assembler."""

    master_path, master_raw = read_bytes(master, "master manuscript")
    try:
        master_text = master_raw.decode("utf-8")
    except UnicodeError as error:
        raise ManuscriptValuesError("master manuscript must be UTF-8") from error
    tokens = set(TOKEN_PATTERN.findall(master_text))
    require(tokens, "master manuscript contains no machine-fill tokens")
    require(
        not (HUMAN_METADATA_TOKENS - tokens),
        "master manuscript is missing required human-metadata tokens",
    )

    graph = _json_source(
        evidence_graph,
        "evidence graph",
        identity_field="manifest_identity_sha256",
    )
    artifacts = _validate_evidence_graph(graph, evidence_artifacts)

    proposal = _json_source(
        model_contract_proposal,
        "model-contract proposal",
        identity_field="model_contract_identity_sha256",
    )
    require(
        proposal.path == artifacts["model_contract_proposal"].path,
        "model-contract proposal must be the named evidence artifact",
    )
    require(
        proposal.file_sha256 == graph.payload.get("model_contract_proposal_sha256")
        and proposal.logical_identity_sha256
        == graph.payload.get("model_contract_proposal_identity_sha256"),
        "evidence graph/model-contract proposal binding mismatch",
    )
    require(
        proposal.payload.get("formal_release_status") == "passed_proposal_not_official",
        "model-contract proposal is not a passed unapplied proposal",
    )
    promotion = proposal.payload.get("promotion")
    require(
        isinstance(promotion, Mapping)
        and promotion.get("schema_version") == "PHAxis-model-contract-promotion-1.0"
        and promotion.get("status") == "validated_proposal_not_applied"
        and promotion.get("official_apply_performed") is False,
        "model-contract proposal promotion guard changed",
    )
    guard_final_payload("model-contract proposal", proposal.payload)
    stageb_binding = promotion.get("stageb_binding")
    require(isinstance(stageb_binding, Mapping), "model-contract proposal Stage-B binding missing")
    hair_expert_id = stageb_binding.get("expert_id")
    require(
        isinstance(hair_expert_id, str)
        and hair_expert_id == "PHAxis-StageB-train399-five-seed",
        "model-contract proposal hair expert is not the neutral final train399 identity",
    )
    try:
        canonical_public_identity = validate_proposal_public_identity(proposal.payload)
    except ContractError as error:
        raise ManuscriptValuesError(
            "model-contract proposal public identity is not canonically derived"
        ) from error
    public_model_bundle_id = canonical_public_identity["model_bundle_id"]
    root_expert = proposal.payload.get("root_expert")
    provider_role = canonical_public_identity["root_provider_role"]
    public_root_expert_id = canonical_public_identity["root_expert_id"]
    root_bundle_identity = (
        root_expert.get("bundle_identity_sha256")
        if isinstance(root_expert, Mapping)
        else None
    )
    root_bundle_authority = (
        root_expert.get("root_bundle_authority")
        if isinstance(root_expert, Mapping)
        else None
    )
    require(
        isinstance(public_model_bundle_id, str)
        and public_model_bundle_id.startswith(MODEL_BUNDLE_PREFIX)
        and proposal.payload.get("model_bundle_id") == public_model_bundle_id
        and provider_role == ROOT_PROVIDER_ROLE
        and isinstance(root_expert, Mapping)
        and root_expert.get("provider_role") == provider_role
        and isinstance(public_root_expert_id, str)
        and public_root_expert_id.startswith(ROOT_EXPERT_PREFIX)
        and root_expert.get("expert_id") == public_root_expert_id
        and is_sha256(root_bundle_identity)
        and isinstance(root_bundle_authority, Mapping)
        and root_bundle_authority.get("bundle_identity_sha256")
        == root_bundle_identity,
        "model-contract proposal public model/root identity is invalid",
    )
    fusion = artifacts["fusion"].payload
    traits = artifacts["traits"].payload
    require(
        fusion.get("hair_identity_count_expert") == hair_expert_id
        and traits.get("hair_identity_count_expert") == hair_expert_id,
        "proposal/fusion/traits hair expert mismatch",
    )
    require(
        fusion.get("model_bundle_id") == public_model_bundle_id
        and fusion.get("root_expert") == public_root_expert_id
        and traits.get("model_bundle_id") == public_model_bundle_id
        and traits.get("root_expert_id") == public_root_expert_id,
        "proposal/fusion/traits public model/root identity mismatch",
    )

    inputs = _json_source(
        figure_inputs,
        "figure inputs",
        identity_field="figure_input_assembly_identity_sha256",
    )
    require(
        inputs.path == artifacts["figure_inputs"].path
        and inputs.file_sha256 == artifacts["figure_inputs"].file_sha256
        and inputs.logical_identity_sha256
        == artifacts["figure_inputs"].logical_identity_sha256,
        "figure inputs must be the named evidence artifact",
    )
    require(inputs.payload.get("schema_version") == FIGURE_INPUT_SCHEMA, "figure input schema changed")
    require(
        inputs.payload.get("assembler_schema_version") == FIGURE_ASSEMBLER_SCHEMA,
        "figure inputs were not made by the production assembler",
    )
    require(inputs.payload.get("status") == "final", "figure inputs are not final")
    guard_final_payload("figure inputs", inputs.payload)
    require(
        inputs.payload.get("model_contract_proposal_sha256") == proposal.file_sha256
        and inputs.payload.get("model_contract_proposal_identity_sha256")
        == proposal.logical_identity_sha256,
        "figure inputs/model-contract proposal binding mismatch",
    )
    require(
        inputs.payload.get("model_contract_public_identity")
        == {
            "model_bundle_id": public_model_bundle_id,
            "root_expert_id": public_root_expert_id,
            "root_provider_role": provider_role,
        },
        "figure inputs/model-contract public identity mismatch",
    )
    expected_core_hashes = {
        role: artifacts[role].file_sha256 for role in CORE_FIGURE_SOURCE_ROLES
    }
    require(
        inputs.payload.get("source_summary_sha256") == expected_core_hashes,
        "figure inputs/evidence graph core source closure mismatch",
    )

    assembly = _json_source(figure_assembly_summary, "figure assembly summary")
    require(
        assembly.payload.get("schema_version") == FIGURE_ASSEMBLER_SCHEMA,
        "figure assembly summary schema changed",
    )
    require(
        assembly.payload.get("status") == "completed_final",
        "figure assembly summary is not final",
    )
    guard_final_payload("figure assembly summary", assembly.payload)
    require(
        assembly.payload.get("figure_inputs_sha256") == inputs.file_sha256
        and assembly.payload.get("figure_input_assembly_identity_sha256")
        == inputs.logical_identity_sha256,
        "figure assembly summary does not bind figure inputs",
    )

    figures = artifacts["figures"].payload
    require(
        figures.get("figure_input_manifest_sha256") == inputs.file_sha256
        and figures.get("figure_input_assembly_identity_sha256")
        == inputs.logical_identity_sha256,
        "final figure suite does not bind figure inputs",
    )

    resource_records = inputs.payload.get("resources")
    require(
        isinstance(resource_records, Mapping)
        and set(resource_records) == set(FIGURE_RESOURCE_ROLES),
        "figure resource roles are not the exact 22-role narrative/assignment/audit contract",
    )
    resources: dict[str, FileSource] = {}
    for role in sorted(resource_records):
        record = resource_records[role]
        require(isinstance(record, Mapping), f"resource {role}: record malformed")
        resources[role] = _resolve_manifest_file(
            inputs.path,
            record,
            f"figure resource {role}",
            container_identity_sha256=str(inputs.logical_identity_sha256),
        )
    require(
        assembly.payload.get("resource_sha256")
        == {role: source.file_sha256 for role, source in resources.items()},
        "figure assembly summary resource hash map mismatch",
    )
    decision_source = _json_source(
        resources["narrative_decision"].path,
        "figure narrative decision",
    )
    try:
        decision = validate_narrative_decision(decision_source.payload)
    except ValueError as error:
        raise ManuscriptValuesError("figure narrative decision is invalid") from error
    decision_identity = decision["narrative_decision_identity_sha256"]
    require(
        inputs.payload.get("narrative_decision_identity_sha256") == decision_identity
        and inputs.payload.get("narrative_branch_id") == decision["branch_id"]
        and assembly.payload.get("narrative_decision_identity_sha256") == decision_identity
        and assembly.payload.get("narrative_branch_id") == decision["branch_id"]
        and figures.get("narrative_decision_identity_sha256") == decision_identity
        and figures.get("narrative_branch_id") == decision["branch_id"]
        and figures.get("title_contract") == title_contract(decision),
        "stage36/stage37 narrative-decision or title contract differs",
    )

    source_records = inputs.payload.get("source_inputs")
    require(
        isinstance(source_records, Mapping)
        and set(source_records) == set(FIGURE_SOURCE_INPUT_ROLES),
        "figure source-input roles are not exact",
    )
    source_inputs: dict[str, FileSource] = {}
    for role in sorted(source_records):
        record = source_records[role]
        require(isinstance(record, Mapping), f"source input {role}: record malformed")
        source_inputs[role] = _resolve_manifest_file(
            inputs.path,
            record,
            f"figure source input {role}",
            container_identity_sha256=str(inputs.logical_identity_sha256),
        )

    provenance_records = inputs.payload.get("provenance_receipts")
    require(
        isinstance(provenance_records, Mapping)
        and set(provenance_records) == set(FIGURE_PROVENANCE_ROLES),
        "figure provenance receipt roles are not exact",
    )
    provenance: dict[str, JsonSource] = {}
    for role in sorted(provenance_records):
        record = provenance_records[role]
        require(isinstance(record, Mapping), f"provenance {role}: record malformed")
        source_file = _resolve_manifest_file(
            inputs.path,
            record,
            f"figure provenance {role}",
            container_identity_sha256=str(inputs.logical_identity_sha256),
        )
        identity_field = record.get("identity_field")
        require(isinstance(identity_field, str) and identity_field, f"{role}: identity field missing")
        source = _json_source(source_file.path, f"figure provenance {role}")
        identity = validate_sealed_identity(source.payload, identity_field, role)
        require(identity == record.get("identity_sha256"), f"{role}: provenance identity mismatch")
        guard_final_payload(
            f"figure provenance {role}",
            source.payload,
            historical_authority=role == "historical_development",
        )
        provenance[role] = JsonSource(
            role=source.role,
            path=source.path,
            raw=source.raw,
            payload=source.payload,
            file_sha256=source.file_sha256,
            logical_identity_sha256=identity,
        )

    human = _json_source(
        human_metadata,
        "human metadata",
        identity_field="human_metadata_identity_sha256",
    )
    human_values, human_identity = validate_human_metadata(human.payload)
    require(human.logical_identity_sha256 == human_identity, "human metadata identity drift")
    source_release, source_release_metadata, software_release_binding = (
        _load_source_release_authority(
            source_release_manifest,
            human_values=human_values,
        )
    )

    bundle = _json_source(
        model_bundle_manifest,
        "model bundle manifest",
        identity_field="model_bundle_manifest_identity_sha256",
    )
    require(
        bundle.payload.get("schema_version") == MODEL_BUNDLE_MANIFEST_SCHEMA
        and bundle.payload.get("status") == "completed_final_immutable_bundle",
        "model bundle manifest is not final",
    )
    guard_final_payload("model bundle manifest", bundle.payload)
    require(
        bundle.payload.get("model_contract_proposal_sha256") == proposal.file_sha256
        and bundle.payload.get("model_contract_proposal_identity_sha256")
        == proposal.logical_identity_sha256,
        "model bundle/proposal binding mismatch",
    )
    require(
        bundle.payload.get("model_bundle_id") == public_model_bundle_id
        and bundle.payload.get("root_expert_id") == public_root_expert_id
        and bundle.payload.get("root_bundle_identity_sha256")
        == root_bundle_identity
        and bundle.payload.get("hair_identity_count_expert") == hair_expert_id,
        "model bundle public identity differs from proposal",
    )
    members = bundle.payload.get("members")
    require(isinstance(members, list) and members, "model bundle members missing")
    require(bundle.payload.get("member_count") == len(members), "model bundle member count mismatch")
    require(is_sha256(bundle.payload.get("bundle_sha256")), "model bundle SHA-256 missing")
    require(
        isinstance(bundle.payload.get("bundle_size_bytes"), int)
        and bundle.payload["bundle_size_bytes"] > 0,
        "model bundle size missing",
    )

    clean = _json_source(
        clean_install_receipt,
        "clean install receipt",
        identity_field="clean_install_receipt_identity_sha256",
    )
    require(
        clean.payload.get("schema_version") == CLEAN_INSTALL_RECEIPT_SCHEMA
        and clean.payload.get("status") == "completed_final_clean_install",
        "clean install verification is not final",
    )
    guard_final_payload("clean install receipt", clean.payload)
    require(
        clean.payload.get("model_contract_proposal_sha256") == proposal.file_sha256
        and clean.payload.get("model_contract_proposal_identity_sha256")
        == proposal.logical_identity_sha256,
        "clean install/proposal binding mismatch",
    )
    require(
        clean.payload.get("model_bundle_id") == public_model_bundle_id
        and clean.payload.get("root_expert_id") == public_root_expert_id
        and clean.payload.get("root_bundle_identity_sha256")
        == root_bundle_identity
        and clean.payload.get("hair_identity_count_expert") == hair_expert_id,
        "clean install public identity differs from proposal",
    )
    require(
        is_sha256(clean.payload.get("example_output_identity_sha256")),
        "clean install example output identity missing",
    )

    return BuildContext(
        master_path=master_path,
        master_raw=master_raw,
        master_text=master_text,
        evidence_graph=graph,
        evidence_artifacts=artifacts,
        figure_inputs=inputs,
        figure_assembly_summary=assembly,
        model_contract_proposal=proposal,
        human_metadata=human,
        human_values=human_values,
        model_bundle_manifest=bundle,
        clean_install_receipt=clean,
        source_release_manifest=source_release,
        source_release_metadata=source_release_metadata,
        software_release_cross_binding=software_release_binding,
        narrative_decision=decision,
        model_bundle_id=public_model_bundle_id,
        root_expert_id=public_root_expert_id,
        root_bundle_identity_sha256=str(root_bundle_identity),
        hair_identity_count_expert=hair_expert_id,
        resources=resources,
        source_inputs=source_inputs,
        provenance_receipts=provenance,
    )


def human_metadata_template() -> dict[str, Any]:
    """Return an intentionally invalid, clearly labelled completion template."""

    return {
        "schema_version": HUMAN_METADATA_SCHEMA,
        "status": "INCOMPLETE_DO_NOT_USE",
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "values": {
            token: (
                "DEFERRED_AUTHOR_VERIFICATION"
                if token in BIOLOGICAL_ACQUISITION_TOKENS
                else None
            )
            for token in sorted(HUMAN_METADATA_TOKENS)
        },
        "human_metadata_identity_sha256": None,
    }


def human_metadata_report(error: HumanMetadataError) -> dict[str, Any]:
    return {
        "schema_version": HUMAN_METADATA_REPORT_SCHEMA,
        "status": "blocked_missing_or_invalid_human_metadata",
        "missing_fields": list(error.missing),
        "extra_fields": list(error.extra),
        "invalid_fields": dict(error.invalid),
        "required_fields": sorted(HUMAN_METADATA_TOKENS),
        "template": human_metadata_template(),
        "formal_values_build_allowed": False,
    }


def source_cell_identity(
    *,
    source_role: str,
    file_sha256: str,
    locator: Mapping[str, Any],
    value: Any,
) -> str:
    require(is_sha256(file_sha256), f"{source_role}: invalid source file SHA-256")
    return sha256_json(
        {
            "source_role": source_role,
            "source_file_sha256": file_sha256,
            "locator": dict(locator),
            "value": value,
        }
    )


def seal_derivation(derivation: Mapping[str, Any]) -> dict[str, Any]:
    sealed = deepcopy(dict(derivation))
    sealed.pop("derivation_identity_sha256", None)
    sealed["derivation_identity_sha256"] = sha256_json(sealed)
    return sealed


def derivation_source(
    *,
    source_role: str,
    source_file_sha256: str,
    container_identity_sha256: str,
    locator: Mapping[str, Any],
    source_value: Any,
    authority_class: str,
    source_logical_identity_sha256: str | None = None,
) -> dict[str, Any]:
    require(is_sha256(container_identity_sha256), f"{source_role}: container identity missing")
    record: dict[str, Any] = {
        "source_role": source_role,
        "source_file_sha256": source_file_sha256,
        "container_identity_sha256": container_identity_sha256,
        "locator": deepcopy(dict(locator)),
        "source_value": source_value,
        "authority_class": authority_class,
    }
    if source_logical_identity_sha256 is not None:
        require(
            is_sha256(source_logical_identity_sha256),
            f"{source_role}: source logical identity missing",
        )
        record["source_logical_identity_sha256"] = source_logical_identity_sha256
    record["source_cell_identity_sha256"] = source_cell_identity(
        source_role=source_role,
        file_sha256=source_file_sha256,
        locator=record["locator"],
        value=source_value,
    )
    return record


def _source_file_registry(context: BuildContext) -> dict[str, Any]:
    records: dict[str, Any] = {}

    def add(role: str, source: JsonSource | FileSource) -> None:
        require(role not in records, f"duplicate source-file registry role: {role}")
        record: dict[str, Any] = {"sha256": source.file_sha256}
        if isinstance(source, JsonSource) and source.logical_identity_sha256 is not None:
            record["logical_identity_sha256"] = source.logical_identity_sha256
        if isinstance(source, FileSource):
            record["container_identity_sha256"] = source.container_identity_sha256
        records[role] = record

    add("evidence_graph", context.evidence_graph)
    for role, source in sorted(context.evidence_artifacts.items()):
        add(f"evidence_artifact:{role}", source)
    add("figure_inputs", context.figure_inputs)
    add("figure_assembly_summary", context.figure_assembly_summary)
    for role, source in sorted(context.resources.items()):
        add(f"figure_resource:{role}", source)
    for role, source in sorted(context.source_inputs.items()):
        add(f"figure_source_input:{role}", source)
    for role, source in sorted(context.provenance_receipts.items()):
        add(f"figure_provenance:{role}", source)
    add("human_metadata", context.human_metadata)
    add("source_release_manifest", context.source_release_manifest)
    add("source_release_metadata", context.source_release_metadata)
    add("model_bundle_manifest", context.model_bundle_manifest)
    add("clean_install_receipt", context.clean_install_receipt)
    return records


def assemble_values_payload(
    *,
    context: BuildContext,
    token_contract: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble and self-seal a complete values payload from derived entries."""

    contract_rows = token_contract.get("tokens")
    require(isinstance(contract_rows, Mapping), "token source contract is malformed")
    missing = sorted(set(contract_rows) - set(entries))
    extra = sorted(set(entries) - set(contract_rows))
    require(not missing and not extra, f"token key mismatch; missing={missing}, extra={extra}")
    normalized_entries = deepcopy(dict(entries))
    derivation_contract: dict[str, Any] = {}
    for token in sorted(contract_rows):
        entry = normalized_entries[token]
        require(isinstance(entry, Mapping), f"{token}: derived entry malformed")
        require(set(entry) == {"value", "source_role", "derivation"}, f"{token}: derived entry schema mismatch")
        require(
            entry.get("source_role") == contract_rows[token].get("source_role"),
            f"{token}: derived entry source role mismatch",
        )
        derivation = entry.get("derivation")
        require(isinstance(derivation, Mapping), f"{token}: derivation missing")
        validate_derivation(derivation, token)
        derivation_contract[token] = {
            "operation": derivation["operation"],
            "source_roles": [source["source_role"] for source in derivation["sources"]],
            "locators": [source["locator"] for source in derivation["sources"]],
        }
    selected_decision = validate_narrative_decision(context.narrative_decision)
    publication_titles = title_contract(selected_decision)
    payload: dict[str, Any] = {
        "schema_version": VALUES_SCHEMA,
        "builder_schema_version": VALUES_BUILDER_SCHEMA,
        "status": "final_values_machine_derived_locked",
        "master_sha256": sha256_bytes(context.master_raw),
        "evidence_graph_file_sha256": context.evidence_graph.file_sha256,
        "evidence_graph_identity_sha256": context.evidence_graph.logical_identity_sha256,
        "figure_inputs_file_sha256": context.figure_inputs.file_sha256,
        "figure_input_assembly_identity_sha256": context.figure_inputs.logical_identity_sha256,
        "figure_assembly_summary_file_sha256": context.figure_assembly_summary.file_sha256,
        "model_contract_proposal_sha256": context.model_contract_proposal.file_sha256,
        "model_contract_proposal_identity_sha256": context.model_contract_proposal.logical_identity_sha256,
        "model_bundle_id": context.model_bundle_id,
        "root_expert_id": context.root_expert_id,
        "root_bundle_identity_sha256": context.root_bundle_identity_sha256,
        "hair_identity_count_expert": context.hair_identity_count_expert,
        "human_metadata_file_sha256": context.human_metadata.file_sha256,
        "human_metadata_identity_sha256": context.human_metadata.logical_identity_sha256,
        "model_bundle_manifest_file_sha256": context.model_bundle_manifest.file_sha256,
        "model_bundle_manifest_identity_sha256": context.model_bundle_manifest.logical_identity_sha256,
        "clean_install_receipt_file_sha256": context.clean_install_receipt.file_sha256,
        "clean_install_receipt_identity_sha256": context.clean_install_receipt.logical_identity_sha256,
        "source_release_manifest_file_sha256": context.source_release_manifest.file_sha256,
        "source_release_tree_identity_sha256": context.source_release_manifest.logical_identity_sha256,
        "source_release_metadata_file_sha256": context.source_release_metadata.file_sha256,
        "source_release_metadata_identity_sha256": context.source_release_metadata.logical_identity_sha256,
        "software_release_cross_binding_identity_sha256": context.software_release_cross_binding[
            "cross_binding_identity_sha256"
        ],
        "narrative_decision_identity_sha256": selected_decision[
            "narrative_decision_identity_sha256"
        ],
        "narrative_branch_id": selected_decision["branch_id"],
        "publication_title_contract": publication_titles,
        "token_contract_identity_sha256": token_contract.get("contract_identity_sha256"),
        "token_derivation_contract_identity_sha256": sha256_json(derivation_contract),
        "source_files": _source_file_registry(context),
        "historical_source_policy": {
            "allowed_token_prefixes": list(HISTORICAL_TOKEN_PREFIXES),
            "development_or_comparator_semantics_required": True,
            "historical_sources_may_satisfy_final_tokens": False,
        },
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "values": normalized_entries,
    }
    payload["values_identity_sha256"] = sha256_json(payload)
    # Re-run the compiler-facing validator before the payload can be written.
    validate_values_payload(
        payload,
        master_raw=context.master_raw,
        evidence_graph_raw=context.evidence_graph.raw,
        evidence_graph_identity_sha256=str(context.evidence_graph.logical_identity_sha256),
        token_contract=token_contract,
    )
    return payload


def validate_derivation(derivation: Mapping[str, Any], token: str) -> None:
    identity = derivation.get("derivation_identity_sha256")
    require(is_sha256(identity), f"{token}: derivation identity missing")
    unsigned = deepcopy(dict(derivation))
    unsigned.pop("derivation_identity_sha256", None)
    require(sha256_json(unsigned) == identity, f"{token}: derivation identity mismatch")
    operation = derivation.get("operation")
    require(isinstance(operation, str) and operation, f"{token}: derivation operation missing")
    sources = derivation.get("sources")
    require(isinstance(sources, list) and sources, f"{token}: derivation sources missing")
    for index, source in enumerate(sources):
        require(isinstance(source, Mapping), f"{token}: derivation source {index} malformed")
        require(is_sha256(source.get("source_file_sha256")), f"{token}: source file hash missing")
        require(is_sha256(source.get("source_cell_identity_sha256")), f"{token}: source cell hash missing")
        source_role = source.get("source_role")
        locator = source.get("locator")
        require(isinstance(source_role, str) and source_role, f"{token}: source role missing")
        require(isinstance(locator, Mapping), f"{token}: source locator missing")
        expected_cell_identity = source_cell_identity(
            source_role=source_role,
            file_sha256=str(source["source_file_sha256"]),
            locator=locator,
            value=source.get("source_value"),
        )
        require(
            source.get("source_cell_identity_sha256") == expected_cell_identity,
            f"{token}: source cell identity mismatch",
        )
        authority = source.get("authority_class")
        require(
            authority in {"final_machine", "historical_development_comparator", "human_external"},
            f"{token}: invalid source authority class",
        )


def _render_scalar(token: str, value: Any) -> str:
    require(value is not None, f"{token}: null value is forbidden")
    require(not isinstance(value, (dict, list)), f"{token}: value must be scalar")
    if isinstance(value, float):
        require(math.isfinite(value), f"{token}: NaN or infinity is forbidden")
    if isinstance(value, str):
        rendered = value
        require(bool(rendered.strip()), f"{token}: empty value is forbidden")
        require(
            rendered.strip().casefold() not in FORBIDDEN_LITERAL_VALUES,
            f"{token}: null/NaN/infinity text is forbidden",
        )
    elif isinstance(value, (int, float, bool)):
        rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
    else:
        raise ManuscriptValuesError(f"{token}: unsupported value type")
    lowered = rendered.casefold()
    require(
        not any(marker in lowered for marker in FORBIDDEN_TEXT_MARKERS),
        f"{token}: TODO/TBD/provisional value is forbidden",
    )
    if token in HUMAN_METADATA_TOKENS:
        require(
            not any(marker in lowered for marker in DEFERRED_HUMAN_METADATA_MARKERS),
            f"{token}: deferred external metadata is forbidden",
        )
    require(
        RESIDUAL_TOKEN_PATTERN.search(rendered) is None,
        f"{token}: residual manuscript token is forbidden",
    )
    require(
        not any(marker in lowered for marker in INTERNAL_PROVIDER_ABI_MARKERS)
        and STALE_PUBLIC_VERSION_PATTERN.search(rendered) is None,
        f"{token}: internal ABI or stale PHAxis public version is forbidden",
    )
    if token == "PHAXIS_RELEASE_TAG":
        require(rendered == "v1.0.0", f"{token}: final public release tag must be v1.0.0")
    if token.startswith("FINAL_"):
        require(
            not any(marker in lowered for marker in LEGACY_DEPLOYMENT_MARKERS),
            f"{token}: legacy 443CV deployment value is forbidden",
        )
    if token == "FINAL_HAIR_EXPERT_ID":
        require(
            not any(marker in lowered for marker in ("candidate", "legacy", "fold", "443cv", "rhaxiscc")),
            "FINAL_HAIR_EXPERT_ID is not a neutral final train399 expert identity",
        )
    return rendered


def validate_values_payload(
    payload: Mapping[str, Any],
    *,
    master_raw: bytes,
    evidence_graph_raw: bytes,
    evidence_graph_identity_sha256: str,
    token_contract: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the complete schema-1.1 values receipt for manuscript compile."""

    expected_top = {
        "schema_version",
        "builder_schema_version",
        "status",
        "master_sha256",
        "evidence_graph_file_sha256",
        "evidence_graph_identity_sha256",
        "figure_inputs_file_sha256",
        "figure_input_assembly_identity_sha256",
        "figure_assembly_summary_file_sha256",
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
        "model_bundle_id",
        "root_expert_id",
        "root_bundle_identity_sha256",
        "hair_identity_count_expert",
        "human_metadata_file_sha256",
        "human_metadata_identity_sha256",
        "model_bundle_manifest_file_sha256",
        "model_bundle_manifest_identity_sha256",
        "clean_install_receipt_file_sha256",
        "clean_install_receipt_identity_sha256",
        "source_release_manifest_file_sha256",
        "source_release_tree_identity_sha256",
        "source_release_metadata_file_sha256",
        "source_release_metadata_identity_sha256",
        "software_release_cross_binding_identity_sha256",
        "narrative_decision_identity_sha256",
        "narrative_branch_id",
        "publication_title_contract",
        "token_contract_identity_sha256",
        "token_derivation_contract_identity_sha256",
        "source_files",
        "historical_source_policy",
        "blind_images_used",
        "root_cap_region_statistics_included",
        "values",
        "values_identity_sha256",
    }
    require(set(payload) == expected_top, "manuscript-values top-level schema mismatch")
    require(payload.get("schema_version") == VALUES_SCHEMA, "unsupported manuscript-values schema")
    require(payload.get("builder_schema_version") == VALUES_BUILDER_SCHEMA, "values builder schema changed")
    require(
        payload.get("status") == "final_values_machine_derived_locked",
        "manuscript values are not final and machine-derived",
    )
    require(payload.get("blind_images_used") == 0, "manuscript values are blind-tainted")
    require(
        payload.get("root_cap_region_statistics_included") is False,
        "manuscript values permit root-cap-region statistics",
    )
    identity = payload.get("values_identity_sha256")
    require(is_sha256(identity), "manuscript-values sealed identity missing")
    unsigned = deepcopy(dict(payload))
    unsigned.pop("values_identity_sha256", None)
    require(sha256_json(unsigned) == identity, "manuscript-values sealed identity mismatch")
    require(payload.get("master_sha256") == sha256_bytes(master_raw), "master SHA-256 binding mismatch")
    require(
        payload.get("evidence_graph_file_sha256") == sha256_bytes(evidence_graph_raw),
        "evidence graph file SHA-256 binding mismatch",
    )
    require(
        payload.get("evidence_graph_identity_sha256") == evidence_graph_identity_sha256,
        "evidence graph identity binding mismatch",
    )
    require(
        payload.get("token_contract_identity_sha256")
        == token_contract.get("contract_identity_sha256"),
        "token source-contract identity binding mismatch",
    )
    for field in (
        "figure_inputs_file_sha256",
        "figure_input_assembly_identity_sha256",
        "figure_assembly_summary_file_sha256",
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
        "human_metadata_file_sha256",
        "human_metadata_identity_sha256",
        "model_bundle_manifest_file_sha256",
        "model_bundle_manifest_identity_sha256",
        "clean_install_receipt_file_sha256",
        "clean_install_receipt_identity_sha256",
        "source_release_manifest_file_sha256",
        "source_release_tree_identity_sha256",
        "source_release_metadata_file_sha256",
        "source_release_metadata_identity_sha256",
        "software_release_cross_binding_identity_sha256",
        "narrative_decision_identity_sha256",
        "token_derivation_contract_identity_sha256",
    ):
        require(is_sha256(payload.get(field)), f"invalid or missing binding: {field}")
    require(
        payload.get("narrative_branch_id") in {"A", "B", "C"},
        "values narrative branch is invalid",
    )
    titles = payload.get("publication_title_contract")
    require(isinstance(titles, Mapping), "publication title contract missing")
    title_unsigned = deepcopy(dict(titles))
    title_identity = title_unsigned.pop("title_contract_identity_sha256", None)
    require(
        is_sha256(title_identity)
        and sha256_json(title_unsigned) == title_identity
        and titles.get("narrative_decision_identity_sha256")
        == payload.get("narrative_decision_identity_sha256")
        and titles.get("branch_id") == payload.get("narrative_branch_id")
        and set(titles.get("figures", {})) == {str(index) for index in range(1, 7)}
        and set(titles.get("tables", {})) == {str(index) for index in range(1, 4)},
        "publication title contract is not bound to the narrative decision",
    )
    require(
        isinstance(payload.get("model_bundle_id"), str)
        and re.fullmatch(
            re.escape(MODEL_BUNDLE_PREFIX) + r"[0-9A-F]{20}",
            str(payload["model_bundle_id"]),
        )
        is not None,
        "values public model-bundle identity is invalid",
    )
    root_bundle_identity = payload.get("root_bundle_identity_sha256")
    require(
        isinstance(payload.get("root_expert_id"), str)
        and is_sha256(root_bundle_identity)
        and payload.get("root_expert_id")
        == ROOT_EXPERT_PREFIX + str(root_bundle_identity)[:20].upper(),
        "values public root-expert identity is invalid",
    )
    require(
        is_sha256(root_bundle_identity),
        "values root-bundle authority identity is invalid",
    )
    require(
        payload.get("hair_identity_count_expert")
        == "PHAxis-StageB-train399-five-seed",
        "values public hair-expert identity is invalid",
    )
    policy = payload.get("historical_source_policy")
    require(
        policy
        == {
            "allowed_token_prefixes": list(HISTORICAL_TOKEN_PREFIXES),
            "development_or_comparator_semantics_required": True,
            "historical_sources_may_satisfy_final_tokens": False,
        },
        "historical source policy changed",
    )
    source_files = payload.get("source_files")
    require(isinstance(source_files, Mapping) and source_files, "values source-file registry missing")
    for role, record in source_files.items():
        require(isinstance(role, str) and role, "values source-file role malformed")
        require(isinstance(record, Mapping), f"values source-file record malformed: {role}")
        require(is_sha256(record.get("sha256")), f"values source-file SHA missing: {role}")
    require(
        source_files.get("source_release_manifest", {}).get("sha256")
        == payload.get("source_release_manifest_file_sha256")
        and source_files.get("source_release_manifest", {}).get(
            "logical_identity_sha256"
        )
        == payload.get("source_release_tree_identity_sha256")
        and source_files.get("source_release_metadata", {}).get("sha256")
        == payload.get("source_release_metadata_file_sha256")
        and source_files.get("source_release_metadata", {}).get(
            "logical_identity_sha256"
        )
        == payload.get("source_release_metadata_identity_sha256"),
        "source-release top-level and source-file bindings differ",
    )

    entries = payload.get("values")
    contract_rows = token_contract.get("tokens")
    require(isinstance(entries, Mapping), "values must be one JSON object")
    require(isinstance(contract_rows, Mapping), "token contract rows missing")
    missing = sorted(set(contract_rows) - set(entries))
    extra = sorted(set(entries) - set(contract_rows))
    require(not missing and not extra, f"token key mismatch; missing={missing}, extra={extra}")
    rendered: dict[str, str] = {}
    derivation_contract: dict[str, Any] = {}
    for token in sorted(contract_rows):
        entry = entries[token]
        require(isinstance(entry, Mapping), f"{token}: entry must be one object")
        require(
            set(entry) == {"value", "source_role", "derivation"},
            f"{token}: entry keys must be exactly value/source_role/derivation",
        )
        expected = contract_rows[token]
        require(
            entry.get("source_role") == expected.get("source_role"),
            f"{token}: source_role does not match token-family contract",
        )
        derivation = entry.get("derivation")
        require(isinstance(derivation, Mapping), f"{token}: derivation missing")
        validate_derivation(derivation, token)
        for source in derivation["sources"]:
            registered = source_files.get(source["source_role"])
            require(
                isinstance(registered, Mapping),
                f"{token}: derivation source is absent from the sealed source-file registry: "
                f"{source['source_role']}",
            )
            require(
                registered.get("sha256") == source.get("source_file_sha256"),
                f"{token}: derivation source hash differs from the sealed registry",
            )
            if "source_logical_identity_sha256" in source:
                require(
                    registered.get("logical_identity_sha256")
                    == source.get("source_logical_identity_sha256"),
                    f"{token}: derivation logical identity differs from the sealed registry",
                )
            if "container_identity_sha256" in registered:
                require(
                    registered.get("container_identity_sha256")
                    == source.get("container_identity_sha256"),
                    f"{token}: derivation container identity differs from the sealed registry",
                )
        authorities = {source["authority_class"] for source in derivation["sources"]}
        historical_token = token.startswith(HISTORICAL_TOKEN_PREFIXES)
        if token.startswith("FINAL_"):
            require(
                "historical_development_comparator" not in authorities,
                f"{token}: historical/comparator evidence cannot satisfy FINAL_*",
            )
        if historical_token:
            require(
                authorities == {"historical_development_comparator"},
                f"{token}: explicit historical token lacks historical authority",
            )
        if token in SOFTWARE_RELEASE_TOKENS:
            require(
                authorities == {"human_external", "final_machine"},
                f"{token}: software-release cross-binding authority mismatch",
            )
        elif token in HUMAN_METADATA_TOKENS:
            require(authorities == {"human_external"}, f"{token}: human metadata authority mismatch")
        else:
            require("human_external" not in authorities, f"{token}: machine token uses human metadata")
        rendered[token] = _render_scalar(token, entry.get("value"))
        derivation_contract[token] = {
            "operation": derivation["operation"],
            "source_roles": [source["source_role"] for source in derivation["sources"]],
            "locators": [source["locator"] for source in derivation["sources"]],
        }
    require(
        sha256_json(derivation_contract)
        == payload.get("token_derivation_contract_identity_sha256"),
        "token derivation-contract identity mismatch",
    )
    return rendered


def publish_json_no_overwrite(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    if destination.exists():
        raise ManuscriptValuesError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ManuscriptValuesError(
                f"refusing to overwrite existing output: {destination}"
            ) from error
        except OSError as error:
            raise ManuscriptValuesError(
                f"atomic no-overwrite publication failed: {destination}"
            ) from error
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


__all__ = [
    "BIOLOGICAL_ACQUISITION_TOKENS",
    "BuildContext",
    "CLEAN_INSTALL_RECEIPT_SCHEMA",
    "DOI_TOKENS",
    "EVIDENCE_ARTIFACT_ROLES",
    "EVIDENCE_GRAPH_SCHEMA",
    "FIGURE_ASSEMBLER_SCHEMA",
    "FIGURE_INPUT_SCHEMA",
    "FIGURE_PROVENANCE_ROLES",
    "FIGURE_RESOURCE_ROLES",
    "FIGURE_SOURCE_INPUT_ROLES",
    "HISTORICAL_TOKEN_PREFIXES",
    "HAIR_ATTACHMENT_ASSURANCE_TOKENS",
    "HUMAN_METADATA_REPORT_SCHEMA",
    "HUMAN_METADATA_SCHEMA",
    "HUMAN_METADATA_TOKENS",
    "HumanMetadataError",
    "ManuscriptValuesError",
    "MODEL_BUNDLE_MANIFEST_SCHEMA",
    "ROOT_TRAIT_ASSURANCE_TOKENS",
    "ROOT_CONTINUITY_ASSURANCE_TOKENS",
    "SAME_HARDWARE_RUNTIME_TOKENS",
    "SOFTWARE_RELEASE_TOKENS",
    "TOKEN_PATTERN",
    "TOKEN_FAMILY_RULES",
    "URL_TOKENS",
    "VALUES_BUILDER_SCHEMA",
    "VALUES_SCHEMA",
    "canonical_json_bytes",
    "build_token_source_contract",
    "assemble_values_payload",
    "guard_final_payload",
    "human_metadata_report",
    "human_metadata_template",
    "is_sha256",
    "load_build_context",
    "publish_json_no_overwrite",
    "read_bytes",
    "read_json_object",
    "require",
    "seal_derivation",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "source_cell_identity",
    "derivation_source",
    "token_rule",
    "validate_derivation",
    "validate_human_metadata",
    "validate_wt_secondary_source_inputs",
    "validate_values_payload",
    "validate_sealed_identity",
    "walk",
]
