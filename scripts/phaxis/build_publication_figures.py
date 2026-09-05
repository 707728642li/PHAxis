#!/usr/bin/env python3
"""Build the unique six-figure PHAxis 1.0.0 manuscript suite.

The command has two deliberately different modes:

``final``
    Requires a fully closed train399 -> exact283 evidence chain.  Every
    receipt and plotting resource is named explicitly and SHA-256 verified.
    Any development/provisional marker, blind access, root-cap-region claim,
    missing observability evidence, or indirect timing record stops the build.

``provisional``
    Exists only for layout tests.  All six files are prefixed ``PROVISIONAL_``,
    carry an on-canvas watermark, and the receipt forbids submission use.

The builder never discovers inputs, imports a legacy implementation, reads a
blind dataset, or starts a GPU program.  It consumes only sealed JSON/CSV and
explicitly named, hash-verified review images.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path as MatplotlibPath  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.contracts import ContractError  # noqa: E402
from phaxis.hair_stageb.evaluation_inference import (  # noqa: E402
    EVALUATION_ARTIFACT_ROLE,
    EVALUATION_DETECTION_SCHEMA,
)
from phaxis.public_identity import validate_proposal_public_identity  # noqa: E402
from phaxis.root_trait_assurance import ROOT_TRAIT_FAMILY_ORDER  # noqa: E402
from phaxis.publication_evidence import (  # noqa: E402
    FIGURE_SOURCE_INPUT_ROLES,
    MAIN_FIGURE_RESOURCE_ROLES,
    MAIN_FIGURE_STEMS,
    SUPPLEMENTARY_FIGURE_RESOURCE_ROLES,
    SUPPLEMENTARY_FIGURE_STEMS,
    SUPPLEMENTARY_FIGURE_TITLES,
    WT_SECONDARY_COHORT_ROLES,
    WT_SECONDARY_ENDPOINTS,
    WT_SECONDARY_RESOURCE_ROLES,
    figure_suite_identity_preimage,
    supplementary_figure_contract,
    validate_wt_secondary_analysis_binding,
    validate_wt_secondary_evidence,
)
from phaxis.supplementary_tables import (  # noqa: E402
    BUNDLE_DIRECTORY as SUPPLEMENTARY_TABLE_DIRECTORY,
    BUNDLE_RECEIPT as SUPPLEMENTARY_TABLE_RECEIPT,
    FINAL_STATUS as FINAL_SUPPLEMENTARY_TABLE_STATUS,
    PROVISIONAL_STATUS as PROVISIONAL_SUPPLEMENTARY_TABLE_STATUS,
    SupplementaryTableError,
    TABLE_SPECS as SUPPLEMENTARY_TABLE_SPECS,
    materialize_supplementary_table_data_bundle,
)
from phaxis.multitrait_atlas import (  # noqa: E402
    MEASUREMENT_FAMILY_ORDER,
    MEASUREMENT_FAMILY_TRAIT_IDS,
    MultitraitAtlasError,
    descriptive_heatmap_matrices,
    validate_multitrait_atlas_against_sources,
)
from phaxis.constants import HAIR_WORKING_UM_PER_PX  # noqa: E402
from phaxis.hair_stageb.preprocess import (  # noqa: E402
    make_input_channels,
    resample_to_physical_scale,
    to_gray,
)
from phaxis.publication_style import (  # noqa: E402
    PALETTE,
    check_text_budget,
    configure_publication_style,
    panel_label,
    save_figure_bundle,
)
from phaxis.narrative_decision import (  # noqa: E402
    ENDPOINT_CONTRACT as NARRATIVE_ENDPOINT_CONTRACT,
    validate_narrative_decision,
)
from phaxis.publication_titles import (  # noqa: E402
    figure_title,
    title_contract,
)


SCHEMA_VERSION = "PHAxis-publication-figure-suite-1.0"
INPUT_SCHEMA_VERSION = "PHAxis-manuscript-figure-inputs-2.0"
ASSEMBLER_SCHEMA_VERSION = "PHAxis-publication-figure-input-assembly-1.0"
RECEIPT_ROLES = (
    "train399_evaluation",
    "root_exact283",
    "stageb",
    "fusion",
    "traits",
    "cohorts",
    "analysis",
    "profiles",
)
RESOURCE_ROLES = (
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
FIGURE_STEMS = MAIN_FIGURE_STEMS
SUPPLEMENTARY_STEMS = SUPPLEMENTARY_FIGURE_STEMS
# Compatibility alias retained for downstream code that referred to the
# original sole supplementary atlas; it is now formally Figure S9.
SUPPLEMENTARY_STEM = SUPPLEMENTARY_STEMS[8]
COMPARATORS = (
    "stageb_train399",
    "legacy_hybrid_endpoint_complete_identity_layer",
)
COMPARATOR_LABELS = {
    COMPARATORS[0]: "Stage B train399",
    COMPARATORS[1]: "Legacy Hybrid identity\nendpoint-complete layer",
}
STAGEB_DETECTION_SCHEMA = EVALUATION_DETECTION_SCHEMA
LEGACY_HYBRID_COMPARATOR_SCHEMA = (
    "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
)
LEGACY_HYBRID_IDENTITY_VARIANT = "hybrid_verified_increment"
LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256 = (
    "ede309b8a828aec35be64d9f8afbc2ac9bf92b5a9e1b1b262d5acf603a746f36"
)
HISTORICAL_COMPARATOR = "historical_family_isolated_oof443"
GROUP_ORDER = ("RHD6_EV_22C", "RHD6_EV_30C", "RHD6_OE_22C", "RHD6_OE_30C")
GROUP_LABELS = {
    "RHD6_EV_22C": "EV 22 °C",
    "RHD6_EV_30C": "EV 30 °C",
    "RHD6_OE_22C": "OE-labelled 22 °C",
    "RHD6_OE_30C": "OE-labelled 30 °C",
}
GROUP_COLOURS = {
    "RHD6_EV_22C": PALETTE["navy"],
    "RHD6_EV_30C": PALETTE["teal"],
    "RHD6_OE_22C": PALETTE["plum"],
    "RHD6_OE_30C": PALETTE["orange"],
}
GROUP_FACTOR_STYLES = {
    "RHD6_EV_22C": {
        "colour": PALETTE["navy"],
        "marker": "o",
        "linestyle": "-",
        "filled": True,
    },
    "RHD6_EV_30C": {
        "colour": PALETTE["navy"],
        "marker": "s",
        "linestyle": "--",
        "filled": False,
    },
    "RHD6_OE_22C": {
        "colour": PALETTE["plum"],
        "marker": "o",
        "linestyle": "-",
        "filled": True,
    },
    "RHD6_OE_30C": {
        "colour": PALETTE["plum"],
        "marker": "s",
        "linestyle": "--",
        "filled": False,
    },
}
CASE_ROLES = ("representative", "low_contrast", "curved_dense", "continuity", "fail_closed")
FIGURE4_LOCKED_ANCHOR_TASK_IDS = {
    "low_contrast": "RHSCU-aa5b6e37df15821f",
    "curved_dense": "RHSCU-bbf649822174e0a2",
}
OVERLAY_CASE_SELECTION_BASIS = "preselected_morphology_acquisition_challenge_roles"
OVERLAY_CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE = (
    "overlay_pixels_and_morphology_evidence_cards_before_output_organization"
)
PRIMARY_ENDPOINTS = (
    "local_hair_count_1_4mm",
    "local_median_hair_length_um_1_4mm",
    "first_hair_ge40um_distance_from_distal_point_um",
    "median_root_width_um",
    "visible_root_axis_length_um",
)
EFFECT_ORDER = ("OE_vs_EV", "30C_vs_22C", "interaction")
PHENOTYPE_EFFECT_COHORT_ORDER = ("primary_clean261", "sensitivity_full283")
ENDPOINT_LABELS = {
    PRIMARY_ENDPOINTS[0]: "Visible-hair count [1,4) mm",
    PRIMARY_ENDPOINTS[1]: "Conditional projected length",
    PRIMARY_ENDPOINTS[2]: "First observed ≥40 µm hair distance",
    PRIMARY_ENDPOINTS[3]: "Median apparent root width",
    PRIMARY_ENDPOINTS[4]: "Visible root-axis extent",
}
WT_ENDPOINT_CODES = dict(
    zip(WT_SECONDARY_ENDPOINTS, ("N", "L", "F", "W", "A"), strict=True)
)
WT_ENDPOINT_COLOURS = dict(
    zip(
        WT_SECONDARY_ENDPOINTS,
        (
            PALETTE["navy"],
            PALETTE["teal"],
            PALETTE["plum"],
            PALETTE["orange"],
            PALETTE["gold"],
        ),
        strict=True,
    )
)
ROOT_TRAIT_FAMILY_LABELS = {
    "axis_extent": "Axis extent",
    "axis_shape": "Axis shape",
    "projected_area": "Projected area",
    "global_width_distribution": "Global width distribution",
    "axial_width_pattern": "Axial width pattern",
    "centerline_curvature": "Centerline curvature",
}
ROOT_TRAIT_FAMILY_COLOURS = dict(
    zip(
        ROOT_TRAIT_FAMILY_ORDER,
        (
            PALETTE["navy"],
            PALETTE["teal"],
            PALETTE["plum"],
            PALETTE["orange"],
            PALETTE["gold"],
            PALETTE["grey"],
        ),
        strict=True,
    )
)
ROOT_TRAIT_SHORT_LABELS = {
    "R01": "Visible axis length",
    "R02": "Axis chord",
    "R03": "Chord tortuosity",
    "R04": "Straightness",
    "R05": "Projected area",
    "R06": "Area per root mm",
    "R07": "Median width",
    "R08": "Width P10",
    "R09": "Width Q25",
    "R10": "Width Q75",
    "R11": "Width P90",
    "R12": "Width CV",
    "R13": "Distal-third width",
    "R14": "Middle-third width",
    "R15": "Shootward-third width",
    "R16": "Shootward:distal width",
    "R17": "Width axial slope",
    "R18": "Median curvature",
    "R19": "P95 curvature",
}
HAIR_TRAIT_SHORT_LABELS = {
    "H01": "Visible-hair count",
    "H02": "Conditional mean length",
    "H03": "Conditional median length",
    "H04": "Supported length sum",
    "H05": "Visible-hair density",
    "H06": "First visible-hair position",
    "H07": "Elongation-qualified boundary",
    "H08": "Local visible-hair count",
    "H09": "Local visible-hair density",
    "H10": "Local conditional mean length",
    "H11": "Local conditional median length",
    "H12": "Local supported length per mm",
    "H13": "Visible attachment span",
}
ATLAS_TRAIT_SHORT_LABELS = {
    **ROOT_TRAIT_SHORT_LABELS,
    **HAIR_TRAIT_SHORT_LABELS,
}
MEASUREMENT_FAMILY_DISPLAY = {
    "visible_hair_abundance": "Visible-hair abundance",
    "conditional_projected_length": "Conditional projected length",
    "axial_deployment": "Axial deployment",
    "visible_root_extent": "Visible-root extent",
    "root_form_trajectory": "Root form / trajectory",
}
MEASUREMENT_FAMILY_COLOURS = {
    "visible_hair_abundance": PALETTE["navy"],
    "conditional_projected_length": PALETTE["plum"],
    "axial_deployment": PALETTE["gold"],
    "visible_root_extent": PALETTE["teal"],
    "root_form_trajectory": PALETTE["orange"],
}
FIGURE_5_SUBMISSION_WIDTH_MM = 178.0
FIGURE_5_SUBMISSION_HEIGHT_MM = 148.0
FIGURE_5_MIN_TEXT_PT = 6.0
FIGURE_5_MIN_SYMBOL_DIAMETER_PT = 6.0
FIGURE_5_MIN_SYMBOL_DIAMETER_300DPI_PX = 25.0
FORBIDDEN_FINAL_MARKERS = (
    "provisional",
    "development_only",
    "development-only",
    "not_for_submission",
    "not for submission",
    "blocked_pending",
)


class FigureSuiteError(RuntimeError):
    """A plotting input or final evidence gate is not manuscript-safe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureSuiteError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value[key]
            yield from _walk(value[key], path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            yield path, item
            yield from _walk(item, path)


def _read_object(path: Path, role: str) -> dict[str, Any]:
    _require(path.is_file(), f"{role}: missing JSON: {path}")
    _require(not path.is_symlink(), f"{role}: symlink inputs are forbidden")
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise FigureSuiteError(f"{role}: invalid UTF-8 JSON object") from error
    return payload


def _receipt_paths(namespace: argparse.Namespace) -> dict[str, Path]:
    return {
        role: Path(getattr(namespace, role)).resolve()
        for role in RECEIPT_ROLES
    }


def _resource_paths(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, str]]:
    resources = manifest.get("resources")
    _require(isinstance(resources, Mapping), "figure inputs: resources map missing")
    _require(set(resources) == set(RESOURCE_ROLES), "figure inputs: resource roles are incomplete")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role in RESOURCE_ROLES:
        record = resources[role]
        _require(isinstance(record, Mapping), f"resource {role}: record must be an object")
        raw_path = record.get("path")
        expected = record.get("sha256")
        _require(isinstance(raw_path, str) and raw_path, f"resource {role}: path missing")
        _require(_is_sha256(expected), f"resource {role}: SHA-256 missing")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        _require(path.is_file(), f"resource {role}: file missing: {path}")
        _require(not path.is_symlink(), f"resource {role}: symlinks are forbidden")
        _require("blind" not in str(path).casefold(), f"resource {role}: blind-labelled path refused")
        observed = sha256_file(path)
        _require(observed == expected, f"resource {role}: SHA-256 mismatch")
        paths[role] = path
        hashes[role] = observed
    return paths, hashes


def _resolve_manifest_file(
    manifest_path: Path, record: Mapping[str, Any], role: str
) -> Path:
    raw_path = record.get("path")
    expected = record.get("sha256")
    _require(isinstance(raw_path, str) and raw_path, f"{role}: path missing")
    _require(_is_sha256(expected), f"{role}: file SHA-256 missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    _require(path.is_file(), f"{role}: file missing: {path}")
    _require(not path.is_symlink(), f"{role}: symlink inputs are forbidden")
    _require("blind" not in str(path).casefold(), f"{role}: blind-labelled path refused")
    _require(sha256_file(path) == expected, f"{role}: file SHA-256 mismatch")
    return path


def _source_input_paths(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, str]]:
    records = manifest.get("source_inputs")
    _require(isinstance(records, Mapping), "figure-input source table map missing")
    _require(
        set(records) == set(FIGURE_SOURCE_INPUT_ROLES),
        "figure-input source role set is not exact",
    )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role in FIGURE_SOURCE_INPUT_ROLES:
        record = records[role]
        _require(isinstance(record, Mapping), f"source input {role}: malformed record")
        path = _resolve_manifest_file(manifest_path, record, f"source input {role}")
        paths[role] = path
        hashes[role] = sha256_file(path)
    return paths, hashes


def _self_sealed_identity(path: Path) -> dict[str, str]:
    """Return the sole top-level self-sealed identity when one is present."""

    if path.suffix.casefold() != ".json":
        return {}
    try:
        payload = _read_object(path, f"identity source {path.name}")
    except FigureSuiteError:
        return {}
    candidates: list[tuple[str, str]] = []
    for field, value in payload.items():
        if not (field.endswith("identity_sha256") and _is_sha256(value)):
            continue
        unsigned = deepcopy(payload)
        unsigned.pop(field, None)
        if sha256_json(unsigned) == value:
            candidates.append((str(field), str(value)))
    _require(len(candidates) <= 1, f"{path.name}: ambiguous self-sealed identity")
    if not candidates:
        return {}
    field, value = candidates[0]
    return {"identity_field": field, "identity_sha256": value}


def _supplementary_table_sources(
    *,
    source_inputs: Mapping[str, Path],
    resources: Mapping[str, Path],
    receipt_paths: Mapping[str, Path],
    proposal_path: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    sources: dict[str, Path] = {
        **{f"source/{role}": path for role, path in source_inputs.items()},
        **{
            f"resource/{role}": resources[role]
            for role in (
                "trait_contract",
                "development_per_image",
                "development_tolerance",
                "development_threshold",
                "assurance_metrics",
                "assurance_pairs",
                "assurance_support",
                "multitrait_atlas",
                "workflow_stages",
                "runtime_summary",
                "runtime_per_image",
            )
        },
        **{f"receipt/{role}": path for role, path in receipt_paths.items()},
        "proposal/model_contract_proposal": proposal_path,
    }
    required = {
        role
        for spec in SUPPLEMENTARY_TABLE_SPECS
        for role in spec["source_roles"]
    }
    _require(required <= set(sources), "supplementary Table/Data authority route is incomplete")
    sources = {role: sources[role] for role in sorted(required)}
    identities = {
        role: identity
        for role, path in sources.items()
        if (identity := _self_sealed_identity(path))
    }
    return sources, identities


def _validate_multitrait_atlas_resource(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    resources: Mapping[str, Path],
) -> str:
    """Recompute the 32-trait atlas from its hash-locked source tables."""

    source_inputs = manifest.get("source_inputs")
    _require(isinstance(source_inputs, Mapping), "multitrait atlas source map missing")
    source_manifest_roles = {
        "clean_traits": "clean_traits",
        "full_traits": "full_traits",
        "canonical_image_traits": "full_image_traits",
        "analysis_primary_table": "analysis_primary_table",
        "analysis_sensitivity_table": "analysis_sensitivity_table",
    }
    source_paths: dict[str, Path] = {}
    for atlas_role, manifest_role in source_manifest_roles.items():
        record = source_inputs.get(manifest_role)
        _require(
            isinstance(record, Mapping),
            f"multitrait atlas source missing: {manifest_role}",
        )
        source_paths[atlas_role] = _resolve_manifest_file(
            manifest_path, record, f"multitrait_atlas/{manifest_role}"
        )
    contract = _read_object(resources["trait_contract"], "multitrait trait contract")
    atlas = _read_object(resources["multitrait_atlas"], "multitrait atlas")
    try:
        validate_multitrait_atlas_against_sources(
            atlas,
            trait_contract=contract,
            clean_traits=pd.read_csv(source_paths["clean_traits"]),
            full_traits=pd.read_csv(source_paths["full_traits"]),
            canonical_image_traits=pd.read_csv(
                source_paths["canonical_image_traits"]
            ),
            primary_analysis=pd.read_csv(source_paths["analysis_primary_table"]),
            sensitivity_analysis=pd.read_csv(
                source_paths["analysis_sensitivity_table"]
            ),
            source_sha256={
                "trait_contract": sha256_file(resources["trait_contract"]),
                **{
                    role: sha256_file(source_paths[role])
                    for role in source_manifest_roles
                },
            },
        )
    except (MultitraitAtlasError, OSError, UnicodeError, pd.errors.ParserError) as error:
        raise FigureSuiteError(f"multitrait atlas validation failed: {error}") from error
    return str(atlas["atlas_identity_sha256"])


def _validate_assembler_receipt(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    final: bool,
) -> tuple[str, dict[str, dict[str, str]]]:
    _require(
        manifest.get("assembler_schema_version") == ASSEMBLER_SCHEMA_VERSION,
        "figure inputs were not produced by the production assembler schema",
    )
    identity = manifest.get("figure_input_assembly_identity_sha256")
    _require(_is_sha256(identity), "figure-input assembly identity missing")
    unsigned = deepcopy(dict(manifest))
    unsigned.pop("figure_input_assembly_identity_sha256", None)
    _require(
        sha256_json(unsigned) == identity,
        "figure-input assembly identity does not seal the complete manifest",
    )
    declared_supplementary = manifest.get("supplementary_figure_contract")
    expected_supplementary = supplementary_figure_contract()
    expected_supplementary["contract_identity_sha256"] = sha256_json(
        expected_supplementary
    )
    _require(
        declared_supplementary == expected_supplementary,
        "figure-input supplementary S1--S9 contract changed",
    )
    lineage = manifest.get("resource_lineage")
    _require(
        isinstance(lineage, Mapping) and set(lineage) == set(RESOURCE_ROLES),
        "figure-input resource lineage is incomplete",
    )
    provenance = manifest.get("provenance_receipts")
    expected_roles = {
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
    }
    _require(
        isinstance(provenance, Mapping) and set(provenance) == expected_roles,
        "figure-input provenance receipts are incomplete",
    )
    bindings: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for role in sorted(expected_roles):
        record = provenance[role]
        _require(isinstance(record, Mapping), f"{role}: provenance record malformed")
        path = _resolve_manifest_file(manifest_path, record, role)
        payload = _read_object(path, role)
        field = record.get("identity_field")
        declared_identity = record.get("identity_sha256")
        _require(isinstance(field, str) and field, f"{role}: identity field missing")
        _require(_is_sha256(declared_identity), f"{role}: logical identity missing")
        _require(payload.get(field) == declared_identity, f"{role}: logical identity differs")
        unsigned_payload = deepcopy(payload)
        unsigned_payload.pop(field, None)
        _require(
            sha256_json(unsigned_payload) == declared_identity,
            f"{role}: logical identity does not seal the complete receipt",
        )
        _guard_no_blind_or_root_cap_region(role, payload)
        if final:
            _guard_no_final_markers(role, payload)
            _require(
                "provisional" not in str(path).casefold(),
                f"{role}: provisional path refused in final mode",
            )
        payloads[role] = payload
        bindings[role] = {
            "sha256": str(record["sha256"]),
            "identity_sha256": str(declared_identity),
        }

    assurance = payloads["measurement_assurance"]
    _require(
        assurance.get("schema_version") == "PHAxis-measurement-assurance-receipt-1.0"
        and assurance.get("status") == "completed_locked_qc_development_assurance"
        and assurance.get("scope")
        == "QC-development measurement assurance; non-independent"
        and assurance.get("independent_accuracy_claim_allowed") is False,
        "measurement assurance is not sealed non-independent QC-development evidence",
    )
    source_inputs = manifest.get("source_inputs")
    _require(isinstance(source_inputs, Mapping), "figure-input source table map missing")
    assurance_hashes = assurance.get("source_table_sha256")
    _require(isinstance(assurance_hashes, Mapping), "measurement assurance table hashes missing")
    for short, role in (
        ("metrics", "assurance_metrics"),
        ("pairs", "assurance_pairs"),
        ("support", "assurance_support"),
        ("topology", "assurance_topology"),
    ):
        record = source_inputs.get(role)
        _require(isinstance(record, Mapping), f"source input missing: {role}")
        _resolve_manifest_file(manifest_path, record, f"source_{role}")
        _require(
            record.get("sha256") == assurance_hashes.get(short),
            f"measurement assurance does not bind source {short} table",
        )
    prediction_provenance = manifest.get("train399_prediction_input_provenance")
    _require(
        isinstance(prediction_provenance, Mapping)
        and set(prediction_provenance)
        == {
            "task_order_identity_sha256",
            "stageb_train399",
            "legacy_hybrid_endpoint_complete_identity_layer",
        }
        and _is_sha256(prediction_provenance.get("task_order_identity_sha256")),
        "train399 ordered prediction-input provenance is incomplete",
    )
    stageb_prediction = prediction_provenance.get("stageb_train399")
    legacy_prediction = prediction_provenance.get(
        "legacy_hybrid_endpoint_complete_identity_layer"
    )
    _require(
        isinstance(stageb_prediction, Mapping)
        and stageb_prediction.get("schema_version") == STAGEB_DETECTION_SCHEMA
        and stageb_prediction.get("artifact_role") == EVALUATION_ARTIFACT_ROLE
        and stageb_prediction.get("production_consumption_allowed") is False
        and stageb_prediction.get("fusion_consumption_allowed") is False
        and stageb_prediction.get("traits_consumption_allowed") is False
        and _is_sha256(
            stageb_prediction.get("evaluation_inference_summary_sha256")
        )
        and _is_sha256(
            stageb_prediction.get(
                "evaluation_inference_summary_identity_sha256"
            )
        )
        and _is_sha256(
            stageb_prediction.get("evaluation_gate_identity_sha256")
        )
        and _is_sha256(
            stageb_prediction.get("ordered_file_set_identity_sha256")
        ),
        "Stage-B QC44 eval-only prediction authority is not hash-locked/nonproduction",
    )
    _require(
        isinstance(legacy_prediction, Mapping)
        and legacy_prediction.get("evidence_role")
        == "locked_legacy_development_comparator"
        and legacy_prediction.get("schema_version")
        == LEGACY_HYBRID_COMPARATOR_SCHEMA
        and legacy_prediction.get("identity_hair_variant")
        == LEGACY_HYBRID_IDENTITY_VARIANT
        and legacy_prediction.get("count_hair_variant")
        == LEGACY_HYBRID_IDENTITY_VARIANT
        and legacy_prediction.get("endpoint_complete_identity_layer") is True
        and legacy_prediction.get("phaxis_payload_allowed") is False
        and legacy_prediction.get("stageb_identity_source_allowed") is False
        and _is_sha256(
            legacy_prediction.get("ordered_file_set_identity_sha256")
        )
        and legacy_prediction.get("prediction_set_identity_sha256")
        == legacy_prediction.get("ordered_file_set_identity_sha256")
        and legacy_prediction.get("expected_prediction_set_identity_sha256")
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
        and legacy_prediction.get("ordered_file_set_identity_sha256")
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256,
        "legacy Hybrid comparator schema/variant/set identity is not locked",
    )
    return str(identity), bindings


def _guard_no_blind_or_root_cap_region(role: str, payload: Mapping[str, Any]) -> None:
    found_blind_guard = False
    for path, value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if leaf == "blind_images_used":
            found_blind_guard = True
            _require(value == 0, f"{role}: {path} must be 0")
        if leaf in {"root_cap_region_output", "root_cap_region_statistics_included"}:
            _require(value is False, f"{role}: {path} must be false")
        if leaf == "canonical_annotations_read":
            _require(value is False, f"{role}: canonical annotations were read")
        if leaf == "condition_metadata_used_for_routing":
            _require(value is False, f"{role}: condition metadata routed inference")
    _require(found_blind_guard, f"{role}: explicit blind_images_used guard missing")


def _guard_no_final_markers(role: str, payload: Mapping[str, Any]) -> None:
    for path, value in _walk(payload):
        if isinstance(value, str):
            lowered = value.casefold()
            _require(
                not any(marker in lowered for marker in FORBIDDEN_FINAL_MARKERS),
                f"{role}: non-final marker at {path}",
            )


def _validate_train399_prediction_contract(evaluation: Mapping[str, Any]) -> None:
    rows = evaluation.get("per_image")
    _require(
        isinstance(rows, list) and len(rows) == 44,
        "train399 evaluation per-image scope is not QC44",
    )
    task_order = [str(row.get("task_id")) for row in rows]
    _require(len(set(task_order)) == 44, "train399 evaluation task order is invalid")
    locks = evaluation.get("prediction_input_locks")
    _require(isinstance(locks, Mapping), "train399 prediction input locks missing")
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
            ),
            f"train399 {list_field} is not the ordered QC44 file lock",
        )
        _require(
            locks.get(identity_field) == sha256_json(records),
            f"train399 {identity_field} does not seal its ordered locks",
        )
    comparator = evaluation.get("comparator_contract", {}).get("hybrid_max")
    _require(
        isinstance(comparator, Mapping)
        and comparator.get("evidence_role")
        == "locked_legacy_development_comparator"
        and comparator.get("schema_version") == LEGACY_HYBRID_COMPARATOR_SCHEMA
        and comparator.get("identity_hair_variant")
        == LEGACY_HYBRID_IDENTITY_VARIANT
        and comparator.get("count_hair_variant")
        == LEGACY_HYBRID_IDENTITY_VARIANT
        and comparator.get("endpoint_complete_identity_layer") is True
        and comparator.get("phaxis_payload_allowed") is False
        and comparator.get("stageb_identity_source_allowed") is False
        and comparator.get("prediction_set_identity_sha256")
        == locks.get("hybrid_prediction_set_identity_sha256")
        and comparator.get("expected_prediction_set_identity_sha256")
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
        and locks.get("hybrid_prediction_set_identity_sha256")
        == LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256,
        "legacy Hybrid comparator semantics are not explicit",
    )


def _validate_final_receipts(paths: Mapping[str, Path], payloads: Mapping[str, dict[str, Any]]) -> None:
    evaluation = payloads["train399_evaluation"]
    training = evaluation.get("training_contract")
    _require(
        evaluation.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
        "train399 evaluation: unsupported schema",
    )
    _require(evaluation.get("status") == "completed", "train399 evaluation is incomplete")
    hierarchy = evaluation.get("metric_hierarchy")
    _require(
        isinstance(hierarchy, Mapping)
        and hierarchy.get("primary")
        == "one-to-one tolerant biological-hair presence; bidirectional partial centreline coverage without endpoint gates"
        and hierarchy.get("primary_minimum_truth_coverage") == 0.25
        and hierarchy.get("primary_minimum_prediction_coverage") == 0.25
        and hierarchy.get("primary_minimum_direction_cosine") == 0.0,
        "train399 evaluation primary metric contract changed",
    )
    _require(isinstance(training, Mapping), "train399 evaluation: training contract missing")
    _require(training.get("training_images") == 399, "train399 evaluation: wrong training scope")
    _require(training.get("validation_images") == 44, "train399 evaluation: wrong QC scope")
    _require(
        training.get("validation_labels_used_for_gradient_or_early_stopping") is False,
        "train399 evaluation: QC labels entered optimization",
    )
    _require(
        evaluation.get("independent_accuracy_claim_allowed") is False,
        "train399 evaluation: independent-test claim is forbidden",
    )
    bootstrap = evaluation.get("paired_bootstrap_95ci")
    _require(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("method") == "paired image-level nonparametric bootstrap"
        and bootstrap.get("repetitions") == 10000
        and bootstrap.get("seed") == 20260828
        and isinstance(
            bootstrap.get("delta_stageb_train399_minus_hybrid", {}).get(
                "biological_presence_f1_20um"
            ),
            Mapping,
        ),
        "train399 primary paired bootstrap contract changed",
    )
    _validate_train399_prediction_contract(evaluation)

    root = payloads["root_exact283"]
    _require(
        root.get("schema_version") == "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "root exact283: unsupported schema",
    )
    _require(root.get("status") == "pass_exact_283", "root exact283 gate did not pass")
    _require(root.get("fresh_portable_raw_image_rerun_completed") is True, "root exact283 rerun missing")
    layers = root.get("layers")
    expected_layers = {"v12_strip_root_mask", "v20_root_polygon", "final_hybrid_root_mask"}
    _require(isinstance(layers, Mapping) and set(layers) == expected_layers, "root layers incomplete")
    for layer, record in layers.items():
        _require(isinstance(record, Mapping), f"root layer {layer}: invalid record")
        _require(
            record.get("exact") == 283
            and record.get("expected") == 283
            and record.get("mismatch_count") == 0
            and record.get("gate_pass") is True,
            f"root layer {layer}: exact283 failed",
        )

    expected_schema = {
        "stageb": "PHAxis-StageB-inference-run-1.1",
        "fusion": "PHAxis-fusion-run-1.1",
        "traits": "PHAxis-trait-export-1.0",
        "cohorts": "PHAxis-biological-cohorts-1.0",
        "analysis": "PHAxis-exploratory-biological-analysis-1.0",
        "profiles": "PHAxis-distal-axis-profile-export-1.0.0",
    }
    for role, schema in expected_schema.items():
        _require(payloads[role].get("schema_version") == schema, f"{role}: unsupported schema")
    _require(payloads["stageb"].get("status") == "completed", "Stage B full283 is incomplete")
    _require(payloads["stageb"].get("images") == 283, "Stage B is not exact full283")
    _require(payloads["fusion"].get("status") == "completed", "fusion full283 is incomplete")
    _require(payloads["fusion"].get("images") == 283, "fusion is not exact full283")
    _require(payloads["traits"].get("status") == "completed", "trait export is incomplete")
    _require(payloads["traits"].get("tasks") == 283, "trait export is not exact full283")
    _require(
        payloads["fusion"].get("source_stageb_summary_sha256") == sha256_file(paths["stageb"]),
        "fusion does not bind the named Stage B summary",
    )
    stageb_expert = payloads["stageb"].get("detection_model_metadata", {}).get("expert_id")
    _require(isinstance(stageb_expert, str) and stageb_expert, "Stage B expert identity missing")
    _require(
        payloads["fusion"].get("hair_identity_count_expert") == stageb_expert,
        "fusion expert differs from Stage B",
    )
    _require(
        payloads["traits"].get("hair_identity_count_expert") == stageb_expert,
        "trait expert differs from Stage B",
    )
    counts = payloads["cohorts"].get("counts")
    _require(isinstance(counts, Mapping), "cohort counts missing")
    _require(counts.get("biological_full") == 283, "cohort full scope changed")
    _require(counts.get("biological_clean") == 261, "cohort clean scope changed")
    cohort_inputs = payloads["cohorts"].get("input_sha256")
    _require(
        isinstance(cohort_inputs, Mapping)
        and cohort_inputs.get("trait_export_summary") == sha256_file(paths["traits"]),
        "cohorts do not bind the named trait summary",
    )
    analysis = payloads["analysis"]
    _require(
        analysis.get("status") == "completed_exploratory_clean_primary_full_sensitivity",
        "analysis clean/full status is incomplete",
    )
    _require(analysis.get("primary_cohort") == "primary_clean261", "analysis primary is not clean261")
    _require(analysis.get("sensitivity_cohort") == "sensitivity_full283", "analysis sensitivity is not full283")
    _require(
        analysis.get("cohort_build_summary_sha256") == sha256_file(paths["cohorts"]),
        "analysis does not bind the named cohort summary",
    )
    profiles = payloads["profiles"]
    _require(profiles.get("status") == "completed", "clean261 profiles are incomplete")
    _require(profiles.get("tasks") == 261, "profile source-unit scope changed")
    _require(
        profiles.get("locked_1_4mm_trait_crosscheck_tasks") == 261
        and profiles.get("locked_1_4mm_trait_crosscheck_mismatches") == 0,
        "profile/trait crosscheck failed",
    )


def _read_table(path: Path, role: str, columns: Sequence[str]) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except Exception as error:
        raise FigureSuiteError(f"{role}: unreadable CSV") from error
    missing = [column for column in columns if column not in table.columns]
    _require(not missing, f"{role}: missing columns {missing}")
    _require(len(table) > 0, f"{role}: empty table")
    return table


def _validate_wt_figure_resources(
    *,
    resources: Mapping[str, Path],
    manifest: Mapping[str, Any],
    analysis_summary: Mapping[str, Any],
) -> dict[str, Any]:
    columns = {
        "wt_within_experiment_contrasts": (
            "cohort", "cohort_role", "endpoint", "experiment_key",
            "developmental_day", "developmental_day_status", "effect_scale",
            "log_effect_30C_over_22C", "log_effect_standard_error",
            "sampling_variance", "estimate_30C_over_22C", "ci95_low",
            "ci95_high", "p_value_model", "p_value_model_BH_FDR",
            "reject_model_BH_FDR_0p05", "multiplicity_family",
            "analysis_status", "not_estimable_reason", "meta_eligible",
            "meta_exclusion_reason", "inference_status",
        ),
        "wt_within_day_meta_analysis": (
            "cohort", "cohort_role", "endpoint", "developmental_day",
            "k_eligible_experiments", "eligible_experiments", "model",
            "effect_scale", "log_effect_30C_over_22C",
            "log_effect_standard_error_hartung_knapp",
            "estimate_30C_over_22C", "ci95_low", "ci95_high",
            "p_value_hartung_knapp", "p_value_hartung_knapp_BH_FDR",
            "reject_hartung_knapp_BH_FDR_0p05", "multiplicity_family",
            "analysis_status", "not_estimable_reason",
            "cross_day_pooling_performed", "unknown_day_contrasts_included",
            "inference_status",
        ),
        "wt_temperature_qc_flow": (
            "cohort", "cohort_role", "experiment_key", "developmental_day",
            "developmental_day_status", "endpoint", "base_gate_pass",
            "endpoint_gate_pass", "model_status", "not_estimable_reason",
            "phenotype_outlier_filter_applied",
        ),
    }
    frames = {
        role: _read_table(resources[role], role, columns[role])
        for role in WT_SECONDARY_RESOURCE_ROLES
    }
    source_inputs = manifest.get("source_inputs")
    _require(isinstance(source_inputs, Mapping), "WT source-input map missing")
    table_hashes = {
        role: sha256_file(resources[role]) for role in WT_SECONDARY_RESOURCE_ROLES
    }
    for role in WT_SECONDARY_RESOURCE_ROLES:
        record = source_inputs.get(role)
        _require(
            isinstance(record, Mapping)
            and record.get("sha256") == table_hashes[role],
            f"WT resource/source hash binding differs: {role}",
        )
    try:
        evidence = validate_wt_secondary_evidence(
            contrasts=frames["wt_within_experiment_contrasts"].to_dict(
                "records"
            ),
            meta=frames["wt_within_day_meta_analysis"].to_dict("records"),
            flow=frames["wt_temperature_qc_flow"].to_dict("records"),
        )
        binding = validate_wt_secondary_analysis_binding(
            analysis_summary=analysis_summary,
            evidence_summary=evidence,
            table_sha256=table_hashes,
        )
    except ValueError as error:
        raise FigureSuiteError(
            f"WT secondary evidence validation failed: {error}"
        ) from error
    _require(
        manifest.get("wt_secondary_evidence") == binding,
        "WT secondary figure-input binding differs",
    )
    return binding


def _finite(frame: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        _require(np.isfinite(values).all(), f"{role}: non-finite {column}")


def _ordered_fixed_effects(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Validate and reindex the exact 5 endpoint x 3 effect x 2 cohort grid."""

    index_columns = ["endpoint_key", "effect_key", "cohort"]
    normalized = frame.copy()
    for column in index_columns:
        normalized[column] = normalized[column].astype(str)
    _require(
        not normalized.duplicated(index_columns).any(),
        f"{role}: duplicate fixed endpoint/effect/cohort cell",
    )
    expected_index = pd.MultiIndex.from_product(
        [PRIMARY_ENDPOINTS, EFFECT_ORDER, PHENOTYPE_EFFECT_COHORT_ORDER],
        names=index_columns,
    )
    observed_index = pd.MultiIndex.from_frame(
        normalized[index_columns]
    )
    _require(
        set(observed_index) == set(expected_index),
        f"{role}: fixed 15-effect clean/full grid is missing or contains an unexpected cell",
    )
    return normalized.set_index(index_columns).reindex(expected_index).reset_index()


def _finite_number(value: Any, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FigureSuiteError(f"{role} is not numeric") from error
    _require(np.isfinite(result), f"{role} is not finite")
    return result


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def _prepare_inputs(
    *,
    mode: str,
    figure_inputs: Path,
    receipt_paths: Mapping[str, Path],
    model_contract_proposal: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Any],
    str,
    str,
    str,
    dict[str, dict[str, str]],
]:
    manifest = _read_object(figure_inputs, "figure inputs")
    _require(manifest.get("schema_version") == INPUT_SCHEMA_VERSION, "figure input schema changed")
    _guard_no_blind_or_root_cap_region("figure inputs", manifest)
    assembly_identity, provenance_bindings = _validate_assembler_receipt(
        figure_inputs,
        manifest,
        final=mode == "final",
    )
    receipt_payloads = {
        role: _read_object(path, role) for role, path in receipt_paths.items()
    }
    for role, payload in receipt_payloads.items():
        _guard_no_blind_or_root_cap_region(role, payload)
    source_hashes = {role: sha256_file(receipt_paths[role]) for role in RECEIPT_ROLES}
    _require(
        manifest.get("source_summary_sha256") == source_hashes,
        "figure inputs do not bind the exact eight named receipts",
    )
    proposal = _read_object(model_contract_proposal, "model-contract proposal")
    _guard_no_blind_or_root_cap_region("model-contract proposal", proposal)
    proposal_file_sha256 = sha256_file(model_contract_proposal)
    proposal_identity = proposal.get("model_contract_identity_sha256")
    _require(_is_sha256(proposal_identity), "model-contract proposal identity missing")
    unsigned_proposal = deepcopy(proposal)
    unsigned_proposal.pop("model_contract_identity_sha256", None)
    _require(
        sha256_json(unsigned_proposal) == proposal_identity,
        "model-contract proposal sealed identity mismatch",
    )
    try:
        derived_public_identity = validate_proposal_public_identity(proposal)
    except ContractError as error:
        raise FigureSuiteError(
            "model-contract proposal canonical public identity is invalid"
        ) from error
    _require(
        manifest.get("model_contract_proposal_sha256") == proposal_file_sha256,
        "figure inputs do not bind the model-contract proposal file",
    )
    _require(
        manifest.get("model_contract_proposal_identity_sha256") == proposal_identity,
        "figure inputs do not bind the model-contract proposal identity",
    )
    expected_public_identity = {
        "model_bundle_id": derived_public_identity["model_bundle_id"],
        "root_expert_id": derived_public_identity["root_expert_id"],
        "root_provider_role": derived_public_identity["root_provider_role"],
    }
    _require(
        manifest.get("model_contract_public_identity")
        == expected_public_identity,
        "figure inputs do not propagate proposal-owned public model identities",
    )
    _require(
        manifest.get("model_bundle_id")
        == expected_public_identity["model_bundle_id"]
        and manifest.get("root_expert_id")
        == expected_public_identity["root_expert_id"],
        "figure inputs top-level public model identities differ from proposal",
    )
    resources, resource_hashes = _resource_paths(figure_inputs, manifest)
    _validate_multitrait_atlas_resource(
        manifest_path=figure_inputs,
        manifest=manifest,
        resources=resources,
    )
    _validate_wt_figure_resources(
        resources=resources,
        manifest=manifest,
        analysis_summary=receipt_payloads["analysis"],
    )
    if mode == "final":
        _require(manifest.get("status") == "final", "final build requires final figure inputs")
        _guard_no_final_markers("figure inputs", manifest)
        for role, payload in receipt_payloads.items():
            _guard_no_final_markers(role, payload)
        _guard_no_final_markers("model-contract proposal", proposal)
        _require(
            proposal.get("schema_version") == "PHAxis-model-contract-1.0.0",
            "model-contract proposal schema changed",
        )
        _require(
            proposal.get("formal_release_status") == "passed_proposal_not_official",
            "model-contract proposal has not passed proposal validation",
        )
        promotion = proposal.get("promotion")
        _require(isinstance(promotion, Mapping), "model-contract proposal promotion receipt missing")
        _require(
            promotion.get("schema_version") == "PHAxis-model-contract-promotion-1.0"
            and promotion.get("status") == "validated_proposal_not_applied"
            and promotion.get("official_apply_performed") is False,
            "model-contract proposal was not validated as an unapplied proposal",
        )
        _validate_final_receipts(receipt_paths, receipt_payloads)
        public_root_provider_role = derived_public_identity["root_provider_role"]
        public_root_id = derived_public_identity["root_expert_id"]
        public_model_id = derived_public_identity["model_bundle_id"]
        proposal_root = proposal.get("root_expert")
        root_receipt = receipt_payloads["root_exact283"]
        _require(
            isinstance(public_model_id, str)
            and public_model_id.startswith("PHAXIS-V1.0.0-STRICT-TRAIN399-")
            and public_root_provider_role == "PHAxis-portable-root-provider"
            and isinstance(public_root_id, str)
            and public_root_id.startswith("PHAxis-root-provider-"),
            "proposal-derived public model/root identity is invalid",
        )
        root_field_by_role = {
            "stageb": "root_expert_id",
            "fusion": "root_expert",
            "traits": "root_expert_id",
            "cohorts": "root_expert_id",
            "analysis": "root_expert_id",
            "profiles": "root_expert_id",
        }
        for role, root_field in root_field_by_role.items():
            _require(
                receipt_payloads[role].get("model_bundle_id") == public_model_id
                and receipt_payloads[role].get(root_field) == public_root_id,
                f"{role}: public model/root identity differs from proposal",
            )
        _require(
            isinstance(proposal_root, Mapping)
            and proposal_root.get("bundle_identity_sha256")
            == root_receipt.get("bundle_identity_sha256")
            and proposal_root.get("pipeline_identity_sha256")
            == root_receipt.get("pipeline_identity_sha256")
            and proposal_root.get("fresh_exact283_audit_identity_sha256")
            == root_receipt.get("audit_identity_sha256"),
            "model-contract proposal does not bind the named root-provider bundle/audit receipt",
        )
        prediction_provenance = manifest["train399_prediction_input_provenance"]
        prediction_locks = receipt_payloads["train399_evaluation"][
            "prediction_input_locks"
        ]
        _require(
            prediction_provenance["stageb_train399"][
                "ordered_file_set_identity_sha256"
            ]
            == prediction_locks["stageb_detection_set_identity_sha256"]
            and prediction_provenance[
                "legacy_hybrid_endpoint_complete_identity_layer"
            ]["ordered_file_set_identity_sha256"]
            == prediction_locks["hybrid_prediction_set_identity_sha256"]
            and prediction_provenance["task_order_identity_sha256"]
            == sha256_json(
                [
                    str(row["task_id"])
                    for row in receipt_payloads["train399_evaluation"]["per_image"]
                ]
            ),
            "figure-input QC44 prediction sets differ from the named evaluator receipt",
        )
        _require(
            manifest.get("hair_identity_expert_id")
            == receipt_payloads["stageb"].get(
                "detection_model_metadata", {}
            ).get("expert_id"),
            "figure inputs hair-identity expert differs from Stage B",
        )
    else:
        _require(manifest.get("status") in {"final", "provisional"}, "invalid provisional input status")
    return (
        manifest,
        resources,
        resource_hashes,
        receipt_payloads,
        source_hashes,
        proposal,
        proposal_file_sha256,
        str(proposal_identity),
        assembly_identity,
        provenance_bindings,
    )


def _linear_display(path: Path, lower: float, upper: float) -> np.ndarray:
    _require(math.isfinite(lower) and math.isfinite(upper) and upper > lower, "invalid display range")
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("RGB"), dtype=np.float32)
    array = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
    return array


def _scale_bar(axis: plt.Axes, *, image_shape: Sequence[int], pixels: float, micrometres: float) -> None:
    _require(pixels > 0 and micrometres > 0, "every microscopy panel needs a positive scale bar")
    height, width = image_shape[:2]
    x1 = width * 0.94
    x0 = x1 - pixels
    y = height * 0.93
    axis.plot([x0, x1], [y, y], color="white", linewidth=2.2, solid_capstyle="butt")
    axis.text((x0 + x1) / 2, y - height * 0.035, f"{micrometres:g} µm", color="white", ha="center", va="top", fontsize=6.2)


def _watermark(figure: plt.Figure, provisional: bool) -> None:
    if provisional:
        figure.text(
            0.5,
            0.5,
            "PROVISIONAL — NOT FOR SUBMISSION",
            ha="center",
            va="center",
            rotation=28,
            fontsize=25,
            color="#B42318",
            alpha=0.18,
            fontweight="bold",
            zorder=100,
        )


def _clean_axis(axis: plt.Axes, *, x: bool = False, y: bool = True) -> None:
    if x:
        axis.grid(axis="x", color=PALETTE["light_grey"], linewidth=0.45)
    if y:
        axis.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.45)
    axis.set_axisbelow(True)


def _figure1(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    geometry = _read_object(resources["figure1_geometry"], "Figure 1 geometry")
    traits = _read_object(resources["trait_contract"], "trait contract")
    _guard_no_blind_or_root_cap_region("Figure 1 geometry", geometry)
    _guard_no_blind_or_root_cap_region("trait contract", traits)
    _require(
        traits.get("schema_version") == "PHAxis-trait-contract-1.0.0",
        "trait contract schema changed",
    )
    _require(
        geometry.get("source_image_sha256") == sha256_file(resources["figure1_image"]),
        "Figure 1 geometry does not bind its microscopy image",
    )
    counts = traits.get("counts")
    _require(isinstance(counts, Mapping), "trait contract counts missing")
    _require(counts.get("nonredundant_biological_numeric_fields") == 32, "trait count is not 32")
    _require(counts.get("primary_root_fields") == 19, "root trait count is not 19")
    _require(counts.get("root_hair_fields") == 13, "hair trait count is not 13")
    _require(counts.get("root_cap_region_fields") == 0, "root-cap region traits are forbidden")
    display = geometry.get("display")
    _require(isinstance(display, Mapping) and display.get("kind") == "linear_global", "Figure 1 display must be global linear")
    image = _linear_display(resources["figure1_image"], float(display["lower"]), float(display["upper"]))
    scale = geometry.get("scale_bar")
    _require(isinstance(scale, Mapping), "Figure 1 scale bar missing")

    configure_publication_style()
    figure = plt.figure()
    grid = figure.add_gridspec(2, 6, left=0.045, right=0.97, bottom=0.075, top=0.94, wspace=0.55, hspace=0.45)
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 0:3]),
        figure.add_subplot(grid[1, 3:6]),
    )
    for axis in axes[:2]:
        axis.imshow(image)
        axis.set_axis_off()
        _scale_bar(axis, image_shape=image.shape, pixels=float(scale["pixels"]), micrometres=float(scale["micrometres"]))
    axes[0].set_title("Raw microscopy", loc="left")
    axes[1].set_title("Distal-axis coordinates", loc="left")
    polygon = np.asarray(geometry.get("root_polygon_xy"), dtype=float)
    axis_xy = np.asarray(geometry.get("axis_xy"), dtype=float)
    distal = np.asarray(geometry.get("distal_point_xy"), dtype=float)
    _require(polygon.ndim == 2 and polygon.shape[1] == 2, "Figure 1 root polygon invalid")
    _require(axis_xy.ndim == 2 and axis_xy.shape[1] == 2, "Figure 1 root axis invalid")
    _require(distal.shape == (2,), "Figure 1 distal point invalid")
    axes[1].add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="#19AADC", linewidth=1.0))
    axes[1].plot(axis_xy[:, 0], axis_xy[:, 1], color="#E6E6E6", linewidth=1.0)
    axes[1].scatter([distal[0]], [distal[1]], s=16, color="#DC3CFF", zorder=6)
    for hair in geometry.get("hair_identities", []):
        identity = np.asarray(hair["identity_xy"], dtype=float)
        attachment = np.asarray(hair["attachment_xy"], dtype=float)
        axes[1].plot(identity[:, 0], identity[:, 1], color="#FFCD14", linewidth=0.8)
        axes[1].scatter([attachment[0]], [attachment[1]], s=8, color="#FFFF00", zorder=7)
        if hair.get("length_curve_xy") is not None:
            curve = np.asarray(hair["length_curve_xy"], dtype=float)
            axes[1].plot(curve[:, 0], curve[:, 1], color="#73F55A", linewidth=1.0)

    conceptual = axes[2]
    conceptual.set_xlim(0, 1)
    conceptual.set_ylim(0, 1)
    conceptual.set_axis_off()
    conceptual.plot([0.48, 0.48], [0.12, 0.88], color=PALETTE["navy"], linewidth=5)
    conceptual.scatter([0.48], [0.12], color=PALETTE["plum"], s=25, zorder=5)
    for y, x in ((0.30, 0.20), (0.48, 0.78), (0.69, 0.17), (0.82, 0.78)):
        conceptual.plot([0.48, x], [y, y + 0.05], color=PALETTE["gold"], linewidth=1.2)
        conceptual.scatter([0.48], [y], color=PALETTE["gold"], s=11)
    conceptual.annotate("shootward", (0.52, 0.84), fontsize=7)
    conceptual.annotate("distal point", (0.52, 0.11), fontsize=7)
    conceptual.annotate("attachment", (0.52, 0.46), fontsize=7)
    conceptual.annotate("endpoint-complete", (0.08, 0.76), fontsize=7)
    conceptual.annotate("review-only: no formal traits", (0.03, 0.04), fontsize=6.2, color=PALETTE["grey"])
    conceptual.set_title("Biological primitives", loc="left")

    families = axes[3]
    families.set_axis_off()
    family_rows = (
        ("H08 / N — visible population", "accepted identities in [1,4) mm", PALETTE["teal"]),
        ("H11 / L — supported morphology", "endpoint-complete projected length", PALETTE["plum"]),
        ("H07 / F — deployment boundary", "first elongation-qualified hair", PALETTE["gold"]),
        ("R07 / W — carrying-root calibre", "apparent 2D median width", PALETTE["orange"]),
        ("R01 / A — visible organ extent", "distal-anchored visible axis", PALETTE["navy"]),
    )
    for index, (title, detail, colour) in enumerate(family_rows):
        y = 0.87 - index * 0.17
        families.add_patch(Rectangle((0.03, y - 0.065), 0.94, 0.115, facecolor=colour, alpha=0.13, edgecolor=colour))
        families.text(0.07, y, title, fontweight="bold", va="center", fontsize=7)
        families.text(0.48, y, detail, va="center", fontsize=7)
    families.text(0.03, 0.02, "32 canonical descriptors = 19 root + 13 hair", fontsize=7.3, fontweight="bold")
    families.set_title("Measurement families", loc="left")

    workflow = axes[4]
    workflow.set_axis_off()
    stages = ("Raw image", "Root + scale", "Hair identity", "Fusion", "Traits", "0–5 mm profile")
    xs = np.linspace(0.07, 0.93, len(stages))
    for index, (x, label) in enumerate(zip(xs, stages, strict=True)):
        workflow.add_patch(Rectangle((x - 0.07, 0.43), 0.14, 0.22, facecolor=PALETTE["pale_teal"], edgecolor=PALETTE["teal"]))
        workflow.text(x, 0.54, label.replace(" ", "\n", 1), ha="center", va="center", fontsize=6.4)
        if index + 1 < len(stages):
            workflow.add_patch(FancyArrowPatch((x + 0.07, 0.54), (xs[index + 1] - 0.07, 0.54), arrowstyle="->", mutation_scale=8, color=PALETTE["ink"]))
    workflow.text(0.05, 0.22, "Measurements retain\nsupport state", fontsize=7)
    workflow.text(0.05, 0.10, "One distal landmark;\ncondition-agnostic analysis", fontsize=7)
    workflow.set_title("Hash-linked measurement workflow", loc="left")
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index), x=-0.08, y=1.04)
    _watermark(figure, provisional)
    return figure


def _draw_matcher_contract(axis: plt.Axes) -> None:
    """Draw the locked partial-curve matcher rather than a point-distance proxy."""

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    truth_x = np.asarray([0.08, 0.20, 0.34, 0.49, 0.65, 0.82, 0.92])
    truth_y = np.asarray([0.25, 0.34, 0.49, 0.61, 0.66, 0.61, 0.53])
    prediction_x = np.asarray([0.10, 0.22, 0.36, 0.51, 0.67, 0.83, 0.90])
    prediction_y = np.asarray([0.17, 0.27, 0.42, 0.54, 0.58, 0.54, 0.47])
    axis.plot(
        truth_x,
        truth_y,
        color=PALETTE["navy"],
        linewidth=1.8,
        label="Annotated centreline",
    )
    axis.plot(
        prediction_x,
        prediction_y,
        color=PALETTE["teal"],
        linewidth=1.8,
        linestyle="--",
        label="Predicted centreline",
    )
    # Four of seven displayed resampling locations are mutually supported;
    # the production matcher uses 32 equally spaced samples on each curve.
    supported = slice(2, 6)
    axis.scatter(
        truth_x[supported],
        truth_y[supported],
        facecolor="white",
        edgecolor=PALETTE["gold"],
        linewidth=0.8,
        s=18,
        zorder=4,
    )
    axis.scatter(
        prediction_x[supported],
        prediction_y[supported],
        facecolor="white",
        edgecolor=PALETTE["gold"],
        linewidth=0.8,
        marker="s",
        s=16,
        zorder=4,
    )
    for index in range(2, 6):
        axis.plot(
            [truth_x[index], prediction_x[index]],
            [truth_y[index], prediction_y[index]],
            color=PALETTE["gold"],
            linewidth=0.55,
            alpha=0.75,
        )
    for x, y in (
        (truth_x[:2], truth_y[:2]),
        (prediction_x[:2], prediction_y[:2]),
    ):
        axis.add_patch(
            FancyArrowPatch(
                (float(x[0]), float(y[0])),
                (float(x[1]), float(y[1])),
                arrowstyle="->",
                mutation_scale=7,
                color=PALETTE["ink"],
                linewidth=0.8,
            )
        )
    axis.text(0.05, 0.90, "32-point equal-arc resampling", fontsize=6.2)
    axis.text(0.05, 0.81, "truth→prediction support ≥25%", fontsize=6.2)
    axis.text(0.05, 0.73, "prediction→truth support ≥25%", fontsize=6.2)
    axis.text(0.57, 0.37, "paired distance ≤20 µm", fontsize=6.1, ha="center")
    axis.text(0.05, 0.04, "proximal directions non-opposing (cosine ≥0)", fontsize=6.1)
    axis.text(0.94, 0.56, "truth", ha="right", fontsize=5.8, color=PALETTE["navy"])
    axis.text(0.94, 0.42, "prediction", ha="right", fontsize=5.8, color=PALETTE["teal"])
    axis.set_title("Matcher contract", loc="left")


def _figure2(resources: Mapping[str, Path], provisional: bool, final: bool) -> plt.Figure:
    per_image = _read_table(
        resources["development_per_image"],
        "development per-image",
        (
            "source_unit",
            "source_unit_order",
            "family_key",
            "comparator",
            "gt_count",
            "predicted_count",
            "biological_presence_tp_5um",
            "biological_presence_tp_10um",
            "biological_presence_tp_20um",
            "prediction_input_sha256",
            "prediction_input_set_identity_sha256",
            "prediction_input_schema_version",
            "identity_hair_variant",
            "evidence_role",
        ),
    )
    tolerance = _read_table(
        resources["development_tolerance"],
        "development tolerance",
        (
            "comparator",
            "tolerance_um",
            "precision",
            "recall",
            "f1",
            "ci_low",
            "ci_high",
            "paired_delta_stageb_minus_legacy_f1",
            "paired_delta_ci_low",
            "paired_delta_ci_high",
        ),
    )
    threshold = _read_table(resources["development_threshold"], "development threshold", ("threshold", "f1_20um", "count_mae", "selected"))
    strata = _read_table(resources["development_strata"], "development strata", ("dimension", "stratum", "comparator", "f1_20um", "ci_low", "ci_high", "n_images"))
    assurance = _read_table(
        resources["assurance_metrics"],
        "development attachment assurance",
        ("domain", "metric_key", "label", "value", "ci_low", "ci_high", "unit", "n"),
    )
    _finite(
        per_image,
        (
            "source_unit_order",
            "gt_count",
            "predicted_count",
            "biological_presence_tp_5um",
            "biological_presence_tp_10um",
            "biological_presence_tp_20um",
        ),
        "development per-image",
    )
    _finite(tolerance, ("tolerance_um", "precision", "recall", "f1", "ci_low", "ci_high"), "development tolerance")
    _require(set(tolerance["comparator"]) == set(COMPARATORS), "development comparators changed")
    _require(set(pd.to_numeric(tolerance["tolerance_um"])) == {5, 10, 20}, "matcher tolerances must be 5/10/20 µm")
    if final:
        required_presence_columns = {
            "primary_metric",
            "minimum_truth_coverage",
            "minimum_prediction_coverage",
            "minimum_direction_cosine",
            "endpoint_gate_used",
        }
        _require(
            required_presence_columns.issubset(tolerance.columns)
            and set(tolerance["primary_metric"])
            == {"one_to_one_tolerant_biological_hair_presence"}
            and np.allclose(pd.to_numeric(tolerance["minimum_truth_coverage"]), 0.25)
            and np.allclose(pd.to_numeric(tolerance["minimum_prediction_coverage"]), 0.25)
            and np.allclose(pd.to_numeric(tolerance["minimum_direction_cosine"]), 0.0)
            and not _bool_series(tolerance["endpoint_gate_used"]).any(),
            "final Figure 2 primary metric is not tolerant biological-hair presence",
        )
        _require(
            {
                "selection_metric",
                "attachment_proxy_f1_20um",
                "straight_base_to_tip_presence_proxy_used",
                "distal_endpoint_or_length_used_as_selection_gate",
            }.issubset(threshold.columns)
            and set(threshold["selection_metric"])
            == {"tolerant_biological_presence_f1_20um"}
            and _bool_series(
                threshold["straight_base_to_tip_presence_proxy_used"]
            ).all()
            and not _bool_series(
                threshold["distal_endpoint_or_length_used_as_selection_gate"]
            ).any(),
            "operating-point selection is not the locked biological-presence metric",
        )
        for comparator in COMPARATORS:
            subset = per_image[per_image["comparator"] == comparator].sort_values(
                "source_unit_order"
            )
            _require(subset["source_unit"].nunique() == 44, f"{comparator}: QC-development scope changed")
            _require(
                list(pd.to_numeric(subset["source_unit_order"]).astype(int))
                == list(range(44)),
                f"{comparator}: ordered QC44 task identity changed",
            )
            _require(
                subset["prediction_input_sha256"].map(_is_sha256).all()
                and subset["prediction_input_set_identity_sha256"].nunique() == 1
                and _is_sha256(
                    subset["prediction_input_set_identity_sha256"].iloc[0]
                ),
                f"{comparator}: prediction-file authority is not hash-locked",
            )
            _require(
                (
                    pd.to_numeric(subset["biological_presence_tp_5um"])
                    <= pd.to_numeric(subset["biological_presence_tp_10um"])
                ).all()
                and (
                    pd.to_numeric(subset["biological_presence_tp_10um"])
                    <= pd.to_numeric(subset["biological_presence_tp_20um"])
                ).all(),
                f"{comparator}: tolerance sufficient statistics are not monotone",
            )
        stage_rows = per_image[per_image["comparator"] == COMPARATORS[0]]
        legacy_rows = per_image[per_image["comparator"] == COMPARATORS[1]]
        _require(
            set(stage_rows["prediction_input_schema_version"])
            == {STAGEB_DETECTION_SCHEMA}
            and set(stage_rows["evidence_role"])
            == {EVALUATION_ARTIFACT_ROLE}
            and set(legacy_rows["prediction_input_schema_version"])
            == {LEGACY_HYBRID_COMPARATOR_SCHEMA}
            and set(legacy_rows["identity_hair_variant"])
            == {LEGACY_HYBRID_IDENTITY_VARIANT}
            and set(legacy_rows["evidence_role"])
            == {"locked_legacy_development_comparator"},
            "QC44 comparator schema/legacy identity semantics changed",
        )
        _require(per_image.groupby("source_unit")["family_key"].nunique().max() == 1, "family identity drift")
        for comparator in COMPARATORS:
            subset = per_image[per_image["comparator"] == comparator]
            for tolerance_um in (5, 10, 20):
                tp = int(
                    pd.to_numeric(
                        subset[f"biological_presence_tp_{tolerance_um}um"]
                    ).sum()
                )
                n_pred = int(pd.to_numeric(subset["predicted_count"]).sum())
                n_gt = int(pd.to_numeric(subset["gt_count"]).sum())
                pooled = tolerance[
                    (tolerance["comparator"] == comparator)
                    & (pd.to_numeric(tolerance["tolerance_um"]) == tolerance_um)
                ]
                _require(len(pooled) == 1, f"{comparator}@{tolerance_um}: pooled row missing")
                precision, recall, f1 = (
                    tp / n_pred,
                    tp / n_gt,
                    2 * tp / (n_pred + n_gt),
                )
                _require(
                    all(
                        math.isclose(
                            observed,
                            float(pooled.iloc[0][field]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for observed, field in (
                            (precision, "precision"),
                            (recall, "recall"),
                            (f1, "f1"),
                        )
                    ),
                    f"{comparator}@{tolerance_um}: figure metric is not recomputable from per-image TP",
                )
    selected = threshold[threshold["selected"].astype(str).str.casefold().isin({"true", "1", "yes"})]
    _require(len(selected) == 1, "threshold table must contain one selected operating point")

    assignment = _read_object(resources["qcdev_assignment"], "QC-development assignment")
    assignment_identity = assignment.get("assignment_identity_sha256")
    unsigned_assignment = deepcopy(assignment)
    unsigned_assignment.pop("assignment_identity_sha256", None)
    _require(
        assignment.get("schema_version") == "PHAxis-qcdev-instance-assignment-1.0"
        and assignment.get("blind_images_used") == 0
        and _is_sha256(assignment_identity)
        and sha256_json(unsigned_assignment) == assignment_identity,
        "QC-development assignment authority is invalid",
    )
    assignments = assignment.get("assignments")
    _require(isinstance(assignments, list) and assignments, "QC-development assignment rows missing")
    selected_assignment = next(
        (row for row in assignments if row.get("source_unit") == assignment.get("display_source_unit")),
        None,
    )
    _require(isinstance(selected_assignment, Mapping), "QC-development display assignment missing")

    configure_publication_style()
    figure, axes_grid = plt.subplots(2, 2)
    figure.subplots_adjust(wspace=0.34, hspace=0.48, bottom=0.14)
    axes = list(axes_grid.ravel())

    axis = axes[0]
    matches = selected_assignment["matches"]
    matched_pred = {int(row["predicted_index"]) for row in matches}
    matched_truth = {int(row["annotated_index"]) for row in matches}
    for index, polyline in enumerate(selected_assignment["annotated_polylines_xy_um"]):
        points = np.asarray(polyline, dtype=float)
        axis.plot(points[:, 0], points[:, 1], color=PALETTE["navy"] if index in matched_truth else PALETTE["plum"], linewidth=1.4, linestyle="-" if index in matched_truth else ":")
    for index, polyline in enumerate(selected_assignment["predicted_polylines_xy_um"]):
        points = np.asarray(polyline, dtype=float)
        axis.plot(points[:, 0], points[:, 1], color=PALETTE["teal"] if index in matched_pred else PALETTE["orange"], linewidth=1.1, linestyle="--")
    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis()
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_title(f"Actual assignment: {selected_assignment['source_unit']}", loc="left")
    axis.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["navy"], label="matched truth"),
            Line2D([0], [0], color=PALETTE["teal"], linestyle="--", label="matched prediction"),
            Line2D([0], [0], color=PALETTE["plum"], linestyle=":", label="unmatched truth"),
            Line2D([0], [0], color=PALETTE["orange"], linestyle="--", label="unmatched prediction"),
        ],
        fontsize=5.5,
        frameon=False,
        loc="best",
    )
    _clean_axis(axis)

    axis = axes[1]
    primary_rows = tolerance[pd.to_numeric(tolerance["tolerance_um"]) == 20].set_index("comparator")
    _require(set(primary_rows.index) == set(COMPARATORS), "primary 20-µm rows missing")
    labels = ["PHAxis", "Frozen predecessor"]
    positions = np.arange(2)
    values = np.asarray([float(primary_rows.loc[key, "f1"]) for key in COMPARATORS])
    low = np.asarray([float(primary_rows.loc[key, "ci_low"]) for key in COMPARATORS])
    high = np.asarray([float(primary_rows.loc[key, "ci_high"]) for key in COMPARATORS])
    axis.errorbar(positions, values, yerr=[values - low, high - values], fmt="o", color=PALETTE["teal"], capsize=3)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Biological-presence F1 @20 µm")
    primary_delta = primary_rows.loc[COMPARATORS[0]]
    axis.text(0.03, 0.08, f"paired ΔF1 {float(primary_delta['paired_delta_stageb_minus_legacy_f1']):+.3f}\n95% CI [{float(primary_delta['paired_delta_ci_low']):+.3f}, {float(primary_delta['paired_delta_ci_high']):+.3f}]", transform=axis.transAxes, fontsize=6.2)
    axis.set_title("Primary identity recovery", loc="left")
    _clean_axis(axis)

    stage = per_image[per_image["comparator"] == COMPARATORS[0]].copy()
    axis = axes[2]
    maximum = max(float(stage["gt_count"].max()), float(stage["predicted_count"].max()), 1.0)
    axis.scatter(stage["gt_count"], stage["predicted_count"], color=PALETTE["teal"], alpha=0.72, s=14)
    axis.plot([0, maximum], [0, maximum], color=PALETTE["ink"], linestyle="--", linewidth=0.8)
    error = stage["predicted_count"] - stage["gt_count"]
    mae = float(np.mean(np.abs(error)))
    axis.text(0.04, 0.93, f"MAE {mae:.2f}", transform=axis.transAxes, va="top", fontsize=7)
    axis.set_xlabel("Annotated count")
    axis.set_ylabel("Predicted count")
    axis.set_title("Per-source visible-hair count", loc="left")
    _clean_axis(axis)

    stage_pair = stage.set_index("source_unit")
    legacy_pair = per_image[
        per_image["comparator"] == COMPARATORS[1]
    ].set_index("source_unit")
    _require(
        set(stage_pair.index) == set(legacy_pair.index),
        "paired image-level comparator scope changed",
    )
    stage_pair = stage_pair.sort_index()
    legacy_pair = legacy_pair.reindex(stage_pair.index)
    stage_denominator = (
        pd.to_numeric(stage_pair["predicted_count"])
        + pd.to_numeric(stage_pair["gt_count"])
    ).to_numpy(float)
    legacy_denominator = (
        pd.to_numeric(legacy_pair["predicted_count"])
        + pd.to_numeric(legacy_pair["gt_count"])
    ).to_numpy(float)
    stage_image_f1 = np.divide(
        2.0
        * pd.to_numeric(stage_pair["biological_presence_tp_20um"]).to_numpy(float),
        stage_denominator,
        out=np.full(len(stage_pair), np.nan),
        where=stage_denominator > 0,
    )
    legacy_image_f1 = np.divide(
        2.0
        * pd.to_numeric(legacy_pair["biological_presence_tp_20um"]).to_numpy(float),
        legacy_denominator,
        out=np.full(len(legacy_pair), np.nan),
        where=legacy_denominator > 0,
    )
    paired_valid = np.isfinite(stage_image_f1) & np.isfinite(legacy_image_f1)
    _require(paired_valid.any(), "paired image-level F1 display has no finite images")
    axis = axes[3]
    count_delta = (
        np.abs(pd.to_numeric(stage_pair["predicted_count"]) - pd.to_numeric(stage_pair["gt_count"]))
        - np.abs(pd.to_numeric(legacy_pair["predicted_count"]) - pd.to_numeric(legacy_pair["gt_count"]))
    ).to_numpy(float)
    f1_delta = stage_image_f1 - legacy_image_f1
    axis.scatter(
        f1_delta[paired_valid],
        count_delta[paired_valid],
        color=PALETTE["teal"],
        alpha=0.72,
        s=14,
    )
    axis.axvline(0, color=PALETTE["grey"], linestyle="--", linewidth=0.8)
    axis.axhline(0, color=PALETTE["grey"], linestyle="--", linewidth=0.8)
    improved = int(
        np.sum(stage_image_f1[paired_valid] > legacy_image_f1[paired_valid])
    )
    tied = int(
        np.sum(
            np.isclose(
                stage_image_f1[paired_valid],
                legacy_image_f1[paired_valid],
                rtol=0.0,
                atol=1e-12,
            )
        )
    )
    axis.text(
        0.04,
        0.95,
        f"Stage B higher in {improved}/{int(paired_valid.sum())}; tied {tied}",
        transform=axis.transAxes,
        va="top",
        fontsize=6.2,
    )
    axis.set_xlabel("ΔF1 (PHAxis − predecessor)")
    axis.set_ylabel("Δ absolute count error\n(PHAxis − predecessor)")
    axis.set_title("Paired source-image changes", loc="left")
    _clean_axis(axis)
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index), x=-0.18, y=1.07)
    figure.text(
        0.05,
        0.02,
        "QC-development44 selected the operating point; tolerance, attachment, Bland–Altman and strata evidence are retained in Figs S2–S3.",
        fontsize=7,
        color=PALETTE["grey"],
    )
    _watermark(figure, provisional)
    return figure


def _metric_card(axis: plt.Axes, rows: pd.DataFrame, title: str) -> None:
    axis.set_axis_off()
    axis.set_title(title, loc="left")
    for index, (_, row) in enumerate(rows.iterrows()):
        y = 0.66 - index * (0.72 / max(len(rows), 1))
        value = float(row["value"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        unit = str(row["unit"])
        # Metric names and their intervals are stacked instead of competing
        # for one narrow row.  This remains legible for the five continuity
        # metrics at final two-column print width and keeps the evidence-unit
        # denominator explicit.
        axis.text(
            0.03,
            y + 0.030,
            str(row["label"]),
            va="center",
            fontsize=6.0,
        )
        unit_text = "" if unit == "fraction" else f" {unit}"
        axis.text(
            0.03,
            y - 0.030,
            f"{value:.3g} [{low:.3g}, {high:.3g}]{unit_text} · n={int(row['n'])}",
            ha="left",
            va="center",
            fontsize=5.8,
        )
        axis.plot([0.03, 0.97], [y - 0.065, y - 0.065], color=PALETTE["light_grey"], linewidth=0.5)


def _figure3(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    metrics = _read_table(
        resources["assurance_metrics"],
        "assurance metrics",
        (
            "domain", "metric_key", "label", "value", "ci_low", "ci_high",
            "unit", "n", "instances", "evidence_role",
            "scale_visible_truth_n", "scale_trusted_metadata_n",
            "scale_absence_test_n", "scale_absence_specificity_status",
            "scale_fail_closed_evidence_basis",
        ),
    )
    pairs = _read_table(resources["assurance_pairs"], "assurance pairs", ("pair_type", "source_unit", "observed", "predicted", "unit"))
    support = _read_table(resources["assurance_support"], "assurance support", ("condition_code", "support_fraction", "supported_hairs", "identity_hairs", "source_units"))
    _finite(metrics, ("value", "ci_low", "ci_high", "n"), "assurance metrics")
    prohibited = metrics["metric_key"].astype(str).str.casefold().str.contains("root_cap.*region", regex=True)
    _require(not prohibited.any(), "root-cap region metrics are forbidden")
    required_domains = {
        "root",
        "root_continuity",
        "hair_attachment",
        "distal",
        "scale",
        "conditional_length",
        "root_trait",
        "provider_equivalence",
    }
    _require(required_domains.issubset(set(metrics["domain"])), "assurance evidence domains incomplete")
    required_metrics = {
        "root_dice",
        "root_boundary_f1",
        "root_hd95_um",
        "distal_median_error_um",
        "distal_pck",
        "scale_detection_coverage",
        "scale_geometry_endpoint_error_um",
        "scale_relative_error_percent",
        "conditional_length_mae_um",
        "conditional_length_ccc",
        "matched_endpoint_error_um",
        "matched_trajectory_continuity",
        "endpoint_complete_support_fraction",
        "root_trait_agreement",
        "axis_containment_median",
        "axis_containment_min",
        "unsupported_attachment_n",
        "provider_exact_fraction",
        "root_continuity_maximum_single_component_coverage_mean",
        "root_continuity_maximum_single_component_coverage_median",
        "root_continuity_best_component_gap_median_um",
        "root_continuity_break_free_rate",
        "root_continuity_visible_axis_extent_mae_um",
        "hair_attachment_qualified_precision_20um",
        "hair_attachment_qualified_recall_20um",
        "hair_attachment_qualified_f1_20um",
        "hair_attachment_error_median_um",
        "hair_attachment_error_p95_um",
    }
    _require(
        required_metrics.issubset(set(metrics["metric_key"])),
        "assurance metric set is incomplete",
    )

    configure_publication_style()
    figure = plt.figure(figsize=(7.2, 7.0))
    grid = figure.add_gridspec(
        3,
        3,
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.96,
        height_ratios=(1.0, 1.0, 1.12),
        hspace=0.72,
        wspace=0.48,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2]),
        figure.add_subplot(grid[2, :]),
    ]
    _metric_card(
        axes[0],
        metrics[metrics["domain"] == "root"],
        "Primary-root body accuracy",
    )
    continuity_keys = [
        "root_continuity_maximum_single_component_coverage_mean",
        "root_continuity_maximum_single_component_coverage_median",
        "root_continuity_best_component_gap_median_um",
        "root_continuity_break_free_rate",
        "root_continuity_visible_axis_extent_mae_um",
    ]
    continuity_rows = metrics[
        metrics["metric_key"].astype(str).isin(continuity_keys)
    ].copy()
    continuity_rows["metric_key"] = pd.Categorical(
        continuity_rows["metric_key"],
        categories=continuity_keys,
        ordered=True,
    )
    continuity_rows = continuity_rows.sort_values("metric_key")
    _require(
        len(continuity_rows) == len(continuity_keys),
        "root-continuity display family is incomplete",
    )
    _metric_card(
        axes[1],
        continuity_rows,
        "Continuous visible-root axis",
    )

    axis = axes[2]
    scale_pairs = pairs[pairs["pair_type"] == "scale"]
    _require(len(scale_pairs) > 0, "scale agreement pairs missing")
    scale_metric_rows = metrics[
        metrics["metric_key"].isin(
            (
                "scale_detection_coverage",
                "scale_geometry_endpoint_error_um",
                "scale_relative_error_percent",
            )
        )
    ].set_index("metric_key")
    _require(
        len(scale_metric_rows) == 3,
        "scale coverage/localization/calibration cells are not unique",
    )
    scale_context = metrics.iloc[0]
    visible_n = int(scale_context["scale_visible_truth_n"])
    trusted_n = int(scale_context["scale_trusted_metadata_n"])
    absence_n = int(scale_context["scale_absence_test_n"])
    _require(
        (visible_n, trusted_n, absence_n) == (37, 7, 0)
        and str(scale_context["scale_absence_specificity_status"])
        == "not_estimable_no_absent_or_untrusted_scale_cases"
        and str(scale_context["scale_fail_closed_evidence_basis"])
        == "software_contract_and_unit_tests",
        "scale applicability annotation contract changed",
    )
    maximum = max(float(scale_pairs[["observed", "predicted"]].max().max()), 1e-6)
    axis.scatter(scale_pairs["observed"], scale_pairs["predicted"], color=PALETTE["gold"], s=14, alpha=0.75)
    axis.plot([0, maximum], [0, maximum], color=PALETTE["ink"], linestyle="--")
    axis.set_xlabel(r"Reference µm px$^{-1}$")
    axis.set_ylabel(r"PHAxis µm px$^{-1}$")
    axis.set_title("Distal landmark and scale", loc="left")
    coverage_row = scale_metric_rows.loc["scale_detection_coverage"]
    localization_row = scale_metric_rows.loc["scale_geometry_endpoint_error_um"]
    calibration_row = scale_metric_rows.loc["scale_relative_error_percent"]
    distal_error = metrics[
        metrics["metric_key"] == "distal_median_error_um"
    ]
    distal_pck = metrics[metrics["metric_key"] == "distal_pck"]
    _require(
        len(distal_error) == len(distal_pck) == 1,
        "distal error/PCK cells are not unique",
    )
    axis.text(
        0.03,
        0.97,
        f"Tip {float(distal_error.iloc[0]['value']):.3g} µm; "
        f"PCK25 {float(distal_pck.iloc[0]['value']):.3g}\n"
        f"Scale {int(coverage_row['instances'])}/{int(coverage_row['n'])}; "
        f"error {float(calibration_row['value']):.3g}%\n"
        "7 metadata; absence not estimable",
        transform=axis.transAxes,
        va="top",
        fontsize=5.6,
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5},
    )
    _clean_axis(axis)

    axis = axes[3]
    length_pairs = pairs[pairs["pair_type"] == "conditional_length"]
    _require(len(length_pairs) > 0, "conditional-length pairs missing")
    maximum = max(float(length_pairs[["observed", "predicted"]].max().max()), 1.0)
    axis.scatter(length_pairs["observed"], length_pairs["predicted"], color=PALETTE["plum"], s=12, alpha=0.65)
    axis.plot([0, maximum], [0, maximum], color=PALETTE["ink"], linestyle="--")
    axis.set_xlabel("Annotated complete length (µm)")
    axis.set_ylabel("Matched curve length (µm)")
    axis.set_title("Conditional-length agreement", loc="left")
    length_mae = metrics[metrics["metric_key"] == "conditional_length_mae_um"]
    length_ccc = metrics[metrics["metric_key"] == "conditional_length_ccc"]
    _require(
        len(length_mae) == len(length_ccc) == 1,
        "conditional-length MAE/CCC assurance cells are not unique",
    )
    axis.text(
        0.04,
        0.94,
        f"MAE {float(length_mae.iloc[0]['value']):.3g} µm\n"
        f"CCC {float(length_ccc.iloc[0]['value']):.3g}",
        transform=axis.transAxes,
        va="top",
        fontsize=6.4,
    )
    _clean_axis(axis)

    axis = axes[4]
    ordered = support.set_index("condition_code").reindex(GROUP_ORDER).dropna(how="all").reset_index()
    _require(len(ordered) > 0, "conditional-length group support missing")
    colours = [GROUP_COLOURS.get(code, PALETTE["grey"]) for code in ordered["condition_code"]]
    bars = axis.bar(np.arange(len(ordered)), ordered["support_fraction"], color=colours)
    axis.set_xticks(np.arange(len(ordered)), [GROUP_LABELS.get(code, code) for code in ordered["condition_code"]], rotation=18)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Endpoint-complete fraction")
    for bar, row in zip(bars, ordered.to_dict("records"), strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{int(row['supported_hairs'])}/{int(row['identity_hairs'])}\nn={int(row['source_units'])}", ha="center", fontsize=6.1)
    axis.set_title("Length observability support", loc="left")
    _clean_axis(axis)

    axis = axes[5]
    root_pairs = pairs[pairs["pair_type"].astype(str) == "root_trait"].copy()
    _require(
        len(root_pairs) >= 38
        and {"trait_id", "trait_family"}.issubset(root_pairs.columns)
        and root_pairs["trait_id"].nunique() == 19,
        "Figure 3 root-trait source-unit pairs are incomplete",
    )
    root_summary: list[dict[str, Any]] = []
    for trait_id, rows in root_pairs.groupby("trait_id", sort=False):
        observed = pd.to_numeric(rows["observed"]).to_numpy(float)
        predicted = pd.to_numeric(rows["predicted"]).to_numpy(float)
        root_summary.append(
            {
                "trait_id": str(trait_id),
                "trait_family": str(rows.iloc[0]["trait_family"]),
                "ccc": _sample_ccc(observed, predicted),
            }
        )
    root_agreement = pd.DataFrame(root_summary).sort_values("trait_id")
    _require(
        set(root_agreement["trait_id"]) == set(ROOT_TRAIT_SHORT_LABELS)
        and set(root_agreement["trait_family"]) == set(ROOT_TRAIT_FAMILY_ORDER),
        "Figure 3 canonical root-trait agreement identities changed",
    )
    x = np.arange(len(root_agreement))
    ccc = pd.to_numeric(root_agreement["ccc"], errors="coerce").to_numpy(float)
    colours = [
        ROOT_TRAIT_FAMILY_COLOURS[str(family)]
        for family in root_agreement["trait_family"]
    ]
    finite = np.isfinite(ccc)
    axis.axhline(0.0, color=PALETTE["light_grey"], linewidth=0.8)
    axis.axhline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=0.8)
    axis.vlines(x[finite], 0.0, ccc[finite], color=np.asarray(colours)[finite], linewidth=1.2)
    axis.scatter(
        x[finite],
        ccc[finite],
        c=np.asarray(colours)[finite],
        s=26,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for position in x[~finite]:
        axis.text(position, -0.95, "NA", ha="center", va="bottom", fontsize=5.0)
    axis.set_xticks(
        x,
        [
            f"{trait_id}\n{ROOT_TRAIT_SHORT_LABELS[trait_id]}"
            for trait_id in root_agreement["trait_id"]
        ],
        rotation=58,
        ha="right",
        fontsize=4.6,
    )
    axis.set_xlim(-0.7, len(root_agreement) - 0.3)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Source-unit CCC")
    axis.set_title(
        "Agreement across all 19 primary-root descriptors",
        loc="left",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=ROOT_TRAIT_FAMILY_COLOURS[family],
                markeredgecolor="white",
                label=ROOT_TRAIT_FAMILY_LABELS[family],
            )
            for family in ROOT_TRAIT_FAMILY_ORDER
        ],
        frameon=False,
        fontsize=5.0,
        ncol=3,
        loc="lower left",
    )
    _clean_axis(axis)
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.07)
    figure.text(
        0.08,
        0.025,
        "Attachment assurance is shown with individual-hair recovery in Fig. 2; deployment equivalence is retained in Fig. 6 and Fig. S5. Trait-wise native-unit MAE and bias are in Fig. S4.",
        fontsize=6.2,
        color=PALETTE["grey"],
    )
    _watermark(figure, provisional)
    return figure


def _verify_overlay_rows(selection: pd.DataFrame, base: Path, final: bool) -> list[dict[str, Any]]:
    required = (
        "case_id", "case_role", "source_path", "source_sha256", "overlay_path",
        "overlay_sha256", "scale_bar_um", "scale_bar_px", "display_lower",
        "display_upper", "selection_rule", "case_selection_basis",
        "random_or_representative_performance_sample",
        "experimental_condition_metadata_used_for_rendering",
        "experimental_condition_metadata_used_for_evidence_assembly",
        "experimental_condition_metadata_used_for_evidence_assembly_scope",
        "full_cohort_review_overlay_path",
        "full_cohort_review_overlay_sha256",
        "overlay_bytes_reused_from_full_cohort_review_export",
        "formal_statistics_eligible", "root_boundary_colour", "axis_colour",
        "distal_colour", "length_curve_colour", "identity_vector_colour",
        "hair_base_colour", "visible_endpoint_colour",
        "inset_required", "inset_rule", "inset_x0", "inset_y0",
        "inset_x1", "inset_y1", "inset_geometry_sha256",
    )
    missing = [column for column in required if column not in selection.columns]
    _require(not missing, f"overlay selection: missing columns {missing}")
    _require(set(selection["case_role"]) == set(CASE_ROLES), "overlay case roles changed")
    _require(selection["case_role"].value_counts().eq(1).all(), "overlay roles must be unique")
    records: list[dict[str, Any]] = []
    for row in selection.to_dict("records"):
        record = dict(row)
        expected_colours = {
            "root_boundary_colour": "#19aadc",
            "axis_colour": "#e6e6e6",
            "distal_colour": "#dc3cff",
            "length_curve_colour": "#73f55a",
            "identity_vector_colour": "#ffcd14",
            "hair_base_colour": "#ffff00",
            "visible_endpoint_colour": "#ff6919",
        }
        for field, expected in expected_colours.items():
            _require(
                str(record[field]).casefold() == expected,
                f"{record['case_id']}: overlay semantic colour changed: {field}",
            )
        _require(
            str(record["length_curve_colour"]).casefold()
            != str(record["identity_vector_colour"]).casefold(),
            f"{record['case_id']}: identity and length colours are not distinct",
        )
        _require(
            str(record["experimental_condition_metadata_used_for_evidence_assembly_scope"])
            == OVERLAY_CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE,
            f"{record['case_id']}: condition-metadata evidence-assembly scope changed",
        )
        for prefix in ("source", "overlay"):
            path = Path(str(record[f"{prefix}_path"]))
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            digest = record[f"{prefix}_sha256"]
            _require(_is_sha256(digest), f"{record['case_id']}: invalid {prefix} SHA")
            _require(path.is_file() and not path.is_symlink(), f"{record['case_id']}: missing {prefix} image")
            _require("blind" not in str(path).casefold(), f"{record['case_id']}: blind-labelled image refused")
            _require(sha256_file(path) == digest, f"{record['case_id']}: {prefix} image hash mismatch")
            record[f"{prefix}_resolved"] = path
        if final:
            _require(
                record["overlay_sha256"]
                == record["full_cohort_review_overlay_sha256"]
                and str(
                    record[
                        "overlay_bytes_reused_from_full_cohort_review_export"
                    ]
                ).strip().casefold()
                in {"true", "1", "yes"}
                and str(record["full_cohort_review_overlay_path"]).startswith(
                    "full283_review_overlays/"
                ),
                f"{record['case_id']}: paper overlay is not the exact283 review PNG authority",
            )
            _require(
                str(record["case_selection_basis"]) == OVERLAY_CASE_SELECTION_BASIS,
                f"{record['case_id']}: overlay case-selection basis changed",
            )
            for field in (
                "random_or_representative_performance_sample",
                "experimental_condition_metadata_used_for_rendering",
                "experimental_condition_metadata_used_for_evidence_assembly",
            ):
                _require(
                    str(record[field]).strip().casefold() in {"false", "0", "no"},
                    f"{record['case_id']}: overlay contract requires {field}=false",
                )
            for value in record.values():
                if isinstance(value, str):
                    _require("provisional" not in value.casefold(), f"{record['case_id']}: provisional marker")
        if record["case_role"] == "fail_closed":
            _require(str(record["formal_statistics_eligible"]).casefold() in {"false", "0", "no"}, "fail-closed case entered formal statistics")
        inset_required = str(record["inset_required"]).strip().casefold() in {
            "true", "1", "yes"
        }
        _require(
            inset_required
            == (record["case_role"] in {"low_contrast", "curved_dense"}),
            f"{record['case_id']}: deterministic inset role changed",
        )
        if record["case_role"] in FIGURE4_LOCKED_ANCHOR_TASK_IDS:
            _require(
                str(record.get("task_id"))
                == FIGURE4_LOCKED_ANCHOR_TASK_IDS[record["case_role"]],
                f"{record['case_id']}: prelocked Fig.4 anchor task ID changed",
            )
        if inset_required:
            _require(
                _is_sha256(record["inset_geometry_sha256"])
                and str(record["inset_rule"]) != "not_applicable",
                f"{record['case_id']}: deterministic inset identity missing",
            )
        records.append(record)
    return sorted(records, key=lambda row: CASE_ROLES.index(row["case_role"]))


def _figure4(resources: Mapping[str, Path], provisional: bool, final: bool) -> tuple[plt.Figure, list[dict[str, Any]]]:
    selection = _read_table(resources["overlay_selection"], "overlay selection", ("case_id", "case_role"))
    records = _verify_overlay_rows(selection, resources["overlay_selection"].parent, final)
    audit = _read_table(
        resources["overlay_audit"],
        "overlay audit",
        (
            "schema_version", "case_id", "case_role", "task_id",
            "source_image_sha256", "prediction_sha256", "formal_state",
            "axis_in_root_coverage_fraction",
            "axis_single_component_coverage_fraction",
            "longest_unsupported_axis_gap_um", "formal_identity_count",
            "endpoint_complete_support_count",
            "endpoint_complete_support_fraction",
            "distal_window_1_4mm_eligible", "distal_window_1_4mm_reason",
            "profile_0_5mm_eligible", "profile_0_5mm_reason",
            "downstream_eligible", "downstream_reason",
            "condition_metadata_used",
        ),
    )
    _require(
        set(audit["schema_version"].astype(str))
        == {"PHAxis-Fig4-case-audit-2.0"},
        "Fig.4 audit is not the 2.0 continuity/eligibility contract",
    )
    _require(not audit["case_id"].duplicated().any(), "Fig.4 audit case IDs are duplicated")
    audit_by_case = audit.set_index("case_id")
    _require(set(audit_by_case.index) == {row["case_id"] for row in records}, "Fig.4 overlay/audit case sets differ")
    configure_publication_style()
    figure, axes_grid = plt.subplots(
        len(records), 3, gridspec_kw={"width_ratios": [1.0, 1.0, 0.86]}
    )
    role_titles = {
        "representative": "Representative eligible",
        "low_contrast": "Low-contrast upper root",
        "curved_dense": "Curved dense-hair zone",
        "continuity": "Root-continuity case",
        "fail_closed": "Fail-closed review-only",
    }
    for row_index, record in enumerate(records):
        pair_shape: tuple[int, ...] | None = None
        for column_index, prefix in enumerate(("source", "overlay")):
            axis = axes_grid[row_index, column_index]
            image = _linear_display(Path(record[f"{prefix}_resolved"]), float(record["display_lower"]), float(record["display_upper"]))
            if pair_shape is None:
                pair_shape = tuple(image.shape)
            else:
                _require(
                    tuple(image.shape) == pair_shape,
                    f"{record['case_id']}: source/overlay image shapes differ",
                )
            axis.imshow(image)
            axis.set_axis_off()
            _scale_bar(axis, image_shape=image.shape, pixels=float(record["scale_bar_px"]), micrometres=float(record["scale_bar_um"]))
            if column_index == 0:
                axis.set_title(f"{role_titles[record['case_role']]} — source", loc="left", fontsize=7)
            else:
                axis.set_title("PHAxis overlay", loc="left", fontsize=7)
            if row_index == 0:
                panel_label(axis, "a" if column_index == 0 else "b", x=-0.04, y=1.03)
            inset_required = str(record["inset_required"]).strip().casefold() in {
                "true", "1", "yes"
            }
            if inset_required:
                x0, y0, x1, y1 = (
                    int(float(record[field]))
                    for field in ("inset_x0", "inset_y0", "inset_x1", "inset_y1")
                )
                _require(
                    0 <= x0 < x1 <= image.shape[1]
                    and 0 <= y0 < y1 <= image.shape[0],
                    f"{record['case_id']}: deterministic inset is outside image",
                )
                axis.add_patch(
                    Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor=PALETTE["orange"], linewidth=0.8,
                    )
                )
                inset_axis = axis.inset_axes([0.61, 0.53, 0.36, 0.42])
                inset_axis.imshow(image[y0:y1, x0:x1])
                inset_axis.set_xticks([])
                inset_axis.set_yticks([])
                inset_axis.set_title("locked inset", fontsize=4.5, pad=1.0)
                for spine in inset_axis.spines.values():
                    spine.set_color(PALETTE["orange"])
                    spine.set_linewidth(0.8)
        card = audit_by_case.loc[record["case_id"]]
        _require(str(card["task_id"]) == str(record["task_id"]), f"{record['case_id']}: audit task differs")
        _require(str(card["prediction_sha256"]) == str(record["prediction_sha256"]), f"{record['case_id']}: audit prediction differs")
        _require(str(card["condition_metadata_used"]).strip().casefold() in {"false", "0", "no"}, f"{record['case_id']}: condition metadata entered audit")
        card_axis = axes_grid[row_index, 2]
        card_axis.set_axis_off()
        formal = str(card["formal_state"]) == "formal"
        _require(formal == (record["case_role"] != "fail_closed"), f"{record['case_id']}: audit formal state differs")
        downstream_eligible = str(card["downstream_eligible"]).strip().casefold() in {
            "true", "1", "yes"
        }
        _require(
            downstream_eligible == formal,
            f"{record['case_id']}: audit downstream eligibility differs from formal state",
        )
        if formal:
            axis_text = f"{float(card['axis_in_root_coverage_fraction']):.3f}"
            component_text = f"{float(card['axis_single_component_coverage_fraction']):.3f}"
            gap_text = f"{float(card['longest_unsupported_axis_gap_um']):.1f} µm"
            identity_text = str(int(float(card["formal_identity_count"])))
            support_count = int(float(card["endpoint_complete_support_count"]))
            support_fraction = card["endpoint_complete_support_fraction"]
            support_text = (
                f"{support_count}/{identity_text} ({float(support_fraction):.1%})"
                if not pd.isna(support_fraction)
                else f"{support_count}/{identity_text} (NA)"
            )
        else:
            formal_value_fields = (
                "axis_in_root_coverage_fraction",
                "axis_single_component_coverage_fraction",
                "longest_unsupported_axis_gap_um",
                "formal_identity_count",
                "endpoint_complete_support_count",
                "endpoint_complete_support_fraction",
            )
            _require(
                all(pd.isna(card[field]) for field in formal_value_fields),
                f"{record['case_id']}: review-only audit carries formal values",
            )
            axis_text = component_text = gap_text = identity_text = support_text = "NA"
        window_eligible = str(card["distal_window_1_4mm_eligible"]).strip().casefold() in {
            "true", "1", "yes"
        }
        profile_eligible = str(card["profile_0_5mm_eligible"]).strip().casefold() in {
            "true", "1", "yes"
        }
        card_axis.text(0.03, 0.94, f"{record['task_id']}", va="top", fontsize=6.2, fontweight="bold")
        card_axis.text(
            0.03,
            0.79,
            "\n".join(
                (
                    f"state  {card['formal_state']}",
                    f"axis-in-root  {axis_text}",
                    f"one component  {component_text}",
                    f"unsupported gap  {gap_text}",
                    f"formal identity n  {identity_text}",
                    f"endpoint support  {support_text}",
                    f"[1,4) mm  {'eligible' if window_eligible else 'no'} — {card['distal_window_1_4mm_reason']}",
                    f"[0,5) mm  {'eligible' if profile_eligible else 'no'} — {card['profile_0_5mm_reason']}",
                    f"downstream  {'eligible' if formal else 'excluded'} — {card['downstream_reason']}",
                )
            ),
            va="top",
            fontsize=4.7,
            linespacing=1.24,
            wrap=True,
        )
        card_axis.add_patch(Rectangle((0.0, 0.04), 0.98, 0.92, fill=False, edgecolor=PALETTE["grey"], linewidth=0.7, transform=card_axis.transAxes))
        if row_index == 0:
            card_axis.set_title("Machine audit", loc="left", fontsize=7)
            panel_label(card_axis, "c", x=-0.04, y=1.03)
    handles = (
        Line2D([0], [0], color="#19AADC", label="Root boundary"),
        Line2D([0], [0], color="#E6E6E6", label="Ordered axis"),
        Line2D([0], [0], marker="o", color="#DC3CFF", linestyle="none", label="Distal point"),
        Line2D([0], [0], color="#73F55A", label="Length curve"),
        Line2D([0], [0], color="#FFCD14", label="Identity vector"),
    )
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    figure.text(0.04, 0.015, "Preselected morphology challenges; not a performance sample. Orange boxes/insets are axis-geometry-derived and identical in source/overlay views.", fontsize=6.5, color=PALETTE["grey"])
    _watermark(figure, provisional)
    return figure, records


def _jitter(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.17, 0.17, count)


def _figure5_submission_readability_qa(
    figure: plt.Figure,
    *,
    raw_axes: Sequence[plt.Axes],
    effect_axes: Sequence[plt.Axes],
    profile_axes: Sequence[plt.Axes],
    provisional: bool,
) -> dict[str, Any]:
    """Prove that Figure 5 remains readable at its submitted physical size.

    This is deliberately a rendering gate rather than a style suggestion.  It
    catches the failure mode in which a large interactive canvas is later
    reduced to 178 x 148 mm and seemingly acceptable 4--5 pt labels become
    unreadable.  The complete 32-trait heatmap is owned by Figure S9; the main
    figure must instead preserve legible endpoint, effect and spatial-profile
    evidence.
    """

    _require(len(raw_axes) == 5, "Figure 5 must retain five endpoint panels")
    _require(len(effect_axes) == 5, "Figure 5 must retain five effect panels")
    _require(len(profile_axes) == 3, "Figure 5 must retain three profile panels")
    figure.set_size_inches(
        FIGURE_5_SUBMISSION_WIDTH_MM / 25.4,
        FIGURE_5_SUBMISSION_HEIGHT_MM / 25.4,
        forward=True,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    text_artists = [
        artist
        for artist in figure.findobj(match=matplotlib.text.Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    submitted_text = [
        artist
        for artist in text_artists
        if not (
            provisional
            and artist.get_text() == "PROVISIONAL — NOT FOR SUBMISSION"
        )
    ]
    _require(bool(submitted_text), "Figure 5 contains no submission text")
    minimum_font_pt = min(
        float(artist.get_fontsize()) for artist in submitted_text
    )
    _require(
        minimum_font_pt + 1e-9 >= FIGURE_5_MIN_TEXT_PT,
        "Figure 5 contains text below the 6-point submission floor",
    )

    allowed_text_rgb = {
        tuple(matplotlib.colors.to_rgba(PALETTE["ink"])[:3]),
        tuple(matplotlib.colors.to_rgba(PALETTE["grey"])[:3]),
        tuple(matplotlib.colors.to_rgba("black")[:3]),
    }
    coloured_text = []
    for artist in submitted_text:
        rgb = tuple(matplotlib.colors.to_rgba(artist.get_color())[:3])
        if rgb not in allowed_text_rgb:
            coloured_text.append(artist.get_text())
    _require(
        not coloured_text,
        f"Figure 5 uses category-coloured text: {coloured_text[:3]}",
    )

    expected_panel_labels = {f"({letter})" for letter in "abcde"}
    rendered_panel_labels = {
        artist.get_text()
        for axis in [*raw_axes, *effect_axes, *profile_axes]
        for artist in axis.texts
        if artist.get_text().startswith("(") and artist.get_text().endswith(")")
    }
    _require(
        rendered_panel_labels == expected_panel_labels,
        "Figure 5 panel labels must be exactly (a)--(e)",
    )

    figure_bbox = figure.bbox
    outside = []
    canvas_insets_px: list[float] = []
    for artist in submitted_text:
        bbox = artist.get_window_extent(renderer=renderer)
        canvas_insets_px.extend(
            (
                bbox.x0 - figure_bbox.x0,
                bbox.y0 - figure_bbox.y0,
                figure_bbox.x1 - bbox.x1,
                figure_bbox.y1 - bbox.y1,
            )
        )
        if (
            bbox.x0 < figure_bbox.x0 - 1.0
            or bbox.y0 < figure_bbox.y0 - 1.0
            or bbox.x1 > figure_bbox.x1 + 1.0
            or bbox.y1 > figure_bbox.y1 + 1.0
        ):
            outside.append(artist.get_text())
    _require(
        not outside,
        f"Figure 5 text leaves the submission canvas: {outside[:3]}",
    )
    minimum_text_canvas_inset_px = min(canvas_insets_px)
    _require(
        minimum_text_canvas_inset_px >= 3.0,
        "Figure 5 text is too close to the submission-canvas edge: "
        f"{minimum_text_canvas_inset_px:.3f} px at 100 dpi",
    )

    def _minimum_axis_extent_mm(
        axes: Sequence[plt.Axes],
    ) -> tuple[float, float]:
        return (
            min(axis.get_position().width for axis in axes)
            * FIGURE_5_SUBMISSION_WIDTH_MM,
            min(axis.get_position().height for axis in axes)
            * FIGURE_5_SUBMISSION_HEIGHT_MM,
        )

    raw_extent = _minimum_axis_extent_mm(raw_axes)
    effect_extent = _minimum_axis_extent_mm(effect_axes)
    profile_extent = _minimum_axis_extent_mm(profile_axes)
    _require(
        raw_extent[0] >= 23.0 and raw_extent[1] >= 22.0,
        f"Figure 5 endpoint panels are undersized at submission scale: {raw_extent}",
    )
    _require(
        effect_extent[0] >= 23.0 and effect_extent[1] >= 19.5,
        f"Figure 5 effect panels are undersized at submission scale: {effect_extent}",
    )
    _require(
        profile_extent[0] >= 41.0 and profile_extent[1] >= 22.0,
        f"Figure 5 profile panels are undersized at submission scale: {profile_extent}",
    )

    def _overlap_pairs(artists: Sequence[matplotlib.text.Text]) -> list[str]:
        boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
        overlaps: list[str] = []
        for left_index, left_box in enumerate(boxes):
            for right_index in range(left_index + 1, len(boxes)):
                right_box = boxes[right_index]
                overlap_width = min(left_box.x1, right_box.x1) - max(
                    left_box.x0, right_box.x0
                )
                overlap_height = min(left_box.y1, right_box.y1) - max(
                    left_box.y0, right_box.y0
                )
                if overlap_width > 0.5 and overlap_height > 0.5:
                    overlaps.append(
                        f"{artists[left_index].get_text()} | "
                        f"{artists[right_index].get_text()}"
                    )
        return overlaps

    tick_overlaps: list[str] = []
    for axis in [*raw_axes, *effect_axes, *profile_axes]:
        for ticks in (axis.get_xticklabels(), axis.get_yticklabels()):
            visible_ticks = [
                tick for tick in ticks if tick.get_visible() and tick.get_text().strip()
            ]
            tick_overlaps.extend(_overlap_pairs(visible_ticks))
    _require(
        not tick_overlaps,
        f"Figure 5 tick labels overlap at submission scale: {tick_overlaps[:3]}",
    )
    profile_annotation_overlaps: list[str] = []
    for axis in profile_axes:
        annotations = [
            artist
            for artist in axis.texts
            if artist.get_visible()
            and artist.get_gid()
            in {"profile-denominator", "profile-denominator-header"}
        ]
        profile_annotation_overlaps.extend(_overlap_pairs(annotations))
    _require(
        not profile_annotation_overlaps,
        "Figure 5 profile denominator annotations overlap at submission scale: "
        f"{profile_annotation_overlaps[:3]}",
    )
    required_profile_data_clearance_px = 1.5
    profile_marker_collisions: list[str] = []
    profile_line_collisions: list[str] = []
    profile_marker_clearances_px: list[float] = []
    for axis in profile_axes:
        annotations = [
            artist
            for artist in axis.texts
            if artist.get_visible()
            and artist.get_gid() == "profile-denominator"
        ]
        marker_boxes: list[Bbox] = []
        line_segments: list[np.ndarray] = []
        for line in axis.lines:
            data = np.asarray(line.get_xydata(), dtype=float)
            data = data[np.all(np.isfinite(data), axis=1)]
            if not len(data):
                continue
            display = axis.transData.transform(data)
            marker = line.get_marker()
            if marker not in (None, "", " ", "None", "none"):
                radius_px = (
                    float(line.get_markersize()) * float(figure.dpi) / 72.0 / 2.0
                )
                marker_boxes.extend(
                    Bbox.from_extents(
                        point[0] - radius_px,
                        point[1] - radius_px,
                        point[0] + radius_px,
                        point[1] + radius_px,
                    )
                    for point in display
                )
            line_segments.extend(
                np.asarray((left, right), dtype=float)
                for left, right in zip(display[:-1], display[1:], strict=True)
            )
        for annotation in annotations:
            annotation_box = annotation.get_window_extent(renderer=renderer)
            padded_annotation_box = Bbox.from_extents(
                annotation_box.x0 - required_profile_data_clearance_px,
                annotation_box.y0 - required_profile_data_clearance_px,
                annotation_box.x1 + required_profile_data_clearance_px,
                annotation_box.y1 + required_profile_data_clearance_px,
            )
            for marker_box in marker_boxes:
                dx = max(
                    annotation_box.x0 - marker_box.x1,
                    marker_box.x0 - annotation_box.x1,
                    0.0,
                )
                dy = max(
                    annotation_box.y0 - marker_box.y1,
                    marker_box.y0 - annotation_box.y1,
                    0.0,
                )
                clearance = float(math.hypot(dx, dy))
                profile_marker_clearances_px.append(clearance)
                if clearance + 1e-9 < required_profile_data_clearance_px:
                    profile_marker_collisions.append(annotation.get_text())
            for segment in line_segments:
                if MatplotlibPath(segment).intersects_bbox(
                    padded_annotation_box,
                    filled=False,
                ):
                    profile_line_collisions.append(annotation.get_text())
    _require(
        not profile_marker_collisions,
        "Figure 5 profile annotations lack marker clearance: "
        f"{profile_marker_collisions[:3]}",
    )
    _require(
        not profile_line_collisions,
        "Figure 5 profile annotations lack line clearance: "
        f"{profile_line_collisions[:3]}",
    )
    minimum_profile_marker_clearance_px = min(profile_marker_clearances_px)
    marker_diameters_pt = [
        float(line.get_markersize())
        for axis in [*raw_axes, *effect_axes, *profile_axes]
        for line in axis.lines
        if line.get_visible()
        and line.get_marker() not in (None, "", " ", "None", "none")
    ]
    marker_diameters_pt.extend(
        float(math.sqrt(size))
        for axis in raw_axes
        for collection in axis.collections
        if isinstance(collection, matplotlib.collections.PathCollection)
        for size in collection.get_sizes()
        if float(size) > 0.0
    )
    _require(marker_diameters_pt, "Figure 5 contains no measurable data symbols")
    minimum_symbol_diameter_pt = min(marker_diameters_pt)
    minimum_symbol_diameter_300dpi_px = (
        minimum_symbol_diameter_pt / 72.0 * 300.0
    )
    _require(
        minimum_symbol_diameter_pt + 1e-9 >= FIGURE_5_MIN_SYMBOL_DIAMETER_PT,
        "Figure 5 data symbols are below the 6-point submission floor: "
        f"{minimum_symbol_diameter_pt:.3f} pt",
    )
    _require(
        minimum_symbol_diameter_300dpi_px + 1e-9
        >= FIGURE_5_MIN_SYMBOL_DIAMETER_300DPI_PX,
        "Figure 5 data symbols are too small at 300-dpi submission scale",
    )
    # H11 needs three compact lines to retain condition identity, the
    # non-null/formal source-root denominator and the endpoint-complete/
    # accepted-hair denominator.  These exact labels are exempt only from the
    # generic two-line tick budget; the stricter submission-size checks above
    # still enforce 6-pt text, canvas containment and zero overlap.
    h11_support_tick_exemptions = tuple(
        item.get_text() for item in raw_axes[1].get_xticklabels()
    )
    try:
        check_text_budget(
            figure,
            explicit_exemptions=h11_support_tick_exemptions,
        )
    except RuntimeError as error:
        raise FigureSuiteError(
            f"Figure 5 violates the submission text budget: {error}"
        ) from error

    expected_sentinel_prefixes = ("N · H08", "L · H11", "F · H07", "W · R07", "A · R01")
    rendered_raw_titles = tuple(axis.get_title(loc="left") for axis in raw_axes)
    _require(
        all(
            title.startswith(prefix)
            for title, prefix in zip(
                rendered_raw_titles, expected_sentinel_prefixes, strict=True
            )
        ),
        "Figure 5 neutral N/L/F/W/A sentinel badges are missing or reordered",
    )
    h11_tick_text = tuple(item.get_text() for item in raw_axes[1].get_xticklabels())
    h07_tick_text = tuple(item.get_text() for item in raw_axes[2].get_xticklabels())
    _require(
        all("\nn " in text and "\nL " in text for text in h11_tick_text),
        "Figure 5 H11 endpoint-complete/accepted-hair support is missing",
    )
    _require(
        all("\nF " in text for text in h07_tick_text),
        "Figure 5 H07 observable/formal-root denominator is missing",
    )
    _require(
        all(axis.get_xscale() == "log" for axis in effect_axes),
        "Figure 5 effect panels must use a log ratio axis",
    )
    effect_limits = [axis.get_xlim() for axis in effect_axes]
    _require(
        all(
            np.allclose(effect_limits[0], limits, rtol=0.0, atol=1e-12)
            for limits in effect_limits[1:]
        ),
        "Figure 5 effect panels must share one ratio-axis range",
    )

    report = {
        "status": "pass_submission_size_readability",
        "width_mm": FIGURE_5_SUBMISSION_WIDTH_MM,
        "height_mm": FIGURE_5_SUBMISSION_HEIGHT_MM,
        "minimum_text_pt": minimum_font_pt,
        "minimum_required_text_pt": FIGURE_5_MIN_TEXT_PT,
        "minimum_data_symbol_diameter_pt": minimum_symbol_diameter_pt,
        "minimum_required_data_symbol_diameter_pt": (
            FIGURE_5_MIN_SYMBOL_DIAMETER_PT
        ),
        "minimum_data_symbol_diameter_300dpi_px": (
            minimum_symbol_diameter_300dpi_px
        ),
        "minimum_required_data_symbol_diameter_300dpi_px": (
            FIGURE_5_MIN_SYMBOL_DIAMETER_300DPI_PX
        ),
        "panel_labels": sorted(rendered_panel_labels),
        "category_coloured_text_count": len(coloured_text),
        "outside_canvas_text_count": len(outside),
        "minimum_text_canvas_inset_px_at_100dpi": minimum_text_canvas_inset_px,
        "overlapping_tick_pair_count": len(tick_overlaps),
        "overlapping_profile_annotation_pair_count": len(
            profile_annotation_overlaps
        ),
        "profile_annotation_required_data_clearance_px_at_100dpi": (
            required_profile_data_clearance_px
        ),
        "profile_annotation_minimum_marker_clearance_px_at_100dpi": (
            minimum_profile_marker_clearance_px
        ),
        "profile_annotation_marker_collision_count": len(
            profile_marker_collisions
        ),
        "profile_annotation_line_collision_count": len(profile_line_collisions),
        "text_budget_pass": True,
        "endpoint_panel_minimum_extent_mm": list(raw_extent),
        "effect_panel_minimum_extent_mm": list(effect_extent),
        "profile_panel_minimum_extent_mm": list(profile_extent),
        "main_figure_descriptor_heatmap_rows": 0,
        "complete_32_descriptor_heatmap_figure": "Figure S9",
        "endpoint_panels": len(raw_axes),
        "effect_slots": len(effect_axes) * len(EFFECT_ORDER),
        "effect_axis_scale": "log2_ratio",
        "effect_axis_common_xlim": list(effect_limits[0]),
        "sentinel_badges": list(expected_sentinel_prefixes),
        "h11_support_annotation": "n non-null/formal source roots; L endpoint-complete/accepted hairs",
        "h11_tick_text_budget_exception": "three_line_semantic_denominator_labels_pass_canvas_overlap_and_font_gates",
        "h07_observability_annotation": "F observable/formal source roots",
        "profile_panels": len(profile_axes),
    }
    setattr(figure, "_phaxis_submission_readability_qa", report)
    return report


def _figure5(resources: Mapping[str, Path], provisional: bool, final: bool) -> plt.Figure:
    points = _read_table(resources["phenotype_points"], "phenotype points", ("source_unit", "cohort", "condition_code", "formal_eligible", "endpoint_key", "value", "unit"))
    effects = _read_table(resources["phenotype_effects"], "phenotype effects", ("cohort", "endpoint_key", "effect_key", "estimate", "ci_low", "ci_high", "endpoint_n"))
    support = _read_table(
        resources["assurance_support"],
        "assurance support",
        (
            "condition_code",
            "support_fraction",
            "supported_hairs",
            "identity_hairs",
            "source_units",
        ),
    )
    profiles = _read_table(resources["axial_profiles"], "axial profiles", ("cohort", "condition_code", "bin_start_mm", "bin_end_mm", "metric_key", "estimate", "ci_low", "ci_high", "eligible_n", "length_supported_n"))
    effects = _ordered_fixed_effects(effects, "phenotype effects")
    _require(set(PRIMARY_ENDPOINTS).issubset(set(points["endpoint_key"])), "five prespecified endpoints missing")
    _require({"primary_clean261", "sensitivity_full283"}.issubset(set(effects["cohort"])), "clean/full effect scopes missing")
    if final:
        _require(set(points["cohort"]) == {"primary_clean261"}, "main raw points must use clean261 only")
        _require(set(profiles["cohort"]) == {"primary_clean261"}, "main profiles must use clean261 only")
    _finite(points, ("value",), "phenotype points")
    _finite(effects, ("estimate", "ci_low", "ci_high", "endpoint_n"), "phenotype effects")
    _finite(
        support,
        (
            "support_fraction",
            "supported_hairs",
            "identity_hairs",
            "source_units",
        ),
        "assurance support",
    )
    _require(
        len(support) == len(GROUP_ORDER)
        and support["condition_code"].astype(str).is_unique
        and set(support["condition_code"].astype(str)) == set(GROUP_ORDER),
        "Figure 5 assurance support must contain the four exact D15 conditions",
    )
    support = support.copy()
    for column in ("supported_hairs", "identity_hairs", "source_units"):
        numeric = pd.to_numeric(support[column], errors="raise").to_numpy(float)
        _require(
            np.allclose(numeric, np.rint(numeric), rtol=0.0, atol=1e-12),
            f"Figure 5 assurance support {column} is not integral",
        )
        support[column] = np.rint(numeric).astype(int)
    support_fraction = pd.to_numeric(
        support["support_fraction"], errors="raise"
    ).to_numpy(float)
    supported_hairs = support["supported_hairs"].to_numpy(int)
    identity_hairs = support["identity_hairs"].to_numpy(int)
    formal_source_units = support["source_units"].to_numpy(int)
    _require(
        np.all(identity_hairs > 0)
        and np.all(formal_source_units > 0)
        and np.all(supported_hairs >= 0)
        and np.all(supported_hairs <= identity_hairs)
        and np.allclose(
            support_fraction,
            supported_hairs / identity_hairs,
            rtol=0.0,
            atol=1e-12,
        ),
        "Figure 5 assurance support counts/fractions are inconsistent",
    )
    support_by_group = {
        str(row["condition_code"]): row for row in support.to_dict("records")
    }
    point_keys = points[["endpoint_key", "condition_code", "source_unit"]].astype(str)
    _require(
        not point_keys.duplicated().any(),
        "Figure 5 phenotype points contain duplicate endpoint/condition/source rows",
    )
    expected_endpoint_contract = tuple(
        (NARRATIVE_ENDPOINT_CONTRACT[endpoint]["sentinel"], NARRATIVE_ENDPOINT_CONTRACT[endpoint]["badge"])
        for endpoint in PRIMARY_ENDPOINTS
    )
    _require(
        expected_endpoint_contract
        == (("H08", "N"), ("H11", "L"), ("H07", "F"), ("R07", "W"), ("R01", "A")),
        "Figure 5 sentinel/badge endpoint contract changed",
    )

    atlas_payload = _read_object(resources["multitrait_atlas"], "main multitrait atlas")
    atlas_matrices = descriptive_heatmap_matrices(atlas_payload)
    atlas_labels = list(atlas_matrices["trait_labels"])
    atlas_medians = np.asarray(atlas_matrices["standardized_medians"], dtype=float)
    atlas_coverage = np.asarray(atlas_matrices["condition_coverage"], dtype=float)
    _require(
        atlas_medians.shape == atlas_coverage.shape == (32, 4),
        "main 32-descriptor atlas matrices changed",
    )
    atlas_ids = [str(label).split(" ", 1)[0] for label in atlas_labels]
    atlas_index = {trait_id: index for index, trait_id in enumerate(atlas_ids)}
    ordered_trait_ids = tuple(
        trait_id
        for family in MEASUREMENT_FAMILY_ORDER
        for trait_id in MEASUREMENT_FAMILY_TRAIT_IDS[family]
    )
    _require(
        len(ordered_trait_ids) == 32
        and set(ordered_trait_ids) == set(ATLAS_TRAIT_SHORT_LABELS)
        and set(atlas_index) == set(ordered_trait_ids),
        "main atlas trait/family identity changed",
    )
    atlas_order = [atlas_index[trait_id] for trait_id in ordered_trait_ids]
    # Reindex and validate the full atlas even though its complete heatmaps are
    # intentionally rendered only in Figure S9.  This preserves the sealed
    # input contract without forcing 32 rows of 4--5 pt text into the main
    # figure.
    atlas_medians = atlas_medians[atlas_order]
    atlas_coverage = atlas_coverage[atlas_order]
    _require(
        np.isfinite(atlas_medians).any() and np.isfinite(atlas_coverage).any(),
        "main Figure 5 atlas authority contains no finite evidence",
    )

    effect_numeric = effects[["estimate", "ci_low", "ci_high"]].apply(
        pd.to_numeric, errors="raise"
    )
    _require(
        (effect_numeric > 0.0).to_numpy().all(),
        "Figure 5 ratio estimates and intervals must be strictly positive for log display",
    )
    _require(
        (
            (effect_numeric["ci_low"] <= effect_numeric["estimate"])
            & (effect_numeric["estimate"] <= effect_numeric["ci_high"])
        ).all(),
        "Figure 5 ratio estimate must lie within its model-based interval",
    )
    effect_low = float(effect_numeric["ci_low"].min())
    effect_high = float(effect_numeric["ci_high"].max())
    effect_log_low = min(math.log2(effect_low), 0.0)
    effect_log_high = max(math.log2(effect_high), 0.0)
    effect_span = max(effect_log_high - effect_log_low, math.log2(1.1))
    effect_xlim = (
        2.0 ** (effect_log_low - 0.10 * effect_span),
        2.0 ** (effect_log_high + 0.10 * effect_span),
    )

    configure_publication_style()
    figure = plt.figure(
        figsize=(
            FIGURE_5_SUBMISSION_WIDTH_MM / 25.4,
            FIGURE_5_SUBMISSION_HEIGHT_MM / 25.4,
        )
    )
    grid = figure.add_gridspec(
        3,
        1,
        left=0.088,
        right=0.978,
        bottom=0.18,
        top=0.925,
        height_ratios=(1.18, 1.00, 1.22),
        hspace=0.82,
    )
    raw_grid = grid[0].subgridspec(1, 5, wspace=0.42)
    effect_grid = grid[1].subgridspec(1, 5, wspace=0.43)
    profile_grid = grid[2].subgridspec(1, 3, wspace=0.40)
    raw_axes = [
        figure.add_subplot(raw_grid[0, index])
        for index in range(5)
    ]
    effect_axes = [
        figure.add_subplot(effect_grid[0, index])
        for index in range(5)
    ]
    abundance_axis = figure.add_subplot(profile_grid[0, 0])
    support_axis = figure.add_subplot(profile_grid[0, 1])
    length_axis = figure.add_subplot(profile_grid[0, 2])

    endpoint_titles = {
        PRIMARY_ENDPOINTS[0]: "N · H08\nVisible-hair count\n[1,4) mm",
        PRIMARY_ENDPOINTS[1]: "L · H11\nConditional median length\n[1,4) mm",
        PRIMARY_ENDPOINTS[2]: "F · H07\nFirst observed ≥40 µm hair\nfrom distal point",
        PRIMARY_ENDPOINTS[3]: "W · R07\nMedian apparent\nroot width",
        PRIMARY_ENDPOINTS[4]: "A · R01\nVisible root-axis\nextent",
    }
    for endpoint_index, (axis, endpoint) in enumerate(zip(raw_axes, PRIMARY_ENDPOINTS, strict=True)):
        subset = points[(points["endpoint_key"] == endpoint) & points["formal_eligible"].astype(str).str.casefold().isin({"true", "1", "yes"})]
        for group_index, group in enumerate(GROUP_ORDER):
            values = pd.to_numeric(subset.loc[subset["condition_code"] == group, "value"])
            style = GROUP_FACTOR_STYLES[group]
            axis.scatter(
                np.full(len(values), group_index)
                + _jitter(len(values), 8200 + endpoint_index * 11 + group_index),
                values,
                marker=str(style["marker"]),
                facecolors=(
                    str(style["colour"])
                    if bool(style["filled"])
                    else "none"
                ),
                edgecolors=str(style["colour"]),
                linewidths=0.65,
                s=36,
                alpha=0.25,
            )
            if len(values):
                axis.plot(
                    [group_index - 0.22, group_index + 0.22],
                    [values.median(), values.median()],
                    color=str(style["colour"]),
                    linestyle=str(style["linestyle"]),
                    linewidth=1.2,
                )
        group_ns = [int((subset["condition_code"] == group).sum()) for group in GROUP_ORDER]
        tick_labels: list[str] = []
        for group, observed_n in zip(GROUP_ORDER, group_ns, strict=True):
            group_label = (
                f"{'EV' if group.startswith('RHD6_EV') else 'OE'}"
                f"{'22' if group.endswith('22C') else '30'}"
            )
            support_row = support_by_group[group]
            formal_n = int(support_row["source_units"])
            if endpoint == PRIMARY_ENDPOINTS[1]:
                _require(
                    observed_n <= formal_n,
                    f"{group}: H11 non-null root count exceeds the formal denominator",
                )
                tick_labels.append(
                    f"{group_label}\nn {observed_n}/{formal_n}\n"
                    f"L {int(support_row['supported_hairs'])}/{int(support_row['identity_hairs'])}"
                )
            elif endpoint == PRIMARY_ENDPOINTS[2]:
                _require(
                    observed_n <= formal_n,
                    f"{group}: H07 observable root count exceeds the formal denominator",
                )
                tick_labels.append(f"{group_label}\nF {observed_n}/{formal_n}")
            else:
                tick_labels.append(f"{group_label}\nn={observed_n}")
        axis.set_xticks(
            range(4),
            tuple(tick_labels),
            fontsize=6.0,
        )
        axis.set_title(endpoint_titles[endpoint], loc="left", fontsize=6.5)
        axis.set_ylabel(
            str(subset["unit"].iloc[0]) if len(subset) else "",
            fontsize=6.2,
        )
        axis.tick_params(axis="y", labelsize=6.2)
        _clean_axis(axis)
    panel_label(raw_axes[0], "(a)", x=-0.22, y=1.10)

    endpoint_short = {
        PRIMARY_ENDPOINTS[0]: "Count",
        PRIMARY_ENDPOINTS[1]: "Projected length",
        PRIMARY_ENDPOINTS[2]: "First observed hair",
        PRIMARY_ENDPOINTS[3]: "Apparent root width",
        PRIMARY_ENDPOINTS[4]: "Visible-axis extent",
    }
    effect_short = {
        "OE_vs_EV": "OE-labelled",
        "30C_vs_22C": "Temp",
        "interaction": "OE-labelled × Temp",
    }
    effect_rows = effects[effects["cohort"] == "primary_clean261"].copy()
    effect_rows["label"] = [
        f"{endpoint_short.get(endpoint, endpoint)} · {effect_short.get(effect, effect)}"
        for endpoint, effect in zip(
            effect_rows["endpoint_key"], effect_rows["effect_key"], strict=True
        )
    ]
    effect_rows = effect_rows.reset_index(drop=True)
    full_endpoint_n = {
        (str(row["endpoint_key"]), str(row["effect_key"])): int(row["endpoint_n"])
        for row in effects[effects["cohort"] == "sensitivity_full283"].to_dict("records")
    }
    effect_tick_labels = ("Construct", "Temperature", "Interaction")
    for endpoint_index, (axis, endpoint) in enumerate(
        zip(effect_axes, PRIMARY_ENDPOINTS, strict=True)
    ):
        endpoint_effects = effect_rows[
            effect_rows["endpoint_key"] == endpoint
        ].copy()
        _require(
            list(endpoint_effects["effect_key"]) == list(EFFECT_ORDER),
            f"Figure 5 effect order changed for {endpoint}",
        )
        y = np.arange(len(EFFECT_ORDER), dtype=float)
        for cohort, marker_face, y_offset in (
            ("primary_clean261", PALETTE["teal"], -0.09),
            ("sensitivity_full283", "white", 0.09),
        ):
            rows = effects[
                (effects["cohort"] == cohort)
                & (effects["endpoint_key"] == endpoint)
            ].copy()
            lookup = {
                str(row["effect_key"]): row for row in rows.to_dict("records")
            }
            aligned = [lookup[effect] for effect in EFFECT_ORDER]
            estimates = np.asarray([float(row["estimate"]) for row in aligned])
            lows = np.asarray([float(row["ci_low"]) for row in aligned])
            highs = np.asarray([float(row["ci_high"]) for row in aligned])
            axis.errorbar(
                estimates,
                y + y_offset,
                xerr=[estimates - lows, highs - estimates],
                fmt="o",
                color=PALETTE["teal"],
                markerfacecolor=marker_face,
                markeredgecolor=PALETTE["teal"],
                markersize=6.0,
                capsize=3.0,
            )
        clean_n = int(endpoint_effects.iloc[0]["endpoint_n"])
        full_n = full_endpoint_n[(endpoint, EFFECT_ORDER[0])]
        axis.axvline(
            1.0,
            color=PALETTE["ink"],
            linestyle="--",
            linewidth=0.8,
        )
        axis.set_xscale("log", base=2)
        axis.set_xlim(*effect_xlim)
        axis.set_xticks(
            (effect_xlim[0], 1.0, effect_xlim[1]),
            tuple(f"{value:.2g}" for value in (effect_xlim[0], 1.0, effect_xlim[1])),
            fontsize=6.0,
        )
        axis.set_yticks(y, effect_tick_labels, fontsize=6.0)
        axis.invert_yaxis()
        axis.set_title(
            f"n clean/full={clean_n}/{full_n}",
            loc="left",
            fontsize=6.2,
        )
        if endpoint_index == 2:
            axis.set_xlabel(
                "Ratio (log scale; 95% model-based interval)", fontsize=6.4
            )
        _clean_axis(axis, x=True, y=False)
    panel_label(effect_axes[0], "(b)", x=-0.22, y=1.12)
    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PALETTE["teal"],
                markeredgecolor=PALETTE["teal"],
                label="Clean-cohort D15",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=PALETTE["teal"],
                label="Full-cohort D15 sensitivity",
            ),
        ),
        frameon=False,
        fontsize=6.2,
        ncol=2,
        loc="center right",
        bbox_to_anchor=(0.975, 0.458),
    )

    profile_specs = (
        (abundance_axis, "identity_abundance", "Hair identities per bin", "(c)"),
        (support_axis, "length_support_fraction", "Length-support fraction", "(d)"),
        (length_axis, "conditional_median_length_um", "Conditional projected length (µm)", "(e)"),
    )
    for axis, metric, ylabel, letter in profile_specs:
        denominator_by_bin: dict[float, list[tuple[int, int]]] = {}
        for group in GROUP_ORDER:
            subset = profiles[(profiles["metric_key"] == metric) & (profiles["condition_code"] == group)].sort_values("bin_start_mm")
            if not len(subset):
                continue
            style = GROUP_FACTOR_STYLES[group]
            x = (pd.to_numeric(subset["bin_start_mm"]) + pd.to_numeric(subset["bin_end_mm"])) / 2
            values = pd.to_numeric(subset["estimate"], errors="coerce")
            low = pd.to_numeric(subset["ci_low"], errors="coerce")
            high = pd.to_numeric(subset["ci_high"], errors="coerce")
            valid = np.isfinite(values) & (pd.to_numeric(subset["eligible_n"], errors="coerce") > 0)
            axis.plot(
                np.asarray(x)[valid],
                np.asarray(values)[valid],
                marker=str(style["marker"]),
                markerfacecolor=(
                    str(style["colour"])
                    if bool(style["filled"])
                    else "white"
                ),
                markeredgecolor=str(style["colour"]),
                linestyle=str(style["linestyle"]),
                markersize=6.0,
                color=str(style["colour"]),
                label=GROUP_LABELS[group],
            )
            axis.fill_between(
                np.asarray(x)[valid],
                np.asarray(low)[valid],
                np.asarray(high)[valid],
                color=str(style["colour"]),
                alpha=0.10,
            )
            for xpos, n, ln in zip(
                np.asarray(x)[valid],
                np.asarray(subset["eligible_n"])[valid],
                np.asarray(subset["length_supported_n"])[valid],
                strict=True,
            ):
                denominator_by_bin.setdefault(float(xpos), []).append(
                    (int(n), int(ln))
                )
        lower, upper = axis.get_ylim()
        axis.set_ylim(lower, upper + 0.22 * max(upper - lower, 1e-9))
        for xpos, denominators in sorted(denominator_by_bin.items()):
            eligible = [row[0] for row in denominators]
            supported = [row[1] for row in denominators]
            eligible_range = (
                str(min(eligible))
                if min(eligible) == max(eligible)
                else f"{min(eligible)}–{max(eligible)}"
            )
            if metric == "identity_abundance":
                annotation = eligible_range
            else:
                support_range = (
                    str(min(supported))
                    if min(supported) == max(supported)
                    else f"{min(supported)}–{max(supported)}"
                )
                annotation = f"{eligible_range}\n{support_range}"
            artist = axis.annotate(
                annotation,
                xy=(xpos, 0.985),
                xycoords=axis.get_xaxis_transform(),
                fontsize=6.0,
                ha="center",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.15,
                },
            )
            artist.set_gid("profile-denominator")
        header = axis.annotate(
            "n" if metric == "identity_abundance" else "n\nL",
            # Keep the schema key outside the first-bin numeric column.  The
            # right-aligned offset gives all three profile panels the same
            # compact n/L row guide without colliding with the first range.
            xy=(-0.035, 0.985),
            xycoords=axis.transAxes,
            fontsize=6.0,
            fontweight="bold",
            ha="right",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.15,
            },
        )
        header.set_gid("profile-denominator-header")
        axis.set_xlim(0, 5)
        axis.set_xticks(
            np.arange(0.5, 5.0, 1.0),
            ("0–1", "1–2", "2–3", "3–4", "4–5"),
            fontsize=6.2,
        )
        axis.tick_params(axis="y", labelsize=6.2)
        axis.set_xlabel("Distance from distal point (mm)", fontsize=6.4)
        axis.set_ylabel(ylabel, fontsize=6.4)
        _clean_axis(axis)
        panel_label(axis, letter, x=-0.14, y=1.10)
    figure.legend(
        handles=tuple(
            Line2D(
                [0],
                [0],
                marker=str(GROUP_FACTOR_STYLES[group]["marker"]),
                markerfacecolor=(
                    str(GROUP_FACTOR_STYLES[group]["colour"])
                    if bool(GROUP_FACTOR_STYLES[group]["filled"])
                    else "white"
                ),
                markeredgecolor=str(GROUP_FACTOR_STYLES[group]["colour"]),
                color=str(GROUP_FACTOR_STYLES[group]["colour"]),
                linestyle=str(GROUP_FACTOR_STYLES[group]["linestyle"]),
                label=(
                    f"{'EV' if group.startswith('RHD6_EV') else 'OE'} "
                    f"{'22°C' if group.endswith('22C') else '30°C'}"
                ),
            )
            for group in GROUP_ORDER
        ),
        frameon=False,
        fontsize=6.0,
        ncol=4,
        columnspacing=1.1,
        handletextpad=0.45,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
    )
    _watermark(figure, provisional)
    _figure5_submission_readability_qa(
        figure,
        raw_axes=raw_axes,
        effect_axes=effect_axes,
        profile_axes=(abundance_axis, support_axis, length_axis),
        provisional=provisional,
    )
    return figure


def _supplementary_placeholder(stem: str, reason: str) -> plt.Figure:
    configure_publication_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.0, 4.3))
    axis.set_axis_off()
    axis.add_patch(
        Rectangle(
            (0.03, 0.08),
            0.94,
            0.84,
            transform=axis.transAxes,
            facecolor="#F3F4F6",
            edgecolor=PALETTE["grey"],
            linewidth=1.2,
        )
    )
    axis.text(
        0.5,
        0.61,
        stem.replace("_", " "),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.43,
        "FINAL EVIDENCE PENDING\nNo quantitative value has been substituted",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=PALETTE["plum"],
    )
    axis.text(
        0.5,
        0.20,
        str(reason)[:240],
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=6,
        color=PALETTE["grey"],
        wrap=True,
    )
    _watermark(figure, True)
    return figure


def _guarded_supplementary(
    *, stem: str, final: bool, builder: Any
) -> plt.Figure:
    try:
        return builder()
    except FigureSuiteError as error:
        if final:
            raise
        return _supplementary_placeholder(stem, str(error))


def _sample_ccc(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    _require(
        observed.size == predicted.size,
        "CCC requires paired source units",
    )
    if observed.size < 2:
        return None
    covariance = float(np.cov(observed, predicted, ddof=1)[0, 1])
    denominator = float(
        np.var(observed, ddof=1)
        + np.var(predicted, ddof=1)
        + (np.mean(observed) - np.mean(predicted)) ** 2
    )
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    value = float(2.0 * covariance / denominator)
    return value if math.isfinite(value) else None


def _supplementary_s1(
    resources: Mapping[str, Path], receipts: Mapping[str, Mapping[str, Any]], provisional: bool
) -> plt.Figure:
    geometry = _read_object(resources["figure1_geometry"], "S1 geometry")
    stageb = receipts["stageb"]
    _require(stageb.get("status") == "completed", "S1 Stage-B receipt is incomplete")
    metadata = stageb.get("detection_model_metadata")
    _require(isinstance(metadata, Mapping), "S1 detection model metadata missing")
    scale = geometry.get("scale_bar")
    _require(isinstance(scale, Mapping), "S1 physical scale is missing")
    pixels = _finite_number(scale.get("pixels"), "S1 scale pixels")
    micrometres = _finite_number(scale.get("micrometres"), "S1 scale micrometres")
    _require(pixels > 0 and micrometres > 0, "S1 physical scale is invalid")
    um_per_px = micrometres / pixels
    with Image.open(resources["figure1_image"]) as opened:
        source_image = np.asarray(opened)
    source_gray = to_gray(source_image)
    grayscale, _scale_factor = resample_to_physical_scale(
        source_gray,
        um_per_px,
        HAIR_WORKING_UM_PER_PX,
    )
    channels = make_input_channels(
        grayscale,
        HAIR_WORKING_UM_PER_PX,
        n_channels=3,
    )
    _require(channels.shape[0] == 3, "S1 production preprocessing did not return three channels")
    intensity, local, ridge = channels
    _require(np.isfinite(channels).all(), "S1 production preprocessing returned nonfinite values")
    configure_publication_style()
    figure, axes = plt.subplots(2, 4, figsize=(7.0, 4.9))
    image_panels = (
        (grayscale, f"Working grayscale\n({HAIR_WORKING_UM_PER_PX:g} µm px⁻¹)", "gray"),
        (intensity, "Robust intensity\nproduction channel", "gray"),
        (local, "40-µm local contrast\nproduction channel", "coolwarm"),
        (ridge, "2.5/7-µm dark ridge\nproduction channel", "magma"),
    )
    for axis, (array, title, cmap) in zip(axes[0], image_panels, strict=True):
        axis.imshow(array, cmap=cmap)
        axis.set_title(title, fontsize=7)
        axis.set_axis_off()
    architecture = axes[1, 0]
    architecture.set_axis_off()
    architecture.set_title("Shared encoder–decoder", fontsize=7)
    for index, (x, width, label, colour) in enumerate(
        (
            (0.04, 0.20, "3-channel\ninput", PALETTE["pale_teal"]),
            (0.31, 0.25, "ResNet34\nencoder", "#DBEAFE"),
            (0.63, 0.30, "U-Net-type\ndecoder", "#F3E8FF"),
        )
    ):
        architecture.add_patch(Rectangle((x, 0.35), width, 0.32, facecolor=colour, edgecolor=PALETTE["ink"]))
        architecture.text(x + width / 2, 0.51, label, ha="center", va="center", fontsize=6)
        if index < 2:
            architecture.annotate("", (x + width + 0.06, 0.51), (x + width, 0.51), arrowprops={"arrowstyle": "->"})
    architecture.set_xlim(0, 1)
    architecture.set_ylim(0, 1)
    head_names = (
        "base heatmap",
        "base offset",
        "base direction",
        "base length",
        "tip heatmap",
        "tip offset",
        "line support",
        "local flow",
        "root support",
    )
    head_axis = axes[1, 1]
    head_axis.set_axis_off()
    head_axis.set_title("Nine tensor heads", fontsize=7)
    for index, name in enumerate(head_names):
        row, column = divmod(index, 3)
        head_axis.text(0.02 + column * 0.33, 0.84 - row * 0.30, name, fontsize=5.0, bbox={"boxstyle": "round,pad=0.14", "fc": "#F9FAFB", "ec": PALETTE["grey"]})
    target_axis = axes[1, 2]
    target_axis.set_axis_off()
    target_axis.set_title("Vector-derived targets", fontsize=7)
    target_axis.plot([0.18, 0.80], [0.72, 0.30], color=PALETTE["teal"], linewidth=2)
    target_axis.scatter([0.18, 0.80], [0.72, 0.30], c=[PALETTE["gold"], PALETTE["plum"]], s=30)
    target_axis.annotate("direction", (0.58, 0.45), (0.34, 0.66), arrowprops={"arrowstyle": "->"}, fontsize=6)
    target_axis.text(0.08, 0.08, "single centreline → point / line / direction fields", fontsize=5.5)
    target_axis.set_xlim(0, 1)
    target_axis.set_ylim(0, 1)
    identity_axis = axes[1, 3]
    identity_axis.set_axis_off()
    identity_axis.set_title("Sealed expert identity", fontsize=7)
    identity_axis.text(
        0.02,
        0.90,
        f"Expert: {metadata.get('expert_id', 'sealed')}\n"
        f"Members: {metadata.get('ensemble_members', 5)}\n"
        "Input scale: physically normalized\n"
        "Output: working-grid coordinates\n"
        f"Checkpoint set: {str(stageb.get('summary_identity_sha256', 'sealed'))[:12]}…",
        va="top",
        fontsize=6,
        linespacing=1.5,
    )
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.12, y=1.04)
    _watermark(figure, provisional)
    return figure


def _supplementary_s2(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    per_image = _read_table(
        resources["development_per_image"],
        "S2 development per image",
        ("source_unit", "comparator", "gt_count", "predicted_count"),
    )
    tolerance = _read_table(
        resources["development_tolerance"],
        "S2 tolerance",
        ("comparator", "tolerance_um", "precision", "recall", "f1"),
    )
    threshold = _read_table(
        resources["development_threshold"],
        "S2 threshold",
        ("threshold", "f1_20um", "count_mae", "selected"),
    )
    strata = _read_table(
        resources["development_strata"],
        "S2 strata",
        ("dimension", "stratum", "f1_20um", "ci_low", "ci_high", "n_images"),
    )
    flow = _read_table(resources["cohort_flow"], "S2 cohort flow", ("node_id", "count"))
    configure_publication_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axis = axes[0, 0]
    flow_index = flow.set_index("node_id")["count"]
    _require({"train399", "qcdevelopment44"}.issubset(flow_index.index), "S2 split counts missing")
    values = [int(flow_index["train399"]), int(flow_index["qcdevelopment44"])]
    axis.bar([0, 1], values, color=[PALETTE["teal"], PALETTE["plum"]])
    axis.set_xticks([0, 1], ["Train", "QC-development"])
    axis.set_ylabel("Source images")
    axis.set_title("Family-isolated 399/44 split", loc="left")
    for x, value in enumerate(values):
        axis.text(x, value + 7, str(value), ha="center", fontsize=7)
    _clean_axis(axis)

    axis = axes[0, 1]
    ordered_threshold = threshold.sort_values("threshold")
    axis.plot(ordered_threshold["threshold"], ordered_threshold["f1_20um"], marker="o", color=PALETTE["teal"], label="Biological-presence F1@20 µm")
    selected = ordered_threshold[_bool_series(ordered_threshold["selected"])]
    _require(len(selected) == 1, "S2 selected threshold is not unique")
    axis.axvline(float(selected.iloc[0]["threshold"]), color=PALETTE["plum"], linestyle="--", label="Selected")
    twin = axis.twinx()
    twin.plot(ordered_threshold["threshold"], ordered_threshold["count_mae"], marker="s", color=PALETTE["gold"], label="Count MAE")
    axis.set_xlabel("Score threshold")
    axis.set_ylabel("F1")
    twin.set_ylabel("Count MAE")
    axis.set_title("Locked operating point", loc="left")
    _clean_axis(axis)

    axis = axes[0, 2]
    for comparator, colour in zip(COMPARATORS, (PALETTE["teal"], PALETTE["grey"]), strict=True):
        rows = tolerance[tolerance["comparator"] == comparator].sort_values("tolerance_um")
        _require(len(rows) == 3, f"S2 {comparator} tolerance family incomplete")
        axis.plot(rows["tolerance_um"], rows["f1"], marker="o", color=colour, label=COMPARATOR_LABELS[comparator].replace("\n", " "))
    axis.set_xticks([5, 10, 20])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Tolerance (µm)")
    axis.set_ylabel("Biological-presence F1")
    axis.set_title("Localization sensitivity", loc="left")
    axis.legend(frameon=False, fontsize=5.2)
    _clean_axis(axis)

    axis = axes[1, 0]
    stageb_rows = per_image[per_image["comparator"] == COMPARATORS[0]]
    _require(stageb_rows["source_unit"].nunique() == 44, "S2 Stage-B per-image scope is not QC44")
    axis.scatter(stageb_rows["gt_count"], stageb_rows["predicted_count"], s=10, alpha=0.65, color=PALETTE["teal"])
    maximum = float(stageb_rows[["gt_count", "predicted_count"]].max().max())
    axis.plot([0, maximum], [0, maximum], linestyle="--", color=PALETTE["ink"])
    axis.set_xlabel("Annotated count")
    axis.set_ylabel("Predicted count")
    axis.set_title("Per-image count agreement", loc="left")
    _clean_axis(axis)

    for axis, dimension, title in (
        (axes[1, 1], "annotation", "Annotation-source strata"),
        (axes[1, 2], "density", "Hair-count strata"),
    ):
        rows = strata[strata["dimension"].astype(str) == dimension].copy()
        if rows.empty and dimension == "density":
            rows = strata[strata["dimension"].astype(str).isin({"hair_count", "density"})].copy()
        _require(not rows.empty, f"S2 {dimension} strata missing")
        y = np.arange(len(rows))
        values = pd.to_numeric(rows["f1_20um"]).to_numpy(float)
        low = values - pd.to_numeric(rows["ci_low"]).to_numpy(float)
        high = pd.to_numeric(rows["ci_high"]).to_numpy(float) - values
        axis.errorbar(values, y, xerr=[low, high], fmt="o", color=PALETTE["plum"], capsize=2)
        axis.set_yticks(y, [f"{row['stratum']} (n={int(row['n_images'])})" for row in rows.to_dict("records")], fontsize=5.5)
        axis.set_xlim(0, 1)
        axis.set_xlabel("F1@20 µm")
        axis.set_title(title, loc="left")
        _clean_axis(axis, x=True, y=False)
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.06)
    _watermark(figure, provisional)
    return figure


def _supplementary_s3(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    metrics = _read_table(resources["assurance_metrics"], "S3 metrics", ("domain", "metric_key", "label", "value", "ci_low", "ci_high", "unit", "n"))
    pairs = _read_table(resources["assurance_pairs"], "S3 pairs", ("pair_type", "observed", "predicted", "endpoint_error_um", "trajectory_continuity"))
    support = _read_table(resources["assurance_support"], "S3 support", ("condition_code", "support_fraction", "supported_hairs", "identity_hairs", "source_units"))
    tolerance = _read_table(resources["development_tolerance"], "S3 tolerance", ("comparator", "tolerance_um", "precision", "recall", "f1"))
    configure_publication_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axis = axes[0, 0]
    rows = tolerance[tolerance["comparator"] == COMPARATORS[0]].sort_values("tolerance_um")
    _require(len(rows) == 3, "S3 tolerant-presence rows incomplete")
    x = np.arange(3)
    width = 0.24
    for offset, key, colour in ((-width, "precision", PALETTE["navy"]), (0, "recall", PALETTE["teal"]), (width, "f1", PALETTE["plum"])):
        axis.bar(x + offset, rows[key], width=width, label=key.title(), color=colour)
    axis.set_xticks(x, [f"{int(value)} µm" for value in rows["tolerance_um"]])
    axis.set_ylim(0, 1)
    axis.set_title("Tolerant biological identity", loc="left")
    axis.legend(frameon=False, fontsize=5.5)
    _clean_axis(axis)

    length = pairs[pairs["pair_type"] == "conditional_length"].copy()
    _require(len(length) >= 2, "S3 conditional-length pairs missing")
    observed = pd.to_numeric(length["observed"]).to_numpy(float)
    predicted = pd.to_numeric(length["predicted"]).to_numpy(float)
    axis = axes[0, 1]
    maximum = max(float(np.max(observed)), float(np.max(predicted)))
    axis.scatter(observed, predicted, s=10, alpha=0.6, color=PALETTE["plum"])
    axis.plot([0, maximum], [0, maximum], "--", color=PALETTE["ink"])
    axis.set_xlabel("Reference length (µm)")
    axis.set_ylabel("PHAxis length (µm)")
    axis.set_title(f"Conditional length; CCC={_sample_ccc(observed, predicted):.2f}", loc="left")
    _clean_axis(axis)

    axis = axes[0, 2]
    endpoint = pd.to_numeric(length["endpoint_error_um"], errors="coerce").dropna()
    _require(len(endpoint) > 0, "S3 endpoint-error evidence missing")
    axis.hist(endpoint, bins=min(12, max(4, len(endpoint) // 2)), color=PALETTE["gold"], alpha=0.8)
    axis.axvline(float(endpoint.median()), color=PALETTE["ink"], linestyle="--")
    axis.set_xlabel("Distal endpoint error (µm)")
    axis.set_ylabel("Matched hairs")
    axis.set_title("Endpoint-complete geometry", loc="left")
    _clean_axis(axis)

    axis = axes[1, 0]
    continuity = pd.to_numeric(length["trajectory_continuity"], errors="coerce").dropna()
    _require(len(continuity) > 0, "S3 trajectory-continuity evidence missing")
    axis.hist(continuity, bins=np.linspace(0, 1, 11), color=PALETTE["teal"], alpha=0.8)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Hair-curve continuity")
    axis.set_ylabel("Matched hairs")
    axis.set_title("Hair-curve trajectory continuity", loc="left")
    _clean_axis(axis)

    axis = axes[1, 1]
    ordered = support.set_index("condition_code").reindex(GROUP_ORDER).dropna(how="all").reset_index()
    _require(len(ordered) == 4, "S3 four-condition support incomplete")
    axis.bar(np.arange(4), ordered["support_fraction"], color=[GROUP_COLOURS[code] for code in ordered["condition_code"]])
    axis.set_xticks(np.arange(4), [GROUP_LABELS[code] for code in ordered["condition_code"]], rotation=20)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Endpoint-complete fraction")
    axis.set_title("Group-specific length support", loc="left")
    _clean_axis(axis)

    formal_attachment_keys = [
        "hair_attachment_qualified_precision_20um",
        "hair_attachment_qualified_recall_20um",
        "hair_attachment_qualified_f1_20um",
        "hair_attachment_error_median_um",
        "hair_attachment_error_p95_um",
    ]
    formal_attachment = metrics[
        metrics["metric_key"].astype(str).isin(formal_attachment_keys)
    ].copy()
    formal_attachment["metric_key"] = pd.Categorical(
        formal_attachment["metric_key"],
        categories=formal_attachment_keys,
        ordered=True,
    )
    formal_attachment = formal_attachment.sort_values("metric_key")
    _require(
        len(formal_attachment) == len(formal_attachment_keys),
        "S3 formal attachment metric family is incomplete",
    )
    _metric_card(axes[1, 2], formal_attachment, "Formal attachment assurance")
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.06)
    _watermark(figure, provisional)
    return figure


def _supplementary_s4(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    pairs = _read_table(resources["assurance_pairs"], "S4 pairs", ("pair_type", "source_unit", "observed", "predicted", "unit"))
    root = pairs[pairs["pair_type"].astype(str) == "root_trait"].copy()
    _require(len(root) >= 38, "S4 root-trait source-unit pairs are absent")
    _require({"trait_id", "trait_key", "trait_family"}.issubset(root.columns), "S4 root-trait identity columns missing")
    _require(root["trait_id"].nunique() == 19, "S4 does not cover all 19 root descriptors")
    _finite(root, ("observed", "predicted"), "S4 root-trait pairs")
    summary_rows = []
    for trait_id, rows in root.groupby("trait_id", sort=False):
        _require(
            rows["trait_key"].astype(str).nunique() == 1
            and rows["trait_family"].astype(str).nunique() == 1
            and rows["unit"].astype(str).nunique() == 1,
            f"S4 {trait_id}: trait identity/family/unit drift",
        )
        observed = pd.to_numeric(rows["observed"]).to_numpy(float)
        predicted = pd.to_numeric(rows["predicted"]).to_numpy(float)
        summary_rows.append({
            "trait_id": str(trait_id),
            "trait_key": str(rows.iloc[0]["trait_key"]),
            "trait_family": str(rows.iloc[0]["trait_family"]),
            "unit": str(rows.iloc[0]["unit"]),
            "n": len(rows),
            "mae": float(np.mean(np.abs(predicted - observed))),
            "bias": float(np.mean(predicted - observed)),
            "ccc": _sample_ccc(observed, predicted),
        })
    summary = pd.DataFrame(summary_rows).sort_values("trait_id")
    _require(
        set(summary["trait_id"].astype(str)) == set(ROOT_TRAIT_SHORT_LABELS),
        "S4 canonical root-trait identifiers changed",
    )
    configure_publication_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 5.3))
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.87, wspace=0.34)
    _require(
        set(summary["trait_family"].astype(str)) == set(ROOT_TRAIT_FAMILY_ORDER),
        "S4 six prespecified root-assurance subgroups changed",
    )
    family_groups = np.asarray(ROOT_TRAIT_FAMILY_ORDER, dtype=object).reshape(3, 2)
    for axis, families in zip(axes, family_groups, strict=True):
        rows = summary[summary["trait_family"].isin(list(families))].copy()
        y = np.arange(len(rows))
        ccc = pd.to_numeric(rows["ccc"], errors="coerce").to_numpy(float)
        estimable = np.isfinite(ccc)
        point_colours = [
            ROOT_TRAIT_FAMILY_COLOURS[str(value)]
            for value in rows["trait_family"]
        ]
        if estimable.any():
            axis.scatter(
                ccc[estimable],
                y[estimable],
                c=np.asarray(point_colours)[estimable],
                s=26,
                edgecolor="white",
                linewidth=0.4,
            )
        for position in y[~estimable]:
            axis.text(
                0.015,
                position,
                "NA",
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=6,
                fontweight="bold",
                color=PALETTE["grey"],
            )
        axis.axvline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=0.8)
        axis.set_yticks(
            y,
            [
                f"{row.trait_id} {ROOT_TRAIT_SHORT_LABELS[str(row.trait_id)]}\n"
                f"n={row.n}; MAE={row.mae:.2g} {row.unit}"
                + ("; CCC=NA" if not math.isfinite(float(row.ccc)) else "")
                for row in rows.itertuples()
            ],
            fontsize=4.8,
        )
        axis.set_xlim(-1.05, 1.05)
        axis.set_xticks((-1.0, 0.0, 1.0))
        axis.set_xlabel("Trait-wise CCC")
        axis.set_title(
            " / ".join(ROOT_TRAIT_FAMILY_LABELS[str(value)] for value in families),
            loc="left",
            fontsize=7,
        )
        axis.invert_yaxis()
        _clean_axis(axis, x=True, y=False)
    figure.suptitle("Nineteen primary-root descriptors: source-unit agreement", fontsize=9, fontweight="bold")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=ROOT_TRAIT_FAMILY_COLOURS[family],
                markeredgecolor="white",
                label=ROOT_TRAIT_FAMILY_LABELS[family],
            )
            for family in ROOT_TRAIT_FAMILY_ORDER
        ],
        frameon=False,
        fontsize=5.2,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
    )
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index), x=-0.22, y=1.04)
    _watermark(figure, provisional)
    return figure


def _supplementary_s5(
    resources: Mapping[str, Path], receipts: Mapping[str, Mapping[str, Any]], provisional: bool
) -> plt.Figure:
    root = receipts["root_exact283"]
    stageb = receipts["stageb"]
    layers = root.get("layers")
    _require(isinstance(layers, Mapping) and len(layers) == 3, "S5 root-provider layers missing")
    layer_rows = []
    for name, record in layers.items():
        _require(isinstance(record, Mapping), f"S5 root layer {name} malformed")
        exact = int(record.get("exact", -1))
        expected = int(record.get("expected", -2))
        _require(exact == expected == 283 and record.get("gate_pass") is True, f"S5 root layer {name} is not exact283")
        layer_rows.append((str(name), exact))
    metrics = _read_table(resources["assurance_metrics"], "S5 metrics", ("domain", "metric_key", "label", "value", "unit"))
    shared = stageb.get("shared_input_acceleration")
    _require(isinstance(shared, Mapping), "S5 shared-input runtime audit missing")
    path_counts = shared.get("runtime_path_counts")
    _require(isinstance(path_counts, Mapping) and bool(path_counts), "S5 runtime-path counts missing")
    configure_publication_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axis = axes[0, 0]
    axis.bar(np.arange(3), [row[1] for row in layer_rows], color=[PALETTE["teal"], PALETTE["navy"], PALETTE["plum"]])
    axis.set_xticks(np.arange(3), [row[0].replace("_", "\n") for row in layer_rows], fontsize=5)
    axis.set_ylim(0, 310)
    axis.set_ylabel("Exact source images")
    axis.set_title("Portable provider equivalence", loc="left")
    _clean_axis(axis)

    axis = axes[0, 1]
    axis.set_axis_off()
    formal_keys = [
        "root_continuity_maximum_single_component_coverage_mean",
        "root_continuity_maximum_single_component_coverage_median",
        "root_continuity_best_component_gap_median_um",
        "root_continuity_break_free_rate",
        "root_continuity_visible_axis_extent_mae_um",
        "hair_attachment_qualified_precision_20um",
        "hair_attachment_qualified_recall_20um",
        "hair_attachment_qualified_f1_20um",
        "hair_attachment_error_median_um",
        "hair_attachment_error_p95_um",
    ]
    selected = metrics[metrics["metric_key"].astype(str).isin(formal_keys)].copy()
    selected["metric_key"] = pd.Categorical(
        selected["metric_key"], categories=formal_keys, ordered=True
    )
    selected = selected.sort_values("metric_key")
    _require(
        len(selected) == len(formal_keys),
        "S5 same-component/formal-attachment metric family is incomplete",
    )
    short_labels = {
        "root_continuity_maximum_single_component_coverage_mean": "One-component cover · mean",
        "root_continuity_maximum_single_component_coverage_median": "One-component cover · median",
        "root_continuity_best_component_gap_median_um": "Best-component gap · median",
        "root_continuity_break_free_rate": "Break-free rate",
        "root_continuity_visible_axis_extent_mae_um": "Visible-axis extent MAE",
        "hair_attachment_qualified_precision_20um": "Attachment P@20 µm",
        "hair_attachment_qualified_recall_20um": "Attachment R@20 µm",
        "hair_attachment_qualified_f1_20um": "Attachment F1@20 µm",
        "hair_attachment_error_median_um": "Matched error · median",
        "hair_attachment_error_p95_um": "Matched error · P95",
    }
    y = 0.93
    for row in selected.to_dict("records"):
        axis.text(
            0.04,
            y,
            f"{short_labels[str(row['metric_key'])]}  {float(row['value']):.4g} {row['unit']}",
            fontsize=4.9,
            va="top",
        )
        y -= 0.085
    axis.set_title("Same-component + formal attachment", loc="left")

    axis = axes[0, 2]
    axis.set_axis_off()
    axis.set_title("Exact tiled coverage contract", loc="left")
    axis.add_patch(Rectangle((0.08, 0.12), 0.84, 0.74, facecolor="#F9FAFB", edgecolor=PALETTE["ink"]))
    for x, y, colour in ((0.10, 0.38, "#DBEAFE"), (0.36, 0.38, "#DCFCE7"), (0.10, 0.14, "#F3E8FF"), (0.36, 0.14, "#FEF3C7")):
        axis.add_patch(Rectangle((x, y), 0.52, 0.46, facecolor=colour, edgecolor=PALETTE["grey"], alpha=0.65))
    axis.text(
        0.5,
        0.04,
        "Physically normalized overlapping tiles\nboundary-aligned origins; one deterministic weighted stitch",
        ha="center",
        fontsize=5.8,
    )

    axis = axes[1, 0]
    names = list(path_counts)
    values = [int(path_counts[name]) for name in names]
    axis.bar(np.arange(len(names)), values, color=PALETTE["teal"])
    axis.set_xticks(np.arange(len(names)), [name.replace("_", "\n") for name in names], fontsize=5)
    axis.set_ylabel("Executed images")
    axis.set_title("Sealed ensemble execution paths", loc="left")
    _clean_axis(axis)

    axis = axes[1, 1]
    axis.set_axis_off()
    axis.set_title("Numerical-equivalence boundary", loc="left")
    axis.text(
        0.03,
        0.88,
        "Reference order\nmember-sequential × original/flip\n\n"
        "Accelerated order\nshared transferred tile batch × canonical members\n\n"
        "Release rule\noptimization is enabled only under its sealed equivalence gate",
        va="top",
        fontsize=6.2,
        linespacing=1.4,
    )

    axis = axes[1, 2]
    axis.set_axis_off()
    axis.set_title("Immutable runtime authorities", loc="left")
    axis.text(
        0.03,
        0.90,
        f"Root audit\n{str(root.get('audit_identity_sha256', 'sealed'))[:18]}…\n\n"
        f"Stage-B summary\n{str(stageb.get('summary_identity_sha256', 'sealed'))[:18]}…\n\n"
        f"Checkpoint count\n{len(stageb.get('checkpoint_sha256', [])) or 5}",
        va="top",
        fontsize=6.3,
    )
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.06)
    _watermark(figure, provisional)
    return figure


def _supplementary_s6(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    selection = _read_table(resources["overlay_selection"], "S6 overlay selection", ("case_id", "case_role", "source_path", "source_sha256", "overlay_path", "overlay_sha256"))
    records = _verify_overlay_rows(selection, resources["overlay_selection"].parent, not provisional)
    _require(len(records) == 5, "S6 sealed gallery is not five unique review roles")
    configure_publication_style()
    figure, axes = plt.subplots(5, 2, figsize=(7.0, 9.0))
    role_titles = {
        "representative": "Representative routine",
        "low_contrast": "Low contrast",
        "curved_dense": "Curved + dense",
        "continuity": "Continuity-completed",
        "fail_closed": "Review-only fail-closed",
    }
    for row_index, record in enumerate(records):
        for column, prefix in enumerate(("source", "overlay")):
            axis = axes[row_index, column]
            array = _linear_display(
                Path(record[f"{prefix}_resolved"]),
                float(record["display_lower"]),
                float(record["display_upper"]),
            )
            axis.imshow(array)
            axis.set_axis_off()
            _scale_bar(
                axis,
                image_shape=array.shape,
                pixels=float(record["scale_bar_px"]),
                micrometres=float(record["scale_bar_um"]),
            )
            axis.set_title(
                f"{role_titles[str(record['case_role'])]} — {prefix}",
                fontsize=6.4,
                loc="left",
            )
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.05, y=1.04)
    _watermark(figure, provisional)
    return figure


def _supplementary_s7(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    points = _read_table(resources["phenotype_points"], "S7 points", ("source_unit", "cohort", "condition_code", "endpoint_key", "value", "unit"))
    effects = _read_table(resources["phenotype_effects"], "S7 effects", ("cohort", "endpoint_key", "effect_key", "estimate", "ci_low", "ci_high", "endpoint_n"))
    support = _read_table(resources["assurance_support"], "S7 support", ("condition_code", "support_fraction", "supported_hairs", "identity_hairs", "source_units"))
    profiles = _read_table(resources["axial_profiles"], "S7 profiles", ("cohort", "condition_code", "bin_start_mm", "metric_key", "estimate", "eligible_n", "length_supported_n"))
    effects = _ordered_fixed_effects(effects, "S7 effects")
    clean = effects[effects["cohort"] == "primary_clean261"].copy()
    full = effects[effects["cohort"] == "sensitivity_full283"].copy()
    keys = ["endpoint_key", "effect_key"]
    joined = clean.merge(
        full,
        on=keys,
        suffixes=("_clean", "_full"),
        validate="one_to_one",
        sort=False,
    )
    _require(len(joined) == 15, "S7 clean/full fixed effect family does not close")
    expected_keys = [
        (endpoint, effect)
        for endpoint in PRIMARY_ENDPOINTS
        for effect in EFFECT_ORDER
    ]
    _require(
        list(zip(joined["endpoint_key"], joined["effect_key"], strict=True))
        == expected_keys,
        "S7 effect rows are not in the locked endpoint/effect order",
    )
    configure_publication_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 5.2))
    figure.subplots_adjust(wspace=0.42, hspace=0.52, bottom=0.13)
    axis = axes[0, 0]
    effect_colours = {
        "OE_vs_EV": PALETTE["teal"],
        "30C_vs_22C": PALETTE["orange"],
        "interaction": PALETTE["plum"],
    }
    axis.scatter(
        joined["estimate_clean"],
        joined["estimate_full"],
        c=[effect_colours[str(value)] for value in joined["effect_key"]],
        s=22,
    )
    lower = min(float(joined["estimate_clean"].min()), float(joined["estimate_full"].min()))
    upper = max(float(joined["estimate_clean"].max()), float(joined["estimate_full"].max()))
    axis.plot([lower, upper], [lower, upper], "--", color=PALETTE["ink"])
    axis.set_xlabel("Clean-cohort D15 ratio")
    axis.set_ylabel("Full-cohort D15 sensitivity ratio")
    axis.set_title("Effect sensitivity", loc="left")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=effect_colours[effect],
                markeredgecolor="none",
                label=label,
            )
            for effect, label in (
                ("OE_vs_EV", "Construct label"),
                ("30C_vs_22C", "Temperature"),
                ("interaction", "Interaction"),
            )
        ],
        frameon=False,
        fontsize=4.6,
        loc="upper left",
    )
    _clean_axis(axis)

    axis = axes[0, 1]
    y = np.arange(len(joined))
    direction = np.sign(joined["estimate_clean"] - 1) == np.sign(joined["estimate_full"] - 1)
    axis.barh(y, joined["estimate_full"] - joined["estimate_clean"], color=np.where(direction, PALETTE["teal"], PALETTE["plum"]))
    axis.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    endpoint_short = {
        PRIMARY_ENDPOINTS[0]: "Count",
        PRIMARY_ENDPOINTS[1]: "Length",
        PRIMARY_ENDPOINTS[2]: "First hair",
        PRIMARY_ENDPOINTS[3]: "Root width",
        PRIMARY_ENDPOINTS[4]: "Axis extent",
    }
    effect_short = {
        "OE_vs_EV": "C",
        "30C_vs_22C": "T",
        "interaction": "C×T",
    }
    axis.set_yticks(
        y,
        [
            f"{endpoint_short[str(row.endpoint_key)]} · "
            f"{effect_short[str(row.effect_key)]}"
            for row in joined.itertuples()
        ],
        fontsize=4.4,
    )
    axis.set_xlabel("Full − clean ratio")
    axis.set_title(f"Direction retained: {int(direction.sum())}/15", loc="left")
    axis.invert_yaxis()
    _clean_axis(axis, x=True, y=False)

    axis = axes[0, 2]
    endpoint = PRIMARY_ENDPOINTS[2]
    endpoint_points = points[
        (points["endpoint_key"] == endpoint)
        & (points["cohort"] == "primary_clean261")
    ]
    _require(not endpoint_points.empty, "S7 first-hair raw points missing")
    compact_group_labels = ("EV\n22 °C", "EV\n30 °C", "OE\n22 °C", "OE\n30 °C")
    for index, condition in enumerate(GROUP_ORDER):
        values = pd.to_numeric(endpoint_points[endpoint_points["condition_code"] == condition]["value"])
        axis.scatter(np.full(len(values), index) + _jitter(len(values), 800 + index), values, s=8, alpha=0.55, color=GROUP_COLOURS[condition])
    axis.set_xticks(np.arange(4), compact_group_labels)
    axis.set_ylabel("First ≥40-µm hair distance (µm)")
    axis.set_title("Observable axial deployment", loc="left")
    _clean_axis(axis)

    axis = axes[1, 0]
    _require(
        support["condition_code"].astype(str).is_unique
        and set(support["condition_code"].astype(str)) == set(GROUP_ORDER),
        "S7 support must contain one row per archived condition",
    )
    ordered = support.set_index("condition_code").reindex(GROUP_ORDER).reset_index()
    bars = axis.bar(np.arange(4), ordered["support_fraction"], color=[GROUP_COLOURS[value] for value in GROUP_ORDER])
    axis.set_xticks(np.arange(4), compact_group_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Endpoint-complete fraction")
    axis.set_title("Length observability", loc="left")
    for bar, row in zip(bars, ordered.to_dict("records"), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{int(row['supported_hairs'])}/{int(row['identity_hairs'])}\n"
            f"n={int(row['source_units'])}",
            ha="center",
            va="bottom",
            fontsize=5,
        )
    _clean_axis(axis)

    for axis, metric, title in (
        (axes[1, 1], "identity_abundance", "Distal abundance"),
        (axes[1, 2], "length_support_fraction", "Distal length support"),
    ):
        rows = profiles[
            (profiles["metric_key"] == metric)
            & (profiles["cohort"] == "primary_clean261")
        ]
        _require(not rows.empty, f"S7 profile missing: {metric}")
        for condition in GROUP_ORDER:
            selected = rows[rows["condition_code"] == condition].sort_values("bin_start_mm")
            axis.plot(selected["bin_start_mm"] + 0.5, selected["estimate"], marker="o", markersize=2, color=GROUP_COLOURS[condition], label=GROUP_LABELS[condition])
        axis.set_xlabel("Distance from distal point (mm)")
        axis.set_ylabel(
            "Hair identities per bin"
            if metric == "identity_abundance"
            else "Length-support fraction"
        )
        axis.set_title(title, loc="left")
        _clean_axis(axis)
    axes[1, 2].legend(frameon=False, fontsize=4.6)
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.06)
    _watermark(figure, provisional)
    return figure


def _supplementary_s8(resources: Mapping[str, Path], provisional: bool) -> plt.Figure:
    runtime = _read_object(resources["runtime_summary"], "S8 runtime")
    trace = _read_table(resources["runtime_per_image"], "S8 per-image trace", ("source_unit", "wall_seconds", "megapixels", "io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"))
    current_latency = runtime.get("sequential_latency_full283")
    baseline_latency = runtime.get("baseline_sequential_latency_full283")
    current_batch = runtime.get("production_batch_full283")
    baseline_batch = runtime.get("baseline_production_batch_full283")
    _require(all(isinstance(value, Mapping) for value in (current_latency, baseline_latency, current_batch, baseline_batch)), "S8 two-system runtime blocks missing")
    for label, block in (("PHAxis", current_batch), ("legacy workflow", baseline_batch)):
        _require(
            all(block.get(key) is not None for key in ("batch_wall_seconds", "images_per_min", "megapixels_per_second", "peak_vram_mib", "mean_gpu_utilization_pct")),
            f"S8 {label} memory/utilization evidence missing",
        )
    configure_publication_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axis = axes[0, 0]
    axis.hist(trace["wall_seconds"], bins=18, color=PALETTE["teal"], alpha=0.8)
    axis.axvline(float(current_latency["median_seconds_per_image"]), color=PALETTE["ink"], linestyle="--", label="Median")
    axis.axvline(float(current_latency["p95_seconds_per_image"]), color=PALETTE["plum"], linestyle=":", label="P95")
    axis.set_xlabel("Per-image wall time (s)")
    axis.set_ylabel("Images")
    axis.set_title("Direct sequential latency", loc="left")
    axis.legend(frameon=False, fontsize=5)
    _clean_axis(axis)

    axis = axes[0, 1]
    systems = ("PHAxis", "Legacy workflow")
    wall = [float(current_batch["batch_wall_seconds"]), float(baseline_batch["batch_wall_seconds"])]
    axis.bar(systems, wall, color=[PALETTE["teal"], PALETTE["grey"]])
    axis.set_ylabel("Batch wall time (s), 283 images")
    axis.set_title("Production batch", loc="left")
    _clean_axis(axis)

    axis = axes[0, 2]
    x = np.arange(2)
    axis.bar(x - 0.16, [current_batch["images_per_min"], baseline_batch["images_per_min"]], 0.32, label="images min⁻¹", color=PALETTE["navy"])
    twin = axis.twinx()
    twin.bar(x + 0.16, [current_batch["megapixels_per_second"], baseline_batch["megapixels_per_second"]], 0.32, label="MP s⁻¹", color=PALETTE["gold"])
    axis.set_xticks(x, systems)
    axis.set_ylabel("Images min⁻¹")
    twin.set_ylabel("MP s⁻¹")
    axis.set_title("Observed throughput", loc="left")
    _clean_axis(axis)

    axis = axes[1, 0]
    stage_values = trace[["io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"]].median()
    axis.bar(np.arange(4), stage_values, color=[PALETTE["grey"], PALETTE["gold"], PALETTE["teal"], PALETTE["plum"]])
    axis.set_xticks(np.arange(4), ["I/O", "Pre", "Inference", "Post"], rotation=18)
    axis.set_ylabel("Median seconds per image")
    axis.set_title("Non-overlapping stage decomposition", loc="left")
    _clean_axis(axis)

    axis = axes[1, 1]
    axis.bar(x - 0.17, [current_batch["peak_vram_mib"], baseline_batch["peak_vram_mib"]], 0.34, color=PALETTE["plum"], label="Peak VRAM MiB")
    twin = axis.twinx()
    twin.bar(x + 0.17, [current_batch["mean_gpu_utilization_pct"], baseline_batch["mean_gpu_utilization_pct"]], 0.34, color=PALETTE["teal"], label="Mean utilization %")
    axis.set_xticks(x, systems)
    axis.set_ylabel("Peak VRAM (MiB)")
    twin.set_ylabel("GPU utilization (%)")
    axis.set_title("Measured accelerator use", loc="left")
    _clean_axis(axis)

    axis = axes[1, 2]
    axis.scatter(trace["megapixels"], trace["wall_seconds"], c=trace["inference_seconds"], cmap="viridis", s=10, alpha=0.65)
    axis.set_xlabel("Image size (MP)")
    axis.set_ylabel("Wall time (s)")
    axis.set_title("I/O-inclusive scaling", loc="left")
    _clean_axis(axis)
    for index, axis in enumerate(axes.ravel()):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.06)
    _watermark(figure, provisional)
    return figure


def _draw_wt_secondary_forest(
    axis: plt.Axes,
    *,
    contrasts: pd.DataFrame,
    meta: pd.DataFrame,
    unknown_day: bool,
) -> dict[str, int]:
    """Draw experiment estimates and only eligible same-day pooled diamonds."""

    frame = contrasts.copy()
    frame["developmental_day"] = pd.to_numeric(
        frame["developmental_day"], errors="coerce"
    )
    frame = frame[
        frame["developmental_day"].isna()
        if unknown_day
        else frame["developmental_day"].notna()
    ]
    cohort_labels = {
        "primary_clean261": "Clean",
        "sensitivity_full283": "Full sens.",
    }
    endpoint_offsets = dict(
        zip(WT_SECONDARY_ENDPOINTS, np.linspace(-0.26, 0.26, 5), strict=True)
    )
    rows: list[tuple[str, str, int | None, str | None]] = []
    blocks: list[tuple[str, int | None]] = []
    if unknown_day:
        blocks = [
            (cohort, None)
            for cohort in WT_SECONDARY_COHORT_ROLES
            if (frame["cohort"].astype(str) == cohort).any()
        ]
    else:
        blocks = sorted(
            {
                (str(row.cohort), int(row.developmental_day))
                for row in frame.itertuples()
            },
            key=lambda item: (
                list(WT_SECONDARY_COHORT_ROLES).index(item[0]), item[1]
            ),
        )
    for cohort, day in blocks:
        selected = frame[frame["cohort"].astype(str).eq(cohort)]
        if day is not None:
            selected = selected[selected["developmental_day"].eq(day)]
        experiments = sorted(selected["experiment_key"].astype(str).unique())
        for experiment in experiments:
            day_text = "day ?" if day is None else f"day {day}"
            rows.append(
                (
                    f"{cohort_labels[cohort]} · {day_text} · {experiment}",
                    cohort,
                    day,
                    experiment,
                )
            )
        if not unknown_day:
            rows.append(
                (
                    f"{cohort_labels[cohort]} · day {day} · pooled",
                    cohort,
                    day,
                    None,
                )
            )
    _require(rows, "WT forest panel contains no rows")

    plotted_limits: list[float] = [1.0]
    diamond_count = 0
    not_estimable_meta_count = 0
    experiment_point_count = 0
    for y, (_label, cohort, day, experiment) in enumerate(rows):
        if experiment is not None:
            selected = frame[
                frame["cohort"].astype(str).eq(cohort)
                & frame["experiment_key"].astype(str).eq(experiment)
            ]
            if day is not None:
                selected = selected[selected["developmental_day"].eq(day)]
            missing_codes: list[str] = []
            for endpoint in WT_SECONDARY_ENDPOINTS:
                endpoint_rows = selected[selected["endpoint"].astype(str).eq(endpoint)]
                _require(len(endpoint_rows) == 1, "WT forest experiment endpoint row changed")
                record = endpoint_rows.iloc[0]
                if str(record["analysis_status"]) != "estimated":
                    missing_codes.append(WT_ENDPOINT_CODES[endpoint])
                    continue
                estimate = float(record["estimate_30C_over_22C"])
                low = float(record["ci95_low"])
                high = float(record["ci95_high"])
                plotted_limits.extend((low, high))
                y_position = y + endpoint_offsets[endpoint]
                colour = WT_ENDPOINT_COLOURS[endpoint]
                filled = cohort == "primary_clean261"
                axis.plot([low, high], [y_position, y_position], color=colour, linewidth=0.75)
                axis.scatter(
                    [estimate],
                    [y_position],
                    s=14,
                    marker="o",
                    facecolors=colour if filled else "white",
                    edgecolors=colour,
                    linewidths=0.65,
                    zorder=3,
                )
                experiment_point_count += 1
            if missing_codes:
                axis.text(
                    0.99,
                    y,
                    "NE: " + ",".join(missing_codes),
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=4.1,
                    color=PALETTE["grey"],
                )
            continue

        selected_meta = meta[
            meta["cohort"].astype(str).eq(cohort)
            & pd.to_numeric(meta["developmental_day"], errors="coerce").eq(day)
        ]
        _require(
            set(selected_meta["endpoint"].astype(str))
            == set(WT_SECONDARY_ENDPOINTS)
            and len(selected_meta) == 5,
            "WT same-day pooled five-endpoint grid changed",
        )
        not_estimable: list[str] = []
        for endpoint in WT_SECONDARY_ENDPOINTS:
            record = selected_meta[
                selected_meta["endpoint"].astype(str).eq(endpoint)
            ].iloc[0]
            k = int(record["k_eligible_experiments"])
            if str(record["analysis_status"]) != "estimated":
                not_estimable_meta_count += 1
                qualifier = "k<3" if k < 3 else "model"
                not_estimable.append(
                    f"{WT_ENDPOINT_CODES[endpoint]}(k={k};{qualifier})"
                )
                continue
            _require(k >= 3, "WT pooled diamond would have k<3")
            estimate = float(record["estimate_30C_over_22C"])
            low = float(record["ci95_low"])
            high = float(record["ci95_high"])
            plotted_limits.extend((low, high))
            y_position = y + endpoint_offsets[endpoint]
            half_height = 0.09
            colour = WT_ENDPOINT_COLOURS[endpoint]
            axis.add_patch(
                Polygon(
                    (
                        (low, y_position),
                        (estimate, y_position - half_height),
                        (high, y_position),
                        (estimate, y_position + half_height),
                    ),
                    closed=True,
                    facecolor=(
                        colour if cohort == "primary_clean261" else "white"
                    ),
                    edgecolor=colour,
                    linewidth=0.8,
                    zorder=3,
                )
            )
            diamond_count += 1
        if not_estimable:
            axis.text(
                0.99,
                y,
                "Not estimable: " + ", ".join(not_estimable),
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=4.0,
                color=PALETTE["grey"],
            )

    positive = np.asarray([value for value in plotted_limits if value > 0], dtype=float)
    lower = float(2 ** (math.floor(math.log2(float(positive.min()))) - 0.15))
    upper = float(2 ** (math.ceil(math.log2(float(positive.max()))) + 0.15))
    if not lower < 1.0 < upper:
        lower = min(lower, 0.8)
        upper = max(upper, 1.25)
    axis.set_xscale("log", base=2)
    axis.set_xlim(lower, upper)
    axis.axvline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=0.75)
    axis.set_yticks(np.arange(len(rows)), [row[0] for row in rows], fontsize=4.2)
    axis.set_ylim(len(rows) - 0.45, -0.55)
    axis.tick_params(axis="x", labelsize=5.0)
    axis.set_xlabel("Within-experiment 30 °C / 22 °C ratio (95% interval)", fontsize=5.5)
    axis.set_title(
        (
            "Unknown developmental day — descriptive experiment contrasts only; never pooled"
            if unknown_day
            else "Developmental-day blocks — experiment contrasts and eligible same-day REML/HK"
        ),
        loc="left",
        fontsize=6.3,
    )
    _clean_axis(axis, x=True, y=False)
    return {
        "experiment_point_count": experiment_point_count,
        "pooled_diamond_count": diamond_count,
        "not_estimable_meta_rows_rendered": not_estimable_meta_count,
        "unknown_day": int(unknown_day),
    }


def _supplementary_multitrait_atlas(
    resources: Mapping[str, Path], provisional: bool
) -> plt.Figure:
    """Render all 32 descriptors, fixed D15 effects, and secondary WT blocks."""

    payload = _read_object(resources["multitrait_atlas"], "multitrait atlas")
    matrices = descriptive_heatmap_matrices(payload)
    labels = matrices["trait_labels"]
    standardized_medians = matrices["standardized_medians"]
    relative_iqrs = matrices["relative_iqrs"]
    condition_coverage = matrices["condition_coverage"]
    effect_labels_raw = matrices["effect_trait_labels"]
    effects = matrices["effect_estimates"]
    contrasts = _read_table(
        resources["wt_within_experiment_contrasts"],
        "S9 WT within-experiment contrasts",
        (
            "cohort", "endpoint", "experiment_key", "developmental_day",
            "analysis_status", "estimate_30C_over_22C", "ci95_low", "ci95_high",
        ),
    )
    meta = _read_table(
        resources["wt_within_day_meta_analysis"],
        "S9 WT same-day meta-analysis",
        (
            "cohort", "endpoint", "developmental_day",
            "k_eligible_experiments", "analysis_status",
            "estimate_30C_over_22C", "ci95_low", "ci95_high",
        ),
    )
    flow = _read_table(
        resources["wt_temperature_qc_flow"],
        "S9 WT model-QC flow",
        (
            "cohort", "experiment_key", "endpoint", "developmental_day",
            "model_status", "phenotype_outlier_filter_applied",
        ),
    )
    try:
        wt_contract = validate_wt_secondary_evidence(
            contrasts=contrasts.to_dict("records"),
            meta=meta.to_dict("records"),
            flow=flow.to_dict("records"),
        )
    except ValueError as error:
        raise FigureSuiteError(f"S9 WT secondary evidence is invalid: {error}") from error
    _require(
        standardized_medians.shape
        == relative_iqrs.shape
        == condition_coverage.shape
        == (32, 4),
        "multitrait raw-condition matrices changed",
    )
    _require(effects.shape == (5, 6), "fixed 15-effect matrix changed")
    display_labels = [
        label.split(" ", 1)[0] + "  " + label.split(" ", 1)[1].replace("_", " ")
        for label in labels
    ]
    endpoint_names = {
        "local_hair_count_1_4mm": "Local visible-hair count",
        "local_median_hair_length_um_1_4mm": "Local conditional median length",
        "first_hair_ge40um_distance_from_distal_point_um": "First observed ≥40-µm hair position",
        "median_root_width_um": "Median apparent root width",
        "visible_root_axis_length_um": "Visible root-axis extent",
    }
    effect_display_labels = [
        f"{label.split(' ', 1)[0]}  {endpoint_names[label.split(' ', 1)[1]]}"
        for label in effect_labels_raw
    ]
    log_effects = np.where(effects > 0, np.log2(effects), np.nan)
    configure_publication_style()
    figure = plt.figure(figsize=(8.2, 12.8))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(5.2, 1.2, 3.25),
        hspace=0.47,
        wspace=0.58,
    )
    raw_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    effect_axis = figure.add_subplot(grid[1, :])
    wt_grid = grid[2, :].subgridspec(1, 2, width_ratios=(2.05, 1.0), wspace=0.42)
    wt_known_axis = figure.add_subplot(wt_grid[0, 0])
    wt_unknown_axis = figure.add_subplot(wt_grid[0, 1])
    condition_labels = ("EV 22°C", "EV 30°C", "OE 22°C", "OE 30°C")
    missing_colour = "#E5E7EB"
    median_cmap = plt.get_cmap("RdBu_r").with_extremes(bad=missing_colour)
    iqr_cmap = plt.get_cmap("YlOrBr").with_extremes(bad=missing_colour)
    coverage_cmap = plt.get_cmap("YlGnBu").copy()
    effect_cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#E5E7EB")
    finite_medians = np.abs(standardized_medians[np.isfinite(standardized_medians)])
    median_limit = max(
        1.0,
        float(np.max(finite_medians)) if finite_medians.size else 1.0,
    )
    median_image = raw_axes[0].imshow(
        np.ma.masked_invalid(standardized_medians), aspect="auto",
        interpolation="nearest", vmin=-median_limit, vmax=median_limit,
        cmap=median_cmap,
    )
    iqr_image = raw_axes[1].imshow(
        np.ma.masked_invalid(relative_iqrs), aspect="auto",
        interpolation="nearest", vmin=0.0, vmax=1.0, cmap=iqr_cmap,
    )
    coverage_image = raw_axes[2].imshow(
        np.ma.masked_invalid(condition_coverage), aspect="auto",
        interpolation="nearest", vmin=0.0, vmax=1.0, cmap=coverage_cmap,
    )
    for index, (axis, title) in enumerate(
        zip(
            raw_axes,
            (
                "Raw condition median\n(within-descriptor z score)",
                "Raw condition IQR\n(fraction of trait-wise maximum)",
                "Condition-level\nnon-null coverage",
            ),
            strict=True,
        )
    ):
        axis.set_xticks(range(4), condition_labels, rotation=35, ha="right", fontsize=5.3)
        axis.set_yticks(range(32), display_labels if index == 0 else [], fontsize=4.5)
        axis.set_title(title, fontsize=7.1, fontweight="bold", pad=7)
        axis.axhline(18.5, color="#111827", linewidth=0.8)
        axis.set_ylim(31.5, -0.5)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    finite_effects = np.abs(log_effects[np.isfinite(log_effects)])
    effect_limit = max(0.25, float(np.max(finite_effects)) if finite_effects.size else 0.25)
    effect_image = effect_axis.imshow(
        np.ma.masked_invalid(log_effects), aspect="auto", interpolation="nearest",
        vmin=-effect_limit, vmax=effect_limit, cmap=effect_cmap,
    )
    effect_labels = [
        f"{cohort}\n{effect}"
        for cohort in ("Clean", "Full")
        for effect in ("OE:EV", "30:22 °C", "Interaction")
    ]
    effect_axis.set_xticks(range(6), effect_labels, rotation=25, ha="right", fontsize=5.7)
    effect_axis.set_yticks(range(5), effect_display_labels, fontsize=5.4)
    effect_axis.set_title(
        "D15 fixed five-endpoint / 15-effect family only (log2 ratio; unchanged by WT)",
        fontsize=7.3,
        fontweight="bold",
        pad=7,
    )
    effect_axis.tick_params(length=0)
    for spine in effect_axis.spines.values():
        spine.set_visible(False)

    median_bar = figure.colorbar(median_image, ax=raw_axes[0], fraction=0.06, pad=0.04)
    median_bar.set_label("Within-trait z score", fontsize=5.5)
    median_bar.ax.tick_params(labelsize=5)
    iqr_bar = figure.colorbar(iqr_image, ax=raw_axes[1], fraction=0.06, pad=0.04)
    iqr_bar.set_label("Relative IQR", fontsize=5.5)
    iqr_bar.ax.tick_params(labelsize=5)
    coverage_bar = figure.colorbar(coverage_image, ax=raw_axes[2], fraction=0.06, pad=0.04)
    coverage_bar.set_label("Non-null fraction", fontsize=5.5)
    coverage_bar.ax.tick_params(labelsize=5)
    effect_bar = figure.colorbar(effect_image, ax=effect_axis, fraction=0.025, pad=0.03)
    effect_bar.set_label("log2 ratio", fontsize=6)
    effect_bar.ax.tick_params(labelsize=5)
    effect_axis.text(
        0.5,
        -0.33,
        "The complete 192-slot D15 effect-status ledger is in Table S9; WT is a separate secondary family below.",
        transform=effect_axis.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
    )

    known_contract = _draw_wt_secondary_forest(
        wt_known_axis, contrasts=contrasts, meta=meta, unknown_day=False
    )
    unknown_contract = _draw_wt_secondary_forest(
        wt_unknown_axis, contrasts=contrasts, meta=meta, unknown_day=True
    )
    _require(
        unknown_contract["pooled_diamond_count"] == 0,
        "unknown-day WT panel contains a pooled diamond",
    )
    figure.legend(
        handles=tuple(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=WT_ENDPOINT_COLOURS[endpoint],
                markeredgecolor=WT_ENDPOINT_COLOURS[endpoint],
                label=f"{WT_ENDPOINT_CODES[endpoint]} · {ENDPOINT_LABELS[endpoint]}",
            )
            for endpoint in WT_SECONDARY_ENDPOINTS
        )
        + (
            Line2D(
                [0], [0], marker="D", linestyle="none", color=PALETTE["ink"],
                markerfacecolor="white", label="Same-day REML/HK pooled estimate",
            ),
        ),
        frameon=False,
        fontsize=4.7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
    )
    panel_label(raw_axes[0], "a", x=-0.76, y=1.03)
    panel_label(raw_axes[1], "b", x=-0.16, y=1.03)
    panel_label(raw_axes[2], "c", x=-0.16, y=1.03)
    panel_label(effect_axis, "d", x=-0.09, y=1.06)
    panel_label(wt_known_axis, "e", x=-0.11, y=1.03)
    panel_label(wt_unknown_axis, "f", x=-0.18, y=1.03)
    setattr(
        figure,
        "_phaxis_wt_secondary_figure_contract",
        {
            **wt_contract,
            "known_day_panel": known_contract,
            "unknown_day_panel": unknown_contract,
            "unknown_day_is_descriptive_only": True,
            "D15_fixed_effect_family_changed": False,
        },
    )
    _watermark(figure, provisional)
    return figure


def _flow_boxes(axis: plt.Axes, boxes: Sequence[tuple[float, float, str, str]], arrows: Sequence[tuple[int, int]], title: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    for x, y, label, colour in boxes:
        axis.add_patch(Rectangle((x - 0.14, y - 0.10), 0.28, 0.20, facecolor=colour, alpha=0.14, edgecolor=colour))
        axis.text(x, y, label, ha="center", va="center", fontsize=7)
    for start, end in arrows:
        x0, y0, _, _ = boxes[start]
        x1, y1, _, _ = boxes[end]
        axis.add_patch(FancyArrowPatch((x0, y0 - 0.10), (x1, y1 + 0.10), arrowstyle="->", mutation_scale=8, color=PALETTE["ink"]))
    axis.set_title(title, loc="left")


def _figure6(
    resources: Mapping[str, Path],
    receipts: Mapping[str, dict[str, Any]],
    source_hashes: Mapping[str, str],
    provisional: bool,
    final: bool,
) -> plt.Figure:
    flow = _read_table(resources["cohort_flow"], "cohort flow", ("node_id", "label", "count", "parent_id", "role"))
    workflow = _read_table(resources["workflow_stages"], "workflow stages", ("stage_order", "stage_name", "receipt_role", "output_identity_sha256"))
    runtime = _read_object(resources["runtime_summary"], "runtime summary")
    per_image = _read_table(resources["runtime_per_image"], "runtime per-image", ("source_unit", "wall_seconds", "megapixels", "io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"))
    _guard_no_blind_or_root_cap_region("runtime summary", runtime)
    _finite(per_image, ("wall_seconds", "megapixels", "io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"), "runtime per-image")
    _require(
        runtime.get("schema_version") == "PHAxis-manuscript-two-mode-runtime-input-1.0"
        and runtime.get("measurement_scope")
        == "raw_image_to_final_traits_and_profiles_direct"
        and runtime.get("batch_latency_is_never_derived_per_image") is True,
        "runtime is not the sealed two-mode direct benchmark input",
    )
    sequential = runtime.get("sequential_latency_full283")
    production = runtime.get("production_batch_full283")
    baseline_sequential = runtime.get("baseline_sequential_latency_full283")
    baseline_production = runtime.get("baseline_production_batch_full283")
    latency_comparison = runtime.get("latency_comparison")
    production_comparison = runtime.get("production_comparison")
    _require(
        all(
            isinstance(value, Mapping)
            for value in (
                sequential,
                production,
                baseline_sequential,
                baseline_production,
                latency_comparison,
                production_comparison,
            )
        ),
        "runtime two-mode or frozen-v1 comparison block missing",
    )
    latency_mode = runtime.get("latency_mode")
    _require(
        latency_mode in {"sequential_persistent_full283", "sequential_cold_cli_full283"}
        and sequential.get("benchmark_mode") == latency_mode
        and baseline_sequential.get("benchmark_mode") == latency_mode
        and latency_comparison.get("benchmark_mode") == latency_mode
        and production.get("benchmark_mode") == "production_batch_full283",
        "runtime A/B benchmark modes changed or were mislabeled",
    )
    _require(
        sequential.get("stage_timing_semantics") == "nonoverlapping_wall_components"
        and production.get("stage_timing_semantics")
        == "nonoverlapping_wall_components",
        "runtime stage timing is not a non-overlapping wall decomposition",
    )
    _require(
        runtime.get("per_image_csv_sha256") == sha256_file(resources["runtime_per_image"]),
        "runtime summary does not bind the per-image table",
    )
    stage_sum = per_image[
        ["io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"]
    ].sum(axis=1)
    _require(
        stage_sum.between(
            pd.to_numeric(per_image["wall_seconds"]) * 0.98,
            pd.to_numeric(per_image["wall_seconds"]) * 1.02,
        ).all(),
        "sequential per-source stage components do not reconstruct direct wall",
    )
    if final:
        _require(
            runtime.get("status") == "completed_two_mode_direct_full283"
            and sequential.get("status") == "completed_direct_full283"
            and production.get("status") == "completed_direct_full283"
            and baseline_sequential.get("status") == "completed_direct_full283"
            and baseline_production.get("status") == "completed_direct_full283",
            "final two-mode runtime or frozen-v1 benchmark is incomplete",
        )
        _require(
            sequential.get("images") == 283
            and production.get("images") == 283
            and per_image["source_unit"].nunique() == 283,
            "runtime scope is not full283",
        )
        _require(
            latency_comparison.get("comparable") is True
            and production_comparison.get("comparable") is True,
            "final frozen-v1 comparison is not like-for-like",
        )
    flow_counts = {str(row["node_id"]): int(row["count"]) for row in flow.to_dict("records")}
    for node in ("human443", "train399", "qcdevelopment44", "bio_full", "overlap", "bio_clean", "formal", "review_only"):
        _require(node in flow_counts, f"cohort flow missing {node}")
    _require(flow_counts["human443"] == flow_counts["train399"] + flow_counts["qcdevelopment44"], "HumanCurated split does not close")
    _require(flow_counts["bio_full"] == flow_counts["overlap"] + flow_counts["bio_clean"], "bio clean overlap flow does not close")
    _require(flow_counts["bio_full"] == flow_counts["formal"] + flow_counts["review_only"], "formal/review flow does not close")
    if final:
        _require(flow_counts["train399"] == 399 and flow_counts["qcdevelopment44"] == 44, "train/QC flow differs from evaluation")
        counts = receipts["cohorts"]["counts"]
        _require(flow_counts["bio_full"] == counts["biological_full"], "bio full flow differs from receipt")
        _require(flow_counts["bio_clean"] == counts["biological_clean"], "bio clean flow differs from receipt")
    _require(workflow["stage_order"].is_unique, "workflow stage order is not unique")
    _require(set(workflow["receipt_role"]).issubset(set(RECEIPT_ROLES)), "workflow references an unknown receipt")
    _require(workflow["output_identity_sha256"].map(_is_sha256).all(), "workflow identities are not SHA-256")

    configure_publication_style()
    figure, axes_grid = plt.subplots(2, 3)
    axes = list(axes_grid.ravel())
    axis = axes[0]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.set_title("Bytewise-separated evidence roles", loc="left")
    role_boxes = (
        (0.06, 0.66, 0.26, 0.20, f"HumanCurated\n{flow_counts['human443']}", PALETTE["navy"]),
        (0.39, 0.72, 0.22, 0.15, f"Train\n{flow_counts['train399']}", PALETTE["teal"]),
        (0.70, 0.72, 0.24, 0.15, f"QC-development\n{flow_counts['qcdevelopment44']}", PALETTE["gold"]),
        (0.06, 0.16, 0.26, 0.20, f"Application\n{flow_counts['bio_full']}", PALETTE["navy"]),
        (0.39, 0.19, 0.22, 0.15, f"SHA overlap\n{flow_counts['overlap']}", PALETTE["orange"]),
        (0.70, 0.19, 0.24, 0.15, f"Clean primary\n{flow_counts['bio_clean']}", PALETTE["teal"]),
    )
    for x, y, width, height, label_text, colour in role_boxes:
        axis.add_patch(
            Rectangle(
                (x, y), width, height, facecolor=colour, alpha=0.14,
                edgecolor=colour,
            )
        )
        axis.text(x + width / 2.0, y + height / 2.0, label_text, ha="center", va="center", fontsize=6.5)
    for start, ends in ((0, (1, 2)), (3, (4, 5))):
        x0, y0, width0, height0, _label, _colour = role_boxes[start]
        for end in ends:
            x1, y1, _width1, height1, _label1, _colour1 = role_boxes[end]
            axis.add_patch(
                FancyArrowPatch(
                    (x0 + width0, y0 + height0 / 2.0),
                    (x1, y1 + height1 / 2.0),
                    arrowstyle="->", mutation_scale=7, color=PALETTE["ink"],
                )
            )
    axis.text(
        0.06, 0.05,
        f"Application disposition: formal {flow_counts['formal']} | review-only {flow_counts['review_only']}",
        fontsize=6.2, color=PALETTE["grey"],
    )

    _flow_boxes(
        axes[1],
        (
            (0.50, 0.84, "External laboratory\nraw image + calibration manifest", PALETTE["navy"]),
            (0.50, 0.53, "PHAxis\nroot → identity → fusion", PALETTE["teal"]),
            (0.50, 0.18, "Reusable outputs\noverlay | hair rows | 32 traits | profiles", PALETTE["plum"]),
        ),
        ((0, 1), (1, 2)),
        "Raw image to reusable plant phenotype",
    )

    axis = axes[2]
    axis.set_axis_off()
    axis.set_title("Hash-linked raw-to-profile stages", loc="left")
    rows = workflow.sort_values("stage_order")
    for index, row in enumerate(rows.to_dict("records")):
        y = 0.90 - index * (0.78 / max(len(rows) - 1, 1))
        axis.add_patch(Rectangle((0.05, y - 0.05), 0.56, 0.10, facecolor=PALETTE["pale_teal"], edgecolor=PALETTE["teal"]))
        axis.text(0.08, y, str(row["stage_name"]), va="center", fontsize=6.5)
        axis.text(0.64, y, str(row["output_identity_sha256"])[:10], va="center", fontsize=6.0, family="monospace")
        if index + 1 < len(rows):
            axis.add_patch(FancyArrowPatch((0.33, y - 0.05), (0.33, y - 0.11), arrowstyle="->", mutation_scale=7, color=PALETTE["ink"]))

    axis = axes[3]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.set_title("Declared reproducibility contract", loc="left")
    contract_roles = (
        "Source", "License", "Use docs", "Model",
        "Input schema", "Output schemas", "Expected example",
    )
    for index, label_text in enumerate(contract_roles):
        column = index % 2
        row = index // 2
        x = 0.05 + column * 0.48
        y = 0.84 - row * 0.19
        axis.add_patch(
            Rectangle(
                (x, y - 0.07), 0.42, 0.13,
                facecolor=PALETTE["pale_teal"], edgecolor=PALETTE["teal"],
            )
        )
        axis.text(x + 0.21, y, label_text, ha="center", va="center", fontsize=6.5)
    short_hashes = [source_hashes[role][:7] for role in RECEIPT_ROLES]
    axis.text(
        0.05, 0.08,
        "Scientific evidence chain  " + " → ".join(short_hashes[:4])
        + "\n" + " " * 28 + " → ".join(short_hashes[4:]),
        fontsize=5.2, family="monospace",
    )
    axis.text(
        0.05, 0.015,
        "Stage-37 contract view; later release and clean-install gates are not implied.",
        fontsize=5.4, color=PALETTE["grey"],
    )

    axis = axes[4]
    values = pd.to_numeric(per_image["wall_seconds"])
    axis.hist(values, bins=min(18, max(5, int(np.sqrt(len(values))))), color=PALETTE["teal"], alpha=0.82)
    median = float(values.median())
    p95 = float(values.quantile(0.95))
    axis.axvline(median, color=PALETTE["ink"], linewidth=0.9, label=f"Median {median:.2f} s")
    axis.axvline(p95, color=PALETTE["orange"], linestyle="--", linewidth=0.9, label=f"P95 {p95:.2f} s")
    axis.set_xlabel("Direct wall time per image (s)")
    axis.set_ylabel("Source units")
    latency_label = "persistent" if latency_mode == "sequential_persistent_full283" else "cold CLI"
    axis.set_title(f"Observed end-to-end latency ({latency_label})", loc="left")
    axis.legend(frameon=False)
    _clean_axis(axis)

    axis = axes[5]
    production_stages = production.get("stage_timings")
    _require(
        isinstance(production_stages, list) and production_stages,
        "production batch stage timings missing",
    )
    stage_labels = tuple(str(row.get("stage")) for row in production_stages)
    stage_values = np.asarray(
        [_finite_number(row.get("wall_seconds"), f"production stage {label}") for row, label in zip(production_stages, stage_labels, strict=True)]
    )
    _require((stage_values >= 0).all() and stage_values.sum() > 0, "runtime stage durations invalid")
    batch_wall = _finite_number(production.get("batch_wall_seconds"), "production batch wall")
    _require(stage_values.sum() <= batch_wall * 1.02, "production stages exceed direct batch wall")
    stage_colours = (PALETTE["grey"], PALETTE["gold"], PALETTE["teal"], PALETTE["plum"], PALETTE["orange"], PALETTE["navy"])
    axis.barh([0], [stage_values[0]], color=stage_colours[0], label=stage_labels[0])
    left = stage_values[0]
    for index, (value, label) in enumerate(zip(stage_values[1:], stage_labels[1:], strict=True), start=1):
        colour = stage_colours[index % len(stage_colours)]
        axis.barh([0], [value], left=[left], color=colour, label=label)
        left += value
    axis.set_yticks([])
    axis.set_ylim(-0.78, 0.65)
    axis.set_xlabel("Direct production-batch wall components (s)")
    axis.set_title("Batch composition and like-for-like reuse speed", loc="left")
    axis.legend(frameon=False, ncol=2, loc="upper center", fontsize=5.2)
    hardware = production.get("hardware", {})
    if isinstance(hardware, Mapping):
        hardware_text = str(hardware.get("gpu_names") or hardware.get("gpus") or hardware.get("platform") or "hardware identity sealed")
    else:
        hardware_text = "hardware identity sealed"
    axis.text(
        0.02, 0.05,
        f"{float(production.get('images_per_min')):.3g} images min⁻¹ | "
        f"{float(production.get('megapixels_per_second')):.3g} MP s⁻¹ | "
        f"legacy/PHAxis median {float(latency_comparison.get('median_latency_speedup_frozen_v1_over_phaxis')):.3g}× | "
        f"batch wall {float(production_comparison.get('batch_wall_speedup_frozen_v1_over_phaxis')):.3g}×\n"
        f"direct batch {batch_wall:.2f} s / 283 | {hardware_text}",
        fontsize=5.6, transform=axis.transAxes,
    )
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index), x=-0.16, y=1.06)
    _watermark(figure, provisional)
    return figure


def _source_groups() -> dict[str, tuple[str, ...]]:
    groups = {
        stem: tuple(roles)
        for stem, roles in zip(
            FIGURE_STEMS,
            MAIN_FIGURE_RESOURCE_ROLES,
            strict=True,
        )
    }
    groups.update(
        {
            stem: tuple(roles)
            for stem, roles in zip(
                SUPPLEMENTARY_STEMS,
                SUPPLEMENTARY_FIGURE_RESOURCE_ROLES,
                strict=True,
            )
        }
    )
    return groups


def _copy_portable_overlay_assets(
    *,
    selection_source: Path,
    selection_copy: Path,
    source_data_root: Path,
    staging: Path,
) -> dict[str, str]:
    selection = _read_table(
        selection_source,
        "source-data overlay selection",
        (
            "case_id",
            "case_role",
            "source_path",
            "source_sha256",
            "overlay_path",
            "overlay_sha256",
        ),
    )
    source_root = selection_source.parent.resolve()
    destination_root = source_data_root.resolve()
    hashes: dict[str, str] = {}
    for row in selection.to_dict("records"):
        case_id = str(row["case_id"])
        for prefix in ("source", "overlay"):
            raw_value = str(row[f"{prefix}_path"]).strip()
            relative = Path(raw_value)
            _require(raw_value != "", f"{case_id}: empty {prefix} source-data path")
            _require(
                not relative.is_absolute() and relative.drive == "",
                f"{case_id}: absolute {prefix} source-data path is forbidden",
            )
            _require(
                relative.parts
                and all(part not in {"", ".", ".."} for part in relative.parts),
                f"{case_id}: non-portable {prefix} source-data path",
            )
            _require(
                "blind" not in raw_value.casefold(),
                f"{case_id}: blind-labelled {prefix} source-data path refused",
            )
            expected = str(row[f"{prefix}_sha256"])
            _require(_is_sha256(expected), f"{case_id}: invalid {prefix} source-data SHA")

            cursor = source_root
            for part in relative.parts:
                cursor = cursor / part
                _require(
                    not cursor.is_symlink(),
                    f"{case_id}: symlink {prefix} source-data path is forbidden",
                )
            source = (source_root / relative).resolve()
            _require(
                source.is_relative_to(source_root),
                f"{case_id}: {prefix} source-data path escapes its resource root",
            )
            _require(
                source.is_file(),
                f"{case_id}: missing {prefix} source-data image",
            )
            _require(
                sha256_file(source) == expected,
                f"{case_id}: {prefix} source-data image hash mismatch",
            )

            destination = source_data_root / relative
            _require(
                destination.resolve().is_relative_to(destination_root),
                f"{case_id}: {prefix} source-data destination escapes bundle",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                _require(
                    destination.is_file()
                    and not destination.is_symlink()
                    and sha256_file(destination) == expected,
                    f"{case_id}: colliding {prefix} source-data destination",
                )
            else:
                shutil.copyfile(source, destination)
            _require(
                sha256_file(destination) == expected,
                f"{case_id}: copied {prefix} source-data image hash mismatch",
            )
            key = str(destination.relative_to(staging)).replace("\\", "/")
            hashes[key] = expected

    copied_selection = _read_table(
        selection_copy,
        "copied source-data overlay selection",
        ("case_id", "case_role"),
    )
    _verify_overlay_rows(copied_selection, selection_copy.parent, final=False)
    return hashes


def _copy_source_data(
    staging: Path,
    resources: Mapping[str, Path],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    root = staging / "source_data"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, str]] = {}
    for figure, roles in _source_groups().items():
        hashes: dict[str, str] = {}
        for role in roles:
            source = resources[role]
            destination = root / f"{figure}_{role}{source.suffix.lower()}"
            shutil.copyfile(source, destination)
            hashes[str(destination.relative_to(staging)).replace("\\", "/")] = sha256_file(destination)
            if role == "overlay_selection":
                hashes.update(
                    _copy_portable_overlay_assets(
                        selection_source=source,
                        selection_copy=destination,
                        source_data_root=root,
                        staging=staging,
                    )
                )
        result[figure] = hashes

    expected: dict[str, str] = {}
    for figure, hashes in result.items():
        for path, digest in hashes.items():
            _require(
                path not in expected or expected[path] == digest,
                f"source-data hash collision across figure maps: {figure}: {path}",
            )
            expected[path] = digest
    observed = {
        str(path.relative_to(staging)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    _require(
        observed == expected,
        "physical source-data files do not exactly close the declared figure maps",
    )
    return result, observed


def _legends_and_alt_text(
    *,
    provisional: bool,
    runtime: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    selected = validate_narrative_decision(decision)
    titles = title_contract(selected)["figures"]
    banner = "**PROVISIONAL — NOT FOR SUBMISSION.**\n\n" if provisional else ""
    return f"""# PHAxis 1.0.0 publication-figure legends and alt text

{banner}## Figure 1. {titles['1']}

**Legend.** (a) Raw Arabidopsis microscopy image with physical scale bar. (b) Visible main-root body, ordered axis, one distal root landmark at s=0, hair attachments, accepted identity vectors, and endpoint-complete curves under the same global linear display. (c) Root-hair observations accrue information hierarchically: identity supports abundance and position, whereas an endpoint-complete one-to-one curve additionally supports projected morphology. (d) Thirty-two canonical descriptors: 19 primary-root and 13 root-hair descriptors organized into five measurement families: visible-hair abundance, conditional projected length, axial deployment, visible-root extent, and root form/trajectory. Some descriptors are normalized or mathematically coupled views of the same biological primitive and are not presented as statistically independent phenotypes. (e) Hash-linked measurement workflow. The distal output is one landmark, and condition labels do not route inference.

**Alt text.** A raw root image is paired with a coloured coordinate overlay. Schematics distinguish accepted hair identity from an endpoint-complete projected-length curve, group 32 descriptors into five measurement families, and connect raw image, root, hair identity, fusion, traits, and axial profiles.

## Figure 2. {titles['2']}

**Legend.** (a) A hash-bound QC-development44 assignment exposes the actual one-to-one truth/prediction pairs produced by the production matcher at 20 µm; unmatched truth and prediction instances remain visible. Biological-hair presence requires bidirectional partial-centreline support and a non-opposing proximal direction, but neither distal-endpoint coincidence nor complete-line overlap. (b) Precision, recall and F1 with image-level bootstrap intervals and the paired Stage-B-minus-legacy F1 difference. (c) Per-image visible-hair count agreement, summarized by count MAE. (d) Paired change in biological-presence F1 versus change in absolute count error. Operating-point selection, the 5/10/20-µm tolerance family and historical family-isolated OOF443 strata remain in Supplementary Fig. S2. QC-development44 is overlay-visible development evidence, not independent evidence or an independent test set.

**Alt text.** Four panels show real one-to-one centreline assignments, QC-development44 biological-presence performance, visible-hair count agreement, and paired comparison with the predecessor; QC-development44 is not an independent test set.

## Figure 3. {titles['3']}

**Legend.** (a) Annotated primary-root Dice, boundary F1 and HD95. (b) Maximum single-component coverage, best-component unsupported gap, break-free rate and visible-axis extent error on the final root mask used for trait extraction; no evaluator-side bridge is added. (c) Distal-point physical error and PCK together with physical-scale detection, line-endpoint localization and conditional calibration agreement. The full 38-visible-bar + 6-trusted-metadata applicability statement remains in Table 2; absence specificity is not estimable and fail-closed behaviour is a software contract plus unit-test result. (d) Reference versus predicted length for one-to-one length-resolved hair curves. (e) Application-group fraction of accepted hairs that supports projected length. (f) Source-unit concordance across all 19 primary-root descriptors, grouped by the six prespecified root-assurance subgroups; native-unit MAE and bias are in Supplementary Fig. S4. Attachment assurance is paired with hair recovery in Fig. 2, while portable geometry-engine equivalence remains in Fig. 6 and Supplementary Fig. S5.

**Alt text.** Root-body and continuous-axis assurance cards, a combined distal-landmark and scale plot, matched-length agreement, group-specific length coverage, and a full-width lollipop plot for all 19 primary-root descriptors summarize plant-facing measurement validity.

## Figure 4. {titles['4']}

**Legend.** Five preselected acquisition-challenge roles include routine morphology, low contrast (`RHSCU-aa5b6e37df15821f`), curved dense hair (`RHSCU-bbf649822174e0a2`), root continuity, and a fail-closed review-only image. Each row contains the source image, a whole-image overlay and a hash-bound audit-2.0 card. Orange boxes show deterministic axis-geometry-derived insets for the two prelocked difficult anchors; source and overlay use identical crop coordinates and retain whole-image context. Cards report axis-in-root coverage, maximum single-component support, longest unsupported gap, formal identity n, endpoint-complete n/fraction, [1,4)-mm window and [0,5)-mm profile eligibility, and the governing reason. Review-only formal metrics remain null rather than zero. These morphology-driven illustrations are not a performance sample. Experimental-condition metadata entered neither prediction, overlay pixels nor morphology-evidence values on the cards; only after pixels were fixed did it organise review directories and assign formal/review labels. Cyan marks the root boundary, white the ordered axis, magenta the distal point, green an endpoint-complete projected-length curve, and amber a Stage-B identity vector that is not itself a length. An orange point on an amber vector marks only the Stage-B vector terminus; only the terminus of a green one-to-one matched curve supports endpoint-complete length.

**Alt text.** Five source/overlay/audit triplets progress from routine morphology through low-contrast, curved dense-hair, continuity and fail-closed examples. Deterministic insets mark the two named difficult anchors, and every audit card distinguishes union root support, one-component continuity, endpoint support and downstream eligibility.

## Figure 5. {titles['5']}

**Legend.** (a) Clean-cohort D15 source-unit observations are ordered as N (H08, visible population), L (H11, supported morphology), F (H07, deployment boundary), W (R07, carrying-root calibre), and A (R01, visible organ extent), with non-null n for every archived condition. (b) The fixed 15-effect family is shown in the same N/L/F/W/A order: construct label, temperature and interaction estimates with 95% model-based intervals; filled points show clean-cohort D15 estimates and hollow points show full-cohort D15 overlap-inclusion sensitivity. A headline cell requires the clean interval to exclude the null and the Full283 point estimate to retain the clean direction. This immutable decision yields Branch {selected['branch_id']}; distal profiles never select or veto it. (c) Visible-hair abundance per eligible 1-mm distal-axis bin. (d) Length-support fraction by bin. (e) Per-image conditional projected length by bin. Exact condition-specific denominators remain in the source-data table. Missing bins are not zero-filled and no profile hypothesis tests are performed. The complete four-condition map and all 32 descriptors remain in Supplementary Fig. S9.

**Alt text.** N/L/F/W/A endpoint distributions and clean/full effect estimates are followed by descriptive distal-axis profiles; the hash-bound Branch {selected['branch_id']} decision is made only from the fixed 15-effect family.

## Figure 6. {titles['6']}

**Legend.** (a) Family-isolated development and bytewise-separated application roles. (b) An external laboratory supplies a raw image and calibration manifest and receives inspectable overlays, hair-instance rows, 32 image-level descriptors and distal-axis profiles. (c) Hash-linked scientific stages bind the raw-image-to-profile journey. (d) The declared reuse contract names source, license, use documentation, model, input schema, output schemas and an expected example; this stage-37 view does not imply that later release or clean-install gates have passed. (e) Mode A is the observed 283-image `{runtime.get('latency_mode', 'undeclared')}` raw-to-final per-image latency distribution; cold-CLI measurements include per-image startup and are never relabelled persistent. (f) Mode B is a separate direct `production_batch_full283` run with non-overlapping batch-stage wall components, throughput and like-for-like legacy RHAxis/RhizoWeave v1.0 workflow speedups. Speed comparisons require exact 283 source/image locks, hardware identity, latency mode, I/O-inclusive scope and no-cache direct runs. Timing scope is `{runtime.get('measurement_scope', 'undeclared')}` and includes raw-image I/O through traits and profiles.

**Alt text.** Development and application evidence roles lead into an external-laboratory raw-image-to-atlas journey, a hash-linked scientific workflow, a declared reuse contract, directly measured latency and production-batch timing, and like-for-like predecessor speed comparisons.

## Supplementary Figure S1. Stage-B physical input representation, multihead architecture, and target contract

**Legend.** The sealed representative input is transformed into robust intensity, 40-µm local contrast and 2.5/7-µm dark-ridge channels using its physical scale. Schematics show the shared ResNet34 encoder–decoder, nine tensor heads, and vector-derived point, line, direction and root-support targets. Base direction and base length jointly encode the base-to-tip vector supervision. Architecture text is method-contract context; image-derived panels and expert identity are recomputed from the hash-locked image, geometry and Stage-B receipt.

**Alt text.** Four physically scaled image representations are followed by model, output-field, target and sealed-identity schematics.

## Supplementary Figure S2. Family-isolated split, operating-point selection, and development strata

**Legend.** (a) The 399/44 family-isolated split. (b) QC-development44 operating-threshold F1 and count MAE. (c) QC-development44 biological-presence precision, recall and F1 across 5, 10 and 20 µm tolerances. (d) QC-development44 per-image count agreement. (e) Historical OOF443 annotation-source strata. (f) Historical OOF443 density/hair-count strata. Both cohorts are development evidence, not independent-test accuracy evidence.

**Alt text.** Split counts, threshold and tolerance curves, count agreement and development strata are displayed from sealed sufficient-statistic tables.

## Supplementary Figure S3. Identity, formal attachment, endpoint, and conditional-length assurance

**Legend.** Tolerant biological presence is separated from conditional endpoint-complete geometry. (a) QC-development44 presence precision, recall and F1 across physical tolerances. (b) Matched endpoint-complete projected-length agreement and CCC. (c) Distal endpoint-error distribution. (d) Hair-curve trajectory continuity, distinct from primary-root connected-component continuity. (e) Application-cohort group-specific endpoint-complete support with explicit identity and length-supported denominators. (f) Formal QC-development44 attachment-qualified precision, recall and F1 at 20 µm, plus median and P95 attachment error on identities returned by the biological-presence matcher. Base-only proxies are excluded from panel f.

**Alt text.** Detection, formally qualified attachment, matched length, endpoint, continuity and support panels distinguish hair identity, base localization and the subset eligible for length measurement.

## Supplementary Figure S4. Agreement of 19 derived primary-root descriptors

**Legend.** All 19 canonical primary-root descriptors are regrouped into the six prespecified assurance subgroups: axis extent, axis shape, projected area, global width distribution, axial width pattern and centerline curvature. Each estimable point is the source-unit paired CCC and is coloured categorically by assurance subgroup; the row label gives trait ID, an abbreviated descriptor, eligible n, native-unit MAE and unit. When CCC is not estimable, the row is retained and marked NA rather than omitted. Missing any canonical root descriptor or assurance subgroup fails the final build.

**Alt text.** Three grouped panels retain every root descriptor, use six subgroup colours for estimable CCC points, report native-unit MAE in labels and explicitly mark non-estimable CCC as NA.

## Supplementary Figure S5. Root-provider equivalence, same-component root continuity, formal attachment, and tiled-inference assurance

**Legend.** Three portable root-provider layers must each reproduce all 283 source images exactly. The adjacent annotated QC-development panel reports the complete formal continuity family—mean and median maximum single-connected-component coverage, median unsupported gap on the best component, break-free rate and visible-axis-extent MAE—and the complete formal attachment family—20-µm attachment-qualified precision, recall and F1 plus median and P95 attachment error on all identities returned by the biological-presence matcher. Union coverage and base-only threshold-selection proxies are excluded from this formal panel. Other panels retain the deterministic boundary-covering tile/stitch contract, sealed Stage-B execution-path counts and immutable receipt identities. Exact implementation equivalence is not relabelled annotated biological accuracy.

**Alt text.** All-283-image provider-equivalence bars sit beside same-component root-continuity and formally matched hair-attachment metrics, a tile-stitch schematic, execution-path counts and sealed identities.

## Supplementary Figure S6. Expanded acquisition-challenge overlay gallery

**Legend.** The five sealed, morphology-driven roles—routine morphology, low contrast, curved dense hair, continuity-completed and review-only fail-closed—are shown as source/whole-image-overlay pairs. Every source and overlay panel is rendered with its sealed whole-image linear display limits and receives a physical scale bar whose pixel length and micrometre value come from the same sealed case record. The low-contrast and curved dense-hair rows are the preselected acquisition-challenge images `RHSCU-aa5b6e37df15821f` and `RHSCU-bbf649822174e0a2`. The gallery is not a performance sample. Experimental-condition metadata entered neither prediction nor overlay pixels; only after pixels were fixed did it organise review directories and assign formal/review labels. Amber is a Stage-B identity vector and its orange terminus is not length evidence; endpoint-complete length requires the terminus of a green one-to-one matched curve.

**Alt text.** Five rows pair whole source and overlay images from routine through difficult and fail-closed review cases; both panels in every pair carry the same calibrated physical scale bar.

## Supplementary Figure S7. Clean-cohort D15 analysis, full-cohort D15 sensitivity, and observability

**Legend.** (a) Clean-cohort D15 primary and full-cohort D15 overlap-inclusion sensitivity estimates are paired for the exact 15-effect family in prespecified endpoint order and, within endpoint, construct–temperature–interaction order; colour identifies the effect type. (b) Full-cohort D15 sensitivity minus clean-cohort D15 primary differences are shown in the same order, with colour indicating whether direction relative to the null ratio is retained. (c) Clean-cohort D15 source-unit first-observed-≥40-µm-hair distances are shown; this conditional quantity is absent when no qualifying hair is observed. (d) Endpoint-complete support among accepted identities by archived experimental group is labelled as supported/identity hairs and source-unit n. (e) Clean-cohort D15 source-unit visible-hair abundance along the distal axis. (f) Clean-cohort D15 source-unit length-support fraction along the distal axis. Only panels (a) and (b) compare the clean-cohort D15 primary analysis with full-cohort D15 sensitivity; missing measurements remain missing rather than being filled with zero. First-hair observability and edge-censoring statuses remain in Table S9.

**Alt text.** Ordered clean-cohort D15 primary/full-cohort D15 sensitivity effect comparisons are accompanied by conditional clean-cohort D15 first-hair observations, supported/identity denominators, and clean-cohort D15 distal-axis abundance and length-support profiles.

## Supplementary Figure S8. Direct runtime, memory, utilization, and I/O decomposition

**Legend.** (a) PHAxis per-image end-to-end latency distribution with median and P95. (b) Observed 283-image production-batch wall time for PHAxis and the frozen legacy comparator. (c) Images min⁻¹ and MP s⁻¹. (d) PHAxis non-overlapping median stage times for I/O, preprocessing, inference and postprocessing. (e) Peak VRAM and mean GPU utilization for both workflows. (f) Per-image megapixels versus wall time, coloured by inference time. Missing telemetry fails the final build.

**Alt text.** Latency, batch wall, throughput, stage timing, accelerator use and image-size scaling compare like-for-like direct workflows.

## Supplementary Figure S9. Clean-cohort D15 32-descriptor phenotype map and block/day-stratified WT temperature secondary evidence

**Legend.** (a) Within-descriptor standardized clean-cohort D15 raw medians across EV-22°C, EV-30°C, OE-22°C, and OE-30°C for all 32 canonical descriptors in trait-contract order (R01–R19, H01–H13). (b) Raw condition IQR divided by the largest finite IQR within each descriptor. (c) Non-null source-unit fraction by descriptor and condition; a measured zero remains observed and an unobservable summary remains blank. (d) Clean-cohort D15 primary and full-cohort D15 sensitivity construct, temperature, and interaction ratios for only the prespecified five-endpoint/15-effect family; the WT secondary family neither selects nor alters this D15 family or its narrative branch. (e) WT 30°C/22°C ratios are shown first as within-experiment estimates with 95% intervals, then as developmental-day-specific random-effects REML/Hartung–Knapp diamonds only when at least three same-day experiments are eligible for that endpoint. A typed `Not estimable` label replaces, rather than visually implying, every ineligible pooled estimate. (f) WT experiments with unknown developmental day remain descriptive within-experiment contrasts and are never pooled; consequently this panel contains no pooled diamond. N, L, F, W, and A denote local visible-hair count, local conditional projected length, first observed ≥40-µm hair position, median apparent root width, and visible root-axis extent. Clean261 and overlap-inclusive Full283 sensitivity estimates remain separate throughout. Table S9 retains native-unit median/Q25/Q75/IQR, observability, the complete 192-slot D15 effect-status ledger, and the separate WT experiment/day/QC ledgers.

**Alt text.** Three 32-row heatmaps show clean-cohort D15 four-condition median patterns, relative IQR, and coverage for every root and root-hair descriptor. A compact five-row panel preserves the fixed D15 15-effect family. Two separate WT forest panels show known-day experiment estimates and eligible same-day pooled estimates, followed by unknown-day descriptive experiment estimates with no pooling.
"""


def _rebase_bundle_paths(bundle: Mapping[str, Any], staging: Path, output: Path) -> dict[str, Any]:
    rebased = deepcopy(dict(bundle))
    files = rebased.get("files", {})
    rebased["files"] = {
        key: str((output / Path(value).resolve().relative_to(staging.resolve())).resolve())
        for key, value in files.items()
    }
    return rebased


def build_figure_suite(
    *,
    mode: str,
    figure_inputs: str | Path,
    model_contract_proposal: str | Path,
    output: str | Path,
    receipt_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Validate all inputs, render six figures, and atomically publish a suite."""
    _require(mode in {"final", "provisional"}, "mode must be final or provisional")
    figure_inputs_path = Path(figure_inputs).resolve()
    model_contract_proposal_path = Path(model_contract_proposal).resolve()
    output_path = Path(output).resolve()
    _require(not output_path.exists(), f"output already exists: {output_path}")
    if mode == "final":
        lowered_output = str(output_path).casefold()
        _require(
            not any(marker in lowered_output for marker in FORBIDDEN_FINAL_MARKERS),
            "final output path contains a provisional/development marker",
        )
    paths = {role: Path(receipt_paths[role]).resolve() for role in RECEIPT_ROLES}
    (
        manifest,
        resources,
        resource_hashes,
        receipts,
        source_hashes,
        _proposal,
        proposal_file_sha256,
        proposal_identity_sha256,
        assembly_identity_sha256,
        figure_input_provenance,
    ) = _prepare_inputs(
        mode=mode,
        figure_inputs=figure_inputs_path,
        receipt_paths=paths,
        model_contract_proposal=model_contract_proposal_path,
    )
    final = mode == "final"
    provisional = not final
    decision = validate_narrative_decision(
        _read_object(resources["narrative_decision"], "narrative decision")
    )
    _require(
        manifest.get("narrative_decision_identity_sha256")
        == decision["narrative_decision_identity_sha256"]
        and manifest.get("narrative_branch_id") == decision["branch_id"],
        "figure inputs do not bind the narrative decision",
    )
    wt_secondary_evidence = _validate_wt_figure_resources(
        resources=resources,
        manifest=manifest,
        analysis_summary=receipts["analysis"],
    )
    locked_titles = title_contract(decision)
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary component short on Windows: the output bundle already
    # contains long, explicit publication stems and must remain below MAX_PATH
    # even when the caller chooses a descriptive output directory.
    staging = Path(tempfile.mkdtemp(prefix=".figures-", dir=parent)).resolve()
    try:
        source_inputs, supplementary_source_hashes = _source_input_paths(
            figure_inputs_path, manifest
        )
        supplementary_sources, supplementary_source_identities = (
            _supplementary_table_sources(
                source_inputs=source_inputs,
                resources=resources,
                receipt_paths=paths,
                proposal_path=model_contract_proposal_path,
            )
        )
        try:
            supplementary_table_bundle = (
                materialize_supplementary_table_data_bundle(
                    output=staging / SUPPLEMENTARY_TABLE_DIRECTORY,
                    status=(
                        FINAL_SUPPLEMENTARY_TABLE_STATUS
                        if final
                        else PROVISIONAL_SUPPLEMENTARY_TABLE_STATUS
                    ),
                    source_paths=supplementary_sources,
                    source_identities=supplementary_source_identities,
                    figure_input_manifest_sha256=sha256_file(figure_inputs_path),
                    figure_input_assembly_identity_sha256=assembly_identity_sha256,
                    model_contract_proposal_identity_sha256=proposal_identity_sha256,
                )
            )
        except SupplementaryTableError as error:
            raise FigureSuiteError(
                f"supplementary Table/Data S1--S10 materialization failed: {error}"
            ) from error
        source_data, source_data_files = _copy_source_data(staging, resources)
        source_data_identity_sha256 = sha256_json(
            {
                "figure_source_data_sha256": source_data,
                "physical_source_data_sha256": source_data_files,
            }
        )
        figures: list[plt.Figure] = []
        figures.append(_figure1(resources, provisional))
        figures.append(_figure2(resources, provisional, final))
        figures.append(_figure3(resources, provisional))
        figure4, overlay_records = _figure4(resources, provisional, final)
        figures.append(figure4)
        figures.append(_figure5(resources, provisional, final))
        figures.append(_figure6(resources, receipts, source_hashes, provisional, final))
        _require(len(figures) == 6, "internal six-figure route changed")

        figure_records: dict[str, dict[str, Any]] = {}
        figure_hashes: dict[str, dict[str, Any]] = {}
        for index, (stem, figure) in enumerate(zip(FIGURE_STEMS, figures, strict=True)):
            filename_stem = f"PROVISIONAL_{stem}" if provisional else stem
            height = 188.0 if index == 3 else 148.0 if index == 4 else 122.0
            bundle = save_figure_bundle(
                figure,
                staging / filename_stem,
                width_mm=178.0,
                height_mm=height,
                check_edge_ink=True,
            )
            rebased = _rebase_bundle_paths(bundle, staging, output_path)
            hashes: dict[str, Any] = dict(bundle["sha256"])
            hashes["source_data"] = source_data[stem]
            figure_hashes[stem] = hashes
            figure_records[stem] = {
                "number": index + 1,
                "title": locked_titles["figures"][str(index + 1)],
                "status": "final" if final else "provisional_not_for_submission",
                "bundle": rebased,
                "source_data_sha256": source_data[stem],
            }

        supplementary_builders = (
            lambda: _supplementary_s1(resources, receipts, provisional),
            lambda: _supplementary_s2(resources, provisional),
            lambda: _supplementary_s3(resources, provisional),
            lambda: _supplementary_s4(resources, provisional),
            lambda: _supplementary_s5(resources, receipts, provisional),
            lambda: _supplementary_s6(resources, provisional),
            lambda: _supplementary_s7(resources, provisional),
            lambda: _supplementary_s8(resources, provisional),
            lambda: _supplementary_multitrait_atlas(resources, provisional),
        )
        supplementary_heights_mm = (148.0, 165.0, 175.0, 190.0, 165.0, 234.0, 175.0, 165.0, 300.0)
        _require(
            len(supplementary_builders)
            == len(SUPPLEMENTARY_STEMS)
            == len(SUPPLEMENTARY_FIGURE_TITLES)
            == 9,
            "internal ordered S1--S9 route changed",
        )
        supplementary_records: dict[str, dict[str, Any]] = {}
        supplementary_bundle_hashes: dict[str, dict[str, Any]] = {}
        supplementary_contract = manifest["supplementary_figure_contract"]
        contract_records = supplementary_contract["figures"]
        for index, (stem, title, builder, height_mm, contract_record) in enumerate(
            zip(
                SUPPLEMENTARY_STEMS,
                SUPPLEMENTARY_FIGURE_TITLES,
                supplementary_builders,
                supplementary_heights_mm,
                contract_records,
                strict=True,
            ),
            start=1,
        ):
            supplementary_figure = _guarded_supplementary(
                stem=stem,
                final=final,
                builder=builder,
            )
            filename_stem = f"PROVISIONAL_{stem}" if provisional else stem
            bundle = save_figure_bundle(
                supplementary_figure,
                staging / filename_stem,
                width_mm=178.0,
                height_mm=height_mm,
                check_edge_ink=True,
            )
            rebased = _rebase_bundle_paths(bundle, staging, output_path)
            hashes: dict[str, Any] = dict(bundle["sha256"])
            hashes["source_data"] = source_data[stem]
            supplementary_bundle_hashes[stem] = hashes
            supplementary_records[stem] = {
                "number": f"S{index}",
                "title": title,
                "status": "final" if final else "provisional_not_for_submission",
                "bundle": rebased,
                "source_data_sha256": source_data[stem],
                "resource_roles": list(contract_record["resource_roles"]),
                "receipt_roles": list(contract_record["receipt_roles"]),
                "receipt_file_sha256": {
                    role: source_hashes[role]
                    for role in contract_record["receipt_roles"]
                },
            }
        supplementary_identity_sha256 = sha256_json(
            supplementary_bundle_hashes
        )
        multitrait_atlas_payload = _read_object(
            resources["multitrait_atlas"], "multitrait atlas"
        )

        runtime = _read_object(resources["runtime_summary"], "runtime summary")
        legends = _legends_and_alt_text(
            provisional=provisional,
            runtime=runtime,
            decision=decision,
        )
        legends_path = staging / "figure_legends_and_alt_text.md"
        legends_path.write_text(legends, encoding="utf-8", newline="\n")
        overlay_source_hashes = {
            str(record["case_id"]): {
                "source": str(record["source_sha256"]),
                "overlay": str(record["overlay_sha256"]),
            }
            for record in overlay_records
        }
        source_hash_manifest = {
            "schema_version": "PHAxis-manuscript-figure-source-hashes-1.0",
            "status": "final" if final else "provisional_not_for_submission",
            "receipt_file_sha256": source_hashes,
            "model_contract_proposal_sha256": proposal_file_sha256,
            "model_contract_proposal_identity_sha256": proposal_identity_sha256,
            "model_contract_public_identity": manifest[
                "model_contract_public_identity"
            ],
            "model_bundle_id": manifest["model_contract_public_identity"][
                "model_bundle_id"
            ],
            "root_expert_id": manifest["model_contract_public_identity"][
                "root_expert_id"
            ],
            "hair_identity_expert_id": receipts["stageb"][
                "detection_model_metadata"
            ]["expert_id"],
            "figure_input_manifest_sha256": sha256_file(figure_inputs_path),
            "figure_input_assembly_identity_sha256": assembly_identity_sha256,
            "figure_input_provenance_receipts": figure_input_provenance,
            "train399_prediction_input_provenance": manifest[
                "train399_prediction_input_provenance"
            ],
            "resource_file_sha256": resource_hashes,
            "figure_source_data_sha256": source_data,
            "physical_source_data_sha256": source_data_files,
            "source_data_identity_sha256": source_data_identity_sha256,
            "multitrait_atlas_identity_sha256": multitrait_atlas_payload[
                "atlas_identity_sha256"
            ],
            "wt_secondary_evidence": wt_secondary_evidence,
            "supplementary_figure_bundle_sha256": supplementary_bundle_hashes,
            "supplementary_figure_bundle_identity_sha256": supplementary_identity_sha256,
            "supplementary_figure_contract_identity_sha256": supplementary_contract[
                "contract_identity_sha256"
            ],
            "supplementary_table_source_input_sha256": supplementary_source_hashes,
            "supplementary_table_source_authority_sha256": supplementary_table_bundle[
                "source_authority_sha256"
            ],
            "supplementary_table_source_authority_identity": supplementary_table_bundle[
                "source_authority_identity"
            ],
            "supplementary_table_bundle_receipt_sha256": supplementary_table_bundle[
                "receipt_sha256"
            ],
            "supplementary_table_bundle_identity_sha256": supplementary_table_bundle[
                "bundle_identity_sha256"
            ],
            "supplementary_table_bundle_sha256": supplementary_table_bundle[
                "bundle_file_sha256"
            ],
            "overlay_image_sha256": overlay_source_hashes,
            "narrative_decision_identity_sha256": decision[
                "narrative_decision_identity_sha256"
            ],
            "narrative_branch_id": decision["branch_id"],
            "title_contract": locked_titles,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        }
        source_hash_manifest["manifest_identity_sha256"] = sha256_json(
            source_hash_manifest
        )
        source_hash_path = staging / "source_hashes.json"
        atomic_write_json(source_hash_path, source_hash_manifest)
        status = "final_sealed_strict_train399_only" if final else "provisional_not_for_submission"
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "formal_train399_only_gate_passed": final,
            "deployment_figures_generated": True,
            "deployment_figures_provisional": provisional,
            "submission_use_allowed": final,
            "source_summary_sha256": source_hashes,
            "model_contract_proposal_sha256": proposal_file_sha256,
            "model_contract_proposal_identity_sha256": proposal_identity_sha256,
            "model_contract_public_identity": manifest[
                "model_contract_public_identity"
            ],
            "model_bundle_id": manifest["model_contract_public_identity"][
                "model_bundle_id"
            ],
            "root_expert_id": manifest["model_contract_public_identity"][
                "root_expert_id"
            ],
            "hair_identity_expert_id": receipts["stageb"][
                "detection_model_metadata"
            ]["expert_id"],
            "figure_input_manifest_sha256": sha256_file(figure_inputs_path),
            "figure_input_assembly_identity_sha256": assembly_identity_sha256,
            "figure_input_provenance_receipts": figure_input_provenance,
            "train399_prediction_input_provenance": manifest[
                "train399_prediction_input_provenance"
            ],
            "measurement_assurance_receipt_sha256": figure_input_provenance[
                "measurement_assurance"
            ]["sha256"],
            "measurement_assurance_identity_sha256": figure_input_provenance[
                "measurement_assurance"
            ]["identity_sha256"],
            "figure_resource_sha256": resource_hashes,
            "figure_source_data_sha256": source_data,
            "physical_source_data_sha256": source_data_files,
            "source_data_identity_sha256": source_data_identity_sha256,
            "multitrait_atlas_identity_sha256": multitrait_atlas_payload[
                "atlas_identity_sha256"
            ],
            "wt_secondary_evidence": wt_secondary_evidence,
            "overlay_image_sha256": overlay_source_hashes,
            "narrative_decision_identity_sha256": decision[
                "narrative_decision_identity_sha256"
            ],
            "narrative_branch_id": decision["branch_id"],
            "title_contract": locked_titles,
            "source_hashes_manifest_sha256": sha256_file(source_hash_path),
            "figures": figure_records,
            "figure_bundle_sha256": figure_hashes,
            "supplementary_figures": supplementary_records,
            "supplementary_figure_bundle_sha256": supplementary_bundle_hashes,
            "supplementary_figure_bundle_identity_sha256": supplementary_identity_sha256,
            "supplementary_figure_contract": supplementary_contract,
            "supplementary_figure_contract_identity_sha256": supplementary_contract[
                "contract_identity_sha256"
            ],
            "supplementary_tables": supplementary_table_bundle["items"],
            "supplementary_table_bundle_receipt": (
                Path(SUPPLEMENTARY_TABLE_DIRECTORY) / SUPPLEMENTARY_TABLE_RECEIPT
            ).as_posix(),
            "supplementary_table_bundle_receipt_sha256": supplementary_table_bundle[
                "receipt_sha256"
            ],
            "supplementary_table_bundle_identity_sha256": supplementary_table_bundle[
                "bundle_identity_sha256"
            ],
            "supplementary_table_bundle_sha256": supplementary_table_bundle[
                "bundle_file_sha256"
            ],
            "supplementary_table_source_input_sha256": supplementary_source_hashes,
            "supplementary_table_source_authority_sha256": supplementary_table_bundle[
                "source_authority_sha256"
            ],
            "supplementary_table_source_authority_identity": supplementary_table_bundle[
                "source_authority_identity"
            ],
            "legends_alt_text_sha256": sha256_file(legends_path),
            "claim_contract": {
                "main_figure_count": 6,
                "supplementary_figure_count": 9,
                "supplementary_table_data_resource_count": 10,
                "trait_count": 32,
                "root_cap_region_statistics_included": False,
                "root_cap_region_output": False,
                "condition_metadata_used_for_routing": False,
                "canonical_annotations_read": False,
                "qcdevelopment44_independent_test_claim": False,
                "clean261_accuracy_claim": False,
                "provider_equivalence_is_accuracy_claim": False,
                "profile_hypothesis_tests_added": False,
                "profiles_select_or_veto_narrative_branch": False,
                "wt_secondary_alters_D15_fixed_effect_family": False,
                "wt_cross_day_pooling_performed": False,
                "wt_unknown_day_meta_analysis_performed": False,
                "wt_clean_full_pooling_performed": False,
                "narrative_decision_identity_sha256": decision[
                    "narrative_decision_identity_sha256"
                ],
            },
            "blind_images_used": 0,
        }
        summary["figure_suite_identity_sha256"] = sha256_json(
            figure_suite_identity_preimage(
                status="final" if final else "provisional",
                figure_hashes=figure_hashes,
                source_hashes=source_hashes,
                figure_input_assembly_identity_sha256=assembly_identity_sha256,
                model_contract_proposal_identity_sha256=proposal_identity_sha256,
                model_contract_public_identity=manifest[
                    "model_contract_public_identity"
                ],
                train399_prediction_input_provenance=manifest[
                    "train399_prediction_input_provenance"
                ],
                supplementary_table_bundle_identity_sha256=(
                    supplementary_table_bundle["bundle_identity_sha256"]
                ),
                supplementary_table_bundle_receipt_sha256=(
                    supplementary_table_bundle["receipt_sha256"]
                ),
            )
        )
        atomic_write_json(staging / "figure_assembly_summary.json", summary)
        os.replace(staging, output_path)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("final", "provisional"), required=True)
    parser.add_argument("--figure-inputs", type=Path, required=True, help="explicit hash-locked figure-input manifest")
    parser.add_argument("--model-contract-proposal", type=Path, required=True, help="validated, sealed, unapplied model-contract proposal")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; must not exist")
    for role in RECEIPT_ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}", dest=role, type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_figure_suite(
        mode=args.mode,
        figure_inputs=args.figure_inputs,
        model_contract_proposal=args.model_contract_proposal,
        output=args.output,
        receipt_paths=_receipt_paths(args),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
