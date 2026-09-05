"""Deterministic, fail-closed PHAxis source-release construction.

The existing ``release/PHAxis_V1_0_Source_20260828`` tree is a historical
snapshot.  This module deliberately never reads from that directory.  Every
copied byte comes from a small, path-bounded allowlist in the project working
tree; release-only metadata is generated here.

Release gating imports only the PHAxis pure-contract/identity helpers in
addition to Python's standard library. Those helpers are intentionally free of
Torch/CUDA and never deserialize model checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

from phaxis import _toml_compat as tomllib
from phaxis.public_identity import (
    MODEL_BUNDLE_PREFIX,
    ROOT_EXPERT_PREFIX,
    ROOT_PROVIDER_ROLE,
    validate_proposal_public_identity,
)
from phaxis.errors import ContractError


SCHEMA_VERSION = "PHAxis-source-release-manifest-2.0"
GATE_SCHEMA_VERSION = "PHAxis-source-release-gate-1.0"
RELEASE_VERSION = "1.0.0"
MANIFEST_NAME = "SOURCE_MANIFEST.json"
BLOCKED_RECEIPT_NAME = "BLOCKED_DEVELOPMENT_STAGING_DO_NOT_RELEASE.json"
FORMAL_RECEIPT_NAME = "FORMAL_RELEASE_GATE_RECEIPT.json"
RELEASE_HUMAN_METADATA_NAME = "RELEASE_HUMAN_METADATA.json"
RELEASE_HUMAN_METADATA_TEMPLATE_NAME = "RELEASE_HUMAN_METADATA_TEMPLATE.json"
RELEASE_HUMAN_METADATA_SCHEMA = "PHAxis-release-human-metadata-1.3"
SOFTWARE_CITATION_TITLE = (
    "PHAxis: physically calibrated primary-root and root-hair phenomics"
)
SBOM_NAME = "SBOM.cdx.json"
THIRD_PARTY_INVENTORY_NAME = "THIRD_PARTY_LICENSES.json"
THIRD_PARTY_NOTICES_NAME = "THIRD_PARTY_NOTICES.md"
ZENODO_METADATA_NAME = ".zenodo.json"
FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z; content is intentionally timeless.
SOURCE_RELEASE_STAGING_PREFIX = ".source-release-staging-"
GITHUB_ACTION_PINS = {
    "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "actions/download-artifact": "d3f86a106a0bac45b974a628896c90dbdf5c8093",
    "pypa/gh-action-pypi-publish": "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
}
LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256 = (
    "ede309b8a828aec35be64d9f8afbc2ac9bf92b5a9e1b1b262d5acf603a746f36"
)
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

PROHIBITED_SUFFIXES = {
    ".bmp",
    ".ckpt",
    ".engine",
    ".gif",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
    ".webp",
}
PROHIBITED_PATH_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "blind",
    "build",
    "data",
    "dist",
    "legacy_project",
    "predictions",
    "weights",
}


@dataclass(frozen=True)
class TreeRule:
    """One path-bounded allowlist rule."""

    source: str
    destination: str
    suffixes: tuple[str, ...]


@dataclass(frozen=True)
class ProjectedEvidenceSpec:
    """One exact release-safe JSON authority projection."""

    source: str
    destination: str
    evidence_role: str
    validator_kind: str
    expected_source_receipt_sha256: str | None = None
    expected_payload_identity_field: str | None = None
    expected_payload_identity_sha256: str | None = None
    expected_projected_payload_identity_sha256: str | None = None


@dataclass(frozen=True)
class ThirdPartyDependency:
    """One direct, declared dependency and its upstream license authority."""

    name: str
    requirement: str
    scopes: tuple[str, ...]
    license_expression: str
    project_url: str
    license_note: str = ""


# This is a direct-dependency declaration inventory, not a claim about the
# transitive closure of a future resolver run.  Every row is cross-checked
# against the generated pyproject.toml.  Exact resolved artifacts remain
# hash-locked by the offline dependency stage.
THIRD_PARTY_DEPENDENCIES: tuple[ThirdPartyDependency, ...] = (
    ThirdPartyDependency("build", "build>=1.2,<2", ("build",), "MIT", "https://pypi.org/project/build/"),
    ThirdPartyDependency("imageio", "imageio>=2.35,<3", ("deployment",), "BSD-2-Clause", "https://pypi.org/project/imageio/"),
    ThirdPartyDependency("joblib", "joblib>=1.4,<2", ("deployment",), "BSD-3-Clause", "https://pypi.org/project/joblib/"),
    ThirdPartyDependency("matplotlib", "matplotlib>=3.8,<4", ("analysis", "deployment", "test"), "PSF-2.0", "https://pypi.org/project/matplotlib/"),
    ThirdPartyDependency("numpy", "numpy>=1.26,<3", ("core",), "BSD-3-Clause", "https://pypi.org/project/numpy/"),
    ThirdPartyDependency(
        "opencv-python-headless",
        "opencv-python-headless>=4.9,<6",
        ("analysis", "deployment", "inference", "test", "visualization"),
        "LicenseRef-opencv-python-headless-wheel-multiple",
        "https://pypi.org/project/opencv-python-headless/",
        (
            "Upstream reports the wrapper as MIT and OpenCV as Apache-2.0; "
            "published wheels also bundle FFmpeg under LGPL-2.1 and may contain "
            "additional artifact-specific notices. Preserve the wheel's "
            "LICENSE-3RD-PARTY.txt and audit the exact locked artifact."
        ),
    ),
    ThirdPartyDependency("packaging", "packaging>=24,<26", ("core",), "Apache-2.0 OR BSD-2-Clause", "https://pypi.org/project/packaging/"),
    ThirdPartyDependency("pandas", "pandas>=2.2,<4", ("analysis", "deployment", "test"), "BSD-3-Clause", "https://pypi.org/project/pandas/"),
    ThirdPartyDependency("Pillow", "Pillow>=10,<13", ("analysis", "deployment", "inference", "publication", "test", "visualization"), "MIT-CMU", "https://pypi.org/project/Pillow/"),
    ThirdPartyDependency("pytest", "pytest>=8,<10", ("test",), "MIT", "https://pypi.org/project/pytest/"),
    ThirdPartyDependency("python-docx", "python-docx>=1.1,<2", ("publication", "test"), "MIT", "https://pypi.org/project/python-docx/"),
    ThirdPartyDependency("scikit-image", "scikit-image>=0.24,<0.27", ("analysis", "deployment", "test"), "BSD-3-Clause", "https://pypi.org/project/scikit-image/"),
    ThirdPartyDependency("scikit-learn", "scikit-learn>=1.5,<2", ("deployment",), "BSD-3-Clause", "https://pypi.org/project/scikit-learn/"),
    ThirdPartyDependency("scipy", "scipy>=1.11,<2", ("core",), "BSD-3-Clause", "https://pypi.org/project/scipy/"),
    ThirdPartyDependency("setuptools", "setuptools>=77", ("build-system",), "MIT", "https://pypi.org/project/setuptools/"),
    ThirdPartyDependency("statsmodels", "statsmodels>=0.14,<1", ("analysis", "deployment", "test"), "BSD-3-Clause", "https://pypi.org/project/statsmodels/"),
    ThirdPartyDependency("tifffile", "tifffile>=2024.8,<2027", ("analysis", "deployment", "inference", "test"), "BSD-3-Clause", "https://pypi.org/project/tifffile/"),
    ThirdPartyDependency("timm", "timm>=1.0.28,<2", ("deployment", "inference", "test"), "Apache-2.0", "https://pypi.org/project/timm/"),
    ThirdPartyDependency("torch", "torch>=2.6,<3", ("deployment", "inference", "test"), "BSD-3-Clause", "https://pypi.org/project/torch/"),
    ThirdPartyDependency("torchvision", "torchvision>=0.21,<1", ("deployment", "inference", "test"), "BSD-3-Clause", "https://pypi.org/project/torchvision/"),
    ThirdPartyDependency("twine", "twine>=6,<7", ("build",), "Apache-2.0", "https://pypi.org/project/twine/"),
    ThirdPartyDependency("wheel", "wheel>=0.45", ("build-system",), "MIT", "https://pypi.org/project/wheel/"),
)

# Tomli is vendored rather than declared as an external Python 3.10 runtime
# dependency.  These rows are the byte authority for the unmodified pure-Python
# files from Tomli 2.4.0.  Keeping the hashes here makes an intentional vendor
# upgrade explicit and lets both the source builder and standalone verifier
# reject a re-sealed but altered vendored component.
VENDORED_TOMLI_FILES: tuple[Mapping[str, Any], ...] = (
    {
        "path": "src/phaxis/_vendor/tomli/LICENSE.txt",
        "bytes": 1072,
        "sha256": "b80816b0d530b8accb4c2211783790984a6e3b61922c2b5ee92f3372ab2742fe",
        "role": "license_text",
    },
    {
        "path": "src/phaxis/_vendor/tomli/__init__.py",
        "bytes": 314,
        "sha256": "6a1b438c6240d8cff0595bc6a73c78609b56c6b581c7aa84f861f9f946281020",
        "role": "vendored_source",
    },
    {
        "path": "src/phaxis/_vendor/tomli/_parser.py",
        "bytes": 25958,
        "sha256": "b717804cb137cc7c99faeb215ed61fad9dcba08b3b273405d96d8a2f583024f8",
        "role": "vendored_source",
    },
    {
        "path": "src/phaxis/_vendor/tomli/_re.py",
        "bytes": 3396,
        "sha256": "a12359fe294523a72112e434d58452a14c9d050affa2417f9927474e4166bfdd",
        "role": "vendored_source",
    },
    {
        "path": "src/phaxis/_vendor/tomli/_types.py",
        "bytes": 254,
        "sha256": "f864c6d9552a929c7032ace654ee05ef26ca75d21b027b801d77e65907138b74",
        "role": "vendored_source",
    },
    {
        "path": "src/phaxis/_vendor/tomli/py.typed",
        "bytes": 26,
        "sha256": "f0f8f2675695a10a5156fb7bd66bafbaae6a13e8d315990af862c792175e6e67",
        "role": "typing_marker",
    },
)
VENDORED_TOMLI_COMPONENT: Mapping[str, Any] = {
    "name": "tomli",
    "version": "2.4.0",
    "purl": "pkg:pypi/tomli@2.4.0",
    "project_url": "https://pypi.org/project/tomli/2.4.0/",
    "package_path": "src/phaxis/_vendor/tomli",
    "license_expression": "MIT",
    "license_text_path": "src/phaxis/_vendor/tomli/LICENSE.txt",
    "relationship": "vendored_source_no_site_fallback",
    "runtime_scope": "python_version < '3.11' including isolated no-site verification",
}


# These rules are intentionally narrow.  A new top-level project folder never
# enters a source release merely because it exists.  Within the PHAxis package,
# contracts and tests, new source files of the declared type are authoritative
# and should not silently go stale in a hand-maintained snapshot.
TREE_RULES: tuple[TreeRule, ...] = (
    TreeRule("src/phaxis", "src/phaxis", (".py",)),
    TreeRule("configs/phaxis/v1_0", "configs/phaxis/v1_0", (".json",)),
    TreeRule("tests/phaxis", "tests/phaxis", (".py",)),
    TreeRule("tests/phaxis/fixtures", "tests/phaxis/fixtures", (".json",)),
)

DOCUMENT_FILES: tuple[str, ...] = (
    "docs/phaxis/PHAXIS_BIOLOGICAL_ACQUISITION_METADATA_COMPLETION_CN_20260829.md",
    "docs/phaxis/PHAXIS_GITHUB_PYPI_RELEASE_GUIDE_CN_20260828.md",
    "docs/phaxis/PHAXIS_MANUSCRIPT_FIGURE_INPUT_CONTRACT_20260828.md",
    "docs/phaxis/PHAXIS_MANUSCRIPT_REFERENCE_AUDIT_20260828.md",
    "docs/phaxis/PHAXIS_MEASUREMENT_ASSURANCE_CONTRACT_CN_20260829.md",
    "docs/phaxis/PHAXIS_SUBMISSION_DOCX_LAYOUT_QA_20260828.md",
    "docs/phaxis/PHAXIS_SUPPLEMENTARY_DOCX_LAYOUT_QA_20260829.md",
    "docs/phaxis/PHAXIS_MANUSCRIPT_VALUES_COMPILER_CONTRACT_20260828.md",
    "docs/phaxis/PHAXIS_REUSE_HANDOVER_PACKAGE_CONTRACT_CN_20260828.md",
    "docs/phaxis/PHAXIS_ROOT_HAIR_BIOLOGICAL_METRIC_PROTOCOL_20260828.md",
    "docs/phaxis/PHAXIS_VERSION_IDENTITY_POLICY_20260828.md",
    "docs/phaxis/TRAIT_CONTRACT_CN.md",
    "docs/phaxis/USER_GUIDE.md",
)

# These are compilation inputs for the later manuscript-value stages, not public
# source-distribution documentation.  They contain deliberate ``{{TOKEN}}``
# slots until stage 50 and therefore must never be copied into the stage-40
# GitHub/PyPI source authority.
UNCOMPILED_MANUSCRIPT_FILES: tuple[str, ...] = (
    "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md",
    "docs/phaxis/PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md",
)

# Historical OOF/QC scripts that import an external RHAxiscc checkout and the
# frozen Hybrid-Max analysis implementation are deliberately absent.  The
# included analysis wrapper is the PHAxis-owned equivalent.  The formal
# train399 evaluator is also scanned to prevent reintroducing external source.
SCRIPT_FILES: tuple[str, ...] = (
    "scripts/phaxis/analyze_biological_cohorts.py",
    "scripts/phaxis/analyze_distal_axis_profiles.py",
    "scripts/phaxis/assemble_post_training_release_manifest.py",
    "scripts/phaxis/assemble_handover_build_contract.py",
    "scripts/phaxis/audit_biological_analysis_equivalence.py",
    "scripts/phaxis/audit_stage22_h11_raw_median_amendment.py",
    "scripts/phaxis/audit_root_provider_reference283.py",
    "scripts/phaxis/benchmark_full_workflow.py",
    "scripts/phaxis/benchmark_stageb_shared_input.py",
    "scripts/phaxis/build_analysis_workflow_manifest.py",
    "scripts/phaxis/build_benchmark_artifact_inventory.py",
    "scripts/phaxis/build_biological_cohorts.py",
    "scripts/phaxis/build_clean_install_expected_identity.py",
    "scripts/phaxis/build_clean_install_sample_manifest.py",
    "scripts/phaxis/build_condition_blinded_overlay_evidence.py",
    "scripts/phaxis/build_clean_install_verification.py",
    "scripts/phaxis/build_direct_benchmark_provider_descriptor.py",
    "scripts/phaxis/build_historical_oof443_publication_evidence.py",
    "scripts/phaxis/build_handover_benchmark_manifest.py",
    "scripts/phaxis/build_handover_dataset_manifest.py",
    "scripts/phaxis/build_handover_image_manifest.py",
    "scripts/phaxis/build_handover_model_asset_manifest.py",
    "scripts/phaxis/build_handover_model_source_manifest.py",
    "scripts/phaxis/build_handover_package.py",
    "scripts/phaxis/build_humancurated_inference_manifest.py",
    "scripts/phaxis/build_post_training_release_stage_contract.py",
    "scripts/phaxis/build_manuscript_evidence_manifest.py",
    "scripts/phaxis/build_paper_first_release_evidence_graph.py",
    "scripts/phaxis/build_manuscript_values.py",
    "scripts/phaxis/build_submission_docx.py",
    "scripts/phaxis/build_supplementary_docx.py",
    "scripts/phaxis/build_supplementary_table_data_bundle.py",
    "scripts/phaxis/build_measurement_assurance_evidence.py",
    "scripts/phaxis/build_production_manifest.py",
    "scripts/phaxis/build_publication_figure_inputs.py",
    "scripts/phaxis/build_publication_figures.py",
    "scripts/phaxis/build_qcdev44_root_provider_inputs.py",
    "scripts/phaxis/build_release_case_prelocks.py",
    "scripts/phaxis/build_release_distributions.py",
    "scripts/phaxis/build_root_provider_bundle.py",
    "scripts/phaxis/build_root_provider_reference283.py",
    "scripts/phaxis/build_stageb_train399_candidate_bundle.py",
    "scripts/phaxis/build_stageb_train399_qcdev44_candidate_pool.py",
    "scripts/phaxis/build_source_release.py",
    "scripts/phaxis/check_post_training_release_topology.py",
    "scripts/phaxis/compile_manuscript.py",
    "scripts/phaxis/compile_supplementary_manuscript.py",
    "scripts/phaxis/compare_hair_experts_biological_presence.py",
    "scripts/phaxis/evaluate_stageb_train399_qcdev44.py",
    "scripts/phaxis/export_cohort_distal_axis_profiles.py",
    "scripts/phaxis/export_distal_axis_profiles.py",
    "scripts/phaxis/export_traits.py",
    "scripts/phaxis/promote_model_contract.py",
    "scripts/phaxis/render_distal_axis_profile_figure.py",
    "scripts/phaxis/render_overlays.py",
    "scripts/phaxis/run_external_direct_benchmark.py",
    "scripts/phaxis/run_root_provider.py",
    "scripts/phaxis/run_post_training_release.py",
    "scripts/phaxis/run_stageb_evaluation_inference.py",
    "scripts/phaxis/run_stageb_inference.py",
    "scripts/phaxis/run_stageb_train399_gpu_queue.ps1",
    "scripts/phaxis/select_stageb_train399_operating_point.py",
    "scripts/phaxis/source_release_common.py",
    "scripts/phaxis/handover_package_common.py",
    "scripts/phaxis/handover_manifest_producers.py",
    "scripts/phaxis/materialize_figure1_geometry.py",
    "scripts/phaxis/materialize_verified_root_provider_bundle.py",
    "scripts/phaxis/materialize_offline_dependencies.py",
    "scripts/phaxis/render_docx_with_word_com_windows.ps1",
    "scripts/phaxis/render_manuscript_bundle.py",
    "scripts/phaxis/run_cli.py",
    "scripts/phaxis/train_stageb_train399.py",
    "scripts/phaxis/verify_root_provider_bundle.py",
    "scripts/phaxis/verify_manuscript_artifacts.py",
    "scripts/phaxis/validate_manuscript_visual_qa.py",
    "scripts/phaxis/verify_handover_package.py",
    "scripts/phaxis/verify_source_release.py",
)

PUBLIC_CARD_FILES: tuple[str, ...] = ("MODEL_CARD.md", "DATA_CARD.md")

SINGLE_FILES: tuple[str, ...] = (
    (
        "LICENSE",
        "configs/phaxis/v1_0/locked_qcdevelopment44_ids.txt",
        "src/phaxis/_vendor/tomli/LICENSE.txt",
        "src/phaxis/_vendor/tomli/py.typed",
    )
    + PUBLIC_CARD_FILES
    + DOCUMENT_FILES
    + SCRIPT_FILES
)

MAPPED_FILES: tuple[tuple[str, str], ...] = (
    (
        "outputs/phaxis_rhaxiscc_metric_parity_20260828/audit.json",
        "evidence/evaluation_metric_parity_audit.json",
    ),
    (
        "models/phaxis_stageb_train399_v1_0_20260828/"
        "AMP_BACKWARD_RETRY_AMENDMENT_20260829.json",
        "evidence/stageb_amp_backward_retry_amendment.json",
    ),
)

AMP_AMENDMENT_SOURCE = MAPPED_FILES[1][0]
AMP_AMENDMENT_DESTINATION = MAPPED_FILES[1][1]

# The H11 amendment is a deliberately narrow, immutable statistical authority.
# Both anchors must remain available inside a standalone source distribution:
# its verifier cannot rely on the mixed development workspace to distinguish an
# authentic authority from a semantically plausible payload whose internal
# identity and release manifest were recomputed by an attacker.
H11_RAW_MEDIAN_AMENDMENT_AUTHORITY_SHA256 = (
    "82570646cc28357e0a48b5c333ac9c978da76521695dfd643d6d103196393896"
)
H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256 = (
    "378b19e9b44d2bc563eee5bf4a8b864094bac803a0e5e9ee30d0fe99ece337bd"
)
H11_HISTORICAL_COHORT_ANALYSIS_CONTRACT_SHA256 = (
    "aaf5fb794986e705c6245217f88f67d5459b476f965a6185f12123cefd3625bf"
)
PRE_AMENDMENT_EQUIVALENCE_AUTHORITY_SHA256 = (
    "10314999405b66cd5f1b6042cfc18bf6e3625e64b2aeb6ee9d7d1e04ef7137e7"
)
PRE_AMENDMENT_EQUIVALENCE_PROJECTED_IDENTITY_SHA256 = (
    "d8353acf6416dd5b0100b59ab50a5008742b676b9bf55359c9ba47ee45db252c"
)

PROJECTED_EVIDENCE_SPECS: tuple[ProjectedEvidenceSpec, ...] = (
    ProjectedEvidenceSpec(
        source=(
            "outputs/phaxis_biological_analysis_native_modelspec_audit_final_"
            "20260828/equivalence_audit.json"
        ),
        destination="evidence/biological_analysis_equivalence_audit.json",
        evidence_role=(
            "pre_amendment_biological_equivalence_historical_baseline"
        ),
        validator_kind="pre_amendment_biological_equivalence",
        expected_source_receipt_sha256=(
            PRE_AMENDMENT_EQUIVALENCE_AUTHORITY_SHA256
        ),
        expected_projected_payload_identity_sha256=(
            PRE_AMENDMENT_EQUIVALENCE_PROJECTED_IDENTITY_SHA256
        ),
    ),
    ProjectedEvidenceSpec(
        source=(
            "outputs/phaxis_stage22_H11_raw_median_gap_audit_r4_20260831/"
            "amendment_audit.json"
        ),
        destination="evidence/h11_raw_median_amendment_audit.json",
        evidence_role="h11_raw_median_contract_amendment_current",
        validator_kind="h11_raw_median_amendment",
        expected_source_receipt_sha256=(
            H11_RAW_MEDIAN_AMENDMENT_AUTHORITY_SHA256
        ),
        expected_payload_identity_field="amendment_audit_identity_sha256",
        expected_payload_identity_sha256=(
            H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256
        ),
    ),
)

if (
    len(PROJECTED_EVIDENCE_SPECS) != 2
    or len({spec.source for spec in PROJECTED_EVIDENCE_SPECS}) != 2
    or len({spec.destination for spec in PROJECTED_EVIDENCE_SPECS}) != 2
    or len({spec.evidence_role for spec in PROJECTED_EVIDENCE_SPECS}) != 2
    or any(
        spec.expected_source_receipt_sha256 is not None
        and re.fullmatch(
            r"[0-9a-f]{64}", spec.expected_source_receipt_sha256
        )
        is None
        for spec in PROJECTED_EVIDENCE_SPECS
    )
    or any(
        (spec.expected_payload_identity_field is None)
        != (spec.expected_payload_identity_sha256 is None)
        or (
            spec.expected_payload_identity_sha256 is not None
            and re.fullmatch(
                r"[0-9a-f]{64}", spec.expected_payload_identity_sha256
            )
            is None
        )
        for spec in PROJECTED_EVIDENCE_SPECS
    )
    or any(
        spec.expected_projected_payload_identity_sha256 is not None
        and re.fullmatch(
            r"[0-9a-f]{64}",
            spec.expected_projected_payload_identity_sha256,
        )
        is None
        for spec in PROJECTED_EVIDENCE_SPECS
    )
):
    raise RuntimeError("projected evidence specification is not an exact two-role set")

# Compatibility view for callers that only need source/destination pairs.
PROJECTED_JSON_FILES: tuple[tuple[str, str], ...] = tuple(
    (spec.source, spec.destination) for spec in PROJECTED_EVIDENCE_SPECS
)

WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
)
USER_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:Users|data|home)/[^/\s]+/"
)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PROJECT_URL_LABELS = ("Homepage", "Repository", "Issues", "Documentation")


class SourceReleaseError(RuntimeError):
    """A source-release policy, gate, or integrity check failed."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _biological_presence_matcher_contract_ok(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and dict(payload) == BIOLOGICAL_PRESENCE_MATCHER_CONTRACT
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pre_amendment_biological_equivalence_receipt_ok(
    payload: Mapping[str, Any],
) -> bool:
    if (
        "release_projection" in payload
        and isinstance(payload.get("release_projection"), Mapping)
        and payload["release_projection"].get("evidence_role")
        == PROJECTED_EVIDENCE_SPECS[0].evidence_role
        and not _release_projection_metadata_ok(
            PROJECTED_EVIDENCE_SPECS[0], payload
        )
    ):
        return False
    tables = payload.get("tables")
    return bool(
        payload.get("schema_version")
        == "PHAxis-biological-analysis-native-equivalence-audit-1.0"
        and payload.get("status") == "passed"
        and _exact_int(payload.get("blind_images_used"), 0)
        and payload.get("canonical_annotations_read") is False
        and payload.get("production_wrapper_imports_frozen_predecessor") is False
        and payload.get("tables_equivalent") is True
        and payload.get("tables_byte_identical") is True
        and _exact_int(payload.get("total_differing_cells"), 0)
        and isinstance(tables, dict)
        and len(tables) == 6
        and all(
            isinstance(record, dict)
            and record.get("equivalent") is True
            and record.get("byte_identical") is True
            for record in tables.values()
        )
    )


def _biological_equivalence_receipt_ok(payload: Mapping[str, Any]) -> bool:
    """Backward-compatible alias for the historical pre-amendment authority."""

    return _pre_amendment_biological_equivalence_receipt_ok(payload)


def _finite_real(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _exact_int(value: Any, expected: int) -> bool:
    """Reject JSON booleans where an exact integer contract is required."""

    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == expected
    )


def _h11_raw_median_amendment_receipt_ok(payload: Mapping[str, Any]) -> bool:
    """Validate the narrow, identity-sealed current H11 amendment authority."""

    if (
        "release_projection" in payload
        and isinstance(payload.get("release_projection"), Mapping)
        and payload["release_projection"].get("evidence_role")
        == PROJECTED_EVIDENCE_SPECS[1].evidence_role
        and not _release_projection_metadata_ok(
            PROJECTED_EVIDENCE_SPECS[1], payload
        )
    ):
        return False
    identity = payload.get("amendment_audit_identity_sha256")
    if identity != H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256:
        return False
    unsigned = dict(payload)
    unsigned.pop("amendment_audit_identity_sha256", None)
    unsigned.pop("release_projection", None)
    try:
        if _sha256_json(unsigned) != identity:
            return False
    except (TypeError, ValueError):
        return False

    expected_h11_tables = {
        "primary_clean_exploratory_factorial_tests.csv": {
            "n": 37,
            "cell_counts": {"EV22": 3, "EV30": 8, "OE22": 17, "OE30": 9},
        },
        "full283_sensitivity_factorial_tests.csv": {
            "n": 49,
            "cell_counts": {"EV22": 6, "EV30": 12, "OE22": 19, "OE30": 12},
        },
    }
    unchanged_tables = {
        "clean_vs_full_effect_stability.csv",
        "robust_sensitivity.csv",
        "primary_group_summaries.csv",
        "primary_model_qc_flow.csv",
    }
    expected_effects = {
        "construct_OE_minus_EV",
        "temperature_30C_minus_22C",
        "construct_by_temperature_interaction",
    }
    baseline = payload.get("pre_amendment_baseline")
    locked = payload.get("locked_inputs")
    implementation = payload.get("implementation_sha256")
    change = payload.get("change_contract")
    tables = payload.get("tables")
    h11_summary = payload.get("H11_raw_median_companion")
    if not (
        payload.get("schema_version")
        == "PHAxis-stage22-H11-raw-median-amendment-audit-1.0"
        and payload.get("status") == "passed"
        and payload.get("artifact_role")
        == "h11_raw_median_contract_amendment_current"
        and payload.get("protected_primary_inference_equivalent") is True
        and payload.get("unaffected_tables_byte_identical") is True
        and payload.get("non_h11_existing_fields_equivalent") is True
        and payload.get("candidate_schema_extension_exact") is True
        and _exact_int(payload.get("unauthorized_differing_cells"), 0)
        and payload.get("separate_hypothesis_test_added") is False
        and _exact_int(payload.get("new_hypothesis_tests_added"), 0)
        and payload.get("D15_fixed_effect_family_changed") is False
        and _exact_int(payload.get("gpu_programs_started"), 0)
        and payload.get("canonical_annotations_read") is False
        and payload.get("condition_metadata_used_for_routing") is False
        and payload.get("root_cap_region_statistics_included") is False
        and _exact_int(payload.get("blind_images_used"), 0)
        and isinstance(baseline, Mapping)
        and baseline.get("authority_path")
        == (
            "outputs/phaxis_biological_analysis_native_modelspec_audit_final_"
            "20260828/equivalence_audit.json"
        )
        and baseline.get("authority_sha256")
        == "10314999405b66cd5f1b6042cfc18bf6e3625e64b2aeb6ee9d7d1e04ef7137e7"
        and baseline.get("authority_schema_version")
        == "PHAxis-biological-analysis-native-equivalence-audit-1.0"
        and baseline.get("authority_status") == "passed"
        and baseline.get("candidate_analysis")
        == "outputs/phaxis_biological_analysis_native_modelspec_candidate_20260828"
        and isinstance(locked, Mapping)
        and set(locked)
        == {
            "analysis_contract_sha256",
            "cohort_lock_sha256",
            "cohort_summary_sha256",
            "model_spec_sha256",
            "primary_traits_sha256",
            "sensitivity_traits_sha256",
        }
        and all(_is_sha256(value) for value in locked.values())
        # Frozen cohort-generation authority from the historical baseline;
        # this is not the later workspace analysis contract.
        and locked.get("analysis_contract_sha256")
        == H11_HISTORICAL_COHORT_ANALYSIS_CONTRACT_SHA256
        and isinstance(implementation, Mapping)
        and set(implementation)
        == {
            "audit_producer",
            "audit_test",
            "biological_analysis",
            "biological_analysis_wrapper",
            "multitrait_atlas",
            "publication_figure_input_builder",
        }
        and all(_is_sha256(value) for value in implementation.values())
        and isinstance(change, Mapping)
        and change.get("endpoint")
        == "local_median_hair_length_um_1_4mm"
        and change.get("raw_effect_estimand")
        == "equal_margin_2x2_factorial_cell_raw_median_difference"
        and change.get("raw_effect_interval_method")
        == (
            "source_root_within_cell_stratified_bootstrap_percentile_"
            "2p5_97p5_numpy_linear"
        )
        and _exact_int(change.get("bootstrap_replicates"), 5000)
        and _exact_int(change.get("base_seed"), 20260823)
        and _exact_int(change.get("effective_seed"), 20271264)
        and change.get("stable_seed_offset_token") == "raw_median_bootstrap"
        and change.get("source_unit") == "source_root"
        and change.get("source_root_identity_policy")
        == "task_id_preferred_then_source_image_sha256_fallback"
        and change.get("cell_summary") == "median"
        and change.get("percentile_interval") == [0.025, 0.975]
        and change.get("numpy_quantile_method") == "linear"
        and change.get("construct_effect")
        == (
            "0.5*((median_OE_22C-median_EV_22C)+"
            "(median_OE_30C-median_EV_30C))"
        )
        and change.get("temperature_effect")
        == (
            "0.5*((median_EV_30C-median_EV_22C)+"
            "(median_OE_30C-median_OE_22C))"
        )
        and change.get("interaction_effect")
        == (
            "(median_OE_30C-median_OE_22C)-"
            "(median_EV_30C-median_EV_22C)"
        )
        and change.get("changed_existing_columns_whitelist")
        == [
            "raw_effect_estimate",
            "raw_effect_ci95_low",
            "raw_effect_ci95_high",
            "standardized_effect",
            "standardized_ci95_low",
            "standardized_ci95_high",
        ]
        and change.get("added_provenance_columns")
        == [
            "raw_effect_estimand",
            "raw_effect_interval_method",
            "raw_effect_bootstrap_replicates",
            "raw_effect_bootstrap_seed",
        ]
        and change.get("separate_hypothesis_test_added") is False
        and change.get("D15_fixed_effect_family_changed") is False
        and isinstance(h11_summary, Mapping)
        and h11_summary.get("validated") is True
        and h11_summary.get("cohort_tables")
        == list(expected_h11_tables)
        and _exact_int(h11_summary.get("effect_rows"), 6)
        and h11_summary.get("independent_point_and_interval_recomputation")
        is True
        and isinstance(tables, Mapping)
        and set(tables) == set(expected_h11_tables) | unchanged_tables
    ):
        return False

    effect_value_fields = {
        "historical_raw_mean_contrast",
        "raw_effect_estimate",
        "raw_effect_ci95_low",
        "raw_effect_ci95_high",
        "standardized_effect",
        "standardized_ci95_low",
        "standardized_ci95_high",
    }
    for table_name, expected in expected_h11_tables.items():
        record = tables.get(table_name)
        if not isinstance(record, Mapping):
            return False
        h11 = record.get("H11")
        if not (
            record.get("policy")
            == "protected_exact_with_independently_validated_H11_companion_amendment"
            and record.get("passed") is True
            and _exact_int(record.get("rows"), 15)
            and record.get("row_identity_exact") is True
            and record.get("candidate_schema_extension_exact") is True
            and record.get("protected_primary_inference_exact") is True
            and _exact_int(record.get("protected_differing_cells"), 0)
            and record.get("non_h11_existing_fields_exact") is True
            and _exact_int(record.get("non_h11_differing_cells"), 0)
            and _exact_int(record.get("allowed_h11_changed_cells"), 18)
            and _exact_int(record.get("unauthorized_differing_cells"), 0)
            and record.get("unauthorized_differences") == []
            and record.get("H11_raw_median_contract_exact") is True
            and record.get("H11_independent_numeric_recomputation_exact") is True
            and record.get("historical_H11_raw_point_was_mean_contrast") is True
            and record.get("non_H11_raw_mean_provenance_exact") is True
            and _is_sha256(record.get("baseline_sha256"))
            and _is_sha256(record.get("candidate_sha256"))
            and isinstance(h11, Mapping)
            and _exact_int(h11.get("n"), expected["n"])
            and _exact_int(h11.get("unique_source_roots"), expected["n"])
            and h11.get("source_root_identity_field") == "task_id"
            and h11.get("source_root_identity_policy")
            == "task_id_preferred_then_source_image_sha256_fallback"
            and h11.get("all_four_cells_nonempty") is True
            and h11.get("cell_counts") == expected["cell_counts"]
            and _exact_int(h11.get("bootstrap_replicates"), 5000)
            and _exact_int(h11.get("base_seed"), 20260823)
            and _exact_int(h11.get("effective_seed"), 20271264)
            and _finite_real(h11.get("sample_standard_deviation"))
            and float(h11["sample_standard_deviation"]) > 0
            and isinstance(h11.get("cell_medians"), Mapping)
            and set(h11["cell_medians"]) == {"EV22", "EV30", "OE22", "OE30"}
            and all(_finite_real(value) for value in h11["cell_medians"].values())
            and isinstance(h11.get("cell_means"), Mapping)
            and set(h11["cell_means"]) == {"EV22", "EV30", "OE22", "OE30"}
            and all(_finite_real(value) for value in h11["cell_means"].values())
            and isinstance(h11.get("effects"), Mapping)
            and set(h11["effects"]) == expected_effects
        ):
            return False
        sd = float(h11["sample_standard_deviation"])
        cell_medians = {
            cell: float(value) for cell, value in h11["cell_medians"].items()
        }
        cell_means = {
            cell: float(value) for cell, value in h11["cell_means"].items()
        }
        expected_raw_median_effects = {
            "construct_OE_minus_EV": 0.5
            * (
                (cell_medians["OE22"] - cell_medians["EV22"])
                + (cell_medians["OE30"] - cell_medians["EV30"])
            ),
            "temperature_30C_minus_22C": 0.5
            * (
                (cell_medians["EV30"] - cell_medians["EV22"])
                + (cell_medians["OE30"] - cell_medians["OE22"])
            ),
            "construct_by_temperature_interaction": (
                (cell_medians["OE30"] - cell_medians["OE22"])
                - (cell_medians["EV30"] - cell_medians["EV22"])
            ),
        }
        expected_historical_raw_mean_effects = {
            "construct_OE_minus_EV": 0.5
            * (
                (cell_means["OE22"] - cell_means["EV22"])
                + (cell_means["OE30"] - cell_means["EV30"])
            ),
            "temperature_30C_minus_22C": 0.5
            * (
                (cell_means["EV30"] - cell_means["EV22"])
                + (cell_means["OE30"] - cell_means["OE22"])
            ),
            "construct_by_temperature_interaction": (
                (cell_means["OE30"] - cell_means["OE22"])
                - (cell_means["EV30"] - cell_means["EV22"])
            ),
        }
        for effect in expected_effects:
            values = h11["effects"].get(effect)
            if not (
                isinstance(values, Mapping)
                and set(values) == effect_value_fields
                and all(_finite_real(value) for value in values.values())
                and float(values["raw_effect_ci95_low"])
                <= float(values["raw_effect_ci95_high"])
            ):
                return False
            if not math.isclose(
                float(values["raw_effect_estimate"]),
                expected_raw_median_effects[effect],
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(values["historical_raw_mean_contrast"]),
                expected_historical_raw_mean_effects[effect],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
            for raw_field, standardized_field in (
                ("raw_effect_estimate", "standardized_effect"),
                ("raw_effect_ci95_low", "standardized_ci95_low"),
                ("raw_effect_ci95_high", "standardized_ci95_high"),
            ):
                if not math.isclose(
                    float(values[raw_field]),
                    float(values[standardized_field]) * sd,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    return False

    for table_name in unchanged_tables:
        record = tables.get(table_name)
        if not (
            isinstance(record, Mapping)
            and record.get("policy") == "byte_identical"
            and record.get("passed") is True
            and record.get("byte_identical") is True
            and _exact_int(record.get("unauthorized_differing_cells"), 0)
            and _is_sha256(record.get("baseline_sha256"))
            and record.get("candidate_sha256") == record.get("baseline_sha256")
        ):
            return False
    return True


def _is_sha256(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value))


def _is_orcid(value: Any) -> bool:
    """Validate a canonical ORCID URI, including its ISO 7064 check digit."""

    if not isinstance(value, str):
        return False
    match = re.fullmatch(
        r"https://orcid\.org/(\d{4})-(\d{4})-(\d{4})-(\d{3}[\dX])",
        value,
    )
    if match is None:
        return False
    compact = "".join(match.groups())
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    result = (12 - total % 11) % 11
    expected = "X" if result == 10 else str(result)
    return compact[-1] == expected


def _is_release_doi(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", value, re.IGNORECASE)
    )


def _is_release_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value and parsed.year >= 2026


def _pretty_json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SourceReleaseError(f"required JSON receipt is absent: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceReleaseError(f"invalid JSON receipt: {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise SourceReleaseError(f"JSON receipt is not an object: {path.name}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("\r\n", "\n").replace("\r", "\n"))


def _posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_root_git_control_path(path: Path, root: Path) -> bool:
    """Identify only checkout metadata rooted exactly at ``root/.git``.

    GitHub Actions necessarily adds this control plane after checking out the
    authored, manifest-closed source tree. A nested ``docs/.git`` or any other
    unmanifested payload remains a hard failure.
    """

    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0].casefold() == ".git"


def _safe_relative(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SourceReleaseError(f"unsafe allowlisted relative path: {value!r}")
    return relative


def _destination(root: Path, relative: str) -> Path:
    parts = _safe_relative(relative).parts
    return root.joinpath(*parts)


def _validate_vendored_tomli(root: Path) -> None:
    """Reject any drift from the audited Tomli 2.4.0 pure-Python file set."""

    vendor_root = _destination(root, str(VENDORED_TOMLI_COMPONENT["package_path"]))
    if not vendor_root.is_dir() or vendor_root.is_symlink():
        raise SourceReleaseError("vendored Tomli package directory is absent or invalid")
    for record in VENDORED_TOMLI_FILES:
        relative = str(record["path"])
        path = _destination(root, relative)
        if not path.is_file() or path.is_symlink():
            raise SourceReleaseError(f"vendored Tomli file is absent or invalid: {relative}")
        if (
            path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise SourceReleaseError(
                f"vendored Tomli 2.4.0 file identity mismatch: {relative}"
            )
    expected_vendor_paths = {
        str(record["path"]) for record in VENDORED_TOMLI_FILES
    }
    actual_vendor_paths = {
        path.relative_to(root).as_posix()
        for path in vendor_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not PROHIBITED_PATH_PARTS.intersection(
            part.casefold() for part in path.relative_to(vendor_root).parts
        )
    }
    if actual_vendor_paths != expected_vendor_paths:
        missing = sorted(expected_vendor_paths - actual_vendor_paths)
        extra = sorted(actual_vendor_paths - expected_vendor_paths)
        raise SourceReleaseError(
            "vendored Tomli file closure differs from the audited 2.4.0 source "
            f"(missing={missing}, extra={extra})"
        )


def _absolute_host_path_markers(text: str) -> tuple[str, ...]:
    """Return content-level host-path classes present in UTF-8 release text."""

    markers: list[str] = []
    for label, pattern in (
        ("Windows drive", WINDOWS_ABSOLUTE_PATH_RE),
        ("user POSIX", USER_POSIX_ABSOLUTE_PATH_RE),
    ):
        if pattern.search(text):
            markers.append(label)
    return tuple(markers)


def _release_safe_json_projection(
    *, payload: Any, project_root: Path
) -> tuple[Any, int]:
    """Rebase in-project absolute paths and reject every other host path."""

    root_text = project_root.resolve().as_posix().rstrip("/")
    root_folded = root_text.casefold()
    rebased = 0

    def project(value: Any) -> Any:
        nonlocal rebased
        if isinstance(value, dict):
            return {str(key): project(item) for key, item in value.items()}
        if isinstance(value, list):
            return [project(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            folded = normalized.casefold()
            if folded == root_folded:
                rebased += 1
                return "project:."
            if folded.startswith(root_folded + "/"):
                rebased += 1
                return "project:" + normalized[len(root_text) + 1 :]
            markers = _absolute_host_path_markers(value)
            if markers:
                raise SourceReleaseError(
                    "authority JSON contains a host-absolute path outside the "
                    f"project root ({', '.join(markers)})"
                )
        return value

    return project(payload), rebased


def _release_projection_metadata_ok(
    spec: ProjectedEvidenceSpec,
    payload: Mapping[str, Any],
) -> bool:
    """Validate the exact, role-bound metadata of one projected authority."""

    metadata = payload.get("release_projection")
    observed_source_receipt_sha256 = (
        metadata.get("source_receipt_sha256")
        if isinstance(metadata, Mapping)
        else None
    )
    source_receipt_sha256_ok = bool(
        observed_source_receipt_sha256
        == spec.expected_source_receipt_sha256
        if spec.expected_source_receipt_sha256 is not None
        else _is_sha256(observed_source_receipt_sha256)
    )
    return bool(
        isinstance(metadata, dict)
        and set(metadata)
        == {
            "schema_version",
            "evidence_role",
            "source_path",
            "source_receipt_sha256",
            "rebased_project_absolute_paths",
            "policy",
        }
        and metadata.get("schema_version")
        == "PHAxis-release-safe-evidence-projection-1.0"
        and metadata.get("evidence_role") == spec.evidence_role
        and metadata.get("source_path") == spec.source
        and source_receipt_sha256_ok
        and isinstance(metadata.get("rebased_project_absolute_paths"), int)
        and not isinstance(metadata.get("rebased_project_absolute_paths"), bool)
        and metadata.get("rebased_project_absolute_paths") >= 0
        and metadata.get("policy")
        == "project_absolute_paths_to_project_relative_posix"
    )


def _projected_evidence_payload_ok(
    spec: ProjectedEvidenceSpec, payload: Mapping[str, Any]
) -> bool:
    if (
        "release_projection" in payload
        and not _release_projection_metadata_ok(spec, payload)
    ):
        return False
    if (
        spec.expected_payload_identity_field is not None
        and payload.get(spec.expected_payload_identity_field)
        != spec.expected_payload_identity_sha256
    ):
        return False
    if (
        spec.expected_projected_payload_identity_sha256 is not None
        and "release_projection" in payload
    ):
        unsigned_projection = dict(payload)
        unsigned_projection.pop("release_projection", None)
        try:
            if (
                _sha256_json(unsigned_projection)
                != spec.expected_projected_payload_identity_sha256
            ):
                return False
        except (TypeError, ValueError):
            return False
    if spec.validator_kind == "pre_amendment_biological_equivalence":
        return _pre_amendment_biological_equivalence_receipt_ok(payload)
    if spec.validator_kind == "h11_raw_median_amendment":
        return _h11_raw_median_amendment_receipt_ok(payload)
    return False


def _project_evidence_receipt(
    *, project_root: Path, source: Path, spec: ProjectedEvidenceSpec
) -> dict[str, Any]:
    """Create one deterministic, role-bound public evidence projection."""

    payload = _read_object(source)
    source_receipt_sha256 = _sha256_file(source)
    if (
        spec.expected_source_receipt_sha256 is not None
        and source_receipt_sha256 != spec.expected_source_receipt_sha256
    ):
        raise SourceReleaseError(
            f"{spec.evidence_role}: authority source SHA-256 differs from "
            "the immutable standalone release anchor"
        )
    if not _projected_evidence_payload_ok(spec, payload):
        raise SourceReleaseError(
            f"{spec.evidence_role}: authority is not a release-eligible pass"
        )
    projected, rebased = _release_safe_json_projection(
        payload=payload,
        project_root=project_root,
    )
    if not isinstance(projected, dict) or not _projected_evidence_payload_ok(
        spec, projected
    ):
        raise SourceReleaseError(
            f"{spec.evidence_role}: release-safe projection changed authority semantics"
        )
    if spec.expected_projected_payload_identity_sha256 is not None:
        try:
            projected_identity_sha256 = _sha256_json(projected)
        except (TypeError, ValueError) as error:
            raise SourceReleaseError(
                f"{spec.evidence_role}: release-safe projected payload is not canonical"
            ) from error
        if (
            projected_identity_sha256
            != spec.expected_projected_payload_identity_sha256
        ):
            raise SourceReleaseError(
                f"{spec.evidence_role}: release-safe projected payload identity "
                "differs from the immutable standalone release anchor"
            )
    source_relative = source.resolve().relative_to(project_root.resolve()).as_posix()
    if source_relative != spec.source:
        raise SourceReleaseError(
            f"{spec.evidence_role}: authority source path differs from exact spec"
        )
    projected["release_projection"] = {
        "schema_version": "PHAxis-release-safe-evidence-projection-1.0",
        "evidence_role": spec.evidence_role,
        "source_path": source_relative,
        "source_receipt_sha256": source_receipt_sha256,
        "rebased_project_absolute_paths": rebased,
        "policy": "project_absolute_paths_to_project_relative_posix",
    }
    if not _projected_evidence_payload_ok(spec, projected):
        raise SourceReleaseError(
            f"{spec.evidence_role}: projected receipt failed semantic validation"
        )
    return projected


def _project_biological_equivalence_receipt(
    *, project_root: Path, source: Path
) -> dict[str, Any]:
    """Compatibility wrapper for the historical baseline projection."""

    return _project_evidence_receipt(
        project_root=project_root,
        source=source,
        spec=PROJECTED_EVIDENCE_SPECS[0],
    )


def collect_allowlisted_sources(project_root: Path) -> list[tuple[Path, str]]:
    """Return sorted ``(authoritative source, release relative path)`` pairs."""

    root = project_root.resolve()
    _validate_vendored_tomli(root)
    pairs: list[tuple[Path, str]] = []
    for rule in TREE_RULES:
        source_root = _destination(root, rule.source)
        if not source_root.is_dir():
            raise SourceReleaseError(f"allowlisted source directory is absent: {rule.source}")
        if source_root.is_symlink():
            raise SourceReleaseError(f"allowlisted source directory is a symlink: {rule.source}")
        candidates = [
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in rule.suffixes
        ]
        symlinks = [path for path in candidates if path.is_symlink()]
        if symlinks:
            relative = symlinks[0].relative_to(root).as_posix()
            raise SourceReleaseError(f"allowlisted source file is a symlink: {relative}")
        selected = sorted(
            path
            for path in candidates
            if not path.is_symlink()
            and not PROHIBITED_PATH_PARTS.intersection(
                part.casefold() for part in path.relative_to(source_root).parts
            )
        )
        if not selected:
            raise SourceReleaseError(f"allowlisted source directory is empty: {rule.source}")
        for path in selected:
            relative_inside = path.relative_to(source_root).as_posix()
            destination = str(PurePosixPath(rule.destination) / relative_inside)
            pairs.append((path, destination))
    for relative in SINGLE_FILES:
        source = _destination(root, relative)
        if not source.is_file():
            raise SourceReleaseError(f"allowlisted source file is absent: {relative}")
        if source.is_symlink():
            raise SourceReleaseError(f"allowlisted source file is a symlink: {relative}")
        pairs.append((source, relative.replace("\\", "/")))
    for source_relative, destination_relative in MAPPED_FILES:
        source = _destination(root, source_relative)
        if not source.is_file():
            raise SourceReleaseError(
                f"allowlisted evidence file is absent: {source_relative}"
            )
        if source.is_symlink():
            raise SourceReleaseError(
                f"allowlisted evidence file is a symlink: {source_relative}"
            )
        pairs.append((source, destination_relative))

    destinations = [destination.casefold() for _source, destination in pairs]
    duplicates = sorted(
        destination
        for destination in set(destinations)
        if destinations.count(destination) > 1
    )
    if duplicates:
        raise SourceReleaseError(f"allowlist destination collision: {duplicates}")
    return sorted(pairs, key=lambda item: item[1])


def _check(
    checks: list[dict[str, Any]], code: str, passed: bool, detail: str
) -> None:
    checks.append({"code": code, "passed": bool(passed), "detail": detail})


def _candidate_gate(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        payload = _read_object(path)
    except SourceReleaseError:
        _check(checks, "train399_candidate_present", False, "candidate manifest is absent or invalid")
        return None
    _check(checks, "train399_candidate_present", True, "candidate manifest is present")
    schema_ok = payload.get("schema_version") == "PHAxis-StageB-train399-candidate-bundle-1.0"
    status_ok = payload.get("status") == "candidate_gate_passed_not_promoted"
    guards_ok = (
        payload.get("candidate_only") is True
        and payload.get("automatic_promotion_performed") is False
        and payload.get("official_constants_modified") is False
        and payload.get("official_model_contract_modified") is False
        and payload.get("blind_images_used") == 0
    )
    identity_payload = payload.get("identity_payload")
    candidate_identity = payload.get("candidate_bundle_identity_sha256")
    identity_ok = (
        isinstance(identity_payload, dict)
        and _is_sha256(candidate_identity)
        and _sha256_json(identity_payload) == candidate_identity
    )
    unsigned = dict(payload)
    manifest_identity = unsigned.pop("candidate_manifest_identity_sha256", None)
    manifest_identity_ok = (
        _is_sha256(manifest_identity) and _sha256_json(unsigned) == manifest_identity
    )
    members = identity_payload.get("members") if isinstance(identity_payload, dict) else None
    training_lock = (
        identity_payload.get("training_lock")
        if isinstance(identity_payload, dict)
        else None
    )
    checkpoint_hashes = (
        [member.get("checkpoint_sha256") for member in members]
        if isinstance(members, list) and all(isinstance(member, dict) for member in members)
        else []
    )
    metadata = payload.get("detection_model_metadata")
    member_hashes_ok = (
        len(checkpoint_hashes) == 5
        and len(set(checkpoint_hashes)) == 5
        and all(_is_sha256(value) for value in checkpoint_hashes)
        and [member.get("member_index") for member in members] == list(range(5))
    )
    lock_hashes_ok = (
        isinstance(training_lock, dict)
        and all(
            _is_sha256(training_lock.get(field))
            for field in (
                "dataset_manifest_sha256",
                "split_manifest_sha256",
                "dataset_split_identity_sha256",
                "integrity_manifest_sha256",
            )
        )
        and identity_payload.get("training_lock_identity_sha256")
        == _sha256_json(training_lock)
    )
    metadata_ok = (
        isinstance(metadata, dict)
        and metadata.get("ensemble_members") == 5
        and metadata.get("training_images") == 399
        and metadata.get("validation_images") == 44
        and metadata.get("validation_labels_used_for_gradient_or_early_stopping")
        is False
        and metadata.get("blind_images_used") == 0
        and metadata.get("checkpoint_sha256") == checkpoint_hashes
        and metadata.get("candidate_bundle_identity_sha256") == candidate_identity
    )
    training_ok = (
        isinstance(identity_payload, dict)
        and identity_payload.get("ensemble_members") == 5
        and identity_payload.get("training_images") == 399
        and identity_payload.get("validation_images") == 44
        and identity_payload.get("validation_labels_used_for_gradient_or_early_stopping") is False
        and identity_payload.get("blind_images_used") == 0
        and isinstance(members, list)
        and len(members) == 5
        and member_hashes_ok
        and lock_hashes_ok
        and metadata_ok
    )
    _check(checks, "train399_candidate_schema_status", schema_ok and status_ok, "candidate schema and non-promoting status are locked")
    _check(checks, "train399_candidate_guards", guards_ok, "candidate promotion, blind, and mutation guards pass")
    _check(checks, "train399_candidate_identity", identity_ok and manifest_identity_ok, "candidate logical and complete-receipt identities verify")
    _check(checks, "train399_candidate_scope", training_ok, "candidate binds five members to train399 with QCdev44 excluded from optimization")
    return payload


def _selection_gate(
    path: Path,
    candidate: Mapping[str, Any] | None,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        payload = _read_object(path)
    except SourceReleaseError:
        _check(checks, "train399_selection_present", False, "selection receipt is absent or invalid")
        return None
    _check(checks, "train399_selection_present", True, "selection receipt is present")
    schema_status_ok = (
        payload.get("schema_version") == "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3"
        and payload.get("status") == "completed"
        and payload.get("images") == 44
    )
    guards_ok = (
        payload.get("blind_images_used") == 0
        and payload.get("independent_accuracy_claim_allowed") is False
        and payload.get("validation_labels_used_for_gradient_or_early_stopping") is False
        and payload.get("straight_base_to_tip_presence_proxy_evaluated_during_selection") is True
        and payload.get("distal_endpoint_error_used_as_selection_gate") is False
        and payload.get("complete_line_overlap_used_as_selection_gate") is False
        and payload.get("length_error_used_as_selection_gate") is False
        and payload.get("manual_hair_width_assumed") is False
    )
    matcher = payload.get("primary_matcher_contract")
    matcher_ok = bool(
        _biological_presence_matcher_contract_ok(matcher)
        and payload.get("primary_matcher_contract_sha256") == _sha256_json(matcher)
    )
    unsigned = dict(payload)
    selection_identity = unsigned.pop("selection_receipt_identity_sha256", None)
    identity_ok = _is_sha256(selection_identity) and _sha256_json(unsigned) == selection_identity
    candidate_ok = bool(
        candidate
        and payload.get("candidate_bundle_identity_sha256")
        == candidate.get("candidate_bundle_identity_sha256")
    )
    selection_contract = payload.get("selection_contract")
    candidate_identity_payload = (
        candidate.get("identity_payload") if isinstance(candidate, dict) else None
    )
    candidate_selection_contract = (
        candidate_identity_payload.get("operating_point_selection_contract")
        if isinstance(candidate_identity_payload, dict)
        else None
    )
    protocol_ok = bool(
        isinstance(selection_contract, dict)
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
        ]
    )
    threshold_metrics = payload.get("threshold_metrics")
    metrics_ok = bool(
        isinstance(threshold_metrics, list)
        and len(threshold_metrics) == 10
        and payload.get("selected") in threshold_metrics
        and all(
            isinstance(row, dict)
            and isinstance(row.get("tolerant_biological_presence_20um"), dict)
            and isinstance(row.get("identity_attachment_proxy_20um"), dict)
            and isinstance(row.get("count_mae"), (int, float))
            and isinstance(row.get("count_bias"), (int, float))
            and isinstance(row.get("per_image"), list)
            and len(row["per_image"]) == 44
            for row in threshold_metrics
        )
    )
    receipt_locks_ok = all(
        _is_sha256(payload.get(field))
        for field in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "dataset_split_identity_sha256",
            "integrity_manifest_sha256",
            "canonical_ground_truth_lock_identity_sha256",
        )
    )
    lock_ok = False
    if candidate:
        identity = candidate.get("identity_payload")
        training_lock = identity.get("training_lock") if isinstance(identity, dict) else None
        if isinstance(training_lock, dict):
            lock_ok = all(
                payload.get(receipt_field) == training_lock.get(lock_field)
                for receipt_field, lock_field in (
                    ("dataset_manifest_sha256", "dataset_manifest_sha256"),
                    ("split_manifest_sha256", "split_manifest_sha256"),
                    ("dataset_split_identity_sha256", "dataset_split_identity_sha256"),
                    ("integrity_manifest_sha256", "integrity_manifest_sha256"),
                )
            )
    _check(checks, "train399_selection_schema_status", schema_status_ok, "selection receipt is the completed locked QCdev44 protocol")
    _check(checks, "train399_selection_guards", guards_ok, "selection remains development-only and blind-clean")
    _check(checks, "train399_selection_primary_matcher", matcher_ok, "selection binds the 20-um tolerant biological-presence matcher")
    _check(checks, "train399_selection_protocol", protocol_ok, "candidate and receipt bind the same primary metric and count-aware tie-breaks")
    _check(checks, "train399_selection_metrics", metrics_ok, "all thresholds carry biological, attachment, count, and per-image sufficient statistics")
    _check(checks, "train399_selection_identity", identity_ok, "selection logical identity verifies")
    _check(checks, "train399_candidate_selection_binding", candidate_ok and lock_ok and receipt_locks_ok, "selection binds the candidate and its dataset/split/integrity locks")
    return payload


def _evaluation_v12_metric_contract_ok(payload: Mapping[str, Any]) -> bool:
    hierarchy = payload.get("metric_hierarchy")
    overall = payload.get("overall")
    bootstrap = payload.get("paired_bootstrap_95ci")
    per_image = payload.get("per_image")
    locks = payload.get("prediction_input_locks")
    comparator_root = payload.get("comparator_contract")
    primary_matcher = (
        hierarchy.get("primary_matcher_contract")
        if isinstance(hierarchy, dict)
        else None
    )
    if not (
        isinstance(hierarchy, dict)
        and hierarchy.get("primary_minimum_truth_coverage") == 0.25
        and hierarchy.get("primary_minimum_prediction_coverage") == 0.25
        and hierarchy.get("primary_minimum_direction_cosine") == 0.0
        and hierarchy.get("primary_tolerance_um") == 20.0
        and "without endpoint gates" in str(hierarchy.get("primary", ""))
        and _biological_presence_matcher_contract_ok(primary_matcher)
        and hierarchy.get("primary_matcher_contract_sha256")
        == _sha256_json(primary_matcher)
        and isinstance(overall, dict)
        and set(overall) == {"stageb_train399", "hybrid_max"}
        and isinstance(bootstrap, dict)
        and bootstrap.get("method") == "paired image-level nonparametric bootstrap"
        and bootstrap.get("repetitions") == 10_000
        and bootstrap.get("seed") == 20260828
        and isinstance(per_image, list)
        and len(per_image) == 44
        and isinstance(locks, dict)
        and isinstance(comparator_root, dict)
    ):
        return False
    for expert in ("stageb_train399", "hybrid_max"):
        record = overall.get(expert)
        if not isinstance(record, dict) or record.get("images") != 44:
            return False
        for metric in (
            "tolerant_biological_presence",
            "identity_attachment_proxy",
            "strict_whole_line_correspondence",
        ):
            values = record.get(metric)
            if not isinstance(values, dict) or set(values) != {"5", "10", "20"}:
                return False
    delta = bootstrap.get("delta_stageb_train399_minus_hybrid")
    if not isinstance(delta, dict) or not isinstance(
        delta.get("biological_presence_f1_20um"), dict
    ):
        return False
    ordered_task_ids: list[str] = []
    for row in per_image:
        if not isinstance(row, dict):
            return False
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in ordered_task_ids:
            return False
        ordered_task_ids.append(task_id)
        for expert in ("stageb_train399", "hybrid_max"):
            expert_row = row.get(expert)
            presence = (
                expert_row.get("biological_presence_tp")
                if isinstance(expert_row, dict)
                else None
            )
            if not isinstance(presence, dict):
                return False
            try:
                tolerances = {float(value) for value in presence}
            except (TypeError, ValueError):
                return False
            if tolerances != {5.0, 10.0, 20.0}:
                return False
    for list_field, identity_field in (
        ("stageb_detection_files", "stageb_detection_set_identity_sha256"),
        ("hybrid_prediction_files", "hybrid_prediction_set_identity_sha256"),
    ):
        records = locks.get(list_field)
        identity = locks.get(identity_field)
        if not (
            isinstance(records, list)
            and len(records) == 44
            and all(
                isinstance(record, dict)
                and set(record) == {"task_id", "sha256"}
                and _is_sha256(record.get("sha256"))
                for record in records
            )
            and [record["task_id"] for record in records] == ordered_task_ids
            and _is_sha256(identity)
            and identity == _sha256_json(records)
        ):
            return False
    comparator = comparator_root.get("hybrid_max")
    expected_hybrid_identity = (
        LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
    )
    return bool(
        isinstance(comparator, dict)
        and comparator.get("evidence_role") == "locked_legacy_development_comparator"
        and comparator.get("schema_version")
        == "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
        and comparator.get("identity_hair_variant") == "hybrid_verified_increment"
        and comparator.get("count_hair_variant") == "hybrid_verified_increment"
        and comparator.get("endpoint_complete_identity_layer") is True
        and comparator.get("phaxis_payload_allowed") is False
        and comparator.get("stageb_identity_source_allowed") is False
        and comparator.get("prediction_set_identity_sha256")
        == locks.get("hybrid_prediction_set_identity_sha256")
        and comparator.get("prediction_set_identity_sha256")
        == expected_hybrid_identity
        and comparator.get("expected_prediction_set_identity_sha256")
        == expected_hybrid_identity
    )


def _evaluation_gate(
    path: Path,
    candidate_path: Path,
    selection_path: Path,
    candidate: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        payload = _read_object(path)
    except SourceReleaseError:
        _check(checks, "train399_evaluation_present", False, "evaluation receipt is absent or invalid")
        return None
    _check(checks, "train399_evaluation_present", True, "evaluation receipt is present")
    contract = payload.get("training_contract")
    inputs = payload.get("inputs_sha256")
    schema_status_ok = (
        payload.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
        and payload.get("status") == "completed"
    )
    guards_ok = (
        payload.get("blind_images_used") == 0
        and payload.get("independent_accuracy_claim_allowed") is False
        and isinstance(contract, dict)
        and contract.get("training_images") == 399
        and contract.get("validation_images") == 44
        and contract.get("validation_labels_used_for_gradient_or_early_stopping") is False
    )
    file_binding_ok = bool(
        isinstance(inputs, dict)
        and candidate_path.is_file()
        and selection_path.is_file()
        and inputs.get("candidate_manifest") == _sha256_file(candidate_path)
        and inputs.get("selection_receipt") == _sha256_file(selection_path)
    )
    logical_binding_ok = bool(
        candidate
        and selection
        and isinstance(contract, dict)
        and contract.get("candidate_bundle_identity_sha256")
        == candidate.get("candidate_bundle_identity_sha256")
        and contract.get("selection_receipt_identity_sha256")
        == selection.get("selection_receipt_identity_sha256")
        and _is_sha256(contract.get("selected_model_metadata_identity_sha256"))
    )
    checkpoint_binding_ok = False
    if candidate and isinstance(contract, dict):
        metadata = candidate.get("detection_model_metadata")
        expected = metadata.get("checkpoint_sha256") if isinstance(metadata, dict) else None
        observed = contract.get("checkpoint_sha256")
        checkpoint_binding_ok = (
            isinstance(expected, list)
            and isinstance(observed, list)
            and len(expected) == len(observed) == 5
            and expected == observed
            and all(_is_sha256(value) for value in expected)
        )
    evidence_lock_ok = False
    if selection and isinstance(inputs, dict):
        evidence_lock_ok = all(
            inputs.get(input_field) == selection.get(receipt_field)
            for input_field, receipt_field in (
                ("dataset_manifest", "dataset_manifest_sha256"),
                ("split_manifest", "split_manifest_sha256"),
                ("integrity_manifest", "integrity_manifest_sha256"),
                (
                    "canonical_ground_truth_lock_identity",
                    "canonical_ground_truth_lock_identity_sha256",
                ),
            )
        ) and _is_sha256(inputs.get("selected_model_metadata"))
    metric_contract_ok = _evaluation_v12_metric_contract_ok(payload)
    _check(checks, "train399_evaluation_schema_status", schema_status_ok, "evaluation receipt is complete and uses the formal schema")
    _check(checks, "train399_evaluation_guards", guards_ok, "evaluation is train399/QCdev44-scoped, blind-clean, and not independent accuracy")
    _check(checks, "train399_receipt_file_binding", file_binding_ok, "evaluation binds exact candidate and selection receipt file hashes")
    _check(checks, "train399_receipt_logical_binding", logical_binding_ok and checkpoint_binding_ok and evidence_lock_ok, "candidate, selection, evaluation, checkpoints, and locked evaluation inputs cross-bind")
    _check(
        checks,
        "train399_evaluation_metric_contract",
        metric_contract_ok,
        (
            "evaluation locks the evaluator-1.2 tolerant biological-presence, "
            "exact10000 bootstrap, prediction-file sets, and legacy comparator contract"
        ),
    )
    return payload


def _root_provider_gate(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        payload = _read_object(path)
    except SourceReleaseError:
        _check(checks, "root_provider_exact283_present", False, "fresh exact283 root-provider receipt is absent or invalid")
        return None
    _check(checks, "root_provider_exact283_present", True, "fresh exact283 root-provider receipt is present")
    layers = payload.get("layers")
    layer_ok = isinstance(layers, dict) and set(layers) == {
        "v12_strip_root_mask",
        "v20_root_polygon",
        "final_hybrid_root_mask",
    }
    if layer_ok:
        layer_ok = all(
            isinstance(record, dict)
            and record.get("exact") == 283
            and record.get("expected") == 283
            and record.get("mismatch_count") == 0
            and record.get("mismatch_task_ids") == []
            and record.get("gate_pass") is True
            for record in layers.values()
        )
    schema_status_ok = (
        payload.get("schema_version")
        == "PHAxis-root-provider-fresh-reference283-audit-1.0"
        and payload.get("status") == "pass_exact_283"
    )
    guards_ok = (
        payload.get("fresh_portable_raw_image_rerun_completed") is True
        and payload.get("fresh_283_exact_reproduction_claim_allowed") is True
        and payload.get("pipeline_raw_image_provenance_gate") is True
        and payload.get("pipeline_stage_evidence_gate") is True
        and payload.get("canonical_annotations_read") is False
        and payload.get("blind_images_used") == 0
    )
    identity_payload = {
        key: payload.get(key)
        for key in (
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
    identity_ok = (
        _is_sha256(payload.get("audit_identity_sha256"))
        and _sha256_json(identity_payload) == payload.get("audit_identity_sha256")
        and all(
            _is_sha256(payload.get(field))
            for field in (
                "reference_identity_sha256",
                "fresh_reference_identity_sha256",
                "bundle_identity_sha256",
                "pipeline_identity_sha256",
            )
        )
        and isinstance(payload.get("prepared_radius_fallback_task_ids"), list)
        and isinstance(
            payload.get("attachment_supported_extension_rescue_task_ids"), list
        )
    )
    source_ok = payload.get("source_image_mismatch_task_ids") == []
    _check(checks, "root_provider_exact283_schema_status", schema_status_ok, "root-provider fresh audit status is pass_exact_283")
    _check(checks, "root_provider_exact283_layers", layer_ok and source_ok, "all three root layers and source identities are exact for 283/283")
    _check(checks, "root_provider_exact283_guards", guards_ok, "portable raw-image provenance, stage evidence, blind, and annotation guards pass")
    _check(checks, "root_provider_exact283_identity", identity_ok, "root-provider audit logical identity verifies")
    return payload


def _evaluator_self_contained(project_root: Path, checks: list[dict[str, Any]]) -> bool:
    path = project_root / "scripts/phaxis/evaluate_stageb_train399_qcdev44.py"
    if not path.is_file():
        _check(checks, "train399_evaluator_self_contained", False, "formal train399 evaluator is absent")
        return False
    text = path.read_text(encoding="utf-8")
    prohibited = (
        "--rhaxiscc-code",
        "from rhaxiscc",
        "import rhaxiscc",
        "args.rhaxiscc_code",
    )
    hits = [token for token in prohibited if token in text]
    passed = not hits
    detail = (
        "train399 evaluator uses only project-owned matcher/evaluation code"
        if passed
        else "train399 evaluator still imports matcher/evaluation code from --rhaxiscc-code"
    )
    _check(checks, "train399_evaluator_self_contained", passed, detail)
    return passed


def _official_contract_current_evidence_ok(
    *,
    model: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evaluation_path: Path,
    root_payload: Mapping[str, Any],
    root_path: Path,
    expected_stageb_binding: Mapping[str, Any],
) -> bool:
    development = model.get("development_evidence")
    qcdev = development.get("qcdev44") if isinstance(development, dict) else None
    source = qcdev.get("source") if isinstance(qcdev, dict) else None
    comparator = (
        qcdev.get("same_run_historical_endpoint_complete_comparator")
        if isinstance(qcdev, dict)
        else None
    )
    comparator_source = (
        comparator.get("source_prediction_contract")
        if isinstance(comparator, dict)
        else None
    )
    evaluation_comparator = evaluation.get("comparator_contract")
    evaluation_comparator = (
        evaluation_comparator.get("hybrid_max")
        if isinstance(evaluation_comparator, dict)
        else None
    )
    evaluation_overall = evaluation.get("overall")
    evaluation_stageb = (
        evaluation_overall.get("stageb_train399")
        if isinstance(evaluation_overall, dict)
        else None
    )
    hair = model.get("hair_identity_count_expert")
    root = model.get("root_expert")
    serialized = json.dumps(model, ensure_ascii=False, sort_keys=True).casefold()
    forbidden_legacy = (
        "checkpoint_sha256_in_fold_order",
        "each_oof_scored_image_excluded_from_its_scoring_model",
        "development_algorithm_candidate_only",
        "oof443_stratified",
        "rhaxis_nextgen_hybrid_max/predictions",
        '"evaluation_path"',
    )
    return bool(
        isinstance(development, dict)
        and set(development) == {"qcdev44"}
        and isinstance(qcdev, dict)
        and qcdev.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
        and qcdev.get("metric_hierarchy") == evaluation.get("metric_hierarchy")
        and qcdev.get("stageb_train399") == evaluation_stageb
        and qcdev.get("paired_bootstrap_95ci")
        == evaluation.get("paired_bootstrap_95ci")
        and qcdev.get("prediction_input_locks")
        == evaluation.get("prediction_input_locks")
        and isinstance(source, dict)
        and source.get("evaluation_sha256") == _sha256_file(evaluation_path)
        and source.get("evaluation_content_identity_sha256")
        == _sha256_json(evaluation)
        and comparator_source == evaluation_comparator
        and isinstance(hair, dict)
        and hair.get("checkpoint_sha256_in_member_order")
        == expected_stageb_binding.get("checkpoint_sha256")
        and hair.get("expert_id") == expected_stageb_binding.get("expert_id")
        and isinstance(root, dict)
        and root.get("fresh_exact283_receipt_sha256") == _sha256_file(root_path)
        and root.get("fresh_exact283_audit_identity_sha256")
        == root_payload.get("audit_identity_sha256")
        and root.get("reference_identity_sha256")
        == root_payload.get("reference_identity_sha256")
        and root.get("fresh_reference_identity_sha256")
        == root_payload.get("fresh_reference_identity_sha256")
        and root.get("pipeline_identity_sha256")
        == root_payload.get("pipeline_identity_sha256")
        and root.get("bundle_identity_sha256")
        == root_payload.get("bundle_identity_sha256")
        and isinstance(root.get("root_bundle_authority"), Mapping)
        and root["root_bundle_authority"].get("bundle_identity_sha256")
        == root_payload.get("bundle_identity_sha256")
        and all(token not in serialized for token in forbidden_legacy)
    )


def _official_model_contract_promotion_gate(
    *,
    model: Mapping[str, Any] | None,
    model_path: Path,
    root_payload: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    root_path: Path,
    candidate_path: Path,
    selection_path: Path,
    evaluation_path: Path,
    checks: list[dict[str, Any]],
) -> bool:
    passed = False
    if model and root_payload and candidate and selection and evaluation:
        unsigned = dict(model)
        model_identity = unsigned.pop("model_contract_identity_sha256", None)
        promotion = model.get("promotion")
        red_lines = model.get("red_lines")
        candidate_metadata = candidate.get("detection_model_metadata")
        evaluation_contract = evaluation.get("training_contract")
        selected = selection.get("selected")
        expected_binding = None
        if (
            isinstance(candidate_metadata, dict)
            and isinstance(evaluation_contract, dict)
            and isinstance(selected, dict)
        ):
            expected_binding = {
                "expert_id": candidate_metadata.get("expert_id"),
                "checkpoint_sha256": candidate_metadata.get("checkpoint_sha256"),
                "selected_score_threshold": selected.get("threshold"),
                "candidate_bundle_identity_sha256": candidate.get(
                    "candidate_bundle_identity_sha256"
                ),
                "selection_receipt_identity_sha256": selection.get(
                    "selection_receipt_identity_sha256"
                ),
                "selected_model_metadata_identity_sha256": evaluation_contract.get(
                    "selected_model_metadata_identity_sha256"
                ),
            }
        expected_sources = {
            "train399_candidate": _sha256_file(candidate_path),
            "train399_selection": _sha256_file(selection_path),
            "train399_evaluation": _sha256_file(evaluation_path),
            "root_exact283": _sha256_file(root_path),
        }
        expected_identities = {
            "candidate_bundle_identity_sha256": candidate.get(
                "candidate_bundle_identity_sha256"
            ),
            "selection_receipt_identity_sha256": selection.get(
                "selection_receipt_identity_sha256"
            ),
            "selected_model_metadata_identity_sha256": (
                evaluation_contract.get("selected_model_metadata_identity_sha256")
                if isinstance(evaluation_contract, dict)
                else None
            ),
            "root_exact283_audit_identity_sha256": root_payload.get(
                "audit_identity_sha256"
            ),
        }
        final_sources = (
            promotion.get("final_receipt_source_sha256")
            if isinstance(promotion, dict)
            else None
        )
        final_identities = (
            promotion.get("final_receipt_identity_sha256")
            if isinstance(promotion, dict)
            else None
        )
        current_evidence_ok = bool(
            expected_binding is not None
            and _official_contract_current_evidence_ok(
                model=model,
                evaluation=evaluation,
                evaluation_path=evaluation_path,
                root_payload=root_payload,
                root_path=root_path,
                expected_stageb_binding=expected_binding,
            )
        )
        try:
            validate_proposal_public_identity(model)
            public_identity_ok = True
        except ContractError:
            public_identity_ok = False
        _check(
            checks,
            "model_contract_current_evaluator_evidence",
            current_evidence_ok,
            (
                "official contract contains only evaluator-1.2 qcdev evidence, "
                "fresh exact283 root bindings, and train399 member hashes"
            ),
        )
        passed = bool(
            _is_sha256(model_identity)
            and _sha256_json(unsigned) == model_identity
            and isinstance(promotion, dict)
            and promotion.get("schema_version")
            == "PHAxis-model-contract-promotion-1.0"
            and promotion.get("status") == "applied_formal_release"
            and promotion.get("official_apply_performed") is True
            and promotion.get("formal_gate_source_sha256") == expected_sources
            and promotion.get("formal_gate_identity_sha256") == expected_identities
            and expected_binding is not None
            and current_evidence_ok
            and public_identity_ok
            and promotion.get("stageb_binding") == expected_binding
            and _is_sha256(promotion.get("proposal_file_sha256"))
            and _is_sha256(promotion.get("proposal_identity_sha256"))
            and _is_sha256(
                promotion.get("expected_source_model_contract_sha256")
            )
            and isinstance(final_sources, dict)
            and set(final_sources) == {"stageb", "fusion", "traits", "evidence"}
            and all(_is_sha256(value) for value in final_sources.values())
            and isinstance(final_identities, dict)
            and set(final_identities) == {"stageb", "fusion", "traits", "evidence"}
            and all(_is_sha256(value) for value in final_identities.values())
            and isinstance(red_lines, dict)
            and red_lines.get("blind_images_used") == 0
            and red_lines.get("canonical_annotations_read_during_inference") is False
            and red_lines.get("condition_metadata_used_for_routing") is False
            and red_lines.get(
                "validation_labels_used_for_training_by_current_five_member_deployment_ensemble"
            )
            is False
            and red_lines.get("formal_train399_only_stageb_weights_available") is True
            and red_lines.get("independent_accuracy_claimed") is False
            and red_lines.get("root_cap_region_statistics_included") is False
        )
    else:
        _check(
            checks,
            "model_contract_current_evaluator_evidence",
            False,
            "official contract or formal Gate inputs are absent",
        )
    _check(
        checks,
        "model_contract_applied_promotion_authority",
        passed,
        (
            "official model contract is a sealed CAS-applied authority binding "
            "the exact283/train399 Gate and final StageB/fusion/traits/evidence"
        ),
    )
    return passed


def _final_public_identity_gate(
    *,
    model: Mapping[str, Any] | None,
    fusion_path: Path,
    traits_path: Path,
    checks: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Verify final producer receipts carry the proposal's public identities."""

    payloads: dict[str, dict[str, Any] | None] = {"fusion": None, "traits": None}
    for role, path in (("fusion", fusion_path), ("traits", traits_path)):
        try:
            payloads[role] = _read_object(path)
        except SourceReleaseError:
            payloads[role] = None

    promotion = model.get("promotion") if isinstance(model, Mapping) else None
    root_expert = model.get("root_expert") if isinstance(model, Mapping) else None
    model_bundle_id = model.get("model_bundle_id") if isinstance(model, Mapping) else None
    root_expert_id = (
        root_expert.get("expert_id") if isinstance(root_expert, Mapping) else None
    )
    root_provider_role = (
        root_expert.get("provider_role") if isinstance(root_expert, Mapping) else None
    )
    proposal_sha256 = (
        promotion.get("proposal_file_sha256")
        if isinstance(promotion, Mapping)
        else None
    )
    proposal_identity = (
        promotion.get("proposal_identity_sha256")
        if isinstance(promotion, Mapping)
        else None
    )
    final_sources = (
        promotion.get("final_receipt_source_sha256")
        if isinstance(promotion, Mapping)
        else None
    )
    final_identities = (
        promotion.get("final_receipt_identity_sha256")
        if isinstance(promotion, Mapping)
        else None
    )
    final_public_identities = (
        promotion.get("final_receipt_public_identity")
        if isinstance(promotion, Mapping)
        else None
    )
    expected_public_identities = {
        role: {
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
        }
        for role in ("fusion", "traits")
    }
    expected = {
        "fusion": {
            "identity_field": "summary_identity_sha256",
            "model_bundle_field": "model_bundle_id",
            "root_expert_field": "root_expert",
        },
        "traits": {
            "identity_field": "export_identity_sha256",
            "model_bundle_field": "model_bundle_id",
            "root_expert_field": "root_expert_id",
        },
    }
    passed = bool(
        isinstance(model_bundle_id, str)
        and model_bundle_id.startswith(MODEL_BUNDLE_PREFIX)
        and isinstance(root_expert_id, str)
        and root_expert_id.startswith(ROOT_EXPERT_PREFIX)
        and root_provider_role == ROOT_PROVIDER_ROLE
        and _is_sha256(proposal_sha256)
        and _is_sha256(proposal_identity)
        and isinstance(final_sources, Mapping)
        and isinstance(final_identities, Mapping)
        and final_public_identities == expected_public_identities
    )
    for role, path in (("fusion", fusion_path), ("traits", traits_path)):
        payload = payloads[role]
        policy = expected[role]
        identity_field = policy["identity_field"]
        if not isinstance(payload, dict):
            passed = False
            continue
        unsigned = dict(payload)
        sealed_identity = unsigned.pop(identity_field, None)
        passed = bool(
            passed
            and _is_sha256(sealed_identity)
            and sealed_identity == _sha256_json(unsigned)
            and final_sources.get(role) == _sha256_file(path)
            and final_identities.get(role) == sealed_identity
            and payload.get("model_contract_proposal_sha256") == proposal_sha256
            and payload.get("model_contract_proposal_identity_sha256")
            == proposal_identity
            and payload.get(policy["model_bundle_field"]) == model_bundle_id
            and payload.get(policy["root_expert_field"]) == root_expert_id
        )
    _check(
        checks,
        "final_fusion_traits_public_identity",
        passed,
        (
            "sealed final fusion and traits receipts bind the applied proposal "
            "and its exact model-bundle/root-provider public identities"
        ),
    )
    return payloads["fusion"], payloads["traits"]


def _release_human_metadata_gate(
    *,
    path: Path,
    project_root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate separately sealed, author-supplied public release metadata."""

    try:
        payload = _read_object(path)
    except SourceReleaseError:
        _check(
            checks,
            "release_human_metadata_present",
            False,
            "sealed author-supplied release metadata is absent or invalid",
        )
        _check(
            checks,
            "release_human_metadata_schema_identity",
            False,
            "release metadata schema and logical identity are not available",
        )
        _check(
            checks,
            "release_human_metadata_public_coordinates",
            False,
            "author, maintainer, GitHub, PyPI, and project URL coordinates are not available",
        )
        _check(
            checks,
            "release_human_metadata_rights",
            False,
            "source redistribution authorization and separate-asset boundaries are not available",
        )
        return None

    _check(
        checks,
        "release_human_metadata_present",
        True,
        "sealed author-supplied release metadata is present",
    )
    expected_keys = {
        "schema_version",
        "status",
        "product",
        "product_version",
        "distribution",
        "authors",
        "maintainers",
        "project_urls",
        "release_coordinates",
        "rights",
        "metadata_identity_sha256",
    }
    unsigned = dict(payload)
    identity = unsigned.pop("metadata_identity_sha256", None)
    schema_identity_ok = bool(
        set(payload) == expected_keys
        and payload.get("schema_version") == RELEASE_HUMAN_METADATA_SCHEMA
        and payload.get("status") == "author_verified_release_authority"
        and payload.get("product") == "PHAxis"
        and payload.get("product_version") == RELEASE_VERSION
        and payload.get("distribution") == "phaxis"
        and _is_sha256(identity)
        and _sha256_json(unsigned) == identity
    )
    _check(
        checks,
        "release_human_metadata_schema_identity",
        schema_identity_ok,
        "release metadata has the exact public schema and sealed logical identity",
    )

    def people_ok(value: Any) -> bool:
        return bool(
            isinstance(value, list)
            and value
            and all(
                isinstance(person, dict)
                and set(person)
                == {
                    "display_name",
                    "given_names",
                    "family_names",
                    "email",
                    "affiliation",
                    "orcid",
                }
                and all(
                    isinstance(person.get(field), str)
                    and bool(person[field].strip())
                    and "\n" not in person[field]
                    and "\r" not in person[field]
                    for field in (
                        "display_name",
                        "given_names",
                        "family_names",
                    )
                )
                and isinstance(person.get("email"), str)
                and bool(EMAIL_RE.fullmatch(person["email"]))
                and isinstance(person.get("affiliation"), str)
                and bool(person["affiliation"].strip())
                and "\n" not in person["affiliation"]
                and "\r" not in person["affiliation"]
                and (
                    person.get("orcid") is None
                    or _is_orcid(person.get("orcid"))
                )
                for person in value
            )
        )

    urls = payload.get("project_urls")
    coordinates = payload.get("release_coordinates")
    urls_ok = bool(
        isinstance(urls, dict)
        and set(urls) == set(PROJECT_URL_LABELS)
        and all(
            isinstance(urls.get(label), str)
            and urls[label].startswith("https://")
            and "\n" not in urls[label]
            and "\r" not in urls[label]
            for label in PROJECT_URL_LABELS
        )
    )
    repository_url = urls.get("Repository") if isinstance(urls, dict) else None
    repository_base = (
        repository_url.rstrip("/") if isinstance(repository_url, str) else ""
    )
    coordinates_ok = bool(
        isinstance(coordinates, dict)
        and set(coordinates)
        == {
            "github_repository_url",
            "github_release_tag",
            "github_release_url",
            "pypi_project",
            "pypi_version",
            "pypi_project_url",
            "release_date",
            "release_doi",
        }
        and re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            repository_base,
        )
        is not None
        and coordinates.get("github_repository_url") == repository_base
        and coordinates.get("github_release_tag") == f"v{RELEASE_VERSION}"
        and coordinates.get("github_release_url")
        == f"{repository_base}/releases/tag/v{RELEASE_VERSION}"
        and coordinates.get("pypi_project") == "phaxis"
        and coordinates.get("pypi_version") == RELEASE_VERSION
        and coordinates.get("pypi_project_url")
        == f"https://pypi.org/project/phaxis/{RELEASE_VERSION}/"
        and _is_release_date(coordinates.get("release_date"))
        and _is_release_doi(coordinates.get("release_doi"))
    )
    public_coordinates_ok = bool(
        people_ok(payload.get("authors"))
        and people_ok(payload.get("maintainers"))
        and urls_ok
        and coordinates_ok
        and not _absolute_host_path_markers(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        and "REQUIRED_" not in json.dumps(payload, ensure_ascii=False)
    )
    _check(
        checks,
        "release_human_metadata_public_coordinates",
        public_coordinates_ok,
        (
            "author/maintainer display/given/family names, affiliations, "
            "optional canonical ORCIDs, and exact GitHub, PyPI, DOI, release-date, and project "
            "coordinates verify"
        ),
    )

    rights = payload.get("rights")
    license_path = project_root / "LICENSE"
    rights_ok = bool(
        isinstance(rights, dict)
        and set(rights)
        == {
            "source_license_spdx",
            "license_file_sha256",
            "source_release_authorized",
            "model_weights_included",
            "images_included",
            "annotations_included",
            "separate_asset_rights_not_conferred",
        }
        and rights.get("source_license_spdx") == "Apache-2.0"
        and license_path.is_file()
        and rights.get("license_file_sha256") == _sha256_file(license_path)
        and rights.get("source_release_authorized") is True
        and rights.get("model_weights_included") is False
        and rights.get("images_included") is False
        and rights.get("annotations_included") is False
        and rights.get("separate_asset_rights_not_conferred") is True
    )
    _check(
        checks,
        "release_human_metadata_rights",
        rights_ok,
        "Apache-2.0 source authorization is explicit and model/data rights remain separate",
    )
    return payload if all(
        (schema_identity_ok, public_coordinates_ok, rights_ok)
    ) else None


def inspect_formal_release_gate(
    *,
    project_root: Path,
    root_provider_receipt: Path | None = None,
    train399_candidate_manifest: Path | None = None,
    train399_selection_receipt: Path | None = None,
    train399_evaluation_receipt: Path | None = None,
    final_fusion_summary: Path | None = None,
    final_traits_summary: Path | None = None,
    release_human_metadata: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every release gate without importing project runtime modules."""

    root = project_root.resolve()
    checks: list[dict[str, Any]] = []
    model_path = root / "configs/phaxis/v1_0/model_contract.json"
    try:
        model = _read_object(model_path)
    except SourceReleaseError:
        model = None
    _check(
        checks,
        "model_contract_formal_release_status",
        bool(model and model.get("formal_release_status") == "passed"),
        "model_contract formal_release_status must equal passed",
    )

    root_payload = _root_provider_gate(
        root_provider_receipt or Path("__missing_root_provider_receipt__"), checks
    )
    candidate_path = train399_candidate_manifest or Path("__missing_candidate__")
    selection_path = train399_selection_receipt or Path("__missing_selection__")
    evaluation_path = train399_evaluation_receipt or Path("__missing_evaluation__")
    candidate = _candidate_gate(candidate_path, checks)
    selection = _selection_gate(selection_path, candidate, checks)
    evaluation = _evaluation_gate(
        evaluation_path,
        candidate_path,
        selection_path,
        candidate,
        selection,
        checks,
    )
    _official_model_contract_promotion_gate(
        model=model,
        model_path=model_path,
        root_payload=root_payload,
        candidate=candidate,
        selection=selection,
        evaluation=evaluation,
        root_path=root_provider_receipt or Path("__missing_root_provider_receipt__"),
        candidate_path=candidate_path,
        selection_path=selection_path,
        evaluation_path=evaluation_path,
        checks=checks,
    )
    fusion_path = final_fusion_summary or Path("__missing_final_fusion_summary__")
    traits_path = final_traits_summary or Path("__missing_final_traits_summary__")
    fusion_payload, traits_payload = _final_public_identity_gate(
        model=model,
        fusion_path=fusion_path,
        traits_path=traits_path,
        checks=checks,
    )
    metadata_path = release_human_metadata or Path("__missing_release_human_metadata__")
    release_metadata = _release_human_metadata_gate(
        path=metadata_path,
        project_root=root,
        checks=checks,
    )
    evaluator_ok = _evaluator_self_contained(root, checks)
    passed = all(record["passed"] for record in checks)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "formal_release_allowed": passed,
        "checks": checks,
        "release_human_metadata": release_metadata,
        "artifacts": {
            "model_contract": (
                {
                    "sha256": _sha256_file(model_path),
                    "formal_release_status": model.get("formal_release_status"),
                }
                if model and model_path.is_file()
                else None
            ),
            "root_provider_exact283": (
                {
                    "sha256": _sha256_file(root_provider_receipt),
                    "audit_identity_sha256": root_payload.get("audit_identity_sha256"),
                    "reference_identity_sha256": root_payload.get("reference_identity_sha256"),
                }
                if root_payload and root_provider_receipt and root_provider_receipt.is_file()
                else None
            ),
            "train399_candidate": (
                {
                    "sha256": _sha256_file(candidate_path),
                    "candidate_bundle_identity_sha256": candidate.get("candidate_bundle_identity_sha256"),
                    "candidate_manifest_identity_sha256": candidate.get("candidate_manifest_identity_sha256"),
                }
                if candidate and candidate_path.is_file()
                else None
            ),
            "train399_selection": (
                {
                    "sha256": _sha256_file(selection_path),
                    "selection_receipt_identity_sha256": selection.get("selection_receipt_identity_sha256"),
                }
                if selection and selection_path.is_file()
                else None
            ),
            "train399_evaluation": (
                {
                    "sha256": _sha256_file(evaluation_path),
                    "schema_version": evaluation.get("schema_version"),
                }
                if evaluation and evaluation_path.is_file()
                else None
            ),
            "final_fusion": (
                {
                    "sha256": _sha256_file(fusion_path),
                    "summary_identity_sha256": fusion_payload.get(
                        "summary_identity_sha256"
                    ),
                    "model_bundle_id": fusion_payload.get("model_bundle_id"),
                    "root_expert_id": fusion_payload.get("root_expert"),
                }
                if fusion_payload and fusion_path.is_file()
                else None
            ),
            "final_traits": (
                {
                    "sha256": _sha256_file(traits_path),
                    "export_identity_sha256": traits_payload.get(
                        "export_identity_sha256"
                    ),
                    "model_bundle_id": traits_payload.get("model_bundle_id"),
                    "root_expert_id": traits_payload.get("root_expert_id"),
                }
                if traits_payload and traits_path.is_file()
                else None
            ),
            "release_human_metadata": (
                {
                    "sha256": _sha256_file(metadata_path),
                    "metadata_identity_sha256": release_metadata.get(
                        "metadata_identity_sha256"
                    ),
                    "github_release_url": release_metadata[
                        "release_coordinates"
                    ]["github_release_url"],
                    "pypi_project_url": release_metadata[
                        "release_coordinates"
                    ]["pypi_project_url"],
                }
                if release_metadata and metadata_path.is_file()
                else None
            ),
            "train399_evaluator_self_contained": evaluator_ok,
        },
    }


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _pyproject(
    *, formal: bool, release_metadata: Mapping[str, Any] | None = None
) -> str:
    classifier = (
        "Development Status :: 5 - Production/Stable"
        if formal
        else "Development Status :: 4 - Beta"
    )
    people_metadata = ""
    project_urls = ""
    if release_metadata is not None:
        people_lines: list[str] = []
        for field in ("authors", "maintainers"):
            people_lines.append(f"{field} = [")
            for person in release_metadata[field]:
                people_lines.append(
                    "  {name = "
                    f"{_toml_string(person['display_name'])}, email = "
                    f"{_toml_string(person['email'])}}},"
                )
            people_lines.append("]")
        people_metadata = "\n".join(people_lines) + "\n"
        url_lines = ["[project.urls]"]
        for label in PROJECT_URL_LABELS:
            url_lines.append(
                f"{label} = {_toml_string(release_metadata['project_urls'][label])}"
            )
        project_urls = "\n" + "\n".join(url_lines) + "\n"
    return f'''[build-system]
requires = ["setuptools>=77", "wheel>=0.45"]
build-backend = "setuptools.build_meta"

[project]
name = "phaxis"
version = "{RELEASE_VERSION}"
description = "Reproducible Arabidopsis primary-root and root-hair phenotyping"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
license-files = ["LICENSE", "src/phaxis/_vendor/tomli/LICENSE.txt"]
{people_metadata}keywords = ["Arabidopsis", "plant phenomics", "primary root", "root hairs", "microscopy"]
classifiers = [
  "{classifier}",
  "Intended Audience :: Science/Research",
  "Operating System :: OS Independent",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
  "numpy>=1.26,<3",
  "packaging>=24,<26",
  "scipy>=1.11,<2",
]

[project.optional-dependencies]
visualization = ["opencv-python-headless>=4.9,<6", "Pillow>=10,<13"]
inference = [
  "opencv-python-headless>=4.9,<6",
  "Pillow>=10,<13",
  "tifffile>=2024.8,<2027",
  "timm>=1.0.28,<2",
  "torch>=2.6,<3",
  "torchvision>=0.21,<1",
]
deployment = [
  "imageio>=2.35,<3",
  "joblib>=1.4,<2",
  "matplotlib>=3.8,<4",
  "opencv-python-headless>=4.9,<6",
  "pandas>=2.2,<4",
  "Pillow>=10,<13",
  "scikit-image>=0.24,<0.27",
  "scikit-learn>=1.5,<2",
  "statsmodels>=0.14,<1",
  "tifffile>=2024.8,<2027",
  "timm>=1.0.28,<2",
  "torch>=2.6,<3",
  "torchvision>=0.21,<1",
]
analysis = [
  "matplotlib>=3.8,<4",
  "opencv-python-headless>=4.9,<6",
  "pandas>=2.2,<4",
  "Pillow>=10,<13",
  "scikit-image>=0.24,<0.27",
  "statsmodels>=0.14,<1",
  "tifffile>=2024.8,<2027",
]
publication = [
  "Pillow>=10,<13",
  "python-docx>=1.1,<2",
]
test = [
  "matplotlib>=3.8,<4",
  "opencv-python-headless>=4.9,<6",
  "pandas>=2.2,<4",
  "Pillow>=10,<13",
  "python-docx>=1.1,<2",
  "pytest>=8,<10",
  "scikit-image>=0.24,<0.27",
  "statsmodels>=0.14,<1",
  "tifffile>=2024.8,<2027",
  "timm>=1.0.28,<2",
  "torch>=2.6,<3",
  "torchvision>=0.21,<1",
]
build = ["build>=1.2,<2", "twine>=6,<7"]

[project.scripts]
phaxis = "phaxis.cli:main"
{project_urls}

[tool.setuptools]
package-dir = {{"" = "src"}}
include-package-data = false

[tool.setuptools.package-data]
"phaxis._vendor.tomli" = ["LICENSE.txt", "py.typed"]

[tool.setuptools.packages.find]
where = ["src"]
include = ["phaxis", "phaxis.*"]

[tool.pytest.ini_options]
addopts = "-ra --strict-config --strict-markers -p no:cacheprovider -p source_release_pytest"
testpaths = ["tests/phaxis"]
pythonpath = ["src"]
'''


def _manifest_in(*, formal: bool, release_metadata: bool = False) -> str:
    receipt = FORMAL_RECEIPT_NAME if formal else BLOCKED_RECEIPT_NAME
    metadata = (
        RELEASE_HUMAN_METADATA_NAME
        if release_metadata
        else RELEASE_HUMAN_METADATA_TEMPLATE_NAME
    )
    zenodo = f"include {ZENODO_METADATA_NAME}\n" if formal else ""
    return f"""include .gitattributes
include .gitignore
{zenodo}include LICENSE
include NOTICE
include {THIRD_PARTY_NOTICES_NAME}
include {THIRD_PARTY_INVENTORY_NAME}
include {SBOM_NAME}
include README.md
include MODEL_CARD.md
include DATA_CARD.md
include CITATION.cff
include CHANGELOG.md
include CODE_OF_CONDUCT.md
include CONTRIBUTING.md
include SECURITY.md
include SUPPORT.md
include docs/phaxis/TRAIT_CONTRACT_CN.md
include docs/phaxis/USER_GUIDE.md
include SOURCE_MANIFEST.json
include source_release_pytest.py
include {receipt}
include {metadata}
recursive-include .github *.md *.yml *.yaml
recursive-include configs/phaxis *.json *.txt
recursive-include docs/phaxis *.md
recursive-include evidence *.json
recursive-include scripts/phaxis *.py *.ps1
recursive-include src/phaxis/_vendor/tomli LICENSE.txt py.typed
recursive-include tests/phaxis *.py
recursive-include tests/phaxis/fixtures *.json
recursive-exclude * __pycache__ *.py[cod]
global-exclude *.pt *.pth *.ckpt *.onnx *.engine *.joblib *.pkl *.pickle
global-exclude *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp
"""


def _workflow_readme() -> str:
    return '''## Batch analysis workflow

`phaxis analyze --manifest workflow.json --output analysis-output` is strictly
plan-only by default: it validates every hash and cross-binding, prints the
deterministic root-provider → Stage-B → fusion → traits → distal-axis-profile
plan, and does not create the analysis output directory. Add `--plan-output
plan.json` to save the plan; that file must not already exist.

For schema compatibility the manifest key remains `model_contract_proposal`.
Its locked path may name either the unapplied proposal or the applied official contract.
An official contract is accepted only after its complete seal,
canonical public identity, Stage-B/root authorities, and reconstructed original
proposal receipt all verify; downstream outputs continue to carry that original
proposal receipt.

Execution requires the explicit flag `--execute`. An existing output is
rejected unless `--execute --resume` is used. Resume additionally requires a
valid `workflow_state.json` bound to the same manifest and plan; completed
stages are reused only after their output-tree identities verify, and drift or
tampering fails closed. `--resume` without `--execute`, or resume against an
absent output, is invalid. Optional `--review-overlays` produces review-only
overlays and never enables condition-based routing.

```console
phaxis analyze --manifest workflow.json --output analysis-output
phaxis analyze --manifest workflow.json --output analysis-output --execute
phaxis analyze --manifest workflow.json --output analysis-output --execute --resume
```

There is currently no separate manifest-construction command. The complete
required-key template is below. Every `SHA256_REQUIRED` must be replaced with
the SHA-256 of the referenced file. Paths may be relative to `workflow.json`.
GPU placeholder strings must be replaced with non-negative physical GPU
indices. The template is deliberately invalid until real asset identities are
supplied; it does not invent release hashes or device assignments.

```json
{
  "schema_version": "PHAxis-analysis-workflow-manifest-1.0",
  "model_contract_proposal": {
    "path": "receipts/model-contract-proposal.json",
    "sha256": "SHA256_REQUIRED"
  },
  "root_provider": {
    "project": "assets/root-provider-project",
    "python_executable": "assets/root-provider-python/python",
    "bundle": {
      "path": "assets/root-provider-bundle",
      "registry_sha256": "SHA256_REQUIRED",
      "bundle_identity_sha256": "SHA256_REQUIRED"
    },
    "input_manifest": {
      "path": "inputs/root-provider-input.csv",
      "sha256": "SHA256_REQUIRED"
    },
    "acquisition_gate": {
      "path": "locks/acquisition-gate.json",
      "sha256": "SHA256_REQUIRED"
    },
    "deployment_metadata": {
      "path": "locks/deployment-metadata.json",
      "sha256": "SHA256_REQUIRED"
    },
    "canonical_manifest": {
      "path": "locks/canonical-manifest.csv",
      "sha256": "SHA256_REQUIRED"
    },
    "deployment_manifest": {
      "path": "locks/deployment-manifest.csv",
      "sha256": "SHA256_REQUIRED"
    },
    "deployment_lock": {
      "path": "locks/deployment-lock.json",
      "sha256": "SHA256_REQUIRED"
    },
    "reference_registry": {
      "path": "locks/root-provider-reference-registry.json",
      "sha256": "SHA256_REQUIRED"
    },
    "image_root": "inputs/images",
    "v1_physical_gpus": ["REPLACE_WITH_GPU_INDEX"],
    "q8_physical_gpus": ["REPLACE_WITH_GPU_INDEX"]
  },
  "stageb": {
    "input_manifest": {
      "path": "inputs/stageb-input.csv",
      "sha256": "SHA256_REQUIRED"
    },
    "image_root": "inputs/images",
    "checkpoints": [
      {"path": "assets/train399/member-1.pt", "sha256": "SHA256_REQUIRED"},
      {"path": "assets/train399/member-2.pt", "sha256": "SHA256_REQUIRED"},
      {"path": "assets/train399/member-3.pt", "sha256": "SHA256_REQUIRED"},
      {"path": "assets/train399/member-4.pt", "sha256": "SHA256_REQUIRED"},
      {"path": "assets/train399/member-5.pt", "sha256": "SHA256_REQUIRED"}
    ],
    "candidate_manifest": {
      "path": "receipts/train399-candidate-manifest.json",
      "sha256": "SHA256_REQUIRED"
    },
    "selected_model_metadata": {
      "path": "receipts/train399-selected-model-metadata.json",
      "sha256": "SHA256_REQUIRED"
    },
    "selection_receipt": {
      "path": "receipts/train399-selection-receipt.json",
      "sha256": "SHA256_REQUIRED"
    },
    "physical_gpu": "REPLACE_WITH_GPU_INDEX",
    "internal_device": "cuda:0",
    "shared_input_acceleration": false
  },
  "traits": {
    "metadata_csv": {
      "path": "inputs/traits-metadata.csv",
      "sha256": "SHA256_REQUIRED"
    }
  },
  "distal_axis_profiles": {
    "contract_json": {
      "path": "contracts/axial-profile-contract.json",
      "sha256": "SHA256_REQUIRED"
    }
  },
  "review_overlays": {"enabled": false},
  "guards": {
    "condition_metadata_used_for_routing": false,
    "canonical_annotations_read": false,
    "blind_images_used": 0,
    "root_cap_region_output": false
  },
  "manifest_identity_sha256": "COMPUTE_LAST_WITH_phaxis.io.sha256_json"
}
```

Seal the final object only after replacing every placeholder:

```python
from phaxis.io import atomic_write_json, read_json, sha256_json

payload = read_json("workflow.json")
payload.pop("manifest_identity_sha256", None)
payload["manifest_identity_sha256"] = sha256_json(payload)
atomic_write_json("workflow.json", payload)
```

The source archive does not contain the external root-provider bundle or its
six locked control files, source images, model weights, or deployment receipts.
The root bundle directory must contain `root_provider_bundle.json` whose file
hash and `bundle_identity_sha256` match the manifest. Stage-B requires exactly
five distinct last-epoch train399 checkpoints whose hashes match the candidate
manifest, plus the mutually bound candidate manifest, selected-model metadata,
and selection receipt. The formal source-release gate separately requires the
fresh exact283 root-provider audit and train399 development-evaluation receipt;
a workflow manifest does not replace those publication gates.

Optional root-provider shard/concurrency/batch fields and Stage-B memory limits
use the defaults documented in `phaxis.workflow` when omitted. All guards shown
above are mandatory and immutable.
'''


def _asset_access_readme(
    *, formal: bool, release_metadata: Mapping[str, Any] | None
) -> str:
    """Explain model-asset acquisition without inventing a public coordinate."""

    if not formal:
        return f"""## Model assets

This blocked development tree has no authorized public model-asset download
coordinate. Do not infer one from local paths or from
`{RELEASE_HUMAN_METADATA_TEMPLATE_NAME}`. A formal release README is generated
only after the author-verified release page and rights record are sealed.
"""

    release_url = release_metadata["release_coordinates"]["github_release_url"]
    documentation_url = release_metadata["project_urls"]["Documentation"]
    return f"""## Model assets

The wheel and source archive contain no weights, microscopy images, or root-
provider bundle. Obtain the separately authorized **PHAxis {RELEASE_VERSION}
model asset bundle** from the [matching release page]({release_url}) and follow
the [asset and workflow documentation]({documentation_url}). Use an asset only
when its supplied manifest binds the same PHAxis version and every SHA-256
required by your workflow manifest. If that release page does not provide an
authorized asset manifest, production inference is not yet reproducible; do
not substitute a similarly named checkpoint or an asset from another release.
"""


def _readme(
    *, formal: bool, release_metadata: Mapping[str, Any] | None = None
) -> str:
    if formal:
        banner = "PHAxis 1.0.0 source release"
        status = (
            "This tree passed the formal source-release gate recorded in "
            f"`{FORMAL_RECEIPT_NAME}`."
        )
        published_install = f"""
After the author-controlled PyPI publication succeeds, the equivalent
published-wheel install is:

```console
python -m pip install "phaxis[deployment]=={RELEASE_VERSION}"
```
"""
    else:
        banner = "BLOCKED DEVELOPMENT STAGING — DO NOT PUBLISH"
        status = (
            "This tree exists only because the builder was invoked with "
            "`--allow-blocked-development-staging`. The failed gates are recorded in "
            f"`{BLOCKED_RECEIPT_NAME}`. It is test evidence, not release authority."
        )
        published_install = ""

    return f"""# {banner}

{status}

PHAxis is a contract-driven system for physically calibrated *Arabidopsis
thaliana* primary-root and root-hair phenotyping. It combines a
continuity-aware primary-root geometry ensemble with a five-member root-hair
identity/count expert and one-to-one conditional-length linkage. PHAxis 1.0.0
is the sole public software and model-system version; implementation labels and
receipt schema names are not additional products.

PHAxis reports exactly **32 canonical image-derived descriptors**: 19 for the
visible primary root and 13 for root hairs. The 82-column canonical image table
also carries identity, calibration, observability, QC, reason-code, and
provenance fields; it does not report 82 phenotypes. Root-cap output is one
distal/root-cap coordinate point, not a segmented root-cap region.

## Install and verify

Verify the authored source bytes before installation, then create a clean
environment and install this source tree. In blocked mode this remains local
test evidence and is not publishable.

```console
python -B scripts/phaxis/verify_source_release.py .
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install ".[deployment]"
python -m pip check
python -m phaxis --version
python -m phaxis --help
python -m phaxis analyze --help
```

{published_install}

Expected version output is `PHAxis {RELEASE_VERSION}`. The package contains no
model weights, microscopy images, or annotations. Full analysis additionally
requires the separately authorized model assets and their exact manifests.

## Minimal workflow

Planning is the safe default. It verifies a sealed workflow manifest and prints
a deterministic plan without starting inference, accessing CUDA, or creating
the analysis output directory:

```console
phaxis analyze --manifest workflow.json --output analysis-output
```

Execution requires explicit `--execute`. An interrupted run can be continued
only when its manifest, plan, state, and completed outputs retain the same
identities:

```console
phaxis analyze --manifest workflow.json --output analysis-output --execute
phaxis analyze --manifest workflow.json --output analysis-output --execute --resume
```

Export the 32 descriptors from already fused predictions with the exact model
contract that authorized those predictions:

```console
phaxis export-traits \
  --predictions analysis-output/fusion/predictions \
  --metadata metadata.csv \
  --model-contract official-contract.json \
  --output exported-traits
```

The complete installation, asset, planning, execution, resume, output, and
failure semantics are in `docs/phaxis/USER_GUIDE.md`. The bilingual phenotype
catalogue is `docs/phaxis/TRAIT_CONTRACT_CN.md`. Run
`phaxis <command> --help` for the installed CLI contract.

{_asset_access_readme(formal=formal, release_metadata=release_metadata)}

## Reproducibility and evidence boundaries

The 399-image training partition and family-isolated 44-image same-domain
development partition support training and operating-point selection; they are
not independent external accuracy evidence. Application images do not provide
dense root-hair accuracy truth. Formal benchmarks belong only in sealed
release receipts bound to the PHAxis 1.0.0 model identity. Blind/final-
validation material is not required or permitted for source verification.

From an untouched authored source tree, verify exact bytes and run the portable
CPU contracts with:

```console
python -B scripts/phaxis/verify_source_release.py .
python -B -m pip install -e ".[test]"
python -B -m pytest tests/phaxis -q
```

The exact source verifier is for the authored source tree, not an unpacked
sdist whose build backend adds standard packaging metadata. Distribution
receipts separately bind wheel/sdist archive hashes and audit generated
members. Source, model weights, and biological data retain separate release
and rights authorities.

## GitHub and PyPI metadata

The source tree supplies PHAxis-specific citation, changelog, contribution,
conduct, security, support, issue, pull-request, and SHA-pinned CPU-CI files.
Formal author, maintainer, repository, documentation, issue-tracker, PyPI,
release, DOI, and asset coordinates are rendered only from author-verified
metadata. The builder never invents or uploads them.
"""
def _notice() -> str:
    return """PHAxis 1.0.0

Copyright 2026 PHAxis contributors.
Licensed under the Apache License, Version 2.0.

Model weights and datasets are separate assets and are not included or licensed
by this source-only distribution.

Direct third-party dependency and vendored-component notices are recorded in
THIRD_PARTY_NOTICES.md, the machine-readable license inventory is
THIRD_PARTY_LICENSES.json, and the CycloneDX software bill of materials is
SBOM.cdx.json. Those files distinguish the exact vendored Tomli source from
declared dependency ranges; resolved external artifacts remain governed by
their upstream licenses and the formal hash-locked dependency materialization
receipt.
"""


def _vendored_tomli_inventory() -> dict[str, Any]:
    files = [dict(record) for record in VENDORED_TOMLI_FILES]
    license_record = next(
        record for record in files if record["role"] == "license_text"
    )
    return {
        **dict(VENDORED_TOMLI_COMPONENT),
        "source_file_count": len(files),
        "source_files": files,
        "source_tree_identity_sha256": _sha256_json(files),
        "license_text_sha256": license_record["sha256"],
        "source_authority": "Tomli 2.4.0 pure-Python distribution file set",
        "source_bytes_unmodified": True,
    }


def _third_party_inventory() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for dependency in THIRD_PARTY_DEPENDENCIES:
        record: dict[str, Any] = {
            "name": dependency.name,
            "requirement": dependency.requirement,
            "scopes": list(dependency.scopes),
            "license_expression": dependency.license_expression,
            "project_url": dependency.project_url,
            "relationship": "declared_direct_dependency",
        }
        if dependency.license_note:
            record["license_note"] = dependency.license_note
        records.append(record)
    vendored = [_vendored_tomli_inventory()]
    payload: dict[str, Any] = {
        "schema_version": "PHAxis-third-party-license-inventory-1.0",
        "status": "complete_declared_direct_dependency_inventory",
        "product": "PHAxis",
        "product_version": RELEASE_VERSION,
        "inventory_scope": (
            "declared_direct_dependencies_build_requirements_and_vendored_components"
        ),
        "dependency_count": len(records),
        "dependencies": records,
        "vendored_component_count": len(vendored),
        "vendored_components": vendored,
        "resolved_transitive_dependency_claimed": False,
        "artifact_specific_license_review_required": True,
        "exact_resolved_artifacts_authority": (
            "offline dependency materialization receipt and wheelhouse hashes"
        ),
    }
    payload["inventory_identity_sha256"] = _sha256_json(payload)
    return payload


def _third_party_notices() -> str:
    lines = [
        "# PHAxis 1.0.0 third-party notices",
        "",
        (
            "This inventory covers direct dependency declarations and build "
            "requirements in the generated PHAxis source release. It does not "
            "replace license texts shipped by resolved wheels, and it does not "
            "claim that a future resolver's transitive closure is unchanged."
        ),
        "",
        "| Distribution | Declared requirement | Scope(s) | License expression | Upstream | Note |",
        "|---|---|---|---|---|---|",
    ]
    for dependency in THIRD_PARTY_DEPENDENCIES:
        scopes = ", ".join(dependency.scopes)
        lines.append(
            f"| {dependency.name} | `{dependency.requirement}` | {scopes} | "
            f"`{dependency.license_expression}` | {dependency.project_url} | "
            f"{dependency.license_note or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Vendored fallback",
            "",
            (
                "PHAxis vendors the unmodified pure-Python source files from "
                "Tomli 2.4.0 for Python 3.10 and isolated no-site source "
                "verification. The exact files and SHA-256 values are recorded "
                "in THIRD_PARTY_LICENSES.json and SBOM.cdx.json. The complete "
                "MIT license is retained at "
                "`src/phaxis/_vendor/tomli/LICENSE.txt` and in the wheel."
            ),
            "",
            (
                "The exact platform-specific deployment wheelhouse is materialized "
                "later by the formal dependency-lock stage with one SHA-256 per "
                "artifact. Users and redistributors must retain each artifact's "
                "upstream license and notice files."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _cyclonedx_sbom() -> dict[str, Any]:
    root_ref = f"pkg:pypi/phaxis@{RELEASE_VERSION}"
    components: list[dict[str, Any]] = []
    dependency_refs: list[str] = []
    for dependency in THIRD_PARTY_DEPENDENCIES:
        normalized = re.sub(r"[-_.]+", "-", dependency.name).casefold()
        bom_ref = f"pkg:pypi/{normalized}"
        dependency_refs.append(bom_ref)
        components.append(
            {
                "type": "library",
                "bom-ref": bom_ref,
                "name": dependency.name,
                "purl": bom_ref,
                "licenses": [
                    {"expression": dependency.license_expression}
                ],
                "externalReferences": [
                    {
                        "type": "distribution",
                        "url": dependency.project_url,
                    }
                ],
                "properties": [
                    {
                        "name": "phaxis:declared-requirement",
                        "value": dependency.requirement,
                    },
                    {
                        "name": "phaxis:dependency-scopes",
                        "value": ",".join(dependency.scopes),
                    },
                    {
                        "name": "phaxis:resolution-status",
                        "value": "declared-version-range-not-resolved-in-source-sbom",
                    },
                    *(
                        [
                            {
                                "name": "phaxis:license-note",
                                "value": dependency.license_note,
                            }
                        ]
                        if dependency.license_note
                        else []
                    ),
                ],
            }
        )
    vendored = _vendored_tomli_inventory()
    vendored_ref = str(vendored["purl"])
    dependency_refs.append(vendored_ref)
    components.append(
        {
            "type": "library",
            "bom-ref": vendored_ref,
            "name": vendored["name"],
            "version": vendored["version"],
            "purl": vendored["purl"],
            "licenses": [{"expression": vendored["license_expression"]}],
            "externalReferences": [
                {"type": "distribution", "url": vendored["project_url"]}
            ],
            "properties": [
                {
                    "name": "phaxis:relationship",
                    "value": vendored["relationship"],
                },
                {
                    "name": "phaxis:vendored-package-path",
                    "value": vendored["package_path"],
                },
                {
                    "name": "phaxis:vendored-source-file-count",
                    "value": str(vendored["source_file_count"]),
                },
                {
                    "name": "phaxis:vendored-source-tree-identity-sha256",
                    "value": vendored["source_tree_identity_sha256"],
                },
                {
                    "name": "phaxis:vendored-license-text-path",
                    "value": vendored["license_text_path"],
                },
                {
                    "name": "phaxis:vendored-license-text-sha256",
                    "value": vendored["license_text_sha256"],
                },
                {
                    "name": "phaxis:runtime-scope",
                    "value": vendored["runtime_scope"],
                },
            ],
        }
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": (
            "urn:uuid:"
            + str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"https://pypi.org/project/phaxis/{RELEASE_VERSION}/declared-direct-sbom",
                )
            )
        ),
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "phaxis",
                "version": RELEASE_VERSION,
                "purl": root_ref,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            },
            "properties": [
                {
                    "name": "phaxis:sbom-scope",
                    "value": (
                        "declared-direct-dependencies-build-requirements-and-"
                        "vendored-components"
                    ),
                },
                {
                    "name": "phaxis:resolved-transitive-closure-claimed",
                    "value": "false",
                },
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": dependency_refs},
            *({"ref": reference, "dependsOn": []} for reference in dependency_refs),
        ],
    }


def _citation_cff(release_metadata: Mapping[str, Any] | None) -> str:
    """Render citation metadata without inventing public repository coordinates."""

    people = (
        release_metadata.get("authors")
        if isinstance(release_metadata, Mapping)
        else None
    )
    if not isinstance(people, list) or not people:
        people = [
            {
                "display_name": "PHAxis contributors",
                "given_names": "PHAxis",
                "family_names": "contributors",
            }
        ]
    lines = [
        "cff-version: 1.2.0",
        'message: "If you use PHAxis, please cite the software release and the accompanying article."',
        f"title: {_toml_string(SOFTWARE_CITATION_TITLE)}",
        "type: software",
        f'version: "{RELEASE_VERSION}"',
        "license: Apache-2.0",
        "authors:",
    ]
    for person in people:
        family = str(person.get("family_names", "")).strip() or "contributors"
        given = str(person.get("given_names", "")).strip()
        lines.append(f"  - family-names: {_toml_string(family)}")
        if given:
            lines.append(f"    given-names: {_toml_string(given)}")
        email = person.get("email")
        if isinstance(email, str) and EMAIL_RE.fullmatch(email):
            lines.append(f"    email: {_toml_string(email)}")
        affiliation = person.get("affiliation")
        if isinstance(affiliation, str) and affiliation.strip():
            lines.append(f"    affiliation: {_toml_string(affiliation.strip())}")
        orcid = person.get("orcid")
        if _is_orcid(orcid):
            lines.append(f"    orcid: {_toml_string(str(orcid))}")
    if isinstance(release_metadata, Mapping):
        urls = release_metadata.get("project_urls")
        if isinstance(urls, Mapping):
            repository = urls.get("Repository")
            homepage = urls.get("Homepage")
            if isinstance(repository, str):
                lines.append(f"repository-code: {_toml_string(repository)}")
            if isinstance(homepage, str):
                lines.append(f"url: {_toml_string(homepage)}")
        coordinates = release_metadata.get("release_coordinates")
        if isinstance(coordinates, Mapping):
            release_date = coordinates.get("release_date")
            release_doi = coordinates.get("release_doi")
            if _is_release_date(release_date):
                lines.append(f"date-released: {_toml_string(str(release_date))}")
            if _is_release_doi(release_doi):
                lines.append(f"doi: {_toml_string(str(release_doi))}")
    return "\n".join(lines) + "\n"


def _zenodo_metadata(release_metadata: Mapping[str, Any] | None) -> str:
    """Render Zenodo metadata from the same human authority as CITATION.cff.

    ``release_doi`` is deliberately treated as the DOI for the immutable
    1.0.0 record.  Zenodo assigns a distinct concept DOI to the record family;
    that identifier cannot be inferred from a version DOI and is therefore not
    fabricated in this pre-deposition file.
    """

    if not isinstance(release_metadata, Mapping):
        payload: dict[str, Any] = {
            "title": SOFTWARE_CITATION_TITLE,
            "upload_type": "software",
            "version": RELEASE_VERSION,
            "access_right": "open",
            "license": "Apache-2.0",
            "creators": [{"name": "contributors, PHAxis"}],
            "notes": (
                "BLOCKED DEVELOPMENT STAGING. The author-verified version DOI, "
                "release date, creators, and repository coordinates are absent; "
                "do not archive this tree in Zenodo."
            ),
        }
        return _pretty_json_text(payload)

    coordinates = release_metadata["release_coordinates"]
    creators: list[dict[str, str]] = []
    for person in release_metadata["authors"]:
        creator = {
            "name": (
                f"{str(person['family_names']).strip()}, "
                f"{str(person['given_names']).strip()}"
            ),
            "affiliation": str(person["affiliation"]).strip(),
        }
        if person.get("orcid") is not None:
            creator["orcid"] = str(person["orcid"]).removeprefix("https://orcid.org/")
        creators.append(creator)
    payload = {
        "title": SOFTWARE_CITATION_TITLE,
        "description": (
            "PHAxis 1.0.0 provides reproducible, physically calibrated "
            "Arabidopsis primary-root and root-hair phenotyping."
        ),
        "upload_type": "software",
        "version": RELEASE_VERSION,
        "publication_date": coordinates["release_date"],
        "doi": coordinates["release_doi"],
        "access_right": "open",
        "license": "Apache-2.0",
        "creators": creators,
        "keywords": [
            "Arabidopsis thaliana",
            "plant phenomics",
            "primary root",
            "root hair",
            "computer vision",
        ],
        "related_identifiers": [
            {
                "identifier": coordinates["github_release_url"],
                "relation": "isAlternateIdentifier",
            },
            {
                "identifier": coordinates["pypi_project_url"],
                "relation": "isAlternateIdentifier",
            },
        ],
        "notes": (
            "The doi field is the author-verified version DOI for the immutable "
            "PHAxis 1.0.0 record. Zenodo's concept DOI identifies the complete "
            "version family; it is intentionally not inferred from the version "
            "DOI and must be verified from the published Zenodo record."
        ),
    }
    return _pretty_json_text(payload)


def _release_workflow(
    *, formal: bool, release_metadata: Mapping[str, Any] | None
) -> str:
    """Render the exact-tag, receipt-closed Trusted Publishing workflow.

    The local stage-41/49/60 products must first be uploaded as assets on a
    protected *draft* GitHub release.  The workflow verifies those bytes and
    their cross-receipt hashes before passing only the wheel/sdist to PyPI's OIDC
    publisher.  A blocked staging tree contains the same audited workflow shape
    but cannot pass ``PHAXIS_RELEASE_ENABLED``.
    """

    enabled = formal and isinstance(release_metadata, Mapping)
    if enabled:
        coordinates = release_metadata["release_coordinates"]
        repository = str(coordinates["github_repository_url"]).removeprefix(
            "https://github.com/"
        )
        release_url = str(coordinates["github_release_url"])
        release_doi = str(coordinates["release_doi"])
    else:
        repository = "BLOCKED/DO_NOT_PUBLISH"
        release_url = "https://example.invalid/BLOCKED_DO_NOT_PUBLISH"
        release_doi = "BLOCKED_DO_NOT_PUBLISH"

    template = r'''name: PHAxis 1.0.0 trusted release

on:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: phaxis-1.0.0-production-release
  cancel-in-progress: false

env:
  PHAXIS_RELEASE_ENABLED: "__RELEASE_ENABLED__"
  PHAXIS_EXPECTED_REPOSITORY: "__EXPECTED_REPOSITORY__"
  PHAXIS_EXPECTED_RELEASE_URL: "__EXPECTED_RELEASE_URL__"
  PHAXIS_EXPECTED_VERSION_DOI: "__EXPECTED_VERSION_DOI__"
  PHAXIS_RELEASE_TAG: "v1.0.0"

jobs:
  verify-release-authority:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Check out the exact release tag
        uses: actions/checkout@__CHECKOUT_SHA__ # v4.2.2
        with:
          fetch-depth: 0
          persist-credentials: false
      - name: Set up Python
        uses: actions/setup-python@__SETUP_PYTHON_SHA__ # v5.6.0
        with:
          python-version: "3.12"
          cache: pip
      - name: Verify repository, tag, source Gate, and package identity
        shell: bash
        run: |
          set -euo pipefail
          test "$PHAXIS_RELEASE_ENABLED" = "true"
          test "$GITHUB_REPOSITORY" = "$PHAXIS_EXPECTED_REPOSITORY"
          test "$GITHUB_REF" = "refs/tags/v1.0.0"
          test "$(git rev-list -n 1 v1.0.0)" = "$GITHUB_SHA"
          test "$(git tag --points-at HEAD --list v1.0.0)" = "v1.0.0"
          python -B scripts/phaxis/verify_source_release.py .
          python -B - <<'PY'
          import json
          from pathlib import Path
          import tomllib
          project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
          assert project["name"] == "phaxis"
          assert project["version"] == "1.0.0"
          assert project["scripts"] == {"phaxis": "phaxis.cli:main"}
          gate = json.loads(Path("FORMAL_RELEASE_GATE_RECEIPT.json").read_text(encoding="utf-8"))
          assert gate["status"] == "passed" and gate["formal_release_allowed"] is True
          assert gate["blind_images_used"] == 0
          PY
      - name: Download the pre-staged draft-release authority
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          mkdir authority-assets
          gh release view v1.0.0 --repo "$PHAXIS_EXPECTED_REPOSITORY" \
            --json isDraft,tagName,url,targetCommitish > authority-assets/github_release.json
          for pattern in \
            'phaxis-1.0.0-py3-none-any.whl' \
            'phaxis-1.0.0.tar.gz' \
            'phaxis-1.0.0.cdx.json' \
            'phaxis-1.0.0-THIRD_PARTY_NOTICES.md' \
            'phaxis-1.0.0-THIRD_PARTY_LICENSES.json' \
            'release_asset_inventory.json' \
            'SHA256SUMS' \
            'distribution_receipt.json' \
            'clean_install_receipt.json' \
            'release_finalization.json'; do
            gh release download v1.0.0 --repo "$PHAXIS_EXPECTED_REPOSITORY" \
              --dir authority-assets --pattern "$pattern"
          done
      - name: Verify distribution, clean-install, and finalization cross-bindings
        shell: bash
        run: |
          set -euo pipefail
          python -B - <<'PY'
          from copy import deepcopy
          import hashlib
          import json
          import os
          from pathlib import Path
          import re

          root = Path(".")
          assets = Path("authority-assets")

          def load(path):
              return json.loads(path.read_text(encoding="utf-8"))

          def digest(path):
              return hashlib.sha256(path.read_bytes()).hexdigest()

          def logical(payload):
              return hashlib.sha256(json.dumps(
                  payload, ensure_ascii=False, allow_nan=False,
                  sort_keys=True, separators=(",", ":"),
              ).encode("utf-8")).hexdigest()

          def sealed(payload, field):
              unsigned = deepcopy(payload)
              observed = unsigned.pop(field)
              assert observed == logical(unsigned), field

          source = load(root / "SOURCE_MANIFEST.json")
          gate = load(root / "FORMAL_RELEASE_GATE_RECEIPT.json")
          metadata = load(root / "RELEASE_HUMAN_METADATA.json")
          zenodo = load(root / ".zenodo.json")
          distribution = load(assets / "distribution_receipt.json")
          inventory = load(assets / "release_asset_inventory.json")
          clean = load(assets / "clean_install_receipt.json")
          finalization = load(assets / "release_finalization.json")
          release_view = load(assets / "github_release.json")

          assert source["schema_version"] == "PHAxis-source-release-manifest-2.0"
          assert source["release_mode"] == "formal"
          assert source["distribution"] == "phaxis" and source["version"] == "1.0.0"
          assert gate["status"] == "passed" and gate["formal_release_allowed"] is True
          assert gate["blind_images_used"] == 0
          assert metadata["release_coordinates"]["github_release_tag"] == "v1.0.0"
          assert metadata["release_coordinates"]["github_release_url"] == os.environ["PHAXIS_EXPECTED_RELEASE_URL"]
          assert metadata["release_coordinates"]["release_doi"] == os.environ["PHAXIS_EXPECTED_VERSION_DOI"]
          assert zenodo["version"] == "1.0.0" and zenodo["doi"] == os.environ["PHAXIS_EXPECTED_VERSION_DOI"]
          assert "concept DOI" in zenodo["notes"] and "not inferred" in zenodo["notes"]
          assert release_view["isDraft"] is True
          assert release_view["tagName"] == "v1.0.0"
          assert release_view["url"] == os.environ["PHAXIS_EXPECTED_RELEASE_URL"]

          sealed(distribution, "distribution_identity_sha256")
          assert distribution["schema_version"] == "PHAxis-release-distributions-1.0"
          assert distribution["status"] == "completed_wheel_sdist_verified"
          assert distribution["distribution"] == "phaxis" and distribution["version"] == "1.0.0"
          assert distribution["twine_check_passed"] is True
          assert distribution["wheel_archive_audit"]["record_verified"] is True
          assert distribution["wheel_archive_audit"]["source_package_hashes_verified"] is True
          assert distribution["wheel_archive_audit"]["metadata_license_files"] == [
              "LICENSE", "src/phaxis/_vendor/tomli/LICENSE.txt"
          ]
          assert distribution["wheel_archive_audit"]["pep639_license_member_count"] == 2
          assert distribution["wheel_archive_audit"]["license_file_hashes_verified"] is True
          toolchain = deepcopy(distribution["build_toolchain"])
          toolchain_identity = toolchain.pop("build_toolchain_identity_sha256")
          assert toolchain_identity == logical(toolchain)
          assert toolchain["exact_versions_recorded"] is True
          assert set(toolchain["packages"]) == {"build", "setuptools", "twine", "wheel"}
          assert distribution["blind_images_used"] == 0
          for command in distribution["commands"]:
              for argument in command["argv"]:
                  assert not re.match(r"^[A-Za-z]:[\\/]", argument)
                  assert not argument.startswith(("/", "\\\\"))
          assert distribution["source_release_manifest_sha256"] == digest(root / "SOURCE_MANIFEST.json")
          artifacts = {row["filename"]: row for row in distribution["artifacts"]}
          assert set(artifacts) == {"phaxis-1.0.0-py3-none-any.whl", "phaxis-1.0.0.tar.gz"}
          for name, row in artifacts.items():
              path = assets / name
              assert row["sha256"] == digest(path) and row["bytes"] == path.stat().st_size

          sealed(inventory, "release_asset_inventory_identity_sha256")
          assert distribution["release_asset_inventory"]["sha256"] == digest(assets / "release_asset_inventory.json")
          assert inventory["source_release_manifest_sha256"] == digest(root / "SOURCE_MANIFEST.json")
          checksum_rows = {
              line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0]
              for line in (assets / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
              if line.strip()
          }
          for row in inventory["assets"]:
              assert checksum_rows[row["filename"]] == row["sha256"]
              assert digest(assets / row["filename"]) == row["sha256"]

          sealed(clean, "clean_install_receipt_identity_sha256")
          assert clean["schema_version"] == "PHAxis-clean-install-verification-1.0"
          assert clean["status"] == "completed_final_clean_install"
          assert clean["product"] == "PHAxis" and clean["product_version"] == "1.0.0"
          assert clean["source_release_manifest_sha256"] == digest(root / "SOURCE_MANIFEST.json")
          assert clean["formal_wheel"]["sha256"] == artifacts["phaxis-1.0.0-py3-none-any.whl"]["sha256"]
          assert clean["formal_wheel"]["source_package_hashes_verified"] is True
          assert clean["formal_wheel"]["metadata_license_files"] == [
              "LICENSE", "src/phaxis/_vendor/tomli/LICENSE.txt"
          ]
          assert clean["formal_wheel"]["pep639_license_member_count"] == 2
          assert clean["formal_wheel"]["license_file_hashes_verified"] is True
          assert clean["installation"]["pip_check_passed"] is True
          assert clean["blind_images_used"] == 0

          sealed(finalization, "release_finalization_identity_sha256")
          assert finalization["schema_version"] == "PHAxis-post-training-release-finalization-1.0"
          assert finalization["status"] == "completed_formal_release_closure"
          assert finalization["formal_release_closed"] is True and finalization["terminal_stage"] is True
          upstream = finalization["upstream_receipt_sha256"]
          assert upstream["source_release"] == digest(root / "SOURCE_MANIFEST.json")
          assert upstream["distributions"] == digest(assets / "distribution_receipt.json")
          assert upstream["clean_install"] == digest(assets / "clean_install_receipt.json")
          assert finalization["software_supply_chain_closure_included"] is True
          assert finalization["blind_images_used"] == 0
          registry_reference = finalization["release_authority_registry_path"]
          assert not re.match(r"^[A-Za-z]:[\\/]", registry_reference)
          assert not registry_reference.startswith(("/", "\\\\"))

          verified = {
              "schema_version": "PHAxis-GitHub-PyPI-publication-authority-1.0",
              "status": "verified_pending_trusted_publication",
              "tag": "v1.0.0",
              "git_commit": os.environ["GITHUB_SHA"],
              "source_manifest_sha256": digest(root / "SOURCE_MANIFEST.json"),
              "distribution_receipt_sha256": digest(assets / "distribution_receipt.json"),
              "clean_install_receipt_sha256": digest(assets / "clean_install_receipt.json"),
              "release_finalization_sha256": digest(assets / "release_finalization.json"),
              "artifacts": [
                  {"filename": name, "sha256": row["sha256"], "bytes": row["bytes"]}
                  for name, row in sorted(artifacts.items())
              ],
              "blind_images_used": 0,
          }
          (assets / "publication_authority.json").write_text(
              json.dumps(verified, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
              encoding="utf-8",
          )
          PY
          python -B -m pip install --upgrade pip
          python -B -m pip install "twine>=6,<7"
          python -B -m twine check authority-assets/phaxis-1.0.0-py3-none-any.whl authority-assets/phaxis-1.0.0.tar.gz
          python -B -m venv "$RUNNER_TEMP/phaxis-release-smoke"
          "$RUNNER_TEMP/phaxis-release-smoke/bin/python" -B -m pip install authority-assets/phaxis-1.0.0-py3-none-any.whl
          "$RUNNER_TEMP/phaxis-release-smoke/bin/python" -B -m pip check
          test "$("$RUNNER_TEMP/phaxis-release-smoke/bin/python" -B -m phaxis --version)" = "PHAxis 1.0.0"
          mkdir verified-dist verified-authority
          cp authority-assets/phaxis-1.0.0-py3-none-any.whl authority-assets/phaxis-1.0.0.tar.gz verified-dist/
          cp authority-assets/publication_authority.json verified-authority/
      - name: Seal verified distributions for the publish job
        uses: actions/upload-artifact@__UPLOAD_ARTIFACT_SHA__ # v4.6.2
        with:
          name: phaxis-1.0.0-verified-distributions
          path: verified-dist/
          if-no-files-found: error
          retention-days: 7
      - name: Seal verified publication authority
        uses: actions/upload-artifact@__UPLOAD_ARTIFACT_SHA__ # v4.6.2
        with:
          name: phaxis-1.0.0-publication-authority
          path: verified-authority/
          if-no-files-found: error
          retention-days: 7

  publish-pypi:
    needs: verify-release-authority
    if: github.ref == 'refs/tags/v1.0.0'
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/phaxis/1.0.0/
    permissions:
      contents: read
      id-token: write
    steps:
      - name: Retrieve exact verified distributions
        uses: actions/download-artifact@__DOWNLOAD_ARTIFACT_SHA__ # v4.3.0
        with:
          name: phaxis-1.0.0-verified-distributions
          path: dist
      - name: Publish through PyPI Trusted Publishing
        uses: pypa/gh-action-pypi-publish@__PYPI_PUBLISH_SHA__ # v1.14.2
        with:
          packages-dir: dist
          verify-metadata: true
          attestations: true
          skip-existing: false
          verbose: true

  publish-github-release:
    needs: publish-pypi
    if: github.ref == 'refs/tags/v1.0.0'
    runs-on: ubuntu-latest
    environment: github-release
    permissions:
      contents: write
    steps:
      - name: Retrieve verified publication authority
        uses: actions/download-artifact@__DOWNLOAD_ARTIFACT_SHA__ # v4.3.0
        with:
          name: phaxis-1.0.0-publication-authority
          path: authority
      - name: Attach workflow provenance and publish the draft release
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          python -B - <<'PY'
          import hashlib
          import json
          import os
          from pathlib import Path
          authority = json.loads(Path("authority/publication_authority.json").read_text(encoding="utf-8"))
          assert authority["tag"] == "v1.0.0"
          assert authority["git_commit"] == os.environ["GITHUB_SHA"]
          provenance = {
              **authority,
              "status": "trusted_publication_jobs_authorized",
              "github_repository": os.environ["GITHUB_REPOSITORY"],
              "github_workflow": os.environ["GITHUB_WORKFLOW"],
              "github_run_id": os.environ["GITHUB_RUN_ID"],
              "github_run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
          }
          provenance["provenance_identity_sha256"] = hashlib.sha256(json.dumps(
              provenance, ensure_ascii=False, allow_nan=False,
              sort_keys=True, separators=(",", ":"),
          ).encode("utf-8")).hexdigest()
          Path("PHAXIS_PUBLISH_PROVENANCE.json").write_text(
              json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
              encoding="utf-8",
          )
          PY
          gh release upload v1.0.0 PHAXIS_PUBLISH_PROVENANCE.json --repo "$PHAXIS_EXPECTED_REPOSITORY"
          gh release edit v1.0.0 --repo "$PHAXIS_EXPECTED_REPOSITORY" --draft=false --verify-tag
'''
    replacements = {
        "__RELEASE_ENABLED__": "true" if enabled else "false",
        "__EXPECTED_REPOSITORY__": repository,
        "__EXPECTED_RELEASE_URL__": release_url,
        "__EXPECTED_VERSION_DOI__": release_doi,
        "__CHECKOUT_SHA__": GITHUB_ACTION_PINS["actions/checkout"],
        "__SETUP_PYTHON_SHA__": GITHUB_ACTION_PINS["actions/setup-python"],
        "__UPLOAD_ARTIFACT_SHA__": GITHUB_ACTION_PINS["actions/upload-artifact"],
        "__DOWNLOAD_ARTIFACT_SHA__": GITHUB_ACTION_PINS["actions/download-artifact"],
        "__PYPI_PUBLISH_SHA__": GITHUB_ACTION_PINS[
            "pypa/gh-action-pypi-publish"
        ],
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    if re.search(r"__[A-Z0-9_]+__", template):
        raise SourceReleaseError("release workflow contains an unresolved marker")
    return template


def _community_files(
    *, formal: bool, release_metadata: Mapping[str, Any] | None
) -> dict[str, str]:
    """Return GitHub/PyPI-facing community files for the isolated PHAxis tree."""

    release_state = "release" if formal else "release candidate"
    files = {
        ".gitattributes": """# Preserve byte-identical manifest closure on every platform.
* text=auto eol=lf
*.py text eol=lf
*.pyi text eol=lf
*.md text eol=lf
*.json text eol=lf
*.toml text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.cff text eol=lf
*.csv text eol=lf
*.tsv text eol=lf
*.ps1 text eol=lf
*.sh text eol=lf
*.bat text eol=lf
*.cmd text eol=lf
*.png -text
*.jpg -text
*.jpeg -text
*.tif -text
*.tiff -text
*.pt -text
*.pth -text
*.ckpt -text
*.onnx -text
*.engine -text
*.whl -text
""",
        ".gitignore": """# Local environments and generated Python state
.venv/
venv/
env/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/

# Packaging and local analysis outputs
build/
dist/
*.egg-info/
analysis-output/
workflow_state.json
""",
        "CITATION.cff": _citation_cff(release_metadata),
        "CHANGELOG.md": f"""# Changelog

## PHAxis {RELEASE_VERSION}

This {release_state} establishes the first public PHAxis model-system identity:

- continuity-aware primary-root, distal-point, and physical-scale measurement;
- a five-member root-hair identity/count expert trained on the locked
  399-image partition;
- one-to-one attachment-aware fusion and conditional projected length;
- 32 canonical plant-facing root and root-hair descriptors;
- exact-manifest batch execution, provenance receipts, figures, and trait export.

Historical component and provider-ABI labels remain receipt provenance and are
not additional public versions.
""",
        "CONTRIBUTING.md": """# Contributing to PHAxis

PHAxis welcomes focused improvements to Arabidopsis root and root-hair
measurement, reproducibility, documentation, and usability. Open an issue
before a large change so its biological measurement contract and evidence plan
are explicit.

Create an isolated Python environment, then run:

```console
python -B -m pip install -e ".[test]"
python -B -m pytest tests/phaxis -q
```

Pull requests should include tests, exact commands, and the affected
measurement/provenance contract. Never submit private images, annotations,
weights, credentials, local machine paths, or blind/final-validation material.
Condition labels and biological outcomes must not be introduced into model
routing or threshold selection.
""",
        "CODE_OF_CONDUCT.md": """# PHAxis community conduct

We are committed to a respectful, inclusive, and scientifically constructive
community. Discuss people and their work professionally; welcome questions;
credit contributions; protect confidential data; and distinguish evidence from
opinion. Harassment, discrimination, intimidation, or disclosure of private
information is not acceptable. Maintainers may remove material or restrict
participation when necessary to protect contributors and users.
""",
        "SECURITY.md": """# Security policy

Security reports for the supported PHAxis 1.0.x series should be submitted
privately through the repository's GitHub Security Advisory interface. Do not
open a public issue for a suspected credential exposure, arbitrary-code
execution path, dependency compromise, or disclosure of private biological
assets. Include the affected version, a minimal non-confidential reproducer,
and impact. Never attach private microscopy data, annotations, weights, tokens,
or blind/final-validation material.
""",
        "SUPPORT.md": (
            f"""# Support

For usage questions, reproducible software defects, and feature requests, use
the [PHAxis issue tracker]({release_metadata['project_urls']['Issues']}). Read
the [documentation]({release_metadata['project_urls']['Documentation']}) first
and include `phaxis --version`, the operating system, the exact command, and
non-confidential receipt hashes. Security concerns belong in the private route
described in `SECURITY.md`. Do not attach private images, annotations, weights,
credentials, host paths, or blind/final-validation material.
"""
            if formal and release_metadata is not None
            else f"""# Support

This is a blocked development-staging tree, not a published support authority.
Do not use placeholder coordinates from
`{RELEASE_HUMAN_METADATA_TEMPLATE_NAME}`. The formal source release will render
author-verified issue and documentation links here. Security reports must use
the private route named by that formal repository; never post credentials or
private biological assets publicly.
"""
        ),
        ".github/ISSUE_TEMPLATE/bug_report.yml": """name: Bug report
description: Report a reproducible PHAxis software or measurement defect.
title: "[Bug]: "
labels: [bug]
body:
  - type: markdown
    attributes:
      value: Do not upload private images, annotations, weights, credentials, or blind/final-validation material.
  - type: input
    id: version
    attributes:
      label: PHAxis version
      placeholder: Output of phaxis --version
    validations:
      required: true
  - type: dropdown
    id: component
    attributes:
      label: Component
      options:
        - primary-root, distal-point, or scale provider
        - root-hair identity and count
        - attachment-aware fusion or conditional length
        - trait or distal-axis profile export
        - batch workflow, installation, or benchmark
        - documentation or other
    validations:
      required: true
  - type: textarea
    id: reproduce
    attributes:
      label: Reproduction steps
      description: Include the exact command, sanitized paths, and input schema.
    validations:
      required: true
  - type: textarea
    id: observed
    attributes:
      label: Observed behavior
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment and provenance
      description: OS, Python, CUDA/GPU when applicable, model-bundle identity, and receipt hashes.
    validations:
      required: true
  - type: checkboxes
    id: privacy
    attributes:
      label: Data safety
      options:
        - label: I removed credentials, private biological material, local machine paths, and blind/final-validation data.
          required: true
""",
        ".github/ISSUE_TEMPLATE/feature_request.yml": """name: Feature request
description: Propose a plant-facing PHAxis measurement or usability improvement.
title: "[Feature]: "
labels: [enhancement]
body:
  - type: textarea
    id: plant_question
    attributes:
      label: Plant-science question
      description: Describe the biological measurement or workflow need before proposing an implementation.
    validations:
      required: true
  - type: dropdown
    id: scope
    attributes:
      label: Scope
      options:
        - root geometry, distal point, or scale
        - root-hair identity, count, or attachment
        - conditional projected length
        - phenotype ontology or distal-axis profiles
        - visualization, benchmark, or reproducibility
        - packaging, documentation, or other
    validations:
      required: true
  - type: textarea
    id: evidence
    attributes:
      label: Acceptance evidence
      description: State the dataset role, biological metric, expected output, and reproducible test.
    validations:
      required: true
  - type: checkboxes
    id: integrity
    attributes:
      label: Evidence integrity
      options:
        - label: The proposal does not require tuning on blind/final-validation outcomes.
          required: true
        - label: The proposal preserves explicit units, observability, and provenance for affected traits.
          required: true
""",
        ".github/pull_request_template.md": """## Scientific purpose

Describe the plant-facing measurement or software change and why it matters.

## Evidence

- Tests and exact commands:
- Measurement/identity contracts affected:
- Visual or benchmark evidence, when applicable:
- Configuration, model, and receipt identities:

## Checklist

- [ ] PHAxis 1.0.0 remains the sole public model-system identity.
- [ ] Tests, documentation, schema, and changelog are updated where required.
- [ ] No private images, annotations, weights, credentials, host paths, or blind/final-validation material were added.
- [ ] Frozen historical components were not overwritten.
- [ ] Units, observability, expert identity, and provenance remain explicit.
- [ ] Dependencies and third-party assets have documented licenses and provenance.
""",
        ".github/workflows/ci.yml": """name: PHAxis CPU contracts

on:
  push:
  pull_request:

permissions:
  contents: read

concurrency:
  group: phaxis-ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - name: Check out source
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Set up Python
        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Verify exact source-tree closure, including no-site isolation
        env:
          PYTHONDONTWRITEBYTECODE: "1"
        run: |
          python -B scripts/phaxis/verify_source_release.py .
          python -B -S scripts/phaxis/verify_source_release.py .
          python -B -I -S scripts/phaxis/verify_source_release.py .
      - name: Install PHAxis and test dependencies
        run: |
          python -B -m pip install --upgrade pip
          python -B -m pip install ".[test]"
      - name: Run PHAxis CPU contract suite
        env:
          PYTHONDONTWRITEBYTECODE: "1"
        run: python -B -m pytest tests/phaxis -q

  package:
    runs-on: ubuntu-latest
    steps:
      - name: Check out source
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Set up Python
        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: "3.12"
          cache: pip
      - name: Verify exact source-tree closure
        env:
          PYTHONDONTWRITEBYTECODE: "1"
        run: python -B scripts/phaxis/verify_source_release.py .
      - name: Build and inspect distributions
        run: |
          python -B -m pip install --upgrade pip
          python -B -m pip install ".[build]"
          python -B -m build
          python -B -m twine check dist/*
      - name: Smoke-test a clean wheel install
        run: |
          python -B -m venv "$RUNNER_TEMP/phaxis-clean"
          "$RUNNER_TEMP/phaxis-clean/bin/python" -B -m pip install dist/*.whl
          "$RUNNER_TEMP/phaxis-clean/bin/python" -B -m pip check
          "$RUNNER_TEMP/phaxis-clean/bin/python" -B -m phaxis --help
""",
        ".github/workflows/release.yml": _release_workflow(
            formal=formal,
            release_metadata=release_metadata,
        ),
    }
    if formal:
        files[ZENODO_METADATA_NAME] = _zenodo_metadata(release_metadata)
    return files


def _source_release_pytest_plugin() -> str:
    return '''"""Transparent skips for checks that require excluded repository state."""

from __future__ import annotations

import sys
from pathlib import Path
import shutil

import pytest


sys.dont_write_bytecode = True

_REPOSITORY_ONLY = {
    (
        "tests/phaxis/test_stageb_train399_contract.py::"
        "test_locked_audit_uses_qc_family_swap_and_excludes_val_from_gradient"
    ): "requires the private train399 dataset-audit receipt excluded from source releases",
    (
        "tests/phaxis/test_stageb_train399_gpu_queue.py::"
        "test_cpu_plan_locks_physical_to_internal_device"
    ): "requires the original repository's locked envs/rhpheno/python.exe path",
    (
        "tests/phaxis/test_stageb_amp_amendment.py::"
        "test_amp_amendment_binds_failure_source_restart_and_legacy_zero_retry"
    ): (
        "requires private failed-attempt and train399 checkpoint files excluded "
        "from source releases; the mapped amendment evidence is verified separately"
    ),
    (
        "tests/phaxis/test_direct_benchmark_provider.py::"
        "test_descriptor_builder_seals_real_four_mode_closures_and_is_create_only"
    ): (
        "requires the excluded frozen predecessor runtime, acquisition gate, "
        "and runtime configuration"
    ),
    (
        "tests/phaxis/test_direct_benchmark_provider.py::"
        "test_descriptor_builder_check_subprocess_is_explicitly_cpu_only"
    ): (
        "requires the excluded frozen predecessor runtime, acquisition gate, "
        "and runtime configuration"
    ),
    (
        "tests/phaxis/test_direct_benchmark_provider.py::"
        "test_provider_object_seal_rejects_any_descriptor_tamper"
    ): (
        "requires the excluded frozen predecessor runtime, acquisition gate, "
        "and runtime configuration"
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        base_nodeid = item.nodeid.split("[", 1)[0].replace("\\\\", "/")
        reason = _REPOSITORY_ONLY.get(base_nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_sessionfinish(session):
    root = Path(str(session.config.rootpath)).resolve()
    if not (root / "SOURCE_MANIFEST.json").is_file():
        return
    caches = sorted(
        (path for path in root.rglob("__pycache__") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for cache in caches:
        resolved = cache.resolve()
        if resolved.name == "__pycache__" and root in resolved.parents:
            shutil.rmtree(resolved)
    pytest_cache = (root / ".pytest_cache").resolve()
    if pytest_cache.name == ".pytest_cache" and root in pytest_cache.parents:
        shutil.rmtree(pytest_cache, ignore_errors=True)
'''


def _release_human_metadata_template() -> dict[str, Any]:
    return {
        "schema_version": RELEASE_HUMAN_METADATA_SCHEMA,
        "status": "BLOCKED_TEMPLATE_NOT_AUTHORITY",
        "product": "PHAxis",
        "product_version": RELEASE_VERSION,
        "distribution": "phaxis",
        "authors": [
            {
                "display_name": "REQUIRED_AUTHOR_DISPLAY_NAME",
                "given_names": "REQUIRED_AUTHOR_GIVEN_NAMES",
                "family_names": "REQUIRED_AUTHOR_FAMILY_NAMES",
                "email": "REQUIRED_AUTHOR_EMAIL",
                "affiliation": "REQUIRED_AUTHOR_AFFILIATION",
                "orcid": None,
            }
        ],
        "maintainers": [
            {
                "display_name": "REQUIRED_MAINTAINER_DISPLAY_NAME",
                "given_names": "REQUIRED_MAINTAINER_GIVEN_NAMES",
                "family_names": "REQUIRED_MAINTAINER_FAMILY_NAMES",
                "email": "REQUIRED_MAINTAINER_EMAIL",
                "affiliation": "REQUIRED_MAINTAINER_AFFILIATION",
                "orcid": None,
            }
        ],
        "project_urls": {
            "Homepage": "REQUIRED_HTTPS_HOMEPAGE",
            "Repository": "REQUIRED_HTTPS_GITHUB_REPOSITORY",
            "Issues": "REQUIRED_HTTPS_ISSUES_URL",
            "Documentation": "REQUIRED_HTTPS_DOCUMENTATION_URL",
        },
        "release_coordinates": {
            "github_repository_url": "REQUIRED_HTTPS_GITHUB_REPOSITORY",
            "github_release_tag": f"v{RELEASE_VERSION}",
            "github_release_url": "REQUIRED_HTTPS_GITHUB_RELEASE_URL",
            "pypi_project": "phaxis",
            "pypi_version": RELEASE_VERSION,
            "pypi_project_url": f"https://pypi.org/project/phaxis/{RELEASE_VERSION}/",
            "release_date": "REQUIRED_RELEASE_DATE_YYYY_MM_DD",
            "release_doi": "REQUIRED_RELEASE_DOI",
        },
        "rights": {
            "source_license_spdx": "Apache-2.0",
            "license_file_sha256": "REQUIRED_LICENSE_FILE_SHA256",
            "source_release_authorized": False,
            "model_weights_included": False,
            "images_included": False,
            "annotations_included": False,
            "separate_asset_rights_not_conferred": True,
        },
        "metadata_identity_sha256": "COMPUTE_AFTER_AUTHOR_VERIFICATION",
    }


def _generated_sources(
    staging: Path, *, gate: Mapping[str, Any], formal: bool
) -> dict[str, str]:
    release_metadata = gate.get("release_human_metadata")
    if release_metadata is not None and not isinstance(release_metadata, Mapping):
        raise SourceReleaseError("release human metadata gate payload is invalid")
    if formal and release_metadata is None:
        raise SourceReleaseError("formal release lacks sealed release human metadata")
    generated = {
        "pyproject.toml": _pyproject(
            formal=formal,
            release_metadata=release_metadata,
        ),
        "MANIFEST.in": _manifest_in(
            formal=formal,
            release_metadata=release_metadata is not None,
        ),
        "README.md": _readme(
            formal=formal,
            release_metadata=release_metadata,
        ),
        "NOTICE": _notice(),
        THIRD_PARTY_NOTICES_NAME: _third_party_notices(),
        THIRD_PARTY_INVENTORY_NAME: _pretty_json_text(_third_party_inventory()),
        SBOM_NAME: _pretty_json_text(_cyclonedx_sbom()),
        "source_release_pytest.py": _source_release_pytest_plugin(),
        **_community_files(
            formal=formal,
            release_metadata=release_metadata,
        ),
    }
    for relative, content in generated.items():
        _write_text(staging / relative, content)
    receipt_name = FORMAL_RECEIPT_NAME if formal else BLOCKED_RECEIPT_NAME
    receipt = dict(gate)
    receipt["release_mode"] = "formal" if formal else "blocked_development_staging"
    receipt["builder_override_used"] = not formal
    if not formal:
        receipt["warning"] = "BLOCKED DEVELOPMENT STAGING — DO NOT PUBLISH OR UPLOAD"
    _write_json(staging / receipt_name, receipt)
    if release_metadata is not None:
        metadata_name = RELEASE_HUMAN_METADATA_NAME
        metadata_payload = dict(release_metadata)
        metadata_origin = "generated:validated_release_human_metadata"
    else:
        metadata_name = RELEASE_HUMAN_METADATA_TEMPLATE_NAME
        metadata_payload = _release_human_metadata_template()
        metadata_origin = "generated:blocked_release_human_metadata_template"
    _write_json(staging / metadata_name, metadata_payload)
    return {
        **{relative: "generated:source_release_common" for relative in generated},
        receipt_name: "generated:formal_release_gate",
        metadata_name: metadata_origin,
    }


def _normalize_metadata(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda item: (len(item.parts), item.as_posix()), reverse=True)
    for path in paths:
        if path.is_file():
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass
        try:
            os.utime(path, (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)
        except (OSError, NotImplementedError):
            pass
    try:
        os.utime(root, (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)
    except (OSError, NotImplementedError):
        pass


def _manifest(staging: Path, origins: Mapping[str, str], *, formal: bool) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in staging.rglob("*") if item.is_file()),
        key=lambda item: _posix_relative(item, staging),
    ):
        relative = _posix_relative(path, staging)
        if relative == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "origin": origins[relative],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "distribution": "phaxis",
        "version": RELEASE_VERSION,
        "release_mode": "formal" if formal else "blocked_development_staging",
        "source_policy": "explicit_path_bounded_allowlist",
        "files": files,
        "tree_identity_sha256": _sha256_json(files),
    }


def _private_staging_path(destination: Path) -> Path:
    """Return a short sibling path without repeating the public tree name."""

    return destination.parent / f"{SOURCE_RELEASE_STAGING_PREFIX}{uuid.uuid4().hex}"


def build_source_release(
    *,
    project_root: Path,
    output: Path,
    allow_blocked_development_staging: bool = False,
    root_provider_receipt: Path | None = None,
    train399_candidate_manifest: Path | None = None,
    train399_selection_receipt: Path | None = None,
    train399_evaluation_receipt: Path | None = None,
    final_fusion_summary: Path | None = None,
    final_traits_summary: Path | None = None,
    release_human_metadata: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one new source tree at ``output``."""

    root = project_root.resolve()
    destination = output.resolve()
    if destination == root or root in destination.parents and destination.name in {"src", "scripts", "tests", "configs"}:
        raise SourceReleaseError("output may not replace a project source directory")
    if destination.exists():
        if not destination.is_dir():
            raise SourceReleaseError("output exists and is not a directory")
        if any(destination.iterdir()):
            raise SourceReleaseError("output directory must be new or empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    gate = inspect_formal_release_gate(
        project_root=root,
        root_provider_receipt=(root_provider_receipt.resolve() if root_provider_receipt else None),
        train399_candidate_manifest=(train399_candidate_manifest.resolve() if train399_candidate_manifest else None),
        train399_selection_receipt=(train399_selection_receipt.resolve() if train399_selection_receipt else None),
        train399_evaluation_receipt=(train399_evaluation_receipt.resolve() if train399_evaluation_receipt else None),
        final_fusion_summary=(final_fusion_summary.resolve() if final_fusion_summary else None),
        final_traits_summary=(final_traits_summary.resolve() if final_traits_summary else None),
        release_human_metadata=(release_human_metadata.resolve() if release_human_metadata else None),
    )
    formal = bool(gate["formal_release_allowed"])
    if not formal and not allow_blocked_development_staging:
        failed = ", ".join(record["code"] for record in gate["checks"] if not record["passed"])
        raise SourceReleaseError(
            "formal release gate is blocked; use --allow-blocked-development-staging "
            f"only for an unmistakably non-release tree. Failed checks: {failed}"
        )

    staging = _private_staging_path(destination)
    if staging.exists():
        raise SourceReleaseError(f"unexpected staging collision: {staging.name}")
    staging.mkdir()
    try:
        origins: dict[str, str] = {}
        for source, relative in collect_allowlisted_sources(root):
            target = _destination(staging, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            origins[relative] = f"project:{source.relative_to(root).as_posix()}"
        for evidence_spec in PROJECTED_EVIDENCE_SPECS:
            source_relative = evidence_spec.source
            destination_relative = evidence_spec.destination
            source = _destination(root, source_relative)
            if not source.is_file():
                raise SourceReleaseError(
                    f"projected evidence file is absent: {source_relative}"
                )
            if source.is_symlink():
                raise SourceReleaseError(
                    f"projected evidence file is a symlink: {source_relative}"
                )
            projected = _project_evidence_receipt(
                project_root=root,
                source=source,
                spec=evidence_spec,
            )
            _write_json(_destination(staging, destination_relative), projected)
            origins[destination_relative] = f"derived-project:{source_relative}"
        origins.update(_generated_sources(staging, gate=gate, formal=formal))
        manifest = _manifest(staging, origins, formal=formal)
        _write_json(staging / MANIFEST_NAME, manifest)
        _normalize_metadata(staging)
        verify_source_release(staging, project_root=root)
        if destination.exists():
            destination.rmdir()  # exact target was already proved empty above
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_manifest(root: Path, failures: list[str], project_root: Path | None) -> dict[str, Any] | None:
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = _read_object(manifest_path)
    except SourceReleaseError as error:
        failures.append(str(error))
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported source manifest schema")
    records = manifest.get("files")
    if not isinstance(records, list):
        failures.append("manifest files is not a list")
        return manifest
    record_paths = [record.get("path") for record in records if isinstance(record, dict)]
    if len(record_paths) != len(records) or len(set(record_paths)) != len(record_paths):
        failures.append("manifest paths are invalid or duplicated")
        return manifest
    if record_paths != sorted(record_paths):
        failures.append("manifest paths are not sorted")
    observed = sorted(
        _posix_relative(path, root)
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not _is_root_git_control_path(path, root)
    )
    if observed != sorted(record_paths):
        for relative in sorted(set(observed) - set(record_paths)):
            failures.append(f"unmanifested file: {relative}")
        for relative in sorted(set(record_paths) - set(observed)):
            failures.append(f"manifested file absent: {relative}")
    for record in records:
        if not isinstance(record, dict):
            continue
        relative = record.get("path")
        try:
            path = _destination(root, str(relative))
        except SourceReleaseError as error:
            failures.append(str(error))
            continue
        if not path.is_file():
            continue
        if path.stat().st_size != record.get("bytes"):
            failures.append(f"manifest byte-size mismatch: {relative}")
        if _sha256_file(path) != record.get("sha256"):
            failures.append(f"manifest SHA-256 mismatch: {relative}")
        origin = record.get("origin")
        if not isinstance(origin, str):
            failures.append(f"manifest origin missing: {relative}")
        elif project_root is not None and origin.startswith("project:"):
            source_relative = origin.removeprefix("project:")
            try:
                source = _destination(project_root, source_relative)
            except SourceReleaseError as error:
                failures.append(str(error))
                continue
            if not source.is_file():
                failures.append(f"authoritative source absent: {source_relative}")
            elif _sha256_file(source) != record.get("sha256"):
                failures.append(f"release file is stale relative to authority: {relative}")
    if project_root is not None:
        expected_origins = {
            destination: f"project:{source.relative_to(project_root).as_posix()}"
            for source, destination in collect_allowlisted_sources(project_root)
        }
        observed_origins = {
            str(record.get("path")): record.get("origin")
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("origin"), str)
            and record["origin"].startswith("project:")
        }
        for relative in sorted(set(expected_origins) - set(observed_origins)):
            failures.append(
                f"current allowlisted project source absent from release: {relative}"
            )
        for relative in sorted(set(observed_origins) - set(expected_origins)):
            failures.append(
                f"release contains a no-longer-allowlisted project source: {relative}"
            )
        for relative in sorted(set(expected_origins) & set(observed_origins)):
            if observed_origins[relative] != expected_origins[relative]:
                failures.append(f"manifest project origin changed: {relative}")
    if manifest.get("tree_identity_sha256") != _sha256_json(records):
        failures.append("manifest tree identity mismatch")
    return manifest


def _verify_packaging(
    root: Path, failures: list[str], *, formal: bool
) -> None:
    pyproject_path = root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        failures.append(f"invalid pyproject.toml: {error}")
        return
    if not isinstance(pyproject, Mapping):
        failures.append("pyproject.toml document root is not a table")
        return
    raw_project = pyproject.get("project")
    if not isinstance(raw_project, Mapping):
        failures.append("pyproject project table is absent or invalid")
        project: Mapping[str, Any] = {}
    else:
        project = raw_project
    if project.get("name") != "phaxis" or project.get("version") != RELEASE_VERSION:
        failures.append("pyproject distribution identity is not phaxis 1.0.0")
    scripts = project.get("scripts")
    if not isinstance(scripts, Mapping) or scripts.get("phaxis") != "phaxis.cli:main":
        failures.append("pyproject does not install the phaxis CLI")
    if project.get("license-files") != [
        "LICENSE",
        "src/phaxis/_vendor/tomli/LICENSE.txt",
    ]:
        failures.append("pyproject does not retain the vendored Tomli MIT license")
    expected_by_scope: dict[str, list[str]] = {}
    for dependency in THIRD_PARTY_DEPENDENCIES:
        for scope in dependency.scopes:
            expected_by_scope.setdefault(scope, []).append(dependency.requirement)
    build_system = pyproject.get("build-system")
    build_requires = (
        build_system.get("requires") if isinstance(build_system, Mapping) else None
    )
    core_dependencies = project.get("dependencies")
    observed_by_scope: dict[str, list[str]] = {}
    if isinstance(build_requires, list) and all(
        isinstance(item, str) for item in build_requires
    ):
        observed_by_scope["build-system"] = list(build_requires)
    if isinstance(core_dependencies, list) and all(
        isinstance(item, str) for item in core_dependencies
    ):
        observed_by_scope["core"] = list(core_dependencies)
    optional = project.get("optional-dependencies")
    if isinstance(optional, Mapping):
        for scope, requirements in optional.items():
            if isinstance(requirements, list) and all(
                isinstance(item, str) for item in requirements
            ):
                observed_by_scope[str(scope)] = list(requirements)
    if set(observed_by_scope) != set(expected_by_scope) or any(
        sorted(observed_by_scope.get(scope, [])) != sorted(requirements)
        for scope, requirements in expected_by_scope.items()
    ):
        failures.append(
            "pyproject dependency declarations differ from the third-party license inventory"
        )
    if formal:
        for field in ("authors", "maintainers"):
            people = project.get(field)
            if not isinstance(people, list) or not people:
                failures.append(f"formal pyproject lacks {field}")
        urls = project.get("urls")
        if not isinstance(urls, dict) or set(urls) != set(PROJECT_URL_LABELS):
            failures.append("formal pyproject lacks canonical project URLs")
    tool = pyproject.get("tool")
    setuptools = tool.get("setuptools") if isinstance(tool, Mapping) else None
    if not isinstance(setuptools, Mapping):
        setuptools = {}
    if setuptools.get("package-dir") != {"": "src"}:
        failures.append("setuptools src-layout package-dir is missing")
    package_data = setuptools.get("package-data")
    if not isinstance(package_data, Mapping) or package_data.get(
        "phaxis._vendor.tomli"
    ) != ["LICENSE.txt", "py.typed"]:
        failures.append("setuptools package-data omits the vendored Tomli license/typing marker")
    packages = setuptools.get("packages")
    find = packages.get("find") if isinstance(packages, Mapping) else None
    include = find.get("include", []) if isinstance(find, Mapping) else []
    if "phaxis" not in include or "phaxis.*" not in include:
        failures.append("setuptools package discovery does not include phaxis and subpackages")
    try:
        _validate_vendored_tomli(root)
    except SourceReleaseError as error:
        failures.append(str(error))
    for relative in ("src/phaxis/__init__.py", "src/phaxis/__main__.py", "src/phaxis/cli.py"):
        if not (root / relative).is_file():
            failures.append(f"required installable package file absent: {relative}")
    community = _community_files(
        formal=formal,
        release_metadata=None,
    )
    for relative in community:
        if not (root / relative).is_file():
            failures.append(f"required GitHub/PyPI community file absent: {relative}")
    for relative in PUBLIC_CARD_FILES:
        if not (root / relative).is_file():
            failures.append(f"required public model/data card absent: {relative}")
    supply_chain_expected = {
        "NOTICE": _notice(),
        THIRD_PARTY_NOTICES_NAME: _third_party_notices(),
        THIRD_PARTY_INVENTORY_NAME: _pretty_json_text(_third_party_inventory()),
        SBOM_NAME: _pretty_json_text(_cyclonedx_sbom()),
    }
    for relative, expected_content in supply_chain_expected.items():
        path = root / relative
        try:
            observed_content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            failures.append(
                f"required PHAxis supply-chain file is unreadable: {relative}: {error}"
            )
        else:
            if observed_content != expected_content:
                failures.append(
                    f"PHAxis supply-chain file is not canonical: {relative}"
                )
            if "RHPheno" in observed_content or "RHAxis 0." in observed_content:
                failures.append(
                    f"PHAxis supply-chain file contains legacy product metadata: {relative}"
                )
    citation_path = root / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8")
        if (
            'version: "1.0.0"' not in citation
            or f"title: {_toml_string(SOFTWARE_CITATION_TITLE)}" not in citation
        ):
            failures.append("CITATION.cff does not identify PHAxis 1.0.0")
    workflow_path = root / ".github/workflows/ci.yml"
    if workflow_path.is_file():
        workflow = workflow_path.read_text(encoding="utf-8")
        for action in ("actions/checkout", "actions/setup-python"):
            immutable_action = f"{action}@{GITHUB_ACTION_PINS[action]}"
            if immutable_action not in workflow:
                failures.append(f"CI workflow lacks immutable action pin: {immutable_action}")
        if "actions/checkout@v" in workflow or "actions/setup-python@v" in workflow:
            failures.append("CI workflow uses a mutable GitHub Action tag")
        for required_ci_contract in (
            "os: [ubuntu-latest, windows-latest]",
            'python-version: ["3.10", "3.11", "3.12"]',
            "python -B scripts/phaxis/verify_source_release.py .",
            "python -B -S scripts/phaxis/verify_source_release.py .",
            "python -B -I -S scripts/phaxis/verify_source_release.py .",
            "python -B -m pytest tests/phaxis -q",
            "python -B -m build",
            "python -B -m twine check dist/*",
            "-B -m pip install dist/*.whl",
            "-B -m pip check",
            "-B -m phaxis --help",
        ):
            if required_ci_contract not in workflow:
                failures.append(
                    f"CI workflow lacks required contract: {required_ci_contract}"
                )
    release_workflow_path = root / ".github/workflows/release.yml"
    if release_workflow_path.is_file():
        release_workflow = release_workflow_path.read_text(encoding="utf-8")
        action_uses = re.findall(
            r"(?m)^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)", release_workflow
        )
        if not action_uses:
            failures.append("release workflow does not invoke pinned actions")
        for action, reference in action_uses:
            if re.fullmatch(r"[0-9a-f]{40}", reference) is None:
                failures.append(
                    f"release workflow action is not full-SHA pinned: {action}@{reference}"
                )
            expected_reference = GITHUB_ACTION_PINS.get(action)
            if expected_reference is None or reference != expected_reference:
                failures.append(
                    f"release workflow action is not the canonical pin: {action}@{reference}"
                )
        for action, reference in GITHUB_ACTION_PINS.items():
            if f"{action}@{reference}" not in release_workflow:
                failures.append(
                    f"release workflow lacks immutable action pin: {action}@{reference}"
                )
        for required_release_contract in (
            "workflow_dispatch:",
            "refs/tags/v1.0.0",
            'PHAXIS_RELEASE_TAG: "v1.0.0"',
            '"git_commit": os.environ["GITHUB_SHA"]',
            "FORMAL_RELEASE_GATE_RECEIPT.json",
            "distribution_receipt.json",
            "clean_install_receipt.json",
            "release_finalization.json",
            'for command in distribution["commands"]:',
            'registry_reference = finalization["release_authority_registry_path"]',
            'environment:\n      name: pypi',
            "id-token: write",
            "attestations: true",
            "skip-existing: false",
            'assert authority["git_commit"] == os.environ["GITHUB_SHA"]',
            "gh release upload v1.0.0 PHAXIS_PUBLISH_PROVENANCE.json",
            "gh release edit v1.0.0",
            "--draft=false --verify-tag",
        ):
            if required_release_contract not in release_workflow:
                failures.append(
                    "release workflow lacks required publication contract: "
                    f"{required_release_contract}"
                )
        expected_enabled = (
            'PHAXIS_RELEASE_ENABLED: "true"'
            if formal
            else 'PHAXIS_RELEASE_ENABLED: "false"'
        )
        if expected_enabled not in release_workflow:
            failures.append("release workflow formal/blocked enablement is inconsistent")
        if formal and (
            "BLOCKED_DO_NOT_PUBLISH" in release_workflow
            or "BLOCKED/DO_NOT_PUBLISH" in release_workflow
        ):
            failures.append("formal release workflow contains blocked coordinates")
        predeclared_commit_variable = "PHAXIS_EXPECTED_" + "GIT_COMMIT"
        if predeclared_commit_variable in release_workflow:
            failures.append("release workflow predeclares the tag commit")
        if re.search(r"__[A-Z0-9_]+__", release_workflow):
            failures.append("release workflow contains unresolved action/authority markers")
    if formal:
        zenodo_path = root / ZENODO_METADATA_NAME
        try:
            zenodo = _read_object(zenodo_path)
            human_metadata = _read_object(root / RELEASE_HUMAN_METADATA_NAME)
        except SourceReleaseError as error:
            failures.append(str(error))
        else:
            coordinates = human_metadata.get("release_coordinates")
            authors = human_metadata.get("authors")
            creators = zenodo.get("creators")
            zenodo_ok = bool(
                isinstance(coordinates, Mapping)
                and zenodo.get("upload_type") == "software"
                and zenodo.get("version") == RELEASE_VERSION
                and zenodo.get("publication_date") == coordinates.get("release_date")
                and zenodo.get("doi") == coordinates.get("release_doi")
                and zenodo.get("license") == "Apache-2.0"
                and zenodo.get("access_right") == "open"
                and isinstance(authors, list)
                and isinstance(creators, list)
                and len(creators) == len(authors)
                and "conceptdoi" not in zenodo
                and "concept DOI" in str(zenodo.get("notes"))
                and "not inferred" in str(zenodo.get("notes"))
                and any(
                    isinstance(row, Mapping)
                    and row.get("identifier")
                    == coordinates.get("github_release_url")
                    for row in zenodo.get("related_identifiers", [])
                )
                and any(
                    isinstance(row, Mapping)
                    and row.get("identifier") == coordinates.get("pypi_project_url")
                    for row in zenodo.get("related_identifiers", [])
                )
                and not any(
                    isinstance(row, Mapping)
                    and "/commit/" in str(row.get("identifier", ""))
                    for row in zenodo.get("related_identifiers", [])
                )
            )
            if not zenodo_ok:
                failures.append(
                    ".zenodo.json does not preserve version/concept DOI semantics "
                    "and the release-human authority"
                )


def _verify_projected_evidence(
    root: Path,
    failures: list[str],
    *,
    project_root: Path | None,
    manifest: Mapping[str, Any] | None,
) -> None:
    observed_by_role: dict[str, Mapping[str, Any]] = {}
    records = manifest.get("files") if isinstance(manifest, Mapping) else None
    for evidence_spec in PROJECTED_EVIDENCE_SPECS:
        source_relative = evidence_spec.source
        destination_relative = evidence_spec.destination
        release_path = _destination(root, destination_relative)
        try:
            observed = _read_object(release_path)
        except SourceReleaseError as error:
            failures.append(f"{evidence_spec.evidence_role}: {error}")
            continue
        observed_by_role[evidence_spec.evidence_role] = observed
        if not _projected_evidence_payload_ok(evidence_spec, observed):
            failures.append(
                f"{evidence_spec.evidence_role}: projected evidence semantic validation failed"
            )
        metadata_ok = _release_projection_metadata_ok(evidence_spec, observed)
        if not metadata_ok:
            failures.append(
                f"{evidence_spec.evidence_role}: release projection metadata is invalid"
            )
        matching_records = (
            [
                item
                for item in records
                if isinstance(item, dict)
                and item.get("path") == destination_relative
            ]
            if isinstance(records, list)
            else []
        )
        if (
            len(matching_records) != 1
            or matching_records[0].get("origin")
            != f"derived-project:{source_relative}"
        ):
            failures.append(
                f"{evidence_spec.evidence_role}: release projection origin is invalid"
            )
        if project_root is not None:
            source = _destination(project_root, source_relative)
            try:
                expected = _project_evidence_receipt(
                    project_root=project_root,
                    source=source,
                    spec=evidence_spec,
                )
            except SourceReleaseError as error:
                failures.append(f"{evidence_spec.evidence_role}: {error}")
            else:
                if observed != expected:
                    failures.append(
                        f"{evidence_spec.evidence_role}: release projection is stale "
                        "relative to authority"
                    )

    if set(observed_by_role) == {
        spec.evidence_role for spec in PROJECTED_EVIDENCE_SPECS
    }:
        historical = observed_by_role[
            "pre_amendment_biological_equivalence_historical_baseline"
        ]
        amendment = observed_by_role[
            "h11_raw_median_contract_amendment_current"
        ]
        historical_projection = historical.get("release_projection")
        baseline = amendment.get("pre_amendment_baseline")
        if not (
            isinstance(historical_projection, Mapping)
            and isinstance(baseline, Mapping)
            and baseline.get("authority_sha256")
            == historical_projection.get("source_receipt_sha256")
            == "10314999405b66cd5f1b6042cfc18bf6e3625e64b2aeb6ee9d7d1e04ef7137e7"
            and baseline.get("authority_path")
            == PROJECTED_EVIDENCE_SPECS[0].source
        ):
            failures.append(
                "historical-to-current H11 amendment evidence cross-binding is invalid"
            )


def _verify_amp_amendment_evidence(
    root: Path,
    failures: list[str],
    *,
    manifest: Mapping[str, Any] | None,
) -> None:
    """Verify the narrowly mapped numerical amendment without shipping models."""

    release_path = _destination(root, AMP_AMENDMENT_DESTINATION)
    try:
        payload = _read_object(release_path)
    except SourceReleaseError as error:
        failures.append(str(error))
        return

    failure = payload.get("superseded_failed_attempt")
    root_cause = payload.get("root_cause")
    policy = payload.get("amended_numeric_policy")
    scientific = payload.get("unchanged_scientific_contract")
    legacy = payload.get("legacy_zero_retry_normalization")
    implementation = payload.get("implementation")
    semantic_ok = bool(
        payload.get("schema_version")
        == "PHAxis-StageB-train399-AMP-backward-amendment-1.0"
        and payload.get("status")
        == "applied_before_authoritative_seed3_optimizer_trajectory"
        and isinstance(failure, Mapping)
        and failure.get("seed") == 2026082803
        and failure.get("completed_epoch") == 0
        and failure.get("global_step_at_failure") == 6
        and failure.get("authoritative_checkpoint_created") is False
        and failure.get("blind_images_used") == 0
        and _is_sha256(failure.get("failure_receipt_sha256"))
        and isinstance(root_cause, Mapping)
        and root_cause.get("failure_class") == "fp16_scaled_backward_overflow"
        and root_cause.get("loss_was_finite") is True
        and root_cause.get("oom") is False
        and root_cause.get("data_or_target_nonfinite") is False
        and isinstance(policy, Mapping)
        and policy.get("contract_policy_string")
        == "fail_closed_no_optimizer_step_skip"
        and policy.get("initial_scale") == 1024.0
        and policy.get("backoff_factor") == 0.5
        and policy.get("maximum_backward_retries_per_batch") == 16
        and policy.get("same_forward_graph_replayed") is True
        and policy.get("forward_recomputed") is False
        and policy.get("batchnorm_buffers_updated_again") is False
        and policy.get("rng_or_data_order_advanced") is False
        and policy.get("optimizer_step_before_finite_unscaled_gradient") is False
        and policy.get("optimizer_steps_skipped_due_nonfinite_gradients") == 0
        and policy.get("failure_after_retry_exhaustion") is True
        and isinstance(scientific, Mapping)
        and scientific.get("training_images") == 399
        and scientific.get("excluded_qcdevelopment_images") == 44
        and scientific.get("family_key_overlap") == 0
        and scientific.get("architecture_changed") is False
        and scientific.get("loss_objective_changed") is False
        and scientific.get("augmentation_or_sampler_changed") is False
        and scientific.get(
            "validation_used_for_gradient_early_stopping_or_retry"
        )
        is False
        and scientific.get("blind_images_used") == 0
        and isinstance(legacy, Mapping)
        and legacy.get("seeds") == [2026082801, 2026082802]
        and legacy.get("normalized_amp_backward_retry_count") == [0, 0]
        and isinstance(implementation, Mapping)
        and _is_sha256(implementation.get("training_source_sha256"))
    )
    if not semantic_ok:
        failures.append("Stage-B AMP backward amendment semantics are invalid")

    training_source = root / "src/phaxis/hair_stageb/training.py"
    if (
        not training_source.is_file()
        or not isinstance(implementation, Mapping)
        or _sha256_file(training_source)
        != implementation.get("training_source_sha256")
    ):
        failures.append(
            "Stage-B AMP amendment does not bind the released training source"
        )

    records = manifest.get("files") if isinstance(manifest, Mapping) else None
    record = (
        next(
            (
                item
                for item in records
                if isinstance(item, Mapping)
                and item.get("path") == AMP_AMENDMENT_DESTINATION
            ),
            None,
        )
        if isinstance(records, list)
        else None
    )
    if not isinstance(record, Mapping) or record.get("origin") != (
        f"project:{AMP_AMENDMENT_SOURCE}"
    ):
        failures.append("Stage-B AMP amendment manifest origin is invalid")


def _verify_generated_metadata(root: Path, failures: list[str], *, formal: bool) -> None:
    receipt_name = FORMAL_RECEIPT_NAME if formal else BLOCKED_RECEIPT_NAME
    try:
        receipt = _read_object(root / receipt_name)
    except SourceReleaseError as error:
        failures.append(str(error))
        receipt = {}
    release_metadata = receipt.get("release_human_metadata")
    if release_metadata is not None and not isinstance(release_metadata, dict):
        failures.append("gate receipt release human metadata is invalid")
        release_metadata = None
    if formal and release_metadata is None:
        failures.append("formal generated metadata lacks release human authority")
    expected = {
        "pyproject.toml": _pyproject(
            formal=formal,
            release_metadata=release_metadata,
        ),
        "MANIFEST.in": _manifest_in(
            formal=formal,
            release_metadata=release_metadata is not None,
        ),
        "README.md": _readme(
            formal=formal,
            release_metadata=release_metadata,
        ),
        "NOTICE": _notice(),
        THIRD_PARTY_NOTICES_NAME: _third_party_notices(),
        THIRD_PARTY_INVENTORY_NAME: _pretty_json_text(_third_party_inventory()),
        SBOM_NAME: _pretty_json_text(_cyclonedx_sbom()),
        "source_release_pytest.py": _source_release_pytest_plugin(),
        **_community_files(
            formal=formal,
            release_metadata=release_metadata,
        ),
    }
    for relative, content in expected.items():
        path = root / relative
        try:
            observed = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            failures.append(
                f"generated release metadata is unreadable: {relative}: {error}"
            )
            continue
        if observed != content:
            failures.append(f"generated release metadata is not canonical: {relative}")
    if release_metadata is not None:
        metadata_name = RELEASE_HUMAN_METADATA_NAME
        expected_metadata = release_metadata
        forbidden_metadata = RELEASE_HUMAN_METADATA_TEMPLATE_NAME
    else:
        metadata_name = RELEASE_HUMAN_METADATA_TEMPLATE_NAME
        expected_metadata = _release_human_metadata_template()
        forbidden_metadata = RELEASE_HUMAN_METADATA_NAME
    try:
        observed_metadata = _read_object(root / metadata_name)
    except SourceReleaseError as error:
        failures.append(str(error))
    else:
        if observed_metadata != expected_metadata:
            failures.append("generated release human metadata is not canonical")
    if release_metadata is not None:
        metadata_checks: list[dict[str, Any]] = []
        validated = _release_human_metadata_gate(
            path=root / metadata_name,
            project_root=root,
            checks=metadata_checks,
        )
        if validated is None or not all(
            check["passed"] for check in metadata_checks
        ):
            failures.append("generated release human metadata fails its public gate")
        artifacts = receipt.get("artifacts")
        artifact = (
            artifacts.get("release_human_metadata")
            if isinstance(artifacts, dict)
            else None
        )
        if not isinstance(artifact, dict) or artifact.get(
            "metadata_identity_sha256"
        ) != release_metadata.get("metadata_identity_sha256"):
            failures.append("release human metadata artifact identity is unbound")
    if (root / forbidden_metadata).exists():
        failures.append("release tree contains conflicting human metadata files")


def _verify_public_phenotype_catalog(root: Path, failures: list[str]) -> None:
    """Fail closed when the plant-facing catalogue drifts or leaks internals."""

    public_paths = (
        Path("README.md"),
        Path("MODEL_CARD.md"),
        Path("DATA_CARD.md"),
        Path("docs/phaxis/TRAIT_CONTRACT_CN.md"),
        Path("docs/phaxis/USER_GUIDE.md"),
    )
    public_text: dict[Path, str] = {}
    for relative in public_paths:
        path = root / relative
        try:
            public_text[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            failures.append(
                f"public phenotype document is absent or unreadable: {relative.as_posix()}: {error}"
            )

    if len(public_text) != len(public_paths):
        return

    catalog_relative = Path("docs/phaxis/TRAIT_CONTRACT_CN.md")
    forbidden_names = (
        "rhaxiscc",
        "rhaxis_nextgen",
        "hybrid-max",
        "stage-b",
        "stage b",
    )
    for relative, text in public_text.items():
        folded = text.casefold()
        for token in forbidden_names:
            if token in folded:
                failures.append(
                    f"public phenotype document exposes internal name {token!r}: "
                    f"{relative.as_posix()}"
                )
        if re.search(r"(?<![a-z0-9_])v2(?:\.0)?(?![a-z0-9_])", folded):
            failures.append(
                f"public phenotype document exposes a non-public v2 label: {relative.as_posix()}"
            )
        if relative != catalog_relative and re.search(
            r"(?<![a-z0-9_])r(?:1[6-9]|[2-9][0-9])(?![a-z0-9_])", folded
        ):
            failures.append(
                f"public user document exposes an internal run revision: {relative.as_posix()}"
            )

    catalog = public_text[catalog_relative]
    folded_catalog = catalog.casefold()
    for token in ("outputs/", "models/", "scripts/phaxis/"):
        if token in folded_catalog:
            failures.append(
                f"plant-facing catalogue exposes an internal project path {token!r}"
            )
    compact_catalog = " ".join(catalog.split())
    for phrase in (
        "five-member root-hair identity/count expert",
        "32 canonical image-derived descriptors",
        "The root-cap representation is exactly one distal/root-cap point",
        "no root-cap region",
        "H06、H07、H13",
        "`[1,4) mm`",
        "does not report 82 phenotypes",
    ):
        if phrase not in compact_catalog:
            failures.append(f"plant-facing catalogue lacks required wording: {phrase!r}")
    for usage_token in (
        "--model-contract <official-contract.json>",
        "read_model_contract_authority(\"<official-contract.json>\")",
        "model_contract_proposal=authority.receipt_fields()",
        "model_contract_public_identity=authority.public_identity_fields()",
    ):
        if usage_token not in compact_catalog:
            failures.append(
                "plant-facing catalogue usage example drifted from the public API: "
                f"{usage_token!r}"
            )

    try:
        contract = _read_object(root / "configs/phaxis/v1_0/trait_contract.json")
    except SourceReleaseError as error:
        failures.append(str(error))
        return
    roots = contract.get("primary_root_traits")
    hairs = contract.get("root_hair_traits")
    if not isinstance(roots, list) or not isinstance(hairs, list):
        failures.append("trait contract lacks primary-root/root-hair row lists")
        return
    rows = [*roots, *hairs]
    if len(roots) != 19 or len(hairs) != 13 or len(rows) != 32:
        failures.append("trait contract/catalog is not exactly 19 root + 13 hair descriptors")
        return
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
    for row in rows:
        if not isinstance(row, dict):
            failures.append("trait contract contains a non-object descriptor row")
            continue
        try:
            marker = (
                f"| {row['id']} | {row['display_name_cn']}<br>{row['display_name_en']} | "
                f"`{row['field']}` | {unit_cells[row['unit']]} |"
            )
        except (KeyError, TypeError) as error:
            failures.append(f"trait contract row cannot bind the bilingual catalogue: {error}")
            continue
        if catalog.count(marker) != 1:
            failures.append(
                f"plant-facing catalogue does not contain exactly one row for {row.get('id')!r}"
            )

    for relative in (
        Path("README.md"),
        Path("MODEL_CARD.md"),
        Path("DATA_CARD.md"),
        Path("docs/phaxis/USER_GUIDE.md"),
    ):
        compact = " ".join(public_text[relative].split())
        if "32 canonical" not in compact or "does not report 82 phenotypes" not in compact:
            failures.append(
                f"public card does not distinguish 32 descriptors from the 82-column schema: "
                f"{relative.as_posix()}"
            )

    for relative in (Path("README.md"), Path("docs/phaxis/USER_GUIDE.md")):
        compact = " ".join(public_text[relative].split())
        for required_command in (
            "phaxis analyze --manifest workflow.json --output analysis-output",
            "phaxis export-traits",
            "--model-contract official-contract.json",
            "phaxis --version",
        ):
            if required_command not in compact:
                failures.append(
                    "public usage document lacks a reusable CLI contract: "
                    f"{relative.as_posix()}: {required_command!r}"
                )


def _verify_boundary(root: Path, failures: list[str], *, formal: bool) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_root_git_control_path(path, root):
            continue
        relative = path.relative_to(root)
        if path.is_symlink():
            failures.append(f"source release may not contain symlinks: {relative.as_posix()}")
        folded_parts = {part.casefold() for part in relative.parts}
        if PROHIBITED_PATH_PARTS.intersection(folded_parts):
            failures.append(f"prohibited generated/data/legacy path: {relative.as_posix()}")
        if path.suffix.casefold() in PROHIBITED_SUFFIXES:
            failures.append(f"prohibited image/model/data artifact: {relative.as_posix()}")
        if any(part.casefold().endswith(".egg-info") for part in relative.parts):
            failures.append(f"generated packaging metadata present: {relative.as_posix()}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            failures.append(f"non-UTF-8 release payload: {relative.as_posix()}")
        except OSError as error:
            failures.append(
                f"unreadable release payload: {relative.as_posix()}: {error}"
            )
        else:
            markers = _absolute_host_path_markers(text)
            if markers:
                failures.append(
                    "host-absolute path content "
                    f"({', '.join(markers)}): {relative.as_posix()}"
                )
    for relative in ("src/rhaxis_nextgen", "src/rhizoweave", "scripts/analyze_six_condition_hybrid_max.py"):
        if (root / relative).exists():
            failures.append(f"legacy/non-PHAxis implementation copied: {relative}")
    for relative in UNCOMPILED_MANUSCRIPT_FILES:
        if (root / relative).exists():
            failures.append(
                "uncompiled manuscript template copied into public source: "
                f"{relative}"
            )
    evaluator = root / "scripts/phaxis/evaluate_stageb_train399_qcdev44.py"
    if formal and evaluator.is_file():
        text = evaluator.read_text(encoding="utf-8")
        if any(
            token in text
            for token in ("--rhaxiscc-code", "from rhaxiscc", "import rhaxiscc", "args.rhaxiscc_code")
        ):
            failures.append("formal tree evaluator depends on an external RHAxiscc source path")
    parity_path = root / "evidence/evaluation_metric_parity_audit.json"
    if not parity_path.is_file():
        failures.append("project-owned evaluation metric parity receipt is absent")
    else:
        try:
            parity = _read_object(parity_path)
        except SourceReleaseError as error:
            failures.append(str(error))
        else:
            reference = parity.get("reference")
            parity_ok = (
                parity.get("schema_version")
                == "PHAxis-RHAxiscc-evaluation-metric-parity-audit-1.0"
                and parity.get("status") == "passed"
                and parity.get("production_evaluator_external_code_dependency") is False
                and parity.get("blind_images_used") == 0
                and parity.get("strict_presence_mismatches") == 0
                and parity.get("evaluate_image_maximum_absolute_numeric_difference") == 0.0
                and parity.get("prf_maximum_absolute_numeric_difference") == 0.0
                and isinstance(reference, dict)
                and all(_is_sha256(value) for value in reference.values())
            )
            if not parity_ok:
                failures.append("project-owned evaluation metric parity receipt is not a zero-difference pass")
    for evidence_spec in PROJECTED_EVIDENCE_SPECS:
        evidence_path = _destination(root, evidence_spec.destination)
        if not evidence_path.is_file() or evidence_path.is_symlink():
            failures.append(
                f"{evidence_spec.evidence_role}: required projected evidence is absent"
            )
            continue
        try:
            evidence = _read_object(evidence_path)
        except SourceReleaseError as error:
            failures.append(f"{evidence_spec.evidence_role}: {error}")
            continue
        if not _projected_evidence_payload_ok(evidence_spec, evidence):
            failures.append(
                f"{evidence_spec.evidence_role}: projected evidence is not a "
                "blind-clean release-eligible pass"
            )
            continue
        if evidence_spec.validator_kind == "h11_raw_median_amendment":
            implementation = evidence.get("implementation_sha256")
            locked = evidence.get("locked_inputs")
            source_bindings = {
                "audit_producer": (
                    "scripts/phaxis/audit_stage22_h11_raw_median_amendment.py"
                ),
                "audit_test": "tests/phaxis/test_h11_raw_median_amendment_audit.py",
                "biological_analysis": "src/phaxis/biological_analysis.py",
                "biological_analysis_wrapper": (
                    "scripts/phaxis/analyze_biological_cohorts.py"
                ),
                "multitrait_atlas": "src/phaxis/multitrait_atlas.py",
                "publication_figure_input_builder": (
                    "scripts/phaxis/build_publication_figure_inputs.py"
                ),
            }
            if not isinstance(implementation, Mapping):
                failures.append("H11 amendment implementation bindings are absent")
            else:
                for field, relative in source_bindings.items():
                    path = _destination(root, relative)
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or implementation.get(field) != _sha256_file(path)
                    ):
                        failures.append(
                            f"H11 amendment source binding differs: {field}"
                        )
            model_spec_path = root / "configs/phaxis/v1_0/biological_model_spec.json"
            if not (
                isinstance(locked, Mapping)
                and model_spec_path.is_file()
                and locked.get("model_spec_sha256") == _sha256_file(model_spec_path)
            ):
                failures.append("H11 amendment model-spec binding differs")


def verify_source_release(
    release_root: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify exact manifest closure, authority freshness, boundary, and packaging."""

    root = release_root.resolve()
    if not root.is_dir():
        raise SourceReleaseError(f"source release directory is absent: {root}")
    failures: list[str] = []
    authority = project_root.resolve() if project_root is not None else None
    manifest = _verify_manifest(root, failures, authority)
    release_mode = manifest.get("release_mode") if manifest else None
    formal = release_mode == "formal"
    if formal:
        receipt_path = root / FORMAL_RECEIPT_NAME
        if not receipt_path.is_file() or (root / BLOCKED_RECEIPT_NAME).exists():
            failures.append("formal tree lacks its sole formal gate receipt")
        else:
            receipt = _read_object(receipt_path)
            if receipt.get("status") != "passed" or receipt.get("formal_release_allowed") is not True:
                failures.append("formal gate receipt is not passed")
    elif release_mode == "blocked_development_staging":
        receipt_path = root / BLOCKED_RECEIPT_NAME
        if not receipt_path.is_file() or (root / FORMAL_RECEIPT_NAME).exists():
            failures.append("blocked tree lacks its sole blocked receipt")
        else:
            receipt = _read_object(receipt_path)
            if (
                receipt.get("status") != "blocked"
                or receipt.get("formal_release_allowed") is not False
                or receipt.get("builder_override_used") is not True
                or "DO NOT PUBLISH" not in str(receipt.get("warning"))
            ):
                failures.append("blocked receipt is not unmistakably fail-closed")
    else:
        failures.append("manifest release_mode is invalid")
    _verify_packaging(root, failures, formal=formal)
    _verify_projected_evidence(
        root,
        failures,
        project_root=authority,
        manifest=manifest,
    )
    _verify_amp_amendment_evidence(root, failures, manifest=manifest)
    _verify_generated_metadata(root, failures, formal=formal)
    _verify_boundary(root, failures, formal=formal)
    _verify_public_phenotype_catalog(root, failures)
    if failures:
        raise SourceReleaseError("source release verification failed:\n- " + "\n- ".join(failures))
    return {
        "schema_version": "PHAxis-source-release-verification-2.0",
        "status": "passed",
        "release_mode": release_mode,
        "files": len(manifest["files"]) if manifest else 0,
        "tree_identity_sha256": manifest.get("tree_identity_sha256") if manifest else None,
        "phaxis_cli_entry_point": "phaxis.cli:main",
        "phaxis_package_included": True,
        "prohibited_artifacts": 0,
    }


__all__ = [
    "BLOCKED_RECEIPT_NAME",
    "FORMAL_RECEIPT_NAME",
    "MANIFEST_NAME",
    "SourceReleaseError",
    "build_source_release",
    "collect_allowlisted_sources",
    "inspect_formal_release_gate",
    "verify_source_release",
]
