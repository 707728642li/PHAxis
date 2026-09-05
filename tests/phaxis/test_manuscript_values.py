from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "phaxis"))

import build_manuscript_values as builder  # noqa: E402
import build_manuscript_evidence_manifest as evidence_builder  # noqa: E402
import build_publication_figure_inputs as figure_inputs_builder  # noqa: E402
import phaxis.manuscript_values as values_core  # noqa: E402
from phaxis.evaluation_metrics import concordance_correlation  # noqa: E402
from phaxis.hair_attachment_assurance import (  # noqa: E402
    build_hair_attachment_assurance,
)
from phaxis.manuscript_values import (  # noqa: E402
    BIOLOGICAL_ACQUISITION_TOKENS,
    BuildContext,
    EVIDENCE_ARTIFACT_ROLES,
    FIGURE_PROVENANCE_ROLES,
    FIGURE_RESOURCE_ROLES,
    FIGURE_SOURCE_INPUT_ROLES,
    FileSource,
    HUMAN_METADATA_SCHEMA,
    HUMAN_METADATA_TOKENS,
    HumanMetadataError,
    JsonSource,
    ManuscriptValuesError,
    assemble_values_payload,
    build_token_source_contract,
    publish_json_no_overwrite,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_human_metadata,
    validate_values_payload,
    validate_wt_secondary_source_inputs,
)
from phaxis.manuscript_contract import (  # noqa: E402
    ABSTRACT_WORD_LIMIT,
    abstract_word_count,
    require_abstract_within_limit,
)
from phaxis.root_trait_assurance import (  # noqa: E402
    ROOT_TRAIT_ASSURANCE_TOKENS,
    ROOT_TRAIT_FAMILY_BY_FIELD,
    ROOT_TRAIT_PREDICTION_DEFINITION,
    ROOT_TRAIT_REFERENCE_DEFINITION,
    RootTraitAssuranceError,
    build_root_trait_assurance,
)
from phaxis.root_continuity_assurance import (  # noqa: E402
    build_root_continuity_assurance,
)
from phaxis.narrative_decision import (  # noqa: E402
    NarrativeDecisionError,
    build_narrative_decision,
)
from tests.phaxis.test_supplementary_table_data_bundle import (  # noqa: E402
    _wt_secondary_authorities,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_ccc_derivations_match_their_declared_authorities() -> None:
    observed = np.asarray([1.0, 2.0, 4.0, 8.0])
    predicted = np.asarray([2.0, 2.5, 4.5, 9.5])
    count_ccc = float(builder._ccc_rows(observed[None, :], predicted[None, :])[0])
    assurance_ccc = float(
        builder._ccc_sample_rows(observed[None, :], predicted[None, :])[0]
    )
    assert count_ccc == concordance_correlation(observed, predicted)
    assert assurance_ccc == figure_inputs_builder._ccc(observed, predicted)
    assert count_ccc != assurance_ccc


def test_values_routes_match_the_frozen_evidence_and_figure_contracts() -> None:
    assert EVIDENCE_ARTIFACT_ROLES == evidence_builder.ROLE_ORDER
    assert FIGURE_RESOURCE_ROLES == figure_inputs_builder.RESOURCE_ROLES
    assert FIGURE_RESOURCE_ROLES == evidence_builder.FIGURE_RESOURCE_ROLES
    assert FIGURE_SOURCE_INPUT_ROLES == evidence_builder.FIGURE_SOURCE_INPUT_ROLES
    assert FIGURE_PROVENANCE_ROLES == evidence_builder.FIGURE_PROVENANCE_ROLES


def _sealed_json(
    root: Path,
    role: str,
    payload: dict,
    identity_field: str,
) -> JsonSource:
    payload = deepcopy(payload)
    payload[identity_field] = sha256_json(payload)
    path = _write_json(root / f"{role}.json", payload)
    raw = path.read_bytes()
    return JsonSource(
        role=role,
        path=path,
        raw=raw,
        payload=payload,
        file_sha256=sha256_bytes(raw),
        logical_identity_sha256=payload[identity_field],
    )


def _plain_json(root: Path, role: str, payload: dict) -> JsonSource:
    path = _write_json(root / f"{role}.json", payload)
    raw = path.read_bytes()
    return JsonSource(
        role=role,
        path=path,
        raw=raw,
        payload=payload,
        file_sha256=sha256_bytes(raw),
    )


def _csv(root: Path, role: str, frame: pd.DataFrame, container: str) -> FileSource:
    path = root / f"{role}.csv"
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return FileSource(
        role=role,
        path=path,
        file_sha256=sha256_file(path),
        container_identity_sha256=container,
    )


def _json_file(root: Path, role: str, payload: dict, container: str) -> FileSource:
    path = _write_json(root / f"{role}.json", payload)
    return FileSource(
        role=role,
        path=path,
        file_sha256=sha256_file(path),
        container_identity_sha256=container,
    )


def _human_values() -> dict[str, str]:
    values = {token: "Author-verified external statement." for token in HUMAN_METADATA_TOKENS}
    for token in list(values):
        if token.endswith("_URL"):
            values[token] = f"https://example.org/{token.casefold()}"
        elif token.endswith("_DOI"):
            values[token] = "10.1234/phaxis.test"
    values["PHAXIS_RELEASE_TAG"] = "v1.0.0"
    values["PHAXIS_SOFTWARE_LICENSE"] = "Apache-2.0"
    values["PHAXIS_MODEL_LICENSE"] = "CC-BY-4.0"
    values["HUMANCURATED443_LICENSE"] = "CC-BY-4.0"
    return values


def _source_release_fixture(
    root: Path,
    *,
    pyproject_repository: str | None = None,
    citation_doi: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    human = _human_values()
    license_path = root / "LICENSE"
    license_path.write_text("Apache License 2.0 fixture\n", encoding="utf-8")
    pyproject_path = root / "pyproject.toml"
    pyproject_path.write_text(
        "[project]\n"
        'name = "phaxis"\n'
        'version = "1.0.0"\n'
        'license = "Apache-2.0"\n\n'
        "[project.urls]\n"
        f'Repository = "{pyproject_repository or human["PHAXIS_REPOSITORY_URL"]}"\n',
        encoding="utf-8",
    )
    citation_path = root / "CITATION.cff"
    citation_path.write_text(
        "cff-version: 1.2.0\n"
        'title: "PHAxis fixture"\n'
        'version: "1.0.0"\n'
        "license: Apache-2.0\n"
        f'repository-code: "{human["PHAXIS_REPOSITORY_URL"]}"\n'
        f'doi: "{citation_doi or human["PHAXIS_RELEASE_DOI"]}"\n',
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "PHAxis-release-human-metadata-1.3",
        "status": "author_verified_release_authority",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "distribution": "phaxis",
        "project_urls": {"Repository": human["PHAXIS_REPOSITORY_URL"]},
        "release_coordinates": {
            "github_repository_url": human["PHAXIS_REPOSITORY_URL"],
            "github_release_tag": human["PHAXIS_RELEASE_TAG"],
            "release_doi": human["PHAXIS_RELEASE_DOI"],
        },
        "rights": {
            "source_license_spdx": human["PHAXIS_SOFTWARE_LICENSE"],
            "source_release_authorized": True,
            "license_file_sha256": sha256_file(license_path),
        },
    }
    metadata["metadata_identity_sha256"] = sha256_json(metadata)
    metadata_path = _write_json(root / "RELEASE_HUMAN_METADATA.json", metadata)
    records = []
    for path in (metadata_path, license_path, pyproject_path, citation_path):
        records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "origin": "generated:test",
            }
        )
    manifest = {
        "schema_version": "PHAxis-source-release-manifest-2.0",
        "distribution": "phaxis",
        "version": "1.0.0",
        "release_mode": "formal",
        "source_policy": "explicit_path_bounded_allowlist",
        "files": records,
        "tree_identity_sha256": sha256_json(records),
    }
    return _write_json(root / "SOURCE_MANIFEST.json", manifest)


def test_source_release_coordinates_cross_bind_metadata_pyproject_and_citation(
    tmp_path: Path,
) -> None:
    manifest_path = _source_release_fixture(tmp_path / "source-release")
    manifest, metadata, binding = values_core._load_source_release_authority(
        manifest_path,
        human_values=_human_values(),
    )
    assert manifest.logical_identity_sha256 == manifest.payload["tree_identity_sha256"]
    assert metadata.logical_identity_sha256 == metadata.payload["metadata_identity_sha256"]
    assert binding["repository_url"] == _human_values()["PHAXIS_REPOSITORY_URL"]
    assert binding["release_tag"] == "v1.0.0"
    assert binding["version"] == "1.0.0"
    assert binding["release_doi"] == _human_values()["PHAXIS_RELEASE_DOI"]
    assert binding["software_license"] == "Apache-2.0"
    assert "git_commit" not in binding


@pytest.mark.parametrize(
    ("token", "value", "message"),
    [
        ("PHAXIS_REPOSITORY_URL", "https://example.org/wrong", "repository URL differs"),
        ("PHAXIS_RELEASE_TAG", "v1.0.1", "release tag/version differs"),
        ("PHAXIS_RELEASE_DOI", "10.1234/wrong", "release DOI differs"),
        ("PHAXIS_SOFTWARE_LICENSE", "MIT", "software license differs"),
    ],
)
def test_source_release_rejects_manuscript_coordinate_drift(
    tmp_path: Path,
    token: str,
    value: str,
    message: str,
) -> None:
    manifest_path = _source_release_fixture(tmp_path / "source-release")
    human = _human_values()
    human[token] = value
    with pytest.raises(ManuscriptValuesError, match=message):
        values_core._load_source_release_authority(
            manifest_path,
            human_values=human,
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"pyproject_repository": "https://example.org/wrong"}, "pyproject public coordinates differ"),
        ({"citation_doi": "10.1234/wrong"}, "CITATION.cff public coordinates differ"),
    ],
)
def test_source_release_rejects_packaging_coordinate_drift(
    tmp_path: Path,
    fixture_kwargs: dict[str, str],
    message: str,
) -> None:
    manifest_path = _source_release_fixture(
        tmp_path / "source-release",
        **fixture_kwargs,
    )
    with pytest.raises(ManuscriptValuesError, match=message):
        values_core._load_source_release_authority(
            manifest_path,
            human_values=_human_values(),
        )


@lru_cache(maxsize=1)
def _component_assurance_fixture() -> tuple[dict, dict]:
    source_units = [f"qc-{index:02d}" for index in range(44)]
    root_records = []
    hair_records = []
    for index, source_unit in enumerate(source_units):
        source_sha = sha256_json(["qc-image", index])
        endpoint = 100.0 if index % 4 == 0 else 72.0 + index % 7
        root_records.append(
            {
                "pair_type": "primary_root_continuity",
                "source_unit": source_unit,
                "source_image_sha256": source_sha,
                "coordinate_space": "physical_um_xy",
                "reference_axis_definition": figure_inputs_builder._root_continuity.ROOT_CONTINUITY_REFERENCE_DEFINITION,
                "prediction_axis_definition": figure_inputs_builder._root_continuity.ROOT_CONTINUITY_PREDICTION_DEFINITION,
                "reference_axis_artifact_sha256": sha256_json(
                    ["root-reference", index]
                ),
                "prediction_axis_artifact_sha256": sha256_json(
                    ["root-prediction", index]
                ),
                "reference_axis_xy_um": [[0.0, 0.0], [100.0, 0.0]],
                "predicted_axis_components_xy_um": [
                    [[0.0, 0.0], [endpoint, 0.0]]
                ],
            }
        )
        offset = float(index % 5)
        hair_records.append(
            {
                "pair_type": "hair_attachment",
                "source_unit": source_unit,
                "source_image_sha256": source_sha,
                "coordinate_space": "physical_um_xy",
                "polyline_orientation": "attachment_to_visible_distal_endpoint",
                "annotation_artifact_sha256": sha256_json(
                    ["hair-reference", index]
                ),
                "prediction_artifact_sha256": sha256_json(
                    ["hair-prediction", index]
                ),
                "annotated_polylines_xy_um": [
                    [[0.0, 0.0], [0.0, 30.0]],
                    [[20.0, 0.0], [20.0, 28.0]],
                ],
                "predicted_polylines_xy_um": [
                    [[offset, 0.0], [offset, 30.0]],
                    [
                        [20.0 + offset / 2.0, 0.0],
                        [20.0 + offset / 2.0, 28.0],
                    ],
                ],
            }
        )
    authority = sha256_json({"authority": "synthetic-qcdev44"})
    return (
        build_root_continuity_assurance(
            records=root_records,
            source_units=source_units,
            reference_authority_sha256=authority,
            prediction_authority_identity_sha256=sha256_json(
                {"root-prediction": "synthetic"}
            ),
        ),
        build_hair_attachment_assurance(
            records=hair_records,
            source_units=source_units,
            annotation_authority_sha256=authority,
            prediction_authority_identity_sha256=sha256_json(
                {"hair-prediction": "synthetic"}
            ),
        ),
    )


def _component_metric_rows(root: dict, hair: dict) -> list[dict]:
    _, expected_root = figure_inputs_builder._validate_root_continuity_assurance(
        root
    )
    _, expected_hair = figure_inputs_builder._validate_hair_attachment_assurance(
        hair
    )
    root_specs = {
        "root_continuity_reference_axis_coverage_mean": (
            "Mean union reference-axis coverage",
            "fraction",
            "diagnostic_only_union_coverage",
        ),
        "root_continuity_maximum_single_component_coverage_mean": (
            "Mean maximum single-component root coverage",
            "fraction",
            "formal_measurement_assurance",
        ),
        "root_continuity_maximum_single_component_coverage_median": (
            "Median maximum single-component root coverage",
            "fraction",
            "formal_measurement_assurance",
        ),
        "root_continuity_best_component_gap_median_um": (
            "Median longest gap on the best root component",
            "um",
            "formal_measurement_assurance",
        ),
        "root_continuity_break_free_rate": (
            "Break-free root image rate",
            "fraction",
            "formal_measurement_assurance",
        ),
        "root_continuity_visible_axis_extent_mae_um": (
            "Visible root-axis extent MAE",
            "um",
            "formal_measurement_assurance",
        ),
    }
    hair_specs = {
        "hair_attachment_qualified_precision_20um": (
            "Attachment-qualified precision @20 µm",
            "fraction",
            "formal_measurement_assurance",
        ),
        "hair_attachment_qualified_recall_20um": (
            "Attachment-qualified recall @20 µm",
            "fraction",
            "formal_measurement_assurance",
        ),
        "hair_attachment_qualified_f1_20um": (
            "Attachment-qualified F1 @20 µm",
            "fraction",
            "formal_measurement_assurance",
        ),
        "hair_attachment_error_median_um": (
            "Median base error on formal hair identities",
            "um",
            "formal_measurement_assurance",
        ),
        "hair_attachment_error_p95_um": (
            "P95 base error on formal hair identities",
            "um",
            "formal_measurement_assurance",
        ),
    }
    qualified = hair["summary"]["formal_matched_attachment_accuracy"][
        "attachment_qualified_identity"
    ]
    formal_match_n = hair["summary"]["formal_matched_attachment_accuracy"][
        "attachment_position_error_on_all_formal_identity_matches"
    ]["n"]
    instances = {
        **{key: 44 for key in root_specs},
        "hair_attachment_qualified_precision_20um": qualified["n_pred"],
        "hair_attachment_qualified_recall_20um": qualified["n_gt"],
        "hair_attachment_qualified_f1_20um": qualified["n_pred"]
        + qualified["n_gt"],
        "hair_attachment_error_median_um": formal_match_n,
        "hair_attachment_error_p95_um": formal_match_n,
    }
    rows = []
    for key, (label, unit, publication_role) in (
        root_specs | hair_specs
    ).items():
        expected = (expected_root | expected_hair)[key]
        rows.append(
            {
                "domain": (
                    "root_continuity" if key in root_specs else "hair_attachment"
                ),
                "metric_key": key,
                "label": label,
                "value": expected["value"],
                "ci_low": expected["ci_low"],
                "ci_high": expected["ci_high"],
                "unit": unit,
                "n": 44,
                "instances": instances[key],
                "ci_method": "image/source-unit nonparametric bootstrap",
                "bootstrap_repetitions": 10_000,
                "bootstrap_seed": 20_260_828,
                "evidence_role": "annotated_qc_development_non_independent",
                "publication_metric_role": publication_role,
            }
        )
    return rows


def _development(root: Path, container: str) -> dict[str, FileSource]:
    rows = []
    for index in range(44):
        truth = 50 + index
        for comparator, predicted, offsets in (
            (builder.STAGEB_COMPARATOR, truth + 1, (-5, -3, -1)),
            (builder.LEGACY_COMPARATOR, truth - 2, (-8, -6, -4)),
        ):
            rows.append(
                {
                    "source_unit": f"qc-{index:02d}",
                    "source_unit_order": index,
                    "family_key": f"family-{index:02d}",
                    "comparator": comparator,
                    "gt_count": truth,
                    "predicted_count": predicted,
                    "biological_presence_tp_5um": truth + offsets[0],
                    "biological_presence_tp_10um": truth + offsets[1],
                    "biological_presence_tp_20um": truth + offsets[2],
                    "prediction_input_sha256": sha256_json([comparator, index]),
                    "prediction_input_set_identity_sha256": sha256_json([comparator, "set"]),
                    "prediction_input_schema_version": "synthetic-locked-1.0",
                    "identity_hair_variant": comparator,
                    "evidence_role": (
                        "selected_train399_qcdevelopment_evaluation"
                        if comparator == builder.STAGEB_COMPARATOR
                        else "locked_legacy_development_comparator"
                    ),
                }
            )
    per_image = pd.DataFrame(rows)
    generator = np.random.default_rng(builder.BOOTSTRAP_SEED)
    sampled = generator.integers(0, 44, size=(builder.BOOTSTRAP_REPETITIONS, 44))
    result: dict[str, dict[int, np.ndarray | float]] = {}
    tolerance_rows = []
    for comparator in (builder.STAGEB_COMPARATOR, builder.LEGACY_COMPARATOR):
        selected = per_image[per_image["comparator"] == comparator].sort_values("source_unit_order")
        pred = selected["predicted_count"].to_numpy(np.int64)
        truth = selected["gt_count"].to_numpy(np.int64)
        result[comparator] = {}
        for tolerance_um in (5, 10, 20):
            tp = selected[f"biological_presence_tp_{tolerance_um}um"].to_numpy(np.int64)
            precision, recall, f1 = builder._prf(int(tp.sum()), int(pred.sum()), int(truth.sum()))
            boot_precision, boot_recall, boot_f1 = builder._prf(
                tp[sampled].sum(axis=1), pred[sampled].sum(axis=1), truth[sampled].sum(axis=1)
            )
            result[comparator][tolerance_um] = boot_f1
            low, high = np.quantile(boot_f1, (0.025, 0.975))
            tolerance_rows.append(
                {
                    "comparator": comparator,
                    "tolerance_um": tolerance_um,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_delta_stageb_minus_legacy_f1": 0.0,
                    "paired_delta_ci_low": 0.0,
                    "paired_delta_ci_high": 0.0,
                    "ci_method": "image-level nonparametric bootstrap",
                    "bootstrap_repetitions": builder.BOOTSTRAP_REPETITIONS,
                    "primary_metric": "one_to_one_tolerant_biological_hair_presence",
                    "minimum_truth_coverage": 0.25,
                    "minimum_prediction_coverage": 0.25,
                    "minimum_direction_cosine": 0.0,
                    "endpoint_gate_used": False,
                }
            )
    for tolerance_um in (5, 10, 20):
        delta = (
            result[builder.STAGEB_COMPARATOR][tolerance_um]
            - result[builder.LEGACY_COMPARATOR][tolerance_um]
        )
        low, high = np.quantile(delta, (0.025, 0.975))
        point = float(
            pd.DataFrame(tolerance_rows)
            .query("comparator == @builder.STAGEB_COMPARATOR and tolerance_um == @tolerance_um")["f1"]
            .iloc[0]
            - pd.DataFrame(tolerance_rows)
            .query("comparator == @builder.LEGACY_COMPARATOR and tolerance_um == @tolerance_um")["f1"]
            .iloc[0]
        )
        for row in tolerance_rows:
            if row["tolerance_um"] == tolerance_um:
                row["paired_delta_stageb_minus_legacy_f1"] = point
                row["paired_delta_ci_low"] = low
                row["paired_delta_ci_high"] = high
    threshold = pd.DataFrame(
        [
            {"threshold": 0.15, "f1_20um": 0.8, "count_mae": 2.0, "selected": False},
            {"threshold": 0.20, "f1_20um": 0.9, "count_mae": 1.0, "selected": True},
            {"threshold": 0.25, "f1_20um": 0.85, "count_mae": 1.5, "selected": False},
        ]
    )
    threshold["attachment_proxy_f1_20um"] = [0.72, 0.75, 0.74]
    threshold["selection_metric"] = "tolerant_biological_presence_f1_20um"
    threshold["straight_base_to_tip_presence_proxy_used"] = True
    threshold["distal_endpoint_or_length_used_as_selection_gate"] = False
    strata = pd.DataFrame(
        [
            {
                "dimension": "density",
                "stratum": "very_dense_ge200",
                "comparator": builder.HISTORICAL_COMPARATOR,
                "f1_20um": 0.64,
                "ci_low": 0.60,
                "ci_high": 0.68,
                "n_images": 55,
                "count_bias": -12.5,
                "precision": 0.7,
                "recall": 0.59,
                "primary_metric": "one_to_one_tolerant_biological_hair_presence",
                "ci_method": "image-level nonparametric bootstrap",
                "bootstrap_repetitions": 10_000,
            }
        ]
    )
    return {
        "development_per_image": _csv(root, "development_per_image", per_image, container),
        "development_tolerance": _csv(root, "development_tolerance", pd.DataFrame(tolerance_rows), container),
        "development_threshold": _csv(root, "development_threshold", threshold, container),
        "development_strata": _csv(root, "development_strata", strata, container),
    }


def _assurance(
    root: Path,
    container: str,
    *,
    trait_contract: dict,
    trait_contract_file_sha256: str,
) -> tuple[
    dict[str, FileSource],
    dict[str, FileSource],
    dict,
    dict[str, str],
    dict,
    dict,
]:
    root_continuity, hair_attachment = (
        deepcopy(value) for value in _component_assurance_fixture()
    )
    length_pairs = []
    for index in range(12):
        observed = 100.0 + 8 * index
        predicted = observed + (-3.0 + index % 4)
        length_pairs.append(
            {
                "pair_type": "conditional_length",
                "source_unit": f"hair-{index:02d}",
                "observed": observed,
                "predicted": predicted,
                "unit": "um",
                "endpoint_error_um": 8.0 + index,
                "trajectory_continuity": 0.80 + index / 100,
            }
        )
    root_pairs = []
    root_records = trait_contract["primary_root_traits"]
    for trait_index, trait in enumerate(root_records):
        field = str(trait["field"])
        for index in range(44):
            observed_trait = 10.0 + trait_index * 3.0 + index * (0.2 + trait_index / 100)
            predicted_trait = observed_trait * (0.995 + trait_index / 20_000) + (index % 3 - 1) * 0.03
            source_unit = f"qc-{index:02d}"
            root_pairs.append(
                {
                    "pair_type": "root_trait",
                    "source_unit": source_unit,
                    "pair_id": f"{source_unit}:{field}",
                    "trait_id": str(trait["id"]),
                    "trait_key": field,
                    "trait_family": ROOT_TRAIT_FAMILY_BY_FIELD[field],
                    "observed": observed_trait,
                    "predicted": predicted_trait,
                    "unit": str(trait["unit"]),
                    "reference_observable": True,
                    "prediction_observable": True,
                    "agreement_eligible": True,
                    "ineligibility_reason": "",
                    "reference_definition": ROOT_TRAIT_REFERENCE_DEFINITION,
                    "prediction_definition": ROOT_TRAIT_PREDICTION_DEFINITION,
                    "source_image_sha256": sha256_json(["qc-image", index]),
                    "endpoint_error_um": np.nan,
                    "trajectory_continuity": np.nan,
                }
            )
    scale_pairs = []
    for index in range(37):
        observed_scale = 1.5 + index / 100
        predicted_scale = observed_scale * 1.01
        scale_pairs.append(
            {
                "pair_type": "scale",
                "source_unit": f"qc-scale-{index:02d}",
                "pair_id": f"qc-scale-{index:02d}:scale",
                "trait_key": "um_per_px",
                "observed": observed_scale,
                "predicted": predicted_scale,
                "unit": "um_per_px",
                "relative_error_percent": abs(predicted_scale - observed_scale)
                / observed_scale
                * 100.0,
                "scale_line_endpoint_error_um": 2.0 + index / 10,
                "source_image_sha256": sha256_json(["qc-scale-image", index]),
                "endpoint_error_um": np.nan,
                "trajectory_continuity": np.nan,
            }
        )
    pairs = pd.DataFrame([*scale_pairs, *length_pairs, *root_pairs])
    length_frame = pairs[pairs["pair_type"] == "conditional_length"]
    observed = length_frame["observed"].to_numpy(float)
    predicted = length_frame["predicted"].to_numpy(float)
    authority_identities = {
        "canonical_ground_truth": sha256_json({"canonical": "qcdev44"}),
        "qcdev_fusion_prediction_ordered_file_set": sha256_json(
            [{"task_id": f"qc-{index:02d}"} for index in range(44)]
        ),
        "root_continuity_assurance": root_continuity[
            "root_continuity_assurance_identity_sha256"
        ],
        "hair_attachment_assurance": hair_attachment[
            "hair_attachment_assurance_identity_sha256"
        ],
    }
    root_trait_assurance = build_root_trait_assurance(
        pairs=root_pairs,
        trait_contract=trait_contract,
        source_units=[f"qc-{index:02d}" for index in range(44)],
        trait_contract_file_sha256=trait_contract_file_sha256,
        reference_authority_sha256=authority_identities["canonical_ground_truth"],
        prediction_authority_identity_sha256=authority_identities[
            "qcdev_fusion_prediction_ordered_file_set"
        ],
        bootstrap_repetitions=100,
        bootstrap_seed=builder.BOOTSTRAP_SEED,
    )
    computed = {
        "conditional_length_mae_um": float(np.mean(np.abs(predicted - observed))),
        "conditional_length_bias_um": float(np.mean(predicted - observed)),
        "conditional_length_ccc": float(builder._ccc_sample_rows(observed[None, :], predicted[None, :])[0]),
        "matched_endpoint_error_um": float(np.median(length_frame["endpoint_error_um"])),
        "matched_trajectory_continuity": float(np.mean(length_frame["trajectory_continuity"])),
    }
    values = {
        "root_dice": 0.96,
        "root_boundary_f1": 0.91,
        "root_hd95_um": 22.0,
        "distal_median_error_um": 14.0,
        "distal_pck": 0.95,
        "scale_detection_coverage": 37 / 38,
        "scale_geometry_endpoint_error_um": float(
            np.median([row["scale_line_endpoint_error_um"] for row in scale_pairs])
        ),
        "scale_relative_error_percent": float(
            np.median([row["relative_error_percent"] for row in scale_pairs])
        ),
        **computed,
        "endpoint_complete_support_fraction": 0.7,
        "root_trait_agreement": float(
            np.median(
                [
                    row["ccc"]
                    for row in root_trait_assurance["trait_rows"]
                    if row["ccc"] is not None
                ]
            )
        ),
        "axis_containment_median": 1.0,
        "axis_containment_min": 0.98,
        "unsupported_attachment_n": 0.0,
        "provider_exact_fraction": 1.0,
    }
    metric_rows = []
    for key, value in values.items():
        denominator = (
            283
            if key == "provider_exact_fraction"
            else 12
            if key in computed
            else 37
            if key
            in {
                "scale_geometry_endpoint_error_um",
                "scale_relative_error_percent",
            }
            else 38
            if key == "scale_detection_coverage"
            else 44
        )
        metric_rows.append(
            {
                "domain": "synthetic_assurance",
                "metric_key": key,
                "label": key,
                "value": value,
                "ci_low": value if key == "unsupported_attachment_n" else value - 0.01,
                "ci_high": value if key == "unsupported_attachment_n" else value + 0.01,
                "unit": "synthetic",
                "n": denominator,
                "instances": (
                    44 * 19
                    if key == "root_trait_agreement"
                    else 37
                    if key == "scale_detection_coverage"
                    else denominator
                ),
                "ci_method": "image/source-unit nonparametric bootstrap",
                "bootstrap_repetitions": 10000,
                "bootstrap_seed": 20260828,
                "evidence_role": (
                    "exact_portable_provider_equivalence"
                    if key == "provider_exact_fraction"
                    else "application_observability_non_accuracy"
                    if key
                    in {
                        "endpoint_complete_support_fraction",
                        "axis_containment_median",
                        "axis_containment_min",
                        "unsupported_attachment_n",
                    }
                    else "annotated_qc_development_non_independent"
                ),
                "publication_metric_role": "other_assurance",
            }
        )
    metric_rows.extend(_component_metric_rows(root_continuity, hair_attachment))
    support = pd.DataFrame(
        [
            {
                "condition_code": group,
                "support_fraction": supported / 250,
                "supported_hairs": supported,
                "identity_hairs": 250,
                "source_units": 12,
                "support_semantics": "endpoint-complete matched subset; absent length is not zero",
            }
            for group, supported in zip(builder.GROUPS, (160, 170, 180, 190), strict=True)
        ]
    )
    topology = pd.DataFrame(
        [
            {
                "source_unit": f"formal-{index:03d}",
                "axis_containment_fraction": 0.98 if index == 0 else 1.0,
                "unsupported_attachment_n": 0,
                "identity_hair_n": 4,
            }
            for index in range(261)
        ]
    )
    resources = {
        "assurance_metrics": _csv(root, "assurance_metrics", pd.DataFrame(metric_rows), container),
        "assurance_pairs": _csv(root, "assurance_pairs", pairs, container),
        "assurance_support": _csv(root, "assurance_support", support, container),
    }
    source_inputs = {"assurance_topology": _csv(root, "assurance_topology", topology, container)}
    return (
        resources,
        source_inputs,
        root_trait_assurance,
        authority_identities,
        root_continuity,
        hair_attachment,
    )


def _biology(root: Path, container: str) -> tuple[dict[str, FileSource], dict[str, FileSource]]:
    points = []
    clean_rows = []
    full_rows = []
    for endpoint_index, endpoint in enumerate(builder.ENDPOINTS.values()):
        for group_index, group in enumerate(builder.GROUPS):
            for replicate in range(3):
                points.append(
                    {
                        "source_unit": f"clean-{group_index}-{replicate}",
                        "cohort": "primary_clean261",
                        "condition_code": group,
                        "formal_eligible": True,
                        "endpoint_key": endpoint,
                        "value": 10 + endpoint_index * 20 + group_index * 3 + replicate,
                        "unit": "count" if endpoint_index == 0 else "um",
                    }
                )
    for group_index, group in enumerate(builder.GROUPS):
        for replicate in range(3):
            row = {
                "task_id": f"clean-{group_index}-{replicate}",
                "experiment_key": "D15_8d",
                "condition_code": group,
                "study_role": "rhd6_factorial_8d_primary",
                "formal_statistics_eligible": True,
                "hair_count": 80 + group_index,
                "hair_length_measurement_hair_count": 5,
            }
            for endpoint_index, endpoint in enumerate(builder.ENDPOINTS.values()):
                row[endpoint] = 10 + endpoint_index * 20 + group_index * 3 + replicate
            clean_rows.append(row)
        for replicate in range(4):
            row = {
                "task_id": f"full-{group_index}-{replicate}",
                "experiment_key": "D15_8d",
                "condition_code": group,
                "study_role": "rhd6_factorial_8d_primary",
                "formal_statistics_eligible": True,
                "hair_count": 82 + group_index,
                "hair_length_measurement_hair_count": 6,
            }
            for endpoint_index, endpoint in enumerate(builder.ENDPOINTS.values()):
                row[endpoint] = 11 + endpoint_index * 20 + group_index * 3 + replicate
            full_rows.append(row)
    normalized_effects = []
    primary_rows = []
    sensitivity_rows = []
    for cohort, output, offset, n in (
        ("primary_clean261", primary_rows, 0.0, 12),
        ("sensitivity_full283", sensitivity_rows, 0.02, 16),
    ):
        for endpoint_index, endpoint in enumerate(builder.ENDPOINTS.values()):
            for effect_index, (effect_label, raw_effect) in enumerate(builder.EFFECTS.values()):
                estimate = 0.82 + endpoint_index * 0.05 + effect_index * 0.04 + offset
                low, high = estimate - 0.08, estimate + 0.08
                is_h11 = endpoint == builder.H11_ENDPOINT
                h11_raw = (6.0, 3.0, 0.0)
                raw_estimate = (
                    h11_raw[effect_index]
                    if is_h11
                    else (estimate - 1.0) * 50.0
                )
                raw_low, raw_high = raw_estimate - 2.0, raw_estimate + 2.0
                raw_estimand = (
                    builder.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                    if is_h11
                    else builder.RAW_EFFECT_OLS_MEAN_CONTRAST
                )
                raw_interval = (
                    builder.RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                    if is_h11
                    else builder.RAW_EFFECT_HC3_INTERVAL
                )
                raw_replicates = builder.H11_RAW_BOOTSTRAP_REPLICATES if is_h11 else 0
                raw_seed = (
                    builder.raw_median_bootstrap_seed(
                        seed=builder.H11_RAW_BOOTSTRAP_BASE_SEED,
                        field=builder.H11_ENDPOINT,
                        component="continuous",
                    )
                    if is_h11
                    else None
                )
                normalized_effects.append(
                    {
                        "cohort": cohort,
                        "endpoint_key": endpoint,
                        "effect_key": effect_label,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "endpoint_n": n,
                        "effect_scale": "ratio",
                        "raw_effect_estimate": raw_estimate,
                        "raw_effect_ci_low": raw_low,
                        "raw_effect_ci_high": raw_high,
                        "raw_effect_estimand": raw_estimand,
                        "raw_effect_interval_method": raw_interval,
                        "raw_effect_bootstrap_replicates": raw_replicates,
                        "raw_effect_bootstrap_seed": raw_seed,
                        "standardized_effect": raw_estimate / 50.0,
                        "standardized_ci_low": raw_low / 50.0,
                        "standardized_ci_high": raw_high / 50.0,
                    }
                )
                output.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "effect": raw_effect,
                        "n": n,
                        "estimate": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value_model_BH_FDR": 0.04 + effect_index / 100,
                        "effect_scale": "ratio",
                        "causal_treatment_claim_allowed": False,
                        "raw_effect_estimate": raw_estimate,
                        "raw_effect_ci95_low": raw_low,
                        "raw_effect_ci95_high": raw_high,
                        "raw_effect_estimand": raw_estimand,
                        "raw_effect_interval_method": raw_interval,
                        "raw_effect_bootstrap_replicates": raw_replicates,
                        "raw_effect_bootstrap_seed": raw_seed,
                        "standardized_effect": raw_estimate / 50.0,
                        "standardized_ci95_low": raw_low / 50.0,
                        "standardized_ci95_high": raw_high / 50.0,
                    }
                )
    resources = {
        "phenotype_points": _csv(root, "phenotype_points", pd.DataFrame(points), container),
        "phenotype_effects": _csv(root, "phenotype_effects", pd.DataFrame(normalized_effects), container),
    }
    inputs = {
        "analysis_primary_table": _csv(root, "analysis_primary_table", pd.DataFrame(primary_rows), container),
        "analysis_sensitivity_table": _csv(root, "analysis_sensitivity_table", pd.DataFrame(sensitivity_rows), container),
        "clean_traits": _csv(root, "clean_traits", pd.DataFrame(clean_rows), container),
        "full_traits": _csv(root, "full_traits", pd.DataFrame(full_rows), container),
    }
    return resources, inputs


def _multitrait_atlas(trait_contract: dict) -> dict:
    family_by_trait = {
        trait_id: family
        for family, trait_ids in builder.MEASUREMENT_FAMILY_TRAIT_IDS.items()
        for trait_id in trait_ids
    }
    canonical = [
        *trait_contract["primary_root_traits"],
        *trait_contract["root_hair_traits"],
    ]
    descriptors = []
    for ordinal, record in enumerate(canonical, start=1):
        trait_id = str(record["id"])
        peak_group_index = (ordinal - 1) % len(builder.GROUPS)
        condition_summaries = {}
        for group_index, group in enumerate(builder.GROUPS):
            total = 3
            non_null = 0 if trait_id == "H12" and group_index == 0 else total
            median = None
            status = "not_estimated_no_finite_source_units"
            if non_null:
                median = (
                    float(31 + group_index * 3)
                    if str(record["field"]) == builder.H11_ENDPOINT
                    else float(100 + ordinal - abs(group_index - peak_group_index) * 5)
                )
                status = builder.CONDITION_SUMMARY_STATUS
            condition_summaries[group] = {
                "source_unit_total": total,
                "non_null_source_unit_n": non_null,
                "observability_fraction": non_null / total,
                "summary_status": status,
                "median": median,
            }
        cohort_records = {}
        for cohort, offset, n in (
            ("primary_clean261", 0.0, 12),
            ("sensitivity_full283", 0.02, 16),
        ):
            effects = {}
            field = str(record["field"])
            if field in builder.ENDPOINTS.values():
                endpoint_index = list(builder.ENDPOINTS.values()).index(field)
                for effect_index, (effect_key, _raw_effect) in enumerate(
                    builder.EFFECTS.values()
                ):
                    estimate = 0.82 + endpoint_index * 0.05 + effect_index * 0.04 + offset
                    is_h11 = field == builder.H11_ENDPOINT
                    raw_estimate = (
                        (6.0, 3.0, 0.0)[effect_index]
                        if is_h11
                        else (estimate - 1.0) * 50.0
                    )
                    raw_low, raw_high = raw_estimate - 2.0, raw_estimate + 2.0
                    effects[effect_key] = {
                        "status": "estimated_fixed_15_effect_family",
                        "estimate": estimate,
                        "ci95_low": estimate - 0.08,
                        "ci95_high": estimate + 0.08,
                        "endpoint_n": n,
                        "effect_scale": "ratio",
                        "raw_effect_estimate": raw_estimate,
                        "raw_effect_ci95_low": raw_low,
                        "raw_effect_ci95_high": raw_high,
                        "raw_effect_estimand": (
                            builder.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                            if is_h11
                            else builder.RAW_EFFECT_OLS_MEAN_CONTRAST
                        ),
                        "raw_effect_interval_method": (
                            builder.RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                            if is_h11
                            else builder.RAW_EFFECT_HC3_INTERVAL
                        ),
                        "raw_effect_bootstrap_replicates": (
                            builder.H11_RAW_BOOTSTRAP_REPLICATES if is_h11 else 0
                        ),
                        "raw_effect_bootstrap_seed": (
                            builder.raw_median_bootstrap_seed(
                                seed=builder.H11_RAW_BOOTSTRAP_BASE_SEED,
                                field=builder.H11_ENDPOINT,
                                component="continuous",
                            )
                            if is_h11
                            else None
                        ),
                        "standardized_effect": raw_estimate / 50.0,
                        "standardized_ci95_low": raw_low / 50.0,
                        "standardized_ci95_high": raw_high / 50.0,
                        "not_estimable_reason": None,
                    }
            cohort_records[cohort] = {
                "condition_summaries": deepcopy(condition_summaries),
                "effects": effects,
            }
        descriptors.append(
            {
                "ordinal": ordinal,
                "trait_id": trait_id,
                "field": str(record["field"]),
                "measurement_family": family_by_trait[trait_id],
                "cohorts": cohort_records,
            }
        )
    payload = {
        "schema_version": builder.MULTITRAIT_ATLAS_SCHEMA_VERSION,
        "status": "completed_source_derived_32_trait_atlas",
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "descriptor_count": 32,
        "root_descriptor_count": 19,
        "hair_descriptor_count": 13,
        "cohort_order": ["primary_clean261", "sensitivity_full283"],
        "condition_order": list(builder.GROUPS),
        "measurement_family_order": list(builder.MEASUREMENT_FAMILY_ORDER),
        "prespecified_inferential_endpoint_fields": list(builder.ATLAS_PRIMARY_ENDPOINTS),
        "effect_order": list(builder.ATLAS_EFFECT_KEYS),
        "estimated_effect_slot_count": 30,
        "not_estimated_effect_slot_count": 162,
        "descriptors": descriptors,
    }
    payload["atlas_identity_sha256"] = sha256_json(payload)
    return payload


def _profiles(root: Path, container: str) -> FileSource:
    rows = []
    for metric_index, metric in enumerate(
        ("identity_abundance", "conditional_median_length_um", "length_support_fraction")
    ):
        for group_index, group in enumerate(builder.GROUPS):
            for start in range(5):
                value = 0.6 + group_index * 0.04 + start * 0.01 if metric_index == 2 else 20 + metric_index * 100 + group_index * 4 + start
                rows.append(
                    {
                        "cohort": "primary_clean261",
                        "condition_code": group,
                        "bin_start_mm": start,
                        "bin_end_mm": start + 1,
                        "metric_key": metric,
                        "estimate": value,
                        "ci_low": value - 0.02,
                        "ci_high": value + 0.02,
                        "eligible_n": 12,
                        "length_supported_n": 8,
                        "bootstrap_repetitions": 10_000,
                        "unit_of_analysis": "source image/root",
                    }
                )
    return _csv(root, "axial_profiles", pd.DataFrame(rows), container)


def _effect_pattern_inputs(
    *,
    endpoint_n: int = 12,
    estimates: tuple[float, float, float] = (0.9, 1.1, 0.99),
    intervals: tuple[tuple[float, float], ...] = (
        (0.8, 1.2),
        (0.9, 1.3),
        (0.85, 1.15),
    ),
    full_estimates: tuple[float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], tuple[object, ...]]]:
    endpoint = builder.ENDPOINTS["ABUNDANCE"]
    points = pd.DataFrame(
        [
            {
                "source_unit": f"{group}-{replicate}",
                "cohort": "primary_clean261",
                "condition_code": group,
                "formal_eligible": True,
                "endpoint_key": endpoint,
                "value": 10 + group_index + replicate,
                "unit": "count",
            }
            for group_index, group in enumerate(builder.GROUPS)
            for replicate in range(3)
        ]
    )
    effects = pd.DataFrame(
        [
            {
                "cohort": "primary_clean261",
                "endpoint_key": endpoint,
                "effect_key": resource_effect,
                "endpoint_n": endpoint_n,
            }
            for resource_effect, _raw_effect in builder.EFFECTS.values()
        ]
    )
    if full_estimates is None:
        full_estimates = estimates
    cache: dict[tuple[str, str], tuple[object, ...]] = {}
    for index, effect_label in enumerate(builder.EFFECTS):
        low, high = intervals[index]
        cache[("ABUNDANCE", effect_label)] = (
            estimates[index],
            low,
            high,
            0.5,
            [],
            full_estimates[index],
            low,
            high,
            [],
        )
    return points, effects, cache


def _runtime(root: Path, container: str) -> tuple[FileSource, FileSource]:
    rows = []
    for index in range(283):
        wall = 7.0 + (index % 10) / 10
        rows.append(
            {
                "source_unit": f"image-{index:03d}",
                "wall_seconds": wall,
                "megapixels": 32.0,
                "io_seconds": 0.5,
                "preprocess_seconds": 0.7,
                "inference_seconds": 4.5,
                "postprocess_seconds": wall - 5.7,
            }
        )
    frame = pd.DataFrame(rows)
    per_image = _csv(root, "runtime_per_image", frame, container)
    hardware = {
        "host": "synthetic-host",
        "processor": "synthetic CPU",
        "gpus": [{"physical_index": 1, "name": "Synthetic RTX 3090"}],
    }
    common = {
        "images": 283,
        "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "fresh_direct_run": True,
        "resume_or_cache_used": False,
        "hardware": hardware,
        "peak_vram_mib": 4096.0,
    }
    latency = {
        **common,
        "benchmark_mode": "sequential_persistent_full283",
        "median_seconds_per_image": float(frame["wall_seconds"].median()),
        "p95_seconds_per_image": float(frame["wall_seconds"].quantile(0.95)),
    }
    production = {
        **common,
        "benchmark_mode": "production_batch_full283",
        "per_image_latency_reported": False,
        "batch_wall_seconds": 720.0,
        "images_per_min": 283 * 60 / 720,
        "megapixels_per_second": 283 * 32 / 720,
        "stage_timings": [
            {"stage": "root_provider", "wall_seconds": 100.0},
            {"stage": "stageb_train399", "wall_seconds": 400.0},
            {"stage": "fusion", "wall_seconds": 50.0},
            {"stage": "traits", "wall_seconds": 40.0},
            {"stage": "distal_axis_profiles", "wall_seconds": 30.0},
        ],
    }
    baseline_latency = {
        **latency,
        "median_seconds_per_image": latency["median_seconds_per_image"] * 2.0,
        "p95_seconds_per_image": latency["p95_seconds_per_image"] * 2.0,
    }
    baseline_production = {
        **production,
        "batch_wall_seconds": production["batch_wall_seconds"] * 2.0,
        "images_per_min": production["images_per_min"] / 2.0,
        "megapixels_per_second": production["megapixels_per_second"] / 2.0,
    }
    payload = {
        "schema_version": "PHAxis-manuscript-two-mode-runtime-input-1.0",
        "status": "completed_two_mode_direct_full283",
        "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        "latency_mode": "sequential_persistent_full283",
        "sequential_latency_full283": latency,
        "production_batch_full283": production,
        "baseline_sequential_latency_full283": baseline_latency,
        "baseline_production_batch_full283": baseline_production,
        "latency_comparison": {
            "status": "comparable_direct_full283",
            "comparable": True,
            "noncomparability_reasons": [],
            "benchmark_mode": "sequential_persistent_full283",
            "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
            "same_283_source_manifest_hardware_and_io_scope": True,
            "historical_component_runtime_used_as_full_baseline": False,
            "median_latency_speedup_frozen_v1_over_phaxis": 2.0,
        },
        "production_comparison": {
            "status": "comparable_direct_full283",
            "comparable": True,
            "noncomparability_reasons": [],
            "benchmark_mode": "production_batch_full283",
            "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
            "same_283_source_manifest_hardware_and_io_scope": True,
            "historical_component_runtime_used_as_full_baseline": False,
            "batch_wall_speedup_frozen_v1_over_phaxis": 2.0,
        },
        "per_image_csv_sha256": per_image.file_sha256,
        "batch_latency_is_never_derived_per_image": True,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    return _json_file(root, "runtime_summary", payload, container), per_image


def _context(tmp_path: Path) -> BuildContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    master_path = PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    master_raw = master_path.read_bytes()
    container = sha256_json({"figure_input_assembly": "synthetic-final"})
    graph = _sealed_json(
        tmp_path,
        "evidence_graph",
        {"schema_version": "PHAxis-manuscript-release-evidence-graph-1.1", "blind_images_used": 0, "root_cap_region_statistics_included": False},
        "manifest_identity_sha256",
    )
    expert = "PHAxis-StageB-train399-five-seed"
    root_bundle_identity = "d" * 64
    model_id = "PHAXIS-V1.0.0-STRICT-TRAIN399-" + "A" * 20
    root_id = "PHAxis-root-provider-" + root_bundle_identity[:20].upper()
    proposal = _sealed_json(
        tmp_path,
        "model_contract_proposal",
        {
            "formal_release_status": "passed_proposal_not_official",
            "promotion": {
                "stageb_binding": {"expert_id": expert},
                "status": "validated_proposal_not_applied",
                "official_apply_performed": False,
            },
            "model_bundle_id": model_id,
            "root_expert": {
                "provider_role": "PHAxis-portable-root-provider",
                "expert_id": root_id,
                "bundle_identity_sha256": root_bundle_identity,
                "root_bundle_authority": {
                    "bundle_identity_sha256": root_bundle_identity,
                },
            },
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "model_contract_identity_sha256",
    )
    traits = _sealed_json(
        tmp_path,
        "traits",
        {
            "tasks": 283,
            "formal_statistics_eligible": 280,
            "review_only": 3,
            "hair_identities": 1000,
            "endpoint_complete_length_identities": 700,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "export_identity_sha256",
    )
    profiles = _sealed_json(
        tmp_path,
        "profiles",
        {
            "locked_1_4mm_trait_crosscheck_tasks": 261,
            "locked_1_4mm_trait_crosscheck_mismatches": 0,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "export_identity_sha256",
    )
    artifacts = {}
    for role in EVIDENCE_ARTIFACT_ROLES:
        if role == "model_contract_proposal":
            artifacts[role] = proposal
        elif role == "traits":
            artifacts[role] = traits
        elif role == "profiles":
            artifacts[role] = profiles
        else:
            artifacts[role] = _sealed_json(
                tmp_path,
                f"artifact_{role}",
                {"role": role, "blind_images_used": 0, "root_cap_region_statistics_included": False},
                "identity_sha256",
            )
    figure_inputs = _sealed_json(
        tmp_path,
        "figure_inputs",
        {"blind_images_used": 0, "root_cap_region_statistics_included": False},
        "figure_input_assembly_identity_sha256",
    )
    assembly = _plain_json(
        tmp_path,
        "figure_assembly_summary",
        {"status": "completed_final", "blind_images_used": 0, "root_cap_region_statistics_included": False},
    )
    human_payload = {
        "schema_version": HUMAN_METADATA_SCHEMA,
        "status": "complete_author_verified_external_metadata",
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "values": _human_values(),
    }
    human = _sealed_json(tmp_path, "human_metadata", human_payload, "human_metadata_identity_sha256")
    bundle = _sealed_json(
        tmp_path,
        "model_bundle_manifest",
        {
            "member_count": 7,
            "members": [{"role": f"member-{index}"} for index in range(7)],
            "bundle_sha256": "b" * 64,
            "bundle_size_bytes": 10_485_760,
            "model_bundle_id": model_id,
            "root_expert_id": root_id,
            "root_bundle_identity_sha256": root_bundle_identity,
            "hair_identity_count_expert": expert,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "model_bundle_manifest_identity_sha256",
    )
    clean_install = _sealed_json(
        tmp_path,
        "clean_install_receipt",
        {
            "example_output_identity_sha256": "c" * 64,
            "model_bundle_id": model_id,
            "root_expert_id": root_id,
            "root_bundle_identity_sha256": root_bundle_identity,
            "hair_identity_count_expert": expert,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "clean_install_receipt_identity_sha256",
    )
    resources = _development(tmp_path, container)
    trait_contract_payload = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(encoding="utf-8")
    )
    trait_contract_resource = _json_file(
        tmp_path, "trait_contract", trait_contract_payload, container
    )
    (
        assurance_resources,
        source_inputs,
        root_trait_assurance,
        assurance_authorities,
        root_continuity_assurance,
        hair_attachment_assurance,
    ) = _assurance(
        tmp_path,
        container,
        trait_contract=trait_contract_payload,
        trait_contract_file_sha256=trait_contract_resource.file_sha256,
    )
    resources.update(assurance_resources)
    resources["multitrait_atlas"] = _json_file(
        tmp_path,
        "multitrait_atlas",
        _multitrait_atlas(trait_contract_payload),
        container,
    )
    biology_resources, biology_inputs = _biology(tmp_path, container)
    resources.update(biology_resources)
    source_inputs.update(biology_inputs)
    wt_paths, wt_summary = _wt_secondary_authorities(tmp_path / "wt_secondary")
    wt_sources = {
        role.removeprefix("source/"): FileSource(
            role=role.removeprefix("source/"),
            path=path,
            file_sha256=sha256_file(path),
            container_identity_sha256=container,
        )
        for role, path in wt_paths.items()
    }
    source_inputs.update(wt_sources)
    resources.update(wt_sources)
    analysis_path = _write_json(tmp_path / "analysis.json", wt_summary)
    analysis_raw = analysis_path.read_bytes()
    artifacts["analysis"] = JsonSource(
        role="analysis",
        path=analysis_path,
        raw=analysis_raw,
        payload=wt_summary,
        file_sha256=sha256_bytes(analysis_raw),
        logical_identity_sha256=wt_summary["analysis_identity_sha256"],
    )
    phenotype_effect_frame = pd.read_csv(resources["phenotype_effects"].path)
    phenotype_effect_records = phenotype_effect_frame.astype(object).where(
        pd.notna(phenotype_effect_frame), None
    ).to_dict("records")
    narrative_decision = build_narrative_decision(
        phenotype_effect_records,
        source_sha256={
            "phenotype_effects": sha256_json(
                phenotype_effect_records
            )
        },
    )
    resources["narrative_decision"] = _json_file(
        tmp_path,
        "narrative_decision",
        narrative_decision,
        container,
    )
    figure_inputs.payload["narrative_decision_identity_sha256"] = (
        narrative_decision["narrative_decision_identity_sha256"]
    )
    figure_inputs.payload["narrative_branch_id"] = narrative_decision["branch_id"]
    assembly.payload["narrative_decision_identity_sha256"] = narrative_decision[
        "narrative_decision_identity_sha256"
    ]
    assembly.payload["narrative_branch_id"] = narrative_decision["branch_id"]
    resources["axial_profiles"] = _profiles(tmp_path, container)
    runtime_summary, runtime_per_image = _runtime(tmp_path, container)
    resources["runtime_summary"] = runtime_summary
    resources["runtime_per_image"] = runtime_per_image
    resources["trait_contract"] = trait_contract_resource
    fields = [
        record["field"]
        for record in (
            trait_contract_payload["primary_root_traits"]
            + trait_contract_payload["root_hair_traits"]
        )
    ]
    clean_trait_frame = pd.read_csv(biology_inputs["clean_traits"].path)
    full_trait_frame = pd.read_csv(biology_inputs["full_traits"].path)
    biological_trait_rows = {
        str(row["task_id"]): row
        for row in [
            *clean_trait_frame.to_dict("records"),
            *full_trait_frame.to_dict("records"),
        ]
    }
    biological_task_ids = list(biological_trait_rows)
    filler_task_ids = [
        f"image-{index:03d}"
        for index in range(283 - len(biological_task_ids))
    ]
    image_rows = []
    for index, task_id in enumerate([*biological_task_ids, *filler_task_ids]):
        row = {"task_id": task_id, "physical_units_valid": True}
        for field_index, field in enumerate(fields):
            row[field] = None if field_index == 0 and index >= 280 else field_index + index / 100
        if task_id in biological_trait_rows:
            row["visible_root_axis_length_um"] = biological_trait_rows[task_id][
                "visible_root_axis_length_um"
            ]
        row["shootward_endpoint_border_visible"] = bool(index % 3)
        image_rows.append(row)
    source_inputs["full_image_traits"] = _csv(tmp_path, "full_image_traits", pd.DataFrame(image_rows), container)
    source_inputs["sensitivity_profiles_summary"] = _json_file(
        tmp_path,
        "sensitivity_profiles_summary",
        {
            "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
            "status": "completed",
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        container,
    )
    hair_crosscheck_locks = [
        {
            "task_id": row["source_unit"],
            "n_pred": row["formal_matched_attachment_accuracy"][
                "formal_biological_presence"
            ]["n_pred"],
            "n_gt": row["formal_matched_attachment_accuracy"][
                "formal_biological_presence"
            ]["n_gt"],
            "biological_presence_tp_20um": row[
                "formal_matched_attachment_accuracy"
            ]["formal_biological_presence"]["tp"],
            "hair_attachment_row_identity_sha256": row["row_identity_sha256"],
        }
        for row in hair_attachment_assurance["per_image"]
    ]
    assurance_authorities[
        "qcdev_stageb_biological_presence_20um_crosscheck"
    ] = sha256_json(hair_crosscheck_locks)
    hair_formal = hair_attachment_assurance["summary"][
        "formal_matched_attachment_accuracy"
    ]
    hair_qualified = hair_formal["attachment_qualified_identity"]
    hair_presence = hair_formal["formal_biological_presence"]
    assurance_receipt = _sealed_json(
        tmp_path,
        "measurement_assurance",
        {
            "measurement_contract": {
                "distal_pck_threshold_um": 25.0,
                "root_trait_provider_equivalence_used_as_accuracy": False,
                "scale_coverage_denominator": "visible_annotated_scale_bar_cases",
                "scale_localization_denominator": "detected_visible_scale_bars",
                "scale_calibration_denominator": "detected_visible_scale_bars",
                "scale_absence_specificity_status": "not_estimable_no_absent_or_untrusted_scale_cases",
                "scale_fail_closed_evidence_basis": "software_contract_and_unit_tests",
            },
            "scale_applicability": {
                "qcdevelopment_images": 44,
                "visible_annotated_scale_bar_cases": 38,
                "trusted_metadata_without_visible_bar_cases": 6,
                "absent_or_untrusted_scale_truth_cases": 0,
                "absence_specificity_status": "not_estimable_no_absent_or_untrusted_scale_cases",
                "fail_closed_evidence_basis": "software_contract_and_unit_tests",
                "empirical_absence_specificity_claimed": False,
            },
            "counts": {
                "qcdevelopment_images": 44,
                "visible_scale_bars": 38,
                "trusted_metadata_without_visible_bar_cases": 6,
                "absent_or_untrusted_scale_truth_cases": 0,
                "detected_scale_bars": 37,
                "scale_localization_pairs": 37,
                "scale_calibration_pairs": 37,
                "root_continuity_source_units": 44,
                "root_continuity_break_free_images": root_continuity_assurance[
                    "summary"
                ]["break_free_images"],
                "root_continuity_union_coverage_hides_fragmentation_images": root_continuity_assurance[
                    "summary"
                ]["union_coverage_hides_fragmentation_images"],
                "hair_attachment_source_units": 44,
                "hair_attachment_predicted_hairs": hair_qualified["n_pred"],
                "hair_attachment_annotated_hairs": hair_qualified["n_gt"],
                "hair_attachment_formal_identity_matches": hair_presence["tp"],
                "hair_attachment_qualified_true_positives_20um": hair_qualified[
                    "tp"
                ],
                "hair_attachment_evaluator_crosschecked_source_units": 44,
            },
            "source_authority_identity_sha256": assurance_authorities,
            "source_authority_sha256": {
                "trait_contract": trait_contract_resource.file_sha256,
            },
            "source_table_sha256": {
                "pairs": resources["assurance_pairs"].file_sha256,
                "root_traits": sha256_json(root_trait_assurance["trait_rows"]),
            },
            "root_trait_assurance": root_trait_assurance,
            "component_receipts": {
                "root_continuity": {
                    "identity_sha256": root_continuity_assurance[
                        "root_continuity_assurance_identity_sha256"
                    ]
                },
                "hair_attachment": {
                    "identity_sha256": hair_attachment_assurance[
                        "hair_attachment_assurance_identity_sha256"
                    ]
                },
            },
            "qcdev_stageb_biological_presence_20um_crosscheck_locks": hair_crosscheck_locks,
            "root_continuity_assurance": root_continuity_assurance,
            "hair_attachment_assurance": hair_attachment_assurance,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        },
        "measurement_assurance_identity_sha256",
    )
    release_metadata = _sealed_json(
        tmp_path,
        "release_human_metadata",
        {
            "schema_version": "PHAxis-release-human-metadata-1.3",
            "status": "author_verified_release_authority",
            "product": "PHAxis",
            "product_version": "1.0.0",
            "distribution": "phaxis",
            "project_urls": {
                "Repository": _human_values()["PHAXIS_REPOSITORY_URL"],
            },
            "release_coordinates": {
                "github_repository_url": _human_values()["PHAXIS_REPOSITORY_URL"],
                "github_release_tag": "v1.0.0",
                "release_doi": _human_values()["PHAXIS_RELEASE_DOI"],
            },
            "rights": {
                "source_license_spdx": "Apache-2.0",
                "source_release_authorized": True,
            },
        },
        "metadata_identity_sha256",
    )
    release_files = [
        {
            "path": "RELEASE_HUMAN_METADATA.json",
            "sha256": release_metadata.file_sha256,
        }
    ]
    source_release = _plain_json(
        tmp_path,
        "source_release_manifest",
        {
            "schema_version": "PHAxis-source-release-manifest-2.0",
            "distribution": "phaxis",
            "version": "1.0.0",
            "release_mode": "formal",
            "files": release_files,
            "tree_identity_sha256": sha256_json(release_files),
        },
    )
    source_release = JsonSource(
        role=source_release.role,
        path=source_release.path,
        raw=source_release.raw,
        payload=source_release.payload,
        file_sha256=source_release.file_sha256,
        logical_identity_sha256=source_release.payload["tree_identity_sha256"],
    )
    release_cross_binding = {
        "cross_binding_identity_sha256": sha256_json(
            {
                "repository": _human_values()["PHAXIS_REPOSITORY_URL"],
                "tag": "v1.0.0",
            }
        )
    }
    return BuildContext(
        master_path=master_path,
        master_raw=master_raw,
        master_text=master_raw.decode("utf-8"),
        evidence_graph=graph,
        evidence_artifacts=artifacts,
        figure_inputs=figure_inputs,
        figure_assembly_summary=assembly,
        model_contract_proposal=proposal,
        human_metadata=human,
        human_values=_human_values(),
        model_bundle_manifest=bundle,
        clean_install_receipt=clean_install,
        source_release_manifest=source_release,
        source_release_metadata=release_metadata,
        software_release_cross_binding=release_cross_binding,
        narrative_decision=narrative_decision,
        model_bundle_id=model_id,
        root_expert_id=root_id,
        root_bundle_identity_sha256=root_bundle_identity,
        hair_identity_count_expert=expert,
        resources=resources,
        source_inputs=source_inputs,
        provenance_receipts={"measurement_assurance": assurance_receipt},
    )


def _built(tmp_path: Path) -> tuple[BuildContext, dict, dict]:
    context = _context(tmp_path)
    contract = build_token_source_contract(context.master_text)
    entries = builder.derive_entries(context, contract)
    payload = assemble_values_payload(context=context, token_contract=contract, entries=entries)
    return context, contract, payload


def test_all_current_master_tokens_are_machine_derived_and_deterministic(tmp_path: Path) -> None:
    context, contract, first = _built(tmp_path / "fixture")
    wt_secondary = validate_wt_secondary_source_inputs(context)
    second_entries = builder.derive_entries(context, contract)
    second = assemble_values_payload(context=context, token_contract=contract, entries=second_entries)
    assert len(contract["tokens"]) == len(first["values"])
    assert len(contract["tokens"]) >= 180
    assert not any(token.startswith("FINAL_WT") for token in contract["tokens"])
    assert wt_secondary["schema_version"] == "PHAxis-WT-temperature-secondary-1.0"
    assert wt_secondary["D15_fixed_effect_rows"] == 15
    assert wt_secondary["D15_narrative_branch_changed"] is False
    assert wt_secondary["FINAL_WT_tokens_created"] is False
    assert wt_secondary["cross_day_pooling_performed"] is False
    assert wt_secondary["unknown_day_meta_analysis_performed"] is False
    assert wt_secondary["clean_full_pooling_performed"] is False
    assert wt_secondary["typed_blocks"] == {
        "wt_gate_flow": "wt_temperature_qc_flow",
        "wt_experiment_contrasts": "wt_within_experiment_contrasts",
        "wt_same_day_meta": "wt_within_day_meta_analysis",
    }
    for role in (
        "wt_temperature_qc_flow",
        "wt_within_experiment_contrasts",
        "wt_within_day_meta_analysis",
    ):
        assert f"figure_resource:{role}" in first["source_files"]
        assert f"figure_source_input:{role}" in first["source_files"]
    assert first["values"]["FINAL_D15_RAW_ABUNDANCE_BY_CELL"]["value"] == (
        "EV-22°C 11.0 [10.5–11.5]; EV-30°C 14.0 [13.5–14.5]; "
        "OE-22°C 17.0 [16.5–17.5]; OE-30°C 20.0 [19.5–20.5]"
    )
    assert first["values"]["FINAL_D15_RAW_ROOT_LENGTH_BY_CELL"]["value"] == (
        "EV-22°C 91.0 [90.5–91.5]; EV-30°C 94.0 [93.5–94.5]; "
        "OE-22°C 97.0 [96.5–97.5]; OE-30°C 100.0 [99.5–100.5]"
    )
    assert first == second
    assert first["model_bundle_id"].startswith("PHAXIS-V1.0.0-STRICT-TRAIN399-")
    assert first["root_expert_id"].startswith("PHAxis-root-provider-")
    assert first["hair_identity_count_expert"] == "PHAxis-StageB-train399-five-seed"
    assert all(entry["derivation"]["sources"] for entry in first["values"].values())
    assert all(
        source["source_cell_identity_sha256"]
        for entry in first["values"].values()
        for source in entry["derivation"]["sources"]
    )
    assert first["values"]["FINAL_D15_CLEAN_POOL_CELL_N"]["value"] == "3 / 3 / 3 / 3"
    assert first["values"]["FINAL_D15_CLEAN_FORMAL_CELL_N"]["value"] == "3 / 3 / 3 / 3"
    assert first["values"]["FINAL_D15_FULL_POOL_CELL_N"]["value"] == "4 / 4 / 4 / 4"
    assert first["values"]["FINAL_D15_CLEAN_LENGTH_CELL_N"]["value"] == "3 / 3 / 3 / 3"
    assert first["values"]["FINAL_D15_FIRST_HAIR_OBSERVABILITY_BY_CELL"]["value"] == (
        "3/3 (100.0%); 3/3 (100.0%); 3/3 (100.0%); 3/3 (100.0%)"
    )
    assert first["values"]["FINAL_D15_VISIBLE_AXIS_CENSORING_BY_CELL"]["value"] == (
        "1/3 (33.3%); 1/3 (33.3%); 1/3 (33.3%); 1/3 (33.3%)"
    )
    assert first["values"]["FINAL_SCALE_DETECTION_COVERAGE"]["value"] == "0.974"
    assert first["values"]["FINAL_SCALE_DETECTED_N"]["value"] == 37
    assert first["values"]["FINAL_SCALE_VALIDATION_N"]["value"] == 38
    scale_cell_identities = {
        first["values"][token]["derivation"]["sources"][0][
            "source_cell_identity_sha256"
        ]
        for token in (
            "FINAL_SCALE_DETECTION_COVERAGE",
            "FINAL_SCALE_DETECTED_N",
            "FINAL_SCALE_VALIDATION_N",
        )
    }
    assert len(scale_cell_identities) == 1
    assert first["values"]["FINAL_BENCHMARK_LATENCY_MODE_LABEL"]["value"] == (
        "sequential persistent-process full-workflow latency "
        "(283 images; process startup excluded per image)"
    )
    assert first["values"]["FINAL_E2E_FROZEN_V1_BATCH_TOTAL_MIN"]["value"] == "24.00"
    assert first["values"][
        "FINAL_E2E_BATCH_SPEEDUP_FROZEN_V1_OVER_PHAXIS"
    ]["value"] == "2.00"
    assert first["values"]["FINAL_E2E_FROZEN_V1_MEDIAN_IMAGE_S"]["value"] == "14.80"
    assert first["values"][
        "FINAL_E2E_MEDIAN_LATENCY_SPEEDUP_FROZEN_V1_OVER_PHAXIS"
    ]["value"] == "2.00"
    assert "FINAL_ABUNDANCE_CONSTRUCT_FULL_SENSITIVITY" not in first["values"]
    abundance_pattern = first["values"]["FINAL_D15_ABUNDANCE_PATTERN"]["value"]
    length_pattern = first["values"]["FINAL_D15_LENGTH_PATTERN"]["value"]
    abstract_synthesis = first["values"]["FINAL_D15_ABSTRACT_SYNTHESIS"]
    synthesis = first["values"]["FINAL_DISCUSSION_BIOLOGICAL_SYNTHESIS"]["value"]
    assert "RHD6_" not in abundance_pattern
    assert "OE-labelled:EV contrast was lower" in abundance_pattern
    assert "clean-cohort interval below the no-difference ratio" in abundance_pattern
    assert "Full283 estimate pointing in the same direction" in abundance_pattern
    assert "source-unit n=" not in abundance_pattern
    assert abundance_pattern.index("OE-labelled:EV contrast") < abundance_pattern.index(
        "30:22°C contrast"
    ) < abundance_pattern.index("construct-by-temperature interaction")
    assert "endpoint-complete support" in length_pattern
    assert abstract_synthesis["value"] == (
        "clean-cohort H08/N visible population was lower for the "
        "OE-versus-EV construct-label contrast"
    )
    assert builder.text_word_count(str(abstract_synthesis["value"])) <= 15
    assert abstract_synthesis["derivation"]["parameters"]["narrative_branch_id"] == "B"
    assert abstract_synthesis["derivation"]["parameters"][
        "narrative_decision_identity_sha256"
    ] == first["narrative_decision_identity_sha256"]
    assert abstract_synthesis["derivation"]["parameters"][
        "profiles_select_or_veto_narrative_branch"
    ] is False
    assert abstract_synthesis["derivation"]["parameters"]["headline_cell"] == {
        "endpoint_key": "local_hair_count_1_4mm",
        "effect_key": "OE_vs_EV",
        "sentinel": "H08",
        "badge": "N",
        "layer": "visible population",
        "clean_direction": "lower",
    }
    assert abstract_synthesis["derivation"]["parameters"][
        "headline_uses_effect_magnitude_or_p_value"
    ] is False
    assert abstract_synthesis["derivation"]["parameters"][
        "branch_b_common_direction_claimed"
    ] is False
    for token in (
        "FINAL_CLEAN_FULL_SAME_DIRECTION_N",
        "FINAL_CLEAN_FULL_UNSTABLE_EFFECTS",
    ):
        direction_parameters = first["values"][token]["derivation"]["parameters"]
        assert direction_parameters == {
            "fixed_effect_family_n": 15,
            "comparison_basis": "clean_vs_Full283_ratio_point_estimate_direction",
            "direction_states": ["lower", "null", "higher"],
            "null": 1.0,
            "intervals_used": False,
        }
    assert "multidimensional reorganization" not in synthesis.casefold()
    assert "In the primary hair-change layer," in synthesis
    assert "local visible-hair abundance and conditional projected hair length were lower" in synthesis
    assert "Along the distal axis," in synthesis
    assert "In carrying-root context," in synthesis
    assert synthesis.index("In the primary hair-change layer,") < synthesis.index(
        "Along the distal axis,"
    ) < synthesis.index("In carrying-root context,")
    assert "event frequency remained an observability descriptor" in synthesis
    assert "only its positive conditional distance entered the fixed models" in synthesis
    assert "rather than a count or directional vote across correlated descriptors" in synthesis
    forbidden_reader_terms = {
        "headline rule",
        "fixed decision",
        "sensitivity-unstable",
        "clean/full-concordant",
        "headline-supported",
        "coordinated phenotype",
        "coordinated d15 remodeling",
    }
    assert all(term not in abundance_pattern.casefold() for term in forbidden_reader_terms)
    assert all(
        term not in str(abstract_synthesis["value"]).casefold()
        for term in forbidden_reader_terms
    )
    assert all(term not in synthesis.casefold() for term in forbidden_reader_terms)
    synthesis_parameters = first["values"]["FINAL_DISCUSSION_BIOLOGICAL_SYNTHESIS"][
        "derivation"
    ]["parameters"]
    assert synthesis_parameters["narrative_layer_order"] == [
        "primary_hair_change",
        "spatial_location",
        "supporting_root_context",
    ]
    assert synthesis_parameters["correlated_descriptor_vote_used"] is False
    assert (
        synthesis_parameters[
            "correlated_descriptor_count_used_as_biological_conclusion"
        ]
        is False
    )
    atlas_summary = first["values"]["FINAL_MULTITRAIT_ATLAS_SUMMARY"]
    assert "condition-resolved 32-descriptor atlas" in atlas_summary["value"]
    for label in builder.MEASUREMENT_FAMILY_COMPACT_LABELS.values():
        assert label in atlas_summary["value"]
    assert "five non-redundant endpoints and 15 effects" in atlas_summary["value"]
    assert "other 27" not in atlas_summary["value"]
    assert "missing cells remain unfilled" in atlas_summary["value"]
    assert "no cross-trait ranking or directional vote was used" in atlas_summary["value"]
    assert len(atlas_summary["value"].split()) <= 65
    atlas_parameters = atlas_summary["derivation"]["parameters"]
    assert atlas_parameters["canonical_descriptor_n"] == 32
    assert atlas_parameters["modeled_endpoint_n"] == 5
    assert atlas_parameters["fixed_effect_family_n"] == 15
    assert atlas_parameters["measurement_family_order"] == list(
        builder.MEASUREMENT_FAMILY_ORDER
    )
    assert atlas_parameters["family_coverage"]["conditional_projected_length"][
        "incomplete_trait_n"
    ] == 1
    assert atlas_parameters["cross_trait_directional_vote_used"] is False
    assert atlas_parameters["missing_values_zero_filled"] is False


def test_h11_reviewer_surfaces_are_three_way_bound_without_changing_primary_effects(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path / "fixture")
    sources = builder.Sources(context)
    normalized = sources.table("figure_resource", "phenotype_effects")
    primary = sources.table("figure_source_input", "analysis_primary_table")
    atlas = sources.json_resource("multitrait_atlas")
    row, raw, evidence = builder._effect_sources(
        sources,
        normalized,
        primary,
        atlas,
        cohort="primary_clean261",
        endpoint=builder.H11_ENDPOINT,
        resource_effect="OE_vs_EV",
        raw_effect="construct_OE_minus_EV",
    )
    assert row["estimate"] == pytest.approx(0.87)
    assert row["ci_low"] == pytest.approx(0.79)
    assert row["ci_high"] == pytest.approx(0.95)
    assert raw["p_value_model_BH_FDR"] == pytest.approx(0.04)
    assert raw["raw_effect_estimate"] == pytest.approx(6.0)
    assert raw["raw_effect_bootstrap_replicates"] == 5000
    assert {
        record["source_role"] for record in evidence
    } == {
        "figure_resource:phenotype_effects",
        "figure_source_input:analysis_primary_table",
        "figure_resource:multitrait_atlas",
    }
    assert {record["source_file_sha256"] for record in evidence} == {
        context.resources["phenotype_effects"].file_sha256,
        context.source_inputs["analysis_primary_table"].file_sha256,
        context.resources["multitrait_atlas"].file_sha256,
    }


@pytest.mark.parametrize(
    ("surface", "column", "delta", "message"),
    (
        (
            "normalized",
            "raw_effect_ci_low",
            0.5,
            "normalized/raw/atlas companion differs",
        ),
        (
            "atlas",
            "standardized_ci95_high",
            0.5,
            "normalized/raw/atlas companion differs",
        ),
    ),
)
def test_h11_three_way_numeric_companion_drift_fails_closed(
    tmp_path: Path,
    surface: str,
    column: str,
    delta: float,
    message: str,
) -> None:
    context = _context(tmp_path / f"fixture-{surface}")
    sources = builder.Sources(context)
    normalized = sources.table("figure_resource", "phenotype_effects")
    primary = sources.table("figure_source_input", "analysis_primary_table")
    atlas_source = sources.json_resource("multitrait_atlas")
    atlas_payload = deepcopy(atlas_source.payload)
    if surface == "normalized":
        target = normalized.index[
            normalized["cohort"].eq("primary_clean261")
            & normalized["endpoint_key"].eq(builder.H11_ENDPOINT)
            & normalized["effect_key"].eq("OE_vs_EV")
        ]
        assert len(target) == 1
        normalized.loc[target[0], column] += delta
    else:
        descriptor = next(
            record
            for record in atlas_payload["descriptors"]
            if record["field"] == builder.H11_ENDPOINT
        )
        descriptor["cohorts"]["primary_clean261"]["effects"]["OE_vs_EV"][
            column
        ] += delta
        atlas_source = JsonSource(
            role=atlas_source.role,
            path=atlas_source.path,
            raw=atlas_source.raw,
            payload=atlas_payload,
            file_sha256=atlas_source.file_sha256,
            logical_identity_sha256=atlas_source.logical_identity_sha256,
        )
    with pytest.raises(ManuscriptValuesError, match=message):
        builder._effect_sources(
            sources,
            normalized,
            primary,
            atlas_source,
            cohort="primary_clean261",
            endpoint=builder.H11_ENDPOINT,
            resource_effect="OE_vs_EV",
            raw_effect="construct_OE_minus_EV",
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("raw_effect_estimand", "wrong_estimand"),
        ("raw_effect_interval_method", "wrong_interval"),
        ("raw_effect_bootstrap_replicates", 4999),
        ("raw_effect_bootstrap_seed", 1),
    ),
)
def test_h11_contract_fails_even_when_all_three_surfaces_share_the_same_wrong_metadata(
    tmp_path: Path,
    field: str,
    wrong_value: object,
) -> None:
    context = _context(tmp_path / f"fixture-{field}")
    sources = builder.Sources(context)
    normalized = sources.table("figure_resource", "phenotype_effects")
    primary = sources.table("figure_source_input", "analysis_primary_table")
    atlas_source = sources.json_resource("multitrait_atlas")
    normalized_column = field
    normalized_target = normalized.index[
        normalized["cohort"].eq("primary_clean261")
        & normalized["endpoint_key"].eq(builder.H11_ENDPOINT)
        & normalized["effect_key"].eq("OE_vs_EV")
    ]
    raw_target = primary.index[
        primary["cohort"].eq("primary_clean261")
        & primary["endpoint"].eq(builder.H11_ENDPOINT)
        & primary["effect"].eq("construct_OE_minus_EV")
    ]
    assert len(normalized_target) == len(raw_target) == 1
    normalized.loc[normalized_target[0], normalized_column] = wrong_value
    primary.loc[raw_target[0], field] = wrong_value
    atlas_payload = deepcopy(atlas_source.payload)
    descriptor = next(
        record
        for record in atlas_payload["descriptors"]
        if record["field"] == builder.H11_ENDPOINT
    )
    descriptor["cohorts"]["primary_clean261"]["effects"]["OE_vs_EV"][
        field
    ] = wrong_value
    changed_atlas = JsonSource(
        role=atlas_source.role,
        path=atlas_source.path,
        raw=atlas_source.raw,
        payload=atlas_payload,
        file_sha256=atlas_source.file_sha256,
        logical_identity_sha256=atlas_source.logical_identity_sha256,
    )
    with pytest.raises(
        ManuscriptValuesError,
        match="H11 raw-median bootstrap contract changed",
    ):
        builder._effect_sources(
            sources,
            normalized,
            primary,
            changed_atlas,
            cohort="primary_clean261",
            endpoint=builder.H11_ENDPOINT,
            resource_effect="OE_vs_EV",
            raw_effect="construct_OE_minus_EV",
        )


def test_manuscript_rejects_non_h11_median_companion_on_all_three_surfaces(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path / "fixture-non-h11")
    sources = builder.Sources(context)
    normalized = sources.table("figure_resource", "phenotype_effects")
    primary = sources.table("figure_source_input", "analysis_primary_table")
    atlas_source = sources.json_resource("multitrait_atlas")
    endpoint = "local_hair_count_1_4mm"
    normalized.loc[
        normalized["cohort"].eq("primary_clean261")
        & normalized["endpoint_key"].eq(endpoint)
        & normalized["effect_key"].eq("OE_vs_EV"),
        "raw_effect_estimand",
    ] = builder.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
    primary.loc[
        primary["cohort"].eq("primary_clean261")
        & primary["endpoint"].eq(endpoint)
        & primary["effect"].eq("construct_OE_minus_EV"),
        "raw_effect_estimand",
    ] = builder.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
    atlas_payload = deepcopy(atlas_source.payload)
    descriptor = next(
        record for record in atlas_payload["descriptors"] if record["field"] == endpoint
    )
    descriptor["cohorts"]["primary_clean261"]["effects"]["OE_vs_EV"][
        "raw_effect_estimand"
    ] = builder.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
    changed_atlas = JsonSource(
        role=atlas_source.role,
        path=atlas_source.path,
        raw=atlas_source.raw,
        payload=atlas_payload,
        file_sha256=atlas_source.file_sha256,
        logical_identity_sha256=atlas_source.logical_identity_sha256,
    )
    with pytest.raises(
        ManuscriptValuesError,
        match="non-H11 raw-mean companion contract changed",
    ):
        builder._effect_sources(
            sources,
            normalized,
            primary,
            changed_atlas,
            cohort="primary_clean261",
            endpoint=endpoint,
            resource_effect="OE_vs_EV",
            raw_effect="construct_OE_minus_EV",
        )


def test_all_2_power_15_headline_masks_select_exactly_one_bounded_branch() -> None:
    branch_counts = {"A": 0, "B": 0, "C": 0}
    effect_labels = {
        "OE_vs_EV": "OE-versus-EV construct-label contrast",
        "30C_vs_22C": "30-versus-22°C temperature contrast",
        "interaction": "construct-by-temperature interaction",
    }
    for mask in range(1 << 15):
        rows = []
        for endpoint_index, endpoint in enumerate(builder.NARRATIVE_ENDPOINT_ORDER):
            for effect_index, effect in enumerate(builder.NARRATIVE_EFFECT_ORDER):
                supported = bool(mask & (1 << (endpoint_index * 3 + effect_index)))
                for cohort in ("primary_clean261", "sensitivity_full283"):
                    rows.append(
                        {
                            "endpoint_key": endpoint,
                            "effect_key": effect,
                            "cohort": cohort,
                            "estimate": 1.2 if supported else 1.0,
                            "ci_low": 1.1 if supported and cohort == "primary_clean261" else 0.9,
                            "ci_high": 1.3 if supported else 1.1,
                            "endpoint_n": 10,
                            "effect_scale": "ratio",
                        }
                    )
        decision = build_narrative_decision(rows, source_sha256={"fixture": "a" * 64})
        branch = decision["branch_id"]
        branch_counts[branch] += 1
        expected_b = any(
            any(mask & (1 << (endpoint * 3 + effect)) for endpoint in range(3))
            and any(mask & (1 << (endpoint * 3 + effect)) for endpoint in range(3, 5))
            for effect in range(3)
        )
        assert branch == ("B" if expected_b else "A" if mask else "C")
        synthesis, _sources, parameters = builder._abstract_biology_synthesis(
            decision,
            {"source_role": "fixture"},
        )
        assert parameters["narrative_branch_id"] == branch
        assert parameters["narrative_decision_identity_sha256"] == decision[
            "narrative_decision_identity_sha256"
        ]
        assert builder.text_word_count(synthesis) <= 15
        supported_cells = [
            cell for cell in decision["cells"] if cell["headline_supported"]
        ]
        if supported_cells:
            selected = supported_cells[0]
            expected = (
                f"clean-cohort {selected['sentinel']}/{selected['badge']} "
                f"{selected['layer']} was {selected['clean_direction']} for the "
                f"{effect_labels[selected['effect_key']]}"
            )
            assert synthesis == expected
            assert parameters["headline_cell"] == {
                key: selected[key]
                for key in (
                    "endpoint_key",
                    "effect_key",
                    "sentinel",
                    "badge",
                    "layer",
                    "clean_direction",
                )
            }
            assert parameters["headline_priority_endpoint_order"] == list(
                builder.NARRATIVE_ENDPOINT_ORDER
            )
            assert parameters["headline_priority_effect_order"] == list(
                builder.NARRATIVE_EFFECT_ORDER
            )
        else:
            assert branch == "C"
            assert synthesis == (
                "no clean-cohort endpoint–effect cell met the predefined "
                "cross-cohort support rule"
            )
            assert parameters["headline_cell"] is None
            assert "higher" not in synthesis and "lower" not in synthesis
        if branch == "B":
            assert not any(
                term in synthesis.casefold()
                for term in (
                    "across layers",
                    "both layers",
                    "common direction",
                    "shared direction",
                    "co-direction",
                    "effect-aligned",
                )
            )
            assert parameters["branch_b_common_direction_claimed"] is False
        assert not any(
            term in synthesis.casefold()
            for term in (
                "headline rule",
                "fixed decision",
                "sensitivity-unstable",
                "clean/full-concordant",
                "headline-supported",
                "coordinated phenotype",
                "coordinated d15 remodeling",
            )
        )

    assert sum(branch_counts.values()) == 1 << 15
    assert branch_counts["C"] == 1
    assert all(branch_counts[branch] > 0 for branch in ("A", "B"))


def _directional_narrative_rows(
    supported_directions: dict[tuple[str, str], str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for endpoint in builder.NARRATIVE_ENDPOINT_ORDER:
        for effect in builder.NARRATIVE_EFFECT_ORDER:
            direction = supported_directions.get((endpoint, effect))
            estimate = 0.8 if direction == "lower" else 1.2 if direction == "higher" else 1.0
            clean_low = 0.7 if direction == "lower" else 1.1 if direction == "higher" else 0.9
            clean_high = 0.9 if direction == "lower" else 1.3 if direction == "higher" else 1.1
            for cohort in ("primary_clean261", "sensitivity_full283"):
                rows.append(
                    {
                        "endpoint_key": endpoint,
                        "effect_key": effect,
                        "cohort": cohort,
                        "estimate": estimate,
                        "ci_low": clean_low if cohort == "primary_clean261" else 0.7,
                        "ci_high": clean_high if cohort == "primary_clean261" else 1.3,
                        "endpoint_n": 10,
                        "effect_scale": "ratio",
                    }
                )
    return rows


def test_branch_b_headline_reports_one_real_direction_without_cross_layer_vote() -> None:
    endpoint_n, _endpoint_l, _endpoint_f, endpoint_w, _endpoint_a = (
        builder.NARRATIVE_ENDPOINT_ORDER
    )
    decision = build_narrative_decision(
        _directional_narrative_rows(
            {
                (endpoint_n, "OE_vs_EV"): "lower",
                (endpoint_w, "OE_vs_EV"): "higher",
            }
        ),
        source_sha256={"fixture": "b" * 64},
    )
    assert decision["branch_id"] == "B"
    synthesis, _sources, parameters = builder._abstract_biology_synthesis(
        decision,
        {"source_role": "fixture"},
    )
    assert synthesis == (
        "clean-cohort H08/N visible population was lower for the "
        "OE-versus-EV construct-label contrast"
    )
    assert "higher" not in synthesis
    assert "R07/W" not in synthesis
    assert parameters["headline_cell"]["clean_direction"] == "lower"
    assert parameters["branch_b_common_direction_claimed"] is False


def test_abstract_headline_rejects_a_tampered_or_invalid_clean_direction() -> None:
    endpoint = builder.NARRATIVE_ENDPOINT_ORDER[0]
    decision = build_narrative_decision(
        _directional_narrative_rows({(endpoint, "OE_vs_EV"): "lower"}),
        source_sha256={"fixture": "c" * 64},
    )
    tampered = deepcopy(decision)
    tampered["cells"][0]["clean_direction"] = "sideways"
    tampered.pop("narrative_decision_identity_sha256")
    tampered["narrative_decision_identity_sha256"] = sha256_json(tampered)
    with pytest.raises(
        NarrativeDecisionError,
        match="narrative decision differs from fixed decision rule",
    ):
        builder._abstract_biology_synthesis(
            tampered,
            {"source_role": "fixture"},
        )


def test_values_builder_fails_closed_at_250_words_before_payload_sealing() -> None:
    within = "## Abstract\n\n" + " ".join(["root"] * 248) + " {{VALUE}}\n\n## 1. Results\n"
    assert builder._require_builder_abstract_within_limit(
        within, {"VALUE": {"value": "root"}}
    ) == 249
    over = "## Abstract\n\n" + " ".join(["root"] * 249) + " {{VALUE}}\n\n## 1. Results\n"
    with pytest.raises(
        ManuscriptValuesError,
        match=r"Plant Phenomics abstract word limit exceeded: 250 > 249",
    ):
        builder._require_builder_abstract_within_limit(
            over, {"VALUE": {"value": "root"}}
        )


def test_successor_master_has_sub_250_word_worst_case_abstract_and_seven_keywords() -> None:
    master = (
        PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    rendered = master.replace(
        "{{FINAL_D15_ABSTRACT_SYNTHESIS}}",
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
    )
    rendered = re.sub(r"\{\{[^{}]+\}\}", "0", rendered)
    assert abstract_word_count(rendered) < 250
    keyword_line = next(
        line for line in master.splitlines() if line.startswith("**Keywords:**")
    )
    keywords = [
        value.strip()
        for value in keyword_line.removeprefix("**Keywords:**").split(";")
    ]
    assert keywords == [
        "Arabidopsis thaliana",
        "root hair",
        "primary root",
        "plant phenomics",
        "spatial phenotype atlas",
        "quantitative microscopy",
        "reproducible software",
    ]


@pytest.mark.parametrize(
    "synthesis",
    (
        "clean-cohort H08/N visible population was lower for the "
        "OE-versus-EV construct-label contrast",
        "no clean-cohort endpoint–effect cell met the predefined cross-cohort "
        "support rule",
    ),
)
def test_abstract_synthesis_renders_grammatically_in_abstract_and_results(
    synthesis: str,
) -> None:
    master = (
        PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    rendered = master.replace("{{FINAL_D15_ABSTRACT_SYNTHESIS}}", synthesis)
    abstract = rendered.split("## Abstract", maxsplit=1)[1].split(
        "## 1. Introduction", maxsplit=1
    )[0]
    result_34 = rendered.split("### 3.4 ", maxsplit=1)[1].split(
        "### 3.5 ", maxsplit=1
    )[0]
    assert f"use case, {synthesis}." in abstract
    assert f"grid indicated that {synthesis}." in result_34
    assert "showed D15 showed" not in rendered
    assert not synthesis.startswith(("D15 ", "showed "))


def test_optional_reader_scale_localization_and_applicability_tokens_are_closed(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    context = _context(tmp_path / "scale-reader-tokens")
    added = "\n".join(
        f"{{{{{token}}}}}"
        for token in (
            "FINAL_SCALE_LOCALIZATION_ERROR_UM",
            "FINAL_SCALE_LOCALIZATION_N",
            "FINAL_SCALE_LOCALIZATION_CI",
            "FINAL_SCALE_APPLICABILITY_STATEMENT",
            "FINAL_SCALE_ABSENCE_SPECIFICITY_STATUS",
            "FINAL_SCALE_FAIL_CLOSED_EVIDENCE_STATEMENT",
        )
    )
    master_text = context.master_text + "\n" + added
    changed = replace(
        context,
        master_raw=master_text.encode("utf-8"),
        master_text=master_text,
    )
    contract = build_token_source_contract(changed.master_text)
    values = builder.derive_entries(changed, contract)
    assert values["FINAL_SCALE_LOCALIZATION_ERROR_UM"]["value"] == "3.80"
    assert values["FINAL_SCALE_LOCALIZATION_N"]["value"] == 37
    assert "scale_geometry_endpoint_error_um" in values[
        "FINAL_SCALE_LOCALIZATION_CI"
    ]["value"]
    assert "38 images contained an annotated visible scale bar" in values[
        "FINAL_SCALE_APPLICABILITY_STATEMENT"
    ]["value"]
    assert values["FINAL_SCALE_ABSENCE_SPECIFICITY_STATUS"]["value"] == (
        "not_estimable_no_absent_or_untrusted_scale_cases"
    )
    assert "software contract covered by unit tests" in values[
        "FINAL_SCALE_FAIL_CLOSED_EVIDENCE_STATEMENT"
    ]["value"]
    localization_sources = {
        source["source_role"]
        for source in values["FINAL_SCALE_LOCALIZATION_ERROR_UM"]["derivation"][
            "sources"
        ]
    }
    assert {
        "figure_resource:assurance_metrics",
        "figure_resource:assurance_pairs",
        "figure_provenance:measurement_assurance",
    }.issubset(localization_sources)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "coverage_drift",
            "scale detection coverage does not equal instances/n",
        ),
        (
            "relative_error_denominator_drift",
            "scale localization/calibration denominator does not equal detected count",
        ),
    ),
)
def test_scale_detection_value_instances_and_conditional_denominator_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    from dataclasses import replace

    context = _context(tmp_path / mutation)
    original = context.resources["assurance_metrics"]
    frame = pd.read_csv(original.path, float_precision="round_trip")
    if mutation == "coverage_drift":
        index = frame[frame["metric_key"] == "scale_detection_coverage"].index[0]
        frame.loc[index, "value"] = 0.5
    else:
        index = frame[frame["metric_key"] == "scale_relative_error_percent"].index[0]
        frame.loc[index, "n"] = 42
    changed_metrics = _csv(
        tmp_path / mutation,
        "assurance_metrics_changed",
        frame,
        str(original.container_identity_sha256),
    )
    resources = dict(context.resources)
    resources["assurance_metrics"] = changed_metrics
    changed = replace(context, resources=resources)
    contract = build_token_source_contract(changed.master_text)
    with pytest.raises(ManuscriptValuesError, match=message):
        builder.derive_entries(changed, contract)


def test_latency_mode_registry_has_two_distinct_readable_labels() -> None:
    persistent = builder._latency_mode_label("sequential_persistent_full283")
    cold_cli = builder._latency_mode_label("sequential_cold_cli_full283")
    assert persistent != cold_cli
    assert "startup excluded per image" in persistent
    assert "startup included per image" in cold_cli
    assert "_" not in persistent
    assert "_" not in cold_cli
    with pytest.raises(ManuscriptValuesError, match="latency mode invalid"):
        builder._latency_mode_label("ambiguous_sequential_mode")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("not_comparable", "latency comparison is not comparable_direct_full283"),
        (
            "batch_speedup_drift",
            "production comparison speedup differs from frozen-v1/PHAxis base walls",
        ),
        (
            "latency_speedup_drift",
            "latency comparison speedup differs from frozen-v1/PHAxis base medians",
        ),
        ("baseline_scope_drift", "frozen-v1 runtime excludes scope or uses cache"),
    ),
)
def test_runtime_comparison_is_recomputed_and_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    from dataclasses import replace

    context = _context(tmp_path / mutation)
    original = context.resources["runtime_summary"]
    payload = json.loads(original.path.read_text(encoding="utf-8"))
    if mutation == "not_comparable":
        payload["latency_comparison"]["status"] = "not_comparable"
        payload["latency_comparison"]["comparable"] = False
        payload["latency_comparison"]["noncomparability_reasons"] = ["hardware_identity_sha256"]
    elif mutation == "batch_speedup_drift":
        payload["production_comparison"][
            "batch_wall_speedup_frozen_v1_over_phaxis"
        ] = 2.01
    elif mutation == "latency_speedup_drift":
        payload["latency_comparison"][
            "median_latency_speedup_frozen_v1_over_phaxis"
        ] = 1.99
    else:
        payload["baseline_sequential_latency_full283"]["measurement_scope"] = (
            "component_only_noncomparable"
        )
    changed_runtime = _json_file(
        tmp_path / mutation,
        "runtime_summary_changed",
        payload,
        str(original.container_identity_sha256),
    )
    resources = dict(context.resources)
    resources["runtime_summary"] = changed_runtime
    changed = replace(context, resources=resources)
    contract = build_token_source_contract(changed.master_text)
    with pytest.raises(ManuscriptValuesError, match=message):
        builder.derive_entries(changed, contract)


def test_effect_pattern_treats_intervals_crossing_one_as_neutral() -> None:
    points, effects, cache = _effect_pattern_inputs()
    pattern, selected = builder._effect_qualified_pattern(
        prefix="ABUNDANCE",
        endpoint=builder.ENDPOINTS["ABUNDANCE"],
        points=points,
        effects=effects,
        effect_cache=cache,
    )
    assert len(selected) == 12
    assert "source-unit n=" not in pattern
    assert pattern.count("clean-cohort interval spanning the no-difference ratio") == 3
    assert pattern.count("Full283 estimate retained the same direction") == 3
    assert pattern.index("OE-labelled:EV contrast") < pattern.index(
        "30:22°C contrast"
    ) < pattern.index("construct-by-temperature interaction")


def test_clean_full_direction_uses_mutually_exclusive_point_estimate_states() -> None:
    # Confidence intervals are deliberately crossing the ratio null: they are
    # irrelevant to this sensitivity descriptor, which compares point direction.
    clean = {"estimate": 1.2, "ci_low": 0.8, "ci_high": 1.4}
    full = {"estimate": 1.1, "ci_low": 0.7, "ci_high": 1.5}
    assert builder._ratio_point_direction(clean["estimate"]) == "higher"
    assert builder._ratio_point_direction(full["estimate"]) == "higher"
    assert builder._ratio_point_direction(0.8) == "lower"
    assert builder._ratio_point_direction(1.2) == "higher"
    assert builder._ratio_point_direction(1.0) == "null"
    assert builder._ratio_point_direction(0.8) != builder._ratio_point_direction(1.2)
    assert builder._ratio_point_direction(1.0) == builder._ratio_point_direction(1.0)
    assert builder._ratio_point_direction(1.0) not in {
        builder._ratio_point_direction(0.8),
        builder._ratio_point_direction(1.2),
    }
    assert builder._ratio_point_relation(0.8, 0.9) == ("same", "lower", "lower")
    assert builder._ratio_point_relation(1.2, 0.8) == (
        "opposite",
        "higher",
        "lower",
    )
    assert builder._ratio_point_relation(1.0, 1.2) == (
        "null_transition",
        "null",
        "higher",
    )


def test_effect_pattern_withholds_clean_full_direction_reversal() -> None:
    points, effects, cache = _effect_pattern_inputs(
        estimates=(0.8, 1.05, 0.95),
        intervals=((0.7, 0.9), (0.9, 1.2), (0.8, 1.1)),
        full_estimates=(1.2, 1.06, 0.94),
    )
    pattern, _selected = builder._effect_qualified_pattern(
        prefix="ABUNDANCE",
        endpoint=builder.ENDPOINTS["ABUNDANCE"],
        points=points,
        effects=effects,
        effect_cache=cache,
    )
    assert "Full283 estimate pointed in the opposite direction" in pattern
    assert "OE-labelled:EV contrast was lower" in pattern
    assert "sensitivity-unstable" not in pattern
    assert "headline rule" not in pattern


@pytest.mark.parametrize(
    ("clean_estimate", "full_estimate", "interval", "expected_fragment"),
    [
        (1.2, 1.0, (1.1, 1.3), "Full283 estimate was at no difference"),
        (0.8, 1.0, (0.7, 0.9), "Full283 estimate was at no difference"),
        (
            1.0,
            1.2,
            (0.9, 1.1),
            "Full283 estimate was higher, so the two point-estimate states did not match",
        ),
        (
            1.0,
            0.8,
            (0.9, 1.1),
            "Full283 estimate was lower, so the two point-estimate states did not match",
        ),
        (
            1.0,
            1.0,
            (0.9, 1.1),
            "both point estimates were at the no-difference ratio",
        ),
    ],
)
def test_effect_pattern_distinguishes_null_transition_from_opposite_direction(
    clean_estimate: float,
    full_estimate: float,
    interval: tuple[float, float],
    expected_fragment: str,
) -> None:
    points, effects, cache = _effect_pattern_inputs(
        estimates=(clean_estimate,) * 3,
        intervals=(interval,) * 3,
        full_estimates=(full_estimate,) * 3,
    )
    pattern, _selected = builder._effect_qualified_pattern(
        prefix="ABUNDANCE",
        endpoint=builder.ENDPOINTS["ABUNDANCE"],
        points=points,
        effects=effects,
        effect_cache=cache,
    )
    assert expected_fragment in pattern
    assert "opposite direction" not in pattern


def test_effect_pattern_rejects_nonclosing_endpoint_denominator() -> None:
    points, effects, cache = _effect_pattern_inputs(endpoint_n=11)
    with pytest.raises(
        ManuscriptValuesError,
        match="four-cell non-null counts do not close to effect endpoint n",
    ):
        builder._effect_qualified_pattern(
            prefix="ABUNDANCE",
            endpoint=builder.ENDPOINTS["ABUNDANCE"],
            points=points,
            effects=effects,
            effect_cache=cache,
        )


def test_profile_spike_is_reported_as_a_descriptive_condition_peak(
    tmp_path: Path,
) -> None:
    source = _profiles(tmp_path, "f" * 64)
    frame = pd.read_csv(source.path)
    spike = (
        (frame["metric_key"] == "identity_abundance")
        & (frame["condition_code"] == builder.GROUPS[-1])
        & (frame["bin_start_mm"] == 4)
    )
    frame.loc[spike, ["estimate", "ci_low", "ci_high"]] = [1_000_000, 999_999, 1_000_001]
    pattern, selected = builder._profile_pattern(frame, "identity_abundance")
    assert len(selected) == 20
    assert "source-unit visible-hair abundance profile" in pattern
    assert "descriptive peak bins were" in pattern
    assert "OE-30°C [4,5) mm (1000000.0)" in pattern
    assert "OE-30°C 32.0→1000000.0 (rose)" in pattern
    assert pattern.count(". First-to-last-bin changes were ") == 1
    assert "profile contrast" not in pattern


def test_profile_support_reports_condition_order_peak_trend_and_range(
    tmp_path: Path,
) -> None:
    source = _profiles(tmp_path, "a" * 64)
    frame = pd.read_csv(source.path)
    pattern, _selected = builder._profile_pattern(frame, "length_support_fraction")
    offsets = [pattern.index(builder.SHORT_GROUP_LABELS[group]) for group in builder.GROUPS]
    assert offsets == sorted(offsets)
    assert "source-unit endpoint-complete length support profile" in pattern
    assert pattern.count(". First-to-last-bin changes were ") == 1
    assert pattern.count("(rose; five-bin range") == 4
    assert "EV-22°C 60.0%→64.0% (rose; five-bin range 60.0%–64.0%)" in pattern
    assert "OE-30°C 72.0%→76.0% (rose; five-bin range 72.0%–76.0%)" in pattern
    assert "eligible n" not in pattern


def test_profile_rejects_duplicate_bin_even_when_row_count_is_unchanged(
    tmp_path: Path,
) -> None:
    source = _profiles(tmp_path, "b" * 64)
    frame = pd.read_csv(source.path)
    selected = (
        (frame["metric_key"] == "identity_abundance")
        & (frame["condition_code"] == builder.GROUPS[0])
        & (frame["bin_start_mm"] == 0)
    )
    frame.loc[selected, ["bin_start_mm", "bin_end_mm"]] = [1, 2]
    with pytest.raises(
        ManuscriptValuesError,
        match=r"prespecified \[0,5\) mm bins changed",
    ):
        builder._profile_pattern(frame, "identity_abundance")


def test_multitrait_atlas_summary_uses_all_five_families_without_directional_voting() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    payload = _multitrait_atlas(contract)
    text, parameters = builder._multitrait_atlas_fingerprint(payload)
    assert [text.index(builder.MEASUREMENT_FAMILY_COMPACT_LABELS[family]) for family in builder.MEASUREMENT_FAMILY_ORDER] == sorted(
        text.index(builder.MEASUREMENT_FAMILY_COMPACT_LABELS[family])
        for family in builder.MEASUREMENT_FAMILY_ORDER
    )
    assert "condition-resolved 32-descriptor atlas" in text
    assert "missing cells remain unfilled" in text
    assert "no cross-trait ranking or directional vote was used" in text
    assert len(text.split()) <= 65
    length = parameters["family_coverage"]["conditional_projected_length"]
    assert length["descriptor_n"] == 6
    assert length["fully_observed_trait_n"] == 5
    assert length["incomplete_trait_n"] == 1
    assert length["observability_min"] == 0.0
    assert parameters["cross_trait_directional_vote_used"] is False
    assert "unique_maximum_by_condition" not in length


def test_multitrait_fingerprint_rejects_observability_denominator_drift() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    payload = _multitrait_atlas(contract)
    payload["descriptors"][0]["cohorts"]["primary_clean261"][
        "condition_summaries"
    ][builder.GROUPS[0]]["observability_fraction"] = 0.5
    payload.pop("atlas_identity_sha256")
    payload["atlas_identity_sha256"] = sha256_json(payload)
    with pytest.raises(
        ManuscriptValuesError,
        match="observability denominator does not close",
    ):
        builder._multitrait_atlas_fingerprint(payload)


def test_values_compiler_rejects_nonzero_profile_trait_crosscheck(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    context = _context(tmp_path / "profile-mismatch")
    original = context.evidence_artifacts["profiles"]
    payload = deepcopy(original.payload)
    payload.pop("export_identity_sha256")
    payload["locked_1_4mm_trait_crosscheck_mismatches"] = 1
    changed_profile = _sealed_json(
        tmp_path / "profile-mismatch",
        "profiles_with_mismatch",
        payload,
        "export_identity_sha256",
    )
    artifacts = dict(context.evidence_artifacts)
    artifacts["profiles"] = changed_profile
    changed = replace(context, evidence_artifacts=artifacts)
    contract = build_token_source_contract(changed.master_text)
    with pytest.raises(
        ManuscriptValuesError,
        match=r"profile-to-trait \[1,4\) mm crosscheck has one or more mismatches",
    ):
        builder.derive_entries(changed, contract)


def test_optional_root_trait_assurance_tokens_are_stable_and_machine_derived(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path / "root-trait-tokens")
    extra = "\n" + "\n".join(f"{{{{{token}}}}}" for token in ROOT_TRAIT_ASSURANCE_TOKENS)
    modified = context.master_text + extra
    from dataclasses import replace

    modified_context = replace(
        context,
        master_raw=modified.encode("utf-8"),
        master_text=modified,
    )
    contract = build_token_source_contract(modified)
    derived = builder.derive_entries(modified_context, contract)
    assert derived["FINAL_ROOT_TRAIT_VALIDATED_N"]["value"] == 19
    assert derived["FINAL_ROOT_TRAIT_VALIDATION_IMAGE_N"]["value"] == 44
    assert derived["FINAL_ROOT_TRAIT_ELIGIBLE_N_RANGE"]["value"] == "44–44"
    assert derived["FINAL_ROOT_TRAIT_OBSERVABILITY_RANGE_PERCENT"]["value"] == "100.0–100.0"
    assert derived["FINAL_ROOT_TRAIT_CCC_ESTIMABLE_N"]["value"] == 19
    assert "canonical mask-plus-distal-point reference" in derived[
        "FINAL_ROOT_TRAIT_AGREEMENT_SUMMARY"
    ]["value"]
    assert "provider_equivalence_used_as_accuracy" in derived[
        "FINAL_ROOT_TRAIT_AGREEMENT_SUMMARY"
    ]["derivation"]["parameters"]


@pytest.mark.parametrize("mutation", ["missing_trait", "denominator_drift", "value_hash_drift"])
def test_values_compiler_rejects_incomplete_or_drifted_root_trait_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    from dataclasses import replace

    context = _context(tmp_path / mutation)
    pair_source = context.resources["assurance_pairs"]
    pairs = pd.read_csv(pair_source.path, float_precision="round_trip")
    if mutation == "missing_trait":
        pairs = pairs[
            ~(
                (pairs["pair_type"].astype(str) == "root_trait")
                & (pairs["trait_id"].astype(str) == "R19")
            )
        ]
    elif mutation == "denominator_drift":
        index = pairs[
            (pairs["pair_type"].astype(str) == "root_trait")
            & (pairs["trait_id"].astype(str) == "R01")
        ].index[0]
        pairs = pairs.drop(index=index)
    else:
        index = pairs[
            (pairs["pair_type"].astype(str) == "root_trait")
            & (pairs["trait_id"].astype(str) == "R01")
        ].index[0]
        pairs.loc[index, "predicted"] = float(pairs.loc[index, "predicted"]) + 1.0
    replacement = _csv(
        tmp_path / mutation,
        f"assurance_pairs_{mutation}",
        pairs,
        pair_source.container_identity_sha256,
    )
    resources = dict(context.resources)
    resources["assurance_pairs"] = replacement
    original_receipt = context.provenance_receipts["measurement_assurance"]
    receipt_payload = deepcopy(original_receipt.payload)
    receipt_payload.pop("measurement_assurance_identity_sha256")
    receipt_payload["source_table_sha256"]["pairs"] = replacement.file_sha256
    changed_receipt = _sealed_json(
        tmp_path / mutation,
        f"measurement_assurance_{mutation}",
        receipt_payload,
        "measurement_assurance_identity_sha256",
    )
    changed = replace(
        context,
        resources=resources,
        provenance_receipts={"measurement_assurance": changed_receipt},
    )
    contract = build_token_source_contract(changed.master_text)
    with pytest.raises(RootTraitAssuranceError, match="denominator|identity drift"):
        builder.derive_entries(changed, contract)


def test_values_compiler_rejects_provider_equivalence_as_root_trait_accuracy(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    context = _context(tmp_path / "provider-masquerade")
    original = context.provenance_receipts["measurement_assurance"]
    payload = deepcopy(original.payload)
    payload.pop("measurement_assurance_identity_sha256")
    payload["root_trait_assurance"]["evidence_role"] = (
        "exact_portable_provider_equivalence"
    )
    payload["root_trait_assurance"]["provider_equivalence_used_as_accuracy"] = True
    payload["root_trait_assurance"]["root_trait_assurance_identity_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload["root_trait_assurance"].items()
            if key != "root_trait_assurance_identity_sha256"
        }
    )
    changed_receipt = _sealed_json(
        tmp_path / "provider-masquerade",
        "measurement_assurance_changed",
        payload,
        "measurement_assurance_identity_sha256",
    )
    changed = replace(
        context,
        provenance_receipts={"measurement_assurance": changed_receipt},
    )
    contract = build_token_source_contract(changed.master_text)
    with pytest.raises(RootTraitAssuranceError, match="cannot masquerade"):
        builder.derive_entries(changed, contract)


def test_nullable_visibility_flags_are_excluded_not_coerced_to_censored() -> None:
    observed, visible = builder._nullable_bool(
        pd.Series([True, False, None, ""]),
        "test visibility",
    )
    assert observed.tolist() == [True, True, False, False]
    assert visible.tolist() == [True, False, False, False]
    assert builder._fmt_cell_fraction([1, 0, 2, 1], [2, 1, 4, 2]) == (
        "1/2 (50.0%); 0/1 (0.0%); 2/4 (50.0%); 1/2 (50.0%)"
    )


def test_source_cell_or_file_drift_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path / "fixture")
    path = context.resources["development_per_image"].path
    frame = pd.read_csv(path)
    frame.loc[0, "biological_presence_tp_20um"] -= 1
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    contract = build_token_source_contract(context.master_text)
    with pytest.raises(ManuscriptValuesError, match="source file drift"):
        builder.derive_entries(context, contract)


def test_sealed_derivation_cell_tamper_is_rejected(tmp_path: Path) -> None:
    context, contract, payload = _built(tmp_path / "fixture")
    changed = deepcopy(payload)
    derivation = changed["values"]["FINAL_ROOT_DICE"]["derivation"]
    derivation["sources"][0]["source_value"] = "tampered-cell"
    derivation.pop("derivation_identity_sha256")
    derivation["derivation_identity_sha256"] = sha256_json(derivation)
    changed.pop("values_identity_sha256")
    changed["values_identity_sha256"] = sha256_json(changed)
    with pytest.raises(ManuscriptValuesError, match="source cell identity mismatch"):
        validate_values_payload(
            changed,
            master_raw=context.master_raw,
            evidence_graph_raw=context.evidence_graph.raw,
            evidence_graph_identity_sha256=str(
                context.evidence_graph.logical_identity_sha256
            ),
            token_contract=contract,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_machine_token_is_rejected(tmp_path: Path, mutation: str) -> None:
    context = _context(tmp_path / "fixture")
    contract = build_token_source_contract(context.master_text)
    entries = builder.derive_entries(context, contract)
    if mutation == "missing":
        entries.pop("FINAL_ROOT_DICE")
    else:
        entries["EXTRA_MACHINE_TOKEN"] = deepcopy(entries["FINAL_ROOT_DICE"])
    with pytest.raises(ManuscriptValuesError, match="token key mismatch"):
        assemble_values_payload(context=context, token_contract=contract, entries=entries)


def test_missing_human_metadata_has_explicit_fields_and_cannot_validate() -> None:
    payload = {
        "schema_version": HUMAN_METADATA_SCHEMA,
        "status": "complete_author_verified_external_metadata",
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "values": _human_values(),
    }
    payload["values"].pop("FINAL_FUNDING_STATEMENT")
    payload["human_metadata_identity_sha256"] = sha256_json(payload)
    with pytest.raises(HumanMetadataError) as captured:
        validate_human_metadata(payload)
    assert captured.value.missing == ("FINAL_FUNDING_STATEMENT",)


def test_human_metadata_rejects_stale_public_release_tag() -> None:
    payload = {
        "schema_version": HUMAN_METADATA_SCHEMA,
        "status": "complete_author_verified_external_metadata",
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "values": _human_values(),
    }
    payload["values"]["PHAXIS_RELEASE_TAG"] = "v1.0"
    payload["human_metadata_identity_sha256"] = sha256_json(payload)
    with pytest.raises(HumanMetadataError, match="invalid human metadata fields") as captured:
        validate_human_metadata(payload)
    assert captured.value.invalid == {
        "PHAXIS_RELEASE_TAG": "invalid_public_version_tag"
    }


def test_structured_acquisition_metadata_is_exact_and_deferred_values_fail_closed() -> None:
    manuscript = (
        PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    template = json.loads(
        (
            PROJECT_ROOT
            / "configs/phaxis/v1_0/POST_TRAINING_MANUSCRIPT_METADATA_TEMPLATE.json"
        ).read_text(encoding="utf-8")
    )
    assert "FINAL_BIOLOGICAL_ACQUISITION_METHODS" not in manuscript
    assert "FINAL_BIOLOGICAL_ACQUISITION_METHODS" not in HUMAN_METADATA_TOKENS
    assert BIOLOGICAL_ACQUISITION_TOKENS <= set(template["values"])
    assert all(
        manuscript.count(f"{{{{{token}}}}}") == 1
        for token in BIOLOGICAL_ACQUISITION_TOKENS
    )
    assert {
        template["values"][token] for token in BIOLOGICAL_ACQUISITION_TOKENS
    } == {"DEFERRED_AUTHOR_VERIFICATION"}

    payload = deepcopy(template)
    payload["status"] = "complete_author_verified_external_metadata"
    payload["values"] = _human_values()
    deferred_token = sorted(BIOLOGICAL_ACQUISITION_TOKENS)[0]
    payload["values"][deferred_token] = "DEFERRED_AUTHOR_VERIFICATION"
    payload["human_metadata_identity_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "human_metadata_identity_sha256"}
    )
    with pytest.raises(HumanMetadataError, match="invalid human metadata fields") as captured:
        validate_human_metadata(payload)
    assert captured.value.invalid == {deferred_token: "placeholder_or_nonfinal"}


def test_cli_missing_human_file_writes_invalid_completion_template(tmp_path: Path) -> None:
    output = tmp_path / "values.json"
    report = tmp_path / "missing-human.json"
    argv = [
        "--master",
        str(PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"),
        "--evidence-graph",
        str(tmp_path / "not-reached-evidence.json"),
        "--figure-inputs",
        str(tmp_path / "not-reached-figures.json"),
        "--figure-assembly-summary",
        str(tmp_path / "not-reached-assembly.json"),
        "--model-contract-proposal",
        str(tmp_path / "not-reached-proposal.json"),
        "--human-metadata",
        str(tmp_path / "missing-human-source.json"),
        "--model-bundle-manifest",
        str(tmp_path / "not-reached-bundle.json"),
        "--clean-install-receipt",
        str(tmp_path / "not-reached-install.json"),
        "--output",
        str(output),
        "--missing-human-report",
        str(report),
    ]
    for role in EVIDENCE_ARTIFACT_ROLES:
        argv.extend(["--evidence-artifact", f"{role}={tmp_path / (role + '.json')}"])
    assert builder.main(argv) == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert set(payload["missing_fields"]) == set(HUMAN_METADATA_TOKENS)
    assert payload["formal_values_build_allowed"] is False
    assert payload["template"]["status"] == "INCOMPLETE_DO_NOT_USE"
    assert {
        payload["template"]["values"][token]
        for token in BIOLOGICAL_ACQUISITION_TOKENS
    } == {"DEFERRED_AUTHOR_VERIFICATION"}
    assert not output.exists()


def test_provisional_and_candidate_final_values_are_rejected(tmp_path: Path) -> None:
    context, contract, payload = _built(tmp_path / "fixture")
    for token, value, message in (
        ("FINAL_ROOT_DICE", "provisional 0.9", "provisional value is forbidden"),
        ("FINAL_ROOT_DICE", "443CV 0.9", "legacy 443CV deployment value"),
        (
            "FINAL_ROOT_DICE",
            "PHAXIS-V1.0-FROZEN-ROOT-PROVIDER",
            "internal ABI or stale PHAxis public version",
        ),
        ("FINAL_HAIR_EXPERT_ID", "candidate-five-fold", "neutral final train399"),
        (
            "FINAL_BIOLOGICAL_ACCESSION",
            "DEFERRED_AUTHOR_VERIFICATION",
            "deferred external metadata is forbidden",
        ),
    ):
        changed = deepcopy(payload)
        changed["values"][token]["value"] = value
        changed.pop("values_identity_sha256")
        changed["values_identity_sha256"] = sha256_json(changed)
        with pytest.raises(ManuscriptValuesError, match=message):
            validate_values_payload(
                changed,
                master_raw=context.master_raw,
                evidence_graph_raw=context.evidence_graph.raw,
                evidence_graph_identity_sha256=str(context.evidence_graph.logical_identity_sha256),
                token_contract=contract,
            )


def test_public_root_identity_must_be_derived_from_stable_bundle(tmp_path: Path) -> None:
    context, contract, payload = _built(tmp_path / "fixture")
    changed = deepcopy(payload)
    changed["root_expert_id"] = "PHAxis-root-provider-" + "E" * 20
    changed.pop("values_identity_sha256")
    changed["values_identity_sha256"] = sha256_json(changed)
    with pytest.raises(ManuscriptValuesError, match="public root-expert identity"):
        validate_values_payload(
            changed,
            master_raw=context.master_raw,
            evidence_graph_raw=context.evidence_graph.raw,
            evidence_graph_identity_sha256=str(
                context.evidence_graph.logical_identity_sha256
            ),
            token_contract=contract,
        )


def test_atomic_values_publish_refuses_overwrite(tmp_path: Path) -> None:
    _context_value, _contract, payload = _built(tmp_path / "fixture")
    output = tmp_path / "values.json"
    publish_json_no_overwrite(output, payload)
    owned = output.read_bytes()
    with pytest.raises(ManuscriptValuesError, match="refusing to overwrite"):
        publish_json_no_overwrite(output, payload)
    assert output.read_bytes() == owned
    assert not list(tmp_path.glob(".*.tmp"))
