#!/usr/bin/env python3
"""Build the PHAxis manuscript value receipt from sealed source cells.

This command is intentionally not a key/value templating utility.  Every
machine value is recomputed from the publication figure bundle or a named
formal receipt; author-controlled release and submission metadata enter only
through the separate, sealed human-metadata document.  The resulting values
receipt records both whole-file hashes and selected-cell identities.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.manuscript_values import (  # noqa: E402
    BuildContext,
    EVIDENCE_ARTIFACT_ROLES,
    HAIR_ATTACHMENT_ASSURANCE_TOKENS,
    HISTORICAL_TOKEN_PREFIXES,
    HUMAN_METADATA_TOKENS,
    HumanMetadataError,
    JsonSource,
    FileSource,
    ManuscriptValuesError,
    ROOT_CONTINUITY_ASSURANCE_TOKENS,
    assemble_values_payload,
    build_token_source_contract,
    canonical_json_bytes,
    derivation_source,
    human_metadata_report,
    load_build_context,
    publish_json_no_overwrite,
    read_json_object,
    require,
    seal_derivation,
    sha256_file,
    sha256_json,
    validate_wt_secondary_source_inputs,
)
from phaxis.manuscript_contract import (  # noqa: E402
    ABSTRACT_WORD_LIMIT,
    ManuscriptTextContractError,
    require_abstract_within_limit,
    text_word_count,
)
from phaxis.multitrait_atlas import (  # noqa: E402
    CONDITION_SUMMARY_STATUS,
    EFFECT_KEYS as ATLAS_EFFECT_KEYS,
    H11_ENDPOINT,
    H11_RAW_BOOTSTRAP_BASE_SEED,
    H11_RAW_BOOTSTRAP_REPLICATES,
    MEASUREMENT_FAMILY_ORDER,
    MEASUREMENT_FAMILY_TRAIT_IDS,
    PRIMARY_ENDPOINTS as ATLAS_PRIMARY_ENDPOINTS,
    SCHEMA_VERSION as MULTITRAIT_ATLAS_SCHEMA_VERSION,
)
from phaxis.biological_analysis import (  # noqa: E402
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    raw_median_bootstrap_seed,
)
from phaxis.root_trait_assurance import (  # noqa: E402
    ROOT_TRAIT_ASSURANCE_TOKENS,
    ROOT_TRAIT_FAMILY_ORDER,
    validate_root_trait_assurance,
)
from phaxis.narrative_decision import (  # noqa: E402
    ENDPOINT_CONTRACT as NARRATIVE_ENDPOINT_CONTRACT,
    ENDPOINT_ORDER as NARRATIVE_ENDPOINT_ORDER,
    EFFECT_ORDER as NARRATIVE_EFFECT_ORDER,
    validate_narrative_decision,
)


STAGEB_COMPARATOR = "stageb_train399"
LEGACY_COMPARATOR = "legacy_hybrid_endpoint_complete_identity_layer"
HISTORICAL_COMPARATOR = "historical_family_isolated_oof443"
GROUPS = ("RHD6_EV_22C", "RHD6_EV_30C", "RHD6_OE_22C", "RHD6_OE_30C")
GROUP_LABELS = {
    "RHD6_EV_22C": "EV-labelled plants at 22 °C",
    "RHD6_EV_30C": "EV-labelled plants at 30 °C",
    "RHD6_OE_22C": "OE-labelled plants at 22 °C",
    "RHD6_OE_30C": "OE-labelled plants at 30 °C",
}
ENDPOINTS = {
    "ABUNDANCE": "local_hair_count_1_4mm",
    "LENGTH": "local_median_hair_length_um_1_4mm",
    "FIRST_HAIR": "first_hair_ge40um_distance_from_distal_point_um",
    "ROOT_WIDTH": "median_root_width_um",
    "ROOT_LENGTH": "visible_root_axis_length_um",
}
BIOLOGICAL_NARRATIVE_LAYERS = {
    "primary_hair_change": ("ABUNDANCE", "LENGTH"),
    "spatial_location": ("FIRST_HAIR",),
    "supporting_root_context": ("ROOT_WIDTH", "ROOT_LENGTH"),
}
EFFECTS = {
    "CONSTRUCT": ("OE_vs_EV", "construct_OE_minus_EV"),
    "TEMPERATURE": ("30C_vs_22C", "temperature_30C_minus_22C"),
    "INTERACTION": ("interaction", "construct_by_temperature_interaction"),
}
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_828
SCALE_ABSENCE_SPECIFICITY_STATUS = (
    "not_estimable_no_absent_or_untrusted_scale_cases"
)
SCALE_FAIL_CLOSED_EVIDENCE_BASIS = "software_contract_and_unit_tests"
ROOT_CONTINUITY_METRIC_KEYS = (
    "root_continuity_maximum_single_component_coverage_mean",
    "root_continuity_maximum_single_component_coverage_median",
    "root_continuity_best_component_gap_median_um",
    "root_continuity_break_free_rate",
    "root_continuity_visible_axis_extent_mae_um",
)
HAIR_ATTACHMENT_METRIC_KEYS = (
    "hair_attachment_qualified_precision_20um",
    "hair_attachment_qualified_recall_20um",
    "hair_attachment_qualified_f1_20um",
    "hair_attachment_error_median_um",
    "hair_attachment_error_p95_um",
)

SHORT_GROUP_LABELS = {
    "RHD6_EV_22C": "EV-22°C",
    "RHD6_EV_30C": "EV-30°C",
    "RHD6_OE_22C": "OE-22°C",
    "RHD6_OE_30C": "OE-30°C",
}
MEASUREMENT_FAMILY_LABELS = {
    "visible_hair_abundance": "visible-hair abundance",
    "conditional_projected_length": "endpoint-supported hair length",
    "axial_deployment": "distal-axis deployment",
    "visible_root_extent": "visible-root extent",
    "root_form_trajectory": "root form and trajectory",
}
MEASUREMENT_FAMILY_COMPACT_LABELS = {
    "visible_hair_abundance": "hair abundance",
    "conditional_projected_length": "supported hair length",
    "axial_deployment": "axial deployment",
    "visible_root_extent": "root extent",
    "root_form_trajectory": "root form/trajectory",
}

# Decimal machine values occupy two counter tokens.  The concise 15-word
# synthesis budget keeps every one of the 31 reachable final renderings below
# 250 words even after those actual values replace their placeholders.
# This local budget is only an early diagnostic: derive_entries() also renders
# every actual token value into the master and enforces the journal-facing
# contract on that final abstract.
ABSTRACT_BIOLOGY_SYNTHESIS_WORD_LIMIT = 15


def _finite(value: Any, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ManuscriptValuesError(f"{role}: finite number required") from error
    require(math.isfinite(result), f"{role}: finite number required")
    return result


def _integer(value: Any, role: str) -> int:
    result = _finite(value, role)
    require(result.is_integer(), f"{role}: integer required")
    return int(result)


def _bool_series(series: pd.Series, role: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    require(
        normalized.isin({"true", "false", "1", "0", "yes", "no"}).all(),
        f"{role}: invalid boolean cell",
    )
    return normalized.isin({"true", "1", "yes"})


def _fmt(value: float, digits: int = 3) -> str:
    require(math.isfinite(float(value)), "cannot format non-finite value")
    result = f"{float(value):.{digits}f}"
    if result.startswith("-0.") and float(result) == 0:
        result = result[1:]
    return result


def _fmt_ci(low: float, high: float, digits: int = 3) -> str:
    require(low <= high, "confidence interval is reversed")
    return f"{_fmt(low, digits)}–{_fmt(high, digits)}"


def _fmt_percent(value: float, digits: int = 1) -> str:
    return _fmt(100.0 * value, digits)


def _observed_numeric(series: pd.Series, role: str) -> pd.Series:
    """Return the non-null finite-numeric mask without treating absence as zero."""

    numeric = pd.to_numeric(series, errors="coerce")
    declared = series.notna() & series.astype(str).str.strip().ne("")
    require(
        not bool((declared & numeric.isna()).any()),
        f"{role}: non-null value is not numeric",
    )
    observed = numeric.notna()
    require(
        np.isfinite(numeric[observed].to_numpy(dtype=float)).all(),
        f"{role}: non-finite numeric value",
    )
    return observed


def _nullable_bool(series: pd.Series, role: str) -> tuple[pd.Series, pd.Series]:
    """Return (is_observed, boolean_value); null flags remain non-evaluable."""

    normalized = series.astype(str).str.strip().str.casefold()
    observed = series.notna() & normalized.ne("")
    require(
        normalized[observed].isin({"true", "false", "1", "0", "yes", "no"}).all(),
        f"{role}: invalid non-null boolean cell",
    )
    values = normalized.isin({"true", "1", "yes"})
    return observed, values


def _cell_counts(frame: pd.DataFrame, *, mask: pd.Series | None = None) -> list[int]:
    selected = frame if mask is None else frame.loc[mask]
    counts = selected["condition_code"].astype(str).value_counts()
    return [int(counts.get(group, 0)) for group in GROUPS]


def _fmt_cell_counts(counts: Sequence[int]) -> str:
    require(len(counts) == len(GROUPS), "condition-cell count vector is not four cells")
    require(all(isinstance(value, int) and value >= 0 for value in counts), "invalid condition-cell count")
    return " / ".join(str(value) for value in counts)


def _fmt_cell_fraction(numerators: Sequence[int], denominators: Sequence[int]) -> str:
    require(
        len(numerators) == len(denominators) == len(GROUPS),
        "condition-cell fraction vector is not four cells",
    )
    cells: list[str] = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        require(0 <= numerator <= denominator and denominator > 0, "invalid condition-cell fraction")
        cells.append(f"{numerator}/{denominator} ({_fmt_percent(numerator / denominator)}%)")
    return "; ".join(cells)


def _fmt_cell_median_iqr(
    frame: pd.DataFrame,
    *,
    field: str,
    expected_counts: Sequence[int],
    digits: int,
) -> str:
    """Render four source-unit distributions in the locked biological order."""

    require(len(expected_counts) == len(GROUPS), "raw-summary denominator is not four cells")
    short_labels = ("EV-22°C", "EV-30°C", "OE-22°C", "OE-30°C")
    cells: list[str] = []
    for group, label, expected in zip(GROUPS, short_labels, expected_counts, strict=True):
        selected = pd.to_numeric(
            frame.loc[frame["condition_code"].astype(str) == group, field],
            errors="coerce",
        ).dropna()
        require(
            len(selected) == expected,
            f"{field}: raw-summary cell denominator drift for {group}",
        )
        if expected == 0:
            cells.append(f"{label} not estimable")
            continue
        values = selected.to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"{field}: raw-summary cell is non-finite")
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75), method="linear")
        cells.append(
            f"{label} {_fmt(float(median), digits)} "
            f"[{_fmt(float(q25), digits)}–{_fmt(float(q75), digits)}]"
        )
    return "; ".join(cells)


def _canonical_cell(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float):
            require(math.isfinite(value), "source table contains infinity")
        return value
    return str(value)


def _read_csv(source: FileSource, role: str) -> pd.DataFrame:
    with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ManuscriptValuesError(f"{role}: empty CSV") from error
    require(header and len(header) == len(set(header)), f"{role}: duplicate/empty header")
    try:
        frame = pd.read_csv(
            source.path,
            encoding="utf-8-sig",
            float_precision="round_trip",
        )
    except Exception as error:
        raise ManuscriptValuesError(f"{role}: CSV cannot be parsed") from error
    require(len(frame) > 0, f"{role}: empty CSV")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], role: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{role}: missing columns {missing}")


class Sources:
    """Read sealed bundle files and create selected-cell provenance records."""

    def __init__(self, context: BuildContext) -> None:
        self.context = context
        self._tables: dict[tuple[str, str], pd.DataFrame] = {}
        self._json: dict[str, JsonSource] = {}

    def table(self, namespace: str, role: str) -> pd.DataFrame:
        key = (namespace, role)
        if key not in self._tables:
            mapping = (
                self.context.resources if namespace == "figure_resource" else self.context.source_inputs
            )
            require(role in mapping, f"required {namespace} table missing: {role}")
            require(
                sha256_file(mapping[role].path) == mapping[role].file_sha256,
                f"{namespace}:{role}: source file drift after context validation",
            )
            self._tables[key] = _read_csv(mapping[role], f"{namespace}:{role}")
        return self._tables[key].copy()

    def json_resource(self, role: str) -> JsonSource:
        if role not in self._json:
            require(role in self.context.resources, f"required JSON resource missing: {role}")
            source = self.context.resources[role]
            require(
                sha256_file(source.path) == source.file_sha256,
                f"figure_resource:{role}: source file drift after context validation",
            )
            path, raw, payload = read_json_object(source.path, f"figure resource {role}")
            self._json[role] = JsonSource(
                role=f"figure_resource:{role}",
                path=path,
                raw=raw,
                payload=payload,
                file_sha256=source.file_sha256,
                logical_identity_sha256=None,
            )
        return self._json[role]

    def table_cells(
        self,
        *,
        namespace: str,
        role: str,
        selected: pd.DataFrame,
        columns: Sequence[str],
        locator: Mapping[str, Any],
        authority: str = "final_machine",
        sort_by: Sequence[str] = (),
    ) -> dict[str, Any]:
        mapping = self.context.resources if namespace == "figure_resource" else self.context.source_inputs
        source = mapping[role]
        _require_columns(selected, columns, f"{namespace}:{role} selection")
        scoped = selected.loc[:, list(columns)].copy()
        if sort_by:
            scoped = scoped.sort_values(list(sort_by), kind="stable")
        rows = [
            {column: _canonical_cell(row[column]) for column in columns}
            for row in scoped.to_dict("records")
        ]
        cell_value = {
            "selected_row_count": len(rows),
            "selected_column_count": len(columns),
            "selected_cells_identity_sha256": sha256_json(rows),
        }
        return derivation_source(
            source_role=f"{namespace}:{role}",
            source_file_sha256=source.file_sha256,
            container_identity_sha256=source.container_identity_sha256,
            locator={**dict(locator), "columns": list(columns), "sort_by": list(sort_by)},
            source_value=cell_value,
            authority_class=authority,
        )

    def json_cell(
        self,
        *,
        role: str,
        source: JsonSource,
        pointer: Sequence[str | int],
        authority: str = "final_machine",
    ) -> dict[str, Any]:
        value: Any = source.payload
        for part in pointer:
            if isinstance(part, int):
                require(isinstance(value, list) and 0 <= part < len(value), f"{role}: JSON pointer missing")
                value = value[part]
            else:
                require(isinstance(value, Mapping) and part in value, f"{role}: JSON pointer missing: {part}")
                value = value[part]
        if role.startswith("figure_resource:"):
            resource_role = role.split(":", 1)[1]
            require(
                resource_role in self.context.resources,
                f"{role}: resource container is absent",
            )
            container = self.context.resources[
                resource_role
            ].container_identity_sha256
        else:
            container = (
                source.logical_identity_sha256
                or self.context.evidence_graph.logical_identity_sha256
            )
        require(container is not None, f"{role}: JSON source lacks a container identity")
        return derivation_source(
            source_role=role,
            source_file_sha256=source.file_sha256,
            source_logical_identity_sha256=source.logical_identity_sha256,
            container_identity_sha256=str(container),
            locator={"kind": "json_pointer", "pointer": "/" + "/".join(map(str, pointer))},
            source_value=deepcopy(value),
            authority_class=authority,
        )


class EntryBuilder:
    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = contract
        self.entries: dict[str, dict[str, Any]] = {}

    def add(
        self,
        token: str,
        value: str | int | float | bool,
        operation: str,
        sources: Sequence[Mapping[str, Any]],
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        require(token in self.contract["tokens"], f"unknown manuscript token: {token}")
        require(token not in self.entries, f"duplicate manuscript value: {token}")
        derivation: dict[str, Any] = {
            "operation": operation,
            "sources": [deepcopy(dict(source)) for source in sources],
        }
        if parameters:
            derivation["parameters"] = deepcopy(dict(parameters))
        self.entries[token] = {
            "value": value,
            "source_role": self.contract["tokens"][token]["source_role"],
            "derivation": seal_derivation(derivation),
        }


def _prf(tp: np.ndarray | int, predicted: np.ndarray | int, truth: np.ndarray | int) -> tuple[Any, Any, Any]:
    tp_array = np.asarray(tp, dtype=float)
    pred_array = np.asarray(predicted, dtype=float)
    truth_array = np.asarray(truth, dtype=float)
    precision = np.divide(tp_array, pred_array, out=np.zeros_like(tp_array), where=pred_array > 0)
    recall = np.divide(tp_array, truth_array, out=np.zeros_like(tp_array), where=truth_array > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    if f1.ndim == 0:
        return float(precision), float(recall), float(f1)
    return precision, recall, f1


def _ccc_rows(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Population CCC used by the formal Stage-B count evaluator."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    require(observed.shape == predicted.shape and observed.ndim == 2, "CCC matrix shape mismatch")
    mean_o = observed.mean(axis=1)
    mean_p = predicted.mean(axis=1)
    centered_o = observed - mean_o[:, None]
    centered_p = predicted - mean_p[:, None]
    covariance = np.mean(centered_o * centered_p, axis=1)
    denominator = (
        np.mean(centered_o**2, axis=1)
        + np.mean(centered_p**2, axis=1)
        + (mean_o - mean_p) ** 2
        + 1e-12
    )
    return 2.0 * covariance / denominator


def _ccc_sample_rows(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Sample-moment CCC used by the sealed measurement-assurance producer."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    require(observed.shape == predicted.shape and observed.ndim == 2, "CCC matrix shape mismatch")
    mean_o = observed.mean(axis=1)
    mean_p = predicted.mean(axis=1)
    centered_o = observed - mean_o[:, None]
    centered_p = predicted - mean_p[:, None]
    require(observed.shape[1] >= 2, "CCC requires at least two paired observations")
    divisor = observed.shape[1] - 1
    covariance = np.sum(centered_o * centered_p, axis=1) / divisor
    denominator = (
        np.sum(centered_o**2, axis=1) / divisor
        + np.sum(centered_p**2, axis=1) / divisor
        + (mean_o - mean_p) ** 2
    )
    require(np.all(denominator > 0.0), "CCC denominator is zero")
    return 2.0 * covariance / denominator


def _selected_source(
    sources: Sources,
    *,
    namespace: str,
    role: str,
    selected: pd.DataFrame,
    columns: Sequence[str],
    filters: Mapping[str, Any],
    authority: str = "final_machine",
    sort_by: Sequence[str] = (),
) -> dict[str, Any]:
    return sources.table_cells(
        namespace=namespace,
        role=role,
        selected=selected,
        columns=columns,
        locator={"kind": "csv_selection", "filters": dict(filters)},
        authority=authority,
        sort_by=sort_by,
    )


def _derive_human(context: BuildContext, sources: Sources, entries: EntryBuilder) -> None:
    release_pointers: dict[str, tuple[str, ...]] = {
        "PHAXIS_REPOSITORY_URL": ("project_urls", "Repository"),
        "PHAXIS_RELEASE_TAG": ("release_coordinates", "github_release_tag"),
        "PHAXIS_RELEASE_DOI": ("release_coordinates", "release_doi"),
        "PHAXIS_SOFTWARE_LICENSE": ("rights", "source_license_spdx"),
    }
    for token in sorted(HUMAN_METADATA_TOKENS):
        source = sources.json_cell(
            role="human_metadata",
            source=context.human_metadata,
            pointer=("values", token),
            authority="human_external",
        )
        bound_sources = [source]
        operation = "author_verified_external_metadata"
        parameters: dict[str, Any] | None = None
        if token in release_pointers:
            bound_sources.extend(
                [
                    sources.json_cell(
                        role="source_release_metadata",
                        source=context.source_release_metadata,
                        pointer=release_pointers[token],
                    ),
                    sources.json_cell(
                        role="source_release_manifest",
                        source=context.source_release_manifest,
                        pointer=("tree_identity_sha256",),
                    ),
                ]
            )
            operation = "cross_bind_author_verified_coordinate_to_formal_source_release"
            parameters = {
                "software_release_cross_binding_identity_sha256": context.software_release_cross_binding[
                    "cross_binding_identity_sha256"
                ],
                "exact_equality_required": True,
            }
        entries.add(
            token,
            context.human_values[token],
            operation,
            bound_sources,
            parameters=parameters,
        )


def _derive_release(context: BuildContext, sources: Sources, entries: EntryBuilder) -> None:
    proposal_source = sources.json_cell(
        role="evidence_artifact:model_contract_proposal",
        source=context.model_contract_proposal,
        pointer=("promotion", "stageb_binding", "expert_id"),
    )
    entries.add(
        "FINAL_HAIR_EXPERT_ID",
        context.hair_identity_count_expert,
        "copy_proposal_owned_neutral_public_hair_expert_id",
        [proposal_source],
    )
    bundle = context.model_bundle_manifest
    for token, field, operation in (
        ("FINAL_MODEL_BUNDLE_MEMBER_N", "member_count", "count_sealed_bundle_members"),
        ("FINAL_MODEL_BUNDLE_SHA256", "bundle_sha256", "copy_sealed_bundle_sha256"),
        ("FINAL_MODEL_BUNDLE_SIZE", "bundle_size_bytes", "format_sealed_bundle_size_bytes"),
    ):
        raw = bundle.payload[field]
        value: str | int = raw
        if field == "bundle_size_bytes":
            size = _integer(raw, field)
            value = f"{size / (1024**2):.1f} MiB ({size} bytes)"
        entries.add(
            token,
            value,
            operation,
            [sources.json_cell(role="model_bundle_manifest", source=bundle, pointer=(field,))],
        )
    clean = context.clean_install_receipt
    entries.add(
        "FINAL_CLEAN_INSTALL_EXAMPLE_IDENTITY",
        str(clean.payload["example_output_identity_sha256"]),
        "copy_clean_install_example_output_identity",
        [sources.json_cell(role="clean_install_receipt", source=clean, pointer=("example_output_identity_sha256",))],
    )


def _derive_development(context: BuildContext, sources: Sources, entries: EntryBuilder) -> None:
    per_image = sources.table("figure_resource", "development_per_image")
    tolerance = sources.table("figure_resource", "development_tolerance")
    threshold = sources.table("figure_resource", "development_threshold")
    strata = sources.table("figure_resource", "development_strata")
    per_columns = (
        "source_unit", "source_unit_order", "family_key", "comparator", "gt_count",
        "predicted_count", "biological_presence_tp_5um", "biological_presence_tp_10um",
        "biological_presence_tp_20um", "prediction_input_sha256",
        "prediction_input_set_identity_sha256", "prediction_input_schema_version",
        "identity_hair_variant", "evidence_role",
    )
    _require_columns(per_image, per_columns, "development_per_image")
    require(len(per_image) == 88, "development_per_image is not two comparators x QC44")
    require(set(per_image["comparator"].astype(str)) == {STAGEB_COMPARATOR, LEGACY_COMPARATOR}, "development comparator set changed")
    require(per_image.groupby("comparator")["source_unit"].nunique().eq(44).all(), "development comparator source-unit scope changed")
    source_sets = per_image.groupby("comparator")["prediction_input_set_identity_sha256"].nunique()
    require(source_sets.eq(1).all(), "development comparator prediction-set identity is mixed")
    _require_columns(
        tolerance,
        (
            "comparator", "tolerance_um", "precision", "recall", "f1", "ci_low", "ci_high",
            "paired_delta_stageb_minus_legacy_f1", "paired_delta_ci_low", "paired_delta_ci_high",
            "ci_method", "bootstrap_repetitions", "primary_metric", "minimum_truth_coverage",
            "minimum_prediction_coverage", "minimum_direction_cosine", "endpoint_gate_used",
        ),
        "development_tolerance",
    )
    require(len(tolerance) == 6, "development tolerance table is not two comparators x three tolerances")
    require(set(pd.to_numeric(tolerance["tolerance_um"])) == {5, 10, 20}, "development tolerances changed")
    require(pd.to_numeric(tolerance["bootstrap_repetitions"]).eq(BOOTSTRAP_REPETITIONS).all(), "development bootstrap repetitions changed")
    require((tolerance["ci_method"].astype(str) == "image-level nonparametric bootstrap").all(), "development CI method changed")
    require((tolerance["primary_metric"].astype(str) == "one_to_one_tolerant_biological_hair_presence").all(), "development primary metric changed")
    require(pd.to_numeric(tolerance["minimum_truth_coverage"]).eq(0.25).all(), "truth coverage threshold changed")
    require(pd.to_numeric(tolerance["minimum_prediction_coverage"]).eq(0.25).all(), "prediction coverage threshold changed")
    require(pd.to_numeric(tolerance["minimum_direction_cosine"]).eq(0.0).all(), "direction contract changed")
    require(~_bool_series(tolerance["endpoint_gate_used"], "endpoint_gate_used").any(), "endpoint gate entered biological-presence metric")

    scoped: dict[str, pd.DataFrame] = {}
    for comparator in (STAGEB_COMPARATOR, LEGACY_COMPARATOR):
        selected = per_image[per_image["comparator"].astype(str) == comparator].sort_values("source_unit_order", kind="stable")
        require(selected["source_unit"].astype(str).is_unique, f"{comparator}: duplicate source units")
        scoped[comparator] = selected.reset_index(drop=True)
    require(list(scoped[STAGEB_COMPARATOR]["source_unit"].astype(str)) == list(scoped[LEGACY_COMPARATOR]["source_unit"].astype(str)), "paired development source-unit order differs")
    require(np.array_equal(pd.to_numeric(scoped[STAGEB_COMPARATOR]["gt_count"]), pd.to_numeric(scoped[LEGACY_COMPARATOR]["gt_count"])), "development truth counts differ by comparator")

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_indices = generator.integers(0, 44, size=(BOOTSTRAP_REPETITIONS, 44))
    results: dict[str, dict[str, Any]] = {}
    for comparator, frame in scoped.items():
        gt = pd.to_numeric(frame["gt_count"]).to_numpy(dtype=np.int64)
        pred = pd.to_numeric(frame["predicted_count"]).to_numpy(dtype=np.int64)
        require(np.all(gt >= 0) and np.all(pred >= 0), f"{comparator}: negative count")
        boot_gt = gt[sampled_indices]
        boot_pred = pred[sampled_indices]
        count_error = pred - gt
        boot_mae = np.mean(np.abs(boot_pred - boot_gt), axis=1)
        boot_bias = np.mean(boot_pred - boot_gt, axis=1)
        boot_ccc = _ccc_rows(boot_gt, boot_pred)
        result: dict[str, Any] = {
            "gt": gt,
            "pred": pred,
            "mae": float(np.mean(np.abs(count_error))),
            "bias": float(np.mean(count_error)),
            "ccc": float(_ccc_rows(gt[None, :], pred[None, :])[0]),
            "mae_boot": boot_mae,
            "bias_boot": boot_bias,
            "ccc_boot": boot_ccc,
        }
        for tolerance_um in (5, 10, 20):
            tp = pd.to_numeric(frame[f"biological_presence_tp_{tolerance_um}um"]).to_numpy(dtype=np.int64)
            require(np.all((tp >= 0) & (tp <= np.minimum(gt, pred))), f"{comparator}@{tolerance_um}: impossible TP")
            precision, recall, f1 = _prf(int(tp.sum()), int(pred.sum()), int(gt.sum()))
            boot_tp = tp[sampled_indices].sum(axis=1)
            boot_pred_sum = boot_pred.sum(axis=1)
            boot_gt_sum = boot_gt.sum(axis=1)
            boot_precision, boot_recall, boot_f1 = _prf(boot_tp, boot_pred_sum, boot_gt_sum)
            result[f"prf_{tolerance_um}"] = (precision, recall, f1)
            result[f"prf_boot_{tolerance_um}"] = (boot_precision, boot_recall, boot_f1)
            row = tolerance[
                (tolerance["comparator"].astype(str) == comparator)
                & pd.to_numeric(tolerance["tolerance_um"]).eq(tolerance_um)
            ]
            require(len(row) == 1, f"{comparator}@{tolerance_um}: normalized row missing")
            record = row.iloc[0]
            for key, observed in (("precision", precision), ("recall", recall), ("f1", f1)):
                require(math.isclose(_finite(record[key], key), observed, abs_tol=1e-12, rel_tol=0), f"{comparator}@{tolerance_um}: {key} differs from sufficient statistics")
            low, high = np.quantile(boot_f1, (0.025, 0.975))
            require(math.isclose(_finite(record["ci_low"], "ci_low"), float(low), abs_tol=1e-12, rel_tol=0) and math.isclose(_finite(record["ci_high"], "ci_high"), float(high), abs_tol=1e-12, rel_tol=0), f"{comparator}@{tolerance_um}: F1 CI differs from sufficient statistics")
        results[comparator] = result

    per_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="development_per_image",
        selected=per_image,
        columns=per_columns,
        filters={"comparators": [STAGEB_COMPARATOR, LEGACY_COMPARATOR], "source_units": 44},
        sort_by=("source_unit_order", "comparator"),
    )
    tolerance_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="development_tolerance",
        selected=tolerance,
        columns=tuple(tolerance.columns),
        filters={"tolerances_um": [5, 10, 20]},
        sort_by=("comparator", "tolerance_um"),
    )
    common_sources = [per_source, tolerance_source]
    stageb = results[STAGEB_COMPARATOR]
    legacy = results[LEGACY_COMPARATOR]
    for tolerance_um in (5, 10, 20):
        precision, recall, f1 = stageb[f"prf_{tolerance_um}"]
        if tolerance_um == 20:
            boot_precision, boot_recall, boot_f1 = stageb[f"prf_boot_{tolerance_um}"]
            for token, value, boot in (
                ("FINAL_HAIR_PRECISION_20UM", precision, boot_precision),
                ("FINAL_HAIR_RECALL_20UM", recall, boot_recall),
                ("FINAL_HAIR_F1_20UM", f1, boot_f1),
            ):
                entries.add(token, _fmt(value), "pooled_one_to_one_biological_presence", common_sources, parameters={"tolerance_um": 20})
                low, high = np.quantile(boot, (0.025, 0.975))
                entries.add(f"{token}_CI", _fmt_ci(float(low), float(high)), "paired_image_bootstrap_percentile_interval", common_sources, parameters={"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED})
        else:
            entries.add(f"FINAL_HAIR_F1_{tolerance_um}UM", _fmt(f1), "pooled_one_to_one_biological_presence", common_sources, parameters={"tolerance_um": tolerance_um})
    entries.add("FINAL_QCDEV_IMAGE_N", 44, "count_unique_qcdevelopment_source_units", [per_source])
    entries.add("FINAL_QCDEV_ANNOTATED_HAIR_N", int(stageb["gt"].sum()), "sum_ground_truth_hair_identities", [per_source])
    for token, key in (
        ("FINAL_HAIR_COUNT_MAE", "mae"),
        ("FINAL_HAIR_COUNT_BIAS", "bias"),
        ("FINAL_HAIR_COUNT_CCC", "ccc"),
    ):
        entries.add(token, _fmt(stageb[key]), f"image_level_count_{key}", [per_source])
    count_ci = []
    for label, key in (("MAE", "mae_boot"), ("bias", "bias_boot"), ("CCC", "ccc_boot")):
        low, high = np.quantile(stageb[key], (0.025, 0.975))
        count_ci.append(f"{label} {_fmt_ci(float(low), float(high))}")
    entries.add("FINAL_HAIR_COUNT_METRICS_CI", "; ".join(count_ci), "paired_image_bootstrap_count_metric_intervals", [per_source], parameters={"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED})

    for token, key in (
        ("LOCKED_LEGACY_HYBRID_IDENTITY_COUNT_MAE", "mae"),
        ("LOCKED_LEGACY_HYBRID_IDENTITY_COUNT_BIAS", "bias"),
        ("LOCKED_LEGACY_HYBRID_IDENTITY_COUNT_CCC", "ccc"),
    ):
        legacy_source = deepcopy(per_source)
        legacy_source["authority_class"] = "historical_development_comparator"
        # authority is sealed into the derivation source; rebuild its cell seal.
        legacy_source = _selected_source(
            sources,
            namespace="figure_resource",
            role="development_per_image",
            selected=scoped[LEGACY_COMPARATOR],
            columns=per_columns,
            filters={"comparator": LEGACY_COMPARATOR, "source_units": 44},
            authority="historical_development_comparator",
            sort_by=("source_unit_order",),
        )
        entries.add(token, _fmt(legacy[key]), f"legacy_same_matcher_image_count_{key}", [legacy_source])
    legacy_prf = legacy["prf_20"]
    legacy_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="development_per_image",
        selected=scoped[LEGACY_COMPARATOR],
        columns=per_columns,
        filters={"comparator": LEGACY_COMPARATOR, "tolerance_um": 20},
        authority="historical_development_comparator",
        sort_by=("source_unit_order",),
    )
    for token, value in (
        ("LOCKED_LEGACY_HYBRID_IDENTITY_PRECISION_20UM", legacy_prf[0]),
        ("LOCKED_LEGACY_HYBRID_IDENTITY_RECALL_20UM", legacy_prf[1]),
        ("LOCKED_LEGACY_HYBRID_IDENTITY_F1_20UM", legacy_prf[2]),
    ):
        entries.add(token, _fmt(value), "legacy_same_matcher_pooled_biological_presence", [legacy_source])

    f1_delta_boot = stageb["prf_boot_20"][2] - legacy["prf_boot_20"][2]
    f1_delta = stageb["prf_20"][2] - legacy["prf_20"][2]
    mae_delta_boot = stageb["mae_boot"] - legacy["mae_boot"]
    mae_delta = stageb["mae"] - legacy["mae"]
    entries.add("FINAL_STAGEB_MINUS_LEGACY_F1_DELTA", _fmt(f1_delta), "paired_stageb_minus_legacy_f1", common_sources)
    low, high = np.quantile(f1_delta_boot, (0.025, 0.975))
    entries.add("FINAL_STAGEB_MINUS_LEGACY_F1_DELTA_CI", _fmt_ci(float(low), float(high)), "paired_image_bootstrap_stageb_minus_legacy_f1", common_sources, parameters={"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED})
    entries.add("FINAL_STAGEB_MINUS_LEGACY_MAE_DELTA", _fmt(mae_delta), "paired_stageb_minus_legacy_count_mae", [per_source])
    low, high = np.quantile(mae_delta_boot, (0.025, 0.975))
    entries.add("FINAL_STAGEB_MINUS_LEGACY_MAE_DELTA_CI", _fmt_ci(float(low), float(high)), "paired_image_bootstrap_stageb_minus_legacy_mae", [per_source], parameters={"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED})

    _require_columns(
        threshold,
        (
            "threshold",
            "f1_20um",
            "attachment_proxy_f1_20um",
            "count_mae",
            "selected",
            "selection_metric",
            "straight_base_to_tip_presence_proxy_used",
            "distal_endpoint_or_length_used_as_selection_gate",
        ),
        "development_threshold",
    )
    selected_threshold = threshold[_bool_series(threshold["selected"], "selected threshold")]
    require(len(selected_threshold) == 1, "threshold table must have exactly one selected row")
    require(
        (
            threshold["selection_metric"].astype(str)
            == "tolerant_biological_presence_f1_20um"
        ).all(),
        "threshold selection metric changed",
    )
    require(
        _bool_series(
            threshold["straight_base_to_tip_presence_proxy_used"],
            "selection presence-proxy guard",
        ).all()
        and not _bool_series(
            threshold["distal_endpoint_or_length_used_as_selection_gate"],
            "selection endpoint/length guard",
        ).any(),
        "threshold biological-presence proxy/Gate semantics changed",
    )
    threshold_source = _selected_source(sources, namespace="figure_resource", role="development_threshold", selected=threshold, columns=tuple(threshold.columns), filters={"selection": "exactly_one_locked_row"}, sort_by=("threshold",))
    entries.add("FINAL_STAGEB_SCORE_THRESHOLD", _fmt(_finite(selected_threshold.iloc[0]["threshold"], "threshold"), 4), "copy_unique_locked_selected_threshold", [threshold_source])

    _require_columns(strata, ("dimension", "stratum", "comparator", "f1_20um", "count_bias"), "development_strata")
    dense = strata[
        (strata["dimension"].astype(str) == "density")
        & (strata["stratum"].astype(str) == "very_dense_ge200")
        & (strata["comparator"].astype(str) == HISTORICAL_COMPARATOR)
    ]
    require(len(dense) == 1, "historical very-dense OOF443 row missing")
    dense_source = _selected_source(sources, namespace="figure_resource", role="development_strata", selected=dense, columns=tuple(strata.columns), filters={"dimension": "density", "stratum": "very_dense_ge200", "comparator": HISTORICAL_COMPARATOR}, authority="historical_development_comparator")
    entries.add("HISTORICAL_OOF_VERY_DENSE_F1", _fmt(_finite(dense.iloc[0]["f1_20um"], "dense F1")), "copy_historical_development_stratum_f1", [dense_source])
    entries.add("HISTORICAL_OOF_VERY_DENSE_COUNT_BIAS", _fmt(_finite(dense.iloc[0]["count_bias"], "dense count bias")), "copy_historical_development_stratum_count_bias", [dense_source])


def _metric_row(metrics: pd.DataFrame, key: str) -> pd.Series:
    row = metrics[metrics["metric_key"].astype(str) == key]
    require(len(row) == 1, f"assurance metric must occur exactly once: {key}")
    return row.iloc[0]


def _component_assurance_receipts(
    receipt: JsonSource, metrics: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    def sealed(field: str, identity_field: str, schema: str) -> dict[str, Any]:
        value = receipt.payload.get(field)
        require(isinstance(value, Mapping), f"{field}: embedded receipt missing")
        payload = deepcopy(dict(value))
        unsigned = deepcopy(payload)
        identity = unsigned.pop(identity_field, None)
        require(
            isinstance(identity, str)
            and len(identity) == 64
            and sha256_json(unsigned) == identity,
            f"{field}: embedded receipt identity drift",
        )
        require(
            payload.get("schema_version") == schema
            and payload.get("status") == "completed"
            and payload.get("evidence_role")
            == "annotated_qc_development_non_independent"
            and payload.get("independent_accuracy_claim_allowed") is False
            and payload.get("blind_images_used") == 0
            and payload.get("provider_equivalence_used_as_accuracy") is False
            and _integer(payload.get("source_unit_total"), f"{field} source-unit total")
            == 44,
            f"{field}: schema, evidence role, or QC-development44 denominator drift",
        )
        bootstrap = payload.get("bootstrap")
        require(
            isinstance(bootstrap, Mapping)
            and bootstrap.get("method")
            == "source-image nonparametric percentile bootstrap"
            and bootstrap.get("unit") == "source_image"
            and _integer(bootstrap.get("repetitions"), f"{field} bootstrap repetitions")
            == 10_000
            and _integer(bootstrap.get("seed"), f"{field} bootstrap seed")
            == 20_260_828,
            f"{field}: source-image bootstrap contract drift",
        )
        rows = payload.get("per_image")
        require(
            isinstance(rows, list)
            and len(rows) == 44
            and payload.get("per_image_set_identity_sha256") == sha256_json(rows),
            f"{field}: sealed per-image sufficient-statistic set drift",
        )
        return payload

    root = sealed(
        "root_continuity_assurance",
        "root_continuity_assurance_identity_sha256",
        "PHAxis-primary-root-continuity-assurance-1.0",
    )
    hair = sealed(
        "hair_attachment_assurance",
        "hair_attachment_assurance_identity_sha256",
        "PHAxis-hair-attachment-assurance-1.0",
    )
    require(
        root.get("source_unit_set_identity_sha256")
        == hair.get("source_unit_set_identity_sha256"),
        "root-continuity and hair-attachment source-image denominators differ",
    )
    components = receipt.payload.get("component_receipts")
    require(
        isinstance(components, Mapping)
        and components.get("root_continuity", {}).get("identity_sha256")
        == root["root_continuity_assurance_identity_sha256"]
        and components.get("hair_attachment", {}).get("identity_sha256")
        == hair["hair_attachment_assurance_identity_sha256"],
        "component audit metadata does not bind the embedded assurance receipts",
    )
    crosscheck = receipt.payload.get(
        "qcdev_stageb_biological_presence_20um_crosscheck_locks"
    )
    authorities = receipt.payload.get("source_authority_identity_sha256")
    hair_rows = hair["per_image"]
    require(
        isinstance(crosscheck, list)
        and len(crosscheck) == len(hair_rows) == 44
        and isinstance(authorities, Mapping)
        and authorities.get("qcdev_stageb_biological_presence_20um_crosscheck")
        == sha256_json(crosscheck),
        "hair production/evaluator biological-presence crosscheck identity drift",
    )
    for lock, row in zip(crosscheck, hair_rows, strict=True):
        formal = row["formal_matched_attachment_accuracy"][
            "formal_biological_presence"
        ]
        require(
            isinstance(lock, Mapping)
            and lock.get("task_id") == row.get("source_unit")
            and int(lock.get("n_pred", -1)) == int(formal["n_pred"])
            and int(lock.get("n_gt", -1)) == int(formal["n_gt"])
            and int(lock.get("biological_presence_tp_20um", -1))
            == int(formal["tp"])
            and lock.get("hair_attachment_row_identity_sha256")
            == row.get("row_identity_sha256"),
            "hair production/evaluator per-source biological-presence drift",
        )
    require(
        "publication_metric_role" in metrics.columns,
        "assurance metric publication-role column missing",
    )
    formal_keys = {*ROOT_CONTINUITY_METRIC_KEYS, *HAIR_ATTACHMENT_METRIC_KEYS}
    formal = metrics[metrics["metric_key"].astype(str).isin(formal_keys)]
    require(
        len(formal) == len(formal_keys)
        and set(formal["metric_key"].astype(str)) == formal_keys
        and set(formal["publication_metric_role"].astype(str))
        == {"formal_measurement_assurance"}
        and str(
            _metric_row(
                metrics, "root_continuity_reference_axis_coverage_mean"
            )["publication_metric_role"]
        )
        == "diagnostic_only_union_coverage",
        "formal component metrics or union-diagnostic role drift",
    )
    return root, hair


def _derive_root_trait_assurance(
    *,
    context: BuildContext,
    sources: Sources,
    entries: EntryBuilder,
    pairs: pd.DataFrame,
    receipt: JsonSource,
    trait_contract_source: JsonSource,
) -> dict[str, Any]:
    """Validate all R01--R19 evidence and expose optional stable manuscript tokens."""

    root_pairs = pairs[pairs["pair_type"].astype(str) == "root_trait"].copy()
    _require_columns(
        root_pairs,
        (
            "source_unit",
            "pair_id",
            "trait_id",
            "trait_key",
            "trait_family",
            "observed",
            "predicted",
            "unit",
            "reference_observable",
            "prediction_observable",
            "agreement_eligible",
            "ineligibility_reason",
            "reference_definition",
            "prediction_definition",
            "source_image_sha256",
        ),
        "root-trait assurance pairs",
    )
    source_units = sorted(set(root_pairs["source_unit"].astype(str)))
    require(len(source_units) == 44, "root-trait assurance is not exact QC-development44")
    receipt_payload = receipt.payload
    assurance_payload = receipt_payload.get("root_trait_assurance")
    require(isinstance(assurance_payload, Mapping), "sealed per-trait root assurance is missing")
    table_hashes = receipt_payload.get("source_table_sha256")
    require(
        isinstance(table_hashes, Mapping)
        and table_hashes.get("pairs")
        == context.resources["assurance_pairs"].file_sha256
        and isinstance(table_hashes.get("root_traits"), str)
        and len(str(table_hashes["root_traits"])) == 64
        and all(
            character in "0123456789abcdef"
            for character in str(table_hashes["root_traits"])
        ),
        "root-trait summary/pair table hash closure is missing or drifted",
    )
    authority_identities = receipt_payload.get("source_authority_identity_sha256")
    require(isinstance(authority_identities, Mapping), "root-trait assurance authority identities missing")
    reference_authority = authority_identities.get("canonical_ground_truth")
    prediction_authority = authority_identities.get(
        "qcdev_fusion_prediction_ordered_file_set"
    )
    trait_resource = context.resources.get("trait_contract")
    require(trait_resource is not None, "trait-contract figure resource missing")
    authority_hashes = receipt_payload.get("source_authority_sha256")
    require(
        isinstance(authority_hashes, Mapping)
        and authority_hashes.get("trait_contract") == trait_resource.file_sha256,
        "root-trait assurance does not bind the named trait-contract file",
    )
    canonical = validate_root_trait_assurance(
        assurance_payload,
        pairs=root_pairs.to_dict("records"),
        trait_contract=trait_contract_source.payload,
        source_units=source_units,
        trait_contract_file_sha256=trait_resource.file_sha256,
        reference_authority_sha256=str(reference_authority),
        prediction_authority_identity_sha256=str(prediction_authority),
    )
    require(canonical["trait_count"] == 19, "root-trait assurance does not cover R01--R19")
    require(canonical["family_count"] == len(ROOT_TRAIT_FAMILY_ORDER), "root-trait family coverage drift")
    require(
        receipt_payload.get("measurement_contract", {}).get(
            "root_trait_provider_equivalence_used_as_accuracy"
        )
        is False,
        "provider equivalence was allowed to masquerade as root-trait accuracy",
    )

    requested = set(entries.contract["tokens"]).intersection(ROOT_TRAIT_ASSURANCE_TOKENS)
    if not requested:
        return canonical
    pair_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="assurance_pairs",
        selected=root_pairs,
        columns=tuple(root_pairs.columns),
        filters={
            "pair_type": "root_trait",
            "trait_ids": "R01--R19",
            "source_units": 44,
            "provider_equivalence_used_as_accuracy": False,
        },
        sort_by=("trait_id", "source_unit"),
    )
    summary_source = sources.json_cell(
        role="figure_provenance:measurement_assurance",
        source=receipt,
        pointer=("root_trait_assurance",),
    )
    contract_source = sources.json_cell(
        role="figure_resource:trait_contract",
        source=trait_contract_source,
        pointer=("primary_root_traits",),
    )
    bound_sources = [pair_source, summary_source, contract_source]
    trait_rows = canonical["trait_rows"]
    eligible = [int(row["eligible_source_units"]) for row in trait_rows]
    observability = [float(row["observability_fraction"]) for row in trait_rows]
    ccc_values = [float(row["ccc"]) for row in trait_rows if row["ccc"] is not None]
    require(ccc_values, "no root-trait CCC is estimable")
    ccc_median = float(np.median(ccc_values))
    ccc_low, ccc_high = min(ccc_values), max(ccc_values)
    family_parts = []
    for row in canonical["family_rows"]:
        ids = row["trait_ids"]
        identifier = ids[0] if len(ids) == 1 else f"{ids[0]}–{ids[-1]}"
        if row["median_ccc"] is None:
            statistic = "CCC not estimable; native-unit MAE reported"
        else:
            statistic = f"median CCC {_fmt(float(row['median_ccc']), 3)}"
        family_parts.append(f"{row['trait_family']} ({identifier}): {statistic}")
    token_values: dict[str, tuple[Any, str]] = {
        "FINAL_ROOT_TRAIT_VALIDATED_N": (19, "count_sealed_root_trait_rows"),
        "FINAL_ROOT_TRAIT_VALIDATION_IMAGE_N": (44, "copy_root_trait_source_unit_denominator"),
        "FINAL_ROOT_TRAIT_ELIGIBLE_N_RANGE": (
            f"{min(eligible)}–{max(eligible)}",
            "range_traitwise_eligible_source_units",
        ),
        "FINAL_ROOT_TRAIT_OBSERVABILITY_RANGE_PERCENT": (
            f"{_fmt_percent(min(observability))}–{_fmt_percent(max(observability))}",
            "range_traitwise_observability_percent",
        ),
        "FINAL_ROOT_TRAIT_CCC_ESTIMABLE_N": (
            len(ccc_values),
            "count_traits_with_estimable_ccc",
        ),
        "FINAL_ROOT_TRAIT_CCC_MEDIAN": (
            _fmt(ccc_median, 3),
            "median_traitwise_ccc",
        ),
        "FINAL_ROOT_TRAIT_CCC_RANGE": (
            f"{_fmt(ccc_low, 3)}–{_fmt(ccc_high, 3)}",
            "range_traitwise_ccc",
        ),
        "FINAL_ROOT_TRAIT_AGREEMENT_SUMMARY": (
            f"All 19 primary-root traits were compared with the canonical mask-plus-distal-point reference across 44 QC-development images; trait-wise eligible n ranged {min(eligible)}–{max(eligible)}, CCC was estimable for {len(ccc_values)}/19 traits with median {_fmt(ccc_median, 3)} (range {_fmt(ccc_low, 3)}–{_fmt(ccc_high, 3)}), and native-unit MAE and bias were retained for every trait.",
            "compose_root_trait_agreement_summary",
        ),
        "FINAL_ROOT_TRAIT_FAMILY_SUMMARY": (
            "; ".join(family_parts),
            "compose_six_root_trait_family_summary",
        ),
    }
    for token in ROOT_TRAIT_ASSURANCE_TOKENS:
        if token in requested:
            value, operation = token_values[token]
            entries.add(
                token,
                value,
                operation,
                bound_sources,
                parameters={
                    "truth_reference": canonical["truth_reference"],
                    "accuracy_evidence_role": canonical["evidence_role"],
                    "provider_equivalence_used_as_accuracy": False,
                },
            )
    return canonical


def _derive_assurance(context: BuildContext, sources: Sources, entries: EntryBuilder) -> None:
    metrics = sources.table("figure_resource", "assurance_metrics")
    pairs = sources.table("figure_resource", "assurance_pairs")
    support = sources.table("figure_resource", "assurance_support")
    topology = sources.table("figure_source_input", "assurance_topology")
    receipt = context.provenance_receipts.get("measurement_assurance")
    require(receipt is not None, "measurement assurance provenance receipt missing")
    trait_contract_source = sources.json_resource("trait_contract")
    contract = trait_contract_source.payload
    require(contract.get("schema_version") == "PHAxis-trait-contract-1.0.0", "trait contract schema changed")
    _require_columns(
        metrics,
        (
            "domain", "metric_key", "value", "ci_low", "ci_high", "unit",
            "n", "instances", "evidence_role", "ci_method",
            "bootstrap_repetitions", "bootstrap_seed",
            "publication_metric_role",
        ),
        "assurance_metrics",
    )
    required = {
        "root_dice", "root_boundary_f1", "root_hd95_um", "distal_median_error_um",
        "distal_pck", "scale_detection_coverage",
        "scale_geometry_endpoint_error_um", "scale_relative_error_percent",
        "conditional_length_mae_um", "conditional_length_bias_um", "conditional_length_ccc",
        "matched_endpoint_error_um", "matched_trajectory_continuity",
        "endpoint_complete_support_fraction", "axis_containment_median", "axis_containment_min",
        "unsupported_attachment_n", "root_trait_agreement", "provider_exact_fraction",
        "root_continuity_reference_axis_coverage_mean",
        *ROOT_CONTINUITY_METRIC_KEYS,
        *HAIR_ATTACHMENT_METRIC_KEYS,
    }
    require(required.issubset(set(metrics["metric_key"].astype(str))), "assurance metric set incomplete")
    root_continuity_receipt, hair_attachment_receipt = (
        _component_assurance_receipts(receipt, metrics)
    )
    metric_sources: dict[str, dict[str, Any]] = {}
    for key in required:
        row = metrics[metrics["metric_key"].astype(str) == key]
        metric_sources[key] = _selected_source(sources, namespace="figure_resource", role="assurance_metrics", selected=row, columns=tuple(metrics.columns), filters={"metric_key": key})

    root_summary = root_continuity_receipt["summary"]
    root_ci = root_summary["bootstrap_95ci"]
    root_component_map = {
        "root_continuity_reference_axis_coverage_mean": "reference_axis_coverage_mean",
        "root_continuity_maximum_single_component_coverage_mean": "maximum_single_component_coverage_mean",
        "root_continuity_maximum_single_component_coverage_median": "maximum_single_component_coverage_median",
        "root_continuity_best_component_gap_median_um": "longest_unsupported_gap_um_on_best_component_median",
        "root_continuity_break_free_rate": "break_free_image_rate",
        "root_continuity_visible_axis_extent_mae_um": "visible_axis_extent_error_um_mae",
    }
    hair_formal = hair_attachment_receipt["summary"][
        "formal_matched_attachment_accuracy"
    ]
    hair_ci = hair_formal["bootstrap_95ci"]
    hair_identity = hair_formal["attachment_qualified_identity"]
    hair_errors = hair_formal[
        "attachment_position_error_on_all_formal_identity_matches"
    ]
    hair_component_map = {
        "hair_attachment_qualified_precision_20um": (
            "formal_attachment_precision",
            hair_identity["precision"],
            hair_identity["n_pred"],
        ),
        "hair_attachment_qualified_recall_20um": (
            "formal_attachment_recall",
            hair_identity["recall"],
            hair_identity["n_gt"],
        ),
        "hair_attachment_qualified_f1_20um": (
            "formal_attachment_f1",
            hair_identity["f1"],
            hair_identity["n_pred"] + hair_identity["n_gt"],
        ),
        "hair_attachment_error_median_um": (
            "formal_attachment_error_median_um",
            hair_errors["median_um"],
            hair_errors["n"],
        ),
        "hair_attachment_error_p95_um": (
            "formal_attachment_error_p95_um",
            hair_errors["p95_um"],
            hair_errors["n"],
        ),
    }
    for key, bootstrap_key in root_component_map.items():
        row = _metric_row(metrics, key)
        interval = root_ci[bootstrap_key]
        require(
            _integer(row["n"], f"{key} source-image denominator") == 44
            and _integer(row["instances"], f"{key} instance denominator") == 44
            and str(row["evidence_role"])
            == "annotated_qc_development_non_independent"
            and math.isclose(
                _finite(row["value"], key),
                _finite(interval["point_estimate"], f"{key} receipt point"),
                abs_tol=1e-12,
                rel_tol=0.0,
            )
            and math.isclose(
                _finite(row["ci_low"], f"{key} CI low"),
                _finite(interval["ci_low_2_5"], f"{key} receipt CI low"),
                abs_tol=1e-12,
                rel_tol=0.0,
            )
            and math.isclose(
                _finite(row["ci_high"], f"{key} CI high"),
                _finite(interval["ci_high_97_5"], f"{key} receipt CI high"),
                abs_tol=1e-12,
                rel_tol=0.0,
            ),
            f"{key}: figure metric differs from embedded source-image assurance",
        )
    for key, (bootstrap_key, point, instances) in hair_component_map.items():
        row = _metric_row(metrics, key)
        interval = hair_ci[bootstrap_key]
        require(
            point is not None
            and _integer(row["n"], f"{key} source-image denominator") == 44
            and _integer(row["instances"], f"{key} instance denominator")
            == int(instances)
            and str(row["evidence_role"])
            == "annotated_qc_development_non_independent"
            and math.isclose(
                _finite(row["value"], key), float(point), abs_tol=1e-12, rel_tol=0.0
            )
            and math.isclose(
                _finite(row["ci_low"], f"{key} CI low"),
                _finite(interval["ci_low_2_5"], f"{key} receipt CI low"),
                abs_tol=1e-12,
                rel_tol=0.0,
            )
            and math.isclose(
                _finite(row["ci_high"], f"{key} CI high"),
                _finite(interval["ci_high_97_5"], f"{key} receipt CI high"),
                abs_tol=1e-12,
                rel_tol=0.0,
            ),
            f"{key}: figure metric differs from embedded formal attachment assurance",
        )

    _require_columns(
        pairs,
        (
            "pair_type",
            "source_unit",
            "pair_id",
            "observed",
            "predicted",
            "unit",
            "endpoint_error_um",
            "trajectory_continuity",
            "relative_error_percent",
            "scale_line_endpoint_error_um",
            "source_image_sha256",
        ),
        "assurance_pairs",
    )
    length_pairs = pairs[pairs["pair_type"].astype(str) == "conditional_length"]
    require(len(length_pairs) >= 2, "conditional-length assurance pairs missing")
    observed_length = pd.to_numeric(length_pairs["observed"], errors="coerce").to_numpy(float)
    predicted_length = pd.to_numeric(length_pairs["predicted"], errors="coerce").to_numpy(float)
    endpoint_error = pd.to_numeric(length_pairs["endpoint_error_um"], errors="coerce").to_numpy(float)
    trajectory = pd.to_numeric(length_pairs["trajectory_continuity"], errors="coerce").to_numpy(float)
    require(
        np.isfinite(observed_length).all()
        and np.isfinite(predicted_length).all()
        and np.isfinite(endpoint_error).all()
        and np.isfinite(trajectory).all(),
        "conditional-length assurance pairs contain non-finite cells",
    )
    pair_values = {
        "conditional_length_mae_um": float(np.mean(np.abs(predicted_length - observed_length))),
        "conditional_length_bias_um": float(np.mean(predicted_length - observed_length)),
        "conditional_length_ccc": float(
            _ccc_sample_rows(observed_length[None, :], predicted_length[None, :])[0]
        ),
        "matched_endpoint_error_um": float(np.median(endpoint_error)),
        "matched_trajectory_continuity": float(np.mean(trajectory)),
    }
    for key, observed in pair_values.items():
        require(
            math.isclose(
                _finite(_metric_row(metrics, key)["value"], key),
                observed,
                abs_tol=1e-12,
                rel_tol=0,
            ),
            f"{key}: assurance metrics/pairs mismatch",
        )
    length_pairs_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="assurance_pairs",
        selected=length_pairs,
        columns=tuple(pairs.columns),
        filters={"pair_type": "conditional_length"},
        sort_by=("source_unit",),
    )
    root_trait_assurance = _derive_root_trait_assurance(
        context=context,
        sources=sources,
        entries=entries,
        pairs=pairs,
        receipt=receipt,
        trait_contract_source=trait_contract_source,
    )
    root_trait_metric = _metric_row(metrics, "root_trait_agreement")
    trait_ccc = [
        float(row["ccc"])
        for row in root_trait_assurance["trait_rows"]
        if row["ccc"] is not None
    ]
    require(trait_ccc, "root-trait aggregate has no estimable trait-wise CCC")
    require(
        str(root_trait_metric["evidence_role"])
        == "annotated_qc_development_non_independent"
        and _integer(root_trait_metric["n"], "root-trait aggregate denominator")
        == 44
        and _integer(
            root_trait_metric["instances"], "root-trait aggregate pair count"
        )
        == 44 * 19
        and math.isclose(
            _finite(root_trait_metric["value"], "root-trait aggregate CCC"),
            float(np.median(trait_ccc)),
            abs_tol=1e-12,
            rel_tol=0,
        ),
        "root-trait aggregate is not the annotated 19-trait/44-image summary",
    )

    scale_detection = _metric_row(metrics, "scale_detection_coverage")
    scale_coverage = _finite(
        scale_detection["value"], "scale detection coverage"
    )
    scale_validation_n = _integer(
        scale_detection["n"], "scale detection validation denominator"
    )
    scale_detected_n = _integer(
        scale_detection["instances"], "scale detection detected count"
    )
    require(
        scale_validation_n > 0
        and 0 <= scale_detected_n <= scale_validation_n
        and 0.0 <= scale_coverage <= 1.0,
        "scale detection coverage/count contract is invalid",
    )
    require(
        math.isclose(
            scale_coverage,
            scale_detected_n / scale_validation_n,
            abs_tol=1e-12,
            rel_tol=0,
        ),
        "scale detection coverage does not equal instances/n",
    )
    scale_relative_error = _metric_row(metrics, "scale_relative_error_percent")
    scale_localization = _metric_row(
        metrics, "scale_geometry_endpoint_error_um"
    )
    scale_applicability = receipt.payload.get("scale_applicability")
    scale_counts = receipt.payload.get("counts")
    scale_contract = receipt.payload.get("measurement_contract")
    require(
        isinstance(scale_applicability, Mapping)
        and isinstance(scale_counts, Mapping)
        and isinstance(scale_contract, Mapping),
        "scale applicability/count/measurement contract receipt is missing",
    )
    visible_scale_n = _integer(
        scale_applicability.get("visible_annotated_scale_bar_cases"),
        "visible annotated scale-bar count",
    )
    trusted_metadata_n = _integer(
        scale_applicability.get("trusted_metadata_without_visible_bar_cases"),
        "trusted-metadata scale count",
    )
    absence_test_n = _integer(
        scale_applicability.get("absent_or_untrusted_scale_truth_cases"),
        "absent/untrusted scale-test count",
    )
    require(
        _integer(
            scale_applicability.get("qcdevelopment_images"),
            "scale applicability QC-development count",
        )
        == 44
        and visible_scale_n == 37
        and trusted_metadata_n == 7
        and absence_test_n == 0
        and visible_scale_n + trusted_metadata_n + absence_test_n == 44
        and scale_applicability.get("absence_specificity_status")
        == SCALE_ABSENCE_SPECIFICITY_STATUS
        and scale_applicability.get("fail_closed_evidence_basis")
        == SCALE_FAIL_CLOSED_EVIDENCE_BASIS
        and scale_applicability.get("empirical_absence_specificity_claimed")
        is False,
        "scale applicability must close as 37 visible + 7 trusted metadata + 0 absence-test cases",
    )
    require(
        scale_contract.get("scale_coverage_denominator")
        == "visible_annotated_scale_bar_cases"
        and scale_contract.get("scale_localization_denominator")
        == "detected_visible_scale_bars"
        and scale_contract.get("scale_calibration_denominator")
        == "detected_visible_scale_bars"
        and scale_contract.get("scale_absence_specificity_status")
        == SCALE_ABSENCE_SPECIFICITY_STATUS
        and scale_contract.get("scale_fail_closed_evidence_basis")
        == SCALE_FAIL_CLOSED_EVIDENCE_BASIS,
        "scale measurement/applicability contract changed",
    )
    require(
        scale_validation_n == visible_scale_n
        and _integer(scale_counts.get("visible_scale_bars"), "receipt visible scale bars")
        == visible_scale_n
        and _integer(
            scale_counts.get("trusted_metadata_without_visible_bar_cases"),
            "receipt trusted-metadata scale cases",
        )
        == trusted_metadata_n
        and _integer(
            scale_counts.get("absent_or_untrusted_scale_truth_cases"),
            "receipt absent/untrusted scale cases",
        )
        == absence_test_n
        and _integer(scale_counts.get("detected_scale_bars"), "receipt detected scales")
        == scale_detected_n,
        "scale metric and receipt denominators disagree",
    )
    scale_pairs = pairs[pairs["pair_type"].astype(str) == "scale"].copy()
    require(
        len(scale_pairs)
        == scale_pairs["source_unit"].nunique()
        == scale_pairs["pair_id"].nunique()
        == scale_pairs["source_image_sha256"].nunique()
        == scale_detected_n,
        "scale pairs do not equal the detected visible-bar denominator",
    )
    scale_metric_rows = metrics[
        metrics["metric_key"].astype(str).isin(
            {
                "scale_detection_coverage",
                "scale_geometry_endpoint_error_um",
                "scale_relative_error_percent",
            }
        )
    ]
    require(
        len(scale_metric_rows) == 3
        and all(
            str(row["ci_method"])
            == "image/source-unit nonparametric bootstrap"
            and _integer(
                row["bootstrap_repetitions"],
                f"{row['metric_key']} bootstrap repetitions",
            )
            == BOOTSTRAP_REPETITIONS
            and _integer(
                row["bootstrap_seed"], f"{row['metric_key']} bootstrap seed"
            )
            == BOOTSTRAP_SEED
            and math.isfinite(_finite(row["ci_low"], f"{row['metric_key']} CI low"))
            and math.isfinite(_finite(row["ci_high"], f"{row['metric_key']} CI high"))
            and _finite(row["ci_low"], f"{row['metric_key']} CI low")
            <= _finite(row["ci_high"], f"{row['metric_key']} CI high")
            for row in scale_metric_rows.to_dict("records")
        ),
        "scale coverage/localization/calibration metrics lack source-image bootstrap intervals",
    )
    observed_scale = pd.to_numeric(scale_pairs["observed"], errors="coerce").to_numpy(float)
    predicted_scale = pd.to_numeric(scale_pairs["predicted"], errors="coerce").to_numpy(float)
    stored_relative_error = pd.to_numeric(
        scale_pairs["relative_error_percent"], errors="coerce"
    ).to_numpy(float)
    localization_error = pd.to_numeric(
        scale_pairs["scale_line_endpoint_error_um"], errors="coerce"
    ).to_numpy(float)
    recomputed_relative_error = (
        np.abs(predicted_scale - observed_scale) / observed_scale * 100.0
    )
    require(
        scale_detected_n >= 2
        and np.isfinite(observed_scale).all()
        and np.isfinite(predicted_scale).all()
        and np.isfinite(stored_relative_error).all()
        and np.isfinite(localization_error).all()
        and bool((observed_scale > 0).all())
        and np.allclose(
            stored_relative_error,
            recomputed_relative_error,
            rtol=0.0,
            atol=1e-12,
        ),
        "scale pair sufficient statistics are invalid",
    )
    require(
        _integer(scale_relative_error["n"], "scale relative-error denominator")
        == _integer(scale_relative_error["instances"], "scale relative-error instances")
        == _integer(scale_counts.get("scale_calibration_pairs"), "scale calibration pairs")
        == scale_detected_n
        and _integer(scale_localization["n"], "scale localization denominator")
        == _integer(scale_localization["instances"], "scale localization instances")
        == _integer(scale_counts.get("scale_localization_pairs"), "scale localization pairs")
        == scale_detected_n,
        "scale localization/calibration denominator does not equal detected count",
    )
    require(
        math.isclose(
            _finite(scale_relative_error["value"], "scale relative error"),
            float(np.median(recomputed_relative_error)),
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        and math.isclose(
            _finite(scale_localization["value"], "scale localization error"),
            float(np.median(localization_error)),
            abs_tol=1e-12,
            rel_tol=0.0,
        ),
        "scale localization/calibration metrics do not match pair sufficient statistics",
    )
    scale_pairs_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="assurance_pairs",
        selected=scale_pairs,
        columns=tuple(pairs.columns),
        filters={"pair_type": "scale"},
        sort_by=("source_unit",),
    )
    scale_applicability_source = sources.json_cell(
        role="figure_provenance:measurement_assurance",
        source=receipt,
        pointer=("scale_applicability",),
    )

    direct = {
        "FINAL_ROOT_DICE": ("root_dice", 3),
        "FINAL_ROOT_BOUNDARY_F1": ("root_boundary_f1", 3),
        "FINAL_ROOT_HD95_UM": ("root_hd95_um", 2),
        "FINAL_DISTAL_MEDIAN_ERROR_UM": ("distal_median_error_um", 2),
        "FINAL_DISTAL_PCK": ("distal_pck", 3),
        "FINAL_SCALE_DETECTION_COVERAGE": ("scale_detection_coverage", 3),
        "FINAL_SCALE_LOCALIZATION_ERROR_UM": (
            "scale_geometry_endpoint_error_um",
            2,
        ),
        "FINAL_SCALE_RELATIVE_ERROR_PERCENT": ("scale_relative_error_percent", 2),
        "FINAL_MATCHED_LENGTH_MAE_UM": ("conditional_length_mae_um", 2),
        "FINAL_MATCHED_LENGTH_BIAS_UM": ("conditional_length_bias_um", 2),
        "FINAL_MATCHED_LENGTH_CCC": ("conditional_length_ccc", 3),
        "FINAL_MATCHED_ENDPOINT_ERROR_UM": ("matched_endpoint_error_um", 2),
        "FINAL_MATCHED_TRAJECTORY_CONTINUITY": ("matched_trajectory_continuity", 3),
        "FINAL_AXIS_CONTAINMENT_MEDIAN": ("axis_containment_median", 3),
        "FINAL_AXIS_CONTAINMENT_MIN": ("axis_containment_min", 3),
    }
    for token, (key, digits) in direct.items():
        if token not in entries.contract["tokens"]:
            continue
        bound_sources = [metric_sources[key]]
        if key in pair_values:
            bound_sources.append(length_pairs_source)
        if key in {
            "scale_detection_coverage",
            "scale_geometry_endpoint_error_um",
            "scale_relative_error_percent",
        }:
            bound_sources.append(scale_applicability_source)
        if key in {
            "scale_geometry_endpoint_error_um",
            "scale_relative_error_percent",
        }:
            bound_sources.append(scale_pairs_source)
        entries.add(token, _fmt(_finite(_metric_row(metrics, key)["value"], key), digits), "copy_recomputed_assurance_metric", bound_sources, parameters={"metric_key": key})
    for token, key in (
        ("FINAL_ROOT_VALIDATION_N", "root_dice"),
        ("FINAL_DISTAL_VALIDATION_N", "distal_median_error_um"),
        ("FINAL_SCALE_VALIDATION_N", "scale_detection_coverage"),
        ("FINAL_SCALE_LOCALIZATION_N", "scale_geometry_endpoint_error_um"),
        ("FINAL_MATCHED_LENGTH_N", "conditional_length_mae_um"),
    ):
        if token not in entries.contract["tokens"]:
            continue
        token_sources = [metric_sources[key]]
        if key.startswith("scale_"):
            token_sources.append(scale_applicability_source)
        if key == "scale_geometry_endpoint_error_um":
            token_sources.append(scale_pairs_source)
        entries.add(token, _integer(_metric_row(metrics, key)["n"], key), "copy_assurance_metric_denominator", token_sources, parameters={"metric_key": key})
    entries.add(
        "FINAL_SCALE_DETECTED_N",
        scale_detected_n,
        "copy_scale_detection_instances_after_coverage_count_crosscheck",
        [
            metric_sources["scale_detection_coverage"],
            scale_applicability_source,
            scale_pairs_source,
        ],
        parameters={"metric_key": "scale_detection_coverage", "count_field": "instances"},
    )
    for token, keys in (
        ("FINAL_ROOT_METRICS_CI", ("root_dice", "root_boundary_f1", "root_hd95_um")),
        ("FINAL_DISTAL_METRICS_CI", ("distal_median_error_um", "distal_pck")),
        ("FINAL_SCALE_ERROR_CI", ("scale_relative_error_percent",)),
        ("FINAL_SCALE_LOCALIZATION_CI", ("scale_geometry_endpoint_error_um",)),
        ("FINAL_MATCHED_LENGTH_METRICS_CI", ("conditional_length_mae_um", "conditional_length_bias_um", "conditional_length_ccc")),
    ):
        if token not in entries.contract["tokens"]:
            continue
        labels = []
        for key in keys:
            row = _metric_row(metrics, key)
            labels.append(f"{key} {_fmt_ci(_finite(row['ci_low'], key), _finite(row['ci_high'], key))}")
        interval_sources = [metric_sources[key] for key in keys]
        if any(key in pair_values for key in keys):
            interval_sources.append(length_pairs_source)
        if any(key.startswith("scale_") for key in keys):
            interval_sources.append(scale_applicability_source)
        if any(
            key
            in {
                "scale_geometry_endpoint_error_um",
                "scale_relative_error_percent",
            }
            for key in keys
        ):
            interval_sources.append(scale_pairs_source)
        entries.add(token, "; ".join(labels), "format_assurance_percentile_intervals", interval_sources, parameters={"metric_keys": list(keys)})

    root_component_receipt_source = sources.json_cell(
        role="figure_provenance:measurement_assurance",
        source=receipt,
        pointer=("root_continuity_assurance",),
    )
    hair_component_receipt_source = sources.json_cell(
        role="figure_provenance:measurement_assurance",
        source=receipt,
        pointer=("hair_attachment_assurance",),
    )
    component_tokens = {
        "FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEAN": (
            "root_continuity_maximum_single_component_coverage_mean",
            3,
            root_component_receipt_source,
        ),
        "FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEDIAN": (
            "root_continuity_maximum_single_component_coverage_median",
            3,
            root_component_receipt_source,
        ),
        "FINAL_ROOT_CONTINUITY_LONGEST_UNSUPPORTED_GAP_UM_ON_BEST_COMPONENT_MEDIAN": (
            "root_continuity_best_component_gap_median_um",
            2,
            root_component_receipt_source,
        ),
        "FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE": (
            "root_continuity_break_free_rate",
            3,
            root_component_receipt_source,
        ),
        "FINAL_ROOT_CONTINUITY_VISIBLE_AXIS_EXTENT_MAE_UM": (
            "root_continuity_visible_axis_extent_mae_um",
            2,
            root_component_receipt_source,
        ),
        "FINAL_HAIR_ATTACHMENT_QUALIFIED_PRECISION_AT_20UM": (
            "hair_attachment_qualified_precision_20um",
            3,
            hair_component_receipt_source,
        ),
        "FINAL_HAIR_ATTACHMENT_QUALIFIED_RECALL_AT_20UM": (
            "hair_attachment_qualified_recall_20um",
            3,
            hair_component_receipt_source,
        ),
        "FINAL_HAIR_ATTACHMENT_QUALIFIED_F1_AT_20UM": (
            "hair_attachment_qualified_f1_20um",
            3,
            hair_component_receipt_source,
        ),
        "FINAL_HAIR_ATTACHMENT_FORMAL_MATCHED_ERROR_MEDIAN_UM": (
            "hair_attachment_error_median_um",
            2,
            hair_component_receipt_source,
        ),
        "FINAL_HAIR_ATTACHMENT_FORMAL_MATCHED_ERROR_P95_UM": (
            "hair_attachment_error_p95_um",
            2,
            hair_component_receipt_source,
        ),
    }
    for token, (key, digits, receipt_source) in component_tokens.items():
        if token not in entries.contract["tokens"]:
            continue
        entries.add(
            token,
            _fmt(_finite(_metric_row(metrics, key)["value"], key), digits),
            "copy_recomputed_component_assurance_metric",
            [metric_sources[key], receipt_source],
            parameters={
                "metric_key": key,
                "evidence_role": "annotated_qc_development_non_independent",
                "bootstrap_unit": "source_image",
                "base_proxy_used_as_formal_accuracy": False,
                "union_coverage_used_as_formal_continuity": False,
            },
        )
    if "FINAL_ROOT_CONTINUITY_VALIDATION_N" in entries.contract["tokens"]:
        entries.add(
            "FINAL_ROOT_CONTINUITY_VALIDATION_N",
            int(root_continuity_receipt["source_unit_total"]),
            "copy_root_continuity_source_image_denominator",
            [
                metric_sources[ROOT_CONTINUITY_METRIC_KEYS[0]],
                root_component_receipt_source,
            ],
            parameters={"bootstrap_unit": "source_image"},
        )
    if "FINAL_HAIR_ATTACHMENT_VALIDATION_N" in entries.contract["tokens"]:
        entries.add(
            "FINAL_HAIR_ATTACHMENT_VALIDATION_N",
            int(hair_attachment_receipt["source_unit_total"]),
            "copy_hair_attachment_source_image_denominator",
            [
                metric_sources[HAIR_ATTACHMENT_METRIC_KEYS[0]],
                hair_component_receipt_source,
            ],
            parameters={"bootstrap_unit": "source_image"},
        )
    for token, value, metric_key, denominator in (
        (
            "FINAL_HAIR_ATTACHMENT_PREDICTED_N",
            int(hair_identity["n_pred"]),
            "hair_attachment_qualified_precision_20um",
            "all predicted hair identities",
        ),
        (
            "FINAL_HAIR_ATTACHMENT_ANNOTATED_N",
            int(hair_identity["n_gt"]),
            "hair_attachment_qualified_recall_20um",
            "all annotated hair identities",
        ),
        (
            "FINAL_HAIR_ATTACHMENT_QUALIFIED_TP_N",
            int(hair_identity["tp"]),
            "hair_attachment_qualified_f1_20um",
            "formal biological-presence identities with attachment error <=20 um",
        ),
    ):
        if token not in entries.contract["tokens"]:
            continue
        entries.add(
            token,
            value,
            "copy_formal_attachment_identity_denominator",
            [metric_sources[metric_key], hair_component_receipt_source],
            parameters={
                "metric_key": metric_key,
                "denominator": denominator,
                "base_only_rematching": False,
            },
        )
    if "FINAL_HAIR_ATTACHMENT_FORMAL_MATCH_N" in entries.contract["tokens"]:
        entries.add(
            "FINAL_HAIR_ATTACHMENT_FORMAL_MATCH_N",
            int(hair_errors["n"]),
            "copy_formal_biological_identity_match_denominator",
            [
                metric_sources["hair_attachment_error_median_um"],
                hair_component_receipt_source,
            ],
            parameters={
                "denominator": "all formal biological-presence matches; no base-only rematching"
            },
        )
    for token, keys, receipt_source in (
        (
            "FINAL_ROOT_CONTINUITY_METRICS_CI",
            ROOT_CONTINUITY_METRIC_KEYS,
            root_component_receipt_source,
        ),
        (
            "FINAL_HAIR_ATTACHMENT_METRICS_CI",
            HAIR_ATTACHMENT_METRIC_KEYS,
            hair_component_receipt_source,
        ),
    ):
        if token not in entries.contract["tokens"]:
            continue
        labels = [
            f"{key} {_fmt_ci(_finite(_metric_row(metrics, key)['ci_low'], key), _finite(_metric_row(metrics, key)['ci_high'], key))}"
            for key in keys
        ]
        entries.add(
            token,
            "; ".join(labels),
            "format_component_assurance_source_image_percentile_intervals",
            [*[metric_sources[key] for key in keys], receipt_source],
            parameters={
                "metric_keys": list(keys),
                "bootstrap_repetitions": 10_000,
                "bootstrap_seed": 20_260_828,
                "bootstrap_unit": "source_image",
            },
        )

    optional_scale_text = {
        "FINAL_SCALE_APPLICABILITY_STATEMENT": (
            f"{visible_scale_n} images contained an annotated visible scale bar and "
            f"{trusted_metadata_n} used trusted metadata; no absent or untrusted "
            "scale-truth case was available",
            "compose_scale_applicability_statement",
        ),
        "FINAL_SCALE_ABSENCE_SPECIFICITY_STATUS": (
            SCALE_ABSENCE_SPECIFICITY_STATUS,
            "copy_scale_absence_specificity_status",
        ),
        "FINAL_SCALE_FAIL_CLOSED_EVIDENCE_STATEMENT": (
            "Fail-closed behaviour is a software contract covered by unit tests; "
            "empirical absence specificity is not estimable in QC-development44.",
            "compose_scale_fail_closed_evidence_statement",
        ),
    }
    for token, (value, operation) in optional_scale_text.items():
        if token in entries.contract["tokens"]:
            entries.add(
                token,
                value,
                operation,
                [scale_applicability_source],
                parameters={
                    "visible_scale_n": visible_scale_n,
                    "trusted_metadata_n": trusted_metadata_n,
                    "absence_test_n": absence_test_n,
                    "absence_specificity_status": SCALE_ABSENCE_SPECIFICITY_STATUS,
                    "fail_closed_evidence_basis": SCALE_FAIL_CLOSED_EVIDENCE_BASIS,
                },
            )

    threshold_um = receipt.payload.get("measurement_contract", {}).get("distal_pck_threshold_um")
    require(_finite(threshold_um, "distal PCK threshold") == 25.0, "distal PCK threshold changed")
    entries.add("FINAL_DISTAL_PCK_THRESHOLD_UM", _fmt(float(threshold_um), 0), "copy_locked_distal_pck_threshold", [sources.json_cell(role="figure_provenance:measurement_assurance", source=receipt, pointer=("measurement_contract", "distal_pck_threshold_um"))])

    _require_columns(topology, ("source_unit", "axis_containment_fraction", "unsupported_attachment_n", "identity_hair_n"), "assurance_topology")
    require(len(topology) == topology["source_unit"].nunique() == 261, "assurance topology is not exact clean261")
    containment = pd.to_numeric(topology["axis_containment_fraction"], errors="coerce")
    unsupported = pd.to_numeric(topology["unsupported_attachment_n"], errors="coerce")
    require(np.isfinite(containment).all() and ((containment >= 0) & (containment <= 1)).all(), "invalid axis containment")
    require(np.isfinite(unsupported).all() and (unsupported >= 0).all(), "invalid unsupported attachment count")
    topology_source = _selected_source(sources, namespace="figure_source_input", role="assurance_topology", selected=topology, columns=("source_unit", "axis_containment_fraction", "unsupported_attachment_n", "identity_hair_n"), filters={"formal_clean_source_units": 261}, sort_by=("source_unit",))
    crosschecks = {
        "axis_containment_median": float(np.median(containment)),
        "axis_containment_min": float(np.min(containment)),
        "unsupported_attachment_n": float(unsupported.sum()),
    }
    for key, value in crosschecks.items():
        require(math.isclose(_finite(_metric_row(metrics, key)["value"], key), value, abs_tol=1e-12, rel_tol=0), f"{key}: metrics/topology mismatch")
    entries.add("FINAL_UNSUPPORTED_ATTACHMENT_N", int(unsupported.sum()), "sum_formal_unsupported_attachments", [topology_source, metric_sources["unsupported_attachment_n"]])

    traits = context.evidence_artifacts["traits"]
    trait_source = sources.json_cell(role="evidence_artifact:traits", source=traits, pointer=("hair_identities",))
    total_hairs = _integer(traits.payload.get("hair_identities"), "formal hair identities")
    complete_hairs = _integer(traits.payload.get("endpoint_complete_length_identities"), "endpoint complete identities")
    require(0 <= complete_hairs <= total_hairs, "trait identity counts impossible")
    entries.add("FINAL_TOTAL_HAIR_IDENTITIES", total_hairs, "copy_formal_trait_export_identity_count", [trait_source])
    entries.add("FINAL_ENDPOINT_COMPLETE_IDENTITY_N", complete_hairs, "copy_formal_endpoint_complete_identity_count", [sources.json_cell(role="evidence_artifact:traits", source=traits, pointer=("endpoint_complete_length_identities",))])
    fraction = complete_hairs / total_hairs
    require(math.isclose(fraction, _finite(_metric_row(metrics, "endpoint_complete_support_fraction")["value"], "support fraction"), abs_tol=1e-12, rel_tol=0), "trait summary/assurance support fraction mismatch")
    entries.add("FINAL_ENDPOINT_COMPLETE_IDENTITY_PERCENT", _fmt_percent(fraction), "endpoint_complete_identity_fraction_percent", [trait_source, metric_sources["endpoint_complete_support_fraction"]])
    _require_columns(
        support,
        ("condition_code", "support_fraction", "supported_hairs", "identity_hairs", "source_units"),
        "assurance_support",
    )
    require(
        len(support) == 4 and set(support["condition_code"].astype(str)) == set(GROUPS),
        "assurance support is not the four locked D15 cells",
    )
    support_fraction = pd.to_numeric(support["support_fraction"], errors="coerce")
    require(
        np.isfinite(support_fraction).all()
        and ((support_fraction >= 0) & (support_fraction <= 1)).all(),
        "assurance support fraction is invalid",
    )
    support_source = _selected_source(
        sources,
        namespace="figure_resource",
        role="assurance_support",
        selected=support,
        columns=tuple(support.columns),
        filters={"conditions": list(GROUPS), "semantics": "endpoint_complete_matched_subset"},
        sort_by=("condition_code",),
    )
    entries.add(
        "FINAL_D15_LENGTH_SUPPORT_MIN_PERCENT",
        _fmt_percent(float(support_fraction.min())),
        "minimum_condition_endpoint_complete_support_percent",
        [support_source],
    )
    entries.add(
        "FINAL_D15_LENGTH_SUPPORT_MAX_PERCENT",
        _fmt_percent(float(support_fraction.max())),
        "maximum_condition_endpoint_complete_support_percent",
        [support_source],
    )
    formal_n = _integer(traits.payload.get("formal_statistics_eligible"), "formal trait rows")
    review_n = _integer(traits.payload.get("review_only"), "review-only trait rows")
    require(formal_n + review_n == _integer(traits.payload.get("tasks"), "trait tasks") == 283, "formal/review trait partition is not exact283")
    entries.add("FINAL_FORMAL_IMAGE_N", formal_n, "copy_trait_export_formal_count", [sources.json_cell(role="evidence_artifact:traits", source=traits, pointer=("formal_statistics_eligible",))])
    entries.add("FINAL_REVIEW_ONLY_IMAGE_N", review_n, "copy_trait_export_review_only_count", [sources.json_cell(role="evidence_artifact:traits", source=traits, pointer=("review_only",))])

    exact = _metric_row(metrics, "provider_exact_fraction")
    require(
        str(exact["evidence_role"]) == "exact_portable_provider_equivalence",
        "provider exactness is not explicitly separated from annotated accuracy",
    )
    exact_n = _integer(exact["n"], "provider exact denominator")
    equivalent = int(round(exact_n * _finite(exact["value"], "provider exact fraction")))
    require(0 <= equivalent <= exact_n and exact_n == 283, "provider equivalence scope changed")
    entries.add("FINAL_ROOT_PROVIDER_EQUIVALENT_N", equivalent, "provider_exact_fraction_times_denominator", [metric_sources["provider_exact_fraction"]])
    entries.add("FINAL_ROOT_PROVIDER_NONEQUIVALENT_N", exact_n - equivalent, "provider_nonequivalent_complement", [metric_sources["provider_exact_fraction"]])

    full_image_traits = sources.table("figure_source_input", "full_image_traits")
    trait_records = list(contract.get("primary_root_traits", [])) + list(contract.get("root_hair_traits", []))
    fields = [str(record.get("field")) for record in trait_records if isinstance(record, Mapping)]
    require(len(fields) == len(set(fields)) == 32, "trait contract is not exact 32 unique fields")
    _require_columns(full_image_traits, ("task_id", "physical_units_valid", *fields), "full_image_traits")
    require(len(full_image_traits) == full_image_traits["task_id"].nunique() == 283, "full image traits are not exact283")
    coverage = [int(pd.to_numeric(full_image_traits[field], errors="coerce").notna().sum()) for field in fields]
    full_trait_source = _selected_source(sources, namespace="figure_source_input", role="full_image_traits", selected=full_image_traits, columns=("task_id", "physical_units_valid", *fields), filters={"source_units": 283, "trait_fields": 32}, sort_by=("task_id",))
    contract_cell = sources.json_cell(role="figure_resource:trait_contract", source=trait_contract_source, pointer=("counts", "nonredundant_biological_numeric_fields"))
    entries.add("FINAL_TRAIT_COVERAGE_MIN_N", min(coverage), "minimum_nonmissing_count_across_32_traits", [full_trait_source, contract_cell])
    entries.add("FINAL_TRAIT_COVERAGE_MAX_N", max(coverage), "maximum_nonmissing_count_across_32_traits", [full_trait_source, contract_cell])
    scale_eligible = int(_bool_series(full_image_traits["physical_units_valid"], "physical_units_valid").sum())
    entries.add("FINAL_SCALE_ELIGIBLE_N", scale_eligible, "count_application_images_passing_physical_scale_gate", [full_trait_source])


def _effect_sources(
    sources: Sources,
    resource_effects: pd.DataFrame,
    source_table: pd.DataFrame,
    atlas_source: JsonSource,
    *,
    cohort: str,
    endpoint: str,
    resource_effect: str,
    raw_effect: str,
) -> tuple[pd.Series, pd.Series, list[dict[str, Any]]]:
    resource_row = resource_effects[
        (resource_effects["cohort"].astype(str) == cohort)
        & (resource_effects["endpoint_key"].astype(str) == endpoint)
        & (resource_effects["effect_key"].astype(str) == resource_effect)
    ]
    raw_row = source_table[
        (source_table["cohort"].astype(str) == cohort)
        & (source_table["endpoint"].astype(str) == endpoint)
        & (source_table["effect"].astype(str) == raw_effect)
    ]
    require(len(resource_row) == len(raw_row) == 1, f"{cohort}/{endpoint}/{raw_effect}: effect row missing")
    for resource_key, raw_key in (("estimate", "estimate"), ("ci_low", "ci95_low"), ("ci_high", "ci95_high"), ("endpoint_n", "n")):
        require(math.isclose(_finite(resource_row.iloc[0][resource_key], resource_key), _finite(raw_row.iloc[0][raw_key], raw_key), abs_tol=1e-12, rel_tol=0), f"{cohort}/{endpoint}/{raw_effect}: normalized effect differs from source")
    resource_record = resource_row.iloc[0]
    raw_record = raw_row.iloc[0]
    require(
        str(resource_record["effect_scale"]) == str(raw_record["effect_scale"]),
        f"{cohort}/{endpoint}/{raw_effect}: normalized effect scale differs from source",
    )

    descriptors = atlas_source.payload.get("descriptors")
    require(isinstance(descriptors, list), "multitrait atlas descriptors missing")
    atlas_matches = [
        (index, descriptor)
        for index, descriptor in enumerate(descriptors)
        if isinstance(descriptor, Mapping) and str(descriptor.get("field")) == endpoint
    ]
    require(len(atlas_matches) == 1, f"{cohort}/{endpoint}: atlas descriptor is not unique")
    descriptor_index, descriptor = atlas_matches[0]
    cohorts = descriptor.get("cohorts")
    require(isinstance(cohorts, Mapping), f"{cohort}/{endpoint}: atlas cohort map missing")
    atlas_cohort = cohorts.get(cohort)
    require(isinstance(atlas_cohort, Mapping), f"{cohort}/{endpoint}: atlas cohort missing")
    atlas_effects = atlas_cohort.get("effects")
    require(isinstance(atlas_effects, Mapping), f"{cohort}/{endpoint}: atlas effects missing")
    atlas_record = atlas_effects.get(resource_effect)
    require(isinstance(atlas_record, Mapping), f"{cohort}/{endpoint}/{raw_effect}: atlas effect missing")

    numeric_bindings = (
        ("estimate", "estimate", "estimate"),
        ("ci_low", "ci95_low", "ci95_low"),
        ("ci_high", "ci95_high", "ci95_high"),
        ("endpoint_n", "n", "endpoint_n"),
        ("raw_effect_estimate", "raw_effect_estimate", "raw_effect_estimate"),
        ("raw_effect_ci_low", "raw_effect_ci95_low", "raw_effect_ci95_low"),
        ("raw_effect_ci_high", "raw_effect_ci95_high", "raw_effect_ci95_high"),
        ("standardized_effect", "standardized_effect", "standardized_effect"),
        ("standardized_ci_low", "standardized_ci95_low", "standardized_ci95_low"),
        ("standardized_ci_high", "standardized_ci95_high", "standardized_ci95_high"),
    )
    for resource_key, raw_key, atlas_key in numeric_bindings:
        resource_value = _finite(resource_record[resource_key], resource_key)
        raw_value = _finite(raw_record[raw_key], raw_key)
        atlas_value = _finite(atlas_record.get(atlas_key), atlas_key)
        require(
            math.isclose(resource_value, raw_value, abs_tol=1e-12, rel_tol=0)
            and math.isclose(resource_value, atlas_value, abs_tol=1e-12, rel_tol=0),
            f"{cohort}/{endpoint}/{raw_effect}: normalized/raw/atlas companion differs",
        )

    string_bindings = (
        ("effect_scale", "effect_scale", "effect_scale"),
        ("raw_effect_estimand", "raw_effect_estimand", "raw_effect_estimand"),
        ("raw_effect_interval_method", "raw_effect_interval_method", "raw_effect_interval_method"),
    )
    for resource_key, raw_key, atlas_key in string_bindings:
        values = {
            str(resource_record[resource_key]),
            str(raw_record[raw_key]),
            str(atlas_record.get(atlas_key)),
        }
        require(
            len(values) == 1,
            f"{cohort}/{endpoint}/{raw_effect}: normalized/raw/atlas semantic label differs",
        )
    resource_replicates = _integer(
        resource_record["raw_effect_bootstrap_replicates"], "raw companion replicates"
    )
    raw_replicates = _integer(
        raw_record["raw_effect_bootstrap_replicates"], "raw companion replicates"
    )
    atlas_replicates = _integer(
        atlas_record.get("raw_effect_bootstrap_replicates"), "raw companion replicates"
    )
    require(
        resource_replicates == raw_replicates == atlas_replicates,
        f"{cohort}/{endpoint}/{raw_effect}: normalized/raw/atlas bootstrap count differs",
    )

    def _optional_seed(value: Any) -> int | None:
        if value is None or pd.isna(value) or str(value).strip().casefold() in {"", "none", "null", "nan"}:
            return None
        return _integer(value, "raw companion bootstrap seed")

    seeds = (
        _optional_seed(resource_record["raw_effect_bootstrap_seed"]),
        _optional_seed(raw_record["raw_effect_bootstrap_seed"]),
        _optional_seed(atlas_record.get("raw_effect_bootstrap_seed")),
    )
    require(
        seeds[0] == seeds[1] == seeds[2],
        f"{cohort}/{endpoint}/{raw_effect}: normalized/raw/atlas bootstrap seed differs",
    )

    raw_estimand = str(raw_record["raw_effect_estimand"])
    raw_interval = str(raw_record["raw_effect_interval_method"])
    if endpoint == H11_ENDPOINT:
        expected_seed = raw_median_bootstrap_seed(
            seed=H11_RAW_BOOTSTRAP_BASE_SEED,
            field=H11_ENDPOINT,
            component="continuous",
        )
        require(
            raw_estimand == RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
            and raw_interval == RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
            and raw_replicates == H11_RAW_BOOTSTRAP_REPLICATES
            and seeds[0] == expected_seed,
            f"{cohort}/{endpoint}/{raw_effect}: H11 raw-median bootstrap contract changed",
        )
        summaries = atlas_cohort.get("condition_summaries")
        require(isinstance(summaries, Mapping), f"{cohort}/{endpoint}: H11 condition summaries missing")
        medians = []
        for condition in GROUPS:
            summary = summaries.get(condition)
            require(isinstance(summary, Mapping), f"{cohort}/{endpoint}/{condition}: H11 summary missing")
            medians.append(_finite(summary.get("median"), f"{condition} H11 median"))
        ev22, ev30, oe22, oe30 = medians
        expected_raw = {
            "OE_vs_EV": 0.5 * ((oe22 - ev22) + (oe30 - ev30)),
            "30C_vs_22C": 0.5 * ((ev30 - ev22) + (oe30 - oe22)),
            "interaction": (oe30 - oe22) - (ev30 - ev22),
        }[resource_effect]
        require(
            math.isclose(
                _finite(raw_record["raw_effect_estimate"], "H11 raw effect"),
                expected_raw,
                abs_tol=1e-12,
                rel_tol=0,
            ),
            f"{cohort}/{endpoint}/{raw_effect}: H11 raw effect is not the four-cell median contrast",
        )
    else:
        require(
            raw_estimand == RAW_EFFECT_OLS_MEAN_CONTRAST
            and raw_interval == RAW_EFFECT_HC3_INTERVAL
            and raw_replicates == 0
            and seeds[0] is None,
            f"{cohort}/{endpoint}/{raw_effect}: non-H11 raw-mean companion contract changed",
        )

    raw_triplet = tuple(
        _finite(raw_record[key], key)
        for key in ("raw_effect_estimate", "raw_effect_ci95_low", "raw_effect_ci95_high")
    )
    standardized_triplet = tuple(
        _finite(raw_record[key], key)
        for key in ("standardized_effect", "standardized_ci95_low", "standardized_ci95_high")
    )
    implied_scales = []
    for raw_value, standardized_value in zip(raw_triplet, standardized_triplet, strict=True):
        if abs(standardized_value) <= 1e-12:
            require(abs(raw_value) <= 1e-12, f"{cohort}/{endpoint}/{raw_effect}: standardized zero is inconsistent")
        else:
            implied_scales.append(raw_value / standardized_value)
    require(
        implied_scales
        and all(value > 0 and math.isfinite(value) for value in implied_scales)
        and all(
            math.isclose(value, implied_scales[0], rel_tol=1e-12, abs_tol=1e-12)
            for value in implied_scales[1:]
        ),
        f"{cohort}/{endpoint}/{raw_effect}: standardized companion does not use one positive source-unit SD",
    )
    resource_source = _selected_source(sources, namespace="figure_resource", role="phenotype_effects", selected=resource_row, columns=tuple(resource_effects.columns), filters={"cohort": cohort, "endpoint_key": endpoint, "effect_key": resource_effect})
    source_role = "analysis_primary_table" if cohort == "primary_clean261" else "analysis_sensitivity_table"
    raw_source = _selected_source(sources, namespace="figure_source_input", role=source_role, selected=raw_row, columns=tuple(source_table.columns), filters={"cohort": cohort, "endpoint": endpoint, "effect": raw_effect})
    atlas_effect_source = sources.json_cell(
        role="figure_resource:multitrait_atlas",
        source=atlas_source,
        pointer=("descriptors", descriptor_index, "cohorts", cohort, "effects", resource_effect),
    )
    return resource_record, raw_record, [resource_source, raw_source, atlas_effect_source]


def _effect_interpretation(estimate: float, low: float, high: float) -> tuple[str, str]:
    direction = "higher" if estimate > 1 else "lower" if estimate < 1 else "unchanged"
    if low > 1:
        interpretation = "positive association with an interval above no difference"
    elif high < 1:
        interpretation = "negative association with an interval below no difference"
    else:
        interpretation = "an estimate whose interval spanned no difference"
    return direction, interpretation


def _endpoint_point_contract(points: pd.DataFrame, endpoint: str) -> tuple[pd.DataFrame, tuple[int, ...]]:
    selected = points[
        (points["cohort"].astype(str) == "primary_clean261")
        & (points["endpoint_key"].astype(str) == endpoint)
        & _bool_series(points["formal_eligible"], "formal_eligible")
    ]
    require(set(selected["condition_code"].astype(str)) == set(GROUPS), f"{endpoint}: primary points lost one or more D15 cells")
    values = pd.to_numeric(selected["value"], errors="coerce")
    require(values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all(), f"{endpoint}: non-finite source-unit value")
    cell_n = tuple(
        int((selected["condition_code"].astype(str) == group).sum())
        for group in GROUPS
    )
    require(all(value > 0 for value in cell_n), f"{endpoint}: empty primary condition cell")
    return selected, cell_n


def _effect_qualified_pattern(
    *,
    prefix: str,
    endpoint: str,
    points: pd.DataFrame,
    effects: pd.DataFrame,
    effect_cache: Mapping[tuple[str, str], tuple[Any, ...]],
) -> tuple[str, pd.DataFrame]:
    """Narrate one endpoint's three fixed effects in biological reading order.

    The source-unit counts are still required to close exactly against the
    fitted endpoint denominator, but they belong in the table-level denominator
    statement rather than being repeated in each endpoint sentence.  Each
    sentence exposes the three decisions a reader needs: clean point-estimate
    state, interval position relative to one, and Full283 state sensitivity.
    """

    require(prefix in ENDPOINTS and ENDPOINTS[prefix] == endpoint, f"{endpoint}: endpoint order changed")
    selected, cell_n = _endpoint_point_contract(points, endpoint)
    clean_rows = effects[
        (effects["cohort"].astype(str) == "primary_clean261")
        & (effects["endpoint_key"].astype(str) == endpoint)
    ]
    endpoint_ns = set(pd.to_numeric(clean_rows["endpoint_n"], errors="raise").astype(int))
    require(len(clean_rows) == 3 and len(endpoint_ns) == 1, f"{endpoint}: clean effect denominator mismatch")
    require(sum(cell_n) == next(iter(endpoint_ns)), f"{endpoint}: four-cell non-null counts do not close to effect endpoint n")

    effect_names = {
        "CONSTRUCT": "OE-labelled:EV contrast",
        "TEMPERATURE": "30:22°C contrast",
        "INTERACTION": "construct-by-temperature interaction",
    }
    sentences: list[str] = []
    for effect_label in EFFECTS:
        require((prefix, effect_label) in effect_cache, f"{endpoint}/{effect_label}: effect cache missing")
        estimate, low, high, _q, _bound, full_estimate, _full_low, _full_high, _full_bound = effect_cache[(prefix, effect_label)]
        name = effect_names[effect_label]
        point_relation, clean_point_state, full_point_state = _ratio_point_relation(
            estimate,
            full_estimate,
        )
        direction = (
            "higher"
            if clean_point_state == "higher"
            else "lower"
            if clean_point_state == "lower"
            else "at no difference"
        )
        full_state_phrase = (
            "at no difference"
            if full_point_state == "null"
            else full_point_state
        )
        if low > 1.0:
            interval = "above"
        elif high < 1.0:
            interval = "below"
        else:
            interval = "spanning"

        article = "the" if not sentences else "The"
        if interval in {"above", "below"} and point_relation == "same":
            sentence = (
                f"{article} {name} was {direction}, with its clean-cohort interval {interval} "
                "the no-difference ratio and the Full283 estimate pointing in the same "
                "direction."
            )
        elif interval in {"above", "below"} and point_relation == "opposite":
            sentence = (
                f"{article} {name} was {direction}, and its clean-cohort interval remained "
                f"{interval} the no-difference ratio, whereas the Full283 estimate pointed "
                "in the opposite direction."
            )
        elif interval in {"above", "below"}:
            sentence = (
                f"{article} {name} was {direction}, and its clean-cohort interval remained "
                f"{interval} the no-difference ratio, whereas the Full283 estimate was "
                f"{full_state_phrase}."
            )
        elif point_relation == "same" and clean_point_state == "null":
            sentence = (
                f"{article} {name} was at no difference, with a clean-cohort interval "
                "spanning the no-difference ratio; both point estimates were at the "
                "no-difference ratio."
            )
        elif point_relation == "same":
            sentence = (
                f"{article} {name} was {direction}, with a clean-cohort interval spanning "
                "the no-difference ratio; the Full283 estimate retained the same direction."
            )
        elif point_relation == "opposite":
            sentence = (
                f"{article} {name} was {direction}, with a clean-cohort interval spanning the "
                "no-difference ratio; the Full283 estimate pointed in the opposite direction."
            )
        else:
            sentence = (
                f"{article} {name} was {direction}, with a clean-cohort interval spanning the "
                f"no-difference ratio; the Full283 estimate was {full_state_phrase}, so the "
                "two point-estimate states did not match."
            )
        sentences.append(sentence)

    # Closure is intentionally validated above even though n is reported once,
    # outside the endpoint narratives, in the fixed-family table contract.
    require(sum(cell_n) == next(iter(endpoint_ns)), f"{endpoint}: endpoint count closure changed")
    return " ".join(sentences), selected


def _abstract_biology_synthesis(
    decision: Mapping[str, Any],
    decision_source: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Render one real, fixed-priority biological headline from stage 36.

    A/B headlines name the first supported cell in the locked
    N->L->F->W->A by construct->temperature->interaction order.  This keeps
    the abstract independent of effect magnitude or p-value and, for branch
    B, reports one cell without implying that cross-layer directions agree.
    """

    selected = validate_narrative_decision(decision)
    branch = selected["branch_id"]
    supported_cells = [
        cell for cell in selected["cells"] if cell["headline_supported"]
    ]
    headline_cell: dict[str, str] | None = None
    if branch in {"A", "B"}:
        require(
            bool(supported_cells),
            f"narrative branch {branch} has no supported headline cell",
        )
        cell = supported_cells[0]
        endpoint = str(cell["endpoint_key"])
        effect = str(cell["effect_key"])
        direction = str(cell["clean_direction"])
        require(
            endpoint in NARRATIVE_ENDPOINT_ORDER,
            "abstract headline endpoint escaped the locked order",
        )
        require(
            effect in NARRATIVE_EFFECT_ORDER,
            "abstract headline effect escaped the locked order",
        )
        require(
            direction in {"higher", "lower"},
            "supported abstract headline has no directional clean estimate",
        )
        contract = NARRATIVE_ENDPOINT_CONTRACT[endpoint]
        require(
            str(cell["sentinel"]) == contract["sentinel"]
            and str(cell["badge"]) == contract["badge"]
            and str(cell["layer"]) == contract["layer"],
            "abstract headline endpoint contract changed",
        )
        effect_label = {
            "OE_vs_EV": "OE-versus-EV construct-label contrast",
            "30C_vs_22C": "30-versus-22°C temperature contrast",
            "interaction": "construct-by-temperature interaction",
        }[effect]
        endpoint_label = f"{contract['sentinel']}/{contract['badge']} {contract['layer']}"
        text = f"clean-cohort {endpoint_label} was {direction} for the {effect_label}"
        headline_cell = {
            "endpoint_key": endpoint,
            "effect_key": effect,
            "sentinel": str(contract["sentinel"]),
            "badge": str(contract["badge"]),
            "layer": str(contract["layer"]),
            "clean_direction": direction,
        }
    else:
        require(
            branch == "C" and not supported_cells,
            "narrative branch C contains a supported headline cell",
        )
        text = (
            "no clean-cohort endpoint–effect cell met the predefined cross-cohort "
            "support rule"
        )
    require(
        text_word_count(text) <= ABSTRACT_BIOLOGY_SYNTHESIS_WORD_LIMIT,
        "abstract biological synthesis exceeded its 15-word internal budget",
    )
    return (
        text,
        [deepcopy(dict(decision_source))],
        {
            "selection_rule": selected["decision_rule"],
            "narrative_decision_identity_sha256": selected[
                "narrative_decision_identity_sha256"
            ],
            "narrative_branch_id": branch,
            "support_mask_bits": selected["support_mask_bits"],
            "supported_layers": selected["supported_layers"],
            "headline_cell": headline_cell,
            "headline_priority_endpoint_order": list(NARRATIVE_ENDPOINT_ORDER),
            "headline_priority_effect_order": list(NARRATIVE_EFFECT_ORDER),
            "headline_uses_effect_magnitude_or_p_value": False,
            "branch_b_common_direction_claimed": False,
            "profiles_select_or_veto_narrative_branch": False,
            "correlated_descriptor_count_used_as_biological_conclusion": False,
            "maximum_words": ABSTRACT_BIOLOGY_SYNTHESIS_WORD_LIMIT,
        },
    )


def _validated_narrative_decision(
    context: BuildContext,
    sources: Sources,
    effect_cache: Mapping[tuple[str, str], tuple[Any, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load stage36's decision and close every decision cell to stage50 inputs."""

    decision_source = sources.json_resource("narrative_decision")
    try:
        decision = validate_narrative_decision(decision_source.payload)
    except ValueError as error:
        raise ManuscriptValuesError("narrative decision failed deterministic validation") from error
    require(
        decision["narrative_decision_identity_sha256"]
        == context.figure_inputs.payload.get("narrative_decision_identity_sha256")
        and decision["branch_id"]
        == context.figure_inputs.payload.get("narrative_branch_id")
        == context.figure_assembly_summary.payload.get("narrative_branch_id"),
        "stage50 did not consume the stage36 narrative decision",
    )
    prefix_by_endpoint = {value: key for key, value in ENDPOINTS.items()}
    effect_label_by_key = {value[0]: key for key, value in EFFECTS.items()}
    require(
        list(decision["endpoint_order"]) == list(NARRATIVE_ENDPOINT_ORDER)
        and list(decision["effect_order"]) == list(NARRATIVE_EFFECT_ORDER),
        "narrative N/L/F/W/A or effect order changed",
    )
    for cell in decision["cells"]:
        endpoint = str(cell["endpoint_key"])
        effect = str(cell["effect_key"])
        require(
            endpoint in prefix_by_endpoint and effect in effect_label_by_key,
            "narrative decision contains an unknown sentinel/effect cell",
        )
        cached = effect_cache[(prefix_by_endpoint[endpoint], effect_label_by_key[effect])]
        clean = cell["clean"]
        full = cell["full283"]
        for observed, expected, label in (
            (cached[0], clean["estimate"], "clean estimate"),
            (cached[1], clean["ci_low"], "clean CI low"),
            (cached[2], clean["ci_high"], "clean CI high"),
            (cached[5], full["estimate"], "full estimate"),
            (cached[6], full["ci_low"], "full CI low"),
            (cached[7], full["ci_high"], "full CI high"),
        ):
            require(
                math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12),
                f"narrative decision {endpoint}/{effect} {label} differs from stage50 evidence",
            )
    identity_source = sources.json_cell(
        role="figure_resource:narrative_decision",
        source=decision_source,
        pointer=("narrative_decision_identity_sha256",),
    )
    return decision, identity_source


def _require_builder_abstract_within_limit(
    master_text: str,
    entries: Mapping[str, Mapping[str, Any]],
) -> int:
    """Render actual values and fail before sealing an over-length abstract."""

    rendered = master_text
    for token in sorted(entries):
        entry = entries[token]
        require(isinstance(entry, Mapping), f"{token}: derived entry malformed")
        require("value" in entry, f"{token}: derived entry value missing")
        rendered = rendered.replace(f"{{{{{token}}}}}", str(entry["value"]))
    try:
        return require_abstract_within_limit(rendered, limit=ABSTRACT_WORD_LIMIT)
    except ManuscriptTextContractError as error:
        raise ManuscriptValuesError(str(error)) from error


def _join_readable(items: Sequence[str]) -> str:
    require(bool(items), "cannot join an empty narrative list")
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _multitrait_atlas_fingerprint(
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Describe coverage of all 32 traits without cross-trait biological voting.

    The atlas intentionally contains related, reciprocal, and deterministically
    scaled descriptors.  A family-level vote over condition-wise maxima would
    therefore give redundant descriptors extra biological weight and would mix
    quantities whose directional meanings differ.  This summary validates all
    raw condition cells and reports coverage only; biological synthesis remains
    restricted to the five prespecified, non-redundant endpoints.
    """

    require(
        payload.get("schema_version") == MULTITRAIT_ATLAS_SCHEMA_VERSION
        and payload.get("status") == "completed_source_derived_32_trait_atlas",
        "multitrait atlas is not the completed source-derived v2 contract",
    )
    require(payload.get("blind_images_used") == 0, "multitrait atlas blind guard changed")
    require(
        payload.get("root_cap_region_statistics_included") is False,
        "root-cap region statistics entered the multitrait narrative",
    )
    require(
        payload.get("descriptor_count") == 32
        and payload.get("root_descriptor_count") == 19
        and payload.get("hair_descriptor_count") == 13,
        "multitrait atlas descriptor counts changed",
    )
    require(
        payload.get("condition_order") == list(GROUPS)
        and payload.get("measurement_family_order") == list(MEASUREMENT_FAMILY_ORDER),
        "multitrait atlas condition/family order changed",
    )
    require(
        payload.get("prespecified_inferential_endpoint_fields")
        == list(ATLAS_PRIMARY_ENDPOINTS)
        and tuple(ATLAS_PRIMARY_ENDPOINTS) == tuple(ENDPOINTS.values())
        and payload.get("effect_order") == list(ATLAS_EFFECT_KEYS)
        and payload.get("estimated_effect_slot_count") == 30
        and payload.get("not_estimated_effect_slot_count") == 162,
        "multitrait atlas no longer preserves the clean/full fixed 15-effect family",
    )
    identity = payload.get("atlas_identity_sha256")
    require(isinstance(identity, str) and len(identity) == 64, "multitrait atlas identity missing")
    unsigned = deepcopy(dict(payload))
    unsigned.pop("atlas_identity_sha256", None)
    require(sha256_json(unsigned) == identity, "multitrait atlas identity drift")

    descriptors = payload.get("descriptors")
    require(isinstance(descriptors, list) and len(descriptors) == 32, "multitrait atlas descriptors missing")
    expected_family_by_trait = {
        trait_id: family
        for family, trait_ids in MEASUREMENT_FAMILY_TRAIT_IDS.items()
        for trait_id in trait_ids
    }
    require(len(expected_family_by_trait) == 32, "measurement-family ontology is incomplete")
    observed_traits: set[str] = set()
    family_records: dict[str, list[Mapping[str, Any]]] = {
        family: [] for family in MEASUREMENT_FAMILY_ORDER
    }
    for descriptor in descriptors:
        require(isinstance(descriptor, Mapping), "multitrait atlas descriptor malformed")
        trait_id = str(descriptor.get("trait_id", ""))
        family = str(descriptor.get("measurement_family", ""))
        require(
            trait_id in expected_family_by_trait
            and expected_family_by_trait[trait_id] == family
            and trait_id not in observed_traits,
            f"multitrait atlas trait/family membership changed: {trait_id}",
        )
        require(str(descriptor.get("field", "")), f"{trait_id}: atlas field missing")
        observed_traits.add(trait_id)
        family_records[family].append(descriptor)
    require(observed_traits == set(expected_family_by_trait), "multitrait atlas trait set changed")

    family_coverage: dict[str, Any] = {}
    for family in MEASUREMENT_FAMILY_ORDER:
        records = family_records[family]
        require(
            {str(record["trait_id"]) for record in records}
            == set(MEASUREMENT_FAMILY_TRAIT_IDS[family]),
            f"{family}: descriptor membership changed",
        )
        incomplete_trait_n = 0
        coverage: list[float] = []
        for descriptor in records:
            trait_id = str(descriptor["trait_id"])
            cohorts = descriptor.get("cohorts")
            require(
                isinstance(cohorts, Mapping) and "primary_clean261" in cohorts,
                f"{trait_id}: clean atlas cohort missing",
            )
            clean = cohorts["primary_clean261"]
            require(isinstance(clean, Mapping), f"{trait_id}: clean atlas cohort malformed")
            summaries = clean.get("condition_summaries")
            require(
                isinstance(summaries, Mapping) and set(summaries) == set(GROUPS),
                f"{trait_id}: four-condition summaries changed",
            )
            fully_observed = True
            for group in GROUPS:
                summary = summaries[group]
                require(isinstance(summary, Mapping), f"{trait_id}/{group}: condition summary malformed")
                total = summary.get("source_unit_total")
                non_null = summary.get("non_null_source_unit_n")
                observed_fraction = summary.get("observability_fraction")
                require(
                    isinstance(total, int)
                    and not isinstance(total, bool)
                    and total > 0
                    and isinstance(non_null, int)
                    and not isinstance(non_null, bool)
                    and 0 <= non_null <= total,
                    f"{trait_id}/{group}: observability counts invalid",
                )
                fraction = _finite(observed_fraction, f"{trait_id}/{group} observability")
                require(
                    0.0 <= fraction <= 1.0
                    and math.isclose(fraction, non_null / total, rel_tol=1e-12, abs_tol=1e-12),
                    f"{trait_id}/{group}: observability denominator does not close",
                )
                coverage.append(fraction)
                if non_null == 0:
                    fully_observed = False
                    require(
                        summary.get("median") is None
                        and summary.get("summary_status")
                        == "not_estimated_no_finite_source_units",
                        f"{trait_id}/{group}: unobserved condition acquired a median",
                    )
                    continue
                require(
                    summary.get("summary_status") == CONDITION_SUMMARY_STATUS,
                    f"{trait_id}/{group}: observed condition summary status changed",
                )
                _finite(summary.get("median"), f"{trait_id}/{group} median")
            if not fully_observed:
                incomplete_trait_n += 1

        fully_observed_trait_n = len(records) - incomplete_trait_n
        require(coverage, f"{family}: atlas coverage is empty")
        family_coverage[family] = {
            "descriptor_n": len(records),
            "fully_observed_trait_n": fully_observed_trait_n,
            "incomplete_trait_n": incomplete_trait_n,
            "observability_min": min(coverage),
            "observability_max": max(coverage),
        }

    overall_observability_min = min(
        record["observability_min"] for record in family_coverage.values()
    )
    overall_observability_max = max(
        record["observability_max"] for record in family_coverage.values()
    )
    family_text = _join_readable(
        [
            MEASUREMENT_FAMILY_COMPACT_LABELS[family]
            for family in MEASUREMENT_FAMILY_ORDER
        ]
    )
    text = (
        "the condition-resolved 32-descriptor atlas retained native-unit raw "
        f"summaries across {family_text}; overall measurement coverage was "
        f"{_fmt_percent(overall_observability_min)}–"
        f"{_fmt_percent(overall_observability_max)}%. Biological synthesis remained "
        "prespecified to five non-redundant endpoints and 15 effects; missing cells "
        "remain unfilled, and no cross-trait ranking or directional vote was used"
    )
    require(len(text.split()) <= 65, "multitrait atlas main-text summary exceeded 65 words")
    return text, {
        "canonical_descriptor_n": 32,
        "modeled_endpoint_n": 5,
        "fixed_effect_family_n": 15,
        "condition_order": list(GROUPS),
        "measurement_family_order": list(MEASUREMENT_FAMILY_ORDER),
        "family_coverage": family_coverage,
        "overall_observability_min": overall_observability_min,
        "overall_observability_max": overall_observability_max,
        "cross_trait_native_unit_values_pooled": False,
        "cross_trait_directional_vote_used": False,
        "biological_synthesis_endpoint_fields": list(ATLAS_PRIMARY_ENDPOINTS),
        "missing_values_zero_filled": False,
    }


def _ratio_point_direction(value: Any) -> str:
    """Classify a ratio point estimate into one mutually exclusive state."""

    estimate = _finite(value, "ratio point estimate")
    if estimate < 1.0:
        return "lower"
    if estimate > 1.0:
        return "higher"
    return "null"


def _ratio_point_relation(clean_value: Any, full_value: Any) -> tuple[str, str, str]:
    """Relate clean and Full283 ratio states without treating null as opposite."""

    clean_state = _ratio_point_direction(clean_value)
    full_state = _ratio_point_direction(full_value)
    if clean_state == full_state:
        relation = "same"
    elif {clean_state, full_state} == {"lower", "higher"}:
        relation = "opposite"
    else:
        relation = "null_transition"
    return relation, clean_state, full_state


def _derive_biology(
    context: BuildContext,
    sources: Sources,
    entries: EntryBuilder,
) -> tuple[dict[tuple[str, str], tuple[Any, ...]], dict[str, Any]]:
    points = sources.table("figure_resource", "phenotype_points")
    effects = sources.table("figure_resource", "phenotype_effects")
    primary = sources.table("figure_source_input", "analysis_primary_table")
    sensitivity = sources.table("figure_source_input", "analysis_sensitivity_table")
    clean = sources.table("figure_source_input", "clean_traits")
    full = sources.table("figure_source_input", "full_traits")
    full_image = sources.table("figure_source_input", "full_image_traits")
    atlas_source = sources.json_resource("multitrait_atlas")
    atlas_summary, atlas_parameters = _multitrait_atlas_fingerprint(
        atlas_source.payload
    )
    _require_columns(points, ("source_unit", "cohort", "condition_code", "formal_eligible", "endpoint_key", "value", "unit"), "phenotype_points")
    companion_resource_columns = (
        "raw_effect_estimate", "raw_effect_ci_low", "raw_effect_ci_high",
        "raw_effect_estimand", "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates", "raw_effect_bootstrap_seed",
        "standardized_effect", "standardized_ci_low", "standardized_ci_high",
    )
    _require_columns(effects, ("cohort", "endpoint_key", "effect_key", "estimate", "ci_low", "ci_high", "endpoint_n", "effect_scale", *companion_resource_columns), "phenotype_effects")
    effect_columns = (
        "cohort", "endpoint", "effect", "n", "estimate", "ci95_low",
        "ci95_high", "p_value_model_BH_FDR", "effect_scale",
        "causal_treatment_claim_allowed", "raw_effect_estimate",
        "raw_effect_ci95_low", "raw_effect_ci95_high", "raw_effect_estimand",
        "raw_effect_interval_method", "raw_effect_bootstrap_replicates",
        "raw_effect_bootstrap_seed", "standardized_effect",
        "standardized_ci95_low", "standardized_ci95_high",
    )
    _require_columns(primary, effect_columns, "analysis_primary_table")
    _require_columns(sensitivity, effect_columns, "analysis_sensitivity_table")
    require(len(effects) == 30, "phenotype effect table is not clean/full x fixed 15 family")
    require(set(effects["cohort"].astype(str)) == {"primary_clean261", "sensitivity_full283"}, "effect cohort set changed")

    effect_cache: dict[
        tuple[str, str],
        tuple[
            float,
            float,
            float,
            float,
            list[dict[str, Any]],
            float,
            float,
            float,
            list[dict[str, Any]],
        ],
    ] = {}
    endpoint_direction_concordance = {prefix: 0 for prefix in ENDPOINTS}
    for prefix, endpoint in ENDPOINTS.items():
        for effect_label, (resource_effect, raw_effect) in EFFECTS.items():
            row, raw, bound = _effect_sources(sources, effects, primary, atlas_source, cohort="primary_clean261", endpoint=endpoint, resource_effect=resource_effect, raw_effect=raw_effect)
            full_row, _full_raw, full_bound = _effect_sources(sources, effects, sensitivity, atlas_source, cohort="sensitivity_full283", endpoint=endpoint, resource_effect=resource_effect, raw_effect=raw_effect)
            estimate = _finite(row["estimate"], "primary estimate")
            low = _finite(row["ci_low"], "primary CI")
            high = _finite(row["ci_high"], "primary CI")
            q = _finite(raw["p_value_model_BH_FDR"], "primary BH q")
            require(0 <= q <= 1 and 0 < low <= high, "effect interval/q invalid")
            full_estimate = _finite(full_row["estimate"], "full estimate")
            full_low = _finite(full_row["ci_low"], "full CI")
            full_high = _finite(full_row["ci_high"], "full CI")
            full_endpoint_n = _integer(full_row["endpoint_n"], "full endpoint n")
            direction, interpretation = _effect_interpretation(estimate, low, high)
            point_relation, _clean_point_state, _full_point_state = _ratio_point_relation(
                estimate,
                full_estimate,
            )
            direction_concordant = point_relation == "same"
            endpoint_direction_concordance[prefix] += int(direction_concordant)
            token_base = f"FINAL_{prefix}_{effect_label}"
            entries.add(f"{token_base}_RATIO", _fmt(estimate), "copy_primary_effect_ratio", bound, parameters={"endpoint": endpoint, "effect": raw_effect})
            entries.add(f"{token_base}_CI", _fmt_ci(low, high), "format_primary_effect_interval", bound, parameters={"endpoint": endpoint, "effect": raw_effect})
            # The compact main table retains all 15 ratios and intervals, while
            # q values and the expanded full283 records live in Supplementary
            # Data S9.  Derive those prose tokens only when a manuscript edition
            # explicitly requests them; effect_cache below still retains the
            # complete fixed-family evidence for narratives and figures.
            if f"{token_base}_Q" in entries.contract["tokens"]:
                entries.add(f"{token_base}_Q", _fmt(q), "copy_fixed_family_BH_adjusted_q", bound, parameters={"family_size": 15})
            if f"{token_base}_FULL_SENSITIVITY" in entries.contract["tokens"]:
                entries.add(
                    f"{token_base}_FULL_SENSITIVITY",
                    (
                        f"{_fmt(full_estimate)} ({_fmt_ci(full_low, full_high)}); "
                        f"endpoint n={full_endpoint_n}; clean/full point-estimate state "
                        f"{ {'same': 'concordant', 'opposite': 'opposite', 'null_transition': 'null-transition'}[point_relation] }"
                    ),
                    "format_full283_sensitivity_effect_interval_denominator_and_point_state_relation",
                    full_bound,
                    parameters={
                        "endpoint": endpoint,
                        "effect": raw_effect,
                        "null_ratio": 1.0,
                    },
                )
            if f"{token_base}_INTERPRETATION" in entries.contract["tokens"]:
                entries.add(f"{token_base}_INTERPRETATION", interpretation, "classify_effect_interval_relative_to_one", bound, parameters={"null_ratio": 1.0})
            if effect_label in {"CONSTRUCT", "TEMPERATURE"} and prefix == "ABUNDANCE":
                entries.add(f"{token_base}_DIRECTION", direction, "classify_effect_direction_relative_to_one", bound, parameters={"null_ratio": 1.0})
            effect_cache[(prefix, effect_label)] = (
                estimate,
                low,
                high,
                q,
                bound,
                full_estimate,
                full_low,
                full_high,
                full_bound,
            )

    pattern_sources: dict[str, dict[str, Any]] = {}
    patterns: dict[str, str] = {}
    for prefix, endpoint in ENDPOINTS.items():
        patterns[prefix], selected = _effect_qualified_pattern(
            prefix=prefix,
            endpoint=endpoint,
            points=points,
            effects=effects,
            effect_cache=effect_cache,
        )
        pattern_sources[prefix] = _selected_source(sources, namespace="figure_resource", role="phenotype_points", selected=selected, columns=tuple(points.columns), filters={"cohort": "primary_clean261", "endpoint_key": endpoint, "conditions": list(GROUPS)}, sort_by=("condition_code", "source_unit"))
    support_min_value = entries.entries["FINAL_D15_LENGTH_SUPPORT_MIN_PERCENT"]
    support_max_value = entries.entries["FINAL_D15_LENGTH_SUPPORT_MAX_PERCENT"]
    patterns["LENGTH"] += (
        " This projected-length contrast was conditional on endpoint-complete "
        f"support of {support_min_value['value']}%–{support_max_value['value']}% across cells"
    )
    pattern_evidence = {
        prefix: [
            pattern_sources[prefix],
            *[
                source
                for effect_label in EFFECTS
                for source in (
                    *effect_cache[(prefix, effect_label)][4],
                    *effect_cache[(prefix, effect_label)][8],
                )
            ],
        ]
        for prefix in ENDPOINTS
    }
    entries.add("FINAL_D15_ABUNDANCE_PATTERN", patterns["ABUNDANCE"], "narrate_fixed_order_effect_direction_interval_and_sensitivity", pattern_evidence["ABUNDANCE"], parameters={"effect_order": list(EFFECTS), "endpoint_n_reported_elsewhere": True})
    entries.add(
        "FINAL_D15_LENGTH_PATTERN",
        patterns["LENGTH"],
        "narrate_fixed_order_effect_direction_interval_sensitivity_and_length_support",
        [
            *pattern_evidence["LENGTH"],
            *support_min_value["derivation"]["sources"],
            *support_max_value["derivation"]["sources"],
        ],
        parameters={"effect_order": list(EFFECTS), "conditional_on_endpoint_complete_support": True, "endpoint_n_reported_elsewhere": True},
    )
    entries.add("FINAL_FIRST_HAIR_PATTERN", patterns["FIRST_HAIR"], "narrate_fixed_order_effect_direction_interval_and_sensitivity", pattern_evidence["FIRST_HAIR"], parameters={"effect_order": list(EFFECTS), "event_observability_mode": "descriptive_only", "modeled_component": "conditional_positive_distance_only", "endpoint_n_reported_elsewhere": True})
    entries.add("FINAL_ROOT_WIDTH_PATTERN", patterns["ROOT_WIDTH"], "narrate_fixed_order_effect_direction_interval_and_sensitivity", pattern_evidence["ROOT_WIDTH"], parameters={"effect_order": list(EFFECTS), "endpoint_n_reported_elsewhere": True})
    entries.add("FINAL_ROOT_LENGTH_PATTERN", patterns["ROOT_LENGTH"], "narrate_fixed_order_effect_direction_interval_and_sensitivity", pattern_evidence["ROOT_LENGTH"], parameters={"effect_order": list(EFFECTS), "endpoint_n_reported_elsewhere": True})
    narrative_decision, narrative_decision_source = _validated_narrative_decision(
        context,
        sources,
        effect_cache,
    )
    abstract_synthesis, abstract_sources, abstract_parameters = _abstract_biology_synthesis(
        narrative_decision,
        narrative_decision_source,
    )
    entries.add(
        "FINAL_D15_ABSTRACT_SYNTHESIS",
        abstract_synthesis,
        "fixed_priority_interval_and_sensitivity_gated_abstract_synthesis",
        abstract_sources,
        parameters=abstract_parameters,
    )

    required_trait_columns = (
        "task_id", "experiment_key", "condition_code", "study_role", "formal_statistics_eligible",
        "hair_count", "hair_length_measurement_hair_count", *ENDPOINTS.values(),
    )
    _require_columns(clean, required_trait_columns, "clean_traits")
    _require_columns(full, required_trait_columns, "full_traits")
    clean_pool = clean[
        (clean["experiment_key"].astype(str) == "D15_8d")
        & (clean["study_role"].astype(str) == "rhd6_factorial_8d_primary")
        & clean["condition_code"].astype(str).isin(GROUPS)
    ]
    full_pool = full[
        (full["experiment_key"].astype(str) == "D15_8d")
        & (full["study_role"].astype(str) == "rhd6_factorial_8d_primary")
        & full["condition_code"].astype(str).isin(GROUPS)
    ]
    clean_scope = clean_pool[
        _bool_series(clean_pool["formal_statistics_eligible"], "clean formal eligibility")
    ]
    full_scope = full_pool[
        _bool_series(full_pool["formal_statistics_eligible"], "full formal eligibility")
    ]
    require(set(clean_scope["condition_code"].astype(str)) == set(full_scope["condition_code"].astype(str)) == set(GROUPS), "D15 scope lost a condition")
    clean_pool_source = _selected_source(sources, namespace="figure_source_input", role="clean_traits", selected=clean_pool, columns=required_trait_columns, filters={"cohort": "primary_clean261", "experiment_key": "D15_8d", "formal": "not_applied"}, sort_by=("task_id",))
    full_pool_source = _selected_source(sources, namespace="figure_source_input", role="full_traits", selected=full_pool, columns=required_trait_columns, filters={"cohort": "sensitivity_full283", "experiment_key": "D15_8d", "formal": "not_applied"}, sort_by=("task_id",))
    clean_source = _selected_source(sources, namespace="figure_source_input", role="clean_traits", selected=clean_scope, columns=required_trait_columns, filters={"cohort": "primary_clean261", "experiment_key": "D15_8d", "formal": True}, sort_by=("task_id",))
    full_source = _selected_source(sources, namespace="figure_source_input", role="full_traits", selected=full_scope, columns=required_trait_columns, filters={"cohort": "sensitivity_full283", "experiment_key": "D15_8d", "formal": True}, sort_by=("task_id",))
    entries.add("FINAL_D15_CLEAN_FORMAL_N", len(clean_scope), "count_clean_D15_formal_source_units", [clean_source])
    entries.add("FINAL_D15_FULL_FORMAL_N", len(full_scope), "count_full_D15_formal_source_units", [full_source])
    clean_counts = clean_scope["condition_code"].astype(str).value_counts()
    full_counts = full_scope["condition_code"].astype(str).value_counts()
    for group, suffix in (("RHD6_EV_22C", "EV22"), ("RHD6_EV_30C", "EV30"), ("RHD6_OE_22C", "OE22"), ("RHD6_OE_30C", "OE30")):
        entries.add(f"FINAL_D15_CLEAN_{suffix}_N", int(clean_counts[group]), "count_condition_source_units", [clean_source], parameters={"condition_code": group})
        entries.add(f"FINAL_D15_FULL_{suffix}_N", int(full_counts[group]), "count_condition_source_units", [full_source], parameters={"condition_code": group})
    entries.add("FINAL_D15_MIN_CELL_N", int(clean_counts.min()), "minimum_clean_D15_condition_cell_size", [clean_source])

    clean_pool_cells = _cell_counts(clean_pool)
    clean_formal_cells = _cell_counts(clean_scope)
    full_pool_cells = _cell_counts(full_pool)
    full_formal_cells = _cell_counts(full_scope)
    require(sum(clean_formal_cells) == len(clean_scope), "clean formal cell counts do not close")
    require(sum(full_formal_cells) == len(full_scope), "full formal cell counts do not close")
    entries.add("FINAL_D15_CLEAN_POOL_CELL_N", _fmt_cell_counts(clean_pool_cells), "ordered_D15_condition_pool_counts_before_formal_gate", [clean_pool_source], parameters={"condition_order": list(GROUPS)})
    entries.add("FINAL_D15_CLEAN_FORMAL_CELL_N", _fmt_cell_counts(clean_formal_cells), "ordered_D15_condition_counts_after_formal_gate", [clean_source], parameters={"condition_order": list(GROUPS)})
    entries.add("FINAL_D15_FULL_POOL_CELL_N", _fmt_cell_counts(full_pool_cells), "ordered_D15_sensitivity_pool_counts_before_formal_gate", [full_pool_source], parameters={"condition_order": list(GROUPS)})
    entries.add("FINAL_D15_FULL_FORMAL_CELL_N", _fmt_cell_counts(full_formal_cells), "ordered_D15_sensitivity_counts_after_formal_gate", [full_source], parameters={"condition_order": list(GROUPS)})

    endpoint_masks = {
        prefix: _observed_numeric(clean_scope[field], f"clean D15 {field}")
        for prefix, field in ENDPOINTS.items()
    }
    endpoint_cells = {
        prefix: _cell_counts(clean_scope, mask=mask)
        for prefix, mask in endpoint_masks.items()
    }
    require(endpoint_cells["ROOT_WIDTH"] == endpoint_cells["ROOT_LENGTH"], "root width/axis-length cell denominators differ; one root-cell token is invalid")
    for token, prefix in (
        ("FINAL_D15_CLEAN_COUNT_CELL_N", "ABUNDANCE"),
        ("FINAL_D15_CLEAN_LENGTH_CELL_N", "LENGTH"),
        ("FINAL_D15_CLEAN_ROOT_CELL_N", "ROOT_WIDTH"),
        ("FINAL_D15_CLEAN_FIRST_HAIR_CELL_N", "FIRST_HAIR"),
    ):
        entries.add(token, _fmt_cell_counts(endpoint_cells[prefix]), "ordered_nonmissing_endpoint_counts_by_D15_condition", [clean_source], parameters={"endpoint": ENDPOINTS[prefix], "condition_order": list(GROUPS)})
    for token, prefix, digits in (
        ("FINAL_D15_RAW_ABUNDANCE_BY_CELL", "ABUNDANCE", 1),
        ("FINAL_D15_RAW_LENGTH_BY_CELL", "LENGTH", 1),
        ("FINAL_D15_RAW_FIRST_HAIR_BY_CELL", "FIRST_HAIR", 1),
        ("FINAL_D15_RAW_ROOT_WIDTH_BY_CELL", "ROOT_WIDTH", 1),
        ("FINAL_D15_RAW_ROOT_LENGTH_BY_CELL", "ROOT_LENGTH", 1),
    ):
        entries.add(
            token,
            _fmt_cell_median_iqr(
                clean_scope,
                field=ENDPOINTS[prefix],
                expected_counts=endpoint_cells[prefix],
                digits=digits,
            ),
            "ordered_source_unit_median_IQR_by_D15_condition",
            [clean_source],
            parameters={
                "endpoint": ENDPOINTS[prefix],
                "condition_order": list(GROUPS),
                "quantiles": [0.25, 0.5, 0.75],
                "null_values_excluded_not_zero_filled": True,
                "digits": digits,
            },
        )
    entries.add(
        "FINAL_D15_FIRST_HAIR_OBSERVABILITY_BY_CELL",
        _fmt_cell_fraction(endpoint_cells["FIRST_HAIR"], clean_formal_cells),
        "ordered_first_hair_observable_over_formal_counts_by_D15_condition",
        [clean_source],
        parameters={"endpoint": ENDPOINTS["FIRST_HAIR"], "condition_order": list(GROUPS)},
    )

    _require_columns(
        full_image,
        ("task_id", "visible_root_axis_length_um", "shootward_endpoint_border_visible"),
        "full_image_traits root visibility",
    )
    require(full_image["task_id"].astype(str).is_unique, "full image traits contain duplicate task IDs")
    visibility_raw = full_image[
        full_image["task_id"].astype(str).isin(clean_scope["task_id"].astype(str))
    ][["task_id", "visible_root_axis_length_um", "shootward_endpoint_border_visible"]]
    require(len(visibility_raw) == len(clean_scope), "clean D15 tasks are missing from full image traits")
    visibility_source = _selected_source(
        sources,
        namespace="figure_source_input",
        role="full_image_traits",
        selected=visibility_raw,
        columns=("task_id", "visible_root_axis_length_um", "shootward_endpoint_border_visible"),
        filters={"task_ids": "clean_formal_D15", "null_flags_excluded_from_denominator": True},
        sort_by=("task_id",),
    )
    visibility = clean_scope[["task_id", "condition_code", "visible_root_axis_length_um"]].merge(
        visibility_raw,
        on="task_id",
        how="left",
        validate="one_to_one",
        suffixes=("_traits", "_image"),
    )
    traits_visible = pd.to_numeric(visibility["visible_root_axis_length_um_traits"], errors="coerce")
    image_visible = pd.to_numeric(visibility["visible_root_axis_length_um_image"], errors="coerce")
    comparable = traits_visible.notna() & image_visible.notna()
    require(
        traits_visible.isna().equals(image_visible.isna())
        and np.allclose(
            traits_visible[comparable].to_numpy(),
            image_visible[comparable].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        ),
        "clean traits/full image visible-axis lengths differ",
    )
    flag_observed, border_visible = _nullable_bool(
        visibility["shootward_endpoint_border_visible"],
        "shootward endpoint border visibility",
    )
    evaluable = comparable & flag_observed
    censored = evaluable & ~border_visible
    censor_denominators = _cell_counts(visibility, mask=evaluable)
    censor_numerators = _cell_counts(visibility, mask=censored)
    entries.add(
        "FINAL_D15_VISIBLE_AXIS_CENSORING_BY_CELL",
        _fmt_cell_fraction(censor_numerators, censor_denominators),
        "ordered_shootward_edge_lower_bound_fraction_by_D15_condition",
        [clean_source, visibility_source],
        parameters={"censoring_definition": "shootward_endpoint_border_visible=false", "condition_order": list(GROUPS)},
    )

    endpoint_n: dict[str, int] = {}
    for prefix, endpoint in ENDPOINTS.items():
        rows = effects[(effects["cohort"].astype(str) == "primary_clean261") & (effects["endpoint_key"].astype(str) == endpoint)]
        values = set(pd.to_numeric(rows["endpoint_n"]).astype(int))
        require(len(rows) == 3 and len(values) == 1, f"{endpoint}: effect denominator mismatch")
        endpoint_n[prefix] = next(iter(values))
        require(sum(endpoint_cells[prefix]) == endpoint_n[prefix], f"{endpoint}: cell denominators do not sum to effect endpoint n")
    entries.add("FINAL_D15_CLEAN_LENGTH_N", endpoint_n["LENGTH"], "copy_fixed_endpoint_denominator", effect_cache[("LENGTH", "CONSTRUCT")][4])
    require(endpoint_n["ROOT_WIDTH"] == endpoint_n["ROOT_LENGTH"], "root width/length endpoint denominators differ but manuscript has one root N token")
    length_scoped = clean_scope[pd.to_numeric(clean_scope[ENDPOINTS["LENGTH"]], errors="coerce").notna()]
    length_hairs = int(pd.to_numeric(length_scoped["hair_length_measurement_hair_count"], errors="raise").sum())
    entries.add("FINAL_D15_CLEAN_LENGTH_HAIR_N", length_hairs, "sum_endpoint_complete_hairs_in_length_endpoint_source_units", [clean_source])

    same_direction = 0
    unstable: list[str] = []
    primary_effect_source_records: list[dict[str, Any]] = []
    for prefix, endpoint in ENDPOINTS.items():
        for effect_label, (resource_effect, _raw_effect) in EFFECTS.items():
            primary_value = effect_cache[(prefix, effect_label)][0]
            full_value = effect_cache[(prefix, effect_label)][5]
            primary_direction = _ratio_point_direction(primary_value)
            full_direction = _ratio_point_direction(full_value)
            same = primary_direction == full_direction
            same_direction += int(same)
            if not same:
                unstable.append(f"{endpoint}:{resource_effect}")
            primary_effect_source_records.extend(effect_cache[(prefix, effect_label)][4])
            primary_effect_source_records.extend(effect_cache[(prefix, effect_label)][8])
    direction_parameters = {
        "fixed_effect_family_n": 15,
        "comparison_basis": "clean_vs_Full283_ratio_point_estimate_direction",
        "direction_states": ["lower", "null", "higher"],
        "null": 1.0,
        "intervals_used": False,
    }
    entries.add(
        "FINAL_CLEAN_FULL_SAME_DIRECTION_N",
        same_direction,
        "count_clean_full_effects_with_same_ratio_point_estimate_direction",
        primary_effect_source_records,
        parameters=direction_parameters,
    )
    entries.add(
        "FINAL_CLEAN_FULL_UNSTABLE_EFFECTS",
        "; ".join(unstable) if unstable else "none of the fixed 15 effects",
        "list_clean_full_effect_direction_disagreements",
        primary_effect_source_records,
        parameters=direction_parameters,
    )
    entries.add(
        "FINAL_MULTITRAIT_ATLAS_SUMMARY",
        atlas_summary,
        "derive_condition_resolved_five_family_fingerprint_from_32_trait_atlas",
        [
            sources.json_cell(
                role="figure_resource:multitrait_atlas",
                source=atlas_source,
                pointer=("atlas_identity_sha256",),
            )
        ],
        parameters=atlas_parameters,
    )
    return effect_cache, narrative_decision


def _profile_pattern(frame: pd.DataFrame, metric: str) -> tuple[str, pd.DataFrame]:
    metric_labels = {
        "identity_abundance": "visible-hair abundance",
        "conditional_median_length_um": "conditional projected hair length",
        "length_support_fraction": "endpoint-complete length support",
    }
    require(metric in metric_labels, f"unknown distal profile metric: {metric}")
    selected = frame[(frame["cohort"].astype(str) == "primary_clean261") & (frame["metric_key"].astype(str) == metric)]
    require(len(selected) == 20 and set(selected["condition_code"].astype(str)) == set(GROUPS), f"{metric}: profile is not 4 conditions x 5 bins")
    for column in ("eligible_n", "length_supported_n", "bootstrap_repetitions"):
        values = pd.to_numeric(selected[column], errors="coerce")
        require(values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all(), f"{metric}: non-finite profile {column}")
    estimate = pd.to_numeric(selected["estimate"])
    low = pd.to_numeric(selected["ci_low"])
    high = pd.to_numeric(selected["ci_high"])
    eligible = pd.to_numeric(selected["eligible_n"]).astype(int)
    supported = pd.to_numeric(selected["length_supported_n"]).astype(int)
    triplet = np.column_stack((estimate, low, high))
    finite_triplet = np.isfinite(triplet)
    require(
        np.all(finite_triplet.all(axis=1) | (~finite_triplet).all(axis=1)),
        f"{metric}: incoherent structural missingness",
    )
    missing = ~finite_triplet.all(axis=1)
    if metric == "identity_abundance":
        require(not missing.any(), "identity abundance cannot be structurally missing")
    elif metric == "conditional_median_length_um":
        require(
            np.array_equal(missing, supported.to_numpy() == 0),
            "conditional length missingness differs from zero length support",
        )
    else:
        identity = frame[
            (frame["cohort"].astype(str) == "primary_clean261")
            & (frame["metric_key"].astype(str) == "identity_abundance")
        ].copy()
        zero_cells = {
            (str(row.condition_code), float(row.bin_start_mm))
            for row in identity.itertuples()
            if float(row.estimate) == 0.0
        }
        observed_missing = {
            (str(row.condition_code), float(row.bin_start_mm))
            for row in selected.loc[missing].itertuples()
        }
        require(
            observed_missing == zero_cells,
            "length-support missingness differs from zero-identity bins",
        )
    finite = ~missing
    require(
        ((low[finite] <= estimate[finite]) & (estimate[finite] <= high[finite])).all(),
        f"{metric}: estimate outside interval",
    )
    require((eligible > 0).all(), f"{metric}: profile bin lacks eligible source units")
    require(((supported >= 0) & (supported <= eligible)).all(), f"{metric}: length support exceeds eligibility")
    require((pd.to_numeric(selected["bootstrap_repetitions"]).astype(int) == BOOTSTRAP_REPETITIONS).all(), f"{metric}: bootstrap contract changed")
    require(set(selected["unit_of_analysis"].astype(str)) == {"one_source_image_root_unit"}, f"{metric}: unit of analysis changed")
    starts = pd.to_numeric(selected["bin_start_mm"], errors="coerce")
    ends = pd.to_numeric(selected["bin_end_mm"], errors="coerce")
    require(
        starts.notna().all()
        and ends.notna().all()
        and np.isfinite(starts.to_numpy(dtype=float)).all()
        and np.isfinite(ends.to_numpy(dtype=float)).all(),
        f"{metric}: profile bin coordinates are non-finite",
    )
    expected_bins = {(float(start), float(start + 1)) for start in range(5)}
    for group in GROUPS:
        group_rows = selected[selected["condition_code"].astype(str) == group]
        observed_bins = {
            (float(row.bin_start_mm), float(row.bin_end_mm))
            for row in group_rows.itertuples(index=False)
        }
        require(
            len(group_rows) == 5 and observed_bins == expected_bins,
            f"{metric}/{group}: prespecified [0,5) mm bins changed",
        )

    if metric == "length_support_fraction":
        require(
            ((estimate[finite] >= 0.0) & (estimate[finite] <= 1.0)).all()
            and ((low[finite] >= 0.0) & (high[finite] <= 1.0)).all(),
            "length_support_fraction: estimate/interval left [0,1]",
        )
    if metric == "conditional_median_length_um":
        require(
            (supported[finite] > 0).all(),
            "conditional length reported without a supported source unit",
        )

    def format_value(value: float) -> str:
        if metric == "length_support_fraction":
            return f"{_fmt_percent(value)}%"
        if metric == "conditional_median_length_um":
            return f"{_fmt(value, 1)} µm"
        return _fmt(value, 1)

    peak_items: list[str] = []
    trend_items: list[str] = []
    for group in GROUPS:
        group_rows = selected[selected["condition_code"].astype(str) == group].copy()
        group_rows = group_rows.sort_values("bin_start_mm", kind="stable")
        group_rows = group_rows[
            pd.to_numeric(group_rows["estimate"], errors="coerce").notna()
        ]
        require(len(group_rows) > 0, f"{metric}/{group}: no observable profile bin")
        values = pd.to_numeric(group_rows["estimate"], errors="raise").to_numpy(dtype=float)
        bin_starts = pd.to_numeric(group_rows["bin_start_mm"], errors="raise").to_numpy(dtype=float)
        bin_ends = pd.to_numeric(group_rows["bin_end_mm"], errors="raise").to_numpy(dtype=float)
        maximum = float(np.max(values))
        peak_indices = [
            index
            for index, value in enumerate(values)
            if math.isclose(float(value), maximum, rel_tol=1e-12, abs_tol=1e-12)
        ]
        peak_bins = [
            f"[{_fmt(bin_starts[index], 0)},{_fmt(bin_ends[index], 0)}) mm"
            for index in peak_indices
        ]
        peak_text = _join_readable(peak_bins)
        first = float(values[0])
        last = float(values[-1])
        if last > first and not math.isclose(last, first, rel_tol=1e-12, abs_tol=1e-12):
            trend = "rose"
        elif last < first and not math.isclose(last, first, rel_tol=1e-12, abs_tol=1e-12):
            trend = "fell"
        else:
            trend = "remained unchanged"
        support_range = ""
        if metric == "length_support_fraction":
            support_range = (
                f"; five-bin range {format_value(float(np.min(values)))}–"
                f"{format_value(float(np.max(values)))}"
            )
        peak_items.append(
            f"{SHORT_GROUP_LABELS[group]} {peak_text} ({format_value(maximum)})"
        )
        trend_items.append(
            f"{SHORT_GROUP_LABELS[group]} {format_value(first)}→{format_value(last)} "
            f"({trend}{support_range})"
        )
    pattern = (
        f"a condition-resolved source-unit {metric_labels[metric]} profile whose "
        "descriptive peak bins were "
        + "; ".join(peak_items)
        + ". First-to-last-observed-bin changes were "
        + "; ".join(trend_items)
    )
    require(
        pattern.count(". First-to-last-observed-bin changes were ") == 1,
        f"{metric}: profile narrative sentence contract changed",
    )
    return pattern, selected


def _derive_profiles(context: BuildContext, sources: Sources, entries: EntryBuilder) -> None:
    profiles = sources.table("figure_resource", "axial_profiles")
    columns = ("cohort", "condition_code", "bin_start_mm", "bin_end_mm", "metric_key", "estimate", "ci_low", "ci_high", "eligible_n", "length_supported_n", "bootstrap_repetitions", "unit_of_analysis")
    _require_columns(profiles, columns, "axial_profiles")
    patterns: dict[str, str] = {}
    bound: dict[str, dict[str, Any]] = {}
    for metric in ("identity_abundance", "conditional_median_length_um", "length_support_fraction"):
        pattern, selected = _profile_pattern(profiles, metric)
        patterns[metric] = pattern
        bound[metric] = _selected_source(sources, namespace="figure_resource", role="axial_profiles", selected=selected, columns=columns, filters={"cohort": "primary_clean261", "metric_key": metric}, sort_by=("condition_code", "bin_start_mm"))
    entries.add("FINAL_D15_AXIAL_ABUNDANCE_PATTERN", patterns["identity_abundance"], "derive_condition_specific_descriptive_peak_and_boundary_trend", [bound["identity_abundance"]], parameters={"metric": "identity_abundance", "condition_order": list(GROUPS), "profile_inference_performed": False})
    entries.add("FINAL_D15_AXIAL_LENGTH_PATTERN", patterns["conditional_median_length_um"], "derive_condition_specific_descriptive_peak_and_boundary_trend", [bound["conditional_median_length_um"]], parameters={"metric": "conditional_median_length_um", "condition_order": list(GROUPS), "conditional_on_endpoint_complete_support": True, "profile_inference_performed": False})
    entries.add("FINAL_D15_AXIAL_SUPPORT_PATTERN", patterns["length_support_fraction"], "derive_condition_specific_descriptive_peak_boundary_trend_and_range", [bound["length_support_fraction"]], parameters={"metric": "length_support_fraction", "condition_order": list(GROUPS), "profile_inference_performed": False})
    entries.add(
        "FINAL_D15_AXIAL_PATTERN",
        (
            "condition-resolved source-unit profiles located the descriptive abundance "
            "and conditional-length maxima and first-to-last-bin changes, while the "
            "support profile showed where endpoint-complete length evidence was available. "
            "These are spatial summaries across five fixed 1-mm bins, not tested "
            "between-condition trajectories or developmental-time curves"
        ),
        "combine_condition_resolved_descriptive_profile_contracts",
        [bound["identity_abundance"], bound["conditional_median_length_um"], bound["length_support_fraction"]],
        parameters={"condition_order": list(GROUPS), "profile_inference_performed": False, "developmental_time_interpretation_allowed": False},
    )
    abundance = profiles[(profiles["cohort"].astype(str) == "primary_clean261") & (profiles["metric_key"].astype(str) == "identity_abundance")]
    length = profiles[(profiles["cohort"].astype(str) == "primary_clean261") & (profiles["metric_key"].astype(str) == "conditional_median_length_um")]
    eligible_bins = int(pd.to_numeric(abundance["eligible_n"], errors="raise").sum())
    length_observations = int(pd.to_numeric(length["length_supported_n"], errors="raise").sum())
    entries.add("FINAL_PROFILE_CLEAN_ELIGIBLE_BIN_N", eligible_bins, "sum_source_unit_bin_eligibility", [bound["identity_abundance"]])
    entries.add("FINAL_PROFILE_CLEAN_LENGTH_OBSERVATION_N", length_observations, "sum_source_unit_bins_with_length_observations", [bound["conditional_median_length_um"]])
    receipt = context.evidence_artifacts["profiles"]
    total = _integer(receipt.payload.get("locked_1_4mm_trait_crosscheck_tasks"), "profile crosscheck total")
    mismatches = _integer(receipt.payload.get("locked_1_4mm_trait_crosscheck_mismatches"), "profile crosscheck mismatches")
    require(0 <= mismatches <= total, "profile crosscheck count impossible")
    require(mismatches == 0, "profile-to-trait [1,4) mm crosscheck has one or more mismatches")
    total_source = sources.json_cell(role="evidence_artifact:profiles", source=receipt, pointer=("locked_1_4mm_trait_crosscheck_tasks",))
    mismatch_source = sources.json_cell(role="evidence_artifact:profiles", source=receipt, pointer=("locked_1_4mm_trait_crosscheck_mismatches",))
    entries.add("FINAL_PROFILE_CROSSCHECK_TOTAL_N", total, "copy_profile_trait_crosscheck_total", [total_source])
    entries.add("FINAL_PROFILE_CROSSCHECK_MISMATCH_N", mismatches, "copy_profile_trait_crosscheck_mismatches", [mismatch_source])
    entries.add("FINAL_PROFILE_CROSSCHECK_MATCH_N", total - mismatches, "profile_trait_crosscheck_complement", [total_source, mismatch_source])


def _hardware_text(hardware: Mapping[str, Any]) -> str:
    gpus = hardware.get("gpus")
    names: list[str] = []
    if isinstance(gpus, list):
        for item in gpus:
            if isinstance(item, Mapping) and isinstance(item.get("name"), str):
                names.append(str(item["name"]))
    if not names and isinstance(hardware.get("gpu_names"), list):
        names = [str(item) for item in hardware["gpu_names"]]
    require(names, "benchmark hardware has no GPU names")
    host = str(hardware.get("host", "recorded host"))
    processor = str(hardware.get("processor", "recorded CPU")).strip() or "recorded CPU"
    return f"{host}; {processor}; {', '.join(names)}"


def _latency_mode_label(mode: str) -> str:
    labels = {
        "sequential_persistent_full283": (
            "sequential persistent-process full-workflow latency "
            "(283 images; process startup excluded per image)"
        ),
        "sequential_cold_cli_full283": (
            "sequential cold-CLI full-workflow latency "
            "(283 images; process startup included per image)"
        ),
    }
    require(mode in labels, "latency mode invalid")
    return labels[mode]


def _derive_runtime(context: BuildContext, sources: Sources, entries: EntryBuilder) -> None:
    runtime_source = sources.json_resource("runtime_summary")
    payload = runtime_source.payload
    require(payload.get("schema_version") == "PHAxis-manuscript-two-mode-runtime-input-1.0" and payload.get("status") == "completed_two_mode_direct_full283", "runtime summary is not final two-mode full283")
    require(payload.get("measurement_scope") == "raw_image_to_final_traits_and_profiles_direct", "runtime scope changed")
    latency = payload.get("sequential_latency_full283")
    production = payload.get("production_batch_full283")
    baseline_latency = payload.get("baseline_sequential_latency_full283")
    baseline_production = payload.get("baseline_production_batch_full283")
    latency_comparison = payload.get("latency_comparison")
    production_comparison = payload.get("production_comparison")
    require(
        all(
            isinstance(record, Mapping)
            for record in (
                latency,
                production,
                baseline_latency,
                baseline_production,
                latency_comparison,
                production_comparison,
            )
        ),
        "runtime modes or frozen-v1 comparison blocks missing",
    )
    require(latency.get("images") == production.get("images") == 283, "runtime modes are not exact283")
    latency_mode = str(latency.get("benchmark_mode"))
    require(latency_mode in {"sequential_persistent_full283", "sequential_cold_cli_full283"}, "latency mode invalid")
    require(payload.get("latency_mode") == latency_mode, "runtime latency mode is inconsistent")
    require(production.get("benchmark_mode") == "production_batch_full283" and production.get("per_image_latency_reported") is False, "production runtime mode invalid")
    for record in (latency, production):
        require(record.get("includes_io") is True and record.get("includes_preprocess") is True and record.get("includes_stitching_fusion_traits_profiles") is True and record.get("fresh_direct_run") is True and record.get("resume_or_cache_used") is False, "runtime excludes scope or uses cache")
    require(
        baseline_latency.get("images") == baseline_production.get("images") == 283
        and baseline_latency.get("benchmark_mode") == latency_mode
        and baseline_production.get("benchmark_mode") == "production_batch_full283"
        and baseline_production.get("per_image_latency_reported") is False,
        "frozen-v1 runtime modes are not matching direct exact283 modes",
    )
    for record in (baseline_latency, baseline_production):
        require(
            record.get("measurement_scope") == payload.get("measurement_scope")
            and record.get("includes_io") is True
            and record.get("includes_preprocess") is True
            and record.get("includes_stitching_fusion_traits_profiles") is True
            and record.get("fresh_direct_run") is True
            and record.get("resume_or_cache_used") is False,
            "frozen-v1 runtime excludes scope or uses cache",
        )
    for label, comparison, expected_mode in (
        ("latency", latency_comparison, latency_mode),
        ("production", production_comparison, "production_batch_full283"),
    ):
        require(
            comparison.get("status") == "comparable_direct_full283"
            and comparison.get("comparable") is True
            and comparison.get("noncomparability_reasons") == []
            and comparison.get("benchmark_mode") == expected_mode
            and comparison.get("measurement_scope") == payload.get("measurement_scope")
            and comparison.get("same_283_source_manifest_hardware_and_io_scope") is True
            and comparison.get("historical_component_runtime_used_as_full_baseline") is False,
            f"{label} comparison is not comparable_direct_full283",
        )
    per_image = sources.table("figure_resource", "runtime_per_image")
    per_columns = ("source_unit", "wall_seconds", "megapixels", "io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds")
    _require_columns(per_image, per_columns, "runtime_per_image")
    require(len(per_image) == per_image["source_unit"].nunique() == 283, "runtime per-image table is not 283 unique sources")
    numeric = per_image[list(per_columns[1:])].apply(pd.to_numeric, errors="coerce")
    require(np.isfinite(numeric.to_numpy()).all() and (numeric >= 0).all().all(), "runtime per-image table contains invalid timing")
    component_sum = numeric[["io_seconds", "preprocess_seconds", "inference_seconds", "postprocess_seconds"]].sum(axis=1)
    require((component_sum <= numeric["wall_seconds"] * 1.02 + 1e-12).all(), "runtime components exceed direct wall")
    median = float(numeric["wall_seconds"].median())
    p95 = float(numeric["wall_seconds"].quantile(0.95))
    require(math.isclose(median, _finite(latency.get("median_seconds_per_image"), "runtime median"), abs_tol=1e-12, rel_tol=0), "runtime median differs from per-image table")
    require(math.isclose(p95, _finite(latency.get("p95_seconds_per_image"), "runtime p95"), abs_tol=1e-12, rel_tol=0), "runtime P95 differs from per-image table")
    table_source = _selected_source(sources, namespace="figure_resource", role="runtime_per_image", selected=per_image, columns=per_columns, filters={"source_units": 283, "mode": latency["benchmark_mode"]}, sort_by=("source_unit",))
    runtime_cell = lambda pointer: sources.json_cell(role="figure_resource:runtime_summary", source=runtime_source, pointer=pointer)
    entries.add(
        "FINAL_BENCHMARK_LATENCY_MODE_LABEL",
        _latency_mode_label(latency_mode),
        "map_single_sealed_latency_mode_to_human_readable_label",
        [runtime_cell(("latency_mode",))],
    )
    entries.add("FINAL_E2E_MEDIAN_IMAGE_S", _fmt(median, 2), "median_direct_per_source_wall", [table_source])
    entries.add("FINAL_E2E_P95_IMAGE_S", _fmt(p95, 2), "p95_direct_per_source_wall", [table_source])
    wall = _finite(production.get("batch_wall_seconds"), "production batch wall")
    baseline_wall = _finite(
        baseline_production.get("batch_wall_seconds"),
        "frozen-v1 production batch wall",
    )
    baseline_median = _finite(
        baseline_latency.get("median_seconds_per_image"),
        "frozen-v1 runtime median",
    )
    require(
        wall > 0 and median > 0 and baseline_wall > 0 and baseline_median > 0,
        "runtime and frozen-v1 base timings must be positive",
    )
    batch_speedup = baseline_wall / wall
    latency_speedup = baseline_median / median
    declared_batch_speedup = _finite(
        production_comparison.get("batch_wall_speedup_frozen_v1_over_phaxis"),
        "production comparison speedup",
    )
    declared_latency_speedup = _finite(
        latency_comparison.get("median_latency_speedup_frozen_v1_over_phaxis"),
        "latency comparison speedup",
    )
    require(
        declared_batch_speedup > 0
        and math.isclose(
            declared_batch_speedup,
            batch_speedup,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "production comparison speedup differs from frozen-v1/PHAxis base walls",
    )
    require(
        declared_latency_speedup > 0
        and math.isclose(
            declared_latency_speedup,
            latency_speedup,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "latency comparison speedup differs from frozen-v1/PHAxis base medians",
    )
    entries.add("FINAL_E2E_TOTAL_MIN", _fmt(wall / 60.0, 2), "convert_direct_batch_wall_seconds_to_minutes", [runtime_cell(("production_batch_full283", "batch_wall_seconds"))])
    entries.add(
        "FINAL_E2E_FROZEN_V1_BATCH_TOTAL_MIN",
        _fmt(baseline_wall / 60.0, 2),
        "convert_frozen_v1_direct_batch_wall_seconds_to_minutes",
        [runtime_cell(("baseline_production_batch_full283", "batch_wall_seconds"))],
    )
    entries.add(
        "FINAL_E2E_BATCH_SPEEDUP_FROZEN_V1_OVER_PHAXIS",
        _fmt(batch_speedup, 2),
        "recompute_frozen_v1_over_phaxis_batch_wall_speedup",
        [
            runtime_cell(("production_batch_full283", "batch_wall_seconds")),
            runtime_cell(("baseline_production_batch_full283", "batch_wall_seconds")),
            runtime_cell(("production_comparison", "batch_wall_speedup_frozen_v1_over_phaxis")),
        ],
    )
    entries.add(
        "FINAL_E2E_FROZEN_V1_MEDIAN_IMAGE_S",
        _fmt(baseline_median, 2),
        "copy_frozen_v1_direct_median_per_source_wall",
        [runtime_cell(("baseline_sequential_latency_full283", "median_seconds_per_image"))],
    )
    entries.add(
        "FINAL_E2E_MEDIAN_LATENCY_SPEEDUP_FROZEN_V1_OVER_PHAXIS",
        _fmt(latency_speedup, 2),
        "recompute_frozen_v1_over_phaxis_median_latency_speedup",
        [
            runtime_cell(("sequential_latency_full283", "median_seconds_per_image")),
            runtime_cell(("baseline_sequential_latency_full283", "median_seconds_per_image")),
            runtime_cell(("latency_comparison", "median_latency_speedup_frozen_v1_over_phaxis")),
        ],
    )
    entries.add("FINAL_E2E_IMAGES_PER_MIN", _fmt(_finite(production.get("images_per_min"), "images/min"), 2), "copy_direct_batch_throughput", [runtime_cell(("production_batch_full283", "images_per_min"))])
    entries.add("FINAL_E2E_MP_PER_S", _fmt(_finite(production.get("megapixels_per_second"), "MP/s"), 2), "copy_direct_batch_pixel_throughput", [runtime_cell(("production_batch_full283", "megapixels_per_second"))])
    peak = max(_finite(latency.get("peak_vram_mib"), "latency peak VRAM"), _finite(production.get("peak_vram_mib"), "production peak VRAM"))
    entries.add("FINAL_E2E_PEAK_GPU_GB", _fmt(peak / 1024.0, 2), "maximum_observed_peak_vram_mib_to_gib", [runtime_cell(("sequential_latency_full283", "peak_vram_mib")), runtime_cell(("production_batch_full283", "peak_vram_mib"))])
    hardware = production.get("hardware")
    require(isinstance(hardware, Mapping) and hardware == latency.get("hardware"), "runtime modes used different hardware")
    hardware_text = _hardware_text(hardware)
    entries.add("FINAL_BENCHMARK_HARDWARE", hardware_text, "format_sealed_runtime_hardware", [runtime_cell(("production_batch_full283", "hardware"))])
    stages = production.get("stage_timings")
    require(isinstance(stages, list), "production stage timings missing")
    by_stage = {str(row.get("stage")): _finite(row.get("wall_seconds"), "stage wall") for row in stages if isinstance(row, Mapping)}
    require(set(by_stage) == {"root_provider", "stageb_train399", "fusion", "traits", "distal_axis_profiles"}, "production stage set changed")
    assembly = by_stage["fusion"] + by_stage["traits"] + by_stage["distal_axis_profiles"]
    for token, value, names in (
        ("FINAL_RUNTIME_ROOT_PERCENT", by_stage["root_provider"], ["root_provider"]),
        ("FINAL_RUNTIME_STAGEB_PERCENT", by_stage["stageb_train399"], ["stageb_train399"]),
        ("FINAL_RUNTIME_ASSEMBLY_PERCENT", assembly, ["fusion", "traits", "distal_axis_profiles"]),
    ):
        entries.add(token, _fmt_percent(value / wall), "sum_named_nonoverlapping_stage_walls_over_batch_wall", [runtime_cell(("production_batch_full283", "stage_timings")), runtime_cell(("production_batch_full283", "batch_wall_seconds"))], parameters={"stages": names})


def _derive_narratives(
    entries: EntryBuilder,
    sources: Sources,
    effect_cache: Mapping[tuple[str, str], tuple[Any, ...]],
    decision: Mapping[str, Any],
) -> None:
    """Compose a plant-facing Discussion synthesis in a fixed layer order."""

    axial = entries.entries["FINAL_D15_AXIAL_PATTERN"]
    endpoint_entries = {
        "ABUNDANCE": entries.entries["FINAL_D15_ABUNDANCE_PATTERN"],
        "LENGTH": entries.entries["FINAL_D15_LENGTH_PATTERN"],
        "FIRST_HAIR": entries.entries["FINAL_FIRST_HAIR_PATTERN"],
        "ROOT_WIDTH": entries.entries["FINAL_ROOT_WIDTH_PATTERN"],
        "ROOT_LENGTH": entries.entries["FINAL_ROOT_LENGTH_PATTERN"],
    }
    biology_sources = [
        *axial["derivation"]["sources"],
        *entries.entries["FINAL_D15_ABSTRACT_SYNTHESIS"]["derivation"]["sources"],
        *[
            source
            for prefix in ENDPOINTS
            for source in endpoint_entries[prefix]["derivation"]["sources"]
        ],
    ]
    endpoint_labels = {
        "ABUNDANCE": "local visible-hair abundance",
        "LENGTH": "conditional projected hair length",
        "FIRST_HAIR": "first observed ≥40-µm hair position",
        "ROOT_WIDTH": "apparent primary-root width",
        "ROOT_LENGTH": "visible primary-root extent",
    }
    selected_decision = validate_narrative_decision(decision)
    endpoint_label_by_key = {value: key for key, value in ENDPOINTS.items()}
    effect_label_by_key = {value[0]: key for key, value in EFFECTS.items()}
    supported: list[tuple[str, str, str]] = [
        (
            endpoint_label_by_key[str(cell["endpoint_key"])],
            effect_label_by_key[str(cell["effect_key"])],
            str(cell["clean_direction"]),
        )
        for cell in selected_decision["cells"]
        if cell["headline_supported"]
    ]

    def layer_headlines(layer_endpoints: Sequence[str]) -> list[str]:
        grouped: dict[tuple[str, str], list[str]] = {}
        for endpoint_label, effect_label, direction in supported:
            if endpoint_label not in layer_endpoints:
                continue
            grouped.setdefault((effect_label, direction), []).append(
                endpoint_labels[endpoint_label]
            )
        result: list[str] = []
        for effect_label in EFFECTS:
            for direction in ("higher", "lower"):
                labels = grouped.get((effect_label, direction))
                if not labels:
                    continue
                subjects = _join_readable(labels)
                plural = len(labels) > 1
                if effect_label == "CONSTRUCT":
                    comparison = (
                        f"{direction} in OE-labelled than EV source units when averaged "
                        "equally across temperatures"
                    )
                elif effect_label == "TEMPERATURE":
                    comparison = (
                        f"{direction} at 30°C than 22°C when averaged equally "
                        "across construct labels"
                    )
                else:
                    side = "above" if direction == "higher" else "below"
                    comparison = (
                        "associated with a construct-by-temperature ratio of ratios "
                        f"{side} one"
                    )
                result.append(
                    f"{subjects} {'were' if plural else 'was'} {comparison}"
                )
        return result

    primary_headlines = layer_headlines(
        BIOLOGICAL_NARRATIVE_LAYERS["primary_hair_change"]
    )
    spatial_headlines = layer_headlines(
        BIOLOGICAL_NARRATIVE_LAYERS["spatial_location"]
    )
    root_headlines = layer_headlines(
        BIOLOGICAL_NARRATIVE_LAYERS["supporting_root_context"]
    )

    primary_summary = (
        ". ".join(primary_headlines)
        if primary_headlines
        else (
            "local visible-hair abundance and conditional projected length remained "
            "descriptive across the archived contrasts"
        )
    )
    spatial_summary = (
        ". ".join(spatial_headlines)
        if spatial_headlines
        else (
            "the elongation-qualified first-hair position remained descriptive across "
            "the archived contrasts"
        )
    )
    root_summary = (
        ". ".join(root_headlines)
        if root_headlines
        else (
            "apparent primary-root width and visible primary-root extent remained "
            "descriptive across the archived contrasts"
        )
    )
    branch_statement = {
        "A": "D15 resolved layer-specific phenotype associations across the root–hair interface",
        "B": "D15 resolved same-effect, endpoint-specific phenotype associations across hair and carrying-root layers",
        "C": (
            "D15 mapped five complementary dimensions of the root–hair interface without "
            "a single dominant condition-associated signature"
        ),
    }[selected_decision["branch_id"]]
    synthesis = (
        f"{branch_statement}. In the primary hair-change layer, {primary_summary}. "
        f"Along the distal axis, {spatial_summary}. Across the three distal profiles, "
        f"{axial['value']}. "
        "The first-hair event frequency remained an observability descriptor, and only "
        "its positive conditional distance entered the fixed models. "
        f"In carrying-root context, {root_summary}. "
        "Associations highlighted above combined a clean-cohort interval excluding the "
        "no-difference ratio with a Full283 estimate in the same direction; all other "
        "contrasts remain available as descriptive or sensitivity results. Root-context "
        "interpretation followed extent, caliber, taper, and trajectory rather than a "
        "count or directional vote across correlated descriptors"
    )
    headlines = [*primary_headlines, *spatial_headlines, *root_headlines]
    entries.add(
        "FINAL_DISCUSSION_BIOLOGICAL_SYNTHESIS",
        synthesis,
        "compose_fixed_family_biological_synthesis_with_interval_and_sensitivity_gates",
        biology_sources,
        parameters={
            "fixed_endpoint_n": 5,
            "fixed_effect_n": 15,
            "narrative_decision_identity_sha256": selected_decision[
                "narrative_decision_identity_sha256"
            ],
            "narrative_branch_id": selected_decision["branch_id"],
            "support_mask_bits": selected_decision["support_mask_bits"],
            "supported_association_n": len(supported),
            "headline_group_n": len(headlines),
            "narrative_layer_order": list(BIOLOGICAL_NARRATIVE_LAYERS),
            "supported_association_n_by_layer": {
                "primary_hair_change": len(primary_headlines),
                "spatial_location": len(spatial_headlines),
                "supporting_root_context": len(root_headlines),
            },
            "profile_inference_performed": False,
            "first_hair_observability_mode": "descriptive_only",
            "modeled_first_hair_component": "conditional_positive_distance_only",
            "root_context_descriptor_n": 19,
            "correlated_descriptor_vote_used": False,
            "correlated_descriptor_count_used_as_biological_conclusion": False,
        },
    )


def derive_entries(context: BuildContext, token_contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive every current master token from mutually sealed sources."""

    wt_secondary = validate_wt_secondary_source_inputs(context)
    require(
        wt_secondary["D15_fixed_effect_rows"] == 15
        and wt_secondary["D15_narrative_branch_changed"] is False
        and wt_secondary["FINAL_WT_tokens_created"] is False,
        "WT secondary values integration escaped its supplementary-only family",
    )
    sources = Sources(context)
    entries = EntryBuilder(token_contract)
    _derive_human(context, sources, entries)
    _derive_release(context, sources, entries)
    _derive_development(context, sources, entries)
    _derive_assurance(context, sources, entries)
    effect_cache, narrative_decision = _derive_biology(context, sources, entries)
    _derive_profiles(context, sources, entries)
    _derive_runtime(context, sources, entries)
    _derive_narratives(entries, sources, effect_cache, narrative_decision)
    missing = sorted(set(token_contract["tokens"]) - set(entries.entries))
    extra = sorted(set(entries.entries) - set(token_contract["tokens"]))
    require(not missing and not extra, f"token derivation coverage mismatch; missing={missing}, extra={extra}")
    for token in entries.entries:
        if token.startswith("FINAL_"):
            authorities = {
                source["authority_class"]
                for source in entries.entries[token]["derivation"]["sources"]
            }
            require("historical_development_comparator" not in authorities, f"{token}: historical source leaked into FINAL token")
    _require_builder_abstract_within_limit(context.master_text, entries.entries)
    return entries.entries


def build_values(
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
    source_release_manifest: str | Path | None = None,
    output: str | Path,
) -> dict[str, Any]:
    if not Path(human_metadata).resolve().is_file():
        raise HumanMetadataError(
            "human metadata file is missing",
            missing=sorted(HUMAN_METADATA_TOKENS),
        )
    require(
        source_release_manifest is not None,
        "final manuscript values require --source-release-manifest",
    )
    context = load_build_context(
        master=master,
        evidence_graph=evidence_graph,
        evidence_artifacts=evidence_artifacts,
        figure_inputs=figure_inputs,
        figure_assembly_summary=figure_assembly_summary,
        model_contract_proposal=model_contract_proposal,
        human_metadata=human_metadata,
        model_bundle_manifest=model_bundle_manifest,
        clean_install_receipt=clean_install_receipt,
        source_release_manifest=source_release_manifest,
    )
    token_contract = build_token_source_contract(context.master_text)
    entries = derive_entries(context, token_contract)
    payload = assemble_values_payload(context=context, token_contract=token_contract, entries=entries)
    publish_json_no_overwrite(output, payload)
    return payload


def _artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--evidence-artifact requires ROLE=PATH")
    role, raw = value.split("=", 1)
    if role not in EVIDENCE_ARTIFACT_ROLES or not raw:
        raise argparse.ArgumentTypeError(f"invalid evidence artifact role/path: {value}")
    return role, Path(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--evidence-graph", required=True, type=Path)
    parser.add_argument("--evidence-artifact", action="append", required=True, type=_artifact, metavar="ROLE=PATH")
    parser.add_argument("--figure-inputs", required=True, type=Path)
    parser.add_argument("--figure-assembly-summary", required=True, type=Path)
    parser.add_argument("--model-contract-proposal", required=True, type=Path)
    parser.add_argument("--human-metadata", required=True, type=Path)
    parser.add_argument("--model-bundle-manifest", required=True, type=Path)
    parser.add_argument("--clean-install-receipt", required=True, type=Path)
    parser.add_argument(
        "--source-release-manifest",
        type=Path,
        help=(
            "formal SOURCE_MANIFEST.json; optional at argument parsing but required "
            "for the final values status"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--missing-human-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    artifact_pairs = list(args.evidence_artifact)
    artifacts = dict(artifact_pairs)
    if len(artifacts) != len(artifact_pairs) or set(artifacts) != set(EVIDENCE_ARTIFACT_ROLES):
        parser.error(f"--evidence-artifact roles must be exact: {list(EVIDENCE_ARTIFACT_ROLES)}")
    try:
        payload = build_values(
            master=args.master,
            evidence_graph=args.evidence_graph,
            evidence_artifacts=artifacts,
            figure_inputs=args.figure_inputs,
            figure_assembly_summary=args.figure_assembly_summary,
            model_contract_proposal=args.model_contract_proposal,
            human_metadata=args.human_metadata,
            model_bundle_manifest=args.model_bundle_manifest,
            clean_install_receipt=args.clean_install_receipt,
            source_release_manifest=args.source_release_manifest,
            output=args.output,
        )
    except HumanMetadataError as error:
        report_path = args.missing_human_report or Path(str(args.output) + ".missing-human-metadata.json")
        publish_json_no_overwrite(report_path, human_metadata_report(error))
        print(f"human metadata blocked; report: {report_path}", file=sys.stderr)
        return 2
    except ManuscriptValuesError as error:
        print(f"manuscript values build refused: {error}", file=sys.stderr)
        return 2
    print(payload["values_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
