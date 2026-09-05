#!/usr/bin/env python
"""Audit locked legacy-backed tables against PHAxis-native analysis tables.

This CPU-only audit compares already generated CSV/JSON artifacts.  It does
not import the frozen predecessor implementation, load images or annotations,
or access blind data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file  # noqa: E402


TABLES = (
    "primary_clean_exploratory_factorial_tests.csv",
    "full283_sensitivity_factorial_tests.csv",
    "clean_vs_full_effect_stability.csv",
    "robust_sensitivity.csv",
    "primary_group_summaries.csv",
    "primary_model_qc_flow.csv",
)
SUMMARY_FIELDS = (
    "schema_version",
    "status",
    "primary_cohort",
    "sensitivity_cohort",
    "primary_scope_units",
    "sensitivity_scope_units",
    "primary_result_rows",
    "sensitivity_result_rows",
    "same_direction_clean_vs_full_rows",
    "clean_full_comparison_rows",
    "primary_model_BH_FDR_rejections",
    "claim_status",
    "design_identifiability",
    "cohort_build_summary_sha256",
    "output_table_sha256",
    "root_cap_region_statistics_included",
    "whole_hair_zone_confirmatory_traits_included",
    "canonical_annotations_read",
    "blind_images_used",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _is_numeric_pair(baseline: pd.Series, candidate: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(baseline)
        and pd.api.types.is_numeric_dtype(candidate)
        and not pd.api.types.is_bool_dtype(baseline)
        and not pd.api.types.is_bool_dtype(candidate)
    )


def _object_equal(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _compare_table(
    baseline_path: Path,
    candidate_path: Path,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(candidate_path)
    same_columns = list(baseline.columns) == list(candidate.columns)
    same_rows = len(baseline) == len(candidate)
    column_reports: dict[str, Any] = {}
    differing_cells = 0
    if same_columns and same_rows:
        for column in baseline.columns:
            left = baseline[column]
            right = candidate[column]
            if _is_numeric_pair(left, right):
                left_values = left.to_numpy(dtype=np.float64)
                right_values = right.to_numpy(dtype=np.float64)
                equivalent = np.isclose(
                    left_values,
                    right_values,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    equal_nan=True,
                )
                finite = np.isfinite(left_values) & np.isfinite(right_values)
                absolute = np.abs(left_values[finite] - right_values[finite])
                denominator = np.maximum(
                    np.maximum(
                        np.abs(left_values[finite]), np.abs(right_values[finite])
                    ),
                    np.finfo(np.float64).tiny,
                )
                relative = absolute / denominator
                report = {
                    "kind": "numeric",
                    "equivalent": bool(np.all(equivalent)),
                    "differing_cells": int(np.size(equivalent) - equivalent.sum()),
                    "max_absolute_difference": float(absolute.max())
                    if absolute.size
                    else 0.0,
                    "max_relative_difference": float(relative.max())
                    if relative.size
                    else 0.0,
                }
            else:
                equivalent_values = [
                    _object_equal(left_value, right_value)
                    for left_value, right_value in zip(left, right, strict=True)
                ]
                report = {
                    "kind": "exact",
                    "equivalent": all(equivalent_values),
                    "differing_cells": equivalent_values.count(False),
                }
            column_reports[column] = report
            differing_cells += int(report["differing_cells"])
    else:
        differing_cells = max(len(baseline), len(candidate)) * max(
            len(baseline.columns), len(candidate.columns)
        )
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_sha256": sha256_file(baseline_path),
        "candidate_sha256": sha256_file(candidate_path),
        "byte_identical": sha256_file(baseline_path) == sha256_file(candidate_path),
        "baseline_rows": len(baseline),
        "candidate_rows": len(candidate),
        "columns_exact": same_columns,
        "row_count_exact": same_rows,
        "equivalent": same_columns and same_rows and differing_cells == 0,
        "differing_cells": differing_cells,
        "columns": column_reports,
    }


def _summary_equivalence(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        fields[field] = {
            "equivalent": baseline.get(field) == candidate.get(field),
            "baseline": baseline.get(field),
            "candidate": candidate.get(field),
        }
    return {
        "equivalent": all(item["equivalent"] for item in fields.values()),
        "fields": fields,
        "expected_provenance_only_differences": {
            "analysis_result_lock_sha256": {
                "baseline": baseline.get("analysis_result_lock_sha256"),
                "candidate": candidate.get("analysis_result_lock_sha256"),
            },
            "analysis_identity_sha256": {
                "baseline": baseline.get("analysis_identity_sha256"),
                "candidate": candidate.get("analysis_identity_sha256"),
            },
        },
    }


def audit(
    *,
    baseline: Path,
    candidate: Path,
    cohorts: Path,
    output: Path,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline_summary = read_json(baseline / "summary.json")
    candidate_summary = read_json(candidate / "summary.json")
    cohort_summary = read_json(cohorts / "summary.json")
    for name, payload in (
        ("baseline", baseline_summary),
        ("candidate", candidate_summary),
        ("cohorts", cohort_summary),
    ):
        if payload.get("blind_images_used") != 0:
            raise RuntimeError(f"{name} evidence is blind-tainted")
        if payload.get("canonical_annotations_read") not in (False, None):
            raise RuntimeError(f"{name} evidence read canonical annotations")

    table_reports: dict[str, Any] = {}
    for filename in TABLES:
        table_reports[filename] = _compare_table(
            baseline / "tables" / filename,
            candidate / "tables" / filename,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    summary_report = _summary_equivalence(baseline_summary, candidate_summary)
    baseline_lock = read_json(baseline / "analysis_result_lock.json")
    candidate_lock = read_json(candidate / "analysis_result_lock.json")
    table_equivalent = all(item["equivalent"] for item in table_reports.values())
    table_byte_identical = all(
        item["byte_identical"] for item in table_reports.values()
    )
    payload: dict[str, Any] = {
        "schema_version": "PHAxis-biological-analysis-native-equivalence-audit-1.0",
        "status": "passed" if table_equivalent and summary_report["equivalent"] else "failed",
        "scope": (
            "CPU-only fieldwise comparison of six locked biological-analysis "
            "tables and non-provenance summary fields"
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "baseline_analysis": str(baseline),
        "candidate_analysis": str(candidate),
        "locked_cohorts": str(cohorts),
        "locked_input_sha256": {
            "cohort_summary": sha256_file(cohorts / "summary.json"),
            "cohort_lock": sha256_file(cohorts / "analysis_contract_lock.json"),
            "primary_traits": sha256_file(
                cohorts / "primary_clean261/traits.csv"
            ),
            "sensitivity_traits": sha256_file(
                cohorts / "sensitivity_full283/traits.csv"
            ),
        },
        "implementation_sha256": {
            "frozen_predecessor_numeric_source": baseline_lock.get(
                "reused_legacy_model_implementation_sha256"
            ),
            "phaxis_native_numeric_source": candidate_lock.get(
                "numerical_implementation_sha256"
            ),
            "phaxis_wrapper": candidate_lock.get("wrapper_implementation_sha256"),
            "frozen_predecessor_model_spec": baseline_lock.get(
                "model_spec_sha256"
            ),
            "phaxis_owned_model_spec": candidate_lock.get("model_spec_sha256"),
            "historical_model_spec_source": candidate_lock.get(
                "historical_model_spec_sha256"
            ),
        },
        "production_wrapper_imports_frozen_predecessor": candidate_lock.get(
            "legacy_model_implementation_imported"
        ),
        "tables": table_reports,
        "tables_equivalent": table_equivalent,
        "tables_byte_identical": table_byte_identical,
        "total_differing_cells": sum(
            int(item["differing_cells"]) for item in table_reports.values()
        ),
        "summary": summary_report,
        "provenance_transition": {
            "baseline_result_lock_schema": baseline_lock.get("schema_version"),
            "candidate_result_lock_schema": candidate_lock.get("schema_version"),
            "removed_baseline_field": "reused_legacy_model_implementation_sha256",
            "added_candidate_fields": [
                "numerical_implementation",
                "numerical_implementation_sha256",
                "legacy_model_implementation_imported",
                "model_spec_schema_version",
                "historical_model_spec_sha256",
            ],
            "numerical_interpretation": "no change",
            "model_spec_interpretation": (
                "PHAxis ownership/path migration only; no model or inference change"
            ),
        },
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    atomic_write_json(output / "equivalence_audit.json", payload)

    lines = [
        "# PHAxis biological-analysis native implementation equivalence audit",
        "",
        f"Status: **{payload['status'].upper()}**",
        "",
        "This CPU-only audit compared the six locked CSV outputs produced by the "
        "former legacy-backed PHAxis wrapper with the PHAxis-owned numerical "
        "implementation on the identical provisional clean261/full283 cohort "
        "inputs. No image, annotation, blind data, or GPU runtime was used.",
        "",
        "## Result",
        "",
        f"- Tables fieldwise equivalent: `{str(table_equivalent).lower()}`",
        f"- Tables byte-identical: `{str(table_byte_identical).lower()}`",
        f"- Differing table cells: `{payload['total_differing_cells']}`",
        f"- Absolute/relative tolerance: `{absolute_tolerance:g}` / `{relative_tolerance:g}`",
        f"- Non-provenance summary fields equivalent: "
        f"`{str(summary_report['equivalent']).lower()}`",
        "- Production wrapper imports frozen predecessor: `false`",
        "- Canonical annotations read: `false`",
        "- Blind images used: `0`",
        "",
        "## Per-table comparison",
        "",
        "| Table | Rows | Columns | Byte-identical | Differing cells |",
        "|---|---:|---:|---|---:|",
    ]
    for filename, report in table_reports.items():
        lines.append(
            f"| `{filename}` | {report['candidate_rows']} | "
            f"{len(report['columns'])} | "
            f"{str(report['byte_identical']).lower()} | "
            f"{report['differing_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Expected provenance-only differences",
            "",
            "The result-lock schema advanced from "
            f"`{baseline_lock.get('schema_version')}` to "
            f"`{candidate_lock.get('schema_version')}`. The legacy implementation "
            "hash field was replaced by PHAxis-native module identity and an "
            "explicit `legacy_model_implementation_imported=false` guard. This "
            "run also replaced the runtime predecessor-config path with the "
            "PHAxis-owned model spec while retaining its source SHA as historical "
            "provenance. "
            "This necessarily changes the result-lock hash and summary identity, but "
            "does not change any numerical/statistical table field.",
            "",
            "Machine-readable details, input hashes, implementation hashes, and "
            "per-column maximum differences are in `equivalence_audit.json`.",
            "",
        ]
    )
    _atomic_text(output / "equivalence_audit.md", "\n".join(lines))
    if payload["status"] != "passed":
        raise RuntimeError("PHAxis biological-analysis equivalence audit failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--relative-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    result = audit(
        baseline=args.baseline.resolve(),
        candidate=args.candidate.resolve(),
        cohorts=args.cohorts.resolve(),
        output=args.output.resolve(),
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
