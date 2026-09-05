from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "phaxis"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import backfill_paper_first_manuscript as builder  # noqa: E402
import build_paper_first_biology_artifacts as focused_builder  # noqa: E402
from phaxis.traits import HAIR_TRAIT_FIELDS, ROOT_TRAIT_FIELDS  # noqa: E402


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return builder._sha256_file(path)


def _seal(payload: dict, field: str) -> None:
    payload.pop(field, None)
    payload[field] = builder._sha256_json(payload)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _master(path: Path) -> None:
    abstract = builder._slot("BIOLOGY_CURRENT_LINEAGE", "D15_ABSTRACT_SYNTHESIS")
    discussion = builder._slot("BIOLOGY_CURRENT_LINEAGE", "DISCUSSION_PARAGRAPH_D15")
    figure = builder._slot("FIGURE5_CURRENT_LINEAGE", "SOURCE_TABLE_IDENTITY")
    table = [
        builder._slot("TABLE3_CURRENT_LINEAGE", name)
        for name in builder.TABLE3_SLOTS
    ]
    excluded = {abstract, discussion, figure, *table}
    result_tokens = sorted(builder.EXPECTED_SCIENCE_TOKENS - excluded)
    admin = sorted(builder.EXPECTED_ADMIN_TOKENS) + [
        "[[RELEASE_METADATA:RELEASE_DOI]]"
    ]
    text = "\n".join(
        [
            "# Synthetic PHAxis manuscript",
            "",
            "## Abstract",
            f"PHAxis {abstract}.",
            "",
            "## 1. Introduction",
            "Synthetic editorial frame.",
            "",
            "## 2. Materials and Methods",
            "The identity evaluation used a 20-µm biological-presence tolerance.",
            "Production used the fixed 40-µm fusion attachment boundary.",
            "This was distinct from the 20-µm formal attachment-assurance threshold.",
            "PHAxis does not segment or quantify a root-cap region.",
            "",
            "## 3. Results",
            *result_tokens,
            "A root-cap region was neither segmented nor scored.",
            "",
            "## 4. Discussion",
            discussion,
            "",
            "## 5. Conclusions",
            "Synthetic conclusion.",
            "",
            "## 13. Main Figure Legends",
            "### Figure 5. Synthetic biological figure",
            f"Data source: {figure}.",
            "",
            "## 14. Main Tables",
            "### Table 3. Synthetic five-sentinel table",
            *table,
            "Root-cap area is not a PHAxis phenotype.",
            "",
            "## 15. Supplementary Information",
            *admin,
            "",
        ]
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def _metric(
    key: str,
    value: float,
    low: float,
    high: float,
    *,
    unit: str,
    n: int,
    instances: int,
    domain: str,
) -> dict:
    return {
        "domain": domain,
        "metric_key": key,
        "label": key,
        "value": value,
        "ci_low": low,
        "ci_high": high,
        "unit": unit,
        "n": n,
        "instances": instances,
        "definition": "synthetic current-lineage sufficient statistic",
        "ci_method": "image/source-unit nonparametric bootstrap",
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 20260828,
    }


def _assurance(root: Path) -> tuple[Path, Path, dict]:
    metrics = [
        _metric("conditional_length_mae_um", 8.0, 7.0, 9.0, unit="um", n=44, instances=283, domain="conditional_length"),
        _metric("conditional_length_bias_um", 1.0, 0.0, 2.0, unit="um", n=44, instances=283, domain="conditional_length"),
        _metric("conditional_length_ccc", 0.90, 0.85, 0.95, unit="ccc", n=44, instances=283, domain="conditional_length"),
        _metric("matched_endpoint_error_um", 5.0, 4.0, 6.0, unit="um", n=44, instances=283, domain="conditional_length"),
        _metric("matched_trajectory_continuity", 0.88, 0.84, 0.92, unit="fraction", n=44, instances=283, domain="conditional_length"),
        _metric("endpoint_complete_support_fraction", 0.50, 0.45, 0.55, unit="fraction", n=261, instances=566, domain="conditional_length"),
        _metric("root_continuity_maximum_single_component_coverage_mean", 0.98, 0.97, 0.99, unit="fraction", n=44, instances=44, domain="root_continuity"),
        _metric("root_continuity_maximum_single_component_coverage_median", 0.99, 0.98, 1.00, unit="fraction", n=44, instances=44, domain="root_continuity"),
        _metric("root_continuity_best_component_gap_median_um", 5.0, 4.0, 6.0, unit="um", n=44, instances=44, domain="root_continuity"),
        _metric("root_continuity_break_free_rate", 0.90, 0.82, 0.96, unit="fraction", n=44, instances=44, domain="root_continuity"),
        _metric("root_continuity_visible_axis_extent_mae_um", 10.0, 8.0, 12.0, unit="um", n=44, instances=44, domain="root_continuity"),
        _metric("hair_attachment_qualified_precision_20um", 0.80, 0.75, 0.85, unit="fraction", n=44, instances=400, domain="hair_attachment"),
        _metric("hair_attachment_qualified_recall_20um", 0.81, 0.76, 0.86, unit="fraction", n=44, instances=380, domain="hair_attachment"),
        _metric("hair_attachment_qualified_f1_20um", 0.805, 0.77, 0.84, unit="fraction", n=44, instances=780, domain="hair_attachment"),
        _metric("hair_attachment_error_median_um", 4.0, 3.0, 5.0, unit="um", n=44, instances=360, domain="hair_attachment"),
        _metric("hair_attachment_error_p95_um", 12.0, 10.0, 14.0, unit="um", n=44, instances=360, domain="hair_attachment"),
    ]
    metrics_path = root / "assurance_metrics.csv"
    _csv(metrics_path, metrics)
    roles = {
        row["metric_key"]: (
            "application_observability_non_accuracy"
            if row["metric_key"] == "endpoint_complete_support_fraction"
            else "annotated_qc_development_non_independent"
        )
        for row in metrics
    }
    receipt = {
        "schema_version": builder.ASSURANCE_SCHEMA,
        "status": "completed_locked_qc_development_assurance",
        "scope": "QC-development measurement assurance; non-independent",
        "metric_evidence_role_by_key": roles,
        "source_table_sha256": {
            "metrics": _sha(metrics_path),
            "pairs": _hash("pairs"),
            "support": _hash("support"),
            "topology": _hash("topology"),
            "root_traits": _hash("root-traits"),
        },
        "measurement_contract": {
            "conditional_length_base_match_tolerance_um": 20.0,
            "matched_trajectory_tolerance_um": 20.0,
            "hair_attachment_formal_tolerance_um": 20.0,
            "root_cap_region_output": False,
        },
        "counts": {
            "conditional_length_pairs": 283,
            "conditional_length_source_units": 44,
            "root_continuity_source_units": 44,
            "hair_attachment_source_units": 44,
            "hair_attachment_formal_identity_matches": 360,
            "application_support_source_units": 261,
            "application_formal_identity_hairs": 566,
        },
        "independent_accuracy_claim_allowed": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    _seal(receipt, "measurement_assurance_identity_sha256")
    receipt_path = root / "assurance_receipt.json"
    _json(receipt_path, receipt)
    return receipt_path, metrics_path, receipt


def _application(root: Path, model_bundle: str, root_expert: str, hair_expert: str) -> tuple[Path, Path, dict]:
    rows: list[dict] = []
    trait_fields = [*ROOT_TRAIT_FIELDS, *HAIR_TRAIT_FIELDS]
    for index in range(283):
        row = {
            "task_id": f"APP-{index:03d}",
            "formal_statistics_eligible": index < 260,
            "root_cap_region_output": False,
            "root_cap_area_used": False,
            "condition_metadata_used_for_routing": False,
            "canonical_annotations_read_during_inference": False,
            "blind_images_used": 0,
            "model_bundle_id": model_bundle,
            "root_expert_id": root_expert,
            "hair_identity_count_expert_id": hair_expert,
            **{field: float(index + 1) for field in trait_fields},
            "hair_count": 2,
            "hair_length_measurement_hair_count": 1,
        }
        # Exercise a real coverage range without fabricating zero measurements.
        if index < 3:
            row[ROOT_TRAIT_FIELDS[-1]] = ""
        rows.append(row)
    image_path = root / "application_image_traits.csv"
    _csv(image_path, rows)
    summary = {
        "schema_version": builder.APPLICATION_SCHEMA,
        "status": "completed",
        "tasks": 283,
        "formal_statistics_eligible": 260,
        "review_only": 23,
        "hair_identities": 566,
        "endpoint_complete_length_identities": 283,
        "nonduplicate_biological_numeric_traits": 32,
        "root_trait_fields": list(ROOT_TRAIT_FIELDS),
        "hair_trait_fields": list(HAIR_TRAIT_FIELDS),
        "image_traits_sha256": _sha(image_path),
        "root_cap_region_statistics_included": False,
        "hair_identity_count_expert": hair_expert,
        "model_bundle_id": model_bundle,
        "root_expert_id": root_expert,
        "model_contract_proposal_identity_sha256": _hash("proposal"),
        "condition_metadata_used_for_model_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    _seal(summary, "export_identity_sha256")
    summary_path = root / "application_summary.json"
    _json(summary_path, summary)
    return summary_path, image_path, summary


def _focused_entries() -> dict[str, dict]:
    entries: dict[str, dict] = {}

    def add(key: str, value: object, *, authority: str = "synthetic_source") -> None:
        entries[key] = {
            "value": value,
            "derivation": {
                "method": "synthetic_current_lineage_sufficient_statistic",
                "source_file_sha256": {authority: _hash(authority)},
                "parameters": {},
            },
        }

    add("FINAL_D15_LENGTH_SUPPORT_MIN_PERCENT", "45.0")
    add("FINAL_D15_LENGTH_SUPPORT_MAX_PERCENT", "55.0")
    add("FINAL_D15_CLEAN_LENGTH_HAIR_N", 283)
    add("FINAL_D15_FIRST_HAIR_OBSERVABILITY_BY_CELL", "8/10 (80.0%) / 9/10 (90.0%) / 8/10 (80.0%) / 9/10 (90.0%)")
    add("FINAL_D15_VISIBLE_AXIS_CENSORING_BY_CELL", "1/10 (10.0%) / 1/10 (10.0%) / 2/10 (20.0%) / 2/10 (20.0%)")
    add("FINAL_D15_AXIAL_ABUNDANCE_PATTERN", "a condition-resolved visible-abundance profile with distinct spatial maxima")
    add("FINAL_D15_AXIAL_LENGTH_PATTERN", "a condition-resolved supported-length profile with distinct spatial maxima")
    add("FINAL_D15_AXIAL_SUPPORT_PATTERN", "a condition-resolved complete-curve-support profile with explicit denominators")
    add("FINAL_PROFILE_CLEAN_ELIGIBLE_BIN_N", 200)
    add("FINAL_PROFILE_CLEAN_LENGTH_OBSERVATION_N", 160)
    add("FINAL_PROFILE_CROSSCHECK_MATCH_N", 80, authority="primary_profile_export_summary")
    add("FINAL_PROFILE_CROSSCHECK_TOTAL_N", 80, authority="primary_profile_export_summary")
    add("FINAL_D15_CLEAN_POOL_CELL_N", "11 / 11 / 11 / 11", authority="cohort_condition_counts")
    add("FINAL_D15_CLEAN_FORMAL_CELL_N", "10 / 10 / 10 / 10")
    add("FINAL_D15_FULL_POOL_CELL_N", "12 / 12 / 12 / 12", authority="cohort_condition_counts")

    for sentinel, contract in builder.ENDPOINTS.items():
        add(str(contract["count"]), "10 / 10 / 10 / 10")
        add(str(contract["raw"]), "1.0 [0.8–1.2] / 1.1 [0.9–1.3] / 1.2 [1.0–1.4] / 1.3 [1.1–1.5]")
        add(str(contract["pattern"]), f"the {sentinel} fixed effect family retained explicit clean/full states")
        for effect in builder.EFFECT_ORDER:
            supported = sentinel == "H08" and effect == "CONSTRUCT"
            ratio = "1.200" if supported else "0.900"
            interval = "1.100–1.300" if supported else "0.800–1.100"
            prefix = contract["prefix"]
            add(f"FINAL_{prefix}_{effect}_RATIO", ratio)
            add(f"FINAL_{prefix}_{effect}_CI", interval)
            add(
                f"FINAL_{prefix}_{effect}_FULL_SENSITIVITY",
                "1.100 (1.000–1.200); endpoint n=48; clean/full point-estimate state concordant",
            )
    assert builder.FOCUSED_BASE_KEYS <= set(entries)
    return entries


def _focused(
    root: Path,
    assurance_sha: str,
    model_bundle: str,
    root_expert: str,
    hair_expert: str,
) -> tuple[Path, Path, dict, dict]:
    entries = _focused_entries()
    values = {
        "schema_version": builder.FOCUSED_VALUES_SCHEMA,
        "status": "completed_current_train399_exact283_paper_first",
        "scope": "Fig5_and_Table3_machine_values_only",
        "entries": entries,
        "entry_count": len(entries),
        "narrative_decision": {
            "branch_id": "A",
            "identity_sha256": _hash("narrative"),
            "profiles_select_or_veto_branch": False,
            "support_mask_bits": "1" + "0" * 14,
            "clean_directions": ["higher", *("lower" for _ in range(14))],
        },
        "model_contract_proposal_sha256": _hash("proposal-file"),
        "model_contract_proposal_identity_sha256": _hash("proposal"),
        "model_bundle_id": model_bundle,
        "root_expert_id": root_expert,
        "hair_identity_count_expert_id": hair_expert,
        "input_file_sha256": {
            "assurance_receipt": assurance_sha,
            "synthetic_source": _hash("synthetic_source"),
            "cohort_condition_counts": _hash("cohort_condition_counts"),
            "primary_profile_export_summary": _hash("primary_profile_export_summary"),
        },
        "claim_contract": {
            "current_train399_exact283_only": True,
            "clean261_is_primary": True,
            "full283_is_overlap_inclusion_sensitivity": True,
            "qcdevelopment44_independent_accuracy_claim": False,
            "profile_hypothesis_tests_added": False,
            "root_cap_region_statistics_included": False,
            "benchmark_values_included": False,
            "author_metadata_included": False,
            "release_packaging_status_claimed": False,
        },
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    _seal(values, "values_identity_sha256")
    values_path = root / "focused_values.json"
    _json(values_path, values)
    receipt = {
        "schema_version": builder.FOCUSED_RECEIPT_SCHEMA,
        "status": "completed_current_train399_exact283_paper_first",
        "output_file_sha256": {"manuscript_values": _sha(values_path)},
        "fig5_source_package_identity_sha256": _hash("figure5-source"),
        "manuscript_values_identity_sha256": values["values_identity_sha256"],
        "model_contract_proposal_identity_sha256": _hash("proposal"),
        "model_bundle_id": model_bundle,
        "root_expert_id": root_expert,
        "hair_identity_count_expert_id": hair_expert,
        "benchmark_required": False,
        "release_packaging_required": False,
        "author_metadata_required": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    _seal(receipt, "receipt_identity_sha256")
    receipt_path = root / "focused_receipt.json"
    _json(receipt_path, receipt)
    return values_path, receipt_path, values, receipt


def _dataset(root: Path) -> tuple[dict[str, Path], dict[str, str], dict]:
    root.mkdir(parents=True)
    master = root / "manuscript_frame.md"
    _master(master)
    assurance_receipt, assurance_metrics, assurance = _assurance(root)
    model_bundle = "PHAXIS-V1.0.0-STRICT-TRAIN399-" + _hash("bundle")[:20].upper()
    root_expert = "PHAxis-root-provider-" + _hash("root")[:20].upper()
    hair_expert = "PHAxis-StageB-train399-five-seed"
    application_summary, application_image_traits, _ = _application(
        root, model_bundle, root_expert, hair_expert
    )
    focused_values, focused_receipt, values, receipt = _focused(
        root,
        _sha(assurance_receipt),
        model_bundle,
        root_expert,
        hair_expert,
    )
    paths = {
        "master": master,
        "focused_values": focused_values,
        "focused_receipt": focused_receipt,
        "application_summary": application_summary,
        "application_image_traits": application_image_traits,
        "assurance_receipt": assurance_receipt,
        "assurance_metrics": assurance_metrics,
    }
    hashes = {role: _sha(path) for role, path in paths.items()}
    return paths, hashes, {"focused_values": values, "focused_receipt": receipt, "assurance": assurance}


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rebind_focused(paths: dict[str, Path], hashes: dict[str, str], values: dict, receipt: dict) -> None:
    values["entry_count"] = len(values["entries"])
    _seal(values, "values_identity_sha256")
    _json(paths["focused_values"], values)
    receipt["output_file_sha256"]["manuscript_values"] = _sha(paths["focused_values"])
    receipt["manuscript_values_identity_sha256"] = values["values_identity_sha256"]
    _seal(receipt, "receipt_identity_sha256")
    _json(paths["focused_receipt"], receipt)
    hashes["focused_values"] = _sha(paths["focused_values"])
    hashes["focused_receipt"] = _sha(paths["focused_receipt"])


def test_exact_73_backfill_is_atomic_deterministic_and_preserves_admin_slots(tmp_path: Path) -> None:
    paths, hashes, _payloads = _dataset(tmp_path / "synthetic_inputs")
    first = tmp_path / "paper_output_one"
    second = tmp_path / "paper_output_two"
    receipt = builder.backfill_paper_first_manuscript(
        inputs=paths,
        expected_sha256=hashes,
        output=first,
    )
    builder.backfill_paper_first_manuscript(
        inputs=paths,
        expected_sha256=hashes,
        output=second,
    )
    assert receipt["scientific_slot_count"] == 73
    assert receipt["administrative_slot_count_remaining"] == 21
    assert receipt["gpu_program_started"] is False
    assert _tree_hashes(first) == _tree_hashes(second)
    rendered = (first / "PHAXIS_PLANT_PHENOMICS_SCIENCE_COMPLETE.md").read_text(
        encoding="utf-8"
    )
    assert not any(token in rendered for token in builder.EXPECTED_SCIENCE_TOKENS)
    residual = [match.group(0) for match in builder.SLOT_PATTERN.finditer(rendered)]
    original_admin = [
        match.group(0)
        for match in builder.SLOT_PATTERN.finditer(paths["master"].read_text(encoding="utf-8"))
        if match.group(1) in builder.ADMIN_FAMILIES
    ]
    assert residual == original_admin
    audit = json.loads((first / "CONSISTENCY_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed_exact_73_scientific_slot_backfill"
    assert audit["effect_display_crosscheck_count"] == 15
    assert all(audit["effect_display_crosschecks"].values())
    assert audit["threshold_contract_um"]["identity_evaluation_um"] == 20.0
    assert audit["threshold_contract_um"]["identity_to_geometry_fusion_um"] == 40.0
    assert audit["threshold_contract_um"]["formal_attachment_assurance_um"] == 20.0
    assert audit["root_cap_region_trait_count"] == 0
    assert "higher H08/N visible-hair abundance for the OE-labelled:EV contrast" in rendered

    with pytest.raises(builder.BackfillError, match="overwrite"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=first,
        )


def test_missing_science_primitives_and_forbidden_entry_fail_closed(tmp_path: Path) -> None:
    paths, hashes, payloads = _dataset(tmp_path / "synthetic_inputs")
    values = deepcopy(payloads["focused_values"])
    receipt = deepcopy(payloads["focused_receipt"])
    values["entries"].pop("FINAL_D15_FULL_POOL_CELL_N")
    _rebind_focused(paths, hashes, values, receipt)
    with pytest.raises(builder.BackfillError, match="missing current-lineage primitives"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=tmp_path / "missing_output",
        )
    assert not (tmp_path / "missing_output").exists()

    paths, hashes, payloads = _dataset(tmp_path / "second_synthetic_inputs")
    values = deepcopy(payloads["focused_values"])
    receipt = deepcopy(payloads["focused_receipt"])
    values["entries"]["FINAL_BENCHMARK_SPEEDUP"] = {
        "value": "99x",
        "derivation": {"method": "forbidden", "source_file_sha256": {}, "parameters": {}},
    }
    _rebind_focused(paths, hashes, values, receipt)
    with pytest.raises(builder.BackfillError, match="forbidden benchmark"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=tmp_path / "forbidden_output",
        )


def test_hash_root_cap_threshold_and_forbidden_path_guards(tmp_path: Path) -> None:
    paths, hashes, _payloads = _dataset(tmp_path / "synthetic_inputs")
    wrong = dict(hashes)
    wrong["master"] = "f" * 64
    with pytest.raises(builder.BackfillError, match="SHA-256 mismatch"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=wrong,
            output=tmp_path / "hash_output",
        )

    marked = tmp_path / "legacy_source" / "manuscript.md"
    marked.parent.mkdir()
    shutil.copyfile(paths["master"], marked)
    marked_paths = dict(paths)
    marked_paths["master"] = marked
    marked_hashes = dict(hashes)
    marked_hashes["master"] = _sha(marked)
    with pytest.raises(builder.BackfillError, match="labelled path refused"):
        builder.backfill_paper_first_manuscript(
            inputs=marked_paths,
            expected_sha256=marked_hashes,
            output=tmp_path / "path_output",
        )

    # A self-consistent but scientifically forbidden root-cap row is rejected.
    fields, rows = builder._read_csv(paths["application_image_traits"], "synthetic")
    rows[0]["root_cap_region_output"] = "True"
    _csv(paths["application_image_traits"], rows)
    app = json.loads(paths["application_summary"].read_text(encoding="utf-8"))
    app["image_traits_sha256"] = _sha(paths["application_image_traits"])
    _seal(app, "export_identity_sha256")
    _json(paths["application_summary"], app)
    hashes["application_image_traits"] = _sha(paths["application_image_traits"])
    hashes["application_summary"] = _sha(paths["application_summary"])
    with pytest.raises(builder.BackfillError, match="root-cap region entered"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=tmp_path / "root_cap_output",
        )


def test_rejects_threshold_conflation_even_when_receipts_are_resealed(tmp_path: Path) -> None:
    paths, hashes, payloads = _dataset(tmp_path / "synthetic_inputs")
    assurance = deepcopy(payloads["assurance"])
    assurance["measurement_contract"]["hair_attachment_formal_tolerance_um"] = 40.0
    _seal(assurance, "measurement_assurance_identity_sha256")
    _json(paths["assurance_receipt"], assurance)
    hashes["assurance_receipt"] = _sha(paths["assurance_receipt"])

    values = deepcopy(payloads["focused_values"])
    receipt = deepcopy(payloads["focused_receipt"])
    values["input_file_sha256"]["assurance_receipt"] = hashes["assurance_receipt"]
    _rebind_focused(paths, hashes, values, receipt)
    with pytest.raises(builder.BackfillError, match="threshold/root-cap contract changed"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=tmp_path / "threshold_output",
        )


def test_rejects_rounded_only_narrative_and_unbound_new_authority(tmp_path: Path) -> None:
    paths, hashes, payloads = _dataset(tmp_path / "synthetic_inputs")
    values = deepcopy(payloads["focused_values"])
    receipt = deepcopy(payloads["focused_receipt"])
    values["narrative_decision"].pop("support_mask_bits")
    _rebind_focused(paths, hashes, values, receipt)
    with pytest.raises(builder.BackfillError, match="unrounded 15-cell support mask"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=tmp_path / "rounded_only_output",
        )

    paths, hashes, payloads = _dataset(tmp_path / "second_synthetic_inputs")
    values = deepcopy(payloads["focused_values"])
    receipt = deepcopy(payloads["focused_receipt"])
    values["entries"]["FINAL_D15_FULL_POOL_CELL_N"]["derivation"][
        "source_file_sha256"
    ]["cohort_condition_counts"] = "f" * 64
    _rebind_focused(paths, hashes, values, receipt)
    with pytest.raises(builder.BackfillError, match="sealed input authority map"):
        builder.backfill_paper_first_manuscript(
            inputs=paths,
            expected_sha256=hashes,
            output=tmp_path / "unbound_authority_output",
        )


def test_real_focused_adapter_schema_is_directly_consumable(tmp_path: Path) -> None:
    """Exercise the real focused adapter, not a hand-written values/receipt pair."""

    tests_dir = PROJECT_ROOT / "tests" / "phaxis"
    sys.path.insert(0, str(tests_dir))
    import test_paper_first_biology_artifacts as focused_fixture  # noqa: PLC0415

    source_root = tmp_path / "joint_current_inputs"
    focused_inputs, focused_hashes, proposal = focused_fixture._dataset(source_root)

    # Both programs consume the same assurance receipt. Enrich the adapter's
    # synthetic authority with the manuscript-facing metrics contract while
    # retaining its support and lineage bindings.
    _unused_receipt, metrics_path, manuscript_assurance = _assurance(
        tmp_path / "joint_assurance_tables"
    )
    assurance = json.loads(
        focused_inputs.assurance_receipt.read_text(encoding="utf-8")
    )
    assurance["measurement_contract"] = manuscript_assurance[
        "measurement_contract"
    ]
    assurance["counts"] = manuscript_assurance["counts"]
    assurance["metric_evidence_role_by_key"] = manuscript_assurance[
        "metric_evidence_role_by_key"
    ]
    assurance["source_table_sha256"].update(
        {
            key: value
            for key, value in manuscript_assurance["source_table_sha256"].items()
            if key != "support"
        }
    )
    _seal(assurance, "measurement_assurance_identity_sha256")
    _json(focused_inputs.assurance_receipt, assurance)
    focused_hashes["assurance_receipt"] = _sha(focused_inputs.assurance_receipt)

    focused_output = tmp_path / "focused_current_output"
    focused_builder.build_paper_first_biology_artifacts(
        inputs=focused_inputs,
        expected_sha256=focused_hashes,
        expected_model_bundle_id=proposal["model_bundle_id"],
        expected_root_expert_id=proposal["root_expert"]["expert_id"],
        output=focused_output,
    )

    master = tmp_path / "manuscript_frame.md"
    _master(master)
    application_summary, application_image_traits, application = _application(
        tmp_path / "joint_application",
        proposal["model_bundle_id"],
        proposal["root_expert"]["expert_id"],
        focused_builder.STAGEB_EXPERT_ID,
    )
    application["model_contract_proposal_identity_sha256"] = proposal[
        "model_contract_identity_sha256"
    ]
    _seal(application, "export_identity_sha256")
    _json(application_summary, application)

    paths = {
        "master": master,
        "focused_values": focused_output / "manuscript_values.json",
        "focused_receipt": focused_output / "receipt.json",
        "application_summary": application_summary,
        "application_image_traits": application_image_traits,
        "assurance_receipt": focused_inputs.assurance_receipt,
        "assurance_metrics": metrics_path,
    }
    hashes = {role: _sha(path) for role, path in paths.items()}
    receipt = builder.backfill_paper_first_manuscript(
        inputs=paths,
        expected_sha256=hashes,
        output=tmp_path / "joint_paper_output",
    )
    assert receipt["scientific_slot_count"] == 73
    assert receipt["administrative_slot_count_remaining"] == 21
