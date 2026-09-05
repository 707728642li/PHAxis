from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/phaxis/audit_stage22_h11_raw_median_amendment.py"
AUTHORITY_ROOT = (
    PROJECT_ROOT / "outputs/phaxis_stage22_H11_raw_median_gap_audit_r4_20260831"
)
HISTORICAL_R2_ROOT = (
    PROJECT_ROOT / "outputs/phaxis_stage22_H11_raw_median_gap_audit_r2_20260831"
)
HISTORICAL_R2_FILE_SHA256 = {
    "amendment_audit.json": (
        "3e63c1f69e28614995bb520d6a7a6006f0f9ea91afdb7f517a3e005d712a0ef1"
    ),
    "recomputed_tables/clean_vs_full_effect_stability.csv": (
        "b29872b3a1e4a70f6d74c93a239067ec7a6e755c161bcd72595a7587440902e1"
    ),
    "recomputed_tables/full283_sensitivity_factorial_tests.csv": (
        "dc3b050c3b8fe064ff3d860e1deb36962bf998524125c5b976c5fdccf60e5cf9"
    ),
    "recomputed_tables/primary_clean_exploratory_factorial_tests.csv": (
        "7611829372d934e822f3515ea864c6fe3e67a7969f45824fbc09c8384eb11b7f"
    ),
    "recomputed_tables/primary_group_summaries.csv": (
        "7d74131e283442ec4c32ad9a23b11ade2c7642c54993728a319bde6d05bfc9ce"
    ),
    "recomputed_tables/primary_model_qc_flow.csv": (
        "24f24eb6327c98c78e888c9bec25fd02cf5cffc61f94a739ec235d9e5fe25baf"
    ),
    "recomputed_tables/robust_sensitivity.csv": (
        "4e4fdd2a92bdca6280ca7806783efe65127dffb4273b8b15ab84bb31adacfd61"
    ),
}
BASELINE_ROOT = (
    PROJECT_ROOT
    / "outputs/phaxis_biological_analysis_native_modelspec_candidate_20260828"
)
COHORT_ROOT = PROJECT_ROOT / "outputs/phaxis_v1_0_biological_cohorts_run1_final"
MODEL_SPEC = PROJECT_ROOT / "configs/phaxis/v1_0/biological_model_spec.json"


def _module():
    spec = importlib.util.spec_from_file_location("h11_amendment_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(module, table: str):
    receipt = json.loads(
        (AUTHORITY_ROOT / "amendment_audit.json").read_text(encoding="utf-8")
    )
    baseline = pd.read_csv(BASELINE_ROOT / "tables" / table)
    candidate = pd.read_csv(AUTHORITY_ROOT / "recomputed_tables" / table)
    independent = receipt["tables"][table]["H11"]
    report = module._compare_factorial_table(
        baseline, candidate, independent=independent, tolerance=1e-12
    )
    assert report["passed"] is True
    return baseline, candidate, independent


def test_real_authority_is_narrow_pass_and_identity_sealed() -> None:
    module = _module()
    payload = json.loads(
        (AUTHORITY_ROOT / "amendment_audit.json").read_text(encoding="utf-8")
    )
    identity = payload.pop("amendment_audit_identity_sha256")
    assert identity == module.sha256_json(payload)
    assert payload["schema_version"] == module.SCHEMA_VERSION
    assert payload["status"] == "passed"
    assert payload["protected_primary_inference_equivalent"] is True
    assert payload["unaffected_tables_byte_identical"] is True
    assert payload["non_h11_existing_fields_equivalent"] is True
    assert payload["candidate_schema_extension_exact"] is True
    assert payload["H11_raw_median_companion"]["validated"] is True
    assert payload["H11_raw_median_companion"]["effect_rows"] == 6
    assert payload["unauthorized_differing_cells"] == 0
    assert payload["blind_images_used"] == 0
    assert payload["implementation_sha256"]["audit_test"] == module.sha256_file(
        Path(__file__)
    )
    assert payload["implementation_sha256"][
        "publication_figure_input_builder"
    ] == module.sha256_file(
        PROJECT_ROOT / "scripts/phaxis/build_publication_figure_inputs.py"
    )


def test_historical_r2_authority_remains_byte_immutable() -> None:
    module = _module()
    observed = {
        path.relative_to(HISTORICAL_R2_ROOT).as_posix(): module.sha256_file(path)
        for path in HISTORICAL_R2_ROOT.rglob("*")
        if path.is_file()
    }
    assert observed == HISTORICAL_R2_FILE_SHA256
    historical = json.loads(
        (HISTORICAL_R2_ROOT / "amendment_audit.json").read_text(encoding="utf-8")
    )
    assert historical["amendment_audit_identity_sha256"] == (
        "3cee831d07c8e8c28cffa5eb61b09bc7086c4cd6169222d6c11b9e3059c1c9d5"
    )


def test_factorial_comparator_rejects_protected_and_non_h11_drift() -> None:
    module = _module()
    baseline, candidate, independent = _fixture(module, module.PRIMARY_TABLE)

    protected = candidate.copy()
    row = protected.index[protected["endpoint"].eq(module.H11_ENDPOINT)][0]
    protected.loc[row, "estimate"] *= 1.01
    report = module._compare_factorial_table(
        baseline, protected, independent=independent, tolerance=1e-12
    )
    assert report["passed"] is False
    assert report["protected_differing_cells"] == 1
    assert report["unauthorized_differing_cells"] == 1

    non_h11 = candidate.copy()
    row = non_h11.index[~non_h11["endpoint"].eq(module.H11_ENDPOINT)][0]
    non_h11.loc[row, "raw_effect_estimate"] += 1.0
    report = module._compare_factorial_table(
        baseline, non_h11, independent=independent, tolerance=1e-12
    )
    assert report["passed"] is False
    assert report["non_h11_differing_cells"] == 1


def test_factorial_comparator_rejects_fake_median_and_provenance_drift() -> None:
    module = _module()
    baseline, candidate, independent = _fixture(module, module.SENSITIVITY_TABLE)

    fake = candidate.copy()
    row = fake.index[fake["endpoint"].eq(module.H11_ENDPOINT)][0]
    fake.loc[row, "raw_effect_estimate"] = baseline.loc[
        row, "raw_effect_estimate"
    ]
    report = module._compare_factorial_table(
        baseline, fake, independent=independent, tolerance=1e-12
    )
    assert report["passed"] is False
    assert report["H11_independent_numeric_recomputation_exact"] is False

    provenance = candidate.copy()
    provenance.loc[
        provenance["endpoint"].eq(module.H11_ENDPOINT),
        "raw_effect_bootstrap_replicates",
    ] = 4999
    report = module._compare_factorial_table(
        baseline, provenance, independent=independent, tolerance=1e-12
    )
    assert report["passed"] is False
    assert report["H11_raw_median_contract_exact"] is False

    reordered = candidate.iloc[::-1].reset_index(drop=True)
    report = module._compare_factorial_table(
        baseline, reordered, independent=independent, tolerance=1e-12
    )
    assert report["passed"] is False
    assert report["row_identity_exact"] is False


def test_independent_h11_is_source_root_order_invariant() -> None:
    module = _module()
    model_spec = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
    traits = pd.read_csv(COHORT_ROOT / "primary_clean261/traits.csv")
    first = module._independent_h11(traits, model_spec=model_spec)
    second = module._independent_h11(
        traits.sample(frac=1.0, random_state=781).reset_index(drop=True),
        model_spec=model_spec,
    )
    assert first["cell_counts"] == second["cell_counts"]
    assert first["cell_medians"] == second["cell_medians"]
    assert first["cell_means"] == second["cell_means"]
    for effect in module.EFFECTS:
        for field in (
            "raw_effect_estimate",
            "raw_effect_ci95_low",
            "raw_effect_ci95_high",
            "historical_raw_mean_contrast",
        ):
            assert first["effects"][effect][field] == second["effects"][effect][field]
        for field in (
            "standardized_effect",
            "standardized_ci95_low",
            "standardized_ci95_high",
        ):
            assert abs(
                first["effects"][effect][field]
                - second["effects"][effect][field]
            ) < 1e-14
    assert first["effective_seed"] == 20271264
    assert first["bootstrap_replicates"] == 5000


def test_change_whitelist_is_exact() -> None:
    module = _module()
    receipt = json.loads(
        (AUTHORITY_ROOT / "amendment_audit.json").read_text(encoding="utf-8")
    )
    contract = deepcopy(receipt["change_contract"])
    assert tuple(contract["changed_existing_columns_whitelist"]) == (
        "raw_effect_estimate",
        "raw_effect_ci95_low",
        "raw_effect_ci95_high",
        "standardized_effect",
        "standardized_ci95_low",
        "standardized_ci95_high",
    )
    assert tuple(contract["added_provenance_columns"]) == (
        "raw_effect_estimand",
        "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates",
        "raw_effect_bootstrap_seed",
    )
    assert contract["separate_hypothesis_test_added"] is False
    assert contract["D15_fixed_effect_family_changed"] is False
