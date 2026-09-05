from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest

from phaxis.manuscript_values import HUMAN_METADATA_TOKENS, TOKEN_PATTERN, sha256_json
from phaxis.publication_evidence import SUPPLEMENTARY_FIGURE_STEMS
from phaxis.publication_titles import FIGURE_STATIC_TITLES, TABLE_STATIC_TITLES
from phaxis.release_topology import MANDATORY_STAGE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/phaxis/build_publication_missing_results_matrix.py"
MASTER = PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
SUPPLEMENT = PROJECT_ROOT / "docs/phaxis/PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md"
STAGE_CONTRACT = PROJECT_ROOT / "configs/phaxis/v1_0/post_training_release_stage_contract_1_8.json"
SUBMISSION = PROJECT_ROOT / "manuscript/phaxis_v1_0/SUBMISSION_TITLE_METADATA_TEMPLATE_2_0.json"
MANUSCRIPT_METADATA = PROJECT_ROOT / "configs/phaxis/v1_0/POST_TRAINING_MANUSCRIPT_METADATA_TEMPLATE.json"
RELEASE_METADATA = PROJECT_ROOT / "configs/phaxis/v1_0/POST_TRAINING_RELEASE_AUTHOR_METADATA_TEMPLATE.json"
HANDOVER_METADATA = PROJECT_ROOT / "configs/phaxis/v1_0/POST_TRAINING_HANDOVER_ATTESTATION_TEMPLATE.json"
HISTORICAL_R6_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R6"
)
HISTORICAL_R8_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R8"
)
HISTORICAL_R9_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R9"
)
HISTORICAL_R10_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R10"
)
HISTORICAL_R11_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R11"
)
HISTORICAL_R12_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R12"
)
HISTORICAL_R13_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260830_R13"
)
HISTORICAL_R14_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260831_R14"
)
HISTORICAL_R15_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260831_R15"
)
CURRENT_R16_BUNDLE = (
    PROJECT_ROOT
    / "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260831_R16"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "build_publication_missing_results_matrix", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _build():
    module = _module()
    payload = module.build_matrix(
        master=MASTER,
        supplement=SUPPLEMENT,
        stage_contract=STAGE_CONTRACT,
        submission_metadata=SUBMISSION,
        manuscript_metadata_template=MANUSCRIPT_METADATA,
        release_metadata_template=RELEASE_METADATA,
        handover_attestation_template=HANDOVER_METADATA,
    )
    return module, payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_default_targets_release_control_1_8(tmp_path: Path) -> None:
    module = _module()
    args = module._parser().parse_args(["--output", str(tmp_path / "bundle")])
    assert args.stage_contract == STAGE_CONTRACT


def test_matrix_exhaustively_covers_current_tokens_figures_and_tables() -> None:
    _, payload = _build()
    master_text = MASTER.read_text(encoding="utf-8")
    supplement_text = SUPPLEMENT.read_text(encoding="utf-8")
    tokens = TOKEN_PATTERN.findall(master_text)

    assert payload["schema_version"] == "PHAxis-publication-missing-results-matrix-1.0"
    assert payload["status"] == "complete_non_result_publication_gap_audit"
    assert payload["formal_scientific_result_receipt"] is False
    assert payload["scientific_values_present"] is False
    assert payload["gpu_program_started"] is False
    assert payload["image_or_annotation_data_read"] is False
    assert payload["blind_images_used"] == 0
    assert payload["root_cap_region_statistics_included"] is False

    assert [row["token"] for row in payload["tokens"]] == sorted(set(tokens))
    assert sum(row["occurrence_count"] for row in payload["tokens"]) == len(tokens)
    assert payload["inventory_summary"]["main_unique_token_count"] == len(set(tokens))
    assert payload["inventory_summary"]["main_token_occurrence_count"] == len(tokens)
    assert not TOKEN_PATTERN.findall(supplement_text)
    assert payload["inventory_summary"]["supplement_unique_token_count"] == 0

    assert len(payload["main_figures"]) == 6
    assert [row["number"] for row in payload["main_figures"]] == [str(index) for index in range(1, 7)]
    assert [row["stem"] for row in payload["supplementary_figures"]] == list(
        SUPPLEMENTARY_FIGURE_STEMS
    )
    assert len(payload["main_tables"]) == 3
    assert len(payload["supplementary_tables_and_data"]) == 10


def test_token_routes_separate_source_readiness_from_final_materialization() -> None:
    _, payload = _build()
    by_token = {row["token"]: row for row in payload["tokens"]}
    by_stage = {row["name"]: row for row in payload["used_release_stages"]}

    assert set(HUMAN_METADATA_TOKENS) == {
        token for token, row in by_token.items() if row["authority_class"] == "human_external"
    }
    assert payload["inventory_summary"]["human_external_token_count"] == len(
        HUMAN_METADATA_TOKENS
    )
    assert payload["inventory_summary"]["historical_comparator_token_count"] == sum(
        row["authority_class"] == "historical_development_comparator"
        for row in payload["tokens"]
    )
    assert payload["inventory_summary"]["final_machine_token_count"] == sum(
        row["authority_class"] == "final_machine" for row in payload["tokens"]
    )

    assert by_token["FINAL_HAIR_F1_20UM"]["source_evidence_segment"] == "gpu1_scientific_prefix"
    assert by_token["FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE"]["source_evidence_stages"] == [
        "measurement_assurance"
    ]
    assert by_token["FINAL_E2E_IMAGES_PER_MIN"]["source_evidence_segment"] == "after_gpu0_release"
    assert by_token["FINAL_MODEL_BUNDLE_SHA256"]["source_evidence_segment"] == (
        "post_gpu0_and_human_authority"
    )
    assert by_token["FINAL_CLEAN_INSTALL_EXAMPLE_IDENTITY"]["source_evidence_stages"] == [
        "clean_install"
    ]
    assert by_token["PHAXIS_RELEASE_DOI"]["source_evidence_segment"] == (
        "human_external_authority"
    )

    for row in payload["tokens"]:
        assert row["manuscript_value_stage"] == "values"
        assert row["final_token_materialization_segment"] == (
            "post_gpu0_and_human_authority"
        )
        for stage in row["source_evidence_stages"]:
            assert stage in MANDATORY_STAGE_ORDER
            assert stage in by_stage

    assert by_stage["overlay_evidence"]["index"] == 27
    assert by_stage["benchmark_phaxis_production"]["index"] == 28
    assert by_stage["benchmark_phaxis_production"]["physical_gpus"] == [0]


def test_all_final_plates_wait_for_runtime_but_preserve_prefix_source_roles() -> None:
    _, payload = _build()
    figures = [*payload["main_figures"], *payload["supplementary_figures"]]
    assert all(row["sealed_figure_stage"] == "figures" for row in figures)
    assert all(row["sealed_artifact_segment"] == "after_gpu0_release" for row in figures)
    assert all(len(row["expected_files"]) == 3 for row in figures)

    by_number = {row["number"]: row for row in figures}
    assert by_number["1"]["source_evidence_segment"] == "gpu1_scientific_prefix"
    assert by_number["5"]["source_evidence_segment"] == "gpu1_scientific_prefix"
    assert by_number["6"]["source_evidence_segment"] == "after_gpu0_release"
    assert by_number["S8"]["source_evidence_segment"] == "after_gpu0_release"
    assert by_number["S9"]["source_evidence_segment"] == "gpu1_scientific_prefix"


def test_supplementary_table_data_materialization_route_is_closed() -> None:
    _, payload = _build()
    assert payload["publication_code_gaps"] == []
    closed = payload["closed_publication_code_gaps"]
    assert len(closed) == 1
    assert closed[0]["code"] == "SUPPLEMENTARY_TABLE_DATA_MATERIALIZER_ABSENT"
    assert closed[0]["formal_materializer_stage"] == "figures"
    assert payload["inventory_summary"]["supplementary_table_or_data_materializer_count"] == 10
    assert all(
        row["formal_materializer_stage"] == "figures"
        and row["formal_output_path"].startswith(
            "{run_dir}/figures/output/supplementary_tables_and_data/S"
        )
        and row["formal_item_receipt"].endswith("/item_receipt.json")
        and row["formal_bundle_receipt"].endswith("/bundle_receipt.json")
        and row["submission_use_allowed"] is True
        for row in payload["supplementary_tables_and_data"]
    )

    topology_source = (PROJECT_ROOT / "src/phaxis/release_topology.py").read_text(encoding="utf-8")
    supplement_compiler = (
        PROJECT_ROOT / "scripts/phaxis/compile_supplementary_manuscript.py"
    ).read_text(encoding="utf-8")
    figure_builder = (
        PROJECT_ROOT / "scripts/phaxis/build_publication_figures.py"
    ).read_text(encoding="utf-8")
    assert '"figures"' in topology_source
    assert "materialize_supplementary_table_data_bundle" in figure_builder
    assert "validate_supplementary_table_data_bundle" in supplement_compiler
    assert "numeric_or_author_values_inserted\": 0" in supplement_compiler


def test_bundle_is_create_only_deterministic_and_self_sealed(tmp_path: Path) -> None:
    module, payload = _build()
    first = tmp_path / "first"
    second = tmp_path / "second"
    module.write_bundle(first, payload)
    module.write_bundle(second, payload)

    for name in ("matrix.json", "tokens.csv", "README.md", "bundle_receipt.json"):
        assert _sha256(first / name) == _sha256(second / name)

    stored = json.loads((first / "matrix.json").read_text(encoding="utf-8"))
    identity = stored.pop("matrix_identity_sha256")
    assert identity == sha256_json(stored)

    receipt = json.loads((first / "bundle_receipt.json").read_text(encoding="utf-8"))
    receipt_identity = receipt.pop("bundle_identity_sha256")
    assert receipt_identity == sha256_json(receipt)
    assert receipt["file_sha256"] == {
        "README.md": _sha256(first / "README.md"),
        "matrix.json": _sha256(first / "matrix.json"),
        "tokens.csv": _sha256(first / "tokens.csv"),
    }
    assert receipt["gpu_program_started"] is False
    assert receipt["blind_images_used"] == 0

    with (first / "tokens.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(set(TOKEN_PATTERN.findall(MASTER.read_text(encoding="utf-8"))))
    assert {row["token"] for row in rows} == set(TOKEN_PATTERN.findall(MASTER.read_text(encoding="utf-8")))

    with pytest.raises(module.MatrixError, match="refusing to overwrite"):
        module.write_bundle(first, payload)


def test_checked_r15_remains_an_immutable_pre_p2_snapshot() -> None:
    _, payload = _build()
    expected_names = [
        "README.md",
        "bundle_receipt.json",
        "matrix.json",
        "tokens.csv",
    ]
    assert HISTORICAL_R15_BUNDLE.is_dir()
    assert sorted(path.name for path in HISTORICAL_R15_BUNDLE.iterdir()) == expected_names
    assert {
        name: _sha256(HISTORICAL_R15_BUNDLE / name) for name in expected_names
    } == {
        "README.md": "2cdab8f5419ee4fbcba2bc941d36cd8b08146ac8f5899641e7ec6c79c0dfd67a",
        "bundle_receipt.json": (
            "b631cf6be5d3eaca153fa61d67d956bdea2120592f6ae9f0cae5c9be7d9dc1e9"
        ),
        "matrix.json": "292b7f94d4e98a4594835a734a0cb6815df4925f2168797cd9c1f7cc0f2542d4",
        "tokens.csv": "ce2a8c2d04ad3b16fd4453bc0a08ea9728b5bd8343488f8998baa8492254dc13",
    }
    checked = json.loads(
        (HISTORICAL_R15_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert checked["matrix_identity_sha256"] == (
        "10dc189affb3aa1369294d1cfaa06fd7abb365246530290a9a3dae22b897de81"
    )
    assert checked["source_files"]["stage_contract"] == {
        "path": "configs/phaxis/v1_0/post_training_release_stage_contract_1_8.json",
        "sha256": _sha256(STAGE_CONTRACT),
    }
    assert checked["source_files"]["master"]["sha256"] == (
        "a535f1894025092bc02c742f3c593fa5770690471ae7572c8a05cd1973d0111b"
    )
    assert checked["source_files"]["publication_figure_builder"]["sha256"] == (
        "c7ec91bff90432ade7b8f9fb2a645ec4e6947cdbbb722f225d28ba4a50d7ca3e"
    )
    assert checked["matrix_identity_sha256"] != payload["matrix_identity_sha256"]


def test_checked_r16_is_the_exact_reproducible_current_bundle(tmp_path: Path) -> None:
    module, payload = _build()
    expected_names = [
        "README.md",
        "bundle_receipt.json",
        "matrix.json",
        "tokens.csv",
    ]
    assert CURRENT_R16_BUNDLE.is_dir()
    assert sorted(path.name for path in CURRENT_R16_BUNDLE.iterdir()) == expected_names
    assert {
        name: _sha256(CURRENT_R16_BUNDLE / name) for name in expected_names
    } == {
        "README.md": "3a233f99f3730bc7e0f0fd58670440ce30e6f4d022b1865230c68af2018c4d4e",
        "bundle_receipt.json": (
            "293259f1b0aa511718de889a181e0be80837c34ae6764c995948adfd7c0ec398"
        ),
        "matrix.json": "8cb447f573c1e94ed29307177421bf1d64ebcfc0f149ef1931961297d742e5da",
        "tokens.csv": "ce2a8c2d04ad3b16fd4453bc0a08ea9728b5bd8343488f8998baa8492254dc13",
    }

    regenerated = tmp_path / "regenerated-r16"
    module.write_bundle(regenerated, payload)
    for name in expected_names:
        assert (regenerated / name).read_bytes() == (CURRENT_R16_BUNDLE / name).read_bytes()

    checked = json.loads(
        (CURRENT_R16_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert checked["matrix_identity_sha256"] == (
        "3c08ab5bcf1e23d30efaf6fd2a375fcddaf8d95a39ed25d9d2dcb41b8cc4f461"
    )
    assert checked["source_files"]["stage_contract"] == {
        "path": "configs/phaxis/v1_0/post_training_release_stage_contract_1_8.json",
        "sha256": _sha256(STAGE_CONTRACT),
    }
    assert checked["source_files"]["master"]["sha256"] == _sha256(MASTER)
    assert checked["source_files"]["matrix_builder"]["sha256"] == _sha256(SCRIPT)
    assert checked["source_files"]["publication_figure_builder"]["sha256"] == _sha256(
        PROJECT_ROOT / "scripts/phaxis/build_publication_figures.py"
    )
    assert checked["source_files"]["release_topology"]["sha256"] == _sha256(
        PROJECT_ROOT / "src/phaxis/release_topology.py"
    )
    assert checked["formal_scientific_result_receipt"] is False
    assert checked["scientific_values_present"] is False
    assert checked["gpu_program_started"] is False
    assert checked["blind_images_used"] == 0


def test_checked_r6_r8_r9_r10_r11_r12_and_r13_remain_immutable_historical_snapshots() -> None:
    _, payload = _build()
    assert HISTORICAL_R6_BUNDLE.is_dir()
    assert sorted(path.name for path in HISTORICAL_R6_BUNDLE.iterdir()) == [
        "README.md",
        "bundle_receipt.json",
        "matrix.json",
        "tokens.csv",
    ]
    checked = json.loads(
        (HISTORICAL_R6_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert checked["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_4.json"
    )
    assert payload["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_8.json"
    )
    assert checked["matrix_identity_sha256"] != payload["matrix_identity_sha256"]
    historical_r8 = json.loads(
        (HISTORICAL_R8_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert historical_r8["matrix_identity_sha256"] == (
        "72238f4f04e346694faa0baad30b802857ef4d21dc57cf75cc57a332c0799fcf"
    )
    assert historical_r8["matrix_identity_sha256"] != payload["matrix_identity_sha256"]
    historical_r9 = json.loads(
        (HISTORICAL_R9_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert historical_r9["matrix_identity_sha256"] == (
        "69f4f6e432c6841925cfcddaceaa93c3313e532b972711de8a35d472777a64d8"
    )
    assert historical_r9["matrix_identity_sha256"] != payload["matrix_identity_sha256"]
    historical_r10 = json.loads(
        (HISTORICAL_R10_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert historical_r10["matrix_identity_sha256"] == (
        "ba574813cfc3753f25a1d37723778cf2e6f6ab4dc6920264ad9ff911d56559ae"
    )
    assert historical_r10["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_5.json"
    )
    assert historical_r10["matrix_identity_sha256"] != payload["matrix_identity_sha256"]
    historical_r11 = json.loads(
        (HISTORICAL_R11_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert historical_r11["matrix_identity_sha256"] == (
        "688c63b626a67c041a5237440b9f5d7b104623521c3989a553e7793093b135d0"
    )
    assert historical_r11["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_6.json"
    )
    assert historical_r11["matrix_identity_sha256"] != payload["matrix_identity_sha256"]
    historical_r12 = json.loads(
        (HISTORICAL_R12_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert historical_r12["matrix_identity_sha256"] == (
        "0d1eb0f1921d075690a6f695790ba388ba866ffb7e53fcc67c6f35df178d17c3"
    )
    assert historical_r12["source_files"]["stage_contract"]["sha256"] == (
        "3019f41be108c970644682702af32fc5fa791add29dfd632af60cb009187ec3f"
    )
    assert historical_r12["matrix_identity_sha256"] != payload["matrix_identity_sha256"]
    historical_r13 = json.loads(
        (HISTORICAL_R13_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert historical_r13["matrix_identity_sha256"] == (
        "1b2770e31f7321930f9d2bc03a52c01040f6603116d93aae1a846a346ef6b8c0"
    )
    assert historical_r13["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_7.json"
    )
    assert historical_r13["matrix_identity_sha256"] != payload["matrix_identity_sha256"]


def test_checked_r14_remains_an_immutable_historical_snapshot() -> None:
    _, payload = _build()
    assert HISTORICAL_R14_BUNDLE.is_dir()
    expected_names = [
        "README.md",
        "bundle_receipt.json",
        "matrix.json",
        "tokens.csv",
    ]
    assert sorted(path.name for path in HISTORICAL_R14_BUNDLE.iterdir()) == expected_names
    assert {
        name: _sha256(HISTORICAL_R14_BUNDLE / name) for name in expected_names
    } == {
        "README.md": "6fc75fd28082b27b3d009c9778bc5ea46b338f07468e5f341d45ff4eedd020ee",
        "bundle_receipt.json": (
            "bad53a8474ef5044f63ab0c46a2852a713f4cd3745458bff6a536bb483999613"
        ),
        "matrix.json": "5ec03bf50a6e0f3fdcc445ee7abd407034be623dbf7d7248c8af3d3349c33a13",
        "tokens.csv": "59085cc17c7285b44c5b83d09247543e3f1a757232b3e01d15e27bbd9631a720",
    }

    checked = json.loads(
        (HISTORICAL_R14_BUNDLE / "matrix.json").read_text(encoding="utf-8")
    )
    assert checked["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_7.json"
    )
    assert checked["matrix_identity_sha256"] == (
        "4bcb382c100ba1adf19ea0d76b15f7f6996862296a073a37e6511fcd95a8466d"
    )
    assert payload["source_files"]["stage_contract"]["path"].endswith(
        "post_training_release_stage_contract_1_8.json"
    )
    assert checked["matrix_identity_sha256"] != payload["matrix_identity_sha256"]


def test_matrix_identity_detects_any_route_or_source_change() -> None:
    _, payload = _build()
    original_identity = payload["matrix_identity_sha256"]
    unsigned = deepcopy(payload)
    unsigned.pop("matrix_identity_sha256")
    assert sha256_json(unsigned) == original_identity

    altered = deepcopy(unsigned)
    altered["tokens"][0]["source_evidence_stages"] = ["benchmark_same_hardware"]
    assert sha256_json(altered) != original_identity


def test_main_table_headings_and_machine_token_subsets_are_closed() -> None:
    _, payload = _build()
    table_by_number = {row["number"]: row for row in payload["main_tables"]}
    assert table_by_number["1"]["machine_token_count"] == 0
    assert table_by_number["1"]["figure_resource_roles"] == ["trait_contract"]
    assert table_by_number["2"]["machine_token_count"] > 20
    assert "FINAL_HAIR_F1_20UM" in table_by_number["2"]["machine_tokens"]
    assert "FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE" in table_by_number["2"]["machine_tokens"]
    assert table_by_number["3"]["machine_token_count"] == 53
    assert "FINAL_ABUNDANCE_CONSTRUCT_RATIO" in table_by_number["3"]["machine_tokens"]
    assert {
        "FINAL_D15_ABUNDANCE_PATTERN",
        "FINAL_D15_LENGTH_PATTERN",
        "FINAL_FIRST_HAIR_PATTERN",
        "FINAL_ROOT_WIDTH_PATTERN",
        "FINAL_ROOT_LENGTH_PATTERN",
    } <= set(table_by_number["3"]["machine_tokens"])
    assert "PHAXIS_RELEASE_DOI" not in table_by_number["3"]["machine_tokens"]
    for number, title in TABLE_STATIC_TITLES.items():
        assert table_by_number[str(number)]["title"] == title
    assert table_by_number["3"]["title"] == (
        "⟦RESULT SLOT → publication_title_contract.tables.3⟧"
    )

    master = MASTER.read_text(encoding="utf-8")
    assert len(re.findall(r"^### Table [1-3]\. ", master, re.MULTILINE)) == 3
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    assert len(re.findall(r"^### Table S(?:[1-9]|10)\. ", supplement, re.MULTILINE)) == 10


def test_main_figure_titles_come_from_the_single_publication_authority() -> None:
    _, payload = _build()
    by_number = {row["number"]: row for row in payload["main_figures"]}
    for number, title in FIGURE_STATIC_TITLES.items():
        assert by_number[str(number)]["title"] == title
    assert by_number["5"]["title"] == (
        "⟦RESULT SLOT → publication_title_contract.figures.5⟧"
    )
    authority = payload["source_files"]["publication_title_authority"]
    assert authority["path"] == "src/phaxis/publication_titles.py"
    assert authority["sha256"] == _sha256(
        PROJECT_ROOT / "src/phaxis/publication_titles.py"
    )
