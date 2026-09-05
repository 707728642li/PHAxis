from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "phaxis"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_manuscript_evidence_manifest import (  # noqa: E402
    build_manuscript_evidence_manifest,
)
import promote_model_contract as promotion_module  # noqa: E402
import source_release_common as source_release_policy  # noqa: E402
from phaxis.publication_evidence import figure_suite_identity_preimage  # noqa: E402
from phaxis.supplementary_tables import (  # noqa: E402
    BUNDLE_RECEIPT as SUPPLEMENTARY_TABLE_BUNDLE_RECEIPT,
    FINAL_STATUS as FINAL_SUPPLEMENTARY_TABLE_STATUS,
    TABLE_SPECS as SUPPLEMENTARY_TABLE_SPECS,
    materialize_supplementary_table_data_bundle,
)
from phaxis.contracts import ContractError  # noqa: E402
from phaxis.evaluation_metrics import (  # noqa: E402
    biological_hair_presence_matcher_contract,
)
from phaxis.io import sha256_file, sha256_json  # noqa: E402
from phaxis.model_contract_binding import (  # noqa: E402
    APPLIED_OFFICIAL_LIFECYCLE,
    read_model_contract_authority,
    read_model_contract_proposal,
)
from promote_model_contract import (  # noqa: E402
    PromotionError,
    apply_model_contract_promotion,
    build_model_contract_proposal,
    recover_application_receipt,
)
from test_manuscript_evidence_manifest import (  # noqa: E402
    _canonical_hash,
    _file_hash,
    _fixture,
    _fixture_legacy_prediction_identity,
    _kwargs,
    _seal,
    _write,
)


def test_promotion_cli_is_directly_runnable_from_an_isolated_checkout() -> None:
    """The release/promotion entry point must not need an installed PHAxis."""

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT_ROOT / "promote_model_contract.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "--current-model-contract" in result.stdout


def _synthetic_hybrid_lock_identity() -> str:
    records = [
        {
            "task_id": f"QCDEV-{index:02d}",
            "sha256": _canonical_hash(["hybrid", f"QCDEV-{index:02d}"]),
        }
        for index in range(44)
    ]
    return _canonical_hash(records)


@pytest.fixture(autouse=True)
def _use_synthetic_hybrid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a synthetic lock set without weakening the production fixed identity."""

    identity = _synthetic_hybrid_lock_identity()
    monkeypatch.setattr(
        source_release_policy,
        "LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256",
        identity,
    )
    monkeypatch.setattr(
        sys.modules["build_manuscript_evidence_manifest"],
        "LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256",
        identity,
    )


def _current_contract(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "model_contract.json",
        {
            "schema_version": "PHAxis-model-contract-1.0.0",
            "product": "PHAxis",
            "product_version": "1.0.0",
            "model_bundle_id": "DEVELOPMENT-CANDIDATE",
            "formal_release_status": "blocked_pending_strict_train399",
            "legacy_sentinel": "MUST_NOT_SURVIVE_PROMOTION",
            "expert_boundary": {"hair_identity_and_count": "legacy-443CV"},
            "root_expert": {
                "root_cap_region_output": False,
                "final_candidate_registry": "stale/legacy/provider.json",
                "final_candidate_registry_sha256": "a" * 64,
            },
            "hair_identity_count_expert": {
                "current_checkpoint_role": (
                    "development_algorithm_candidate_only; not formal PHAxis "
                    "deployment weights"
                ),
                "training_scope": "five family-grouped folds spanning all 443 images",
                "each_oof_scored_image_excluded_from_its_scoring_model": True,
                "checkpoint_sha256_in_fold_order": [
                    "a36c48802a2ed1120602319dc9e6c6d386cc64d87d90dacd421a24d77faafd35",
                    "de3d32e99c65e4c9d9a785b974aadc8b1cde8ae15f90644ee8af28102466ab41",
                    "d6dfe0b245fbe1c9af8ad56f153ad59d0941b8c9e13a452c8ef4de88ec868311",
                    "342271324c4b3d6c3a149133747b512ff4955d2f397fc399dae5bdc0fa364e6a",
                    "cc09a97c81cba2cc33f3c8269a8332afc2c915821b3b5805cfb03c87879b0d5",
                ],
                "legacy_fusion_prediction_path": (
                    "outputs/rhaxis_nextgen_hybrid_max/predictions"
                ),
                "working_um_per_px": 2.0,
                "output_stride": 2,
                "window": 1024,
                "overlap": 256,
                "batch": 4,
                "nms_kernel": 5,
                "horizontal_flip_tta": True,
                "use_trace": False,
                "root_gate_um": [-90.0, 25.0],
                "maximum_instances": 4000,
                "precision_mode": "fp32_locked",
            },
            "development_evidence": {
                "qcdev44": {
                    "evaluation_path": "stale/legacy/qcdev44.json",
                    "strict_presence_f1_at_20um": -999.0,
                },
                "oof443": {"legacy_sentinel": "MUST_NOT_SURVIVE_PROMOTION"},
                "oof443_stratified": {
                    "path": "stale/legacy/oof443.json",
                    "sha256": "c" * 64,
                },
            },
            "data_contract": {},
            "red_lines": {
                "blind_images_used": 0,
                "formal_train399_only_stageb_weights_available": False,
            },
        },
    )


def _metric(tp: int, predicted: int = 110, truth: int = 100) -> dict[str, object]:
    precision = tp / predicted
    recall = tp / truth
    return {
        "tp": tp,
        "n_pred": predicted,
        "n_gt": truth,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall),
    }


def _upgrade_evaluation_1_2(paths: dict[str, Path]) -> dict[str, object]:
    evaluation_path = paths["train399_evaluation"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["schema_version"] = (
        "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
    )
    evaluation["scope"] = (
        "locked overlay-visible QC-development44; not independent accuracy"
    )
    primary_matcher = biological_hair_presence_matcher_contract()
    evaluation["metric_hierarchy"] = {
        "primary": (
            "one-to-one tolerant biological-hair presence; bidirectional "
            "partial centreline coverage without endpoint gates"
        ),
        "primary_minimum_truth_coverage": 0.25,
        "primary_minimum_prediction_coverage": 0.25,
        "primary_minimum_direction_cosine": 0.0,
        "secondary": (
            "attachment/base identity; strict whole-line correspondence; "
            "distal endpoint and length geometry"
        ),
        "primary_tolerance_um": 20.0,
        "primary_matcher_contract": primary_matcher,
        "primary_matcher_contract_sha256": sha256_json(primary_matcher),
        "coordinate_evaluation": "physical_um_after_per_axis_realized_resize_correction",
    }

    def expert(offset: int) -> dict[str, object]:
        return {
            "images": 44,
            "predicted_hairs": 110 - offset,
            "ground_truth_hairs": 100,
            "tolerant_biological_presence": {
                "5": _metric(70 - offset, 110 - offset),
                "10": _metric(80 - offset, 110 - offset),
                "20": _metric(90 - offset, 110 - offset),
            },
            "identity_attachment_proxy": {
                "5": _metric(65 - offset, 110 - offset),
                "10": _metric(75 - offset, 110 - offset),
                "20": _metric(85 - offset, 110 - offset),
            },
            "strict_whole_line_correspondence": {
                "5": _metric(60 - offset, 110 - offset),
                "10": _metric(70 - offset, 110 - offset),
                "20": _metric(80 - offset, 110 - offset),
            },
            "count": {
                "mae": 3.25 + offset,
                "bias": 0.5 + offset,
                "pearson_r": 0.91 - offset / 100.0,
                "ccc": 0.89 - offset / 100.0,
            },
        }

    evaluation["overall"] = {
        "stageb_train399": expert(0),
        "hybrid_max": expert(10),
    }
    stageb_primary = evaluation["overall"]["stageb_train399"][
        "tolerant_biological_presence"
    ]["20"]
    hybrid_primary = evaluation["overall"]["hybrid_max"][
        "tolerant_biological_presence"
    ]["20"]
    evaluation["delta_stageb_train399_minus_hybrid"] = {
        "biological_presence_f1_20um": (
            stageb_primary["f1"] - hybrid_primary["f1"]
        ),
        "identity_f1_20um": 0.1,
        "strict_f1_20um": 0.1,
        "count_mae": (
            evaluation["overall"]["stageb_train399"]["count"]["mae"]
            - evaluation["overall"]["hybrid_max"]["count"]["mae"]
        ),
        "count_bias": -10.0,
        "count_ccc": 0.1,
    }
    interval = {"lower_2_5": 0.7, "upper_97_5": 0.9}
    bootstrap_metrics = {
        "biological_presence_f1_20um": dict(interval),
        "identity_f1_20um": dict(interval),
        "count_mae": {"lower_2_5": 2.0, "upper_97_5": 5.0},
        "count_ccc": dict(interval),
    }
    evaluation["paired_bootstrap_95ci"] = {
        "method": "paired image-level nonparametric bootstrap",
        "repetitions": 10_000,
        "seed": 20260828,
        "experts": {
            "stageb_train399": bootstrap_metrics,
            "hybrid_max": deepcopy(bootstrap_metrics),
        },
        "delta_stageb_train399_minus_hybrid": deepcopy(bootstrap_metrics),
    }
    evaluation["per_image"] = [
        {
            "task_id": f"QCDEV-{index:02d}",
            "stageb_train399": {
                "n_pred": 3,
                "n_gt": 3,
                "biological_presence_tp": {"5.0": 1, "10.0": 2, "20.0": 3},
            },
            "hybrid_max": {
                "n_pred": 3,
                "n_gt": 3,
                "biological_presence_tp": {"5.0": 1, "10.0": 2, "20.0": 2},
            },
        }
        for index in range(44)
    ]
    stageb_locks = [
        {"task_id": row["task_id"], "sha256": _canonical_hash(["stageb", row["task_id"]])}
        for row in evaluation["per_image"]
    ]
    hybrid_locks = [
        {"task_id": row["task_id"], "sha256": _canonical_hash(["hybrid", row["task_id"]])}
        for row in evaluation["per_image"]
    ]
    evaluation["prediction_input_locks"] = {
        "stageb_detection_files": stageb_locks,
        "stageb_detection_set_identity_sha256": _canonical_hash(stageb_locks),
        "hybrid_prediction_files": hybrid_locks,
        "hybrid_prediction_set_identity_sha256": _canonical_hash(hybrid_locks),
    }
    evaluation_summary_sha256 = _canonical_hash("evaluation-inference-summary-file")
    evaluation_summary_identity = _canonical_hash(
        "evaluation-inference-summary-identity"
    )
    evaluation_gate_identity = _canonical_hash("evaluation-inference-gate")
    evaluation["evaluation_inference_authority"] = {
        "schema_version": (
            "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0"
        ),
        "artifact_role": (
            "locked_qcdevelopment44_full_geometry_evaluation_only_not_deployable"
        ),
        "evaluation_detection_schema_version": (
            "PHAxis-StageB-train399-QCdev44-evaluation-only-full-geometry-1.0"
        ),
        "evaluation_inference_summary_sha256": evaluation_summary_sha256,
        "evaluation_inference_summary_identity_sha256": evaluation_summary_identity,
        "evaluation_gate_identity_sha256": evaluation_gate_identity,
        "evaluation_detection_set_identity_sha256": _canonical_hash(stageb_locks),
        "model_contract_proposal_required_for_artifact": False,
        "model_contract_proposal_present": False,
        "production_consumption_allowed": False,
        "fusion_consumption_allowed": False,
        "traits_consumption_allowed": False,
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    evaluation["inputs_sha256"][
        "evaluation_inference_summary"
    ] = evaluation_summary_sha256
    evaluation["training_contract"][
        "evaluation_gate_identity_sha256"
    ] = evaluation_gate_identity
    evaluation["training_contract"][
        "evaluation_inference_summary_identity_sha256"
    ] = evaluation_summary_identity
    evaluation["comparator_contract"] = {
        "hybrid_max": {
            "evidence_role": "locked_legacy_development_comparator",
            "schema_version": (
                "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
            ),
            "identity_hair_variant": "hybrid_verified_increment",
            "count_hair_variant": "hybrid_verified_increment",
            "endpoint_complete_identity_layer": True,
            "phaxis_payload_allowed": False,
            "stageb_identity_source_allowed": False,
            "prediction_set_identity_sha256": _canonical_hash(hybrid_locks),
            "expected_prediction_set_identity_sha256": _canonical_hash(
                hybrid_locks
            ),
        }
    }
    _write(evaluation_path, evaluation)
    return evaluation


def _checkpoint_files(tmp_path: Path) -> list[Path]:
    result = []
    for index in range(5):
        path = tmp_path / f"checkpoint-{index}.pt"
        path.write_bytes(f"checkpoint-{index}".encode("utf-8"))
        result.append(path)
    return result


def _proposal_kwargs(
    *, current: Path, paths: dict[str, Path], checkpoints: list[Path], output: Path
) -> dict:
    return {
        "current_model_contract": current,
        "train399_candidate": paths["train399_candidate"],
        "train399_selection": paths["train399_selection"],
        "train399_evaluation": paths["train399_evaluation"],
        "root_exact283": paths["root_exact283"],
        "checkpoints": checkpoints,
        "output": output,
    }


def _rebind_final_receipts(paths: dict[str, Path], proposal_path: Path) -> None:
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    model_bundle_id = proposal["model_bundle_id"]
    root_expert_id = proposal["root_expert"]["expert_id"]
    binding = {
        "model_contract_proposal_sha256": _file_hash(proposal_path),
        "model_contract_proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
    }

    stageb = json.loads(paths["stageb"].read_text(encoding="utf-8"))
    stageb.pop("summary_identity_sha256")
    stageb.update(binding)
    stageb["model_bundle_id"] = model_bundle_id
    stageb["root_expert_id"] = root_expert_id
    _seal(stageb, "summary_identity_sha256")
    _write(paths["stageb"], stageb)

    fusion = json.loads(paths["fusion"].read_text(encoding="utf-8"))
    fusion.pop("summary_identity_sha256")
    fusion.update(binding)
    fusion["model_bundle_id"] = model_bundle_id
    fusion["root_expert"] = root_expert_id
    fusion["source_stageb_summary_sha256"] = _file_hash(paths["stageb"])
    _seal(fusion, "summary_identity_sha256")
    _write(paths["fusion"], fusion)

    traits = json.loads(paths["traits"].read_text(encoding="utf-8"))
    traits.pop("export_identity_sha256")
    traits.update(binding)
    traits["model_bundle_id"] = model_bundle_id
    traits["root_expert_id"] = root_expert_id
    _seal(traits, "export_identity_sha256")
    _write(paths["traits"], traits)

    cohorts = json.loads(paths["cohorts"].read_text(encoding="utf-8"))
    cohorts.pop("cohort_build_identity_sha256")
    cohorts.update(binding)
    cohorts["model_bundle_id"] = model_bundle_id
    cohorts["root_expert_id"] = root_expert_id
    cohorts["input_sha256"]["trait_export_summary"] = _file_hash(paths["traits"])
    _seal(cohorts, "cohort_build_identity_sha256")
    _write(paths["cohorts"], cohorts)

    analysis = json.loads(paths["analysis"].read_text(encoding="utf-8"))
    analysis.pop("analysis_identity_sha256")
    analysis.update(binding)
    analysis["model_bundle_id"] = model_bundle_id
    analysis["root_expert_id"] = root_expert_id
    analysis["cohort_build_summary_sha256"] = _file_hash(paths["cohorts"])
    _seal(analysis, "analysis_identity_sha256")
    _write(paths["analysis"], analysis)

    profiles = json.loads(paths["profiles"].read_text(encoding="utf-8"))
    profiles.pop("export_identity_sha256")
    profiles.update(binding)
    profiles["model_bundle_id"] = model_bundle_id
    profiles["root_expert_id"] = root_expert_id
    _seal(profiles, "export_identity_sha256")
    _write(paths["profiles"], profiles)

    source_summary = {
        "train399_evaluation": _file_hash(paths["train399_evaluation"]),
        "root_exact283": _file_hash(paths["root_exact283"]),
        **{
            role: _file_hash(paths[role])
            for role in (
                "stageb",
                "fusion",
                "traits",
                "cohorts",
                "analysis",
                "profiles",
            )
        },
    }
    public_identity = {
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_expert_id,
        "root_provider_role": proposal["root_expert"]["provider_role"],
    }
    evaluation = json.loads(
        paths["train399_evaluation"].read_text(encoding="utf-8")
    )
    prediction_locks = evaluation["prediction_input_locks"]
    evaluation_authority = evaluation["evaluation_inference_authority"]
    comparator = evaluation["comparator_contract"]["hybrid_max"]
    prediction_provenance = {
        "task_order_identity_sha256": _canonical_hash(
            [row["task_id"] for row in evaluation["per_image"]]
        ),
        "stageb_train399": {
            "schema_version": evaluation_authority[
                "evaluation_detection_schema_version"
            ],
            "artifact_role": evaluation_authority["artifact_role"],
            "evaluation_inference_summary_sha256": evaluation_authority[
                "evaluation_inference_summary_sha256"
            ],
            "evaluation_inference_summary_identity_sha256": evaluation_authority[
                "evaluation_inference_summary_identity_sha256"
            ],
            "evaluation_gate_identity_sha256": evaluation_authority[
                "evaluation_gate_identity_sha256"
            ],
            "production_consumption_allowed": False,
            "fusion_consumption_allowed": False,
            "traits_consumption_allowed": False,
            "ordered_file_set_identity_sha256": prediction_locks[
                "stageb_detection_set_identity_sha256"
            ],
        },
        "legacy_hybrid_endpoint_complete_identity_layer": {
            **comparator,
            "ordered_file_set_identity_sha256": prediction_locks[
                "hybrid_prediction_set_identity_sha256"
            ],
        },
    }
    figure_inputs = json.loads(
        paths["figure_inputs"].read_text(encoding="utf-8")
    )
    figure_inputs.pop("figure_input_assembly_identity_sha256")
    figure_inputs.update(binding)
    figure_inputs["source_summary_sha256"] = source_summary
    figure_inputs["model_contract_public_identity"] = public_identity
    figure_inputs["model_bundle_id"] = model_bundle_id
    figure_inputs["root_expert_id"] = root_expert_id
    figure_inputs["hair_identity_expert_id"] = stageb[
        "detection_model_metadata"
    ]["expert_id"]
    figure_inputs["train399_prediction_input_provenance"] = prediction_provenance
    _seal(figure_inputs, "figure_input_assembly_identity_sha256")
    _write(paths["figure_inputs"], figure_inputs)

    supplementary_source_paths: dict[str, Path] = {}
    for spec in SUPPLEMENTARY_TABLE_SPECS:
        for authority_role in spec["source_roles"]:
            authority_kind, role = authority_role.split("/", 1)
            if authority_kind == "source":
                record = figure_inputs["source_inputs"][role]
                supplementary_source_paths[authority_role] = (
                    paths["figure_inputs"].parent / record["path"]
                ).resolve()
            elif authority_kind == "resource":
                record = figure_inputs["resources"][role]
                supplementary_source_paths[authority_role] = (
                    paths["figure_inputs"].parent / record["path"]
                ).resolve()
            elif authority_kind == "receipt":
                supplementary_source_paths[authority_role] = paths[role]
            elif authority_kind == "proposal":
                supplementary_source_paths[authority_role] = proposal_path
            else:  # pragma: no cover - fixed exact-ten test contract
                raise AssertionError(authority_role)
    promoted_table_directory = "supplementary_tables_and_data_promoted"
    supplementary_table_bundle = materialize_supplementary_table_data_bundle(
        output=paths["figures"].parent / promoted_table_directory,
        status=FINAL_SUPPLEMENTARY_TABLE_STATUS,
        source_paths=supplementary_source_paths,
        source_identities={},
        figure_input_manifest_sha256=_file_hash(paths["figure_inputs"]),
        figure_input_assembly_identity_sha256=figure_inputs[
            "figure_input_assembly_identity_sha256"
        ],
        model_contract_proposal_identity_sha256=proposal[
            "model_contract_identity_sha256"
        ],
    )

    figures = json.loads(paths["figures"].read_text(encoding="utf-8"))
    figures.update(binding)
    figures["source_summary_sha256"] = source_summary
    figures["model_contract_public_identity"] = public_identity
    figures["model_bundle_id"] = model_bundle_id
    figures["root_expert_id"] = root_expert_id
    figures["figure_input_manifest_sha256"] = _file_hash(paths["figure_inputs"])
    figures["figure_input_assembly_identity_sha256"] = figure_inputs[
        "figure_input_assembly_identity_sha256"
    ]
    figures["train399_prediction_input_provenance"] = prediction_provenance
    figures["supplementary_tables"] = supplementary_table_bundle["items"]
    figures["supplementary_table_bundle_receipt"] = (
        f"{promoted_table_directory}/{SUPPLEMENTARY_TABLE_BUNDLE_RECEIPT}"
    )
    figures["supplementary_table_bundle_receipt_sha256"] = (
        supplementary_table_bundle["receipt_sha256"]
    )
    figures["supplementary_table_bundle_identity_sha256"] = (
        supplementary_table_bundle["bundle_identity_sha256"]
    )
    figures["supplementary_table_bundle_sha256"] = supplementary_table_bundle[
        "bundle_file_sha256"
    ]
    figures["supplementary_table_source_input_sha256"] = {
        role: record["sha256"]
        for role, record in figure_inputs["source_inputs"].items()
    }
    figures["supplementary_table_source_authority_sha256"] = (
        supplementary_table_bundle["source_authority_sha256"]
    )
    figures["supplementary_table_source_authority_identity"] = (
        supplementary_table_bundle["source_authority_identity"]
    )
    # Promotion rebinds the eight formal receipts to the proposal-owned public
    # model identity.  Every supplementary S1--S9 record must therefore follow
    # those new receipt bytes as well; retaining the pre-proposal fixture hashes
    # would correctly fail the publication evidence gate.
    for supplementary_record in figures["supplementary_figures"].values():
        supplementary_record["receipt_file_sha256"] = {
            role: source_summary[role]
            for role in supplementary_record["receipt_roles"]
        }
    figures["figure_suite_identity_sha256"] = _canonical_hash(
        figure_suite_identity_preimage(
            status="final",
            figure_hashes=figures["figure_bundle_sha256"],
            source_hashes=source_summary,
            figure_input_assembly_identity_sha256=figure_inputs[
                "figure_input_assembly_identity_sha256"
            ],
            model_contract_proposal_identity_sha256=proposal[
                "model_contract_identity_sha256"
            ],
            model_contract_public_identity=public_identity,
                train399_prediction_input_provenance=figures[
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
    _write(paths["figures"], figures)
    paths["model_contract_proposal"] = proposal_path


def _closed_fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _fixture(tmp_path)
    _upgrade_evaluation_1_2(paths)
    current = _current_contract(tmp_path)
    checkpoints = _checkpoint_files(tmp_path)
    proposal_path = tmp_path / "model_contract.proposal.json"
    proposal = build_model_contract_proposal(
        **_proposal_kwargs(
            current=current,
            paths=paths,
            checkpoints=checkpoints,
            output=proposal_path,
        )
    )
    _rebind_final_receipts(paths, proposal_path)
    evidence_path = tmp_path / "manuscript_evidence.json"
    build_manuscript_evidence_manifest(**_kwargs(paths, evidence_path))
    return {
        "paths": paths,
        "current": current,
        "checkpoints": checkpoints,
        "proposal": proposal_path,
        "proposal_payload": proposal,
        "evidence": evidence_path,
        "expected_current_sha256": _file_hash(current),
    }


def _apply_kwargs(fixture: dict, receipt: Path) -> dict:
    paths = fixture["paths"]
    return {
        "current_model_contract": fixture["current"],
        "expected_current_sha256": fixture["expected_current_sha256"],
        "proposal": fixture["proposal"],
        "train399_candidate": paths["train399_candidate"],
        "train399_selection": paths["train399_selection"],
        "train399_evaluation": paths["train399_evaluation"],
        "root_exact283": paths["root_exact283"],
        "checkpoints": fixture["checkpoints"],
        "stageb_summary": paths["stageb"],
        "fusion_summary": paths["fusion"],
        "traits_summary": paths["traits"],
        "manuscript_evidence_manifest": fixture["evidence"],
        "application_receipt": receipt,
    }


def test_default_writes_unapplied_deterministic_proposal_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evaluation = _upgrade_evaluation_1_2(paths)
    current = _current_contract(tmp_path)
    checkpoints = _checkpoint_files(tmp_path)
    before = current.read_bytes()
    first = tmp_path / "proposal-a.json"
    second = tmp_path / "proposal-b.json"
    one = build_model_contract_proposal(
        **_proposal_kwargs(current=current, paths=paths, checkpoints=checkpoints, output=first)
    )
    two = build_model_contract_proposal(
        **_proposal_kwargs(current=current, paths=paths, checkpoints=checkpoints, output=second)
    )
    assert one == two
    assert first.read_bytes() == second.read_bytes()
    assert current.read_bytes() == before
    assert one["formal_release_status"] == "passed_proposal_not_official"
    assert one["promotion"]["official_apply_performed"] is False
    serialized = json.dumps(one, ensure_ascii=False)
    assert "MUST_NOT_SURVIVE_PROMOTION" not in serialized
    assert "checkpoint_sha256_in_fold_order" not in serialized
    assert "a36c48802a2ed1120602319dc9e6c6d386cc64d87d90dacd421a24d77faafd35" not in serialized
    assert "development_algorithm_candidate_only" not in serialized
    assert "rhaxis_nextgen_hybrid_max/predictions" not in serialized
    assert "each_oof_scored_image_excluded_from_its_scoring_model" not in serialized
    assert "stale/legacy" not in serialized
    assert set(one["development_evidence"]) == {"qcdev44"}
    qcdev44 = one["development_evidence"]["qcdev44"]
    acceptance = qcdev44["development_acceptance_gate"]
    assert acceptance["gate_pass"] is True
    assert acceptance["precision_noninferiority_margin_absolute"] == 0.0
    assert all(acceptance["passed"].values())
    assert "not an independent accuracy" in acceptance["scope"]
    assert qcdev44["metric_hierarchy"] == evaluation["metric_hierarchy"]
    assert qcdev44["stageb_train399"] == evaluation["overall"]["stageb_train399"]
    assert qcdev44["paired_bootstrap_95ci"] == evaluation["paired_bootstrap_95ci"]
    assert qcdev44["paired_bootstrap_95ci"][
        "delta_stageb_train399_minus_hybrid"
    ]["biological_presence_f1_20um"] == evaluation["paired_bootstrap_95ci"][
        "delta_stageb_train399_minus_hybrid"
    ]["biological_presence_f1_20um"]
    assert qcdev44["prediction_input_locks"] == evaluation[
        "prediction_input_locks"
    ]
    comparator = qcdev44["same_run_historical_endpoint_complete_comparator"][
        "source_prediction_contract"
    ]
    assert comparator["schema_version"] == (
        "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
    )
    assert comparator["identity_hair_variant"] == "hybrid_verified_increment"
    assert comparator["count_hair_variant"] == "hybrid_verified_increment"
    assert comparator["prediction_set_identity_sha256"] == evaluation[
        "prediction_input_locks"
    ]["hybrid_prediction_set_identity_sha256"]
    assert comparator["expected_prediction_set_identity_sha256"] == evaluation[
        "prediction_input_locks"
    ]["hybrid_prediction_set_identity_sha256"]
    assert qcdev44["source"]["evaluation_sha256"] == _file_hash(
        paths["train399_evaluation"]
    )
    assert qcdev44["source"]["evaluation_content_identity_sha256"] == _canonical_hash(
        evaluation
    )
    assert "final_candidate_registry" not in one["root_expert"]
    unsigned = dict(one)
    identity = unsigned.pop("model_contract_identity_sha256")
    assert identity == _canonical_hash(unsigned)


def test_proposal_rejects_evaluator_without_per_image_biological_presence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    evaluation = _upgrade_evaluation_1_2(paths)
    del evaluation["per_image"][0]["stageb_train399"]["biological_presence_tp"]
    _write(paths["train399_evaluation"], evaluation)
    current = _current_contract(tmp_path)
    with pytest.raises(PromotionError, match="train399_evaluation_metric_contract"):
        build_model_contract_proposal(
            **_proposal_kwargs(
                current=current,
                paths=paths,
                checkpoints=_checkpoint_files(tmp_path),
                output=tmp_path / "new-proposal.json",
            )
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("remove_authority", "evaluation-only inference authority is absent"),
        ("make_deployable", "evaluation-only inference is circular or deployable"),
        ("detach_summary_hash", "source hashes do not bind"),
    ),
)
def test_proposal_requires_sealed_nonproduction_evaluation_inference_authority(
    tmp_path: Path, mutation: str, message: str
) -> None:
    paths = _fixture(tmp_path)
    evaluation = _upgrade_evaluation_1_2(paths)
    if mutation == "remove_authority":
        del evaluation["evaluation_inference_authority"]
    elif mutation == "make_deployable":
        evaluation["evaluation_inference_authority"][
            "fusion_consumption_allowed"
        ] = True
    elif mutation == "detach_summary_hash":
        evaluation["inputs_sha256"]["evaluation_inference_summary"] = "f" * 64
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    _write(paths["train399_evaluation"], evaluation)
    with pytest.raises(PromotionError, match=message):
        build_model_contract_proposal(
            **_proposal_kwargs(
                current=_current_contract(tmp_path),
                paths=paths,
                checkpoints=_checkpoint_files(tmp_path),
                output=tmp_path / "new-proposal.json",
            )
        )


@pytest.mark.parametrize(
    "mutation",
    ("f1_not_improved", "recall_not_improved", "precision_collapsed", "mae_not_improved"),
)
def test_proposal_enforces_directional_qcdev44_development_acceptance_gate(
    tmp_path: Path, mutation: str
) -> None:
    paths = _fixture(tmp_path)
    evaluation = _upgrade_evaluation_1_2(paths)
    stageb = evaluation["overall"]["stageb_train399"]
    hybrid = evaluation["overall"]["hybrid_max"]
    if mutation in {"recall_not_improved", "precision_collapsed"}:
        new_predicted = 90 if mutation == "recall_not_improved" else 130
        stageb["predicted_hairs"] = new_predicted
        for family in (
            "tolerant_biological_presence",
            "identity_attachment_proxy",
            "strict_whole_line_correspondence",
        ):
            for tolerance, record in stageb[family].items():
                tp = int(record["tp"])
                if family == "tolerant_biological_presence" and tolerance == "20":
                    tp = 80 if mutation == "recall_not_improved" else 100
                stageb[family][tolerance] = _metric(
                    tp, predicted=new_predicted, truth=100
                )
    elif mutation == "f1_not_improved":
        stageb["tolerant_biological_presence"]["20"] = _metric(
            84, predicted=110, truth=100
        )
    elif mutation == "mae_not_improved":
        stageb["count"]["mae"] = hybrid["count"]["mae"]
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    evaluation["delta_stageb_train399_minus_hybrid"][
        "biological_presence_f1_20um"
    ] = (
        stageb["tolerant_biological_presence"]["20"]["f1"]
        - hybrid["tolerant_biological_presence"]["20"]["f1"]
    )
    evaluation["delta_stageb_train399_minus_hybrid"]["count_mae"] = (
        stageb["count"]["mae"] - hybrid["count"]["mae"]
    )
    _write(paths["train399_evaluation"], evaluation)
    with pytest.raises(
        PromotionError, match="QCdevelopment44 development acceptance Gate failed"
    ):
        build_model_contract_proposal(
            **_proposal_kwargs(
                current=_current_contract(tmp_path),
                paths=paths,
                checkpoints=_checkpoint_files(tmp_path),
                output=tmp_path / "new-proposal.json",
            )
        )


def test_proposal_rejects_stale_prediction_file_lock(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evaluation = _upgrade_evaluation_1_2(paths)
    evaluation["prediction_input_locks"]["hybrid_prediction_files"][0][
        "sha256"
    ] = _canonical_hash("STALE_FUSED_PREDICTION_SENTINEL")
    _write(paths["train399_evaluation"], evaluation)
    with pytest.raises(PromotionError, match="train399_evaluation_metric_contract"):
        build_model_contract_proposal(
            **_proposal_kwargs(
                current=_current_contract(tmp_path),
                paths=paths,
                checkpoints=_checkpoint_files(tmp_path),
                output=tmp_path / "new-proposal.json",
            )
        )


def test_proposal_rejects_self_consistent_but_wrong_hybrid_prediction_set(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    evaluation = _upgrade_evaluation_1_2(paths)
    locks = evaluation["prediction_input_locks"]["hybrid_prediction_files"]
    locks[0]["sha256"] = _canonical_hash("DIFFERENT_COMPLETE_HYBRID_SET")
    wrong_identity = _canonical_hash(locks)
    evaluation["prediction_input_locks"][
        "hybrid_prediction_set_identity_sha256"
    ] = wrong_identity
    comparator = evaluation["comparator_contract"]["hybrid_max"]
    comparator["prediction_set_identity_sha256"] = wrong_identity
    comparator["expected_prediction_set_identity_sha256"] = wrong_identity
    _write(paths["train399_evaluation"], evaluation)

    with pytest.raises(PromotionError, match="train399_evaluation_metric_contract"):
        build_model_contract_proposal(
            **_proposal_kwargs(
                current=_current_contract(tmp_path),
                paths=paths,
                checkpoints=_checkpoint_files(tmp_path),
                output=tmp_path / "new-proposal.json",
            )
        )


def test_apply_revalidates_and_cas_replaces_official_contract(tmp_path: Path) -> None:
    fixture = _closed_fixture(tmp_path)
    receipt_path = tmp_path / "promotion_application.json"
    proposal_bytes = fixture["proposal"].read_bytes()
    final, receipt = apply_model_contract_promotion(
        **_apply_kwargs(fixture, receipt_path)
    )
    assert final["formal_release_status"] == "passed"
    assert final["promotion"]["official_apply_performed"] is True
    assert json.loads(fixture["current"].read_text(encoding="utf-8")) == final
    assert receipt_path.is_file()
    assert receipt["status"] == "applied"
    assert fixture["proposal"].read_bytes() == proposal_bytes
    public_identity = {
        "model_bundle_id": final["model_bundle_id"],
        "root_expert_id": final["root_expert"]["expert_id"],
    }
    assert final["promotion"]["final_receipt_public_identity"] == {
        "fusion": public_identity,
        "traits": public_identity,
    }
    binding = read_model_contract_authority(fixture["current"])
    assert binding.authority_lifecycle == APPLIED_OFFICIAL_LIFECYCLE
    assert binding.authority_file_sha256 == sha256_file(fixture["current"])
    assert binding.authority_identity_sha256 == final[
        "model_contract_identity_sha256"
    ]
    assert binding.receipt_fields() == {
        "model_contract_proposal_sha256": final["promotion"][
            "proposal_file_sha256"
        ],
        "model_contract_proposal_identity_sha256": final["promotion"][
            "proposal_identity_sha256"
        ],
    }
    with pytest.raises(ContractError, match="not the passed, unapplied candidate"):
        read_model_contract_proposal(fixture["current"])
    final_serialized = json.dumps(final, ensure_ascii=False)
    for forbidden in (
        "a36c48802a2ed1120602319dc9e6c6d386cc64d87d90dacd421a24d77faafd35",
        "development_algorithm_candidate_only",
        "rhaxis_nextgen_hybrid_max/predictions",
        "checkpoint_sha256_in_fold_order",
        "oof443_stratified",
    ):
        assert forbidden not in final_serialized


def test_receipt_replace_crash_is_recoverable_from_official_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _closed_fixture(tmp_path)
    receipt_path = tmp_path / "promotion_application.json"
    real_replace = promotion_module.os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic derivative-receipt replace crash")
        return real_replace(source, destination)

    monkeypatch.setattr(promotion_module.os, "replace", fail_second_replace)
    with pytest.raises(PromotionError, match="official contract was atomically applied"):
        apply_model_contract_promotion(**_apply_kwargs(fixture, receipt_path))
    official = json.loads(fixture["current"].read_text(encoding="utf-8"))
    assert official["formal_release_status"] == "passed"
    assert official["promotion"]["official_apply_performed"] is True
    assert not receipt_path.exists()

    recovered = recover_application_receipt(
        applied_model_contract=fixture["current"], output=receipt_path
    )
    assert recovered["status"] == "applied"
    assert recovered["final_model_contract_sha256"] == _file_hash(fixture["current"])
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == recovered


@pytest.mark.parametrize(
    "missing_role",
    ("stageb_summary", "fusion_summary", "traits_summary", "manuscript_evidence_manifest"),
)
def test_apply_rejects_each_missing_final_receipt(tmp_path: Path, missing_role: str) -> None:
    fixture = _closed_fixture(tmp_path)
    kwargs = _apply_kwargs(fixture, tmp_path / "not-written.json")
    kwargs[missing_role] = tmp_path / f"missing-{missing_role}.json"
    with pytest.raises(PromotionError, match="file does not exist"):
        apply_model_contract_promotion(**kwargs)
    assert json.loads(fixture["current"].read_text(encoding="utf-8"))[
        "formal_release_status"
    ].startswith("blocked")


def test_apply_rejects_cross_hash_and_different_proposal(tmp_path: Path) -> None:
    fixture = _closed_fixture(tmp_path)
    paths = fixture["paths"]
    fusion = json.loads(paths["fusion"].read_text(encoding="utf-8"))
    fusion.pop("summary_identity_sha256")
    fusion["source_stageb_summary_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    _seal(fusion, "summary_identity_sha256")
    _write(paths["fusion"], fusion)
    with pytest.raises(PromotionError, match="source StageB SHA mismatch"):
        apply_model_contract_promotion(
            **_apply_kwargs(fixture, tmp_path / "cross-hash.json")
        )

    fixture = _closed_fixture(tmp_path / "other")
    other = json.loads(fixture["proposal"].read_text(encoding="utf-8"))
    other.pop("model_contract_identity_sha256")
    other["proposal_variant"] = "different-sealed-proposal"
    _seal(other, "model_contract_identity_sha256")
    other_path = _write(tmp_path / "other-proposal.json", other)
    kwargs = _apply_kwargs(fixture, tmp_path / "wrong-proposal.json")
    kwargs["proposal"] = other_path
    with pytest.raises(PromotionError, match="deterministic sanitized contract"):
        apply_model_contract_promotion(**kwargs)


@pytest.mark.parametrize(
    ("role", "field", "message"),
    (
        ("fusion", "model_bundle_id", "fusion public model-bundle"),
        ("traits", "root_expert_id", "traits public model-bundle"),
    ),
)
def test_apply_rejects_final_public_identity_mismatch(
    tmp_path: Path, role: str, field: str, message: str
) -> None:
    fixture = _closed_fixture(tmp_path)
    path = fixture["paths"][role]
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity_field = (
        "summary_identity_sha256" if role == "fusion" else "export_identity_sha256"
    )
    payload.pop(identity_field)
    payload[field] = "PHAXIS-V1.0.0-STRICT-TRAIN399-DIFFERENT"
    _seal(payload, identity_field)
    _write(path, payload)
    with pytest.raises(PromotionError, match=message):
        apply_model_contract_promotion(
            **_apply_kwargs(fixture, tmp_path / f"{role}-public-identity.json")
        )


def test_apply_rejects_current_contract_cas_drift(tmp_path: Path) -> None:
    fixture = _closed_fixture(tmp_path)
    current = json.loads(fixture["current"].read_text(encoding="utf-8"))
    current["unrelated_concurrent_change"] = True
    _write(fixture["current"], current)
    with pytest.raises(PromotionError, match="SHA drifted before validation"):
        apply_model_contract_promotion(
            **_apply_kwargs(fixture, tmp_path / "drift.json")
        )


@pytest.mark.parametrize(
    ("role", "field", "invalid", "message"),
    (
        ("stageb", "blind_images_used", 1, "blind_images_used must be 0"),
        (
            "traits",
            "root_cap_region_statistics_included",
            True,
            "root_cap_region_statistics_included must be false",
        ),
    ),
)
def test_apply_rejects_release_red_line_violation(
    tmp_path: Path, role: str, field: str, invalid, message: str
) -> None:
    fixture = _closed_fixture(tmp_path)
    paths = fixture["paths"]
    payload = json.loads(paths[role].read_text(encoding="utf-8"))
    identity = "summary_identity_sha256" if role == "stageb" else "export_identity_sha256"
    payload.pop(identity)
    payload[field] = invalid
    _seal(payload, identity)
    _write(paths[role], payload)
    with pytest.raises(PromotionError, match=message):
        apply_model_contract_promotion(
            **_apply_kwargs(fixture, tmp_path / "guard.json")
        )
