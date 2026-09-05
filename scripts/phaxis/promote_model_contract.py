#!/usr/bin/env python
"""Propose or CAS-apply the PHAxis 1.0.0 formal model contract.

The default mode writes a new, conspicuously unapplied contract candidate from
the fresh exact283 and strict train399 Gates.  It never edits the official
contract.  ``--apply`` additionally requires the final downstream evidence,
revalidates every binding, and atomically replaces the official contract only
when its bytes still match ``--expected-current-sha256``.  The official
contract embeds the complete promotion evidence and is authoritative; its
separate application receipt is deterministic and recoverable after a crash.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

# A source checkout must be directly runnable before PHAxis has been installed.
# ``source_release_common`` imports pure helpers from ``src/phaxis`` at module
# import time, so establish the checkout import root first rather than relying
# on an ambient PYTHONPATH or a previous editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import source_release_common as source_release_policy
from phaxis.hair_stageb.evaluation_inference import (  # noqa: E402
    EVALUATION_ARTIFACT_ROLE,
    EVALUATION_DETECTION_SCHEMA,
    EVALUATION_RUN_SCHEMA,
)
from phaxis.public_identity import (  # noqa: E402
    MODEL_BUNDLE_PREFIX,
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    ROOT_EXPERT_PREFIX,
    ROOT_PROVIDER_ROLE,
    derive_public_identity,
)
from build_manuscript_evidence_manifest import (
    EvidenceManifestError,
    SCHEMA_VERSION as EVIDENCE_GRAPH_SCHEMA,
    _guard_final_summary,
    _prediction_map,
    _require_proposal_binding,
    _sealed_identity,
    _validate_formal_gate_receipts,
    _validate_model_contract_proposal,
    sha256_file,
    sha256_json,
)


PROMOTION_SCHEMA = "PHAxis-model-contract-promotion-1.0"
APPLICATION_SCHEMA = "PHAxis-model-contract-promotion-application-1.0"
MODEL_CONTRACT_SCHEMA = "PHAxis-model-contract-1.0.0"
QCDEV44_DEVELOPMENT_ACCEPTANCE_GATE = {
    "schema_version": "PHAxis-QCdev44-development-acceptance-gate-1.0",
    "scope": (
        "deterministic overlay-visible development acceptance; not an "
        "independent accuracy or significance claim"
    ),
    "primary_metric": "tolerant_biological_presence_f1_20um",
    "requirements": {
        "stageb_f1_strictly_greater_than_locked_legacy_hybrid": True,
        "stageb_recall_strictly_greater_than_locked_legacy_hybrid": True,
        "stageb_precision_not_lower_than_locked_legacy_hybrid": True,
        "stageb_count_mae_strictly_lower_than_locked_legacy_hybrid": True,
    },
    "precision_noninferiority_margin_absolute": 0.0,
    "paired_bootstrap_role": (
        "descriptive post-selection uncertainty; not a promotion significance gate"
    ),
}


class PromotionError(RuntimeError):
    """A proposal, final evidence receipt, or compare-and-swap Gate failed."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionError(message)


def _read(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"{role}: file does not exist: {resolved}")
    _require(not resolved.is_symlink(), f"{role}: symlink inputs are forbidden")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"{role}: invalid UTF-8 JSON object") from error
    _require(isinstance(payload, dict), f"{role}: input must be one JSON object")
    return resolved, payload


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _require(not path.exists(), f"output appeared during validation: {path}")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _gate_inputs(
    *,
    train399_candidate: str | Path,
    train399_selection: str | Path,
    train399_evaluation: str | Path,
    root_exact283: str | Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    arguments = {
        "train399_candidate": train399_candidate,
        "train399_selection": train399_selection,
        "train399_evaluation": train399_evaluation,
        "root_exact283": root_exact283,
    }
    paths: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for role, value in arguments.items():
        paths[role], payloads[role] = _read(value, role)
    _require(len(set(paths.values())) == 4, "formal Gate inputs must be distinct files")
    try:
        _validate_formal_gate_receipts(paths, payloads)
    except EvidenceManifestError as error:
        raise PromotionError(str(error)) from error
    return paths, payloads


def _gate_stageb_binding(payloads: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = payloads["train399_candidate"]
    selection = payloads["train399_selection"]
    evaluation = payloads["train399_evaluation"]
    metadata = candidate.get("detection_model_metadata")
    selected = selection.get("selected")
    contract = evaluation.get("training_contract")
    _require(isinstance(metadata, Mapping), "candidate detection metadata is absent")
    _require(isinstance(selected, Mapping), "selection operating point is absent")
    _require(isinstance(contract, Mapping), "evaluation training contract is absent")
    expert = metadata.get("expert_id")
    checkpoints = metadata.get("checkpoint_sha256")
    threshold = selected.get("threshold")
    _require(isinstance(expert, str) and bool(expert), "candidate expert_id is absent")
    _require(
        metadata.get("deployment_role") == "candidate_gate_passed_not_promoted",
        "candidate metadata no longer has the non-promoting Gate role",
    )
    _require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 5
        and len(set(checkpoints)) == 5
        and all(_is_sha256(value) for value in checkpoints),
        "candidate checkpoints are not five distinct SHA-256 values",
    )
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold)),
        "selected threshold is not finite",
    )
    _require(contract.get("checkpoint_sha256") == checkpoints, "evaluation checkpoint binding differs")
    _require(
        contract.get("candidate_bundle_identity_sha256")
        == candidate.get("candidate_bundle_identity_sha256"),
        "evaluation candidate identity differs",
    )
    _require(
        contract.get("selection_receipt_identity_sha256")
        == selection.get("selection_receipt_identity_sha256"),
        "evaluation selection identity differs",
    )
    selected_identity = contract.get("selected_model_metadata_identity_sha256")
    _require(_is_sha256(selected_identity), "evaluation selected metadata identity is invalid")
    return {
        "expert_id": expert,
        "checkpoint_sha256": list(checkpoints),
        "selected_score_threshold": float(threshold),
        "candidate_bundle_identity_sha256": candidate["candidate_bundle_identity_sha256"],
        "selection_receipt_identity_sha256": selection["selection_receipt_identity_sha256"],
        "selected_model_metadata_identity_sha256": selected_identity,
    }


def _checkpoint_files(
    values: Sequence[str | Path], expected: Sequence[str]
) -> tuple[list[Path], list[str]]:
    _require(len(values) == 5, "exactly five checkpoint files are required")
    paths = [Path(value).resolve() for value in values]
    _require(len(set(paths)) == 5, "checkpoint paths must be distinct")
    for path in paths:
        _require(path.is_file(), f"checkpoint file is absent: {path}")
        _require(not path.is_symlink(), f"checkpoint symlinks are forbidden: {path}")
    observed = [sha256_file(path) for path in paths]
    _require(observed == list(expected), "checkpoint file SHA/order differs from candidate Gate")
    return paths, observed


def _validate_current_contract(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == MODEL_CONTRACT_SCHEMA, "unsupported model contract schema")
    _require(payload.get("product") == "PHAxis", "model contract product is not PHAxis")
    _require(payload.get("product_version") == "1.0.0", "model contract version is not 1.0.0")
    status = str(payload.get("formal_release_status", ""))
    _require("passed" not in status.casefold(), "official model contract is already passed")


def _finite_number(value: Any, *, role: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{role} is not finite",
    )
    return float(value)


def _validated_tolerance_metrics(payload: Any, *, role: str) -> dict[str, Any]:
    _require(isinstance(payload, Mapping), f"{role} is absent")
    _require(set(payload) == {"5", "10", "20"}, f"{role} must contain 5/10/20 um")
    copied: dict[str, Any] = {}
    for tolerance in ("5", "10", "20"):
        record = payload[tolerance]
        _require(isinstance(record, Mapping), f"{role}@{tolerance} is invalid")
        for field in ("tp", "n_pred", "n_gt"):
            value = record.get(field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{role}@{tolerance}.{field} is invalid",
            )
        for field in ("precision", "recall", "f1"):
            value = _finite_number(record.get(field), role=f"{role}@{tolerance}.{field}")
            _require(0.0 <= value <= 1.0, f"{role}@{tolerance}.{field} is outside [0,1]")
        tp = int(record["tp"])
        n_pred = int(record["n_pred"])
        n_gt = int(record["n_gt"])
        _require(
            tp <= min(n_pred, n_gt),
            f"{role}@{tolerance} has impossible true-positive counts",
        )
        expected_precision = tp / n_pred if n_pred else 0.0
        expected_recall = tp / n_gt if n_gt else 0.0
        expected_f1 = (
            2.0 * expected_precision * expected_recall
            / (expected_precision + expected_recall)
            if expected_precision + expected_recall
            else 0.0
        )
        for field, expected in (
            ("precision", expected_precision),
            ("recall", expected_recall),
            ("f1", expected_f1),
        ):
            _require(
                math.isclose(
                    float(record[field]), expected, rel_tol=0.0, abs_tol=1e-12
                ),
                f"{role}@{tolerance}.{field} differs from sufficient statistics",
            )
        copied[tolerance] = deepcopy(dict(record))
    return copied


def _qcdev44_development_acceptance_gate(
    *,
    stageb: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    declared_delta: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply direction-only development acceptance without independence claims."""

    stageb_primary = stageb["tolerant_biological_presence"]["20"]
    hybrid_primary = hybrid["tolerant_biological_presence"]["20"]
    observed = {
        "stageb_f1": float(stageb_primary["f1"]),
        "legacy_hybrid_f1": float(hybrid_primary["f1"]),
        "f1_delta": float(stageb_primary["f1"])
        - float(hybrid_primary["f1"]),
        "stageb_recall": float(stageb_primary["recall"]),
        "legacy_hybrid_recall": float(hybrid_primary["recall"]),
        "recall_delta": float(stageb_primary["recall"])
        - float(hybrid_primary["recall"]),
        "stageb_precision": float(stageb_primary["precision"]),
        "legacy_hybrid_precision": float(hybrid_primary["precision"]),
        "precision_delta": float(stageb_primary["precision"])
        - float(hybrid_primary["precision"]),
        "stageb_count_mae": float(stageb["count"]["mae"]),
        "legacy_hybrid_count_mae": float(hybrid["count"]["mae"]),
        "count_mae_delta": float(stageb["count"]["mae"])
        - float(hybrid["count"]["mae"]),
    }
    _require(
        math.isclose(
            float(declared_delta.get("biological_presence_f1_20um")),
            observed["f1_delta"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(declared_delta.get("count_mae")),
            observed["count_mae_delta"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "evaluation declared primary/count deltas differ from pooled metrics",
    )
    passed = {
        "stageb_f1_strictly_greater_than_locked_legacy_hybrid": (
            observed["f1_delta"] > 0.0
        ),
        "stageb_recall_strictly_greater_than_locked_legacy_hybrid": (
            observed["recall_delta"] > 0.0
        ),
        "stageb_precision_not_lower_than_locked_legacy_hybrid": (
            observed["precision_delta"] >= 0.0
        ),
        "stageb_count_mae_strictly_lower_than_locked_legacy_hybrid": (
            observed["count_mae_delta"] < 0.0
        ),
    }
    failed = [name for name, value in passed.items() if not value]
    _require(
        not failed,
        "QCdevelopment44 development acceptance Gate failed: "
        + ", ".join(failed),
    )
    return {
        **deepcopy(QCDEV44_DEVELOPMENT_ACCEPTANCE_GATE),
        "observed": observed,
        "passed": passed,
        "gate_pass": True,
    }


def _qcdev44_development_evidence(
    evaluation_path: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract only current evaluator-1.2 evidence; never inherit legacy blocks."""

    _require(
        evaluation.get("schema_version")
        == "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
        "formal proposal requires the evaluator 1.2 receipt",
    )
    _require(evaluation.get("status") == "completed", "evaluator 1.2 receipt is incomplete")
    _require(
        evaluation.get("blind_images_used") == 0
        and evaluation.get("independent_accuracy_claim_allowed") is False,
        "evaluator 1.2 receipt is blind-tainted or overclaims independence",
    )
    hierarchy = evaluation.get("metric_hierarchy")
    _require(isinstance(hierarchy, Mapping), "evaluator 1.2 metric hierarchy is absent")
    _require(
        hierarchy.get("primary_minimum_truth_coverage") == 0.25
        and hierarchy.get("primary_minimum_prediction_coverage") == 0.25
        and hierarchy.get("primary_minimum_direction_cosine") == 0.0
        and hierarchy.get("primary_tolerance_um") == 20.0
        and "without endpoint gates" in str(hierarchy.get("primary", "")),
        "evaluator 1.2 tolerant biological-presence contract changed",
    )
    overall = evaluation.get("overall")
    _require(
        isinstance(overall, Mapping)
        and set(overall) == {"stageb_train399", "hybrid_max"},
        "evaluator 1.2 overall expert set changed",
    )
    validated_experts: dict[str, Any] = {}
    for expert in ("stageb_train399", "hybrid_max"):
        record = overall[expert]
        _require(isinstance(record, Mapping), f"evaluation overall.{expert} is invalid")
        _require(record.get("images") == 44, f"evaluation overall.{expert} is not QCdev44")
        for field in ("predicted_hairs", "ground_truth_hairs"):
            value = record.get(field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"evaluation overall.{expert}.{field} is invalid",
            )
        copied = {
            "images": 44,
            "predicted_hairs": record["predicted_hairs"],
            "ground_truth_hairs": record["ground_truth_hairs"],
            "tolerant_biological_presence": _validated_tolerance_metrics(
                record.get("tolerant_biological_presence"),
                role=f"evaluation overall.{expert}.tolerant_biological_presence",
            ),
            "identity_attachment_proxy": _validated_tolerance_metrics(
                record.get("identity_attachment_proxy"),
                role=f"evaluation overall.{expert}.identity_attachment_proxy",
            ),
            "strict_whole_line_correspondence": _validated_tolerance_metrics(
                record.get("strict_whole_line_correspondence"),
                role=f"evaluation overall.{expert}.strict_whole_line_correspondence",
            ),
        }
        count = record.get("count")
        _require(isinstance(count, Mapping), f"evaluation overall.{expert}.count is absent")
        for field in ("mae", "bias", "pearson_r", "ccc"):
            _finite_number(count.get(field), role=f"evaluation overall.{expert}.count.{field}")
        copied["count"] = deepcopy(dict(count))
        for metric_family in (
            "tolerant_biological_presence",
            "identity_attachment_proxy",
            "strict_whole_line_correspondence",
        ):
            _require(
                all(
                    metric["n_pred"] == record["predicted_hairs"]
                    and metric["n_gt"] == record["ground_truth_hairs"]
                    for metric in copied[metric_family].values()
                ),
                f"evaluation overall.{expert}.{metric_family} counts differ from pooled totals",
            )
        validated_experts[expert] = copied

    per_image = evaluation.get("per_image")
    _require(isinstance(per_image, list) and len(per_image) == 44, "evaluation per_image is not exact44")
    ordered_task_ids: list[str] = []
    for index, row in enumerate(per_image):
        _require(isinstance(row, Mapping), f"evaluation per_image[{index}] is invalid")
        task_id = row.get("task_id")
        _require(
            isinstance(task_id, str) and task_id and task_id not in ordered_task_ids,
            f"evaluation per_image[{index}].task_id is empty or duplicated",
        )
        ordered_task_ids.append(task_id)
        for expert in ("stageb_train399", "hybrid_max"):
            expert_row = row.get(expert)
            presence = (
                expert_row.get("biological_presence_tp")
                if isinstance(expert_row, Mapping)
                else None
            )
            _require(
                isinstance(presence, Mapping)
                and {float(value) for value in presence} == {5.0, 10.0, 20.0},
                f"evaluation per_image[{index}].{expert} lacks 5/10/20 biological presence",
            )

    bootstrap = evaluation.get("paired_bootstrap_95ci")
    _require(isinstance(bootstrap, Mapping), "evaluation paired bootstrap is absent")
    _require(
        bootstrap.get("method") == "paired image-level nonparametric bootstrap"
        and isinstance(bootstrap.get("repetitions"), int)
        and bootstrap.get("repetitions") == 10_000
        and bootstrap.get("seed") == 20260828,
        "evaluation paired bootstrap contract changed",
    )
    bootstrap_experts = bootstrap.get("experts")
    _require(
        isinstance(bootstrap_experts, Mapping)
        and "stageb_train399" in bootstrap_experts,
        "evaluation StageB bootstrap evidence is absent",
    )
    stageb_bootstrap = bootstrap_experts["stageb_train399"]
    _require(isinstance(stageb_bootstrap, Mapping), "evaluation StageB bootstrap is invalid")
    _require(
        {
            "biological_presence_f1_20um",
            "identity_f1_20um",
            "count_mae",
            "count_ccc",
        }.issubset(stageb_bootstrap),
        "evaluation StageB bootstrap metric set is incomplete",
    )
    bootstrap_delta = bootstrap.get("delta_stageb_train399_minus_hybrid")
    _require(
        isinstance(bootstrap_delta, Mapping)
        and "biological_presence_f1_20um" in bootstrap_delta,
        "evaluation primary biological-presence delta bootstrap is absent",
    )
    primary_delta_interval = bootstrap_delta["biological_presence_f1_20um"]
    _require(
        isinstance(primary_delta_interval, Mapping),
        "evaluation primary biological-presence delta CI is invalid",
    )
    for field in ("lower_2_5", "upper_97_5"):
        _finite_number(
            primary_delta_interval.get(field),
            role=f"evaluation primary biological-presence delta CI.{field}",
        )

    prediction_locks = evaluation.get("prediction_input_locks")
    _require(isinstance(prediction_locks, Mapping), "evaluation prediction input locks are absent")
    validated_prediction_locks: dict[str, Any] = {}
    for list_field, identity_field in (
        ("stageb_detection_files", "stageb_detection_set_identity_sha256"),
        ("hybrid_prediction_files", "hybrid_prediction_set_identity_sha256"),
    ):
        records = prediction_locks.get(list_field)
        identity = prediction_locks.get(identity_field)
        _require(
            isinstance(records, list) and len(records) == 44,
            f"evaluation {list_field} is not exact44",
        )
        _require(
            all(
                isinstance(record, Mapping)
                and set(record) == {"task_id", "sha256"}
                and _is_sha256(record.get("sha256"))
                for record in records
            ),
            f"evaluation {list_field} contains an invalid file lock",
        )
        _require(
            [record["task_id"] for record in records] == ordered_task_ids,
            f"evaluation {list_field} task order differs from per_image",
        )
        _require(
            _is_sha256(identity) and identity == sha256_json(records),
            f"evaluation {identity_field} does not seal its ordered file locks",
        )
        validated_prediction_locks[list_field] = deepcopy(records)
        validated_prediction_locks[identity_field] = identity

    evaluation_inference = evaluation.get("evaluation_inference_authority")
    _require(
        isinstance(evaluation_inference, Mapping),
        "evaluation-only inference authority is absent",
    )
    _require(
        evaluation_inference.get("schema_version") == EVALUATION_RUN_SCHEMA
        and evaluation_inference.get("artifact_role") == EVALUATION_ARTIFACT_ROLE
        and evaluation_inference.get("evaluation_detection_schema_version")
        == EVALUATION_DETECTION_SCHEMA,
        "evaluation-only inference schema/role changed",
    )
    for field in (
        "evaluation_inference_summary_sha256",
        "evaluation_inference_summary_identity_sha256",
        "evaluation_gate_identity_sha256",
        "evaluation_detection_set_identity_sha256",
    ):
        _require(
            _is_sha256(evaluation_inference.get(field)),
            f"evaluation-only inference {field} is invalid",
        )
    _require(
        evaluation_inference["evaluation_detection_set_identity_sha256"]
        == validated_prediction_locks["stageb_detection_set_identity_sha256"],
        "evaluation-only inference detection set differs from evaluator locks",
    )
    _require(
        evaluation_inference.get("model_contract_proposal_required_for_artifact")
        is False
        and evaluation_inference.get("model_contract_proposal_present") is False
        and evaluation_inference.get("production_consumption_allowed") is False
        and evaluation_inference.get("fusion_consumption_allowed") is False
        and evaluation_inference.get("traits_consumption_allowed") is False,
        "evaluation-only inference is circular or deployable",
    )
    _require(
        evaluation_inference.get("canonical_annotations_read_during_inference")
        is False
        and evaluation_inference.get("condition_metadata_used_for_routing") is False
        and evaluation_inference.get("independent_accuracy_claim_allowed") is False
        and evaluation_inference.get("blind_images_used") == 0,
        "evaluation-only inference violates information boundaries",
    )

    comparator_contracts = evaluation.get("comparator_contract")
    comparator_contract = (
        comparator_contracts.get("hybrid_max")
        if isinstance(comparator_contracts, Mapping)
        else None
    )
    expected_hybrid_identity = (
        source_release_policy.LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256
    )
    _require(
        validated_prediction_locks["hybrid_prediction_set_identity_sha256"]
        == expected_hybrid_identity,
        "evaluation legacy Hybrid prediction set is not the fixed QCdev44 comparator",
    )
    expected_comparator_contract = {
        "evidence_role": "locked_legacy_development_comparator",
        "schema_version": (
            "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
        ),
        "identity_hair_variant": "hybrid_verified_increment",
        "count_hair_variant": "hybrid_verified_increment",
        "endpoint_complete_identity_layer": True,
        "phaxis_payload_allowed": False,
        "stageb_identity_source_allowed": False,
        "prediction_set_identity_sha256": expected_hybrid_identity,
        "expected_prediction_set_identity_sha256": expected_hybrid_identity,
    }
    _require(
        isinstance(comparator_contract, Mapping)
        and dict(comparator_contract) == expected_comparator_contract,
        "evaluation legacy Hybrid comparator contract changed or is cross-bound",
    )

    overall_delta = evaluation.get("delta_stageb_train399_minus_hybrid")
    _require(isinstance(overall_delta, Mapping), "evaluation overall delta map is absent")
    _finite_number(
        overall_delta.get("biological_presence_f1_20um"),
        role="evaluation overall biological-presence F1 delta",
    )
    _finite_number(
        overall_delta.get("count_mae"),
        role="evaluation overall count-MAE delta",
    )
    development_acceptance_gate = _qcdev44_development_acceptance_gate(
        stageb=validated_experts["stageb_train399"],
        hybrid=validated_experts["hybrid_max"],
        declared_delta=overall_delta,
    )

    contract = evaluation.get("training_contract")
    inputs = evaluation.get("inputs_sha256")
    _require(isinstance(contract, Mapping), "evaluation training contract is absent")
    _require(isinstance(inputs, Mapping), "evaluation source hash map is absent")
    _require(
        inputs.get("evaluation_inference_summary")
        == evaluation_inference["evaluation_inference_summary_sha256"],
        "evaluation source hashes do not bind the evaluation-inference summary",
    )
    _require(
        contract.get("evaluation_gate_identity_sha256")
        == evaluation_inference["evaluation_gate_identity_sha256"]
        and contract.get("evaluation_inference_summary_identity_sha256")
        == evaluation_inference["evaluation_inference_summary_identity_sha256"],
        "evaluation training contract does not bind evaluation-only inference",
    )
    return {
        "role": "locked_qcdevelopment44_non_independent_development_evidence",
        "schema_version": evaluation["schema_version"],
        "scope": evaluation.get("scope"),
        "images": 44,
        "metric_hierarchy": deepcopy(dict(hierarchy)),
        "stageb_train399": validated_experts["stageb_train399"],
        "same_run_historical_endpoint_complete_comparator": {
            "role": "historical_comparator_recomputed_by_evaluator_1_2",
            "source_prediction_contract": deepcopy(expected_comparator_contract),
            **validated_experts["hybrid_max"],
        },
        "delta_stageb_train399_minus_hybrid": deepcopy(
            dict(overall_delta)
        ),
        "development_acceptance_gate": development_acceptance_gate,
        "paired_bootstrap_95ci": deepcopy(dict(bootstrap)),
        "prediction_input_locks": validated_prediction_locks,
        "evaluation_inference_authority": deepcopy(dict(evaluation_inference)),
        "source": {
            "evaluation_sha256": sha256_file(evaluation_path),
            "evaluation_content_identity_sha256": sha256_json(evaluation),
            "inputs_sha256": deepcopy(dict(inputs)),
            "candidate_bundle_identity_sha256": contract.get(
                "candidate_bundle_identity_sha256"
            ),
            "selection_receipt_identity_sha256": contract.get(
                "selection_receipt_identity_sha256"
            ),
            "selected_model_metadata_identity_sha256": contract.get(
                "selected_model_metadata_identity_sha256"
            ),
            "stageb_detection_set_identity_sha256": validated_prediction_locks[
                "stageb_detection_set_identity_sha256"
            ],
            "hybrid_prediction_set_identity_sha256": validated_prediction_locks[
                "hybrid_prediction_set_identity_sha256"
            ],
            "evaluation_inference_summary_sha256": evaluation_inference[
                "evaluation_inference_summary_sha256"
            ],
            "evaluation_inference_summary_identity_sha256": evaluation_inference[
                "evaluation_inference_summary_identity_sha256"
            ],
            "evaluation_gate_identity_sha256": evaluation_inference[
                "evaluation_gate_identity_sha256"
            ],
        },
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }


def _proposal_payload(
    *,
    current: Mapping[str, Any],
    current_sha256: str,
    paths: Mapping[str, Path],
    payloads: Mapping[str, dict[str, Any]],
    binding: Mapping[str, Any],
    checkpoint_sha256: Sequence[str],
) -> dict[str, Any]:
    candidate = payloads["train399_candidate"]
    root_receipt = payloads["root_exact283"]
    identity_payload = candidate.get("identity_payload")
    training_lock = (
        identity_payload.get("training_lock")
        if isinstance(identity_payload, Mapping)
        else None
    )
    _require(isinstance(training_lock, Mapping), "candidate training lock is absent")
    qcdev44 = _qcdev44_development_evidence(
        paths["train399_evaluation"], payloads["train399_evaluation"]
    )
    public_identity = derive_public_identity(
        binding,
        root_bundle_identity_sha256=root_receipt["bundle_identity_sha256"],
    )
    public_system_identity = public_identity["public_system_identity_sha256"]
    model_bundle_id = public_identity["model_bundle_id"]
    root_expert_id = public_identity["root_expert_id"]

    current_hair = current.get("hair_identity_count_expert")
    runtime: dict[str, Any] = {}
    if isinstance(current_hair, Mapping):
        for field in (
            "working_um_per_px",
            "output_stride",
            "window",
            "overlap",
            "batch",
            "nms_kernel",
            "horizontal_flip_tta",
            "use_trace",
            "root_gate_um",
            "maximum_instances",
            "precision_mode",
        ):
            if field in current_hair:
                runtime[field] = deepcopy(current_hair[field])

    proposal: dict[str, Any] = {
        "schema_version": MODEL_CONTRACT_SCHEMA,
        "product": "PHAxis",
        "product_version": "1.0.0",
        "model_bundle_id": model_bundle_id,
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public_system_identity,
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "formal_release_status": "passed_proposal_not_official",
        "expert_boundary": {
            "root_point_scale_continuity_statistics": root_expert_id,
            "hair_identity_and_count": binding["expert_id"],
            "hair_length": (
                "endpoint-complete centreline one-to-one associated with a "
                "Stage-B identity"
            ),
        },
        "root_expert": {
            "provider_role": public_identity["root_provider_role"],
            "expert_id": root_expert_id,
            "portable_raw_image_provider_status": "fresh_exact283_passed",
            "fresh_exact283_receipt_sha256": sha256_file(paths["root_exact283"]),
            "fresh_exact283_audit_identity_sha256": root_receipt["audit_identity_sha256"],
            "reference_identity_sha256": root_receipt["reference_identity_sha256"],
            "fresh_reference_identity_sha256": root_receipt[
                "fresh_reference_identity_sha256"
            ],
            "bundle_identity_sha256": root_receipt["bundle_identity_sha256"],
            "pipeline_identity_sha256": root_receipt["pipeline_identity_sha256"],
            "root_bundle_authority": {
                "binding": "transitively_sealed_by_fresh_exact283_pipeline_identity",
                "bundle_identity_sha256": root_receipt["bundle_identity_sha256"],
                "pipeline_identity_sha256": root_receipt["pipeline_identity_sha256"],
            },
            "root_cap_region_output": False,
        },
        "hair_identity_count_expert": {
            "current_checkpoint_role": "formal_train399_only_deployment_candidate",
            "training_scope": (
                "five fixed-seed members trained on train399; "
                "QCdevelopment44 excluded from optimization"
            ),
            "deployment_ensemble_used_qcdev44_labels_in_some_members": False,
            "strict_train399_only_retraining_gate": "passed_proposal_not_official",
            "score_threshold": binding["selected_score_threshold"],
            "checkpoint_policy": "five_seed_train399_last_epoch_60",
            "checkpoint_sha256_in_member_order": list(checkpoint_sha256),
            "expert_id": binding["expert_id"],
            **runtime,
        },
        "length_association": {
            "maximum_base_distance_um": 20.0,
            "matching": "one_to_one_Hungarian",
            "formal_geometry": "length_hairs[*].points_xy",
            "forbidden_formal_geometry": "identity_hairs[*].points_xy",
            "stageb_predicted_length_role": "diagnostic_only",
        },
        "development_evidence": {"qcdev44": qcdev44},
        "data_contract": {
            "train_images": 399,
            "development_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "dataset_manifest_sha256": training_lock.get("dataset_manifest_sha256"),
            "split_manifest_sha256": training_lock.get("split_manifest_sha256"),
            "dataset_split_identity_sha256": training_lock.get(
                "dataset_split_identity_sha256"
            ),
            "integrity_manifest_sha256": training_lock.get(
                "integrity_manifest_sha256"
            ),
            "root_hair_annotation": "one centreline polyline per visible hair trunk",
        },
        "red_lines": {
            "blind_images_used": 0,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "validation_labels_used_for_training_by_current_five_member_deployment_ensemble": False,
            "formal_train399_only_stageb_weights_available": True,
            "independent_accuracy_claimed": False,
            "root_cap_region_statistics_included": False,
            "pyRootHair_called_or_copied": False,
            "legacy_v1_runtime_dependency": False,
            "rhaxiscc_runtime_dependency": False,
        },
    }
    proposal["promotion"] = {
        "schema_version": PROMOTION_SCHEMA,
        "status": "validated_proposal_not_applied",
        "official_apply_performed": False,
        "source_model_contract_sha256": current_sha256,
        "formal_gate_source_sha256": {
            role: sha256_file(paths[role])
            for role in (
                "train399_candidate",
                "train399_selection",
                "train399_evaluation",
                "root_exact283",
            )
        },
        "formal_gate_identity_sha256": {
            "candidate_bundle_identity_sha256": binding[
                "candidate_bundle_identity_sha256"
            ],
            "selection_receipt_identity_sha256": binding[
                "selection_receipt_identity_sha256"
            ],
            "selected_model_metadata_identity_sha256": binding[
                "selected_model_metadata_identity_sha256"
            ],
            "root_exact283_audit_identity_sha256": payloads["root_exact283"][
                "audit_identity_sha256"
            ],
        },
        "checkpoint_file_sha256_in_member_order": list(checkpoint_sha256),
        "stageb_binding": dict(binding),
        "required_final_receipts_before_apply": [
            "stageb_summary",
            "fusion_summary",
            "traits_summary",
            "manuscript_evidence_manifest",
        ],
    }
    proposal["model_contract_identity_sha256"] = sha256_json(proposal)
    return proposal


def build_model_contract_proposal(
    *,
    current_model_contract: str | Path,
    train399_candidate: str | Path,
    train399_selection: str | Path,
    train399_evaluation: str | Path,
    root_exact283: str | Path,
    checkpoints: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Write a new passed-contract candidate without modifying the official file."""

    destination = Path(output).resolve()
    _require(not destination.exists(), f"output already exists: {destination}")
    current_path, current = _read(current_model_contract, "current_model_contract")
    _validate_current_contract(current)
    paths, payloads = _gate_inputs(
        train399_candidate=train399_candidate,
        train399_selection=train399_selection,
        train399_evaluation=train399_evaluation,
        root_exact283=root_exact283,
    )
    binding = _gate_stageb_binding(payloads)
    _checkpoint_paths, checkpoint_hashes = _checkpoint_files(
        checkpoints, binding["checkpoint_sha256"]
    )
    proposal = _proposal_payload(
        current=current,
        current_sha256=sha256_file(current_path),
        paths=paths,
        payloads=payloads,
        binding=binding,
        checkpoint_sha256=checkpoint_hashes,
    )
    _atomic_json_new(destination, proposal)
    return proposal


def _validate_proposal_against_gate(
    *,
    proposal_path: Path,
    proposal: Mapping[str, Any],
    current: Mapping[str, Any],
    current_sha256: str,
    paths: Mapping[str, Path],
    payloads: Mapping[str, dict[str, Any]],
    binding: Mapping[str, Any],
    checkpoint_hashes: Sequence[str],
) -> tuple[str, str]:
    try:
        proposal_identity, _ = _validate_model_contract_proposal(proposal)
    except EvidenceManifestError as error:
        raise PromotionError(str(error)) from error
    expected_proposal = _proposal_payload(
        current=current,
        current_sha256=current_sha256,
        paths=paths,
        payloads=payloads,
        binding=binding,
        checkpoint_sha256=checkpoint_hashes,
    )
    _require(
        dict(proposal) == expected_proposal,
        "proposal is not the deterministic sanitized contract for this Gate",
    )
    promotion = proposal["promotion"]
    _require(
        promotion.get("source_model_contract_sha256") == current_sha256,
        "proposal belongs to a different current model contract",
    )
    expected_sources = {
        role: sha256_file(paths[role])
        for role in (
            "train399_candidate",
            "train399_selection",
            "train399_evaluation",
            "root_exact283",
        )
    }
    _require(
        promotion.get("formal_gate_source_sha256") == expected_sources,
        "proposal formal Gate file binding mismatch",
    )
    _require(promotion.get("stageb_binding") == binding, "proposal StageB binding mismatch")
    _require(
        promotion.get("checkpoint_file_sha256_in_member_order")
        == list(checkpoint_hashes),
        "proposal checkpoint file binding mismatch",
    )
    expected_root = payloads["root_exact283"].get("audit_identity_sha256")
    _require(
        promotion.get("formal_gate_identity_sha256", {}).get(
            "root_exact283_audit_identity_sha256"
        )
        == expected_root,
        "proposal root exact283 identity mismatch",
    )
    return sha256_file(proposal_path), proposal_identity


def _validate_final_receipts(
    *,
    paths: Mapping[str, Path],
    payloads: Mapping[str, dict[str, Any]],
    binding: Mapping[str, Any],
    proposal_sha256: str,
    proposal_identity_sha256: str,
    model_bundle_id: str,
    root_expert_id: str,
) -> None:
    stageb, fusion, traits = (
        payloads[role] for role in ("stageb", "fusion", "traits")
    )
    for role in ("stageb", "fusion", "traits"):
        try:
            if role == "stageb":
                # Stage B is a hair-identity/count inference receipt, not a
                # root-cap output or phenotype schema.  Requiring it to carry
                # a root-cap field would retroactively change a hash-sealed
                # scientific result without adding any protection.  Instead,
                # fail closed on the Stage-B schema/status/blind scope and on
                # any root-cap field that might ever be introduced.  The
                # fusion, trait and manuscript receipts below still require
                # an explicit ``False`` root-cap-region guard.
                stageb_payload = payloads[role]
                _require(
                    stageb_payload.get("schema_version")
                    == "PHAxis-StageB-inference-run-1.1"
                    and stageb_payload.get("status") == "completed",
                    "stageb: unsupported or incomplete final summary",
                )
                _require(
                    stageb_payload.get("blind_images_used") == 0,
                    "stageb: blind_images_used must be 0",
                )
                for field in (
                    "root_cap_region_output",
                    "root_cap_region_statistics_included",
                ):
                    _require(
                        stageb_payload.get(field) in {None, False},
                        f"stageb: {field} must be absent or false",
                    )
            else:
                _guard_final_summary(role, payloads[role])
            _require_proposal_binding(
                role,
                payloads[role],
                proposal_sha256=proposal_sha256,
                proposal_identity_sha256=proposal_identity_sha256,
            )
            _sealed_identity(
                payloads[role],
                "summary_identity_sha256" if role in {"stageb", "fusion"} else "export_identity_sha256",
                role,
            )
        except EvidenceManifestError as error:
            raise PromotionError(str(error)) from error
    selected = stageb.get("detection_model_metadata")
    _require(isinstance(selected, Mapping), "stageb selected metadata is absent")
    for field, expected in binding.items():
        observed = (
            selected.get(field)
            if field not in {"selected_score_threshold"}
            else selected.get("selected_score_threshold")
        )
        if field == "selected_score_threshold":
            _require(
                isinstance(observed, (int, float))
                and math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12),
                "stageb selected threshold differs from proposal",
            )
        else:
            _require(observed == expected, f"stageb selected metadata differs: {field}")
    _require(
        selected.get("deployment_role") == "candidate_gate_passed_not_promoted",
        "stageb selected metadata did not retain the candidate Gate role",
    )
    _require(stageb.get("images") == 283, "stageb final image count is not 283")
    _require(stageb.get("checkpoint_sha256") == binding["checkpoint_sha256"], "stageb checkpoint mismatch")
    _require(
        math.isclose(float(stageb.get("score_threshold", float("nan"))), float(binding["selected_score_threshold"]), rel_tol=0.0, abs_tol=1e-12),
        "stageb summary threshold mismatch",
    )
    _require(fusion.get("images") == 283, "fusion final image count is not 283")
    _require(
        fusion.get("model_bundle_id") == model_bundle_id
        and fusion.get("root_expert") == root_expert_id,
        "fusion public model-bundle/root-provider identity differs from proposal",
    )
    _require(
        fusion.get("source_stageb_summary_sha256") == sha256_file(paths["stageb"]),
        "fusion source StageB SHA mismatch",
    )
    _require(fusion.get("hair_identity_count_expert") == binding["expert_id"], "fusion expert mismatch")
    predictions = _prediction_map(fusion)
    _require(traits.get("tasks") == 283, "traits final task count is not 283")
    _require(
        traits.get("model_bundle_id") == model_bundle_id
        and traits.get("root_expert_id") == root_expert_id,
        "traits public model-bundle/root-provider identity differs from proposal",
    )
    _require(traits.get("prediction_sha256") == predictions, "traits prediction map differs from fusion")
    _require(traits.get("hair_identity_count_expert") == binding["expert_id"], "traits expert mismatch")

    evidence = payloads["evidence"]
    _require(
        evidence.get("schema_version")
        == EVIDENCE_GRAPH_SCHEMA
        and evidence.get("status") == "passed_formal_evidence_graph"
        and evidence.get("formal_release_evidence_closed") is True,
        "manuscript evidence graph is not a formal pass",
    )
    _require(evidence.get("blind_images_used") == 0, "evidence graph is blind-tainted")
    _require(
        evidence.get("root_cap_region_statistics_included") is False,
        "evidence graph includes root-cap-region statistics",
    )
    try:
        _sealed_identity(evidence, "manifest_identity_sha256", "evidence")
    except EvidenceManifestError as error:
        raise PromotionError(str(error)) from error
    _require(
        evidence.get("model_contract_proposal_sha256") == proposal_sha256
        and evidence.get("model_contract_proposal_identity_sha256")
        == proposal_identity_sha256,
        "evidence graph belongs to a different model-contract proposal",
    )
    _require(evidence.get("stageb_binding") == binding, "evidence StageB binding mismatch")
    artifacts = evidence.get("artifacts")
    _require(isinstance(artifacts, Mapping), "evidence artifact graph is absent")
    expected_files = {
        "model_contract_proposal": proposal_sha256,
        "train399_candidate": sha256_file(paths["train399_candidate"]),
        "train399_selection": sha256_file(paths["train399_selection"]),
        "train399_evaluation": sha256_file(paths["train399_evaluation"]),
        "root_exact283": sha256_file(paths["root_exact283"]),
        "stageb": sha256_file(paths["stageb"]),
        "fusion": sha256_file(paths["fusion"]),
        "traits": sha256_file(paths["traits"]),
    }
    for role, digest in expected_files.items():
        record = artifacts.get(role)
        _require(
            isinstance(record, Mapping) and record.get("source_file_sha256") == digest,
            f"evidence artifact file binding differs: {role}",
        )
    figure_record = artifacts.get("figures")
    declared = (
        figure_record.get("declared_sha256_identities")
        if isinstance(figure_record, Mapping)
        else None
    )
    _require(isinstance(declared, Mapping), "evidence figure proposal binding is absent")
    _require(
        proposal_sha256 in declared.values()
        and proposal_identity_sha256 in declared.values(),
        "final figure receipt does not explicitly bind the same proposal",
    )


def _application_receipt_payload(
    final_contract: Mapping[str, Any], *, final_contract_sha256: str
) -> dict[str, Any]:
    promotion = final_contract.get("promotion")
    _require(isinstance(promotion, Mapping), "applied contract promotion block is absent")
    final_sources = promotion.get("final_receipt_source_sha256")
    final_identities = promotion.get("final_receipt_identity_sha256")
    _require(isinstance(final_sources, Mapping), "applied contract final source hashes are absent")
    _require(isinstance(final_identities, Mapping), "applied contract final identities are absent")
    receipt: dict[str, Any] = {
        "schema_version": APPLICATION_SCHEMA,
        "status": "applied",
        "official_model_contract_replaced": True,
        "expected_previous_model_contract_sha256": promotion.get(
            "expected_source_model_contract_sha256"
        ),
        "proposal_file_sha256": promotion.get("proposal_file_sha256"),
        "proposal_identity_sha256": promotion.get("proposal_identity_sha256"),
        "final_model_contract_sha256": final_contract_sha256,
        "final_model_contract_identity_sha256": final_contract.get(
            "model_contract_identity_sha256"
        ),
        "final_evidence_manifest_sha256": final_sources.get("evidence"),
        "final_evidence_manifest_identity_sha256": final_identities.get("evidence"),
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    _require(
        all(
            _is_sha256(receipt.get(field))
            for field in (
                "expected_previous_model_contract_sha256",
                "proposal_file_sha256",
                "proposal_identity_sha256",
                "final_model_contract_sha256",
                "final_model_contract_identity_sha256",
                "final_evidence_manifest_sha256",
                "final_evidence_manifest_identity_sha256",
            )
        ),
        "applied contract lacks receipt-recovery hashes",
    )
    receipt["application_identity_sha256"] = sha256_json(receipt)
    return receipt


def recover_application_receipt(
    *, applied_model_contract: str | Path, output: str | Path
) -> dict[str, Any]:
    """Deterministically rebuild the derivative receipt from official authority."""

    contract_path, contract = _read(applied_model_contract, "applied_model_contract")
    _require(contract.get("schema_version") == MODEL_CONTRACT_SCHEMA, "unsupported applied contract schema")
    _require(contract.get("formal_release_status") == "passed", "model contract is not officially passed")
    promotion = contract.get("promotion")
    _require(
        isinstance(promotion, Mapping)
        and promotion.get("schema_version") == PROMOTION_SCHEMA
        and promotion.get("status") == "applied_formal_release"
        and promotion.get("official_apply_performed") is True,
        "model contract is not an applied promotion authority",
    )
    try:
        _sealed_identity(contract, "model_contract_identity_sha256", "applied_model_contract")
    except EvidenceManifestError as error:
        raise PromotionError(str(error)) from error
    receipt = _application_receipt_payload(
        contract, final_contract_sha256=sha256_file(contract_path)
    )
    _atomic_json_new(Path(output).resolve(), receipt)
    return receipt


def apply_model_contract_promotion(
    *,
    current_model_contract: str | Path,
    expected_current_sha256: str,
    proposal: str | Path,
    train399_candidate: str | Path,
    train399_selection: str | Path,
    train399_evaluation: str | Path,
    root_exact283: str | Path,
    checkpoints: Sequence[str | Path],
    stageb_summary: str | Path,
    fusion_summary: str | Path,
    traits_summary: str | Path,
    manuscript_evidence_manifest: str | Path,
    application_receipt: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """CAS-replace the official contract after revalidating final evidence."""

    expected = str(expected_current_sha256).lower()
    _require(_is_sha256(expected), "expected-current-sha256 is invalid")
    receipt_path = Path(application_receipt).resolve()
    _require(not receipt_path.exists(), f"output already exists: {receipt_path}")
    current_path, current = _read(current_model_contract, "current_model_contract")
    _validate_current_contract(current)
    current_sha = sha256_file(current_path)
    _require(current_sha == expected, "current model contract SHA drifted before validation")
    proposal_path, proposal_payload = _read(proposal, "model_contract_proposal")
    _require(proposal_path != current_path, "proposal must be a separate unapplied file")
    gate_paths, gate_payloads = _gate_inputs(
        train399_candidate=train399_candidate,
        train399_selection=train399_selection,
        train399_evaluation=train399_evaluation,
        root_exact283=root_exact283,
    )
    binding = _gate_stageb_binding(gate_payloads)
    _checkpoint_paths, checkpoint_hashes = _checkpoint_files(
        checkpoints, binding["checkpoint_sha256"]
    )
    proposal_sha, proposal_identity = _validate_proposal_against_gate(
        proposal_path=proposal_path,
        proposal=proposal_payload,
        current=current,
        current_sha256=current_sha,
        paths=gate_paths,
        payloads=gate_payloads,
        binding=binding,
        checkpoint_hashes=checkpoint_hashes,
    )
    model_bundle_id = proposal_payload.get("model_bundle_id")
    root_expert = proposal_payload.get("root_expert")
    root_expert_id = (
        root_expert.get("expert_id")
        if isinstance(root_expert, Mapping)
        else None
    )
    root_provider_role = (
        root_expert.get("provider_role")
        if isinstance(root_expert, Mapping)
        else None
    )
    _require(
        isinstance(model_bundle_id, str)
        and model_bundle_id.startswith(MODEL_BUNDLE_PREFIX)
        and isinstance(root_expert_id, str)
        and root_expert_id.startswith(ROOT_EXPERT_PREFIX)
        and root_provider_role == ROOT_PROVIDER_ROLE,
        "proposal public model-bundle/root-provider identity is invalid",
    )
    final_paths = dict(gate_paths)
    final_payloads = dict(gate_payloads)
    for role, value in (
        ("stageb", stageb_summary),
        ("fusion", fusion_summary),
        ("traits", traits_summary),
        ("evidence", manuscript_evidence_manifest),
    ):
        final_paths[role], final_payloads[role] = _read(value, role)
    _require(
        len(set(final_paths.values()) | {current_path, proposal_path})
        == len(final_paths) + 2,
        "promotion inputs must be distinct files",
    )
    _validate_final_receipts(
        paths=final_paths,
        payloads=final_payloads,
        binding=binding,
        proposal_sha256=proposal_sha,
        proposal_identity_sha256=proposal_identity,
        model_bundle_id=model_bundle_id,
        root_expert_id=root_expert_id,
    )

    final_contract = deepcopy(proposal_payload)
    final_contract.pop("model_contract_identity_sha256", None)
    final_contract["formal_release_status"] = "passed"
    final_contract["hair_identity_count_expert"]["current_checkpoint_role"] = (
        "formal_train399_only_deployment"
    )
    final_contract["hair_identity_count_expert"][
        "strict_train399_only_retraining_gate"
    ] = "passed"
    final_contract["promotion"] = {
        **dict(final_contract["promotion"]),
        "status": "applied_formal_release",
        "official_apply_performed": True,
        "proposal_file_sha256": proposal_sha,
        "proposal_identity_sha256": proposal_identity,
        "expected_source_model_contract_sha256": expected,
        "final_receipt_source_sha256": {
            role: sha256_file(final_paths[role])
            for role in ("stageb", "fusion", "traits", "evidence")
        },
        "final_receipt_identity_sha256": {
            "stageb": final_payloads["stageb"]["summary_identity_sha256"],
            "fusion": final_payloads["fusion"]["summary_identity_sha256"],
            "traits": final_payloads["traits"]["export_identity_sha256"],
            "evidence": final_payloads["evidence"]["manifest_identity_sha256"],
        },
        "final_receipt_public_identity": {
            role: {
                "model_bundle_id": model_bundle_id,
                "root_expert_id": root_expert_id,
            }
            for role in ("fusion", "traits")
        },
    }
    final_contract["model_contract_identity_sha256"] = sha256_json(final_contract)
    final_bytes = (
        json.dumps(final_contract, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    final_sha = hashlib.sha256(final_bytes).hexdigest()
    receipt = _application_receipt_payload(
        final_contract, final_contract_sha256=final_sha
    )

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    contract_temp = current_path.parent / f".{current_path.name}.{uuid.uuid4().hex}.tmp"
    receipt_temp = receipt_path.parent / f".{receipt_path.name}.{uuid.uuid4().hex}.tmp"
    official_replaced = False
    try:
        with contract_temp.open("xb") as handle:
            handle.write(final_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with receipt_temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _require(not receipt_path.exists(), f"output appeared during validation: {receipt_path}")
        _require(
            sha256_file(current_path) == expected,
            "current model contract SHA drifted during validation",
        )
        os.replace(contract_temp, current_path)
        official_replaced = True
        os.replace(receipt_temp, receipt_path)
    except BaseException as error:
        for temporary in (contract_temp, receipt_temp):
            if temporary.exists():
                temporary.unlink()
        if official_replaced:
            raise PromotionError(
                "official contract was atomically applied but derivative receipt "
                "publication failed; recover it with --recover-application-receipt"
            ) from error
        raise
    return final_contract, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-model-contract", required=True)
    parser.add_argument("--train399-candidate")
    parser.add_argument("--train399-selection")
    parser.add_argument("--train399-evaluation")
    parser.add_argument("--root-exact283")
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--recover-application-receipt", action="store_true")
    parser.add_argument("--proposal")
    parser.add_argument("--expected-current-sha256")
    parser.add_argument("--stageb-summary")
    parser.add_argument("--fusion-summary")
    parser.add_argument("--traits-summary")
    parser.add_argument("--manuscript-evidence-manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.recover_application_receipt:
        try:
            forbidden = (
                args.apply,
                args.train399_candidate,
                args.train399_selection,
                args.train399_evaluation,
                args.root_exact283,
                args.checkpoint,
                args.proposal,
                args.expected_current_sha256,
                args.stageb_summary,
                args.fusion_summary,
                args.traits_summary,
                args.manuscript_evidence_manifest,
            )
            _require(not any(forbidden), "recovery mode accepts only current contract and output")
            receipt = recover_application_receipt(
                applied_model_contract=args.current_model_contract,
                output=args.output,
            )
            print(receipt["application_identity_sha256"])
            return 0
        except PromotionError as error:
            print(f"PHAxis model-contract promotion blocked: {error}")
            return 2
    base_required = {
        "train399_candidate": args.train399_candidate,
        "train399_selection": args.train399_selection,
        "train399_evaluation": args.train399_evaluation,
        "root_exact283": args.root_exact283,
        "checkpoints": args.checkpoint,
    }
    base_missing = [name for name, value in base_required.items() if not value]
    if base_missing:
        print(
            "PHAxis model-contract promotion blocked: missing required inputs: "
            + ", ".join(base_missing)
        )
        return 2
    common = {
        "current_model_contract": args.current_model_contract,
        **base_required,
    }
    try:
        if not args.apply:
            forbidden = (
                args.proposal,
                args.expected_current_sha256,
                args.stageb_summary,
                args.fusion_summary,
                args.traits_summary,
                args.manuscript_evidence_manifest,
            )
            _require(not any(forbidden), "apply-only inputs require --apply")
            result = build_model_contract_proposal(**common, output=args.output)
            print(result["model_contract_identity_sha256"])
            return 0
        required = {
            "proposal": args.proposal,
            "expected_current_sha256": args.expected_current_sha256,
            "stageb_summary": args.stageb_summary,
            "fusion_summary": args.fusion_summary,
            "traits_summary": args.traits_summary,
            "manuscript_evidence_manifest": args.manuscript_evidence_manifest,
        }
        missing = [name for name, value in required.items() if not value]
        _require(not missing, "--apply missing required inputs: " + ", ".join(missing))
        final, _receipt = apply_model_contract_promotion(
            **common,
            **required,
            application_receipt=args.output,
        )
        print(final["model_contract_identity_sha256"])
        return 0
    except PromotionError as error:
        print(f"PHAxis model-contract promotion blocked: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
