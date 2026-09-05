"""Command-line interface for the PHAxis 1.0.0 runtime."""

from __future__ import annotations

import argparse
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

from .constants import PRODUCT_NAME, PRODUCT_VERSION
from .contracts import (
    ContractError,
    validate_hybrid_prediction,
    validate_stageb_detection_payload,
)
from .fusion import fuse_hybrid_root_with_stageb_hairs
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .model_contract_binding import (
    ModelContractProposalBinding,
    read_model_contract_authority,
    require_receipt_binding,
    validate_stageb_proposal_binding,
)
from .traits import export_traits


def _version() -> str:
    try:
        return distribution_version("phaxis")
    except Exception:
        return PRODUCT_VERSION


def _materialize_locked_artifacts(
    prediction: dict[str, Any], *, source_root: Path, output_root: Path
) -> None:
    pairs = (
        ("root_mask_relpath", "root_mask_sha256"),
        ("root_axis_geometry_relpath", "root_axis_geometry_sha256"),
        ("root_continuity_added_mask_relpath", "root_continuity_added_mask_sha256"),
        ("root_width_reference_mask_relpath", "root_width_reference_mask_sha256"),
        (
            "root_width_reference_axis_geometry_relpath",
            "root_width_reference_axis_geometry_sha256",
        ),
    )
    for relpath_field, digest_field in pairs:
        relpath = prediction.get(relpath_field)
        expected_digest = prediction.get(digest_field)
        if not relpath or not expected_digest:
            continue
        source = source_root / str(relpath)
        destination = output_root / str(relpath)
        if destination.exists():
            if sha256_file(destination) != str(expected_digest):
                raise ContractError(f"existing output artifact hash mismatch: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != str(expected_digest):
            raise ContractError(f"copied artifact hash mismatch: {destination}")


def _require_public_identity(
    payload: dict[str, Any],
    binding: ModelContractProposalBinding,
    *,
    role: str,
    root_field: str,
) -> None:
    """Require a component payload to carry the proposal-owned public IDs."""

    public = binding.public_identity_fields()
    if (
        payload.get("model_bundle_id") != public["model_bundle_id"]
        or payload.get(root_field) != public["root_expert_id"]
    ):
        raise ContractError(f"{role}: public model/root identity differs from proposal")


def _require_stageb_proposal_match(
    payload: dict[str, Any], binding: ModelContractProposalBinding, *, role: str
) -> None:
    """Bind a serialized train399 detection to the exact proposal Stage-B member set."""

    require_receipt_binding(payload, binding, role=role)
    _require_public_identity(payload, binding, role=role, root_field="root_expert_id")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ContractError(f"{role}: Stage-B model metadata is absent")
    expected = binding.stageb_binding
    comparisons = {
        "expert_id": "expert_id",
        "checkpoint_sha256": "checkpoint_sha256",
        "selected_score_threshold": "selected_score_threshold",
        "candidate_bundle_identity_sha256": "candidate_bundle_identity_sha256",
        "selection_receipt_identity_sha256": "selection_receipt_identity_sha256",
        "selected_model_metadata_identity_sha256": (
            "selected_model_metadata_identity_sha256"
        ),
    }
    for model_field, proposal_field in comparisons.items():
        observed = model.get(model_field)
        authorized = expected.get(proposal_field)
        if model_field == "selected_score_threshold":
            try:
                matches = abs(float(observed) - float(authorized)) <= 1e-12
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == authorized
        if not matches:
            raise ContractError(f"{role}: proposal mismatch in {model_field}")


def _fuse(args: argparse.Namespace) -> int:
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    public_identity = proposal_binding.public_identity_fields()
    hybrid_prediction_root = Path(args.hybrid_predictions)
    hybrid_artifact_root = Path(args.hybrid_root)
    hair_detection_root = Path(args.hair_detections)
    output_root = Path(args.output)
    prediction_paths = sorted(hybrid_prediction_root.glob("*.json"))
    if args.task_id:
        selected = set(args.task_id)
        prediction_paths = [path for path in prediction_paths if path.stem in selected]
        missing = selected - {path.stem for path in prediction_paths}
        if missing:
            raise ContractError(
                f"requested root-provider task IDs are absent: {sorted(missing)}"
            )
    if not prediction_paths:
        raise ContractError(
            f"no PHAxis root-provider predictions in {hybrid_prediction_root}"
        )
    output_prediction_root = output_root / "predictions"
    if output_prediction_root.exists() and any(output_prediction_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_prediction_root}")
    records: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        hybrid = read_json(prediction_path)
        task_id = str(hybrid["task_id"])
        hair_path = hair_detection_root / f"{task_id}.json"
        if not hair_path.is_file():
            raise ContractError(f"missing Stage-B detections: {hair_path}")
        stageb = read_json(hair_path)
        _require_stageb_proposal_match(
            stageb,
            proposal_binding,
            role=f"Stage-B detection {task_id}",
        )
        fused = fuse_hybrid_root_with_stageb_hairs(
            hybrid,
            stageb,
            hybrid_artifact_root=hybrid_artifact_root,
            model_contract_proposal=proposal_binding.receipt_fields(),
            model_contract_public_identity=public_identity,
            maximum_attachment_boundary_error_um=args.attachment_tolerance_um,
            physical_scale_contract=args.physical_scale_contract,
        )
        phaxis = fused.get("phaxis")
        if not isinstance(phaxis, dict):
            raise ContractError(f"{task_id}: fused prediction has no PHAxis provenance")
        fused["phaxis"] = {
            **phaxis,
            **proposal_binding.receipt_fields(),
            "model_bundle_id": public_identity["model_bundle_id"],
            "root_expert": public_identity["root_expert_id"],
        }
        _materialize_locked_artifacts(
            fused, source_root=hybrid_artifact_root, output_root=output_root
        )
        atomic_write_json(output_prediction_root / f"{task_id}.json", fused)
        records.append(
            {
                "task_id": task_id,
                "hair_identity_count_expert": fused["phaxis"][
                    "hair_identity_count_expert"
                ],
                "hybrid_identity_count": fused["phaxis"][
                    "previous_hybrid_identity_count"
                ],
                "phaxis_identity_count": fused["phaxis"][
                    "formal_stageb_identity_count"
                ],
                "attachment_valid_fraction": fused["phaxis"][
                    "attachment_valid_fraction"
                ],
                "root_lock_sha256": fused["phaxis"]["root_lock_sha256"],
                "matched_endpoint_complete_lengths": fused["phaxis"][
                    "length_identity_association"
                ]["matched_length_identities"],
                "prediction_sha256": sha256_file(
                    output_prediction_root / f"{task_id}.json"
                ),
            }
        )
    hair_experts = sorted(
        {str(record["hair_identity_count_expert"]) for record in records}
    )
    if len(hair_experts) != 1:
        raise ContractError(
            f"mixed Stage-B expert identities in one fusion batch: {hair_experts}"
        )
    summary = {
        "schema_version": "PHAxis-fusion-run-1.1",
        "status": "completed",
        "software": {"name": PRODUCT_NAME, "version": _version()},
        "model_bundle_id": public_identity["model_bundle_id"],
        "root_expert": public_identity["root_expert_id"],
        "hair_identity_count_expert": hair_experts[0],
        "images": len(records),
        "hybrid_identity_count": sum(r["hybrid_identity_count"] for r in records),
        "phaxis_identity_count": sum(r["phaxis_identity_count"] for r in records),
        "matched_endpoint_complete_lengths": sum(
            r["matched_endpoint_complete_lengths"] for r in records
        ),
        "hybrid_prediction_root": str(hybrid_prediction_root.resolve()),
        "hybrid_artifact_root": str(hybrid_artifact_root.resolve()),
        "stageb_detection_root": str(hair_detection_root.resolve()),
        "source_hybrid_summary_sha256": (
            sha256_file(hybrid_artifact_root / "summary.json")
            if (hybrid_artifact_root / "summary.json").is_file()
            else None
        ),
        "source_stageb_summary_sha256": (
            sha256_file(hair_detection_root.parent / "summary.json")
            if (hair_detection_root.parent / "summary.json").is_file()
            else None
        ),
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_output": False,
        "records": records,
        "blind_images_used": 0,
        **proposal_binding.receipt_fields(),
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output_root / "fusion_summary.json", summary)
    print(json.dumps(summary | {"records": f"{len(records)} per-image records"}, ensure_ascii=False, indent=2))
    return 0


def _verify_prediction(args: argparse.Namespace) -> int:
    prediction = read_json(args.prediction)
    validate_hybrid_prediction(prediction, artifact_root=args.artifact_root)
    print(
        json.dumps(
            {
                "status": "valid",
                "task_id": prediction["task_id"],
                "schema_version": prediction.get("schema_version"),
                "blind_images_used": prediction["blind_images_used"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _infer_hairs(args: argparse.Namespace) -> int:
    if len(args.checkpoint) != 5:
        raise ContractError("exactly five Stage-B checkpoints are required")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not args.task_id or args.source_um_per_px <= 0:
        raise ContractError("task_id and a positive source_um_per_px are required")
    gate_values = (
        args.model_contract_proposal,
        args.candidate_manifest,
        args.selected_model_metadata,
        args.selection_receipt,
    )
    if not all(value is not None for value in gate_values):
        raise ContractError(
            "formal train399 inference requires --model-contract-proposal, "
            "--candidate-manifest, --selected-model-metadata and "
            "--selection-receipt together; legacy 443CV fallback is forbidden"
        )
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    candidate_manifest = read_json(args.candidate_manifest)
    selected_model_metadata = read_json(args.selected_model_metadata)
    selection_receipt = read_json(args.selection_receipt)
    checkpoint_paths = [Path(path) for path in args.checkpoint]
    validate_stageb_proposal_binding(
        proposal_binding,
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest=candidate_manifest,
        selected_model_metadata_path=args.selected_model_metadata,
        selected_model_metadata=selected_model_metadata,
        selection_receipt_path=args.selection_receipt,
        selection_receipt=selection_receipt,
        checkpoints=checkpoint_paths,
    )
    if not str(args.device).startswith("cuda"):
        raise ContractError("PHAxis 1.0.0 locked Stage-B inference requires CUDA")
    preflight = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    import tifffile
    import torch

    from .hair_stageb.runtime import StageBEnsemble
    from .hair_stageb.serialization import make_detection_payload

    if not torch.cuda.is_available():
        raise ContractError("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    image_path = Path(args.image)
    image_sha256 = sha256_file(image_path)
    started = time.perf_counter()
    ensemble = StageBEnsemble(
        checkpoint_paths,
        device=args.device,
        use_amp=False,
        candidate_manifest=args.candidate_manifest,
        selected_model_metadata=args.selected_model_metadata,
        selection_receipt=args.selection_receipt,
    )
    image = tifffile.imread(image_path)
    prediction = ensemble.predict(image, source_um_per_px=args.source_um_per_px)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    payload = make_detection_payload(
        task_id=args.task_id,
        source_image_sha256=image_sha256,
        source_um_per_px=args.source_um_per_px,
        prediction=prediction,
        precision_mode="fp32_locked",
        model_metadata=ensemble.detection_model_metadata,
        score_threshold=(
            ensemble.score_threshold
            if ensemble.detection_model_metadata is not None
            else None
        ),
    )
    payload.pop("detection_identity_sha256", None)
    payload.update(proposal_binding.receipt_fields())
    payload.update(proposal_binding.public_identity_fields())
    payload["detection_identity_sha256"] = sha256_json(payload)
    _require_stageb_proposal_match(
        payload,
        proposal_binding,
        role=f"Stage-B detection {args.task_id}",
    )
    validate_stageb_detection_payload(
        payload,
        expected_task_id=args.task_id,
        expected_image_sha256=image_sha256,
        expected_model_metadata=ensemble.detection_model_metadata,
    )
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "task_id": args.task_id,
                "detections": payload["n"],
                "output": str(output.resolve()),
                "wall_seconds_including_model_load_and_io": seconds,
                "peak_allocated_vram_mib": float(
                    torch.cuda.max_memory_allocated() / 2**20
                ),
                "nvidia_smi_preflight_captured": bool(preflight),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _export_traits(args: argparse.Namespace) -> int:
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    prediction_paths = sorted(Path(args.predictions).glob("*.json"))
    if not prediction_paths:
        raise ContractError(f"no PHAxis predictions in {args.predictions}")
    for path in prediction_paths:
        prediction = read_json(path)
        phaxis = prediction.get("phaxis")
        if not isinstance(phaxis, dict):
            raise ContractError(f"{path.name}: PHAxis provenance is absent")
        require_receipt_binding(
            phaxis,
            proposal_binding,
            role=f"prediction {path.name}",
        )
        _require_public_identity(
            phaxis,
            proposal_binding,
            role=f"prediction {path.name}",
            root_field="root_expert",
        )
    result = export_traits(
        prediction_root=args.predictions,
        metadata_csv=args.metadata,
        output=args.output,
        model_contract_proposal=proposal_binding.receipt_fields(),
        model_contract_public_identity=proposal_binding.public_identity_fields(),
    )
    print(
        json.dumps(
            result | {"prediction_sha256": f"{result['tasks']} records"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    """Plan or explicitly execute the installable end-to-end workflow."""

    from .workflow import build_analysis_plan, run_analysis

    if args.resume and not args.execute:
        raise ContractError("--resume is valid only together with explicit --execute")
    if args.execute:
        result = run_analysis(
            args.manifest,
            output=args.output,
            resume=bool(args.resume),
            review_overlays=args.review_overlays,
        )
    else:
        result = build_analysis_plan(
            args.manifest,
            output=args.output,
            review_overlays=args.review_overlays,
        )
    if args.plan_output is not None:
        plan_output = Path(args.plan_output).resolve()
        if plan_output.exists():
            raise FileExistsError(f"refusing to overwrite plan output: {plan_output}")
        atomic_write_json(plan_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phaxis",
        description="PHAxis 1.0.0: primary-root and root-hair phenotyping",
    )
    parser.add_argument("--version", action="version", version=f"PHAxis {_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run CPU-only synthetic fusion/trait numerical example")
    demo.add_argument("--output", required=True, help="new demo directory")
    demo.set_defaults(handler=_demo)
    report = subparsers.add_parser("report", help="create an offline report from existing trait tables")
    report.add_argument("--traits", required=True, help="directory containing image_traits.csv")
    report.add_argument("--output", required=True, help="new report directory")
    report.set_defaults(handler=_report)
    analyze = subparsers.add_parser(
        "analyze",
        help=(
            "plan the locked raw-image -> roots -> root-hair -> traits workflow; "
            "execution requires --execute"
        ),
    )
    analyze.add_argument(
        "--manifest",
        required=True,
        help="sealed PHAxis analysis-workflow manifest",
    )
    analyze.add_argument(
        "--output",
        required=True,
        help="new analysis directory (or the bound directory when resuming)",
    )
    analyze.add_argument(
        "--plan-output",
        default=None,
        help="optional new JSON file receiving the printed plan/state",
    )
    analyze.add_argument(
        "--review-overlays",
        action="store_true",
        default=None,
        help="enable optional review-only overlays (never used for routing)",
    )
    analyze.add_argument(
        "--execute",
        action="store_true",
        help="execute the plan; omission is strictly plan-only",
    )
    analyze.add_argument(
        "--resume",
        action="store_true",
        help="resume a verified interrupted run; valid only with --execute",
    )
    analyze.set_defaults(handler=_analyze)
    fuse = subparsers.add_parser(
        "fuse",
        help="combine locked PHAxis root-provider outputs with root-hair detections",
    )
    root_predictions = fuse.add_mutually_exclusive_group(required=True)
    root_predictions.add_argument(
        "--root-predictions",
        dest="hybrid_predictions",
        metavar="ROOT_PREDICTIONS",
        help="directory of locked PHAxis root-provider prediction JSON files",
    )
    root_predictions.add_argument(
        "--hybrid-predictions",
        dest="hybrid_predictions",
        help=argparse.SUPPRESS,
    )
    root_artifacts = fuse.add_mutually_exclusive_group(required=True)
    root_artifacts.add_argument(
        "--root-artifacts",
        dest="hybrid_root",
        metavar="ROOT_ARTIFACTS",
        help="root-provider artifact directory referenced by the predictions",
    )
    root_artifacts.add_argument(
        "--hybrid-root",
        dest="hybrid_root",
        help=argparse.SUPPRESS,
    )
    fuse.add_argument(
        "--hair-detections",
        required=True,
        help="directory of locked PHAxis root-hair detection JSON files",
    )
    fuse_contract = fuse.add_mutually_exclusive_group(required=True)
    fuse_contract.add_argument(
        "--model-contract",
        dest="model_contract_proposal",
        metavar="CONTRACT",
        help="sealed PHAxis model-contract proposal or applied official contract",
    )
    fuse_contract.add_argument(
        "--model-contract-proposal",
        dest="model_contract_proposal",
        help=argparse.SUPPRESS,
    )
    fuse.add_argument("--output", required=True, help="new fused-output directory")
    fuse.add_argument("--attachment-tolerance-um", type=float, default=40.0)
    fuse.add_argument(
        "--physical-scale-contract",
        choices=("strict_root_provider", "stageb_reference_evaluation"),
        default="strict_root_provider",
        help=(
            "strict deployment scale agreement (default), or the locked "
            "Stage-B acquisition scale for QC-development geometry evaluation"
        ),
    )
    fuse.add_argument("--task-id", action="append", default=[])
    fuse.set_defaults(handler=_fuse)
    verify = subparsers.add_parser("verify-prediction", help="verify root locks and provenance")
    verify.add_argument("--prediction", required=True)
    verify.add_argument("--artifact-root", required=True)
    verify.set_defaults(handler=_verify_prediction)
    infer = subparsers.add_parser(
        "infer-hairs",
        help="run the locked PHAxis root-hair identity/count expert on one image",
    )
    infer.add_argument("--image", required=True)
    infer.add_argument("--task-id", required=True)
    infer.add_argument("--source-um-per-px", type=float, required=True)
    infer.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="one selected ensemble checkpoint; repeat exactly five times",
    )
    infer_contract = infer.add_mutually_exclusive_group()
    infer_contract.add_argument(
        "--model-contract",
        dest="model_contract_proposal",
        metavar="CONTRACT",
        default=None,
        help=(
            "sealed PHAxis 1.0.0 model-contract authority "
            "(unapplied proposal or applied official contract)"
        ),
    )
    infer_contract.add_argument(
        "--model-contract-proposal",
        dest="model_contract_proposal",
        help=argparse.SUPPRESS,
    )
    infer.add_argument(
        "--candidate-manifest",
        default=None,
        help="strict train399 candidate Gate receipt",
    )
    infer.add_argument(
        "--selected-model-metadata",
        default=None,
        help="train399 metadata bound to the selected QC-development operating point",
    )
    infer.add_argument(
        "--selection-receipt",
        default=None,
        help="selection receipt whose file and logical identities are rechecked",
    )
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--output", required=True)
    infer.set_defaults(handler=_infer_hairs)
    traits = subparsers.add_parser(
        "export-traits", help="export canonical 32-trait and per-hair tables"
    )
    traits.add_argument(
        "--predictions",
        required=True,
        help="directory of fused PHAxis prediction JSON files",
    )
    traits.add_argument(
        "--metadata",
        required=True,
        help="CSV containing task IDs and biological sample metadata",
    )
    traits_contract = traits.add_mutually_exclusive_group(required=True)
    traits_contract.add_argument(
        "--model-contract",
        dest="model_contract_proposal",
        metavar="CONTRACT",
        help="sealed PHAxis model-contract proposal or applied official contract",
    )
    traits_contract.add_argument(
        "--model-contract-proposal",
        dest="model_contract_proposal",
        help=argparse.SUPPRESS,
    )
    traits.add_argument("--output", required=True, help="new trait-export directory")
    traits.set_defaults(handler=_export_traits)
    return parser


def _demo(args: argparse.Namespace) -> int:
    from .local_demo import run_demo
    print(json.dumps(run_demo(args.output), indent=2))
    return 0


def _report(args: argparse.Namespace) -> int:
    from .offline_report import build_report
    print(json.dumps(build_report(args.traits, args.output), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ContractError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"PHAxis error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
