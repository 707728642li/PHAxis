"""Audit, cache and train the leak-free PHAxis Stage-B train399 ensemble.

Examples
--------
Audit the locked release (run its bundled verifier first)::

    python -B scripts/phaxis/train_stageb_train399.py audit

Materialize the project-local read-only 2 um/px training cache::

    python -B scripts/phaxis/train_stageb_train399.py cache \
      --readonly-reuse-root <readonly-reuse-root>/2umpx

Train one formal independent seed on the first visible CUDA device::

    python -B scripts/phaxis/train_stageb_train399.py train \
      --seed 2026082801 --device cuda:0

There is intentionally no argument that accepts an initialization checkpoint.
The only checkpoint load path is atomic resume of the same seed/contract.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.hair_stageb.training import (  # noqa: E402
    FORMAL_SEEDS,
    StageBTrain399Config,
    train_one_seed,
)
from phaxis.hair_stageb.training_data import (  # noqa: E402
    load_locked_records,
    materialize_image_cache,
    training_records,
    write_dataset_audit,
)
from phaxis.io import sha256_file  # noqa: E402


DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "rhaxis_arabidopsis_human_curated443_v1_0"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "phaxis_stageb_train399_dataset_audit_run1"
    / "dataset_audit.json"
)
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "rhaxis_nextgen"
    / "splits"
    / "qc_development_v1_0"
    / "split_manifest.csv"
)
DEFAULT_SPLIT_LOCK = DEFAULT_SPLIT_MANIFEST.parent / "split_lock.json"
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "outputs"
    / "phaxis_stageb_train399_cache_canonical_v1_0"
    / "2umpx"
)
DEFAULT_MODEL_ROOT = (
    PROJECT_ROOT / "models" / "phaxis_stageb_train399_v1_0_20260828"
)
WRITE_SIDE_EFFECT_INCIDENT = {
    "status": "recovered_exact",
    "command": (
        ".\\envs\\rhpheno\\python.exe "
        "data\\rhaxis_arabidopsis_human_curated443_v1_0\\tools\\"
        "verify_rhaxis_human_curated443_dataset.py "
        "data\\rhaxis_arabidopsis_human_curated443_v1_0"
    ),
    "cause": (
        "dataset-bundled verifier writes verification_report.json after successful checks"
    ),
    "content_difference": "verified_utc only",
    "before_recovery_sha256": (
        "699b36acda4d5e912a8f2a37be0c558107d1d306d3e09a06db177c36df7aa61b"
    ),
    "recovery_source": str(
        PROJECT_ROOT
        / "models"
        / "rhaxis_nextgen_candidate_20260821"
        / "dataset_contract_snapshot"
        / "verification_report.json"
    ),
    "recovery_source_sha256": (
        "2e97d5f0938df5497faca25ce7c3fad420692c0a19f4c2ffb1d970b79294687a"
    ),
    "current_canonical_sha256": (
        "2e97d5f0938df5497faca25ce7c3fad420692c0a19f4c2ffb1d970b79294687a"
    ),
    "recovery_exact": True,
    "model_training_had_started": False,
    "model_checkpoints_affected": 0,
    "future_verification_policy": (
        "read-only integrity-manifest rehash; never call the write-back entry point"
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "cache", "train"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_SPLIT_LOCK)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--readonly-reuse-root",
        type=Path,
        default=None,
        help="optional validated 2um/px cache; arrays are copied, never modified",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, default=FORMAL_SEEDS[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="non-formal 1-epoch/1-crop run written below outputs, not models",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "audit":
        records, audit = write_dataset_audit(
            args.dataset,
            args.audit_output,
            split_manifest=args.split_manifest,
            split_lock=args.split_lock,
            rehash_integrity=True,
            write_side_effect_incident=WRITE_SIDE_EFFECT_INCIDENT,
        )
    else:
        if not args.audit_output.is_file():
            raise FileNotFoundError(
                f"locked read-only audit missing; run the audit action first: {args.audit_output}"
            )
        records, current_identity = load_locked_records(
            args.dataset,
            split_manifest=args.split_manifest,
            split_lock=args.split_lock,
        )
        audit = json.loads(args.audit_output.read_text(encoding="utf-8"))
        for field in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "integrity_manifest_sha256",
            "dataset_split_identity_sha256",
            "locked_split_identity_sha256",
            "train_ids_sha256",
            "excluded_val_ids_sha256",
        ):
            if audit.get(field) != current_identity.get(field):
                raise RuntimeError(f"locked dataset audit is stale at field {field}")
        current_report_sha256 = sha256_file(args.dataset / "verification_report.json")
        if audit.get("verification_report_sha256") != current_report_sha256:
            raise RuntimeError("canonical verification_report changed after audit lock")
        if audit.get("integrity_recheck", {}).get("status") != "passed":
            raise RuntimeError("locked dataset audit lacks a passed full integrity recheck")
        if audit.get("write_side_effect_incident", {}).get("recovery_exact") is not True:
            raise RuntimeError("dataset write-side-effect recovery is not exact")
        audit["dataset_audit_sha256"] = sha256_file(args.audit_output)
    print(
        f"[audit] {audit['train_records']} train / "
        f"{audit['excluded_val_records']} excluded val; "
        f"families overlap={audit['family_key_overlap']}",
        flush=True,
    )
    if args.action == "audit":
        print(args.audit_output.resolve())
        return

    train = training_records(records)
    cache_audit = materialize_image_cache(
        train,
        args.cache_root,
        target_um_per_px=2.0,
        readonly_reuse_root=args.readonly_reuse_root,
        hash_arrays=True,
    )
    cache_audit_path = args.cache_root.parent / "cache_audit.json"
    cache_audit["cache_audit_sha256"] = sha256_file(cache_audit_path)
    print(
        f"[cache] {cache_audit['records']} train arrays identity="
        f"{cache_audit['cache_identity_sha256']}",
        flush=True,
    )
    if args.action == "cache":
        print(cache_audit_path.resolve())
        return

    config = StageBTrain399Config()
    if args.workers is not None:
        config = replace(config, workers=args.workers)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    formal = not args.smoke
    if args.smoke:
        config = replace(
            config,
            epochs=1,
            crops_per_image=1,
            warmup_epochs=1,
            fixed_last_epoch_policy=False,
        )
        run_dir = (
            PROJECT_ROOT
            / "outputs"
            / "phaxis_stageb_train399_smoke_run2"
            / f"seed_{args.seed}"
        )
    else:
        run_dir = args.model_root / f"seed_{args.seed}"
    receipt = train_one_seed(
        records=train,
        dataset_audit=audit,
        cache_audit=cache_audit,
        cache_root=args.cache_root,
        run_dir=run_dir,
        project_root=PROJECT_ROOT,
        seed=args.seed,
        device=args.device,
        config=config,
        formal=formal,
        resume=args.resume,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
