"""Assemble a hash-sealed PHAxis post-training release manifest.

This module is deliberately CPU-only.  It separates immutable authorities from
stage-produced artefacts, reports incomplete prerequisites without requiring a
pre-existing release manifest, and publishes a formal manifest with an atomic
create-only commit.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping

from .hair_stageb.candidate_bundle import FORMAL_TRAIN399_SEEDS
from .io import read_json, sha256_file, sha256_json
from .release_orchestrator import (
    DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA,
    EXTERNAL_AUTHORITY_CLASSES,
    KNOWN_STAGE_SCHEMAS,
    MANIFEST_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    build_release_plan,
    validate_deferred_human_authority_contract,
)
from .release_topology import (
    MANDATORY_STAGE_ORDER,
    ReleaseTopologyError,
    require_manifest_stage_dependencies,
    validate_release_topology,
)


ASSEMBLY_CONFIG_SCHEMA = "PHAxis-post-training-release-assembly-config-1.0"
STAGE_TEMPLATE_SCHEMA = "PHAxis-post-training-release-stage-contract-template-1.0"
READINESS_SCHEMA = "PHAxis-post-training-release-readiness-1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(
    r"(?:REQUIRED(?:_|\b)|COMPUTE_AFTER|BLOCKED_TEMPLATE|TO_BE_FILLED|<[^>]+>)",
    re.IGNORECASE,
)


class ReleaseManifestAssemblyError(RuntimeError):
    """A release manifest prerequisite or create-only publication Gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseManifestAssemblyError(message)


def _resolve(value: str, *, workspace: Path) -> Path:
    _require(isinstance(value, str) and bool(value), "path is absent")
    _require("*" not in value and "?" not in value, f"glob paths are forbidden: {value}")
    path = Path(value.replace("{workspace}", str(workspace)))
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _directory_lock(path: Path) -> dict[str, Any]:
    _require(path.is_dir(), f"directory authority is missing: {path}")
    records: list[dict[str, Any]] = []
    for member in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        _require(not member.is_symlink(), f"symlink is forbidden: {member}")
        if member.is_file():
            records.append(
                {
                    "path": member.relative_to(path).as_posix(),
                    "size_bytes": member.stat().st_size,
                    "sha256": sha256_file(member),
                }
            )
    return {
        "kind": "directory",
        "sha256": sha256_json(records),
        "files": len(records),
        "size_bytes": sum(item["size_bytes"] for item in records),
        "members": records,
    }


def _path_lock(path: Path, kind: str) -> dict[str, Any]:
    if kind == "file":
        _require(path.is_file(), f"file authority is missing: {path}")
        _require(not path.is_symlink(), f"symlink is forbidden: {path}")
        return {
            "kind": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _require(kind == "directory", f"unsupported authority kind: {kind}")
    return _directory_lock(path)


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReleaseManifestAssemblyError(f"refusing to overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _placeholder_locations(value: Any, prefix: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            locations.extend(_placeholder_locations(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_placeholder_locations(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and _PLACEHOLDER.search(value):
        locations.append(prefix)
    return locations


def _validate_author_metadata_authority(path: Path) -> None:
    """Reject extant author files that are still completion templates."""

    metadata = read_json(path)
    placeholders = _placeholder_locations(metadata)
    _require(
        not placeholders,
        f"author metadata still contains placeholders: {placeholders}",
    )
    status = metadata.get("status")
    _require(isinstance(status, str) and status, "author metadata status is absent")
    _require(
        not re.search(r"(?:BLOCKED|INCOMPLETE|TEMPLATE|DO_NOT_USE)", status, re.IGNORECASE),
        f"author metadata status is not final: {status}",
    )


def _deferred_external_spec(
    name: str,
    raw_spec: Mapping[str, Any],
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize an assembler descriptor without hashing unfinished bytes."""

    deferred = raw_spec.get("deferred_authority")
    _require(isinstance(deferred, Mapping), f"deferred authority contract is invalid: {name}")
    _require(
        raw_spec.get("authority_class") == "author_metadata",
        f"only author_metadata may be deferred: {name}",
    )
    _require(raw_spec.get("kind") == "file", f"deferred authority must be a file: {name}")
    expected_keys = {
        "schema_version",
        "document_schema_version",
        "final_status",
        "first_consumer_stage",
        "human_authority_id",
        "identity_field",
        "status_field",
        "target_path",
    }
    _require(
        set(deferred) == expected_keys,
        f"deferred authority descriptor fields are invalid: {name}",
    )
    _require(
        deferred.get("schema_version")
        == DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA,
        f"deferred authority contract schema is invalid: {name}",
    )
    _require(
        deferred.get("first_consumer_stage") in MANDATORY_STAGE_ORDER
        and MANDATORY_STAGE_ORDER.index(str(deferred["first_consumer_stage"]))
        > MANDATORY_STAGE_ORDER.index("official_apply"),
        f"deferred human first consumer must follow official_apply: {name}",
    )
    target_path = deferred.get("target_path")
    _require(
        isinstance(target_path, str)
        and target_path.replace("\\", "/").startswith(
            "{run_dir}/human_authorities/"
        )
        and ".." not in Path(target_path.replace("{run_dir}/", "")).parts,
        f"deferred authority target must be run-scoped under human_authorities/: {name}",
    )
    draft_template = _resolve(str(raw_spec.get("path", "")), workspace=workspace)
    normalized = {
        "path": target_path,
        "kind": "file",
        "authority_class": "author_metadata",
        "deferred": True,
        "deferred_contract_schema_version": deferred.get("schema_version"),
        "human_authority_id": deferred.get("human_authority_id"),
        "document_schema_version": deferred.get("document_schema_version"),
        "status_field": deferred.get("status_field"),
        "final_status": deferred.get("final_status"),
        "identity_field": deferred.get("identity_field"),
        "first_consumer_stage": deferred.get("first_consumer_stage"),
        "draft_template_path": str(draft_template),
    }
    try:
        validate_deferred_human_authority_contract(
            normalized,
            authority_name=name,
        )
    except Exception as error:
        raise ReleaseManifestAssemblyError(str(error)) from error
    return normalized


def _training_receipt_errors(receipt_path: Path, checkpoint_path: Path, seed: int) -> list[str]:
    missing: list[str] = []
    if not receipt_path.is_file():
        missing.append("completion_receipt_missing")
    if not checkpoint_path.is_file():
        missing.append("checkpoint_missing")
    if missing:
        return missing
    try:
        receipt = read_json(receipt_path)
    except (OSError, ValueError, TypeError) as error:
        return [f"completion_receipt_unreadable:{error}"]
    expected = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": "completed",
        "formal_training": True,
        "seed": seed,
        "epochs": 60,
        "steps_per_epoch": 399,
        "global_steps": 23940,
        "nvidia_smi_preflight_status": "passed",
        "nvidia_smi_training_monitor_status": "passed",
        "validation_evaluated_during_training": False,
        "blind_images_used": 0,
    }
    errors = [f"field_mismatch:{field}" for field, expected_value in expected.items() if receipt.get(field) != expected_value]
    observed_sha = sha256_file(checkpoint_path)
    if receipt.get("checkpoint_sha256") != observed_sha:
        errors.append("checkpoint_sha256_mismatch")
    receipt_checkpoint = receipt.get("checkpoint")
    if not isinstance(receipt_checkpoint, str) or not receipt_checkpoint:
        errors.append("checkpoint_path_absent")
    else:
        named = Path(receipt_checkpoint)
        if not named.is_absolute():
            named = receipt_path.parent / named
        if named.resolve() != checkpoint_path.resolve():
            errors.append("checkpoint_path_mismatch")
    return errors


def _validate_template_command_authorities(
    stages: list[Any],
    *,
    workspace: Path,
    run_dir: Path,
) -> None:
    names = [str(stage["name"]) for stage in stages]
    indices = {name: index for index, name in enumerate(names)}
    resolved_run_dir = run_dir.resolve()
    seen_artifact_paths: dict[Path, tuple[str, str]] = {}
    for index, raw_stage in enumerate(stages):
        _require(isinstance(raw_stage, Mapping), "stage-contract stage is not an object")
        stage = raw_stage
        name = str(stage["name"])
        inputs = stage.get("inputs")
        _require(isinstance(inputs, list), f"{name}: stage inputs are absent")
        external_refs = {
            str(reference["external"])
            for reference in inputs
            if isinstance(reference, Mapping) and "external" in reference
        }
        stage_refs = {
            str(reference["stage"])
            for reference in inputs
            if isinstance(reference, Mapping) and "stage" in reference
        }
        command = stage.get("command")
        if command is not None:
            _require(isinstance(command, list), f"{name}: command is not argv")
            serialized = json.dumps(command, ensure_ascii=False)
            token_external_refs = set(
                re.findall(
                    r"\{(?:external|external_sha256):([^{}]+)\}",
                    serialized,
                )
            )
            _require(
                token_external_refs <= external_refs,
                f"{name}: command uses undeclared external authorities: "
                f"{sorted(token_external_refs - external_refs)}",
            )
            run_owners = set(
                re.findall(r"\{run_dir\}[\\/]+([^\\/={}]+)", serialized)
            )
            for owner in sorted(run_owners):
                _require(owner in indices, f"{name}: command uses unknown run-stage path: {owner}")
                if owner == name:
                    continue
                _require(
                    indices[owner] < index,
                    f"{name}: command uses future-stage authority: {owner}",
                )
                _require(
                    owner in stage_refs,
                    f"{name}: command uses undeclared stage authority: {owner}",
                )
        artifacts = stage.get("artifacts")
        _require(isinstance(artifacts, list), f"{name}: artifacts are absent")
        artifact_names: set[str] = set()
        artifact_kinds: dict[str, str] = {}
        for artifact in artifacts:
            _require(isinstance(artifact, Mapping), f"{name}: artifact is invalid")
            artifact_name = artifact.get("name")
            raw_path = artifact.get("path")
            kind = artifact.get("kind")
            _require(
                isinstance(artifact_name, str) and artifact_name,
                f"{name}: artifact name is invalid",
            )
            _require(
                artifact_name not in artifact_names,
                f"{name}: duplicate artifact name: {artifact_name}",
            )
            _require(
                isinstance(raw_path, str) and bool(raw_path),
                f"{name}: artifact path is invalid",
            )
            _require(kind in {"file", "directory"}, f"{name}: artifact kind is invalid")
            assert isinstance(raw_path, str)
            owners = re.findall(r"\{run_dir\}[\\/]+([^\\/={}]+)", raw_path)
            _require(
                not owners or owners == [name],
                f"{name}: artifact path must be owned by its producing stage",
            )
            expanded = raw_path.replace("{workspace}", str(workspace)).replace(
                "{run_dir}", str(resolved_run_dir)
            )
            _require(
                "{" not in expanded and "}" not in expanded,
                f"{name}: artifact path contains an unknown placeholder",
            )
            _require(
                "*" not in expanded and "?" not in expanded,
                f"{name}: artifact path contains forbidden glob syntax",
            )
            resolved_path = Path(expanded)
            if not resolved_path.is_absolute():
                resolved_path = resolved_run_dir / resolved_path
            resolved_path = resolved_path.resolve()
            try:
                relative_artifact = resolved_path.relative_to(resolved_run_dir)
            except ValueError as error:
                raise ReleaseManifestAssemblyError(
                    f"{name}: artifact path leaves the release run directory"
                ) from error
            _require(
                bool(relative_artifact.parts)
                and relative_artifact.parts[0] == name,
                f"{name}: artifact path must be owned by its producing stage",
            )
            _require(
                resolved_path not in seen_artifact_paths,
                f"stage artifact path reused by {name}/{artifact_name}: "
                f"{seen_artifact_paths.get(resolved_path)}",
            )
            seen_artifact_paths[resolved_path] = (name, artifact_name)
            artifact_names.add(artifact_name)
            artifact_kinds[artifact_name] = str(kind)
        receipt = stage.get("receipt")
        _require(isinstance(receipt, Mapping), f"{name}: receipt contract is absent")
        receipt_artifact = receipt.get("artifact")
        _require(
            receipt_artifact in artifact_names,
            f"{name}: receipt artifact is undeclared",
        )
        _require(
            artifact_kinds.get(str(receipt_artifact)) == "file",
            f"{name}: receipt artifact must be a JSON file",
        )


def _load_config(config_path: str | Path) -> tuple[Path, dict[str, Any], Path]:
    source = Path(config_path).resolve()
    _require(source.is_file(), f"assembly config is missing: {source}")
    config = read_json(source)
    _require(config.get("schema_version") == ASSEMBLY_CONFIG_SCHEMA, "unsupported assembly config schema")
    workspace_raw = config.get("workspace")
    _require(isinstance(workspace_raw, str) and workspace_raw, "workspace is absent")
    workspace = Path(workspace_raw)
    if not workspace.is_absolute():
        workspace = source.parent / workspace
    workspace = workspace.resolve()
    _require(workspace.is_dir(), f"workspace is missing: {workspace}")
    return source, config, workspace


def configured_output_root(config_path: str | Path) -> Path:
    """Resolve the create-only release output root declared by the config."""

    _source, config, workspace = _load_config(config_path)
    return _resolve(str(config.get("output_root", "")), workspace=workspace)


def inspect_release_readiness(config_path: str | Path) -> dict[str, Any]:
    """Return a complete, non-formal readiness report without requiring a manifest."""

    source, config, workspace = _load_config(config_path)
    blockers: list[dict[str, Any]] = []
    authorities_result: dict[str, Any] = {}
    authorities = config.get("authorities")
    if not isinstance(authorities, Mapping) or not authorities:
        blockers.append({"code": "AUTHORITIES_ABSENT"})
        authorities = {}
    deferred_authorities: dict[str, dict[str, Any]] = {}
    for name, raw_spec in sorted(authorities.items()):
        result: dict[str, Any] = {"ready": False}
        try:
            _require(
                name not in set(MANDATORY_STAGE_ORDER),
                f"stage-derived output cannot be an external authority: {name}",
            )
            _require(isinstance(raw_spec, Mapping), "authority spec is not an object")
            authority_class = raw_spec.get("authority_class")
            kind = raw_spec.get("kind")
            _require(authority_class in EXTERNAL_AUTHORITY_CLASSES, f"invalid authority_class: {authority_class}")
            if "deferred_authority" in raw_spec:
                normalized = _deferred_external_spec(
                    str(name), raw_spec, workspace=workspace
                )
                deferred_authorities[str(name)] = normalized
                result.update(
                    ready=True,
                    status="deferred_until_first_consumer",
                    authority_class="author_metadata",
                    target_path=normalized["path"],
                    draft_template_path=normalized["draft_template_path"],
                    human_authority_id=normalized["human_authority_id"],
                    document_schema_version=normalized["document_schema_version"],
                    first_consumer_stage=normalized["first_consumer_stage"],
                    unfinished_template_bytes_locked_by_manifest=False,
                    exact_final_bytes_locked_by_first_consumer_sentinel=True,
                )
                authorities_result[str(name)] = result
                continue
            path = _resolve(str(raw_spec.get("path", "")), workspace=workspace)
            lock = _path_lock(path, str(kind))
            if authority_class == "author_metadata":
                _require(kind == "file", "author metadata authority must be a file")
                _validate_author_metadata_authority(path)
            result.update(
                ready=True,
                path=str(path),
                authority_class=authority_class,
                **lock,
            )
        except (ReleaseManifestAssemblyError, OSError, ValueError) as error:
            result["error"] = str(error)
            blockers.append({"code": "EXTERNAL_AUTHORITY_NOT_READY", "authority": name, "detail": str(error)})
        authorities_result[str(name)] = result

    author_result: dict[str, Any] = {"ready": False}
    author_raw = config.get("author_metadata_template")
    try:
        author_path = _resolve(str(author_raw or ""), workspace=workspace)
        matching_deferred = [
            name
            for name, raw_spec in authorities.items()
            if isinstance(raw_spec, Mapping)
            and "deferred_authority" in raw_spec
            and _resolve(str(raw_spec.get("path", "")), workspace=workspace)
            == author_path
        ]
        if matching_deferred:
            author_result = {
                "ready": True,
                "status": "deferred_until_first_consumer",
                "path": str(author_path),
                "deferred_authorities": matching_deferred,
                "unfinished_template_bytes_locked_by_manifest": False,
            }
        else:
            metadata = read_json(author_path)
            placeholders = _placeholder_locations(metadata)
            _require(not placeholders, f"author metadata still contains placeholders: {placeholders}")
            _require(metadata.get("status") not in {"BLOCKED_TEMPLATE_NOT_AUTHORITY", "blocked"}, "author metadata is a blocked template")
            author_result = {
                "ready": True,
                "path": str(author_path),
                "sha256": sha256_file(author_path),
                "placeholder_locations": [],
            }
    except (ReleaseManifestAssemblyError, OSError, ValueError, TypeError) as error:
        author_result["error"] = str(error)
        blockers.append({"code": "AUTHOR_METADATA_NOT_FINAL", "detail": str(error)})

    training_result: list[dict[str, Any]] = []
    members = config.get("training_members")
    if not isinstance(members, list):
        members = []
    seen: set[int] = set()
    for member in members:
        seed = member.get("seed") if isinstance(member, Mapping) else None
        result: dict[str, Any] = {"seed": seed, "ready": False}
        if seed not in FORMAL_TRAIN399_SEEDS or seed in seen:
            errors = ["invalid_or_duplicate_seed"]
        else:
            seen.add(seed)
            receipt_path = _resolve(str(member.get("completion_receipt", "")), workspace=workspace)
            checkpoint_path = _resolve(str(member.get("checkpoint", "")), workspace=workspace)
            errors = _training_receipt_errors(receipt_path, checkpoint_path, int(seed))
            result.update(
                completion_receipt=str(receipt_path),
                checkpoint=str(checkpoint_path),
            )
            if not errors:
                result.update(
                    ready=True,
                    completion_receipt_sha256=sha256_file(receipt_path),
                    checkpoint_sha256=sha256_file(checkpoint_path),
                )
        if errors:
            result["errors"] = errors
            blockers.append({"code": "TRAINING_MEMBER_NOT_COMPLETE", "seed": seed, "details": errors})
        training_result.append(result)
    missing_seeds = sorted(set(FORMAL_TRAIN399_SEEDS) - seen)
    for seed in missing_seeds:
        blockers.append({"code": "TRAINING_MEMBER_NOT_CONFIGURED", "seed": seed})

    manifest_fields_result: dict[str, Any] = {"ready": False, "errors": []}
    manifest_fields = config.get("manifest_fields")
    if not isinstance(manifest_fields, Mapping):
        manifest_fields_result["errors"].append("manifest_fields_absent")
    else:
        expected_guards = {
            "product": "PHAxis",
            "product_version": "1.0.0",
            "blind_images_used": 0,
            "canonical_annotations_read": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_statistics_included": False,
            "frozen_v1_read_only": True,
        }
        for field, expected in expected_guards.items():
            if manifest_fields.get(field) != expected:
                manifest_fields_result["errors"].append(f"guard_mismatch:{field}")
        for field in ("run_id", "dataset_audit_input", "qcdev_manifest_input", "locked_val_ids_input"):
            value = manifest_fields.get(field)
            if not isinstance(value, str) or not value:
                manifest_fields_result["errors"].append(f"field_absent:{field}")
            elif field != "run_id" and value not in authorities:
                manifest_fields_result["errors"].append(f"authority_reference_missing:{field}:{value}")
        frozen_names = manifest_fields.get("frozen_v1_inputs")
        if not isinstance(frozen_names, list) or not frozen_names:
            manifest_fields_result["errors"].append("frozen_v1_inputs_absent")
        else:
            for name in frozen_names:
                spec = authorities.get(name)
                if not isinstance(spec, Mapping) or spec.get("authority_class") != "frozen_read_only_asset":
                    manifest_fields_result["errors"].append(f"frozen_v1_authority_invalid:{name}")
    manifest_fields_result["ready"] = not manifest_fields_result["errors"]
    if manifest_fields_result["errors"]:
        blockers.append(
            {
                "code": "MANIFEST_FIELDS_NOT_READY",
                "details": list(manifest_fields_result["errors"]),
            }
        )

    template_result: dict[str, Any] = {"ready": False}
    template_raw = config.get("stage_contract_template")
    try:
        template_path = _resolve(str(template_raw or ""), workspace=workspace)
        template = read_json(template_path)
        _require(template.get("schema_version") == STAGE_TEMPLATE_SCHEMA, "unsupported stage-contract template schema")
        stages = template.get("stages")
        _require(isinstance(stages, list), "stage-contract stages are absent")
        names = tuple(stage.get("name") for stage in stages if isinstance(stage, Mapping))
        _require(names == MANDATORY_STAGE_ORDER, "stage-contract order differs from current producer topology")
        template_run_dir = _resolve(str(config.get("output_root", "")), workspace=workspace)
        _validate_template_command_authorities(
            stages,
            workspace=workspace,
            run_dir=template_run_dir,
        )
        stage_inputs = {
            str(stage["name"]): {
                str(reference["stage"])
                for reference in stage.get("inputs", [])
                if isinstance(reference, Mapping) and "stage" in reference
            }
            for stage in stages
        }
        require_manifest_stage_dependencies(stage_inputs)
        input_external_refs = {
            str(reference["external"])
            for stage in stages
            for reference in stage.get("inputs", [])
            if isinstance(reference, Mapping) and "external" in reference
        }
        token_external_refs = {
            match
            for match in re.findall(
                r"\{(?:external|external_sha256):([^{}]+)\}",
                json.dumps(stages, ensure_ascii=False),
            )
        }
        unknown_external_refs = sorted(
            {
                name
                for name in input_external_refs | token_external_refs
                if name not in authorities
            }
        )
        project_root_raw = config.get("project_root", str(workspace))
        project_root = _resolve(str(project_root_raw), workspace=workspace)
        topology = validate_release_topology(project_root=project_root)
        template_result = {
            "ready": True,
            "path": str(template_path),
            "sha256": sha256_file(template_path),
            "stage_count": len(stages),
            "stage_order": list(names),
            "producer_source_checks": topology["real_producer_source_checks"],
            "unconfigured_external_authorities": unknown_external_refs,
        }
        blockers.extend(
            {
                "code": "EXTERNAL_AUTHORITY_NOT_CONFIGURED",
                "authority": name,
                "required_by_stage_contract": True,
            }
            for name in unknown_external_refs
        )
    except (ReleaseManifestAssemblyError, ReleaseTopologyError, OSError, ValueError, TypeError) as error:
        template_result["error"] = str(error)
        blockers.append({"code": "STAGE_CONTRACT_NOT_READY", "detail": str(error)})

    pending_stages = list(MANDATORY_STAGE_ORDER)
    try:
        output_root = configured_output_root(source)
        output_root_result: dict[str, Any] = {
            "path": str(output_root),
            "ready": not output_root.exists(),
            "create_only": True,
        }
        if output_root.exists():
            output_root_result["error"] = "configured output root already exists"
            blockers.append(
                {
                    "code": "OUTPUT_ROOT_NOT_NEW",
                    "detail": str(output_root),
                }
            )
    except (ReleaseManifestAssemblyError, OSError, ValueError) as error:
        output_root_result = {"ready": False, "error": str(error), "create_only": True}
        blockers.append({"code": "OUTPUT_ROOT_NOT_READY", "detail": str(error)})
    if blockers:
        status = "blocked_current_prerequisites"
    elif deferred_authorities:
        status = "ready_to_assemble_science_prefix_human_gate_deferred"
    else:
        status = "ready_to_assemble"
    ready_statuses = {
        "ready_to_assemble",
        "ready_to_assemble_science_prefix_human_gate_deferred",
    }
    first_deferred_consumer = None
    if deferred_authorities:
        first_deferred_consumer = min(
            (
                item["first_consumer_stage"]
                for item in deferred_authorities.values()
            ),
            key=lambda name: MANDATORY_STAGE_ORDER.index(str(name)),
        )
    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA,
        "status": status,
        "formal_manifest_present_or_required_for_check": False,
        "formal_release_allowed": status in ready_statuses,
        "formal_release_completion_requires_deferred_human_authorities": bool(
            deferred_authorities
        ),
        "expected_pause_is_not_training_or_algorithm_failure": bool(
            deferred_authorities
        ),
        "expected_first_deferred_consumer_stage": first_deferred_consumer,
        "deferred_human_authority_names": sorted(deferred_authorities),
        "cpu_only": True,
        "config": {"path": str(source), "sha256": sha256_file(source)},
        "workspace": str(workspace),
        "manifest_schema_target": MANIFEST_SCHEMA,
        "author_metadata": author_result,
        "external_authorities": authorities_result,
        "training_members": training_result,
        "manifest_fields": manifest_fields_result,
        "stage_contract": template_result,
        "configured_output_root": output_root_result,
        "blockers": blockers,
        "pending_derived_stages": pending_stages,
        "pending_derived_assets_are_not_external_authorities": True,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "frozen_v1_read_only": True,
    }
    payload["readiness_identity_sha256"] = sha256_json(payload)
    return payload


def _external_specs(config: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in sorted(config["authorities"].items()):
        if "deferred_authority" in spec:
            result[str(name)] = _deferred_external_spec(
                str(name), spec, workspace=workspace
            )
            continue
        path = _resolve(str(spec["path"]), workspace=workspace)
        lock = _path_lock(path, str(spec["kind"]))
        result[str(name)] = {
            "path": str(path),
            "kind": lock["kind"],
            "sha256": lock["sha256"],
            "authority_class": str(spec["authority_class"]),
        }
    return result


def _materialize_template_value(value: Any, external: Mapping[str, Mapping[str, Any]]) -> Any:
    """Resolve assembler-only authority/schema tokens, preserving run tokens."""

    if isinstance(value, Mapping):
        return {key: _materialize_template_value(child, external) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize_template_value(child, external) for child in value]
    if not isinstance(value, str):
        return value
    known_match = re.fullmatch(r"\{known_stage_schema:([^{}]+)\}", value)
    if known_match:
        stage = known_match.group(1)
        _require(stage in KNOWN_STAGE_SCHEMAS, f"unknown stage schema token: {stage}")
        return KNOWN_STAGE_SCHEMAS[stage]
    for name, spec in external.items():
        value = value.replace(f"{{external:{name}}}", str(spec["path"]))
        sha_token = f"{{external_sha256:{name}}}"
        if sha_token in value:
            _require(
                "sha256" in spec,
                f"deferred human authority has no pre-final byte SHA: {name}",
            )
            value = value.replace(sha_token, str(spec["sha256"]))
    unresolved = re.findall(r"\{(?:external|external_sha256|known_stage_schema):[^{}]+\}", value)
    _require(not unresolved, f"unresolved stage-contract token: {unresolved}")
    return value


def build_release_manifest_object(config_path: str | Path) -> dict[str, Any]:
    """Build but do not publish the complete formal manifest object."""

    readiness = inspect_release_readiness(config_path)
    _require(
        readiness["status"]
        in {
            "ready_to_assemble",
            "ready_to_assemble_science_prefix_human_gate_deferred",
        },
        f"release prerequisites are blocked: {readiness['blockers']}",
    )
    _source, config, workspace = _load_config(config_path)
    template_path = _resolve(str(config["stage_contract_template"]), workspace=workspace)
    template = read_json(template_path)
    manifest_fields = config.get("manifest_fields")
    _require(isinstance(manifest_fields, Mapping), "manifest_fields are absent")
    external = _external_specs(config, workspace)
    payload = deepcopy(dict(manifest_fields))
    payload.update(
        {
            "schema_version": MANIFEST_SCHEMA,
            "workspace": str(workspace),
            "external_inputs": external,
            "training_members": [],
            "stages": _materialize_template_value(template["stages"], external),
        }
    )
    for member in config["training_members"]:
        payload["training_members"].append(
            {
                "seed": int(member["seed"]),
                "completion_receipt_input": str(member["completion_receipt_input"]),
                "checkpoint_input": str(member["checkpoint_input"]),
            }
        )
    payload["manifest_identity_sha256"] = sha256_json(payload)
    return payload


def assemble_release_manifest(
    config_path: str | Path,
    output_path: str | Path,
    *,
    run_dir: str | Path,
    candidate_builder: Callable[..., dict[str, Any]] | None = None,
    validate_plan: bool = True,
) -> dict[str, Any]:
    """Validate and atomically publish one new formal manifest."""

    output = Path(output_path).resolve()
    _require(not output.exists(), f"refusing to overwrite: {output}")
    payload = build_release_manifest_object(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.preflight.", suffix=".json", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if validate_plan:
            kwargs: dict[str, Any] = {}
            if candidate_builder is not None:
                kwargs["candidate_builder"] = candidate_builder
            build_release_plan(temporary, run_dir, **kwargs)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_write_new_json(output, payload)
    return {
        "schema_version": "PHAxis-post-training-release-manifest-assembly-receipt-1.0",
        "status": "formal_manifest_created_no_overwrite",
        "manifest": str(output),
        "manifest_file_sha256": sha256_file(output),
        "manifest_identity_sha256": payload["manifest_identity_sha256"],
        "stage_count": len(payload["stages"]),
        "stage_order": [stage["name"] for stage in payload["stages"]],
        "run_dir": str(Path(run_dir).resolve()),
        "blind_images_used": 0,
    }


__all__ = [
    "ASSEMBLY_CONFIG_SCHEMA",
    "READINESS_SCHEMA",
    "STAGE_TEMPLATE_SCHEMA",
    "ReleaseManifestAssemblyError",
    "assemble_release_manifest",
    "build_release_manifest_object",
    "configured_output_root",
    "inspect_release_readiness",
]
