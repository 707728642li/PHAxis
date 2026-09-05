from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest

from phaxis.contracts import ContractError
from phaxis.io import sha256_json
from phaxis.public_identity import (
    MODEL_BUNDLE_PREFIX,
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    ROOT_EXPERT_PREFIX,
    derive_public_identity,
    validate_proposal_public_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stageb() -> dict[str, object]:
    return {
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "checkpoint_sha256": [sha256_json(index) for index in range(5)],
        "selected_score_threshold": 0.225,
        "candidate_bundle_identity_sha256": sha256_json("candidate"),
        "selection_receipt_identity_sha256": sha256_json("selection"),
        "selected_model_metadata_identity_sha256": sha256_json("metadata"),
    }


def _proposal() -> dict[str, object]:
    stageb = _stageb()
    audit = sha256_json("root-audit")
    pipeline = sha256_json("runtime-and-hardware-specific-pipeline")
    bundle = sha256_json("stable-root-bundle")
    public = derive_public_identity(
        stageb, root_bundle_identity_sha256=bundle
    )
    return {
        "model_bundle_id": public["model_bundle_id"],
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public["root_expert_id"],
            "hair_identity_and_count": stageb["expert_id"],
        },
        "root_expert": {
            "provider_role": public["root_provider_role"],
            "expert_id": public["root_expert_id"],
            "fresh_exact283_audit_identity_sha256": audit,
            "bundle_identity_sha256": bundle,
            "pipeline_identity_sha256": pipeline,
            "root_bundle_authority": {
                "bundle_identity_sha256": bundle,
                "pipeline_identity_sha256": pipeline,
            },
        },
        "promotion": {
            "stageb_binding": stageb,
            "formal_gate_identity_sha256": {
                "root_exact283_audit_identity_sha256": audit,
            },
        },
    }


def test_public_identity_is_stable_to_runtime_pipeline_but_not_bundle() -> None:
    proposal = _proposal()
    observed = validate_proposal_public_identity(proposal)
    assert observed["model_bundle_id"].startswith(MODEL_BUNDLE_PREFIX)
    assert observed["root_expert_id"].startswith(ROOT_EXPERT_PREFIX)

    different_pipeline = deepcopy(proposal)
    replacement = sha256_json("different-gpu-and-sharding")
    different_pipeline["root_expert"]["pipeline_identity_sha256"] = replacement
    different_pipeline["root_expert"]["root_bundle_authority"][
        "pipeline_identity_sha256"
    ] = replacement
    assert validate_proposal_public_identity(different_pipeline) == observed

    different_bundle = derive_public_identity(
        proposal["promotion"]["stageb_binding"],
        root_bundle_identity_sha256=sha256_json("different-root-bundle"),
    )
    assert different_bundle["root_expert_id"] != observed["root_expert_id"]
    assert different_bundle["model_bundle_id"] != observed["model_bundle_id"]


def test_stageb_change_changes_system_bundle_but_not_root_expert() -> None:
    proposal = _proposal()
    observed = validate_proposal_public_identity(proposal)
    changed = deepcopy(proposal["promotion"]["stageb_binding"])
    changed["selection_receipt_identity_sha256"] = sha256_json("new-selection")
    derived = derive_public_identity(
        changed,
        root_bundle_identity_sha256=proposal["root_expert"][
            "bundle_identity_sha256"
        ],
    )
    assert derived["model_bundle_id"] != observed["model_bundle_id"]
    assert derived["root_expert_id"] == observed["root_expert_id"]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("model_bundle_id",), MODEL_BUNDLE_PREFIX + "WRONG"),
        (("root_expert", "expert_id"), ROOT_EXPERT_PREFIX + "WRONG"),
        (("public_system_identity", "identity_sha256"), "0" * 64),
        (("root_expert", "root_bundle_authority", "bundle_identity_sha256"), "1" * 64),
    ),
)
def test_proposal_public_identity_rejects_self_sealed_arbitrary_ids(
    path: tuple[str, ...], value: str
) -> None:
    proposal = _proposal()
    target = proposal
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(ContractError, match="public model/root identity"):
        validate_proposal_public_identity(proposal)


def test_public_entry_documents_use_only_phaxis_1_0_identity_and_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (PROJECT_ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    data_card = (PROJECT_ROOT / "DATA_CARD.md").read_text(encoding="utf-8")
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    user_guide = (PROJECT_ROOT / "docs/phaxis/USER_GUIDE.md").read_text(
        encoding="utf-8"
    )
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    support = (PROJECT_ROOT / "SUPPORT.md").read_text(encoding="utf-8")
    combined = "\n".join(
        (
            readme,
            model_card,
            data_card,
            citation,
            changelog,
            user_guide,
            contributing,
            security,
            support,
        )
    )

    if (PROJECT_ROOT / "SOURCE_MANIFEST.json").is_file():
        # A development source staging deliberately leads with a non-publishable
        # warning, while a formal tree leads with its release identity.  Both
        # must still name PHAxis 1.0.0 as the sole public product in the body.
        assert readme.startswith(
            (
                "# PHAxis 1.0.0 source release\n",
                "# BLOCKED DEVELOPMENT STAGING — DO NOT PUBLISH\n",
            )
        )
        assert "PHAxis 1.0.0 is the sole public product" in readme
    else:
        assert readme.startswith("# PHAxis 1.0.0\n")
    assert model_card.startswith("# PHAxis 1.0.0 model card\n")
    assert data_card.startswith("# PHAxis 1.0.0 data card\n")
    assert re.search(
        r'^version:\s*["\']?1\.0\.0["\']?\s*$', citation, flags=re.MULTILINE
    )
    assert 'title: "PHAxis:' in citation
    assert "0.9.0rc1" not in combined
    assert "tolerant one-to-one biological" in combined
    assert "independent external accuracy" in combined
    assert "one distal/root-cap" in combined
    assert "399-image training partition" in combined
    assert "283-image application" in combined
    assert "five-member root-hair identity/count expert" in combined
    assert "--model-contract official-contract.json" in readme
    assert "docs/phaxis/USER_GUIDE.md" in readme
    folded = combined.casefold()
    for forbidden in (
        "stage-b",
        "stage b",
        "hybrid-max",
        "rhaxiscc",
        "rhaxis_nextgen",
        "0.9.0rc1",
    ):
        assert forbidden not in folded
    for document in (readme, model_card, data_card, user_guide, changelog):
        assert re.search(
            r"(?<![a-z0-9_])r(?:1[6-9]|[2-9][0-9])(?![a-z0-9_])",
            document.casefold(),
        ) is None


def test_workspace_release_registry_quarantines_superseded_snapshot() -> None:
    release_root = PROJECT_ROOT / "release"
    if not release_root.is_dir():
        # The independent formal source tree deliberately excludes the mixed
        # workspace's historical release directory.
        assert (PROJECT_ROOT / "SOURCE_MANIFEST.json").is_file()
        return

    registry_path = release_root / "RELEASE_AUTHORITY_REGISTRY.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["schema_version"] == "PHAxis-release-authority-registry-1.0"
    assert registry["status"] == (
        "formal_release_not_yet_materialized_pending_scientific_and_human_authority_gates"
    )
    assert registry["public_identity"] == {
        "product": "PHAxis",
        "version": "1.0.0",
        "distribution": "phaxis",
        "import_namespace": "phaxis",
        "cli": "phaxis",
        "release_tag": "v1.0.0",
    }
    assert registry["current_formal_source_release"] is None
    assert registry["current_formal_release_gate_receipt"] is None
    assert registry["blind_images_used"] == 0
    assert registry["formal_release_generator"]["path"] == (
        "scripts/phaxis/build_source_release.py"
    )
    assert registry["formal_release_generator"]["sha256"] == _file_sha256(
        PROJECT_ROOT / registry["formal_release_generator"]["path"]
    )
    assert registry["formal_release_generator"]["common_path"] == (
        "scripts/phaxis/source_release_common.py"
    )
    assert registry["formal_release_generator"]["common_sha256"] == _file_sha256(
        PROJECT_ROOT / registry["formal_release_generator"]["common_path"]
    )
    assert registry["release_control"]["stage_contract_sha256"] == _file_sha256(
        PROJECT_ROOT
        / registry["release_control"]["stage_contract"]
    )
    assert registry["release_control"]["assembly_config_sha256"] == _file_sha256(
        PROJECT_ROOT
        / registry["release_control"]["assembly_config"]
    )
    assert registry["release_control"]["orchestrator_sha256"] == _file_sha256(
        PROJECT_ROOT
        / registry["release_control"]["orchestrator"]
    )
    registered_config = json.loads(
        (
            PROJECT_ROOT
            / registry["release_control"]["assembly_config"]
        ).read_text(encoding="utf-8")
    )
    assert registered_config["stage_contract_template"] == (
        registry["release_control"]["stage_contract"]
    )
    for path_field, hash_field in (
        ("topology", "topology_sha256"),
        ("stage_contract_generator", "stage_contract_generator_sha256"),
        ("manifest_assembler", "manifest_assembler_sha256"),
        ("assembler_cli", "assembler_cli_sha256"),
    ):
        assert registry["release_control"][hash_field] == _file_sha256(
            PROJECT_ROOT / registry["release_control"][path_field]
        )

    excluded = registry["excluded_non_authorities"]
    assert len(excluded) == 1
    record = excluded[0]
    assert record["path"] == "release/PHAxis_V1_0_Source_20260828"
    assert record["status"] == "superseded_historical_snapshot_do_not_ship"
    assert record["formal_release_allowed"] is False

    snapshot = PROJECT_ROOT / record["path"]
    marker_path = PROJECT_ROOT / record["quarantine_marker"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "superseded_historical_snapshot_do_not_ship"
    assert marker["public_release_authority"] is False
    assert marker["formal_release_allowed"] is False
    assert marker["installation_allowed"] is False
    assert marker["citation_allowed"] is False
    assert marker["submission_use_allowed"] is False
    assert marker["blind_images_used"] == 0
    assert marker["original_snapshot_sha256"]["SOURCE_MANIFEST.json"] == record[
        "source_manifest_sha256"
    ]
    assert marker["quarantined_readme_sha256"] == _file_sha256(
        snapshot / "README.md"
    )
    assert (snapshot / "README.md").read_text(encoding="utf-8").startswith(
        "# SUPERSEDED DEVELOPMENT SNAPSHOT — DO NOT SHIP, INSTALL, CITE, OR PUBLISH\n"
    )
    assert not (snapshot / "FORMAL_RELEASE_GATE_RECEIPT.json").exists()
    assert (release_root / "README.md").read_text(encoding="utf-8").startswith(
        "# PHAxis release authority\n"
    )
