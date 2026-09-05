from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "phaxis"
sys.path.insert(0, str(SCRIPT_ROOT))

from compile_manuscript import (  # noqa: E402
    EXPECTED_EVIDENCE_ROLES,
    ManuscriptCompileError,
    build_token_source_contract,
    compile_manuscript,
)
from phaxis.manuscript_values import (  # noqa: E402
    HUMAN_METADATA_TOKENS,
    SAME_HARDWARE_RUNTIME_TOKENS,
    SOFTWARE_RELEASE_TOKENS,
    derivation_source,
    seal_derivation,
)


def _canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload, *, allow_nan: bool = False) -> Path:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=allow_nan,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _evidence_graph(path: Path) -> Path:
    graph = {
        "schema_version": "PHAxis-manuscript-release-evidence-graph-1.1",
        "status": "passed_formal_evidence_graph",
        "formal_release_evidence_closed": True,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "cohort_identities": {"full": "full283", "primary": "clean261"},
        "stageb_binding": {"expert_id": "synthetic-final-expert"},
        "artifacts": {
            role: {"source_file_sha256": hashlib.sha256(role.encode()).hexdigest()}
            for role in EXPECTED_EVIDENCE_ROLES
        },
        "figure_table_identities": {},
    }
    graph["manifest_identity_sha256"] = _canonical_hash(graph)
    return _write_json(path, graph)


def _master(path: Path) -> Path:
    path.write_text(
        "# PHAxis compiler fixture\n\n"
        "## Abstract\n\n"
        "PHAxis measures visible root and root-hair traits.\n\n"
        "## 1. Results\n\n"
        "Hair F1 {{FINAL_HAIR_F1_20UM}}. Root Dice {{FINAL_ROOT_DICE}}.\n"
        "Release {{PHAXIS_RELEASE_TAG}}.\n"
        "Contributions: {{FINAL_AUTHOR_CONTRIBUTIONS}}\n\n"
        "### Figure 1. ⟦RESULT SLOT → publication_title_contract.figures.1⟧\n\n"
        "### Figure 2. ⟦RESULT SLOT → publication_title_contract.figures.2⟧\n\n"
        "### Figure 3. ⟦RESULT SLOT → publication_title_contract.figures.3⟧\n\n"
        "### Figure 4. ⟦RESULT SLOT → publication_title_contract.figures.4⟧\n\n"
        "### Figure 5. ⟦RESULT SLOT → publication_title_contract.figures.5⟧\n\n"
        "### Figure 6. ⟦RESULT SLOT → publication_title_contract.figures.6⟧\n\n"
        "### Table 1. ⟦RESULT SLOT → publication_title_contract.tables.1⟧\n\n"
        "### Table 2. ⟦RESULT SLOT → publication_title_contract.tables.2⟧\n\n"
        "### Table 3. ⟦RESULT SLOT → publication_title_contract.tables.3⟧\n",
        encoding="utf-8",
    )
    return path


def _values(master: Path, evidence: Path, path: Path) -> dict:
    contract = build_token_source_contract(master.read_text(encoding="utf-8"))
    resolved = {
        "FINAL_HAIR_F1_20UM": "0.91",
        "FINAL_ROOT_DICE": "0.95",
        "PHAXIS_RELEASE_TAG": "v1.0.0",
        "FINAL_AUTHOR_CONTRIBUTIONS": "A.B. designed and validated the study.",
    }
    entries = {}
    source_release_manifest_sha = hashlib.sha256(b"source-release-manifest").hexdigest()
    source_release_tree_identity = hashlib.sha256(b"source-release-tree").hexdigest()
    source_release_metadata_sha = hashlib.sha256(b"source-release-metadata").hexdigest()
    source_release_metadata_identity = hashlib.sha256(b"source-release-metadata-id").hexdigest()
    software_cross_binding_identity = hashlib.sha256(b"software-cross-binding").hexdigest()
    source_files = {
        "source_release_manifest": {
            "sha256": source_release_manifest_sha,
            "logical_identity_sha256": source_release_tree_identity,
        },
        "source_release_metadata": {
            "sha256": source_release_metadata_sha,
            "logical_identity_sha256": source_release_metadata_identity,
        },
    }
    derivation_contract = {}
    for token, row in contract["tokens"].items():
        physical_role = f"synthetic:{token}"
        source_sha = hashlib.sha256(f"source:{token}".encode()).hexdigest()
        container = hashlib.sha256(f"container:{token}".encode()).hexdigest()
        source = derivation_source(
            source_role=physical_role,
            source_file_sha256=source_sha,
            container_identity_sha256=container,
            locator={"kind": "synthetic_fixture_cell", "token": token},
            source_value=resolved[token],
            authority_class=(
                "human_external"
                if token in HUMAN_METADATA_TOKENS
                else "final_machine"
            ),
        )
        sources = [source]
        if token in SOFTWARE_RELEASE_TOKENS:
            sources.append(
                derivation_source(
                    source_role="source_release_metadata",
                    source_file_sha256=source_release_metadata_sha,
                    source_logical_identity_sha256=source_release_metadata_identity,
                    container_identity_sha256=source_release_tree_identity,
                    locator={"kind": "release_coordinate", "token": token},
                    source_value=resolved[token],
                    authority_class="final_machine",
                )
            )
        derivation = seal_derivation(
            {"operation": "synthetic_hash_bound_fixture", "sources": sources}
        )
        entries[token] = {
            "value": resolved[token],
            "source_role": row["source_role"],
            "derivation": derivation,
        }
        source_files[physical_role] = {
            "sha256": source_sha,
            "container_identity_sha256": container,
        }
        derivation_contract[token] = {
            "operation": derivation["operation"],
            "source_roles": [item["source_role"] for item in sources],
            "locators": [item["locator"] for item in sources],
        }
    digest = hashlib.sha256(b"synthetic-binding").hexdigest()
    narrative_identity = hashlib.sha256(b"synthetic-narrative-decision").hexdigest()
    title_contract = {
        "narrative_decision_identity_sha256": narrative_identity,
        "branch_id": "C",
        "figures": {str(index): f"Fixture figure {index}" for index in range(1, 7)},
        "tables": {str(index): f"Fixture table {index}" for index in range(1, 4)},
    }
    title_contract["title_contract_identity_sha256"] = _canonical_hash(title_contract)
    payload = {
        "schema_version": "PHAxis-manuscript-values-1.2",
        "builder_schema_version": "PHAxis-manuscript-values-builder-1.1",
        "status": "final_values_machine_derived_locked",
        "master_sha256": _file_hash(master),
        "evidence_graph_file_sha256": _file_hash(evidence),
        "evidence_graph_identity_sha256": json.loads(
            evidence.read_text(encoding="utf-8")
        )["manifest_identity_sha256"],
        "figure_inputs_file_sha256": digest,
        "figure_input_assembly_identity_sha256": digest,
        "figure_assembly_summary_file_sha256": digest,
        "model_contract_proposal_sha256": digest,
        "model_contract_proposal_identity_sha256": digest,
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-" + "A" * 20,
        "root_expert_id": "PHAxis-root-provider-" + digest[:20].upper(),
        "root_bundle_identity_sha256": digest,
        "hair_identity_count_expert": "PHAxis-StageB-train399-five-seed",
        "human_metadata_file_sha256": digest,
        "human_metadata_identity_sha256": digest,
        "model_bundle_manifest_file_sha256": digest,
        "model_bundle_manifest_identity_sha256": digest,
        "clean_install_receipt_file_sha256": digest,
        "clean_install_receipt_identity_sha256": digest,
        "source_release_manifest_file_sha256": source_release_manifest_sha,
        "source_release_tree_identity_sha256": source_release_tree_identity,
        "source_release_metadata_file_sha256": source_release_metadata_sha,
        "source_release_metadata_identity_sha256": source_release_metadata_identity,
        "software_release_cross_binding_identity_sha256": software_cross_binding_identity,
        "narrative_decision_identity_sha256": narrative_identity,
        "narrative_branch_id": "C",
        "publication_title_contract": title_contract,
        "token_contract_identity_sha256": contract["contract_identity_sha256"],
        "token_derivation_contract_identity_sha256": _canonical_hash(
            derivation_contract
        ),
        "source_files": source_files,
        "historical_source_policy": {
            "allowed_token_prefixes": [
                "HISTORICAL_OOF_",
                "LOCKED_LEGACY_HYBRID_IDENTITY_",
            ],
            "development_or_comparator_semantics_required": True,
            "historical_sources_may_satisfy_final_tokens": False,
        },
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "values": entries,
    }
    payload["values_identity_sha256"] = _canonical_hash(payload)
    _write_json(path, payload)
    return payload


def _reseal(payload: dict) -> None:
    payload.pop("values_identity_sha256", None)
    payload["values_identity_sha256"] = _canonical_hash(payload)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    master = _master(tmp_path / "master.md")
    evidence = _evidence_graph(tmp_path / "evidence.json")
    values = tmp_path / "values.json"
    _values(master, evidence, values)
    return master, evidence, values


def test_compiles_deterministically_and_writes_hash_closed_receipt(tmp_path: Path) -> None:
    master, evidence, values = _fixture(tmp_path)
    first_output = tmp_path / "compiled-a.md"
    second_output = tmp_path / "compiled-b.md"
    first_receipt = tmp_path / "receipt-a.json"
    second_receipt = tmp_path / "receipt-b.json"
    first = compile_manuscript(
        master=master,
        evidence_graph=evidence,
        values=values,
        output=first_output,
        receipt=first_receipt,
    )
    second = compile_manuscript(
        master=master,
        evidence_graph=evidence,
        values=values,
        output=second_output,
        receipt=second_receipt,
    )
    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_receipt.read_bytes() == second_receipt.read_bytes()
    assert b"{{" not in first_output.read_bytes()
    assert first["token_count"] == 4
    assert first["output_sha256"] == _file_hash(first_output)
    unsigned = dict(first)
    identity = unsigned.pop("receipt_identity_sha256")
    assert identity == _canonical_hash(unsigned)
    assert first["blind_images_used"] == 0
    assert first["root_cap_region_statistics_included"] is False
    assert first["abstract_word_count"] == 7
    assert first["abstract_word_limit"] == 249
    assert first["abstract_word_limit_passed"] is True
    assert first["publication_title_slot_count"] == 9
    assert first["narrative_branch_id"] == "C"
    assert first["software_release_cross_binding_identity_sha256"] == hashlib.sha256(
        b"software-cross-binding"
    ).hexdigest()
    assert not list(tmp_path.glob(".*.tmp"))


def test_rejects_final_abstract_over_plant_phenomics_limit(tmp_path: Path) -> None:
    master, evidence, values = _fixture(tmp_path)
    text = master.read_text(encoding="utf-8")
    text = text.replace(
        "PHAxis measures visible root and root-hair traits.",
        " ".join(["root"] * 250),
    )
    master.write_text(text, encoding="utf-8")
    payload = json.loads(values.read_text(encoding="utf-8"))
    payload["master_sha256"] = _file_hash(master)
    _reseal(payload)
    _write_json(values, payload)
    output = tmp_path / "blocked-overlength-abstract.md"
    with pytest.raises(
        ManuscriptCompileError,
        match=r"Plant Phenomics abstract word limit exceeded: 250 > 249",
    ):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("tampered_input", ["master", "evidence"])
def test_rejects_input_tampering_after_values_binding(
    tmp_path: Path, tampered_input: str
) -> None:
    master, evidence, values = _fixture(tmp_path)
    if tampered_input == "master":
        master.write_text(master.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        message = "master SHA-256 binding mismatch"
    else:
        graph = json.loads(evidence.read_text(encoding="utf-8"))
        graph.pop("manifest_identity_sha256")
        graph["tampered_but_resealed"] = True
        graph["manifest_identity_sha256"] = _canonical_hash(graph)
        _write_json(evidence, graph)
        message = "evidence graph file SHA-256 binding mismatch"
    output = tmp_path / "blocked.md"
    with pytest.raises(ManuscriptCompileError, match=message):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=output,
        )
    assert not output.exists()


def test_rejects_unfilled_author_metadata(tmp_path: Path) -> None:
    master, evidence, values = _fixture(tmp_path)
    payload = json.loads(values.read_text(encoding="utf-8"))
    payload["values"]["FINAL_AUTHOR_CONTRIBUTIONS"]["value"] = ""
    _reseal(payload)
    _write_json(values, payload)
    with pytest.raises(ManuscriptCompileError, match="empty value is forbidden"):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=tmp_path / "blocked.md",
        )


def test_rejects_provisional_value(tmp_path: Path) -> None:
    master, evidence, values = _fixture(tmp_path)
    payload = json.loads(values.read_text(encoding="utf-8"))
    payload["values"]["FINAL_HAIR_F1_20UM"]["value"] = "provisional 0.91"
    _reseal(payload)
    _write_json(values, payload)
    with pytest.raises(ManuscriptCompileError, match="provisional value is forbidden"):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=tmp_path / "blocked.md",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "token key mismatch"),
        ("extra", "token key mismatch"),
        ("null", "null value is forbidden"),
        ("string_nan", "null/NaN/infinity text is forbidden"),
        ("todo", "TODO/TBD/provisional value is forbidden"),
        ("tbd", "TODO/TBD/provisional value is forbidden"),
        ("residual", "residual manuscript token is forbidden"),
    ],
)
def test_rejects_incomplete_or_placeholder_value_sets(
    tmp_path: Path, mutation: str, message: str
) -> None:
    master, evidence, values = _fixture(tmp_path)
    payload = json.loads(values.read_text(encoding="utf-8"))
    entries = payload["values"]
    if mutation == "missing":
        entries.pop("FINAL_ROOT_DICE")
    elif mutation == "extra":
        entries["EXTRA_TOKEN"] = {
            "value": "unsupported",
            "source_role": "template_registry",
        }
    elif mutation == "null":
        entries["FINAL_ROOT_DICE"]["value"] = None
    elif mutation == "string_nan":
        entries["FINAL_ROOT_DICE"]["value"] = "NaN"
    elif mutation == "todo":
        entries["FINAL_ROOT_DICE"]["value"] = "TODO"
    elif mutation == "tbd":
        entries["FINAL_ROOT_DICE"]["value"] = "TBD"
    else:
        entries["FINAL_ROOT_DICE"]["value"] = "{{FINAL_ROOT_DICE}}"
    _reseal(payload)
    _write_json(values, payload)
    with pytest.raises(ManuscriptCompileError, match=message):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=tmp_path / "blocked.md",
        )


def test_rejects_nonfinite_json_value(tmp_path: Path) -> None:
    master, evidence, values = _fixture(tmp_path)
    payload = json.loads(values.read_text(encoding="utf-8"))
    payload["values"]["FINAL_ROOT_DICE"]["value"] = float("nan")
    _write_json(values, payload, allow_nan=True)
    with pytest.raises(ManuscriptCompileError, match="without duplicate keys or NaN"):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=tmp_path / "blocked.md",
        )


@pytest.mark.parametrize("existing", ["output", "receipt"])
def test_refuses_to_overwrite_output_or_receipt(tmp_path: Path, existing: str) -> None:
    master, evidence, values = _fixture(tmp_path)
    output = tmp_path / "owned.md"
    receipt = tmp_path / "owned.receipt.json"
    owned = output if existing == "output" else receipt
    owned.write_text("owned\n", encoding="utf-8")
    with pytest.raises(ManuscriptCompileError, match="refusing to overwrite"):
        compile_manuscript(
            master=master,
            evidence_graph=evidence,
            values=values,
            output=output,
            receipt=receipt,
        )
    assert owned.read_text(encoding="utf-8") == "owned\n"
    other = receipt if existing == "output" else output
    assert not other.exists()


def test_current_master_tokens_all_have_machine_source_contract() -> None:
    master = (
        PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    contract = build_token_source_contract(master)
    assert len(contract["tokens"]) >= 180
    assert contract["tokens"]["FINAL_AUTHOR_CONTRIBUTIONS"]["source_role"] == (
        "author_submission_metadata"
    )
    assert all(
        contract["tokens"][token]["family"]
        == "same_hardware_frozen_v1_benchmark"
        for token in SAME_HARDWARE_RUNTIME_TOKENS
    )
    assert "UPPER_SNAKE_CASE" not in contract["tokens"]


def test_continuity_and_attachment_tokens_precede_broad_root_and_hair_families() -> None:
    master = (
        "{{FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE}}\n"
        "{{FINAL_HAIR_ATTACHMENT_QUALIFIED_F1_AT_20UM}}\n"
        "{{FINAL_ROOT_DICE}}\n"
        "{{FINAL_HAIR_F1_20UM}}\n"
    )
    rows = build_token_source_contract(master)["tokens"]
    assert rows["FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE"] == {
        "family": "root_continuity_assurance",
        "source_role": "measurement_assurance",
        "required_evidence_roles": ["figures"],
    }
    assert rows["FINAL_HAIR_ATTACHMENT_QUALIFIED_F1_AT_20UM"] == {
        "family": "hair_attachment_assurance",
        "source_role": "measurement_assurance",
        "required_evidence_roles": ["figures"],
    }
    assert rows["FINAL_ROOT_DICE"]["family"] == "root_distal_scale_assurance"
    assert rows["FINAL_HAIR_F1_20UM"]["family"] == "train399_evaluation"
