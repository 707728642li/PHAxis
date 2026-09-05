"""Emit the explicit current-topology PHAxis post-training release argv contract.

This builder is CPU-only and does not inspect result contents or run a stage.
The generated JSON contains only existing project entry points, explicit input
authorities and run-scoped derived output paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.release_topology import MANDATORY_STAGE_ORDER, STAGE_DEPENDENCIES  # noqa: E402
from phaxis.io import atomic_write_json  # noqa: E402


def E(name: str) -> dict[str, str]:
    return {"external": name}


def R(stage: str, artifact: str = "receipt") -> dict[str, str]:
    return {"stage": stage, "artifact": artifact}


def P(stage: str, name: str) -> str:
    return f"{{run_dir}}/{stage}/{name}"


def A(name: str, path: str, kind: str = "file") -> dict[str, str]:
    return {"name": name, "path": path, "kind": kind}


def gpu_contract(physical_gpu: int) -> dict[str, object]:
    if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int):
        raise TypeError("physical GPU must be an integer")
    if physical_gpu < 0:
        raise ValueError("physical GPU must be non-negative")
    return {
        "physical_gpus": [physical_gpu],
        "cuda_visible_devices": str(physical_gpu),
        "internal_device": "cuda:0",
        "estimated_peak_memory_mib": 12288,
        "reserve_memory_mib": 2048,
        "maximum_utilization_pct": 80,
    }


# Preserve the checked-in mixed-card authority as the default: scientific
# inference on physical GPU1 and the frozen-v1-bound same-hardware benchmark /
# clean-install work on physical GPU0.  The scientific card is configurable;
# the comparator remains GPU0 because its frozen read-only runtime enforces
# that mapping internally.
GPU = gpu_contract(1)
GPU0 = gpu_contract(0)

STRICT_GPU_ENV = {"PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU": "1"}

STATUS = {
    "candidate_manifest": "candidate_gate_passed_not_promoted",
    "production_manifest": "completed",
    "direct_benchmark_provider_descriptor": "ready_hash_locked_direct_execution",
    "release_case_prelocks": "completed_result_independent_exact283_case_prelocks",
    "qcdev_candidate_pool": "completed",
    "selection": "completed",
    "qcdev_evaluation_inference": "completed",
    "qcdev_evaluation": "completed",
    "root_provider_exact283": "pass_exact_283",
    "root_bundle_materialization": "pass",
    "proposal": "passed_proposal_not_official",
    "authority_pin": "sealed_unapplied_proposal_for_production",
    "analysis_workflow_manifest": "ready_hash_locked_full_workflow",
    "clean_install_sample_manifest": "completed_real_nonblind_release_example_manifest",
    "qcdev_root_inputs": "completed_locked_exact44_label_free_source_contract",
    "qcdev_root_provider": "completed_uncompared",
    "qcdev_fusion": "completed",
    "production_stageb_exact283": "completed",
    "fusion_exact283": "completed",
    "figure1_geometry_materialization": "completed_from_preselected_case_and_final_prediction",
    "traits_exact283": "completed",
    "cohorts_exact283": "completed_without_fitting_biological_effect_models",
    "biological_analysis": "completed_exploratory_clean_primary_full_sensitivity",
    "profiles_exact283": "completed",
    "profile_analysis": "completed_exploratory_source_unit_profile_summaries",
    "historical_oof_evidence": "completed_locked_historical_oof443_development",
    "measurement_assurance": "completed_locked_qc_development_assurance",
    "overlay_evidence": "completed_locked_preselected_gallery_and_exact_cohort_review_export",
    "benchmark_phaxis_production": "completed_direct_full283",
    "benchmark_frozen_v1_production": "completed_direct_full283",
    "benchmark_phaxis_sequential": "completed_direct_full283",
    "benchmark_frozen_v1_sequential": "completed_direct_full283",
    "benchmark_production_comparison": "comparable_direct_full283",
    "benchmark_sequential_comparison": "comparable_direct_full283",
    "benchmark_same_hardware": "passed",
    "benchmark_artifact_inventory": "completed_explicit_benchmark_inventory",
    "figure_inputs": "completed_final",
    "figures": "final_sealed_strict_train399_only",
    "evidence": "passed_formal_evidence_graph",
    "official_apply": "applied",
    "source_release": "formal",
    "distributions": "completed_wheel_sdist_verified",
    "offline_dependencies": "completed_locked_cp312_win_amd64",
    "handover_dataset_manifest": "created",
    "handover_image_manifest": "created",
    "handover_model_source_manifest": "created",
    "handover_model_asset_manifest": "created",
    "clean_install_expected_identity": "completed_fresh_real_nonblind_reference",
    "handover_benchmark_manifest": "created",
    "clean_install": "completed_final_clean_install",
    "values": "final_values_machine_derived_locked",
    "manuscript": "completed_strict_final_manuscript_compilation",
    "supplementary_manuscript": "completed_strict_final_supplementary_compilation",
    "submission_docx": "completed_final_double_anonymous_submission_bundle",
    "supplementary_docx": "completed_final_anonymized_supplementary_docx",
    "manuscript_artifact_qa": "passed_double_anonymous_three_role_ooxml_closure",
    "manuscript_render": "completed_three_role_word_pdf_and_page_png_render",
    "manuscript_visual_qa": "passed_author_verified_three_role_page_visual_qa",
    "handover_contract": "created",
    "handover": "passed",
    "release_finalize": "completed_formal_release_closure",
}

IDENTITY = {
    "candidate_manifest": "candidate_manifest_identity_sha256",
    "direct_benchmark_provider_descriptor": "descriptor_identity_sha256",
    "selection": "selection_receipt_identity_sha256",
    "proposal": "model_contract_identity_sha256",
    "authority_pin": "authority_pin_identity_sha256",
    "release_case_prelocks": "case_prelock_identity_sha256",
    "clean_install_sample_manifest": "sample_input_suite_identity_sha256",
    "qcdev_root_inputs": "summary_identity_sha256",
    "production_stageb_exact283": "summary_identity_sha256",
    "fusion_exact283": "summary_identity_sha256",
    "figure1_geometry_materialization": "figure1_geometry_materialization_identity_sha256",
    "traits_exact283": "export_identity_sha256",
    "profiles_exact283": "cohort_profile_bundle_identity_sha256",
    "profile_analysis": "analysis_identity_sha256",
    "overlay_evidence": "overlay_selection_identity_sha256",
    "evidence": "manifest_identity_sha256",
    "official_apply": "application_identity_sha256",
    "release_finalize": "release_finalization_identity_sha256",
    "clean_install_expected_identity": "reference_output_identity_sha256",
    "offline_dependencies": "dependency_materialization_identity_sha256",
    "values": "values_identity_sha256",
    "manuscript": "receipt_identity_sha256",
    "supplementary_manuscript": "receipt_identity_sha256",
    "submission_docx": "receipt_identity_sha256",
    "supplementary_docx": "receipt_identity_sha256",
    "manuscript_artifact_qa": "qa_identity_sha256",
    "manuscript_render": "render_identity_sha256",
    "manuscript_visual_qa": "visual_qa_identity_sha256",
}


def stage(
    name: str,
    command: list[str] | None,
    *,
    external: tuple[str, ...] = (),
    extra_inputs: tuple[dict[str, str], ...] = (),
    artifacts: tuple[dict[str, str], ...] = (),
    receipt_path: str | None = None,
    gpu: bool | dict = False,
    cas: dict[str, str] | None = None,
    release_registry_cas: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
) -> dict:
    receipt_path = receipt_path or P(name, "receipt.json")
    inputs = [R(dep) for dep in STAGE_DEPENDENCIES[name]]
    inputs.extend(extra_inputs)
    inputs.extend(E(item) for item in external)
    declared = [A("receipt", receipt_path), *artifacts]
    artifact_names = [str(item["name"]) for item in declared]
    artifact_paths = [str(item["path"]).replace("\\", "/").casefold() for item in declared]
    if len(artifact_names) != len(set(artifact_names)):
        raise ValueError(f"{name}: duplicate artifact name in generated contract")
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError(f"{name}: duplicate artifact path in generated contract")
    receipt = {
        "artifact": "receipt",
        "schema_version": f"{{known_stage_schema:{name}}}",
        "status_field": (
            "formal_release_status"
            if name == "proposal"
            else "release_mode"
            if name == "source_release"
            else "status"
        ),
        "status": STATUS[name],
        "required_fields": {},
    }
    if name in IDENTITY:
        receipt["identity_field"] = IDENTITY[name]
        receipt["identity_seals_complete_object"] = True
    result = {
        "name": name,
        "command": command,
        "cwd": "{workspace}",
        "environment": dict(environment or {}),
        "inputs": inputs,
        "artifacts": declared,
        "receipt": receipt,
    }
    if gpu:
        result["gpu"] = dict(GPU if gpu is True else gpu)
    if cas is not None:
        result["cas"] = cas
    if release_registry_cas is not None:
        result["release_registry_cas"] = release_registry_cas
    return result


def checkpoints() -> list[str]:
    values: list[str] = []
    for seed in range(1, 6):
        values += ["--checkpoint", f"{{external:seed_202608280{seed}_checkpoint}}"]
    return values


def checkpoint_inputs() -> tuple[str, ...]:
    return tuple(f"seed_202608280{seed}_checkpoint" for seed in range(1, 6))


def build(
    *,
    primary_physical_gpu: int = 1,
) -> dict:
    primary_gpu = gpu_contract(primary_physical_gpu)
    primary_gpu_text = str(primary_physical_gpu)
    py = "{python}"
    s = "{workspace}/scripts/phaxis"
    stages: list[dict] = []
    stages.append(stage("candidate_manifest", [py, f"{s}/build_stageb_train399_candidate_bundle.py", *checkpoints(), "--dataset-audit", "{external:dataset_audit}", "--output", P("candidate_manifest", "candidate_manifest.json")], external=("dataset_audit", *checkpoint_inputs(), *(f"seed_202608280{x}_receipt" for x in range(1, 6))), receipt_path=P("candidate_manifest", "candidate_manifest.json")))
    stages.append(stage("production_manifest", [py, f"{s}/build_production_manifest.py", "--analysis-metadata", "{external:raw_analysis_metadata}", "--review-manifest", "{external:raw_review_manifest}", "--root-input-manifest", "{external:root_raw_input_manifest}", "--output", P("production_manifest", "output")], external=("raw_analysis_metadata", "raw_review_manifest", "root_raw_input_manifest"), receipt_path=P("production_manifest", "output/summary.json"), artifacts=(A("manifest_all", P("production_manifest", "output/manifest_all.csv")),)))
    # The descriptor is itself the hash-sealed stage receipt.  Do not declare
    # the same provider.json bytes under a second artifact name: formal plan
    # assembly requires a one-to-one artifact-name/path mapping.
    stages.append(stage("direct_benchmark_provider_descriptor", [py, f"{s}/build_direct_benchmark_provider_descriptor.py", "--project-root", "{workspace}", "--physical-gpu", "0", "--output", P("direct_benchmark_provider_descriptor", "provider.json"), "--assemble"], receipt_path=P("direct_benchmark_provider_descriptor", "provider.json")))
    stages.append(stage("release_case_prelocks", [py, f"{s}/build_release_case_prelocks.py", "--application-manifest", P("production_manifest", "output/manifest_all.csv"), "--output", P("release_case_prelocks", "output")], receipt_path=P("release_case_prelocks", "output/receipt.json"), artifacts=(A("output", P("release_case_prelocks", "output"), "directory"), A("overlay_case_plan", P("release_case_prelocks", "output/overlay_case_plan.csv")), A("figure1_case_selection", P("release_case_prelocks", "output/figure1_case_selection.json")))))
    stages.append(stage("qcdev_candidate_pool", [py, f"{s}/build_stageb_train399_qcdev44_candidate_pool.py", "--manifest", "{external:qcdev_manifest}", *checkpoints(), "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--output", P("qcdev_candidate_pool", "output"), "--device", "cuda:0"], external=("qcdev_manifest", *checkpoint_inputs()), receipt_path=P("qcdev_candidate_pool", "output/summary.json"), artifacts=(A("candidate_pools", P("qcdev_candidate_pool", "output/candidate_pools"), "directory"),), gpu=primary_gpu))
    stages.append(stage("selection", [py, f"{s}/select_stageb_train399_operating_point.py", "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--candidate-pool", P("qcdev_candidate_pool", "output"), "--dataset-root", "{external:canonical443_dataset}", "--dataset-manifest", "{external:dataset_manifest}", "--split-manifest", "{external:split_manifest}", "--selection-receipt", P("selection", "selection_receipt.json"), "--selected-model-metadata", P("selection", "selected_model_metadata.json")], external=("canonical443_dataset", "dataset_manifest", "split_manifest"), receipt_path=P("selection", "selection_receipt.json"), artifacts=(A("selected_model_metadata", P("selection", "selected_model_metadata.json")),)))
    stages.append(stage("qcdev_evaluation_inference", [py, f"{s}/run_stageb_evaluation_inference.py", "--manifest", "{external:qcdev_manifest}", "--locked-val-ids", "{external:locked_val_ids}", *checkpoints(), "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--selected-model-metadata", P("selection", "selected_model_metadata.json"), "--selection-receipt", P("selection", "selection_receipt.json"), "--output", P("qcdev_evaluation_inference", "output"), "--device", "cuda:0"], external=("qcdev_manifest", "locked_val_ids", *checkpoint_inputs()), receipt_path=P("qcdev_evaluation_inference", "output/summary.json"), artifacts=(A("detections", P("qcdev_evaluation_inference", "output/detections"), "directory"),), gpu=primary_gpu))
    stages.append(stage("qcdev_evaluation", [py, f"{s}/evaluate_stageb_train399_qcdev44.py", "--detections", P("qcdev_evaluation_inference", "output/detections"), "--evaluation-inference-summary", P("qcdev_evaluation_inference", "output/summary.json"), "--hybrid-predictions", "{external:qcdev_hybrid_predictions}", "--dataset-root", "{external:canonical443_dataset}", "--dataset-manifest", "{external:dataset_manifest}", "--split-manifest", "{external:split_manifest}", "--locked-val-ids", "{external:locked_val_ids}", "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--selected-model-metadata", P("selection", "selected_model_metadata.json"), "--selection-receipt", P("selection", "selection_receipt.json"), "--output", P("qcdev_evaluation", "evaluation.json")], external=("canonical443_dataset", "dataset_manifest", "split_manifest", "locked_val_ids", "qcdev_hybrid_predictions"), receipt_path=P("qcdev_evaluation", "evaluation.json")))
    root_common = ["--project", "{workspace}", "--bundle", "{external:frozen_root_bundle}", "--acquisition-gate", "{external:root_acquisition_gate}", "--deployment-metadata", "{external:root_deployment_metadata}", "--canonical-manifest", "{external:root_canonical_manifest}", "--deployment-manifest", "{external:root_deployment_manifest}", "--deployment-lock", "{external:root_deployment_lock}", "--image-root", "{external:biological_image_root}", "--v1-physical-gpu", primary_gpu_text, "--q8-physical-gpu", primary_gpu_text, "--strict-physical-gpu", "--execute"]
    root_ext = ("frozen_root_bundle", "root_acquisition_gate", "root_deployment_metadata", "root_canonical_manifest", "root_deployment_manifest", "root_deployment_lock", "biological_image_root")
    stages.append(stage("root_provider_exact283", [py, f"{s}/run_root_provider.py", *root_common, "--input-manifest", "{external:root_raw_input_manifest}", "--reference-registry", "{external:root_reference_registry}", "--output", P("root_provider_exact283", "output")], external=(*root_ext, "root_raw_input_manifest", "root_reference_registry"), receipt_path=P("root_provider_exact283", "output/fresh_reference_audit.json"), artifacts=(A("output", P("root_provider_exact283", "output"), "directory"),), gpu=primary_gpu, environment=STRICT_GPU_ENV))
    stages.append(stage("root_bundle_materialization", [py, f"{s}/materialize_verified_root_provider_bundle.py", "--source-bundle", "{external:frozen_root_bundle}", "--output", P("root_bundle_materialization", "output")], external=("frozen_root_bundle",), receipt_path=P("root_bundle_materialization", "output/verification.json"), artifacts=(A("bundle", P("root_bundle_materialization", "output/bundle"), "directory"), A("bundle_manifest", P("root_bundle_materialization", "output/bundle/root_provider_bundle.json")))))
    stages.append(stage("proposal", [py, f"{s}/promote_model_contract.py", "--current-model-contract", "{external:pending_official_contract}", "--train399-candidate", P("candidate_manifest", "candidate_manifest.json"), "--train399-selection", P("selection", "selection_receipt.json"), "--train399-evaluation", P("qcdev_evaluation", "evaluation.json"), "--root-exact283", P("root_provider_exact283", "output/fresh_reference_audit.json"), *checkpoints(), "--output", P("proposal", "proposal.json")], external=("pending_official_contract", *checkpoint_inputs()), receipt_path=P("proposal", "proposal.json")))
    stages.append(stage("authority_pin", None, receipt_path=P("authority_pin", "authority_pin.json")))
    stages.append(stage("analysis_workflow_manifest", [py, f"{s}/build_analysis_workflow_manifest.py", "--project", "{workspace}", "--bundle", "{external:frozen_root_bundle}", "--root-input-manifest", "{external:root_raw_input_manifest}", "--acquisition-gate", "{external:root_acquisition_gate}", "--deployment-metadata", "{external:root_deployment_metadata}", "--canonical-manifest", "{external:root_canonical_manifest}", "--deployment-manifest", "{external:root_deployment_manifest}", "--deployment-lock", "{external:root_deployment_lock}", "--image-root", "{external:biological_image_root}", "--reference-registry", "{external:root_reference_registry}", "--model-contract-proposal", P("proposal", "proposal.json"), "--stageb-input-manifest", P("production_manifest", "output/manifest_all.csv"), *checkpoints(), "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--selected-model-metadata", P("selection", "selected_model_metadata.json"), "--selection-receipt", P("selection", "selection_receipt.json"), "--traits-metadata", "{external:raw_production_metadata}", "--profile-contract", "{external:static_axial_profile_contract}", "--v1-physical-gpu", "0", "--q8-physical-gpu", "0", "--stageb-physical-gpu", "0", "--stageb-internal-device", "cuda:0", "--output", P("analysis_workflow_manifest", "analysis_workflow_manifest.json")], external=(*root_ext, "root_raw_input_manifest", "root_reference_registry", "raw_production_metadata", "static_axial_profile_contract", *checkpoint_inputs()), receipt_path=P("analysis_workflow_manifest", "analysis_workflow_manifest.json")))
    stages.append(stage("clean_install_sample_manifest", [py, f"{s}/build_clean_install_sample_manifest.py", "--analysis-workflow-manifest", P("analysis_workflow_manifest", "analysis_workflow_manifest.json"), "--case-selection", P("release_case_prelocks", "output/figure1_case_selection.json"), "--output", P("clean_install_sample_manifest", "output")], receipt_path=P("clean_install_sample_manifest", "output/receipt.json"), artifacts=(A("output", P("clean_install_sample_manifest", "output"), "directory"), A("example_manifest", P("clean_install_sample_manifest", "output/release_example_manifest.json")), A("sample_input", P("clean_install_sample_manifest", "output/inputs/sample_source_image.ome.tif")))))
    stages.append(stage("qcdev_root_inputs", [py, f"{s}/build_qcdev44_root_provider_inputs.py", "--manifest", "{external:qcdev_manifest}", "--dataset-root", "{external:canonical443_dataset}", "--dataset-manifest", "{external:dataset_manifest}", "--locked-val-ids", "{external:locked_val_ids}", "--output", P("qcdev_root_inputs", "output")], external=("qcdev_manifest", "canonical443_dataset", "dataset_manifest", "locked_val_ids"), receipt_path=P("qcdev_root_inputs", "output/summary.json"), artifacts=(A("output", P("qcdev_root_inputs", "output"), "directory"), A("root_input_manifest", P("qcdev_root_inputs", "output/root_input_manifest.csv")), A("deployment_metadata", P("qcdev_root_inputs", "output/deployment_metadata.csv")), A("canonical_unit_manifest", P("qcdev_root_inputs", "output/canonical_unit_manifest.csv")), A("deployment_manifest", P("qcdev_root_inputs", "output/deployment_manifest.csv")), A("deployment_lock", P("qcdev_root_inputs", "output/deployment_manifest_lock.json")), A("acquisition_gate", P("qcdev_root_inputs", "output/acquisition_gate.json")))))
    qcdev_root_cmd = [py, f"{s}/run_root_provider.py", "--project", "{workspace}", "--bundle", "{external:frozen_root_bundle}", "--input-manifest", P("qcdev_root_inputs", "output/root_input_manifest.csv"), "--acquisition-gate", P("qcdev_root_inputs", "output/acquisition_gate.json"), "--deployment-metadata", P("qcdev_root_inputs", "output/deployment_metadata.csv"), "--canonical-manifest", P("qcdev_root_inputs", "output/canonical_unit_manifest.csv"), "--deployment-manifest", P("qcdev_root_inputs", "output/deployment_manifest.csv"), "--deployment-lock", P("qcdev_root_inputs", "output/deployment_manifest_lock.json"), "--image-root", "{external:canonical443_dataset}", "--v1-physical-gpu", primary_gpu_text, "--q8-physical-gpu", primary_gpu_text, "--strict-physical-gpu", "--output", P("qcdev_root_provider", "output"), "--execute"]
    stages.append(stage("qcdev_root_provider", qcdev_root_cmd, external=("frozen_root_bundle", "canonical443_dataset"), receipt_path=P("qcdev_root_provider", "output/pipeline_state.json"), artifacts=(A("output", P("qcdev_root_provider", "output"), "directory"),), gpu=primary_gpu, environment=STRICT_GPU_ENV))
    stages.append(stage("qcdev_fusion", [py, f"{s}/run_cli.py", "fuse", "--root-predictions", P("qcdev_root_provider", "output/hybrid/predictions"), "--root-artifacts", P("qcdev_root_provider", "output/hybrid"), "--hair-detections", P("qcdev_evaluation_inference", "output/detections"), "--model-contract", P("authority_pin", "authority_pin.json"), "--output", P("qcdev_fusion", "output")], receipt_path=P("qcdev_fusion", "output/fusion_summary.json"), artifacts=(A("output", P("qcdev_fusion", "output"), "directory"),)))
    stages.append(stage("production_stageb_exact283", [py, f"{s}/run_stageb_inference.py", "--manifest", P("production_manifest", "output/manifest_all.csv"), *checkpoints(), "--output", P("production_stageb_exact283", "output"), "--device", "cuda:0", "--model-contract-proposal", P("authority_pin", "authority_pin.json"), "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--selected-model-metadata", P("selection", "selected_model_metadata.json"), "--selection-receipt", P("selection", "selection_receipt.json")], external=checkpoint_inputs(), receipt_path=P("production_stageb_exact283", "output/summary.json"), artifacts=(A("detections", P("production_stageb_exact283", "output/detections"), "directory"),), gpu=primary_gpu))
    stages.append(stage("fusion_exact283", [py, f"{s}/run_cli.py", "fuse", "--root-predictions", P("root_provider_exact283", "output/hybrid/predictions"), "--root-artifacts", P("root_provider_exact283", "output/hybrid"), "--hair-detections", P("production_stageb_exact283", "output/detections"), "--model-contract", P("authority_pin", "authority_pin.json"), "--output", P("fusion_exact283", "output")], receipt_path=P("fusion_exact283", "output/fusion_summary.json"), artifacts=(A("output", P("fusion_exact283", "output"), "directory"),)))
    stages.append(stage("figure1_geometry_materialization", [py, f"{s}/materialize_figure1_geometry.py", "--case-selection", P("release_case_prelocks", "output/figure1_case_selection.json"), "--application-manifest", P("production_manifest", "output/manifest_all.csv"), "--fusion-root", P("fusion_exact283", "output"), "--output", P("figure1_geometry_materialization", "output")], receipt_path=P("figure1_geometry_materialization", "output/receipt.json"), artifacts=(A("output", P("figure1_geometry_materialization", "output"), "directory"), A("figure1_image", P("figure1_geometry_materialization", "output/figure1_source_image.tif")), A("figure1_geometry", P("figure1_geometry_materialization", "output/figure1_geometry.json")))))
    stages.append(stage("traits_exact283", [py, f"{s}/export_traits.py", "--predictions", P("fusion_exact283", "output/predictions"), "--metadata", "{external:raw_production_metadata}", "--model-contract-proposal", P("authority_pin", "authority_pin.json"), "--output", P("traits_exact283", "output")], external=("raw_production_metadata", "static_trait_contract"), receipt_path=P("traits_exact283", "output/summary.json"), artifacts=(A("output", P("traits_exact283", "output"), "directory"), A("traits", P("traits_exact283", "output/traits.csv")), A("hair_instances", P("traits_exact283", "output/hair_instances.csv")), A("image_traits", P("traits_exact283", "output/image_traits.csv")))))
    stages.append(stage("cohorts_exact283", [py, f"{s}/build_biological_cohorts.py", "--trait-export", P("traits_exact283", "output"), "--analysis-metadata", "{external:raw_analysis_metadata}", "--design-manifest", "{external:raw_study_design}", "--overlap-audit", "{external:immutable_overlap_audit}", "--analysis-contract", "{external:static_analysis_contract}", "--model-contract-proposal", P("proposal", "proposal.json"), "--output", P("cohorts_exact283", "output")], external=("raw_analysis_metadata", "raw_study_design", "immutable_overlap_audit", "static_analysis_contract"), receipt_path=P("cohorts_exact283", "output/summary.json"), artifacts=(A("output", P("cohorts_exact283", "output"), "directory"),)))
    stages.append(stage("biological_analysis", [py, f"{s}/analyze_biological_cohorts.py", "--cohorts", P("cohorts_exact283", "output"), "--analysis-contract", "{external:static_analysis_contract}", "--model-spec", "{external:static_biological_model_spec}", "--model-contract-proposal", P("proposal", "proposal.json"), "--output", P("biological_analysis", "output")], external=("static_analysis_contract", "static_biological_model_spec"), receipt_path=P("biological_analysis", "output/summary.json"), artifacts=(A("output", P("biological_analysis", "output"), "directory"),)))
    stages.append(stage("profiles_exact283", [py, f"{s}/export_cohort_distal_axis_profiles.py", "--cohorts-root", P("cohorts_exact283", "output"), "--contract", "{external:static_axial_profile_contract}", "--model-contract-proposal", P("proposal", "proposal.json"), "--traits-summary", P("traits_exact283", "output/summary.json"), "--output", P("profiles_exact283", "output")], external=("static_axial_profile_contract",), receipt_path=P("profiles_exact283", "output/summary.json"), artifacts=(A("output", P("profiles_exact283", "output"), "directory"), A("primary_summary", P("profiles_exact283", "output/primary_clean261/summary.json")), A("primary_profiles", P("profiles_exact283", "output/primary_clean261/distal_axis_profiles.csv")), A("sensitivity_summary", P("profiles_exact283", "output/sensitivity_full283/summary.json")), A("sensitivity_profiles", P("profiles_exact283", "output/sensitivity_full283/distal_axis_profiles.csv")))))
    stages.append(stage("profile_analysis", [py, f"{s}/analyze_distal_axis_profiles.py", "--primary-profiles", P("profiles_exact283", "output/primary_clean261"), "--sensitivity-profiles", P("profiles_exact283", "output/sensitivity_full283"), "--contract", "{external:static_profile_analysis_contract}", "--model-contract-proposal", P("proposal", "proposal.json"), "--output", P("profile_analysis", "output")], external=("static_profile_analysis_contract",), receipt_path=P("profile_analysis", "output/summary.json"), artifacts=(A("output", P("profile_analysis", "output"), "directory"),)))
    stages.append(stage("historical_oof_evidence", [py, f"{s}/build_historical_oof443_publication_evidence.py", "--oof-pickle", "{external:frozen_historical_oof_pickle}", "--dataset-manifest", "{external:dataset_manifest}", "--split-manifest", "{external:split_manifest}", "--trusted-local-oof-pickle", "--output", P("historical_oof_evidence", "output")], external=("frozen_historical_oof_pickle", "dataset_manifest", "split_manifest"), receipt_path=P("historical_oof_evidence", "output/historical_development_receipt.json"), artifacts=(A("per_image", P("historical_oof_evidence", "output/per_image_sufficient_statistics.csv")),)))
    stages.append(stage("measurement_assurance", [py, f"{s}/build_measurement_assurance_evidence.py", "--train399-evaluation", P("qcdev_evaluation", "evaluation.json"), "--qcdev-stageb-summary", P("qcdev_evaluation_inference", "output/summary.json"), "--qcdev-fusion-summary", P("qcdev_fusion", "output/fusion_summary.json"), "--qcdev-fusion-root", P("qcdev_fusion", "output"), "--application-fusion-summary", P("fusion_exact283", "output/fusion_summary.json"), "--application-fusion-root", P("fusion_exact283", "output"), "--dataset-root", "{external:canonical443_dataset}", "--dataset-manifest", "{external:dataset_manifest}", "--split-manifest", "{external:split_manifest}", "--clean-traits", P("cohorts_exact283", "output/primary_clean261/traits.csv"), "--cohorts-receipt", P("cohorts_exact283", "output/summary.json"), "--root-exact283-receipt", P("root_provider_exact283", "output/fresh_reference_audit.json"), "--output", P("measurement_assurance", "output")], external=("canonical443_dataset", "dataset_manifest", "split_manifest", "static_trait_contract"), receipt_path=P("measurement_assurance", "output/measurement_assurance_receipt.json"), artifacts=(A("output", P("measurement_assurance", "output"), "directory"),)))
    stages.append(stage("overlay_evidence", [py, f"{s}/build_condition_blinded_overlay_evidence.py", "--case-plan", P("release_case_prelocks", "output/overlay_case_plan.csv"), "--application-manifest", P("production_manifest", "output/manifest_all.csv"), "--full-traits", P("traits_exact283", "output/traits.csv"), "--fusion-root", P("fusion_exact283", "output"), "--expected-task-count", "283", "--output", P("overlay_evidence", "output")], receipt_path=P("overlay_evidence", "output/overlay_selection_receipt.json"), artifacts=(A("selection", P("overlay_evidence", "output/overlay_selection.csv")), A("full283_review_overlays", P("overlay_evidence", "output/full283_review_overlays"), "directory"), A("full283_review_index", P("overlay_evidence", "output/full283_review_index.csv")), A("full283_review_checklist", P("overlay_evidence", "output/full283_review_checklist.csv")), A("full283_review_summary", P("overlay_evidence", "output/full283_review_summary.json")), A("full283_review_readme", P("overlay_evidence", "output/README_CN.md")))))
    for name, mode in (("benchmark_phaxis_production", "phaxis_production"), ("benchmark_frozen_v1_production", "frozen_v1_production"), ("benchmark_phaxis_sequential", "phaxis_sequential"), ("benchmark_frozen_v1_sequential", "frozen_v1_sequential")):
        cmd = [py, f"{s}/run_external_direct_benchmark.py", "--project-root", "{workspace}", "--producer-interface", P("direct_benchmark_provider_descriptor", "provider.json"), "--mode", mode, "--source-manifest", P("production_manifest", "output/manifest_all.csv"), "--image-root", "{external:biological_image_root}"]
        cmd += ["--analysis-manifest", P("analysis_workflow_manifest", "analysis_workflow_manifest.json")]
        cmd += ["--workflow-output", P(name, "workflow"), "--output", P(name, "benchmark"), "--cuda-visible-devices", "0", "--execute"]
        ext = ("biological_image_root",) + (("frozen_v1_read_only_assets",) if mode.startswith("frozen") else ())
        extra = [A("workflow", P(name, "workflow"), "directory"), A("benchmark", P(name, "benchmark"), "directory"), A("gpu_telemetry", P(name, "benchmark/gpu_telemetry.json")), A("hardware_preflight", P(name, "benchmark/hardware_preflight.json"))]
        if mode.endswith("sequential"):
            extra.append(A("runtime_per_image", P(name, "benchmark/runtime_per_image.csv")))
        stages.append(stage(name, cmd, external=ext, receipt_path=P(name, "benchmark/runtime_summary.json"), artifacts=tuple(extra), gpu=GPU0, environment=STRICT_GPU_ENV))
    stages.append(stage("benchmark_production_comparison", [py, f"{s}/benchmark_full_workflow.py", "--compare-benchmarks", "--phaxis-summary", P("benchmark_phaxis_production", "benchmark/runtime_summary.json"), "--baseline-receipt", P("benchmark_frozen_v1_production", "benchmark/runtime_summary.json"), "--output", P("benchmark_production_comparison", "comparison.json")], receipt_path=P("benchmark_production_comparison", "comparison.json")))
    stages.append(stage("benchmark_sequential_comparison", [py, f"{s}/benchmark_full_workflow.py", "--compare-benchmarks", "--phaxis-summary", P("benchmark_phaxis_sequential", "benchmark/runtime_summary.json"), "--baseline-receipt", P("benchmark_frozen_v1_sequential", "benchmark/runtime_summary.json"), "--output", P("benchmark_sequential_comparison", "comparison.json")], receipt_path=P("benchmark_sequential_comparison", "comparison.json")))
    stages.append(stage("benchmark_same_hardware", [py, f"{s}/benchmark_full_workflow.py", "--aggregate-same-hardware", "--phaxis-production-summary", P("benchmark_phaxis_production", "benchmark/runtime_summary.json"), "--phaxis-sequential-summary", P("benchmark_phaxis_sequential", "benchmark/runtime_summary.json"), "--frozen-v1-production-summary", P("benchmark_frozen_v1_production", "benchmark/runtime_summary.json"), "--frozen-v1-sequential-summary", P("benchmark_frozen_v1_sequential", "benchmark/runtime_summary.json"), "--production-comparison", P("benchmark_production_comparison", "comparison.json"), "--sequential-comparison", P("benchmark_sequential_comparison", "comparison.json"), "--publish-receipt", "--output", P("benchmark_same_hardware", "receipt.json")]))
    inv_cmd = [py, f"{s}/build_benchmark_artifact_inventory.py", "--project-root", "{workspace}"]
    for role, owner in (("phaxis_production_summary", "benchmark_phaxis_production"), ("v1_production_summary", "benchmark_frozen_v1_production"), ("phaxis_sequential_summary", "benchmark_phaxis_sequential"), ("v1_sequential_summary", "benchmark_frozen_v1_sequential"), ("production_comparison_receipt", "benchmark_production_comparison"), ("sequential_comparison_receipt", "benchmark_sequential_comparison"), ("same_hardware_receipt", "benchmark_same_hardware")):
        filename = "comparison.json" if "comparison" in role else "receipt.json" if role == "same_hardware_receipt" else "benchmark/runtime_summary.json"
        inv_cmd += ["--artifact", f"{role}=model/benchmark/{role}.json={P(owner, filename)}"]
    for owner, package_stem in (("benchmark_phaxis_sequential", "phaxis_sequential"), ("benchmark_frozen_v1_sequential", "v1_sequential")):
        inv_cmd += ["--artifact", f"per_image_latency_csv=model/benchmark/{package_stem}_runtime_per_image.csv={P(owner, 'benchmark/runtime_per_image.csv')}"]
    for owner, package_stem in (("benchmark_phaxis_production", "phaxis_production"), ("benchmark_frozen_v1_production", "v1_production"), ("benchmark_phaxis_sequential", "phaxis_sequential"), ("benchmark_frozen_v1_sequential", "v1_sequential")):
        inv_cmd += ["--artifact", f"gpu_telemetry=model/benchmark/{package_stem}_gpu_telemetry.json={P(owner, 'benchmark/gpu_telemetry.json')}"]
        inv_cmd += ["--artifact", f"hardware_preflight=model/benchmark/{package_stem}_hardware_preflight.json={P(owner, 'benchmark/hardware_preflight.json')}"]
    inv_cmd += ["--output", P("benchmark_artifact_inventory", "inventory.csv"), "--receipt", P("benchmark_artifact_inventory", "receipt.json")]
    stages.append(stage("benchmark_artifact_inventory", inv_cmd, artifacts=(A("inventory", P("benchmark_artifact_inventory", "inventory.csv")),)))
    # Publication assemblers consume only explicit named receipts/resources.
    fig_cmd = [py, f"{s}/build_publication_figure_inputs.py", "--mode", "final", "--output", P("figure_inputs", "output"), "--model-contract-proposal", P("proposal", "proposal.json"), "--train399-selection", P("selection", "selection_receipt.json"), "--split-manifest", "{external:split_manifest}", "--trait-contract", "{external:static_trait_contract}", "--figure1-image", P("figure1_geometry_materialization", "output/figure1_source_image.tif"), "--figure1-geometry", P("figure1_geometry_materialization", "output/figure1_geometry.json"), "--historical-development-receipt", P("historical_oof_evidence", "output/historical_development_receipt.json"), "--historical-oof-per-image", P("historical_oof_evidence", "output/per_image_sufficient_statistics.csv"), "--measurement-assurance-receipt", P("measurement_assurance", "output/measurement_assurance_receipt.json"), "--assurance-metrics", P("measurement_assurance", "output/assurance_metrics.csv"), "--assurance-pairs", P("measurement_assurance", "output/assurance_pairs.csv"), "--assurance-support", P("measurement_assurance", "output/assurance_support.csv"), "--assurance-topology", P("measurement_assurance", "output/assurance_topology.csv"), "--overlay-index-receipt", P("overlay_evidence", "output/overlay_selection_receipt.json"), "--overlay-selection", P("overlay_evidence", "output/overlay_selection.csv"), "--clean-traits", P("cohorts_exact283", "output/primary_clean261/traits.csv"), "--full-traits", P("traits_exact283", "output/traits.csv"), "--full-image-traits", P("traits_exact283", "output/image_traits.csv"), "--analysis-primary-table", P("biological_analysis", "output/tables/primary_clean_exploratory_factorial_tests.csv"), "--analysis-sensitivity-table", P("biological_analysis", "output/tables/full283_sensitivity_factorial_tests.csv"), "--profile-analysis-summary", P("profile_analysis", "output/summary.json"), "--profile-analysis-table", P("profile_analysis", "output/distal_axis_profile_group_summaries.csv"), "--sensitivity-profiles-summary", P("profiles_exact283", "output/sensitivity_full283/summary.json"), "--runtime-latency-summary", P("benchmark_phaxis_sequential", "benchmark/runtime_summary.json"), "--runtime-per-image", P("benchmark_phaxis_sequential", "benchmark/runtime_per_image.csv"), "--runtime-production-summary", P("benchmark_phaxis_production", "benchmark/runtime_summary.json"), "--runtime-latency-comparison", P("benchmark_sequential_comparison", "comparison.json"), "--runtime-production-comparison", P("benchmark_production_comparison", "comparison.json"), "--baseline-runtime-latency-summary", P("benchmark_frozen_v1_sequential", "benchmark/runtime_summary.json"), "--baseline-runtime-per-image", P("benchmark_frozen_v1_sequential", "benchmark/runtime_per_image.csv"), "--baseline-runtime-production-summary", P("benchmark_frozen_v1_production", "benchmark/runtime_summary.json")]
    fig_cmd += ["--train399-candidate", P("candidate_manifest", "candidate_manifest.json"), "--dataset-manifest", "{external:dataset_manifest}", "--image-traits-schema", "{workspace}/configs/phaxis/v1_0/image_traits.schema.json", "--benchmark-same-hardware", P("benchmark_same_hardware", "receipt.json"), "--benchmark-artifact-inventory", P("benchmark_artifact_inventory", "inventory.csv")]
    for seed in range(2026082801, 2026082806):
        fig_cmd += ["--training-receipt", f"{seed}={{external:seed_{seed}_receipt}}"]
    core_map = {"train399_evaluation": P("qcdev_evaluation", "evaluation.json"), "root_exact283": P("root_provider_exact283", "output/fresh_reference_audit.json"), "stageb": P("production_stageb_exact283", "output/summary.json"), "fusion": P("fusion_exact283", "output/fusion_summary.json"), "traits": P("traits_exact283", "output/summary.json"), "cohorts": P("cohorts_exact283", "output/summary.json"), "analysis": P("biological_analysis", "output/summary.json"), "profiles": P("profiles_exact283", "output/primary_clean261/summary.json")}
    for role, path in core_map.items(): fig_cmd += [f"--{role.replace('_','-')}", path]
    stages.append(stage("figure_inputs", fig_cmd, external=("dataset_manifest", "split_manifest", "static_trait_contract", *(f"seed_{seed}_receipt" for seed in range(2026082801, 2026082806))), receipt_path=P("figure_inputs", "output/assembly_summary.json"), artifacts=(A("output", P("figure_inputs", "output"), "directory"), A("manifest", P("figure_inputs", "output/figure_inputs.json")))))
    figures_cmd = [py, f"{s}/build_publication_figures.py", "--mode", "final", "--figure-inputs", P("figure_inputs", "output/figure_inputs.json"), "--model-contract-proposal", P("proposal", "proposal.json"), "--output", P("figures", "output")]
    for role, path in core_map.items(): figures_cmd += [f"--{role.replace('_','-')}", path]
    stages.append(stage("figures", figures_cmd, receipt_path=P("figures", "output/figure_assembly_summary.json"), artifacts=(A("output", P("figures", "output"), "directory"),)))
    evidence_cmd = [py, f"{s}/build_manuscript_evidence_manifest.py", "--model-contract-proposal", P("proposal", "proposal.json"), "--train399-candidate", P("candidate_manifest", "candidate_manifest.json"), "--train399-selection", P("selection", "selection_receipt.json"), "--train399-evaluation", P("qcdev_evaluation", "evaluation.json"), "--root-exact283", P("root_provider_exact283", "output/fresh_reference_audit.json"), "--figure-inputs", P("figure_inputs", "output/figure_inputs.json")]
    for role, path in {"stageb": core_map["stageb"], "fusion": core_map["fusion"], "traits": core_map["traits"], "cohorts": core_map["cohorts"], "analysis": core_map["analysis"], "profiles": core_map["profiles"], "figures": P("figures", "output/figure_assembly_summary.json")}.items(): evidence_cmd += [f"--{role}-summary", path]
    evidence_cmd += ["--output", P("evidence", "evidence.json")]
    stages.append(stage("evidence", evidence_cmd, receipt_path=P("evidence", "evidence.json")))
    apply_cmd = [py, f"{s}/promote_model_contract.py", "--current-model-contract", "{external:pending_official_contract}", "--proposal", P("proposal", "proposal.json"), "--expected-current-sha256", "{external_sha256:pending_official_contract}", *checkpoints(), "--stageb-summary", core_map["stageb"], "--fusion-summary", core_map["fusion"], "--traits-summary", core_map["traits"], "--manuscript-evidence-manifest", P("evidence", "evidence.json"), "--output", P("official_apply", "application_receipt.json"), "--apply"]
    stages.append(stage("official_apply", apply_cmd, external=("pending_official_contract", *checkpoint_inputs()), receipt_path=P("official_apply", "application_receipt.json"), cas={"path": "{external:pending_official_contract}", "expected_sha256": "{external_sha256:pending_official_contract}"}))
    stages.append(stage("source_release", [py, f"{s}/build_source_release.py", "--project-root", "{workspace}", "--root-provider-exact283-receipt", core_map["root_exact283"], "--train399-candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), "--train399-selection-receipt", P("selection", "selection_receipt.json"), "--train399-evaluation-receipt", P("qcdev_evaluation", "evaluation.json"), "--final-fusion-summary", core_map["fusion"], "--final-traits-summary", core_map["traits"], "--release-human-metadata", "{external:release_author_metadata}", "--output", P("source_release", "output")], external=("release_author_metadata",), receipt_path=P("source_release", "output/SOURCE_MANIFEST.json"), artifacts=(A("output", P("source_release", "output"), "directory"),)))
    stages.append(stage("distributions", [py, f"{s}/build_release_distributions.py", "--source-release-root", P("source_release", "output"), "--source-release-manifest", P("source_release", "output/SOURCE_MANIFEST.json"), "--output", P("distributions", "dist"), "--python", py], receipt_path=P("distributions", "dist/distribution_receipt.json"), artifacts=(A("output", P("distributions", "dist"), "directory"), A("wheel", P("distributions", "dist/phaxis-1.0.0-py3-none-any.whl")))))
    stages.append(stage("offline_dependencies", [py, f"{s}/materialize_offline_dependencies.py", "--formal-wheel", P("distributions", "dist/phaxis-1.0.0-py3-none-any.whl"), "--python", py, "--output", P("offline_dependencies", "output"), "--execute"], receipt_path=P("offline_dependencies", "output/receipt.json"), artifacts=(A("output", P("offline_dependencies", "output"), "directory"), A("dependency_lock", P("offline_dependencies", "output/requirements.lock.txt")), A("wheelhouse", P("offline_dependencies", "output/wheelhouse"), "directory"), A("resolved_sbom", P("offline_dependencies", "output/SBOM.resolved.cdx.json")), A("resolved_license_inventory", P("offline_dependencies", "output/THIRD_PARTY_LICENSES.resolved.json")))))
    # Reuse-package manifests are independently materialised with author attestation.
    common_att = ("author_release_attestation",)
    stages.append(stage("handover_dataset_manifest", [py, f"{s}/build_handover_dataset_manifest.py", "--project-root", "{workspace}", "--manual-image-manifest", "{external:manual500_image_manifest}", "--all500-decisions", "{external:all500_decisions}", "--canonical-dataset-root", "{external:canonical443_dataset}", "--canonical-dataset-manifest", "{external:dataset_manifest}", "--canonical-integrity-manifest", "{external:canonical_integrity_manifest}", "--all500-notes", "{external:all500_notes}", "--release-attestation", "{external:author_release_attestation}", "--output", P("handover_dataset_manifest", "manifest.csv"), "--receipt", P("handover_dataset_manifest", "receipt.json"), "--execute"], external=("manual500_image_manifest", "all500_decisions", "canonical443_dataset", "dataset_manifest", "canonical_integrity_manifest", "all500_notes", *common_att), artifacts=(A("manifest", P("handover_dataset_manifest", "manifest.csv")),)))
    stages.append(stage("handover_image_manifest", [py, f"{s}/build_handover_image_manifest.py", "--project-root", "{workspace}", "--deployment-manifest", "{external:root_deployment_manifest}", "--deployment-lock", "{external:root_deployment_lock}", "--image-root", "{external:biological_image_root}", "--release-attestation", "{external:author_release_attestation}", "--output", P("handover_image_manifest", "manifest.csv"), "--receipt", P("handover_image_manifest", "receipt.json"), "--execute"], external=("root_deployment_manifest", "root_deployment_lock", "biological_image_root", *common_att), artifacts=(A("manifest", P("handover_image_manifest", "manifest.csv")),)))
    stages.append(stage("handover_model_source_manifest", [py, f"{s}/build_handover_model_source_manifest.py", "--project-root", "{workspace}", "--source-release-root", P("source_release", "output"), "--source-release-manifest", P("source_release", "output/SOURCE_MANIFEST.json"), "--release-attestation", "{external:author_release_attestation}", "--output", P("handover_model_source_manifest", "manifest.csv"), "--receipt", P("handover_model_source_manifest", "receipt.json"), "--execute"], external=common_att, artifacts=(A("manifest", P("handover_model_source_manifest", "manifest.csv")),)))
    stages.append(stage("handover_model_asset_manifest", [py, f"{s}/build_handover_model_asset_manifest.py", "--project-root", "{workspace}", "--applied-model-contract", "{external:pending_official_contract}", "--candidate-manifest", P("candidate_manifest", "candidate_manifest.json"), *checkpoints(), "--root-provider-bundle-root", P("root_bundle_materialization", "output/bundle"), "--root-provider-bundle-manifest", P("root_bundle_materialization", "output/bundle/root_provider_bundle.json"), "--root-provider-verification-receipt", P("root_bundle_materialization", "output/verification.json"), "--release-example-root", P("clean_install_sample_manifest", "output"), "--bundle-manifest-output", P("handover_model_asset_manifest", "MODEL_BUNDLE_MANIFEST.json"), "--portable-capsule-output", P("handover_model_asset_manifest", "portable_capsule"), "--release-attestation", "{external:author_release_attestation}", "--output", P("handover_model_asset_manifest", "manifest.csv"), "--receipt", P("handover_model_asset_manifest", "receipt.json"), "--execute"], external=("pending_official_contract", *checkpoint_inputs(), *common_att), artifacts=(A("manifest", P("handover_model_asset_manifest", "manifest.csv")), A("bundle_manifest", P("handover_model_asset_manifest", "MODEL_BUNDLE_MANIFEST.json")), A("portable_capsule", P("handover_model_asset_manifest", "portable_capsule"), "directory"))))
    stages.append(stage("clean_install_expected_identity", [py, f"{s}/build_clean_install_expected_identity.py", "--example-manifest", P("handover_model_asset_manifest", "portable_capsule/model/examples/clean_install/release_example_manifest.json"), "--portable-capsule-root", P("handover_model_asset_manifest", "portable_capsule"), "--model-contract-proposal", P("proposal", "proposal.json"), "--applied-model-contract", "{external:pending_official_contract}", "--model-bundle-manifest", P("handover_model_asset_manifest", "MODEL_BUNDLE_MANIFEST.json"), "--source-release-root", P("source_release", "output"), "--formal-wheel", P("distributions", "dist/phaxis-1.0.0-py3-none-any.whl"), "--python", py, "--physical-gpu", "0", "--cuda-visible-devices", "0", "--output", P("clean_install_expected_identity", "output"), "--execute"], external=("pending_official_contract",), receipt_path=P("clean_install_expected_identity", "output/receipt.json"), artifacts=(A("output", P("clean_install_expected_identity", "output"), "directory"), A("expected_identity", P("clean_install_expected_identity", "output/expected_identity.json")), A("reference_analysis", P("clean_install_expected_identity", "output/reference_analysis"), "directory")), gpu=GPU0, environment=STRICT_GPU_ENV))
    stages.append(stage("handover_benchmark_manifest", [py, f"{s}/build_handover_benchmark_manifest.py", "--project-root", "{workspace}", "--same-hardware-receipt", P("benchmark_same_hardware", "receipt.json"), "--artifact-inventory", P("benchmark_artifact_inventory", "inventory.csv"), "--release-attestation", "{external:author_release_attestation}", "--output", P("handover_benchmark_manifest", "manifest.csv"), "--receipt", P("handover_benchmark_manifest", "receipt.json"), "--execute"], external=common_att, artifacts=(A("manifest", P("handover_benchmark_manifest", "manifest.csv")),)))
    stages.append(stage("clean_install", [py, f"{s}/build_clean_install_verification.py", "--project-root", "{workspace}", "--wheel", P("distributions", "dist/phaxis-1.0.0-py3-none-any.whl"), "--source-release-root", P("source_release", "output"), "--applied-model-contract", "{external:pending_official_contract}", "--model-contract-proposal", P("proposal", "proposal.json"), "--model-bundle-manifest", P("handover_model_asset_manifest", "MODEL_BUNDLE_MANIFEST.json"), "--portable-capsule-root", P("handover_model_asset_manifest", "portable_capsule"), "--example-manifest", P("handover_model_asset_manifest", "portable_capsule/model/examples/clean_install/release_example_manifest.json"), "--expected-example-identity", P("clean_install_expected_identity", "output/expected_identity.json"), "--dependency-lock", P("offline_dependencies", "output/requirements.lock.txt"), "--wheelhouse", P("offline_dependencies", "output/wheelhouse"), "--base-python", py, "--work-root", P("clean_install", "work"), "--output", P("clean_install", "receipt.json"), "--cuda-visible-devices", "0", "--execute"], external=("pending_official_contract",), gpu=GPU0, environment=STRICT_GPU_ENV))
    values_cmd = [py, f"{s}/build_manuscript_values.py", "--master", "{external:master_manuscript}", "--evidence-graph", P("evidence", "evidence.json"), "--figure-inputs", P("figure_inputs", "output/figure_inputs.json"), "--figure-assembly-summary", P("figure_inputs", "output/assembly_summary.json"), "--model-contract-proposal", P("proposal", "proposal.json"), "--human-metadata", "{external:author_verified_manuscript_metadata}", "--model-bundle-manifest", P("handover_model_asset_manifest", "MODEL_BUNDLE_MANIFEST.json"), "--clean-install-receipt", P("clean_install", "receipt.json"), "--source-release-manifest", P("source_release", "output/SOURCE_MANIFEST.json")]
    values_evidence_artifacts = {
        "model_contract_proposal": P("proposal", "proposal.json"),
        "train399_candidate": P("candidate_manifest", "candidate_manifest.json"),
        "train399_selection": P("selection", "selection_receipt.json"),
        "train399_evaluation": P("qcdev_evaluation", "evaluation.json"),
        "root_exact283": P("root_provider_exact283", "output/fresh_reference_audit.json"),
        "stageb": core_map["stageb"],
        "fusion": core_map["fusion"],
        "traits": core_map["traits"],
        "cohorts": core_map["cohorts"],
        "analysis": core_map["analysis"],
        "profiles": core_map["profiles"],
        "figure_inputs": P("figure_inputs", "output/figure_inputs.json"),
        "figures": P("figures", "output/figure_assembly_summary.json"),
    }
    for role, path in values_evidence_artifacts.items():
        values_cmd += ["--evidence-artifact", f"{role}={path}"]
    values_cmd += ["--output", P("values", "values.json"), "--missing-human-report", P("values", "missing_human.json")]
    # missing_human.json is a resumable diagnostic work item emitted on a
    # blocked first pass, not a successful-stage artifact.  Declaring it as an
    # output would make the preexisting-output guard deadlock the resumed run.
    stages.append(stage("values", values_cmd, external=("master_manuscript", "author_verified_manuscript_metadata"), receipt_path=P("values", "values.json")))
    stages.append(stage("manuscript", [py, f"{s}/compile_manuscript.py", "--master", "{external:master_manuscript}", "--evidence-graph", P("evidence", "evidence.json"), "--values", P("values", "values.json"), "--output", P("manuscript", "manuscript.md"), "--receipt", P("manuscript", "receipt.json")], external=("master_manuscript",), artifacts=(A("document", P("manuscript", "manuscript.md")),)))
    stages.append(stage("supplementary_manuscript", [py, f"{s}/compile_supplementary_manuscript.py", "--master", "{external:supplement_master}", "--main-manuscript", P("manuscript", "manuscript.md"), "--main-compile-receipt", P("manuscript", "receipt.json"), "--figure-summary", P("figures", "output/figure_assembly_summary.json"), "--output", P("supplementary_manuscript", "supplementary.md"), "--receipt", P("supplementary_manuscript", "receipt.json")], external=("supplement_master",), artifacts=(A("document", P("supplementary_manuscript", "supplementary.md")),)))
    stages.append(stage("submission_docx", [py, f"{s}/build_submission_docx.py", "--mode", "final", "--manuscript", P("manuscript", "manuscript.md"), "--submission-metadata", "{external:submission_title_metadata}", "--compile-receipt", P("manuscript", "receipt.json"), "--figure-summary", P("figures", "output/figure_assembly_summary.json"), "--title-page-output", P("submission_docx", "PHAxis_Plant_Phenomics_title_page.docx"), "--anonymized-main-output", P("submission_docx", "PHAxis_Plant_Phenomics_anonymized_main.docx"), "--receipt", P("submission_docx", "receipt.json")], external=("submission_title_metadata",), artifacts=(A("title_page", P("submission_docx", "PHAxis_Plant_Phenomics_title_page.docx")), A("anonymized_main", P("submission_docx", "PHAxis_Plant_Phenomics_anonymized_main.docx")))))
    stages.append(stage("supplementary_docx", [py, f"{s}/build_supplementary_docx.py", "--mode", "final", "--supplement", P("supplementary_manuscript", "supplementary.md"), "--main-manuscript", P("manuscript", "manuscript.md"), "--main-compile-receipt", P("manuscript", "receipt.json"), "--figure-summary", P("figures", "output/figure_assembly_summary.json"), "--output", P("supplementary_docx", "PHAxis_Plant_Phenomics_anonymized_supplement.docx"), "--receipt", P("supplementary_docx", "receipt.json")], artifacts=(A("anonymized_supplement", P("supplementary_docx", "PHAxis_Plant_Phenomics_anonymized_supplement.docx")),)))
    manuscript_qa_cmd = [py, f"{s}/verify_manuscript_artifacts.py", "--main-master", "{external:master_manuscript}", "--supplement-master", "{external:supplement_master}", "--main-manuscript", P("manuscript", "manuscript.md"), "--main-compile-receipt", P("manuscript", "receipt.json"), "--supplement-manuscript", P("supplementary_manuscript", "supplementary.md"), "--supplement-compile-receipt", P("supplementary_manuscript", "receipt.json"), "--submission-metadata", "{external:submission_title_metadata}", "--figure-summary", P("figures", "output/figure_assembly_summary.json"), "--title-page-docx", P("submission_docx", "PHAxis_Plant_Phenomics_title_page.docx"), "--anonymized-main-docx", P("submission_docx", "PHAxis_Plant_Phenomics_anonymized_main.docx"), "--submission-docx-receipt", P("submission_docx", "receipt.json"), "--anonymized-supplement-docx", P("supplementary_docx", "PHAxis_Plant_Phenomics_anonymized_supplement.docx"), "--supplement-docx-receipt", P("supplementary_docx", "receipt.json"), "--output", P("manuscript_artifact_qa", "receipt.json"), "--upload-manifest", P("manuscript_artifact_qa", "upload-role-manifest.json")]
    stages.append(stage("manuscript_artifact_qa", manuscript_qa_cmd, external=("master_manuscript", "supplement_master", "submission_title_metadata"), artifacts=(A("upload_manifest", P("manuscript_artifact_qa", "upload-role-manifest.json")),)))
    stages.append(stage("manuscript_render", [py, f"{s}/render_manuscript_bundle.py", "--title-page-docx", P("submission_docx", "PHAxis_Plant_Phenomics_title_page.docx"), "--anonymized-main-docx", P("submission_docx", "PHAxis_Plant_Phenomics_anonymized_main.docx"), "--anonymized-supplement-docx", P("supplementary_docx", "PHAxis_Plant_Phenomics_anonymized_supplement.docx"), "--structural-qa", P("manuscript_artifact_qa", "receipt.json"), "--upload-manifest", P("manuscript_artifact_qa", "upload-role-manifest.json"), "--powershell", "powershell.exe", "--pdftoppm", "pdftoppm", "--output", P("manuscript_render", "output")], receipt_path=P("manuscript_render", "output/receipt.json"), artifacts=(A("output", P("manuscript_render", "output"), "directory"), A("title_page_pdf", P("manuscript_render", "output/title_page/title_page.pdf")), A("anonymized_main_pdf", P("manuscript_render", "output/anonymized_main/anonymized_main.pdf")), A("anonymized_supplement_pdf", P("manuscript_render", "output/anonymized_supplement/anonymized_supplement.pdf")), A("visual_qa_template", P("manuscript_render", "output/VISUAL_QA_ATTESTATION_TEMPLATE.json")))))
    # The first invocation deliberately materialises the human work item at
    # VISUAL_QA_ATTESTATION.json and exits nonzero.  It is therefore an input
    # to a resumed invocation, not a successful-stage artifact: declaring it
    # here would make the orchestrator's no-preexisting-output guard prevent
    # the required review -> seal -> resume lifecycle.
    stages.append(stage("manuscript_visual_qa", [py, f"{s}/validate_manuscript_visual_qa.py", "--render-receipt", P("manuscript_render", "output/receipt.json"), "--structural-qa", P("manuscript_artifact_qa", "receipt.json"), "--template", P("manuscript_render", "output/VISUAL_QA_ATTESTATION_TEMPLATE.json"), "--attestation", P("manuscript_visual_qa", "VISUAL_QA_ATTESTATION.json"), "--output", P("manuscript_visual_qa", "receipt.json")]))
    # The contract assembler receives every named sub-manifest explicitly.
    handover_bindings = {"applied_model_contract": "{external:pending_official_contract}", "train399_candidate_manifest": P("candidate_manifest", "candidate_manifest.json"), "train399_selection_receipt": P("selection", "selection_receipt.json"), "train399_evaluation_receipt": P("qcdev_evaluation", "evaluation.json"), "fresh_exact283_receipt": P("root_provider_exact283", "output/fresh_reference_audit.json"), "final_fusion_receipt": core_map["fusion"], "final_traits_receipt": core_map["traits"], "same_hardware_benchmark_receipt": P("benchmark_same_hardware", "receipt.json"), "source_release_manifest": P("source_release", "output/SOURCE_MANIFEST.json"), "clean_install_receipt": P("clean_install", "receipt.json"), "dataset_manifest": P("handover_dataset_manifest", "manifest.csv"), "image_manifest": P("handover_image_manifest", "manifest.csv"), "model_source_manifest": P("handover_model_source_manifest", "manifest.csv"), "model_asset_manifest": P("handover_model_asset_manifest", "manifest.csv"), "benchmark_manifest": P("handover_benchmark_manifest", "manifest.csv"), "trait_contract": "{external:static_trait_contract}"}
    hc = [py, f"{s}/assemble_handover_build_contract.py", "--project-root", "{workspace}"]
    for role, path in handover_bindings.items(): hc += [f"--{role.replace('_','-')}", path]
    hc += [*checkpoints(), "--release-attestation", "{external:author_release_attestation}", "--output", P("handover_contract", "contract.json"), "--receipt", P("handover_contract", "receipt.json"), "--execute"]
    stages.append(stage("handover_contract", hc, external=("pending_official_contract", "static_trait_contract", "author_release_attestation", *checkpoint_inputs()), artifacts=(A("contract", P("handover_contract", "contract.json")),)))
    stages.append(stage("handover", [py, f"{s}/build_handover_package.py", "--project-root", "{workspace}", "--contract", P("handover_contract", "contract.json"), "--output", P("handover", "output")], receipt_path=P("handover", "output/BUILD_RECEIPT.json"), artifacts=(A("output", P("handover", "output"), "directory"),)))
    stages.append(
        stage(
            "release_finalize",
            None,
            external=("release_authority_registry",),
            receipt_path=P("release_finalize", "release_finalization.json"),
            release_registry_cas={
                "external": "release_authority_registry",
                "path": "{external:release_authority_registry}",
                "expected_sha256": "{external_sha256:release_authority_registry}",
            },
        )
    )
    if tuple(item["name"] for item in stages) != MANDATORY_STAGE_ORDER:
        raise RuntimeError("generated stage order drifted from release_topology")
    return {
        "schema_version": "PHAxis-post-training-release-stage-contract-template-1.0",
        "status": "explicit_real_producer_argv_contract",
        "synthetic_commands_present": False,
        "stage_count": len(stages),
        "stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically update the deterministic checked-in stage contract",
    )
    parser.add_argument(
        "--primary-physical-gpu",
        type=int,
        default=1,
        help="physical GPU for QC-development and production inference (default: 1)",
    )
    args = parser.parse_args(argv)
    payload = build(
        primary_physical_gpu=args.primary_physical_gpu,
    )
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        atomic_write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "status": "stage_contract_written",
                    "output": str(args.output.resolve()),
                    "stage_count": payload["stage_count"],
                    "primary_physical_gpu": args.primary_physical_gpu,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
