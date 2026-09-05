"""Authoritative producer topology for the PHAxis formal release chain.

This module records *producer* dependencies, rather than an attractive ordering
of publication artefacts.  It is deliberately CPU-only and imports neither
torch nor CUDA.  The topology is also independent from release receipts: a
structurally valid graph is not evidence that any scientific result exists or
that a formal release Gate passed.

The important authority boundary is two-phase:

* ``official_apply`` atomically establishes the model authority after the
  final model/evidence receipts exist;
* post-apply producers build and verify the source distribution, clean install,
  manuscript and reuse package against that applied authority;
* ``release_finalize`` is the last internal stage and seals the completed
  release, so an applied model contract can never be mistaken for a completed
  software/publication release.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import sha256_file, sha256_json


TOPOLOGY_AUDIT_SCHEMA = "PHAxis-formal-release-producer-topology-audit-1.2"


class ReleaseTopologyError(RuntimeError):
    """The formal producer graph or a producer CLI contract is inconsistent."""


@dataclass(frozen=True)
class ProducerContract:
    """One coarse release stage and the real entry point that owns its receipt.

    ``dependencies`` are stage-to-stage receipt dependencies.  Immutable raw
    data, frozen read-only assets, preregistered scientific contracts and human
    attestations are named in ``external_authorities``; any result-dependent or
    otherwise reproducible release artefact must instead have a real producer.
    ``required_cli_options`` is a deliberately small set of dependency-bearing
    flags whose presence is checked against the real producer source.
    """

    name: str
    producer: str
    execution_class: str
    dependencies: tuple[str, ...] = ()
    external_authorities: tuple[str, ...] = ()
    required_cli_options: tuple[str, ...] = ()
    capability_gap: str | None = None


# This order is a deterministic topological ordering, not a statement that
# independent CPU nodes cannot be run concurrently.  The release orchestrator
# uses it for an auditable, resumable serial commit protocol.
FORMAL_RELEASE_PRODUCERS: tuple[ProducerContract, ...] = (
    ProducerContract(
        "candidate_manifest",
        "scripts/phaxis/build_stageb_train399_candidate_bundle.py",
        "cpu",
        external_authorities=("dataset_audit", "five_training_receipts", "five_checkpoints"),
        required_cli_options=("--checkpoint", "--dataset-audit", "--output"),
    ),
    ProducerContract(
        "production_manifest",
        "scripts/phaxis/build_production_manifest.py",
        "cpu",
        dependencies=("candidate_manifest",),
        external_authorities=("raw_analysis_metadata", "raw_review_manifest", "root_raw_input_manifest"),
        required_cli_options=("--analysis-metadata", "--review-manifest", "--root-input-manifest", "--output"),
    ),
    ProducerContract(
        "direct_benchmark_provider_descriptor",
        "scripts/phaxis/build_direct_benchmark_provider_descriptor.py",
        "cpu",
        dependencies=("production_manifest",),
        required_cli_options=("--project-root", "--physical-gpu", "--output", "--assemble"),
    ),
    ProducerContract(
        "release_case_prelocks",
        "scripts/phaxis/build_release_case_prelocks.py",
        "cpu",
        dependencies=("production_manifest",),
        required_cli_options=("--application-manifest", "--output"),
    ),
    ProducerContract(
        "qcdev_candidate_pool",
        "scripts/phaxis/build_stageb_train399_qcdev44_candidate_pool.py",
        "gpu",
        dependencies=("candidate_manifest",),
        external_authorities=("qcdev44_manifest", "five_checkpoints"),
        required_cli_options=("--manifest", "--checkpoint", "--candidate-manifest", "--output", "--device"),
    ),
    ProducerContract(
        "selection",
        "scripts/phaxis/select_stageb_train399_operating_point.py",
        "cpu",
        dependencies=("candidate_manifest", "qcdev_candidate_pool"),
        external_authorities=("canonical443_dataset", "split_manifest"),
        required_cli_options=(
            "--candidate-manifest",
            "--candidate-pool",
            "--dataset-root",
            "--dataset-manifest",
            "--split-manifest",
            "--selection-receipt",
            "--selected-model-metadata",
        ),
    ),
    ProducerContract(
        "qcdev_evaluation_inference",
        "scripts/phaxis/run_stageb_evaluation_inference.py",
        "gpu",
        dependencies=("candidate_manifest", "selection"),
        external_authorities=("qcdev44_manifest", "locked_val_ids", "five_checkpoints"),
        required_cli_options=("--manifest", "--locked-val-ids", "--candidate-manifest", "--selection-receipt", "--output"),
    ),
    ProducerContract(
        "qcdev_evaluation",
        "scripts/phaxis/evaluate_stageb_train399_qcdev44.py",
        "cpu",
        dependencies=("candidate_manifest", "selection", "qcdev_evaluation_inference"),
        external_authorities=("canonical443_dataset", "split_manifest"),
        required_cli_options=(
            "--detections",
            "--evaluation-inference-summary",
            "--candidate-manifest",
            "--selection-receipt",
            "--output",
        ),
    ),
    ProducerContract(
        "root_provider_exact283",
        "scripts/phaxis/run_root_provider.py",
        "gpu",
        dependencies=("production_manifest",),
        external_authorities=("frozen_root_bundle", "root_raw_provider_inputs", "root_reference_registry"),
        required_cli_options=("--bundle", "--input-manifest", "--reference-registry", "--output", "--v1-physical-gpu", "--q8-physical-gpu", "--strict-physical-gpu", "--execute"),
    ),
    ProducerContract(
        "root_bundle_materialization",
        "scripts/phaxis/materialize_verified_root_provider_bundle.py",
        "cpu",
        dependencies=("root_provider_exact283",),
        external_authorities=("frozen_root_bundle",),
        required_cli_options=("--source-bundle", "--output"),
    ),
    ProducerContract(
        "proposal",
        "scripts/phaxis/promote_model_contract.py",
        "cpu",
        dependencies=("candidate_manifest", "selection", "qcdev_evaluation", "root_provider_exact283"),
        external_authorities=("five_checkpoints", "pending_official_contract"),
        required_cli_options=("--current-model-contract", "--train399-candidate", "--train399-selection", "--train399-evaluation", "--root-exact283", "--output"),
    ),
    ProducerContract(
        "authority_pin",
        "internal:release_orchestrator.authority_pin",
        "internal_cpu",
        dependencies=("proposal",),
    ),
    ProducerContract(
        "analysis_workflow_manifest",
        "scripts/phaxis/build_analysis_workflow_manifest.py",
        "cpu",
        dependencies=(
            "production_manifest",
            "candidate_manifest",
            "selection",
            "proposal",
            "authority_pin",
        ),
        external_authorities=(
            "five_checkpoints",
            "frozen_root_bundle",
            "root_raw_provider_inputs",
            "raw_production_metadata",
            "static_axial_profile_contract",
        ),
        required_cli_options=(
            "--bundle",
            "--root-input-manifest",
            "--model-contract-proposal",
            "--stageb-input-manifest",
            "--checkpoint",
            "--candidate-manifest",
            "--selected-model-metadata",
            "--selection-receipt",
            "--traits-metadata",
            "--profile-contract",
            "--output",
        ),
    ),
    ProducerContract(
        "clean_install_sample_manifest",
        "scripts/phaxis/build_clean_install_sample_manifest.py",
        "cpu",
        dependencies=("analysis_workflow_manifest", "release_case_prelocks"),
        required_cli_options=("--analysis-workflow-manifest", "--case-selection", "--output"),
    ),
    ProducerContract(
        "qcdev_root_inputs",
        "scripts/phaxis/build_qcdev44_root_provider_inputs.py",
        "cpu",
        dependencies=("selection",),
        external_authorities=(
            "qcdev44_manifest",
            "locked_val_ids",
            "canonical443_dataset",
            "dataset_manifest",
        ),
        required_cli_options=(
            "--manifest",
            "--dataset-root",
            "--dataset-manifest",
            "--locked-val-ids",
            "--output",
        ),
    ),
    ProducerContract(
        "qcdev_root_provider",
        "scripts/phaxis/run_root_provider.py",
        "gpu",
        dependencies=("authority_pin", "root_provider_exact283", "qcdev_root_inputs"),
        external_authorities=("frozen_root_bundle",),
        required_cli_options=("--bundle", "--input-manifest", "--output", "--v1-physical-gpu", "--q8-physical-gpu", "--strict-physical-gpu", "--execute"),
    ),
    ProducerContract(
        "qcdev_fusion",
        "scripts/phaxis/run_cli.py",
        "cpu",
        dependencies=("qcdev_evaluation_inference", "qcdev_root_provider", "authority_pin"),
        required_cli_options=("--root-predictions", "--root-artifacts", "--hair-detections", "--model-contract", "--output"),
    ),
    ProducerContract(
        "production_stageb_exact283",
        "scripts/phaxis/run_stageb_inference.py",
        "gpu",
        dependencies=("production_manifest", "candidate_manifest", "selection", "authority_pin"),
        external_authorities=("five_checkpoints",),
        required_cli_options=("--manifest", "--checkpoint", "--output", "--device"),
    ),
    ProducerContract(
        "fusion_exact283",
        "scripts/phaxis/run_cli.py",
        "cpu",
        dependencies=("root_provider_exact283", "production_stageb_exact283", "authority_pin"),
        required_cli_options=("--root-predictions", "--root-artifacts", "--hair-detections", "--model-contract", "--output"),
    ),
    ProducerContract(
        "figure1_geometry_materialization",
        "scripts/phaxis/materialize_figure1_geometry.py",
        "cpu",
        dependencies=("release_case_prelocks", "production_manifest", "fusion_exact283"),
        required_cli_options=("--case-selection", "--application-manifest", "--fusion-root", "--output"),
    ),
    ProducerContract(
        "traits_exact283",
        "scripts/phaxis/export_traits.py",
        "cpu",
        dependencies=("fusion_exact283", "authority_pin"),
        external_authorities=("raw_production_metadata", "static_trait_contract"),
        required_cli_options=("--predictions", "--metadata", "--model-contract-proposal", "--output"),
    ),
    ProducerContract(
        "cohorts_exact283",
        "scripts/phaxis/build_biological_cohorts.py",
        "cpu",
        dependencies=("traits_exact283", "proposal"),
        external_authorities=("raw_study_design", "static_analysis_contract", "immutable_overlap_audit"),
        required_cli_options=("--trait-export", "--analysis-metadata", "--design-manifest", "--overlap-audit", "--analysis-contract", "--model-contract-proposal", "--output"),
    ),
    ProducerContract(
        "biological_analysis",
        "scripts/phaxis/analyze_biological_cohorts.py",
        "cpu",
        dependencies=("cohorts_exact283", "proposal"),
        external_authorities=("static_analysis_contract", "static_biological_model_spec"),
        required_cli_options=("--cohorts", "--analysis-contract", "--model-spec", "--model-contract-proposal", "--output"),
    ),
    ProducerContract(
        "profiles_exact283",
        "scripts/phaxis/export_cohort_distal_axis_profiles.py",
        "cpu",
        dependencies=("traits_exact283", "fusion_exact283", "cohorts_exact283", "proposal"),
        external_authorities=("static_axial_profile_contract",),
        required_cli_options=("--cohorts-root", "--contract", "--model-contract-proposal", "--traits-summary", "--output"),
    ),
    ProducerContract(
        "profile_analysis",
        "scripts/phaxis/analyze_distal_axis_profiles.py",
        "cpu",
        dependencies=("profiles_exact283", "proposal"),
        external_authorities=("static_profile_analysis_contract",),
        required_cli_options=("--primary-profiles", "--sensitivity-profiles", "--contract", "--model-contract-proposal", "--output"),
    ),
    ProducerContract(
        "historical_oof_evidence",
        "scripts/phaxis/build_historical_oof443_publication_evidence.py",
        "cpu",
        dependencies=("candidate_manifest",),
        external_authorities=("frozen_historical_oof_pickle", "canonical443_dataset", "split_manifest"),
        required_cli_options=("--oof-pickle", "--dataset-manifest", "--split-manifest", "--trusted-local-oof-pickle", "--output"),
    ),
    ProducerContract(
        "measurement_assurance",
        "scripts/phaxis/build_measurement_assurance_evidence.py",
        "cpu",
        dependencies=("qcdev_evaluation", "qcdev_evaluation_inference", "qcdev_fusion", "root_provider_exact283", "fusion_exact283", "traits_exact283", "cohorts_exact283"),
        external_authorities=("canonical443_dataset", "split_manifest", "static_trait_contract"),
        required_cli_options=("--train399-evaluation", "--qcdev-stageb-summary", "--qcdev-fusion-summary", "--application-fusion-summary", "--dataset-root", "--cohorts-receipt", "--root-exact283-receipt", "--output"),
    ),
    ProducerContract(
        "overlay_evidence",
        "scripts/phaxis/build_condition_blinded_overlay_evidence.py",
        "cpu",
        dependencies=("production_manifest", "release_case_prelocks", "fusion_exact283", "traits_exact283"),
        required_cli_options=(
            "--case-plan",
            "--application-manifest",
            "--full-traits",
            "--fusion-root",
            "--expected-task-count",
            "--output",
        ),
    ),
    ProducerContract(
        "benchmark_phaxis_production",
        "scripts/phaxis/run_external_direct_benchmark.py",
        "gpu",
        dependencies=("production_manifest", "direct_benchmark_provider_descriptor", "analysis_workflow_manifest", "authority_pin", "profiles_exact283"),
        required_cli_options=("--producer-interface", "--mode", "--source-manifest", "--image-root", "--analysis-manifest", "--workflow-output", "--output", "--cuda-visible-devices", "--execute"),
    ),
    ProducerContract(
        "benchmark_frozen_v1_production",
        "scripts/phaxis/run_external_direct_benchmark.py",
        "gpu",
        dependencies=("production_manifest", "direct_benchmark_provider_descriptor", "analysis_workflow_manifest"),
        external_authorities=("frozen_v1_read_only_assets",),
        required_cli_options=("--producer-interface", "--mode", "--source-manifest", "--image-root", "--analysis-manifest", "--workflow-output", "--output", "--cuda-visible-devices", "--execute"),
    ),
    ProducerContract(
        "benchmark_phaxis_sequential",
        "scripts/phaxis/run_external_direct_benchmark.py",
        "gpu",
        dependencies=("production_manifest", "direct_benchmark_provider_descriptor", "analysis_workflow_manifest", "authority_pin", "profiles_exact283"),
        required_cli_options=("--producer-interface", "--mode", "--source-manifest", "--image-root", "--analysis-manifest", "--workflow-output", "--output", "--cuda-visible-devices", "--execute"),
    ),
    ProducerContract(
        "benchmark_frozen_v1_sequential",
        "scripts/phaxis/run_external_direct_benchmark.py",
        "gpu",
        dependencies=("production_manifest", "direct_benchmark_provider_descriptor", "analysis_workflow_manifest"),
        external_authorities=("frozen_v1_read_only_assets",),
        required_cli_options=("--producer-interface", "--mode", "--source-manifest", "--image-root", "--analysis-manifest", "--workflow-output", "--output", "--cuda-visible-devices", "--execute"),
    ),
    ProducerContract(
        "benchmark_production_comparison",
        "scripts/phaxis/benchmark_full_workflow.py",
        "cpu",
        dependencies=("benchmark_phaxis_production", "benchmark_frozen_v1_production"),
        required_cli_options=("--compare-benchmarks", "--phaxis-summary", "--baseline-receipt", "--output"),
    ),
    ProducerContract(
        "benchmark_sequential_comparison",
        "scripts/phaxis/benchmark_full_workflow.py",
        "cpu",
        dependencies=("benchmark_phaxis_sequential", "benchmark_frozen_v1_sequential"),
        required_cli_options=("--compare-benchmarks", "--phaxis-summary", "--baseline-receipt", "--output"),
    ),
    ProducerContract(
        "benchmark_same_hardware",
        "scripts/phaxis/benchmark_full_workflow.py",
        "cpu",
        dependencies=("benchmark_phaxis_production", "benchmark_frozen_v1_production", "benchmark_phaxis_sequential", "benchmark_frozen_v1_sequential", "benchmark_production_comparison", "benchmark_sequential_comparison"),
        required_cli_options=("--aggregate-same-hardware", "--phaxis-production-summary", "--phaxis-sequential-summary", "--frozen-v1-production-summary", "--frozen-v1-sequential-summary", "--production-comparison", "--sequential-comparison", "--publish-receipt", "--output"),
    ),
    ProducerContract(
        "benchmark_artifact_inventory",
        "scripts/phaxis/build_benchmark_artifact_inventory.py",
        "cpu",
        dependencies=("benchmark_phaxis_production", "benchmark_frozen_v1_production", "benchmark_phaxis_sequential", "benchmark_frozen_v1_sequential", "benchmark_production_comparison", "benchmark_sequential_comparison", "benchmark_same_hardware"),
        required_cli_options=("--project-root", "--artifact", "--output", "--receipt"),
    ),
    ProducerContract(
        "figure_inputs",
        "scripts/phaxis/build_publication_figure_inputs.py",
        "cpu",
        dependencies=("candidate_manifest", "selection", "qcdev_evaluation", "proposal", "root_provider_exact283", "production_stageb_exact283", "fusion_exact283", "figure1_geometry_materialization", "traits_exact283", "cohorts_exact283", "biological_analysis", "profiles_exact283", "profile_analysis", "historical_oof_evidence", "measurement_assurance", "overlay_evidence", "benchmark_phaxis_production", "benchmark_frozen_v1_production", "benchmark_phaxis_sequential", "benchmark_frozen_v1_sequential", "benchmark_production_comparison", "benchmark_sequential_comparison", "benchmark_same_hardware", "benchmark_artifact_inventory"),
        external_authorities=("dataset_manifest", "static_trait_contract", "split_manifest", "five_training_completion_receipts"),
        required_cli_options=("--mode", "--model-contract-proposal", "--train399-candidate", "--dataset-manifest", "--image-traits-schema", "--training-receipt", "--benchmark-same-hardware", "--benchmark-artifact-inventory", "--historical-development-receipt", "--measurement-assurance-receipt", "--overlay-index-receipt", "--runtime-latency-summary", "--runtime-production-summary", "--output"),
    ),
    ProducerContract(
        "figures",
        "scripts/phaxis/build_publication_figures.py",
        "cpu",
        dependencies=("figure_inputs", "proposal", "qcdev_evaluation", "root_provider_exact283", "production_stageb_exact283", "fusion_exact283", "traits_exact283", "cohorts_exact283", "biological_analysis", "profiles_exact283"),
        required_cli_options=("--mode", "--figure-inputs", "--model-contract-proposal", "--output"),
    ),
    ProducerContract(
        "evidence",
        "scripts/phaxis/build_manuscript_evidence_manifest.py",
        "cpu",
        dependencies=("candidate_manifest", "selection", "qcdev_evaluation", "proposal", "root_provider_exact283", "production_stageb_exact283", "fusion_exact283", "traits_exact283", "cohorts_exact283", "biological_analysis", "profiles_exact283", "figure_inputs", "figures"),
        required_cli_options=("--model-contract-proposal", "--train399-candidate", "--train399-selection", "--train399-evaluation", "--root-exact283", "--figure-inputs", "--output"),
    ),
    ProducerContract(
        "official_apply",
        "scripts/phaxis/promote_model_contract.py",
        "cpu_cas",
        dependencies=("proposal", "production_stageb_exact283", "fusion_exact283", "traits_exact283", "evidence"),
        external_authorities=("pending_official_contract", "five_checkpoints"),
        required_cli_options=("--apply", "--expected-current-sha256", "--stageb-summary", "--fusion-summary", "--traits-summary", "--manuscript-evidence-manifest"),
    ),
    ProducerContract(
        "source_release",
        "scripts/phaxis/build_source_release.py",
        "cpu",
        dependencies=("official_apply", "root_provider_exact283", "candidate_manifest", "selection", "qcdev_evaluation", "fusion_exact283", "traits_exact283"),
        external_authorities=("release_author_metadata",),
        required_cli_options=("--root-provider-exact283-receipt", "--train399-candidate-manifest", "--train399-selection-receipt", "--train399-evaluation-receipt", "--final-fusion-summary", "--final-traits-summary", "--release-human-metadata", "--output"),
    ),
    ProducerContract(
        "distributions",
        "scripts/phaxis/build_release_distributions.py",
        "cpu",
        dependencies=("source_release",),
        required_cli_options=("--source-release-root", "--source-release-manifest", "--output", "--python"),
    ),
    ProducerContract(
        "offline_dependencies",
        "scripts/phaxis/materialize_offline_dependencies.py",
        "cpu",
        dependencies=("distributions",),
        required_cli_options=("--formal-wheel", "--python", "--output", "--execute"),
    ),
    ProducerContract(
        "handover_dataset_manifest",
        "scripts/phaxis/build_handover_dataset_manifest.py",
        "cpu",
        dependencies=("official_apply",),
        external_authorities=("immutable_manual500_and_canonical443", "author_release_attestation"),
        required_cli_options=("--project-root", "--release-attestation", "--output", "--receipt", "--execute"),
    ),
    ProducerContract(
        "handover_image_manifest",
        "scripts/phaxis/build_handover_image_manifest.py",
        "cpu",
        dependencies=("official_apply", "production_manifest"),
        external_authorities=("raw_biological283_images", "author_release_attestation"),
        required_cli_options=("--project-root", "--deployment-manifest", "--deployment-lock", "--image-root", "--release-attestation", "--output", "--receipt", "--execute"),
    ),
    ProducerContract(
        "handover_model_source_manifest",
        "scripts/phaxis/build_handover_model_source_manifest.py",
        "cpu",
        dependencies=("source_release",),
        external_authorities=("author_release_attestation",),
        required_cli_options=("--project-root", "--source-release-root", "--source-release-manifest", "--release-attestation", "--output", "--receipt", "--execute"),
    ),
    ProducerContract(
        "handover_model_asset_manifest",
        "scripts/phaxis/build_handover_model_asset_manifest.py",
        "cpu",
        dependencies=("official_apply", "candidate_manifest", "root_provider_exact283", "root_bundle_materialization", "clean_install_sample_manifest"),
        external_authorities=("five_checkpoints", "author_release_attestation"),
        required_cli_options=("--project-root", "--applied-model-contract", "--candidate-manifest", "--checkpoint", "--release-example-root", "--bundle-manifest-output", "--portable-capsule-output", "--release-attestation", "--output", "--receipt", "--execute"),
    ),
    ProducerContract(
        "clean_install_expected_identity",
        "scripts/phaxis/build_clean_install_expected_identity.py",
        "gpu",
        dependencies=("clean_install_sample_manifest", "proposal", "official_apply", "source_release", "distributions", "handover_model_asset_manifest"),
        external_authorities=("pending_official_contract",),
        required_cli_options=("--example-manifest", "--portable-capsule-root", "--model-contract-proposal", "--applied-model-contract", "--model-bundle-manifest", "--source-release-root", "--formal-wheel", "--python", "--physical-gpu", "--cuda-visible-devices", "--output", "--execute"),
    ),
    ProducerContract(
        "handover_benchmark_manifest",
        "scripts/phaxis/build_handover_benchmark_manifest.py",
        "cpu",
        dependencies=("benchmark_same_hardware", "benchmark_artifact_inventory"),
        external_authorities=("author_release_attestation",),
        required_cli_options=("--project-root", "--same-hardware-receipt", "--artifact-inventory", "--release-attestation", "--output", "--receipt", "--execute"),
    ),
    ProducerContract(
        "clean_install",
        "scripts/phaxis/build_clean_install_verification.py",
        "gpu",
        dependencies=("proposal", "official_apply", "source_release", "distributions", "offline_dependencies", "clean_install_sample_manifest", "handover_model_asset_manifest", "clean_install_expected_identity"),
        required_cli_options=(
            "--wheel",
            "--source-release-root",
            "--applied-model-contract",
            "--model-contract-proposal",
            "--model-bundle-manifest",
            "--portable-capsule-root",
            "--base-python",
            "--work-root",
            "--cuda-visible-devices",
            "--execute",
            "--output",
        ),
    ),
    ProducerContract(
        "values",
        "scripts/phaxis/build_manuscript_values.py",
        "cpu",
        dependencies=(
            "candidate_manifest",
            "selection",
            "qcdev_evaluation",
            "proposal",
            "root_provider_exact283",
            "production_stageb_exact283",
            "fusion_exact283",
            "traits_exact283",
            "cohorts_exact283",
            "biological_analysis",
            "profiles_exact283",
            "evidence",
            "figure_inputs",
            "figures",
            "handover_model_asset_manifest",
            "clean_install",
            "source_release",
        ),
        external_authorities=("author_verified_manuscript_metadata", "master_manuscript"),
        required_cli_options=("--evidence-graph", "--figure-inputs", "--figure-assembly-summary", "--model-contract-proposal", "--human-metadata", "--model-bundle-manifest", "--clean-install-receipt", "--source-release-manifest", "--evidence-artifact", "--output"),
    ),
    ProducerContract(
        "manuscript",
        "scripts/phaxis/compile_manuscript.py",
        "cpu",
        dependencies=("evidence", "figures", "values"),
        external_authorities=("master_manuscript",),
        required_cli_options=("--master", "--evidence-graph", "--values", "--output", "--receipt"),
    ),
    ProducerContract(
        "supplementary_manuscript",
        "scripts/phaxis/compile_supplementary_manuscript.py",
        "cpu",
        dependencies=("figures", "manuscript"),
        external_authorities=("supplement_master",),
        required_cli_options=("--master", "--main-manuscript", "--main-compile-receipt", "--figure-summary", "--output", "--receipt"),
    ),
    ProducerContract(
        "submission_docx",
        "scripts/phaxis/build_submission_docx.py",
        "cpu",
        dependencies=("figures", "manuscript"),
        external_authorities=("submission_title_metadata",),
        required_cli_options=("--mode", "--manuscript", "--submission-metadata", "--compile-receipt", "--figure-summary", "--title-page-output", "--anonymized-main-output", "--receipt"),
    ),
    ProducerContract(
        "supplementary_docx",
        "scripts/phaxis/build_supplementary_docx.py",
        "cpu",
        dependencies=("figures", "manuscript", "supplementary_manuscript"),
        required_cli_options=("--mode", "--supplement", "--main-manuscript", "--main-compile-receipt", "--figure-summary", "--output", "--receipt"),
    ),
    ProducerContract(
        "manuscript_artifact_qa",
        "scripts/phaxis/verify_manuscript_artifacts.py",
        "cpu",
        dependencies=("figures", "manuscript", "supplementary_manuscript", "submission_docx", "supplementary_docx"),
        external_authorities=("master_manuscript", "supplement_master", "submission_title_metadata"),
        required_cli_options=("--main-master", "--supplement-master", "--main-manuscript", "--main-compile-receipt", "--supplement-manuscript", "--supplement-compile-receipt", "--submission-metadata", "--figure-summary", "--title-page-docx", "--anonymized-main-docx", "--submission-docx-receipt", "--anonymized-supplement-docx", "--supplement-docx-receipt", "--output", "--upload-manifest"),
    ),
    ProducerContract(
        "manuscript_render",
        "scripts/phaxis/render_manuscript_bundle.py",
        "cpu",
        dependencies=("submission_docx", "supplementary_docx", "manuscript_artifact_qa"),
        required_cli_options=("--title-page-docx", "--anonymized-main-docx", "--anonymized-supplement-docx", "--structural-qa", "--upload-manifest", "--output", "--powershell", "--pdftoppm"),
    ),
    ProducerContract(
        "manuscript_visual_qa",
        "scripts/phaxis/validate_manuscript_visual_qa.py",
        "cpu",
        dependencies=("manuscript_artifact_qa", "manuscript_render"),
        required_cli_options=("--render-receipt", "--structural-qa", "--template", "--attestation", "--output"),
    ),
    ProducerContract(
        "handover_contract",
        "scripts/phaxis/assemble_handover_build_contract.py",
        "cpu",
        dependencies=("official_apply", "candidate_manifest", "selection", "qcdev_evaluation", "root_provider_exact283", "fusion_exact283", "traits_exact283", "benchmark_same_hardware", "source_release", "clean_install", "handover_dataset_manifest", "handover_image_manifest", "handover_model_source_manifest", "handover_model_asset_manifest", "handover_benchmark_manifest"),
        external_authorities=("static_trait_contract", "five_checkpoints", "author_release_attestation"),
        required_cli_options=("--project-root", "--checkpoint", "--release-attestation", "--output", "--receipt", "--execute"),
    ),
    ProducerContract(
        "handover",
        "scripts/phaxis/build_handover_package.py",
        "cpu",
        dependencies=("handover_contract",),
        required_cli_options=("--contract", "--output"),
    ),
    ProducerContract(
        "release_finalize",
        "internal:release_orchestrator.release_finalize",
        "internal_cpu",
        dependencies=("official_apply", "source_release", "distributions", "clean_install", "values", "manuscript", "supplementary_manuscript", "submission_docx", "supplementary_docx", "manuscript_artifact_qa", "manuscript_render", "manuscript_visual_qa", "handover"),
        external_authorities=("release_authority_registry",),
    ),
)


MANDATORY_STAGE_ORDER = tuple(item.name for item in FORMAL_RELEASE_PRODUCERS)
STAGE_DEPENDENCIES = {
    item.name: item.dependencies for item in FORMAL_RELEASE_PRODUCERS
}


def _argument_options(source: Path) -> set[str]:
    """Return literal argparse option strings declared by one real producer."""

    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ReleaseTopologyError(f"cannot inspect producer source: {source}: {error}") from error
    trees = [tree]
    # run_cli.py is a checked-in src-layout bootstrap, not a second parser.  It
    # must import the public CLI entry point literally; inspect that real
    # delegated argparse source so dependency-bearing fuse options remain
    # auditable without relying on caller PYTHONPATH or an installed package.
    if source.name == "run_cli.py":
        delegates_public_cli = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "phaxis.cli"
            and any(alias.name == "main" for alias in node.names)
            for node in ast.walk(tree)
        )
        if not delegates_public_cli:
            raise ReleaseTopologyError(
                "run_cli.py no longer delegates to the public phaxis.cli main"
            )
        delegated = source.parents[2] / "src" / "phaxis" / "cli.py"
        try:
            trees.append(
                ast.parse(
                    delegated.read_text(encoding="utf-8"),
                    filename=str(delegated),
                )
            )
        except (OSError, SyntaxError, UnicodeError) as error:
            raise ReleaseTopologyError(
                f"cannot inspect delegated public CLI source: {delegated}: {error}"
            ) from error
    options: set[str] = set()
    for node in (child for parsed in trees for child in ast.walk(parsed)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    options.add(argument.value)
    return options


def validate_release_topology(
    *,
    project_root: str | Path | None = None,
    producers: Sequence[ProducerContract] = FORMAL_RELEASE_PRODUCERS,
    inspect_real_producers: bool = True,
) -> dict[str, Any]:
    """Validate graph reachability and dependency-bearing real CLI contracts.

    The returned object is a non-formal diagnostic.  It intentionally contains
    no accuracy, latency, throughput, GPU-memory or speedup values and cannot be
    consumed as a formal release receipt.
    """

    names = [item.name for item in producers]
    if not names or len(names) != len(set(names)):
        raise ReleaseTopologyError("producer stage names are absent or duplicated")
    by_name = {item.name: item for item in producers}
    observed: set[str] = set()
    source_checks: list[dict[str, Any]] = []
    root = Path(project_root).resolve() if project_root is not None else None
    for item in producers:
        if item.execution_class not in {
            "cpu",
            "gpu",
            "cpu_cas",
            "cpu_isolated_environment",
            "internal_cpu",
        }:
            raise ReleaseTopologyError(
                f"{item.name}: unsupported execution class: {item.execution_class}"
            )
        if len(item.dependencies) != len(set(item.dependencies)):
            raise ReleaseTopologyError(f"{item.name}: duplicate producer dependency")
        unknown = [name for name in item.dependencies if name not in by_name]
        if unknown:
            raise ReleaseTopologyError(
                f"{item.name}: unknown producer dependencies: {unknown}"
            )
        late = [name for name in item.dependencies if name not in observed]
        if late:
            raise ReleaseTopologyError(
                f"{item.name}: dependency is not reachable in topological order: {late}"
            )
        observed.add(item.name)
        if not inspect_real_producers or item.producer.startswith(("internal:", "module:")):
            continue
        if root is None:
            raise ReleaseTopologyError("project_root is required to inspect real producers")
        path = (root / item.producer).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ReleaseTopologyError(f"{item.name}: producer escapes project root") from error
        if not path.is_file() or path.is_symlink():
            raise ReleaseTopologyError(f"{item.name}: real producer is absent: {path}")
        options = _argument_options(path)
        missing = sorted(set(item.required_cli_options) - options)
        if missing:
            raise ReleaseTopologyError(
                f"{item.name}: real producer CLI lost dependency options: {missing}"
            )
        source_checks.append(
            {
                "stage": item.name,
                "producer": item.producer,
                "producer_sha256": sha256_file(path),
                "required_cli_options": list(item.required_cli_options),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": TOPOLOGY_AUDIT_SCHEMA,
        "status": (
            "structurally_valid_with_declared_producer_gap"
            if any(item.capability_gap for item in producers)
            else "structurally_valid_non_formal"
        ),
        "cpu_only_inspection": True,
        "formal_release_allowed": False,
        "formal_result_receipt": False,
        "scientific_or_performance_results_present": False,
        "producer_stage_order": names,
        "producer_dependencies": {
            item.name: list(item.dependencies) for item in producers
        },
        "external_authorities": {
            item.name: list(item.external_authorities)
            for item in producers
            if item.external_authorities
        },
        "declared_capability_gaps": sorted(
            {str(item.capability_gap) for item in producers if item.capability_gap}
        ),
        "stages_blocked_by_declared_capability_gap": {
            item.name: str(item.capability_gap)
            for item in producers
            if item.capability_gap
        },
        "real_producer_source_checks": source_checks,
        "official_apply_is_authority_checkpoint": True,
        "release_finalize_is_terminal_stage": names[-1] == "release_finalize",
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    payload["topology_audit_identity_sha256"] = sha256_json(payload)
    return payload


def require_manifest_stage_dependencies(
    stage_inputs: Mapping[str, Iterable[str]],
) -> None:
    """Require every real producer edge to be explicit in a release manifest."""

    names = set(stage_inputs)
    if names != set(MANDATORY_STAGE_ORDER):
        missing = sorted(set(MANDATORY_STAGE_ORDER) - names)
        extra = sorted(names - set(MANDATORY_STAGE_ORDER))
        raise ReleaseTopologyError(
            f"manifest stage set differs from producer topology; missing={missing}, extra={extra}"
        )
    for name, dependencies in STAGE_DEPENDENCIES.items():
        declared = set(stage_inputs[name])
        absent = sorted(set(dependencies) - declared)
        if absent:
            raise ReleaseTopologyError(
                f"{name}: real producer dependencies are absent: {absent}"
            )


__all__ = [
    "FORMAL_RELEASE_PRODUCERS",
    "MANDATORY_STAGE_ORDER",
    "ProducerContract",
    "ReleaseTopologyError",
    "STAGE_DEPENDENCIES",
    "TOPOLOGY_AUDIT_SCHEMA",
    "require_manifest_stage_dependencies",
    "validate_release_topology",
]
