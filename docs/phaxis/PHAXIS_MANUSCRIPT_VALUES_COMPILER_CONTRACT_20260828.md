# PHAxis 1.0.0 manuscript-values compiler contract

## Purpose

`scripts/phaxis/build_manuscript_values.py` is the only supported production route from final evidence to the `PHAxis-manuscript-values-1.1` file consumed by `compile_manuscript.py`. It is not a general key/value template filler. The builder derives the exact token inventory from the current master on every invocation and requires one-and-only-one value for every token. After the 30 August 2026 plant-facing Table 3, acquisition-schema, and commit-free release-metadata revisions, the current master audit contains 229 unique tokens; this count is diagnostic, not a permissive schema constant. The 15 q values, full283 sensitivity records and complete 192-slot status ledger remain machine-readable in Supplementary Data S9 rather than being duplicated as main-manuscript prose tokens. The main tokens are closed as follows:

- machine tokens are recomputed from named, hash-verified final receipts or selected cells in the sealed publication figure-input bundle;
- 34 genuinely external fields—author statements, 15 structured biological-acquisition fields, image-availability statements, repository and DOI records, and release/data/model URLs and licences—come from one separately sealed `PHAxis-manuscript-human-metadata-1.0` JSON file;
- the two explicit historical families, `HISTORICAL_OOF_*` and `LOCKED_LEGACY_HYBRID_IDENTITY_*`, may use development/comparator authority only under those names. No historical source may satisfy a `FINAL_*` token.

The builder rereads the master on every invocation. It therefore never accepts a cached token list or an earlier master SHA.

The former free-text `FINAL_BIOLOGICAL_ACQUISITION_METHODS` token is forbidden. Acquisition metadata are split into exact scalar fields for accession; construct/control identity and source; growth medium; photoperiod; growth timeline; temperature-exposure onset and duration; plate/block and plant/source-unit mapping; replication and randomization; imaging device; objective; native pixel sampling; field sampling and stitching; physical calibration; and prephenotype exclusion/consolidation rules. The incomplete template records each unknown field as `DEFERRED_AUTHOR_VERIFICATION`. `deferred` is a forbidden final-value marker, so changing the document status or sealing the template without replacing every deferred field cannot authorize compilation. Values must come from an author-verifiable experimental record; filenames, neighboring experiments, and image appearance are not admissible substitutes. The author-facing evidence and completion requirements for every field are listed in `PHAXIS_BIOLOGICAL_ACQUISITION_METADATA_COMPLETION_CN_20260829.md`.

The two machine-generated biological narrative tokens use a fixed layer order: primary hair change (visible abundance, then supported projected length), spatial location (first-hair deployment and distal profiles), and supporting-root context (width/extent interpreted beside R01–R19). A headline still requires a clean interval excluding the null and a Full283 point estimate in the same direction. The generator may not count correlated derived descriptors, use a cross-trait directional vote, or convert the number of non-null traits into a biological conclusion.

## Required inputs

The command requires explicit paths for the master, final evidence graph, every one of the graph's 13 named artifacts, figure-input manifest, figure-assembly summary, passed unapplied model-contract proposal, human metadata, immutable model-bundle manifest, and clean-install receipt. It does not discover a newest run. The separately named proposal and figure-input paths must resolve to the same files already locked under those graph artifact roles.

The figure input must use schema `PHAxis-manuscript-figure-inputs-2.0`, assembler `PHAxis-publication-figure-input-assembly-1.0`, the exact frozen 19 resource roles (including the source-derived `multitrait_atlas`), the exact source-input/provenance roles, and a valid `figure_input_assembly_identity_sha256`. Required numeric inputs include the QC-development sufficient statistics, 5/10/20-µm tolerance table, historical strata, assurance metrics/pairs/support/topology, canonical 32-column image-trait table, clean/full biological tables, the complete 32-descriptor clean/full support atlas, fixed-family effects, axial profiles, and the two-mode direct runtime receipts. The atlas may estimate effects only for the prespecified five endpoints; all remaining contrast slots must carry a hash-verified not-estimated reason.

The `measurement_assurance` provenance role is a sealed master receipt with schema `PHAxis-measurement-assurance-receipt-1.0`, status `completed_locked_qc_development_assurance`, scope `QC-development measurement assurance; non-independent`, `independent_accuracy_claim_allowed=false`, and `blind_images_used=0`. Its `measurement_assurance_identity_sha256` must seal the complete unsigned payload. It must embed both `root_continuity_assurance` and `hair_attachment_assurance`, and its `component_receipts` object must contain exactly the matching `root_continuity` and `hair_attachment` audit records. These are final-machine QC-development measurement evidence, but explicitly non-independent; neither child receipt permits an independent-accuracy claim.

The proposal owns the public identities. The builder requires:

- hair expert `PHAxis-StageB-train399-five-seed`;
- model ID prefix `PHAXIS-V1.0.0-STRICT-TRAIN399-`;
- root provider role `PHAxis-portable-root-provider` separately from the public root expert ID;
- an instantiated public root expert ID beginning `PHAxis-root-provider-`;
- exact equality between `root_expert.bundle_identity_sha256` and `root_expert.root_bundle_authority.bundle_identity_sha256`.

Pipeline identity remains runtime provenance and is not used to manufacture the public root ID.
Likewise, frozen provider-bundle ABI labels remain internal receipt provenance. They cannot populate a manuscript product-version field or create a second public release name; all public manuscript, figure, table, and availability text uses `PHAxis 1.0.0`.

## Measurement-assurance child-receipt closure

The publication chain validates the two embedded child receipts semantically; checking only a parent-file SHA or a child identity string is insufficient. Before any child-derived `FINAL_*` value is emitted, all of the following must close exactly:

1. **Seal, schema, status, and role.** Each child identity is the SHA-256 of the complete unsigned child payload. Root continuity must use `PHAxis-primary-root-continuity-assurance-1.0`; hair attachment must use `PHAxis-hair-attachment-assurance-1.0`. Both must have `status=completed`, `evidence_role=annotated_qc_development_non_independent`, `independent_accuracy_claim_allowed=false`, `provider_equivalence_used_as_accuracy=false`, and `blind_images_used=0`. Hair attachment additionally requires `val_labels_used_for_training=false`.
2. **Exact source-image denominator.** Each child contains exactly 44 unique QC-development source images, one per source unit. `per_image_set_identity_sha256` must seal the ordered per-image rows and `source_unit_set_identity_sha256` must seal the ordered `{source_unit, source_image_sha256}` set. Root and hair child source-unit-set identities must be equal. Bootstrap resampling is by source image, never by component, matched pair, or individual hair.
3. **Bootstrap contract.** Both children require the source-image nonparametric percentile bootstrap, 10,000 repetitions, seed `20260828`, and two-sided 95% limits at 2.5% and 97.5%. Their per-image bootstrap sufficient statistics, point estimates, intervals, and aggregate summaries are recomputed. The hair child must carry all matches and errors belonging to a resampled image together.
4. **Geometry and authority identity.** Each child carries a valid `input_geometry_set_identity_sha256`, a sealed `input_contract_identity_sha256`, prediction authority, reference/annotation authority, and implementation SHA. The same identities must be duplicated unchanged in the child's provenance block. Root prediction authority must resolve to the sealed final fused root masks delivered to trait extraction. Hair prediction authority must resolve to validated production Stage-B geometry exactly cross-checked against the final fused `identity_hairs` and `count_hairs`. The canonical vector-derived reference/annotation authority is scoring-only and must never be read during inference.
5. **Parent binding.** The master receipt's `source_authority_identity_sha256` map, `component_receipts[*].identity_sha256`, declared identity field, and embedded child identity must agree. The parent's metric-role map, counts, prediction locks, fused-assurance input-set identity, and the child's denominators must also close. Evaluation-only and production Stage-B file sets remain distinct byte authorities while sharing the same locked model/checkpoint/threshold authority.
6. **Auditable reconstruction.** Every component audit record names a basename-only, regular, non-symlink sibling copy of the child receipt and of its portable input contract. The root input contract must use `PHAxis-primary-root-continuity-assurance-input-1.0`; the hair input contract must use `PHAxis-hair-attachment-assurance-input-1.0`. Both file SHA-256 values must match the audit record; the strictly parsed child copy must equal the embedded child object; the input contract must self-seal to the recorded `input_contract_identity_sha256`. Rebuilding the child from that sealed input geometry must reproduce the embedded child exactly. Missing files, path escape, symlink substitution, identity drift, or any reconstruction difference is fatal.
7. **Metric-table equality.** Canonical child results are copied into `assurance_metrics` under `publication_metric_role=formal_measurement_assurance`, except for the explicitly diagnostic union-coverage row. Each value and CI must equal the recomputation from the child per-image sufficient statistics at absolute tolerance `1e-12`; domain, unit, evidence role, source-image `n`, instance denominator, bootstrap metadata, label, and definition must also agree.

The figure-input assembler performs the full sibling audit-copy and geometry reconstruction while it still has the physical measurement-assurance receipt path. The manuscript-values builder then rereads the sealed master receipt from the figure-input provenance registry, revalidates the embedded child seals, semantics, source-image denominator, metric rows, counts, and derivation pointers, and binds the selected child cells into each output entry. The compiler accepts only the resulting source-role contract; it cannot replace this chain with hand-authored values.

## Canonical component metrics and token families

### Primary-root continuity

Root continuity is evaluated in `physical_um_xy` against the ordered visible reference axis derived from the canonical vector root polygon and the annotated distal/root-cap point. Prediction geometry consists of **every 8-connected component** skeletonized from the sealed final fused root foreground. The evaluator may not interpolate, bridge, join, or complete gaps. Reference support uses a 5-µm tolerance and at most 2-µm sampling intervals.

The formal criterion is one connected component: a root is break-free only if at least one individual predicted component supports every reference interval. Union-of-components coverage cannot convert jointly covering fragments into a continuous root. Consequently, `root_continuity_reference_axis_coverage_mean` is retained only as `diagnostic_only_union_coverage` and has no `FINAL_ROOT_CONTINUITY_*` accuracy token. The canonical formal metric rows and token mappings are:

| Canonical metric key | Manuscript token | Denominator / interpretation |
|---|---|---|
| `root_continuity_maximum_single_component_coverage_mean` | `FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEAN` | Mean best-single-component reference coverage; 44 source images / 44 root instances |
| `root_continuity_maximum_single_component_coverage_median` | `FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEDIAN` | Median best-single-component reference coverage; 44 / 44 |
| `root_continuity_best_component_gap_median_um` | `FINAL_ROOT_CONTINUITY_LONGEST_UNSUPPORTED_GAP_UM_ON_BEST_COMPONENT_MEDIAN` | Median longest unsupported gap on the maximum-coverage single component; 44 / 44 |
| `root_continuity_break_free_rate` | `FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE` | Fraction of images having one component that spans every reference interval; 44 / 44 |
| `root_continuity_visible_axis_extent_mae_um` | `FINAL_ROOT_CONTINUITY_VISIBLE_AXIS_EXTENT_MAE_UM` | Mean absolute proximal-to-distal projected-extent error; internal gaps remain separately penalized; 44 / 44 |

`FINAL_ROOT_CONTINUITY_VALIDATION_N` is the sealed 44-source-image denominator. `FINAL_ROOT_CONTINUITY_METRICS_CI` formats the five source-image percentile intervals above and may not include the union diagnostic as formal continuity evidence. The specific `FINAL_ROOT_CONTINUITY_*` rule must be matched before the broader `FINAL_ROOT_*` family so these tokens always use `source_role=measurement_assurance`.

### Formal root-hair attachment

The annotation semantics remain one attachment-to-visible-end centreline polyline per biological root hair, not a dense-width mask. Formal attachment uses the already established one-to-one biological-presence assignments at 20 µm, with bidirectional curve coverage at least 0.25, proximal direction cosine at least 0, proximal arc fraction 0.25, and 32 resampled points. It then asks whether each assigned pair's attachment/base error is at most 20 µm. There is **no second base-only rematching**.

The original biological-identity denominators are preserved: precision uses all `n_pred`, recall uses all `n_gt`, and F1 is recomputed from those pooled counts. Attachment-position median and P95 use all formal biological-presence matches, not only the attachment-qualified subset. The production/fused per-image and pooled `n_pred`, `n_gt`, and biological-presence TP@20 µm must exactly reproduce the sealed evaluation-only Stage-B counts; shared model authority alone is not enough to establish coordinate or decoding equivalence.

| Canonical metric key | Manuscript token | Denominator / interpretation |
|---|---|---|
| `hair_attachment_qualified_precision_20um` | `FINAL_HAIR_ATTACHMENT_QUALIFIED_PRECISION_AT_20UM` | Attachment-qualified formal-identity TP / all predicted identities (`n_pred`) |
| `hair_attachment_qualified_recall_20um` | `FINAL_HAIR_ATTACHMENT_QUALIFIED_RECALL_AT_20UM` | Attachment-qualified formal-identity TP / all annotated identities (`n_gt`) |
| `hair_attachment_qualified_f1_20um` | `FINAL_HAIR_ATTACHMENT_QUALIFIED_F1_AT_20UM` | Pooled F1 retaining `n_pred` and `n_gt`; metric-row instances = `n_pred+n_gt` |
| `hair_attachment_error_median_um` | `FINAL_HAIR_ATTACHMENT_FORMAL_MATCHED_ERROR_MEDIAN_UM` | Median base error across all formal biological-presence matches |
| `hair_attachment_error_p95_um` | `FINAL_HAIR_ATTACHMENT_FORMAL_MATCHED_ERROR_P95_UM` | P95 base error across all formal biological-presence matches |

The family also contains `FINAL_HAIR_ATTACHMENT_VALIDATION_N`, `FINAL_HAIR_ATTACHMENT_PREDICTED_N`, `FINAL_HAIR_ATTACHMENT_ANNOTATED_N`, `FINAL_HAIR_ATTACHMENT_QUALIFIED_TP_N`, `FINAL_HAIR_ATTACHMENT_FORMAL_MATCH_N`, and `FINAL_HAIR_ATTACHMENT_METRICS_CI`. Here `FORMAL_MATCH_N` is biological-presence TP before the attachment-tolerance qualification, whereas `QUALIFIED_TP_N` is the subset whose base error is at most 20 µm. The 5/10/20-µm base-only matching sweep is development-only localization sensitivity, does not select the operating point, and cannot supply any formal attachment token. The specific `FINAL_HAIR_ATTACHMENT_*` rule must precede the broader `FINAL_HAIR_*` family and uses `source_role=measurement_assurance`.

### Physical-scale assurance remains unchanged

QC-development44 closes as 38 images with one annotated visible scale bar, six images with trusted metadata but no visible bar, and zero absent or untrusted-scale truth cases. This applicability split must not be collapsed:

- `FINAL_SCALE_DETECTION_COVERAGE`, `FINAL_SCALE_DETECTED_N`, and `FINAL_SCALE_VALIDATION_N` use the 38 visible-bar cases; detected count is the metric row's `instances` and coverage must equal `instances/n`;
- `FINAL_SCALE_LOCALIZATION_ERROR_UM` and `FINAL_SCALE_LOCALIZATION_N` are conditional on the detected visible bars, and the localization-pair count must equal that detected count;
- `FINAL_SCALE_RELATIVE_ERROR_PERCENT` is likewise conditional on detected visible bars, and its calibration-pair denominator must equal the detected count;
- `FINAL_SCALE_LOCALIZATION_CI` and `FINAL_SCALE_ERROR_CI` are source-image bootstrap intervals with 10,000 repetitions and seed `20260828`;
- `FINAL_SCALE_APPLICABILITY_STATEMENT` must state the 38 + 6 + 0 composition; `FINAL_SCALE_ABSENCE_SPECIFICITY_STATUS` must remain `not_estimable_no_absent_or_untrusted_scale_cases`; and `FINAL_SCALE_FAIL_CLOSED_EVIDENCE_STATEMENT` must distinguish software-contract/unit-test evidence from empirical absence specificity.
- `FINAL_SCALE_ELIGIBLE_N` remains the separately sealed application-image count that passed the locked scale gate; it cannot replace, or be replaced by, any QC-development44 assurance denominator.

The six trusted-metadata cases support physical calibration applicability but are neither visible-bar detection cases nor negative absence cases. Software fail-closed tests cannot be restated as empirical absence specificity.

## Derivation and provenance

Each value entry contains `value`, its manuscript semantic `source_role`, and a sealed derivation. A derivation names the operation and one or more physical source records. Each source record locks:

- physical source role and full-file SHA-256;
- figure-assembly or receipt logical identity;
- an explicit JSON pointer or CSV row/column selection;
- a digest of the exact selected cells;
- authority class: `final_machine`, `historical_development_comparator`, or `human_external`.

The builder independently recomputes the principal sufficient-statistic results. Examples include pooled tolerant-presence precision/recall/F1; formal attachment-qualified precision/recall/F1 and formal-match attachment errors without rematching; single-connected-component root-continuity metrics; fixed-seed 10,000-replicate source-image bootstrap intervals; image-level count MAE/bias/CCC and paired deltas; matched-length agreement from the pair table; topology aggregates from 261 per-source rows; 32-trait coverage; fixed 15-effect clean/full comparisons; profile counts/patterns; and direct runtime medians/stage shares. Scale detection coverage, annotated-image denominator, and detected count are bound to the same assurance-metric row; coverage must equal `instances/n`, and both conditional localization and relative-calibration-error denominators must equal the detected count. For the Fig. 6(f) frozen-v1 comparison, the single sealed latency mode is converted through a fixed two-value human-readable registry; frozen-v1 batch wall minutes and median latency are taken from their named direct summaries, while batch-wall and median-latency speedups are recomputed from the PHAxis/frozen-v1 base timings. Both comparison blocks must remain `comparable_direct_full283`, carry no noncomparability reasons, and preserve the assembly's exact283 same-source/hardware/I/O/no-cache declaration; the recomputed ratios must equal the sealed comparison fields at `1e-12` relative and absolute tolerance. Normalized figure cells must agree with their sealed source tables at tight numerical tolerance.

For Table 3, every four-cell vector is emitted in the fixed order EV-22°C / EV-30°C / OE-22°C / OE-30°C. Block A source-unit medians and IQRs are recomputed directly from the clean formal trait rows, with nulls excluded rather than zero-filled; Block B contains the 15 centered model contrasts. Pool, post-formal-gate, and endpoint-specific non-null counts are recomputed separately and must close to the corresponding total/effect denominator. First-hair observability is rendered as `n/formal n (%)`; shootward-edge censoring is rendered as `not-border-visible/evaluable n (%)`, with null visibility flags excluded rather than counted as false. Each Full283 sensitivity value includes its own endpoint n and whether its direction relative to ratio 1 agrees with the clean analysis.

The output self-seals all entries and the complete source-file registry. It also carries the master, evidence graph, figure assembly, proposal, human metadata, model bundle, clean-install, public model/root/hair IDs, and root-bundle authority identities. Publication uses an atomic no-overwrite hard-link operation.

## Fail-closed rules

The builder and compiler refuse:

- missing, extra, duplicated, null, NaN, infinite, empty, residual-token, `TODO`, `TBD`, or provisional values;
- the retired `FINAL_BIOLOGICAL_ACQUISITION_METHODS` field, any missing structured acquisition field, or any value retaining a `deferred` marker;
- a biological narrative whose layer order is not primary hair change → spatial location → supporting-root context, or whose conclusion is based on the count/directional vote of correlated derived descriptors;
- any source file whose bytes differ from the manifest after validation;
- blind-labelled paths or any receipt with `blind_images_used != 0`;
- root-cap-region statistics, canonical-annotation inference reads, or condition-metadata model routing;
- a candidate/legacy/fold/443CV expert name in `FINAL_HAIR_EXPERT_ID`;
- historical/comparator authority in any `FINAL_*` derivation;
- a missing, extra, unsealed, schema-drifted, role-drifted, non-completed, or non-QC-development root-continuity/hair-attachment child receipt;
- a child source-image set other than the exact 44 unique QC-development images, unequal root/hair source-unit-set identities, or bootstrap units other than source images;
- child audit-copy or input-contract audit-copy path escape, symlink, missing file, SHA drift, self-seal drift, embedded/copy mismatch, or geometry-rebuild mismatch;
- a child input-geometry, reference/annotation authority, prediction authority, implementation, per-image set, or parent authority-map identity that does not close through all audit copies and provenance fields;
- a root-continuity formal metric derived from union-of-components support, evaluator-side bridging/gap completion, or a break-free decision not witnessed by one connected component;
- a formal hair-attachment value obtained by base-only rematching, by promoting the 5/10/20-µm proxy sweep, by replacing the original `n_pred`/`n_gt`, or by restricting formal-match localization error to the attachment-qualified subset;
- production/fused hair geometry whose per-image or pooled `n_pred`, `n_gt`, or biological-presence TP@20 µm differs from the sealed evaluation-only Stage-B result;
- any canonical component metric row whose value, CI, domain, unit, evidence role, publication role, source-image `n`, instance denominator, bootstrap metadata, label, or definition differs from its rebuilt child receipt;
- scale detection coverage that does not reproduce from the same metric row's detected `instances` and annotated-image `n`, or a conditional calibration-error denominator that differs from the detected count;
- a scale-localization denominator that differs from the detected visible-bar count, inclusion of the six metadata-only cases in the visible-bar denominator, treatment of those six cases as absence negatives, any empirical absence-specificity claim from the 38 + 6 + 0 set, or substitution of software tests for empirical specificity;
- a frozen-v1 timing comparison that is not `comparable_direct_full283`, changes the single latency mode or full-workflow scope, or whose reported speedup cannot be recomputed from its bound base timings;
- a source record absent from, or hash-inconsistent with, the sealed source registry;
- output overwrite.

If human metadata are absent, the CLI writes a clearly invalid missing-field report and completion template. The template has `formal_values_build_allowed=false`, null fields, and an incomplete status; it cannot pass formal compilation.

## Command shape

```powershell
.\envs\rhpheno\python.exe scripts\phaxis\build_manuscript_values.py `
  --master docs\phaxis\PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260828.md `
  --evidence-graph <final-evidence-graph.json> `
  --evidence-artifact model_contract_proposal=<proposal.json> `
  --evidence-artifact train399_candidate=<candidate.json> `
  --evidence-artifact train399_selection=<selection.json> `
  --evidence-artifact train399_evaluation=<evaluation.json> `
  --evidence-artifact root_exact283=<root-audit.json> `
  --evidence-artifact stageb=<stageb-summary.json> `
  --evidence-artifact fusion=<fusion-summary.json> `
  --evidence-artifact traits=<traits-summary.json> `
  --evidence-artifact cohorts=<cohorts-summary.json> `
  --evidence-artifact analysis=<analysis-summary.json> `
  --evidence-artifact profiles=<profiles-summary.json> `
  --evidence-artifact figure_inputs=<figure-inputs.json> `
  --evidence-artifact figures=<figure-suite-summary.json> `
  --figure-inputs <figure-inputs.json> `
  --figure-assembly-summary <assembly-summary.json> `
  --model-contract-proposal <proposal.json> `
  --human-metadata <author-verified-human-metadata.json> `
  --model-bundle-manifest <immutable-model-bundle-manifest.json> `
  --clean-install-receipt <clean-install-receipt.json> `
  --output <manuscript-values.json>
```

The current repository intentionally contains no invented final values and does not generate a final compiled manuscript before all named final inputs exist.
