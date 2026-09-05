# CLI reference

Generated from the actual PHAxis 1.0.0 candidate parser.

## phaxis

```text
usage: phaxis [-h] [--version]
              {demo,report,analyze,fuse,verify-prediction,infer-hairs,export-traits}
              ...

PHAxis 1.0.0: primary-root and root-hair phenotyping

positional arguments:
  {demo,report,analyze,fuse,verify-prediction,infer-hairs,export-traits}
    demo                run CPU-only synthetic fusion/trait numerical example
    report              create an offline report from existing trait tables
    analyze             plan the locked raw-image -> roots -> root-hair ->
                        traits workflow; execution requires --execute
    fuse                combine locked PHAxis root-provider outputs with root-
                        hair detections
    verify-prediction   verify root locks and provenance
    infer-hairs         run the locked PHAxis root-hair identity/count expert
                        on one image
    export-traits       export canonical 32-trait and per-hair tables

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

## analyze

```text
usage: phaxis analyze [-h] --manifest MANIFEST --output OUTPUT
                      [--plan-output PLAN_OUTPUT] [--review-overlays]
                      [--execute] [--resume]

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST   sealed PHAxis analysis-workflow manifest
  --output OUTPUT       new analysis directory (or the bound directory when
                        resuming)
  --plan-output PLAN_OUTPUT
                        optional new JSON file receiving the printed
                        plan/state
  --review-overlays     enable optional review-only overlays (never used for
                        routing)
  --execute             execute the plan; omission is strictly plan-only
  --resume              resume a verified interrupted run; valid only with
                        --execute
```

## fuse

```text
usage: phaxis fuse [-h] --root-predictions ROOT_PREDICTIONS --root-artifacts
                   ROOT_ARTIFACTS --hair-detections HAIR_DETECTIONS
                   --model-contract CONTRACT --output OUTPUT
                   [--attachment-tolerance-um ATTACHMENT_TOLERANCE_UM]
                   [--physical-scale-contract {strict_root_provider,stageb_reference_evaluation}]
                   [--task-id TASK_ID]

options:
  -h, --help            show this help message and exit
  --root-predictions ROOT_PREDICTIONS
                        directory of locked PHAxis root-provider prediction
                        JSON files
  --root-artifacts ROOT_ARTIFACTS
                        root-provider artifact directory referenced by the
                        predictions
  --hair-detections HAIR_DETECTIONS
                        directory of locked PHAxis root-hair detection JSON
                        files
  --model-contract CONTRACT
                        sealed PHAxis model-contract proposal or applied
                        official contract
  --output OUTPUT       new fused-output directory
  --attachment-tolerance-um ATTACHMENT_TOLERANCE_UM
  --physical-scale-contract {strict_root_provider,stageb_reference_evaluation}
                        strict deployment scale agreement (default), or the
                        locked Stage-B acquisition scale for QC-development
                        geometry evaluation
  --task-id TASK_ID
```

## infer-hairs

```text
usage: phaxis infer-hairs [-h] --image IMAGE --task-id TASK_ID
                          --source-um-per-px SOURCE_UM_PER_PX --checkpoint
                          CHECKPOINT [--model-contract CONTRACT]
                          [--candidate-manifest CANDIDATE_MANIFEST]
                          [--selected-model-metadata SELECTED_MODEL_METADATA]
                          [--selection-receipt SELECTION_RECEIPT]
                          [--device DEVICE] --output OUTPUT

options:
  -h, --help            show this help message and exit
  --image IMAGE
  --task-id TASK_ID
  --source-um-per-px SOURCE_UM_PER_PX
  --checkpoint CHECKPOINT
                        one selected ensemble checkpoint; repeat exactly five
                        times
  --model-contract CONTRACT
                        sealed PHAxis 1.0.0 model-contract authority
                        (unapplied proposal or applied official contract)
  --candidate-manifest CANDIDATE_MANIFEST
                        strict train399 candidate Gate receipt
  --selected-model-metadata SELECTED_MODEL_METADATA
                        train399 metadata bound to the selected QC-development
                        operating point
  --selection-receipt SELECTION_RECEIPT
                        selection receipt whose file and logical identities
                        are rechecked
  --device DEVICE
  --output OUTPUT
```

## export-traits

```text
usage: phaxis export-traits [-h] --predictions PREDICTIONS --metadata METADATA
                            --model-contract CONTRACT --output OUTPUT

options:
  -h, --help            show this help message and exit
  --predictions PREDICTIONS
                        directory of fused PHAxis prediction JSON files
  --metadata METADATA   CSV containing task IDs and biological sample metadata
  --model-contract CONTRACT
                        sealed PHAxis model-contract proposal or applied
                        official contract
  --output OUTPUT       new trait-export directory
```

## verify-prediction

```text
usage: phaxis verify-prediction [-h] --prediction PREDICTION --artifact-root
                                ARTIFACT_ROOT

options:
  -h, --help            show this help message and exit
  --prediction PREDICTION
  --artifact-root ARTIFACT_ROOT
```

## demo

```text
usage: phaxis demo [-h] --output OUTPUT

options:
  -h, --help       show this help message and exit
  --output OUTPUT  new demo directory
```

## report

```text
usage: phaxis report [-h] --traits TRAITS --output OUTPUT

options:
  -h, --help       show this help message and exit
  --traits TRAITS  directory containing image_traits.csv
  --output OUTPUT  new report directory
```
