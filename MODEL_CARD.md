# PHAxis 1.0.0 model card

## Model description

PHAxis 1.0.0 combines two locked capabilities:

1. a continuity-aware primary-root geometry ensemble that supplies the root
   body, ordered axis, distal/root-cap point, width reference, and physical
   calibration; and
2. a five-member root-hair identity/count expert that detects one biological
   root-hair identity per accepted instance.

Fusion attaches identities to the ordered root axis and, when available,
links each identity to at most one endpoint-complete curve for conditional
projected-length measurement. Public model and root-provider identities are
derived from exact asset and receipt hashes rather than informal model names.

## Intended use

PHAxis measures visible primary-root morphology, root-hair abundance, axial
deployment, and conditional projected length in calibrated *Arabidopsis
thaliana* microscopy images. It is research software, not a medical,
diagnostic, agronomic decision, or safety-critical system.

## Output semantics

- PHAxis exposes 32 canonical image-derived descriptors: 19 visible-primary-
  root descriptors and 13 root-hair descriptors. The canonical 82-column
  export also carries identity, geometry, observability, QC, reason-code, and
  provenance fields; it does not report 82 phenotypes.
- Root-hair count uses tolerant one-to-one biological presence in physical
  units. Endpoint coincidence and strict whole-curve overlap are secondary
  geometric diagnostics.
- Length is conditional: it is reported only for a detected identity linked
  one-to-one to an endpoint-complete curve. An identity vector is not itself a
  length curve.
- The root-cap output is exactly one distal/root-cap coordinate point. PHAxis
  does not segment a root-cap region or report root-cap-region statistics.
- Root continuity is evaluated on the final fused root foreground. Evaluation
  does not bridge disconnected components.
- Unobservable or ineligible measurements remain null with reason codes; they
  are not converted to biological zero.

The exact bilingual phenotype definitions are in
`docs/phaxis/TRAIT_CONTRACT_CN.md`.

## Training and development evidence

- Five expert members are trained only on the locked 399-image training
  partition.
- A separate 44-image, family-isolated, same-domain development partition is
  used for operating-point selection and development evaluation. It is not an
  independent external test.
- Historical same-domain out-of-fold results are algorithm-development
  evidence, not external accuracy.
- The application collection has no dense root-hair accuracy truth; its
  biological analyses are exploratory.
- Blind/final-validation images and outcomes are excluded from training,
  documentation construction, and release preparation.

The five-member expert uses float16 mixed-precision training under a sealed
finite-gradient policy. A numerical overflow may replay backward on the same
retained forward graph after loss-scale backoff; it may not repeat data
sampling, augmentation, forward execution, BatchNorm updates, or optimizer
updates. Exhaustion fails closed. This policy changes neither the model,
sampler, loss, nor fixed training horizon. Blind images used: 0.

## Limitations

Current evidence does not establish external accuracy across laboratories,
species, organs, developmental stages, microscopes, or uncontrolled
acquisition conditions. Dense overlap, weak contrast, unusual root geometry,
cropped roots, scale failure, and incomplete endpoint visibility can reduce
observability. Root-provider exact-equivalence evidence demonstrates portable
artifact reproduction, not segmentation ground-truth accuracy.

## Reproducibility and redistribution

Production predictions must bind the official model contract, five checkpoint
hashes, selection receipt, root-provider identity, calibration, source-image
hashes, and software version. Model weights are separate from the Apache-2.0
source distribution and require independent rights authorization.

No final benchmark should be inferred from this card. Release-specific values
belong in hash-bound benchmark receipts produced after the formal release gate.
