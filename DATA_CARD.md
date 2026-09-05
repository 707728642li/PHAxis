# PHAxis 1.0.0 data card

## Scope

PHAxis 1.0.0 is scoped to calibrated microscopy of *Arabidopsis thaliana*
primary roots and root hairs. External accuracy has not been established for
other species, organs, laboratories, microscopes, or uncontrolled acquisition
conditions.

## Data roles and separation

| Data layer | Permitted role | May tune the model? | Included in source/PyPI? |
|---|---|---:|---:|
| Project-generated synthetic geometry | unit tests, schema checks, engineering regression | only when declared development | audited fixtures only |
| 399-image private training partition | model fitting under rights controls | yes | no |
| 44-image family-isolated development partition | operating-point selection and development evaluation | only within the locked protocol | no |
| 443-image same-domain development corpus | historical algorithm characterization | historical development only | no |
| 261-image SHA-disjoint application subset | primary exploratory biological analysis | no | no |
| Full 283-image application collection | overlap-sensitivity analysis | no | no |
| Future external/blind/final-validation data | independent evaluation only | no | no |

The 44-image development partition and historical 443-image results must not be
reported as independent accuracy. Application images do not provide dense
root-hair accuracy truth. Experimental condition, genotype, and temperature
metadata are joined after inference and must not route or alter predictions.

## Annotation and measurement semantics

The plant-facing catalogue in `docs/phaxis/TRAIT_CONTRACT_CN.md` defines 32
canonical descriptors (19 visible-primary-root and 13 root-hair descriptors),
including observability, null, partial, lower-bound, and censoring semantics.
The canonical 82-column export is a measurement-and-provenance schema; it does
not report 82 phenotypes.

Root-hair centreline annotations represent a single visible centreline per
hair, not a dense biological-width mask. Root-hair identity/count evaluation
therefore uses tolerant one-to-one presence in physical units. Conditional
length requires an accepted identity to link one-to-one to an endpoint-complete
curve. The root-cap representation is one distal/root-cap point; no root-cap
region mask, area, or region statistic is produced.

Synthetic examples validate installation and data contracts, not biological
accuracy. Model-assisted annotations are not automatically independent human
truth and must retain their provenance.

## Leakage and release controls

Training and development splits are source- and family-aware. Development,
blind, application labels, biological metadata, hashes, and outcomes may not
enter fitting, undeclared threshold selection, inference routing, or
result-driven exclusion. Release candidates bind exact dataset, split, model,
and evaluation-code identities.

Private images and annotations are excluded from source packages. Model and
data redistribution require asset-specific authorization and attribution. This
card documents engineering controls, not legal advice.
