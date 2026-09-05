# Methods and validation

PHAxis 1.0.0 uses a continuity-aware primary-root geometry provider, five fixed train399 root-hair models, a selected score threshold of 0.225, identity-first fusion and one-to-one endpoint-complete length assignment. The distal point defines physical axial coordinates. There are 19 root and 13 hair descriptors, not 82 phenotypes.

Model identity: `1b260e05341ece496e228bd073d48945bbe10832e5f71ed77f0e63c41847b910`.

Frozen same-domain QC-development44 evidence: tolerant biological-presence F1=0.8890625, count MAE=8.681818; the historical comparator F1=0.79640298 and MAE=14.431818 use the same matcher. These data describe development performance, not independent external accuracy. Applied exact283 lineage contains 34,612 accepted identities and 12,343 endpoint-complete lengths. Primary biological cohort=261 source roots; D15 formal comparison=47.

Local packaging does not retrain, rerun biological inference or alter frozen statistics. The software demo tests fusion/traits on synthetic geometry, distinct from the frozen microscopy evidence. See MODEL_CARD.md, DATA_CARD.md and the contract JSON in the repository for provenance. Fresh full-image GPU throughput and full-capsule clean-install are not claimed by this CPU packaging check.
