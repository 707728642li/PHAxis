# PHAxis manuscript reference audit — 2026-08-28

> **Historical reference audit.** The `Database/Software Article` label and SPJ-era author rules in this file were superseded on 2026-08-30 by the current Elsevier `Software and Hardware Article` Guide for Authors. Use `docs/phaxis/PHAXIS_CURRENT_GFA_MIGRATION_LOCK_20260830.md` for the live submission contract; retain this file only for reference provenance.

## Scope and rule

This audit checks the recent plant-phenotyping and root-hair references used to
frame the PHAxis manuscript.  Verification used the version-of-record journal
page (or the publisher's current issue page) and the DOI.  Search snippets,
secondary summaries, and software marketing pages are not treated as
bibliographic authority.  Citation of a comparator does not imply that its
software, weights, or data were executed or copied by PHAxis.

## Verified recent references

| Topic | Version-of-record citation | DOI/status | Manuscript role |
|---|---|---|---|
| Heat–RHD6 biology | Du G, et al. *Root Hair Development Is Suppressed by Long-Term Mild Heat Through Down-Regulation of RHD6 and RHD6-like Genes*. Plant, Cell & Environment (2025). | [10.1111/pce.15563](https://doi.org/10.1111/pce.15563), verified | Direct biological motivation for the RHD6 × temperature use case. |
| Individual-hair combinatorial tracing | Pietrzyk P, et al. *DIRT/µ: automated extraction of root hair traits using combinatorial optimization*. Journal of Experimental Botany 76:285–298 (2025; online 2024). | [10.1093/jxb/erae385](https://doi.org/10.1093/jxb/erae385), verified | Recent individual-hair phenotype comparator and motivation for topology-aware tracing. |
| ML root-hair software | Tsang I, et al. *pyRootHair: Machine learning accelerated software for high-throughput phenotyping of plant root hair traits*. GigaScience 15 (2026). | [10.1093/gigascience/giaf141](https://doi.org/10.1093/gigascience/giaf141), verified | Literature context only; PHAxis did not call, copy, train from, or infer with pyRootHair. |
| Rhizotron root-hair area | Themistokleous G, et al. *RootHairAreaFinder: an image processing method for quantifying barley root growth and root hairs simultaneously in a flat rhizotron system*. Plant Methods (2026). | [10.1186/s13007-026-01564-z](https://doi.org/10.1186/s13007-026-01564-z), verified | Contrasts projected hair-area phenotyping with PHAxis individual identity and conditional length. |
| Pose-based root phenotyping | Berrigan EM, et al. *Fast and Efficient Root Phenotyping via Pose Estimation*. Plant Phenomics 6:0175 (2024). | [10.34133/plantphenomics.0175](https://doi.org/10.34133/plantphenomics.0175), verified | High-level precedent for topology/landmark representations and plant-facing traits. |
| Integrated root toolbox | Shi J, et al. *RPT: An integrated root phenotyping toolbox for segmenting and quantifying root system architecture*. Plant Biotechnology Journal 23:2095–2109 (2025). | [10.1111/pbi.70040](https://doi.org/10.1111/pbi.70040), verified | Recent integrated root segmentation and trait-toolbox context. |
| Root-hair longitudinal sizing | Guichard M, et al. *Root Hair Sizer: an algorithm for high throughput recovery of different root hair and root developmental parameters*. Plant Methods 15:104 (2019). | [10.1186/s13007-019-0483-z](https://doi.org/10.1186/s13007-019-0483-z), verified | Precedent for spatially resolved root-hair development along the axis. |
| ML reporting | Walsh I, et al. *DOME: recommendations for supervised machine learning validation in biology*. Nature Methods 18:1122–1127 (2021). | [10.1038/s41592-021-01205-4](https://doi.org/10.1038/s41592-021-01205-4), verified; publisher correction exists | Reporting and evidence-role framework, not a biological comparator. |

## Corrections applied to the master draft

- Replaced the inaccurate shorthand author/title for the Plant Phenomics
  pose-estimation paper with Berrigan *et al.* and the version-of-record title.
- Corrected `Shi R` to `Shi J` and expanded the RPT title to the
  version-of-record title.
- Expanded RootHairAreaFinder and Root Hair Sizer to their official titles.
- Replaced the generic Stetter title with the exact PLOS ONE title.

## Remaining submission-time checks

- Resolve complete author lists, volumes, issues, page/article numbers, and
  journal-required punctuation through the reference manager immediately
  before submission.
- Replace the PHAxis release DOI placeholder only after creation of the
  immutable public archive; do not invent a DOI or repository URL.
- Re-run the DOI resolver audit at final manuscript freeze because 2026
  articles may acquire final volume/page metadata after online publication.

## 2026-08-29 Plant Phenomics positioning update

A second primary-source pass checked the current *Plant Phenomics* and root-platform landscape against PREPs, IHUP, RootXplorer, ChronoRoot 2.0, and Leaf Analyzer. The 2026-08-31 formal 20-reference list contains the four most directly relevant root-hair measurement papers (Root Hair Sizer, DIRT/µ, pyRootHair, and RootHairAreaFinder), Berrigan *et al.* for structured root topology linked to validated traits, Shoaib *et al.* for interpretable algorithm-derived root traits linked to a biological task, DOME, clDice, core biology, experimental-unit statistics, and the PHAxis release slot. U-Net and ResNet remain implementation descriptors with code/model provenance but no longer occupy main-text reference slots. The two same-journal papers were selected for direct scientific relevance, not recency or journal self-citation, and no external performance result is transferred to PHAxis.

The additional verified sources remain documented in `PHAXIS_PLANT_PHENOMICS_LITERATURE_POSITIONING_AUDIT_20260828.md` and informed manuscript structure rather than unsupported performance comparisons. Their common evidentiary pattern—plant problem, explicit measured quantity, layered trait validity, biological use case, and evaluable release—now defines the PHAxis narrative order. RootXplorer supplies a strong biological-platform example; ChronoRoot 2.0 supplies a recent versioned root-platform example; and Leaf Analyzer supplies a same-journal measurement-accuracy/use-case example. PHAxis does not claim their cross-species, temporal, three-dimensional, or external-tool performance scope.

One metadata caution remains outside the current 20-reference main list: Leaf Analyzer is the 2026 issue record `8(1):100145` (published online in 2025), DOI `10.1016/j.plaphe.2025.100145`. If it is substituted into the final list, the reference manager must preserve the version-of-record issue year rather than manually choosing between online and issue years. RootHairAreaFinder remains an accepted early-access version with the verified online date and DOI but without final volume/article metadata at this audit date.

## 2026-08-29 Elsevier-era refresh

The current publisher record and PubMed entry were checked for Di R, Gao P, Li C, Ruan S, Tan F, Yan W, Liang Z, Liu J, Zhang C, and Xu W, *PlantSpecLab: A comprehensive open-source platform for high-throughput plant spectral data processing and phenotypic modeling*, *Plant Phenomics* 8(1):100148 (2026), DOI `10.1016/j.plaphe.2025.100148`. Its integrated workflow, module validation, biological prediction cases, end-to-end time comparison, and versioned source availability make it the closest current editorial analogue for PHAxis. It remains an editorial-structure source rather than a performance comparator; its hyperspectral tasks and numerical results are not transferable to PHAxis.

The 2026 issue record `8(1):100146`, DOI `10.1016/j.plaphe.2025.100146`, was also checked for *Root segmentation beyond species boundaries: A generalizable framework for anatomical analysis*. It supports a staged, domain-aware measurement architecture but does not establish PHAxis cross-species generalization. RootXplorer (`10.1016/j.plaphe.2025.100143`) and GrowScreen-Rhizo 3 (`10.1016/j.plaphe.2026.100213`) reinforce the journal's current preference for system capability, interpretable root traits, biological use, throughput, and accessible workflows in one paper.

No main-list replacement was made in this pass because the 20 current references remain nonredundant and the Database/Software Article limit is already full. PlantSpecLab is the first replacement candidate if a weaker software precedent is removed during final reference-manager reconciliation. The official journal author guide was reread on 2026-08-29 and still specifies a 250-word abstract, no more than 15,000 words, up to ten combined figures/tables, up to 20 references, evaluable software, and one or a few use cases for this article category.

## 2026-08-29 biology-forward editorial verification

The final biology-forward edit rechecked the claims actually used to organize the manuscript against primary pages for Du et al. (Wiley, DOI `10.1111/pce.15563`), Berrigan et al. (*Plant Phenomics*, DOI `10.34133/plantphenomics.0175`), Shoaib et al. (*Plant Phenomics*, DOI `10.1016/j.plaphe.2025.100088`), DIRT/µ (OUP, DOI `10.1093/jxb/erae385`), pyRootHair (OUP, DOI `10.1093/gigascience/giaf141`), ChronoRoot 2.0 (OUP, DOI `10.1093/gigascience/giag018`), PlantSpecLab (Elsevier, DOI `10.1016/j.plaphe.2025.100148`), RootXplorer (publisher-deposited full text, DOI `10.1016/j.plaphe.2025.100143`), and GrowScreen-Rhizo 3 (Elsevier, DOI `10.1016/j.plaphe.2026.100213`). The common structure supported by these sources is a plant problem followed by an explicit measurement representation, trait-level validity, one or more biological use cases, observed workflow efficiency, and evaluable software. It does not support transferring any external paper's accuracy, cross-species, 3D, temporal, causal, or stress-classification scope to PHAxis.

No main-reference slot changed. The current registry remains the bibliographic authority for references 1–20, and the first-citation order remains strict. The new sources serve as editorial analogues in `PHAXIS_PLANT_PHENOMICS_LITERATURE_POSITIONING_AUDIT_20260828.md`; adding any of them to the manuscript would require an explicit 20-slot replacement, registry update, DOI/title revalidation, and a fresh reference-contract test.

## 2026-08-29 full-registry DOI reconciliation

All 19 externally registered DOI records were reread from Crossref and compared with the numbered manuscript and `manuscript_reference_registry.json`; the PHAxis release remains the intentionally unresolved twentieth authority slot. Titles, DOI identities, containers, first-citation order, and complete author lists remain aligned. Online-first versus issue-year differences for DIRT/µ (online 2024; volume 76 in 2025), pyRootHair (online 2025; volume 15 in 2026), and RootHairAreaFinder (2026 early access) remain represented by their version-of-record issue or early-access citation rather than by the Crossref deposit year alone. One proceedings pagination discrepancy was corrected: the IEEE DOI registration for clDice and the author-institution publication record give pages 16555–16564, whereas the CVF open-access repository labels its parallel copy 16560–16569. The manuscript now follows the DOI-registered IEEE proceedings pagination, with DOI `10.1109/CVPR46437.2021.01629` unchanged.
