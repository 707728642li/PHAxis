# PHAxis 1.0.0 六幅投稿主图与证据输入合同

## 1. 唯一生产路径

PHAxis 1.0.0 的投稿主图采用两级、均为 fail-closed 的生产路径：

1. `scripts/phaxis/build_publication_figure_inputs.py` 从显式命名的权威回执和其哈希声明的源表重新计算绘图单元格，原子生成 `PHAxis-manuscript-figure-inputs-2.0`；
2. `scripts/phaxis/build_publication_figures.py` 只消费该装配结果、同一 model-contract proposal 和同一八个核心回执，生成六幅主图及 `PHAxis-publication-figure-suite-1.0` 回执。

两级程序均不搜索“最新”目录、不读取 blind 数据、不运行模型或 GPU、不修改训练/权重/冻结 v1，也不输出根冠区域统计。任一源文件、逻辑身份、单元格重算或跨回执绑定不一致，`final` 模式即停止且不留下可误认的最终目录。`provisional` 只供布局测试：文件名前缀为 `PROVISIONAL_`，画布带水印，且 `submission_use_allowed=false`。

六幅图编号和内容固定为：

1. `Figure_01_biological_measurement_design`：生物测量对象、distal-axis 坐标和 32 trait 家族；
2. `Figure_02_train399_development_evidence`：train399 / QC-development44 的 tolerant biological-hair presence 证据；
3. `Figure_03_measurement_assurance`：root、distal point、scale、conditional length 与拓扑/可观测性保证；
4. `Figure_04_difficult_image_interpretability`：代表、低对比、弯曲致密、连续性和 fail-closed overlay；
5. `Figure_05_exploratory_phenotype_atlas`：clean261-primary / full283-sensitivity 的 RHD6×temperature 表型及 0–5 mm profiles；
6. `Figure_06_reproducibility_and_efficiency`：cohort integrity、哈希工作流和同硬件真实端到端效率。

每幅图输出 PDF、600 dpi PNG 和 300 dpi RGB/LZW TIFF。图源、legend、alt text、逐文件 SHA-256 与逻辑身份随图发布。

## 2. 八个核心权威回执

下表规定最终发布回执必须满足的条件，不声明这些回执或 final283 生产运行在
当前工作树中已经完成。未产生并密封的回执不得用历史、provisional 或手填数字
代替。

装配器和绘图器都必须逐一命名以下八个 JSON；不存在默认路径：

| 角色 | final Gate |
|---|---|
| `train399_evaluation` | evaluator schema 1.2；399 train / 44 QC-development；QC 标签不参与梯度或 early stopping；不允许独立准确率声明 |
| `root_exact283` | 必须提供 fresh portable root-provider 回执，并证明三个 root 层在锁定 283 图参考集逐图 exact；bundle、pipeline、audit 身份齐全 |
| `stageb` | 必须绑定 proposal 的 train399 five-seed expert，并提供密封的 final283 Stage-B 生产回执；本合同不预填完成状态或数值 |
| `fusion` | 与所给 Stage-B 文件 SHA、hair expert、proposal model/root ID 一致 |
| `traits` | 同一 model/root/hair expert；必须由 final283 生产回执证明 canonical image-row 闭合 |
| `cohorts` | full283、SHA-disjoint clean261 与 22 个 byte-overlap 闭合 |
| `analysis` | clean261 primary、full283 sensitivity；exploratory association scope |
| `profiles` | clean261、[0,5) mm；必须由最终回执重算并证明 trait crosscheck，无预填完成数 |

`source_summary_sha256` 必须精确且仅含这八个键，值是命令行所指回执文件的 bytes SHA-256。它不等于回执内部逻辑身份，也不允许用目录哈希代替。

## 3. Proposal 与公开模型身份

proposal 必须满足：

- `schema_version=PHAxis-model-contract-1.0.0`；
- `formal_release_status=passed_proposal_not_official`；
- `promotion.status=validated_proposal_not_applied`；
- `promotion.official_apply_performed=false`；
- `model_contract_identity_sha256` 等于移除该字段后的 canonical JSON SHA-256。

公开模型 ID 不能由手写前缀或运行时 GPU/并发配置决定。所有 consumer 都调用同一 canonical derivation：Stage-B 的 candidate/selection/selected-metadata 三个逻辑身份与稳定 `root_provider_bundle_identity_sha256` 共同派生 `model_bundle_id`；root expert ID 单独由该稳定 bundle 身份派生。公开 bundle 前缀固定为 `PHAXIS-V1.0.0-STRICT-TRAIN399-`。`root_expert.provider_role=PHAxis-portable-root-provider` 只是角色，不得冒充实例化 `root_expert.expert_id`。

proposal 的 `root_expert.bundle_identity_sha256`、`pipeline_identity_sha256`、`fresh_exact283_audit_identity_sha256` 必须逐字段等于本次命名的 `root_exact283` 回执；`root_bundle_authority` 同时绑定 bundle 和 pipeline。pipeline/audit 用于运行验证，不进入公开 ID 预映像。

## 4. 生产级 figure-input 装配

最终 `figure_inputs.json` 固定为：

- `schema_version=PHAxis-manuscript-figure-inputs-2.0`；
- `assembler_schema_version=PHAxis-publication-figure-input-assembly-1.0`；
- `status=final`；
- 八键 `source_summary_sha256`；
- proposal 文件 SHA、proposal 逻辑身份和 canonical public identity；
- train399 Stage-B 与 legacy Hybrid QC44 的 ordered file locks/set identities；
- 恰好 19 个 `resources`；
- 显式 `source_inputs`、`provenance_receipts` 和逐资源 `resource_lineage`；
- `blind_images_used=0`、`root_cap_region_statistics_included=false`；
- `figure_input_assembly_identity_sha256=sha256_json(移除该字段后的完整对象)`。

装配器不得接受已经归一化的手填绘图数值作为权威。19 个绘图资源如下：

| resource | 派生规则 |
|---|---|
| `trait_contract` | hash-locked 32 traits：19 root + 13 hair；root-cap region fields=0 |
| `figure1_image` | full283 中的非 blind 原始图 |
| `figure1_geometry` | source-image 和 final prediction 双哈希绑定的 root polygon、axis、distal point、identity/length geometry |
| `development_per_image` | evaluator1.2 的 44×2 comparator sufficient statistics，保留每个 prediction 文件 SHA 与 ordered set identity |
| `development_tolerance` | 从逐图 TP/n_pred/n_gt 重算 5/10/20 µm P/R/F1 与 paired image-level nonparametric bootstrap CI |
| `development_threshold` | 从 selection receipt 复制完整 biological-presence candidate grid（主 20 µm F1、次级 count MAE/绝对 bias、attachment 定位敏感性），并核唯一 selected threshold |
| `development_strata` | 从 sealed historical OOF443 per-image statistics 重算 quality/density/annotation strata |
| `assurance_metrics` | 从 measurement-assurance pairs/support/topology 逐单元格重算或核对 |
| `assurance_pairs` | scale、conditional-length、19 root-trait agreement 的 source-unit/pair sufficient statistics；root-continuity 与 formally matched attachment 的逐图充分统计由 measurement-assurance receipt 内嵌子回执封存 |
| `assurance_support` | 四个 application condition 的 identity、endpoint-complete support 和 source-unit n |
| `overlay_selection` | producer 重渲染的五类预选形态/采集挑战 source/overlay pairs；实验条件字段不参与渲染或证据组装 |
| `phenotype_points` | 从 clean261 trait rows 派生五个 plant-facing endpoint 原始点 |
| `phenotype_effects` | 从 analysis receipt 哈希声明的 primary/sensitivity tables 提取固定 15-effect family |
| `multitrait_atlas` | 从 canonical trait contract、clean261/full283 cohort membership/hair trait 表、traits receipt 绑定的 exact283 canonical `image_traits.csv`（19 root 字段）和固定分析表重算全部 R01--R19/H01--H13；按 `task_id` 左连接时逐行核 `source_image_sha256`，cohort 已含的 root 值须与 canonical 值/null exact 一致，13 hair 字段不得由 canonical 表覆盖或填补；每项含 `measurement_family`、unit、clean/full 总体支持，并在四个固定 D15 condition 中逐项报告原始未调整 median/Q25/Q75/IQR、非空分子/source-unit 分母与 observability，以及 clean/full × 三种 contrast 的“已估计或明确未估计”记录；公开 schema 固定为 `PHAxis-multitrait-atlas-2.0`，家族顺序键为 `measurement_family_order` |
| `axial_profiles` | 从 profile-analysis table 派生 clean261 四组×五 bin×三指标 |
| `cohort_flow` | 从 evaluator/cohort receipt 重新闭合 443→399/44 与 283→261/22/formal/review |
| `workflow_stages` | 八回执的固定顺序和各自逻辑身份 |
| `runtime_summary` | A/B 两种 direct benchmark、frozen-v1 对照及比较回执 |
| `runtime_per_image` | mode A 的 283 条真实逐图 latency trace；不得由 batch wall 均摊 |

`source_inputs` 必须恰好包含 33 个角色。除原有 19 个角色（`split_manifest`、`historical_oof_per_image`、`assurance_metrics`、`assurance_pairs`、`assurance_support`、`assurance_topology`、`clean_traits`、`full_traits`、`full_image_traits`、`analysis_primary_table`、`analysis_sensitivity_table`、`profile_analysis_table`、`sensitivity_profiles_summary`、`runtime_latency`、`runtime_production`、`runtime_per_image`、`baseline_runtime_latency`、`baseline_runtime_production`、`baseline_runtime_per_image`）外，还必须包含 `dataset_manifest`、`image_traits_schema`、`train399_candidate`、`train399_selection`、`model_contract_proposal`、五个 `training_receipt_seed_2026082801`–`2026082805`、`benchmark_same_hardware`、`benchmark_artifact_inventory`、`runtime_latency_comparison` 与 `runtime_production_comparison`。新增角色为 S1–S10 的 reviewer-facing source-authority closure 服务，不改变科学输入。

`provenance_receipts` 必须恰好包含 10 个已封存回执：`historical_development`、`measurement_assurance`、`overlay_index`、`profile_analysis`、PHAxis 的 latency/production/comparison 两对，以及 frozen-v1 的 latency/production 两个 direct summaries。每一项同时保存文件 path、bytes SHA、identity field 和 logical identity；所有 lineage 只能引用八个核心角色、19 个 source inputs 或这 10 个 provenance roles。

## 5. 三类辅助证据 producer

三个辅助回执必须由生产脚本从原始权威输入生成，不允许人工编写汇总 CSV 后自报 `evidence_role`：

| producer | 输出与范围 |
|---|---|
| `build_historical_oof443_publication_evidence.py` | 从显式可信本地 OOF pickle、dataset manifest 与 split manifest 重算 443 张逐图 biological-presence sufficient statistics；仅 historical algorithm-development evidence |
| `build_measurement_assurance_evidence.py` | 从 canonical QC-development vector、root/scale providers、Stage-B/fusion predictions、clean261 application predictions 重算 assurance metrics/pairs/support/topology；明确 non-independent QC-development 与 non-accuracy observability scope |
| `build_condition_blinded_overlay_evidence.py` | 为兼容既有命令保留的入口文件名；五行 plan 只含 `case_role/task_id`，程序从 full283 manifest、traits、fusion prediction 重新渲染并哈希 source/overlay；输出合同不声明选择阶段盲法 |

measurement-assurance 的四个表必须由 receipt 的 `source_table_sha256.{metrics,pairs,support,topology}` 封存；root-continuity 与 formally matched attachment 子回执还必须各自验证 schema、逻辑 identity、逐图分母和 source-image bootstrap 合同。核心单元格至少包括 root Dice/Boundary F1/HD95、single-component axis coverage/best-component gap/break-free rate/visible-axis extent error、distal error/PCK、scale detection coverage/line-endpoint localization/conditional calibration/applicability、formal attachment-qualified precision/recall/F1 与 attachment error、conditional-length MAE/bias/CCC、matched endpoint error、trajectory continuity、root-trait agreement、endpoint-complete support、axis containment、unsupported attachment 和 provider exact fraction。CCC 统一采用样本协方差/样本方差（`ddof=1`）：

```text
CCC = 2 cov_sample(x,y) / [var_sample(x) + var_sample(y) + (mean(x)-mean(y))^2]
```

## 6. 根毛主指标语义

Fig.2 的正式主指标是 one-to-one tolerant biological hair presence。预测二点线段与人工单中心线在 5/10/20 µm 下匹配；truth 与 prediction 均至少 25% arc coverage，近端方向 cosine ≥0；distal endpoint coincidence 和完整全线重合都不是 presence 的硬门。

Operating-point selection 直接优化与正式评估相同的 20 µm tolerant biological-presence F1：同一次网络前向在固定 0.10 base-score floor 下物化不可变的 base→tip 直线代理池，阈值扫描只过滤该池。主 matcher 使用双向 partial-centreline coverage、非反向近端方向和逐图 Hungarian 一对一分配；人工中心线不被当作有宽度掩膜，distal endpoint、完整曲线重合与 length error 均不是身份 Gate。主 F1 相同后依次用 count MAE、absolute signed bias 和更高的预声明阈值确定唯一 operating point。正式 attachment localization 必须复用主结果的一对一身份匹配，不允许 base-only rematch；formal attachment-qualified accuracy、strict whole-line、endpoint error 和 conditional length 仍是独立层级，不得互相推断。

Root continuity 以 trait extraction 实际接收的 sealed final fused root mask 为预测权威。评价器逐 connected component 骨架化且不自行桥接；正式 `break_free` 只在同一单分量支持整条 reference axis 时成立，多个断片的 union coverage 不得冒充连续性。Scale applicability 固定闭合为 QC-development44 的 38 个 visible-bar、6 个 trusted-metadata 和 0 个 absent/untrusted cases；因此经验 absence specificity 必须写为不可估，软件 fail-closed 负例证据另列。

QC-development44 的区间和 Stage-B-minus-legacy 差值均为 paired image-level nonparametric bootstrap，固定 10,000 次、seed 20260828。family isolation 只用于 split，不是 cluster bootstrap。QC-development44 与 OOF443 都是开发证据，不得包装成独立准确率。

## 7. Profile 与生物学统计

`full_image_traits` 必须是 traits receipt 通过 `image_traits_sha256` 绑定的 exact283 canonical 32-column table，用于逐 trait 重新计算 coverage。clean/full cohort trait tables分别由 cohort receipt 绑定。

profile-analysis receipt、clean261 primary profile summary 和 full283 sensitivity profile summary 必须同时一致地绑定四个身份字段：proposal file SHA、proposal logical identity、`model_bundle_id`、`root_expert_id`。profile analysis 同时绑定两个 profile summary 的文件 SHA 与逻辑身份；不可只验证 primary table SHA。

Fig.5 的推断单位是一张 canonical source image/root。clean261 为 primary，full283 仅作 overlap-contaminated sensitivity。不可观测 profile bin 不连线、不补 0；每 bin 同时给 eligible n 与 length-supported n；profile 不增加 hypothesis test。生物学结果表述为 exploratory association，不作因果处理声明。

## 8. Fig.4 / Fig.S6 overlay 合同

五个角色恰好各一例：`representative`、`low_contrast`、`curved_dense`、`continuity`、`fail_closed`。合同将其明确记录为 `case_selection_basis=preselected_morphology_acquisition_challenge_roles` 且 `random_or_representative_performance_sample=false`，不对选择阶段是否查看过实验条件作无法由五行 plan 证明的声明。可验证的边界仅为 `experimental_condition_metadata_used_for_rendering=false` 和 `experimental_condition_metadata_used_for_evidence_assembly=false`；`fail_closed` 必须 `formal_statistics_eligible=false`。同一 case 的 source/overlay 使用相同线性显示与正的物理 scale bar。

| 对象 | 颜色 | 语义 |
|---|---|---|
| root boundary | `#28D7E5` | 主根外轮廓 |
| ordered axis | `#FFFFFF` | distal-oriented 主根轴 |
| distal point | `#F044A5` | 单一 distal/root-cap 点 |
| endpoint-complete curve | `#48C774` | 一对一关联且可用于 conditional length |
| identity vector | `#F0A202` | 根毛身份；不能直接当作完整长度 |

## 9. Fig.6 双模式 direct benchmark

同一系统必须提供两种不混淆的 benchmark：

- A：`sequential_persistent_full283`，或诚实标注的 `sequential_cold_cli_full283`。前者每图 wall 不含 startup，后者必须包含；输出真实逐图 median/P95 latency 及严格七列 per-image CSV；
- B：`production_batch_full283`。输出直接 batch total wall、images/min、MP/s 与 non-overlapping stage decomposition；`per_image_latency_reported=false`，不得把并行 batch wall 或 stage time 硬均摊到 283 行。

PHAxis 与 legacy RHAxis/RhizoWeave v1.0 workflow 必须在同一 283 source/image locks、同一硬件、同一 latency mode、同一 raw-image-to-final-traits-and-profiles I/O scope、无 cache/resume 下才允许显示 speedup。历史 98.47 min 若只覆盖 root/branch 组件，必须标为 `component_only_noncomparable`，不得与 full workflow 直接比较。

## 10. 最终 figure-suite 身份

figure suite 的逻辑身份只由共享 helper `phaxis.publication_evidence.figure_suite_identity_preimage` 定义：

```python
sha256_json({
    "status": "final",
    "figure_hashes": summary["figure_bundle_sha256"],
    "source_hashes": summary["source_summary_sha256"],
    "figure_input_assembly_identity_sha256": summary["figure_input_assembly_identity_sha256"],
    "model_contract_proposal_identity_sha256": summary["model_contract_proposal_identity_sha256"],
    "model_contract_public_identity": summary["model_contract_public_identity"],
    "train399_prediction_input_provenance": summary["train399_prediction_input_provenance"],
})
```

`multitrait_atlas` 同时封存描述性与推断状态，但两者不得混写。描述性部分固定为 32 descriptor × clean/full 两 cohort × EV-22°C、EV-30°C、OE-22°C、OE-30°C 四 condition，共 256 个 condition-summary slots。每个 slot 均以一张 formal D15 source image/root 为单位，报告 `source_unit_total`、`non_null_source_unit_n`、`observability_fraction` 以及原始未调整的 median、Q25、Q75、IQR、minimum 和 maximum；observability 必须闭合为 `non_null_source_unit_n/source_unit_total`，IQR 必须闭合为 Q75−Q25。真实测得的零保留在分布中；零个有限观测的 slot 必须保留空分布统计并给出 `no_finite_observations_in_formal_D15_condition`，不得把不可观测值填作零。

推断状态部分固定为 32 descriptor × clean/full 两 cohort × 三 contrast，共 192 个 effect slots。只有预设五 endpoint 的 clean/full 30 个 slots 可带 estimate/CI/effect scale/endpoint n；其余 162 个必须为 `not_estimated` 并给出 `trait_not_in_prespecified_five_endpoint_15_effect_family`，不得从描述性 condition 中位数差异虚构推断效应，也不得把 `not_estimated` 解释为零效应或无生物学变化。每个 effect denominator 必须与对应四个 condition-summary 的非空 source-unit n 逐项一致并闭合为其和。资源自身以 `atlas_identity_sha256` 封存，并逐字节绑定六个来源：trait contract、clean/full cohort trait 表、canonical exact283 image-trait 表和两张 analysis 表。

六幅主图与三张主表合同不变。补图采用一个 canonical、有序且恰好九项的合同：

| 编号 | 固定 stem | final 重算来源 |
|---|---|---|
| S1 | `Supplementary_Figure_S01_stageb_input_architecture_targets` | 原图、几何/scale、trait contract、Stage-B receipt |
| S2 | `Supplementary_Figure_S02_split_selection_development_strata` | split/cohort flow、逐图开发统计、tolerance、threshold、strata、evaluator receipt |
| S3 | `Supplementary_Figure_S03_identity_attachment_endpoint_assurance` | tolerance、assurance metrics/pairs/support 及 evaluator/fusion/traits receipts |
| S4 | `Supplementary_Figure_S04_primary_root_trait_agreement` | 19 项 root-trait source-unit pairs、trait contract、root/traits receipts |
| S5 | `Supplementary_Figure_S05_provider_tiling_numerical_equivalence` | root exact283 layers、Stage-B execution-path audit、provider/topology metrics |
| S6 | `Supplementary_Figure_S06_expanded_overlay_gallery` | 五角色预选采集挑战 sealed source/overlay pairs、fusion/traits receipts；不作性能抽样声明 |
| S7 | `Supplementary_Figure_S07_biological_sensitivity_observability` | clean/full points/effects、support、profiles 及 cohorts/analysis/profiles receipts |
| S8 | `Supplementary_Figure_S08_runtime_memory_io` | 283 条 direct trace 与两个系统的 sealed batch/latency/memory/utilization summaries |
| S9 | `Supplementary_Figure_S09_multitrait_atlas_coverage_effect_heatmap` | 六来源重算的 32-trait atlas 及 traits/cohorts/analysis receipts |

每项均输出 PDF、600-dpi PNG 和 300-dpi TIFF，并在唯一正式汇总文件 `figure_assembly_summary.json` 依次保存 `S1`--`S9`、resource roles、receipt roles、receipt file SHA 和 source-data SHA。`supplementary_figure_bundle_identity_sha256=sha256_json(有序九项 bundle SHA map)`；figure-input assembly 与 figure suite 必须逐字相等地携带 `PHAxis-publication-supplementary-figure-contract-1.0` 及其逻辑身份，且 `claim_contract.supplementary_figure_count=9`。

`final` 模式从 hash-locked resources/receipts 重算定量单元格；缺字段、分母不闭合、trait 不全、receipt/hash 漂移或运行时 telemetry 缺失即原子失败。`provisional` 模式允许证据尚未完成，但对应补图只能是同时带全局 provisional 水印和 `FINAL EVIDENCE PENDING / No quantitative value has been substituted` 的占位，且 `submission_use_allowed=false`；不得用默认数或历史数伪装 final。

S9 的主视觉必须覆盖全部 32 项描述量：分别显示 clean261 四条件原始中位数的 descriptor 内标准化图、原始 IQR 相对图和逐条件非空支持率；测得零保持可见，不可观测 slot 保持空白并由 coverage 解释。推断面板只显示固定五 endpoint 的 clean/full 15-effect family，不把其余 162 个预设 `not_estimated` slot 画成占主导的灰色区域；完整 192-slot 状态仍保留在 Table S9。Table S9 由同一 atlas 展开为 256 行 condition-summary、192 行 effect-status ledger 与 cohort/provenance ledger。Supplementary bundle 不计入六幅主图数量。

`figure_assembly_summary.json` 另在顶层输出 `model_bundle_id`、`root_expert_id`、`hair_identity_expert_id`、`multitrait_atlas_identity_sha256`、supplementary figure bundle 身份，以及 create-only S1–S10 Table/Data bundle 的 receipt SHA、bundle identity、逐项 identity、exact-file hashes 与 source-authority map。evidence graph 会反向验证 figures → figure-input assembly → 19 resources / 33 source inputs / auxiliary receipts → 八个核心回执及 proposal；`evidence`、supplement compiler、supplementary DOCX 和 artifact QA 会再次验签同一 S1–S10 closure。仅自洽地重封一个手填 manifest 不能通过。

Table/Data bundle 的物理子目录固定为短且有序的 `S01/`–`S10/`，用以在长的
orchestrator run root 下仍满足 Windows `MAX_PATH`。公开的 Table/Data stem 和 title 不因此
缩写：它们保留在逐项 `item_receipt.json` 中，而顶层 receipt 同时锁定短目录、长 stem、
逐文件 SHA 和逻辑身份。

输出目录必须预先不存在。所有 JSON/CSV/图像完成、重读和哈希验证后，staging 目录才会原子改名为最终目录。
