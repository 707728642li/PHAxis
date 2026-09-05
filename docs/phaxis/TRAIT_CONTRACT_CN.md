# PHAxis 1.0.0 植物表型目录与图像级字段契约 / Plant phenotype catalog and image-level trait contract

## 1. 目的与适用范围

本文件是 PHAxis 1.0.0 在源码树中**唯一人工可读、面向植物学用户的权威表型目录**，同时规定
图像级结果表、字段词典、可观测性和缺失/部分测量/删失语义。交接包顶层的
`PHENOTYPE_CAPABILITIES_CN.md` 必须在构建时从本路径逐字节复制，不能单独维护。
The present document is the **single authoritative, human-readable,
plant-facing phenotype catalogue** in the PHAxis 1.0.0 source tree. It also
defines image-level fields, observability, missingness, partial measurement and
censoring. The handover copy must be byte-identical to this path at build time,
never an independently edited second catalogue.

本目录以锁定的 PHAxis 主根几何专家和五成员根毛身份/计数专家为基础，避免把内部实现名称或
输出列数误写成公开模型版本或表型数量。

- 一行代表一张输入图像中的一条可见主根。
- 正式、非重复的生物学数值描述量固定为 **32 项：19 项主根 + 13 项根毛**；它们不是 32 个
  统计独立的性状。
- 根冠只输出一个 distal/root-cap point，作为轴向距离零点；**不分割、不统计根冠区域**。
- 根毛人工标注和正式输出均按中心线身份解释；不报告根毛宽度、面积或体积。
- 无法观测或不满足测量前提时写 `null`/CSV 空值，不用 0 伪装缺失。
- 处理条件、基因型和温度等实验信息不得参与模型路由；应在推理完成后按 `task_id` 和
  `source_image_sha256` 从独立样本表连接。

机器可读 ID、字段名、中英文名称和单位的最高优先级合同见
`configs/phaxis/v1_0/trait_contract.json`；图像级平面记录的 JSON Schema 见
`configs/phaxis/v1_0/image_traits.schema.json`。若本文件与机器合同发生任何数量、ID、字段、名称或
单位漂移，catalog 回归测试必须失败。Machine-readable IDs, field names, bilingual names and units
remain normative in `trait_contract.json`; the catalogue test rejects any drift.

## 2. 语义图例 / Reporting-semantics legend

- **null / 空值**：结构、尺度或规定的支持条件不可观测或不合格；绝不以 0 代替。
  The measurement is unobservable or ineligible; zero is never used as an
  imputation.
- **measured zero / 已测生物学零**：测量域已完整定义且其中确实没有正式身份，例如合格窗口内
  H08=0。The measurement domain was observed and contains no accepted identity.
- **visible-scope / 可见范围量**：只描述图像中可见主根；触边时不得外推为整株量。
  The value describes the imaged root extent and is not extrapolated beyond the field of view.
- **conditional / 条件长度量**：仅对与正式根毛身份一对一关联的 endpoint-complete 曲线定义。
  Length is conditional on an identity having one endpoint-complete geometry curve.
- **partial / 部分测量**：数值是可测完整曲线子集的和，而非所有身份的总量；必须与支持数、支持率和
  `total_hair_length_is_partial` 一起解释。The numeric total covers only the measured subset.
- **right-censored / 右删失**：视野可能在真实分布结束前终止；当前仅 H13 明确采用该语义。
  The field of view may end before the underlying deployment span ends; H13 is descriptive only.

所有以 µm、mm、µm² 或其组合为单位的量都要求可信物理尺度，否则为 null。根宽是二维投影中的
**表观根体直径**，不是组织学直径；根长是图像内可见测地长度，不是整株总根长。
All physical-unit descriptors require valid calibration. Apparent width is a
two-dimensional image measurement, and visible axis length is not whole-plant total length.

## 3. 19 项非重复主根描述量 / 19 primary-root descriptors

| ID | 中文名称 / English name | 字段 / Field | 单位 / Unit | 计算依据 / Calculation basis | 所需可观测性 / Required observability | 植物学用途 / Plant-biological use | null、部分与删失语义 / Reporting semantics |
|---|---|---|---|---|---|---|---|
| R01 | 可见主根轴长度<br>Visible primary-root axis length | `visible_root_axis_length_um` | µm (`um`) | distal point 至可见 shootward 端的有序轴测地长度。<br>Ordered-axis geodesic from the distal point to the visible shootward endpoint. | 可信尺度、连续根体、distal point 和有序轴。<br>Valid scale, coherent root body, distal point and ordered axis. | 比较图像内主根伸长/可见范围。<br>Visible primary-root elongation or extent. | 不可测为 null；shootward 触边时仅是整根长度的可见下界。<br>Null if unmeasurable; a shootward crop makes it a visible lower bound. |
| R02 | 主根端点弦长<br>Root-axis endpoint chord length | `root_axis_chord_um` | µm (`um`) | distal 与 shootward 两可见端点的欧氏距离。<br>Euclidean distance between the two visible axis endpoints. | R01 的几何支持和两个有限端点。<br>R01 geometry and two finite endpoints. | 区分净位移与实际轴路程。<br>Separates endpoint displacement from travelled axis length. | 不可测为 null；属于 visible-scope，触边时不代表完整根的端点间距。<br>Null if unavailable; field-of-view conditional. |
| R03 | 主根曲折度<br>Root centreline-to-chord tortuosity | `root_centerline_chord_tortuosity` | ratio | R01/R02。<br>R01 divided by R02. | R01、R02 均有效且 R02>0。<br>Finite R01 and positive R02. | 描述整体弯曲、波状生长或机械响应。<br>Global bending, waving or mechanical-response phenotype. | 分母无效为 null；visible-scope 比值，不外推不可见轴段。<br>Null for an invalid chord; ratio describes the visible span only. |
| R04 | 主根直线度<br>Root straightness | `root_straightness` | ratio | R02/R01。<br>R02 divided by R01. | R01、R02 均有效且 R01>0。<br>Positive R01 and finite R02. | 以接近 1 的直观尺度描述主根直行性。<br>Intuitive near-one representation of axial straightness. | 分母无效为 null；与 R03 数学耦合，不是独立发现。<br>Null for invalid length; mathematically coupled to R03. |
| R05 | 主根二维投影面积<br>Primary-root-body projected area | `root_projected_area_um2` | µm² (`um2`) | 不含根毛的主根根体 mask 像素数乘以尺度平方。<br>Hair-excluded root-body mask pixels multiplied by squared calibration. | 可信尺度和有效主根根体 mask。<br>Valid scale and accepted root-body mask. | 表征二维根体占据/形态投入，不作为三维生物量。<br>Two-dimensional root-body occupancy, not three-dimensional biomass. | 不可测为 null；裁边时是可见根体的部分面积。<br>Null if unavailable; cropped roots yield visible partial area. |
| R06 | 单位根长投影面积<br>Projected area per millimetre visible root | `root_projected_area_um2_per_root_mm` | µm²/mm (`um2_per_mm`) | R05/(R01/1000)。<br>R05 divided by R01 in millimetres. | R05 有效且 R01>0。<br>Finite R05 and positive R01. | 在可见长度标准化后概括根体粗细/面积投入。<br>Length-normalized root-body occupancy or calibre. | 任一组成量不可测则 null；只对同一可见范围解释。<br>Null when either component is unavailable; visible-scope normalization. |
| R07 | 主根中位宽度<br>Median apparent primary-root width | `median_root_width_um` | µm (`um`) | 去除两端不稳定区后，轴上 2×EDT 半径的中位数。<br>Median of twice the distance-transform radius along the end-trimmed axis. | 可信尺度、根体 mask/轴和至少一个正的中心宽度样本。<br>Valid scale, mask/axis and at least one positive central width. | 稳健概括二维表观径向生长或根体口径。<br>Robust apparent radial-growth/calibre summary. | 无有效宽度为 null；裁边时仅代表可见支持区。<br>Null without width support; pertains to the visible supported segment. |
| R08 | 主根宽度 P10<br>Apparent primary-root width P10 | `root_width_p10_um` | µm (`um`) | 中心支持区表观宽度第 10 百分位。<br>10th percentile of supported apparent widths. | 与 R07 相同。<br>Same support as R07. | 捕捉较细段和潜在收缩。<br>Characterizes thinner segments or constrictions. | 无有效宽度为 null；visible-scope 分布量。<br>Null without width support; visible-scope distribution. |
| R09 | 主根宽度 Q25<br>Apparent primary-root width Q25 | `root_width_q25_um` | µm (`um`) | 中心支持区表观宽度第 25 百分位。<br>25th percentile of supported apparent widths. | 与 R07 相同。<br>Same support as R07. | 稳健描述宽度分布的下部。<br>Robust lower-width distribution summary. | 无有效宽度为 null；visible-scope 分布量。<br>Null without width support; visible-scope distribution. |
| R10 | 主根宽度 Q75<br>Apparent primary-root width Q75 | `root_width_q75_um` | µm (`um`) | 中心支持区表观宽度第 75 百分位。<br>75th percentile of supported apparent widths. | 与 R07 相同。<br>Same support as R07. | 稳健描述宽度分布的上部。<br>Robust upper-width distribution summary. | 无有效宽度为 null；visible-scope 分布量。<br>Null without width support; visible-scope distribution. |
| R11 | 主根宽度 P90<br>Apparent primary-root width P90 | `root_width_p90_um` | µm (`um`) | 中心支持区表观宽度第 90 百分位。<br>90th percentile of supported apparent widths. | 与 R07 相同。<br>Same support as R07. | 捕捉较粗段、局部膨大或径向响应。<br>Characterizes thicker segments or local swelling. | 无有效宽度为 null；visible-scope 分布量。<br>Null without width support; visible-scope distribution. |
| R12 | 主根宽度变异系数<br>Primary-root width coefficient of variation | `root_width_cv` | ratio | 正中心宽度样本的样本标准差/均值。<br>Sample standard deviation divided by mean of positive central widths. | 至少两个正宽度样本且均值>0。<br>At least two positive widths with a positive mean. | 描述可见轴上根体粗细异质性。<br>Apparent calibre heterogeneity along the visible axis. | 支持不足为 null；无单位且不等于绝对宽度。<br>Null for insufficient support; dimensionless, not absolute width. |
| R13 | distal 三分之一中位宽度<br>Distal-third median apparent root width | `root_width_tip_third_median_um` | µm (`um`) | 可见有序轴 distal 三分之一的宽度中位数。<br>Median apparent width in the distal third of the visible ordered axis. | 中心宽度样本≥6、轴向跨度≥1 mm 且三段均有样本。<br>At least six central samples, ≥1-mm span and support in every third. | 描述远端可见段口径，用于轴向粗细格局。<br>Distal visible calibre for longitudinal patterning. | 支持不足为 null；三等分是图像区间，不是组织学分区。<br>Null if unsupported; image interval, not a histological zone. |
| R14 | 中段三分之一中位宽度<br>Middle-third median apparent root width | `root_width_middle_third_median_um` | µm (`um`) | 可见有序轴中间三分之一的宽度中位数。<br>Median apparent width in the middle third of the visible ordered axis. | 与 R13 相同。<br>Same support as R13. | 描述可见根中段口径和轴向过渡。<br>Middle-span calibre and longitudinal transition. | 支持不足为 null；visible-scope 图像区间。<br>Null if unsupported; visible image interval. |
| R15 | shootward 三分之一中位宽度<br>Shootward-third median apparent root width | `root_width_shootward_third_median_um` | µm (`um`) | 可见有序轴 shootward 三分之一的宽度中位数。<br>Median apparent width in the shootward third of the visible ordered axis. | 与 R13 相同。<br>Same support as R13. | 描述近 shootward 可见段口径；不保证到达下胚轴。<br>Shootward visible calibre; need not reach the hypocotyl. | 支持不足为 null；触边时只代表可见段。<br>Null if unsupported; cropped values describe the visible segment only. |
| R16 | shootward/distal 宽度比<br>Shootward-to-distal apparent-width ratio | `root_width_shootward_to_tip_ratio` | ratio | R15/R13。<br>R15 divided by R13. | R13、R15 有效且 R13>0。<br>Finite R13/R15 and positive R13. | 描述从远端到 shootward 的锥度/粗细梯度。<br>Longitudinal taper or calibre gradient. | 组成量无效为 null；visible-scope 比值。<br>Null when components are unavailable; visible-scope ratio. |
| R17 | 主根轴向宽度斜率<br>Axial apparent-root-width slope | `root_width_axial_slope_um_per_mm` | µm/mm (`um_per_mm`) | 从 distal 向 shootward 的长度加权线性宽度斜率。<br>Length-weighted linear width slope from distal to shootward. | 至少三个正加权样本且轴向跨度≥1 mm。<br>At least three positive weighted samples over ≥1 mm. | 量化宽度变化方向与幅度；不是生长速率。<br>Direction and magnitude of calibre change; not a growth rate. | 支持不足为 null；符号由 distal→shootward 坐标决定。<br>Null if unsupported; sign follows the distal-to-shootward axis. |
| R18 | 主根中位曲率<br>Median primary-root centreline curvature | `root_centerline_curvature_median_rad_per_mm` | rad/mm (`rad_per_mm`) | 250-µm 半窗绝对转向曲率的中位数。<br>Median absolute turning curvature at a 250-µm half-window. | 有序轴跨度≥1 mm 且至少三个有效曲率位置。<br>At least 1-mm span and three valid curvature locations. | 描述典型局部弯曲、波状生长或机械响应。<br>Typical local bending, waving or mechanical response. | 支持不足为 null；visible-scope、多尺度依赖量。<br>Null if unsupported; visible-scope and scale-specific. |
| R19 | 主根曲率 P95<br>Primary-root centreline curvature P95 | `root_centerline_curvature_p95_rad_per_mm` | rad/mm (`rad_per_mm`) | 同一 250-µm 半窗绝对曲率第 95 百分位。<br>95th percentile of absolute curvature at the same half-window. | 与 R18 相同。<br>Same support as R18. | 强调局部急弯、折点或高曲率响应。<br>Highlights focal bends, kinks or high-curvature responses. | 支持不足为 null；visible-scope、多尺度依赖量。<br>Null if unsupported; visible-scope and scale-specific. |

`root_orientation_deg` 反映图像坐标方向而不是植株生物学方向，因此属于采集几何，不计入 19 项。
The root-cap representation is exactly one distal/root-cap point used as the
axial origin; PHAxis reports no root-cap region, area, perimeter or regional descriptor.

## 4. 13 项非重复根毛描述量 / 13 root-hair descriptors

| ID | 中文名称 / English name | 字段 / Field | 单位 / Unit | 计算依据 / Calculation basis | 所需可观测性 / Required observability | 植物学用途 / Plant-biological use | null、部分与删失语义 / Reporting semantics |
|---|---|---|---|---|---|---|---|
| H01 | 根毛数量<br>Visible root-hair identity count | `hair_count` | count | 绑定工作点下去重后的正式根毛身份数。<br>Deduplicated formal root-hair identities at the bound operating point. | 根毛身份/计数专家输出；原始 count 不依赖物理尺度，但正式组间分析仍要求合格行。<br>Identity/count-expert output; raw count is scale-free, formal comparison still requires eligibility. | 根毛形成/丰度的直接图像级指标。<br>Direct image-level hair formation or abundance. | 已观察且无身份时为 0；不得因长度缺失而删除身份。<br>Zero is a measured biological zero; missing length never removes an identity. |
| H02 | 平均根毛长度<br>Conditional mean endpoint-complete hair length | `mean_hair_length_um` | µm (`um`) | 与正式根毛身份一对一匹配的 endpoint-complete 曲线长度均值。<br>Mean length of one-to-one linked endpoint-complete curves. | 可信尺度且至少一个身份具有 endpoint-complete 曲线。<br>Valid scale and at least one linked complete curve. | 比较可完整测量根毛子集的平均伸长。<br>Mean elongation among completely measurable hairs. | 无完整曲线为 null；是 conditional 子集量，不代表所有身份。<br>Null without complete curves; conditional on length support. |
| H03 | 中位根毛长度<br>Conditional median endpoint-complete hair length | `median_hair_length_um` | µm (`um`) | 同 H02 曲线集合的中位数。<br>Median of the H02 endpoint-complete curve set. | 与 H02 相同。<br>Same support as H02. | 对极长根毛更稳健的条件伸长指标。<br>Robust conditional elongation summary. | 无完整曲线为 null；conditional 子集量。<br>Null without complete curves; conditional on length support. |
| H04 | 已测根毛总长度<br>Measured endpoint-complete hair-length total | `total_hair_length_um` | µm (`um`) | endpoint-complete 一对一匹配曲线之和。<br>Sum of one-to-one linked endpoint-complete curves. | 可信尺度；解释正身份样本时必须检查长度支持数/率。<br>Valid scale; positive-identity rows require length-support fields. | 汇总可测根毛伸长投入。<br>Integrated measured hair elongation load. | H01=0 时为 0；H01>0 且无完整曲线时为 null；支持不全时为 numeric partial。<br>Zero only for H01=0; null with identities but no curves; otherwise possibly partial. |
| H05 | 每毫米可见主根的根毛密度<br>Visible-hair density per millimetre visible root | `hair_density_per_mm_visible_root` | count/mm | H01/(R01/1000)。<br>H01 divided by R01 in millimetres. | H01 有效、可信尺度且 R01>0。<br>Valid H01, calibration and positive R01. | 校正不同图像可见根长后的根毛丰度。<br>Abundance normalized for visible-root extent. | 合格且 H01=0 时为 0；分母/尺度不可测为 null；仍是 visible-scope 密度。<br>Zero for observed H01=0; null for invalid scale/denominator. |
| H06 | 第一根可识别根毛距 distal point 距离<br>Distance from distal point to first identifiable hair | `first_hair_distance_from_distal_point_um` | µm (`um`) | 所有有效根毛身份附着点轴向距离的最小值。<br>Minimum ordered-axis distance among valid identity attachments. | 可信尺度、有序轴且至少一个身份通过附着门。<br>Valid scale/axis and at least one attachment-valid identity. | 描述首个可识别根毛的轴向出现位置。<br>Axial deployment of the first identifiable visible hair. | 无有效附着为 null；0 可为 distal 原点处实测距离；不是“第一细胞隆起”。<br>Null without a valid attachment; zero is possible; not the first cellular bulge. |
| H07 | 第一根长度≥40 µm根毛距 distal point 距离<br>Distance from distal point to first >=40 um endpoint-complete hair | `first_hair_ge40um_distance_from_distal_point_um` | µm (`um`) | 有效附着且一对一完整曲线长度≥40 µm 的正式根毛身份中最小轴向距离。<br>Minimum attachment distance among linked endpoint-complete identities with length ≥40 µm. | H06 几何支持、完整曲线链接和 40-µm 长度判定。<br>H06 geometry, one-to-one complete curve and ≥40-µm length. | 稳健表征根毛达到可见伸长期的轴向转变。<br>Axial transition to a clearly elongated visible hair. | 没有满足阈值的完整曲线时为 null，不得解释为“没有根毛”。<br>Null when no linked curve meets the threshold; not evidence of zero hairs. |
| H08 | distal 1--4 mm 窗口根毛数<br>Root-hair identity count in the distal [1,4) mm window | `local_hair_count_1_4mm` | count | 有效根毛身份附着点位于 `[1000,4000)` µm 的身份数。<br>Count of valid identity attachments in `[1000,4000)` µm. | 正式统计合格、可信尺度且 R01≥4000 µm。<br>Formal eligibility, valid scale and R01≥4000 µm. | 固定发育位置窗口内比较根毛形成。<br>Hair formation in a fixed axial context. | 窗口不合格为 null；窗口合格但无身份为 measured zero。<br>Null for an ineligible window; zero for an observed empty window. |
| H09 | distal 1--4 mm 窗口根毛密度<br>Root-hair identity density in the distal [1,4) mm window | `local_hair_density_per_mm_1_4mm` | count/mm | H08/3 mm。<br>H08 divided by the fixed 3-mm window length. | 与 H08 相同。<br>Same eligibility as H08. | 固定轴位的单位根长根毛丰度，适合条件/基因型比较。<br>Position-standardized abundance for treatment or genotype comparisons. | H08=0 时为 0；窗口不合格为 null。<br>Zero when H08=0; null when the window is ineligible. |
| H10 | 1--4 mm 窗口平均根毛长度<br>Conditional mean hair length in the distal [1,4) mm window | `local_mean_hair_length_um_1_4mm` | µm (`um`) | 窗口内匹配身份的 endpoint-complete 曲线均值。<br>Mean linked endpoint-complete length for identities attached in `[1,4)` mm. | H08 窗口合格且至少一个窗口身份有完整曲线。<br>Eligible window and at least one locally linked complete curve. | 固定轴位背景下比较根毛伸长。<br>Hair elongation at a standardized axial position. | 无局部完整曲线为 null，包括 H08=0；conditional 子集量。<br>Null without local complete curves, including H08=0; conditional. |
| H11 | 1--4 mm 窗口中位根毛长度<br>Conditional median hair length in the distal [1,4) mm window | `local_median_hair_length_um_1_4mm` | µm (`um`) | H10 曲线集合的中位数。<br>Median of the H10 locally linked curve set. | 与 H10 相同。<br>Same support as H10. | 固定轴位下对极长值稳健的伸长比较。<br>Robust axial-position-standardized elongation summary. | 无局部完整曲线为 null；conditional 子集量。<br>Null without local complete curves; conditional. |
| H12 | 1--4 mm 单位根长已测总长度<br>Measured hair-length total per root mm in the distal [1,4) mm window | `local_total_hair_length_um_per_root_mm_1_4mm` | µm/mm (`um_per_mm`) | 窗口内 endpoint-complete 曲线和/3 mm。<br>Sum of locally linked endpoint-complete curves divided by 3 mm. | H08 窗口合格；正身份样本还需至少一条局部完整曲线。<br>Eligible window; positive local count also requires a complete curve. | 在固定轴位综合可测根毛数量与伸长。<br>Integrated measured abundance-by-elongation load at fixed position. | H08=0 时为 0；H08>0 且无完整曲线时为 null；支持不全时为 numeric partial。<br>Zero only for H08=0; null without local curves; otherwise possibly partial. |
| H13 | 可见根毛附着跨度<br>Visible attachment span (descriptive, right-censored) | `visible_hair_attachment_span_um_descriptive_right_censored` | µm (`um`) | 有效根毛身份附着轴位的 max−min；一个附着点时为 0。<br>Max-minus-min valid identity attachment position; zero for one attachment. | 可信尺度和至少一个有效附着点。<br>Valid scale and at least one attachment-valid identity. | 探索性描述可见根毛轴向部署范围。<br>Exploratory description of visible axial deployment. | 无有效附着为 null；始终按 right-censored descriptive 解释，禁止作为完整根毛区确认性长度。<br>Null without attachments; always descriptive and right-censored, never confirmatory whole-zone length. |

### 4.1 根毛专家边界是强制契约 / Mandatory cross-expert boundary

PHAxis 的 `identity_hairs[*].points_xy` 是根毛身份/计数专家的 base-to-tip **存在向量**。它用于身份、计数和
附着定位，不代表人工标注意义下的完整远端中心线。即使该身份有
`complete_length_measurement_eligible=true`，也不得直接对这个两点向量计算 H02--H04、H07 或
H10--H12。

完整长度只能来自 `length_hairs[*].points_xy`，并通过以下键一对一回连正式根毛身份：

- `length_hairs[*].identity_source_instance_id` -> `identity_hairs[*].source_instance_id`；
- `identity_hairs[*].length_measurement_source_instance_id` -> `length_hairs[*].source_instance_id`；
- 匹配必须满足锁定的 base 距离 `<=20 µm`。

身份/计数专家输出的两点向量长度只能作为模型诊断量，不能写入正式 `length_um`。这一规则
同时保护用户提出的“只要确认为一根真实根毛即可”的宽容身份口径和完整长度统计的可解释性。

审阅叠加图也执行同一强契约：绿色只画 `length_hairs` 中一对一匹配的真实
endpoint-complete 完整曲线；橙色只画未匹配正式身份的两点存在向量，不能解释为完整长度。
若 `complete_length_measurement_eligible` 标志与实际关联不一致，或一个身份出现重复长度关联，
渲染器 fail closed，不生成会误导审阅者的图。

### 4.2 H06、H07、H13 与固定轴向窗口 / Axial-deployment semantics

distal/root-cap point 只是一个坐标点，并固定为有序轴距离 `0 µm`；它不是根冠区域。
H06 在所有通过附着门的正式根毛身份中找最近轴位，不需要完整远端曲线。H07 是更严格的
复合描述量：身份必须通过附着门、与 endpoint-complete 曲线一对一关联，且该曲线长度
`>=40 µm`。因此 H07 为 null 只表示没有**可完整测量且达到阈值**的身份，不等于 H01=0。
H13 是可见有效附着点的 max−min；它不寻找“根毛区终点”，且
`whole_hair_zone_confirmatory_allowed=false`，所以始终作为右删失探索性描述量。

The distal/root-cap point is a coordinate origin, not a cap region. H06 uses
the nearest attachment-valid root-hair identity and does not require a complete
distal curve. H07 additionally requires a one-to-one endpoint-complete curve
of at least 40 µm, so a null H07 is not evidence of zero identities. H13 is the
span of observed valid attachments, not an inferred whole hair zone, and is
always descriptive and right-censored.

锁定窗口 `[1,4) mm` 相对 distal point 左闭右开：包含 `1.000 mm` 处的附着，排除
`4.000 mm` 处的附着，固定分母为 `3 mm`。只有正式统计合格且 R01 至少 `4 mm` 时 H08--H12
才有定义。`distal_axis_profiles.csv` 的 `[0,1)`、`[1,2)`、`[2,3)`、`[3,4)`、`[4,5)` mm
是 H01--H13 的空间分解；它们不增加图像级 canonical phenotype 的数量。每个 profile bin 只有在
可见轴完整到达该 bin 远端时才可观测，`[1,4)` 三个 bin 的重新聚合必须逐图复现 H08--H12。

The locked `[1,4) mm` window is left-closed and right-open relative to the
distal origin, has a fixed 3-mm denominator, and is eligible only when the
formal visible axis reaches 4 mm. Profile bins are spatial decompositions, not
additional canonical phenotypes, and their `[1,4)` aggregation must reproduce
H08--H12 exactly.

## 5. PHAxis 标准导出接口与结果文件

当前稳定 CLI 为：

```powershell
phaxis export-traits `
  --predictions <PHAxis融合目录>/predictions `
  --metadata <analysis_metadata.csv> `
  --model-contract <official-contract.json> `
  --output <新建或空的导出目录>
```

等价 Python API 为 `phaxis.traits.export_traits`：

```python
from phaxis.model_contract_binding import read_model_contract_authority
from phaxis.traits import export_traits

authority = read_model_contract_authority("<official-contract.json>")

summary = export_traits(
    prediction_root="<PHAxis融合目录>/predictions",
    metadata_csv="<analysis_metadata.csv>",
    output="<新建或空的导出目录>",
    model_contract_proposal=authority.receipt_fields(),
    model_contract_public_identity=authority.public_identity_fields(),
)
```

prediction 与 metadata 的 `task_id` 集合、图像 SHA-256 和物理尺度必须一致；非空输出目录、
盲测污染、根工件哈希漂移、跨图记录、重复/悬空长度链接均 fail closed。

### 5.1 canonical `image_traits.csv`

每图一行、固定 **82 列**。其中按固定顺序完整包含：软件/模型/专家身份与哈希锁、尺度与坐标、
R01--R19、H01--H13、测量支持/QC、根专家来源、路由/标注/盲测安全字段。82 列中只有 32 列是
非重复生物学数值条目；身份、坐标、元数据、QC、支持度和安全列不得被包装成额外表型。
The 82-column canonical export is a measurement-and-provenance schema; it does not report 82 phenotypes.

### 5.2 分析兼容 `traits.csv`

每图一行，包含：

1. 13 项 canonical 根毛条目 H01--H13；
2. 为密度和生物学联表重复携带的 R01 `visible_root_axis_length_um` 与 R07
   `median_root_width_um`；
3. 正式统计资格、长度可测数与比例、窗口资格、附着有效比例等支持/QC 列；
4. 独立联入的实验元数据列。

该表是便于既有分析脚本使用的规范化视图，不是 canonical 完整 82 列表；与下一表按
`task_id + source_image_sha256` 一对一连接也可恢复 32 项。重复的 R01/R07 不增加表型计数。
正式分析默认筛选 `formal_statistics_eligible=true`，而不是仅判断某个单元格是否有数值。

### 5.3 `detailed_root_statistics.csv`

每图一行，包含 19 项 canonical 主根条目 R01--R19，以及根轴/宽度来源、正式资格和样本元数据。
与 `traits.csv` 连接后得到完整的 19+13=32 项非重复结果。

### 5.4 `hair_instances.csv`

每个正式根毛身份一行。至少包含：`task_id`、`source_image_sha256`、
`hair_id/source_instance_id`、身份置信分数、source 坐标存在向量 JSON、轴向附着距离、
边界误差、附着有效性、1--4 mm 窗口状态，以及可空的 `matched_length_id`、
`length_identity_base_match_error_um` 和正式 `length_um`。正式 `length_um` 只能从匹配的
`length_hairs` 曲线计算。

### 5.5 `distal_axis_profiles.csv`（固定空间剖面派生表）

该表从 `traits.csv` 和 `hair_instances.csv` 只读派生以 distal/root-cap point 为 0 的固定 1 mm
轴向剖面。当前锁定 bins 为 `[0,1)`、`[1,2)`、
`[2,3)`、`[3,4)`、`[4,5)` mm；只有正式统计合格且可见主根轴达到 bin 远端的图像才对该 bin
给出数值。每个 bin 分别报告有效附着的身份数/密度，以及 endpoint-complete 一对一匹配子集的
长度支持数、支持比例、条件均值/中位数和已测总长度。

该表是 H01--H13 的空间分解，**不把 5 个位置 bin 计成新的图像级表型**，因此 PHAxis 的
非重复核心表型总数仍为 32。身份专家的两点存在向量绝不作为长度；零身份 bin 的条件长度和
长度支持比例为 null；不可观测 bin 的全部生物数值为 null。导出器还会把 `[1,4)` mm 三个
bin 重新聚合，与锁定的 H08--H12 逐图交叉核对，任何 count/density/conditional-length/null
语义漂移均 fail closed。机器合同见 `configs/phaxis/v1_0/axial_profile_contract.json`。

### 5.6 `summary.json`

保存 schema、任务/正式资格/根毛身份/长度可测数量、32 项字段列表、四个 CSV 的 SHA-256、
metadata 与逐图 prediction SHA-256、专家职责及 `blind_images_used=0` 等安全标记。模型 checkpoint、
GPU、环境与推理配置的运行级 provenance 继续由上游推理/融合 manifest 保存；不能用 trait
summary 替代它。

### 5.7 兼容字段

当前正式 API 不输出 4 个旧精确别名，也不输出内部消融/历史 tier，避免把诊断列误当成植物学
表型。若以后增加兼容导出，必须显式 opt-in，且仍不得改变 32 项 canonical 计数。

## 6. 缺失值和正式统计规则

- 物理尺度 fail-closed：该行强制 `formal_statistics_eligible=false`；现存物理量只能用于 review，
  不得进入正式生物学统计。像素坐标和 QC 仍可保留。
- `distal_window_1_4mm_eligible=false`：H08--H12 全部为 null，不写 0。
- 窗口合格但没有任何正式身份：H08=0、H09=0；H10、H11 为 null，H12=0。这里的 0 是
  已确认“该窗口没有根毛身份”的生物学零。
- 全图没有正式身份：H01=0、H05=0；H06、H07、H13 为 null。
- 全图身份数为 0 时 H04=0，但 `hair_length_measurement_fraction=null` 且
  `attachment_axis_valid_fraction=null`（两个支持率的分母均为 0，不能写成 1.0），
  `total_hair_length_is_partial=false`。这与“有身份但没有长度/附着观测”严格区分。
- 全图身份数大于 0 时，`attachment_axis_valid_fraction` 必须为有限的 `[0,1]` 数值，并与
  `hair_instances.csv` 中有效附着身份数/全部身份数逐图一致；0 身份时在轴向 profile 中也导出为空值。
- 有身份但没有 endpoint-complete 匹配：H02、H03、H04、H07 为 null；若 1--4 mm 窗口内
  H08>0 但没有局部 endpoint-complete 匹配，则 H10、H11、H12 也为 null。此时 0 会伪装成
  生物学零，禁止使用空集合求和生成 0。
- 至少存在一条 endpoint-complete 匹配时，H04/H12 才能报告匹配子集之和；匹配不完整时它们是
  明确标记的 partial measured total，必须与测量数、支持比例和 `total_hair_length_is_partial=true`
  一起解释。
- `shootward_endpoint_border_visible=false`：R01 是可见下界；不自动外推图像外长度。
- `formal_statistics_eligible=false` 的行保留用于复核，不默认进入组间统计。

## 7. 公开实现职责与回归合同 / Public implementation responsibilities

以下行为由公开 CPU contract tests 锁定：

1. `phaxis export-traits` / `phaxis.traits.export_traits` 原子导出 canonical 82 列
   `image_traits.csv`、三个分析表和 `summary.json`；
2. PHAxis 主根几何专家提供主根根体、有序轴、单个 distal/root-cap point、尺度和
   endpoint-complete 几何；不输出根冠区域；
3. H01、H05、H06、H08、H09、H13 只由正式根毛身份/附着派生；
4. H02--H04、H07、H10--H12 只从一对一链接的 `length_hairs` endpoint-complete 曲线派生；
5. 根毛身份专家的两点存在向量不进入正式长度；
6. 同一正式根毛身份最多关联一条 length curve，重复或悬空链接 fail closed；
7. 19 项主根字段和 13 项根毛字段逐项必需，空平均/中位数保留为 null，长度支持数/比例与
   partial flag 同步保存；
8. treatment/condition 元数据不进入模型路由，`blind_images_used=0`。

The public contract separates the primary-root geometry expert from the
five-member root-hair identity/count expert, preserves their one-to-one geometry link, and exports
the same 32 descriptors through both the CLI and Python API.

## 8. 论文和对外说明的固定口径 / Recommended public wording

- “PHAxis reports 32 canonical image-derived descriptors: 19 primary-root descriptors and
  13 root-hair descriptors, organized into five measurement families: visible-hair abundance,
  conditional projected length, axial deployment, visible-root extent, and root form/trajectory.”
- 部分 descriptor 是同一生物学原量的归一化或数学耦合视图（例如 count/density、
  tortuosity/straightness），因此不得表述为 32 个统计独立 phenotype。
- 32 项是软件可导出的候选表型库，不表示一项实验可以把 32 项全部当作确认性终点；论文应按
  生物学问题预先指定主终点并控制多重比较。
- distal/root-cap point 是长度与空间部署的坐标原点，不是根冠区域、面积或分区。
- H13 是右删失描述量，whole-hair-zone 统计不作为当前确认性结论。
- 根毛长度结论必须同时报告 `hair_length_measurement_hair_count`、
  `hair_length_measurement_fraction` 和 `total_hair_length_is_partial`。
