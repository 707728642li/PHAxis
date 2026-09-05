# PHAxis 1.0.0 生物材料与图像采集元数据补全规范

## 用途与边界

本规范用于补全 `configs/phaxis/v1_0/POST_TRAINING_MANUSCRIPT_METADATA_TEMPLATE.json` 中的 15 项生物学采集字段。它不要求新增标注、重新训练、读取 blind/final 数据或推断缺失实验事实。当前模板保持 `INCOMPLETE_DO_NOT_USE`；每个尚未由作者或原始实验记录确认的采集字段保持 `DEFERRED_AUTHOR_VERIFICATION`。

原来的 `FINAL_BIOLOGICAL_ACQUISITION_METHODS` 自由文本字段已经废止。正式 manuscript-values 构建要求下表字段恰好齐全、均为非空 author-verified 字符串，且不含 `deferred`、`TODO`、`TBD`、`provisional` 或残余 manuscript token。仅修改状态或重新计算总哈希不能使 deferred 模板通过。

## 逐字段核验表

| Manuscript token | 必须回答的实验问题 | 可接受的权威依据 | 最低可核验内容 |
|---|---|---|---|
| `FINAL_BIOLOGICAL_ACCESSION` | 使用了哪个 *Arabidopsis thaliana* accession/ecotype？ | 原始实验记录、材料登记、已核实的实验方案 | accession 的标准名称；若不同实验不同，逐实验列出 |
| `FINAL_BIOLOGICAL_CONSTRUCT_CONTROL_IDENTITY_AND_SOURCE` | OE 与 EV 标签分别对应什么构建体和对照，来源为何？ | plasmid/line 登记、构建体图谱、材料转交记录、作者确认 | construct/control 的完整身份、promoter/insert/control 类型、材料来源；档案标签不能被扩写为已验证表达量 |
| `FINAL_BIOLOGICAL_GROWTH_MEDIUM` | 培养基及关键补充物是什么？ | 培养配方、实验记录 | medium 名称/强度、sucrose/agar 等关键成分及浓度、pH（若有记录） |
| `FINAL_BIOLOGICAL_PHOTOPERIOD` | 光周期和光照条件是什么？ | 生长室记录、实验方案 | light/dark 时长；已记录时补充光强与光源，不得估算 |
| `FINAL_BIOLOGICAL_GROWTH_TIMELINE` | 播种、层积、萌发、转移和成像时点如何衔接？ | 时间戳、实验日志 | 关键事件相对时间；明确 D15 是实验标识，不是 plant age |
| `FINAL_BIOLOGICAL_TEMPERATURE_EXPOSURE_ONSET` | 22/30°C 条件从何时开始？ | 生长室/处理日志 | 相对播种、萌发或转移的 onset；未知时不得从文件夹名推断 |
| `FINAL_BIOLOGICAL_TEMPERATURE_EXPOSURE_DURATION` | 温度暴露持续多久、是否连续？ | 同上 | duration、连续/分段方式、成像前是否恢复 |
| `FINAL_BIOLOGICAL_PLATE_BLOCK_AND_PLANT_UNIT` | plate/block、plant、source image/root 的关系是什么？ | plate map、样本登记、成像表 | 每个 source unit 是否对应独立 plant；若不能证明，明确 source-unit 限定，不把 acquisition batch 当 biological plate |
| `FINAL_BIOLOGICAL_REPLICATION_AND_RANDOMIZATION` | 独立重复、block 和随机化如何实施？ | plate map、实验设计表、作者确认 | biological/technical replicate 定义、随机化或其不可恢复状态；不可用图片数量代替独立重复数 |
| `FINAL_BIOLOGICAL_IMAGING_DEVICE` | 使用什么成像设备？ | 设备日志、方法记录、图像 metadata | manufacturer、model 与成像模式；不从图像外观猜测 |
| `FINAL_BIOLOGICAL_IMAGING_OBJECTIVE` | 使用什么物镜/放大设置？ | 设备日志、方法记录 | objective magnification、NA（若有记录）、其他改变采样的光学设置 |
| `FINAL_BIOLOGICAL_NATIVE_PIXEL_SAMPLING` | 原始采样和图像尺寸是什么？ | 原始 TIFF metadata、设备导出记录 | native pixel dimensions、bit depth/channels、原始 µm px⁻¹（若设备权威记录存在） |
| `FINAL_BIOLOGICAL_FIELD_SAMPLING_AND_STITCHING` | 如何选择视野并完成拼接？ | acquisition protocol、stitch log | field/overlap 设计、扫描方向、stitch 软件与版本/关键设置、是否存在 alternative stitches |
| `FINAL_BIOLOGICAL_PHYSICAL_CALIBRATION` | 如何把像素换算为物理单位？ | 可见比例尺、设备 calibration、受信 metadata | calibration 来源及适用范围；比例尺缺失或不可信时的 fail-closed 规则 |
| `FINAL_BIOLOGICAL_EXCLUSION_RULES` | 哪些采集失败或技术版本在表型值产生前被排除/合并？ | 预先或独立于 phenotype 的 QC 记录、duplicate/stitch ledger | corrupt/unreadable、视野/比例尺/几何 gate、exact duplicate 和 alternative-stitch consolidation；不得按 phenotype、残差或显著性排除 |

## 283-image collection 与 D15 的固定关系

283 张图像是完整 application collection，clean261 是排除 22 张 HumanCurated443 byte-identical overlap 后的 primary application cohort。D15 只是在该 collection 内提供四个 archived construct-label-by-temperature condition 的探索性因子用例；不得把 283 或 clean261 全部称为 D15 2 × 2 实验。D15 是实验标识而不是植物日龄。OE/EV 是 archived construct labels；在 construct 身份和表达证据补齐前，不得写成已验证 genotype effect、RHD6 expression effect 或 rescue。

## 正式补全和封存

1. 逐字段查回原始实验记录；无法查回的字段继续保留 deferred，稿件保持不可正式编译。
2. 由实验负责人核对每个值是否回答上表问题，并确认没有从文件名、图像外观、相邻实验或旧稿推断事实。
3. 全部字段补齐后，才可将顶层 `status` 改为 `complete_author_verified_external_metadata`。
4. 保持 `blind_images_used=0` 和 `root_cap_region_statistics_included=false`。
5. 按项目 canonical JSON 规则对移除 `human_metadata_identity_sha256` 后的完整对象计算 SHA-256，再写入该字段。
6. 运行 manuscript-values metadata tests；任何缺字段、额外字段、deferred 标记、残余 token、错误 release tag/URL/DOI 或 identity 漂移均应 fail closed。

## 生物学叙事合同

最终的 D15 摘要与 Discussion synthesis 必须依次解释：

1. **主变化层**：可见根毛 abundance，其次是 endpoint-complete 支持下的 projected length；
2. **空间位置层**：first observed ≥40-µm hair 的条件距离及 0–5 mm distal profiles；
3. **supporting-root context**：apparent width、visible extent，以及 R01–R19 提供的 extent/caliber/taper/trajectory 背景。

相关或互为变换的 descriptors 不能投票；“多少项 trait 同方向/有值”不能成为生物学结论。完整 32-descriptor map 用于可观测性和假设生成，正式 headline 仍只来自固定五 endpoint、clean interval 与 Full283 direction gate。
