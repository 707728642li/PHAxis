# PHAxis 1.0.0 复用交接包构建合同

本合同只定义最终交付树如何被确定性构建和离线验证；它不宣布当前工作区已经满足正式发布条件，也不允许从旧 RHAxis 复用包或历史 PHAxis source snapshot 回填缺失权威。

## 固定边界

- 顶层数据载荷为 `data/`：`data/human_annotated500/` 必须物化精确 500 个 `manual500_source_image`、同任务身份的 500 个 `manual500_raw_return_json`，以及其子集中的精确 443 个 `canonical443_vector_json`；还必须包含 README、DATASET_CARD、数据许可、label schema、build/verification/provenance、全500处置、canonical dataset/split manifest 与 `ALL500_DATA_NOTES_CN.md`。训练/验证可以混合，但逐文件 `task_id`、`dataset_id`、`annotation_kind`、备注、provenance、授权状态、字节数与 SHA-256 必须保留。`data/biological283/images/` 必须是 283 个唯一 task，并逐图携带温度与 genotype/construct（含 RHD6 设计）元数据。
- 顶层模型载荷为 `model/`：正式 source release、使用方法、模型资产、benchmark 与所有权威回执。源码清单中出现 stitch、mosaic 或其他图片拼接/组装组件会拒绝构建；PHAxis 只接收已完成拼接的输入图。
- 根目录生成中文 `README_CN.md` 和中英双语 `PHENOTYPE_CAPABILITIES_CN.md`。后者必须在构建时从项目权威路径 `docs/phaxis/TRAIT_CONTRACT_CN.md` 逐字节复制，而不是另一份可独立修改的目录；它逐项列出 R01--R19 与 H01--H13 的中英文名、单位、计算依据、可观测性、植物学用途和 null/partial/right-censored 语义。根冠只允许一个 distal/root-cap 点；根毛长度只允许来自与 Stage-B 身份一对一关联的 endpoint-complete 中心线。
- 输出目录必须原先不存在。builder 使用同级 staging 目录，完成自验证后才原子改名，不覆盖旧包、冻结 v1 或历史 source snapshot。

## 正式硬门

构建合同 schema 为 `PHAxis-reuse-handover-build-contract-1.0`，自身使用 `contract_identity_sha256` 规范 JSON 自哈希。`bindings` 必须逐项给出项目内相对 `path` 与小写 SHA-256，缺一即 fail closed：

1. 已 CAS apply 的 `applied_model_contract`；
2. train399 candidate、selection、evaluation 三个互绑回执；applied contract 中真实键名固定为 `train399_candidate`、`train399_selection`、`train399_evaluation`；
3. 五个不同且按 `member_index=0..4`、固定 seed `2026082801..05` 顺序与 applied contract 完全一致的 train399 checkpoint；
4. fresh portable exact283 root-provider 回执；
5. 已完成 exact283 的最终 fusion 与 traits 回执，并被 applied contract 反向绑定；
6. exact283 same-hardware benchmark 回执，至少两个运行模式共享同一 hardware identity；
7. formal source-release manifest 与绑定它的 `PHAxis-clean-install-verification-1.0` 测试回执；
8. `dataset_manifest`、`image_manifest`、`model_source_manifest`、`model_asset_manifest`、`benchmark_manifest`；
9. canonical `PHAxis-trait-contract-1.0.0`（19+13、root-cap region=0）。

`scope_attestation` 还必须逐项确认：依法可交付人工标注已完整列入、训练/验证允许混合但 provenance/hash 不丢失、283 生物学设计覆盖温度与 RHD6、图片组装排除、blind/final 分区排除、冻结 v1 不受修改。

五份物化 CSV 的公共必需列为 `source_path,package_path,sha256,bytes,provenance,notes,release_authorized`；人工标注清单另需 `task_id,dataset_id,annotation_kind`，283 图清单另需 `task_id,temperature_c,genotype_or_construct`，模型资产清单另需 `asset_role`，并精确包含五个 `stageb_checkpoint`、一个 `model_bundle_manifest` 和至少一个 `root_provider_asset`。所有 source/binding path 必须是项目内相对路径；builder 拒绝 symlink 链、父目录穿越、Windows ADS/大小写目标冲突和项目外来源。源文件即使原为 NTFS hardlink，也按流式字节复制到 staging，不把源 hardlink 关系带入交付树；复制过程直接对合同 SHA-256/bytes 校验，避免检查与复制之间的文件漂移。

`model_source_manifest` 必须与 formal `SOURCE_MANIFEST.json.files` 在相对路径、大小和 SHA-256 上精确相等。说明图片组装被排除的 Markdown 文档允许存在；只有名称明确属于 stitch/mosaic/image-assembly 的可执行文件或源码会被拒绝，避免误拒绝边界说明文档。

## 使用

仅审计合同，不复制载荷：

```powershell
python -B scripts/phaxis/build_handover_package.py --project-root . --contract <正式合同.json> --check-only
```

所有硬门通过后才允许正式构建到一个全新目录：

```powershell
python -B scripts/phaxis/build_handover_package.py --project-root . --contract <正式合同.json> --output <全新交接目录>
python -B scripts/phaxis/verify_handover_package.py <全新交接目录>
```

verifier 不导入 PHAxis 推理代码；正式包内同时保存自哈希 build contract。离线验证会重新执行 16 个 authority 的 schema、逻辑身份与交叉绑定，重新核对五份物化 CSV 对载荷的闭包、500+500+443 和 exact283 数量、formal source tree 精确闭包、顶层表型目录的 build-time source hash/bytes 回执、32 项目录逐行匹配机器 trait contract，以及图片组装禁令，而不只是信任 builder 写入的 `checks` 字符串。任何增删、重算控制文件后的语义漂移或字节漂移均失败。

实际 producer/缺口的独立审计见 `PHAXIS_HANDOVER_INDEPENDENT_CODE_AUDIT_20260828.md`。正式大包只能在其中标为“待产出”的 authority 与五份 materialisation manifest 均由正式 producer 生成后构建；不得用历史复用包倒填。
