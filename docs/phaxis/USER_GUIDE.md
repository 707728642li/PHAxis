# PHAxis 1.0.0 用户指南 / User guide

本指南面向获得正式 PHAxis 1.0.0 源码包和独立模型资产包的使用者。源码包不含模型权重、
显微图像或人工标注；任何资产都必须有与 PHAxis 1.0.0 一致的 manifest、许可说明和 SHA-256。
This guide assumes an official PHAxis 1.0.0 source package and separately
authorized model assets. The source package contains no weights or biological
images.

PHAxis reports 32 canonical image-derived descriptors. Its 82-column image
table also carries identity, calibration, observability, QC, reason-code, and
provenance fields; it does not report 82 phenotypes.

## 1. 安装与最小自检 / Installation and smoke check

建议为每次发布建立新的虚拟环境。只检查 CLI 可安装基础包；执行完整原图分析安装
`deployment` extra。

```console
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install --upgrade pip
python -m pip install "phaxis[deployment]==1.0.0"
python -m pip check
python -m phaxis --version
python -m phaxis --help
python -m phaxis analyze --help
```

预期版本输出为 `PHAxis 1.0.0`。若从作者提供的独立源码树安装，可把安装目标替换为
`".[deployment]"`；不要在混合研发工作区根目录执行 `pip install .`。

Expected version output is `PHAxis 1.0.0`. When installing from an authored
standalone source tree, replace the install target with `".[deployment]"`.
Do not install from the mixed research workspace.

## 2. 准备输入 / Required inputs

正式工作流使用一个密封的 `PHAxis-analysis-workflow-manifest-1.0` JSON。模型资产发布包应
提供可复制的 manifest 模板和以下内容的准确路径及哈希：

- 已应用的 PHAxis 1.0.0 model contract；
- 主根 provider bundle、其 registry、运行环境和六项输入/部署控制文件；
- 五个互异的根毛身份/计数 expert checkpoints，以及相互绑定的候选、选择和模型元数据回执；
- 每张图像的任务 ID、原图 SHA-256 和可信 `µm/px` 尺度；
- trait metadata CSV 和轴向 profile contract；
- 固定安全字段：condition metadata 不参与路由、canonical annotations 不被读取、
  `blind_images_used=0`、`root_cap_region_output=false`。

Manifest 中的路径可相对 manifest 所在目录。每个文件引用都必须同时给出 SHA-256，完整 JSON
最后用 `manifest_identity_sha256` 自密封。不要猜测 checkpoint、尺度、GPU 编号或哈希；缺失
任一权威时应停止并向资产提供者索取。

The workflow manifest is create-once input authority. Do not hand-edit it after
sealing. A copied manifest remains portable when its relative paths and all
referenced bytes are preserved.

## 3. 先计划，再执行 / Plan before execution

不带 `--execute` 的命令只验证输入、交叉身份和输出路径，并打印确定性计划；不会启动模型、
访问 CUDA 或创建 analysis output。

```console
phaxis analyze --manifest workflow.json --output analysis-output
```

计划通过后，执行前用 `nvidia-smi` 检查显存、利用率和已有进程，选择 manifest 中明确记录的
物理 GPU；不得终止或挂起其他任务。正式执行必须显式加 `--execute`：

```console
phaxis analyze --manifest workflow.json --output analysis-output --execute
```

中断后仅在原 manifest、计划和已完成输出的身份全部一致时恢复：

```console
phaxis analyze --manifest workflow.json --output analysis-output --execute --resume
```

`--resume` 不会忽略损坏、替换或不完整的结果。请为新分析使用新的输出目录；PHAxis
不会覆盖已有非空结果。

## 4. 结果目录 / Outputs

完整工作流在 analysis output 内产生并逐级绑定：

- `root_provider/`：主根根体、连续轴、distal/root-cap 点、宽度与尺度工件；
- `fusion/predictions/`：主根与正式根毛身份/长度链接后的逐图预测；
- `traits/image_traits.csv`：每图固定 82 列的 canonical 记录，其中 32 列是非重复
  image-derived descriptors；
- `traits/traits.csv` 与 `traits/detailed_root_statistics.csv`：便于生物学联表的根毛和
  主根视图；
- `traits/hair_instances.csv`：逐根毛身份、附着、长度支持和 reason-code；
- `distal_axis_profiles/distal_axis_profiles.csv`：固定 1-mm 轴向 bins 的描述性分解；
- 每阶段 summary/receipt 和顶层 `workflow_state.json`：输入、输出、模型与恢复身份。

根冠输出仅为一个 distal/root-cap 点。32 项表型的单位、算法、可观测性和 null/partial/censored
语义见 `TRAIT_CONTRACT_CN.md`。

## 5. 从已有融合结果导出表型 / Export from fused predictions

CLI 会重新验证 prediction 与 official model contract 的公共身份。输出目录必须不存在或为空：

```console
phaxis export-traits \
  --predictions analysis-output/fusion/predictions \
  --metadata metadata.csv \
  --model-contract official-contract.json \
  --output exported-traits
```

`metadata.csv` 的 `task_id` 集合、原图 SHA-256 和尺度必须与 predictions 一致。实验条件只在
推理后联入，不能改变模型路由。正式统计默认筛选
`formal_statistics_eligible=true`，并保留每项支持数、空值原因和部分测量标记。

## 6. 常见失败 / Common fail-closed cases

- 模型、contract、manifest 或图像 SHA-256 不一致；
- 缺少或不可信物理尺度；
- prediction 与 metadata 的 task 集合不一致；
- 输出目录非空，或恢复时 state/plan 身份不一致；
- 一个根毛身份重复链接长度曲线，或把两点存在向量误作完整长度；
- 请求根冠区域统计、条件驱动路由或 blind 输入。

这些错误不会自动降级成可发表结果。保留错误文本和非机密 receipt hashes，按照
`SUPPORT.md` 提交可复现报告。

## 7. 结果解释边界 / Interpretation boundary

PHAxis 描述图像中可见结构。裁边主根长度是下界；未链接完整曲线的根毛仍可计数但没有正式长度；
H13 可见附着跨度始终是右删失描述量。44-image development evidence is not independent external
accuracy, and application images do not provide dense accuracy truth. Consult
`MODEL_CARD.md`, `DATA_CARD.md`, and `TRAIT_CONTRACT_CN.md` before biological
inference.
