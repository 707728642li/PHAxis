# PHAxis 版本与模型身份政策

## 唯一公开版本

本项目对外的软件、论文、命令行、Python 包、模型卡、图表和引文版本统一为
**PHAxis 1.0.0**。公开模型身份必须以
`PHAXIS-V1.0.0-STRICT-TRAIN399-` 开头，并由锁定的 Stage-B train399 候选、
工作点选择回执和稳定根提供器 bundle identity 共同确定；不得手工填写或沿用
开发阶段的静态常量。

公开表面的映射固定如下，后续训练 seed、阈值选择、运行批次、GPU 分配和论文
结果填充均不得改变它：

| 表面 | 唯一值 |
|---|---|
| 产品名称 | `PHAxis` |
| 软件版本 | `1.0.0` |
| Python distribution | `phaxis` |
| import namespace | `phaxis` |
| CLI | `phaxis` |
| Git tag | `v1.0.0` |

当前研发根目录的 `pyproject.toml` 属于冻结 legacy workspace；它不是 PHAxis 的
安装或发布权威。只有正式 source builder 新建并经 verifier 密封的独立目录才可
声明上述映射。论文、README、MODEL_CARD、DATA_CARD、CITATION 和 release
metadata 必须从该独立目录及 applied model contract 读取身份，不能从路径名或
旧根目录元数据推断。

发布目录的唯一机器指针为 `release/RELEASE_AUTHORITY_REGISTRY.json`。在正式
source tree 和 gate receipt 产生前，其 `current_formal_source_release` 必须为
`null`。`release/PHAxis_V1_0_Source_20260828/` 已明确隔离为 superseded
development snapshot；其旧 `pyproject.toml`、CFF、classifier、代码和文档均不得
进入安装、交接、GitHub/PyPI、论文补充材料或引文。

## 内部兼容命名不是产品版本

以下字符串是可复现性所需的历史/内部兼容命名，不是公开产品版本：

- `configs/phaxis/v1_0/`：PHAxis 1.0.0 的内部配置命名空间；
- `PHAXIS-V1.0-FROZEN-V1-V20-Q8-HYBRID-ROOT-20260828`：冻结根提供器 ABI 与
  历史 V1/V20 计算闭包的 provenance 标识；
- `RHAxis`、`RhizoWeave`、`Hybrid-Max`、`V20.12`：被只读复用的前代组件或
  算法分支名称。

这些内部标识可以出现在机器回执、bundle registry 和方法学 provenance 中，
但不得作为论文标题、软件版本、Git tag、PyPI 版本或公开模型版本。论文中如需
提及，必须称为“PHAxis 1.0.0 所锁定的历史根提供器组件”。

## 根提供器身份稳定性

根提供器的公开 expert ID 从稳定的 root-provider bundle identity 派生，而不从
一次运行的 pipeline identity 派生。物理 GPU、分片数、批大小、输出目录和运行
时间可以改变 pipeline identity，但只要 bundle 字节不变，就不能改变公开模型
身份；反之，bundle 中任何受管字节改变都必须产生新的 root expert 与 PHAxis
model-bundle ID，并重新完成相应等价审计。

当前 fresh283 管线使用的根 bundle 为 NTFS hardlink 物化闭包。运行中不得为了
文案美化改写其 contract 或 bundle ID，因为这会同时改变受管字节并破坏已记录
哈希。公开版本统一由上层 PHAxis 1.0.0 身份合同实现，而不是重命名冻结历史
组件来实现。

`nextgen candidate`、`Hybrid-Max final`、`Stage-B seedN`、`runN`、`rNN` 等均为
研发过程或组件角色，不得成为公开模型版本。正式五成员选择完成后，公开模型
bundle ID 由 checkpoint 顺序、选择回执和稳定 root-provider bundle identity
计算得到；此前保持 proposal/deferred 状态，不手写一个“看起来最终”的 ID。

## proposal 与正式应用后的运行权威

模型发布前，运行权威是 sealed、validated-but-unapplied 的 model-contract
proposal。正式 compare-and-swap 应用后，运行权威切换为 sealed official
contract；新输出仍携带原 proposal 的 file SHA-256 与 logical identity，从而与
应用前生成的证据保持同一公共模型身份。promotion 工具继续只接受未应用
proposal，避免把正式合同误当作待晋级候选重复应用。

## 禁止事项

- 不得把 `V1.0`、`v1_0` 或历史 RHAxis 版本显示为 PHAxis 的公开版本；
- 不得用路径名、时间戳或一次 runtime pipeline hash 直接充当公开模型身份；
- 不得在 proposal 应用后因 authority 文件自身 SHA 改变而重命名同一模型；
- 不得为了保持旧 ID 而接受 bundle、checkpoint、selection 或数据划分漂移。
- 不得在论文正文、图表标题、PyPI、GitHub Release 或交接包顶层同时展示多个
  “最终版本”；这些位置只允许 `PHAxis 1.0.0`，历史组件仅在明确的 comparator
  或 provenance 上下文出现。
