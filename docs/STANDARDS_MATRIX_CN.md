# 软件发布标准对照（本地准备范围）

版本：PHAxis 1.0.0；日期：2026-09-05。依据用户指定的《软件发布标准.md》，明确排除 Bioconda。不把本地通过等同于公网发布。

| 标准主题 | 本轮物化内容 | 验证 / 边界 |
|---|---|---|
| 科学正确性 | 正式模型及 promotion 回执核对；保留既有算法和开发评测 | 五模型身份/阈值不变；没有重新做盲测或推理 |
| 数值真值 | 两根毛身份、14 µm 完整长度、未匹配长度缺失、零根毛、重复运行测试 | 真实融合/性状函数；合成输入明确标记 |
| 输入合同 | 原有 sealed manifest、校准、task ID、哈希合同及完整用户指南 | analyze 不加 execute 即预检/计划；不伪造简化生产输入 |
| 输出合同 | 82 列/32 表型字典、逐毛表、轴向表；JSON/CSV 与哈希 | 原有字段不变；缺失长度不改成 0 |
| CLI | help/version/analyze/fuse/infer-hairs/export-traits；新增 demo/report | 公开可复制本地安装与测试命令 |
| HTML | 离线报告、搜索、排序、分页、下载、QC/支持字段、provenance | 桌面/移动端实测；无 CDN 或遥测 |
| 环境复现 | 专属 Conda 打包环境、两个独立干净安装环境、CPU wheelhouse/哈希锁 | Windows CPython 3.12 实测；不声称完整 GPU 环境已测 |
| GitHub 首页 | SVG wordmark/工作流、真实示例报告截图、简洁双语 README | 状态徽章明确 local preparation，非伪造 CI/PyPI 绿标 |
| 文档站 | MkDocs Material，搜索/导航/版本/代码复制，输入输出/教程/FAQ | 本地 strict 构建；Pages/HTTPS 尚未部署 |
| CI | Windows/Linux Python3.10–3.12、CPU/构建/文档/静态检查/安全模板 | 本地已测部分有日志；远程矩阵未运行 |
| 治理 | 贡献/安全/支持/治理、issue/PR 模板、Dependabot/CODEOWNERS 草案 | 真实 GitHub 账号及分支保护需负责人配置 |
| PyPI | wheel+sdist、twine strict、各自离线干净安装/CLI/demo/pip check | 无上传；TestPyPI/OIDC 待授权，上传 job 硬禁用 |
| 供应链 | SHA-256、源码清单、直接依赖 SBOM、CPU 已解析 SBOM与漏洞扫描 | 不伪造远程签名 attestation、DOI 或容器 digest |
| 工作流/容器 | Snakemake 示例、Docker/Apptainer CPU 配方 | 本机无 Docker/Apptainer，未声称运行成功 |
| 信用/元数据 | CITATION 与独立作者元数据模板、模型/数据卡、Apache/Tomli license | 沿用集体贡献者条目；姓名/ORCID/URL/DOI 等待确认 |
| Bioconda | 排除 | 不创建 recipe、不测试、不上传 |

最终日志、安装回执、哈希和核准结果见交付目录 qa/ 与 README_本地交付说明.md。
