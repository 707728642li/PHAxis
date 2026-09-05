# PHAxis 1.0.0 GitHub 与 PyPI 正式发布指南

本文说明如何把已经通过正式 Gate 的 PHAxis 1.0.0 独立源码树发布到
GitHub 与 PyPI。它不授权公开源码、模型或数据，也不允许从当前大型工作区
直接执行 `git add .`。唯一可发布输入是
`scripts/phaxis/build_source_release.py` 原子生成且由
`FORMAL_RELEASE_GATE_RECEIPT.json` 与 `SOURCE_MANIFEST.json` 共同密封的
独立源码目录。

## 1. 唯一公开身份

- 产品、Python distribution、import namespace 与命令分别为
  `PHAxis 1.0.0`、`phaxis`、`phaxis`、`phaxis`。
- Git tag 固定为 `v1.0.0`。
- 历史组件、开发数据简称和 root-provider ABI 名称只表示机器回执中的
  provenance，不建立第二个公开软件版本。
- 模型权重、显微图、人工标注和预测结果不随 Apache-2.0 源码包自动授权。

## 2. 发布前必须具备的机器权威

正式 source builder 必须同时收到并验证：

1. applied PHAxis 1.0.0 model contract；
2. 五个互异 train399 checkpoint 及 candidate、selection、evaluation receipts；
3. fresh exact283 root-provider audit；
4. 显式传入的 final fusion 与 traits receipts；final root-hair、figures 与
   manuscript-evidence receipts 已由 applied model contract 的 promotion
   evidence 逐项哈希绑定，不作为 source builder 的重复命令行参数；
5. 作者确认的 `PHAxis-release-human-metadata-1.3` 权威，包括作者和维护者的
   显示名、明确拆分且不由程序猜测的 given/family names、邮箱、机构、有效
   ORCID（若未提供必须显式为 `null`，不得用占位值），以及 GitHub/PyPI 坐标、release DOI、ISO release date、源码发布授权
   及独立资产边界。未知值必须保持模板状态并在 human gate 处暂停，不得用
   虚构值通过 Gate。

缺少任何一项时 builder 只能生成带有
`BLOCKED_DEVELOPMENT_STAGING_DO_NOT_RELEASE.json` 的开发树；该树不得上传。

## 3. 构建独立源码树

### 3.1 正式 DAG 的 GPU0 用户锁

如果 GPU0 临时要让给更紧急的任务，可在首次启动正式 post-training DAG 时显式加入
`--hold-physical-gpu 0`：

```console
python -B scripts/phaxis/assemble_post_training_release_manifest.py \
  --config <assembly-config.json> \
  --manifest-output <formal-manifest.json> \
  --run-output <formal-run-directory> \
  --launch \
  --hold-physical-gpu 0
```

DAG 会完成此前所有不需要 GPU0 的阶段，并在第一个真正要求物理 GPU0 的阶段前以退出码 `5`
和状态 `paused_for_user_gpu_hold` 正常暂停；暂停前不会为该阶段调用 `nvidia-smi`、GPU probe 或
生产命令。这是用户资源锁，不是训练或算法失败。GPU0 可用后省略 hold，并对同一 manifest 和
run directory 恢复：

```console
python -B scripts/phaxis/run_post_training_release.py \
  --manifest <formal-manifest.json> \
  --output <formal-run-directory> \
  --execute --resume
```

冻结的 v1 同硬件 benchmark 明确绑定物理 GPU0；不得把它改派到 GPU1 来绕过用户锁。GPU1 可以
继续承担其合同允许的其他阶段，但不能替代该 frozen-v1 GPU0 权威。

在项目 conda 环境中运行正式 builder，并显式传入所有 receipt 路径：

```console
python -B scripts/phaxis/build_source_release.py \
  --project-root <project-root> \
  --output <new-empty-phaxis-source-directory> \
  --root-provider-exact283-receipt <fresh-root-audit.json> \
  --train399-candidate-manifest <candidate-manifest.json> \
  --train399-selection-receipt <selection-receipt.json> \
  --train399-evaluation-receipt <evaluation-receipt.json> \
  --final-fusion-summary <fusion-summary.json> \
  --final-traits-summary <traits-summary.json> \
  --release-human-metadata <author-verified-release-metadata.json>
```

随后从项目外观上独立地复验该目录：

```console
python -B scripts/phaxis/verify_source_release.py <new-empty-phaxis-source-directory>
```

正式目录必须只有 `FORMAL_RELEASE_GATE_RECEIPT.json`，不得同时出现 blocked
receipt；`SOURCE_MANIFEST.json` 必须覆盖目录内除自身以外的全部文件字节。
其中 `NOTICE`、`THIRD_PARTY_NOTICES.md`、`THIRD_PARTY_LICENSES.json` 与
`SBOM.cdx.json` 必须由 PHAxis builder 生成并进入同一闭包，不能复制研发工作区
根目录中历史 RHPheno/RHAxis 的同名文件。

## 4. 构建并验证 wheel 与 sdist

发行构建器会依据 `SOURCE_MANIFEST.json` 把独立源码树逐文件复制到私有、一次性的
构建输入目录，并只在该副本内运行 PEP 517；正式源码目录在 stage 41 前后必须保持
逐字节不变。手工复验可在另一个临时副本内运行：

```console
python -B -m pip install ".[build]"
python -B -m build
python -B -m twine check dist/*
```

应得到 `phaxis-1.0.0-py3-none-any.whl` 和对应 sdist。正式 clean-install
executor 会把源码、wheel、完整 `phaxis[deployment]` 离线 wheelhouse、SHA-256
依赖锁和自包含模型 capsule 复制到一次性的 release root，在其中新建不继承
system-site-packages 的 venv。它先以 `--no-index --require-hashes` 安装已锁定的
raw-image→profiles 部署依赖，再以 `--no-index --no-deps` 安装 PHAxis wheel；随后
复验 `pip check`、所有直接部署模块的导入来源、`phaxis --version`、CLI plan、
root subprocess Python、全部运行时绝对路径和一个真实非 blind 示例的输出身份。
这些路径与模块必须全部位于该一次性 release root，不能回退到作者工作区、
`PYTHONPATH` 或用户 site-packages。没有该 sealed clean-install receipt 时不得上传。

源码树中的 `SBOM.cdx.json` 是声明级直接依赖/版本范围权威；随后
`offline_dependencies` stage 必须针对 CPython 3.12/Windows wheelhouse 生成
`SBOM.resolved.cdx.json`（精确版本、wheel SHA-256 与依赖图）及
`THIRD_PARTY_LICENSES.resolved.json`（逐 wheel METADATA 和许可证成员哈希）。
两级 SBOM 含义不得混用；后者仍要求发行者保留并人工复核每个锁定 wheel 的
artifact-specific license/notice，不能把 inventory 当作自动法律许可结论。

## 5. GitHub 仓库准备

只把正式独立源码目录作为新仓库根目录。推荐过程是：

```console
git init --initial-branch=main
git add --all
git commit -m "Release PHAxis 1.0.0"
git remote add origin <author-approved-github-repository-url>
git push -u origin main
git tag -a v1.0.0 -m "PHAxis 1.0.0"
git push origin v1.0.0
```

执行 `git add --all` 前必须确认当前目录就是 source builder 新生成的独立
目录，而不是研发工作区。发布树已提供：

- `CITATION.cff`、`CHANGELOG.md`、`CONTRIBUTING.md`；
- `CODE_OF_CONDUCT.md`、`SECURITY.md`、`SUPPORT.md`；
- `NOTICE`、第三方依赖/许可证 inventory 与 CycloneDX 1.6 SBOM；
- PHAxis 专用 issue 与 pull-request templates；
- Python 3.10/3.11/3.12 CPU CI，且 `checkout` 与 `setup-python` 均按不可变 commit
  SHA 固定；Python 3.10 的普通、`-S` 和 `-I -S` source-verifier 路径使用随
  source manifest 与 wheel 一起封存的、逐字审计的 Tomli 2.4.0 源码副本。
  该副本的文件 SHA-256、MIT 许可证路径和 CycloneDX 组件身份必须同时闭合，
  不依赖 verifier 运行前从网络安装 TOML 解析器。

该跨平台 CI 只证明源码与 CPU 合同在 Ubuntu/Windows 上的兼容性。正式
offline wheelhouse 与 GPU clean-install 回执当前只证明 Windows CPython 3.12，
不得外推成 Linux、其他 Python 版本或其他 GPU 平台上的正式复现证据。

CI 只测试源码，不携带权重、图片、标注、凭据或 blind 数据。公开仓库应启用
GitHub Security Advisories；正式发布动作建议使用受保护 environment。

## 6. PyPI Trusted Publishing

建议由项目负责人在 PyPI 创建 `phaxis` 项目并配置 GitHub OIDC Trusted
Publisher。不要把长期 PyPI token 写入源码、workflow、日志或本地配置文件。
第一次可先使用 TestPyPI 验证元数据和安装，但 TestPyPI/PyPI 的坐标必须与
author-verified release metadata 一致。

正式上传前重新核对：

- distribution=`phaxis`、version=`1.0.0`；
- console entry point=`phaxis = phaxis.cli:main`；
- 作者、维护者的机构/ORCID、Homepage/Repository/Issues/Documentation、
  release DOI/date 坐标正确；
- wheel/sdist SHA-256 与 release checksums 一致；
- wheel `METADATA` 的两个 `License-File` 字段及 `.dist-info/licenses/` 中的
  Apache-2.0、vendored Tomli MIT 文本均已按 `SOURCE_MANIFEST.json` 的 bytes/SHA-256
  验签，且 distribution 与 clean-install 回执均为
  `license_file_hashes_verified=true`；
- wheel 内没有模型、图像、标注、预测、主机绝对路径或凭据。

PyPI 安装验证：

```console
python -B -m pip install phaxis==1.0.0
phaxis --version
phaxis --help
```

## 7. GitHub Release 资产

`v1.0.0` release 至少附加 wheel、sdist、source manifest、checksums、正式
release receipt、clean-install receipt、SBOM/依赖锁和第三方声明。模型 bundle
使用独立下载资产、独立许可、总 SHA-256 与成员 manifest；数据集和 283 图像
按各自授权与 DOI/仓库发布，不嵌入 wheel 或源码归档。

论文、GitHub release、PyPI metadata、model card 和 citation 文件最终必须引用
同一 `PHAxis 1.0.0`、同一 tag 与同一模型/根提供器公共身份。commit 不预写入
待提交的发布元数据；受保护 workflow 从真实 tag checkout 的 `GITHUB_SHA` 生成
`PHAXIS_PUBLISH_PROVENANCE.json`，并在 PyPI 发布后把它附加到 GitHub Release。
