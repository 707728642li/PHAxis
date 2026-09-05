# PHAxis 1.0.0 公开源码候选

PHAxis 将主根几何、根毛身份和完整长度关联起来，在物理尺度下输出 32 项表型及轴向分布。
本目录是已获用户授权上传 GitHub 的源码候选，不等于正式稳定版本或已发布 PyPI；不含 Bioconda。

```console
git clone https://github.com/707728642li/PHAxis.git
cd PHAxis
python -m pip install .
phaxis --version
phaxis demo --output demo-results
```

打开 `demo-results/report.html`。示例使用合成几何调用真实融合及性状导出函数；不下载模型、不访问 GPU、不代表显微图预测精度。
正式原图流程为 `phaxis analyze`，不加 `--execute` 即预检/计划；需要独立授权模型包。
论文 v3.x 是稿件版本，不改变软件 1.0.0 和当前 train399 五模型身份。

## 模型、原图与标注

[研究资产预览版](https://github.com/707728642li/PHAxis/releases/tag/assets-v1.0.0-preview)单独提供五个根毛模型、主根模型包、443 张 HumanCurated 原图及 443 份原始标注，以及 283 张应用原图。两组间有 22 张相同哈希的图片；Clean261 排除了这些重叠图片。资产须全部上传并通过摘要核验后才会公开下载，不随 Git clone 下载。

原图分成 18 个独立 tar.gz 包，下载后分别解压到同一目录，**不需要拼接分包**。完整下载约 10.6 GB，原图解压后约 20.3 GB。详见[下载与校验说明](docs/research-assets.md)；资产授权不自动等同于源码 Apache-2.0 授权。这些文件尚不是已完成全链路 GPU 验收的一键部署胶囊；本轮没有发布 PyPI。

[英文首页](README.md) · [文档](docs/index.md) · [完整中文用户指南](docs/phaxis/USER_GUIDE.md) · [待确认发布事项](docs/releasing.md)
