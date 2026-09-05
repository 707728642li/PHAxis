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

[英文首页](README.md) · [文档](docs/index.md) · [完整中文用户指南](docs/phaxis/USER_GUIDE.md) · [待确认发布事项](docs/releasing.md)
