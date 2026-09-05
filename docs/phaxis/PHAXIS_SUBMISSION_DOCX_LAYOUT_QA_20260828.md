# PHAxis 1.0.0 Plant Phenomics 投稿 DOCX 版式审计（更新至 2026-08-29）

## 结论

第十三轮投稿版式夹具通过 Microsoft Word 真实渲染与逐页视觉复核，取代此前夹具作为
当前版式证据。夹具仅用于验证排版，
`submission_use_allowed=false`，不会被误作正式投稿稿。正式稿仍必须等待无占位符的
编译稿、六幅密封主图、最终模型/分析回执和作者核验元数据。

## 输入与身份

- 主稿：`docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260828.md`
- 主稿 SHA-256：`0a48c9b954f373731ca115a1a493b9f5361ac3f0b25f7578d41798919fdc0df6`
- DOCX：`outputs/phaxis_submission_docx_layout_fixture_r13_20260829/PHAxis_Plant_Phenomics_LAYOUT_FIXTURE_R13.docx`
- DOCX SHA-256：`ea7921fe18c6304643b242c2b8d85f50c9a53d3c50ae74570ffd2f2496e19058`
- 构建回执 identity：`a6d3e8455159a445ecfea664f5beae104b91a105dede4d2847168753cdd58b15`
- 构建回执文件 SHA-256：`b461d60a745fb59d05d7d6ae8633bdb858a2c8cf280736ce5147e4e10ed5bc10`
- PDF：`outputs/phaxis_submission_docx_layout_fixture_r13_20260829/render_word_com/PHAxis_Plant_Phenomics_LAYOUT_FIXTURE_R13.pdf`
- PDF SHA-256：`3df40cedc6eccb4962639d1fabcc14a520d422a5966dadde135aca2a67ff4b67`

## 验证结果

- Word COM 以隐藏、只读、宏关闭方式渲染为 42 页；任务自建 Word 进程正常退出，
  DOCX 与 PDF 均已计算 SHA-256。
- 42 页全部以 Poppler 150 dpi 栅格化。第十二轮的 42 页均以原始清晰度逐页直接复核；
  第十三轮与第十二轮逐页 SHA-256 比较后，仅第 21 页因 Fig. 2 标题同步而发生像素变化，
  该页已再次以原始清晰度直接复核，其余 41 页具有逐像素继承证据。
- 未发现文本、表格或图版裁切、重叠、页外溢出、错误方向或不可读缩放。
- 标题页不带页眉、页脚和行号；正文从第 2 页起具有连续行号、运行页眉和页码。
- 3 张主表使用横向页面、固定 DXA 列宽、重复表头和无竖线设计。第三轮真实渲染曾
  暴露 Table 3 长行跨页后语义片段孤立的问题；当前构建为每个 Word 表格行写入
  `w:cantSplit`，使每个完整性状/对比行保持为不可拆分单元。第 22–30 页确认所有行
  完整且无孤立 `}}` 片段。
- 第六轮逐页检查进一步发现 Table 2 的重复表头可单独出现在第 24 页页底，而首个
  数据行在第 25 页。第七轮为所有表头单元格段落写入 `keepNext`，并以首个数据行的
  `cantSplit` 联合约束初始表头。Word 真实重排后，第 24 页只保留完整表题和说明，
  第 25 页从表头与首个完整数据行共同开始；不存在孤立表头。
- 第八轮编辑后发现三个可改善的真实 Word 版式问题：行内 LaTeX 以普通字符显示，
  表后空段落得到孤立行号，以及两位数参考文献编号与作者名间距不足并把第 19–20 条
  挤到近空白页。第十轮把轴坐标写成稳定的 `x(s)` / `s = 0` 文本；对三个表后
  spacer 写入 `w:suppressLineNumbers`；把参考文献编号 text start/hanging indent 调为
  720/360 twip，并把段后距收紧为 1 pt。真实渲染确认空行号 604、609、626 消失，
  `10. Tsang` 至 `20. PHAxis` 均有清楚间隔，20 条参考文献完整置于第 32 页，原近空白
  spill page 被消除。
- 长机器占位符仍会在夹具中单元格内换行；这是压力测试，正式短值写入后不会保留该
  视觉噪声。
- 20 条参考文献使用真实 Word 编号并位于第 36 页；第 35 页下半部留白来自横向表格
  到纵向参考文献的强制分节，不含孤立行号。6 张主图各有独立整页图版位置，第 37–42
  页无裁切或漂移。
- `Machine-Fill Placeholder Registry` 和内部放置标记不进入 DOCX。
- 当前轮同步纳入正式根毛附着、同一连通组件主根连续性、38 个可见比例尺 + 6 个可信
  元数据比例尺的适用性边界，以及固定五端点/15-effect 统计合同。Fig. 2 标题在主稿、
  图例生成器与 DOCX 中统一为同一最终文本。逐页检查确认这些编辑没有造成新的孤立标题、
  表格断裂、参考文献 spill 或图版分页漂移。
- 当前文稿/图例/主文与补充材料 DOCX 定向组合回归为 `33 passed`；主文构建器继续覆盖确定性、哈希闭包、全部表格行的 `cantSplit` OOXML、
  表头 `keepNext`、重复表头、标题边框、分节首页、占位符/图像漂移、长路径 staging 和元数据拒绝路径。
  新回归还直接验证两位数参考文献 tab/hanging indent 与表后 spacer 的
  `w:suppressLineNumbers` 属性。

## 仍待正式输入

最终 DOCX 构建器继续 fail closed；缺少以下任一项都不生成投稿稿：最终 train399
模型合同、QC-development44/283 图正式分析、六幅 final figure suite、无占位符编译回执，
以及由作者核验的姓名、单位、通讯邮箱、ORCID、基金、贡献和利益冲突信息。

本审计没有启动 GPU、没有读取 blind/final-validation、没有读取部署期 canonical 标注，
`blind_images_used=0`，根冠区域统计仍为 false。较早的 r2–r4 文件仅作为历史回归证据，
第六轮保留为孤立表头的负面排版证据，第八轮保留为参考文献 spill、编号间距和空行号
问题的负面证据；第十二轮保留为第十三轮逐页像素继承的直接视觉基线，但不再是当前
版式权威。
