# PHAxis 补充材料 DOCX 布局与逐页核验记录（2026-08-29）

## 1. 核验对象

- 源文件：`docs/phaxis/PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260828.md`
- 构建器：`scripts/phaxis/build_supplementary_docx.py`
- 补充稿 SHA-256：`28897d11f35d07f4a7250b12fd6c6eae362274c927b5274ab8a16a0631cd155d`
- DOCX：`outputs/phaxis_supplementary_docx_layout_fixture_r8_20260829/PHAxis_Plant_Phenomics_SUPPLEMENT_LAYOUT_FIXTURE_R8.docx`
- DOCX SHA-256：`b7b24e5e80ae0da7352836d8bb00f7433800777db90a799aa9f2f4b13dde5868`
- 构建 receipt identity：`aa023aacc81d5f0ba44954ad90dbb6bf8219e576ab7e12575b7250a83ac830e9`
- 构建 receipt 文件 SHA-256：`14949f8dc1d833f87bd932baade0940e1b73c0ced9776c7722c0dd5cb1511346`
- Word COM PDF：`outputs/phaxis_supplementary_docx_layout_fixture_r8_20260829/render_word_com/PHAxis_Plant_Phenomics_SUPPLEMENT_LAYOUT_FIXTURE_R8.pdf`
- PDF SHA-256：`1da72f89688d3e97f44e6d432f40e88c09f7c207eda30c5341919f593f704bde`
- 页面：22 页，150 dpi RGB PNG 全页栅格化。

该对象是醒目标记的布局样稿，不是最终投稿件，不包含最终作者信息或 S1–S9 正式图版。

## 2. 设计合同

沿用主文唯一的 `narrative_proposal` 预设，并使用命名覆盖
`plant_phenomics_manuscript`：Letter 纸、四边 1 inch、Times New Roman 11 pt、
1.5 倍行距、连续行号、安静的居中页眉与页码。标题页和 S1–S9 图版页不显示行号；
正文保留连续行号。布局样稿在标题页、页眉和每个图版占位页均使用红色
`NOT FOR SUBMISSION` 标记。

## 3. 逐页视觉检查

在 Word 桌面渲染结果中以原始分辨率逐一检查第 1–22 页：

- 第 1 页：与主文同步的新标题、版本、补充图/表数量和主文关系层次清楚；没有标题下横线或模板残留。
- 第 2–7 页：S1–S9 方法完整，正文、公式符号、等宽代码和编号列表未裁切、未重叠。
  生产预处理的三个通道公式、九个 Stage-B tensor heads、32 点双向 tolerant matcher、
  H12 linkage 重建和固定五端点/15-effect 合同均完整可读。
- 第 8–10 页：Figure S1–S9 图注标题与正文层次完整，无孤立标题或溢出；更新后的
  Fig. S9 明确展示全部 32 descriptors 的四条件 raw-median z map、相对 IQR、coverage，
  并把固定 15 effects 独立成紧凑面板。
- 第 11–13 页：Table/Data S1–S10 说明和 Table S9 三块资源合同完整，无裁切；内部
  `Supplementary machine-fill policy` 已按构建合同截断，不进入面向读者的 DOCX。
- 第 14–22 页：九个图版页一页一图，图题、页眉、页码和红色占位标记均稳定；这些大面积留白为正式图版预留，不是布局缺陷。

所有 22 页均未发现文字裁切、对象重叠、异常字体替换、页眉/页码丢失、断裂标题、
错误方向或表格几何问题。Word COM 独占创建的任务进程在导出后被构建包装器清理；
既有 Word 进程未被触碰。

第七轮的 22 张 150 dpi RGB 页面已全部以原始清晰度逐页检查。第八轮将 S3 图版标题
统一为“Identity, formal attachment, endpoint, and conditional-length assurance”，并将 S5
统一为“Root-provider equivalence, same-component root continuity, formal attachment, and
tiled-inference assurance”。逐页 SHA-256 比较显示仅第 16、18 页变化，两页均已重新直接
检查；其余 20 页与第七轮逐像素相同。第 7 页方法结束后的留白以及第 14–22 页图版留白
均为显式分节/正式图版预留，不是意外空白。第八轮为当前代码生成的权威布局证据。

## 4. 长路径回归

首次真实构建发现 Windows 旧式路径上，长投稿文件名与重复临时文件名叠加会超过路径限制。
主文与补充材料的确定性 ZIP 归一化、原子 staging 目录均已改为短的私有临时名；
新增主文和补充材料长文件名回归测试。当前文稿/图例/两份 DOCX 构建器定向组合回归为
`33 passed`；新增回归直接从当前补充稿构建，并验证 Table S9 的四列结构、固定 A/B/C
行、重复表头、表头 `keepNext` 与逐行禁止跨页拆分属性。

## 5. 最终构建要求

最终补充材料只能在以下条件同时满足后生成：最终作者 metadata 已封存；主文为无占位符的
最终编译稿；主文 compiler receipt 哈希闭合；S1–S9 图版全部为最终 train399-only 结果并
逐图校验 SHA-256；主文与补充材料的模型 bundle、根 provider 和 Stage-B 身份完全一致；
`blind_images_used=0` 且不包含根冠区域统计。最终 DOCX 仍须重复 Word/PDF 全页渲染和逐页检查。
