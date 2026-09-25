# source24 Figures A-I 设计汇总

更新日期：2026-09-24。本文记录当前 Figure 设计与实现状态，不把旧版 PPT 或服务器残留图片当作设计依据。

## 数据、比较范围与解读边界

- 数据背景：spatialLIBD 的人类背外侧前额叶皮层（DLPFC）10x Visium 切片，人工标注为 Layer1-6 与白质（WM）；层状结构的生物学背景参考 Maynard et al., *Nature Neuroscience* (2021), DOI: 10.1038/s41593-020-00787-0。
- FigureA/B 是 source24 benchmark 层面的展示，覆盖可用的 12 张切片；FigureC-I 的下游分析只使用 donor 3 的 `151673`、`151674`、`151675`、`151676`。FigureE 的空间例图和 FigureG 的深度方向例图固定使用 `151673`；FigureI 仅分析 `151673`。
- 读取既有 frozen prediction CSV；不重新训练或重新聚类。详细生物学图固定使用 `SparNiche_rep1`（seed 1234）；FigureG/I 的对比方法设计为除 SparNiche 外，按 FigureB 的 source24 平均 ARI 排名选取最高的两个方法，并使用各自的 frozen rep1 结果。FigureD/F/H 则先在 method-section 内平均算法 repeats，repeat 不作为生物学重复。
- 统计/解释单位是同一 donor 内的 section；spot 不是独立的生物学重复，四张 section 也不能提供独立 donor 推断。切片之间未进行物理配准，不构造跨切片空间边。
- 预测 cluster 与人工 Layer/WM 通过最大重叠匈牙利匹配对应。`Unmatched` 可留在审计表中，但不在任何 Figure 里展示。Cell2location/scNiche 的 `obsm['X_C2L']` 是混合 Visium spot 上的参考驱动丰度估计，不是单细胞真值或绝对细胞数。
- 下游图一般各自输出 PDF 与同名 300-DPI PNG，不以 PPT 合图替代独立图文件。

## 逐图设计

| Figure | 核心问题与分析方法 | 当前独立图文件（每个 stem 对应 `.pdf` + `.png`） | 解读边界 |
| --- | --- | --- | --- |
| A | 各切片人工 annotation 与各 frozen 方法预测的空间对照，检查层状形态与边界。 | `comparison_results/source24/Figures/FigureA/` 下的 `ground_truth`、`predictions1`、`predictions2`、`all_predictions` 系列；本地另有 `downstream/source24/Figures/FigureA/source24_overview`。 | 空间外观支持解读，但不代替定量指标；范围是整个 source24 benchmark，不仅 donor 3。 |
| B | frozen 预测的聚类/分类 benchmark：ARI、NMI、FMI、Accuracy、MacroF1；各指标箱线图，并有综合图、条形图、资源图和显著性表。 | `comparison_results/source24/Figures/FigureB/` 的各指标独立 boxplot；另有 `metric_boxplots`、`barplot`、`resource_bars`。 | benchmark 表现与 C-I 的 donor-3 生物学证据是两种不同层次的评价。 |
| C | 将 domain 在四张 section 的出现、预测对照、生物学证据并列审视。第一图为各 domain 的 section spot fraction；第二图为方法平均 ARI 与 boundary F1；第三图汇总 marker、cell type、functional support。 | `domain_occurrence_across_sections`；`direct_frozen_result_comparison`；`integrated_evidence_by_domain`。三组独立文件，不输出合并图。 | “冻结预测”指直接评估既有预测标签，而非重新训练或重新聚类；第三图是描述性证据，不是外部验证或统一显著性评分。 |
| D | cortical marker profile fidelity：以人工层标注为 reference，比较各方法的 profile Spearman，并展示 SparNiche 减 baseline 的配对效应及 donor-stratified hierarchical bootstrap 95% CI。 | `marker_fidelity_boxplot`；`marker_fidelity_paired_contrast`。 | 分子身份支持不等于因果或独立 donor 泛化。 |
| E | GraphST Figure 2F/2I 风格的 marker spatial support：每列一个 Layer/WM，上排预测 domain 的空间位置，下两排是两个 marker 的 spatial plot；另画 marker 效应与检出率气泡图，并保留合并版。 | `marker_spatial_support`；`cortical_marker_bubble`；`marker_biological_evidence`。已删除 enrichment 热图。 | 当前只展示 Layer1、Layer3、Layer5、Layer6、WM；Layer2/4 不展示。背景白、黑色 spot 边框、spot 相对先前约放大 1.5 倍；marker 信号支持定位，但不能证明细胞构成。 |
| F | 用 official scNiche Cell2location 丰度考察预测 domain 对细胞类型组成 profile 的支持；展示 profile Spearman 分布与 SparNiche 减 baseline 的配对效应。 | `celltype_fidelity_boxplot`；`celltype_fidelity_paired_contrast`。 | 丰度依赖 Hodge 等人的人类 MTG snRNA-seq reference（75 transcriptomic cell types）；这是 reference-based fidelity，不是 spot-level gold truth。 |
| G | 细胞类型富集、跨切片复现、皮层深度梯度，以及 SparNiche 与 ARI 最高的两个对比方法在相邻层差异表达、空间和功能证据上的固定结果比较。 | Page 1：`page1_celltype_enrichment`、`page1_cross_section_recurrence`。Page 2：`page2_spatial_abundance`、`page2_depth_direction`、`page2_depth_profiles`、`page2_peak_depth_ordering`。Page 3：`page3_positive_enrichment_recurrence`、`page3_layer_differential`、`page3_section_consistency`、`page3_adjacent_spatial_support`、`page3_adjacent_program_support`、`page3_adjacent_depth_profiles`。无 Page 4。 | Page 2 按空间丰度图 → L1 至 WM 方向箭头 → 深度曲线 → 峰值排序阅读。Page 3 的相邻层比较和 FDR 属同 donor 内探索性 spot-level 分析。 |
| H | curated DLPFC functional program profile fidelity：按相同预测结果计算 program profile Spearman，并比较 SparNiche 减 baseline 的配对效应。 | `functional_fidelity_boxplot`；`functional_fidelity_paired_contrast`。 | 是与参考层相关的表达程序一致性，不是直接的 pathway activation 证据。 |
| I | 在 `151673` 对 SparNiche 与 ARI 最高的两个对比方法预测的 Layer1-6 和 WM 分别做 layer-vs-rest 正向差异表达及 GO Biological Process 富集；用 Layer ↔ GOBP 弦图展示经论文背景筛选且实际显著的连接。 | 设计输出为 `FigureI_GOBP_SparNiche`（粉色）及两个按 ARI 选出的方法各自的 `FigureI_GOBP_<方法名>`（蓝色）。旧四图 `functional_biological_evidence` 已不属于现行设计。 | 论文提供层状生物学背景和候选词条，不提供这套精确 GOBP 富集表；实际显著性来自本次数据和预测。未显著或缺失的层不应被解释为生物学缺失。 |

### FigureC 的 integrated evidence 计算

每个 section-domain 行记录 spot 数，以及三类支持：marker support 为所选 marker 的平均 `log2_fold_change`；cell-type support 为正的 `log2_observed_expected` 的平均；functional support 为正的 program `mean_difference` 的平均。图中再按 domain 对四张 section 的这些数值取均值，分别画三条曲线。三类数值的量纲和来源不同，不能把曲线高度直接当作跨类别可比的综合得分。

### FigureE 的 marker 与视觉规格

| 展示分区 | 两个 marker |
| --- | --- |
| Layer1 | RELN、AQP4 |
| Layer3 | ADCYAP1、FREM3 |
| Layer5 | PCP4、TRABD2A |
| Layer6 | NTNG2、KRT17 |
| WM | MBP、MOBP |

气泡图颜色编码 marker 的平均 `log2_fold_change`，大小编码 domain 内检出比例。空间版中未测到的 marker 标为 `Not measured`。当前绘图函数使用白色背景与黑色边框；`source24_downstream.py` 自动生成的旧 FigureE legend 文案仍称“七层、灰背景”，与绘图实现不一致，应以绘图实现为准。

### FigureG 的相邻层比较

Page 1 仅保留 cell-type enrichment 与 cross-section recurrence。Page 3 的 `page3_layer_differential` 以 `Layer1-Layer2`、`Layer2-Layer3`、`Layer3-Layer4`、`Layer4-Layer5`、`Layer5-Layer6`、`Layer6-WM` 六个相邻层对分面；上调/下调用不同颜色区分。SparNiche 与按 ARI 选出的两个对比方法均使用固定 rep1、四张 section 的共同 spot 集合，并映射到相同 Layer/WM 名称。表达量从 raw counts 作每 spot CP10K，Welch 检验用于 `log1p(CP10K)`，每个 method-section-layer pair 对基因做 BH 校正；图上方向判定要求四张 section 均可估计、方向一致，并满足 FDR < 0.01 与绝对 log2FC > 0.5。缺失层对以 `NA` 留在审计表。空间支持图使用相同 marker expression 底图叠加各方法边界；深度曲线使用共同的 L1-to-WM 参考方向比较 predicted layer occupancy。

### FigureH 的 curated programs

基因集版本为 `source24-curated-dlpfc-v1-2026-09-16`，包含 synaptic transmission、inhibitory signaling、myelination、glial homeostasis、complement inflammation、vascular signaling、extracellular matrix、oxidative phosphorylation、translation。FigureD/F/H 的箱线图与配对效应图均分开输出；method-section 内算法 repeat 先平均，再以四张 donor-3 section 作配对比较。

### FigureI 的富集与弦图规则

SparNiche 与按 ARI 选出的两个对比方法使用 `151673` 的共同 spot。对每个匹配的 Layer/WM，采用 raw X → 每 spot CP10K → `log1p(CP10K)` 上的 Welch layer-vs-rest 检验；挑选 log2FC ≥ 0.25 且 BH FDR ≤ 0.10 的正向差异基因。`clusterProfiler::enrichGO` 使用 `org.Hs.eg.db`、BP ontology 与完整 h5ad 基因背景；只有经过文章背景词条筛选、GO BH FDR ≤ 0.05 且命中基因数 ≥ 3 的连接进入弦图。粉色/蓝色是方法区分，不代表效应方向。候选词条、实际富集结果、入图连接分别保存在 source tables，以便审计。

## 代码、文件与版本状态

- 下游主实现：`D:/baselines/new_model/downstream/DLPFC/source24_downstream.py`；FigureC-I 的入口脚本在同目录，FigureG 三方法比较由 `figure_g_comparison.py` 实现，FigureI 由 `figure_i_gobp.py` 与 `figure_i_gobp_chord.R` 实现。目前这两个比较模块仍硬编码 GraphST/STAGATE；按 ARI 动态选择对比方法尚属文档设计，代码及图片未随本次文档修改而更新。
- 本地已保存 FigureA 概览与 FigureG/I 的独立图：`D:/reference/downstream/source24/Figures/`；FigureG/I 的审计表在 `D:/reference/downstream/source24/source_tables/`。FigureA/B benchmark 原始图目录为服务器 `comparison_results/source24/Figures/`，C-I 下游图目录为服务器 `downstream/source24/Figures/`。
- 上次服务器核对时，`FigureI/functional_biological_evidence.pdf/png` 仍是旧版四图；新版 FigureI 的三方法 GOBP 弦图及 source tables 在本地。本次只整理文档，不更新服务器或 PPT；服务器状态以后续实际同步/复核为准。
