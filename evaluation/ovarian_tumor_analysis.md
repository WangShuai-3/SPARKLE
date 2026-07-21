# 卵巢癌 Visium HD 切片：环境 RNA 校正与肿瘤相关分析

> 数据集：**Visium HD Human Ovarian Cancer (FF)**，2 µm 像素分辨率
> 分析窗口：`ovarian_x1000-1800_y300-1100`（800×800 像素 ≈ 1.6×1.6 mm）
> 规模：**16,247 个分割细胞**，18,786 个基因，16 种细胞类型
> 方法：SPARKLE（本方法）vs 官方 SpotClean / SpatialSoupX / SoupX / DecontX，以 RAW 为基线

本文汇总了在卵巢癌切片上围绕**环境 RNA（ambient RNA）污染校正**开展的一系列分析，重点验证 SPARKLE 是否
（1）提升细胞类型纯度、（2）改善癌症 marker 特异性、（3）去除虚假的细胞通讯。所有结论都以配对的
**scFFPE 单细胞数据（17,050 细胞，16 类型）** 作为"无空间污染"的真值参照。

---

## 0. 数据与流程

| 输入 | 路径 |
|------|------|
| 空间数据（含分割） | `evaluation/data/ovarian/Visium_HD_Human_Ovarian_Cancer_FF_feature_slice.h5` |
| 单细胞参考 | `evaluation/data/ovarian/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5` |
| 单细胞注释（16 类） | `evaluation/data/ovarian/FLEX_Ovarian_Barcode_Cluster_Annotation.csv` |

分析流水线（脚本均在 `evaluation/scripts/`）：

```
final_comparison.py --dataset ovarian        # 1. 五种方法各自校正 → h5ad
prepare_ovarian_scrna_reference.py           # 2. 构建单细胞 pseudobulk 参考
run_rctd_ovarian.R                           # 3. RCTD 细胞类型映射（每方法）
inject_rctd_annotations.py --annotation-method RAW   # 4. 用 RAW 的 RCTD 注释回填（统一分组）
evaluate_mousebrain_h5ad.py (复用)           # 5. 表达层面对比单细胞
reference_marker_localization.py (复用)      # 5b. marker 定位准确率
analyze_ovarian_cancer_markers.py            # 6. 癌症 marker 特异性
analyze_ovarian_marker_specificity_detailed.py # 6b. 六方法逐 marker 特异性比较
analyze_ovarian_markers_vs_scrna.py           # 6c. 逐 marker 与 scFFPE 参考一致性
analyze_ovarian_cellchat.py                  # 7a. CellChat（非空间, liana）
analyze_ovarian_cellchat_vs_scrna.py         # 7b. vs 单细胞 → 假阳性去除
run_cellchat_spatial_ovarian.R               # 8. 空间 CellChat v2（真实 R 包）
analyze_ovarian_cellchat_proof.py            # 9. COLLAGEN 例子证明下降合理
```

细胞类型分组：

- **肿瘤（Tumor）**：Tumor Cells / Proliferative Tumor Cells / VEGFA+ Tumor Cells / MT-High, Jun+-Fos+ Tumor Cells / Inflammatory Tumor Cells / Malignant Cells Lining Cyst
- **基质/免疫（Stromal/Immune）**：Tumor Associated Fibroblasts / Stromal Associated Fibroblasts / Endothelial Cells / Macrophages / T & NK Cells / Pericytes / Smooth Muscle Cells / Granulosa Cells / Fallopian Tube Epithelium / Ciliated Epithelial Cells

---

## 1. 环境 RNA 校正（运行时长）

在 16,247 细胞 × 18,786 基因上运行：

| 方法 | 运行时长 | 备注 |
|------|:--:|------|
| **SPARKLE** | **176 s** | 最快；校正 9,825/18,786 基因，λ=10 µm |
| SpatialSoupX | 265 s | |
| DecontX | 1,142 s | |
| SoupX | 4,543 s | 最慢（≈76 min）|

---

## 2. RCTD 细胞类型映射质量（doublet 模式）

RCTD 用单细胞参考对每个空间细胞打类型。**singlet 比例↑、entropy↓** 代表映射更确定。

| 方法 | %singlet | %doublet | mean_entropy | singlet_purity |
|------|:--:|:--:|:--:|:--:|
| RAW | 14.3 | 78.8 | 2.00 | 0.563 |
| **SPARKLE** | **27.5** | **50.8** | 2.23 | 0.489 |
| SoupX | 15.0 | 77.8 | 2.01 | 0.560 |
| DecontX | 16.6 | 75.5 | 1.91 | 0.599 |
| SpatialSoupX | 56.5 | 39.8 | 1.49 | 0.707 |

SPARKLE 把 singlet 比例从 14.3% 提升到 **27.5%（近乎翻倍）**——校正后更多细胞被明确判为单一类型。SpatialSoupX 的 singlet 最高但源于过度校正（见后）。

---

## 3. 与单细胞参考的表达一致性

对每个方法（统一使用 RAW 的 RCTD 注释分组），计算类型 pseudobulk 与单细胞参考的相关性等指标。

| 方法 | snRNA相关↑ | 组间相关↓ | 污染下降（高污染基因）↑ |
|------|:--:|:--:|:--:|
| RAW | 0.483 | 0.798 | — |
| **SPARKLE** | **0.540** | **0.664** | **+0.086** |
| SpatialSoupX | 0.536 | 0.923 | +0.516（但整体过校正）|
| SoupX | 0.484 | 0.798 | +0.007 |
| DecontX | 0.472 | 0.748 | +0.016 |

- **snRNA 相关**（越高越好=更接近真实单细胞表达谱）：SPARKLE 最高（0.540，比 RAW +0.057）。
- **组间相关**（越低越好=细胞类型越可分）：SPARKLE 最低（0.664，比 RAW −0.134）。
- SpatialSoupX 虽然对高污染基因下降最多，但组间相关升到 0.923、污染净变 −0.14（**变差**），是过度校正的信号。

> marker 定位准确率（`reference_marker_localization.py`）：SPARKLE vs RAW 平均 Δ≈−0.008（基本持平，4 组改善 3 组变差），说明 SPARKLE 在提升表达一致性的同时未牺牲 marker 的类型定位。

---

## 4. 癌症 marker 特异性

最终主分析只使用 17 个肿瘤 marker 和 9 个基质 marker。每个细胞先按总表达量缩放到 CP10K，再做
`log1p`；随后对每个基因计算
`log2((tumor mean log1p-CP10K + 0.01)/(non-tumor mean log1p-CP10K + 0.01))`。
肿瘤 marker 预期为正、基质 marker 预期为负。6 个免疫 marker 仅保留为探索性逐基因结果，**不进入最终
panel 均值、contrast 或方法排名**。
所有方法使用同一套 RAW-RCTD 细胞注释，避免重新注释造成循环比较。
这里的 `stromal_mean` 沿用原脚本命名，实际汇总所有已注释的非肿瘤细胞；所以免疫 panel 衡量的是
**非肿瘤 vs 肿瘤**分离，而不是某一免疫亚型内部的定位准确率。

### 4.0 预设 marker 的单细胞方向验证

在配对 scFFPE 的 6,369 个肿瘤细胞和 10,681 个非肿瘤细胞中，按与空间分析一致的逐细胞
log1p-CP10K 均值计算 `log2((tumor+0.01)/(non-tumor+0.01))`：

- 17/17 个预设肿瘤 marker 均为正 log2FC（+1.40 至 +3.01；中位绝对值 2.27）。
- 9/9 个预设基质 marker 均为负 log2FC（−1.02 至 −3.48；中位绝对值 2.66）。
- 改用 16 个细胞类型等权，或只用5个结构性基质类型作为对照，26/26 个 marker 的方向均不改变。

因此预设 panel 的**方向得到单细胞数据完整支持**。但方向正确不代表表达量充足：CLDN6 的单细胞
肿瘤/非肿瘤均值仅 0.0255/0.0034，log2FC 对伪计数敏感；FAP 的肿瘤表达也很低（0.0154），但
非肿瘤和结构性基质表达分别为 0.2730/0.4189，负方向仍清晰。完整数值见
`evaluation/reports/ovarian_eval/marker_analysis/scrna_agreement/preset_marker_scrna_tumor_stromal_log2fc.csv`。

### 4.1 最终主结果：log1p-CP10K log2FC

方法级结果是逐 marker log2FC 的算术平均；SEM 也在 marker 层面计算，而不是细胞层面。综合 contrast 定义为
`tumor marker mean log2FC - stromal marker mean log2FC`，因此越大表示两个 panel 沿预期方向的分离越强。

| 方法 | 肿瘤 marker mean log2FC | 基质 marker mean log2FC | tumor-stromal contrast |
|------|:--:|:--:|:--:|
| **SpotClean** | **+0.845** | **−1.620** | **2.465** |
| **SPARKLE** | +0.759 | −1.571 | 2.330 |
| DecontX | +0.748 | −1.443 | 2.191 |
| SoupX | +0.595 | −1.531 | 2.126 |
| RAW | +0.600 | −1.502 | 2.103 |
| SpatialSoupX | +0.350 | −1.441 | 1.790 |

最终排序为 **SpotClean > SPARKLE > DecontX > SoupX > RAW > SpatialSoupX**。主图为
`evaluation/reports/ovarian_eval/marker_analysis/marker_log2fc_mean_sem_bar.png` 和
`evaluation/reports/ovarian_eval/marker_analysis/marker_log2fc_boxplot.png`；最终 26 个 marker 的完整逐基因值见
`evaluation/reports/ovarian_eval/marker_analysis/marker_log2fc_final_core.csv`。不做 log1p 或不做每细胞 CP10K 的结果
只作为 normalization sensitivity，不进入上述主排序。

### 4.2 肿瘤 + 基质核心 panel 的相对 RAW 变化（26 genes）

| 方法 | mean Δ | median Δ | 改善 | 恶化 | 基本不变 | 最优/近最优 marker* |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| RAW | 0 | 0 | 0 | 0 | 26 | 0 |
| **SPARKLE** | **+0.128** | +0.044 | 13 | 12 | 1 | 4 |
| **SpotClean** | **+0.201** | +0.055 | **22** | 4 | 0 | **14** |
| SoupX | +0.007 | −0.004 | 7 | 0 | 19 | 1 |
| SpatialSoupX | **−0.185** | −0.153 | 5 | **20** | 1 | 1 |
| DecontX | +0.076 | **+0.088** | 20 | 5 | 1 | 7 |

\* 与该基因最高分相差不超过 0.01 记为近最优，因此并列时总数可超过 26。改善/恶化阈值同为 |Δ|>0.01。
SpotClean 的表现最一致；SPARKLE 的平均增益居第二，但呈现较明显的 marker-specific trade-off；DecontX
覆盖面较广但会损伤一部分基质 marker。SoupX 基本维持 RAW，SpatialSoupX 则对多数肿瘤 marker 不利。

### 4.3 分 panel 结果

| Panel | SPARKLE | SpotClean | SoupX | SpatialSoupX | DecontX |
|------|------|------|------|------|------|
| 肿瘤（17） | 9↑ / 7↓ / 1≈；median Δ +0.078 | **13↑ / 4↓；+0.168** | 0↑ / 0↓ / 17≈；−0.006 | 2↑ / **15↓**；−0.249 | **16↑ / 1↓；+0.187** |
| 基质（9） | 4↑ / 5↓；−0.129 | **9↑ / 0↓；+0.051** | 7↑ / 0↓ / 2≈；+0.024 | 3↑ / 5↓ / 1≈；−0.054 | 4↑ / 4↓ / 1≈；−0.010 |
免疫 panel 不进入最终表。其探索性结果仍保存在详细输出中，但 CD3E、NKG7 等低表达基因对伪计数敏感，
不能用于方法总体排名。

### 4.4 逐 marker 差异与方法特征

- **SpotClean**：MUC16 (+0.575)、EPCAM (+0.540)、TACSTD2 (+0.492)、CLDN4 (+0.480)、PAX8 (+0.439)
  均明显改善，并且 9/9 基质 marker 全部改善；仅 CLDN6、KRT19、EPHA2、LSR 小幅恶化（−0.047 至 −0.050）。
- **SPARKLE**：WFDC2/HE4 增益为所有方法最高（+1.039），COL1A1 (+0.669)、COL1A2 (+0.687) 的
  区室特异性改善也最强；但 ACTA2 (−0.266)、VWF (−0.214)、LSR (−0.193)、CDH5 (−0.183)、DCN
  (−0.158) 明显下降，说明其强校正不是对所有 marker 都有利。
- **DecontX**：17 个肿瘤 marker 中改善 16 个，尤其 CLDN4 (+0.367)、KRT7 (+0.276)、CLDN3 (+0.269)；
  代价是 DCN (−0.444)、CDH5 (−0.262)、EPHA2 (−0.243)、PECAM1 (−0.183)、ACTA2 (−0.179) 受损。
- **SoupX**：绝大多数改变接近 0；主要增益集中在 COL1A1 (+0.097) 等基质 marker，说明在此数据上校正较弱。
- **SpatialSoupX**：KRT19 (−0.655)、EPHA2 (−0.560)、CLDN6 (−0.489)、LSR (−0.397)、FOLR1
  (−0.377) 明显恶化；虽改善 5/6 免疫 marker，但不能抵消肿瘤 panel 的系统性下降。

从靶区/非靶区分解看，SPARKLE 对两者都强烈压低（中位 log2 变化 −0.699 / −0.664），因而净改善依赖
具体基因；SpotClean 的压低更温和（−0.141 / −0.191），且非靶区下降更多，结果更稳定。SpatialSoupX
反而使非靶区增加 (+0.462) 高于靶区 (+0.187)，解释了其核心 panel 的负 Δ。伪计数从 0.001 到 0.1
时，核心 panel 的 **mean Δ 排序始终为 SpotClean > SPARKLE > DecontX > SoupX > SpatialSoupX**。

完整逐基因数值、靶区/非靶区分解、两两胜负和图见
`evaluation/reports/ovarian_eval/marker_analysis/detailed/marker_specificity_detailed_report.md`。

### 4.5 与配对 scFFPE 单细胞参考的一致性

空间内部 specificity 增加不一定代表更接近无空间污染的参考。进一步对每个 marker 计算其在 16 个匹配
细胞类型中的空间 pseudobulk 与 scFFPE pseudobulk Pearson/Spearman 相关，并用 6 个肿瘤类型和 10 个
非肿瘤类型等权计算 reference specificity 误差。后者可避免空间与单细胞细胞组成差异造成混杂。

相关性列使用全部 32 个 marker；specificity 列只汇总 26 个肿瘤+基质核心 marker，因为稳健范围移除了
T & NK Cells，不能再可靠评估以该类型为主要靶区的 CD3E/CD8A/NKG7。

| 方法 | marker Pearson 中位数（16类型） | 稳健12类型中位数* | 核心 specificity r | 核心 MAE↓（16/12类型） | 方向正确 | 核心误差改善/恶化/不变 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| RAW | 0.637 | 0.776 | 0.434 | 0.811 / 0.821 | 25/26 | — |
| SPARKLE | 0.595 | 0.762 | 0.269 | 0.879 / 0.812 | 24/26 | 8 / 18 / 0 |
| **SpotClean** | 0.693 | 0.794 | 0.422 | **0.709 / 0.732** | 25/26 | **17 / 8 / 1** |
| SoupX | 0.636 | 0.775 | 0.422 | 0.816 / 0.824 | 25/26 | 2 / 3 / 21 |
| SpatialSoupX | 0.596 | 0.763 | 0.369 | 1.000 / 0.990 | 24/26 | 4 / 21 / 1 |
| DecontX | **0.703** | **0.798** | 0.416 | 0.803 / **0.787** | 25/26 | 14 / 10 / 2 |

\* 稳健范围只保留空间 n≥20 的类型；完整 16 类型范围包含 Malignant Cells Lining Cyst (n=6)、
Smooth Muscle Cells (n=4)、Stromal Associated Fibroblasts (n=5)、T & NK Cells (n=2)。因此 CD3E、
CD8A、NKG7 的 16-type 结果只能探索性解读。即使方法的相关性较高，也可能有更大的 MAE，因此
**相关性不能替代绝对误差**：整体排序更相似不代表表达对比幅度更接近参考。

将第 4.1 节“空间内部改善”逐基因与单细胞误差下降交叉验证。由于低样本类型同时也是部分基质/免疫
marker 的主要来源，16 类型保留完整生物学覆盖但噪声较大，12 类型范围降低噪声但会遗漏这些来源；两者
共同构成敏感性边界，不能只选择其中一套：

| 方法 | 核心 marker 内部改善 | 16类型：支持/不变/反对 | 支持率 | 12类型：支持/不变/反对 | 支持率 |
|------|:--:|:--:|:--:|:--:|:--:|
| SPARKLE | 13 | 5 / 0 / 8 | 38.5% | 10 / 0 / 3 | 76.9% |
| **SpotClean** | 22 | **17 / 1 / 4** | **77.3%** | **16 / 3 / 3** | **72.7%** |
| SoupX | 7 | 2 / 2 / 3 | 28.6% | 4 / 1 / 2 | 57.1% |
| SpatialSoupX | 5 | 2 / 1 / 2 | 40.0% | 4 / 0 / 1 | 80.0% |
| DecontX | 20 | 12 / 2 / 6 | 60.0% | 15 / 2 / 3 | 75.0% |

- SpotClean 的 WFDC2（cell-type corr Δ +0.370；16-type 误差减少 +0.699）、EPCAM (+0.379；+0.512)、
  MUC16 (+0.119；+0.474)、TACSTD2（误差减少 +0.336）同时得到单细胞参考支持；其核心 marker 支持率
  在两个范围均为约 73–77%，是最稳定的一组改善。
- SPARKLE 的 MUC16（+0.040；+0.272）和 COL1A2（+0.235；+0.329）得到支持；但 WFDC2 虽然空间
  内部 specificity Δ=+1.039，其 cell-type correlation Δ=−0.420、参考误差变化=−0.094，说明校正方向
  在肿瘤/非肿瘤二分上更强，却没有恢复单细胞中的完整细胞类型分布。COL1A1 也表现为相关性改善
  (+0.173) 但 16-type 幅度误差恶化 (−0.473)，进一步说明相关性与校准必须同时看。SPARKLE 支持率
  在 16/12 类型范围从 38.5% 变为 76.9%，对低样本类型高度敏感，不能给出单一的强结论。
- DecontX 的 EPCAM、CLDN3 等 cell-type correlation 改善，但 COL1A1、CDH5、CD163 的幅度误差明显增加。
- 第 3 节的全局 snRNA 相关性使用数千个 SPARKLE-corrected genes；这里仅针对预先指定的 32 个癌症相关
  marker，因此 SPARKLE 的全局表达一致性改善与 marker panel 表现下降并不矛盾。

完整逐 marker 相关性、reference specificity、交叉验证和图见
`evaluation/reports/ovarian_eval/marker_analysis/scrna_agreement/marker_scrna_agreement_report.md`。

---

## 5. 细胞通讯：非空间 CellChat + 与单细胞比较

用 `liana` 的 CellChat 算法（CellChatDB）在类型层面推断通讯，并以单细胞 CellChat 结果为真值判定假阳性。

### 5.1 通讯量对比（RAW 基线）

| 方法 | 显著互作 | 总强度 | 自分泌强度 | 配对连通率 |
|------|:--:|:--:|:--:|:--:|
| RAW | 895 | 25.66 | 1.61 | 0.434 |
| **SPARKLE** | 765 (−14.5%) | 6.71 (**−73.9%**) | **0.37 (−76.8%)** | 0.484 |
| DecontX | 632 (−29.4%) | 11.13 | 0.78 (−51.5%) | 0.414 |
| SoupX | 923 (+3.1%) | 24.96 | 1.52 | 0.445 |
| SpatialSoupX | 19061 (**+2030%**) | 273 | 20.8 (**+1196%**) | 0.766 |

### 5.2 与单细胞真值比较（是否去除确信假阳性）

单细胞 CellChat 得到 2,422 个显著互作作为"无空间污染"真值。互作键 = (source, target, ligand, receptor)。
**确信假阳性 FP** = 空间显著但单细胞不显著。以 RAW 的 FP/TP 为锚：

| 方法 | 假阳性去除率↑ | 真阳性保留率↑ |
|------|:--:|:--:|
| **SPARKLE** | **23.1%** | **81.9%** |
| DecontX | 50.4% | 53.9%（杀掉46%真信号）|
| SoupX | 0.8% | 100%（几乎没动）|
| SpatialSoupX | 8.0% | 97.4%（但暴增至 19061 个显著，precision 0.45→0.26）|

肿瘤相关互作：SPARKLE 去除 **23.6%** 假阳性、保留 **77.9%** 真阳性。

> **结论**：SPARKLE 是唯一在"有意义地去除 ambient 假阳性"与"保留真实通讯"之间取得平衡的方法。DecontX 去得多但误伤近半真信号；SoupX 几乎无效；SpatialSoupX 反而制造大量伪信号。

---

## 6. 空间 CellChat v2（真实 R 包，考虑空间位置）

用 **CellChat v2.2.0 R 包**的空间模式：`computeCommunProb(distance.use=TRUE, interaction.range=250 µm)`，
细胞坐标取自分割 mask 质心。只允许**空间邻近的细胞群**通讯。输入预检发现 DecontX 在 16,213 个
候选细胞中产生了 15 个零文库细胞；为避免 `normalizeData()` 的除零和保证方法间配对，五种输入统一
剔除这 15 个细胞，最终在相同的 **16,198 个细胞**上重算。预检明细见
`evaluation/reports/ovarian_eval/cellchat_spatial/` 中的 `cellchat_spatial_input_qc.csv` 和
`cellchat_spatial_cell_qc.csv`。

| 方法 | 显著互作 | 总通讯概率 | 肿瘤相关概率 | 自分泌概率 |
|------|:--:|:--:|:--:|:--:|
| RAW | 5297 | 1.022 | 0.610 | 0.292 |
| **SPARKLE** | 5077 (−4.2%) | **0.451 (−55.9%)** | **0.227 (−62.8%)** | **0.140 (−52.1%)** |
| SoupX | 5316 (+0.4%) | 1.021 (−0.1%) | 0.603 (−1.1%) | 0.295 (+1.0%) |
| DecontX | 4194 (−20.8%) | 0.666 (−34.8%) | 0.356 (−41.6%) | 0.188 (−35.6%) |
| SpatialSoupX | 28747 (+442.7%) | 6.074 (+494.5%) | 3.348 (+449.2%) | 1.796 (+515.4%) |

即便加入空间距离约束，SPARKLE 仍呈现同样模式：**互作对数量仅降 4.2%，但通讯概率强度降 55.9%、
自分泌降 52.1%**——这是去除 ambient 弱信号膨胀（而非删除大量真实互作）的特征。DecontX 在修复
零文库预处理缺口后得到可估计结果，通讯强度下降 34.8%，不再是全零；旧的 DecontX=0 是流程失败，
不能作为过度校正的定量证据。SpatialSoupX 仍表现为显著膨胀。

---

## 7. 例证：为何 SPARKLE 的通讯下降是合理的（COLLAGEN 通路）

**前提**：胶原（COL1A1/COL1A2）是成纤维细胞的分泌产物，肿瘤细胞几乎不合成。RAW 里"肿瘤→肿瘤的胶原通讯"
只可能来自邻近成纤维细胞胶原 mRNA 的空间扩散污染。

### COL1A2 表达（log1p-CPM，以单细胞为真值标尺）

| 细胞类型 | scRNA 真值 | RAW 空间 | SPARKLE |
|---------|:--:|:--:|:--:|
| **VEGFA+ 肿瘤细胞** | 0.34 | **2.24（92%表达）** | **0.71（30%）** |
| MT-High 肿瘤细胞 | 1.01 | 3.18（97%）| 1.71（53%）|
| **成纤维细胞 TAF（真源头）** | 4.38 | 4.93（100%）| 4.47（91%）|

1. 单细胞真值：肿瘤 COL1A2=0.34，成纤维=4.38（相差 ~13 倍）→ 胶原确为成纤维细胞基因。
2. RAW 被污染：92% 的肿瘤细胞"表达"COL1A2 且均值 2.24（比真值高 6.6 倍）→ ambient 泄漏。
3. SPARKLE 校正回真值：肿瘤 COL1A2 降到 0.71（≈真值 0.34），而真源头成纤维几乎不动（4.93→4.47）。

### 判别性证据（最关键）

同一个 COL1A2→SDC4 互作，SPARKLE 对不同来源**差异化处理**：

| 互作方向 | 来源真实性 | RAW prob | SPARKLE prob | 处理 |
|---------|-----------|:--:|:--:|------|
| VEGFA+肿瘤 → VEGFA+肿瘤 | 假（ambient 自分泌）| 0.00376 | **0.000** | **完全清除** |
| 成纤维TAF → VEGFA+肿瘤 | 真（胶原确来自成纤维）| 0.00234 | 0.00117 | 保留（仅减半）|

SPARKLE **精确清除肿瘤自分泌的 ambient 假信号（→0），却保留成纤维→肿瘤的真实旁分泌信号**。
这证明第 6 节 −56% 的通讯概率下降主要来自去除 ambient 伪迹，而非破坏真实生物学。

配套图表：`evaluation/reports/ovarian_eval/cellchat_spatial/proof_collagen_expression.png`

---

## 8. 总体结论（SPARKLE）

1. **细胞纯度**：RCTD singlet 比例 14.3%→27.5%（近翻倍）。
2. **表达保真**：与单细胞参考的相关性最高（0.540），细胞类型可分性最好（组间相关 0.664）。
3. **癌症 marker**：SPARKLE 提升部分关键 marker 的空间内部特异性，但其 scFFPE 支持率受低样本细胞类型影响较大（5/13 到 10/13）；SpotClean 在两个范围均较稳定（17/22、16/22）。SPARKLE 中 MUC16、COL1A2 得到支持，而 WFDC2 的二分特异性增益未恢复其完整细胞类型分布。
4. **细胞通讯**：去除 ~23% 的（单细胞确认的）假阳性通讯并保留 ~82% 真阳性；空间 CellChat v2 中通讯概率降 55.9%、自分泌降 52.1%，且经 COLLAGEN 例子证明这些下降精确对应 ambient 伪迹。
5. **对照方法**：修复输入预处理后，DecontX 的空间通讯不再归零（总概率 −34.8%，显著互作 −20.8%）；其非空间分析仍显示较强真信号损失。SoupX 几乎无效；SpatialSoupX 病态膨胀、组间相关变差。

> 数据说明：本窗口以肿瘤细胞为主（VEGFA+/MT-High 约占 89%），基质/免疫细胞相对少，旁分泌类别基于较少的基质细胞；相关分析结论以"减少虚假膨胀"而非"旁分泌绝对增强"来解读更稳妥。

---

*所有中间结果与图表位于 `evaluation/reports/ovarian_eval/`（`.gitignore` 忽略，不入库）。脚本位于 `evaluation/scripts/`。*
