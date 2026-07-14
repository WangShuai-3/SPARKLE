# CRC 2-µm 空间转录组：环境 RNA 校正、双分割 RCTD 与单细胞参考分析

> 数据集：**人结直肠癌（CRC）2-µm 空间转录组**
> 分析窗口：`x=14200–15000, y=2750–3550 µm`（800×800 µm）
> 条件：**Proseg** 与 **StarDist** 两套 spot-to-cell 分割结果
> 规模：160,064 spots、2,471,982 UMI、18,085 个基因（全部保留）
> 方法：SPARKLE / SpatialSoupX / SoupX / DecontX，以 RAW 为基线

本文参照 `evaluation/ovarian_tumor_analysis.md` 的评估思路，在同一个 CRC
窗口中平行处理两套分割。重点回答三个问题：

1. 不同校正方法是否提高 RCTD 细胞类型映射的确定性；
2. 校正后的类型 pseudobulk 是否更接近配对的 Pelka CRC 单细胞参考；
3. 上述结论是否在 Proseg 与 StarDist 两种分割下保持一致。

所有 baseline 均使用仓库原始实现和默认算法参数；没有为本次全基因运行增加
降维、减少 `n_init`、缩减 marker 或改写后验计算。

---

## 0. 数据、窗口与流程

### 0.1 CRC spot 范围与选定窗口

注册后的全切片物理坐标范围为：

- x：**13967.49–15338.95 µm**；
- y：**2464.03–3835.49 µm**。

本次使用的 800×800 µm 窗口覆盖全切片 **34.03%** 的 spots，较此前
600×600 µm 快速窗口进一步扩大，同时仍能完成全基因、全方法计算。

| 条件 | 分割细胞 | cell spots | empty spots | empty 比例 |
|------|---------:|-----------:|------------:|-----------:|
| Proseg | 4,503 | 107,013 | 53,051 | 33.14% |
| StarDist | 4,440 | 30,326 | 129,738 | 81.05% |

两个条件的 160,064 spots、2,471,982 UMI 和 18,085 个基因完全相同；仅
spot-to-cell label map 不同。因此条件间差异主要反映 segmentation，而不是
表达输入或裁剪范围差异。

### 0.2 单细胞参考

Pelka CRC CELLxGENE h5ad 以 `ClusterMidway` 为标签。为避免 370,115 个原始
细胞直接进入 RCTD，`prepare_crc_scrna_reference.py` 固定随机种子、按类型平衡
抽样，得到：

- **9,781 个细胞**；
- **37,486 个唯一 gene symbols**；
- **20 种细胞类型**：B、DC、Endo、Epi、EpiT、Fibro、Granulo、ILC、Macro、
  Mast、Mono、NK、Peri、Plasma、Schwann、SmoothMuscle、TCD4、TCD8、Tgd、
  TZBTB16；
- 同一批细胞同时生成 RCTD raw-count reference 和 log1p-CP10K pseudobulk，
  保证两项分析的标签体系一致。

### 0.3 分析流水线

```text
final_comparison.py --dataset crc             # 两种分割 × 五种表达矩阵
prepare_crc_scrna_reference.py                # 平衡单细胞 reference + pseudobulk
run_rctd_crc.R                                # 每个分割、每个方法独立跑 RCTD doublet
inject_rctd_annotations.py --annotation-method RAW
                                               # 固定 RAW 注释，隔离表达校正效应
evaluate_mousebrain_h5ad.py                   # 类型 pseudobulk vs 单细胞 reference
```

RCTD 与空间数据共有 **17,590 个基因**。单细胞相关分析只使用相应条件中
SPARKLE 实际校正、且在 reference 中存在的基因：Proseg 为 5,742 个，StarDist
为 2,632 个。因两套基因集合不同，**绝对相关值只在各自 segmentation 内比较**；
不把 Proseg 0.74 与 StarDist 0.64 直接解释成分割优劣。

---

## 1. 全基因环境 RNA 校正

| 条件 | 方法 | 运行时 | 关键结果 |
|------|------|-------:|----------|
| Proseg | SPARKLE | 33.3 s | λ=20 µm；5,983/18,085 基因通过 R²≥0.01 |
| Proseg | SpatialSoupX | 29.0 s | λ=10 µm；ρ=0.1301；12,573 个候选基因 |
| Proseg | SoupX | 7,034.7 s | 原版 225 簇；ρ=0.0580 |
| Proseg | DecontX | 899.6 s | 原版 10 个 Leiden 簇；平均 contamination=0.585 |
| StarDist | SPARKLE | 26.3 s | λ=10 µm；2,740/18,085 基因通过 R²≥0.01 |
| StarDist | SpatialSoupX | 24.4 s | λ=10 µm；ρ=0.2842；14,459 个候选基因 |
| StarDist | SoupX | 6,218.8 s | 原版 222 簇；ρ=0.0400 |
| StarDist | DecontX | 3,330.0 s | 原版 45 个 Leiden 簇；平均 contamination=0.458 |

SoupX 的主要耗时来自原始全维 `KMeans(n_init=10)`，以及对 1,001 个候选 ρ
逐一遍历 gene×cluster 组合的标量 Gamma 后验；本次没有改变该实现。DecontX
表中时间为脚本记录的主迭代时间，不含前置 normalize/PCA/Leiden。

分割会显著改变方法的内部估计：StarDist 的 empty spot 比例更高，对应
SpatialSoupX 的 ρ 更高；DecontX 也从 Proseg 的 10 簇变为 StarDist 的 45 簇，
运行时间随之增加。

---

## 2. RCTD 细胞类型映射质量

RCTD 使用 doublet mode。`%singlet` 越高表示更多空间细胞被明确映射到单一类型；
`mean_entropy` 越低表示权重更集中。但校正会改变 UMI，RCTD 的内部过滤会使
各方法实际保留的细胞数不同，因此必须同时查看 `n_cells` 和 shared-cell 指标。

### 2.1 Proseg

| 方法 | RCTD cells | %singlet ↑ | %doublet ↓ | entropy ↓ | singlet purity ↑ |
|------|-----------:|-----------:|-----------:|----------:|------------------:|
| RAW | 3,951 | 37.31 | 62.69 | 1.799 | 0.684 |
| **SPARKLE** | **3,940** | **39.97** | **60.03** | **1.781** | 0.683 |
| SoupX | 3,951 | 37.51 | 62.49 | 1.800 | 0.683 |
| DecontX | 1,817 | 44.91 | 55.09 | 1.571 | 0.721 |
| SpatialSoupX | 2,469 | 37.46 | 62.54 | 1.500 | 0.765 |

在与各方法相同的细胞集合上比较 RAW：

| 方法 | shared cells | RAW doublet | 方法 doublet | Δdoublet | Δentropy |
|------|-------------:|------------:|-------------:|---------:|---------:|
| SPARKLE | 3,940 | 62.74% | 60.03% | **−2.72 pp** | **−0.018** |
| SoupX | 3,951 | 62.69% | 62.49% | −0.20 pp | +0.000 |
| DecontX | 1,817 | 61.70% | 55.09% | −6.60 pp | −0.130 |
| SpatialSoupX | 2,463 | 66.59% | 62.61% | −3.98 pp | −0.281 |

SPARKLE 在几乎不损失细胞（3,951→3,940）的前提下，提高 singlet 2.67 个百分点。
DecontX 和 SpatialSoupX 的部分“改善”伴随大量细胞退出 RCTD，不能只看 singlet
或 entropy 判为更优。

### 2.2 StarDist

| 方法 | RCTD cells | %singlet ↑ | %doublet ↓ | entropy ↓ | singlet purity ↑ |
|------|-----------:|-----------:|-----------:|----------:|------------------:|
| RAW | 2,314 | 63.31 | 36.69 | 1.570 | 0.674 |
| **SPARKLE** | **2,140** | **66.31** | **33.69** | **1.550** | 0.672 |
| SoupX | 2,314 | 63.31 | 36.69 | 1.570 | 0.674 |
| DecontX | 1,448 | 65.26 | 34.74 | 1.567 | 0.668 |
| SpatialSoupX | 429 | 71.10 | 28.90 | 1.127 | 0.769 |

shared-cell 比较：

| 方法 | shared cells | RAW doublet | 方法 doublet | Δdoublet | Δentropy |
|------|-------------:|------------:|-------------:|---------:|---------:|
| SPARKLE | 2,140 | 36.73% | 33.69% | **−3.04 pp** | **−0.005** |
| SoupX | 2,314 | 36.69% | 36.69% | 0.00 pp | −0.001 |
| DecontX | 1,448 | 34.94% | 34.74% | −0.21 pp | +0.017 |
| SpatialSoupX | 406 | 35.96% | 29.56% | −6.40 pp | −0.345 |

SPARKLE 再次降低 doublet，且保留 92.5% 的 RAW-RCTD 细胞。SpatialSoupX 虽有
最高 singlet 和最低 entropy，却只保留 **429/2,314（18.5%）** 的 RAW 规模；
这是强烈的选择效应，不能把 71.1% singlet 单独作为质量提升证据。

### 2.3 分割条件差异

RAW 在 Proseg 与 StarDist 中的 singlet 分别为 37.31% 和 63.31%，实际 RCTD
细胞分别为 3,951 和 2,314。这个差异远大于同一分割下多数校正方法的变化，说明
**segmentation 本身是 RCTD 结论的重要条件**。因此本报告不合并两个条件，也不
把某一分割的绝对 singlet 值外推为校正算法的普遍效果。

---

## 3. 与 Pelka 单细胞 reference 的表达一致性

所有方法统一使用 RAW RCTD 的 `first_type` 分组；Unknown 细胞不进入类型
pseudobulk。这样方法间差异来自表达校正，而不是每个方法重新分组。

### 3.1 Proseg（5,742 个 reference-overlap 校正基因）

| 方法 | 单细胞相关 ↑ | Δ vs RAW | 组间相关 ↓ | Δ vs RAW | silhouette ↑ |
|------|-------------:|---------:|------------:|---------:|-------------:|
| RAW | 0.7420 | — | 0.5655 | — | −0.052 |
| **SPARKLE** | **0.7413** | **−0.0007** | **0.5365** | **−0.0290** | −0.059 |
| SoupX | 0.7422 | +0.0003 | 0.5642 | −0.0013 | −0.053 |
| DecontX | 0.7010 | −0.0410 | 0.4271 | −0.1384 | −0.496 |
| SpatialSoupX | 0.6884 | −0.0536 | 0.8142 | +0.2488 | −0.151 |

SPARKLE 基本保持单细胞相关（变化小于 0.001），同时把类型间相关从 0.5655
降到 0.5365，说明表达保真不变而类型可分性提高。SoupX 几乎不改变 RAW。
DecontX 虽降低组间相关，但单细胞相关下降且 silhouette 大幅恶化；SpatialSoupX
同时降低单细胞相关并显著提高组间相关，属于不理想的表达扭曲。

### 3.2 StarDist（2,632 个 reference-overlap 校正基因）

| 方法 | 单细胞相关 ↑ | Δ vs RAW | 组间相关 ↓ | Δ vs RAW | silhouette ↑ |
|------|-------------:|---------:|------------:|---------:|-------------:|
| RAW | 0.6438 | — | 0.3962 | — | −0.024 |
| **SPARKLE** | **0.6722** | **+0.0284** | **0.3275** | **−0.0687** | −0.044 |
| SoupX | 0.6436 | −0.0003 | 0.3948 | −0.0015 | −0.025 |
| DecontX | 0.6173 | −0.0266 | 0.3126 | −0.0837 | −0.292 |
| SpatialSoupX | 0.6786 | +0.0347 | 0.6326 | +0.2364 | −0.053 |

StarDist 中 SPARKLE 同时提升单细胞相关和类型可分性，是两项指标方向一致的
改善。SpatialSoupX 的单细胞相关略高于 SPARKLE，但类型间相关从 0.3962 升至
0.6326，且 RCTD 仅保留 429 个细胞，提示结果受强校正和选择效应影响。DecontX
降低类型间相关的同时损失单细胞表达一致性；SoupX 仍基本等同 RAW。

### 3.3 跨分割一致结论

两种分割共同支持：

1. **SPARKLE**：在 Proseg 中保持 reference 相关、在 StarDist 中提高 reference
   相关；两者都降低类型间相关并降低 RCTD doublet。
2. **SoupX**：与 RAW 几乎重合；全基因计算耗时最长，但变化最小。
3. **DecontX**：两种分割均降低单细胞 reference 相关，并显著减少 RCTD 可用细胞。
4. **SpatialSoupX**：结果高度依赖分割；Proseg 中 reference 相关下降，StarDist
   中虽上升但组间相关恶化、RCTD 保留率极低，存在过度校正/选择效应。

---

## 4. 总体结论

1. **窗口与基因覆盖**：在 800×800 µm、160,064 spots、18,085 全基因上完成了
   Proseg 与 StarDist 的五方法平行评估。
2. **RCTD**：SPARKLE 在两种分割中都降低 shared-cell doublet（Proseg −2.72 pp；
   StarDist −3.04 pp），且保留的细胞显著多于 DecontX/SpatialSoupX。
3. **单细胞保真**：Proseg 中 SPARKLE 与 RAW 基本持平（−0.0007），StarDist 中
   提升 +0.0284；两种分割中类型间相关都下降，方向一致。
4. **分割敏感性**：RAW singlet、可用细胞数、cell/empty spot 比例以及方法内部
   参数均随分割明显变化。双条件报告是必要的，不能只展示一套 segmentation。
5. **baseline 解释**：SoupX 成本高而变化小；DecontX 的参考一致性损失和细胞
   退出提示激进校正；SpatialSoupX 的高 singlet/高 reference 相关不能脱离低保留率
   和更高类型间相关单独解读。

---

## 5. 结果位置

- 五方法 h5ad：`evaluation/reports/h5ad/`
- RCTD：`evaluation/reports/rctd_crc/{proseg,stardist}/`
- 固定 RAW 注释 h5ad：
  `evaluation/reports/h5ad_crc_{proseg,stardist}_annotated_x14200-15000_y2750-3550/`
- 单细胞相关、热图与汇总：
  `evaluation/reports/crc_eval_full/{proseg,stardist}/method_level_new/`

中间矩阵和图表由 `.gitignore` 忽略，不纳入版本库；本报告、CRC 流程脚本和命令
示例纳入版本控制。
