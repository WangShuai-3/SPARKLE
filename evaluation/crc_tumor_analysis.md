# CRC 2-µm 空间转录组：环境 RNA 校正、双分割 RCTD 与单细胞参考分析

> 数据集：**人结直肠癌（CRC）2-µm 空间转录组**
> 分析窗口：`x=14200–15000, y=2750–3550 µm`（800×800 µm）
> 条件：**Proseg** 与 **StarDist** 两套 spot-to-cell 分割结果
> 规模：160,064 spots、2,471,982 UMI、18,085 个基因（全部保留）
> 方法：SPARKLE / SpatialSoupX / SoupX / DecontX，以 RAW 为基线

本文参照 `evaluation/ovarian_tumor_analysis.md` 的评估思路，在同一个 CRC
窗口中平行处理两套分割。重点回答四个问题：

1. 不同校正方法是否提高 RCTD 细胞类型映射的确定性；
2. 校正后的类型 pseudobulk 是否更接近配对的 Pelka CRC 单细胞参考；
3. 考虑细胞质心距离后，空间 CellChat 通讯是否更接近 Pelka 单细胞参考；
4. 上述结论是否在 Proseg 与 StarDist 两种分割下保持一致。

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
run_cellchat_spatial_crc.R                    # 质心距离约束的空间 CellChat v2
analyze_crc_spatial_cellchat_vs_scrna.py      # 空间互作 vs Pelka CellChat reference
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

### 2.4 严格配对的 RCTD score 比较

为避免 2.1/2.2 中细胞保留集合不同造成选择偏差，进一步运行
`analyze_crc_paired_rctd_scores.py`。该分析沿用 3.4 节的跨分割物理细胞配对，
并只保留 Proseg/StarDist × RAW/SPARKLE 四份 RCTD 结果都存在的细胞：最终纳入
**1,328/1,423 对（93.32%）、13 个类型**。RAW 与 SPARKLE 在同一分割内逐细胞
配对，bootstrap 也按共识细胞类型分层并使用相同的重采样索引（10,000 次）。

RCTD doublet mode 的主要连续指标定义为
`score margin = singlet_score - min_score`；spacexr 在 margin < 25 时判为 singlet，
因此 **margin 越低越偏向 singlet**。结果如下：

| 分割 | RAW margin | SPARKLE margin | 配对 Δ | bootstrap 95% 区间 | RAW singlet | SPARKLE singlet | Δ singlet | first_type 一致率 |
|------|-----------:|---------------:|-------:|---------------------:|------------:|----------------:|----------:|------------------:|
| Proseg | 31.074 | 28.090 | **−2.984** | [−3.437, −2.575] | 47.97% | 52.48% | **+4.52 pp** | 97.74% |
| StarDist | 14.573 | 14.051 | **−0.522** | [−0.975, −0.045] | 73.42% | 74.32% | +0.90 pp | 93.52% |

- **Proseg 是明确改善**：63 个细胞由 doublet 转为 singlet、仅 3 个反向变化；
  精确 McNemar/binomial 检验 `p=1.30×10⁻15`，singlet 增量的 bootstrap 95%
  区间为 [+3.39, +5.72] pp。13/13 个细胞类型的平均 margin 都下降，说明结果
  不是由单一大类驱动。
- **StarDist 的连续 margin 有小幅改善，但类别变化证据不足**：62 个细胞转为
  singlet、50 个反向变化，`p=0.299`；singlet 增量区间 [−0.68, +2.48] pp 跨过
  0。13 个类型中 9 个 margin 下降。
- RAW→SPARKLE 后 `first_type` 一致率仍为 93.5%–97.7%，主要变化是 singlet/
  doublet 置信度，而非大范围重写主细胞类型。

敏感性分析将匹配距离改为 3–7.5 µm，并将每类型最少细胞数改为 3、10、20：
Proseg margin Δ 为 −2.67 至 −2.98、singlet 增加 +4.29 至 +4.59 pp；StarDist
margin Δ 为 −0.49 至 −0.70，而 singlet 变化区间在所有设置均跨 0，结论稳定。

这与 3.4 节形成互补：Proseg 的 RCTD singlet 置信度明确改善，但对单细胞参考的
表达相关基本不变；StarDist 的参考相关明确提高，而 RCTD 类别变化较小。两组
指标分别衡量混合类型判定与表达保真，不能互相替代。

> 不同 segmentation 的绝对 RCTD score 还受 UMI 深度和每个细胞所含 spots 数量
> 影响，因此 Proseg 与 StarDist 的绝对 margin 只能作描述，不能据此给分割方法
> 排名。这里的主要推断是每套分割内部 RAW→SPARKLE 的同细胞配对变化。

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

### 3.4 严格配对：相同物理细胞 × 共同 R² 基因

前述 3.1/3.2 在各自 segmentation 内固定 RAW 注释，但两套分割使用的细胞集合和
SPARKLE 校正基因集合仍不同。为进一步控制比较口径，运行
`analyze_crc_paired_shared_r2.py`，增加以下限制：

1. **不能按 cell ID 匹配**：两套分割有 1,450 个同名 ID，但同名细胞质心距离
   中位数约 497 µm，说明 ID 是分割内部编号；
2. 用质心 **双向最近邻**建立一对一映射，并限制距离≤5 µm，得到 3,667 对；
3. 只保留两套 RAW-RCTD 均非 Unknown 且注释一致的细胞；去掉少于 3 个细胞的
   类型后，最终为 **1,423 对细胞、13 个类型**；
4. Proseg 与 StarDist 分别有 5,983 和 2,740 个基因通过 R²≥0.01；交集为
   2,253 个，其中 **2,150 个**存在于 Pelka reference；
5. RAW 与 SPARKLE 使用完全相同的细胞、共识注释和 2,150 个基因。CP10K
   library size 仍从全部 18,085 个基因计算，再截取共同基因；
6. 以相同的细胞重采样索引进行 1,000 次配对 bootstrap。

| 分割 | RAW 相关 | SPARKLE 相关 | Δ | bootstrap 95% 区间 | P(Δ>0) |
|------|---------:|-------------:|--:|---------------------:|-------:|
| Proseg | 0.76813 | 0.76593 | **−0.00221** | [−0.00833, −0.00078] | 0.007 |
| StarDist | 0.70152 | 0.73288 | **+0.03136** | [+0.00700, +0.03130] | 0.998 |

因此，在最严格的同细胞、同注释、同基因口径下，结论不是“两套分割都增加”：

- **StarDist 明确增加**，平均相关提高约 0.0314；
- **Proseg 没有增加**，点估计轻微下降约 0.0022。这个差异很小，主要应解读为
  “保持不变/未改善”，而不是有生物学意义的大幅损失。

敏感性检查将匹配距离改为 3–7.5 µm，并将每类型最少细胞数改为 3、10、20：
Proseg 的 Δ 范围为 −0.0053 至 −0.0018；StarDist 为 +0.0274 至 +0.0339，方向
保持一致。当每类型至少 20 个细胞时，Proseg bootstrap 区间轻微跨过 0，进一步
说明其变化接近零；StarDist 的正向提升仍稳定。

> 配对分析只纳入两套分割注释一致的细胞，代表“分割稳定细胞”而非窗口内全部
> 细胞；这提高了比较的内部可比性，但不应外推到 segmentation-discordant 细胞。

---

## 4. 考虑质心距离的空间 CellChat 与单细胞参考

参照 ovarian 分析，使用 **CellChat v2.2.0** 的空间模式，并设置
`distance.use=TRUE`、`interaction.range=250 µm`、`contact.range=100 µm`、
`nboot=20`、随机种子 42。坐标直接使用分割 mask 的物理质心。每套分割内五种
方法固定使用同一批 RAW-RCTD `first_type` 细胞；少于 10 个细胞的类型在运行前
统一排除：Proseg 为 **3,924 个细胞、14 类**，StarDist 为 **2,290 个细胞、
14 类**。CRC 的肿瘤类型定义为 Pelka `ClusterMidway=EpiT`，正常上皮 `Epi` 不
归入肿瘤。

Pelka 平衡参考的 9,781 个单细胞也用同一版本的 R CellChat 和 CellChatDB.human
运行，但不使用空间距离。空间与单细胞互作均按
`(source, target, ligand, receptor)` 唯一键比较，显著阈值为 `p<0.05`。

### 4.1 空间通讯数量与概率

| 分割 | 方法 | 显著互作 | 总概率 | 自分泌概率 | EpiT 相关概率 |
|------|------|---------:|-------:|-----------:|--------------:|
| Proseg | RAW | 1,167 | 0.2420 | 0.0597 | 0.0648 |
| Proseg | **SPARKLE** | **1,156（−0.9%）** | **0.2215（−8.5%）** | **0.0571（−4.3%）** | **0.0590（−8.9%）** |
| Proseg | SoupX | 1,165 | 0.2331 | 0.0581 | 0.0619 |
| Proseg | SpatialSoupX | 14,977 | 4.7473 | 1.0697 | 0.7377 |
| Proseg | DecontX | 0 | 0 | 0 | 0 |
| StarDist | RAW | 166 | 0.05045 | 0.01700 | 0.02690 |
| StarDist | **SPARKLE** | **159（−4.2%）** | **0.03401（−32.6%）** | **0.00855（−49.7%）** | **0.01258（−53.2%）** |
| StarDist | SoupX | 166 | 0.04919 | 0.01662 | 0.02594 |
| StarDist | SpatialSoupX | 5,908 | 1.5508 | 0.2950 | 0.3746 |
| StarDist | DecontX | 0 | 0 | 0 | 0 |

SPARKLE 保留了绝大多数显著互作，但降低通讯概率；StarDist 的下降明显大于
Proseg。SoupX 仍接近 RAW；SpatialSoupX 在两套分割分别把显著互作放大约
12.8 倍和 35.6 倍；DecontX 没有任何过表达配体–受体对，二者都不宜解读为
可靠的通讯恢复。

不过，**通讯概率下降本身不是改善证据**。例如 StarDist 中下降最大的两项
EpiT→EpiT 信号是 `CEACAM5→CEACAM6` 和 `CEACAM6→CEACAM6`，它们在 Pelka
单细胞参考中均为显著互作。因此不能把 StarDist 的 −53.2% 肿瘤相关概率全部
归因于去除 ambient 假信号。

### 4.2 单细胞参考的互作集合验证

Pelka 参考中有 17,580 个可检验互作，其中 15,119 个在 `p<0.05` 下显著。下面
以每套分割的 RAW 显著互作为锚，区分“单细胞参考阳性”和“单细胞参考阴性”：

| 分割/范围 | RAW TP / FP | SPARKLE TP / FP | RAW-TP 保留 | RAW-FP 去除 | 新增参考阴性 |
|-----------|------------:|-----------------:|------------:|------------:|-------------:|
| Proseg / 全部 | 786 / 14 | 786 / 13 | 781/786（99.36%） | 1/14（7.14%） | 0 |
| Proseg / EpiT 相关 | 246 / 5 | 246 / 5 | **246/246（100%）** | 0/5 | 0 |
| StarDist / 全部 | 120 / 8 | 120 / 8 | **120/120（100%）** | 0/8 | 0 |
| StarDist / EpiT 相关 | 61 / 4 | 61 / 4 | **61/61（100%）** | 0/4 | 0 |

这里 SPARKLE 的当前 TP 数可以与 RAW-TP 保留数不同：Proseg 全部范围丢失 5 个
RAW-TP，同时新增 5 个参考阳性互作，因此总 TP 仍为 786。更重要的是，RAW 的
参考阴性互作只有 14 个和 8 个，分母太小；CRC 不能复现 ovarian 中“去除约 23%
假阳性”的强结论。能可靠支持的是：**没有新增参考阴性互作，且 EpiT 相关的
RAW 参考阳性互作在两套分割中全部保留。**

### 4.3 单细胞参考的通讯概率排序

空间距离核与无空间单细胞结果的绝对概率不可直接比较，因此只比较排序。对
“RAW 显著且单细胞可检验”的固定互作集合，将校正后消失的互作记为概率 0，
计算 Spearman 相关；95% 区间来自按 source-target 细胞类型对分层的 5,000 次
配对 bootstrap。

| 分割/范围 | RAW | SPARKLE | Δ | bootstrap 95% 区间 |
|-----------|----:|--------:|--:|---------------------:|
| Proseg / 全部（n=800） | 0.6126 | 0.5938 | −0.0187 | [−0.0430, +0.0021] |
| Proseg / EpiT 相关（n=251） | 0.6668 | 0.6754 | **+0.0086** | **[+0.0022, +0.0167]** |
| StarDist / 全部（n=128） | 0.5863 | 0.6145 | +0.0282 | [−0.0000, +0.0609] |
| StarDist / EpiT 相关（n=65） | 0.6664 | 0.7123 | +0.0459 | [−0.0119, +0.1089] |

因此 CRC 的空间通讯结论是**局部、有限改善，而非全局明确改善**：两种分割的
EpiT 相关排序都向单细胞参考靠近，但只有 Proseg 的区间明确高于 0；StarDist
方向为正但互作数较少、区间跨 0。Proseg 全部互作没有改善，StarDist 全部互作
仅呈边界正向趋势。结合 4.2，最稳妥的表述是 SPARKLE 基本保留单细胞支持的
肿瘤通讯，并改善部分肿瘤互作的相对排序，但没有充分证据把整体强度下降都称为
去除假阳性。

> 单细胞参考并非绝对真值：它混合多个 donor，且组织解离会丢失空间依赖信号。
> 另外，Proseg 仅 800/1,167、StarDist 仅 128/166 个 RAW 空间显著互作能在参考
> 的测试集合中评估；其余互作不被计作假阳性或真阳性。

---

## 5. 总体结论

1. **窗口与基因覆盖**：在 800×800 µm、160,064 spots、18,085 全基因上完成了
   Proseg 与 StarDist 的五方法平行评估。
2. **RCTD**：普通 shared-cell 汇总中 SPARKLE 在两种分割都降低 doublet；进一步
   固定同一批物理细胞后，Proseg 的 margin 和 singlet 分类均明确改善，StarDist
   仅有较小的 margin 改善，singlet 比例变化未达显著。
3. **单细胞保真**：Proseg 中 SPARKLE 与 RAW 基本持平（−0.0007），StarDist 中
   提升 +0.0284；两种分割中类型间相关都下降，方向一致。
4. **分割敏感性**：RAW singlet、可用细胞数、cell/empty spot 比例以及方法内部
   参数均随分割明显变化。双条件报告是必要的，不能只展示一套 segmentation。
5. **空间通讯**：SPARKLE 在两套分割中保留全部 RAW 的 EpiT 相关参考阳性互作，
   且肿瘤互作概率排序向单细胞参考靠近；但只有 Proseg 的排序增量区间明确高于
   0，参考阴性分母很小，不能声称 CRC 的整体通讯已明确改善。
6. **baseline 解释**：SoupX 成本高而变化小；DecontX 的参考一致性损失和细胞
   退出提示激进校正；SpatialSoupX 的高 singlet/高 reference 相关不能脱离低保留率
   和更高类型间相关单独解读。

---

## 6. 结果位置

- 五方法 h5ad：`evaluation/reports/h5ad/`
- RCTD：`evaluation/reports/rctd_crc/{proseg,stardist}/`
- 固定 RAW 注释 h5ad：
  `evaluation/reports/h5ad_crc_{proseg,stardist}_annotated_x14200-15000_y2750-3550/`
- 单细胞相关、热图与汇总：
  `evaluation/reports/crc_eval_full/{proseg,stardist}/method_level_new/`
- 严格配对同细胞/共同 R² 基因结果：
  `evaluation/reports/crc_eval_paired_shared_r2/`
- 严格配对同细胞 RCTD score 结果：
  `evaluation/reports/crc_eval_paired_rctd/`
- 空间 CellChat 与 Pelka 单细胞参考结果：
  `evaluation/reports/crc_eval_cellchat_spatial/`

中间矩阵和图表由 `.gitignore` 忽略，不纳入版本库；本报告、CRC 流程脚本和命令
示例纳入版本控制。
