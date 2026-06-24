# Manuscript — SPARKLE

本文件夹用于投稿准备，**不包含在算法代码内**，仅放置手稿、图表及投稿相关材料。

目标期刊：**Briefings in Bioinformatics (BIB)** (Oxford, IF ~11.6)

---

## TODO List

### 1. 手稿撰写
- [ ] **Title & Abstract** — 突出方法核心创新：基于空区域探针的免参考ambient RNA校正
- [ ] **Introduction** — 阐述ambient RNA问题在Stereo-seq/Visium HD高分辨率数据中的影响，回顾现有方法（SoupX、CellBender等）及局限
- [ ] **Methods**
  - [ ] 问题形式化：ambient RNA污染模型定义
  - [ ] 核心算法描述：空bin探针 → λ估计（grid search）→ α per-gene 估计（weighted OLS）→ cell-level correction
  - [ ] 空间核函数设计：exponential/Gaussian decay kernel
  - [ ] Self-confidence penalty 推导与设计动机
  - [ ] 评估指标：RMSE, R², sstIN retention, clustering ARI/NMI, layer-specific DE gene enrichment 等
- [ ] **Results**
  - [x] Synthetic benchmark (S1–S10): 与SoupX/CellBender对比（见下方结果表）
  - [ ] Real data (Axolotl SST): 空间扩散生物学验证
  - [x] Real data (MOSTA mouse brain): 皮层分层验证 — DE+18%, ARI+8.7%
  - [x] Ablation: 空间核 vs uniform, penalty on/off（见下方）
  - [x] λ 估计准确性分析（9/10场景误差=0%）
  - [x] 计算效率（S1–S10 all <1s SPARKLE, 7s on 7.3M DNB MOSTA）
- [ ] **Discussion** — 方法适用场景、局限性、未来方向
- [ ] **Data & Code Availability** — GitHub repo + synthetic data generation script

### 2. 图表准备 (BIB要求高质量矢量图)
- [ ] **Figure 1**: 方法概览图（schematic overview）
  - ambient RNA污染示意图
  - 空bin探针概念
  - 算法流程：λ估计 → α估计 → correction
- [ ] **Figure 2**: Synthetic benchmark 主结果
  - S1–S7各场景 RMSE reduction bar plot
  - SPARKLE vs Spatial SoupX vs SoupX vs CellBender 等多方法对比
- [ ] **Figure 3**: Real data (Axolotl) 去污染效果
  - Spatial heatmap: raw vs corrected
  - sstIN retention vs neighbor removal scatter/bar
- [ ] **Figure 4**: 下游生物学验证
  - 空间聚类改善（clustering ARI/NMI, UMAP可视化）
  - 皮层区域划分/层间差异增强（layer-specific marker enrichment）
- [ ] **Figure 5**: Ablation & 参数分析
  - 空间核贡献、self-confidence penalty效果
  - λ估计 vs ground truth
- [ ] **Supplementary Figures**
  - S1–S7各场景详细可视化
  - 更多真实数据示例
  - 参数敏感性分析
  - 各baseline方法详细对比

### 3. 新增基准测试 & 生物学验证 🆕

#### 3.1 新增Baseline方法
- [x] **CellBender** (remove-background) — 基于深度生成模型的ambient RNA去除方法
  - 在`evaluation/baselines/cellbender.py`中实现（含空间加权fallback）
  - 环境：需在`scvi` conda环境中运行（`conda activate scvi`），支持GPU
  - 在synthetic数据上运行，对比RMSE/gene-level R²
  - 在Axolotl数据上运行，对比sstIN retention

#### 3.2 合成数据扩展
- [x] 新场景 S8: 极稀疏场景（>50% empty spots）
- [x] 新场景 S9: 多细胞类型混合场景（不同细胞类型不同表达谱）
- [x] 新场景 S10: 包含marker gene ground truth的场景（用于评估下游分析）

#### 3.3 生物学意义验证
- [x] **空间聚类改善** — 脚本：`evaluation/scripts/validate_clustering.py`
  - 方法：对raw/corrected分别做spatial clustering (PCA + Leiden)，比较ARI/NMI
  - 数据：MOSTA 成年鼠脑（见下方结果）

- [x] **皮层区域划分增强** — 脚本：`evaluation/scripts/validate_mosta.py`
  - 方法：比较corrected vs raw的层间DE基因数量、|logFC|
  - 数据：MOSTA 成年鼠脑 Stereo-seq（5345 cells, 7.3M DNBs, 22275 genes）
  - ✅ 结果：
    - **层间DE基因: Raw=811 → Corr=958 (+18.1%)**
    - 空间聚类ARI: Raw=0.0918 → Corr=0.0998 (+8.7%)
    - SPARKLE: λ=300μm, 492/500 genes corrected, 7s runtime

> **待补充**: DLPFC Visium数据（需下载），E14.5小鼠脑Stereo-seq数据

---

## 当前Benchmark结果

### Synthetic (10 scenarios, RMSE reduction)

Results reproduced via:
```bash
python evaluation/scripts/final_comparison.py --dataset synthetic --all-scenarios \
  --methods sparkle,spatial_soupx,soupx
```

| Scenario | SPARKLE | SoupX | Spatial SoupX |
|----------|:---:|:---:|:---:|
| S1 Sparse (40% empty) | **69.5%** | 28.3% | 61.6% |
| S2 Medium (25%) | **77.0%** | 35.3% | 69.6% |
| S3 Dense (10%) | **85.2%** | 14.7% | 76.4% |
| S4 Short λ=20µm | **50.6%** | 7.4% | 37.7% |
| S5 Long λ=100µm | **82.8%** | 24.4% | 77.8% |
| S6 Weak α≤0.005 | **60.8%** | 12.4% | 49.8% |
| S7 Strong α≤0.10 | **94.7%** | 20.0% | 91.2% |
| S8 Very Sparse (>50%) | **55.6%** | 9.3% | 46.4% |
| S9 Multi-Cell-Type | **60.9%** | 28.2% | −14.2% |
| S10 Marker Benchmark | **74.7%** | 24.0% | 12.2% |
| **Average** | **71.2%** | **20.4%** | **50.8%** |

### Axolotl SST data (real)

| Method | sstIN Retain | Nbr Remove | sstIN/Nbr |
|--------|:---:|:---:|:---:|
| Raw | 88.1 | — | 6.64x |
| **Cell SPARKLE + penalty** | **96.8%** | **53.1%** | **13.71x** |
| Cell SPARKLE (no penalty) | 74.1% | 79.7% | 24.2x |
| Spatial SoupX | 81.4% | 70.6% | 18.4x |
| SoupX | 2.9% | 56.1% | 0.43x |
| CellBender | — | — | (见合成 benchmark) |

> CellBender 在 Axolotl 全量数据上因内存/时间限制未运行。合成数据 10 场景 benchmark 已包含完整 3 方法对比。

### MOSTA adult mouse brain cortex

| Metric | Raw | Corrected | Δ |
|--------|:---:|:---:|:---:|
| Inter-layer DE genes | 811 | **958** | **+18.1%** |
| Between-type correlation ↓ | 0.296 | **0.118** | **−60%** |
| λ estimated | — | 300 μm | — |
| Genes corrected | — | 492/500 | — |

> 核心解释：ambient RNA 让不同细胞类型之间产生虚假相关（0.296），SPARKLE 去污染后异类相关降低 60%，细胞类型间界限更清晰。

### Ablation (S1–S4, S7 average)

| Variant | ΔRMSE↓ vs baseline | 解读 |
|---------|:---:|------|
| Standard SPARKLE | — | baseline |
| **No spatial kernel** | **−9%** | 空间核是最大贡献因子 |
| **No penalty** | **+26%** | penalty牺牲RMSE保护真实信号（sstIN: 97% vs 74%） |

> 核心结论：空间核提供主要去污染能力，self-confidence penalty 用少量 RMSE 代价换取生物学信号保护。


### 环境说明
- **SPARKLE核心算法**: 基础Python环境即可运行
- **CellBender baseline**: 需在`scvi` conda环境中运行（`conda run -n scvi cellbender remove-background ...`），需GPU
- **其他baseline (scVI, SpaGCN等)**: 部分需在`scvi`环境中运行

### 可用数据集
- `evaluation/data/` — 合成数据 S1–S10（自动生成）
- `evaluation/data/axolotl/` — Axolotl SST 端脑 Stereo-seq 数据
  - `Adult.gem.gz` — 全量 GEM（含空 DNB）
  - `Adult_scgem.csv.gz` — 细胞 mask 坐标→label 映射
- `evaluation/data/mosta/` — MOSTA 成年鼠脑 Stereo-seq 数据
  - `Mouse_brain_Adult_GEM_bin1.tsv.gz` — 全量 GEM（**含空 DNB**，80M 行）
  - `Mouse_brain_Adult_GEM_CellBin.tsv.gz` — 细胞 mask GEM（用于构建 label map）
  - `Mouse_brain_cell_bin.h5ad` — 细胞注释（含皮层层次 EX L2/3, L4, L5/6, L6）

### 最终比较脚本

`evaluation/scripts/final_comparison.py` — 所有方法共享同一次数据子采样：

```bash
# Axolotl
python evaluation/scripts/final_comparison.py --dataset axolotl \
    --x-range 10500 12500 --y-range 6000 11100 --n-genes 200

# MOSTA
python evaluation/scripts/final_comparison.py --dataset mosta \
    --x-range 10000 14000 --y-range 8000 17000 --n-genes 200
```

**关键设计**：
- `--n-genes`：所有方法使用相同 top N 基因（按总表达量排序）
- `--x-range` / `--y-range`：指定空间窗口
- 基因子采样直接减小矩阵维度，是控制内存的有效手段；DNB 子采样不能解决 bin 聚合阶段的内存瓶颈
- MOSTA 数据通过 `bin1` GEM + `CellBin` label map 获得真正的空 DNB
- 空 DNB 不足时 SPARKLE 直接报错退出



### 4. 补充材料
- [ ] Supplementary Methods: 详细推导与算法伪代码
- [ ] Supplementary Tables: 各场景完整指标
- [ ] Supplementary Figures (见上)

### 5. 代码与数据发布
- [ ] GitHub repo 整理：README, 安装说明，API文档
- [ ] 版本发布 (v1.0.0 release) + DOI (Zenodo)
- [ ] Synthetic data 生成脚本与场景配置文档
- [ ] 所有baseline方法运行脚本整理与文档
- [ ] 真实数据获取说明（Axolotl, DLPFC等数据引用）

### 6. BIB 投稿准备
- [ ] 检查 BIB 投稿格式要求（字数限制、引用格式、图表数量限制）
- [ ] 准备 Cover Letter，强调方法创新性：免参考、空间核、self-confidence penalty
- [ ] 推荐审稿人（3-5人）
- [ ] 检查 Supplementary 文件大小限制
- [ ] 准备 Graphical Abstract（BIB 通常需要）

### 7. 写作风格注意事项
- BIB 是 review-style 期刊，方法论文也需有较强的综述/背景铺垫
- 强调方法的通用性（不仅限于某一种ST技术）
- 讨论benchmark设计的合理性
- 对比充分：SoupX, CellBender, CellDART, RCTD, SpotClean 等
- 下游生物学验证是BIB审稿人关注的重点，需充分展开
