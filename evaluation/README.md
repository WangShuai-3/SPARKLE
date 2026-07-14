# Evaluation Directory

> **本目录仅用于评测和报告，不包含算法代码。**
> 算法代码位于 `stambient/`，单元测试位于 `tests/`。

## 目录结构

```
evaluation/
├── README.md              # 本文件
├── synthetic/             # 模拟数据生成
│   ├── __init__.py
│   ├── generator.py       # 200×200 DNB 网格 + cell-based ambient RNA 注入
│   └── scenarios.py       # 预定义评测场景 S1–S10
├── scripts/               # 评测脚本
│   ├── run_benchmark.py   # 主评测流程 (合成数据) — 已集成到 final_comparison.py
│   ├── test_axolotl.py    # Axolotl 真实数据测试
│   ├── final_comparison.py# 最终方法对比（含 CRC 双 segmentation）
│   └── visualize.py       # 空间热力图生成 — 待实现
├── baselines/             # 对比方法
│   ├── soupx.py           # 原始 SoupX
│   ├── spatial_soupx.py   # Spatial SoupX (bin 级空间核)
│   └── standalone_decontx.py / decontx.py  # DecontX (per-gene contamination)
├── data/                  # 数据
│   ├── axolotl/           # Axolotl 真实数据
│   ├── mosta/             # MOSTA 成年小鼠脑 Stereo-seq
│   ├── mousebrain/        # 新加入：Stereo-seq 小鼠脑 T304 + snRNA-seq 308 clusters
│   ├── visiumhd/          # Visium HD 人结肠癌
│   ├── ovarian/           # Visium HD 人卵巢癌 (FF) + scFFPE 单细胞 + FLEX 注释
│   └── CRC/               # CRC 2-µm spots + Proseg/StarDist + Pelka scRNA reference
└── reports/               # 评测报告输出
```

## 模拟数据规格

| 参数 | 值 |
|------|-----|
| DNB 网格 | 200×200 = **40,000 DNBs** |
| DNB 间距 | 0.5 μm |
| 基因数 | 500 (其中 80 个高表达基因) |
| 细胞数 | 80–280 (因 empty_fraction 而异) |
| 细胞类型数 | 3–5 种 |
| Marker 基因比例 | 20%–50% |
| 空细胞比例 | 10%–60% |
| 真实 λ | 20 / 50 / 500 μm |
| 泄漏率 α | 0.005–0.10 |

## SPARKLE 运行参数（合成数据）

| 参数 | 值 | 说明 |
|------|-----|------|
| bin_size | 25 | bin 边长 (DNB) |
| max_radius | 300 μm | 邻居搜索半径 |
| lambda_grid | [10,20,30,50,70,100,150,200,300] | λ 候选值 |
| cell_based | True | Cell 级架构 |
| self_confidence_penalty | True | |

## 使用

```bash
# 单个合成数据场景
python evaluation/scripts/final_comparison.py --dataset synthetic --scenario S1

# 全部 10 个合成数据场景（含 DecontX）
python evaluation/scripts/final_comparison.py --dataset synthetic --all-scenarios \
    --methods sparkle,spatial_soupx,soupx,decontx

# Axoltol
python evaluation/scripts/final_comparison.py --dataset axolotl \
    --x-range 10500 12500 --y-range 6000 11100 \
    --n-genes 2000 --n-high-genes 2000 \
    --methods sparkle,spatial_soupx,soupx,decontx

# Mosta
# python evaluation/scripts/final_comparison.py --dataset mosta \
#     --x-range 10000 14000 --y-range 8000 17000 \
#     --n-genes 2000 --n-high-genes 2000 \
#     --methods sparkle,spatial_soupx,soupx,decontx

# Visium HD
python evaluation/scripts/final_comparison.py --dataset visiumhd \
    --n-genes 30000 --n-high-genes 30000 \
    --methods sparkle,spatial_soupx,soupx,decontx

# MouseBrain (T304)
python evaluation/scripts/final_comparison.py --dataset mousebrain \
    --x-range 12500 20000 --y-range 2000 10000 \
    --n-genes 30000 --n-high-genes 30000 \
    --methods sparkle,spatial_soupx,soupx,decontx

# Visium HD (人卵巢癌 FF, 内嵌 segmentation)
python evaluation/scripts/final_comparison.py --dataset ovarian \
    --x-range 1000 1800 --y-range 300 1100 \
    --n-genes 30000 --n-high-genes 30000 \
    --methods sparkle,spatial_soupx,soupx,decontx

# CRC：相同窗口下依次运行 Proseg 与 StarDist 两个 segmentation 条件
python evaluation/scripts/final_comparison.py --dataset crc \
    --crc-segmentation both \
    --x-range 14300 14900 --y-range 2850 3450 \
    --n-genes 2000 --n-high-genes 2000 --cut-genes \
    --max-radius 200 --methods sparkle,spatial_soupx

# resource (using MouseBrain)
python evaluation/scripts/benchmark_resource.py \
    --x-range 6000 20000 --y-range 2000 15000 \
    --n-genes 10000 --n-high-genes 10000 \
    --r2-threshold 0 --plot --n-runs 8
```

## 控制是否保存 h5ad

默认会保存每个方法的 corrected h5ad 到 `evaluation/reports/h5ad/`。若只需要指标而
不需要大文件，可加上 `--no-save-h5ad`：

```bash
python evaluation/scripts/final_comparison.py --dataset synthetic --scenario S1 \
    --methods sparkle --no-save-h5ad
```

## 额外可调参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--max-radius` | 空间邻域搜索半径（µm）。Axolotl 默认 200，其他数据集 300。 | 数据集相关 |
| `--cut-genes` / `--no-cut-genes` | `subsample_data` 是否只保留 top N 基因。默认 `--no-cut-genes`（保留全部基因用于评估）。 | False |

例如，在 Axolotl 上测试更大的 `lambda`（需要同时扩大 `max_radius`）：

```bash
python evaluation/scripts/final_comparison.py --dataset axolotl \
    --x-range 10500 12500 --y-range 6000 11100 \
    --n-genes 2000 --n-high-genes 2000 \
    --methods sparkle --no-save-h5ad \
    --max-radius 500 \
    --lambda-grid 200 300 500 700 1000
```

## Ovarian (Visium HD) RCTD + 单细胞比较评估

针对 Visium HD 人卵巢癌（FF）数据，仿照 MouseBrain 流程做 RCTD 细胞类型映射
以及基于 scFFPE 单细胞参考的表达评估。单细胞文件与注释位于
`evaluation/data/ovarian/`：

- `17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5` — scRNA-seq 计数
- `FLEX_Ovarian_Barcode_Cluster_Annotation.csv` — 每个 barcode 的 `Cell Annotation`（16 类）

完整流程（先跑 `final_comparison.py --dataset ovarian` 生成 h5ad 后）：

```bash
# 1. 构建单细胞类型 pseudobulk 参考 (genes x 16 cell types)
python evaluation/scripts/prepare_ovarian_scrna_reference.py

# 2. RCTD doublet 模式（每个方法）；导出 per-cell first_type 用于回填注释
conda activate r-env
RCTD_MAX_CORES=16 Rscript evaluation/scripts/run_rctd_ovarian.R
#   -> evaluation/reports/rctd_ovarian/{rctd_summary_metrics,rctd_shared_metrics,rctd_doublet_results}.csv
#   -> evaluation/reports/rctd_ovarian/first_type/{Method}_first_type.csv

# 3. 把 RCTD 注释回填进各方法 h5ad（用 RAW 的 first_type 作为共享注释，隔离校正对表达的影响）
python evaluation/scripts/inject_rctd_annotations.py --annotation-method RAW
#   -> evaluation/reports/h5ad_ovarian_annotated/

# 4. 基于单细胞参考的表达评估（复用 mousebrain 评估脚本）
python evaluation/scripts/evaluate_mousebrain_h5ad.py \
    --tag ovarian_x1000-1800_y300-1100 \
    --input-dir evaluation/reports/h5ad_ovarian_annotated \
    --snrna-ref evaluation/data/ovarian/scrna_celltype_pseudobulk.csv \
    --output-dir evaluation/reports/ovarian_eval \
    --methods RAW,SPARKLE,SpatialSoupX,SoupX,DecontX

# 5.（可选）参考 marker 定位准确率（RAW vs SPARKLE）
python evaluation/scripts/reference_marker_localization.py \
    --tag ovarian_x1000-1800_y300-1100 \
    --input-dir evaluation/reports/h5ad_ovarian_annotated \
    --output-dir evaluation/reports/ovarian_eval \
    --snrna-ref evaluation/data/ovarian/scrna_celltype_pseudobulk.csv
```

> RCTD 环境需 `spacexr`, `Seurat`, `hdf5r`（已装于 `r-env`）。RCTD 通过
> `segmentations/cell_segmentation_mask` 计算每个 cell 的 (x,y)，通过 `cell_id`
> 与 h5ad 对应。

## CRC 双 segmentation + RCTD + 单细胞 reference 评估

完整结果与条件内解释见 [`crc_tumor_analysis.md`](crc_tumor_analysis.md)。

CRC 的 `proseg` 和 `stardist` 输入共享相同的 18,085 genes × 470,416 spots
表达矩阵及坐标，只有 spot-to-cell label map 不同，因此作为两个独立条件平行比较。
注册后的全切片范围为 x=**13967.49–15338.95 µm**、
y=**2464.03–3835.49 µm**。

本次报告使用扩大的 `x=14200–15000, y=2750–3550` 窗口（800 × 800 µm）：
共有 160,064 spots 和 2,471,982 UMI，占全切片 spots 的 34.0%。Proseg
包含 4,503 cells（107,013 cell spots、53,051 empty spots），StarDist 包含
4,440 cells（30,326 cell spots、129,738 empty spots）。两个条件使用相同的
spots、表达矩阵和全部 18,085 个基因；差异只来自 segmentation label map。

```bash
# 1. 同一窗口内生成两个 segmentation 条件的 h5ad
python evaluation/scripts/final_comparison.py --dataset crc \
    --crc-segmentation both \
    --x-range 14200 15000 --y-range 2750 3550 \
    --n-genes 18085 --n-high-genes 18085 --no-cut-genes \
    --lambda-grid 10 20 30 50 70 100 150 200 --max-radius 200 \
    --methods sparkle,spatial_soupx,soupx,decontx

# SoupX/DecontX 保持仓库中的原始实现和默认参数；在全部基因上运行时，
# SoupX 的全维 KMeans 与标量后验循环是主要耗时步骤。

# 2. 构建平衡的 Pelka ClusterMidway reference（同时输出 RCTD counts 和 pseudobulk）
python evaluation/scripts/prepare_crc_scrna_reference.py --max-cells-per-type 500

# 3. RCTD doublet mode；两种条件写入独立目录，可同时启动
CRC_SEGMENTATION=proseg \
RCTD_DATASET_TAG=crc_proseg_x14200-15000_y2750-3550 \
RCTD_MAX_CORES=16 Rscript evaluation/scripts/run_rctd_crc.R

CRC_SEGMENTATION=stardist \
RCTD_DATASET_TAG=crc_stardist_x14200-15000_y2750-3550 \
RCTD_MAX_CORES=16 Rscript evaluation/scripts/run_rctd_crc.R

# 禁止本地 PSOCK 端口的沙箱/集群节点使用 RCTD_MAX_CORES=1。

# 4. 用 RAW RCTD first_type 给所有方法回填固定注释，隔离表达校正效应
python evaluation/scripts/inject_rctd_annotations.py \
    --tag crc_proseg_x14200-15000_y2750-3550 \
    --input-dir evaluation/reports/h5ad \
    --first-type-dir evaluation/reports/rctd_crc/proseg/first_type \
    --output-dir evaluation/reports/h5ad_crc_proseg_annotated_x14200-15000_y2750-3550 \
    --annotation-method RAW

python evaluation/scripts/inject_rctd_annotations.py \
    --tag crc_stardist_x14200-15000_y2750-3550 \
    --input-dir evaluation/reports/h5ad \
    --first-type-dir evaluation/reports/rctd_crc/stardist/first_type \
    --output-dir evaluation/reports/h5ad_crc_stardist_annotated_x14200-15000_y2750-3550 \
    --annotation-method RAW

# 5. 两个条件分别与同一个 ClusterMidway pseudobulk reference 做相关分析
python evaluation/scripts/evaluate_mousebrain_h5ad.py \
    --tag crc_proseg_x14200-15000_y2750-3550 \
    --input-dir evaluation/reports/h5ad_crc_proseg_annotated_x14200-15000_y2750-3550 \
    --snrna-ref evaluation/data/CRC/scrna_reference/prepared_cluster_midway/crc_cluster_midway_pseudobulk.csv \
    --output-dir evaluation/reports/crc_eval_full/proseg \
    --methods RAW,SPARKLE,SpatialSoupX,SoupX,DecontX

python evaluation/scripts/evaluate_mousebrain_h5ad.py \
    --tag crc_stardist_x14200-15000_y2750-3550 \
    --input-dir evaluation/reports/h5ad_crc_stardist_annotated_x14200-15000_y2750-3550 \
    --snrna-ref evaluation/data/CRC/scrna_reference/prepared_cluster_midway/crc_cluster_midway_pseudobulk.csv \
    --output-dir evaluation/reports/crc_eval_full/stardist \
    --methods RAW,SPARKLE,SpatialSoupX,SoupX,DecontX

# 6. 严格配对比较：跨分割相同物理细胞 + 两套分割共同通过 R² 的基因
# 默认用 5-µm 双向最近邻、RAW-RCTD 共识注释和 1,000 次配对 bootstrap。
python evaluation/scripts/analyze_crc_paired_shared_r2.py

# 7. 在上述相同物理细胞中配对比较 RAW/SPARKLE 的 RCTD score
# 仅纳入 Proseg/StarDist × RAW/SPARKLE 四份 RCTD 结果都存在的细胞；默认执行
# 10,000 次按共识细胞类型分层的配对 bootstrap。主要指标 score margin 越低，
# 表示 RCTD 越偏向 singlet；跨 segmentation 的绝对 score 仅作描述。
python evaluation/scripts/analyze_crc_paired_rctd_scores.py
```

CRC 注册坐标带有小角度旋转。loader 用注册后的物理坐标筛选窗口并保存 cell
centroid，同时从 `bin_id` 恢复严格的 2-µm 网格供 SPARKLE 分箱；刚体变换保持
距离不变，并避免浮点旋转使 pitch 推断接近零。
