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
│   ├── spotclean.py       # SpotClean cell-only / cell + empty-bin 两种适配
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

### 官方 SpotClean R 包装器

`evaluation/scripts/run_spotclean_official.py` 将分割细胞聚合为 tissue
spots，将固定网格内的 empty DNB 聚合为 background spots；两类 spot 都使用
各自质心，计数保持原始求和、不做 exposure 缩放。Python 随后调用独立脚本
`evaluation/scripts/run_spotclean_official.R`，统计估计与 EM 校正完全由官方
Bioconductor 包的 `SpotClean::createSlide()` 和 `SpotClean::spotclean()` 完成。

真实数据默认复现 Axolotl、MouseBrain、Ovarian 的 final-comparison 窗口及
候选半径。结果命名为 `*_SpotCleanOfficial.h5ad`，避免和早期 Python 近似实现
混淆。官方实现会构造 all-spot 的稠密距离/核矩阵；驱动脚本会记录内存估计，
并默认跳过明显不安全的窗口（仅可通过 `--force-memory` 强制执行）。

```bash
python evaluation/scripts/run_spotclean_official.py --overwrite
```

合成数据还可直接复用已有 h5ad，按细胞计算预测表达与无污染真值在全部
500 个基因上的 Pearson `r²`。该指标对整体倍数缩放不敏感，用于补充 RMSE
的绝对计数误差视角；输出包含细胞级 CSV、场景/方法汇总和两张比较图，并将
mean、median、四分位数及有效细胞数写回 metrics JSON：

```bash
python evaluation/scripts/evaluate_synthetic_cell_r2.py
```

```bash
# 单个合成数据场景
python evaluation/scripts/final_comparison.py --dataset synthetic --scenario S1

# 全部 10 个合成数据场景（含 SpotClean 两种输入方案）
python evaluation/scripts/final_comparison.py --dataset synthetic --all-scenarios \
    --methods sparkle,spatial_soupx,soupx,decontx,spotclean,spotclean_bg

# CPU/GPU 数值一致性与运行时间检查（无 GPU 时验证 fallback）
conda run -n scvi python evaluation/scripts/compare_sparkle_cpu_gpu.py

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
    --r2-threshold 0 --plot --n-runs 8 \
    --backends cpu gpu --require-gpu \
    --gpu-dtype float64 \
    --output evaluation/reports/resource_benchmark_mousebrain_cpu_gpu_final.csv
```

合成数据中同时评估两种 SpotClean 适配：

- `spotclean`（结果名 `SpotClean`）：每个分割细胞视为一个 tissue
  spot，空 DNB/bin 完全不进入。由于没有 background spots 时全局
  bleeding/distal rate 不可辨识，两者默认固定为 0.10（可通过
  `--spotclean-bleed-rate` 和 `--spotclean-distal-rate` 修改）。
- `spotclean_bg`（结果名 `SpotClean-bg`）：将细胞矩阵与 empty-bin
  矩阵拼接，细胞作为 tissue/source spots，empty bins 作为 background
  receiver spots，用总计数估计 bleeding rate、distal rate 和 Gaussian
  bandwidth，再用 EM 将计数重分配回细胞。默认 empty bin 边长为 25 µm
  （`--spotclean-empty-bin-size`）。为满足 SpotClean 各 spot 捕获曝光可比的假设，
  empty-bin 计数按“中位细胞 DNB 面积 / bin 的 empty-DNB 数”归一化。
  若要完全按原始计数矩阵直接拼接，使用
  `--no-spotclean-normalize-empty-exposure`。

两者都使用 SpotClean 的 Gaussian/uniform swapping 模型和 EM 表达重分配，
但是独立 Python 适配，不是对官方 R 包的调用。metrics JSON 会明确保存
`background_used`、参数识别方式和 empty-bin 曝光归一化信息。该对比只修改
评估基线和脚本，不修改 `evaluation/synthetic/generator.py` 或任何模拟场景参数。

若只需为已完成 final comparison 的真实数据窗口补充 cell-only
SpotClean h5ad，而不重跑或覆盖其他方法和 metrics：

```bash
python evaluation/scripts/run_spotclean_real.py \
    --datasets axolotl mousebrain ovarian
```

脚本固定使用已有的三个 final-comparison 窗口，直接复用已有 RAW
h5ad 的细胞、基因、计数和顺序，只从 segmentation/scGEM 恢复细胞质心。
为避免 MouseBrain 27,681 个细胞的全连接矩阵，Gaussian 局部项使用
32-nearest-neighbor 稀疏图，uniform distal 项保持精确；使用 top 50 高计数
基因选 bandwidth，然后对全部基因执行 3 次分批 EM。输出仅新增
`evaluation/reports/h5ad/*_SpotClean.h5ad`。

资源比较默认对每个空间窗口依次运行 CPU 和 GPU，并在 CSV 中保存
`cpu_runtime_sec`、`gpu_runtime_sec`、`gpu_speedup`、两种后端的 host RSS，
以及 `gpu_peak_allocated_mb` 和 `gpu_peak_reserved_mb`。CSV 还包含
`gpu_gene_batch_size`、`gpu_dtype`、`gpu_sparse_format`、两张空间图的 nnz，
以及 `*_cell_aggregation_sec`、`*_empty_bin_assignment_sec`、
`*_empty_expression_aggregation_sec`、`*_empty_graph_build_sec`、
`*_lambda_search_sec`、`*_alpha_estimation_sec`、`*_correction_sec` 等阶段耗时。
其中 allocated 表示张量实际占用，reserved 表示 PyTorch CUDA allocator 向驱动
保留的显存，通常后者更接近运行任务需要预留的显存容量；allocated 是 reserved
的一部分，二者不能相加。memory 图将资源池分为三个 panel：CPU/GPU 运行的 host
RAM 绝对 RSS 峰值、相对运行前基线的 host RAM 增量，以及仅 GPU 才有的 device
VRAM allocated/reserved，避免把系统内存和显存当成同一种可互换资源。
`--plot` 会额外生成 runtime、GPU speedup 和 memory 三类图。只测一种后端可使用
`--backends cpu` 或 `--backends gpu`；在 GPU
节点或 CI 中建议加 `--require-gpu`，避免自动回退被误当成 GPU 基准。
如果已有相同窗口和参数的 CPU CSV，可通过 `--cpu-baseline` 复用；脚本会严格
检查每个 run 的窗口、DNB 数、cell 数、empty DNB 数和 gene 数，再计算 GPU
speedup，避免重复运行耗时较长的 CPU 基准。

GPU 使用一次构建/一次传输的距离 CSR 图；各个 lambda 只在设备上更新权重。
`--gpu-gene-batch-size` 缺省时根据空闲显存自动选择。`--gpu-dtype float64`
用于严格 CPU/GPU 对照；`mixed` 和 `float32` 是近似快速模式，建议同时运行
`compare_sparkle_cpu_gpu.py --gpu-dtype ... --require-gpu` 检查目标规模的误差。

RTX 4090 完整实测结果见根目录 README。empty-DNB 分箱已由逐 bin 全量布尔扫描
改为一次向量化映射：8 个窗口的分箱总时间从 1,768.9s 降至 20.9s（减少
98.8%）。相同新代码下，CPU 总时间为 412.6s，GPU strict-float64 总时间为
182.8s，GPU 总体加速 2.26×；8 个窗口的 λ 与 corrected-gene 数均保持一致。
最大 allocated/reserved 显存仍为 8,261.8/10,798 MiB，这是自适应大 batch
用显存换取吞吐的结果。

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

# 6. Spatial CellChat v2；先做跨方法零文库/有限值预检，再完整重算
/home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
    evaluation/scripts/run_cellchat_spatial_ovarian.R --preflight-only
/home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
    evaluation/scripts/run_cellchat_spatial_ovarian.R
#   -> ovarian_eval/cellchat_spatial/cellchat_spatial_{input_qc,cell_qc}.csv
#   -> ovarian_eval/cellchat_spatial/{Method}_cellchat_spatial.csv
#   -> ovarian_eval/cellchat_spatial/cellchat_spatial_summary.csv

# 7. 从已验证的四方法结果重绘 Figure 05/05b（interaction n 动态计算）
python evaluation/scripts/plot_ovarian_spatial_cellchat.py
#   -> ovarian_eval/figures/fig05_spatial_cellchat_v2.{png,pdf,svg}
#   -> ovarian_eval/figures/fig05b_cellchatv2_average_strength.{png,pdf,svg}
```

> RCTD 环境需 `spacexr`, `Seurat`, `hdf5r`（已装于 `r-env`）。RCTD 通过
> `segmentations/cell_segmentation_mask` 计算每个 cell 的 (x,y)，通过 `cell_id`
> 与 h5ad 对应。`final_comparison.py` 会把每个方法的零文库统计写入 metrics JSON；
> Spatial CellChat 会统一剔除任一方法中的零文库细胞，并在归一化前后拒绝 NaN/Inf，
> 避免把不可估计的流程失败写成全零生物学结果。

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

# 8. 考虑分割细胞质心距离的空间 CellChat v2；两套分割分别运行，五种方法在
# 各分割内固定使用 RAW-RCTD first_type 和完全相同的细胞集合。
CRC_SEGMENTATION=proseg \
  /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
  evaluation/scripts/run_cellchat_spatial_crc.R

CRC_SEGMENTATION=stardist \
  /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
  evaluation/scripts/run_cellchat_spatial_crc.R

# 9. 将空间显著配体-受体互作与同一 R CellChat 生成的 Pelka 单细胞参考比较；
# 同时对固定 RAW 互作集合执行 5,000 次分层配对 bootstrap。
python evaluation/scripts/analyze_crc_spatial_cellchat_vs_scrna.py
```

CRC 注册坐标带有小角度旋转。loader 用注册后的物理坐标筛选窗口并保存 cell
centroid，同时从 `bin_id` 恢复严格的 2-µm 网格供 SPARKLE 分箱；刚体变换保持
距离不变，并避免浮点旋转使 pitch 推断接近零。
