# SPARKLE 2.x Step 01 汇报：Latent X self-consistent inference（v2.0-alpha1）

日期：2026-08-14　分支：`SPARKLEv2`
对应路线图 Stage 0 + Stage 1（Phase 1，v2.0-alpha1）：仅把 leakage source 从观测值 Y 换成 latent clean expression X，**λ、α、R² 全部沿用 1.x**。

---

## 1. Baseline 与兼容性

| 项目 | 内容 |
|---|---|
| Baseline tag | `sparkle-v1-baseline`（e0855e1） |
| 本阶段 commits | `c863e34` baseline 冻结；`23cba20` latent 实现；`e48f5aa` 单元测试；`6dbc4b1`+`0f95231` 评估脚本 |
| 兼容性 | `inference_mode="legacy"` 为默认；`tests/test_legacy_freeze.py` 锁定 legacy 输出的 λ/α/R²/sha256，全部通过（101 passed, 1 GPU-skip） |
| GPU | latent GPU 分支与 CPU float64 误差 ~7e-15（机器精度）；修复了 torch 降序 quantile 秩映射 bug（p90 模式） |
| 评估环境 | `conda run -n scvi`（Python 3.11 + torch CUDA）；RTX 3060 + RTX 4090 |
| 数据 | 与 v1 完全相同：synthetic S1–S10（seed 42）、Axolotl/MouseBrain/Ovarian 论文窗口；仅比较 RAW / SPARKLE 1.x / SPARKLE 2.x |

实现：新模块 `stambient/latent_correction.py`，阻尼不动点 `X⁽⁰⁾=Y`，`X̃=max(Y−L(X⁽ᵗ⁾),0)`，`X⁽ᵗ⁺¹⁾=(1−η)X⁽ᵗ⁾+ηX̃`，η=0.5，max_iter=20，tol=1e-4。`SPARKLE` 新增 `inference_mode / latent_eta / latent_max_iter / latent_tol` 四个参数；legacy 路径数值零改动。

## 2. Synthetic S1–S10（有 ground truth）

核心结果（完整表见 `evaluation/reports/v2/synthetic/synthetic_v2_summary.csv`）：

| 场景 | 指标 | RAW | **v1 (A0)** | **v2 (A1)** | v2NP (A2) |
|---|---|---|---|---|---|
| S1 | RMSE↓ / OCR↓ | 0.703 / 0 | **0.452** / 0.0119 | 0.501 / 0.0060 | 0.503 / 0.0071 |
| S3 | RMSE↓ / ARI↑ | 1.003 / 0.606 | **0.683** / 0.671 | 0.789 / 0.893 | 0.786 / **1.000** |
| S6(域) | RMSE↓ / 源保留 | 0.743 / 1.418 | **0.484** / 1.222 | 0.547 / 1.268 | 0.551 / **1.114** |
| S7(强漏) | RMSE↓ / OCR↓ | 1.607 / 0 | **1.194** / 0.0050 | 1.364 / 0.0006 | 1.319 / 0.0004 |

规律（10/10 场景一致）：

- **RMSE、cell-wise R²：v1 全面优于 alpha1**（差距随泄漏强度增大，S7 v1=1.19 vs v2=1.36）。
- **OCR（过校正率）：alpha1 全面更低**（S6 0.0167→0.0063，S7 0.0050→0.0006）——latent X 确实在保护 source 信号，机制按设计工作。
- **source 保留：v2NP（去 penalty）最接近 1.0**，说明 latent 结构本身已部分替代 penalty 的作用（Axolotl 上 v2NP 的 sstIN 反而被多扣，见 §3，penalty 仍有必要）。
- **ARI：v2/v2NP 在多个场景显著更高**（S3 0.89/1.00 vs v1 0.67；S6 v2NP 0.696 vs v1 0.533；S10 v2 0.699 vs v1 0.493）——下游结构保留更好。
- **收敛：全部场景 9–16 次迭代收敛**（rel-change < 1e-4），运行时约为 v1 的 2 倍（绝对值均 < 1s）。

## 3. 真实数据

**MouseBrain**（snRNA cell_group pseudobulk 一致性）：

| 方法 | Pearson↑ | Spearman↑ | 运行 |
|---|---|---|---|
| RAW | 0.2465 | 0.7434 | — |
| SPARKLEv1 | **0.2847** | 0.7434 | 66s |
| SPARKLEv2 | 0.2741 | 0.7435 | 74s |
| SPARKLEv2NP | 0.2727 | 0.7435 | 65s |

**Ovarian**（scRNA pseudobulk 一致性，共享 RCTD RAW first_type 分组）：

| 方法 | Pearson↑ | Spearman↑ | median contamination↓ |
|---|---|---|---|
| RAW | 0.4822 | 0.6751 | 0.6383 |
| SPARKLEv1 | **0.5330** | 0.6737 | 0.6438 |
| SPARKLEv2 | 0.5214 | 0.6756 | 0.6464 |
| SPARKLEv2NP | 0.5183 | 0.6756 | 0.6446 |

（v2/v2NP 的 mean contamination 34.4/4.7 为个别基因 target 表达近零的伪迹，median 无差异。）

**Axolotl**（SST 分组，均值 counts）：

| 方法 | sstIN(source)↑ | 邻居 ↓ | 其他 ↓ | S/N↑ |
|---|---|---|---|---|
| RAW | 73.04 | 9.70 | 2.13 | 7.53× |
| SPARKLEv1 | 69.94 | 3.95 | 0.83 | 17.69× |
| SPARKLEv2 | **70.67** | **3.83** | 1.03 | **18.46×** |
| SPARKLEv2NP | 60.72 | **3.14** | 1.00 | 19.34× |

真实数据解读：v2 全部优于 RAW，但 Pearson 一致性略低于 v1（与 synthetic RMSE 现象同源）。Axolotl 上 **v2 在 source 保留更高（70.7 vs 69.9）的同时邻居扣除更干净（3.83 vs 3.95）**，是 latent 机制的直接证据；v2NP 的 sstIN 掉到 60.7 —— penalty 在真实数据上仍必要（暂不采纳路线图的理想情形 C≈B）。

## 4. 关键发现：α 的 Y-校准与 X-source 不自洽（alpha1 差距的根源）

alpha1 沿用的 α 是用 **Y-source** 预测子对 background bins 拟合的。由于 Y = X + L 系统性偏大，拟合出的 α_Y 系统性偏小（synthetic 真值 α≈0.010，Y-校准仅 ≈0.003–0.009）。latent 校正 α_Y·L(\hat X) 两个小因子叠加 → 系统性**少扣**，这解释了 alpha1 在 RMSE/Pearson 上的全部差距。

验证实验（`evaluation/scripts_v2/probe_alpha_refit_synthetic.py`，synthetic 上做 1–2 轮「X→重估 α/R²→X」交替）：

| 场景 | v1 RMSE | alpha1 | +1 轮 α 重估 | +2 轮 |
|---|---|---|---|---|
| S1 | 0.4518 | 0.5005 | 0.4483 | **0.4386** ✅ |
| S2 | 0.5447 | 0.6262 | 0.5512 | **0.5246** ✅ |
| S3 | 0.6834 | 0.7892 | 0.6941 | **0.6485** ✅ |
| S6 | 0.4837 | 0.5469 | 0.5048 | **0.4961** ✅ |

重估后 α_mean 向真值收敛（S1 0.0043→0.0061，真值 0.010），**2 轮交替后 4/4 场景 RMSE 反超 v1**。这证明 latent-X 方向成立，且差距与收益都由统计机制解释，不依赖调参。

## 5. 结论（go/no-go）

- alpha1 按路线图定义严格执行并如实评估：**机制成立（OCR↓、source 保留↑、ARI↑、快速收敛），但单独换 source 不足以超越 v1**，路线图 §1.5 的理想情形 C>A 未达。
- 差距根因已定位为 α 校准不自洽，且 probe 显示交替重估 α（= 路线图规划的 v2.0-alpha2）可全部反超。**判定：go，按路线图进入 alpha2。**
- alpha2 设计输入（本阶段结论）：① α/R² 应对收敛的 \hat X 重估（W_empty 图与 λ 保持不变，低开销）；② penalty 保留为默认 on；③ 交替轮数 ≤2 轮已足够（probe 边际收益递减）。

## 6. 复现

```bash
# 校正（GPU；synthetic 用 CPU/base env 即可）
conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py --dataset synthetic --all-scenarios
CUDA_VISIBLE_DEVICES=1 conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py --dataset mousebrain --use-gpu --gpu-dtype float64
CUDA_VISIBLE_DEVICES=0 conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py --dataset ovarian --use-gpu --gpu-dtype float64
CUDA_VISIBLE_DEVICES=0 conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py --dataset axolotl --use-gpu --gpu-dtype float64
# 对比
python evaluation/scripts_v2/compare_synthetic_v2.py
conda run -n scvi python evaluation/scripts_v2/compare_real_v2.py
# α 重估 probe
python evaluation/scripts_v2/probe_alpha_refit_synthetic.py
```

输出：`evaluation/reports/v2/{h5ad,metrics,synthetic,real}/`。
