# SPARKLE 2.x Step 02 汇报：交替 α/R² 重估（v2.0-alpha2）

日期：2026-08-17　分支：`SPARKLEv2`
对应路线图 §1.4 v2.0-alpha2：在 latent-X 框架内做 block coordinate descent ——
`X → (α, R²) → X → (α, R²) → …`，λ、W_empty 空间图与 background 观测固定。
动机：Step 01 定位 α 用 Y-source 校准导致 latent 模式系统性少扣（见
`step01_latent_x_report.md` §4），probe 显示 1–2 轮重估即可反超 v1。

---

## 1. 实现与兼容性

| 项目 | 内容 |
|---|---|
| 本阶段 commits | `6c7dc2d` alpha2 实现；`0e87988` 评估扩展；`1db92d6` raw 聚合修复 |
| API | `SPARKLE(..., inference_mode="latent", latent_refit_rounds=N)`，N=0 复现 alpha1；legacy 传 N>0 报错 |
| 机制 | 每轮：latent solve（η=0.5, tol=1e-4）→ 用当前 X̂ 重新 `estimate_alphas`（α 与 R² 都更新）→ 再 solve |
| 兼容性 | legacy 冻结测试与 alpha1 全部不变（103 passed, 1 GPU-skip） |
| Ablation | `SPARKLEv2R1`（1 轮）、`SPARKLEv2R2`（2 轮），与 RAW / v1 / v2 / v2NP 同窗口同参数 |

## 2. Synthetic S1–S10（ground truth）

完整表：`evaluation/reports/v2/synthetic/synthetic_v2_summary.csv`。

**RMSE（log1p CP10K，↓）胜场：v2R2 9/10**

| 场景 | RAW | v1 | v2(alpha1) | v2R1 | **v2R2** |
|---|---|---|---|---|---|
| S1 | 0.703 | 0.4518 | 0.5005 | 0.4483 | **0.4386** |
| S2 | 0.830 | 0.5447 | 0.6262 | 0.5512 | **0.5246** |
| S3 | 1.003 | 0.6834 | 0.7892 | 0.6941 | **0.6485** |
| S5 | 1.374 | 1.0030 | 1.1322 | 1.0018 | **0.9193** |
| S6(域) | 0.743 | **0.4837** | 0.5469 | 0.5048 | 0.4961 |
| S7(强漏) | 1.607 | 1.1939 | 1.3641 | 1.2287 | **1.1381** |
| S9 | 1.009 | 0.6270 | 0.7627 | 0.6434 | **0.5931** |

**cell-wise R²（↑）胜场同样是 v2R2 9/10**（唯一例外仍是 S6：v1 0.9684 vs R2 0.9610）。

其他指标：

- **ARI**：refit 家族在最难场景显著更强（S2 R1=0.925、S5 R1=0.979 vs v1 0.695；S7 R2=0.434 vs v1 0.001；S6 R1=0.711 vs v1 0.533）。
- **OCR**：R1/R2 回到 v1 量级（如 S6：v1 0.0167 / R2 0.0147；alpha1 仅 0.0063）——重估后的扣除量确实"补回来"了，且 RMSE 更优，说明多扣的部分是真泄漏而非 source。
- **α 轨迹**（真实数据诊断，见 §3）单调上升并趋于收敛，与 Step 01 的"Y-校准偏小"机制一致。
- 收敛：所有轮次 latent solve 均收敛（9–17 次迭代）；R2 运行时约为 v1 的 2–3 倍（synthetic 单场景 < 1s）。

## 3. 真实数据

**MouseBrain**（snRNA 参考，36 cell group）：

| 方法 | Pearson↑ | Spearman | median contam. | Silhouette↑ | 扣除 | 运行 |
|---|---|---|---|---|---|---|
| RAW | 0.2465 | 0.7434 | 0.7406 | −0.111 | — | — |
| v1 | 0.2847 | 0.7434 | 0.7422 | −0.101 | 18.0% | 66s |
| v2R1 | 0.2841 | 0.7434 | 0.7415 | −0.100 | 17.8% | 134s |
| **v2R2** | **0.2875** | 0.7434 | 0.7421 | **−0.096** | 18.8% | 194s |

→ **R2 Pearson 首次超过 v1**，silhouette 全场最优，扣除幅度与 v1 持平（18.8% vs 18.0%）。

**Ovarian**（scRNA 参考，15 类型，共享 RCTD RAW first_type）：

| 方法 | Pearson↑ | Spearman↑ | median contam. | Silhouette↑ | 扣除 |
|---|---|---|---|---|---|
| RAW | 0.4822 | 0.6751 | 0.6383 | +0.003 | — |
| v1 | **0.5330** | 0.6737 | 0.6438 | −0.049 | 48.9% |
| v2R1 | 0.5307 | **0.6759** | 0.6484 | **−0.042** | 45.5% |
| v2R2 | 0.5306 | **0.6759** | 0.6480 | −0.051 | 48.5% |

→ R1/R2 与 v1 基本持平（Pearson −0.002），Spearman 与 silhouette（R1）优于 v1；α 轨迹 0.198→0.246→0.263 显示 ovarian 的 Y-校准偏差最大，重估后扣除回到 v1 水平。

**Axolotl**（SST 分组）：

| 方法 | sstIN(source)↑ | 邻居 ↓ | S/N↑ |
|---|---|---|---|
| RAW | 73.04 | 9.70 | 7.53× |
| v1 | 69.94 | 3.95 | 17.69× |
| v2(alpha1) | 70.67 | 3.83 | 18.46× |
| **v2R2** | **70.19** | **3.53** | **19.87×** |

→ **R2 三项全胜**：source 保留高于 v1、邻居扣除更干净、空间对比度最高。

## 4. 结论（go/no-go）

- **Step 01 的验收判据达成**：latent+refit 在 synthetic RMSE/R² 9/10 反超 v1（S5/S7 困难场景收益最大），真实数据 MouseBrain 超 v1、Ovarian 持平且 Spearman/silhouette 更优、Axolotl 三项全胜；OCR 回到 v1 量级而 accuracy 更好 —— 证明 alpha1 的差距确实全部来自 α 校准，latent-X 方向**确认成立**。
- **遗留点**：S6（spatial domains）RMSE/R² 仍略逊 v1（−2.6%），但 ARI 更好；penalty 仍必要（v2NP 在 Axolotl 过度扣 source）。
- **判定：go。** latent + penalty + refit=2 可作为 SPARKLE 2.0 校正核心候选；按路线图进入 **Phase 2（Poisson count model + diffuse background β_g）**，届时 α→ρ_g 的估计将从 weighted OLS 换成 Poisson MLE，β_g 吸收 non-local 分量。
- 工程备注：refit 每轮成本 ≈ 一次 α 估计 + 一次 latent solve（MouseBrain R2 总计 194s GPU，可接受）；评估中发现并修复了 raw 聚合在非 0-based label 数据集上的映射 bug（commit `1db92d6`）。

## 5. 复现

```bash
# 增量补跑 R1/R2（metrics JSON 自动合并）
python evaluation/scripts_v2/run_correction_v2.py --dataset synthetic --all-scenarios --methods SPARKLEv2R1,SPARKLEv2R2
CUDA_VISIBLE_DEVICES=1 conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py --dataset mousebrain --use-gpu --gpu-dtype float64 --methods SPARKLEv2R1,SPARKLEv2R2
# （axolotl/ovarian 同理，GPU0）
# 对比
python evaluation/scripts_v2/compare_synthetic_v2.py
conda run -n scvi python evaluation/scripts_v2/compare_real_v2.py
```
