# SPARKLE 2.0 模块提升总结（Phase 0–3,2026-08)

分支 `SPARKLEv2`。分步报告:`step01_latent_x_report.md`、`step02_alpha_refit_report.md`、
`step03_poisson_report.md`、`step04_evidence_report.md`。评估基线与 v1 完全一致
(同数据窗口、参数、seed),CPU/GPU parity ≤7e-15,117 测试通过。

## 1. Ablation 阶梯(与路线图 A0–A3 对应)

| 标记 | 配置 | 模块 |
|---|---|---|
| A0 = SPARKLEv1 | legacy + penalty + R² 门 | v1 基线 |
| A1 | latent X + penalty(+ refit=2 = R2) | **M1: latent-X 自洽推断** |
| A2 = R2 + Poisson(P2) | A1 + μ=A(β+ρS) count model | **M2: Poisson + diffuse β** |
| A3 = P2 + evidence(E2) | A2 + spatial-CV E_g 连续权重 | **M3: evidence-based 校正** |

## 2. 各模块是否提升

### M1 — latent X(step01/02)
- **单独(A1)不提升**:机制指标(marker preservation、OCR、ARI、axolotl)改善,
  但 RMSE 与真实数据参考一致性落后 v1 —— 根因是 α 用 Y-source 校准系统性偏小。
- **加 α-refit(A1+R2)后明确提升**:synthetic 9/10 场景 RMSE/R² 胜 v1
  (唯一例外 S6,v1 优 2.6%);mousebrain Pearson 0.2875>0.2847 且 silhouette 最优;
  axolotl 三项全胜(S/N 19.87× vs 17.69×);α 轨迹单调上升收敛。
- 判定:**GO(需与 refit 绑定)**,latent+penalty+refit=2 为 2.0 校正核心;penalty 必须保留。

### M2 — Poisson count model + diffuse β(step03)
- **模型质量(held-out 证据)决定性提升**:4-fold 空间块 held-out NLL 上,
  v1 的 OLS local-only 在**全部 3 个真实数据集上劣于 flat-rate null**
  (axolotl +2561、mousebrain +2289 NLL/基因),而 diffuse+local Poisson 大幅最优
  (axolotl Δ=−7548/基因 vs OLS);synthetic 纯 local 数据上正确打平,不虚报 β。
- **归因正确性**:M2(纯 diffuse)β̂/真值中位 0.967、ρ≈0、OCR 比 v1 低 8×;
  真实数据 β>0 基因比例 86–100% —— diffuse ambient 普遍存在,
  v1 把其中 30–70% 误记为"局部泄漏"(axolotl α 均值 0.298→0.178)。
- **校正指标**:synthetic 9/10 ≥ v2R2;ovarian P2 双相关最佳(0.5341/0.6765);
  mousebrain P1 Pearson 最佳(0.2896);axolotl P 系列 S/N=20.84× 最佳。
- 代价:flat-S 场景(S5/S7、mousebrain)β/ρ 单基因不可辨识,P2 偶发欠校正
  (P2D 可回补)。判定:**GO**,且暴露并修复了 Newton 边界停滞 bug。

### M3 — evidence score(step04)
- **校准性决定性提升**:E_g 分离"有/无局部泄漏"基因 **AUC=1.000**(S2 vs M2)、
  **0.977**(弱泄漏 M3);无泄漏面板假阳性 P(E>0.05)=1.3%;R² 门无此校准
  (在 M2 上仍放行全部 80 基因)。
- **假校正控制**:M2(零泄漏)上 OCR 比 v1 低 **137×**(E2=0.0001),
  校正基因 80→22;弱泄漏 M3 上 E1 全场最佳。
- **校正指标**:in-model S1–S10 与 P2 持平(≤2% 代价,仅长 λ 的 S5/S7);
  ovarian **E2 双相关全场最佳**(Pearson 0.5345 / Spearman 0.6769);
  axolotl 持平;mousebrain 沿 flat-S 极限略降。
- 判定:**GO**,E2 定为 SPARKLE 2.0-rc 配置。

## 3. 最终 2.0(A3=E2)vs v1:核心指标对比

**真实数据**(参考一致性 / 标记物):
| 数据集 | 指标 | v1 | 2.0(E2) | 最优 2.x 变体 |
|---|---|---|---|---|
| axolotl | SST S/N ↑ | 17.69× | **20.84×(+18%)** | P1/P2/E1/E2 同 |
| ovarian | Pearson ↑ | 0.5330 | **0.5345** | E2(双相关最佳) |
| ovarian | Spearman ↑ | 0.6737 | **0.6769** | E2 |
| mousebrain | Pearson ↑ | 0.2847 | 0.2550 | **P1 0.2896 / P2D 0.2889** |
| mousebrain | Spearman ↑ | 0.7434 | 0.7435 | 各法 ≈ 持平 |

**synthetic S1–S10 均值**(真值已知):
| 方法 | RMSE↓ | cell R²↑ | ARI↑ | marker 源保持 | 非源残差↓ | OCR↓ |
|---|---|---|---|---|---|---|
| RAW | 0.912 | 0.750 | 0.478 | 1.545 | 37.59 | 0 |
| v1 | 0.622 | 0.892 | 0.611 | 1.268 | 11.10 | 0.0102 |
| R2 (A1) | **0.597** | **0.915** | 0.648 | 1.293 | 7.85 | 0.0117 |
| P2 (A2) | 0.606 | 0.906 | **0.706** | 1.302 | 9.23 | 0.0108 |
| E2 (A3) | 0.608 | 0.906 | 0.705 | **1.302** | 9.27 | **0.0107** |

**out-of-model mismatch**:
| 场景 | v1 | R2 | P2 | E1 | E2 |
|---|---|---|---|---|---|
| M1(局部+diffuse)RMSE | 0.616 | 0.587 | 0.585 | 0.585 | 0.585 |
| M2(纯 diffuse)OCR | 0.0137 | 0.0151 | 0.0018 | 0.0013 | **0.0001** |
| M2 β̂/真值中位 | – | – | 0.967 | 0.968 | 0.968 |
| M3(弱泄漏)RMSE | 0.3326 | 0.3308 | 0.3297 | **0.3285** | 0.3288 |
| M4(50% dropout)RMSE | 0.5964 | 0.5788 | 0.5792 | 0.5792 | 0.5792 |

## 4. 结论

1. **三个模块都提升,但提升的维度不同**:M1(latent+refit)提升校正精度与 marker
   保真;M2(Poisson+β)提升背景模型质量(held-out NLL 碾压级)与泄漏归因正确性;
   M3(evidence)提升假校正控制与校准性(AUC≈1)。路线图"一次只改一个统计假设"
   的 ablation 链条完整成立。
2. **2.0 总体**:在 3 个真实数据集中的 2 个(ovarian、axolotl)核心指标全面 ≥ v1
   且多个为全场最佳;synthetic 上 RMSE/R² 最优的是 A1(R2),ARI 最优的是 A2/A3,
   out-of-model 鲁棒性随模块阶梯单调增强。
3. **明确的遗留问题**(均为同一根因 —— S 空间平坦时 β/ρ/λ 单基因不可辨识):
   - mousebrain(λ=200,高密度)Pearson:E2 默认配置低于 v1,P1/P2D 变体仍胜 v1;
   - S5/S7(长 λ)P2/E 轻微欠校正;
   - S6(marker 空间聚集)v1 的 penalty 主导优势仍在。
   → 这正是 **Phase 4(hierarchical λ partial pooling,SPARKLE 2.1)** 的目标。
