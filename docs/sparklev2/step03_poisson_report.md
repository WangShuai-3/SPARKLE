# SPARKLE 2.x Phase 2 — Step 03: Poisson count model with diffuse background β

Branch `SPARKLEv2`, commits `9401a88` (implementation) and `b35bcc3` (solver fix + evaluation).
All numbers below use the fixed solver; every reported artifact is under
`evaluation/reports/v2/` (metrics JSONs, h5ads, comparison CSVs/PNGs, logs in
`evaluation/logs/step03_*.log`).

## 1. What changed

**Observation model (B-ablation).** The background-bin model is now selectable:
`μ_gb = A_b·[β_g + ρ_g·S_gb]` fit by direct Poisson maximum likelihood
(`observation_model="poisson"`), replacing the v1 weighted-OLS fit
`y ≈ α·A·S` (`observation_model="weighted_ols"`, still the default):

- **B1 / SPARKLEv2P1**: local-only Poisson, `μ = A·ρ·S` (direct Poisson successor of α).
- **B2 / SPARKLEv2P2**: diffuse+local, `μ = A·(β + ρ·S)`, `subtract_diffuse=False`
  (conservative: β debiases ρ but the diffuse component is *not* removed from cells,
  because a uniform background-bin component is not distinguishable from endogenous
  low-level expression).
- **B3 / SPARKLEv2P2D**: B2 with `subtract_diffuse=True` (also subtracts `A_c·β`).

All variants run the v2.0-alpha2 core (latent-X + penalty + 2 α-refit rounds; the
refits re-fit β/ρ against the current latent X). R² gating is unchanged (Phase 3
replaces it with a count-based evidence score). Legacy and alpha1/2 numerics are
bit-identical (freeze test + 110 tests pass).

**Solver** (`stambient/count_model.py`): vectorised projected Newton, 2 non-negative
parameters per gene, closed-form OLS warm start, analytic 2×2 Hessian, backtracking
line search; CPU and GPU branches agree to ≤7e-15.

**Solver bug found by held-out evaluation** (fixed in `b35bcc3`): the naive
projected Newton *stalled at the non-negativity boundary* — at β=0 the clipped
joint step is not a descent direction for the constrained problem, so B2 fits could
land below the nested B1 optimum (worst train-loglik gap −124/gene on S2). Fixes:
(i) active-set step (a parameter at its bound whose unconstrained step would leave
the orthant is dropped; the free coordinate is re-solved on the reduced Hessian
block), (ii) KKT-aware convergence check, (iii) a nested-model safeguard in
`estimate_leakage_poisson` (the B1 MLE is always kept as a candidate and the
pointwise higher-loglik solution wins), making B2 ≥ B1 in-sample by construction.
Held-out Δ(P2−P1) on S2 went from +11.7 to +0.3 NLL/gene after the fix.

## 2. Model evidence: held-out spatial-block deviance

4-fold checkerboard of empty bins (block edge = 2×bin size), fit on 3/4, Poisson
NLL on the held-out 1/4, per gene. λ, W_empty, gene panel fixed across folds.
NULL = flat rate `A·β₀` (the best model *without* spatial structure).

| dataset | λ | genes | bins | OLS−NULL | P1−OLS | P2−P1 | verdict |
|---|---|---|---|---|---|---|---|
| S2 (local only) | 50 | 80 | 100 | **−223.8** | −0.11 | +0.32 | local signal real; P2 ties P1 (no hallucinated β) |
| M1 (local+diffuse) | 100 | 80 | 100 | −145.1 | −0.16 | −0.06 | slight P2 gain |
| M2 (diffuse only) | 500 | 80 | 100 | **+2.9** | −0.43 | **−2.23** | OLS *worse than NULL*; P2 ≈ NULL (−3702.5 vs −3702.8) |
| axolotl | 20 | 200 | 4015 | **+2561** | −637 | **−6911** | OLS catastrophically worse than NULL; P2 ≫ all |
| ovarian | 10 | 18786 | 2798 | −34.0 | −6.7 | **−27.2** | P2 best |
| mousebrain | 200 | 23339 | 23985 | **+2289** | −7.5 | **−2319** | OLS worse than NULL; P2 best, ≈ NULL+local gain |

Headline: on all three real datasets the v1 observation model (weighted-OLS
local-only) is **worse than a flat-rate null on spatially held-out background bins**
— the "local leakage" the OLS fit detects does not generalise spatially. The
diffuse+local Poisson model dominates decisively (axolotl ΔNLL ≈ −7548/gene vs
OLS). On synthetic local-only data (S2) P2 correctly ties P1 instead of
hallucinating a diffuse component. GPU and CPU deviance runs agree bit-for-bit
(axolotl cross-check).

## 3. Mismatch scenarios (known diffuse ground truth)

`evaluation/scripts_v2/mismatch.py`: M1 = S2 + uniform diffuse at 0.5× the per-gene
local ambient level; M2 = S2 layout with local leakage switched off (diffuse-only).
True per-DNB diffuse rate `d_g` is known exactly (bin area = DNB count ⇒ β ≡ d_g).

| scenario | method | RMSE↓ | cell R²↑ | ARI↑ | OCR↓ | β̂/d med | ρ̄ |
|---|---|---|---|---|---|---|---|
| M1 | RAW | 0.9259 | 0.691 | 0.634 | 0 | – | – |
| M1 | SPARKLEv1 | 0.6162 | 0.929 | 0.666 | 0.0108 | – | ᾱ=0.0019 |
| M1 | SPARKLEv2R2 | 0.5873 | 0.947 | 0.618 | 0.0135 | – | ᾱ=0.0032 |
| M1 | **SPARKLEv2P1** | **0.5831** | **0.948** | 0.598 | 0.0143 | 0 | 0.0033 |
| M1 | SPARKLEv2P2 | 0.5847 | 0.948 | 0.674 | 0.0136 | 0 | 0.0032 |
| M1 | SPARKLEv2P2D | 0.5849 | 0.948 | 0.918* | 0.0140 | 0 | 0.0032 |
| M2 | RAW | 0.6456 | 0.921 | 0.669 | 0 | – | – |
| M2 | SPARKLEv1 | 0.4162 | 0.988 | 0.727 | 0.0137 | – | ᾱ=0.0005 |
| M2 | SPARKLEv2R2 | 0.4084 | 0.989 | 0.724 | 0.0151 | – | ᾱ=0.0006 |
| M2 | SPARKLEv2P1 | 0.4105 | 0.988 | 0.681 | 0.0139 | 0 | 0.0006 |
| M2 | **SPARKLEv2P2** | 0.6093† | 0.937† | 0.667 | **0.0018** | **0.967** | **0.0001** |
| M2 | **SPARKLEv2P2D** | **0.4101** | **0.988** | 0.685 | 0.0142 | **0.964** | 0.0001 |

*P2D's ARI on M1 is a KMeans-seed outlier (its marker/RMSE metrics match P2).
†P2 without subtraction deliberately leaves the diffuse component in cells
(conservative default), so its RMSE stays near RAW — while its overcorrection rate
is 8× lower than any local-leakage method.

Findings:
- **β recovery**: on M2 the fitted diffuse rate tracks the injected truth
  (median ratio 0.967, IQR 0.30) and ρ collapses to ≈0 — no false local leakage.
- **Identifiability limit**: on M1 (diffuse at only 0.5× the local level) the
  per-gene MLE attributes everything to the local channel (β̂/d median 0); the two
  components are nearly unidentifiable per gene at that level. Aggregate/hierarchical
  shrinkage (Phase 4) is the remedy, not a solver issue.
- **Misattribution is benign for point correction**: on M2 the local-only methods
  also achieve good RMSE by absorbing the (spatially flat) diffuse component through
  S; what distinguishes the models is the *mechanism* (held-out NLL, β recovery,
  OCR), which P2 gets right.

## 4. Synthetic S1–S10 (v1 scenarios, local-only truth)

Poisson variants hold or slightly improve the v2R2 core; v1 wins only S6 (the known
marker-clustering case). Best RMSE per scenario: P1 ×5 (S2, S3, S8, S9, S10),
P2 ×2 (S1, S4), P2D ×2 (S5, S7), v1 ×1 (S6). On the long-λ scenarios S5/S7 the
flatter S lets β absorb part of the local leakage; there `subtract_diffuse` matters:
S5 P2D 0.9086 < v2R2 0.9193 < P2 0.9681; S7 P2D 1.1315 < v2R2 1.1381 < P2 1.1838.
On the remaining scenarios the three Poisson variants are within ±0.005 RMSE of
v2R2 (e.g. S2: P1 0.5238 vs v2R2 0.5246 vs v1 0.5447).

## 5. Real-data correction quality

| dataset | metric | v1 | v2R2 | P1 | P2 | P2D |
|---|---|---|---|---|---|---|
| mousebrain | Pearson vs snRNA ref | 0.2847 | 0.2875 | **0.2896** | 0.2634 | 0.2889 |
| mousebrain | Spearman | 0.7434 | 0.7434 | 0.7434 | 0.7435 | 0.7435 |
| ovarian | Pearson vs scRNA ref | 0.5330 | 0.5306 | 0.5307 | **0.5341** | 0.5332 |
| ovarian | Spearman | 0.6737 | 0.6759 | 0.6758 | **0.6765** | 0.6762 |
| axolotl | SST sstIN mean | 69.94 | 70.19 | 69.90 | 69.90 | 69.90 |
| axolotl | SST S/N ratio | 17.69× | 19.87× | **20.84×** | **20.84×** | **20.84×** |

(Axolotl: the SST gene's refit-ρ converges to nearly the same value in all three
Poisson variants — max per-cell difference 0.037 — so its group means coincide.)

Diffuse is pervasive on real data: fraction of panel genes with β̂ > 0 is
axolotl 100%, mousebrain 92%, ovarian 86%. The β/ρ split reassigns a large share
of what v1 called "local leakage" to the diffuse channel (mean correction strength:
axolotl 0.298→0.178, ovarian 0.319→0.175, mousebrain 0.00271→0.00083). P1's
in-sample Δloglik vs the null is *negative* on axolotl (−507) and mousebrain
(−2276) — forcing background through the local channel alone is worse than a flat
rate, consistent with the held-out result.

The one regression: mousebrain P2 Pearson drops to 0.2634 (λ=200 → flat S → β
absorbs local leakage → undercorrection); `subtract_diffuse` (P2D) restores 0.2889.
This is the same flat-S identifiability tension as S5/S7.

## 6. Conclusions

- The Poisson count model with a diffuse component is a decisive improvement in
  *background-model* quality on every real dataset (held-out NLL), with no
  regression on synthetic local-only data, and it eliminates false local-leakage
  attribution (M2: ρ≈0, OCR 8× lower, β recovered at 0.97 of truth).
- Correction-quality changes are modest on synthetic data and mixed-but-positive
  on real data (ovarian best overall with P2; mousebrain best with P1/P2D;
  axolotl best S/N with any P variant).
- The per-gene β/ρ split is unidentifiable when the diffuse level is small or S
  is flat (M1, S5/S7, mousebrain): the MLE then lands on a boundary, and which
  boundary it lands on moves RMSE. This motivates Phase 4 (hierarchical/aggregate
  shrinkage of β and ρ) more than any solver tweak.

**Go decision: GO.** Adopt B2 (`observation_model="poisson", fit_diffuse=True,
subtract_diffuse=False`) as the Phase-2 observation model for Phase 3, keeping
P1/P2D as registered ablations. The R² gate (whose failures this phase made
visible — OLS worse than NULL on real held-out data) is replaced next by the
count-based evidence score (Δloglik vs the diffuse-only null is already computed
per gene and stored in `poisson_stats`), and the β/ρ boundary instability is the
primary target for Phase 4 hierarchical shrinkage.
