#!/usr/bin/env python3
"""重建 Figure 2 最终汇总表(10 scenarios × 5 methods,官方 R 基线)。

数据来源:
- RAW / truth: 由 evaluation.synthetic 生成器现场重生成(seed=42, 20% dropout, 与
  final_comparison.py 完全同一代码路径),保证同一 realization;
- SPARKLE / SpotClean: reports/all_scenarios_dropout20/corrected/{S*}_{method}.npy
  (8/10 产物;两方法本次未变更,并会与 8/10 旧汇总逐值对账验证一致性);
- SoupX / DecontX: reports/{soupx,decontx}_official/synthetic_S*/output/decont.h5
  (8/23 官方 R 产物;rhdf5 存储为 cells×genes,读取后转置回 genes×cells)。

指标口径(与手稿一致,函数直接复用 final_comparison.py):
- rmse_log1p_cp10k: log1p(CP10K) 空间 RMSE vs ground truth;
- r2_mean: 逐 cell 跨 genes Pearson r 的平方的均值;
- reduction_pct: (rmse_raw - rmse_method)/rmse_raw × 100。

输出: Report_V2/Figure2_synthetic/figure2_final_summary_r_impl.csv (50 行)
用法: cd evaluation && python scripts/rebuild_fig2_final_table.py
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from final_comparison import (  # noqa: E402
    _cellwise_r2_mean,
    _rmse_log1p_cp10k,
    compute_cell_expr,
    load_synthetic_scenario_data,
)

REPORTS = os.path.join(EVAL_DIR, "reports")
OUT_CSV = os.path.join(EVAL_DIR, "Report_V2", "Figure2_synthetic",
                       "figure2_final_summary_r_impl.csv")
OLD_SUMMARY = os.path.join(REPORTS, "all_scenarios_dropout20",
                           "log1p_cp10k_rmse_r2_summary.csv")
# 8/23 中间层 RAW 参考值,用于验证同一 realization
RAW_REFERENCE = {"S3": 1.0025574072450725}

METHOD_ORDER = ["RAW", "SoupX", "DecontX", "SpotClean", "SPARKLE"]


def load_r_decont(dataset_dir):
    """读取官方 R decont.h5。rhdf5 写出为 cells×genes,转置回 genes×cells。"""
    path = os.path.join(REPORTS, dataset_dir, "output", "decont.h5")
    with h5py.File(path, "r") as f:
        x = f["X"][:]
    return np.asarray(x.T, dtype=np.float64)


def main():
    old = pd.read_csv(OLD_SUMMARY).set_index(["scenario", "method"])
    rows = []
    for i in range(1, 11):
        sid = f"S{i}"
        data = load_synthetic_scenario_data(sid, seed=42)
        true_expr = data["true_expr"]
        n_cells = true_expr.shape[1]
        raw = compute_cell_expr(data["dnb_expr"], data["dnb_labels"], n_cells)

        mats = {"RAW": raw}
        for m in ("SPARKLE", "SpotClean"):
            mats[m] = np.load(os.path.join(
                REPORTS, "all_scenarios_dropout20", "corrected",
                f"{sid}_{m}.npy"))
        mats["SoupX"] = load_r_decont(f"soupx_official/synthetic_{sid}")
        mats["DecontX"] = load_r_decont(f"decontx_official/synthetic_{sid}")

        rmse_raw = _rmse_log1p_cp10k(raw, true_expr)
        if sid in RAW_REFERENCE:
            assert abs(rmse_raw - RAW_REFERENCE[sid]) < 1e-9, (
                f"{sid} RAW RMSE {rmse_raw} != 中间层参考值 "
                f"{RAW_REFERENCE[sid]}:realization 不一致!")
            print(f"  [{sid}] RAW RMSE 与 8/23 中间层一致 ✓ (同一 realization)")

        for m in METHOD_ORDER:
            mat = np.asarray(mats[m], dtype=np.float64)
            assert mat.shape == true_expr.shape, (
                f"{sid}/{m} 形状 {mat.shape} != truth {true_expr.shape}")
            rmse = _rmse_log1p_cp10k(mat, true_expr)
            r2 = _cellwise_r2_mean(mat, true_expr)
            rows.append({
                "scenario": sid, "dropout": 0.2, "method": m,
                "rmse_log1p_cp10k": rmse,
                "reduction_pct": (rmse_raw - rmse) / rmse_raw * 100.0,
                "r2_mean": r2,
                "implementation": ("official_R_package"
                                   if m in ("SoupX", "DecontX", "SpotClean")
                                   else m),
            })
            # 未变更方法与 8/10 旧汇总逐值对账
            if m in ("RAW", "SPARKLE", "SpotClean"):
                o = old.loc[(sid, m)]
                assert abs(o["rmse_log1p_cp10k"] - rmse) < 1e-6, (
                    f"{sid}/{m} RMSE 与 8/10 汇总不符: {rmse} vs "
                    f"{o['rmse_log1p_cp10k']}")
                assert abs(o["r2_mean"] - r2) < 1e-6, (
                    f"{sid}/{m} R2 与 8/10 汇总不符: {r2} vs {o['r2_mean']}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV} ({len(df)} rows)")

    print("\n=== headline 对账 ===")
    agg = df.groupby("method").agg(
        r2_mean=("r2_mean", "mean"), r2_std=("r2_mean", "std"),
        rmse_mean=("rmse_log1p_cp10k", "mean"),
        reduction_mean=("reduction_pct", "mean")).reindex(METHOD_ORDER)
    print(agg.round(4).to_string())
    rmse_win = df.loc[df.groupby("scenario")["rmse_log1p_cp10k"].idxmin()]
    r2_win = df.loc[df.groupby("scenario")["r2_mean"].idxmax()]
    print("\nRMSE 最优计数:", rmse_win["method"].value_counts().to_dict())
    print("R² 最优计数:", r2_win["method"].value_counts().to_dict())


if __name__ == "__main__":
    main()
