#!/usr/bin/env python3
"""Benchmark correction methods (SPARKLE, SoupX, DecontX, SpotClean) on the
three smallest MouseBrain windows: runtime and peak memory.

Each (method, window) runs in its own subprocess; wall-clock time and peak RSS
are captured with /usr/bin/time -v and stored alongside the result files so a
rerun can reuse completed combinations (--skip-existing).

Usage:
    python evaluation/scripts/benchmark_methods_mousebrain.py \
        [--methods sparkle,soupx,decontx,spotclean] [--skip-existing] [--plot]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = PROJECT_ROOT / "evaluation" / "reports"
OUT_DIR = REPORTS / "method_benchmark_mousebrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    {"id": "w8", "x_range": (12125.0, 13875.0), "y_range": (7687.5, 9312.5), "area_mm2": 0.711, "n_cells": 1831},
    {"id": "w7", "x_range": (11250.0, 14750.0), "y_range": (6875.0, 10125.0), "area_mm2": 2.844, "n_cells": 6804},
    {"id": "w6", "x_range": (10375.0, 15625.0), "y_range": (6062.5, 10937.5), "area_mm2": 6.398, "n_cells": 14577},
]

TIME_BIN = "/usr/bin/time"
WORKER = PROJECT_ROOT / "evaluation" / "scripts" / "benchmark_methods_mousebrain_worker.py"
SPOTCLEAN_WORKER = PROJECT_ROOT / "evaluation" / "scripts" / "benchmark_spotclean_worker.py"


def _result_paths(win, method):
    base = OUT_DIR / f"{win['id']}_{method}"
    return base.with_suffix(".txt"), base.with_suffix(".rss.txt")


def _parse_rss(stderr):
    for line in stderr.splitlines():
        if "Maximum resident set size" in line:
            return int(line.split(":")[-1].strip()) / 1024.0  # KB -> MB
    return None


def run_python_method(method, win, skip_existing):
    x0, x1 = win["x_range"]
    y0, y1 = win["y_range"]
    res_path, rss_path = _result_paths(win, method)
    if skip_existing and res_path.exists() and rss_path.exists():
        info = dict(l.split("=", 1) for l in res_path.read_text().splitlines() if "=" in l)
        peak = float(rss_path.read_text().strip())
        rt = float(info.get("runtime_sec", "nan"))
        print(f"  [{method:>8} {win['id']}] (skip) runtime={rt:8.1f}s  peakRSS={peak:8.1f} MB")
        return rt, peak
    res_path.unlink(missing_ok=True)
    rss_path.unlink(missing_ok=True)
    cmd = [TIME_BIN, "-v", sys.executable, str(WORKER),
           method, str(x0), str(x1), str(y0), str(y1), str(res_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    peak = _parse_rss(proc.stderr)
    if not res_path.exists():
        print(f"  [{method} {win['id']}] FAILED\n{proc.stderr[-2000:]}")
        return None, peak
    info = dict(l.split("=", 1) for l in res_path.read_text().splitlines() if "=" in l)
    rt = float(info.get("runtime_sec", "nan"))
    if peak is not None:
        rss_path.write_text(f"{peak:.1f}")
    print(f"  [{method:>8} {win['id']}] runtime={rt:8.1f}s  peakRSS={peak:8.1f} MB")
    return rt, peak


def run_spotclean(win, skip_existing):
    x0, x1 = win["x_range"]
    y0, y1 = win["y_range"]
    res_path, rss_path = _result_paths(win, "spotclean")
    if skip_existing and res_path.exists() and rss_path.exists():
        info = dict(l.split("=", 1) for l in res_path.read_text().splitlines() if "=" in l)
        peak = float(rss_path.read_text().strip())
        rt = float(info.get("runtime_sec", "nan"))
        print(f"  [SPOTCLEAN {win['id']}] (skip) runtime={rt:8.1f}s  peakRSS={peak:8.1f} MB")
        return rt, peak
    res_path.unlink(missing_ok=True)
    rss_path.unlink(missing_ok=True)
    cmd = [TIME_BIN, "-v", sys.executable,
           str(SPOTCLEAN_WORKER), str(x0), str(x1), str(y0), str(y1), str(res_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    peak = _parse_rss(proc.stderr)
    if not res_path.exists():
        print(f"  [SPOTCLEAN {win['id']}] FAILED\n{proc.stderr[-2000:]}")
        return None, peak
    info = dict(l.split("=", 1) for l in res_path.read_text().splitlines() if "=" in l)
    rt = float(info.get("runtime_sec", "nan"))
    if peak is not None:
        rss_path.write_text(f"{peak:.1f}")
    print(f"  [SPOTCLEAN {win['id']}] runtime={rt:8.1f}s  peakRSS={peak:8.1f} MB")
    return rt, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="sparkle,soupx,decontx,spotclean")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",")]

    rows = []
    for win in WINDOWS:
        for m in methods:
            if m == "spotclean":
                rt, peak = run_spotclean(win, args.skip_existing)
            else:
                rt, peak = run_python_method(m, win, args.skip_existing)
            rows.append({
                "window": win["id"], "area_mm2": win["area_mm2"],
                "n_cells": win["n_cells"], "method": m,
                "runtime_sec": rt, "peak_rss_mb": peak,
            })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "method_benchmark_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    if args.plot:
        _plot(df, OUT_DIR)


def _plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sparkle": "#1f4e79", "soupx": "#bc8e36",
              "decontx": "#457f78", "spotclean": "#8f6aa8"}
    for win in df["window"].unique():
        sub = df[df.window == win].sort_values("runtime_sec")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(sub["method"], sub["runtime_sec"], color=[colors[m] for m in sub["method"]])
        axes[0].set_ylabel("Runtime (s)")
        axes[0].set_title(f"{win} ({sub.iloc[0]['area_mm2']:.2f} mm²)")
        axes[1].bar(sub["method"], sub["peak_rss_mb"], color=[colors[m] for m in sub["method"]])
        axes[1].set_ylabel("Peak RSS (MB)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(out_dir / f"benchmark_{win}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in df["method"].unique():
        s = df[df.method == m].sort_values("area_mm2")
        axes[0].plot(s["area_mm2"], s["runtime_sec"], marker="o", label=m, color=colors[m])
        axes[1].plot(s["area_mm2"], s["peak_rss_mb"], marker="o", label=m, color=colors[m])
    axes[0].set_xlabel("Area (mm²)"); axes[0].set_ylabel("Runtime (s)"); axes[0].legend()
    axes[1].set_xlabel("Area (mm²)"); axes[1].set_ylabel("Peak RSS (MB)"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / "benchmark_combined.png", dpi=150)
    plt.close(fig)
    print(f"Saved plots in {out_dir}")


if __name__ == "__main__":
    main()
