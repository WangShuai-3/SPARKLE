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
│   ├── final_comparison.py# 最终方法对比（支持 axolotl/mosta/visiumhd/synthetic）
│   └── visualize.py       # 空间热力图生成 — 待实现
├── baselines/             # 对比方法
│   ├── soupx.py           # 原始 SoupX
│   └── spatial_soupx.py   # Spatial SoupX (bin 级空间核)
├── data/                  # 数据
│   └── axolotl/           # Axolotl 真实数据
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
| 真实 λ | 20 / 50 / 100 μm |
| 泄漏率 α | 0.005–0.10 |

## SPARKLE 运行参数（合成数据）

| 参数 | 值 | 说明 |
|------|-----|------|
| bin_size | 25 | bin 边长 (DNB) |
| max_radius | 300 μm | 邻居搜索半径 |
| lambda_grid | [10,20,30,50,70,100,150,200,300] | λ 候选值 |
| cell_based | True | Cell 级架构 |
| self_confidence_penalty | False | 合成 benchmark 中关闭以优化 RMSE |

## 使用

```bash
# 单个合成数据场景
python evaluation/scripts/final_comparison.py --dataset synthetic --scenario S1

# 全部 10 个合成数据场景
python evaluation/scripts/final_comparison.py --dataset synthetic --all-scenarios

# Axolotl 真实数据
python evaluation/scripts/test_axolotl.py

# 最终方法对比（真实数据）
python evaluation/scripts/final_comparison.py --dataset axolotl
```
