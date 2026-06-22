# Evaluation Directory

> **本目录仅用于评测和报告，不包含算法代码。**
> 算法代码位于 `stambient/`，单元测试位于 `tests/`。

## 目录结构

```
evaluation/
├── README.md              # 本文件
├── synthetic/             # 模拟数据生成
│   ├── generator.py       #   Voronoi-based 细胞 mask + 环境 RNA 注入
│   └── scenarios.py       #   预定义评测场景
├── scripts/               # 评测脚本
│   ├── run_benchmark.py   #   主评测流程 (合成数据)
│   ├── test_axolotl.py    #   Axolotl 真实数据测试
│   ├── final_comparison.py#   最终方法对比
│   └── visualize.py       #   空间热力图生成
├── baselines/             # 对比方法
│   ├── soupx.py           #   原始 SoupX
│   └── spatial_soupx.py   #   Spatial SoupX (bin 级空间核)
├── data/                  # 数据
│   ├── S1-S7.npz          #   合成数据缓存
│   └── axolotl/           #   Axolotl 真实数据
└── reports/               # 评测报告输出
```

## 模拟数据规格

| 参数 | 值 |
|------|-----|
| DNB 网格 | 200×200 = **40,000 DNBs** |
| DNB 间距 | 0.5 μm |
| 基因数 | 500 (其中 80 个高表达基因) |
| 细胞数 | 108–306 (因场景而异) |
| 空 DNB 占比 | 10%–40% |
| 真实 λ | 20 / 50 / 100 μm |
| 泄漏率 α | 0.001–0.10 |

## SPARKLE 运行参数

| 参数 | 值 | 说明 |
|------|-----|------|
| bin_size | 25 (合成) / 50 (真实) | bin 边长 (DNB) |
| max_radius | 3×λ (合成) / 200μm (真实) | 邻居搜索半径 |
| lambda_grid | [10,20,30,50,70,100,150,200,300] | λ 候选值 |
| cell_based | True (推荐) | Cell 级架构 |
| penalty | `1/(1+(s/p90)²)` (默认) | 自置信惩罚 |

## 使用

```bash
# 合成数据 benchmark
python evaluation/scripts/run_benchmark.py --all

# Axolotl 真实数据
python evaluation/scripts/test_axolotl.py

# 最终方法对比
python evaluation/scripts/final_comparison.py

# 空间热力图
python evaluation/scripts/visualize.py --all
```
