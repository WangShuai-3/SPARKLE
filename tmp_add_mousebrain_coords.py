"""Add mean x,y coordinates to existing MouseBrain h5ad files."""
import gzip
import numpy as np
import scanpy as sc
from pathlib import Path
import pandas as pd

tag = "mousebrain_x12500-17500_y2000-5000"
h5ad_dir = Path("evaluation/reports/h5ad")
gem_path = Path("evaluation/data/mousebrain/total_gene_T304_mouse_f001_2D_mouse1-20230119.txt.gz")
x_range = (12500, 17500)
y_range = (2000, 5000)

# Get cell_ids from one h5ad
adata_ref = sc.read_h5ad(h5ad_dir / f"{tag}_raw.h5ad")
target_cell_ids = set(adata_ref.obs['cell_id'].astype(int))
print(f"Target cell ids: {len(target_cell_ids)}")

# Compute mean coords per cell from GEM
coord_sums = {}
coord_counts = {}
with gzip.open(gem_path, 'rt') as f:
    header = f.readline().strip().split('\t')
    # gene x y umi_count cell_label gene_area rx ry
    for line in f:
        parts = line.rstrip('\n').split('\t')
        x = int(parts[1])
        y = int(parts[2])
        if x_range and not (x_range[0] <= x <= x_range[1]):
            continue
        if y_range and not (y_range[0] <= y <= y_range[1]):
            continue
        cell_label = int(parts[4])
        if cell_label not in target_cell_ids:
            continue
        if cell_label not in coord_sums:
            coord_sums[cell_label] = [0.0, 0.0]
            coord_counts[cell_label] = 0
        coord_sums[cell_label][0] += x
        coord_sums[cell_label][1] += y
        coord_counts[cell_label] += 1

coord_df = pd.DataFrame({
    'cell_id': list(coord_sums.keys()),
    'x': [coord_sums[cid][0] / coord_counts[cid] for cid in coord_sums],
    'y': [coord_sums[cid][1] / coord_counts[cid] for cid in coord_sums],
})
coord_df.set_index('cell_id', inplace=True)
print(f"Computed coords for {len(coord_df)} cells")

# Add coords to each h5ad
methods = ["raw", "SPARKLE", "SpatialSoupX", "SoupX", "DecontX"]
for method in methods:
    path = h5ad_dir / f"{tag}_{method}.h5ad"
    if not path.exists():
        continue
    adata = sc.read_h5ad(path)
    cids = adata.obs['cell_id'].astype(int).values
    adata.obs['x'] = coord_df.loc[cids, 'x'].values
    adata.obs['y'] = coord_df.loc[cids, 'y'].values
    adata.write_h5ad(path)
    print(f"Updated {path}")

print("Done")
