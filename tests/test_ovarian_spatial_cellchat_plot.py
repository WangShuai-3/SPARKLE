from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from evaluation.scripts.plot_ovarian_spatial_cellchat import (
    METHOD_ORDER,
    figure_05b,
    prepare_spatial_cellchat_data,
)


def _write_fixture(input_dir: Path, empty_method: str | None = None) -> None:
    pd.DataFrame(
        {
            "Method": METHOD_ORDER,
            "total_prob": [1.0, 0.98, 0.65, 0.45],
            "autocrine_prob": [0.30, 0.29, 0.19, 0.14],
            "tumor_involving_prob": [0.61, 0.60, 0.36, 0.23],
        }
    ).to_csv(input_dir / "cellchat_spatial_summary.csv", index=False)

    keys = [
        ("A", "B", "L1", "R1"),
        ("B", "A", "L2", "R2"),
        ("A", "A", "L3", "R3"),
    ]
    observed = {
        "RAW": keys,
        "SoupX": keys[:2],
        "DecontX": keys[1:],
        "SPARKLE": keys[::2],
    }
    for method in METHOD_ORDER:
        rows = observed[method]
        frame = pd.DataFrame(rows, columns=["source", "target", "ligand", "receptor"])
        frame["prob"] = [0.1 * (i + 1) for i in range(len(frame))]
        frame["pval"] = 0.01
        if method == empty_method:
            frame = frame.iloc[0:0]
        frame.to_csv(input_dir / f"{method}_cellchat_spatial.csv", index=False)


def test_spatial_interactions_are_nonempty_and_matched(tmp_path: Path):
    _write_fixture(tmp_path)
    prepared = prepare_spatial_cellchat_data(tmp_path)

    counts = prepared["interaction_union"].groupby("method").size().to_dict()
    assert set(counts) == set(METHOD_ORDER)
    assert len(set(counts.values())) == 1
    assert counts["DecontX"] > 0
    assert prepared["interaction_union"].loc[
        prepared["interaction_union"]["method"] == "DecontX", "prob"
    ].gt(0).any()


def test_figure_05b_subtitle_uses_current_interaction_union_size(tmp_path: Path):
    _write_fixture(tmp_path)
    prepared = prepare_spatial_cellchat_data(tmp_path)
    expected = prepared["interaction_summary"]["n"].iloc[0]

    figure = figure_05b(prepared)
    try:
        text = " ".join(item.get_text() for item in figure.axes[0].texts)
        assert f"{expected:,} matched interactions" in text
        assert "5,492 matched interactions" not in text
    finally:
        plt.close(figure)


def test_empty_method_table_is_rejected(tmp_path: Path):
    _write_fixture(tmp_path, empty_method="DecontX")

    with pytest.raises(ValueError, match="non-estimable"):
        prepare_spatial_cellchat_data(tmp_path)
