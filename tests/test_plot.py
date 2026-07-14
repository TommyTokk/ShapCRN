import numpy as np
import pandas as pd

from shapcrn.utils.plot import plot_heatmap


def test_plot_heatmap_supports_current_matplotlib(tmp_path):
    data = pd.DataFrame(
        [[1.0, np.nan]],
        index=["S1"],
        columns=["S1", "S2"],
    )

    figure, axes = plot_heatmap(
        data,
        y_labels=["S1"],
        x_labels=["S1", "S2"],
        colnames_to_index={},
        cmap="viridis",
        save_path=tmp_path,
        img_name="heatmap.png",
    )

    assert figure is not None
    assert axes is not None
    assert (tmp_path / "heatmap.png").is_file()