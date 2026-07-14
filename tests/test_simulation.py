import numpy as np
import pandas as pd

from shapcrn.utils.simulation import get_shapley_values


def test_shapley_values_mask_self_comparisons_with_pandas_3():
    payoff_values = [
        (
            "S1",
            pd.DataFrame(
                {
                    "[S1]": [1.0, 3.0],
                    "[S2]": [2.0, 4.0],
                }
            ),
        ),
        (
            "S2",
            pd.DataFrame(
                {
                    "[S1]": [5.0, 7.0],
                    "[S2]": [6.0, 8.0],
                }
            ),
        ),
    ]

    result = get_shapley_values(
        payoff_values,
        n_combinations=2,
        n_inputs=1,
    )

    assert np.isnan(result.loc["S1", "[S1]"])
    assert np.isnan(result.loc["S2", "[S2]"])
    assert result.loc["S1", "[S2]"] == 3.0
    assert result.loc["S2", "[S1]"] == 6.0