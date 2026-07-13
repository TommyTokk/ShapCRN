import numpy as np
import pandas as pd

from shapcrn import api
from shapcrn.api import SensitivityResult


class FakeRoadRunner:
    def __init__(self):
        self.timeCourseSelections = ["time", "[S1]", "[S2]"]


def _patch_sensitivity_runtime(monkeypatch, sample_sizes):
    monkeypatch.setattr(
        api.simulation_utils, "load_roadrunner_model", lambda *args, **kwargs: FakeRoadRunner()
    )

    def sample(problem, size, **kwargs):
        sample_sizes.append(size)
        return np.zeros((size, problem["num_vars"]))

    monkeypatch.setattr(api.sobol_sample, "sample", sample)
    monkeypatch.setattr(
        api.sensitivity_utils,
        "run_simulation_with_params",
        lambda rr, params, elements, indexes, inputs, **kwargs: np.ones(
            (len(params), len(elements))
        ),
    )

    def frames(problem, outputs, targets):
        indices = pd.DataFrame(
            [{"target": targets[0], "input": problem["names"][0], "S1": 1.0, "ST": 1.0}]
        )
        interactions = pd.DataFrame()
        sobol = {
            targets[0]: {
                "S1": np.array([1.0]),
                "S1_conf": np.array([0.0]),
                "ST": np.array([1.0]),
                "ST_conf": np.array([0.0]),
                "S2": np.array([[np.nan]]),
                "S2_conf": np.array([[np.nan]]),
            }
        }
        return indices, interactions, sobol

    monkeypatch.setattr(api, "_sobol_frames", frames)


def test_sensitivity_uses_base_samples_without_fixed_values(
    monkeypatch, model_path
):
    sample_sizes = []
    _patch_sensitivity_runtime(monkeypatch, sample_sizes)

    result = api.analyze_sensitivity(
        model_path, input_species=["S1"], target_species=["S2"], base_samples=128
    )

    assert isinstance(result, SensitivityResult)
    assert sample_sizes == [128]
    assert result.fixed_comparison is None


def test_convergence_is_followed_by_final_analysis(monkeypatch, model_path):
    sample_sizes = []
    _patch_sensitivity_runtime(monkeypatch, sample_sizes)
    monkeypatch.setattr(
        api.sensitivity_utils,
        "check_convergence",
        lambda *args, **kwargs: {
            "[S2]": {
                "converged_at": 64,
                "diverged_after": None,
                "max_change": {},
                "ci_half_width": {},
            }
        },
    )

    result = api.analyze_sensitivity(
        model_path,
        input_species=["S1"],
        target_species=["S2"],
        base_samples=64,
        check_convergence=True,
    )

    assert sample_sizes == [64, 64]
    assert result.convergence["[S2]"]["converged_at"] == 64
