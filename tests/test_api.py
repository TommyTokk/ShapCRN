import libsbml
import numpy as np
import pandas as pd
import pytest

from shapcrn import api
from shapcrn import (
    ImportanceResult,
    SimulationResult,
    knockin_species,
    knockin_reaction,
    knockout_reaction,
    knockout_species,
    simulate_model,
)
from shapcrn.api import _sobol_frames
from shapcrn.exceptions import AnalysisError, InvalidReactionError, InvalidSpeciesError


def test_simulate_model_returns_requested_points_without_writing(model_path, tmp_path):
    before = set(tmp_path.iterdir())
    result = simulate_model(model_path, end_time=2, points=17)

    assert isinstance(result, SimulationResult)
    assert len(result.data) == 17
    assert result.artifacts == ()
    assert set(tmp_path.iterdir()) == before


def test_simulate_model_writes_only_when_requested(model_path, tmp_path):
    result = simulate_model(model_path, end_time=1, points=5, output_dir=tmp_path)

    assert any(path.name == "simulation.csv" for path in result.artifacts)
    assert all(path.exists() for path in result.artifacts)


def test_model_edits_clone_by_default_and_do_not_write(model_path, tmp_path):
    original = libsbml.readSBMLFromFile(str(model_path)).getModel()
    modified = knockout_species(original, "S1")

    assert original.getSpecies("S1").getInitialConcentration() == 1
    assert modified.getSpecies("S1").getInitialConcentration() == 0
    assert list(tmp_path.iterdir()) == []

    output_path = tmp_path / "knocked.xml"
    knockout_species(original, "S1", output_path=output_path)
    assert output_path.is_file()


def test_knockout_reaction_and_knockin_species(model_path):
    knocked = knockout_reaction(model_path, "R1")
    assert libsbml.formulaToL3String(knocked.getReaction("R1").getKineticLaw().getMath()) == "0"

    knocked_in = knockin_species(model_path, "S1", value=2.5)
    species = knocked_in.getSpecies("S1")
    assert species.getInitialConcentration() == 2.5
    assert species.getBoundaryCondition()
    assert species.getConstant()

    reaction_knockin = knockin_reaction(model_path, "R1", values=[1.5])
    assert reaction_knockin.getReaction("R1_KI") is not None
    assert reaction_knockin.getSpecies("S1_KI").getInitialConcentration() == 1.5


def test_invalid_edit_targets_are_domain_errors(model_path):
    with pytest.raises(InvalidSpeciesError):
        knockout_species(model_path, "missing")
    with pytest.raises(InvalidReactionError):
        knockout_reaction(model_path, "missing")


def test_non_finite_sobol_output_fails_explicitly():
    problem = {"num_vars": 1, "names": ["S1"], "bounds": [[0, 1]]}
    with pytest.raises(AnalysisError, match="non-finite"):
        _sobol_frames(problem, np.array([[1.0], [np.nan]]), ["S2"])


def test_importance_api_returns_data_without_writing(monkeypatch, model_path, tmp_path):
    fake_model = object()
    original = [pd.DataFrame({"[S1]": [1.0], "[S2]": [0.0]})]
    monkeypatch.setattr(
        api.importance_pipeline,
        "model_preparation",
        lambda args: {"sbml_model": fake_model, "knocked_ids": ["S1"]},
    )
    monkeypatch.setattr(api.importance_pipeline, "generate_samples", lambda *a, **k: None)
    monkeypatch.setattr(
        api.importance_pipeline,
        "simulate_original_model",
        lambda *a, **k: (original, ["time", "[S1]", "[S2]"], 10),
    )
    monkeypatch.setattr(
        api.importance_pipeline,
        "simulate_knocked_data",
        lambda *a, **k: [("S1", [original[0]])],
    )
    expected = pd.DataFrame([[0.0]], index=["S1"], columns=["[S2]"])
    monkeypatch.setattr(
        api.simulation_utils,
        "get_relative_variations_log_ratio_no_samples",
        lambda *a, **k: expected,
    )

    result = api.assess_importance(model_path)

    assert isinstance(result, ImportanceResult)
    pd.testing.assert_frame_equal(result.variations, expected)
    assert result.artifacts == ()
    assert list(tmp_path.iterdir()) == []
