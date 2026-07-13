"""Public API for the ShapCRN package."""

from shapcrn import exceptions
from shapcrn.api import (
    ImportanceResult,
    SensitivityResult,
    SimulationResult,
    analyze_sensitivity,
    assess_importance,
    knockin_reaction,
    knockin_species,
    knockout_reaction,
    knockout_species,
    simulate_model,
)
from shapcrn.pipelines.network import create_model_network
from shapcrn.utils import graph, plot, sensitivity, simulation
from shapcrn.utils.sbml import io as sbml_io
from shapcrn.utils.sbml import knock as sbml_knock
from shapcrn.utils.sbml import reactions as sbml_reactions
from shapcrn.utils.sbml import utils as sbml_utils
from shapcrn.utils.sbml.io import load_and_prepare_model, load_model, save_sbml_model
from shapcrn.utils.simulation import (
    load_roadrunner_model,
    simulate,
    simulate_with_steady_state,
)
from shapcrn.utils.utils import parse_args, setup_output_dirs

# Compatibility aliases retained for code written against the pre-release API.
importance_assessment = assess_importance
sensitivity_analysis = analyze_sensitivity

__all__ = [
    "SimulationResult",
    "ImportanceResult",
    "SensitivityResult",
    "simulate_model",
    "assess_importance",
    "analyze_sensitivity",
    "importance_assessment",
    "sensitivity_analysis",
    "knockout_species",
    "knockout_reaction",
    "knockin_species",
    "knockin_reaction",
    "create_model_network",
    "load_model",
    "load_and_prepare_model",
    "save_sbml_model",
    "load_roadrunner_model",
    "simulate",
    "simulate_with_steady_state",
    "parse_args",
    "setup_output_dirs",
    "simulation",
    "plot",
    "sensitivity",
    "graph",
    "sbml_io",
    "sbml_utils",
    "sbml_knock",
    "sbml_reactions",
    "exceptions",
]
