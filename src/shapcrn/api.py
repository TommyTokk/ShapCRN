"""Stable, side-effect-controlled public API for :mod:`shapcrn`."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import libsbml
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from shapcrn import exceptions
from shapcrn.pipelines import importance as importance_pipeline
from shapcrn.utils import plot as plot_utils
from shapcrn.utils import sensitivity as sensitivity_utils
from shapcrn.utils import simulation as simulation_utils
from shapcrn.utils import utils as common_utils
from shapcrn.utils.sbml import io as sbml_io
from shapcrn.utils.sbml import knock as knock_utils
from shapcrn.utils.sbml import reactions as reaction_utils
from shapcrn.utils.sbml import species as species_utils
from shapcrn.utils.sbml import utils as sbml_utils

ModelSource: TypeAlias = str | PathLike[str] | libsbml.Model | libsbml.SBMLDocument
Integrator: TypeAlias = Literal["cvode", "gillespie", "rk4"]


@dataclass(frozen=True)
class SimulationResult:
    """Result of a time-course simulation."""

    data: pd.DataFrame
    steady_state_time: float | None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ImportanceResult:
    """Result of a knockout/knockin importance assessment."""

    variations: pd.DataFrame
    shapley_values: pd.DataFrame | None = None
    perturbation_statistics: dict | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SensitivityResult:
    """Sobol sensitivity results in tidy tabular form."""

    indices: pd.DataFrame
    interactions: pd.DataFrame
    convergence: dict | None = None
    fixed_comparison: pd.DataFrame | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)


def _clone_model(model: libsbml.Model) -> libsbml.Model:
    clone = model.clone()
    if clone is None:
        raise exceptions.InvalidModelFormatError("in-memory model")
    return clone


def _model_from_source(
    source: ModelSource,
    *,
    prepare: bool = False,
    inplace: bool = False,
    log_file: str | PathLike[str] | None = None,
) -> libsbml.Model:
    if isinstance(source, libsbml.Model):
        model = source if inplace else _clone_model(source)
        return (
            reaction_utils.split_all_reversible_reactions(model, log_file)
            if prepare
            else model
        )
    if isinstance(source, libsbml.SBMLDocument):
        model = source.getModel()
        if model is None:
            raise exceptions.InvalidModelFormatError("in-memory document")
        model = model if inplace else _clone_model(model)
        return (
            reaction_utils.split_all_reversible_reactions(model, log_file)
            if prepare
            else model
        )

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SBML model does not exist: {path}")
    document, model = sbml_io.load_and_prepare_model(
        str(path), split_reversible=prepare, log_file=log_file
    )
    del document
    if model is None:
        raise exceptions.InvalidModelFormatError(str(path))
    return model


def _model_name(source: ModelSource, model: libsbml.Model) -> str:
    if isinstance(source, (str, PathLike)):
        return Path(source).stem
    return model.getId() or "model"


def _save_if_requested(
    model: libsbml.Model,
    output_path: str | PathLike[str] | None,
    log_file: str | PathLike[str] | None,
) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sbml_io.save_sbml_model(model, str(path), log_file=log_file)


def simulate_model(
    model: ModelSource,
    *,
    end_time: float = 10.0,
    start_time: float = 0.0,
    integrator: Integrator = "cvode",
    steady_state: bool = False,
    max_time: float = 1000.0,
    step: float = 5.0,
    points: int = 1000,
    threshold: float = 1e-6,
    output_dir: str | PathLike[str] | None = None,
    interactive: bool = False,
    log_file: str | PathLike[str] | None = None,
) -> SimulationResult:
    """Simulate an SBML model, writing artifacts only when ``output_dir`` is set."""
    if points < 2:
        raise exceptions.InvalidArgumentError("points", points, "must be at least 2")
    sbml_model = _model_from_source(model, prepare=True, log_file=log_file)
    rr_model = simulation_utils.load_roadrunner_model(
        sbml_model, integrator=integrator, log_file=log_file
    )
    values, ss_time, columns = simulation_utils.simulate(
        rr_model,
        start_time=start_time,
        end_time=end_time,
        output_rows=points,
        steady_state=steady_state,
        max_end_time=max_time,
        sim_step=step,
        threshold=threshold,
        log_file=log_file,
    )
    data = pd.DataFrame(values, columns=columns)
    artifacts: list[Path] = []

    if output_dir is not None:
        out_dirs = common_utils.setup_output_dirs(
            str(output_dir), _model_name(model, sbml_model)
        )
        csv_path = Path(out_dirs["csv"]) / "simulation.csv"
        data.to_csv(csv_path, index=False)
        artifacts.append(csv_path)
        if interactive:
            html_path = Path(out_dirs["images"]) / "simulation.html"
            plot_utils.plot_results_interactive(
                data,
                model_name="",
                html_dir_path=out_dirs["images"],
                html_name=html_path.name,
                ss_time=ss_time,
                log_file=log_file,
            )
            artifacts.append(html_path)
        else:
            image_path = Path(out_dirs["images"]) / "simulation.png"
            plot_utils.plot_results(
                data,
                img_dir_path=out_dirs["images"],
                img_name=image_path.name,
                ss_time=ss_time,
                log_file=log_file,
            )
            artifacts.append(image_path)

    return SimulationResult(data=data, steady_state_time=ss_time, artifacts=tuple(artifacts))


def _importance_arguments(
    model_path: str,
    *,
    operation: Literal["knockout", "knockin"],
    input_species: Sequence[str] | None,
    knocked_species: Sequence[str] | None,
    target_nodes: Sequence[str] | None,
    preserve_inputs: bool,
    use_perturbations: bool,
    fixed_perturbations: Sequence[float] | None,
    num_samples: int,
    variation: float,
    max_combinations: int | None,
    payoff: Literal["max", "min", "last"],
    end_time: float,
    integrator: Integrator,
    steady_state: bool,
    max_time: float,
    step: float,
    points: int,
    threshold: float,
    n_jobs: int,
    log_file: str | PathLike[str] | None,
) -> dict:
    return {
        "input_path": model_path,
        "operation": operation,
        "input_species_ids": list(input_species or []),
        "knocked_species_ids": list(knocked_species) if knocked_species else None,
        "target_ids": list(target_nodes) if target_nodes else None,
        "preserve_inputs": preserve_inputs,
        "use_perturbations": use_perturbations,
        "max_combinations": max_combinations,
        "use_fixed_perturbations": fixed_perturbations is not None,
        "fixed_perturbations": list(fixed_perturbations) if fixed_perturbations else None,
        "num_samples": num_samples,
        "variation_percentage": variation,
        "perturbations_importance": False,
        "random_perturbations_importance": False,
        "payoff_function": payoff,
        "sim_time": end_time,
        "sim_integrator": integrator,
        "use_steady_state": steady_state,
        "ss_max_time": max_time,
        "ss_sim_steps": step,
        "ss_sim_points": points,
        "ss_threshold": threshold,
        "n_jobs": n_jobs,
        "output_dir": None,
        "log_file": str(log_file) if log_file is not None else None,
    }


def assess_importance(
    model_path: str | PathLike[str],
    *,
    operation: Literal["knockout", "knockin"] = "knockout",
    input_species: Sequence[str] | None = None,
    knocked_species: Sequence[str] | None = None,
    target_nodes: Sequence[str] | None = None,
    preserve_inputs: bool = False,
    use_perturbations: bool = False,
    fixed_perturbations: Sequence[float] | None = None,
    num_samples: int = 5,
    variation: float = 20.0,
    max_combinations: int | None = None,
    payoff: Literal["max", "min", "last"] = "last",
    end_time: float = 10.0,
    integrator: Integrator = "cvode",
    steady_state: bool = False,
    max_time: float = 1000.0,
    step: float = 5.0,
    points: int = 1000,
    threshold: float = 1e-6,
    n_jobs: int = -1,
    seed: int | None = None,
    output_dir: str | PathLike[str] | None = None,
    log_file: str | PathLike[str] | None = None,
) -> ImportanceResult:
    """Assess species importance without requiring an argparse namespace."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"SBML model does not exist: {path}")
    if use_perturbations and not input_species:
        raise exceptions.InvalidArgumentError(
            "input_species", input_species, "required when perturbations are enabled"
        )
    if fixed_perturbations is not None and not fixed_perturbations:
        raise exceptions.InvalidArgumentError(
            "fixed_perturbations", fixed_perturbations, "must not be empty"
        )
    if num_samples < 1:
        raise exceptions.InvalidArgumentError(
            "num_samples", num_samples, "must be at least one"
        )
    if points < 2:
        raise exceptions.InvalidArgumentError("points", points, "must be at least 2")

    args = _importance_arguments(
        str(path),
        operation=operation,
        input_species=input_species,
        knocked_species=knocked_species,
        target_nodes=target_nodes,
        preserve_inputs=preserve_inputs,
        use_perturbations=use_perturbations,
        fixed_perturbations=fixed_perturbations,
        num_samples=num_samples,
        variation=variation,
        max_combinations=max_combinations,
        payoff=payoff,
        end_time=end_time,
        integrator=integrator,
        steady_state=steady_state,
        max_time=max_time,
        step=step,
        points=points,
        threshold=threshold,
        n_jobs=n_jobs,
        log_file=log_file,
    )
    prepared = importance_pipeline.model_preparation(args)
    sbml_model = prepared["sbml_model"]
    knocked_ids = prepared["knocked_ids"]
    samples = importance_pipeline.generate_samples(sbml_model, args, seed=seed)
    original, selections, min_ss_time = importance_pipeline.simulate_original_model(
        sbml_model, knocked_ids, samples, args
    )
    new_values = None
    if operation == "knockin":
        new_values = list(original[0][selections[1:]].max())
    knocked = importance_pipeline.simulate_knocked_data(
        sbml_model,
        knocked_ids,
        samples,
        selections,
        min_ss_time,
        args,
        new_values=new_values,
    )

    shapley_values = None
    perturbation_statistics = None
    if use_perturbations:
        shapley_values = importance_pipeline.run_shap_analysis(
            original,
            knocked,
            len(original),
            len(args["input_species_ids"]),
            payoff=payoff,
            log_file=args["log_file"],
        )
        variations = simulation_utils.get_relative_variations_log_ratio(
            original, knocked, aggregation="median", return_signed=False
        )
        perturbation_statistics = importance_pipeline.assess_perturbation_importance(
            original, knocked, log_file=args["log_file"]
        )
    else:
        variations = simulation_utils.get_relative_variations_log_ratio_no_samples(
            original[0], knocked, return_signed=True
        )

    if target_nodes:
        columns = [
            target if target in variations.columns else f"[{target}]"
            for target in target_nodes
        ]
        missing = [column for column in columns if column not in variations.columns]
        if missing:
            raise exceptions.InvalidSpeciesError(", ".join(missing))
        variations = variations[columns]
        if shapley_values is not None:
            shapley_values = shapley_values[columns]

    artifacts: list[Path] = []
    if output_dir is not None:
        out_dirs = common_utils.setup_output_dirs(str(output_dir), path.stem)
        variations_path = Path(out_dirs["csv"]) / "variations.csv"
        variations.to_csv(variations_path)
        artifacts.append(variations_path)
        variation_image = Path(out_dirs["images"]) / "variations_heatmap.png"
        plot_utils.plot_heatmap(
            variations,
            colnames_to_index={name: index for index, name in enumerate(variations.columns)},
            x_labels=variations.columns,
            y_labels=variations.index,
            title="Relative variations",
            img_name=variation_image.name,
            save_path=out_dirs["images"],
            log_file=args["log_file"],
        )
        artifacts.append(variation_image)
        if shapley_values is not None:
            shapley_path = Path(out_dirs["csv"]) / "shapley_values.csv"
            shapley_values.to_csv(shapley_path)
            artifacts.append(shapley_path)
            normalized, _ = common_utils.normalize_asinh(shapley_values)
            shapley_image = Path(out_dirs["images"]) / "shapley_values_heatmap.png"
            plot_utils.plot_heatmap(
                normalized,
                colnames_to_index={
                    name: index for index, name in enumerate(shapley_values.columns)
                },
                x_labels=shapley_values.columns,
                y_labels=shapley_values.index,
                title="Shapley values (asinh normalized)",
                img_name=shapley_image.name,
                save_path=out_dirs["images"],
                log_file=args["log_file"],
            )
            artifacts.append(shapley_image)

    return ImportanceResult(
        variations=variations,
        shapley_values=shapley_values,
        perturbation_statistics=perturbation_statistics,
        artifacts=tuple(artifacts),
    )


def _sobol_frames(problem: dict, outputs: np.ndarray, targets: Sequence[str]):
    index_rows: list[dict] = []
    interaction_rows: list[dict] = []
    result_dict: dict[str, dict] = {}
    for column, target in enumerate(targets):
        values = outputs[:, column]
        non_finite = int((~np.isfinite(values)).sum())
        if non_finite:
            raise exceptions.AnalysisError(
                f"Sobol design for target '{target}' contains {non_finite} non-finite results"
            )
        result = sobol_analyze.analyze(
            problem, values, calc_second_order=True, print_to_console=False
        )
        result_dict[target] = result
        for index, input_name in enumerate(problem["names"]):
            index_rows.append(
                {
                    "target": target,
                    "input": input_name,
                    "S1": result["S1"][index],
                    "S1_conf": result["S1_conf"][index],
                    "ST": result["ST"][index],
                    "ST_conf": result["ST_conf"][index],
                }
            )
        for left in range(problem["num_vars"]):
            for right in range(left + 1, problem["num_vars"]):
                interaction_rows.append(
                    {
                        "target": target,
                        "input_i": problem["names"][left],
                        "input_j": problem["names"][right],
                        "S2": result["S2"][left, right],
                        "S2_conf": result["S2_conf"][left, right],
                    }
                )
    return (
        pd.DataFrame(index_rows),
        pd.DataFrame(
            interaction_rows,
            columns=["target", "input_i", "input_j", "S2", "S2_conf"],
        ),
        result_dict,
    )


def _fixed_comparison(
    rr_model,
    sampled: np.ndarray,
    model: libsbml.Model,
    input_species: Sequence[str],
    targets: Sequence[str],
    valid_indices: dict,
    fixed_perturbations: Sequence[float],
    log_file,
) -> pd.DataFrame:
    fixed_samples = sbml_utils.get_fixed_combinations(
        model, list(input_species), list(fixed_perturbations), log_file
    )
    fixed_results, _ = simulation_utils.simulate_combinations(
        rr_model,
        sbml_utils.create_combinations(fixed_samples),
        list(input_species),
        min_ss_time=1000,
        end_time=5000,
        max_end_time=5000,
        steady_state=False,
        log_file=log_file,
    )
    rows = []
    for column, target in enumerate(targets):
        selection = f"[{target}]"
        fixed = np.asarray(
            [result[-1, valid_indices[selection]] for result in fixed_results],
            dtype=float,
        )
        random = sampled[:, column]
        fixed = fixed[np.isfinite(fixed)]
        random = random[np.isfinite(random)]
        rows.append(
            {
                "target": target,
                "fixed_count": fixed.size,
                "sampled_count": random.size,
                "fixed_mean": np.mean(fixed) if fixed.size else np.nan,
                "sampled_mean": np.mean(random) if random.size else np.nan,
                "fixed_std": np.std(fixed, ddof=1) if fixed.size > 1 else np.nan,
                "sampled_std": np.std(random, ddof=1) if random.size > 1 else np.nan,
                "wasserstein_distance": (
                    common_utils.wasserstein_1d(fixed, random)
                    if fixed.size and random.size
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_sensitivity(
    model_path: str | PathLike[str],
    *,
    input_species: Sequence[str],
    target_species: Sequence[str] | None = None,
    base_samples: int = 1024,
    perturbation_range: float = 20.0,
    check_convergence: bool = False,
    fixed_perturbations: Sequence[float] | None = None,
    seed: int | None = None,
    n_jobs: int | None = None,
    simulation_end_time: float = 5000.0,
    output_dir: str | PathLike[str] | None = None,
    log_file: str | PathLike[str] | None = None,
) -> SensitivityResult:
    """Run a Sobol analysis and return first-, total-, and second-order indices."""
    if not input_species:
        raise exceptions.InvalidArgumentError(
            "input_species", input_species, "at least one input is required"
        )
    if not isinstance(base_samples, int) or base_samples < 2:
        raise exceptions.InvalidArgumentError(
            "base_samples", base_samples, "must be an integer greater than one"
        )
    path = Path(model_path)
    model = _model_from_source(path, prepare=True, log_file=log_file)
    all_species = species_utils.get_list_of_species_ids(model)
    for species_id in input_species:
        if species_id not in all_species:
            raise exceptions.InvalidSpeciesError(species_id, model.getId())
    targets = list(
        target_species
        or [species_id for species_id in all_species if species_id not in input_species]
    )
    for species_id in targets:
        if species_id not in all_species:
            raise exceptions.InvalidSpeciesError(species_id, model.getId())

    problem = sensitivity_utils.get_problem_parameters(
        model,
        len(input_species),
        list(input_species),
        perturbation_range=perturbation_range,
        log_file=log_file,
    )
    rr_model = simulation_utils.load_roadrunner_model(model, log_file=log_file)
    selections = rr_model.timeCourseSelections
    for target in targets:
        selection = f"[{target}]"
        if selection not in selections:
            selections.append(selection)
    rr_model.timeCourseSelections = selections
    valid_indices = {
        f"[{target}]": rr_model.timeCourseSelections.index(f"[{target}]")
        for target in targets
    }
    valid_elements = list(valid_indices)

    convergence = None
    sample_size = base_samples
    if check_convergence:
        candidates = [value for value in (64, 128, 256, 512, 1024) if value <= base_samples]
        if not candidates:
            candidates = [base_samples]
        convergence_results = {}
        for value in candidates:
            params = sobol_sample.sample(
                problem, value, calc_second_order=True, seed=seed
            )
            outputs = sensitivity_utils.run_simulation_with_params(
                rr_model,
                params,
                valid_elements,
                valid_indices,
                list(input_species),
                log_file=log_file,
                n_processes=n_jobs,
                sim_end_time=simulation_end_time,
            )
            _, _, results = _sobol_frames(problem, outputs, targets)
            convergence_results[value] = {
                f"[{target}]": results[target] for target in targets
            }
        convergence = sensitivity_utils.check_convergence(
            convergence_results,
            valid_elements,
            tol_ci=0.10,
            min_consecutive=2,
            log_file=log_file,
        )
        converged_at = [
            details["converged_at"]
            for details in convergence.values()
            if details["converged_at"] is not None
        ]
        sample_size = max(converged_at) if len(converged_at) == len(targets) else candidates[-1]

    params = sobol_sample.sample(
        problem, sample_size, calc_second_order=True, seed=seed
    )
    outputs = sensitivity_utils.run_simulation_with_params(
        rr_model,
        params,
        valid_elements,
        valid_indices,
        list(input_species),
        log_file=log_file,
        n_processes=n_jobs,
        sim_end_time=simulation_end_time,
    )
    indices, interactions, results = _sobol_frames(problem, outputs, targets)
    fixed_comparison = None
    if fixed_perturbations is not None:
        fixed_comparison = _fixed_comparison(
            rr_model,
            outputs,
            model,
            input_species,
            targets,
            valid_indices,
            fixed_perturbations,
            log_file,
        )

    artifacts: list[Path] = []
    if output_dir is not None:
        out_dirs = common_utils.setup_output_dirs(str(output_dir), path.stem)
        indices_path = Path(out_dirs["csv"]) / "sobol_indices.csv"
        interactions_path = Path(out_dirs["csv"]) / "sobol_interactions.csv"
        report_path = Path(out_dirs["reports"]) / "sensitivity_report.txt"
        indices.to_csv(indices_path, index=False)
        interactions.to_csv(interactions_path, index=False)
        sensitivity_utils.report_sensitivity(
            results, list(problem["names"]), str(report_path)
        )
        artifacts.extend((indices_path, interactions_path, report_path))
        if fixed_comparison is not None:
            fixed_path = Path(out_dirs["csv"]) / "sensitivity_comparison.csv"
            fixed_comparison.to_csv(fixed_path, index=False)
            artifacts.append(fixed_path)
        if convergence is not None:
            convergence_path = Path(out_dirs["reports"]) / "convergence_report.txt"
            sensitivity_utils.convergence_report(convergence, str(convergence_path))
            sensitivity_utils.plot_convergence_single_plot(
                convergence,
                file_name="sensitivity_convergence",
                output_dir=out_dirs["images"],
            )
            artifacts.extend(
                (
                    convergence_path,
                    Path(out_dirs["images"]) / "sensitivity_convergence.png",
                )
            )

    return SensitivityResult(
        indices=indices,
        interactions=interactions,
        convergence=convergence,
        fixed_comparison=fixed_comparison,
        artifacts=tuple(artifacts),
    )


def knockout_species(
    model: ModelSource,
    species_id: str,
    *,
    inplace: bool = False,
    output_path: str | PathLike[str] | None = None,
    log_file: str | PathLike[str] | None = None,
) -> libsbml.Model:
    modified = knock_utils.knockout_species(
        _model_from_source(model, inplace=inplace, log_file=log_file),
        species_id,
        log_file,
    )
    _save_if_requested(modified, output_path, log_file)
    return modified


def knockout_reaction(
    model: ModelSource,
    reaction_id: str,
    *,
    inplace: bool = False,
    output_path: str | PathLike[str] | None = None,
    log_file: str | PathLike[str] | None = None,
) -> libsbml.Model:
    modified = knock_utils.knockout_reaction(
        _model_from_source(model, inplace=inplace, log_file=log_file),
        reaction_id,
        log_file,
    )
    _save_if_requested(modified, output_path, log_file)
    return modified


def knockin_species(
    model: ModelSource,
    species_id: str,
    *,
    value: float | None = None,
    inplace: bool = False,
    output_path: str | PathLike[str] | None = None,
    log_file: str | PathLike[str] | None = None,
) -> libsbml.Model:
    sbml_model = _model_from_source(model, inplace=inplace, log_file=log_file)
    if value is None:
        value = simulation_utils.get_species_peak_value(
            sbml_model, species_id, log_file=log_file
        )
    modified = knock_utils.knockin_species(sbml_model, species_id, value, log_file)
    _save_if_requested(modified, output_path, log_file)
    return modified


def knockin_reaction(
    model: ModelSource,
    reaction_id: str,
    *,
    values: Sequence[float] | None = None,
    inplace: bool = False,
    output_path: str | PathLike[str] | None = None,
    log_file: str | PathLike[str] | None = None,
) -> libsbml.Model:
    sbml_model = _model_from_source(model, inplace=inplace, log_file=log_file)
    reaction = sbml_model.getReaction(reaction_id)
    if reaction is None:
        raise exceptions.InvalidReactionError(reaction_id, sbml_model.getId())
    if values is None:
        values = simulation_utils.get_reactants_peak_values(
            sbml_model, reaction, log_file=log_file
        )
    modified = knock_utils.knockin_reaction(
        sbml_model, reaction, list(values), log_file
    )
    _save_if_requested(modified, output_path, log_file)
    return modified
