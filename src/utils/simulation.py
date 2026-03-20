from decimal import *
import sys

import pandas as pd
from pandas.io.html import pprint_thing
from scipy.special import factorial
from scipy.stats import variation

import roadrunner as rr
import libsbml
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
import multiprocessing as mp
from multiprocessing import Pool
import datetime

from src import exceptions
from src.utils.utils import (
    print_log,
)
from src.utils import plot as plt_ut
from src.utils.sbml.utils import create_combinations

# from utils.sbml_utils import create_samples_combination, generate_species_samples


# KEEP
def load_roadrunner_model(
    sbml_model: libsbml.Model,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12,
    integrator: str = "cvode",
    log_file=None,
) -> rr.RoadRunner:
    """
    Load an SBML model into a RoadRunner instance and configure integrator settings.

    This function creates a RoadRunner instance from an SBML model and configures the
    numerical integrator with specified tolerance values. It supports multiple integrator
    types for different simulation approaches (deterministic ODE, Runge-Kutta, or stochastic).

    Parameters
    ----------
    sbml_model : libsbml.Model or str
        The SBML model to load. Can be either:
        - libsbml.Model: A model object (will be converted to SBML string)
        - str: SBML XML string or file path
    rel_tol : float, optional
        Relative tolerance for the integrator, by default 1e-8
    abs_tol : float, optional
        Absolute tolerance for the integrator, by default 1e-12
    integrator : str, optional
        Integrator type to use for simulations, by default "cvode"
        Supported values:
        - "cvode": CVODE integrator for ODEs (deterministic)
        - "rk4": Fourth-order Runge-Kutta method
        - "gillespie": Gillespie stochastic simulation algorithm
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    roadrunner.RoadRunner
        Configured RoadRunner instance ready for simulation

    Raises
    ------
    SystemExit
        If model loading fails due to invalid SBML or file path errors

    Notes
    -----
    The function automatically handles conversion between libsbml.Model objects and
    SBML string representations required by RoadRunner.

    Integrator tolerances affect simulation accuracy and performance:
    - Lower tolerances increase accuracy but slow down simulations
    - Higher tolerances speed up simulations but may reduce accuracy

    Examples
    --------
    Load model from libsbml.Model object:
    >>> rr_model = load_roadrunner_model(sbml_model, rel_tol=1e-6, abs_tol=1e-10)

    Load model with stochastic integrator:
    >>> rr_model = load_roadrunner_model(
    ...     sbml_model, integrator="gillespie", log_file=log
    ... )

    Load from SBML file path:
    >>> rr_model = load_roadrunner_model("model.xml", integrator="rk4")
    """

    try:
        writer = libsbml.SBMLWriter()
        # Check if input is a libSBML model or a string
        if isinstance(sbml_model, libsbml.Model):
            sbml_doc = writer.writeSBMLToString(sbml_model.getSBMLDocument())
            rr_model = rr.RoadRunner(sbml_doc)
        else:
            # Assume it's a string representation or file path
            rr_model = rr.RoadRunner(sbml_model)
    except Exception as e:
        raise exceptions.ModelError(f"Failed to load SBML model: {str(e)}")

    # Configure integrator settings
    # Setting the relative and absolute tolerance of the model
    rr_model.getIntegrator().setValue("relative_tolerance", rel_tol)
    rr_model.getIntegrator().setValue("absolute_tolerance", abs_tol)

    # Setting the integrator if necessary
    if integrator is not None:
        rr_model.setIntegrator(integrator)

    return rr_model


# KEEP
def simulate(
    rr_model: rr.RoadRunner,
    start_time: float = 0,
    end_time: float = 1000,
    output_rows: int = 100,
    steady_state: bool = False,
    max_end_time: float = 1000,
    sim_step: int = 5,
    threshold: float = 1e-6,
    log_file=None,
) -> tuple:
    """
    Simulate a RoadRunner model with optional steady state detection.

    This function runs either a standard time-course simulation or an adaptive simulation
    that continues until steady state is reached. Steady state detection uses a block-by-block
    approach to monitor concentration changes.

    Parameters
    ----------
    rr_model : roadrunner.RoadRunner
        Configured RoadRunner model instance to simulate
    start_time : float, optional
        Simulation start time, by default 0
    end_time : float, optional
        Simulation end time (used when steady_state=False), by default 1000
    output_rows : int, optional
        Number of output time points in the results, by default 100
    steady_state : bool, optional
        If True, run adaptive simulation until steady state is reached, by default False
    max_end_time : float, optional
        Maximum simulation time when detecting steady state, by default 1000
    sim_step : int, optional
        Time step size for each simulation block during steady state detection, by default 5
    threshold : float, optional
        Convergence threshold for steady state detection (max relative change), by default 1e-6
    log_file : file, optional
        File object for logging simulation progress, by default None

    Returns
    -------
    tuple of (numpy.ndarray, float or None, list)
        A tuple containing:
        - simulation_results : Structured array with time-course data (columns: time, species)
        - steady_state_time : Time when steady state was reached, or None if not reached/not requested
        - colnames : List of column names from the simulation results

    Notes
    -----
    The function automatically sets `nonnegative = True` on the integrator to prevent
    negative concentration values during stochastic simulations.

    When `steady_state=True`:
    - Uses `simulate_with_steady_state()` for adaptive block-by-block simulation
    - Points per block is calculated as max(20, output_rows // (max_end_time / sim_step))
    - Simulation continues until convergence or max_end_time is reached
    - Steady state is detected when all monitored species change by less than threshold

    When `steady_state=False`:
    - Runs standard RoadRunner time-course simulation from start_time to end_time
    - Returns exactly output_rows time points

    Examples
    --------
    Standard time-course simulation:
    >>> results, ss_time, cols = simulate(rr_model, end_time=500, output_rows=200)
    >>> print(f"Simulation completed with {len(cols)} species")

    Steady state simulation:
    >>> results, ss_time, cols = simulate(
    ...     rr_model, steady_state=True, max_end_time=2000,
    ...     threshold=1e-8, log_file=log
    ... )
    >>> if ss_time is not None:
    ...     print(f"Steady state reached at time {ss_time}")
    """
    # Setting nonnegative for stochastic simulations
    rr_model.getIntegrator().nonnegative = True

    if steady_state:
        print_log(
            log_file,
            f"Simulating until steady state (max time: {max_end_time}, threshold: {threshold})",
        )

        # Calculate number of points per block based on output_rows and sim_step
        points_per_block = int(max(20, output_rows // (max_end_time / sim_step)))

        result, ss_time, colnames = simulate_with_steady_state(
            rr_model,
            start_time=start_time,
            max_end_time=max_end_time,
            block_size=sim_step,
            points_per_block=points_per_block,
            threshold=threshold,
        )

        if ss_time is not None:
            print_log(log_file, f"Steady state reached at time: {ss_time}")
        else:
            print_log(
                log_file,
                f"Steady state not reached within maximum time ({max_end_time})",
            )

        return result, ss_time, colnames
    else:
        # Standard simulation
        res = rr_model.simulate(start_time, end_time, output_rows)

        return res, None, res.colnames


def simulate_with_steady_state(
    rr_model: rr.RoadRunner,
    start_time: float = 0,
    max_end_time: float = 1000,
    block_size: int = 10,
    points_per_block: int = 100,
    threshold: float = 1e-12,
    consecutive_checks: int = 3,
    monitor_species: list = None,
    log_file=None,
) -> tuple:
    """
    Simulates a model until steady state is reached using a block-by-block approach.

    Parameters
    ----------
    rr_model : roadrunner.RoadRunner
        RoadRunner model instance to simulate
    start_time : float, optional
        Start time for simulation, by default 0
    max_end_time : float, optional
        Maximum end time if steady state is not reached, by default 1000
    block_size : float, optional
        Size of each simulation block, by default 10
    points_per_block : int, optional
        Number of points to calculate in each block, by default 100
    threshold : float, optional
        Threshold for steady state detection, by default 1e-12
    consecutive_checks : int, optional
        Number of consecutive blocks that must meet threshold for steady state, by default 3
    monitor_species : list of str, optional
        Species to monitor for steady state detection. If None, all floating species are monitored, by default None
    log_file : file, optional
        Optional log file for debugging output, by default None

    Returns
    -------
    tuple of (numpy.ndarray, float or None, list)
        A tuple containing:
        - simulation_results : Structured array with time-course data
        - steady_state_time : Time when steady state was reached, or None if not reached
        - column_names : List of column names from the simulation results

    Notes
    -----
    The function uses adaptive block sizing:
    - When approaching steady state, block size increases up to max_block_size (50) for efficiency
    - When not in steady state, block size decreases down to min_block_size (1.0) for better resolution
    - Steady state is detected when all monitored species show variations below threshold
    for consecutive_checks consecutive blocks
    """
    rr_model.setIntegrator("cvode")
    rr_model.reset()
    all_species = rr_model.model.getFloatingSpeciesIds()
    if monitor_species is None:
        monitor_species = all_species

    # Precompute indices to monitor
    monitor_idx = [all_species.index(s) for s in monitor_species if s in all_species]

    # Prepare column names (including 'time')
    colnames = rr_model.timeCourseSelections.copy()

    current_time = start_time
    all_results = []
    prev_block = None
    steady_state_time = None
    steady_blocks_count = 0
    initial_block_size = block_size
    zero_tol = 1e-30
    min_block_size = 1.0  # Minimum block size to prevent excessive reduction
    max_block_size = 50  # Maximum block size for efficiency
    is_steady = False

    while current_time < max_end_time:
        next_time = min(current_time + block_size, max_end_time)
        print_log(
            log_file,
            f"Simulating from {current_time} to {next_time}, block_size: {block_size}",
        )

        block_results = rr_model.simulate(current_time, next_time, points_per_block)
        all_results.append(block_results)

        if prev_block is not None:
            # Extract concentrations at the last point of each block
            prev_conc = prev_block[-1, 1 : 1 + len(all_species)]
            curr_conc = block_results[-1, 1 : 1 + len(all_species)]

            # Calculate variations for all species
            variations = np.zeros_like(prev_conc)
            small_mask = np.abs(prev_conc) < zero_tol

            # Use absolute change where previous concentration is near zero
            variations[small_mask] = np.abs(
                curr_conc[small_mask] - prev_conc[small_mask]
            )

            # Use relative change elsewhere
            variations[~small_mask] = np.abs(
                (curr_conc[~small_mask] - prev_conc[~small_mask])
                / prev_conc[~small_mask]
            )

            # Validate monitor indices
            monitor_idx = [i for i in monitor_idx if i < len(variations)]
            if not monitor_idx:
                print_log(
                    log_file, "Warning: No valid species to monitor for steady state"
                )
                monitor_idx = list(range(min(len(variations), len(all_species))))

            # Check if ALL monitored species satisfy steady-state condition
            monitored_variations = variations[monitor_idx]
            max_variation = np.max(monitored_variations)
            is_steady = max_variation < threshold

            print_log(
                log_file, f"Max variation among monitored species: {max_variation}"
            )
            print_log(log_file, f"Threshold: {threshold}")
            print_log(log_file, f"Is steady: {is_steady}")

            if is_steady:
                steady_blocks_count += 1
                print_log(
                    log_file,
                    f"Steady block #{steady_blocks_count} at time {current_time}",
                )

                # Check if we have enough consecutive steady blocks
                if steady_blocks_count >= consecutive_checks:
                    steady_state_time = current_time
                    print_log(
                        log_file, f"Steady state reached at time {steady_state_time}"
                    )
                    is_steady = True
                    break

                # Increase block size when approaching steady state for efficiency
                block_size = min(block_size * 1.2, max_block_size)
            else:
                # Reset counter if steady state is broken
                steady_blocks_count = 0
                # Reduce block size for better resolution when not in steady state
                block_size = max(block_size * 0.8, min_block_size)

        prev_block = block_results
        if not is_steady:
            current_time = next_time

    # Concatenate all results, removing duplicate boundary points
    if not all_results:
        full_results = np.empty((0, len(colnames)))
    else:
        # Remove the first row of each subsequent block to avoid time point duplication
        for i in range(1, len(all_results)):
            all_results[i] = all_results[i][1:]
        full_results = np.vstack(all_results)

    return full_results, steady_state_time, colnames


def simulate_samples(
    rr_model: rr.RoadRunner,
    combination: list,
    input_species_id: str,
    max_end_time: float = 1000,
    start_time: float = 0,
    end_time: float = 1000,
    output_rows: int = 100,
    steady_state: bool = False,
) -> tuple:
    """
    Simulate a model with a specific input species concentration combination.

    This function applies a given combination of input species concentrations to a
    RoadRunner model and runs a simulation, supporting both standard time-course
    and steady-state simulations.

    Parameters
    ----------
    rr_model : roadrunner.RoadRunner
        RoadRunner model to simulate
    combination : list of float
        Single combination of input species concentrations to apply
    input_species_id : list of str
        IDs of species considered as inputs (must match length of combination)
    max_end_time : float, optional
        Maximum end time for steady state detection, by default 1000
    start_time : float, optional
        Start time of the simulation, by default 0
    end_time : float, optional
        End time of the simulation (for non-steady-state mode), by default 1000
    output_rows : int, optional
        Number of output rows in the simulation results, by default 100
    steady_state : bool, optional
        If True, simulate until steady state is reached, by default False

    Returns
    -------
    tuple of (numpy.ndarray, float or None, list)
        A tuple containing:
        - simulation_results : Structured array with time-course data
        - steady_state_time : Time when steady state was reached, or None
        - colnames : List of column names from the simulation results

    Notes
    -----
    - The function preserves the original timeCourseSelections of the model
    - For Gillespie integrator, nonnegative is automatically set to True
    - Initial concentrations are set via `setInitConcentration()` before simulation
    - The model is reset before applying new concentrations
    """

    # Save the current selections
    current_selections = rr_model.timeCourseSelections
    rr_model.reset()

    # Set the new concentrations
    if rr_model.getIntegrator().getName() == "gillespie":
        rr_model.getIntegrator().nonnegative = True

    for i in range(len(input_species_id)):
        rr_model.setInitConcentration(input_species_id[i], combination[i])

    # rr_model.regenerateModel()

    # Reset the concentrations
    # Needed because the .reset() method reset also the simulation selections
    rr_model.timeCourseSelections = current_selections

    res = simulate(
        rr_model,
        start_time,
        end_time,
        output_rows,
        steady_state=steady_state,
        max_end_time=max_end_time,
    )

    return res


# KEEP
def process_species_samples(args: tuple) -> tuple:
    """
    Simulate knocked model with perturbations across multiple input combinations.

    This worker function processes a single knocked species by simulating the modified
    (knockout or knockin) model both with and without perturbations. It's designed to be called
    within a multiprocessing context.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements (in order):
        - knocked_species : str
            ID of the species that has been knocked out or knocked in
        - modified_model : libsbml.Model or str
            SBML model with the knockout or knockin applied
        - samples : array-like
            Sample values for input species perturbations
        - input_species_ids : list of str
            IDs of input species to perturb
        - selections : list of str
            Time course selections for the simulation
        - integrator : str
            Name of the integrator to use (e.g., 'cvode', 'gillespie')
        - start_time : float
            Start time for simulations
        - end_time : float
            End time for simulations
        - steady_state : bool
            Whether to simulate until steady state
        - max_end_time : float
            Maximum simulation time for steady state detection
        - min_ss_time : float
            Minimum steady state time observed so far
        - log_file : file
            File object for logging

    Returns
    -------
    tuple of (str, list of pandas.DataFrame)
        A tuple containing:
        - knocked_species : str
            ID of the knocked species
        - knocked_data : list of pandas.DataFrame
            List of DataFrames where:
            - First element: baseline knockout or knockin simulation (no perturbations)
            - Subsequent elements: knockout or knockin simulations with each perturbation combination

    Raises
    ------
    Exception
        If any error occurs during model loading, simulation, or data processing

    Notes
    -----
    - This function is intended to be used with multiprocessing.Pool.map()
    - The modified model is loaded fresh for each call to ensure thread safety
    - All DataFrames exclude the time column (columns start from index 1)
    - The function updates min_ss_time based on steady state detection

    See Also
    --------
    process_species_no_samples : Similar function without perturbations
    process_species_multiprocessing : Orchestrates parallel execution
    """
    try:
        (
            knocked_species,
            modified_model,
            samples,
            input_species_ids,
            selections,
            integrator,
            start_time,
            end_time,
            steady_state,
            max_end_time,
            min_ss_time,
            log_file,
        ) = args

        print_log(log_file, f"Starting processing: {knocked_species}")

        # Loading the modified model
        modified_rr = load_roadrunner_model(
            modified_model, integrator=integrator, log_file=log_file
        )

        # Setting the selections
        modified_rr.timeCourseSelections = selections

        # Simulating the not perturbed model
        modified_model_results, ss_time, colnames = simulate(
            modified_rr,
            start_time=start_time,
            end_time=end_time,
            steady_state=steady_state,
            max_end_time=max_end_time,
        )

        # Resetting the model
        modified_rr.reset()
        modified_rr.timeCourseSelections = selections
        min_ss_time = (
            ss_time if ss_time is not None and ss_time <= min_ss_time else min_ss_time
        )

        # Simulating the modified model with the perturbations
        combinations_knocked_model_results, colnames = simulate_combinations(
            modified_rr,
            create_combinations(samples),
            input_species_ids,
            min_ss_time,
            end_time,
            max_end_time,
            steady_state,
            log_file,
        )

        knocked_data = [
            pd.DataFrame(modified_model_results[:, 1:], columns=colnames[1:])
        ]

        for i in range(len(combinations_knocked_model_results)):
            ko_res_i = combinations_knocked_model_results[i]
            knocked_data.append(pd.DataFrame(ko_res_i[:, 1:], columns=colnames[1:]))

        return (knocked_species, knocked_data)
    except Exception as e:
        raise Exception(f"Error during processing species:\n Error: {e}")


# KEEP
def process_species_no_samples(args: tuple) -> tuple:
    """
    Simulate knocked model without perturbations.

    This worker function processes a single knocked species by simulating the modified
    (knockout or knockin) model without input perturbations. It's designed to be called within a
    multiprocessing context.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements (in order):
        - knocked_species : str
            ID of the species that has been knocked out or knocked in
        - modified_model : libsbml.Model or str
            SBML model with the knockout or knockin applied
        - combinations : array-like
            Not used in this function (for compatibility with parallel processing)
        - input_species_ids : list of str
            Not used in this function (for compatibility)
        - selections : list of str
            Time course selections for the simulation
        - integrator : str
            Name of the integrator to use (e.g., 'cvode', 'gillespie')
        - start_time : float
            Start time for simulation
        - end_time : float
            End time for simulation
        - steady_state : bool
            Whether to simulate until steady state
        - max_end_time : float
            Maximum simulation time for steady state detection
        - min_ss_time : float
            Minimum steady state time observed so far (updated but not returned)
        - log_file : file
            File object for logging

    Returns
    -------
    tuple of (str, pandas.DataFrame)
        A tuple containing:
        - knocked_species : str
            ID of the knocked out or knocked in species
        - knocked_data : pandas.DataFrame
            Simulation results excluding the time column

    Raises
    ------
    Exception
        If any error occurs during model loading, simulation, or data processing

    Notes
    -----
    - This function is intended to be used with multiprocessing.Pool.map()
    - The modified model is loaded fresh for each call to ensure thread safety
    - The DataFrame excludes the time column (columns start from index 1)
    - Unlike process_species_samples, this only runs a single baseline simulation

    See Also
    --------
    process_species_samples : Similar function with perturbations
    process_species_multiprocessing : Orchestrates parallel execution
    """
    try:
        (
            knocked_species,
            modified_model,
            combinations,
            input_species_ids,
            selections,
            integrator,
            start_time,
            end_time,
            steady_state,
            max_end_time,
            min_ss_time,
            log_file,
        ) = args

        print_log(log_file, f"Starting processing: {knocked_species}")

        modified_rr = load_roadrunner_model(
            modified_model, integrator=integrator, log_file=log_file
        )

        # __import__('pprint').pprint(modified_rr.relative_tolerance)

        modified_rr.selections = selections

        knocked_model_results, ss_time, colnames = simulate(
            modified_rr,
            start_time=0 if start_time is None else start_time,
            end_time=end_time,
            steady_state=steady_state,
            max_end_time=max_end_time,
        )

        min_ss_time = (
            ss_time if ss_time is not None and ss_time <= min_ss_time else min_ss_time
        )

        return (
            knocked_species,
            pd.DataFrame(knocked_model_results[:, 1:], columns=colnames[1:]),
        )
    except Exception as e:
        raise exceptions.ModelError(f"Error during processing species:\n Error: {e}")


# KEEP
def process_species_multiprocessing(
    target_ids: list,
    modified_models_dict: dict,
    samples: list,
    input_species_ids: list,
    selections: list,
    integrator: str,
    start_time: float = 0,
    end_time: float = 1000,
    steady_state: bool = False,
    max_end_time: float = 1000,
    min_ss_time: float = 1000,
    log_file=None,
    max_workers: int = None,
    use_perturbations: bool = False,
    preserve_input: bool = False,
) -> list:
    """
    Orchestrate parallel simulation of multiple knocked models using multiprocessing.

    This function distributes the simulation of knocked models across multiple CPU cores,
    automatically determining the optimal number of workers and preparing arguments for
    parallel execution. It supports both perturbation-based and standard knocked analyses.

    Parameters
    ----------
    target_ids : list of str
        List of species IDs to be knocked and analyzed
    modified_models_dict : dict
        Dictionary mapping species IDs to their corresponding knocked models
        Keys are species IDs, values are libsbml.Model or SBML strings
    samples : list or array-like
        Sample values for input species perturbations (used only if use_perturbations=True)
    input_species_ids : list of str
        IDs of input species to perturb
    selections : list of str
        Time course selections for simulations
    integrator : str
        Name of the integrator to use (e.g., 'cvode', 'gillespie')
    start_time : float, optional
        Start time for simulations, by default 0
    end_time : float, optional
        End time for simulations, by default 1000
    steady_state : bool, optional
        Whether to simulate until steady state is reached, by default False
    max_end_time : float, optional
        Maximum simulation time for steady state detection, by default 1000
    min_ss_time : float, optional
        Minimum steady state time threshold, by default 1000
    log_file : file, optional
        File object for logging, by default None
    max_workers : int, optional
        Maximum number of worker processes. If None, uses 40% of available CPU cores
        (capped at 8), by default None
    use_perturbations : bool, optional
        If True, use process_species_samples (with perturbations);
        if False, use process_species_no_samples, by default False
    preserve_input : bool, optional
        Currently unused parameter (reserved for future functionality), by default False

    Returns
    -------
    list of tuple
        List of tuples where each tuple contains:
        - If use_perturbations=True: (species_id, list of pandas.DataFrame)
        - If use_perturbations=False: (species_id, pandas.DataFrame)

    Raises
    ------
    SystemExit
        If a critical error occurs during multiprocessing pool execution

    Notes
    -----
    - Worker allocation: Uses 75% of available CPU cores by default, capped at 8 workers
    - Missing models: Species in target_ids without corresponding entries in
      modified_models_dict are skipped with a warning
    - Thread safety: Each worker loads models independently to avoid shared state issues
    - The function uses multiprocessing.Pool.map() for parallel execution

    Examples
    --------
    Without perturbations:
    >>> results = process_species_multiprocessing(
    ...     target_ids=['S1', 'S2', 'S3'],
    ...     modified_models_dict=ko_models,
    ...     samples=[],
    ...     input_species_ids=[],
    ...     selections=['time', '[S1]', '[S2]'],
    ...     integrator='cvode',
    ...     use_perturbations=False
    ... )

    With perturbations:
    >>> results = process_species_multiprocessing(
    ...     target_ids=['S1', 'S2'],
    ...     modified_models_dict=ko_models,
    ...     samples=perturbation_samples,
    ...     input_species_ids=['Input1', 'Input2'],
    ...     selections=['time', '[S1]', '[S2]'],
    ...     integrator='cvode',
    ...     use_perturbations=True,
    ...     max_workers=4
    ... )

    See Also
    --------
    process_species_samples : Worker function for simulations with perturbations
    process_species_no_samples : Worker function for simulations without perturbations
    """
    # Determine number of workers
    # Use 75% of the available cores
    n_core = int((mp.cpu_count() * 75) / 100)
    print_log(log_file, f" N core: {n_core}")
    if max_workers is None:
        max_workers = min(len(target_ids), n_core, 8)  # Don't use all CPUs

    print_log(
        log_file,
        f"Starting multiprocessing with {max_workers} workers for {len(target_ids)} species",
    )

    # Prepare the arguments
    process_args = []
    i = 0

    for ts in target_ids:
        if ts in modified_models_dict.keys():
            args = (
                ts,
                modified_models_dict[ts],
                samples,
                input_species_ids,
                selections,
                integrator,
                start_time,
                end_time,
                steady_state,
                max_end_time,
                min_ss_time,
                log_file,
            )

            process_args.append(args)
        else:
            print_log(log_file, f"Warning: No knockout model found for species {ts}")
        i += 1

    # Create and run the pool
    try:
        # Implement the logic using Pool and imap
        print_log(log_file, f" Starting pool")
        with Pool(max_workers) as pool:
            if use_perturbations:
                # If perturbations required
                operation = process_species_samples
            else:
                # If perturbations not required
                operation = process_species_no_samples

            print_log(log_file, f"[PROCESS] {operation}")
            # Running the workers
            result = pool.map(operation, process_args)
    except Exception as e:
        raise exceptions.SimulationError(
            f"Error during multiprocessing execution: {str(e)}"
        )

    return result


def get_knockout_variation(
    original_model, ko_models: list, colnames: list, log_file=None
) -> dict:
    """
    Calculate variation and relative variation of species with respect to internal species knockout.

    This version is used when perturbations are not required. It compares the final steady-state
    concentrations between the original model and knockout models to quantify the impact of
    removing each species from the system.

    Parameters
    ----------
    original_model : numpy.ndarray
        Structured array containing simulation results without knockout.
        Expected shape: (n_timepoints, n_species+1) where first column is time.
    ko_models : list of tuple
        List of tuples where each tuple contains:
        - knockedout_species : str
            ID of the species that was knocked out
        - simulation_results : numpy.ndarray
            Simulation results after the species knockout
    colnames : list of str
        Column names from the simulation output, typically in format ['time', '[species1]', '[species2]', ...]
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            'knocked_out_species_id': {
                'affected_species_id': {
                    'variation': float,
                    'relative-variation': float
                }
            }
        }
        - variation: Absolute difference (ko_value - original_value)
        - relative-variation: Relative difference ((ko_value - original_value) / original_value)

        Self-comparisons (knockout species vs itself) have NaN values.

    Notes
    -----
    - Uses epsilon = 1e-20 as threshold for numerical stability
    - Values below epsilon are treated as zero to avoid division errors
    - When original value < epsilon, relative variation is set to inf
    - Only the final time point concentrations are compared
    - Column names are cleaned by removing brackets using regex pattern r"[\[\]]"

    Examples
    --------
    >>> original_sim = simulate(model)
    >>> ko_sims = [(species_id, simulate(ko_model)) for species_id, ko_model in knockouts]
    >>> variations = get_knockout_variation(original_sim, ko_sims, original_sim.colnames)
    >>> print(variations['S1']['S2']['relative-variation'])
    0.25  # S2 increased by 25% when S1 was knocked out

    See Also
    --------
    get_absolute_variations_no_samples : Calculate absolute variations without perturbations
    get_relative_variations_no_samples : Calculate relative variations without perturbations
    """
    variations_dict = {}
    species_idxs = {}
    exp = r"[\[\]]"

    epsilon = 1e-20  # Variable used as values cutoff and to avoid division by 0

    # taking the indices
    for cn in colnames:
        if cn == "time":  # Skipping the time column
            continue
        id = re.sub(exp, "", cn)
        species_idxs[id] = colnames.index(cn)

    for i in range(len(ko_models)):
        # Getting the ko species with the result of the simulation
        ko_species, ko_spec_result = ko_models[i]

        # Init variations dictionary
        variations_dict[ko_species] = {}

        for species in species_idxs.keys():
            if species == ko_species:  # Skipping the species if compared with itself
                variation = np.nan
                relative_variation = np.nan
            else:
                # Getting the last value from the original model
                original_value = original_model[-1, species_idxs[species]]

                # Check if the original value is too small (smaller than epsilon) and setting it to 0
                # example: original_value = 1e-30 < 1e-20 ==> original_value = 0
                original_val = np.where(
                    np.abs(np.float64(original_value)) < epsilon,
                    0,
                    np.float64(original_value),
                )

                # Getting the last value from the knockout results
                ko_last_value = np.float64(ko_spec_result[-1, species_idxs[species]])

                # Check if the ko value is too small (smaller than epsilon) and setting it to 0
                ko_val = np.where(
                    np.abs(np.float64(ko_last_value)) < epsilon,
                    0,
                    np.float64(ko_last_value),
                )

                # Getting the variation
                variation = ko_val - original_val

                # Getting the relative variation
                # Check if dividing by 0, if yes using epsilon instead
                if original_val > epsilon:
                    relative_variation = (ko_val - original_val) / original_val
                else:
                    relative_variation = np.inf

                __import__("pprint").pprint(relative_variation)

                # Load the data in the dictionary
                variations_dict[ko_species][species] = {
                    "variation": variation,
                    "relative-variation": relative_variation,
                }

    return variations_dict


# KEEP
def get_absolute_variations_samples(
    final_results_original_model: list[pd.DataFrame],
    final_results_knocked_model: list[pd.DataFrame],
    epsilon: float = 1e-20,
    log_file=None,
) -> pd.DataFrame:
    """
    Calculate absolute variations between original and knocked model results with perturbations.

    This function computes the root mean square (RMS) of absolute concentration differences
    across multiple perturbation combinations. It compares the final steady-state values
    between original and knocked models for each species across all input perturbations.

    Parameters
    ----------
    final_results_original_model : list of pandas.DataFrame
        List of DataFrames containing original model simulation results for each perturbation.
        Each DataFrame should have species concentrations as columns.
        Length must match the number of perturbation combinations.
    final_results_knocked_model : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked out species
        - ko_data : list of pandas.DataFrame
            List of DataFrames with knockout simulation results for each perturbation
    epsilon : float, optional
        Threshold below which values are considered zero to avoid numerical issues,
        by default 1e-20
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_species) where:
        - Rows: Knocked species IDs
        - Columns: All species IDs (with brackets removed)
        - Values: RMS of absolute variations across all perturbations

        NaN values appear where:
        - Knocked species is compared to itself (diagonal)
        - Infinite values are encountered

    Notes
    -----
    The function calculates variations as:
    1. For each perturbation: var = knocked_value - original_value
    2. Aggregate using RMS: RMS = sqrt(mean(var²))

    Values ≤ epsilon are masked to zero before calculating differences.
    Self-comparisons (knocked species vs itself) are set to NaN to avoid meaningless data.

    The RMS aggregation emphasizes larger variations and provides a single metric
    that summarizes the impact across all perturbations.

    Examples
    --------
    >>> original_results = [df1, df2, df3]  # 3 perturbations
    >>> ko_results = [('S1', [ko_df1, ko_df2, ko_df3]), ('S2', [...])]
    >>> variations = get_absolute_variations_samples(original_results, ko_results)
    >>> print(variations.loc['S1', 'S2'])  # RMS variation of S2 when S1 is knocked out
    0.145

    See Also
    --------
    get_relative_variations_samples : Calculate relative variations with perturbations
    get_absolute_variations_no_samples : Calculate absolute variations without perturbations
    """
    rms = []
    for knocked_species, knocked_data in final_results_knocked_model:
        variations = []

        for c in range(len(knocked_data)):
            knocked_sim_i = knocked_data[c]
            original_sim_i = final_results_original_model[c]

            # Getting the last values
            last_knocked_values = knocked_sim_i.tail(1)
            last_original_values = original_sim_i.tail(1)

            # Masking
            last_knocked_values = last_knocked_values.mask(
                last_knocked_values <= epsilon, 0
            )
            last_original_values = last_original_values.mask(
                last_original_values <= epsilon, 0
            )

            var = last_knocked_values - last_original_values
            variations.append(var)

        variations_df = pd.concat(variations, ignore_index=True)
        variations_rms = np.sqrt((variations_df**2).mean())
        variations_rms.name = knocked_species
        rms.append(variations_rms)

    res_df = pd.DataFrame(rms)

    # Applying the mas for inf values
    res_df = res_df.mask(np.isinf(res_df), np.nan)
    col_names = res_df.columns.str.strip("[]")
    rows = res_df.index.get_indexer(col_names)
    valid = rows != -1
    res_df.values[rows[valid], np.arange(len(res_df.columns))[valid]] = np.nan

    return res_df


# KEEP
def get_absolute_variations_no_samples(
    original_data: pd.DataFrame,
    knocked_data: pd.DataFrame,
    epsilon: float = 1e-20,
    log_file=None,
) -> pd.DataFrame:
    """
    Calculate absolute variations between original and knocked model results without perturbations.

    This function computes the absolute concentration differences at steady state
    between the original model and each knocked model. Unlike the samples version,
    this operates on single simulation runs without input perturbations.

    Parameters
    ----------
    original_data : pandas.DataFrame
        DataFrame containing the original model simulation results.
        Expected to have species concentrations as columns.
        Typically excludes the time column.
    knocked_data : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - ko_info : pandas.DataFrame
            DataFrame with knocked simulation results for that species
    epsilon : float, optional
        Threshold below which values are considered zero to avoid numerical issues,
        by default 1e-20
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_species) where:
        - Rows: Knocked species IDs
        - Columns: All species IDs (with brackets removed)
        - Values: Absolute variation (knocked_value - original_value) at final time point

        NaN values appear where:
        - Knocked species is compared to itself (diagonal)
        - Infinite values are encountered

    Notes
    -----
    The function calculates variations as:
    - var = knocked_final_value - original_final_value

    Processing steps:
    1. Extract final time point values using .tail(1)
    2. Mask values ≤ epsilon to zero
    3. Calculate absolute difference
    4. Set diagonal (self-comparisons) to NaN

    This is the non-perturbation counterpart to get_absolute_variations_samples.

    Examples
    --------
    >>> original_df = pd.DataFrame({'S1': [1.0], 'S2': [2.0], 'S3': [0.5]})
    >>> ko_data = [
    ...     ('S1', pd.DataFrame({'S1': [0.0], 'S2': [2.3], 'S3': [0.6]})),
    ...     ('S2', pd.DataFrame({'S1': [1.1], 'S2': [0.0], 'S3': [0.4]}))
    ... ]
    >>> variations = get_absolute_variations_no_samples(original_df, ko_data)
    >>> print(variations.loc['S1', 'S2'])  # Change in S2 when S1 is knocked out
    0.3

    See Also
    --------
    get_absolute_variations_samples : Calculate absolute variations with perturbations
    get_relative_variations_no_samples : Calculate relative variations without perturbations
    """
    variations = []

    for knocked_species, knocked_info in knocked_data:
        last_knocked_values = knocked_info.tail(1)
        last_original_values = original_data.tail(1)

        # Masking
        last_knocked_values = last_knocked_values.mask(
            last_knocked_values <= epsilon, 0
        )
        last_original_values = last_original_values.mask(
            last_original_values <= epsilon, 0
        )

        var = last_knocked_values - last_original_values
        # rms_vars = np.sqrt(var**2)
        var_series = var.squeeze()
        var_series.name = knocked_species
        variations.append(var_series)

    res_df = pd.DataFrame(variations)

    # Applying the mas for inf values
    res_df = res_df.mask(np.isinf(res_df), np.nan)
    col_names = res_df.columns.str.strip("[]")
    rows = res_df.index.get_indexer(col_names)
    valid = rows != -1
    res_df.values[rows[valid], np.arange(len(res_df.columns))[valid]] = np.nan

    return res_df


# KEEP
def get_relative_variations_samples(
    final_results_original_model: list[pd.DataFrame],
    final_results_knocked_model: list[pd.DataFrame],
    epsilon: float = 1e-20,
    log_file=None,
) -> pd.DataFrame:
    """
    Calculate relative variations between original and knocked model results with perturbations.

    This function computes the root mean square (RMS) of relative concentration changes
    across multiple perturbation combinations. It compares the final steady-state values
    between original and knocked models for each species across all input perturbations,
    normalized by the original values.

    Parameters
    ----------
    final_results_original_model : list of pandas.DataFrame
        List of DataFrames containing original model simulation results for each perturbation.
        Each DataFrame should have species concentrations as columns.
        Length must match the number of perturbation combinations.
    final_results_knocked_model : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - ko_data : list of pandas.DataFrame
            List of DataFrames with knocked simulation results for each perturbation
    epsilon : float, optional
        Threshold below which values are considered zero to avoid numerical issues,
        by default 1e-20
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_species) where:
        - Rows: Knockout species IDs
        - Columns: All species IDs (with brackets removed)
        - Values: RMS of relative variations across all perturbations

        NaN values appear where:
        - knocked species is compared to itself (diagonal)
        - Infinite values are encountered (e.g., division by zero)

    Notes
    -----
    The function calculates relative variations as:
    1. For each perturbation: rel_var = (knocked - original_value) / original_value
    2. Aggregate using RMS: RMS = sqrt(mean(rel_var²))

    Processing steps:
    - Values ≤ epsilon are masked to zero before calculations
    - Relative changes are computed for each perturbation
    - RMS aggregation provides a single metric across all perturbations
    - Self-comparisons (knockout species vs itself) are set to NaN
    - Infinite values (from division by zero) are masked to NaN

    The RMS aggregation emphasizes larger relative variations and provides a robust
    measure that is less sensitive to outliers than simple averaging.

    Examples
    --------
    >>> original_results = [df1, df2, df3]  # 3 perturbations
    >>> ko_results = [('S1', [ko_df1, ko_df2, ko_df3]), ('S2', [...])]
    >>> rel_variations = get_relative_variations_samples(original_results, ko_results)
    >>> print(rel_variations.loc['S1', 'S2'])  # RMS relative change in S2 when S1 is knocked out
    0.125  # S2 changed by ~12.5% on average (RMS) across perturbations

    See Also
    --------
    get_absolute_variations_samples : Calculate absolute variations with perturbations
    get_relative_variations_no_samples : Calculate relative variations without perturbations
    """
    rms = []
    for knocked_species, knocked_data in final_results_knocked_model:
        variations = []

        for c in range(len(knocked_data)):
            knocked_sim_i = knocked_data[c]
            original_sim_i = final_results_original_model[c]

            # Getting the last values
            last_knocked_values = knocked_sim_i.tail(1)
            last_original_values = original_sim_i.tail(1)

            # Masking
            last_knocked_values = last_knocked_values.mask(
                last_knocked_values <= epsilon, 0
            )
            last_original_values = last_original_values.mask(
                last_original_values <= epsilon, 0
            )

            var = (last_knocked_values - last_original_values) / last_original_values
            variations.append(var)

        variations_df = pd.concat(variations, ignore_index=True)
        variations_rms = np.sqrt((variations_df**2).mean())
        variations_rms.name = knocked_species
        rms.append(variations_rms)

    res_df = pd.DataFrame(rms)

    # Applying the mas for inf values
    res_df = res_df.mask(np.isinf(res_df), np.nan)
    col_names = res_df.columns.str.strip("[]")
    rows = res_df.index.get_indexer(col_names)
    valid = rows != -1
    res_df.values[rows[valid], np.arange(len(res_df.columns))[valid]] = np.nan

    return res_df


# KEEP
def get_relative_variations_no_samples(
    original_data: pd.DataFrame,
    knocked_data: pd.DataFrame,
    epsilon: float = 1e-20,
    log_file=None,
) -> pd.DataFrame:
    """
    Calculate relative variations between original and knocked model results without perturbations.

    This function computes the relative concentration changes at steady state between the
    original model and each knocked model. Unlike the samples version, this operates on
    single simulation runs without input perturbations, normalizing changes by the original
    concentrations.

    Parameters
    ----------
    original_data : pandas.DataFrame
        DataFrame containing the original model simulation results.
        Expected to have species concentrations as columns.
        Typically excludes the time column.
    knocked_data : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - ko_info : pandas.DataFrame
            DataFrame with knocked simulation results for that species
    epsilon : float, optional
        Threshold below which values are considered zero to avoid numerical issues,
        by default 1e-20
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_species) where:
        - Rows: Knockout species IDs
        - Columns: All species IDs (with brackets removed)
        - Values: Relative variation ((knocked_value - original_value) / original_value) at final time point

        NaN values appear where:
        - Knocked species is compared to itself (diagonal)
        - Infinite values are encountered (e.g., division by zero)

    Notes
    -----
    The function calculates relative variations as:
    - rel_var = (knocked_value - original_final_value) / original_final_value

    Processing steps:
    1. Extract final time point values using .tail(1)
    2. Mask values ≤ epsilon to zero
    3. Calculate relative difference (normalized by original value)
    4. Set diagonal (self-comparisons) to NaN
    5. Mask infinite values (from division by zero) to NaN

    Relative variations provide scale-independent measures of knockout impact,
    making them suitable for comparing effects across species with different
    concentration magnitudes.

    This is the non-perturbation counterpart to get_relative_variations_samples.

    Examples
    --------
    >>> original_df = pd.DataFrame({'S1': [1.0], 'S2': [2.0], 'S3': [0.5]})
    >>> ko_data = [
    ...     ('S1', pd.DataFrame({'S1': [0.0], 'S2': [2.4], 'S3': [0.6]})),
    ...     ('S2', pd.DataFrame({'S1': [1.1], 'S2': [0.0], 'S3': [0.4]}))
    ... ]
    >>> rel_variations = get_relative_variations_no_samples(original_df, ko_data)
    >>> print(rel_variations.loc['S1', 'S2'])  # Relative change in S2 when S1 is knocked out
    0.2  # S2 increased by 20%

    See Also
    --------
    get_relative_variations_samples : Calculate relative variations with perturbations
    get_absolute_variations_no_samples : Calculate absolute variations without perturbations
    """
    variations = []

    for knocked_species, knocked_info in knocked_data:
        last_knocked_values = knocked_info.tail(1)
        last_original_values = original_data.tail(1)

        # Masking
        last_knocked_values = last_knocked_values.mask(
            last_knocked_values <= epsilon, 0
        )
        last_original_values = last_original_values.mask(
            last_original_values <= epsilon, 0
        )

        var = (last_knocked_values - last_original_values) / last_original_values

        # rms_vars = np.sqrt(var**2)
        var_series = var.squeeze()
        var_series.name = knocked_species
        variations.append(var_series)

    res_df = pd.DataFrame(variations)

    # Applying the mas for inf values
    res_df = res_df.mask(np.isinf(res_df), np.nan)
    col_names = res_df.columns.str.strip("[]")
    rows = res_df.index.get_indexer(col_names)
    valid = rows != -1
    res_df.values[rows[valid], np.arange(len(res_df.columns))[valid]] = np.nan

    return res_df

"""
# TODO: Create other metrics for the variation
A quantitative score (log-ratio or sensitivity coefficient) for species that are present in both conditions
A qualitative score or binary flag for species that switch between zero and non-zero
"""
def get_relative_variations_log_ratio(
    final_results_original_model: list[pd.DataFrame],
    final_results_knocked_model: list[tuple[str, list[pd.DataFrame]]],
    epsilon: float = 1e-20,
    log_file=None,
    aggregation: str = "median",
    log_base: str = "2",
    return_signed: bool = False,
) -> pd.DataFrame:
    """
    Calculate relative variations between original and knocked model results
    using log-ratios, as an alternative to get_relative_variations_samples.

    For each perturbation i:
        d_i = log2( c_knocked_i / c_original_i )

    Aggregated across perturbations via the chosen method.

    Parameters
    ----------
    final_results_original_model : list of pandas.DataFrame
        List of DataFrames containing original model simulation results for each perturbation.
        Each DataFrame should have species concentrations as columns.
        Length must match the number of perturbation combinations.
    final_results_knocked_model : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - ko_data : list of pandas.DataFrame
            List of DataFrames with knocked simulation results for each perturbation
    epsilon : float, optional
        Clip floor to avoid log(0), by default 1e-20.
        Both knocked and original values are clipped to epsilon before the ratio,
        so near-zero / near-zero yields log-ratio ≈ 0 (no spurious inf values).
    log_file : file, optional
        File object for logging operations (currently unused), by default None.
    aggregation : str, optional
        How to aggregate log-ratios across perturbations, by default "median":
        - "median" : median(|d_i|) — robust to outlier perturbations, recommended
        - "mean"   : mean(|d_i|)  — L1, interpretable
        - "rms"    : sqrt(mean(d_i²)) — L2, emphasizes large deviations
        - "max"    : max(|d_i|)   — worst-case deviation across perturbations
    log_base : str, optional
        Logarithm base, by default "2":
        - "2"  : scores in doublings — |score| = 1.0 means exactly twofold change
        - "e"  : scores in nats
        - "10" : scores in decades
    return_signed : bool, optional
        If True, returns signed values (median or mean without absolute value),
        by default False.
        Positive = knockout increases species concentration.
        Negative = knockout decreases species concentration.
        Only valid with aggregation="median" or "mean".

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_species) where:
        - Rows: Knockout species IDs
        - Columns: All species IDs (with brackets removed)
        - Values: Aggregated log-ratio scores across all perturbations

        NaN values appear where:
        - knocked species is compared to itself (diagonal)
        - Infinite values are encountered

    Notes
    -----
    Advantages over get_relative_variations_samples (RMS of arithmetic ratios):
    - Symmetric: log(2x) = -log(0.5x), unlike (knocked - orig) / orig
    - No spurious inf: clipping avoids division by near-zero values
    - Natural scale for concentrations spanning orders of magnitude
    - Median aggregation suppresses influence of outlier perturbations

    Interpretation guide (absolute values; equivalent fold-change anchors):
        Base 2 (log_base="2"):
            |score| < 0.15 : negligible  (< ~10% change)
            |score| ~ 0.5  : moderate    (~41% change)
            |score| ~ 1.0  : strong      (twofold change)
            |score| > 2.0  : very strong (fourfold or more)

        Base e (log_base="e"):
            |score| < 0.095 : negligible  (< ~10% change)
            |score| ~ 0.347 : moderate    (~41% change)
            |score| ~ 0.693 : strong      (twofold change)
            |score| > 1.386 : very strong (fourfold or more)

        Base 10 (log_base="10"):
            |score| < 0.041 : negligible  (< ~10% change)
            |score| ~ 0.151 : moderate    (~41% change)
            |score| ~ 0.301 : strong      (twofold change)
            |score| > 0.602 : very strong (fourfold or more)

    Examples
    --------
    >>> original_results = [df1, df2, df3]  # 3 perturbations
    >>> ko_results = [('S1', [ko_df1, ko_df2, ko_df3]), ('S2', [...])]
    >>> rel_variations = get_relative_variations_log_ratio(original_results, ko_results)
    >>> print(rel_variations.loc['S1', 'S2'])
    1.0  # S2 changed by approximately twofold across perturbations

    See Also
    --------
    get_relative_variations_samples : Original RMS-based relative variations
    get_relative_variations_ks : KS-test based distributional variations
    get_absolute_variations_samples : Calculate absolute variations with perturbations
    """
    log_fns = {"2": np.log2, "e": np.log, "10": np.log10}
    if log_base not in log_fns:
        raise ValueError(f"log_base must be one of {list(log_fns)}. Got: '{log_base}'")
    if aggregation not in {"median", "mean", "rms", "max"}:
        raise ValueError(
            f"aggregation must be one of 'median', 'mean', 'rms', 'max'. Got: '{aggregation}'"
        )
    if return_signed and aggregation not in {"median", "mean"}:
        raise ValueError(
            "return_signed=True is only valid with aggregation='median' or 'mean'."
        )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0. Got: {epsilon}")
    if len(final_results_original_model) == 0:
        raise ValueError(
            "final_results_original_model must contain at least one DataFrame."
        )

    log_fn = log_fns[log_base]
    rows = []
    expected_n_perturbations = len(final_results_original_model)

    for knocked_species, knocked_data in final_results_knocked_model:
        if len(knocked_data) != expected_n_perturbations:
            raise ValueError(
                f"Length mismatch for knocked species '{knocked_species}': "
                f"expected {expected_n_perturbations} simulations, got {len(knocked_data)}."
            )

        log_ratios = []

        # Process each perturbation for the current knocked species
        for c, (ko_df, orig_df) in enumerate(
            zip(knocked_data, final_results_original_model)
        ):
            # Retrieve the last row
            ss_ko = ko_df.tail(1)
            ss_orig = orig_df.tail(1)

            # Ensure columns match and are in the same order
            if not ss_ko.columns.equals(ss_orig.columns):
                missing_in_ko = list(ss_orig.columns.difference(ss_ko.columns))
                missing_in_orig = list(ss_ko.columns.difference(ss_orig.columns))
                if missing_in_ko or missing_in_orig:
                    raise ValueError(
                        f"Column mismatch at perturbation index {c} for knocked species '{knocked_species}'. "
                        f"Missing in knocked: {missing_in_ko}. Missing in original: {missing_in_orig}."
                    )
            # Clip values to epsilon to avoid log(0) issues
            ss_ko = ss_ko.loc[:, ss_orig.columns].clip(lower=epsilon)
            ss_orig = ss_orig.loc[:, ss_orig.columns].clip(lower=epsilon)

            # Compute log-ratio for this perturbation
            lr = log_fn(ss_ko.to_numpy() / ss_orig.to_numpy())
            log_ratios.append(pd.DataFrame(lr, columns=ss_orig.columns))

        # Concatenate log-ratios across perturbations for this knocked species
        lr_df = pd.concat(log_ratios, ignore_index=True)  # (n_perturbations, n_species)

        if return_signed:
            score = lr_df.median() if aggregation == "median" else lr_df.mean()
        else:
            abs_lr = lr_df.abs()
            if aggregation == "median":
                score = abs_lr.median()
            elif aggregation == "mean":
                score = abs_lr.mean()
            elif aggregation == "rms":
                score = np.sqrt((lr_df**2).mean())
            elif aggregation == "max":
                score = abs_lr.max()
            else:
                # Fallback to median by default
                score = abs_lr.median()

        print_log(log_file, f"[DEBUG] score type: {type(score)} | score: {score}")
        score.name = knocked_species
        rows.append(score)

    res_df = pd.DataFrame(rows)
    res_df = res_df.mask(np.isinf(res_df), np.nan)

    col_names = pd.Index(res_df.columns).astype(str).str.strip("[]")
    idx = res_df.index.get_indexer(col_names)
    valid = idx != -1
    res_df.values[idx[valid], np.arange(len(res_df.columns))[valid]] = np.nan

    return res_df


def get_relative_variations_log_ratio_no_samples(
    original_data: pd.DataFrame,
    knocked_data: list[tuple[str, pd.DataFrame]],
    epsilon: float = 1e-20,
    return_signed: bool = False,
    log_base: str = "2",
    log_file=None,
) -> pd.DataFrame:
    """
    Calculate relative variations between original and knocked model results using log-ratios without perturbations.

    This function computes the log-ratio of final concentrations between the original model and each knocked model
    at steady state, without input perturbations. It serves as a non-perturbation counterpart to get_relative_variations_log_ratio.

    Parameters
    ----------
    original_data : pandas.DataFrame
        DataFrame containing the original model simulation results.
        Expected to have species concentrations as columns.
        Typically excludes the time column.
    knocked_data : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - ko_info : pandas.DataFrame
            DataFrame with knocked simulation results for that species
    epsilon : float, optional
        Clip floor to avoid log(0), by default 1e-20.
        Both knocked and original values are clipped to epsilon before the ratio,
        so near-zero / near-zero yields log-ratio ≈ 0 (no spurious inf values).
    return_signed : bool, optional
        If True, returns signed log-ratios.
        If False (default), returns absolute log-ratios.
        Positive = knockout increases species concentration.
        Negative = knockout decreases species concentration.
    log_file : file, optional
        File object for logging operations (currently unused), by default None.
    log_base : str, optional
        Logarithm base, by default "2":
        - "2"  : scores in doublings — |score| = 1.0 means exactly twofold change
        - "e"  : scores in nats
        - "10" : scores in decades

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_species) where:
        - Rows: Knockout species IDs
        - Columns: All species IDs (with brackets removed)
                - Values: Log-ratio scores at steady state
                    (signed if return_signed=True, absolute otherwise)

        NaN values appear where:
        - Knocked species is compared to itself (diagonal)
        - Infinite values are encountered
    Notes
    The function calculates log-ratio variations as:
    - log_ratio = log(knocked_final_value / original_final_value)
    Processing steps:
    1. Extract final time point values using .tail(1)
    2. Clip values ≤ epsilon to avoid log(0) issues
    3. Calculate log-ratio (knocked / original)
    4. Set diagonal (self-comparisons) to NaN
    5. Mask infinite values (from division by zero) to NaN
    Examples
    >>> original_df = pd.DataFrame({'S1': [1.0], 'S2': [2.0], 'S3': [0.5]})
    >>> ko_data = [
    ...     ('S1', pd.DataFrame({'S1': [0.0], 'S2': [2.4], 'S3': [0.6]})),
    ...     ('S2', pd.DataFrame({'S1': [1.1], 'S2': [0.0], 'S3': [0.4]}))
    ... ]
    >>> rel_variations = get_relative_variations_log_ratio_no_samples(original_df, ko_data)
    >>> print(rel_variations.loc['S1', 'S2'])
    1.263  # S2 changed by approximately 2.4/2.0 = 1.2-fold, log2(1.2) ≈ 1.263
    See Also
    get_relative_variations_log_ratio : Log-ratio relative variations with perturbations
    get_relative_variations_no_samples : Relative variations without perturbations (arithmetic ratio)
    get_absolute_variations_no_samples : Absolute variations without perturbations
    """
    log_fns = {"2": np.log2, "e": np.log, "10": np.log10}
    if log_base not in log_fns:
        raise ValueError(f"log_base must be one of {list(log_fns)}. Got: '{log_base}'")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0. Got: {epsilon}")

    log_fn = log_fns[log_base]
    rows = []

    for knocked_species, knocked_info in knocked_data:
        ss_ko = knocked_info.tail(1)
        ss_orig = original_data.tail(1)

        if not ss_ko.columns.equals(ss_orig.columns):
            missing_in_ko = list(ss_orig.columns.difference(ss_ko.columns))
            missing_in_orig = list(ss_ko.columns.difference(ss_orig.columns))
            if missing_in_ko or missing_in_orig:
                raise ValueError(
                    f"Column mismatch for knocked species '{knocked_species}'. "
                    f"Missing in knocked: {missing_in_ko}. Missing in original: {missing_in_orig}."
                )

        ss_ko = ss_ko.loc[:, ss_orig.columns].clip(lower=epsilon)
        ss_orig = ss_orig.loc[:, ss_orig.columns].clip(lower=epsilon)

        lr = log_fn(ss_ko.to_numpy() / ss_orig.to_numpy())
        if not return_signed:
            lr = np.abs(lr)
        lr_series = pd.Series(lr.flatten(), index=ss_orig.columns, name=knocked_species)
        rows.append(lr_series)

    res_df = pd.DataFrame(rows)
    res_df = res_df.mask(np.isinf(res_df), np.nan)

    col_names = pd.Index(res_df.columns).astype(str).str.strip("[]")
    idx = res_df.index.get_indexer(col_names)
    valid = idx != -1
    res_df.values[idx[valid], np.arange(len(res_df.columns))[valid]] = np.nan

    return res_df


# KEEP
def get_payoff_vals(
    final_results_original_model: pd.DataFrame,
    final_results_knocked_model: pd.DataFrame,
    payoff_function: callable,
    epsilon: float = 1e-20,
    log_file=None,
) -> list:
    """
    Calculate payoff differences between original and knocked model results using a custom payoff function.

    This function applies a user-defined payoff function to both original and knocked simulation
    results across multiple perturbation combinations, then computes the difference. This is
    typically used for Shapley value calculations or game-theoretic analyses of knock impacts.

    Parameters
    ----------
    final_results_original_model : list of pandas.DataFrame
        List of DataFrames containing original model simulation results for each perturbation.
        Each DataFrame should have species concentrations as columns and time points as rows.
        Length must match the number of perturbation combinations.
    final_results_knocked_model : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - ko_data : list of pandas.DataFrame
            List of DataFrames with knocked simulation results for each perturbation
    payoff_function : callable
        Function that computes a payoff value from a simulation DataFrame.
        Should accept a pandas.DataFrame and return a numeric value or Series.
        Example: lambda df: df.iloc[-1].sum() for final total concentration
    epsilon : float, optional
        Threshold for numerical stability (currently unused in this function),
        by default 1e-20
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Returns
    -------
    list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - payoffs_df : pandas.DataFrame
            DataFrame of payoff differences (original - knocked) for all perturbations
            Shape: (n_perturbations, n_output_values) depending on payoff_function

    Notes
    -----
    The payoff difference is calculated as:
    - payoff_diff = payoff_function(original_simulation) - payoff_function(knocked_simulation)

    Positive values indicate that the original model had higher payoff (knocked decreased performance).
    Negative values indicate that the knockout model had higher payoff (knocked improved performance).

    This function is commonly used as input to Shapley value calculations, where the payoff
    represents a measure of system performance or output.

    Examples
    --------
    Using sum of final concentrations as payoff:
    >>> def total_concentration(df):
    ...     return df.iloc[-1].sum()
    >>>
    >>> original_results = [df1, df2, df3]
    >>> ko_results = [('S1', [ko_df1, ko_df2, ko_df3]), ('S2', [...])]
    >>> payoffs = get_payoff_vals(original_results, ko_results, total_concentration)
    >>> for ko_species, payoff_df in payoffs:
    ...     print(f"{ko_species}: mean payoff difference = {payoff_df.mean()}")

    Using specific species as payoff:
    >>> def species_S3_payoff(df):
    ...     return df['S3'].iloc[-1]
    >>>
    >>> payoffs = get_payoff_vals(original_results, ko_results, species_S3_payoff)

    See Also
    --------
    get_shapley_values : Compute Shapley values from payoff differences
    """
    res = []

    knocked_species_list = []

    for knocked_species, _ in final_results_knocked_model:
        knocked_species_list.append(knocked_species)

    for knocked_species, knocked_data in final_results_knocked_model:
        payoffs = []

        for c in range(len(knocked_data)):
            knocked_sim_i = knocked_data[c]
            original_sim_i = final_results_original_model[c]

            payoff_original = payoff_function(original_sim_i)
            payoff_knocked = payoff_function(knocked_sim_i)

            payoff_diff = payoff_original - payoff_knocked

            payoffs.append(payoff_diff)

        # Convert payoffs in an unique DataFrame
        payoffs_df = pd.concat(payoffs, ignore_index=True)
        # Append to the results

        res.append((knocked_species, payoffs_df))
    # Return a list of tuple (ko_species, payoffs)
    return res


# KEEP
def get_shapley_values(
    payoff_values: list, n_combinations: int, n_inputs: int, log_file=None
) -> pd.DataFrame:
    """
    Calculate Shapley values from payoff differences across knocked species.

    This function computes Shapley values for each species and output metric by aggregating
    payoff differences across all input perturbation combinations. Shapley values quantify
    the marginal contribution of each knocked species to the overall system behavior.

    Parameters
    ----------
    payoff_values : list of tuple
        List of tuples where each tuple contains:
        - ko_species : str
            ID of the knocked species
        - payoffs_df : pandas.DataFrame
            DataFrame of payoff differences for all perturbations
            Shape: (n_combinations, n_output_metrics)
    n_combinations : int
        Total number of input perturbation combinations used in the analysis
    n_inputs : int
        Number of input species that were perturbed
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape (n_knocked_species, n_output_metrics) where:
        - Rows: Knocked species IDs
        - Columns: Output metric names (depends on payoff_function used)
        - Values: Shapley values representing the contribution of each knockout

    Notes
    -----
    The Shapley value formula used is:

    $$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

    Simplified for this implementation:

    $$\text{Shapley}_{\text{ko}} = \frac{n_{\text{inputs}}! \cdot (n_{\text{combinations}} - n_{\text{inputs}})!}{n_{\text{combinations}}!} \sum_{\text{all combinations}} \text{payoff}_{\text{ko}}$$

    Where:
    - The left factor normalizes the contribution across all possible coalitions
    - The sum aggregates payoff differences across all perturbation combinations
    - Higher positive Shapley values indicate species whose knockout has larger impact

    The Shapley value provides a fair allocation of the total payoff change to each
    knocked-out species, accounting for synergistic and antagonistic interactions.

    Examples
    --------
    Calculate Shapley values after computing payoffs:
    >>> payoffs = get_payoff_vals(original_results, ko_results, payoff_func)
    >>> shapley_df = get_shapley_values(
    ...     payoff_values=payoffs,
    ...     n_combinations=100,
    ...     n_inputs=3
    ... )
    >>> print(shapley_df.loc['S1'])  # Shapley values for knocking out S1
    S2    0.034
    S3    0.021
    S4   -0.012
    Name: S1, dtype: float64

    Interpreting results:
    >>> # Positive Shapley: knockout decreased system performance
    >>> # Negative Shapley: knockout improved system performance
    >>> top_impact = shapley_df.abs().max(axis=1).sort_values(ascending=False)
    >>> print(f"Most impactful knockout: {top_impact.index[0]}")

    See Also
    --------
    get_payoff_vals : Calculate payoff differences for Shapley analysis
    """
    shap_vals = []
    left_factor = (
        factorial(n_inputs) * factorial(n_combinations - n_inputs)
    ) / factorial(n_combinations)

    print_log(
        log_file,
        f"[DEBUG] n_inputs: {n_inputs} | n_combinations: {n_combinations} | lf: {left_factor}",
    )

    # __import__("pprint").pprint(left_factor)

    for knocked_species, payoffs in payoff_values:
        factors = left_factor * payoffs

        sums = factors.sum()
        sums.name = knocked_species
        shap_vals.append(sums)

    shap_df = pd.DataFrame(shap_vals)

    # Set self-comparison entries (knocked species vs itself) to NaN
    col_names = pd.Index(shap_df.columns).astype(str).str.strip("[]")
    idx = shap_df.index.get_indexer(col_names)
    valid = idx != -1
    shap_df.values[idx[valid], np.arange(len(shap_df.columns))[valid]] = np.nan

    return shap_df


def get_no_samples_variations(
    variations_dict,
    all_species,
    ko_species_list,
    variation_type="relative",
    log_file=None,
):
    """
    Create the heatmap with the variations if perturbations are not required

    args:
        - variations_dict: Dictionary containing the variations informations
        - all_species: Set of all the model's species
        - ko_species_list: List of knocked-out species
        - variation_type: Type of variation to get from the dictionary
        - log_file: File where to log informations
    """

    # Updating the all_species set with the model's species
    # for combinations in variations_dict.values():
    #     for species_data in combinations.values():
    #         all_species.update(species_data.keys())

    # all_species = sorted(list(all_species))

    # Creating the result matrix with shape (ko_species, all_species)
    res_matrix = np.zeros((len(ko_species_list), len(all_species)))

    # Looping through the knocked-out species
    for i, ko_species in enumerate(ko_species_list):
        # Extracting the variation dict
        no_sample_combinations = variations_dict[ko_species]["original"]

        # Looping through the species list
        for j, species in enumerate(all_species):
            # If relative variation required
            if variation_type.lower() == "relative":
                res_matrix[i, j] = np.sqrt(
                    no_sample_combinations[species]["relative-variation"] ** 2
                )
            # If variation required
            else:
                res_matrix[i, j] = np.sqrt(
                    no_sample_combinations[species]["variation"] ** 2
                )

    return res_matrix, all_species


def get_variations_hm_samples(
    variations_dict,
    all_species,
    ko_species_list,
    variation_type="relative",
    log_file=None,
):
    """
    Create the variation's heatmap when samples are needed

    Args:
        - variations_dict: Dictionary containing the variations information
        - all_species: List of all the species in the model
        - ko_species_list: List of species that has been knocked out
        - variation_type: Type of variation to plot
        - log_file: File where to print the debug information
    """

    # Updating the all_species set
    # for combinations in variations_dict.values():
    #     for species_data in combinations.values():
    #         all_species.update(species_data.keys())
    #
    # all_species = sorted(list(all_species))

    # Calculate the variation's matrix with shape (ko_species_list, all_species)
    heatmap_data = np.zeros((len(ko_species_list), len(all_species)))

    # Looping through the knocked-out species
    for i, ko_species in enumerate(ko_species_list):
        # Extracting the combinations dict
        combinations = variations_dict[ko_species]

        # Looping through the species
        for j, species in enumerate(all_species):
            variations = []

            # Looping through the combinations
            for combination, combination_data in combinations.items():
                if species in combination_data:
                    if variation_type.lower() == "relative":
                        variations.append(
                            combination_data[species]["relative-variation"]
                        )
                    else:
                        variations.append(combination_data[species]["variation"])

            if variations:
                # Use the RMS to avoid the lost of information
                heatmap_data[i, j] = np.sqrt(np.mean([v**2 for v in variations]))
    return (heatmap_data, all_species)


def get_variations_hm_no_samples(
    variations_dict,
    all_species,
    ko_species_list,
    variation_type="relative",
    log_file=None,
):
    """
    Create the heatmap with the variations information when no samples are needed.

    Args:
        - variation_dict: Dictionary containing the variation for each species
        - all_species: List of all the species in the model
        - ko_species_list: List of species that has been knocked out
        - variation_type: Variation type to use to build the heatmap
        - log_file: File where to print the debug informations
    Returns:
        - A couple (heatmap, all_species)
    """

    # for ko_spec, obj in variations_dict.items():
    #     all_species.update(obj.keys())

    # all_species = sorted(list(all_species))

    # Calculate the variation's matrix
    heatmap_data = np.zeros((len(ko_species_list), len(all_species)))

    for i, ko_species in enumerate(ko_species_list):
        for j, species in enumerate(all_species):
            if ko_species == species:
                # Automatically skip self-variation (KO species vs itself)
                heatmap_data[i, j] = np.nan
                print_log(log_file, f"Skipping KO species {ko_species} for itself.")
            else:
                try:
                    variation_entry = variations_dict[ko_species][species]
                    key = (
                        "relative-variation"
                        if variation_type.lower() == "relative"
                        else "variation"
                    )

                    heatmap_data[i, j] = variation_entry[key]
                except KeyError:
                    # If missing from variations_dict, fill with NaN
                    heatmap_data[i, j] = np.nan
                    print_log(
                        log_file,
                        f"Missing data for KO {ko_species}, species {species}.",
                    )

    return heatmap_data


# KEEP
def simulate_combinations(
    rr: rr.RoadRunner,
    combinations: list,
    input_species_ids: list,
    min_ss_time: float,
    end_time: float = 1000,
    max_end_time: float = 1000,
    steady_state: bool = False,
    log_file=None,
) -> tuple:
    """
    Simulate a RoadRunner model across multiple input perturbation combinations.

    This function runs simulations for each combination of input species concentrations,
    collecting results for perturbation analysis. It supports both standard time-course
    and steady-state simulations, tracking the minimum steady-state time across all runs.

    Parameters
    ----------
    rr : roadrunner.RoadRunner
        Configured RoadRunner model instance to simulate across all combinations.
        The model is not reset between combinations to preserve configuration.
    combinations : list of list
        List of input concentration combinations to simulate.
        Each element is a list of concentration values matching input_species_ids.
        Example: [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]] for 2 input species
    input_species_ids : list of str
        IDs of input species whose concentrations will be perturbed.
        Must match the length of each combination in combinations.
    min_ss_time : float
        Initial minimum steady-state time threshold.
        Updated during execution to track the earliest steady state reached.
    end_time : float, optional
        End time for standard simulations (when steady_state=False), by default 1000
    max_end_time : float, optional
        Maximum simulation time for steady-state detection, by default 1000
    steady_state : bool, optional
        If True, simulate until steady state is reached for each combination,
        by default False
    log_file : file, optional
        File object for logging progress and steady-state times, by default None

    Returns
    -------
    tuple of (list of numpy.ndarray, list of str)
        A tuple containing:
        - samples_simulations_results : list of numpy.ndarray
            List of simulation results, one per combination.
            Each array has shape (n_timepoints, n_species+1) with time as first column
        - colnames : list of str
            Column names from the last simulation (consistent across all runs)

    Notes
    -----
    - The function uses simulate_samples() for each individual combination
    - min_ss_time is tracked but not returned; it's logged if steady_state=True
    - The model state is preserved between combinations (no reset called)
    - Progress is logged after each combination when log_file is provided

    This function is typically used internally by process_species_samples and
    process_species_no_samples for perturbation-based knockout analysis.

    Examples
    --------
    Standard time-course simulations across perturbations:
    >>> rr_model = roadrunner.RoadRunner('model.xml')
    >>> combinations = [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]
    >>> input_ids = ['Input1', 'Input2']
    >>> results, colnames = simulate_combinations(
    ...     rr=rr_model,
    ...     combinations=combinations,
    ...     input_species_ids=input_ids,
    ...     min_ss_time=1000,
    ...     end_time=500,
    ...     steady_state=False
    ... )
    >>> print(f"Simulated {len(results)} combinations")

    Steady-state simulations with logging:
    >>> with open('sim.log', 'w') as log:
    ...     results, colnames = simulate_combinations(
    ...         rr=rr_model,
    ...         combinations=combinations,
    ...         input_species_ids=input_ids,
    ...         min_ss_time=1000,
    ...         max_end_time=2000,
    ...         steady_state=True,
    ...         log_file=log
    ...     )

    See Also
    --------
    simulate_samples : Simulate a single input combination
    simulate : Core simulation function
    process_species_samples : Uses this function for knockout analysis with perturbations
    """

    samples_simulations_results = []
    i = 1
    for comb in combinations:
        # rr.reset()

        # __import__("pprint").pprint(f"comb:{comb}")

        sim_res, ss_time, colnames = simulate_samples(
            rr,
            comb,
            input_species_ids,
            start_time=0,
            end_time=end_time,
            steady_state=steady_state,
            max_end_time=max_end_time,
        )

        # __import__("pprint").pprint(sim_res[:, :])

        min_ss_time = (
            ss_time if ss_time is not None and ss_time <= min_ss_time else min_ss_time
        )
        if steady_state:
            print_log(log_file, f"Min time {min_ss_time}")

        samples_simulations_results.append(sim_res)
    if steady_state:
        print_log(log_file, f"Min ss_time: {min_ss_time}")

    return samples_simulations_results, colnames


def rms_average(array, axis=1, log_file=None):
    """
    Calculate Root Mean Square average, which emphasizes larger values.

    Parameters:
    -----------
    array : numpy.ndarray
        Array of shape (n_simulations, n_timepoints) with perturbed simulations
    axis: Axis to consider

    Returns:
    --------
    numpy.ndarray
        RMS average
    """
    # Square all values, take mean, then take square root
    square = np.square(array)
    print_log(log_file, f"Square: {square}")
    result = np.sqrt(np.nanmean(square, axis=axis))

    return result


def log_transform_average(simulations, epsilon=1e-10):
    """
    Average simulations using log-transform to handle multiplicative perturbations.

    Parameters:
    -----------
    simulations : numpy.ndarray
        Array of shape (m, n) with m simulations and n timepoints
    epsilon : float
        Small value to avoid log(0)

    Returns:
    --------
    numpy.ndarray
        Log-transform averaged simulation of shape (n,)
    """
    # Add small epsilon to avoid log(0) issues
    safe_simulations = simulations + epsilon

    # Log-transform the data
    log_simulations = np.log(safe_simulations)

    # Calculate mean in log space
    mean_log = np.mean(log_simulations, axis=0)

    # Transform back to original space
    mean_simulation = np.exp(mean_log)

    return mean_simulation


def geometric_mean_variation(perturbed_array, original_data):
    """
    Calcola la variazione media usando una trasformazione logaritmica,
    che gestisce correttamente variazioni percentuali opposte come +10% e -10%.

    Esempio:
    +10% (moltiplicatore 1.1) e -10% (moltiplicatore 0.9) hanno una media geometrica di 0.995,
    che corrisponde meglio alla realtà rispetto alla media aritmetica di 1.
    """
    # Converti le variazioni percentuali in moltiplicatori
    ratios = perturbed_array / original_data

    # Calcola la media logaritmica
    log_ratios = np.log(ratios)
    mean_log_ratio = np.nanmean(log_ratios, axis=0)

    # Riconverti in percentuale
    return (np.exp(mean_log_ratio) - 1) * 100


def keq_from_equilibrium_concentrations(
    self, reactant_concs, product_concs, stoichiometry=None
):
    """
    Calculate Keq from equilibrium concentrations
    For reaction: aA + bB ⇌ cC + dD
    Keq = ([C]^c * [D]^d) / ([A]^a * [B]^b)

    Args:
        reactant_concs: List of reactant concentrations at equilibrium
        product_concs: List of product concentrations at equilibrium
        stoichiometry: Dict with stoichiometric coefficients
                      e.g., {'reactants': [1, 1], 'products': [1, 1]}

    Returns:
        Equilibrium constant
    """
    if stoichiometry is None:
        stoichiometry = {
            "reactants": [1] * len(reactant_concs),
            "products": [1] * len(product_concs),
        }

    # Calculate numerator (products)
    numerator = 1
    for i, conc in enumerate(product_concs):
        numerator *= conc ** stoichiometry["products"][i]

    # Calculate denominator (reactants)
    denominator = 1
    for i, conc in enumerate(reactant_concs):
        denominator *= conc ** stoichiometry["reactants"][i]

    keq = numerator / denominator
    return keq


def analyze_directional_variations(variations_percentage):
    """
    analyze directional variations
    """

    pos_variations = []
    neg_variations = []

    for variations in variations_percentage:
        for vp in variations:
            if vp is not None and vp >= 0:
                pos_variations.append(vp)
            else:
                neg_variations.append(vp)

    mean_pos = np.nanmean(pos_variations)
    mean_neg = np.nanmean(neg_variations)

    return mean_pos, mean_neg
