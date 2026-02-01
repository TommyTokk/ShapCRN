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

from src.utils.utils import (
    get_ko_species_importance,
    print_log,
    truncate_small_values,
)
from src.utils import plot_utils as plt_ut
from utils.sbml_utils import create_combinations

# from utils.sbml_utils import create_samples_combination, generate_species_samples


# KEEP
def load_roadrunner_model(
    sbml_model: libsbml.Model, rel_tol: float = 1e-8, abs_tol:float = 1e-12, integrator:str ="cvode", log_file=None
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
        exit(f"Error during model loading: {e}")

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
    start_time: float=0,
    end_time:float=1000,
    output_rows:int=100,
    steady_state:bool=False,
    max_end_time:float=1000,
    sim_step:int=5,
    threshold:float=1e-6,
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
    rr_model,
    start_time=0,
    max_end_time=1000,
    block_size=10,
    points_per_block=100,
    threshold=1e-12,
    consecutive_checks=3,
    monitor_species=None,
    log_file=None,
):
    """
    Simulates a model until steady state is reached using a block-by-block approach.

    Args:
        rr_model: RoadRunner model instance
        start_time: Start time for simulation
        max_end_time: Maximum end time if steady state is not reached
        block_size: Size of each simulation block
        points_per_block: Number of points to calculate in each block
        threshold: Threshold for steady state detection
        monitor_species: Species to monitor for steady state detection
        log_file: Optional log file for debugging output

    Returns:
        Tuple of (simulation_results, steady_state_time, column_names)
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
    rr_model,
    combination,
    input_species_id,
    max_end_time,
    start_time=0,
    end_time=1000,
    output_rows=100,
    steady_state=False,
):
    """
    Simulate the specified model, using the specified input combination

    Args:
        - rr_model: Roadrunner mmodel to simulate
        - combination: Single combination of input species concentrations
        - input_species_id: IDs of species considered as input
        - start_time: Start time of the simulation
        - end_time: End time of the simulation
        - output_rows: Output rows of the simulation

    Returns:
        - Simulation results of the specified combination
    """
    # rr_model.reset()
    #
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
def process_species_samples(args):
    """
    Simulates, using perturbations, the specific modified model with the knockout of the knockedout_species
    """
    try:
        (
            knockedout_species,
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

        print_log(log_file, f"Starting processing: {knockedout_species}")

        # Loading the modified model
        modified_rr = load_roadrunner_model(
            modified_model, integrator=integrator, log_file=log_file
        )

        # Setting the selections
        modified_rr.timeCourseSelections = selections

        # Simulating the not perturbed model
        knockout_model_results, ss_time, colnames = simulate(
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
        combinations_knockout_model_results, colnames = simulate_combinations(
            modified_rr,
            create_combinations(samples),
            input_species_ids,
            min_ss_time,
            end_time,
            max_end_time,
            steady_state,
            log_file,
        )

        ko_data = [pd.DataFrame(knockout_model_results[:, 1:], columns=colnames[1:])]

        for i in range(len(combinations_knockout_model_results)):
            ko_res_i = combinations_knockout_model_results[i]
            ko_data.append(pd.DataFrame(ko_res_i[:, 1:], columns=colnames[1:]))

        return (knockedout_species, ko_data)
    except Exception as e:
        raise Exception(f"Error during processing species:\n Error: {e}")


# KEEP
def process_species_no_samples(args):
    try:
        (
            knockedout_species,
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

        print_log(log_file, knockedout_species)

        print_log(log_file, f"Starting processing: {knockedout_species}")

        modified_rr = load_roadrunner_model(
            modified_model, integrator=integrator, log_file=log_file
        )

        # __import__('pprint').pprint(modified_rr.relative_tolerance)

        modified_rr.selections = selections

        knockout_model_results, ss_time, colnames = simulate(
            modified_rr,
            start_time=0,
            end_time=end_time,
            steady_state=steady_state,
            max_end_time=max_end_time,
        )

        min_ss_time = (
            ss_time if ss_time is not None and ss_time <= min_ss_time else min_ss_time
        )

        return (
            knockedout_species,
            pd.DataFrame(knockout_model_results[:, 1:], columns=colnames[1:]),
        )
    except Exception as e:
        raise Exception(f"Error during processing species :\n Error: {e}")


# KEEP
def process_species_multiprocessing(
    target_ids,
    modified_models_dict,
    samples,
    input_species_ids,
    selections,
    integrator,
    start_time,
    end_time,
    steady_state,
    max_end_time,
    min_ss_time,
    log_file=None,
    max_workers=None,
    use_perturbations=False,
    preserve_input=False,
):
    """
    Use multiprocessing to simulates all the knockouts
    """
    # Determine number of workers
    # Use 75% of the available cores
    n_core = int((mp.cpu_count() * 40) / 100)
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

    # print_log(log_file, f"[DEBUG]{len(process_args)}")

    # Create and run the pool
    try:
        # Implement the logic using Pool and imap
        print_log(log_file, f" Starting pool")
        with Pool() as pool:
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
        print_log(log_file, f"Critical error in multiprocessing : {e}")
        exit(1)

    return result


def get_knockout_variation(original_model, ko_models, colnames, log_file=None):
    """
    Calculate the variation and relative variation of species in respect to an internal species knockout
    This is the version used in case perturbations are not required.
    args:
        - original_model: Results of the simulation without the kncokout
        - ko_models: Array of tuples containing the knocked-out species and the simulation results after the species' knockout
        - colnames: Array containing the name of the columns from the simulation output
        - log_file: File used to print log informations

    returns:
        A dictionary having as key the knocked-out species, and as value, for each species in the model, a dictionary containing the variation and
        relative variation
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
    final_results_original_model,
    final_results_knocked_model,
    epsilon=1e-20,
    log_file=None,
):
    """
    Calculate variations and relative variation between original and knocked model results.

    Args:
        final_results_original_model: DataFrame with original model results
        final_results_knocked_model: List of tuples (species, ko_dfs)
        log_file: Optional logging file (unused in current implementation)

    Returns:
        Dict: Nested dictionary with variations per species and combination
    """
    rms = []
    for ko_species, ko_data in final_results_knocked_model:
        variations = []

        for c in range(len(ko_data)):
            ko_sim_i = ko_data[c]
            original_sim_i = final_results_original_model[c]

            # Getting the last values
            last_ko_values = ko_sim_i.tail(1)
            last_original_values = original_sim_i.tail(1)

            # Masking
            last_ko_values = last_ko_values.mask(last_ko_values <= epsilon, 0)
            last_original_values = last_original_values.mask(
                last_original_values <= epsilon, 0
            )

            var = last_ko_values - last_original_values
            variations.append(var)

        variations_df = pd.concat(variations, ignore_index=True)
        variations_rms = np.sqrt((variations_df**2).mean())
        variations_rms.name = ko_species
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
    original_data, ko_data, epsilon=1e-20, log_file=None
):
    variations = []

    for ko_species, ko_info in ko_data:
        last_ko_values = ko_info.tail(1)
        last_original_values = original_data.tail(1)

        # Masking
        last_ko_values = last_ko_values.mask(last_ko_values <= epsilon, 0)
        last_original_values = last_original_values.mask(
            last_original_values <= epsilon, 0
        )

        var = last_ko_values - last_original_values
        # rms_vars = np.sqrt(var**2)
        var_series = var.squeeze()
        var_series.name = ko_species
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
    final_results_original_model,
    final_results_knocked_model,
    epsilon=1e-20,
    log_file=None,
):
    """
    Calculate variations and relative variation between original and knocked model results.

    Args:
        final_results_original_model: DataFrame with original model results
        final_results_knocked_model: List of tuples (species, ko_dfs)
        log_file: Optional logging file (unused in current implementation)

    Returns:
        Dict: Nested dictionary with variations per species and combination
    """
    rms = []
    for ko_species, ko_data in final_results_knocked_model:
        variations = []

        for c in range(len(ko_data)):
            ko_sim_i = ko_data[c]
            original_sim_i = final_results_original_model[c]

            # Getting the last values
            last_ko_values = ko_sim_i.tail(1)
            last_original_values = original_sim_i.tail(1)

            # Masking
            last_ko_values = last_ko_values.mask(last_ko_values <= epsilon, 0)
            last_original_values = last_original_values.mask(
                last_original_values <= epsilon, 0
            )

            var = (last_ko_values - last_original_values) / last_original_values
            variations.append(var)

        variations_df = pd.concat(variations, ignore_index=True)
        variations_rms = np.sqrt((variations_df**2).mean())
        variations_rms.name = ko_species
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
    original_data, ko_data, epsilon=1e-20, log_file=None
):
    variations = []

    for ko_species, ko_info in ko_data:
        last_ko_values = ko_info.tail(1)
        last_original_values = original_data.tail(1)

        # Masking
        last_ko_values = last_ko_values.mask(last_ko_values <= epsilon, 0)
        last_original_values = last_original_values.mask(
            last_original_values <= epsilon, 0
        )

        var = (last_ko_values - last_original_values) / last_original_values

        # rms_vars = np.sqrt(var**2)
        var_series = var.squeeze()
        var_series.name = ko_species
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
def get_payoff_vals(
    final_results_original_model,
    final_results_knocked_model,
    payoff_function,
    epsilon=1e-20,
    log_file=None,
):
    res = []

    ko_species_list = []

    for ko_species, _ in final_results_knocked_model:
        ko_species_list.append(ko_species)

    for ko_species, ko_data in final_results_knocked_model:
        payoffs = []

        for c in range(len(ko_data)):
            ko_sim_i = ko_data[c]
            original_sim_i = final_results_original_model[c]

            payoff_original = payoff_function(original_sim_i)
            payoff_ko = payoff_function(ko_sim_i)

            payoff_diff = payoff_original - payoff_ko

            payoffs.append(payoff_diff)

        # Convert payoffs in an unique DataFrame
        payoffs_df = pd.concat(payoffs, ignore_index=True)
        # Append to the results

        res.append((ko_species, payoffs_df))
    # Return a list of tuple (ko_species, payoffs)
    return res


# KEEP
def get_shapley_values(payoff_values, n_combinations, n_inputs, log_file=None):
    shap_vals = []
    left_factor = (
        factorial(n_inputs) * factorial(n_combinations - n_inputs)
    ) / factorial(n_combinations)

    # __import__("pprint").pprint(left_factor)

    for ko_species, payoffs in payoff_values:
        factors = left_factor * payoffs

        sums = factors.sum()
        sums.name = ko_species
        shap_vals.append(sums)

    shap_df = pd.DataFrame(shap_vals)

    # __import__("pprint").pprint(shap_df["[species_10]"])

    return shap_df


# KEEP
def generate_values_distance_report(
    distance_matrix,
    correlation_coefficient,
    p_value,
    alternative,
    ko_species_list,
    model_name,
    saving_path,
    threshold=0.2,
    alpha=0.05,
    report_title="VALUES DISTANCE REPORT",
    log_file=None,
):
    """
    Generate a comprehensive distance report.

    Args:
        distance_matrix: 2D numpy array with distance values
        ko_species_list: List of knockout species names
        model_name: Name of the model (for the report filename)
        saving_path: Directory where to save the report
        threshold: Threshold for significant differences
        log_file: Optional log file
    """
    # Check if the destinations folder exists otherwise create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)
        print_log(log_file, f"Created directory: {saving_path}")

    # Determine correlation strength
    abs_r = abs(correlation_coefficient)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    # Determine direction
    if correlation_coefficient > 0:
        direction = "positive"
    elif correlation_coefficient < 0:
        direction = "negative"
    else:
        direction = "no"

    # Statistical significance
    is_significant = p_value < alpha
    significance_level = (
        "highly significant"
        if p_value < 0.001
        else "significant"
        if p_value < 0.01
        else "marginally significant"
    )

    # Hypothesis description
    if alternative == "two-sided":
        null_hyp = "H₀: No linear relationship exists (r = 0)"
        alt_hyp = "H₁: Linear relationship exists (r ≠ 0)"
        p_interpretation = f"probability of observing a correlation this extreme (in either direction) by chance"
    elif alternative == "greater":
        null_hyp = "H₀: No positive relationship (r ≤ 0)"
        alt_hyp = "H₁: Positive relationship exists (r > 0)"
        p_interpretation = (
            f"probability of observing a correlation this large or larger by chance"
        )
    elif alternative == "less":
        null_hyp = "H₀: No negative relationship (r ≥ 0)"
        alt_hyp = "H₁: Negative relationship exists (r < 0)"
        p_interpretation = f"probability of observing a correlation this negative or more negative by chance"
    else:
        raise Exception("Alternative not valid")

    # Build report
    report = f"""CORRELATION RESULTS:
    -------------------
    Correlation coefficient (r): {correlation_coefficient:.6f}
    P-value: {p_value:.6f}
    Test type: {alternative} 
    Significance level (α): {alpha}

    STATISTICAL INTERPRETATION:
    --------------------------
    • Correlation strength: {strength.title()} {direction} correlation
    • Statistical significance: {"" if is_significant else "Not "}{significance_level}
    • Null hypothesis: {null_hyp}
    • Alternative hypothesis: {alt_hyp}\n\n"""

    # Getting additional metrics
    max_diff = np.nanmax(distance_matrix)
    min_diff = np.nanmin(distance_matrix)
    mean_diff = np.nanmean(distance_matrix)
    std_diff = np.nanstd(distance_matrix)

    significant_differences = np.sum(distance_matrix > threshold)
    total_comparisons = distance_matrix.size

    # Calculate knockout importance
    ko_impact, ko_ranking = get_ko_species_importance(
        distance_matrix, ko_species_list, log_file
    )

    # Generate report filename
    report_filename = report_title
    report_path = os.path.join(saving_path, report_filename)

    # Write the report
    try:
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{report_title.upper()}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model: {model_name}\n")
            f.write(
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("=" * 80 + "\n")
            f.write("PEARSON ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(report)

            f.write("=" * 80 + "\n")
            f.write("DISTANCE MATRIX ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Threshold: {threshold}\n\n")

            f.write("MATRIX STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Maximum global Distance: {max_diff}\n")
            f.write(f"Minimum global Distance: {min_diff}\n")
            f.write(f"Mean gloabl Distance: {mean_diff}\n")
            f.write(f"Gloabl standard Deviation: {std_diff}\n")
            # f.write(
            #     f"Significant Differences: {significant_differences}/{total_comparisons}\n"
            # )
            # f.write(
            #     f"Significance Rate: {(significant_differences/total_comparisons)*100:.20f}%\n\n"
            # )

            if ko_impact is not None and ko_ranking is not None:
                f.write("KNOCKOUT SPECIES RANKING:\n")
                f.write("-" * 40 + "\n")
                f.write("Top 10 knockouts most sensitive to parameter uncertainty:\n")
                for i in range(min(10, len(ko_ranking))):
                    ko_idx = ko_ranking[i]
                    ko_name = ko_species_list[ko_idx]
                    impact_score = ko_impact[ko_idx]
                    if np.isnan(impact_score):
                        f.write(f"  {i + 1:2d}. {ko_name:<20} NaN (no valid data)\n")
                    else:
                        f.write(f"  {i + 1:2d}. {ko_name:<20} {impact_score:.20f}\n")
            else:
                f.write("KNOCKOUT SPECIES RANKING: Unable to calculate\n")

            f.write("\n" + "=" * 80 + "\n")

        print_log(log_file, f"Distance report saved to: {report_path}")

    except Exception as e:
        print_log(log_file, f"Error writing report to {report_path}: {e}")
        raise

    return report_path


def generate_pattern_distance_report(
    distance_matrix,
    correlation_coefficient,
    p_value,
    alternative,
    ko_species_list,
    model_name,
    saving_path,
    threshold=0.2,
    alpha=0.05,
    report_title="PATTERN DISTANCE REPORT",
    log_file=None,
):
    """
    Generate a comprehensive distance report.

    Args:
        distance_matrix: 2D numpy array with distance values
        ko_species_list: List of knockout species names
        model_name: Name of the model (for the report filename)
        saving_path: Directory where to save the report
        threshold: Threshold for significant differences
        log_file: Optional log file
    """
    # Check if the destinations folder exists otherwise create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)
        print_log(log_file, f"Created directory: {saving_path}")

    # Determine correlation strength
    abs_r = abs(correlation_coefficient)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    # Determine direction
    if correlation_coefficient > 0:
        direction = "positive"
    elif correlation_coefficient < 0:
        direction = "negative"
    else:
        direction = "no"

    # Statistical significance
    is_significant = p_value < alpha
    significance_level = (
        "highly significant"
        if p_value < 0.001
        else "significant"
        if p_value < 0.01
        else "marginally significant"
    )

    # Hypothesis description
    if alternative == "two-sided":
        null_hyp = "H₀: No linear relationship exists (r = 0)"
        alt_hyp = "H₁: Linear relationship exists (r ≠ 0)"
        p_interpretation = f"probability of observing a correlation this extreme (in either direction) by chance"
    elif alternative == "greater":
        null_hyp = "H₀: No positive relationship (r ≤ 0)"
        alt_hyp = "H₁: Positive relationship exists (r > 0)"
        p_interpretation = (
            f"probability of observing a correlation this large or larger by chance"
        )
    elif alternative == "less":
        null_hyp = "H₀: No negative relationship (r ≥ 0)"
        alt_hyp = "H₁: Negative relationship exists (r < 0)"
        p_interpretation = f"probability of observing a correlation this negative or more negative by chance"
    else:
        raise Exception("Alternative not valid")

    # Build report
    report = f"""CORRELATION RESULTS:
    -------------------
    Correlation coefficient (r): {correlation_coefficient:.6f}
    P-value: {p_value:.6f}
    Test type: {alternative} 
    Significance level (α): {alpha}

    STATISTICAL INTERPRETATION:
    --------------------------
    • Correlation strength: {strength.title()} {direction} correlation
    • Statistical significance: {"" if is_significant else "Not "}{significance_level}
    • Null hypothesis: {null_hyp}
    • Alternative hypothesis: {alt_hyp}\n\n"""

    # Generate report filename
    report_filename = f"{model_name}_pattern_distance_report.txt"
    report_path = os.path.join(saving_path, report_filename)

    # Write the report
    try:
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{report_title.upper()}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model: {model_name}\n")
            f.write(
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("=" * 80 + "\n")
            f.write("PEARSON ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(report)

            f.write("\n" + "=" * 80 + "\n")

        print_log(log_file, f"Distance report saved to: {report_path}")

    except Exception as e:
        print_log(log_file, f"Error writing report to {report_path}: {e}")
        raise

    return report_path


def get_simulations_informations(
    samples_simulations_results,
    original_results,
    combinations,
    colnames,
    log_file=None,
):
    """
    Returns a dictionary with original and perturbation results for each target species.

    Args:
        - samples_simulations_results: Dictionary containing, for each combination, the results of the simulation
        - original_results: Structured array containing the results from the original model
        - combinations: List of combinations
        - colnames: List of colnames of the model
        - log_file: File used to print log informations
    """

    target_indices = {}
    target_ids = []
    missing_species = []

    exp = r"[\[\]]"  # Expression to filter the colnames

    # Pre-compute column indices for all target species
    for cn in colnames:
        s_id = re.sub(exp, "", cn)
        target_indices[s_id] = colnames.index(cn)
        try:
            target_indices[s_id] = colnames.index(cn)
            target_ids.append(s_id)
        except Exception as e:
            print_log(log_file, f"Error finding column for species {s_id}: {e}")
            missing_species.append(s_id)

    # Remove time
    try:
        _ = target_indices.pop("time")
    except Exception as e:
        print_log(log_file, f"Time not present: {e}")

    # Initialize results structure
    table_results = {}

    # Add original results (vectorized)
    table_results["original"] = {}
    for ts in target_ids:
        if ts == "time":  # Skipping time column
            continue
        ts_index = target_indices[ts]

        table_results["original"][ts] = truncate_small_values(
            original_results[-1, ts_index]
        )  # Final concentration

    # Process simulations in batch
    if not combinations or not samples_simulations_results:
        print_log(log_file, "No simulation data to process")
        return table_results

    # Pre-compute combination keys
    combination_keys = ["_".join([f"{val}" for val in combo]) for combo in combinations]

    # Process all simulations at once
    processed_count = 0
    for i, combo_key in enumerate(combination_keys):
        # print_log(log_file, f"Progress: {(counter/len(combination_keys))*100}%")
        # counter += 1

        if i >= len(samples_simulations_results):
            print_log(
                log_file,
                f"MISSING: Simulation {i} missing results data (total available: {len(samples_simulations_results)})",
            )
            continue

        table_results[combo_key] = {}
        simulation_results = samples_simulations_results[i]

        # Extract final concentrations for all target species at once
        for ts in target_ids:
            if ts == "time":
                continue

            ts_index = target_indices[ts]
            try:
                table_results[combo_key][ts] = truncate_small_values(
                    simulation_results[-1, ts_index]
                )
            except (IndexError, TypeError) as e:
                print_log(
                    log_file, f"Error extracting results for sim {i}, species {ts}: {e}"
                )
                table_results[combo_key][ts] = np.nan

        processed_count += 1

    print_log(
        log_file,
        f"Processed {processed_count} simulations for {len(target_indices)} species",
    )

    return table_results


def get_simulations_informations_with_detailed_data(
    samples_simulations_results,
    original_results,
    combinations,
    input_species_ids,
    target_ids,
    colnames,
    log_file=None,
):
    """
    Alternative version that keeps the detailed simulation data structure
    while still being more efficient than the original.
    """

    # Pre-compute column indices
    target_indices = {}
    for ts in target_ids:
        try:
            for col_format in [f"[{ts}]", f"{ts}"]:
                if col_format in colnames:
                    target_indices[ts] = colnames.index(col_format)
                    break
            else:
                continue
        except Exception as e:
            print_log(log_file, f"Species {ts} not present in the results: {e}")
            continue

    if not target_indices:
        return {}

    # Build detailed data structure more efficiently
    target_species_data = {}

    for ts, ts_index in target_indices.items():
        target_species_data[ts] = {
            "original": original_results[:, ts_index],
            "simulations": [],
        }

        # Process all simulations for this species
        for i, combination_values in enumerate(combinations):
            if i >= len(samples_simulations_results):
                continue

            try:
                combination_str = "-".join(
                    [
                        f"{species}:{value:.4f}"
                        for species, value in zip(input_species_ids, combination_values)
                    ]
                )

                simulation_data = {
                    "id": f"sim_{i}_{combination_str}",
                    "combination": combination_values,
                    "results": samples_simulations_results[i][:, ts_index],
                }

                target_species_data[ts]["simulations"].append(simulation_data)

            except Exception as e:
                print_log(
                    log_file,
                    f"Error extracting data for simulation {i}, species {ts}: {e}",
                )

        print_log(
            log_file,
            f"Processed {len(target_species_data[ts]['simulations'])} simulations for species {ts}",
        )

    # Build table structure efficiently
    table_results = {"original": {}}

    # Add original concentrations
    for ts in target_indices:
        table_results["original"][ts] = target_species_data[ts]["original"][-1]

    # Create lookup for faster simulation data access
    sim_lookup = defaultdict(dict)
    for ts, data in target_species_data.items():
        for sim_data in data["simulations"]:
            combo_tuple = tuple(sim_data["combination"])
            sim_lookup[combo_tuple][ts] = sim_data["results"][-1]

    # Add perturbation rows
    for combination_values in combinations:
        combo_key = "_".join([f"{val:.3f}" for val in combination_values])
        combo_tuple = tuple(combination_values)

        table_results[combo_key] = {}
        for ts in target_indices:
            table_results[combo_key][ts] = sim_lookup[combo_tuple].get(ts, np.nan)

    print_log(
        log_file,
        f"Created table structure with {len(table_results)} rows and {len(target_indices)} columns",
    )

    return table_results


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
    rr,
    combinations,
    input_species_ids,
    min_ss_time,
    end_time,
    max_end_time,
    steady_state=False,
    log_file=None,
):
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
