from decimal import *
import roadrunner as rr
import libsbml
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import defaultdict

from src.utils.utils import dict_pretty_print, print_log
from src.utils import plot_utils as plt_ut


def load_roadrunner_model(sbml_model, integrator=None, log_file=None):
    """
    Loads a SBML model into a RoadRunner instance and configures the integrator settings.

    Args:
        sbml_model: The SBML model (libsbml.Model or string)
        rel_tol: Relative tolerance for the integrator (default: 1e-6)
        abs_tol: Absolute tolerance for the integrator (default: 1e-8)

    Returns:
        roadrunner.RoadRunner: Configured RoadRunner instance
    """
    # Check if input is a libSBML model or a string
    if isinstance(sbml_model, libsbml.Model):
        sbml_doc = libsbml.SBMLWriter().writeSBMLToString(sbml_model.getSBMLDocument())
        rr_model = rr.RoadRunner(sbml_doc)
    else:
        # Assume it's a string representation or file path
        rr_model = rr.RoadRunner(sbml_model)

    # Configure integrator settings
    # rr_model.integrator.nonnegative = True
    rr_model.getIntegrator().setValue("relative_tolerance", 1e-8)
    rr_model.getIntegrator().setValue("absolute_tolerance", 1e-10)

    if integrator is not None:
        rr_model.setIntegrator(integrator)

    # print_log(log_file, f"integrator: {rr_model.getIntegrator()}")

    return rr_model


def simulate(
    rr_model,
    start_time=0,
    end_time=100,
    output_rows=100,
    steady_state=False,
    max_end_time=1000,
    sim_step=5,
    threshold=1e-6,
    log_file=None,
):
    """
    Simulate the model with optional steady state detection.
    """
    # Setting nonnegative for stochastic simulations
    if rr_model.getIntegrator().getName() == "gillespie":
        rr_model.getIntegrator().nonnegative = True

    if steady_state:
        print_log(
            log_file,
            f"Simulating until steady state (max time: {max_end_time}, threshold: {threshold})",
        )

        # Calculate a reasonable number of points per block based on output_rows and sim_step
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

        # print_log(log_file, "Normal simulations")

        res = rr_model.simulate(start_time, end_time, output_rows)

        return res, None, res.colnames


def simulate_samples(
    rr_model,
    combination,
    input_species_id,
    max_end_time,
    start_time=0,
    end_time=10,
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
    # Save the current selections
    current_selections = rr_model.selections

    # Set the new concentrations
    if rr_model.getIntegrator().getName() == "gillespie":
        rr_model.getIntegrator().nonnegative = True

    for i in range(len(input_species_id)):
        rr_model.setInitConcentration(input_species_id[i], combination[i])

    # rr_model.regenerateModel()

    # Reset the concentrations
    # Needed because the .reset() method reset also the simulation selections
    rr_model.selections = current_selections

    res = simulate(
        rr_model,
        start_time,
        end_time,
        output_rows,
        steady_state=steady_state,
        max_end_time=max_end_time,
    )

    return res
    # return rr_model.simulate(start_time, end_time, output_rows)


# FIX: Refactor the code
def analyze_simulation_variations(  # TODO: Check correctness
    target_data, original_results, output_dir, log_file=None, target_type=0
):
    """
    Analizza le variazioni delle simulazioni rispetto ai risultati originali.

    Args:
        target_data: Dizionario con i dati delle specie target
        original_results: Risultati della simulazione originale
        output_dir: Directory di output per i grafici
        log_file: File di log
        target_type: Tipo di target (0=specie, 1=reazione)
    """
    precision = int(os.getenv("SIMULATION_PREC", "20"))
    mask_value = 1e-20

    if target_type == 1:  # The target is a reaction
        # In this case original results and target data are lists
        pass
    else:
        for target_species, data in target_data.items():
            original_data = data["original"]
            simulations_data = data["simulations"]
            time = original_results[:, 0]
            print_log(log_file, f"time len: {len(time)}")
            mask = np.abs(original_data) > mask_value

            # Get data results
            perturbed_data_list = [sim["results"] for sim in simulations_data]

            # check if every simulation has the same length
            lengths = [len(sim_data) for sim_data in perturbed_data_list]
            min_length = min(lengths)

            if not all(length == min_length for length in lengths):
                print_log(
                    log_file,
                    f"Warning: Simulation results have different lengths. Truncating to minimum length: {min_length}",
                )

                perturbed_data_list = [
                    sim_data[:min_length] for sim_data in perturbed_data_list
                ]
                original_data = original_data[:min_length]
                time = time[:min_length]
                mask = mask[:min_length]

            # Using numpy array
            perturbed_array = np.array(perturbed_data_list)

            # End value analysis
            last_original_value = original_data[-1]
            last_simulations_values = perturbed_array[
                :, -1
            ]  # Get last values from all simulations

            print_log(
                log_file,
                f"Analyzing {target_species} with {len(last_simulations_values)} simulations",
            )

            # Initialize variations array
            if np.abs(last_original_value) > mask_value:
                last_value_percents_variations = (
                    (last_simulations_values - last_original_value)
                    / last_original_value
                ) * 100
                variation_type = "percent"
            else:
                print_log(
                    log_file,
                    "  Last original value near zero - using absolute differences",
                )
                last_value_percents_variations = (
                    last_simulations_values - last_original_value
                )
                variation_type = "absolute"

            # Calculate statistics
            # print_log(
            #     log_file,
            #     f"Last value percent variations: {last_value_percents_variations}",
            # )

            last_value_rms_variations = rms_average(
                last_value_percents_variations, axis=0
            )

            last_value_std_variations = np.std(
                last_value_percents_variations, mean=last_value_rms_variations
            )

            print_log(
                log_file,
                f"[{target_species}]Last value variations RMS: {last_value_rms_variations}",
            )
            print_log(
                log_file,
                f"[{target_species}]Last value variations std: {last_value_std_variations}",
            )

            # FULL TIME ANALYSIS
            # FIX: Problem with nan values
            # FIX: Problem with average
            variations = np.full_like(perturbed_array, np.nan)
            variations_percentage = np.full_like(perturbed_array, np.nan)

            # Separate masks for percentage and absolute variations
            valid_mask_percent = mask & (np.abs(original_data) > mask_value)
            valid_mask_absolute = mask & (np.abs(original_data) <= mask_value)

            # Calculate percentage variations where original data is significant
            variations[:, valid_mask_percent] = (
                perturbed_array[:, valid_mask_percent]
                - original_data[valid_mask_percent]
            ) / original_data[valid_mask_percent]

            # For points where original data is near zero, use absolute differences
            variations[:, valid_mask_absolute] = (
                perturbed_array[:, valid_mask_absolute]
                - original_data[valid_mask_absolute]
            )

            variations_percentage = variations * 100

            # Only calculate RMS for valid percentage variations
            variations_percentage_rms = np.full(variations_percentage.shape[1], np.nan)
            if np.any(valid_mask_percent):
                variations_percentage_rms[valid_mask_percent] = rms_average(
                    variations_percentage[:, valid_mask_percent], axis=0
                )

            avg_variations_percentage_rms = np.mean(variations_percentage_rms)
            std_variations_percentage_rms = np.std(
                variations_percentage, mean=avg_variations_percentage_rms
            )

            print_log(
                log_file,
                f"[{target_species}]Std variation percentage RMS: {std_variations_percentage_rms:.{precision}f}%",
            )
            print_log(
                log_file,
                f"[{target_species}]Average variations percentage RMS:±{avg_variations_percentage_rms:.{precision}f}%",
            )

            # plotting
            species_dir = os.path.join(output_dir, target_species)
            os.makedirs(species_dir, exist_ok=True)

            # plotting simulations comparison
            plt_ut.plot_all_simulation_traces(
                time, original_data, perturbed_array, target_species, species_dir
            )

            plt_ut.plot_statistical_comparison(
                time,
                original_data,
                perturbed_array,
                avg_variations_percentage_rms,
                target_species,
                species_dir,
            )

            plt_ut.plot_boxplot_distribution(
                perturbed_array, time, target_species, species_dir
            )

            print_log(log_file, f"  Created plots in {species_dir}\n")


def aggregate_by_combination(final_results, log_file=None):
    """
    Aggregates results by combination with detailed information for each target species.

    Returns:
        Dict: {2
            'combination_key': {
                'target_species_1': {
                    'last_original_value': float,
                    'last_perturbed_value': float,
                    'variation': float,
                    'variation_type': str,
                    'original_series': array,
                    'perturbed_series': array
                },
                'target_species_2': { ... },
                ...
            },
            ...
        }
    """
    aggregated_result = {}

    # First collect all unique combinations
    all_combinations = set()
    for target_species, ts_data in final_results.items():
        all_combinations.update(ts_data.keys())

    # For each combination, collect data from all target species
    for combination in all_combinations:
        aggregated_result[combination] = {}

        for target_species, ts_data in final_results.items():
            if combination in ts_data:
                value = ts_data[combination]
                last_original_value = value["original"][-1]
                last_perturbed_value = value["perturbed_result"][-1]

                # Calculate variation
                try:
                    if abs(last_original_value) > 1e-30:  # Avoid division by zero
                        variation = (
                            last_perturbed_value - last_original_value
                        ) / last_original_value
                        variation_type = "relative"
                    else:
                        variation = last_perturbed_value - last_original_value
                        variation_type = "absolute"
                except:
                    variation = last_perturbed_value - last_original_value
                    variation_type = "absolute"

                # Add information for this target species
                aggregated_result[combination][target_species] = {
                    "last_original_value": last_original_value,
                    "last_perturbed_value": last_perturbed_value,
                    "variation": variation,
                    "variation_type": variation_type,
                    "original_series": value["original"],
                    "perturbed_series": value["perturbed_result"],
                }

    for combination, combination_data in aggregated_result.items():
        print_log(log_file, f"combination: {combination}")
        for target_species, ts_data in combination_data.items():
            print_log(log_file, f"  target_species: {target_species}")
            print_log(
                log_file, f"      last_original_value:{ts_data['last_original_value']}"
            )
            print_log(
                log_file,
                f"      last_perturbed_value_value:{ts_data['last_perturbed_value']}",
            )
            print_log(log_file, f"      variation:{ts_data['variation']}")
            print_log(log_file, f"      variation_type:{ts_data['variation_type']}")
    return aggregated_result


def get_simulations_informations(
    samples_simulations_results,
    original_results,
    combinations,
    species_list,
    colnames,
    log_file=None,
):
    """
    Returns a dictionary with original and perturbation results for each target species.
    """

    target_indices = {}
    missing_species = []
    counter = 0

    # Pre-compute column indices for all target species
    for sp in species_list:
        try:
            # Try both formats in one go
            for col_format in [f"[{sp}]", f"{sp}"]:
                if col_format in colnames:
                    # print_log(log_file, f"[GET SIMULATION INFO]{col_format}, {colnames}")

                    target_indices[sp] = colnames.index(col_format)
                    break
            else:
                missing_species.append(sp)
        except Exception as e:
            print_log(log_file, f"Error finding column for species {sp}: {e}")
            missing_species.append(sp)

    if missing_species:
        print_log(log_file, f"Missing species: {missing_species}")

    # Only process species that exist in the data
    # valid_target_ids = [ts for ts in target_ids if ts in target_indices]

    # if not valid_target_ids:
    #     print_log(log_file, "No valid target species found")
    #     return {}

    # Initialize results structure
    table_results = {}

    # Add original results (vectorized)
    table_results["original"] = {}
    for ts in species_list:
        ts_index = target_indices[ts]
        table_results["original"][ts] = original_results[
            -1, ts_index
        ]  # Final concentration

    # Process simulations in batch
    if not combinations or not samples_simulations_results:
        print_log(log_file, "No simulation data to process")
        return table_results

    # Pre-compute combination keys
    combination_keys = [
        "_".join([f"{val:.3f}" for val in combo]) for combo in combinations
    ]

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
        for ts in species_list:
            ts_index = target_indices[ts]
            try:
                table_results[combo_key][ts] = simulation_results[-1, ts_index]
            except (IndexError, TypeError) as e:
                print_log(
                    log_file, f"Error extracting results for sim {i}, species {ts}: {e}"
                )
                table_results[combo_key][ts] = np.nan

        processed_count += 1

    print_log(
        log_file,
        f"Processed {processed_count} simulations for {len(species_list)} species",
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


def get_simulations_variations(
    final_results_original_model, final_results_knocked_model, log_file=None
):
    """
    Calculate variations between original and knocked model results.

    Args:
        final_results_original_model: Dict with original model results
        final_results_knocked_model: List of tuples (species, info_dict)
        log_file: Optional logging file (unused in current implementation)

    Returns:
        Dict: Nested dictionary with variations per species and combination
    """
    variations_dict = {}
    getcontext().prec = 15

    for ko_species, species_dict in final_results_knocked_model:
        variations_dict[ko_species] = {}

        for combination, original_info_dict in final_results_original_model.items():
            variations_dict[ko_species][combination] = {}
            for species, original_value in original_info_dict.items():
                print_log(
                    log_file,
                    f"[GET SIMULATIONS VARIATIONS] ko_species: {ko_species} | species: {species}",
                )
                if species == ko_species:
                    continue

                ko_species_value = species_dict[combination][species]

                variations_dict[ko_species][combination][species] = {
                    "variation": Decimal(ko_species_value) - Decimal(original_value),
                    "relative-variation": (
                        Decimal(ko_species_value) - Decimal(original_value)
                    )
                    / Decimal(original_value),
                }

    for ko_species, ko_species_info_dict in variations_dict.items():
        for combination, combination_info_dict in ko_species_info_dict.items():
            # TODO: Ask if is necessary to remove the knocked out species from the mean
            # print_log(log_file, f"N species: {len(combination_info_dict.keys())}")
            combination_info_dict.pop(ko_species, "speices not in dict")
            # print_log(log_file, f"N species: {len(combination_info_dict.keys())}")

    return variations_dict


def get_variations_mean(
    variations_dict,
    all_species,
    ko_species_list,
    variation_type="relative",
    log_file=None,
):

    for combinations in variations_dict.values():
        for species_data in combinations.values():
            all_species.update(species_data.keys())

    all_species = sorted(list(all_species))

    # Calculate the matrix of mean absolute variations
    heatmap_data = np.zeros((len(ko_species_list), len(all_species)))

    for i, ko_species in enumerate(ko_species_list):
        combinations = variations_dict[ko_species]

        for j, species in enumerate(all_species):
            variations = []
            for combination_data in combinations.values():
                if species in combination_data:
                    if variation_type.lower() == "relative":
                        variations.append(
                            combination_data[species]["relative-variation"]
                        )
                    else:
                        variations.append(combination_data[species]["variation"])

            if variations:
                heatmap_data[i, j] = np.sqrt(np.mean([v**2 for v in variations]))
    return (heatmap_data, all_species)


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
    for i in range(len(combinations)):
        rr.reset()
        # print_log(log_file, f"Simulation nr. {i}")

        sim_res, ss_time, colnames = simulate_samples(
            rr,
            combinations[i],
            input_species_ids,
            start_time=0,
            end_time=end_time,
            steady_state=steady_state,
            max_end_time=max_end_time,
        )
        min_ss_time = (
            ss_time if ss_time is not None and ss_time <= min_ss_time else min_ss_time
        )
        if steady_state:
            print_log(log_file, f"Min time {min_ss_time}")

        samples_simulations_results.append(sim_res)
    if steady_state:
        print_log(log_file, f"Min ss_time: {min_ss_time}")

    return samples_simulations_results


def get_importance_informations(
    target_data, original_results, output_dir, log_file=None
):
    pass


def simulate_with_steady_state(
    rr_model,
    start_time=0,
    max_end_time=1000,
    block_size=10,
    points_per_block=100,
    threshold=1e-6,
    monitor_species=None,
    log_file=None,
):  # TODO: Make it more efficient
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

    Returns:
        Tuple of (simulation_results, steady_state_time, column_names)
    """
    rr_model.setIntegrator("cvode")
    rr_model.reset()
    all_species = rr_model.model.getFloatingSpeciesIds()
    if monitor_species is None:
        monitor_species = all_species
    # Precompute indices to monitor
    monitor_idx = [all_species.index(s) for s in monitor_species]

    # Prepare column names (including 'time')
    colnames = rr_model.timeCourseSelections.copy()

    current_time = start_time
    all_results = []
    prev_block = None
    steady_state_time = None
    steady_blocks_count = 0
    initial_block_size = block_size
    zero_tol = 1e-12
    consecutive_checks = 2
    max_block_size = 50
    # test = rr_model.model.getReactionIds().index("GROWTH")

    while current_time < max_end_time:
        # print(f"Growth Reaction rate: {rr_model.model.getReactionRates()[test]}")
        next_time = min(current_time + block_size, max_end_time)
        print_log(log_file, f"Next time: {next_time}")
        print_log(log_file, f"block_size: {block_size}")
        block_results = rr_model.simulate(current_time, next_time, points_per_block)
        all_results.append(block_results)

        if prev_block is not None:
            # Extract concentrations at the last point
            prev_conc = prev_block[-1, 1 : 1 + len(all_species)]
            curr_conc = block_results[-1, 1 : 1 + len(all_species)]

            variations = np.zeros_like(prev_conc)
            small_mask = np.abs(prev_conc) < zero_tol
            # absolute change where prev is near zero
            variations[small_mask] = np.abs(
                curr_conc[small_mask] - prev_conc[small_mask]
            )
            # relative elsewhere
            variations[~small_mask] = np.abs(
                (curr_conc[~small_mask] - prev_conc[~small_mask])
                / prev_conc[~small_mask]
            )

            # Check if indices are valid
            monitor_idx = [i for i in monitor_idx if i < len(variations)]
            if not monitor_idx:
                print("Warning: No valid species to monitor for steady state")
                monitor_idx = range(min(len(variations), len(all_species)))

            is_steady = False

            # Check steady state for all monitored species
            for idx in monitor_idx:
                print_log(log_file, f"Variation: {variations[idx] < 1e-4}")
                if variations[idx] < 1e-4:
                    is_steady = True
                else:
                    is_steady = False
                    break

            print_log(log_file, f"Is Steady: {is_steady}")

            if is_steady:
                steady_blocks_count += 1
                print_log(log_file, f"Steady block at time {current_time}")

                if steady_blocks_count >= consecutive_checks:
                    steady_state_time = next_time
                    break
            else:
                steady_blocks_count = 0

        if steady_blocks_count > consecutive_checks:
            block_size = min(initial_block_size, block_size * 1.2)
        else:
            block_size = max(block_size * 0.5, max_block_size)

        prev_block = block_results
        current_time = next_time

    # Concatenate, dropping duplicate boundary rows
    if not all_results:
        full_results = np.empty((0, len(colnames)))
    else:
        for i in range(1, len(all_results)):
            # Drop first row of each subsequent block
            all_results[i] = all_results[i][1:]
        full_results = np.vstack(all_results)

    return full_results, steady_state_time, colnames


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


def plot_variation_range(
    time,
    original_data,
    perturbed_array,
    target_species,
    species_dir,
):
    """
    Plot the range of variation using the actual simulations with smallest and largest values.

    Args:
        time: Time points array
        original_data: Original simulation data for the target species
        perturbed_array: Array of all simulation results (shape: n_simulations, n_timepoints)
        target_species: Name of the species being analyzed
        species_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Identifica le simulazioni con valori più piccoli e più grandi
    # Utilizziamo la media di ogni simulazione come criterio
    simulation_means = np.mean(perturbed_array, axis=1)
    min_sim_idx = np.argmin(simulation_means)
    max_sim_idx = np.argmax(simulation_means)

    # Estrai le simulazioni complete con valori minimi e massimi
    min_simulation = perturbed_array[min_sim_idx]
    max_simulation = perturbed_array[max_sim_idx]

    # Plot del range come area riempita tra le due simulazioni
    plt.fill_between(
        time,
        min_simulation,
        max_simulation,
        color="lightblue",
        alpha=0.5,
        label="Simulation Range",
    )

    # Plot delle simulazioni minima e massima come linee
    plt.plot(time, min_simulation, "b-", linewidth=1, alpha=0.7, label="Min Simulation")
    plt.plot(time, max_simulation, "b-", linewidth=1, alpha=0.7, label="Max Simulation")

    # Plot dei dati originali
    plt.plot(time, original_data, "r-", linewidth=2.5, label="Original")

    # Aggiungi etichette e titoli
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{target_species} - Simulation Variation Range")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Aggiungi il numero di simulazioni come annotazione
    n_simulations = perturbed_array.shape[0]
    plt.figtext(
        0.02,
        0.02,
        f"Based on {n_simulations} simulations\nShowing simulations with smallest and largest average values",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(species_dir, f"{target_species}_variation_range.png"))
    plt.close()


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


# def get_concentrations_at_equilibrium(rr_model, species=None, log_file=None):
#
#     floating_species = rr_model.model.getFloatingSpeciesIds()
#     bounded_species = rr_model.model.getBoundarySpeciesIds()
#     tot_species = [f"[{s}]" for s in (floating_species + bounded_species)]
#
#     rr_model.steadyStateSelections = tot_species
#
#     rr_model.conservedMoietyAnalysis = True
#     steady_state = rr_model.steadyState()
#
#     steady_state_concentrations = rr_model.getSteadyStateValuesNamedArray()
#
#     print_log(log_file, f"{steady_state_concentrations.colnames}")
#
#     # TODO: Return an object <species_id, steady_state_concentrations>


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
