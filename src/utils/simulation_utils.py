import roadrunner as rr
import libsbml
import numpy as np
import matplotlib.pyplot as plt
import os
import math


from src.utils.utils import print_log
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
    rr_model.getIntegrator().setValue("relative_tolerance", 1e-8)
    rr_model.getIntegrator().setValue("absolute_tolerance", 1e-10)

    if integrator is not None:
        rr_model.setIntegrator(integrator)

    print_log(log_file, f"integrator: {rr_model.getIntegrator()}")

    return rr_model


def simulate(rr_model, start_time=0, end_time=100, output_rows=100, log_file=None):
    # TODO: Ask if is correct to force a positive value
    if rr_model.getIntegrator().getName() == "gillespie":
        rr_model.getIntegrator().nonnegative = True

    result = rr_model.simulate(start_time, end_time, output_rows)
    return result


def simulate_samples(
    rr_model, combination, input_species_id, start_time=0, end_time=10, output_rows=100
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

    return rr_model.simulate(start_time, end_time, output_rows)


def analyze_simulation_variations(
    target_species_data, original_results, output_dir, log_file=None
):

    precision = int(os.getenv("SIMULATION_PREC", "20"))
    mask_value = 1e-20
    for target_species, data in target_species_data.items():
        original_data = data["original"]
        simulations_data = data["simulations"]
        time = original_results[:, 0]
        mask = np.abs(original_data) > mask_value

        perturbed_data_list = [sim["results"] for sim in simulations_data]
        perturbed_array = np.array(
            perturbed_data_list
        )  # Shape: (n_simulations, n_time_points)

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
            percents_variations = (
                (last_simulations_values - last_original_value) / last_original_value
            ) * 100
            variation_type = "percent"
        else:
            print_log(
                log_file, "  Last original value near zero - using absolute differences"
            )
            percents_variations = last_simulations_values - last_original_value
            variation_type = "absolute"

        # Calculate statistics
        last_value_avg = np.mean(percents_variations)
        last_value_max = np.max(percents_variations)
        last_value_min = np.min(percents_variations)

        # Full timeline analysis
        variations_percentage = np.full_like(perturbed_array, np.nan)
        for sim_idx, simulation in enumerate(perturbed_array):
            valid_mask = mask & (np.abs(original_data) > mask_value)
            variations_percentage[sim_idx, valid_mask] = (
                (simulation[valid_mask] - original_data[valid_mask])
                / original_data[valid_mask]
            ) * 100

        # Calculate timeline statistics (automatically ignores NaNs)
        avg_percentage_variation = np.nanmean(variations_percentage, axis=0)
        overall_avg = np.nanmean(avg_percentage_variation)
        max_avg = np.nanmax(avg_percentage_variation)
        min_avg = np.nanmin(avg_percentage_variation)

        # Reporting
        variation_suffix = "%" if variation_type == "percent" else " units"
        print_log(log_file, f"  Statistics for {target_species}:")
        print_log(
            log_file, f"    - Average overall variation: {overall_avg:.{precision}f}%"
        )
        print_log(
            log_file, f"    - Maximum pointwise increase: {max_avg:.{precision}f}%"
        )
        print_log(
            log_file, f"    - Maximum pointwise decrease: {min_avg:.{precision}f}%"
        )
        print_log(
            log_file,
            f"    - Final value avg variation: {last_value_avg:.{precision}f}{variation_suffix}",
        )
        print_log(
            log_file,
            f"    - Final value max variation: {last_value_max:.{precision}f}{variation_suffix}",
        )
        print_log(
            log_file,
            f"    - Final value min variation: {last_value_min:.{precision}f}{variation_suffix}",
        )

        # Plotting
        species_dir = os.path.join(output_dir, target_species)
        os.makedirs(species_dir, exist_ok=True)

        # Statistical comparison plot
        mean_values = np.nanmean(perturbed_array, axis=0)
        std_values = np.nanstd(perturbed_array, axis=0)
        plt_ut.plot_statistical_comparison(
            time,
            original_data,
            mean_values,
            std_values,
            np.nanmin(perturbed_array, axis=0),
            np.nanmax(perturbed_array, axis=0),
            target_species,
            species_dir,
        )

        # Percentage variation plot
        plt_ut.plot_percentage_variation(
            time,
            avg_percentage_variation,
            variations_percentage,
            target_species,
            species_dir,
        )

        # Boxplot distribution
        plt_ut.plot_boxplot_distribution(
            perturbed_array, time, target_species, species_dir
        )

        print_log(log_file, f"  Created plots in {species_dir}\n")


def simulate_with_steady_state():
    # TODO
    pass


def pearson_correlation(simulation_data, correlation_threshold=0.5):
    """
    Calculate the Pearson correlation of the simulation results and identify important species

    Args:
        - simulation_data: Results of the roadrunner simulation
        - correlation_threshold: Threshold to consider a correlation as significant (default: 0.5)
    """
    # Take the columns names
    cols_names = simulation_data.colnames

    # Convert the simulation data into numpy array
    numpy_sim_data = np.array(simulation_data)

    species_data = numpy_sim_data[:, 1:]
    species_names = cols_names[1:]

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(species_data, rowvar=False)

    return correlation_matrix
