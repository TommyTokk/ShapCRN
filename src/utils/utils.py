import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr

import pandas as pd

import os


import json
import numpy as np


def parse_args():
    """
    Parse command line arguments using a subcommand structure.

    Returns:
        argparse.Namespace: Object containing all parsed arguments
    """
    import argparse

    # Create main parser
    parser = argparse.ArgumentParser(
        description="SBML model analysis and manipulation tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # === SIMULATE command ===
    simulate_parser = subparsers.add_parser("simulate", help="Simulate an SBML model")
    simulate_parser.add_argument("input_path", help="Path to the SBML model file")
    simulate_parser.add_argument(
        "-t", "--time", type=float, default=10, help="Simulation end time (default: 10)"
    )
    simulate_parser.add_argument(
        "-i",
        "--integrator",
        choices=["cvode", "gillespie", "rk4"],
        help="Integrator to use",
    )
    simulate_parser.add_argument(
        "-o",
        "--output",
        default="./imgs",
        help="Output directory for plots (default: ./imgs)",
    )
    # Steady state options for simulate
    simulate_parser.add_argument(
        "--steady-state",
        action="store_true",
        help="Simulate until steady state is reached",
    )
    simulate_parser.add_argument(
        "--max-time",
        type=float,
        default=1000,
        help="Maximum simulation time when seeking steady state (default: 1000)",
    )
    simulate_parser.add_argument(
        "--sim-step",
        type=float,
        default=5,
        help="Time step for steady state check (default: 5)",
    )
    simulate_parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of points in the final profile (default: 1000)",
    )
    simulate_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for steady state detection (default: 1e-12)",
    )

    simulate_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate an intereactive plot instead of a static one",
    )

    # === SIMULATE_SAMPLES command ===
    simulate_samples_parser = subparsers.add_parser(
        "importance_assessment",
        help="Simulate model with different input species concentrations",
    )
    simulate_samples_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    simulate_samples_parser.add_argument(
        "-is",
        "--input_species",
        nargs="+",
        default=None,
        help="One or more species IDs to vary (e.g. -s ACEx GLCx P). If None no samples will be generated",
    )
    simulate_samples_parser.add_argument(
        "-ko",
        "--knockout",
        nargs="+",
        default=None,
        help="One or more IDs to knockout, if empty all species are used",
    )

    simulate_samples_parser.add_argument(
        "-tn",
        "--target-nodes",
        nargs="+",
        default=None,
        help="One or more IDs to print on analysis",
    )
    # simulate_samples_parser.add_argument(
    #     "--mko",
    #     "--multi_ko",
    #     nargs="+",
    #     default=None,
    #     help="Multiple species to Knockout in the same model",
    # )
    simulate_samples_parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples for each species (default: 5)",
    )
    simulate_samples_parser.add_argument(
        "-v",
        "--variation",
        type=float,
        default=20.0,
        help="Percentage variation around initial value (default: 20.0)",
    )
    simulate_samples_parser.add_argument(
        "-t", "--time", type=float, default=10, help="Simulation end time (default: 10)"
    )
    simulate_samples_parser.add_argument(
        "-i",
        "--integrator",
        choices=["cvode", "gillespie", "rk4"],
        help="Integrator to use",
    )

    simulate_samples_parser.add_argument(
        "--preserve-inputs",
        action="store_true",
        default=False,
        help="Preserve the inputs node from been analysed",
    )
    simulate_samples_parser.add_argument(
        "--use-perturbations",
        action="store_true",
        default=False,
        help="Run the analysis using inputs' perturbations",
    )
    simulate_samples_parser.add_argument(
        "-fp",
        "--fixed-perturbations",
        nargs="+",
        help="Perturbation percentages to use in fixed samples combination (WARNING: The number of samples will be equal to the number of parameters)",
    )
    simulate_samples_parser.add_argument(
        "--perturbations-importance",
        action="store_true",
        default=False,
        help="Run the analysis on the importance of using perturbations for the model",
    )

    simulate_samples_parser.add_argument(
        "--random-perturbations-importance",
        action="store_true",
        default=False,
        help="Run analysis on the importance of random perturbations for the model",
    )

    # Steady state options for simulate_samples
    simulate_samples_parser.add_argument(
        "--steady-state",
        action="store_true",
        help="Simulate until steady state is reached",
    )

    simulate_samples_parser.add_argument(
        "--max-time",
        type=float,
        default=1000,
        help="Maximum simulation time when seeking steady state (default: 1000)",
    )
    simulate_samples_parser.add_argument(
        "--sim-step",
        type=float,
        default=5,
        help="Time step for steady state check (default: 5)",
    )
    simulate_samples_parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of points in the final profile (default: 1000)",
    )
    simulate_samples_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for steady state detection (default: 1e-6)",
    )

    simulate_samples_parser.add_argument(
        "-o",
        "--output",
        default="./imgs/samples",
        help="Output directory for plots (default: ./imgs/samples)",
    )

    simulate_samples_parser.add_argument(
        "-so",
        "--save-output",
        action="store_true",
        help="Save importance analysis results",
    )

    # === SENSITIVITY_ANALYSIS command ===
    sens_parser = subparsers.add_parser(
        "sensitivity_analysis",
        help="Run sensitivity analysis between fixed and random perturbations",
    )
    sens_parser.add_argument("input_path", help="Path to the SBML model file")

    sens_parser.add_argument(
        "-is",
        "--input_species",
        nargs="+",
        default=None,
        help="One or more species IDs to vary (e.g. -s ACEx GLCx P). If None no samples will be generated",
    )

    sens_parser.add_argument(
        "-bs",
        "--base-samples",
        type=float,
        default=4096,
        help="Base samples size used to run SOBOL analysis with SALib",
    )

    sens_parser.add_argument(
        "--preserve-inputs",
        action="store_true",
        default=False,
        help="Ignore the input nodes for the analysis",
    )

    sens_parser.add_argument(
        "-fp",
        "--fixed-perturbations",
        nargs="+",
        help="Perturbation percentages to use in fixed samples combination (WARNING: The number of samples will be equal to the number of parameters)",
    )

    sens_parser.add_argument(
        "-cc",
        "--check-convergence",
        action="store_true",
        default=False,
        help="Check convergence with increasing samples (MAX: 4096)",
    )

    # === KNOCKOUT_SPECIES command ===
    knockout_species_parser = subparsers.add_parser(
        "knockout_species", help="Knockout a species in the model"
    )
    knockout_species_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    knockout_species_parser.add_argument(
        "species_id", help="ID of the species to inhibit"
    )
    knockout_species_parser.add_argument(
        "-o",
        "--output",
        default="./models",
        help="Output directory for plots (default: ./models)",
    )

    # === KNOCKOUT_REACTION command ===
    knockout_reaction_parser = subparsers.add_parser(
        "knockout_reaction", help="Knockout a reaction in the model"
    )
    knockout_reaction_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    knockout_reaction_parser.add_argument(
        "reaction_id", help="ID of the reaction to inhibit"
    )
    knockout_reaction_parser.add_argument(
        "-o",
        "--output",
        default="./imgs",
        help="Output directory for plots (default: ./imgs)",
    )

    # === CREATE PETRINET command ==
    # create_petrinet_parser = subparsers.add_parser(
    #     "create_petrinet", help="Create the PetriNet of the given model"
    # )
    #
    # create_petrinet_parser.add_argument("input_path", help="Path to the SBML model")
    #
    # create_petrinet_parser.add_argument(
    #     "-o",
    #     "--output",
    #     default="./imgs/PetriNets",
    #     help="Output directory for plots (default: ./imgs/PetriNets)",
    # )
    #
    # create_petrinet_parser.add_argument(
    #     "-tn", "--tests_number", default=10, help="Number of tests to perform"
    # )
    #
    # create_petrinet_parser.add_argument("-iq", "--increase_quantity", default=5)

    # Common arguments for all commands
    for subparser in [
        simulate_parser,
        simulate_samples_parser,
        knockout_species_parser,
        knockout_reaction_parser,
    ]:
        subparser.add_argument("-l", "--log", help="Path to log file")

    return parser.parse_args()


def print_log(log_file, string):
    current_date = datetime.datetime.now()
    if log_file:
        with open(log_file, "a") as out:
            out.write(f"[{current_date}]: {string}\n")
    else:
        print(f"[{current_date}]: {string}")


def dict_pretty_print(dict_obj):
    """Pretty print a dictionary as formatted JSON.

    Args:
        dict_obj: Dictionary to be printed
    """

    json_formatted_str = json.dumps(dict_obj, indent=2)
    print(json_formatted_str)


def pretty_print_variations(variations_dict, precision=4, show_zero=False):
    """
    Pretty print the variations dictionary in a readable format.

    Args:
        variations_dict: Dict returned by get_simulations_variations
        precision: Number of decimal places to show (default: 4)
        show_zero: Whether to show variations that are zero (default: False)
    """
    if not variations_dict:
        print("No variations to display.")
        return

    print("=" * 80)
    print("SIMULATION VARIATIONS REPORT")
    print("=" * 80)

    for ko_species in sorted(variations_dict.keys()):
        print(f"\n🧬 KNOCKED OUT SPECIES: {ko_species}")
        print("-" * 60)

        combinations = variations_dict[ko_species]
        if not combinations:
            print("  No combinations found.")
            continue

        for combination in sorted(combinations.keys()):
            print(f"\n  📊 Combination: {combination}")
            print("  " + "─" * 50)

            species_variations = combinations[combination]
            if not species_variations:
                print("    No species variations found.")
                continue

            # Sort species by absolute variation value (largest first)
            sorted_species = sorted(
                species_variations.items(),
                key=lambda x: abs(x[1]["variation"]),
                reverse=True,
            )

            displayed_any = False
            for species, data in sorted_species:
                variation = data["variation"]

                # Skip zero variations if show_zero is False
                if not show_zero and abs(variation) < 1e-10:
                    continue

                # Format the variation with appropriate sign and color indicators
                if variation > 0:
                    sign_indicator = "↑"
                    variation_str = f"+{variation:.{precision}f}"
                elif variation < 0:
                    sign_indicator = "↓"
                    variation_str = f"{variation:.{precision}f}"
                else:
                    sign_indicator = "="
                    variation_str = f"{variation:.{precision}f}"

                print(f"    {sign_indicator} {species:<20} {variation_str:>12}")
                displayed_any = True

            if not displayed_any:
                print("    (All variations are zero)")

    print("\n" + "=" * 80)
    print("Legend: ↑ = Increase, ↓ = Decrease, = = No change")
    print("=" * 80)


# === DEBUG ===


def save_shapley_values_to_csv_pivot(shapley_dict, file_path, cols=None, log_file=None):
    """
    Save Shapley values to a CSV file in pivot table format (knockout species as rows, species as columns).

    Args:
        shapley_dict: Dictionary with structure:
                     {ko_species: {species: {"shap": value}}}
        file_path: Path where to save the CSV file
        cols: Optional list of target species to include as columns (if None, use all available)
        log_file: Optional log file for messages

    Returns:
        str: Path of the saved CSV file
    """
    try:
        # Debug: analyze the structure of shapley_dict
        print_log(log_file, "=== SHAPLEY VALUES STRUCTURE DEBUG ===")

        all_ko_species = list(shapley_dict.keys())
        all_available_species = set()

        # Collect all available species across all ko_species
        for ko_species, ko_info in shapley_dict.items():
            species = list(ko_info.keys())
            all_available_species.update(species)

        all_available_species = sorted(list(all_available_species))
        print_log(
            log_file, f"Total unique species available: {len(all_available_species)}"
        )
        print_log(log_file, f"Total ko species: {len(all_ko_species)}")

        # Determine which species to use as columns
        if cols is not None:
            # Filter to only include specified columns that exist in the data
            available_cols = [col for col in cols if col in all_available_species]
            missing_cols = [col for col in cols if col not in all_available_species]

            if missing_cols:
                print_log(
                    log_file,
                    f"Warning: Specified columns not found in data: {missing_cols}",
                )

            if available_cols:
                species_to_use = available_cols
                print_log(
                    log_file, f"Using specified columns: {len(species_to_use)} species"
                )
                print_log(log_file, f"Specified species: {species_to_use}")
            else:
                print_log(
                    log_file,
                    "Warning: None of the specified columns found, using all available",
                )
                species_to_use = all_available_species
        else:
            species_to_use = all_available_species
            print_log(log_file, f"Using all available species: {len(species_to_use)}")

        # Create lists to store the data with consistent structure
        ko_species_list = []
        species_list = []
        shapley_values = []

        # Extract data ensuring all combinations are represented
        for ko_species in all_ko_species:
            ko_info = shapley_dict[ko_species]

            for species in species_to_use:
                ko_species_list.append(ko_species)
                species_list.append(species)

                # Check if this species exists for this ko_species
                if species in ko_info:
                    shap_value = ko_info[species]
                    # Handle special cases
                    if shap_value is None:
                        shapley_values.append(np.nan)
                    elif isinstance(shap_value, (int, float)) and np.isnan(shap_value):
                        shapley_values.append(np.nan)
                    else:
                        shapley_values.append(float(shap_value))
                else:
                    # Species not present for this ko_species
                    shapley_values.append(np.nan)
                    print_log(
                        log_file,
                        f"Missing data for ko_species='{ko_species}', species='{species}'",
                    )

        # Verify all lists have the same length
        expected_length = len(all_ko_species) * len(species_to_use)
        print_log(log_file, f"Expected length: {expected_length}")
        print_log(log_file, f"ko_species_list length: {len(ko_species_list)}")
        print_log(log_file, f"species_list length: {len(species_list)}")
        print_log(log_file, f"shapley_values length: {len(shapley_values)}")

        if not (
            len(ko_species_list)
            == len(species_list)
            == len(shapley_values)
            == expected_length
        ):
            raise ValueError(
                f"Inconsistent data structure: ko={len(ko_species_list)}, species={len(species_list)}, values={len(shapley_values)}, expected={expected_length}"
            )

        # Create DataFrame in long format
        df = pd.DataFrame(
            {
                "knockout_species": ko_species_list,
                "species": species_list,
                "shapley_value": shapley_values,
            }
        )

        print_log(log_file, f"Created DataFrame with shape: {df.shape}")
        print_log(log_file, f"DataFrame columns: {list(df.columns)}")

        # Create pivot table
        pivot_df = df.pivot(
            index="knockout_species", columns="species", values="shapley_value"
        )

        print_log(log_file, f"Created pivot table with shape: {pivot_df.shape}")
        print_log(log_file, f"Pivot table columns: {list(pivot_df.columns)}")

        # Fill NaN values with 0 or keep them as NaN (you can choose)
        # pivot_df = pivot_df.fillna(0)  # Uncomment if you want to fill NaN with 0

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save to CSV
        pivot_df.to_csv(f"{file_path}/shap.csv", float_format="%.10f")

        print_log(log_file, f"Shapley values (pivot format) saved to: {file_path}")
        print_log(
            log_file,
            f"Final CSV contains {len(pivot_df.index)} ko_species and {len(pivot_df.columns)} target_species",
        )

        return file_path

    except Exception as e:
        error_msg = f"Error saving Shapley values (pivot) to CSV: {e}"
        print_log(log_file, error_msg)

        # Additional debug information
        print_log(log_file, "=== DEBUG INFO ===")
        if "shapley_dict" in locals():
            print_log(log_file, f"shapley_dict type: {type(shapley_dict)}")
            if isinstance(shapley_dict, dict):
                print_log(
                    log_file, f"shapley_dict keys: {list(shapley_dict.keys())[:3]}..."
                )  # Show first 3
                if shapley_dict:
                    first_ko = list(shapley_dict.keys())[0]
                    print_log(
                        log_file,
                        f"First ko_species '{first_ko}' data type: {type(shapley_dict[first_ko])}",
                    )
                    if isinstance(shapley_dict[first_ko], dict):
                        print_log(
                            log_file,
                            f"First ko_species targets: {list(shapley_dict[first_ko].keys())[:3]}...",
                        )

        if "cols" in locals():
            print_log(log_file, f"cols parameter: {cols}")

        raise Exception(error_msg)


# === ANALYSIS ===


def pearson_correlation(
    matrix_1, matrix_2, permutations=999, seed=None, alternative="two-sided"
):
    rng = np.random.RandomState(seed)
    a = matrix_1.ravel()
    b = matrix_2.ravel()

    valid = ~np.isnan(a) & ~np.isnan(b)
    a_valid, b_valid = a[valid], b[valid]

    r_obs = pearsonr(a_valid, b_valid)[0]

    count = 0
    for _ in range(permutations):
        b_perm = rng.permutation(b_valid)
        r_perm = pearsonr(a_valid, b_perm)[0]

        # Correct logic based on alternative hypothesis
        if alternative == "two-sided":
            if abs(r_perm) >= abs(r_obs):  # pyright:ignore
                count += 1
        elif alternative == "greater":
            if r_perm >= r_obs:  # pyright:ignore
                count += 1
        elif alternative == "less":
            if r_perm <= r_obs:  # pyright:ignore
                count += 1

    pvalue = (count + 1) / (permutations + 1)
    return r_obs, pvalue


def truncate_small_values(value, threshold=1e-20):
    val = np.float64(value)
    return np.where(np.abs(val) < threshold, 0, val)


def get_active_cells(matrix, log_file=None):

    res = np.where((matrix > 0) & np.isfinite(matrix), 1, 0)

    return res


def get_ko_species_importance(matrix, ko_species_list, log_file=None):
    ko_impact = np.nanmean(matrix, axis=1)
    ko_ranking = np.argsort(ko_impact)[::-1]

    # print_log(log_file, "Top 5 knockouts most sensitive to parameter uncertainty:")
    # for i in range(min(5, len(ko_ranking))):
    #     ko_idx = ko_ranking[i]
    #     ko_name = ko_species_list[ko_idx]
    #     impact_score = ko_impact[ko_idx]
    #     print_log(log_file, f"  {i+1}. {ko_name}: {impact_score:.20f}")

    return ko_impact, ko_ranking


# === NORMALIZATION ===


def minMax_normalize(heatmap_data, log_file=None):

    try:
        global_min = np.nanmax(heatmap_data)
        global_max = np.nanmin(heatmap_data)

        if global_max == global_min:
            return np.zeros_like(heatmap_data)

        if global_max - global_min > 1e-20:
            heatmap_data = (heatmap_data - global_min) / (global_max - global_min)

        return np.clip(heatmap_data, 0.0, 1.0)
    except Exception as e:
        print_log(log_file, f"[ERROR] error during minMax normalization: {e}")
        exit(1)


def z_score_normalize(heatmap_data, log_file=None):

    means = np.mean(heatmap_data, axis=0)
    stds = np.std(heatmap_data, axis=0, ddof=0)

    stds_safe = np.where(stds == 0, 1, stds)

    z_scores = (heatmap_data - means) / 3 * stds_safe

    if log_file:
        log_file.write(f"Column means: {means}\n")
        log_file.write(f"Column stds: {stds}\n")

    return z_scores


def frobenius_norm(matrix, ignore_nan=True):
    """
    Calculate Frobenius norm with robust NaN handling

    Parameters:
    matrix: numpy array
    ignore_nan: if True, ignore NaN values; if False, return NaN if any NaN present
    """

    if not ignore_nan and np.any(np.isnan(matrix)):
        return np.nan

    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(matrix)

    if not np.any(valid_mask):
        # All values are NaN
        return np.nan

    # Calculate Frobenius norm only for valid values
    valid_values = matrix[valid_mask]

    return np.linalg.norm(valid_values)
