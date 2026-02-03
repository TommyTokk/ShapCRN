import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr

import pandas as pd

import os


import json
import numpy as np


def parse_args():
    """
    Parse command line arguments for SBML model analysis tool.

    Sets up argument parsing with multiple subcommands for model simulation,
    sensitivity analysis, knockout/knockin operations, and network visualization.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the selected subcommand and
        all associated options.
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
        "-si",
        "--save-images",
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
        "-pf",
        "--payoff-function",
        choices=["max", "min", "last"],
        default="last",
        help="Function to use to calculate the payoff in the Shapley value",
    )

    simulate_samples_parser.add_argument(
        "-n",
        "--num-samples",
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
        "--use-fixed-perturbations",
        action="store_true",
        default=False,
        help="Run the analysis using fixed perturbations",
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
        "-si",
        "--save-images",
        default="./imgs",
        help="Output directory for plots",
    )

    simulate_samples_parser.add_argument(
        "-gr",
        "--generate-report",
        default="./report",
        help="Output directory for reports",
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

    # === KNOCKIN SPECIES command ===

    knockin_species_parser = subparsers.add_parser(
        "knockin_species", help="Knockin a species in the model"
    )

    knockin_species_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    knockin_species_parser.add_argument(
        "target_species_id", help="ID of the species to knockin"
    )

    # === KNOCKIN REACTION command ===
    knockin_reaction_parser = subparsers.add_parser(
        "knockin_reaction", help="Knockin a reaction in the model"
    )

    knockin_reaction_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )

    knockin_reaction_parser.add_argument(
        "target_reaction_id", help="ID of the reaction to knockin"
    )

    # === CREATE PETRINET command ==
    create_network_parser = subparsers.add_parser(
        "create_network", help="Create the Network of the given model"
    )

    create_network_parser.add_argument("input_path", help="Path to the SBML model")

    create_network_parser.add_argument(
        "-o",
        "--output",
        default="./imgs/PetriNets",
        help="Output directory for plots (default: ./imgs/PetriNets)",
    )

    create_network_parser.add_argument(
        "-sd",
        "--save-dot",
        default=None,
        help="directory where to save the dot code for the network",
    )

    create_network_parser.add_argument(
        "-or",
        "--orientation",
        choices=["TB", "BT", "LR", "RL"],
        default="TB",
        help="Orientation to use for the network: TB -> Top-Bottom, BT -> Bottom-Top, LR -> Left-Right, RL -> Right-Left",
    )

    create_network_parser.add_argument(
        "-l",
        "--layout",
        choices=["dot", "neato", "fdp", "sfdp"],
        default="dot",
        help="Layout used to plot the network",
    )

    create_network_parser.add_argument(
        "-vs",
        "--vertical-spacing",
        type=float,
        default=0.5,
        help="Vertical spacing between ranks",
    )

    create_network_parser.add_argument(
        "-hs",
        "--horizontal-spacing",
        type=float,
        default=0.3,
        help="Horizontal spacing between nodes",
    )

    # Common arguments for all commands
    for subparser in [
        simulate_parser,
        simulate_samples_parser,
        knockout_species_parser,
        knockout_reaction_parser,
    ]:
        subparser.add_argument("-l", "--log", help="Path to log file")

    return parser.parse_args()


def print_log(log_file: str | None, string):
    """
    Log a message with timestamp to file or stdout.

    Parameters
    ----------
    log_file : str or None
        Path to log file. If None, prints to stdout instead.
    string : str
        Message to log.

    Returns
    -------
    None
    """
    current_date = datetime.datetime.now()
    if log_file:
        with open(log_file, "a") as out:
            out.write(f"[{current_date}]: {string}\n")
    else:
        print(f"[{current_date}]: {string}")


def dict_pretty_print(dict_obj):
    """
    Pretty print a dictionary as formatted JSON.

    Parameters
    ----------
    dict_obj : dict
        Dictionary to be printed.

    Returns
    -------
    None
    """

    json_formatted_str = json.dumps(dict_obj, indent=2)
    print(json_formatted_str)


# === DEBUG ===
#
#
#


# === SHAPLEY VALUE ===


def payoff_last(sim_data):
    return sim_data.tail(1)


def payoff_max(sim_data):
    return sim_data.max(axis=0).to_frame().T


def payoff_min(sim_data):
    return sim_data.min(axis=0).to_frame().T


# KEEP
def save_shapley_values_to_csv_pivot(shapley_vals, file_path, cols=None, log_file=None):
    """
    Save Shapley values to a CSV file in pivot table format (knockout species as rows, species as columns).

    Args:
        shapley_vals: DataFrame with the shapley values
        file_path: Path where to save the CSV file
        cols: Optional list of target species to include as columns (if None, use all available)
        log_file: Optional log file for messages

    Returns:
        str: Path of the saved CSV file
    """

    if not os.path.isdir(file_path):
        os.makedirs(file_path, exist_ok=True)

    # Saving
    shapley_vals.to_csv(f"{file_path}/shap.csv")


# === ANALYSIS ===


# KEEP
def pearson_correlation(
    matrix_1, matrix_2, permutations=999, seed=None, alternative="two-sided"
):
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of 'two-sided', 'greater', 'less'")

    # Align: prendiamo l'intersezione di righe e colonne
    if not isinstance(matrix_1, pd.DataFrame) or not isinstance(matrix_2, pd.DataFrame):
        raise TypeError("Both inputs must be pandas.DataFrame")

    common_index = matrix_1.index.intersection(matrix_2.index)
    common_cols = matrix_1.columns.intersection(matrix_2.columns)

    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError("No overlapping index/columns between the two DataFrames")

    a = matrix_1.loc[common_index, common_cols].to_numpy().ravel()
    b = matrix_2.loc[common_index, common_cols].to_numpy().ravel()

    # Keep only pairs where neither is NaN
    valid = ~np.isnan(a) & ~np.isnan(b)
    a_valid = a[valid]
    b_valid = b[valid]

    # If not enough data, return NaN
    if a_valid.size < 2:
        return np.nan, np.nan

    # compute observed Pearson r (handle caso di input costanti)
    try:
        r_obs = pearsonr(a_valid, b_valid)[0]
    except Exception:
        # ad es. ConstantInputWarning / errori numerici -> nessuna correlazione definita
        return np.nan, np.nan

    # Permutation test
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(permutations):
        b_perm = rng.permutation(b_valid)
        try:
            r_perm = pearsonr(a_valid, b_perm)[0]
        except Exception:
            # se la permutazione genera valori costanti, considera r_perm = 0 (o skip)
            r_perm = 0.0

        if alternative == "two-sided":
            if abs(r_perm) >= abs(r_obs):
                count += 1
        elif alternative == "greater":
            if r_perm >= r_obs:
                count += 1
        else:  # "less"
            if r_perm <= r_obs:
                count += 1

    pvalue = (count + 1) / (permutations + 1)
    return r_obs, pvalue


def truncate_small_values(value, threshold=1e-20):
    val = np.float64(value)
    return np.where(np.abs(val) < threshold, 0, val)


def get_active_cells(matrix, log_file=None):
    res = np.where((matrix > 0) & np.isfinite(matrix), 1, 0)

    return res


# KEEP
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


def minMax_normalize(df, epsilon=1e-20, log_file=None):
    try:
        global_min = df.min().min()
        global_max = df.max().max()

        if np.isclose(global_max, global_min, atol=epsilon):
            normalized_df = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        else:
            normalized_df = (df - global_min) / (global_max - global_min)
            normalized_df = normalized_df.clip(0.0, 1.0)

        return normalized_df

    except Exception as e:
        # Log opzionale
        msg = f"[ERROR] error during minMax normalization: {e}"
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        else:
            print(msg)
        raise  # rilancia l'eccezione


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
