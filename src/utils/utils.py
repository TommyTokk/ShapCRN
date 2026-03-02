import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr

import pandas as pd

import os


import json
import numpy as np



def setup_output_dirs(output_root: str, model_name: str) -> dict:
    """
    Create a standardised output directory tree and return the paths.

    Structure::

        <output_root>/
        └── <model_name>/
            ├── images/
            ├── csv/
            └── reports/

    Parameters
    ----------
    output_root : str
        Root output directory (``args.output``).
    model_name : str
        Model name (typically the SBML filename without extension).

    Returns
    -------
    dict
        Keys: ``'root'``, ``'images'``, ``'csv'``, ``'reports'``.
    """
    base = os.path.join(output_root, model_name)
    dirs = {
        "root": base,
        "images": os.path.join(base, "images"),
        "csv": os.path.join(base, "csv"),
        "reports": os.path.join(base, "reports"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def _add_output_arg(parser, default="./results"):
    """Add the single --output root directory argument to a parser."""
    parser.add_argument(
        "-o", "--output", default=default,
        help=f"Root output directory (default: {default}). "
             "Sub-folders (images/, csv/, reports/) are created "
             "automatically under <output>/<model_name>/.",
    )


def _add_simulation_args(parser):
    """Add common simulation arguments (time, integrator) to a parser."""
    parser.add_argument(
        "-t", "--time", type=float, default=10,
        help="Simulation end time (default: 10)",
    )
    parser.add_argument(
        "-i", "--integrator", choices=["cvode", "gillespie", "rk4"],
        help="Integrator to use",
    )


def _add_steady_state_args(parser):
    """Add steady-state related arguments to a parser."""
    parser.add_argument(
        "--steady-state", action="store_true",
        help="Simulate until steady state is reached",
    )
    parser.add_argument(
        "--max-time", type=float, default=1000,
        help="Maximum simulation time when seeking steady state (default: 1000)",
    )
    parser.add_argument(
        "--sim-step", type=float, default=5,
        help="Time step for steady state check (default: 5)",
    )
    parser.add_argument(
        "--points", type=int, default=1000,
        help="Number of points in the final profile (default: 1000)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-6,
        help="Threshold for steady state detection (default: 1e-6)",
    )


def _add_perturbation_args(parser):
    """Add perturbation-related arguments to a parser."""
    parser.add_argument(
        "--preserve-inputs", action="store_true", default=False,
        help="Preserve the input nodes from being analysed",
    )
    parser.add_argument(
        "--fixed-perturbations", nargs="+",
        help="Perturbation percentages to use in fixed samples combination "
             "(WARNING: The number of samples will be equal to the number of parameters)",
    )


def _add_input_species_args(parser):
    """Add input species argument to a parser."""
    parser.add_argument(
        "--input-species", nargs="+", default=None,
        help="One or more species IDs to vary (e.g. --input-species ACEx GLCx P). "
             "If None no samples will be generated",
    )


def _build_simulate_parser(subparsers):
    """Build the 'simulate' subcommand parser."""
    parser = subparsers.add_parser("simulate", help="Simulate an SBML model")
    parser.add_argument("input_path", help="Path to the SBML model file")
    _add_simulation_args(parser)
    _add_output_arg(parser)
    _add_steady_state_args(parser)
    parser.add_argument(
        "--interactive", action="store_true",
        help="Generate an interactive plot instead of a static one",
    )
    parser.add_argument("-l", "--log", help="Path to log file")
    return parser


def _build_importance_assessment_parser(subparsers):
    """Build the 'importance_assessment' subcommand parser."""
    parser = subparsers.add_parser(
        "importance_assessment",
        help="Simulate model with different input species concentrations",
    )

    # -- Positional / Model --
    parser.add_argument("input_path", help="Path to the SBML model file")
    parser.add_argument(
        "--operation", choices=["knockout", "knockin"], default="knockout",
        help="Type of operation to perform on species (default: knockout)",
    )

    # -- Species selection --
    species_group = parser.add_argument_group("Species selection")
    _add_input_species_args(species_group)
    species_group.add_argument(
        "--knocked", nargs="+", default=None,
        help="One or more species IDs to knock (out or in), if empty all species are used",
    )
    species_group.add_argument(
        "--target-nodes", nargs="+", default=None,
        help="One or more IDs to print on analysis",
    )
    species_group.add_argument(
        "--preserve-inputs", action="store_true", default=False,
        help="Preserve the input nodes from being analysed",
    )

    # -- Sampling & perturbation --
    perturbation_group = parser.add_argument_group("Sampling & perturbation")
    perturbation_group.add_argument(
        "-n", "--num-samples", type=int, default=5,
        help="Number of samples for each species (default: 5)",
    )
    perturbation_group.add_argument(
        "-v", "--variation", type=float, default=20.0,
        help="Percentage variation around initial value (default: 20.0)",
    )
    perturbation_group.add_argument(
        "--use-perturbations", action="store_true", default=False,
        help="Run the analysis using inputs' perturbations",
    )
    perturbation_group.add_argument(
        "--use-fixed-perturbations", action="store_true", default=False,
        help="Run the analysis using fixed perturbations",
    )
    perturbation_group.add_argument(
        "--fixed-perturbations", nargs="+",
        help="Perturbation percentages to use in fixed samples combination "
             "(WARNING: The number of samples will be equal to the number of parameters)",
    )
    perturbation_group.add_argument(
        "--perturbations-importance", action="store_true", default=False,
        help="Run the analysis on the importance of using perturbations for the model",
    )
    perturbation_group.add_argument(
        "--random-perturbations-importance", action="store_true", default=False,
        help="Run analysis on the importance of random perturbations for the model",
    )

    # -- Shapley value --
    shapley_group = parser.add_argument_group("Shapley value")
    shapley_group.add_argument(
        "--payoff-function", choices=["max", "min", "last"], default="last",
        help="Function to use to calculate the payoff in the Shapley value",
    )

    # -- Simulation --
    sim_group = parser.add_argument_group("Simulation")
    _add_simulation_args(sim_group)
    _add_steady_state_args(sim_group)

    # -- Output --
    output_group = parser.add_argument_group("Output")
    _add_output_arg(output_group)
    output_group.add_argument("-l", "--log", help="Path to log file")

    return parser


def _build_sensitivity_analysis_parser(subparsers):
    """Build the 'sensitivity_analysis' subcommand parser."""
    parser = subparsers.add_parser(
        "sensitivity_analysis",
        help="Run sensitivity analysis between fixed and random perturbations",
    )
    parser.add_argument("input_path", help="Path to the SBML model file")
    _add_input_species_args(parser)
    parser.add_argument(
        "--base-samples", type=float, default=4096,
        help="Base samples size used to run SOBOL analysis with SALib",
    )
    _add_perturbation_args(parser)
    parser.add_argument(
        "--check-convergence", action="store_true", default=False,
        help="Check convergence with increasing samples (MAX: 4096)",
    )
    return parser


def _build_knockout_species_parser(subparsers):
    """Build the 'knockout_species' subcommand parser."""
    parser = subparsers.add_parser(
        "knockout_species", help="Knockout a species in the model",
    )
    parser.add_argument("input_path", help="Path to the SBML model file")
    parser.add_argument("species_id", help="ID of the species to inhibit")
    _add_output_arg(parser)
    parser.add_argument("-l", "--log", help="Path to log file")
    return parser


def _build_knockout_reaction_parser(subparsers):
    """Build the 'knockout_reaction' subcommand parser."""
    parser = subparsers.add_parser(
        "knockout_reaction", help="Knockout a reaction in the model",
    )
    parser.add_argument("input_path", help="Path to the SBML model file")
    parser.add_argument("reaction_id", help="ID of the reaction to inhibit")
    _add_output_arg(parser)
    parser.add_argument("-l", "--log", help="Path to log file")
    return parser


def _build_knockin_species_parser(subparsers):
    """Build the 'knockin_species' subcommand parser."""
    parser = subparsers.add_parser(
        "knockin_species", help="Knockin a species in the model",
    )
    parser.add_argument("input_path", help="Path to the SBML model file")
    parser.add_argument("target_species_id", help="ID of the species to knockin")
    _add_output_arg(parser)
    parser.add_argument("-l", "--log", help="Path to log file")
    return parser


def _build_knockin_reaction_parser(subparsers):
    """Build the 'knockin_reaction' subcommand parser."""
    parser = subparsers.add_parser(
        "knockin_reaction", help="Knockin a reaction in the model",
    )
    parser.add_argument("input_path", help="Path to the SBML model file")
    parser.add_argument("target_reaction_id", help="ID of the reaction to knockin")
    _add_output_arg(parser)
    parser.add_argument("-l", "--log", help="Path to log file")
    return parser


def _build_create_network_parser(subparsers):
    """Build the 'create_network' subcommand parser."""
    parser = subparsers.add_parser(
        "create_network", help="Create the Network of the given model",
    )
    parser.add_argument("input_path", help="Path to the SBML model")
    _add_output_arg(parser)
    parser.add_argument(
        "--save-dot", default=None,
        help="Directory to save the DOT source code for the network",
    )
    parser.add_argument(
        "-or", "--orientation", choices=["TB", "BT", "LR", "RL"], default="TB",
        help="Orientation to use for the network: TB -> Top-Bottom, "
             "BT -> Bottom-Top, LR -> Left-Right, RL -> Right-Left",
    )
    parser.add_argument(
        "-l", "--layout", choices=["dot", "neato", "fdp", "sfdp"], default="dot",
        help="Layout used to plot the network",
    )
    parser.add_argument(
        "-vs", "--vertical-spacing", type=float, default=0.5,
        help="Vertical spacing between ranks",
    )
    parser.add_argument(
        "-hs", "--horizontal-spacing", type=float, default=0.3,
        help="Horizontal spacing between nodes",
    )
    return parser


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

    parser = argparse.ArgumentParser(
        description="SBML model analysis and manipulation tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    _build_simulate_parser(subparsers)
    _build_importance_assessment_parser(subparsers)
    _build_sensitivity_analysis_parser(subparsers)
    _build_knockout_species_parser(subparsers)
    _build_knockout_reaction_parser(subparsers)
    _build_knockin_species_parser(subparsers)
    _build_knockin_reaction_parser(subparsers)
    _build_create_network_parser(subparsers)

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


def truncate_small_values(value, threshold=1e-20):
    val = np.float64(value)
    return np.where(np.abs(val) < threshold, 0, val)


def get_active_cells(matrix, log_file=None):
    res = np.where((matrix > 0) & np.isfinite(matrix), 1, 0)

    return res


# === NORMALIZATION ===
def normalize_asinh(df, scale=None):
    is_dataframe = isinstance(df, pd.DataFrame)
    vals = df.to_numpy(dtype=float) if is_dataframe else np.asarray(df, dtype=float)
    nan_mask = ~np.isfinite(vals)
    abs_vals = np.abs(vals[np.isfinite(vals)])

    if scale is None:
        s = np.nanpercentile(abs_vals, 75) if abs_vals.size > 0 else 1.0
    else:
        s = float(scale)

    if not np.isfinite(s) or s <= 0:
        s = 1.0

    normalized = np.arcsinh(vals / s)
    normalized[nan_mask] = np.nan

    if is_dataframe:
        normalized = pd.DataFrame(normalized, index=df.index, columns=df.columns)

    return normalized, s


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
