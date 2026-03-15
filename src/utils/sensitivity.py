import multiprocessing
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import roadrunner as rr
from SALib.sample import sobol as sobol_sample
from scipy import stats

from src.utils.utils import print_log


def _simulation_worker(args):
    """
    Worker function for parallel simulation execution.

    Parameters
    ----------
    args : tuple
        (index, param, sbml_string, valid_elements, valid_idxs,
         input_ids, current_selections, sim_end_time)

    Returns
    -------
    tuple
        (index, result_array) where result_array contains final concentrations.
    """
    i, param, sbml_string, valid_elements, valid_idxs, input_ids, current_selections, sim_end_time = args

    rr_local = rr.RoadRunner(sbml_string)
    rr_local.timeCourseSelections = current_selections
    rr_local.reset()

    for j, sp_id in enumerate(input_ids):
        rr_local.setInitConcentration(sp_id, param[j])

    rr_local.timeCourseSelections = current_selections

    sim_res = rr_local.simulate(0, sim_end_time)

    result = np.zeros(len(valid_elements))
    for j, el in enumerate(valid_elements):
        result[j] = sim_res[-1, valid_idxs[el]]

    return (i, result)


def get_problem_parameters(
    sbml_model, n_input_species: int, input_species_ids: list, perturbation_range: int = 20, log_file=None
) -> dict:
    
    """
    Generate problem parameters for Sobol sensitivity analysis with perturbation bounds.

    This function creates a parameter dictionary compatible with SALib's Sobol analysis,
    defining the input variables and their sampling bounds based on percentage perturbations
    around baseline concentrations.

    Parameters
    ----------
    sbml_model : libsbml.Model
        SBML model containing species definitions with initial concentrations.
    n_input_species : int
        Number of input species to include in the sensitivity analysis.
        Should match the length of input_species_ids.
    input_species_ids : list of str
        IDs of input species to perturb for sensitivity analysis.
        Each species must exist in the SBML model.
    perturbation_range : int, optional
        Percentage range for perturbation bounds around baseline concentrations,
        by default 20 (i.e., ±20%)
        Example: perturbation_range=20 creates bounds [conc*0.8, conc*1.2]
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    dict
        SALib-compatible problem dictionary with keys:
        - 'num_vars' : int
            Number of input variables (equals n_input_species)
        - 'names' : list of str
            Names of input variables (equals input_species_ids)
        - 'bounds' : list of [float, float]
            Lower and upper bounds for each input variable
            Bounds are calculated as: [conc*(1-range/100), conc*(1+range/100)]

    Notes
    -----
    The problem dictionary is designed for use with SALib's Sobol sampling methods:
    - sobol.sample() for generating sample points
    - sobol.analyze() for computing sensitivity indices

    Bounds are symmetric around the initial concentration. For a species with
    initial concentration C and perturbation_range R%:
    - Lower bound = C * (1 - R/100)
    - Upper bound = C * (1 + R/100)

    Examples
    --------
    Create problem for 20% perturbation range:
    >>> problem = get_problem_parameters(
    ...     sbml_model=model,
    ...     n_input_species=3,
    ...     input_species_ids=['Input1', 'Input2', 'Input3'],
    ...     perturbation_range=20
    ... )
    >>> print(problem['bounds'])
    [[0.8, 1.2], [1.6, 2.4], [0.4, 0.6]]  # For initial concs [1.0, 2.0, 0.5]

    Create problem with wider perturbation range:
    >>> problem = get_problem_parameters(
    ...     sbml_model=model,
    ...     n_input_species=2,
    ...     input_species_ids=['S1', 'S2'],
    ...     perturbation_range=50,  # ±50% perturbation
    ...     log_file=log
    ... )

    Use with SALib for Sobol sampling:
    >>> from SALib.sample import sobol
    >>> problem = get_problem_parameters(model, 2, ['S1', 'S2'], perturbation_range=20)
    >>> samples = sobol.sample(problem, 1024)

    See Also
    --------
    SALib.sample.sobol : Generate Sobol sequences for sensitivity analysis
    SALib.analyze.sobol : Analyze Sobol sensitivity indices
    run_simulation_with_params : Execute simulations with generated parameter samples
    """
    problem = {}

    problem["num_vars"] = n_input_species
    problem["names"] = input_species_ids

    tmp = []

    for ins in input_species_ids:
        conc = sbml_model.getSpecies(ins).getInitialConcentration()

        # Check for consistency of the initial concentration value
        if conc == 0.0:
            # Provide a tiny functional bound if naturally zero
            tmp.append([0.0, 1e-6]) 
        else:
            lower_bound = conc * (1 - (perturbation_range / 100))
            upper_bound = conc * (1 + (perturbation_range / 100))
            tmp.append([lower_bound, upper_bound])

    problem["bounds"] = tmp

    return problem


def run_simulation_with_params(
    model: rr.RoadRunner, params: np.ndarray, valid_elements: list, valid_idxs: dict,
    input_ids: list, log_file=None, n_processes: int = None, sim_end_time: float = 5000
) -> np.ndarray:
    """
    Run batch simulations with parameter samples and extract final concentrations.

    This function executes multiple RoadRunner simulations using different parameter sets
    (typically from Sobol sampling), collecting the final steady-state concentrations of
    specified output species. Simulations are parallelized across multiple processes.

    Parameters
    ----------
    model : roadrunner.RoadRunner
        Configured RoadRunner model instance to simulate.
        The model configuration is preserved across simulations.
    params : numpy.ndarray
        2D array of parameter samples to simulate.
        Shape: (n_samples, n_input_species)
        Each row represents one parameter set (input species concentrations).
    valid_elements : list of str
        List of output species IDs whose final concentrations should be extracted.
        These are the species to monitor in the sensitivity analysis.
    valid_idxs : dict
        Mapping from species IDs to their column indices in simulation results.
        Format: {species_id: column_index}
    input_ids : list of str
        IDs of input species whose concentrations will be varied.
        Must match the number of columns in params.
    log_file : file, optional
        File object for logging simulation progress, by default None
    n_processes : int, optional
        Number of parallel processes to use. If None, uses CPU count - 1.
        Default is None.
    sim_end_time : float, optional
        End time for each simulation, by default 5000.

    Returns
    -------
    numpy.ndarray
        2D array of final concentrations for monitored species.
        Shape: (n_samples, n_valid_elements)
        RES[i, j] = final concentration of valid_elements[j] for parameter set i

    Notes
    -----
    - The function uses multiprocessing to parallelize simulations
    - Each simulation runs from time 0 to sim_end_time time units
    - Progress is displayed as "Processed X/Y samples" to stdout
    - The model is reset before each simulation to ensure independence
    - Only the final time point concentration is extracted from each simulation

    The function is typically used with Sobol-sampled parameters for sensitivity analysis,
    where the output matrix RES is then passed to SALib.analyze.sobol().

    Examples
    --------
    Run simulations with Sobol samples:
    >>> from SALib.sample import sobol
    >>> problem = get_problem_parameters(model, 2, ['Input1', 'Input2'], perturbation_range=20)
    >>> params = sobol.sample(problem, 1024)
    >>> 
    >>> valid_elements = ['Output1', 'Output2', 'Output3']
    >>> valid_idxs = {sp: i for i, sp in enumerate(['time'] + valid_elements)}
    >>> 
    >>> results = run_simulation_with_params(
    ...     rr=rr_model,
    ...     params=params,
    ...     valid_elements=valid_elements,
    ...     valid_idxs=valid_idxs,
    ...     input_ids=['Input1', 'Input2'],
    ...     n_processes=4
    ... )
    >>> print(results.shape)  # (1024, 3)

    Use results for Sobol analysis:
    >>> from SALib.analyze import sobol
    >>> Si = sobol.analyze(problem, results[:, 0])  # Analyze first output
    >>> print(f"S1 indices: {Si['S1']}")

    See Also
    --------
    get_problem_parameters : Create problem definition for Sobol sampling
    SALib.sample.sobol : Generate Sobol parameter samples
    SALib.analyze.sobol : Compute Sobol sensitivity indices
    """
    current_selections = model.timeCourseSelections
    model.timeCourseSelections = current_selections
    actual_selections = model.timeCourseSelections
    print(f"Requested: {len(current_selections)} | Accepted: {len(actual_selections)}")

    missing = set(current_selections) - set(actual_selections)
    if missing:
        print(f"Model ignored these selections: {missing}")

    # Get SBML string for recreation in worker processes
    sbml_string = model.getCurrentSBML()
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Running simulations with {n_processes} processes...")
    
    # Prepare arguments for worker processes
    args_list = [
        (i, param, sbml_string, valid_elements, valid_idxs, input_ids, current_selections, sim_end_time)
        for i, param in enumerate(params)
    ]
    
    # Initialize result array
    RES = np.zeros([params.shape[0], len(valid_elements)])
    
    # Run simulations in parallel
    with Pool(processes=n_processes) as pool:
        # Use imap_unordered for better progress tracking
        results = []
        for result in pool.imap_unordered(_simulation_worker, args_list, chunksize=10):
            results.append(result)
            sys.stdout.write("\r" + " " * 50)  # Clear the line
            sys.stdout.write(f"\rProcessed {len(results)}/{len(params)} samples")
            sys.stdout.flush()
        
        # Assemble results in correct order
        for i, result in results:
            RES[i, :] = result
    
    print()  # New line after progress
    return RES


import numpy as np


def check_convergence(
    results: dict,
    internal_nodes: list,
    min_consecutive: int = 2,
    tol_change: float = 0.01,
    tol_ci: float = 0.05,
    eps_small: float = 1e-3,
    relative: bool = False,
    log_file=None
) -> dict:
    """
    Determine convergence per node and report error metrics.

    Parameters
    ----------
    results : dict
        Nested dictionary of Sobol analysis results at different sample sizes.
        Format: { N : { node : {'S1': array, 'S1_conf': array,
                                'ST': array, 'ST_conf': array}, ... }, ... }
        where N is the sample size and node is the output node name.
    internal_nodes : list of str
        Names of output nodes to analyze for convergence.
    min_consecutive : int, optional
        Minimum number of consecutive sample sizes that must meet convergence
        criteria before declaring convergence, by default 2.
    tol_change : float, optional
        Maximum allowed change (absolute or relative) between consecutive
        sample sizes, by default 0.01.
    tol_ci : float, optional
        Maximum allowed confidence interval half-width (absolute), by default 0.05.
        Note: this is an absolute threshold. For nodes where all Sobol indices
        are near zero, consider whether a relative CI check is also needed.
    eps_small : float, optional
        Cutoff below which an index is considered negligible and excluded from
        the change calculation, by default 1e-3.
    relative : bool, optional
        If False, uses absolute changes between consecutive sample sizes.
        If True, uses symmetric relative changes (normalised by the mean of the
        two values), by default False.

    Returns
    -------
    dict
        Convergence information per node with structure::

            {
              node: {
                'converged_at':  N (first N of the passing streak) or None,
                'diverged_after': N (first N after which convergence broke) or None,
                'max_change':    {N: float or np.nan, ...},
                'ci_half_width': {N: float or np.nan, ...},
              }
            }

        ``max_change[N]`` is ``np.inf`` for the very first sample size (no
        previous result to compare against) and ``np.nan`` when the node is
        absent from ``results[N]`` or an error occurred.

    Raises
    ------
    TypeError
        If ``results`` is not a dict, or ``internal_nodes`` is not a list.
    """
    if not isinstance(results, dict):
        raise TypeError(f"'results' must be a dict, got {type(results)}")
    if not isinstance(internal_nodes, list):
        raise TypeError(f"'internal_nodes' must be a list, got {type(internal_nodes)}")

    def _nanmax_or(arr: np.ndarray, fallback: float) -> float:
        """Return nanmax of *arr*, or *fallback* when arr is empty or all-NaN."""
        try:
            v = np.nanmax(arr)
            return fallback if np.isnan(v) else float(v)
        except ValueError:
            return fallback
        
    print_log(log_file, f"Confidence interval half-widths: {tol_ci}")

    Ns = sorted(results.keys())
    convergence: dict = {}

    for node in internal_nodes:
        prev: dict | None = None
        converged_at: int | None = None
        diverged_after: int | None = None
        consecutive_count: int = 0
        max_change: dict = {}
        ci_half_width: dict = {}

        for N in Ns:
            # --- node absent at this sample size ---
            if node not in results[N]:
                max_change[N] = np.nan
                ci_half_width[N] = np.nan
                consecutive_count = 0
                prev = None
                continue

            curr = results[N][node]

            try:
                # ------------------------------------------------------------------
                # 1. Confidence-interval width (absolute, max over S1 and ST)
                # ------------------------------------------------------------------
                max_ci = max(
                    _nanmax_or(
                        np.asarray(curr.get("S1_conf", [np.nan]), dtype=float),
                        np.inf,
                    ),
                    _nanmax_or(
                        np.asarray(curr.get("ST_conf", [np.nan]), dtype=float),
                        np.inf,
                    ),
                )
                ci_half_width[N] = max_ci

                # ------------------------------------------------------------------
                # 2. Change relative to previous sample size
                # ------------------------------------------------------------------
                if prev is None:
                    # First sample size — no predecessor to compare against.
                    # Record inf explicitly so callers know no comparison was made.
                    max_delta: float = np.inf
                else:
                    curr_S1 = np.asarray(curr.get("S1", [np.nan]), dtype=float)
                    curr_ST = np.asarray(curr.get("ST", [np.nan]), dtype=float)
                    prev_S1 = np.asarray(prev.get("S1", [np.nan]), dtype=float)
                    prev_ST = np.asarray(prev.get("ST", [np.nan]), dtype=float)

                    # Mask: keep only indices that are non-negligible in at least
                    # one of the two consecutive results.
                    mask = (
                        np.fmax(np.abs(curr_S1), np.abs(prev_S1)) >= eps_small
                    ) | (
                        np.fmax(np.abs(curr_ST), np.abs(prev_ST)) >= eps_small
                    )

                    if not np.any(mask):
                        # All indices negligible — treat as perfectly converged.
                        max_delta = 0.0
                    else:
                        if relative:
                            # Symmetric relative change: avoids instability when
                            # the denominator (prev) is near zero.
                            denom_S1 = np.where(
                                (np.abs(prev_S1) + np.abs(curr_S1)) / 2 < eps_small,
                                eps_small,
                                (np.abs(prev_S1) + np.abs(curr_S1)) / 2,
                            )
                            denom_ST = np.where(
                                (np.abs(prev_ST) + np.abs(curr_ST)) / 2 < eps_small,
                                eps_small,
                                (np.abs(prev_ST) + np.abs(curr_ST)) / 2,
                            )
                            diff_S1 = np.abs(curr_S1 - prev_S1) / denom_S1
                            diff_ST = np.abs(curr_ST - prev_ST) / denom_ST
                        else:
                            diff_S1 = np.abs(curr_S1 - prev_S1)
                            diff_ST = np.abs(curr_ST - prev_ST)

                        delta_s1 = _nanmax_or(diff_S1[mask], 0.0)
                        delta_st = _nanmax_or(diff_ST[mask], 0.0)
                        max_delta = float(np.fmax(delta_s1, delta_st))

                max_change[N] = max_delta

                # ------------------------------------------------------------------
                # 3. Convergence check
                # ------------------------------------------------------------------
                passed = (max_delta < tol_change) and (max_ci < tol_ci)

                if passed:
                    consecutive_count += 1
                    # Record the *start* of the passing streak on first hit.
                    if consecutive_count >= min_consecutive and converged_at is None:
                        streak_start_idx = max(
                            0, Ns.index(N) - consecutive_count + 1
                        )
                        converged_at = Ns[streak_start_idx]
                else:
                    # A failing result after a previously declared convergence
                    # means the series has diverged again.
                    if converged_at is not None and diverged_after is None:
                        diverged_after = N
                    consecutive_count = 0

                # Always advance prev so consecutive Ns are always compared.
                prev = curr

            except (TypeError, ValueError, KeyError) as exc:
                # Narrow exception catch: real bugs in unrelated code will still
                # surface.  Record the failure for this N and reset state.
                print_log(log_file, f"[DEBUG CONVERGENCE ERROR] Node {node} at N={N} failed because: {repr(exc)}")

                max_change[N] = np.nan
                ci_half_width[N] = np.nan
                consecutive_count = 0
                prev = None
                # Optionally log: warnings.warn(f"[{node}@{N}] skipped: {exc}")
                continue

        convergence[node] = {
            "converged_at": converged_at,
            "diverged_after": diverged_after,
            "max_change": max_change,
            "ci_half_width": ci_half_width,
        }

    return convergence



def report_sensitivity(res_dict: dict, parameters: list, report_file: str) -> None:
    """
    Generate a formatted sensitivity analysis report and write it to a file.

    This function creates a comprehensive text report of Sobol sensitivity indices,
    including first-order (S1) and total-order (ST) indices, dominant inputs,
    negligible inputs, and interaction detection for each analyzed node.

    Parameters
    ----------
    res_dict : dict
        Dictionary of Sobol analysis results per node.
        Format: { node_name: {'S1': array, 'S1_conf': array,
                            'ST': array, 'ST_conf': array,
                            'S2': array}, ... }
        where arrays contain sensitivity indices and confidence intervals.
    parameters : list of str
        List of input parameter names corresponding to the sensitivity indices.
        Order must match the order of indices in the result arrays.
    report_file : str
        Path to the output report file where results will be written.

    Returns
    -------
    None
        The function writes results to the specified file and returns nothing.

    Notes
    -----
    The report includes for each node:
    - First-order Sobol indices (S1) with confidence intervals
    - Dominant input identification (highest absolute S1 value)
    - Negligible inputs (|S1| < 0.01)
    - Interaction detection based on second-order indices (S2 > 1e-3)
    - Total-order Sobol indices (ST) with confidence intervals

    The report is written in plain text format with clear section separators.

    Examples
    --------
    Generate sensitivity report:
    >>> res_dict = {
    ...     'Output1': {
    ...         'S1': np.array([0.6, 0.3, 0.05]),
    ...         'S1_conf': np.array([0.02, 0.01, 0.005]),
    ...         'ST': np.array([0.65, 0.35, 0.06]),
    ...         'ST_conf': np.array([0.03, 0.02, 0.007]),
    ...         'S2': np.array([[0, 0.02, 0], [0.02, 0, 0], [0, 0, 0]])
    ...     }
    ... }
    >>> parameters = ['Input1', 'Input2', 'Input3']
    >>> report_sensitivity(res_dict, parameters, 'sensitivity_report.txt')

    See Also
    --------
    check_convergence : Analyze convergence of sensitivity indices
    convergence_report : Generate convergence analysis report
    """
    with open(report_file, "w") as f:
        f.write("====== SENSITIVITY REPORT ANALYSIS ======\n")
        f.write(41 * "=")
        f.write("\n")

        for node, data in res_dict.items():
            f.write(f"=== Sensitivity Report for Node {node} ===\n")
            f.write("First-Order Sobol Indices (S1):\n")
            for i, s1 in enumerate(data["S1"]):
                conf = data["S1_conf"][i]
                f.write(f"  {parameters[i]}: {s1:.5f} ± {conf:.5f}\n")

            dominant_index = np.argmax(np.abs(data["S1"]))
            f.write(
                f"Dominant Input: {parameters[dominant_index]} (S1 = {data['S1'][dominant_index]:.5f})\n"
            )

            negligible = [
                parameters[i] for i, s1 in enumerate(data["S1"]) if np.abs(s1) < 0.01
            ]
            if negligible:
                f.write(f"Negligible Inputs (S1 < 0.01): {', '.join(negligible)}\n")
            else:
                f.write("No negligible inputs (all S1 ≥ 0.01)\n")

            # Check for interactions
            S2_nonzero = np.nansum(np.abs(data["S2"]) > 1e-3)
            if S2_nonzero == 0:
                f.write("Interactions: None detected (all S2 < 1e-3)\n")
            else:
                f.write(
                    f"Interactions: {S2_nonzero} significant interaction(s) detected (S2 > 1e-3)\n",
                )

            f.write("Total-Order Sobol Indices (ST):\n")
            for i, st in enumerate(data["ST"]):
                conf = data["ST_conf"][i]
                f.write(f"  {parameters[i]}: {st:.5f} ± {conf:.5f}\n")
            f.write("=" * 41)
            f.write("\n")


def convergence_report(convergence_informations, report_file):
    """
    {
        'C': {
            'ci_half_width': {64: np.float64(0.24602364788674944),
                     128: np.float64(0.20275722115655403)},
            'converged_at': None,
            'max_change': {64: inf, 128: np.float64(0.07043455196840663)}},
    'Out1': {
            'ci_half_width': {64: np.float64(0.30358699371699416),
                        128: np.float64(0.18037084052882127)},
            'converged_at': None,
            'max_change': {64: inf, 128: np.float64(0.9703688240363703)}},
    'Out2': {
            'ci_half_width': {64: np.float64(0.2285240676628604),
                        128: np.float64(0.16170909631137673)},
            'converged_at': None,
            'max_change': {64: inf, 128: np.float64(0.07642293347579533)}}}
    """
    # __import__("pprint").pprint(convergence_informations)

    conv_info = [
        (node, data["converged_at"])
        for node, data in convergence_informations.items()
    ]

    not_conv = []

    with open(report_file, "w") as f:
        f.write("====== CONVERGENCE ANALYSIS ======\n")
        f.write(41 * "=")
        f.write("\n")

        for node, c_at in conv_info:
            if c_at is None:
                not_conv.append(node)

        if len(not_conv) >= 1:
            f.write("[WARNING]: Not convergence detected whitin 4096 samples\n")
            f.write(
                "[WARNING]: There may be nodes with constant value or that do not converge\n"
            )

        f.write("=== NON_CONVERGENT NODES ===\n")
        f.write(f"{','.join(not_conv)}\n")
        f.write(41 * "=")
        f.write("\n")

        f.write("=== CONVERGENCE INFORMATION ===\n")

        valid_values = []

        for node, value in conv_info:
            if value is not None:
                valid_values.append((node, value))

        if len(valid_values) != 0:
            max_samples_size = np.nanmax([t[1] for t in valid_values])
            f.write(
                f" Max samples size to convergence whithin 4096: {max_samples_size}\n"
            )
        else:
            f.write("All values are NaN. \nNo node reaches convergence\n")

        f.write("=== Convergence values per node ===\n")
        for node, value in conv_info:
            f.write(f"  {node}:{value}\n")

        f.write(41 * "=")


def plot_convergence_single_plot(
    convergence_informations, tol_change=0.01, tol_ci=0.05, file_name=None, output_dir="./results"
):
    """
    Plot both metrics on a single plot with different line styles and colors.
    """
    plt.figure(figsize=(12, 8))  # Aumentata l'altezza per la legenda sotto

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(convergence_informations)))

    for i, (node, info) in enumerate(convergence_informations.items()):
        Ns = sorted(info["max_change"].keys())
        changes = [info["max_change"][N] for N in Ns]
        cis = [info["ci_half_width"][N] for N in Ns]

        # Plot changes (solid line with circles)
        plt.plot(
            Ns,
            changes,
            "-o",
            color=colors[i],
            label=f"{node} - Max Change",
            linewidth=2,
            markersize=6,
        )

        # Plot CIs (dashed line with squares)
        plt.plot(
            Ns,
            cis,
            "--s",
            color=colors[i],
            alpha=0.7,
            label=f"{node} - CI Half-Width",
            linewidth=2,
            markersize=5,
        )

    # Add tolerance lines
    plt.axhline(
        tol_change,
        color="gray",
        linestyle="-",
        linewidth=2,
        label="Change Tolerance",
        alpha=0.8,
    )
    plt.axhline(
        tol_ci, color="red", linestyle="-", linewidth=2, label="CI Tolerance", alpha=0.8
    )

    plt.title("Sobol Index Convergence Diagnostics")
    plt.xlabel("Base sample size N")
    plt.ylabel("Metric Value")

    # Legenda sotto il grafico con più colonne per compattezza
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)

    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()

    if file_name is None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "Convergence_analysis.png"), dpi=300, bbox_inches="tight")
    else:
        path = os.path.join(output_dir, file_name)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "Convergence_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()


def statistical_tests(
    random_samples, fixed_samples, node_names, alpha=0.05, report_file=None
):
    """
    Test whether random perturbations produce significantly different simulation
    outputs compared to fixed perturbations.

    H0: The output distributions from fixed and random perturbations are identical
        (random perturbations provide no additional information).
    H1: The output distributions differ
        (random perturbations capture behaviour that fixed ones miss).

    Three complementary tests are run per node:
    - Two-sample t-test (parametric, compares means)
    - Kolmogorov-Smirnov test (non-parametric, compares full distributions)
    - Mann-Whitney U test (non-parametric, compares medians/ranks)

    A Bonferroni correction is applied across all nodes to control the
    family-wise error rate.

    Parameters
    ----------
    random_samples : numpy.ndarray
        2D array of simulation results with random perturbations.
        Shape: (n_random_samples, n_nodes)
    fixed_samples : numpy.ndarray
        2D array of simulation results with fixed perturbations.
        Shape: (n_fixed_samples, n_nodes)
    node_names : list of str
        Names of the output nodes (one per column in the sample arrays).
    alpha : float, optional
        Significance level before correction, by default 0.05.
    report_file : str or None, optional
        Path to write the report. If None, prints to stdout.

    Returns
    -------
    dict
        Per-node results with structure:
        { node_name: {
            'ttest': {'statistic': float, 'pvalue': float, 'reject_h0': bool},
            'ks':    {'statistic': float, 'pvalue': float, 'reject_h0': bool},
            'mw':    {'statistic': float, 'pvalue': float, 'reject_h0': bool},
            'cohen_d': float,
            'significant': bool   # True if ANY corrected test rejects H0
          }
        }
    """
    num_nodes = len(node_names)
    corrected_alpha = alpha / num_nodes  # Bonferroni correction

    results = {}
    lines = []

    lines.append("=" * 60)
    lines.append("STATISTICAL COMPARISON: RANDOM vs FIXED PERTURBATIONS")
    lines.append("=" * 60)
    lines.append(f"H0: No difference between random and fixed perturbation outputs")
    lines.append(f"H1: Distributions differ (random perturbations are informative)")
    lines.append(f"Significance level (alpha): {alpha}")
    lines.append(f"Bonferroni-corrected alpha ({num_nodes} nodes): {corrected_alpha:.2e}")
    lines.append("")

    significant_nodes = []

    for idx in range(num_nodes):
        node = node_names[idx]
        random_node = random_samples[:, idx]
        fixed_node = fixed_samples[:, idx]

        # --- Tests ---
        t_stat, t_pval = stats.ttest_ind(random_node, fixed_node, equal_var=False)
        ks_stat, ks_pval = stats.ks_2samp(random_node, fixed_node)
        mw_stat, mw_pval = stats.mannwhitneyu(
            random_node, fixed_node, alternative="two-sided"
        )

        # --- Effect size (Cohen's d, pooled std) ---
        n_r, n_f = len(random_node), len(fixed_node)
        pooled_std = np.sqrt(
            ((n_r - 1) * np.var(random_node, ddof=1) + (n_f - 1) * np.var(fixed_node, ddof=1))
            / (n_r + n_f - 2)
        )
        cohen_d = (np.mean(random_node) - np.mean(fixed_node)) / pooled_std if pooled_std > 0 else 0.0

        t_reject = t_pval < corrected_alpha
        ks_reject = ks_pval < corrected_alpha
        mw_reject = mw_pval < corrected_alpha
        any_reject = t_reject or ks_reject or mw_reject

        results[node] = {
            "ttest": {"statistic": t_stat, "pvalue": t_pval, "reject_h0": t_reject},
            "ks": {"statistic": ks_stat, "pvalue": ks_pval, "reject_h0": ks_reject},
            "mw": {"statistic": mw_stat, "pvalue": mw_pval, "reject_h0": mw_reject},
            "cohen_d": cohen_d,
            "significant": any_reject,
        }

        if any_reject:
            significant_nodes.append(node)

        verdict = "REJECT H0 (significant difference)" if any_reject else "FAIL TO REJECT H0"

        lines.append(f"--- {node} ---")
        lines.append(f"  Welch t-test  : t = {t_stat:+.4f},  p = {t_pval:.2e}  {'*' if t_reject else ''}")
        lines.append(f"  KS test       : D = {ks_stat:.4f},   p = {ks_pval:.2e}  {'*' if ks_reject else ''}")
        lines.append(f"  Mann-Whitney U: U = {mw_stat:.0f},  p = {mw_pval:.2e}  {'*' if mw_reject else ''}")
        lines.append(f"  Cohen's d     : {cohen_d:+.4f}")
        lines.append(f"  Verdict       : {verdict}")
        lines.append("")

    # --- Overall conclusion ---
    lines.append("=" * 60)
    lines.append("OVERALL CONCLUSION")
    lines.append("=" * 60)
    n_sig = len(significant_nodes)

    if n_sig == 0:
        lines.append(
            "No significant differences detected across any node."
        )
        lines.append(
            "=> Fixed perturbations appear sufficient; "
            "random perturbations do NOT add information."
        )
    elif n_sig == num_nodes:
        lines.append(
            f"All {num_nodes} nodes show significant differences."
        )
        lines.append(
            "=> Random perturbations capture substantially different behaviour; "
            "they are recommended over fixed perturbations."
        )
    else:
        lines.append(
            f"{n_sig}/{num_nodes} nodes show significant differences: "
            f"{', '.join(significant_nodes)}"
        )
        lines.append(
            "=> Mixed results; random perturbations are informative for some "
            "outputs. Consider using random perturbations for a more complete analysis."
        )

    lines.append("")

    report_text = "\n".join(lines)

    if report_file is not None:
        os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report_text)
    else:
        print(report_text)

    return results
