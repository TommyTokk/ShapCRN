import multiprocessing
from os import path
import os

from matplotlib.cbook import safe_masked_invalid
from utils import simulation_utils
from utils.simulation_utils import load_roadrunner_model, simulate
from utils.utils import print_log
import numpy as np
import sys
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sample
from multiprocessing import Pool

import roadrunner as rr
import libsbml

from scipy import stats


def _simulation_worker(args):
    """
    Worker function for parallel simulation execution.
    
    Parameters
    ----------
    args : tuple
        (index, param, sbml_string, valid_elements, valid_idxs, input_ids, current_selections)
        
    Returns
    -------
    tuple
        (index, result_array) where result_array contains final concentrations
    """
    i, param, sbml_string, valid_elements, valid_idxs, input_ids, current_selections = args
    
    # Create a new RoadRunner instance in this process
    rr_local = rr.RoadRunner(sbml_string)
    rr_local.timeCourseSelections = current_selections
    
    # Reset and set initial concentrations
    rr_local.reset()
    for j in range(len(input_ids)):
        rr_local.setInitConcentration(input_ids[j], param[j])
    
    # Simulate
    sim_res = rr_local.simulate(0, 5000)
    
    # Extract final concentrations
    result = np.zeros(len(valid_elements))
    for j, el in enumerate(valid_elements):
        s_idx = valid_idxs[el]
        result[j] = sim_res[-1, s_idx]
    
    return (i, result)


# KEEP
def get_problem_parameters(
    sbml_model: libsbml.Model, n_input_species: int, input_species_ids: list, range: int=20, log_file=None
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
    range : int, optional
        Percentage range for perturbation bounds around baseline concentrations,
        by default 20 (i.e., ±20%)
        Example: range=20 creates bounds [conc*0.8, conc*1.2]
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
    initial concentration C and range R%:
    - Lower bound = C * (1 - R/100)
    - Upper bound = C * (1 + R/100)

    Examples
    --------
    Create problem for 20% perturbation range:
    >>> problem = get_problem_parameters(
    ...     sbml_model=model,
    ...     n_input_species=3,
    ...     input_species_ids=['Input1', 'Input2', 'Input3'],
    ...     range=20
    ... )
    >>> print(problem['bounds'])
    [[0.8, 1.2], [1.6, 2.4], [0.4, 0.6]]  # For initial concs [1.0, 2.0, 0.5]

    Create problem with wider perturbation range:
    >>> problem = get_problem_parameters(
    ...     sbml_model=model,
    ...     n_input_species=2,
    ...     input_species_ids=['S1', 'S2'],
    ...     range=50,  # ±50% perturbation
    ...     log_file=log
    ... )

    Use with SALib for Sobol sampling:
    >>> from SALib.sample import sobol
    >>> problem = get_problem_parameters(model, 2, ['S1', 'S2'], range=20)
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

        lower_bound = conc * (1 - (range / 100))
        upper_bound = conc * (1 + (range / 100))

        tmp.append([lower_bound, upper_bound])

    problem["bounds"] = tmp

    return problem


# KEEP
def run_simulation_with_params(
    rr: rr.RoadRunner, params: np.ndarray, valid_elements: list, valid_idxs: dict, input_ids:list, log_file=None, n_processes: int=None
) -> np.ndarray:
    """
    Run batch simulations with parameter samples and extract final concentrations.

    This function executes multiple RoadRunner simulations using different parameter sets
    (typically from Sobol sampling), collecting the final steady-state concentrations of
    specified output species. Simulations are parallelized across multiple processes.

    Parameters
    ----------
    rr : roadrunner.RoadRunner
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

    Returns
    -------
    numpy.ndarray
        2D array of final concentrations for monitored species.
        Shape: (n_samples, n_valid_elements)
        RES[i, j] = final concentration of valid_elements[j] for parameter set i

    Notes
    -----
    - The function uses multiprocessing to parallelize simulations
    - Each simulation runs from time 0 to 5000 time units
    - Progress is displayed as "Processed X/Y samples" to stdout
    - The model is reset before each simulation to ensure independence
    - Only the final time point concentration is extracted from each simulation

    The function is typically used with Sobol-sampled parameters for sensitivity analysis,
    where the output matrix RES is then passed to SALib.analyze.sobol().

    Examples
    --------
    Run simulations with Sobol samples:
    >>> from SALib.sample import sobol
    >>> problem = get_problem_parameters(model, 2, ['Input1', 'Input2'], range=20)
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
    current_selections = rr.timeCourseSelections.copy()
    
    # Get SBML string for recreation in worker processes
    sbml_string = rr.getCurrentSBML()
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Running simulations with {n_processes} processes...")
    
    # Prepare arguments for worker processes
    args_list = [
        (i, param, sbml_string, valid_elements, valid_idxs, input_ids, current_selections)
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


# KEEP
def check_convergence(
    results: dict,
    internal_nodes: list,
    min_consecutive: int=2,
    tol_change: float=0.01,
    tol_ci:float=0.05,
    eps_small:float=1e-3,
    relative:bool=False,
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
        criteria before declaring convergence, by default 2
    tol_change : float, optional
        Maximum allowed change (absolute or relative) between consecutive sample sizes,
        by default 0.01
    tol_ci : float, optional
        Maximum allowed confidence interval half-width (absolute), by default 0.05
    eps_small : float, optional
        Cutoff value for determining small/negligible elements, by default 1e-3
    relative : bool, optional
        Flag for calculation type. If False, uses absolute changes.
        If True, uses relative changes, by default False

    Returns
    -------
    dict
        Convergence information per node with structure:
        { node: { 'converged_at': N or None,
                'max_change': {N: value, ...},
                'ci_half_width': {N: value, ...} } }
        where 'converged_at' is the first sample size where convergence is achieved,
        or None if not converged.
    """

    def nanmax_or(arr, fallback):
        try:
            v = np.nanmax(arr)
            if np.isnan(v):
                return fallback
            return v
        except ValueError:
            # all-NaN slice
            return fallback

    convergence = {}
    Ns = sorted(results.keys())

    for node in internal_nodes:
        prev = None
        converged_at = None
        consecutive_count = 0
        max_change = {}
        ci_half_width = {}

        for N in Ns:
            if node not in results[N]:
                max_change[N] = np.nan
                ci_half_width[N] = np.nan
                consecutive_count = 0
                prev = None
                continue

            curr = results[N][node]

            try:
                # CI: Takes the max ignoring Nan values
                max_ci = max(
                    nanmax_or(curr.get("S1_conf", np.array([np.nan])), np.inf),
                    nanmax_or(curr.get("ST_conf", np.array([np.nan])), np.inf),
                )
                ci_half_width[N] = max_ci

                if prev is not None:
                    curr_S1 = np.asarray(curr.get("S1", np.array([np.nan])))
                    curr_ST = np.asarray(curr.get("ST", np.array([np.nan])))
                    prev_S1 = np.asarray(prev.get("S1", np.array([np.nan])))
                    prev_ST = np.asarray(prev.get("ST", np.array([np.nan])))

                    # Maks the relevant values
                    mask_relevant = (
                        np.fmax(np.abs(curr_S1), np.abs(prev_S1)) >= eps_small
                    ) | (np.fmax(np.abs(curr_ST), np.abs(prev_ST)) >= eps_small)

                    if relative:
                        denom_S1 = (np.abs(prev_S1) + np.abs(curr_S1)) / 2
                        denom_ST = (np.abs(prev_ST) + np.abs(curr_ST)) / 2

                        denom_S1 = np.where(denom_S1 < eps_small, eps_small, denom_S1)
                        denom_ST = np.where(denom_ST < eps_small, eps_small, denom_ST)

                        diff_S1 = np.abs(curr_S1 - prev_S1) / denom_S1
                        diff_ST = np.abs(curr_ST - prev_ST) / denom_ST
                    else:
                        diff_S1 = np.abs(curr_S1 - prev_S1)
                        diff_ST = np.abs(curr_ST - prev_ST)

                    # Apply the mask, if no relevant values delta = 0
                    if np.any(mask_relevant):
                        delta_s1 = nanmax_or(diff_S1[mask_relevant], 0.0)
                        delta_st = nanmax_or(diff_ST[mask_relevant], 0.0)
                        max_delta = np.fmax(delta_s1, delta_st)
                    else:
                        max_delta = 0.0
                else:
                    max_delta = np.inf

                max_change[N] = max_delta

                passed = (max_delta < tol_change) and (max_ci < tol_ci)

                # print_log(
                #     "debug",
                #     f"[DEBUG] {node} - {(max_delta < tol_change)} - {(max_ci < tol_ci)} - {passed}",
                # )

                if passed:
                    consecutive_count += 1
                    if consecutive_count >= min_consecutive and converged_at is None:
                        streak_start_idx = max(0, Ns.index(N) - consecutive_count + 1)
                        converged_at = Ns[streak_start_idx]
                        # print_log(
                        #     "debug",
                        #     f"[DEBUG] {node} - {converged_at} - {consecutive_count}",
                        # )
                else:
                    consecutive_count = 0

                prev = curr

            except Exception:
                max_change[N] = np.nan
                ci_half_width[N] = np.nan
                consecutive_count = 0
                prev = None
                continue

        convergence[node] = {
            "converged_at": converged_at,
            "max_change": max_change,
            "ci_half_width": ci_half_width,
        }

    return convergence


def assess_model_linearity(RES, sample_sizes) -> str:
    """
    Assess model linearity to help decide on perturbation suitability.

    This function analyzes the coefficient of variation of model outputs across
    different nodes to determine if the model exhibits linear behavior, which
    helps decide whether perturbation-based methods are appropriate.

    Parameters
    ----------
    RES : numpy.ndarray
        2D array of simulation results.
        Shape: (n_samples, n_nodes)
        Each column represents outputs from one node across all samples.
    sample_sizes : array-like
        Array of sample sizes used in the analysis (currently unused in implementation).

    Returns
    -------
    str
        Overall linearity assessment, one of:
        - "High - perturbations likely sufficient" : >70% of nodes show high linearity
        - "Medium - test both approaches" : 40-70% of nodes show high linearity
        - "Low - full sampling recommended" : <40% of nodes show high linearity

    Notes
    -----
    Linearity is assessed per node based on coefficient of variation (CV):
    - High linearity: CV < 0.5
    - Medium linearity: 0.5 <= CV < 1.5
    - Low linearity: CV >= 1.5

    The function prints detailed linearity information to stdout during execution.
    """

    print(f"\nMODEL LINEARITY ASSESSMENT:")
    print(f"{'=' * 50}")

    # Check if sensitivity indices are stable across different sample sizes
    # (indicating linear behavior)
    linearity_scores = {}

    for node_idx in range(RES.shape[1]):
        node_outputs = RES[:, node_idx]

        # Simple linearity check: coefficient of variation of outputs
        cv = np.std(node_outputs) / (np.mean(np.abs(node_outputs)) + 1e-10)

        if cv < 0.5:
            linearity = "High"
        elif cv < 1.5:
            linearity = "Medium"
        else:
            linearity = "Low"

        linearity_scores[f"node_{node_idx}"] = linearity

    # Overall assessment
    high_linearity_count = sum(1 for l in linearity_scores.values() if l == "High")
    total_nodes = len(linearity_scores)

    if high_linearity_count / total_nodes > 0.7:
        overall_linearity = "High - perturbations likely sufficient"
    elif high_linearity_count / total_nodes > 0.4:
        overall_linearity = "Medium - test both approaches"
    else:
        overall_linearity = "Low - full sampling recommended"

    print(f"Overall model linearity: {overall_linearity}")
    return overall_linearity


# KEEP
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


# KEEP
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

    conv_info = []

    for internal_node, convergence_data in convergence_informations.items():
        conv_info.append((internal_node, convergence_data["converged_at"]))

    __import__("pprint").pprint(conv_info)

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


# KEEP
def plot_convergence_single_plot(
    convergence_informations, tol_change=0.01, tol_ci=0.05, file_name=None
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
        plt.savefig("./imgs/Convergence_analysis.png", dpi=300, bbox_inches="tight")
    else:
        path = f"./imgs/{file_name}"
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/Convergence_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()  # Opzionale: libera la memoria


def convergence_analysis(random_samples, fixed_samples, n_inputs):
    """
    Analyze how random samples converge compared to fixed samples
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for node in range(n_inputs):
        random_node = random_samples[:, node]
        fixed_node = fixed_samples[:, node]

        # Running mean of random samples
        running_mean = np.cumsum(random_node) / np.arange(1, len(random_node) + 1)
        fixed_mean = np.mean(fixed_node)

        # Plot convergence
        axes[node].plot(running_mean, label="Random (running mean)", alpha=0.8)
        axes[node].axhline(
            y=fixed_mean,
            color="red",
            linestyle="--",
            label=f"Fixed mean ({fixed_mean:.4f})",
        )
        axes[node].set_title(f"Node {node} - Convergence Analysis")
        axes[node].set_xlabel("Sample number")
        axes[node].set_ylabel("Running mean")
        axes[node].legend()
        axes[node].grid(True, alpha=0.3)

    plt.tight_layout()


# KEEP
def statistical_tests(random_samples, fixed_samples, num_nodes):
    """
    Perform statistical tests to compare distributions
    """
    print("=== STATISTICAL TESTS ===")

    for node in range(num_nodes):
        random_node = random_samples[:, node]
        fixed_node = fixed_samples[:, node]

        print(f"--- Node {node} ---")

        # Two-sample t-test
        t_stat, t_pval = stats.ttest_ind(random_node, fixed_node)
        print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {t_pval:.2e}")

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(random_node, fixed_node)
        print(f"KS-test: statistic = {ks_stat:.4f}, p-value = {ks_pval:.2e}")

        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_pval = stats.mannwhitneyu(
            random_node, fixed_node, alternative="two-sided"
        )
        print(f"Mann-Whitney U: statistic = {mw_stat:.0f}, p-value = {mw_pval:.2e}")
        print()
