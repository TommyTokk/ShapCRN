from utils.simulation_utils import simulate
from utils.utils import print_log
import numpy as np


def get_problem_parameters(
    sbml_model, n_input_species, input_species_ids, range=20, log_file=None
):
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


def run_simulation_with_params(
    rr, params, input_ids, selections, internal_nodes, log_file=None
):

    RES = np.zeros([params.shape[0], len(internal_nodes)])

    for i, param in enumerate(params):
        # Saving the selections

        rr.reset()

        rr.selections = selections

        # Changing the concentrations
        for j in range(len(input_ids)):
            rr.setInitConcentration(input_ids[j], param[j])

        # Simulate
        sim_res, _, colnames = simulate(rr, 0, 5000, log_file=log_file)

        for j in range(len(internal_nodes)):
            s_id = colnames.index(f"[{internal_nodes[j]}]")

            RES[i, j] = sim_res[-1, s_id]

    return RES


def check_convergence(results, internal_nodes, tol_change=0.01, tol_ci=0.05):
    """
    Determine convergence per node and report error metrics.

    Parameters:
      results (dict): { N : { node : {'S1': array, 'S1_conf': array,
                                      'ST': array, 'ST_conf': array}, ... }, ... }
      internal_nodes (list): names of output nodes to analyze.
      tol_change (float): Max allowed change in index between sample sizes.
      tol_ci (float): Max allowed CI half-width.

    Returns:
      dict: { node: { 'converged_at': N or None,
                      'max_change': {N: value, ...},
                      'ci_half_width': {N: value, ...} } }
    """
    convergence = {}
    Ns = sorted(results.keys())

    for node in internal_nodes:
        node_data = results
        prev = None
        converged_at = None
        max_change = {}
        ci_half_width = {}

        for N in Ns:
            curr = results[N][node]
            # Max CI half-width across S1 and ST
            max_ci = max(curr["S1_conf"].max(), curr["ST_conf"].max())
            ci_half_width[N] = max_ci

            # Max change from previous N
            if prev:
                delta_s1 = abs(curr["S1"] - prev["S1"]).max()
                delta_st = abs(curr["ST"] - prev["ST"]).max()
                max_delta = max(delta_s1, delta_st)
            else:
                max_delta = float("inf")
            max_change[N] = max_delta

            if max_delta < tol_change and max_ci < tol_ci and converged_at is None:
                converged_at = N

            prev = curr

        convergence[node] = {
            "converged_at": converged_at,
            "max_change": max_change,
            "ci_half_width": ci_half_width,
        }

    return convergence


def report_sensitivity(node, data, parameters, log_file=None):
    print_log(log_file, f"\n--- Sensitivity Report for Node {node} ---")
    print_log(log_file, "First-Order Sobol Indices (S1):")
    for i, s1 in enumerate(data["S1"]):
        conf = data["S1_conf"][i]
        print_log(log_file, f"  {parameters[i]}: {s1:.5f} ± {conf:.5f}")

    dominant_index = np.argmax(np.abs(data["S1"]))
    print_log(
        log_file,
        f"\nDominant Input: {parameters[dominant_index]} (S1 = {data['S1'][dominant_index]:.5f})",
    )

    negligible = [parameters[i] for i, s1 in enumerate(data["S1"]) if np.abs(s1) < 0.01]
    if negligible:
        print_log(log_file, f"Negligible Inputs (S1 < 0.01): {', '.join(negligible)}")
    else:
        print_log(log_file, "No negligible inputs (all S1 ≥ 0.01)")

    # Check for interactions
    S2_nonzero = np.nansum(np.abs(data["S2"]) > 1e-3)
    if S2_nonzero == 0:
        print_log(log_file, "Interactions: None detected (all S2 < 1e-3)")
    else:
        print_log(
            log_file,
            f"Interactions: {S2_nonzero} significant interaction(s) detected (S2 > 1e-3)",
        )

    print("\nTotal-Order Sobol Indices (ST):")
    for i, st in enumerate(data["ST"]):
        conf = data["ST_conf"][i]
        print_log(log_file, f"  {parameters[i]}: {st:.5f} ± {conf:.5f}")
    print_log(log_file, "-" * 40)
