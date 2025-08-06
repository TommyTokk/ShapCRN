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
