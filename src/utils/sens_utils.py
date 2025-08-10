from utils import simulation_utils
from utils.simulation_utils import simulate
from utils.utils import print_log
import numpy as np
import sys
import matplotlib.pyplot as plt


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
    rr, params, valid_elements, valid_idxs, input_ids, log_file=None
):

    current_selections = rr.timeCourseSelections.copy()

    __import__("pprint").pprint(f"pre: {len(rr.timeCourseSelections)}")
    RES = np.zeros([params.shape[0], len(valid_elements)])
    processed = 0

    for i, param in enumerate(params):
        sys.stdout.write("\r" + " " * 50)  # Clear the line
        sys.stdout.write(f"\rProcessed {processed}/{len(params)} samples")
        sys.stdout.flush()

        #
        rr.reset()

        # Changing the concentrations
        for j in range(len(input_ids)):
            rr.setInitConcentration(input_ids[j], param[j])

        rr.timeCourseSelections = current_selections

        # Simulate
        sim_res, _, colnames = simulation_utils.simulate(rr, 0, 5000, log_file=log_file)

        # __import__("pprint").pprint(colnames)

        idx = 0

        for el in valid_elements:
            s_idx = valid_idxs[el]
            # print_log(log_file, f"{el} - {idx} - {s_idx} - {colnames.index(el)}")
            RES[i, idx] = sim_res[-1, s_idx]

            idx += 1

        processed += 1

    return RES


def check_convergence(
    results,
    internal_nodes,
    min_consecutive=2,
    tol_change=0.01,
    tol_ci=0.05,
    eps_small=1e-3,
    relative=False,
):
    """
    Determine convergence per node and report error metrics.

    Parameters:
      results (dict): { N : { node : {'S1': array, 'S1_conf': array,
                                      'ST': array, 'ST_conf': array}, ... }, ... }
      internal_nodes (list): names of output nodes to analyze.
      tol_change (float): Max allowed change (assoluta o relativa) tra N consecutivi.
      tol_ci (float): Max allowed CI half-width (assoluta).
      eps_small (float): Soglia sotto cui un indice è considerato trascurabile.
      relative (bool): Se True usa variazione relativa; altrimenti assoluta.

    Returns:
      dict: { node: { 'converged_at': N or None,
                      'max_change': {N: value, ...},
                      'ci_half_width': {N: value, ...} } }
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

                    # Maschera parametri “rilevanti”
                    mask_relevant = (
                        np.fmax(np.abs(curr_S1), np.abs(prev_S1)) >= eps_small
                    ) | (np.fmax(np.abs(curr_ST), np.abs(prev_ST)) >= eps_small)

                    if relative:
                        denom_S1 = np.fmax(np.abs(prev_S1), np.abs(curr_S1))
                        denom_ST = np.fmax(np.abs(prev_ST), np.abs(curr_ST))
                        denom_S1 = np.where(denom_S1 < eps_small, 1.0, denom_S1)
                        denom_ST = np.where(denom_ST < eps_small, 1.0, denom_ST)
                        diff_S1 = np.abs(curr_S1 - prev_S1) / denom_S1
                        diff_ST = np.abs(curr_ST - prev_ST) / denom_ST
                    else:
                        diff_S1 = np.abs(curr_S1 - prev_S1)
                        diff_ST = np.abs(curr_ST - prev_ST)

                    # Applica maschera; se nessun parametro rilevante, delta=0
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

                print_log(
                    "debug",
                    f"[DEBUG] {node} - {(max_delta < tol_change)} - {(max_ci < tol_ci)} - {passed}",
                )

                if passed:
                    consecutive_count += 1
                    if consecutive_count >= min_consecutive and converged_at is None:

                        streak_start_idx = max(0, Ns.index(N) - consecutive_count + 1)
                        converged_at = Ns[streak_start_idx]
                        print_log(
                            "debug",
                            f"[DEBUG] {node} - {converged_at} - {consecutive_count}",
                        )
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


def plot_convergence_single_plot(
    convergence_informations, tol_change=0.01, tol_ci=0.05
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

    plt.title("Sobol Index Convergence Diagnostics - All Nodes")
    plt.xlabel("Base sample size N")
    plt.ylabel("Metric Value")
    
    # Legenda sotto il grafico con più colonne per compattezza
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("./imgs/Convergence_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()  # Opzionale: libera la memoria
