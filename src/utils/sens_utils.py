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

from scipy import stats


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

        # __import__("pprint").pprint(valid_elements)
        for j, el in enumerate(valid_elements):
            s_idx = valid_idxs[el]
            # print_log(log_file, f"{el} - {idx} - {s_idx} - {colnames.index(el)}")
            RES[i, j] = sim_res[-1, s_idx]

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
      eps_small (float): Cutoff value for small elements.
      relative (bool): Flag for calculation type, if False --> Absolute.

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


def assess_model_linearity(RES, sample_sizes):
    """
    Assess model linearity to help decide on perturbation suitability
    """

    print(f"\nMODEL LINEARITY ASSESSMENT:")
    print(f"{'='*50}")

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


def report_sensitivity(res_dict, parameters, report_file):

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

    plt.title("Sobol Index Convergence Diagnostics - All Nodes")
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


def statistical_tests(random_samples, fixed_samples):
    """
    Perform statistical tests to compare distributions
    """
    print("=== STATISTICAL TESTS ===")

    for node in range(3):
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
