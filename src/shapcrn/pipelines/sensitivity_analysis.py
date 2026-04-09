import os
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import libsbml
import pandas as pd

import shapcrn.utils.sensitivity as sens_ut
from shapcrn.utils import utils as ut
from shapcrn import exceptions as ex
from shapcrn.utils.sbml import io as sbml_io
from shapcrn.utils.sbml import utils as sbml_ut
from shapcrn.utils import species as sbml_species
from shapcrn.utils import simulation as sim_ut

N_VALUES = [64, 128, 256, 512, 1024]


def parse_args(args):
    model_path = args.input_path
    input_species_ids = args.input_species
    base_samples = args.base_samples
    perturbations_range = args.perturbation_range
    knock_operation = args.operation
    preserve_input = args.preserve_inputs
    target_species = args.target_species
    fixed_perturbations = args.fixed_perturbations
    check_convergence = args.check_convergence
    log_file = args.log if args.log else None

    parsed_args = {
        "model_path": model_path,
        "input_species_ids": input_species_ids,
        "base_samples": base_samples,
        "perturbation_range": perturbations_range,
        "knock_operation": knock_operation,
        "preserve_input": preserve_input,
        "target_species": target_species,
        "fixed_perturbations": fixed_perturbations,
        "check_convergence": check_convergence,
        "log_file": log_file,
    }

    return parsed_args


def model_preparation(args):
    model_path = args["model_path"]
    input_species_ids = args["input_species_ids"]

    sbml_doc, model = sbml_io.load_and_prepare_model(
        model_path, log_file=args["log_file"]
    )

    # Check if the input species exist in the model
    for species_id in input_species_ids:
        if not model.getSpecies(species_id):
            raise ex.InvalidSpeciesError(
                f"Species '{species_id}' not found in the model."
            )

    return sbml_doc, model


def run_convergence_analysis(
    rr_model,
    input_species,
    problem_specs,
    target_species,
    valid_idxs,
    out_dirs,
    log_file,
):
    convergence_results = {}

    valid_elements = list(valid_idxs.keys())


    for N in N_VALUES:
        params = saltelli.sample(problem_specs, N, calc_second_order=True)

        sim_results = sens_ut.run_simulation_with_params(
            rr_model,
            params,
            valid_elements,
            valid_idxs,
            input_species,
        )

        # Update the convergence results
        convergence_results[N] = {}

        for j, node in enumerate(valid_elements):
            node_data = sim_results[:, j]

            # Check for NaNs and Variance
            num_nans = np.isnan(node_data).sum()
            if num_nans < len(node_data):
                variance = np.var(node_data[~np.isnan(node_data)])
            else:
                variance = 0.0

            print(
                f"[DEBUG] N={N} | Node: {node} | NaNs dropped: {num_nans} | Variance: {variance:.6f}"
            )

            valid_mask = ~np.isnan(node_data)

            # If even ONE NaN is dropped, SALib will fail due to length mismatch
            if valid_mask.sum() > 0:
                try:
                    Si = sobol.analyze(
                        problem_specs,
                        node_data[valid_mask],
                        calc_second_order=True,
                        print_to_console=False,
                    )
                    convergence_results[N][node] = Si
                except Exception as e:
                    ut.print_log(
                        log_file,
                        f"[WARNING] SALib failed for node '{node}' at N={N}. Error: {e}",
                    )
                    convergence_results[N][node] = None
            else:
                ut.print_log(
                    log_file, f"[WARNING] All data is NaN for node '{node}' at N={N}."
                )
                convergence_results[N][node] = None

    convergence_info = sens_ut.check_convergence(
        convergence_results,
        valid_elements,
        tol_ci=0.10,
        min_consecutive=2,
        log_file=log_file,
    )

    return convergence_results, convergence_info

def prepare_new_values(sbml_model, valid_elements, log_file = None):
    pass


def prompt_user_for_N(problem_specs, log_file=None):
    """Prompt the user to choose or confirm the number of base samples (N)."""
    default_N = max(N_VALUES)
    num_evals = default_N * (2 * problem_specs["num_vars"] + 2)

    ut.print_log(log_file, f"[INFO] Convergence check disabled.")
    ut.print_log(log_file, f"[INFO] Using default N: {default_N} ({num_evals} model evaluations)")
    ut.print_log(log_file, f"[INFO] Do you want to use the default N [y/n]?")

    if input().strip().lower() == "y":
        ut.print_log(log_file, f"[INFO] Proceeding with default N: {default_N}")
        return default_N

    ut.print_log(log_file, f"[INFO] Please enter the desired N:")
    while True:
        try:
            user_N = int(input().strip())
        except ValueError:
            ut.print_log(log_file, f"[ERROR] Invalid input. Please enter a valid integer.")
            continue

        if user_N <= max(N_VALUES):
            ut.print_log(log_file, f"[INFO] Proceeding with N: {user_N} ({user_N * (2 * problem_specs['num_vars'] + 2)} model evaluations)")
            return user_N

        user_evals = user_N * (2 * problem_specs["num_vars"] + 2)
        ut.print_log(log_file, f"[WARNING] Entered N is greater than the maximum recommended value.")
        ut.print_log(log_file, f"[INFO] Proceeding with N: {user_N} ({user_evals} model evaluations) [y/n]?")
        if input().strip().lower() == "y":
            return user_N
        ut.print_log(log_file, f"[INFO] Please enter a new N value:")


def sensitivity_analysis(args, out_dirs):
    parsed_args = parse_args(args)

    sbml_doc, sbml_model = model_preparation(parsed_args)

    # Creation of the problem specifications for the sensitivity analysis
    problem_specs = sens_ut.get_problem_parameters(
        sbml_model=sbml_model,
        n_input_species=len(parsed_args["input_species_ids"]),
        input_species_ids=parsed_args["input_species_ids"],
        perturbation_range=parsed_args["perturbation_range"],
        log_file=parsed_args["log_file"],
    )

    ut.print_log(parsed_args["log_file"], f"{problem_specs}")

    # Getting all the species
    all_species = [s.getId() for s in sbml_species.get_list_of_species(sbml_model)]
    input_species = parsed_args["input_species_ids"]

    internal_nodes = set(all_species) - set(input_species)

    if parsed_args["target_species"] is None:
        target_species = list(internal_nodes)
    else:
        target_species = parsed_args["target_species"]
        for species_id in target_species:
            if species_id not in internal_nodes:
                raise ex.InvalidSpeciesError(
                    f"Target species '{species_id}' not found among internal nodes."
                )
    rr = sim_ut.load_roadrunner_model(sbml_model, log_file=parsed_args["log_file"])

    selections = rr.timeCourseSelections

    #Add the target species to the selections if not already present
    for sp in target_species:
        selection_str = f"[{sp}]"
        if selection_str not in selections:
            selections.append(selection_str)

    rr.timeCourseSelections = selections

    valid_idxs = {
        f"[{sp}]": rr.timeCourseSelections.index(f"[{sp}]") for sp in target_species
    }

    # Run the sensitivity analysis
    optimal_N = max(N_VALUES)  # Default to max N if convergence check is not enabled
    if parsed_args["check_convergence"]:
        convergence_results, convergence_info = run_convergence_analysis(
            rr,
            input_species,
            problem_specs,
            target_species,
            valid_idxs,
            out_dirs,
            parsed_args["log_file"],
        )

        valid_elements = list(valid_idxs.keys())

        converged_Ns = [
            info["converged_at"]
            for info in convergence_info.values()
            if info["converged_at"] is not None
        ]
        if len(converged_Ns) == len(valid_elements):
            optimal_N = max(converged_Ns)
            print(f"\n[INFO] All nodes converged. Optimal N found: {optimal_N}")
        else:
            optimal_N = max(N_VALUES)
            print(f"\n[INFO] Not all nodes converged. Defaulting to maximum N: {optimal_N}")

        number_of_samples = optimal_N * ((2 * problem_specs["num_vars"] )+ 2)
        print(f"[INFO] Total number of model evaluations for N={optimal_N}: {number_of_samples}")

        sens_ut.plot_convergence_single_plot(
            convergence_info,
            file_name="sensitivity_convergence",
            output_dir=out_dirs["images"],
        )
    else:

        if parsed_args["fixed_perturbations"] is None:
            ut.print_log(parsed_args["log_file"], f"[ERROR] To evaluate the difference in sensitivity indices with fixed perturbations, the user must provide a set of fixed perturbations using the '--fixed_perturbations' argument.")
            raise ex.InvalidArgumentError("Fixed perturbations not provided.")
        
        optimal_N = prompt_user_for_N(problem_specs, log_file=parsed_args["log_file"])

        params = saltelli.sample(problem_specs, optimal_N, calc_second_order=True)

        # Running the simulations with the sampled parameters
        RES = sens_ut.run_simulation_with_params(
            rr,
            params,
            list(valid_idxs.keys()),
            valid_idxs,
            input_species,
        )

        # Preparing the fixed samples
        fp = [int(p) for p in args.fixed_perturbations]

        fixed_samples = sbml_ut.get_fixed_combinations(
            sbml_model, input_species, fp, parsed_args["log_file"]
        )

        fixed_results, _ = sim_ut.simulate_combinations(
            rr,
            sim_ut.create_combinations(fixed_samples),
            input_species,
            min_ss_time=1000,
            end_time=5000,
            max_end_time=5000,
            steady_state=False,
            log_file=parsed_args["log_file"],
        )

        FIXED_RES = np.zeros(
            [
                len(fixed_results),
                len(valid_idxs.keys()),
            ]
        )

        for i in range(len(fixed_results)):
            sim = fixed_results[i]
            idx = 0

            for j, el in enumerate(valid_idxs.keys()):
                s_idx = valid_idxs[el]
                FIXED_RES[i, j] = sim[-1, s_idx]
                idx += 1

        # Per-species statistics using only final simulation values.
        # Non-finite values are dropped so failed/invalid runs do not skew metrics.
        species_labels = list(valid_idxs.keys())
        eps = 1e-12
        rows = []
        distribution_plot_dir = os.path.join(out_dirs["images"], "distribution_transport")
        plotted_species_count = 0

        for j, species in enumerate(species_labels):
            fixed_vals = FIXED_RES[:, j]
            sampled_vals = RES[:, j]

            # Keep only finite final values for robust comparisons.
            fixed_vals = fixed_vals[np.isfinite(fixed_vals)]
            sampled_vals = sampled_vals[np.isfinite(sampled_vals)]

            n_fixed = int(fixed_vals.size)
            n_sampled = int(sampled_vals.size)

            fixed_mean = np.mean(fixed_vals) if n_fixed > 0 else np.nan
            sampled_mean = np.mean(sampled_vals) if n_sampled > 0 else np.nan

            # Sample std (ddof=1) estimates spread better than population std for finite samples.
            fixed_std = np.std(fixed_vals, ddof=1) if n_fixed > 1 else np.nan
            sampled_std = np.std(sampled_vals, ddof=1) if n_sampled > 1 else np.nan

            fixed_median = np.median(fixed_vals) if n_fixed > 0 else np.nan
            sampled_median = np.median(sampled_vals) if n_sampled > 0 else np.nan

            fixed_iqr = np.percentile(fixed_vals, 75) - np.percentile(fixed_vals, 25) if n_fixed > 0 else np.nan
            sampled_iqr = np.percentile(sampled_vals, 75) - np.percentile(sampled_vals, 25) if n_sampled > 0 else np.nan

            # Signed differences: positive means sampled > fixed.
            delta_mean = sampled_mean - fixed_mean
            delta_std = sampled_std - fixed_std
            delta_median = sampled_median - fixed_median
            delta_iqr = sampled_iqr - fixed_iqr

            # Relative change in mean concentration (%), stabilized around zero baseline.
            relative_mean_change_pct = (
                (delta_mean / (abs(fixed_mean) + eps)) * 100.0
                if np.isfinite(delta_mean)
                else np.nan
            )

            # Standardized mean difference (Cohen's d) using pooled variance.
            pooled_std = np.nan
            if n_fixed > 1 and n_sampled > 1 and np.isfinite(fixed_std) and np.isfinite(sampled_std):
                pooled_var = (
                    ((n_fixed - 1) * (fixed_std ** 2)) + ((n_sampled - 1) * (sampled_std ** 2))
                ) / (n_fixed + n_sampled - 2)
                pooled_std = np.sqrt(pooled_var)

            cohen_d = (
                delta_mean / pooled_std
                if np.isfinite(pooled_std) and pooled_std > eps
                else np.nan
            )

            # Distribution-level shift: Earth Mover distance in original concentration units.
            distribution_shift_w1 = (
                ut.wasserstein_1d(fixed_vals, sampled_vals)
                if n_fixed > 0 and n_sampled > 0
                else np.nan
            )

            # Save a transport-style distribution visualization for this species.
            # It shows fixed/sample marginals and one quantile-based coupling density.
            plot_path = sens_ut.plot_distribution_transport_map(
                fixed_vals,
                sampled_vals,
                species_name=species,
                output_dir=distribution_plot_dir,
                log_file=parsed_args["log_file"],
            )
            if plot_path is not None:
                plotted_species_count += 1

            rows.append(
                {
                    "Species": species,
                    "N_Fixed": n_fixed,
                    "N_Sampled": n_sampled,
                    "Fixed_Final_Mean": fixed_mean,
                    "Fixed_Final_Std": fixed_std,
                    "Fixed_Final_Median": fixed_median,
                    "Fixed_Final_IQR": fixed_iqr,
                    "Sampled_Final_Mean": sampled_mean,
                    "Sampled_Final_Std": sampled_std,
                    "Sampled_Final_Median": sampled_median,
                    "Sampled_Final_IQR": sampled_iqr,
                    "Delta_Final_Mean": delta_mean,
                    "Delta_Final_Std": delta_std,
                    "Delta_Final_Median": delta_median,
                    "Delta_Final_IQR": delta_iqr,
                    "Relative_Mean_Change_Pct": relative_mean_change_pct,
                    "Cohens_d": cohen_d,
                    "Wasserstein_Distance": distribution_shift_w1,
                }
            )

        # Save the results to a CSV file
        results_df = pd.DataFrame(rows)
        results_df.to_csv(os.path.join(out_dirs["csv"], "sensitivity_comparison.csv"), index=False)
        ut.print_log(parsed_args["log_file"], f"[INFO] Sensitivity comparison results saved to: {out_dirs['csv']}")
        ut.print_log(
            parsed_args["log_file"],
            f"[INFO] Distribution transport plots saved for {plotted_species_count}/{len(species_labels)} species in: {distribution_plot_dir}",
        )
            
