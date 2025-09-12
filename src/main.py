#!/usr/bin/env python
import enum
from logging import raiseExceptions
from math import log, nan
import os
import re
import sys
from traceback import print_tb
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.assortativity import correlation
from numpy._typing import _UnknownType
from pandas._libs import iNaT
from pandas.core.generic import pprint_thing
from classes.SBMLHandler import SBMLHandler
import roadrunner
import libsbml
import numpy as np
from dotenv import load_dotenv
import time
import pandas as pd
from pathlib import Path

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from SALib.util import ProblemSpec

from scipy.special import factorial


# Add the src folder path to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as sbml_ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu
from src.utils import utils as ut
from src.utils import plot_utils as plt_ut
from src.utils import sens_utils as sens_ut


def main():
    # load_dotenv()
    # Use the parse_args function to get command-line arguments
    args = ut.parse_args()

    # Setup logging
    log_file = args.log if hasattr(args, "log") else None
    file_path = Path(args.input_path)

    if not file_path.exists():
        ut.print_log(log_file, f"[ERROR] The specified model doesn't exists")

    file_name = os.path.splitext(os.path.basename(args.input_path))[0]

    try:
        if args.command == "simulate":

            # Extracting the paramters
            simulation_time = args.time
            integrator = args.integrator
            use_steady_state = args.steady_state
            ss_max_time = args.max_time
            ss_sim_steps = args.sim_step
            sim_points = args.points
            ss_threshold = args.threshold
            interactive_plot = args.interactive

            # Load the model
            sbml_doc = sbml_ut.load_model(args.input_path)

            # sbml_model = sbml_doc.getModel()
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_doc.getModel())

            ut.print_log(log_file, f"Simulating model: {file_name}")

            # Load and simulate
            if integrator:
                rr = sim_ut.load_roadrunner_model(
                    sbml_model, integrator=integrator, log_file=log_file
                )
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

            res, ss_time, colnames = sim_ut.simulate(
                rr_model=rr,
                start_time=0,
                end_time=simulation_time,
                steady_state=use_steady_state,
                max_end_time=ss_max_time,
                sim_step=ss_sim_steps,
                threshold=ss_threshold,
                log_file=log_file,
            )

            res_df = pd.DataFrame(res, columns=colnames)

            save_path = f"{args.save_images}/{file_name}"

            if interactive_plot:
                plt_ut.plot_results_interactive(
                    res,
                    colnames,
                    model_name=file_name,
                    log_file=log_file,
                )

            else:
                plt_ut.plot_results(
                    res, colnames, save_path, "interactive_model_simulation", log_file
                )

        elif args.command == "importance_assessment":

            # Getting the input species ids
            input_species_ids = args.input_species

            use_perturbations = args.use_perturbations
            use_fixed_perturbations = args.use_fixed_perturbations

            __import__("pprint").pprint(
                (use_fixed_perturbations and args.fixed_perturbations is not None)
            )

            # Check for consistency
            if use_fixed_perturbations:
                if args.fixed_perturbations is None:
                    ut.print_log(
                        log_file,
                        "[ERROR] --use_fixed_perturbations and --fixed_perturbations must be both present to be used",
                    )
                else:
                    fixed_perturbations = [int(p) for p in args.fixed_perturbations]

            num_samples = args.num_samples
            variation = args.variation
            simulation_time = args.time
            integrator = args.integrator
            use_steady_state = args.steady_state
            ss_max_end_time = args.max_time
            ss_min_time = args.max_time
            ss_sim_steps = args.sim_step
            sim_points = args.points
            ss_threshold = args.threshold

            preserve_inputs = args.preserve_inputs

            # Load the model
            sbml_doc = sbml_ut.load_model(args.input_path)

            sbml_model = sbml_doc.getModel()

            # Split all the reversible reactions in two different reactions
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            # Getting the model name
            # file_name = os.path.basename(args.input_path)

            # Get the list of species
            species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

            if args.knockout is None:
                # If no input species are provided all nodes will be used
                ids_to_ko = [s.getId() for s in sbml_model.getListOfSpecies()]
            else:
                # Else only the provided ones will be used
                ids_to_ko = list(set(species_list).intersection(set(args.knockout)))

            if args.preserve_inputs:
                # Removing the input species from the species to analyze
                ids_to_ko = list(set(ids_to_ko) - set(input_species_ids))

            id_to_idx = {}

            for indx, s in enumerate(species_list):
                id_to_idx[s] = indx

            ids_to_ko.sort(key=lambda x: id_to_idx[x])

            # Init the combinations array
            combinations = None

            if use_perturbations:
                if use_fixed_perturbations:  # args.fixed_perturbations is None:
                    ut.print_log(log_file, "Running with fixed samples generations")

                    combinations = sbml_ut.get_fixed_combinations(
                        sbml_model=sbml_model,
                        input_species=input_species_ids,
                        fixed_variations=fixed_perturbations,
                        log_file=log_file,
                    )

                else:
                    ut.print_log(log_file, "Running with random samples generations")

                    # Create the combinations
                    combinations = sbml_ut.generate_species_random_combinations(
                        sbml_model,
                        target_species=input_species_ids,
                        log_file=log_file,
                        n_samples=num_samples,
                        variation=variation,
                    )

            else:
                ut.print_log(log_file, "Running without samples generations")
            # Load the model
            rr = sim_ut.load_roadrunner_model(
                sbml_model, integrator=integrator, log_file=log_file
            )

            # Get the selections
            selections = rr.timeCourseSelections
            for s in species_list:
                if f"[{s}]" not in selections:
                    selections.append(f"[{s}]")

            # Add also the reactions in the selections if target
            for ts in ids_to_ko:
                if ts in sbml_model.getListOfReactions().getId():
                    selections.append(f"{ts}")

            rr.timeCourseSelections = selections

            # Simulate the model
            ut.print_log(log_file, "Simulating original model")
            original_results, ss_time, colnames = sim_ut.simulate(
                rr,
                end_time=args.time,
                start_time=0,
                log_file=log_file,
                steady_state=use_steady_state,
                max_end_time=ss_max_end_time,
            )

            original_df = pd.DataFrame(original_results[:, 1:], columns=colnames[1:])

            colnames_to_index = {}
            for i, el in enumerate(colnames):
                if el == "time":
                    continue
                colnames_to_index[el] = i

            # ut.print_log("orig_res", original_results[:, :])

            min_ss_time = (
                ss_time
                if ss_time is not None and ss_time <= ss_min_time
                else ss_min_time
            )

            if use_perturbations:
                if use_fixed_perturbations:
                    ut.print_log(
                        log_file, "Simulating original model with fixed perturbations"
                    )

                    samples_simulations_results, _ = sim_ut.simulate_combinations(
                        rr,
                        combinations,
                        input_species_ids,
                        ss_min_time,
                        simulation_time,
                        ss_max_end_time,
                        use_steady_state,
                        log_file,
                    )
                else:
                    # Simulate the original model with samples
                    ut.print_log(
                        log_file, "Simulating original model with random samples"
                    )
                    samples_simulations_results, _ = sim_ut.simulate_combinations(
                        rr,
                        combinations,
                        input_species_ids,
                        ss_min_time,
                        simulation_time,
                        ss_max_end_time,
                        use_steady_state,
                        log_file,
                    )

                ut.print_log(
                    log_file, "Getting simulations informations of the original model"
                )

                original_data = [original_df]

                for i in range(len(combinations)):
                    sim_res_i = samples_simulations_results[i]
                    original_data.append(
                        pd.DataFrame(sim_res_i[:, 1:], columns=colnames[1:])
                    )

                ut.print_log(log_file, f"{len(species_list)}")

            knockout_data = []

            # Copying the SBML model
            sbml_str = libsbml.writeSBMLToString(sbml_doc)

            # Create a dict of the ko models
            model_dict = sbml_ut.create_ko_models(
                ids_to_ko, sbml_model, sbml_str, log_file
            )

            # counter = 0

            start_time = time.perf_counter()

            knockout_data = sim_ut.process_species_multiprocessing(
                ids_to_ko,
                model_dict,
                combinations,
                input_species_ids,
                selections,
                integrator,
                start_time=0,
                end_time=simulation_time,
                steady_state=use_steady_state,
                max_end_time=ss_max_end_time,
                min_ss_time=ss_min_time,
                log_file=log_file,
                use_perturbations=use_perturbations,
                preserve_input=preserve_inputs,
            )

            end_time = time.perf_counter()

            # === SHAPLEY VALUE ===
            if use_perturbations:
                payoff_dict = sim_ut.get_payoff_vals(
                    original_data,  # pyright:ignore
                    knockout_data,
                    colnames[1:],
                    log_file=log_file,  # pyright:ignore
                )

                n_combinations = np.power(num_samples, len(input_species_ids)) + 1

                shap_values = sim_ut.get_shapley_values(
                    payoff_dict,
                    n_combinations,
                    len(input_species_ids),
                    log_file=log_file,
                )

                if args.generate_report is not None:
                    save_path = args.generate_report
                    # Create the directory if it doesn't exist
                    os.makedirs(save_path, exist_ok=True)

                    if args.target_nodes:
                        cols = args.target_nodes
                    else:
                        cols = None

                    ut.save_shapley_values_to_csv_pivot(
                        shap_values, save_path, cols=cols, log_file=log_file
                    )

            # === IF NO SAMPLES ===
            if not use_perturbations:

                ut.print_log(log_file, "Printing without samples")

                relative_vars = sim_ut.get_relative_variations_no_samples(
                    original_df,
                    knockout_data,
                    log_file=log_file,
                )

                absolute_vars = sim_ut.get_absolute_variations_no_samples(
                    original_df, knockout_data, log_file=log_file
                )

                # Normalizing with logs

                log_relative = np.log10(relative_vars + 1)
                log_absolute = np.log10(absolute_vars + 1)

                if args.generate_report is not None:

                    saving_path = (
                        args.generate_report
                    )  # Check if the destinations folder exists otherwise create it
                    if not os.path.exists(saving_path):
                        os.makedirs(saving_path, exist_ok=True)
                        ut.print_log(log_file, f"Created directory: {saving_path}")

                    relative_vars.to_csv(f"{saving_path}/relative_variations.csv")

                    absolute_vars.to_csv(f"{saving_path}/absolute_variations.csv")

                if args.save_images is not None:

                    saving_path = args.save_images

                    os.makedirs(saving_path, exist_ok=True)

                    plt_ut.plot_heatmap(
                        log_relative,
                        log_relative.index.tolist(),
                        log_relative.columns.tolist(),
                        colnames_to_index=colnames_to_index,
                        title="Relative variations without perturbations",
                        save_path=f"{saving_path}/No Samples",
                        img_name="Relative variations Heatmap.png",
                    )

                    plt_ut.plot_heatmap(
                        log_absolute,
                        log_absolute.index.tolist(),
                        log_absolute.columns.tolist(),
                        colnames_to_index=colnames_to_index,
                        cmap="plasma",
                        title="Absolute variations without perturbations",
                        save_path=f"{saving_path}/No Samples",
                        img_name="Absolute variations Heatmap.png",
                    )

            else:  # === IF SAMPLES ===
                try:
                    assert original_data is not None  # pyright:ignore
                    ut.print_log(log_file, "Printing with samples")
                    ut.print_log(
                        log_file,
                        f"Time to process species: {(end_time-start_time):.2f}s",
                    )

                    # Getting the variations with no samples

                    # Getting the ko without samples
                    ko_no_samples = []
                    for ko_species, ko_data in knockout_data:
                        ko_or = ko_data[0]

                        ko_no_samples.append((ko_species, ko_or))

                    # Taking the first original
                    ori_no_samples = original_data[0]

                    # Getting the variations without samples
                    no_samples_relative_vars = (
                        sim_ut.get_relative_variations_no_samples(
                            ori_no_samples, ko_no_samples, log_file=log_file
                        )
                    )

                    no_samples_absolute_vars = (
                        sim_ut.get_absolute_variations_no_samples(
                            ori_no_samples, ko_no_samples, log_file=log_file
                        )
                    )

                    # Getting the variations with samples
                    samples_relative_vars = sim_ut.get_relative_variations_samples(
                        original_data, knockout_data
                    )

                    samples_absolute_vars = sim_ut.get_absolute_variations_samples(
                        original_data, knockout_data, log_file=log_file
                    )

                    # Normalizing the variations with log
                    log_no_samples_relative = np.log10(no_samples_relative_vars + 1)
                    log_no_samples_absolute = np.log10(no_samples_absolute_vars + 1)

                    log_samples_relative = np.log10(samples_relative_vars + 1)
                    log_samples_absolute = np.log10(samples_absolute_vars + 1)

                    # Normalizing using minMax
                    # minMax_no_samples_relative = ut.minMax_normalize(
                    #     log_no_samples_relative
                    # )
                    # minMax_no_samples_absolute = ut.minMax_normalize(
                    #     log_no_samples_absolute
                    # )
                    #
                    # minMax_samples_relative = ut.minMax_normalize(
                    #     (log_samples_relative)
                    # )
                    # minMax_samples_absolute = ut.minMax_normalize(log_samples_absolute)

                    if args.save_images is not None:
                        saving_path = args.save_images

                        os.makedirs(saving_path, exist_ok=True)

                        # PLOTTING HEATMAPS
                        plt_ut.plot_heatmap(
                            log_samples_relative,
                            log_samples_relative.index.tolist(),
                            log_no_samples_relative.columns.tolist(),
                            colnames_to_index,
                            title="Relative variations with perturbations (Log scaled)",
                            save_path=saving_path,
                            img_name="Log scaled relative variations Heatmap.png",
                        )

                        plt_ut.plot_heatmap(
                            log_samples_absolute,
                            log_samples_absolute.index.tolist(),
                            log_samples_absolute.columns.tolist(),
                            colnames_to_index,
                            cmap="plasma",
                            title="Absolute variations with perturbations (Log scaled)",
                            save_path=saving_path,
                            img_name="Log scaled absolute variations Heatmap.png",
                        )

                    # SAVING THE VARAITIONS HEATMAPS
                    if args.generate_report:  # Saving the variation heatmaps

                        save_path = args.generate_report

                        os.makedirs(save_path, exist_ok=True)

                        samples_relative_vars.to_csv(
                            f"{save_path}/relative_variations.csv"
                        )

                        samples_absolute_vars.to_csv(
                            f"{save_path}/absolute_variations.csv"
                        )

                    _, ko_ranking_relative = ut.get_ko_species_importance(
                        samples_relative_vars,
                        samples_relative_vars.index.tolist(),
                        log_file=log_file,
                    )

                    _, ko_ranking_absolute = ut.get_ko_species_importance(
                        samples_absolute_vars,
                        samples_absolute_vars.index.tolist(),
                        log_file,
                    )

                    if args.perturbations_importance:
                        ut.print_log(log_file, "=== IMPORTANCE ANALYSIS ===")

                        # Setting the correlation type
                        correlation_type = "two-sided"

                        # GETTING THE RELATIVE VALUES DISTANCES
                        relative_values_distances = np.abs(
                            log_samples_relative - log_no_samples_relative
                        )

                        absolute_values_distances = np.abs(
                            log_samples_absolute - log_no_samples_absolute
                        )

                        __import__("pprint").pprint(log_samples_absolute)
                        __import__("pprint").pprint(log_no_samples_absolute)
                        __import__("pprint").pprint(absolute_values_distances)

                        # GETTING THE PEARSON COEFFICIENT
                        pearson_coefficient_relative, p_value_relative = (
                            ut.pearson_correlation(
                                log_samples_relative, log_no_samples_relative
                            )
                        )

                        pearson_coefficient_absolute, p_value_absolute = (
                            ut.pearson_correlation(
                                log_samples_absolute,
                                log_no_samples_absolute,
                            )
                        )

                        if args.generate_report:
                            saving_path = args.generate_report

                            os.makedirs(saving_path, exist_ok=True)

                            # GETTING THE VALUE DISTANCES REPORT
                            _ = sim_ut.generate_values_distance_report(
                                relative_values_distances,
                                pearson_coefficient_relative,
                                p_value_relative,
                                correlation_type,
                                samples_relative_vars.index.tolist(),
                                file_name,
                                f"{saving_path}/Perturbations importance analysis",
                                report_title="Log scaled value distances report analysis (Relative map)",
                                log_file=log_file,
                            )

                            _ = sim_ut.generate_values_distance_report(
                                absolute_values_distances,
                                pearson_coefficient_absolute,
                                p_value_absolute,
                                correlation_type,
                                samples_absolute_vars.index.tolist(),
                                file_name,
                                f"{saving_path}/Perturbations importance analysis",
                                report_title="Log scaled value distances report analysis (Absolute map)",
                                log_file=log_file,
                            )

                        if args.save_images is not None:
                            saving_path = args.save_images

                            os.makedirs(saving_path, exist_ok=True)

                            # PLOTTING THE VALUES DISTANCES
                            plt_ut.plot_heatmap(
                                relative_values_distances,
                                samples_relative_vars.index.tolist(),
                                samples_relative_vars.columns.tolist(),
                                colnames_to_index,
                                cmap="PiYG_r",
                                save_path=f"{saving_path}/Perturbations importance analysis",
                                title="Perturbations VS No Perturbations Distance (Relative map)",
                                img_name="Relative Perturbations VS No Perturbations distance",
                            )

                            plt_ut.plot_heatmap(
                                absolute_values_distances,
                                samples_absolute_vars.index.tolist(),
                                samples_absolute_vars.index.tolist(),
                                colnames_to_index,
                                cmap="PRGn_r",
                                save_path=f"{saving_path}/Perturbations importance analysis",
                                title="Perturbations VS No Perturbations Distance (Absolute map)",
                                img_name="Absolute Perturbations VS No Perturbations distance",
                            )

                    if args.random_perturbations_importance:
                        ut.print_log(log_file, "=== FIXED SAMPLES ANALYSIS ===")
                        correlation_type = "two-sided"

                        ut.print_log(
                            log_file,
                            "[WARNING] Notice that the number of samples will equals to the number of variations",
                        )

                        if args.fixed_perturbations is None:
                            ut.print_log(
                                log_file,
                                "[ERROR] You need to specify the variations to use setting the -fp parameter",
                            )

                            exit(1)

                        # env_pert = os.getenv("FIXED_PERTURBATIONS").split(
                        #     ","
                        # )  # pyright:ignore
                        #
                        # if env_pert is not None:
                        #     fixed_variations = [np.float64(v) for v in env_pert]
                        # else:
                        #     raise TypeError("Perturbations are None")

                        fp = [int(p) for p in args.fixed_perturbations]

                        ut.print_log(log_file, "Generating fixed combinations")

                        fixed_combinations = sbml_ut.get_fixed_combinations(
                            sbml_model,
                            input_species_ids,
                            fp,
                            log_file,
                        )

                        fixed_samples_results, _ = sim_ut.simulate_combinations(
                            rr,
                            fixed_combinations,
                            input_species_ids,
                            min_ss_time,
                            end_time,
                            ss_max_end_time,
                            use_steady_state,
                            log_file,
                        )

                        fixed_original_data = [
                            pd.DataFrame(original_results[:, 1:], columns=colnames[1:])
                        ]

                        for i in range(len(fixed_combinations)):
                            fixed_original_data.append(
                                pd.DataFrame(
                                    fixed_samples_results[i][:, 1:],
                                    columns=colnames[1:],
                                )
                            )

                        ut.print_log(
                            log_file,
                            " Simulating multiporcessing with fixed samples",
                        )

                        fixed_knockout_data = sim_ut.process_species_multiprocessing(
                            ids_to_ko,
                            model_dict,
                            fixed_combinations,
                            input_species_ids,
                            selections,
                            integrator,
                            start_time=0,
                            end_time=end_time,
                            steady_state=use_steady_state,
                            max_end_time=ss_max_end_time,
                            min_ss_time=min_ss_time,
                            log_file=log_file,
                            preserve_input=args.preserve_inputs,
                            use_perturbations=args.use_perturbations,
                        )

                        fixed_relative_vars = sim_ut.get_relative_variations_samples(
                            fixed_original_data, fixed_knockout_data, log_file=log_file
                        )

                        fixed_absolute_vars = sim_ut.get_absolute_variations_samples(
                            fixed_original_data, fixed_knockout_data, log_file=log_file
                        )

                        # NORMALIZING WITH LOGS
                        fixed_log_relative = np.log10(fixed_relative_vars + 1)
                        fixed_log_absolute = np.log10(fixed_absolute_vars + 1)

                        # NORMALIZING WITH MINMAX
                        normalized_fixed_relative = ut.minMax_normalize(
                            fixed_log_relative
                        )
                        normalized_fixed_absolute = ut.minMax_normalize(
                            fixed_log_absolute
                        )

                        # GETTING THE DISTANCES
                        fixed_relative_values_distances = np.abs(
                            log_samples_relative - fixed_log_relative
                        )

                        fixed_absolute_values_distances = np.abs(
                            log_samples_absolute - fixed_log_absolute
                        )

                        # GETTING THE PEARSON COEFFICIENT

                        (
                            fixed_pearson_coefficient_relative,
                            fixed_p_value_relative,
                        ) = ut.pearson_correlation(
                            log_samples_relative, fixed_log_relative
                        )

                        (
                            fixed_pearson_coefficient_absolute,
                            fixed_p_value_absolute,
                        ) = ut.pearson_correlation(
                            log_samples_absolute, fixed_log_absolute
                        )

                        if args.generate_report is not None:

                            saving_path = args.generate_report

                            os.makedirs(saving_path, exist_ok=True)

                            # GENERATING DISTANCES REPORT
                            _ = sim_ut.generate_values_distance_report(
                                fixed_relative_values_distances,
                                fixed_pearson_coefficient_relative,
                                fixed_p_value_relative,
                                correlation_type,
                                fixed_relative_vars.index.tolist(),
                                file_name,
                                f"{saving_path}/Fixed perturbations importance analysis",
                                report_title="Log scaled Fixed perturbations values distance report analysis (Relative map)",
                                log_file=log_file,
                            )

                            _ = sim_ut.generate_values_distance_report(
                                fixed_absolute_values_distances,
                                fixed_pearson_coefficient_absolute,
                                fixed_p_value_absolute,
                                correlation_type,
                                fixed_absolute_vars.index.tolist(),
                                file_name,
                                f"{saving_path}/Fixed perturbations importance analysis",
                                report_title="Log scaled Fixed perturbations values distance report analysis (Absolute map)",
                                log_file=log_file,
                            )

                        if args.save_images:
                            saving_path = args.save_images

                            os.makedirs(saving_path, exist_ok=True)

                            # PLOTTING VALUES DISTANCES HEATMAPS
                            plt_ut.plot_heatmap(
                                fixed_relative_values_distances,
                                fixed_relative_vars.index.tolist(),
                                species_list,
                                colnames_to_index=colnames_to_index,
                                cmap="cividis",
                                save_path=f"{saving_path}/Random Perturbations Importance Analysis",
                                title="Random Perturbations VS Fixed Perturbations Distance (Relative map)",
                                img_name="Relative Random Perturbations VS Fixed Perturbations distance",
                            )

                            plt_ut.plot_heatmap(
                                fixed_absolute_values_distances,
                                fixed_absolute_vars.index.tolist(),
                                species_list,
                                colnames_to_index=colnames_to_index,
                                cmap="PuOr_r",
                                save_path=f"{saving_path}/Random Perturbations Importance Analysis",
                                title="Random Perturbations VS Fixed Perturbations Distance (Absolute map)",
                                img_name="Absolute Random Perturbations VS Fixed Perturbations distance",
                            )

                except AssertionError as ae:
                    ut.print_log(
                        log_file, f"Error during samples results elaboration: {ae}"
                    )

        elif args.command == "sensitivity_analysis":

            saving_path = f"./report/{file_name}/"

            if not os.path.exists(saving_path):
                os.makedirs(saving_path, exist_ok=True)
                ut.print_log(log_file, "Repository successfully cerated")

            # LOADING MODEL
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()

            # SPLITTING REVERSIBLE REACTIONS
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            # file_name = os.path.basename(args.input_path)

            # GETTING INPUT SPECIES IDS
            input_ids = args.input_species
            all_species_ids = []

            for s in sbml_model.getListOfSpecies():
                if not s.getConstant():
                    all_species_ids.append(s.getId())
                else:
                    continue

            if args.preserve_inputs:

                internal_nodes = list(set(all_species_ids) - set(input_ids))
            else:
                internal_nodes = list(set(all_species_ids))

            rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

            res_dict = {}
            needed_selections = [f"[{n}]" for n in internal_nodes]
            selections = rr.timeCourseSelections
            for sel in needed_selections:
                if sel not in selections:
                    selections.append(sel)
            rr.timeCourseSelections = selections

            # Mappa coerente selezione -> indice in rr.selections
            selection_to_idx = {
                sel: rr.timeCourseSelections.index(sel) for sel in needed_selections
            }

            # Limita ai soli nodi interni effettivamente disponibili
            available_pairs = [
                (n, f"[{n}]") for n in internal_nodes if f"[{n}]" in selection_to_idx
            ]
            available_internal_nodes = [n for (n, _) in available_pairs]
            available_needed_selections = [sel for (_, sel) in available_pairs]

            ut.print_log(
                log_file, f"Available internal nodes: {available_needed_selections}"
            )

            problem = sens_ut.get_problem_parameters(
                sbml_model, len(input_ids), input_ids, 20, log_file=log_file
            )

            spec = ProblemSpec(problem)

            c_sample_size = None

            if args.check_convergence:
                sample_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

                informations_dict = {}

                ut.print_log(log_file, "Starting convergence analysis")

                for i, size in enumerate(sample_sizes):
                    ut.print_log(log_file, f" Running  with {size} samples")
                    start = time.perf_counter()
                    params = sobol_sample.sample(problem, sample_sizes[i])

                    RES_i = sens_ut.run_simulation_with_params(
                        rr,
                        params,
                        available_needed_selections,
                        selection_to_idx,
                        input_ids,
                        log_file,
                    )

                    # cv = sens_ut.assess_model_linearity(RES, size)

                    res_dict_conv = {}

                    for j, node_id in enumerate(available_internal_nodes):
                        Si = sobol_analyze.analyze(problem, RES_i[:, j])
                        res_dict_conv[node_id] = Si

                    informations_dict[size] = res_dict_conv

                    end = time.perf_counter()

                    ut.print_log(
                        log_file, f"{size} samples analyzed in {(end - start)}s"
                    )
                    ut.print_log(log_file, 40 * "=")

                convergence_informations = sens_ut.check_convergence(
                    informations_dict,
                    available_internal_nodes,
                    min_consecutive=2,
                    relative=False,
                )

                for node, node_info in convergence_informations.items():
                    if node_info["converged_at"] is None:
                        c_sample_size = 4096
                        break
                    else:
                        if c_sample_size is None:
                            c_sample_size = 64
                        if node_info["converged_at"] > c_sample_size:
                            c_sample_size = node_info["converged_at"]

                sens_ut.convergence_report(
                    convergence_informations,
                    f"./report/{file_name}/convergence_analysis",
                )

                sens_ut.plot_convergence_single_plot(
                    convergence_informations, file_name=file_name
                )

            # Creating the problem
            #
            base_samples = args.base_samples if c_sample_size is None else c_sample_size

            __import__("pprint").pprint(base_samples)

            params = spec.sample(sobol_sample.sample, base_samples).samples

            # RES = np.zeros([params.shape[0], len(internal_nodes)])

            start = time.perf_counter()

            # __import__("pprint").pprint(f"idxs: {valid_idxs}")

            RES = sens_ut.run_simulation_with_params(
                rr,
                params,
                available_needed_selections,
                selection_to_idx,
                input_ids,
                log_file,
            )

            end = time.perf_counter()

            ut.print_log(log_file, f"Time: {end - start}")
            # Running the fixed simulations

            fp = [int(p) for p in args.fixed_perturbations]

            fixed_combinations = sbml_ut.get_fixed_combinations(
                sbml_model, input_ids, fp, log_file
            )

            # __import__("pprint").pprint(fixed_combinations)

            fixed_samples_results, _ = sim_ut.simulate_combinations(
                rr,
                fixed_combinations,
                input_ids,
                1000,
                5000,
                5000,
                False,
                log_file,
            )

            FIXED_RES = np.zeros(
                [
                    np.array(fixed_combinations).shape[0],
                    len(available_needed_selections),
                ]
            )

            for i, fc in enumerate(fixed_combinations):
                sim = fixed_samples_results[i]
                idx = 0

                for j, el in enumerate(available_needed_selections):
                    s_idx = selection_to_idx[el]
                    FIXED_RES[i, j] = sim[-1, s_idx]
                    idx += 1

            __import__("pprint").pprint(f"{RES.shape} | {FIXED_RES.shape}")

            fixed_mean = np.mean(FIXED_RES, axis=0)
            fixed_std = np.std(FIXED_RES, axis=0)

            random_std = np.std(RES, axis=0)
            random_mean = np.mean(RES, axis=0)

            # mean_diff = np.abs(fixed_mean - random_mean)
            mean_diff = (fixed_mean - random_mean) / fixed_mean

            rms_diff = np.sqrt(np.nanmean(mean_diff**2))
            ut.print_log(
                f"{saving_path}/RandomVsFixed infos",
                f"Max: {np.nanmax(mean_diff)} | Min:{np.nanmin(mean_diff)} | Avg: {rms_diff} | Std. Dev: {np.nanstd(mean_diff)}",
            )

            # __import__("pprint").pprint(
            #     f"Max: {np.nanmax(mean_diff)} | Min{np.nanmin(mean_diff)} | Avg: {np.nanmean(mean_diff)} | Std. Dev: {np.std(mean_diff)}"
            # )

            normalized_diff = mean_diff / random_std

            num_nodes = len(available_needed_selections)

            # sens_ut.convergence_analysis(RES, FIXED_RES, num_nodes)
            sens_ut.statistical_tests(RES, FIXED_RES, num_nodes)

            __import__("pprint").pprint(
                f"fixed mean: {fixed_mean} | random_mean{random_mean} | mean diff: {mean_diff} | normalized diff: {normalized_diff}"
            )

            for j, node_id in enumerate(available_internal_nodes):

                Si = sobol_analyze.analyze(problem, RES[:, j])
                res_dict[node_id] = Si

                # __import__("pprint").pprint(res_dict)

                sens_ut.report_sensitivity(
                    res_dict, input_ids, f"./report/{file_name}/sens_analysis"
                )

        elif args.command == "knockout_species":
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()

            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            # file_name = os.path.basename(args.input_path)

            # FOR DEBUG
            # for specie in sbml_model.getListOfSpecies():
            #     sbml_ut.check_presence(sbml_model, specie.getId(), log_file)

            ut.print_log(
                log_file,
                f"Knocking out species {args.species_id} in model: {file_name}",
            )

            # Inhibit the species
            modified_model = sbml_ut.knockout_species(
                sbml_model, args.species_id, log_file
            )

            # modified_model, r = sbml_ut.knockout_species_via_reaction(
            #     sbml_model, args.species_id, log_file
            # )

            # if r is None:
            #     raise Exception("Knockout species failed")

            operation_name = f"no_species_{args.species_id}"

            ut.print_log(
                log_file,
                f"Searching for updating operations on target species {args.species_id}",
            )

            # Save
            xml_string, output_filename = sbml_ut.save_file(
                file_name, operation_name, modified_model, True, log_file=log_file
            )

            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)

        elif args.command == "knockout_reaction":
            # Load the model
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()
            # file_name = os.path.basename(args.input_path)

            # sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            ut.print_log(
                log_file,
                f"Knocking out reaction {args.reaction_id} in model: {file_name}",
            )

            # Inhibit the reaction
            modified_model = sbml_ut.knockout_reaction(
                sbml_model, args.reaction_id, log_file
            )
            operation_name = f"no_reaction_{args.reaction_id}"

            # Save
            xml_string, output_filename = sbml_ut.save_file(
                file_name, operation_name, modified_model, True, log_file
            )

            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)
        elif args.command == "create_petrinet":
            print(
                "Hi!\n This command is still under construction...\n Thanks for your patience."
            )
            exit(1)
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path).getModel()
            dir_name = os.path.dirname(args.input_path)
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)
            # file_name = os.path.basename(args.input_path)

            species_list = sbml_ut.get_list_of_species(sbml_model)
            reactions_list = sbml_ut.get_list_of_reactions(
                sbml_model, sbml_ut.get_species_dict(species_list)
            )

            ut.print_log(log_file, f"Simulating model: {file_name}")

            rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

            N = nu.get_network_from_sbml(reactions_list, species_list, log_file)
            nu.plot_network(
                graph=N,
                img_dir_path=args.output,
                img_name=f"{os.path.splitext(file_name)[0]}_network",  # Nome del file senza estensione + _network
                log_file=log_file,
            )

    except Exception as e:
        ut.print_log(log_file, f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
