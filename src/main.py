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

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from SALib.util import ProblemSpec


# Add the src folder path to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as sbml_ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu
from src.utils import utils as ut
from src.utils import plot_utils as plt_ut
from src.utils import sens_utils as sens_ut


def main():
    load_dotenv()
    # Use the parse_args function to get command-line arguments
    args = ut.parse_args()

    # Setup logging
    log_file = args.log if hasattr(args, "log") else None

    try:
        # Process based on command
        if args.command == "simulate":
            # Load the model
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_doc.getModel())
            file_name = os.path.basename(args.input_path)

            # sbml_ut.split_reversible_reactions(sbml_model, log_file)
            # sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            ut.print_log(log_file, f"Simulating model: {file_name}")

            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(
                    sbml_model, integrator=args.integrator, log_file=log_file
                )
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

            res, ss_time, colnames = sim_ut.simulate(
                rr,
                end_time=args.time,
                start_time=0,
                log_file=log_file,
                steady_state=args.steady_state,
                max_end_time=args.max_time,
            )

            ut.print_log(log_file, f"Colnames: {colnames}")

            # ut.print_log(log_file, rr["[Px]"])
            # ut.print_log(
            #     log_file,
            #     f"tolerances: abs {rr.integrator.absolute_tolerance} | rel {rr.integrator.relative_tolerance}",
            # )

            __import__("pprint").pprint(res[-1, :])

            # plt_ut.plot_results(
            #     res, colnames, args.output, file_name, log_file, ss_time
            # )
            #
            plt_ut.plot_results_interactive(res, colnames)

        elif args.command == "importance_assessment":

            steady_state = args.steady_state
            max_end_time = args.max_time
            min_ss_time = args.max_time
            ss_time = args.max_time
            end_time = args.time
            integrator = args.integrator

            # Load the model
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()

            # Split all the reversible reactions in two different reactions
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            file_name = os.path.basename(args.input_path)

            input_species_ids = args.input_species

            if args.target_ids is None:
                target_ids = [s.getId() for s in sbml_model.getListOfSpecies()]
            else:
                target_ids = args.target_ids

            if args.preserve_inputs:
                target_ids = list(set(target_ids) - set(input_species_ids))

            target_ids.sort()

            # Get the list of species
            species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

            combinations = None

            if args.use_perturbations:
                ut.print_log(log_file, "Running with samples generations")
                # Generate the samples
                samples = sbml_ut.generate_species_samples(
                    sbml_model,
                    target_species=input_species_ids,
                    log_file=log_file,
                    n_samples=args.num_samples,
                    variation=args.variation,
                )

                # Create the combinations
                combinations = sbml_ut.create_samples_combination(samples, log_file)
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
            for ts in target_ids:
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
                steady_state=args.steady_state,
                max_end_time=args.max_time,
            )

            # ut.print_log("orig_res", original_results[:, :])

            min_ss_time = (
                ss_time
                if ss_time is not None and ss_time <= min_ss_time
                else min_ss_time
            )

            if args.use_perturbations:
                # Simulate the original model with samples
                ut.print_log(log_file, "Simulating original model with samples")
                samples_simulations_results, _ = sim_ut.simulate_combinations(
                    rr,
                    combinations,
                    input_species_ids,
                    min_ss_time,
                    args.time,
                    args.max_time,
                    steady_state,
                    log_file,
                )

                ut.print_log(
                    log_file, "Getting simulations informations of the original model"
                )

                ut.print_log(log_file, f"{len(species_list)}")
                # Getting the information about the original model
                original_model_simulations_info = sim_ut.get_simulations_informations(
                    samples_simulations_results,
                    original_results,
                    combinations,
                    colnames,
                    log_file,
                )

                # for combination, info_dict in original_model_simulations_info.items():
                #     ut.print_log("orig", f"{combination}:")
                #     for species, values in info_dict.items():
                #         ut.print_log("orig", f" {species}: {values}")

            knockout_data = []

            # FOR DEBUG ONLY
            early_stop = 0

            # Copying the SBML model
            sbml_str = libsbml.writeSBMLToString(sbml_doc)

            # Create a dict of the ko models
            model_dict = sbml_ut.create_ko_models(
                target_ids, sbml_model, sbml_str, log_file
            )

            # counter = 0

            start_time = time.perf_counter()

            knockout_data = sim_ut.process_species_multiprocessing(
                target_ids,
                model_dict,
                combinations,
                input_species_ids,
                selections,
                integrator,
                start_time=0,
                end_time=end_time,
                steady_state=steady_state,
                max_end_time=max_end_time,
                min_ss_time=min_ss_time,
                log_file=log_file,
                use_perturbations=args.use_perturbations,
                preserve_input=args.preserve_inputs,
            )

            # for ko_species, comb_dict in knockout_data:
            #     ut.print_log("samples", f"{ko_species}:")
            #     for combination, info_dict in comb_dict.items():
            #         ut.print_log("samples", f"  {combination}:")
            #         for species, value in info_dict.items():
            #             ut.print_log("samples", f"      {species}: {value}")

            end_time = time.perf_counter()

            # === IF NO SAMPLES ===
            if not args.use_perturbations:

                ut.print_log(log_file, "Printing without samples")

                sim_variations = sim_ut.get_knockout_variation(
                    original_results, knockout_data, colnames, log_file
                )

                ko_species_list = list(sim_variations.keys())

                all_species = set()

                relative_map, all_species_rel = sim_ut.get_variations_hm_no_samples(
                    sim_variations, all_species, ko_species_list, log_file=log_file
                )

                abs_map, all_species_abs = sim_ut.get_variations_hm_no_samples(
                    sim_variations, all_species, ko_species_list, "absolute", log_file
                )

                log_rel_map = np.log(relative_map + 1)

                log_abs_map = np.log(abs_map + 1)

                plt_ut.plot_variations_heatmap(
                    log_rel_map,
                    all_species_rel,
                    ko_species_list,
                    title="Relative log variations heatmap",
                )

                plt_ut.plot_variations_heatmap(
                    log_abs_map,
                    all_species_abs,
                    ko_species_list,
                    variation_type="absolute",
                    title="Absolute log variations heatmap",
                )

                if args.save_output:
                    relative_data_frame = pd.DataFrame(
                        relative_map, columns=all_species_rel, index=ko_species_list
                    )
                    absolute_data_frame = pd.DataFrame(
                        abs_map, columns=all_species_abs, index=ko_species_list
                    )

                    relative_data_frame.to_csv(
                        f"./report/{file_name}/relative_variations.csv"
                    )
                    absolute_data_frame.to_csv(
                        f"./report/{file_name}/absolute_variations.csv"
                    )

            else:  # === IF SAMPLES ===
                try:
                    assert original_model_simulations_info is not None  # pyright:ignore
                    ut.print_log(log_file, "Printing with samples")
                    ut.print_log(
                        log_file,
                        f"Time to process species: {(end_time-start_time):.2f}s",
                    )

                    # Get the variation informations
                    variation_dict = sim_ut.get_knockout_variations_samples(
                        original_model_simulations_info,  # pyright:ignore
                        knockout_data,
                        log_file=None,
                    )

                    # Collect all species and knocked-out species
                    all_species = set()
                    ko_species_list = list(variation_dict.keys())

                    # Getting the informations about the original relative variations
                    no_samples_relative_map, all_species_rel = (
                        sim_ut.get_no_samples_variations(
                            variation_dict,
                            all_species,
                            ko_species_list,
                            variation_type="relative",
                            log_file=log_file,
                        )
                    )

                    no_samples_absolute_map, all_species_abs = (
                        sim_ut.get_no_samples_variations(
                            variation_dict,
                            all_species,
                            ko_species_list,
                            variation_type="absolute",
                            log_file=log_file,
                        )
                    )

                    # Getting the samples informations
                    relative_map, all_species_rel = sim_ut.get_variations_hm_samples(
                        variation_dict,
                        all_species,
                        ko_species_list,
                        variation_type="relative",
                        log_file=log_file,
                    )

                    absolute_map, all_species_abs = sim_ut.get_variations_hm_samples(
                        variation_dict,
                        all_species,
                        ko_species_list,
                        variation_type="absolute",
                        log_file=log_file,
                    )

                    # Normaliziing with logs
                    no_samples_log_relative_map = np.log(no_samples_relative_map + 1)
                    no_samples_log_absolute_map = np.log(no_samples_absolute_map + 1)

                    samples_log_relative_map = np.log(relative_map + 1)
                    samples_log_absolute_map = np.log(absolute_map + 1)

                    # Normalizing with minMax
                    normalized_no_samples_relative = ut.minMax_normalize(
                        no_samples_log_relative_map
                    )

                    normalized_no_samples_absolute = ut.minMax_normalize(
                        no_samples_log_absolute_map
                    )

                    normalized_samples_relative_map = ut.minMax_normalize(
                        samples_log_relative_map
                    )

                    normalized_samples_absolute_map = ut.minMax_normalize(
                        samples_log_absolute_map
                    )

                    # Heatmap plotting
                    plt_ut.plot_variations_heatmap(
                        samples_log_relative_map,
                        all_species_rel,
                        ko_species_list,
                        save_path=f"./imgs/{file_name}",
                        variation_type="relative",
                        title="Log relative variations with samples",
                    )

                    plt_ut.plot_variations_heatmap(
                        samples_log_absolute_map,
                        all_species_abs,
                        ko_species_list,
                        save_path=f"./imgs/{file_name}",
                        variation_type="absolute",
                        title="Log absolute variations with samples",
                    )

                    _, ko_ranking = ut.get_ko_species_importance(
                        samples_log_relative_map, ko_species_list, log_file=log_file
                    )

                    ut.print_log(log_file, "Top 5 species most sensitive to knock out:")
                    for i in range(min(5, len(ko_ranking))):
                        ko_idx = ko_ranking[i]
                        ko_name = ko_species_list[ko_idx]
                        impact_score = np.nanmax(samples_log_relative_map[ko_idx])
                        ut.print_log(
                            log_file, f"  {i+1}. {ko_name}: {impact_score:.20f}"
                        )

                    if args.save_output:  # Saving the variation heatmaps
                        relative_data_frame = pd.DataFrame(
                            relative_map, columns=all_species_rel, index=ko_species_list
                        )
                        absolute_data_frame = pd.DataFrame(
                            absolute_map, columns=all_species_abs, index=ko_species_list
                        )

                        relative_data_frame.to_csv(
                            f"./report/{file_name}/relative_variations.csv"
                        )
                        absolute_data_frame.to_csv(
                            f"./report/{file_name}/absolute_variations.csv"
                        )

                    if args.perturbations_importance:
                        ut.print_log(log_file, "Importance analysis")
                        # === SAMPLING IMPORTANCE ANALYSIS ===

                        correlation_type = "two-sided"

                        # Getting the values relative distance
                        relative_values_distance = np.abs(
                            samples_log_relative_map - no_samples_log_relative_map
                        )

                        # Getting the values absolute distance
                        absolute_values_distance = np.abs(
                            absolute_map - no_samples_absolute_map
                        )

                        __import__("pprint").pprint(absolute_map[1, 1])
                        __import__("pprint").pprint(no_samples_absolute_map[1, 1])

                        # __import__("pprint").pprint(
                        #    f"Absolute distance: {absolute_values_distance[0, 1]}"
                        # )

                        # ut.print_log(log_file, relative_values_distance[0, 1])

                        # Getting the pearson_coefficient
                        pearson_coefficient, p_value = ut.pearson_correlation(
                            samples_log_relative_map, no_samples_log_relative_map
                        )

                        # Getting the values distance report
                        _ = sim_ut.generate_values_distance_report(
                            relative_values_distance,
                            pearson_coefficient,
                            p_value,
                            correlation_type,
                            ko_species_list,
                            file_name,
                            f"./report/{file_name}/Perturbations importance analysis",
                            report_title="Values distance report analysis",
                            log_file=log_file,
                        )

                        # Plotting the distance of log normalized matrices
                        plt_ut.plot_variations_heatmap(
                            relative_values_distance,
                            all_species_rel,
                            ko_species_list,
                            save_path=f"./imgs/{file_name}/Perturbations importance analysis",
                            title="Perturbations VS No Perturbations Distance",
                            imgs_name="Perturbations VS No Perturbations distance",
                        )

                        plt_ut.plot_variations_heatmap(
                            absolute_values_distance,
                            all_species_abs,
                            ko_species_list,
                            save_path=f"./imgs/{file_name}/Perturbations importance analysis absolute",
                            title="Perturbations VS No Perturbations Distance absolute",
                            imgs_name="Perturbations VS No Perturbations distance absolute",
                        )

                        # Get the active cells of the matrices
                        active_cells_no_samples_map = ut.get_active_cells(
                            normalized_no_samples_relative, log_file
                        )
                        active_cells_samples_map = ut.get_active_cells(
                            normalized_samples_relative_map, log_file
                        )

                        # Getting the pattern distance report
                        relative_pattern_distance = np.abs(
                            active_cells_no_samples_map - active_cells_samples_map
                        )

                        pattern_pearson_coefficient, pattern_p_value = (
                            ut.pearson_correlation(
                                active_cells_no_samples_map, active_cells_samples_map
                            )
                        )

                        _ = sim_ut.generate_pattern_distance_report(
                            relative_pattern_distance,
                            pattern_pearson_coefficient,
                            pattern_p_value,
                            correlation_type,
                            ko_species_list,
                            file_name,
                            f"./report/{file_name}/Perturbations importance analysis",
                            report_title="Pattern distance report analysis",
                            log_file=log_file,
                        )

                        # Plotting the pattern distnace's matrix
                        plt_ut.plot_variations_heatmap(
                            relative_pattern_distance,
                            all_species_rel,
                            ko_species_list,
                            save_path=f"./imgs/{file_name}/Perturbations importance analysis",
                            title="Relative patterns distance's heatmap",
                            imgs_name="Relative pattern distance's heatmap",
                        )

                    if args.random_perturbations_importance:
                        ut.print_log(log_file, "Random importance analysis")

                        # === FIXED SAMPLES ANALYSIS ===
                        # rr.reset()

                        ut.print_log(log_file, "STARTING FIXED SAMPLES ANALYSIS")
                        ut.print_log(
                            log_file, "Please insert the fixed variations (v1 v2 ...)"
                        )
                        ut.print_log(
                            log_file,
                            "[WARNING] Notice that the number of samples will equals to the number of variations",
                        )

                        env_pert = os.getenv("FIXED_PERTURBATIONS").split(
                            ","
                        )  # pyright:ignore

                        if env_pert is not None:
                            fixed_variations = [np.float64(v) for v in env_pert]
                        else:
                            raise TypeError("Perturbations are None")

                        ut.print_log(log_file, f"{fixed_variations}")

                        ut.print_log(log_file, "Generating fixed combinations")

                        fixed_combinations = sbml_ut.get_fixed_combinations(
                            sbml_model, input_species_ids, fixed_variations, log_file
                        )

                        fixed_samples_results, _ = sim_ut.simulate_combinations(
                            rr,
                            fixed_combinations,
                            input_species_ids,
                            min_ss_time,
                            end_time,
                            max_end_time,
                            steady_state,
                            log_file,
                        )

                        fixed_samples_original_model_informations = (
                            sim_ut.get_simulations_informations(
                                fixed_samples_results,
                                original_results,
                                fixed_combinations,
                                colnames,
                                log_file,
                            )
                        )

                        ut.print_log(
                            log_file, " Simulating multiporcessing with fixed samples"
                        )

                        fixed_knockout_data = sim_ut.process_species_multiprocessing(
                            target_ids,
                            model_dict,
                            fixed_combinations,
                            input_species_ids,
                            selections,
                            integrator,
                            start_time=0,
                            end_time=end_time,
                            steady_state=steady_state,
                            max_end_time=max_end_time,
                            min_ss_time=min_ss_time,
                            log_file=log_file,
                            preserve_input=args.preserve_inputs,
                            use_perturbations=args.use_perturbations,
                        )

                        fixed_variation_dict = sim_ut.get_knockout_variations_samples(
                            fixed_samples_original_model_informations,  # pyright:ignore
                            fixed_knockout_data,
                            log_file=None,
                        )

                        fixed_relative_map, fixed_all_species_rel = (
                            sim_ut.get_variations_hm_samples(
                                fixed_variation_dict,
                                all_species,
                                ko_species_list,
                                variation_type="relative",
                                log_file=log_file,
                            )
                        )

                        ut.print_log(log_file, fixed_relative_map[0, 1])
                        fixed_log_relative_map = np.log(fixed_relative_map + 1)
                        ut.print_log(log_file, fixed_log_relative_map[0, 1])

                        # === DIFFERENCE ANALYSIS WITH SAMPLES ===

                        # Getting the relative distance
                        fixed_relative_values_distance = np.abs(
                            fixed_log_relative_map - samples_log_relative_map
                        )

                        normalized_fixed_relative_map = ut.minMax_normalize(
                            fixed_log_relative_map
                        )

                        # Getting the pearson_coefficient
                        fixed_pearson_coefficient, fixed_p_value = (
                            ut.pearson_correlation(
                                fixed_log_relative_map, samples_log_relative_map
                            )
                        )

                        # Getting the values distance report
                        _ = sim_ut.generate_values_distance_report(
                            fixed_relative_values_distance,
                            fixed_pearson_coefficient,
                            fixed_p_value,
                            correlation_type,
                            ko_species_list,
                            file_name,
                            f"./report/{file_name}/Fixed perturbations importance analysis",
                            report_title="Fixed perturbations values distance report analysis",
                            log_file=log_file,
                        )

                        # Plotting the distance of log normalized matrices
                        plt_ut.plot_variations_heatmap(
                            fixed_relative_values_distance,
                            fixed_all_species_rel,
                            ko_species_list,
                            save_path=f"./imgs/{file_name}/Random Perturbations Importance Analysis",
                            title="Random Perturbations VS Fixed Perturbations Distance",
                            imgs_name="Random Perturbations VS Fixed Perturbations distance",
                        )

                        # Getting active cells of fixed_relative_map
                        fixed_active_cells = ut.get_active_cells(
                            normalized_fixed_relative_map, log_file
                        )

                        # Doing fixed pattern analysis
                        fixed_pattern_distance = np.abs(
                            fixed_active_cells - active_cells_samples_map
                        )

                        __import__("pprint").pprint(fixed_active_cells)
                        __import__("pprint").pprint(active_cells_samples_map)

                        fixed_pattern_pearson_coefficient, fixed_p_value = (
                            ut.pearson_correlation(
                                fixed_active_cells,
                                active_cells_samples_map,
                            )
                        )

                        _ = sim_ut.generate_pattern_distance_report(
                            fixed_pattern_distance,
                            fixed_pattern_pearson_coefficient,
                            fixed_p_value,
                            correlation_type,
                            ko_species_list,
                            file_name,
                            f"./report/{file_name}/Fixed perturbations importance analysis",
                            report_title="Fixed perturbations pattern distance report analysis",
                            log_file=log_file,
                        )

                        # Plotting the pattern distance's matrix
                        plt_ut.plot_variations_heatmap(
                            fixed_pattern_distance,
                            fixed_all_species_rel,
                            ko_species_list,
                            save_path=f"./imgs/{file_name}/Random Perturbations Importance Analysis",
                            title="Fixed perturbations patterns distance's heatmap",
                            imgs_name="Fixed perturbations pattern distance's heatmap",
                        )
                #
                except AssertionError as ae:
                    ut.print_log(
                        log_file, f"Error during samples results elaboration: {ae}"
                    )

        elif args.command == "sensibility_analysis":

            # LOADING MODEL
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()

            # SPLITTING REVERSIBLE REACTIONS
            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            file_name = os.path.basename(args.input_path)

            # GETTING INPUT SPECIES IDS
            input_ids = args.input_species
            all_species_ids = [s.getId() for s in sbml_model.getListOfSpecies()]
            internal_nodes = list(set(all_species_ids) - set(input_ids))

            rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

            # Creating the problem
            problem = sens_ut.get_problem_parameters(
                sbml_model, len(input_ids), input_ids, 20, log_file=log_file
            )

            spec = ProblemSpec(problem)

            params = spec.sample(sobol_sample.sample, 2).samples

            # RES = np.zeros([params.shape[0], len(internal_nodes)])

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
                log_file, f"Available internal nodes: {available_internal_nodes}"
            )

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

            for j, node_id in enumerate(available_internal_nodes):

                Si = sobol_analyze.analyze(problem, RES[:, j])
                res_dict[node_id] = Si

            # __import__("pprint").pprint(res_dict)

            for node, data in res_dict.items():
                sens_ut.report_sensitivity(node, data, input_ids, log_file)

            if args.check_convergence:
                sample_sizes = [64, 128, 256, 1024, 2048, 4096]

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
                    min_consecutive=1,
                    relative=True,
                )

                __import__("pprint").pprint(convergence_informations)

                sens_ut.plot_convergence_single_plot(convergence_informations)

        elif args.command == "knockout_species":
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()

            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            file_name = os.path.basename(args.input_path)

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
                file_name, operation_name, modified_model, True, log_file
            )

            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)

        elif args.command == "knockout_reaction":
            # Load the model
            sbml_doc = sbml_ut.load_model(args.input_path)
            sbml_model = sbml_doc.getModel()
            file_name = os.path.basename(args.input_path)

            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

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
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path).getModel()
            dir_name = os.path.dirname(args.input_path)
            file_name = os.path.basename(args.input_path)

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
