#!/usr/bin/env python
import enum
from math import log, nan
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.assortativity import correlation
from numpy._typing import _UnknownType
from pandas._libs import iNaT
from classes.SBMLHandler import SBMLHandler
import roadrunner
import libsbml
import numpy as np
from dotenv import load_dotenv
import time


# Add the src folder path to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as sbml_ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu
from src.utils import utils as ut
from src.utils import plot_utils as plt_ut


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
            sbml_model = sbml_doc.getModel()
            file_name = os.path.basename(args.input_path)

            # sbml_ut.split_reversible_reactions(sbml_model, log_file)
            # sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            ut.print_log(log_file, f"Simulating model: {file_name}")

            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(sbml_model, args.integrator, log_file)
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file)

            # sim_ut.simulate_to_steady_state(rr, log_file=log_file)
            # steady_state_time = sim_ut.time_to_steady_state_window(rr, 10)

            # ut.print_log(log_file, f"Steady state reached at time: {steady_state_time}")

            # if steady_state_time:
            #   simulation_end_time = steady_state_time
            # else:
            #    simulation_end_time = args.time
            #
            # rr.reset()

            res, ss_time, colnames = sim_ut.simulate(
                rr,
                end_time=args.time,
                start_time=0,
                log_file=log_file,
                steady_state=args.steady_state,
                max_end_time=args.max_time,
            )

            # ut.print_log(log_file, f"Colnames: {colnames}")

            # ut.print_log(log_file, rr["[Px]"])
            # ut.print_log(
            #     log_file,
            #     f"tolerances: abs {rr.integrator.absolute_tolerance} | rel {rr.integrator.relative_tolerance}",
            # )

            plt_ut.plot_results(res, colnames, args.output, file_name, log_file)

        elif args.command == "simulate_samples":

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

            if input_species_ids is not None:
                target_ids = list(set(target_ids) - set(input_species_ids))
                # ut.print_log(log_file, f"{target_ids}")

            # ut.print_log(log_file, f"target ids: {target_ids}")
            target_ids.sort()

            # Get the list of species
            species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

            combinations = None

            if input_species_ids is not None:
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
            rr = sim_ut.load_roadrunner_model(sbml_model, integrator, log_file)

            # Get the selections
            selections = rr.selections
            for s in species_list:
                if f"[{s}]" not in selections:
                    selections.append(f"[{s}]")

            # Add also the reactions in the selections if target
            for ts in target_ids:
                if ts in sbml_model.getListOfReactions().getId():
                    selections.append(f"{ts}")

            rr.selections = selections

            # Simulate the model
            ut.print_log(log_file, "Simulating original model")
            original_results, ss_time, colnames = sim_ut.simulate(
                rr,
                start_time=0,
                end_time=end_time,
                steady_state=steady_state,
                max_end_time=max_end_time,
            )

            # ut.print_log(log_file, f"[MAIN] colnames: {colnames}")

            min_ss_time = (
                ss_time
                if ss_time is not None and ss_time <= min_ss_time
                else min_ss_time
            )

            if input_species_ids is not None:
                # Simulate the original model with samples
                ut.print_log(log_file, "Simulating original model with samples")
                samples_simulations_results, _ = sim_ut.simulate_combinations(
                    rr,
                    combinations,
                    input_species_ids,
                    min_ss_time,
                    end_time,
                    max_end_time,
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

                for combination, info_dict in original_model_simulations_info.items():
                    ut.print_log("orig", f"{combination}:")
                    for species, values in info_dict.items():
                        ut.print_log("orig", f" {species}: {values}")

            knockout_data = []

            # FOR DEBUG ONLY
            early_stop = 0

            # Copying the SBML model
            sbml_str = libsbml.writeSBMLToString(sbml_doc)

            # Create a dict of the ko models
            model_dict = {}
            for ids in target_ids:
                doc_copy = libsbml.readSBMLFromString(sbml_str)  # rebuild in memory
                model_copy = doc_copy.getModel()
                if ids in [s.getId() for s in sbml_model.getListOfSpecies()]:
                    modified_model = sbml_ut.knockout_species(model_copy, ids, log_file)

                elif ids in [r.getId() for r in sbml_model.getListOfReactions()]:
                    modified_model = sbml_ut.knockout_reaction(
                        model_copy, ids, log_file
                    )

                else:
                    raise Exception("Id not present in the model")

                doc_copy.setModel(modified_model)
                model_dict[ids] = libsbml.writeSBMLToString(doc_copy)

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
            )

            for ko_species, comb_dict in knockout_data:
                ut.print_log("samples", f"{ko_species}:")
                for combination, info_dict in comb_dict.items():
                    ut.print_log("samples", f"  {combination}:")
                    for species, value in info_dict.items():
                        ut.print_log("samples", f"      {species}: {value}")

            end_time = time.perf_counter()

            # === IF NO SAMPLES ===
            if input_species_ids is None:

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
                    title="Relative variations heatmap",
                )

                # for i, ko_species in enumerate(ko_species_list):
                #     for j, species in enumerate(all_species_rel):
                #         ut.print_log(
                #             "no_samples",
                #             f"({ko_species}, {species}) {log_rel_map[i,j]}",
                #         )

                plt_ut.plot_variations_heatmap(
                    log_abs_map,
                    all_species_abs,
                    ko_species_list,
                    variation_type="absolute",
                    title="Variation heatmap",
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

                    # for i, ko_species in enumerate(ko_species_list):
                    #     for j, species in enumerate(all_species_rel):
                    #         ut.print_log(
                    #             "samples",
                    #             f"{ko_species, species} {no_samples_log_relative_map[i,j]}",
                    #         )

                    # Get the informations about the original absolute variations
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

                    abs_map, all_species_abs = sim_ut.get_variations_hm_samples(
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
                    samples_log_absolute_map = np.log(abs_map + 1)

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
                        title="Relative variations with samples",
                    )

                    plt_ut.plot_variations_heatmap(
                        samples_log_absolute_map,
                        all_species_rel,
                        ko_species_list,
                        save_path=f"./imgs/{file_name}",
                        variation_type="absolute",
                        title="Variations with samples",
                    )

                    # === SAMPLING IMPORTANCE ANALYSIS ===

                    correlation_type = "two-sided"

                    # Getting the values relative distance
                    relative_values_distance = np.abs(
                        samples_log_relative_map - no_samples_log_relative_map
                    )

                    ut.print_log(log_file, relative_values_distance[0, 1])

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

                    # TODO: Check the fixed samples analysis

                    # === FIXED SAMPLES ANALYSIS ===
                    ut.print_log(
                        log_file, "Do you want to perform fixed samples analysis? [y/n]"
                    )

                    user_choice = input()

                    valids_input = ["y", "n"]

                    while user_choice.lower() not in valids_input:
                        ut.print_log(
                            log_file,
                            "[WARNING] Invalid input, please insert a valid value (y/n)",
                        )
                        user_choice = input()

                    if user_choice.lower() == "y":

                        rr.reset()

                        ut.print_log(log_file, "STARTING FIXED SAMPLES ANALYSIS")
                        ut.print_log(
                            log_file, "Please insert the fixed variations (v1 v2 ...)"
                        )
                        ut.print_log(
                            log_file,
                            "[WARNING] Notice that the number of samples will equals to the number of variations",
                        )

                        fixed_variations = [
                            float(inp) for inp in input().strip().split(" ")
                        ]

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

                        fixed_absolute_map, _ = sim_ut.get_variations_hm_samples(
                            fixed_variation_dict,
                            all_species,
                            ko_species_list,
                            variation_type="absolute",
                            log_file=log_file,
                        )

                        fixed_log_relative_map = np.log(fixed_relative_map + 1)

                        fixed_log_absolute_map = np.log(fixed_absolute_map + 1)

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

                        # Doing pattern analysis
                        fixed_pattern_distance = np.abs(
                            fixed_active_cells - active_cells_samples_map
                        )

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

                except AssertionError as ae:
                    ut.print_log(
                        log_file, f"Error during samples results elaboration: {ae}"
                    )

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

            rr = sim_ut.load_roadrunner_model(sbml_model, log_file)

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
