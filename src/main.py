#!/usr/bin/env python
from math import log
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from numpy._typing import _UnknownType
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
            ut.print_log(
                log_file,
                f"tolerances: abs {rr.integrator.absolute_tolerance} | rel {rr.integrator.relative_tolerance}",
            )
            ut.print_log(
                log_file,
                f"Concentration of species_4:\n {res[:, res.colnames.index('[species_4]')]}",
            )
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

            target_ids = list(set(target_ids) - set(input_species_ids))
            # ut.print_log(log_file, f"{target_ids}")

            # ut.print_log(log_file, f"target ids: {target_ids}")
            target_ids.sort()

            species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

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

            # Load the model
            rr = sim_ut.load_roadrunner_model(sbml_model, integrator, log_file)

            # Get the selections

            selections = rr.selections
            for s in species_list:
                if f"[{s}]" not in selections:
                    selections.append(f"[{s}]")

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

            # Simulate the original model with samples
            ut.print_log(log_file, "Simulating original model with samples")
            samples_simulations_results = sim_ut.simulate_combinations(
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
                colnames[1:],
                log_file,
            )

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

            counter = 0

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

            # for species_to_knockout in target_ids:
            #     ut.print_log(
            #         log_file,
            #         f"Working on species: {species_to_knockout} ({(counter/len(target_ids))*100}%)",
            #     )
            #     counter += 1
            #
            #     modified_model = model_dict[species_to_knockout]
            #
            #     # ut.print_log(log_file, f"model: {modified_model.getName()}")
            #
            #     modified_rr = sim_ut.load_roadrunner_model(
            #         modified_model, integrator, log_file
            #     )
            #
            #     modified_rr.selections = selections
            #
            #     knockout_model_results, ss_time, colnames = sim_ut.simulate(
            #         modified_rr,
            #         start_time=0,
            #         end_time=end_time,
            #         steady_state=steady_state,
            #         max_end_time=max_end_time,
            #     )
            #
            #     min_ss_time = (
            #         ss_time
            #         if ss_time is not None and ss_time <= min_ss_time
            #         else min_ss_time
            #     )
            #
            #     combinations_knockout_model_results = sim_ut.simulate_combinations(
            #         modified_rr,
            #         combinations,
            #         input_species_ids,
            #         min_ss_time,
            #         end_time,
            #         max_end_time,
            #         steady_state,
            #         log_file,
            #     )
            #
            #     # Contains information, for each knockedout species,
            #     # and foreach combination, about the simulation's results
            #     combinations_knockout_model_simualtions_info = (
            #         sim_ut.get_simulations_informations(
            #             combinations_knockout_model_results,
            #             knockout_model_results,
            #             combinations,
            #             colnames[1:],
            #             log_file,
            #         )
            #     )
            #
            #     knockout_data.append(
            #         (species_to_knockout, combinations_knockout_model_simualtions_info)
            #     )

            end_time = time.perf_counter()

            ut.print_log(
                log_file, f"Time to process species: {(end_time-start_time):.2f}s"
            )

            variation_dict = sim_ut.get_simulations_variations(
                original_model_simulations_info, knockout_data, log_file=None
            )

            # Collect all species and knocked-out species
            all_species = set()
            ko_species_list = list(variation_dict.keys())

            relative_map, all_species_rel = sim_ut.get_variations_mean(
                variation_dict,
                all_species,
                ko_species_list,
                variation_type="relative",
                log_file=log_file,
            )

            relative_log_map = np.log10(relative_map + 1)

            abs_map, all_species_abs = sim_ut.get_variations_mean(
                variation_dict,
                all_species,
                ko_species_list,
                variation_type="absolute",
                log_file=log_file,
            )

            abs_log_map = np.log10(abs_map + 1)

            # for i in range(len(target_ids)):
            #     ut.print_log(log_file, f"Id:{target_ids[i]}")
            #     ut.print_log(log_file, f"Relative-Data: {relative_map[i]}")
            #     ut.print_log(log_file, f"Absolute-Data: {abs_map[i]}")
            #     ut.print_log(log_file, "===========================")

            # ut.pretty_print_variations(variation_dict, precision=10, show_zero=True)

            plt_ut.plot_variations_heatmap(
                relative_log_map,
                all_species_rel,
                ko_species_list,
                title="Relative variations heatmap",
            )
            plt_ut.plot_variations_heatmap(
                abs_log_map,
                all_species_abs,
                ko_species_list,
                variation_type="absolute",
                title="Variation heatmap",
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
