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
            sbml_model = sbml_ut.load_model(args.input_path)
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
                f"Concentration of species_0:\n {res[:, res.colnames.index('[species_0]')]}",
            )
            plt_ut.plot_results(res, colnames, args.output, file_name, log_file)

        elif args.command == "simulate_samples":
            # TODO: Check if the method is correct
            steady_state = args.steady_state
            max_end_time = args.max_time
            min_ss_time = args.max_time
            ss_time = args.max_time
            end_time = args.time
            integrator = args.integrator

            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)

            sbml_model = sbml_ut.split_all_reversible_reactions(sbml_model)

            file_name = os.path.basename(args.input_path)

            input_species_ids = args.input_species

            if args.target_ids is None:
                target_ids = [s.getId() for s in sbml_model.getListOfSpecies()]
            else:
                target_ids = args.target_ids

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

            # Load and simulate
            rr = sim_ut.load_roadrunner_model(sbml_model, integrator, log_file)

            # Check if some target species are reaction
            for ts in target_ids:
                if ts in rr.model.getReactionIds():
                    rr.selections = rr.selections + [f"{ts}"]
                elif ts in rr.model.getBoundarySpeciesIds():
                    rr.selections = rr.selections + [f"[{ts}]"]

            ut.print_log(log_file, "Simulating original model")
            original_results, ss_time, colnames = sim_ut.simulate(
                rr,
                start_time=0,
                end_time=end_time,
                steady_state=steady_state,
                max_end_time=max_end_time,
            )

            min_ss_time = (
                ss_time
                if ss_time is not None and ss_time <= min_ss_time
                else min_ss_time
            )

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

            # Get the information about the simulations
            ut.print_log(
                log_file, "Getting simulations informations of the original model"
            )

            final_results_original_model = sim_ut.get_simulations_informations(
                samples_simulations_results,
                original_results,
                combinations,
                target_ids,
                colnames,
                log_file,
            )

            # TODO: Implement difference logic from original
            # Calculate for every knocked out species the final result
            # Create a dictionary having the combinatinos as key
            # And as value the variation in respect to the original
            #

            knocked_data = []
            final_results_knocked_model = None

            early_stop = 0  # FOR DEBUG ONLY

            for species in sbml_model.getListOfSpecies():
                # FOR DEBUG ONLY
                if early_stop == 3:
                    break

                species_id = species.getId()

                ut.print_log(
                    log_file, f"Simulating with knockout of species: {species_id}"
                )

                modified_model = sbml_ut.knockout_species(
                    sbml_model, species_id, log_file
                )

                # FIX: CHECK FOR POSSIBLE BUGS

                knocked_rr = sim_ut.load_roadrunner_model(
                    modified_model, integrator, log_file
                )

                knocked_out_original_results, ss_time, colnames = sim_ut.simulate(
                    knocked_rr,
                    start_time=0,
                    end_time=end_time,
                    steady_state=steady_state,
                    max_end_time=max_end_time,
                )

                min_ss_time = (
                    ss_time
                    if ss_time is not None and ss_time <= min_ss_time
                    else min_ss_time
                )

                knocked_samples_simulations_results = sim_ut.simulate_combinations(
                    knocked_rr,
                    combinations,
                    input_species_ids,
                    min_ss_time,
                    end_time,
                    max_end_time,
                    steady_state,
                    log_file,
                )

                final_results_knocked_model = sim_ut.get_simulations_informations(
                    knocked_samples_simulations_results,
                    knocked_out_original_results,
                    combinations,
                    target_ids,
                    colnames,
                    log_file,
                )

                knocked_data.append((species_id, final_results_knocked_model))

                knocked_rr.reset()

                early_stop += 1

            # TODO: Create the dictionary with every variation

            variations_dict = sim_ut.get_variations_dict(
                final_results_original_model, knocked_data, log_file
            )

            plt_ut.plot_knockdown_effect_heatmap(variations_dict, log_file=log_file)

        elif args.command == "knockout_species":
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)

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
            sbml_model = sbml_ut.load_model(args.input_path)
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
            sbml_model = sbml_ut.load_model(args.input_path)
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
