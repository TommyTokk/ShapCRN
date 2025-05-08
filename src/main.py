from math import log
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import roadrunner
import libsbml
import numpy as np

# Add the src folder path to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as sbml_ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu
from src.utils import utils as ut


def main():
    # Use the parse_args function to get command-line arguments
    args = ut.parse_args()

    # Setup logging
    log_file = args.log if hasattr(args, "log") else None

    try:
        # Process based on command
        if args.command == "simulate":
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)

            ut.print_log(log_file, f"Simulating model: {file_name}")

            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(sbml_model, args.integrator, log_file)
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file)

            res = sim_ut.simulate(
                rr, end_time=args.time, start_time=0, log_file=log_file
            )
            # ut.print_log(log_file, rr["[Px]"])
            # ut.print_log(log_file, f"Concentration of Px:\n {res[:, res.colnames.index("[Px]")]}")
            sim_ut.plot_results(res, args.output, file_name, log_file)

        elif args.command == "simulate_samples":
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)

            input_species_ids = args.input_species
            target_species_ids = args.target_species

            # Generate the samples
            samples = sbml_ut.generate_species_samples(
                sbml_model,
                input_species_ids,
                log_file=log_file,
                n_samples=args.num_samples,
                variation=args.variation,
            )

            # Create the combinations
            combinations = sbml_ut.create_samples_combination(samples, log_file)

            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(sbml_model, args.integrator, log_file)
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file)

            # Check if some target species are boundary species
            for ts in target_species_ids:
                if ts in rr.model.getBoundarySpeciesIds():
                    rr.selections = rr.selections + [f"[{ts}]"]

            # ut.print_log(log_file, rr.timeCourseSelections)
            # ut.print_log(log_file, f"Model floating species: {rr.model.getFloatingSpeciesIds()}")
            # ut.print_log(log_file, f"Model boundary species: {rr.model.getBoundarySpeciesIds()}")
            # exit(1)

            original_results = sim_ut.simulate(rr, start_time=0, end_time=10)

            samples_simulations_results = (
                []
            )  # will contains the results of perturbated inputs

            for i in range(len(combinations)):
                ut.print_log(log_file, f"Simulation nr. {i}")
                # ut.print_log(log_file, f"ACEp' concentration: {rr["ACEp"]}")
                samples_simulations_results.append(
                    sim_ut.simulate_samples(
                        rr,
                        combinations[i],
                        input_species_ids,
                        start_time=0,
                        end_time=10,
                    )
                )

            target_species_data = {}  # Dictionary for species informations

            for ts in target_species_ids:
                try:
                    ts_index = original_results.colnames.index(f"[{ts}]")
                    
                    target_species_data[ts] = {
                        "original": original_results[:, ts_index],
                        "simulations": []  # List containing informations about the simulations
                    }
                    
                    # Adding the results of the simulations
                    for i in range(len(samples_simulations_results)):
                        try:
                            # Getting the info about the combinations for te specified simulation's ID
                            combination_values = combinations[i]
                            combination_str = "-".join([f"{species}:{value:.4f}" for species, value in zip(input_species_ids, combination_values)])
                            
                            # Creating the simulation dictionary
                            simulation_data = {
                                "id": f"sim_{i}_{combination_str}",
                                "combination": combinations[i],  # Saving the combination
                                "results": samples_simulations_results[i][:, ts_index]  # Saving the results
                            }
                            
                            # Adding to the simulations's list
                            target_species_data[ts]["simulations"].append(simulation_data)
                            
                        except Exception as e:
                            ut.print_log(log_file, f"Error extracting data for simulation {i}, species {ts}: {e}")
                            
                    ut.print_log(log_file, f"Processed {len(target_species_data[ts]['simulations'])} simulations for species {ts}")
                    
                except Exception as e:
                    ut.print_log(log_file, f"Species {ts} not present in the results: {e}")
                    continue

            # Analyze data
            ut.analyze_simulation_variations(target_species_data, original_results, args.output, log_file)

        elif args.command == "inhibit_species":
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)

            # FOR DEBUG
            # for specie in sbml_model.getListOfSpecies():
            #     sbml_ut.check_presence(sbml_model, specie.getId(), log_file)

            ut.print_log(
                log_file, f"Inhibiting species {args.species_id} in model: {file_name}"
            )

            # Inhibit the species
            modified_model = sbml_ut.inhibit_species(
                sbml_model, args.species_id, log_file
            )
            operation_name = f"no_species_{args.species_id}"

            ut.print_log(
                log_file,
                f"Searching for updating operations on target species {args.species_id}",
            )
            found = False
            for rule in sbml_model.getListOfRules():
                if rule.getVariable() == args.species_id:
                    ut.print_log(
                        log_file, f"Found rules for {args.species_id}, {rule.getType()}"
                    )
                    found = True
            if not found:
                ut.print_log(log_file, f"No rules has been found for {args.species_id}")
            found = False
            for reaction in sbml_model.getListOfReactions():
                for i in range(reaction.getNumProducts()):
                    product = reaction.getProduct(i)
                    if product.getSpecies() == args.species_id:
                        ut.print_log(
                            log_file,
                            f"target species found as product in reaction {reaction.getId()}",
                        )
                        found = True
            if not found:
                ut.print_log(
                    log_file,
                    f"No reactions has been found having {args.species_id} as product",
                )

            # Save
            xml_string, output_filename = sbml_ut.save_file(
                file_name, operation_name, modified_model, True, log_file
            )

            for specie in sbml_model.getListOfSpecies():
                if specie.getId() == args.species_id:
                    ut.print_log(
                        log_file,
                        f"Concentration of {args.species_id} is {specie.getInitialConcentration()}",
                    )

            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)

        elif args.command == "inhibit_reaction":
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)

            ut.print_log(
                log_file,
                f"Inhibiting reaction {args.reaction_id} in model: {file_name}",
            )

            # Inhibit the reaction
            modified_model = sbml_ut.inhibit_reaction(
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
