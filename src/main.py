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
    log_file = args.log if hasattr(args, 'log') else None


    try:
        # Process based on command
        if args.command == 'simulate':
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)
            
            ut.print_log(log_file, f"Simulating model: {file_name}")
            
            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(sbml_model, args.integrator, log_file)
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file)
            
            res = sim_ut.simulate(rr, end_time=args.time, start_time=0)
            sim_ut.plot_results(res, args.output, file_name, log_file)

        elif args.command == 'simulate_samples':
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)

            input_species_ids = args.input_species
            target_species_ids = args.target_species
            
            #Generate the samples
            samples = sbml_ut.generate_species_samples(sbml_model, input_species_ids, log_file=log_file, n_samples=args.num_samples, variation=args.variation)

            #Create the combinations
            combinations = sbml_ut.create_samples_combination(samples, log_file)

            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(sbml_model, args.integrator, log_file)
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file)

            #Check if some target species are boundary species
            for ts in target_species_ids:
                if ts in rr.model.getBoundarySpeciesIds():
                    rr.selections = rr.selections + [f"[{ts}]"]


            # ut.print_log(log_file, rr.timeCourseSelections)
            # ut.print_log(log_file, f"Model floating species: {rr.model.getFloatingSpeciesIds()}")
            # ut.print_log(log_file, f"Model boundary species: {rr.model.getBoundarySpeciesIds()}")
            # exit(1)

            original_results = sim_ut.simulate(rr, start_time=0, end_time=10)

            samples_simulations_results = []#will contains the results of perturbated inputs

            for i in range(len(combinations)):
                ut.print_log(log_file, f"Simulation nr. {i}")
                samples_simulations_results.append(sim_ut.simulate_samples(rr, combinations[i], input_species_ids, start_time=0, end_time=10))

            #TODO: Handle the simulations data
            #The the colums of the target_output_species
            target_species_original_concentrations = []
            target_species_perturbated_concentrations = []

            for ts in target_species_ids:
                try:
                    ts_index = samples_simulations_results[0].colnames.index(f"[{ts}]")
                    print(ts_index)
                except: 
                    ut.print_log(log_file, f"Species {ts} not present in the results")
                    continue
                target_species_perturbated_concentrations.append(samples_simulations_results[0][:, ts_index])
                target_species_original_concentrations.append(original_results[:, ts_index])

            ut.print_log(log_file, f"{len(target_species_original_concentrations)}, {len(target_species_perturbated_concentrations)}")
            
            #TODO:Compare the original results with the perturbated results
            ut.check_variation(target_species_ids, original_results, target_species_original_concentrations, target_species_perturbated_concentrations, log_file)
                

            
        elif args.command == 'inhibit_species':
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)

            #FOR DEBUG
            # for specie in sbml_model.getListOfSpecies():
            #     sbml_ut.check_presence(sbml_model, specie.getId(), log_file)

            
            ut.print_log(log_file, f"Inhibiting species {args.species_id} in model: {file_name}")
            
            # Inhibit the species
            modified_model = sbml_ut.inhibit_species(sbml_model, args.species_id, log_file)
            operation_name = f"no_species_{args.species_id}"
            
            # Save
            xml_string, output_filename = sbml_ut.save_file(
                file_name, operation_name, modified_model, True, log_file
            )
            
            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)
            
        elif args.command == 'inhibit_reaction':
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)
            
            ut.print_log(log_file, f"Inhibiting reaction {args.reaction_id} in model: {file_name}")
            
            # Inhibit the reaction
            modified_model = sbml_ut.inhibit_reaction(sbml_model, args.reaction_id, log_file)
            operation_name = f"no_reaction_{args.reaction_id}"
            
            # Save
            xml_string, output_filename = sbml_ut.save_file(
                file_name, operation_name, modified_model, True, log_file
            )
            
            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)
        elif args.command == 'create_petrinet':
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            dir_name = os.path.dirname(args.input_path)
            file_name = os.path.basename(args.input_path)

            species_list = sbml_ut.get_list_of_species(sbml_model)
            reactions_list = sbml_ut.get_list_of_reactions(sbml_model, sbml_ut.get_species_dict(species_list))
            
            ut.print_log(log_file, f"Simulating model: {file_name}")
            
            rr = sim_ut.load_roadrunner_model(sbml_model, log_file)

            N = nu.get_network_from_sbml(reactions_list, species_list, log_file)
            nu.plot_network(
                graph=N,
                img_dir_path=args.output, 
                img_name=f"{os.path.splitext(file_name)[0]}_network",  # Nome del file senza estensione + _network
                log_file=log_file
            )

    except Exception as e:
        ut.print_log(log_file, f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
