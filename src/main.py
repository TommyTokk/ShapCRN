import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import roadrunner
import libsbml

# Add the src folder path to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as sbml_ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu
from src.utils import utils as ut

from SBML_batch import PetriNets


def main():
    # Use the parse_args function to get command-line arguments
    args = ut.parse_args()
    
    # Setup logging
    log_file = args.log if hasattr(args, 'log') else None

    try:
        # Process based on command
        if args.command == 'simulate':
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)
            
            ut.print_log(log_file, f"Simulating model: {file_name}")
            
            # Load and simulate
            if args.integrator:
                rr = sim_ut.load_roadrunner_model(sbml_model, args.integrator, log_file)
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file)
            
            res = sim_ut.simulate(rr, end_time=args.time)
            sim_ut.plot_results(res, args.output, file_name, log_file)
            
        elif args.command == 'inhibit_species':
            # Load the model
            sbml_model = sbml_ut.load_model(args.input_path)
            file_name = os.path.basename(args.input_path)
            
            ut.print_log(log_file, f"Inhibiting species {args.species_id} in model: {file_name}")
            
            # Inhibit the species
            modified_model = sbml_ut.inhibit_species(sbml_model, args.species_id, log_file)
            operation_name = f"no_species_{args.species_id}"
            
            # Save if requested
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
            
            # Save if requested
            xml_string, output_filename = sbml_ut.save_file(
                file_name, operation_name, modified_model, True, log_file
            )
            
            # Simulate the modified model
            # rr_modified = sim_ut.load_roadrunner_model(xml_string, log_file)
            # res_modified = sim_ut.simulate(rr_modified)
            # sim_ut.plot_results(res_modified, args.output, output_filename, log_file)

    except Exception as e:
        ut.print_log(log_file, f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
