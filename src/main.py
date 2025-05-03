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
    
    # Extract args
    model_dir_path = args[0]
    operation_type = args[1]
    target_id = args[2]
    save_output = args[3]
    log_file = args[4]
    
    # Extract directory path and file name
    dir_path = os.path.dirname(model_dir_path)
    file_name = os.path.basename(model_dir_path)

    # If dir_path is empty, set current directory
    if dir_path == "":
        dir_path = "."  # "." represents current directory

    ut.print_log(log_file, f"Directory path: {dir_path}")
    ut.print_log(log_file, f"File name: {file_name}")

    try:
        # Load model
        sbml_model = sbml_ut.load_model(model_dir_path)


        species = sbml_ut.get_list_of_species(sbml_model)
        species_ids = [s.get_id() for s in species]
        
        # First simulation - original model
        ut.print_log(log_file, "Simulating original model...")
        rr = sim_ut.load_roadrunner_model(sbml_model)

        #rr.setIntegrator('gillespie')
        res = sim_ut.simulate(rr)
        sim_ut.plot_results(res, "./imgs", file_name, log_file)
        rr.reset()
        if True:
            # Check if we have an operation to perform
            if len(args) > 1:
                
                # Perform the specified operation
                if operation_type == 1:
                    # Inhibit species
                    ut.print_log(log_file, f"Inhibiting species: {target_id}")
                    modified_model = sbml_ut.inhibit_species(sbml_model, target_id, log_file)
                    operation_name = f"no_species_{target_id}"

                elif operation_type == 2:
                    # Inhibit reaction
                    ut.print_log(log_file, f"Inhibiting reaction: {target_id}")
                    modified_model = sbml_ut.inhibit_reaction(sbml_model, target_id, log_file)
                    operation_name = f"no_reaction_{target_id}"

                
                #Save 
                xml_string, output_filename = sbml_ut.save_file(file_name, operation_name, modified_model, save_output, log_file)
                    
                # Second simulation - modified model
                rr_modified = sim_ut.load_roadrunner_model(xml_string)
                res_modified = sim_ut.simulate(rr_modified)
                sim_ut.plot_results(res_modified, "./imgs", output_filename, log_file)
            else:
                exit(f"Error: Failed to convert modified model to XML")

            # list_of_reaction = ut.get_list_of_reactions(sbml_model, ut.get_species_dict(species))
            # N = nu.get_network_from_sbml(list_of_reaction, species)

            # nu.plot_network(N)

    except Exception as e:
        ut.print_log(log_file, f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
