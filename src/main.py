import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import roadrunner
import libsbml

# Add the src folder path to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu

from SBML_batch import PetriNets


def main():
    # Use the parse_args function to get command-line arguments
    args = ut.parse_args()
    
    # Extract model path from args
    model_dir_path = args[0]
    
    # Extract directory path and file name
    dir_path = os.path.dirname(model_dir_path)
    file_name = os.path.basename(model_dir_path)

    # If dir_path is empty, set current directory
    if dir_path == "":
        dir_path = "."  # "." represents current directory

    print(f"Directory path: {dir_path}")
    print(f"File name: {file_name}")

    try:
        # Load model
        sbml_model = ut.load_model(model_dir_path)

        species = ut.get_list_of_species(sbml_model)
        species_ids = [s.get_id() for s in species]
        
        # Prima simulazione - modello originale
        print("Simulating original model...")
        rr = sim_ut.load_roadrunner_model(sbml_model)
        #rr.setIntegrator('gillespie')
        res = sim_ut.simulate(rr)
        sim_ut.plot_results(res, "./imgs", file_name)

        rr.reset()

        # Check if we have an operation to perform
        if len(args) > 1:
            operation_type = args[1]
            target_id = args[2]
            save_output = False
            
            # Check if save_output flag is specified
            if len(args) > 3:
                save_output = args[3]
            
            # Perform the specified operation
            if operation_type == 1:
                # Inhibit species
                print(f"Inhibiting species: {target_id}")
                modified_model = ut.inhibit_species(sbml_model, target_id)
                operation_name = f"no_species_{target_id}"

            elif operation_type == 2:
                # Inhibit reaction
                print(f"Inhibiting reaction: {target_id}")
                modified_model = ut.inhibit_reaction(sbml_model, target_id)
                operation_name = f"no_reaction_{target_id}"
            
            # Generate output filename
            base_name, extension = os.path.splitext(file_name)
            output_filename = f"{base_name}_{operation_name}{extension}"
            output_path = os.path.join("models", output_filename)
            
            # Get the XML representation of the modified model
            xml_string = ut.get_sbml_as_xml(modified_model)
            if xml_string:
                # Save the model only if save_output is True
                if save_output:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'w') as f:
                        f.write(xml_string)
                    print(f"Modified SBML saved to: {output_path}")
                else:
                    print("Modified SBML not saved (use -so flag to save)")
                
                # Seconda simulazione - modello modificato
                print(f"Simulating model after {operation_name}:")
                # Usa il contenuto XML direttamente invece del percorso del file
                rr_modified = sim_ut.load_roadrunner_model(xml_string)
                #rr_modified.setIntegrator('gillespie')
                res_modified = sim_ut.simulate(rr_modified)
                sim_ut.plot_results(res_modified, "./imgs", output_filename)
            else:
                print(f"Error: Failed to convert modified model to XML")

            # list_of_reaction = ut.get_list_of_reactions(sbml_model, ut.get_species_dict(species))
            # N = nu.get_network_from_sbml(list_of_reaction, species)

            # nu.plot_network(N)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
