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
        species_names = [s.get_name() for s in species]
        
        # Prima simulazione - modello originale
        print("Simulating original model...")
        rr = sim_ut.load_roadrunner_model(sbml_model)
        rr.setIntegrator('gillespie')
        res = sim_ut.simulate(rr)
        sim_ut.plot_results(res, species_names, "./imgs", file_name)

        # Check if we have an operation to perform
        if len(args) > 1:
            operation_type = args[1]
            target_id = args[2]
            
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
            
            # Save the modified model to file using get_sbml_as_xml
            xml_string = ut.get_sbml_as_xml(modified_model)
            if xml_string:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(xml_string)
                print(f"Modified SBML saved to: {output_path}")
            
                # Seconda simulazione - modello modificato
                print(f"Simulating model after {operation_name}:")
                # Usa il contenuto XML direttamente invece del percorso del file
                rr_modified = sim_ut.load_roadrunner_model(xml_string)
                rr_modified.setIntegrator('gillespie')
                res_modified = sim_ut.simulate(rr_modified)
                sim_ut.plot_results(res_modified, species_names, "./imgs", output_filename)
            else:
                print(f"Error: Failed to convert modified model to XML")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
