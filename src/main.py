import sbml_utils as ut
import simulation_utils as sim_ut
import numpy as np

def main():
    file_path = ut.parse_args()[0]

    try:
        rr = sim_ut.load_roadrunner_model(file_path=file_path)
        
        sbml_model = ut.load_model(file_path)
        species_list = ut.get_list_of_species(sbml_model)
        species_ids = ut.get_list_of_species_ids(species_list)
        
        print(f"Specie nel modello: {species_ids}")
        
        result = sim_ut.simulate(rr)
        
        sim_ut.plot_results(result, species_ids)
        
    except Exception as e:
        print(f"Erorr during the simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
