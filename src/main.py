import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import roadrunner

# Aggiungi il percorso della cartella src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as ut
from src.utils import simulation_utils as sim_ut
from src.utils import net_utils as nu

from SBML_batch import PetriNets



def main():
    model_dir_path = sys.argv[1]

    # Estrai il percorso della directory e il nome del file
    dir_path = os.path.dirname(model_dir_path)
    file_name = os.path.basename(model_dir_path)

    # Se dir_path è vuoto, imposta la directory corrente
    if dir_path == "":
        dir_path = "."  # "." rappresenta la directory corrente

    print(f"Directory path: {dir_path}")
    print(f"File name: {file_name}")

    try:
        sbml_model = ut.load_model(model_dir_path)
        
        rr = sim_ut.load_roadrunner_model(model_dir_path)

        rr.getIntegrator().setValue('relative_tolerance', 1e-6)
        rr.getIntegrator().setValue('absolute_tolerance', 1e-8)

        species_list = ut.get_list_of_species(sbml_model)
        species_dict = ut.get_species_dict(species_list)
        reaction_list = ut.get_list_of_reactions(sbml_model, species_dict)

        N = nu.get_network_from_sbml(reaction_list, species_list)


        res = rr.simulate()

        sim_ut.plot_results(res, [s.get_name() for s in species_list])
        N, rr = nu.knockout_reaction_node(N, "r1", rr)
        
        nu.plot_network(N)

        rr.getIntegrator().setValue('relative_tolerance', 1e-6)
        rr.getIntegrator().setValue('absolute_tolerance', 1e-8)

        res = rr.simulate()

        sim_ut.plot_results(res, [s.get_name() for s in species_list])

        # for node, data in N.nodes(data=True):
        #     if(data.get('type') == 'reaction'):
        #         r = data.get('reaction')
        #         ut.dict_pretty_print(r.to_dict())



    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
