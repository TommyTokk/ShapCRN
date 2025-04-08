import os
import sys

# Aggiungi il percorso della cartella src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import sbml_utils as ut
from src.utils import simulation_utils as sim_ut
from src.classes.species import Species
from src.classes.reaction import Reaction

def main():
    file_path = ut.parse_args()[0]

    try:
        # Carica il modello SBML
        sbml_model = ut.load_model(file_path)
        
        # Crea gli oggetti Species
        species_dict = {}
        for sbml_species in sbml_model.getListOfSpecies():
            species = Species.from_sbml(sbml_species)
            species_dict[species.id] = species

        
        # Crea gli oggetti Reaction
        reactions = []
        for sbml_reaction in sbml_model.getListOfReactions():
            reaction = Reaction.from_sbml(sbml_reaction, species_dict)
            reactions.append(reaction)
        
        # Simulazione (se necessario)
        rr = sim_ut.load_roadrunner_model(file_path=file_path)
        result = sim_ut.simulate(rr)
        sim_ut.plot_results(result, list(species_dict.keys()))
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
