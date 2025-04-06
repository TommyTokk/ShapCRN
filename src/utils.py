import roadrunner as rr
import sys
from libsbml import *

def parse_args():

    args = []

    if len(sys.argv) != 2:
        exit("Usage: python main.py <path_to_sbml_file>")

    args.append(sys.argv[1])

    return args

def load_model(model_file_path):
    
    reader = SBMLReader()

    document = reader.readSBMLFromFile(model_file_path)

    model = document.getModel()

    return model


# ============
# SPECIES
# ============

def get_list_of_species(SBML_model):
    
    return SBML_model.getListOfSpecies()



def get_list_of_species_names(species_list):
    
    species_names = []

    for s in species_list:
        species_names.append(s.getName())

    return species_names

def specie_to_dict(specie_obj):
    specie_dict = {}

    specie_dict['id'] = specie_obj.getId()
    specie_dict['name'] = specie_obj.getName()
    specie_dict['compartment'] = specie_obj.getCompartment()

    return specie_dict

# ============
# SPECIES
# ============


# ============
# REACTIONS
# ============
def get_list_of_reactions(SBML_model):
    return SBML_model.getListOfReactions()

def get_reactants_dict(list_of_reactions):
    pass

def reactions_to_dict(list_of_reactions):
    pass
# ============
# REACTIONS
# ============



