import roadrunner as rr
import sys
from libsbml import *
import json

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


def dict_pretty_print(dict_obj):
    """Pretty print a dictionary as formatted JSON.
    
    Args:
        dict_obj: Dictionary to be printed
    """
    # Usa json.dumps una sola volta sul dizionario originale
    json_formatted_str = json.dumps(dict_obj, indent=2)
    print(json_formatted_str)


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
    specie_dict['name'] = specie_obj.getName()
    specie_dict['compartment'] = specie_obj.getCompartment()
    specie_dict['stoichiometry'] = ""

    return specie_dict

def species_dict(species_list):
    dict = {}
    for specie in species_list:
        specie_id = specie.getId()
        dict[specie_id] = specie_to_dict(specie)

    return dict




# ============
# SPECIES
# ============


# ============
# REACTIONS
# ============
def get_list_of_reactions(SBML_model):
    return SBML_model.getListOfReactions()

def get_reactants_dict(list_of_reactions, species_dict):
    """
    Creates a dictionary containing all the information about the reactants of the model,
    based on the reaction.

    Args:
        - list_of_reaction: A list of reactions of type libsbml.ListOfReactions
        - species_dict: A dictionary containing the information of the model's species.
        This dictionary should be created using the species_dict() function.

    See Also:
        - species_dict(species_list): Function to create a dictionary of species information
        that is required as input for this function.
    """
    
    reactants_dict = {}

    for reaction in list_of_reactions:
        reactants_dict[reaction.getId()] = {

            'name': reaction.getName(),
            'reactants': []
            
        }

        for reactant in reaction.getListOfReactants():
            specie_id = reactant.getSpecies()
            if specie_id in species_dict:
                s_dict = species_dict[specie_id]
                s_dict['stoichiometry'] = reactant.getStoichiometry()
                reactants_dict[reaction.getId()]['reactants'].append({
                    's_id': specie_id,
                    'info': s_dict
                })


    return reactants_dict


def get_products_dict(list_of_reactions, species_dict):
    """
    Creates a dictionary containing all the information about the products of the model,
    based on the reaction.

    Args:
        - list_of_reaction: A list of reactions of type libsbml.ListOfReactions
        - species_dict: A dictionary containing the information of the model's species.
        This dictionary should be created using the species_dict() function.

    See Also:
        - species_dict(species_list): Function to create a dictionary of species information
        that is required as input for this function.
    """
    
    products_dict = {}

    for reaction in list_of_reactions:
        products_dict[reaction.getId()] = {

            'name': reaction.getName(),
            'products': []
            
        }

        for reactant in reaction.getListOfProducts():
            specie_id = reactant.getSpecies()
            if specie_id in species_dict:
                s_dict = species_dict[specie_id]
                s_dict['stoichiometry'] = reactant.getStoichiometry()
                products_dict[reaction.getId()]['products'].append({
                    's_id': specie_id,
                    'info': s_dict
                })


    return products_dict
            


"""#TODO: Creare la funzione che ritorni un dizionario con tutte le informazioni utili
            delle reazioni"""
def reactions_to_dict(list_of_reactions):
    pass
# ============
# REACTIONS
# ============



