import sys
from libsbml import *
import json

from src.classes.species import Species
from src.classes.reaction import Reaction
from src.classes.function import Function
from src.classes.reagent import Reagent
from src.classes.product import Product

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

    json_formatted_str = json.dumps(dict_obj, indent=2)
    print(json_formatted_str)


# ============
# SPECIES
# ============

def get_list_of_species(SBML_model):
    """
    Get a list of Species objects from the SBML model.
    
    Args:
        SBML_model: SBML model object
        
    Returns:
        list: List of Species objects
    """
    species_list = []
    
    for sbml_species in SBML_model.getListOfSpecies():
        species = Species.from_sbml(sbml_species)
        species_list.append(species)

    return species_list

def get_species_dict(species_list):
    """
    Create a dictionary of Species objects indexed by ID.
    
    Args:
        species_list: List of Species objects
        
    Returns:
        dict: Dictionary of Species objects with species IDs as keys
    """
    dict = {}

    for s in species_list:
        dict[s.get_id()] = s.to_dict()

    return dict

def species_dict_list(species_list):
    """
    Convert a list of Species objects to a list of dictionaries.
    
    Args:
        species_list: List of Species objects
        
    Returns:
        list: List of dictionaries, each representing a species
    """
    return [s.to_dict() for s in species_list]

# ============
# REACTIONS
# ============

def get_list_of_reactions(SBML_model, species_dict):
    """
    Get a list of Reaction objects from the SBML model.
    
    Args:
        SBML_model: SBML model object
        species_dict: Dictionary of Species objects indexed by ID
        
    Returns:
        list: List of Reaction objects
    """
    reactions = []
    for sbml_reaction in SBML_model.getListOfReactions():
        reaction = Reaction.from_sbml(sbml_reaction, species_dict)
        reactions.append(reaction)

    return reactions

def get_reactants_dict(reaction_objects):
    """
    Creates a dictionary containing all the information about the reactants from a list of Reaction objects.
    
    Args:
        reaction_objects: A list of Reaction objects
        
    Returns:
        dict: Dictionary with reactions's id as key and reactions's reagents obj as value
    """

    dict = {}

    for r_obj in reaction_objects:
        dict[r_obj.get_id()] = r_obj.get_reagents()

    return dict



def get_products_dict(reaction_objects):
    """
    Creates a dictionary containing all the information about the products from a list of Reaction objects.
    
    Args:
        reaction_objects: A list of Reaction objects
        
    Returns:
        dict: Dictionary with reaction information and their products
    """
    dict = {}

    for r_obj in reaction_objects:
        dict[r_obj.get_id()] = r_obj.get_products()

    return dict

def reactions_to_dict(reaction_list):
    """
    Convert a list of Reaction objects to a JSON-serializable dictionary.
    
    Args:
        reaction_list: List of Reaction objects
        
    Returns:
        dict: Dictionary representation of the reactions list
    """
    reactions_dict = {}
    
    if not reaction_list:
        return reactions_dict
    
    for reaction in reaction_list:
        reaction_id = reaction.getId()
        reactions_dict[reaction_id] = reaction.to_dict()
    
    return reactions_dict

def print_reactions_as_json(reactions_list):
    """
    Print a list of Reaction objects in JSON format.
    
    Args:
        reactions_list: List of Reaction objects to print
    """
    reactions_dict = reactions_to_dict(reactions_list)
    dict_pretty_print(reactions_dict)

# ============
# FUNCTIONS
# ============

def get_functions_list(SBML_model):
    """
    Get a list of Function objects from the SBML model.
    
    Args:
        SBML_model: SBML model object
        
    Returns:
        list: List of Function objects
    """
    functions_list = []
    
    for sbml_func in SBML_model.getListOfFunctionDefinitions():
        function = Function.from_sbml(sbml_func)
        functions_list.append(function)
    
    return functions_list

def print_functions_as_json(functions_list):
    """
    Print a list of Function objects in JSON format.
    
    Args:
        functions_list: List of Function objects
    """
    functions_dict = {func.getId(): func.to_dict() for func in functions_list}
    dict_pretty_print(functions_dict)

# ============
# FUNCTIONS
# ============



