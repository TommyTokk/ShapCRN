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

    json_formatted_str = json.dumps(dict_obj, indent=2)
    print(json_formatted_str)


# ============
# SPECIES
# ============

def get_list_of_species(SBML_model):
    
    return SBML_model.getListOfSpecies()



def get_list_of_species_ids(species_list):
    
    species_ids = []

    for s in species_list:
        species_ids.append(s.getId())

    return species_ids

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

def get_reaction_components(list_of_reactions, species_dict, component_type='reactants'):
    """
    Creates a dictionary containing information about reactants or products of a model.
    
    Args:
        list_of_reactions: A list of reactions of type libsbml.ListOfReactions
        species_dict: A dictionary containing the information of the model's species.
            This dictionary should be created using the species_dict() function.
        component_type: String indicating the type of components to get ('reactants' or 'products')
        
    Returns:
        dict: Dictionary with reaction information and their components
        
    See Also:
        species_dict(species_list): Function to create a dictionary of species information
        that is required as input for this function.
    """
    # Check if component type is valid
    if component_type not in ['reactants', 'products']:
        raise ValueError("component_type must be either 'reactants' or 'products'")
    
    # Determine which method to call based on the component_type
    get_components = (lambda r: r.getListOfReactants()) if component_type == 'reactants' else \
                    (lambda r: r.getListOfProducts())
    
    result_dict = {}
    
    for reaction in list_of_reactions:
        reaction_id = reaction.getId()
        result_dict[reaction_id] = {
            component_type: []
        }
        
        
        for component in get_components(reaction):
            species_id = component.getSpecies()
            if species_id in species_dict:
                
                species_info = species_dict[species_id].copy()
                species_info['stoichiometry'] = component.getStoichiometry()
                
                result_dict[reaction_id][component_type].append({
                    's_id': species_id,
                    'info': species_info
                })
    
    return result_dict

def get_reactants_dict(list_of_reactions, species_dict):
    """
    Creates a dictionary containing all the information about the reactants of the model.
    
    Args:
        list_of_reactions: A list of reactions of type libsbml.ListOfReactions
        species_dict: A dictionary containing the information of the model's species.
        
    Returns:
        dict: Dictionary with reaction information and their reactants
    """
    return get_reaction_components(list_of_reactions, species_dict, 'reactants')

def get_products_dict(list_of_reactions, species_dict):
    """
    Creates a dictionary containing all the information about the products of the model.
    
    Args:
        list_of_reactions: A list of reactions of type libsbml.ListOfReactions
        species_dict: A dictionary containing the information of the model's species.
        
    Returns:
        dict: Dictionary with reaction information and their products
    """
    return get_reaction_components(list_of_reactions, species_dict, 'products')
            

def reactions_to_dict(list_of_reactions_obj):
    """
    Convert a list of Reaction objects to a JSON-serializable dictionary.
    
    Args:
        list_of_reactions_obj: List of Reaction objects
        
    Returns:
        dict: Dictionary representation of the reactions list
    """
    reactions_dict = {}
    
    if not list_of_reactions_obj:
        return reactions_dict
    
    for reaction in list_of_reactions_obj:
        
        reaction_id = reaction.id
        
        reaction_data = reaction.to_dict()
        
        reactions_dict[reaction_id] = reaction_data
    
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
# REACTIONS
# ============

# ============
# GENERAL
# ============
def get_functions_list(sbml_model):

    functions_formulas = []

    for fd in sbml_model.getListOfFunctionDefinitions():
    #     functions_formulas.append(formulaToL3String(fd.getMath()))

    # return functions_formulas
        print("===========================")
        for i in range(fd.getNumArguments()):
            print(formulaToL3String(fd.getArgument(i)))

        print("===========================")



