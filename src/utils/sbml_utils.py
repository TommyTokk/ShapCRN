import sys
from libsbml import *
import json

from src.classes.species import Species
from src.classes.reaction import Reaction
from src.classes.function import Function
from src.classes.reagent import Reagent
from src.classes.product import Product

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        list: [model_path, operation_type (optional), target_id (optional)]
        
        operation_type:
            1 - Inhibit a species
            2 - Inhibit a reaction
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process SBML models and perform operations such as species or reaction inhibition.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-f', '--file', type=str, required=True, 
                        help='Path to the SBML model file')
    parser.add_argument('-op', '--operation', type=int, choices=[1, 2],
                        help='Operation to perform:\n1 - Inhibit a species\n2 - Inhibit a reaction')
    parser.add_argument('-tn', '--target_id', type=str,
                        help='ID of the target species or reaction to inhibit')
    
    parsed_args = parser.parse_args()
    
    # Convert to the expected return format
    args = [parsed_args.file]
    
    # Add operation_type if provided
    if parsed_args.operation is not None:
        args.append(parsed_args.operation)
        
        # If operation is specified but target_id is not
        if parsed_args.target_id is None:
            parser.error("Error: When using --operation/-op, you must also specify --target_id/-tn")
        
        # Add target_id
        args.append(parsed_args.target_id)
    elif parsed_args.target_id is not None:
        # If target_id is specified but operation is not
        parser.error("Error: When using --target_id/-tn, you must also specify --operation/-op")
    
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

def inhibit_species(sbml_model, target_species_id):
    """
    Removes a species from the products of reactions and sets its initial concentration to 0.
    
    Args:
        sbml_model: SBML model object
        target_species_id: ID of the species to inhibit
        
    Returns:
        Model: Updated SBML model with the inhibited species
    """
    
    # For each reaction in the model
    for reaction in sbml_model.getListOfReactions():
        # If the reaction has products
        if reaction.getNumProducts() > 0:
            # Identify the products to remove
            #This method avoid to get the data structure error for modifying while ciclying on it
            products_to_remove = []
            
            # Collect the IDs of products to remove
            for i in range(reaction.getNumProducts()):
                product = reaction.getProduct(i)
                if product.getSpecies() == target_species_id:
                    products_to_remove.append(product.getSpecies())
            
            # Remove the identified products
            for product_id in products_to_remove:
                reaction.removeProduct(product_id)
    
    # Set the initial concentration to 0.0
    for species in sbml_model.getListOfSpecies():
        if species.getId() == target_species_id:
            result = species.setInitialConcentration(0.0)
            if result:
                print(f"Error setting concentration for {species.getId()}")
    
    return sbml_model

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

def inhibit_reaction(sbml_model, target_reaction_id):
    """
    Remove a reaction from the SBML model and return the updated model.
    
    Args:
        sbml_model: SBML model object
        target_reaction_id: ID of the reaction to remove
        
    Returns:
        Model: Updated SBML model with the reaction removed
    """
    # Verify if the reaction exists
    reaction = sbml_model.getReaction(target_reaction_id)
    
    if reaction is not None:
        # Remove a reaction from the model
        result = sbml_model.removeReaction(target_reaction_id)
        
        if result:
            print(f"Successfully removed reaction {target_reaction_id}")
        else:
            print(f"Error removing reaction {target_reaction_id}: error code {result}")
    else:
        print(f"Reaction {target_reaction_id} not found in the model")

    return sbml_model

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

def save_sbml_model(model, file_path):
    """
    Save an SBML model to a file. The model can be a libsbml.Model, a string (XML),
    or a libsbml.SBMLDocument.
    
    Args:
        model: The SBML model to save. Can be a libsbml.Model, a string (XML), or a libsbml.SBMLDocument.
        file_path: Path where to save the file
        
    Returns:
        bool: True if the save was successful, False otherwise
    """
    # Check the type of the model and convert to SBMLDocument if necessary
    if isinstance(model, Model):
        # If it's a Model, get the associated document
        doc = model.getSBMLDocument()
        if doc is None:
            # If there's no associated document, create a new one
            doc = SBMLDocument(model.getLevel(), model.getVersion())
            doc.setModel(model)
        success = writeSBMLToFile(doc, file_path)
    elif isinstance(model, str):
        # If it's an XML string
        reader = SBMLReader()
        doc = reader.readSBMLFromString(model)
        success = writeSBMLToFile(doc, file_path)
    else:
        # Otherwise, try directly
        success = writeSBMLToFile(model, file_path)
    
    if success:
        print(f"Successfully saved SBML to: {file_path}")
    else:
        print(f"Error: Failed to write SBML file to {file_path}")
        
    return success

def get_sbml_as_xml(model):
    """
    Converts an SBML model to its XML string representation.
    
    Args:
        model: The SBML model. Can be a libsbml.Model, a string (XML), or a libsbml.SBMLDocument.
        
    Returns:
        str: The XML string representation of the SBML model
    """
    # Check the type of the model and convert to SBMLDocument if necessary
    if isinstance(model, Model):
        # If it's a Model, get the associated document
        doc = model.getSBMLDocument()
        if doc is None:
            # If there's no associated document, create a new one
            doc = SBMLDocument(model.getLevel(), model.getVersion())
            doc.setModel(model)
        xml_string = writeSBMLToString(doc)
    elif isinstance(model, str):
        # If it's already an XML string, return it
        return model
    else:
        # Otherwise, try to write it directly to string
        xml_string = writeSBMLToString(model)
    
    if xml_string:
        return xml_string
    else:
        print("Error: Failed to convert SBML to XML string")
        return None



