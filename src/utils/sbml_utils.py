import sys
from libsbml import *
import json
import numpy as np
import itertools

from src.classes.species import Species
from src.classes.reaction import Reaction
from src.classes.function import Function
from src.classes.reagent import Reagent
from src.classes.product import Product

from src.utils.utils import print_log


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


def save_file(file_name, operation_name, model, save_output=False, log_file=None):
    # Generate output filename
    base_name, extension = os.path.splitext(file_name)
    output_filename = f"{base_name}_{operation_name}{extension}"
    output_path = os.path.join("models", output_filename)

    # Get the XML representation of the modified model
    xml_string = get_sbml_as_xml(model, log_file)
    if xml_string:
        # Save the model only if save_output is True
        if save_output:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(xml_string)
            print_log(log_file, f"Modified SBML saved to: {output_path}")
        else:
            print_log(log_file, "Modified SBML not saved (use -so flag to save)")

    return (xml_string, output_filename)


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


def inhibit_species(sbml_model, target_species_id, log_file=None):
    """
    Removes a species from the products of reactions and sets its initial concentration to 0.

    Args:
        sbml_model: SBML model object
        target_species_id: ID of the species to inhibit

    Returns:
        Model: Updated SBML model with the inhibited species
    """

    in_rules = False

    # For each Rules in the model, pin the rules to remove
    for rule in sbml_model.getListOfRules():
        # TODO: Ask if it's a correct approach to set the math to 0
        # if rule.isAssignment():
        if rule.getVariable() == target_species_id:  # Found the updating rule
            in_rules = True
            # Set the math of the target_species costant to 0.0
            # Create the new ASTNode as REAL
            zero_ast = ASTNode(AST_REAL)
            # Set the value of the node
            zero_ast.setValue(0.0)
            # Change the math of the node
            rule.setMath(zero_ast)

    reactions_to_inhibit = []

    if not in_rules:
        print_log(
            log_file,
            f"Species {target_species_id} not found in rules, looking in reactions...",
        )
        # For each reaction in the model
        for reaction in sbml_model.getListOfReactions():
            # If the reaction has products
            # TODO: Notify the changes
            if reaction.getNumProducts() > 0:
                # Identify the products to remove
                products_to_remove = []

                # Collect the IDs of products to remove
                for i in range(reaction.getNumProducts()):
                    product = reaction.getProduct(i)
                    if product.getSpecies() == target_species_id:
                        products_to_remove.append(product.getSpecies())

                # Remove the identified products
                for product_id in products_to_remove:
                    reaction.removeProduct(product_id)
            # Collecting all the reaction that has only the target_species as product
            if reaction.getNumProducts() == 0:
                reactions_to_inhibit.append(reaction.getId())

            # Removing the species also from reactants to avoid errors
            # During the simulation numerical errors accumulate and make the concentration not 0
            if reaction.getNumReactants() > 0:
                reactants_to_remove = []

                for i in range(reaction.getNumReactants()):
                    reactant = reaction.getReactant(i)
                    if reactant.getSpecies() == target_species_id:
                        reactants_to_remove.append(reactant.getSpecies())
                # Remove the identified reactants
                for reactant_id in reactants_to_remove:
                    reaction.removeReactant(reactant_id)

            if reaction.getNumReactants() == 0:
                reactions_to_inhibit.append(reaction.getId())

        # TODO: Ask for wich approach is better
        # V1: Only removes the species from the products list without inhibiting the reaction
        #       (iff the species was the only product)
        #      - In this case reactants concentrations keeps decreasing
        #
        # V2: After removing the species from the product list, if the species was the only product,
        #       also inhibit the reaction

        if True:  # False to exec V1, True to exec V2
            for reaction in reactions_to_inhibit:
                sbml_model = inhibit_reaction(sbml_model, reaction, log_file)

    # Set the initial concentration to 0.0
    for species in sbml_model.getListOfSpecies():
        if species.getId() == target_species_id:
            result = species.setInitialConcentration(0.0)
            # result = species.setConstant(True)
            # print_log(log_file, f"result:{result}")
            if result == LIBSBML_OPERATION_FAILED:
                exit(f"Error setting concentration for {species.getId()}")

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


def inhibit_reaction(sbml_model, target_reaction_id, log_file=None):
    """
    Set the kinetic law of the reaction to 0 in the SBML model and return the updated model.

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
        # result = sbml_model.removeReaction(target_reaction_id)
        # Create the new ASTNode as REAL
        zero_ast = ASTNode(AST_REAL)
        # Set the value of the node
        zero_ast.setValue(0.0)

        # Set the kineticLaw of the reaction to 0
        result = reaction.getKineticLaw().setMath(zero_ast)

        if result == LIBSBML_OPERATION_SUCCESS:
            print_log(log_file, f"Successfully inhibite reaction {target_reaction_id}")
        else:
            print_log(
                log_file,
                f"Error during inhibition reaction {target_reaction_id}: error code {result}",
            )
    else:
        print_log(log_file, f"Reaction {target_reaction_id} not found in the model")

    return sbml_model


# def inhibit_reaction_rr(rr_model, target_reaction_id, log_file=None):
#     # TODO: Ask which of the two methods is better
#     """
#     Set the kineticLaw of a reaction to 0, inhibiting it

#     Args:
#         rr_model: roadrunner model object
#         target_reaction_id: ID of the reaction to remove

#     Returns:
#     """
#     #Set the kineticLaw to 0
#     rr_model.setKineticLaw(target_reaction_id, "0", True)


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


def save_sbml_model(model, file_path, log_file):
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
        print_log(log_file, f"Successfully saved SBML to: {file_path}")
    else:
        print_log(log_file, f"Error: Failed to write SBML file to {file_path}")

    return success


def get_sbml_as_xml(model, log_file=None):
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
        print_log(log_file, "Error: Failed to convert SBML to XML string")
        return None


# TODO: CHECK THIS PART
def generate_species_samples(
    sbml_model,
    target_species=["ACEx", "GLCx", "P"],
    log_file=None,
    n_samples=5,
    variation=20,
):
    res_triple = []

    for ts in target_species:
        # taking intial concentration
        t0_conc = (
            sbml_model.getListOfSpecies().getElementBySId(ts).getInitialConcentration()
        )

        sample_lower_bound = t0_conc - ((variation / 100) * t0_conc)
        sample_upper_bound = t0_conc + ((variation / 100) * t0_conc)

        tmp = []

        for i in range(0, n_samples):
            tmp.append(np.random.uniform(sample_lower_bound, sample_upper_bound))

        res_triple.append(tmp)

    return tuple(res_triple)


def create_samples_combination(input_samples, log_file=None):
    ACEx_samples, GLCx_samples, P_samples = input_samples

    combinantions = list(itertools.product(ACEx_samples, GLCx_samples, P_samples))

    return combinantions


# FOR DEBUG ONLY
def check_for_duplicates(combinations, log_file=None):
    """
    Check if there are duplicate combinations in the list.

    Args:
        combinations: List of combinations to check
        log_file: Optional log file for output

    Returns:
        tuple: (has_duplicates, num_duplicates)
    """
    # Converto ogni combinazione in tupla se già non lo è
    # (le liste non sono hashable, le tuple sì)
    tuple_combinations = [
        tuple(combo) if not isinstance(combo, tuple) else combo
        for combo in combinations
    ]

    # Confronto lunghezze
    original_length = len(combinations)
    unique_combinations = set(tuple_combinations)
    unique_length = len(unique_combinations)

    if original_length == unique_length:
        print_log(log_file, f"No duplicates found in {original_length} combinations.")
        return False, 0
    else:
        # Ci sono duplicati
        num_duplicates = original_length - unique_length
        print_log(
            log_file,
            f"Found {num_duplicates} duplicates in {original_length} combinations.",
        )

        # Trova e stampa alcuni esempi di duplicati
        from collections import Counter

        counts = Counter(tuple_combinations)
        duplicates = {combo: count for combo, count in counts.items() if count > 1}

        print_log(log_file, "Examples of duplicates:")
        for i, (combo, count) in enumerate(duplicates.items()):
            print_log(log_file, f"  Combination {combo} appears {count} times")
            if i >= 2:  # Mostra al massimo 3 esempi
                break

        return True, num_duplicates


# FOR DEBUG ONLY
def check_presence(sbml_model, target_species_id, log_file):
    in_rules = False
    in_reactions = False

    for rule in sbml_model.getListOfRules():
        # TODO: Ask if it's a correct approach to set the math to 0
        # if rule.isAssignment():
        if rule.getVariable() == target_species_id:  # Found the updating rule
            in_rules = True

    for reaction in sbml_model.getListOfReactions():
        for i in range(reaction.getNumProducts()):
            product = reaction.getProduct(i)
            if product.getSpecies() == target_species_id:
                in_reactions = True

    print_log(
        log_file,
        f"species {target_species_id}, both in rules ({in_rules}) and reactions ({in_reactions})? {in_rules and in_reactions}",
    )
