import enum
import os

import libsbml
import json
import numpy as np
import itertools

from src.classes.species import Species
from src.classes.reaction import Reaction
from src.classes.function import Function


from src.utils.utils import print_log

from enum import Enum


class Op(Enum):
    MINUS = libsbml.AST_MINUS
    PLUS = libsbml.AST_PLUS
    PROD = libsbml.AST_TIMES


def load_model(model_file_path):

    reader = libsbml.SBMLReader()

    document = reader.readSBMLFromFile(model_file_path)

    return document


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


def knockout_species(sbml_model, target_species_id, log_file=None):
    """
    Removes a species from the products of reactions and sets its initial concentration to 0.
    """
    in_rules = False

    # For each Rules in the model, pin the rules to remove
    for rule in sbml_model.getListOfRules():
        if rule.getVariable() == target_species_id:  # Found the updating rule
            in_rules = True
            # Set the math of the target_species constant to 0
            zero_ast = libsbml.ASTNode(libsbml.AST_INTEGER)
            zero_ast.setValue(0)
            rule.setMath(zero_ast)
            print_log(log_file, f"Set rule for {target_species_id} to 0")

    reactions_to_knockout = []

    if not in_rules:
        print_log(
            log_file,
            f"Species {target_species_id} not found in rules, looking in reactions...",
        )

        # Collect all reactions that need to be knocked out
        for reaction in sbml_model.getListOfReactions():
            reaction_id = reaction.getId()
            should_knockout = False

            # Check if species is a reactant
            for i in range(reaction.getNumReactants()):
                if reaction.getReactant(i).getSpecies() == target_species_id:
                    should_knockout = True
                    print_log(
                        log_file,
                        f"{target_species_id} found as reactant in {reaction_id}",
                    )
                    break

            if should_knockout:
                reactions_to_knockout.append(reaction_id)

        # Remove products (handle this separately to avoid index issues)
        products_to_remove = []
        for reaction in sbml_model.getListOfReactions():
            num_products = reaction.getNumProducts()

            if (
                reaction.getId() not in reactions_to_knockout
            ):  # Only if not already being knocked out
                for i in range(num_products):
                    if reaction.getProduct(i).getSpecies() == target_species_id:
                        products_to_remove.append((reaction.getId(), i))
                        print_log(
                            log_file,
                            f"{target_species_id} found as product in {reaction.getId()}",
                        )

        # Remove products (from highest index to lowest to avoid index shifting)
        products_by_reaction = {}
        for reaction_id, product_idx in products_to_remove:
            if reaction_id not in products_by_reaction:
                products_by_reaction[reaction_id] = []
            products_by_reaction[reaction_id].append(product_idx)

        for reaction_id, indices in products_by_reaction.items():
            reaction = sbml_model.getReaction(reaction_id)
            # Sort indices in descending order to remove from end to beginning
            for idx in sorted(indices, reverse=True):
                print_log(
                    log_file, f"Removing product at index {idx} from {reaction_id}"
                )
                reaction.removeProduct(idx)
                # if reaction.getNumProducts() == 0:
                #     reactions_to_knockout.append(reaction_id)

        # Knockout the reactions that need to be knocked out
        for reaction_id in reactions_to_knockout:
            print_log(log_file, f"Calling knockout_reaction for {reaction_id}")
            sbml_model = knockout_reaction(sbml_model, reaction_id, log_file)

    # Set the initial concentration to 0.0 and make it constant
    for species in sbml_model.getListOfSpecies():
        if species.getId() == target_species_id:
            result = species.setInitialConcentration(0.0)
            result = species.setConstant(True)
            if result == libsbml.LIBSBML_OPERATION_FAILED:
                print_log(
                    log_file, f"Error setting concentration for {species.getId()}"
                )

    return sbml_model


def knockout_species_via_reaction(sbml_model, target_species_id, log_file=None):

    species = sbml_model.getSpecies(target_species_id)
    species_comp = species.getCompartment()

    # Create the new product species

    p_species = sbml_model.createSpecies()
    p_species.setId(f"ko_{target_species_id}")
    p_species.setCompartment(species.getCompartment())
    p_species.setInitialConcentration(0)
    # p_species.setStoichiometry(1)

    # Create the new reaction
    new_reaction = sbml_model.createReaction()
    new_reaction.setId(f"d_{target_species_id}")
    new_reaction.setName(f"deactivation_of_{target_species_id}")
    new_reaction.setReversible(False)

    # Create the reactant
    react = new_reaction.createReactant()
    react.setSpecies(species.getId())
    react.setStoichiometry(1)
    react.setConstant(species.getConstant())

    # Create the product
    prod = new_reaction.createProduct()
    prod.setSpecies(p_species.getId())
    prod.setStoichiometry(1)
    prod.setConstant(False)

    # Create the kinetic law of the new reaction
    kl_deactivation = new_reaction.createKineticLaw()
    # Create the parameter Id
    ko_id = "k_ko"
    param_ko = kl_deactivation.createParameter()
    param_ko.setId(ko_id)
    param_ko.setValue(1e10)
    param_ko.setConstant(True)
    # Create MathMlL for the reaction
    math_ast = libsbml.parseL3Formula(f"{ko_id} * {species_comp} * {species.getId()}")
    kl_deactivation.setMath(math_ast)

    sbml_model.addReaction(new_reaction)

    return (sbml_model, new_reaction)


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


def split_all_reversible_reactions(model):
    """
    Split all reversible reactions in a model into forward and reverse reactions

    Args:
        model: libSBML Model object

    Returns:
        libSBML Model object: The modified model with all reversible reactions split
    """

    model_comps = [c.getId() for c in model.getListOfCompartments()]

    model_params_dict = {}

    for p in model.getListOfParameters():
        model_params_dict[p.getId()] = p.getValue()

    # Get list of reversible reaction IDs (make a copy since we'll be modifying the model)
    reversible_reaction_ids = []
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        if reaction.getReversible():
            reversible_reaction_ids.append(reaction.getId())

    print(f"Found {len(reversible_reaction_ids)} reversible reactions to split")

    # Split each reversible reaction
    for reaction_id in reversible_reaction_ids:
        forward_reaction, reverse_reaction = split_reversible_reaction(
            model, reaction_id, model_comps, model_params_dict
        )

        model.addReaction(forward_reaction)
        model.addReaction(reverse_reaction)

    return model


def parse_ast_tree(ast_node, log_file=None):
    """
    Recursively parses an ASTNode from libSBML and returns a dictionary representation.
    """

    if ast_node is None:
        return None

    # Base case: if node is a leaf (a number or a variable)
    if ast_node.getNumChildren() == 0:
        if ast_node.isName():
            return {"type": "name", "value": ast_node.getName()}
        elif ast_node.isNumber():
            return {"type": "number", "value": ast_node.getValue()}
        else:
            return {"type": "unknown", "value": ast_node.toFormula()}

    operator = ast_node.getName()
    children = [
        parse_ast_tree(ast_node.getChild(i)) for i in range(ast_node.getNumChildren())
    ]

    return {
        "type": "operator",
        "op": operator,
        "op_type": ast_node.getType(),
        "args": children,
    }


def split_reversible_reaction(
    model, reaction_id, model_compartments, model_parameters_dict, log_file=None
):
    """
    Split a reversible reaction into two irreversible reactions (forward and reverse)

    Args:
        model: libSBML Model object
        reaction_id: ID of the reversible reaction to split

    Returns:
        Tuple containing the two reactions
    """

    # TODO: Takes all the parameters and not only the local ones

    reaction = model.getReaction(reaction_id)

    if reaction is None:
        raise ValueError(f"Reaction with ID '{reaction_id}' not found in the model.")

    if not reaction.getReversible():
        raise ValueError(f"Reaction '{reaction_id}' is already irreversible.")

    reactants = reaction.getListOfReactants()
    products = reaction.getListOfProducts()

    kinetic_law = reaction.getKineticLaw()

    if kinetic_law is None:
        raise ValueError(
            f"Reaction '{reaction_id}' does not have a kinetic law defined."
        )

    # Extract parameters from the kinetic law
    parameters = {}
    for i in range(kinetic_law.getNumParameters()):
        param = kinetic_law.getParameter(i)
        parameters[param.getId()] = param.getValue()

    # Extract the nodes values of AST
    ast_nodes = kinetic_law.getMath().getListOfNodes()
    reaction_comps = []

    for i in range(ast_nodes.getSize()):
        node = ast_nodes.get(i)
        # print_log(log_file, node.isName())
        if node.isName():
            if node.getName() in model_parameters_dict.keys():
                parameters[node.getName()] = model_parameters_dict[node.getName()]
            else:
                for c in model_compartments:
                    if c == node.getName():
                        reaction_comps.append(c)

    print_log(log_file, f"{parameters}")

    reaction_comps_string = " * ".join([c for c in reaction_comps])

    # Creation of forward reaction
    forward_reaction = model.createReaction()
    forward_reaction.setId(f"{reaction_id}_forward")
    forward_reaction.setReversible(False)
    # forward_reaction.setFast(False)

    # Adding the reactants
    for reactant in reactants:
        sr = forward_reaction.createReactant()
        sr.setSpecies(reactant.getSpecies())
        sr.setStoichiometry(reactant.getStoichiometry())
        sr.setConstant(reactant.getConstant())

    # Adding the products
    for product in products:
        sr = forward_reaction.createProduct()
        sr.setSpecies(product.getSpecies())
        sr.setStoichiometry(product.getStoichiometry())
        sr.setConstant(product.getConstant())

    # Define forward kinetic law
    kl_forward = forward_reaction.createKineticLaw()
    # Create MathML for forward rate: k_forward * [A] * [B]
    reactant_species = " * ".join([r.getSpecies() for r in reactants])
    k_forward_id = list(parameters.keys())[
        0
    ]  # Assuming the first parameter corresponds to the forward rate
    math_ast_forward = libsbml.parseL3Formula(
        f"{reaction_comps_string} * {k_forward_id} * {reactant_species}"
    )
    kl_forward.setMath(math_ast_forward)

    # Add parameter k_forward
    param_k_forward = kl_forward.createParameter()
    param_k_forward.setId(k_forward_id)
    param_k_forward.setValue(parameters[k_forward_id])
    param_k_forward.setConstant(True)

    # Create reverse reaction
    reverse_reaction = model.createReaction()
    reverse_reaction.setId(f"{reaction_id}_reverse")
    reverse_reaction.setReversible(False)
    # reverse_reaction.setFast(False)

    for product in products:
        sr = reverse_reaction.createReactant()
        sr.setSpecies(product.getSpecies())
        sr.setStoichiometry(product.getStoichiometry())
        sr.setConstant(product.getConstant())

    for reactant in reactants:
        sr = reverse_reaction.createProduct()
        sr.setSpecies(reactant.getSpecies())
        sr.setStoichiometry(reactant.getStoichiometry())
        sr.setConstant(reactant.getConstant())

    # Define reverse kinetic law
    kl_reverse = reverse_reaction.createKineticLaw()
    # Create MathML for reverse rate: k_reverse * [C]
    product_species = " * ".join([p.getSpecies() for p in products])
    k_reverse_id = list(parameters.keys())[
        1
    ]  # Assuming the second parameter corresponds to the reverse rate
    math_ast_reverse = libsbml.parseL3Formula(
        f"{reaction_comps_string} * {k_reverse_id} * {product_species}"
    )
    kl_reverse.setMath(math_ast_reverse)

    # Add parameter k_reverse
    param_k_reverse = kl_reverse.createParameter()
    param_k_reverse.setId(k_reverse_id)
    param_k_reverse.setValue(parameters[k_reverse_id])
    param_k_reverse.setConstant(True)

    # Remove the original reversible reaction
    model.removeReaction(reaction_id)

    return (forward_reaction, reverse_reaction)


def knockout_reaction(sbml_model, target_reaction_id, log_file=None):
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
        zero_ast = libsbml.ASTNode(libsbml.AST_INTEGER)
        # Set the value of the node
        zero_ast.setValue(0)

        # Set the kinetic law
        kinetic_law = reaction.getKineticLaw()
        if kinetic_law is not None:
            result = kinetic_law.setMath(zero_ast)

            if result == libsbml.LIBSBML_OPERATION_SUCCESS:
                # Force model validation/consistency check
                # sbml_model.checkConsistency()
                print_log(
                    log_file, f"Successfully knocked out reaction {target_reaction_id}"
                )
            else:
                print_log(log_file, f"Error setting kinetic law: {result}")
        else:
            print_log(
                log_file, f"No kinetic law found for reaction {target_reaction_id}"
            )

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
    if isinstance(model, libsbml.Model):
        # If it's a Model, get the associated document
        doc = model.getSBMLDocument()
        if doc is None:
            # If there's no associated document, create a new one
            doc = libsbml.SBMLDocument(model.getLevel(), model.getVersion())
            doc.setModel(model)
        success = libsbml.writeSBMLToFile(doc, file_path)
    elif isinstance(model, str):
        # If it's an XML string
        reader = libsbml.SBMLReader()
        doc = reader.readSBMLFromString(model)
        success = libsbml.writeSBMLToFile(doc, file_path)
    else:
        # Otherwise, try directly
        success = libsbml.writeSBMLToFile(model, file_path)

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
    if isinstance(model, libsbml.Model):
        # If it's a Model, get the associated document
        doc = model.getSBMLDocument()
        if doc is None:
            # If there's no associated document, create a new one
            doc = libsbml.SBMLDocument(model.getLevel(), model.getVersion())
            doc.setModel(model)
        xml_string = libsbml.writeSBMLToString(doc)
    elif isinstance(model, str):
        # If it's already an XML string, return it
        return model
    else:
        # Otherwise, try to write it directly to string
        xml_string = libsbml.writeSBMLToString(model)

    if xml_string:
        return xml_string
    else:
        print_log(log_file, "Error: Failed to convert SBML to XML string")
        return None


# TODO: CHECK THIS PART
# TODO: Change the logic of generating samples
def generate_species_samples(
    sbml_model,
    target_species=[],
    log_file=None,
    n_samples=5,
    variation=20,
):
    res = []

    for ts in target_species:
        # taking initial concentration
        t0_conc = (
            sbml_model.getListOfSpecies().getElementBySId(ts).getInitialConcentration()
        )

        tmp = []

        for i in range(n_samples):
            # Sample multiplication factors between (1-variation/100) and (1+variation/100)
            factor = np.random.uniform(1 - variation / 100, 1 + variation / 100)
            sample = t0_conc + factor
            tmp.append(sample)

        res.append(tmp)

    print_log(log_file, f"[GENERATE SAMPLES]{res}")

    return res


def create_samples_combination(input_samples, log_file=None):
    # input_samples is a 2D array: e.g., [[1, 2], [3, 4], [5, 6]]
    combinations = list(itertools.product(*input_samples))

    # print_log(log_file, f"Combinations: {combinations}")
    return combinations


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


def get_selections(sbml_model, rr_model, target_ids, log_file=None):

    selections = rr_model.selections

    # Check if some target species are reaction
    for ts in target_ids:
        print_log(log_file, f"[GET SELECTIONS]{ts}")
        if ts in [r.getId() for r in sbml_model.getListOfReactions()]:
            selections = selections + [f"{ts}"]
        elif (
            ts in [s.getId() for s in sbml_model.getListOfSpecies()]
            and f"[{ts}]" not in selections
        ):
            selections = selections + [f"[{ts}]"]

    # print_log(log_file, f"[GET SELECTIONS]sel: {selections}")
    return selections


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
