import os
from threading import local
from typing import Generator


import libsbml

import numpy as np
import itertools

from src import exceptions
from src.classes.species import Species
from src.classes.reaction import Reaction
from src.classes.function import Function


from src.utils.utils import print_log

from enum import Enum
import re


class Op(Enum):
    MINUS = libsbml.AST_MINUS
    PLUS = libsbml.AST_PLUS
    PROD = libsbml.AST_TIMES


def load_model(model_file_path: str) -> libsbml.SBMLDocument:
    """
    Load an SBML model from the specified file path.

    Parameters
    ----------
    model_file_path : str
        The file path to the SBML model.

    Returns
    -------
    libsbml.SBMLDocument
        The loaded SBML document.
    """
    reader = libsbml.SBMLReader()

    document = reader.readSBMLFromFile(model_file_path)

    return document


def create_ko_models(
    target_ids: list, sbml_model, sbml_str: str, log_file=None
) -> dict:
    """
    Create the knockout models for the specified target IDs.

    Parameters
    ----------
    target_ids : list
        List of target IDs to knockout.
    sbml_model : libsbml.Model
        The original SBML model.
    sbml_str : str
        The SBML model as a string.
    log_file : file, optional
        File to log information, by default None.

    Returns
    -------
    dict
        A dictionary with target IDs as keys and their corresponding knockout SBML models as values.
    """
    model_dict = {}
    for ids in target_ids:
        doc_copy = libsbml.readSBMLFromString(sbml_str)  # rebuild in memory
        model_copy = doc_copy.getModel()
        if ids in [s.getId() for s in sbml_model.getListOfSpecies()]:
            modified_model = knockout_species(model_copy, ids, log_file)

        elif ids in [r.getId() for r in sbml_model.getListOfReactions()]:
            modified_model = knockout_reaction(model_copy, ids, log_file)

        else:
            raise Exception("Id not present in the model")

        doc_copy.setModel(modified_model)
        model_dict[ids] = libsbml.writeSBMLToString(doc_copy)

    return model_dict


# KEEP
def save_file(
    file_name: str,
    operation_name: str,
    model: libsbml.Model,
    save_output: bool = False,
    save_path: str = "./models",
    log_file=None,
) -> tuple:
    """
    Docstring per save_file

    Parameters
    ----------
    file_name : str
        Name of the input file
    operation_name : str
        Name of the operation performed
    model : libsbml.Model
        SBML model object
    save_output : bool, optional
        Flag to save the output file, by default False
    save_path : str, optional
        Path to save the output file, by default "./models"
    log_file : file, optional
        File to log information, by default None

    Returns
    -------
    tuple
        A tuple containing the XML string of the modified model and the output filename
    """
    # Generate output filename

    base_name, extension = os.path.splitext(file_name)
    print(f"{file_name}")
    print(f"{base_name}, {extension}")

    output_filename = f"{base_name}_{operation_name}{extension}"
    output_path = os.path.join(save_path, output_filename)

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


def get_list_of_species(SBML_model: libsbml.Model) -> list:
    """
    Get a list of Species objects from the SBML model.

    Parameters
    ----------
    SBML_model : libsbml.Model
        SBML model object

    Returns
    -------
    list
        List of Species objects
    """
    species_list = []

    for sbml_species in SBML_model.getListOfSpecies():
        species = Species.from_sbml(sbml_species)
        species_list.append(species)

    return species_list


# KEEP
def get_species_dict(species_list: list) -> dict:
    """
    Create a dictionary of Species objects indexed by ID.

    Parameters
    ----------
    species_list : list
        List of Species objects

    Returns
    -------
    dict
        Dictionary of Species objects indexed by ID
    """
    dict = {}

    for s in species_list:
        dict[s.get_id()] = s.to_dict()

    return dict


def species_dict_list(species_list: list) -> list:
    """
    Convert a list of Species objects to a list of dictionaries.

    Parameters
    ----------
    species_list : list
        List of Species objects

    Returns
    -------
    list
        List of dictionaries representing the species
    """
    return [s.to_dict() for s in species_list]


# KEEP
def knockout_species(
    sbml_model: libsbml.Model, target_species_id: str, log_file=None
) -> libsbml.Model:
    """
    Knockout the target species with the following logic:
        - If the target species is a reactant then remove the reaction
        - If the target species is a product, then remove the species from the products
        - Set the initial concentration of the species to 0

    Parameters
    ----------
    sbml_model : libsbml.Model
        SBML model to use
    target_species_id : str
        Target species's id
    log_file : file, optional
        File to use to print informations

    Returns
    -------
    libsbml.Model
        The modified SBML model
    """
    in_rules = False

    # Set the math of the target_species constant to 0
    zero_ast = libsbml.ASTNode(libsbml.AST_INTEGER)
    zero_ast.setValue(0)

    # For each Rules in the model, pin the rules to remove
    for rule in sbml_model.getListOfRules():
        if rule.getVariable() == target_species_id:  # Found the updating rule
            in_rules = True
            rule.setMath(zero_ast)
            print_log(log_file, f"Set rule for {target_species_id} to 0")

    # Removing every assigment events for the target species
    for event in sbml_model.getListOfEvents():
        eas = event.getListOfEventAssignments()
        for ea in eas:
            if ea.getVariable() == target_species_id:
                print_log(log_file, f"Event assigment for {target_species_id} set to 0")
                eas.setMath(zero_ast)

    # Removing every initial assigments for the target species
    for ias in sbml_model.getListOfInitialAssignments():
        if ias.getSymbol() == target_species_id:
            print_log(log_file, f"Initial assigment for {target_species_id} set to 0")
            ias.setMath(zero_ast)

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
                if reaction.getNumProducts() == 0:
                    reactions_to_knockout.append(reaction_id)

        # Knockout the reactions that need to be knocked out
        for reaction_id in reactions_to_knockout:
            print_log(log_file, f"Calling knockout_reaction for {reaction_id}")
            sbml_model = knockout_reaction(sbml_model, reaction_id, log_file)

    # Set the initial concentration to 0.0 and make it constant
    for species in sbml_model.getListOfSpecies():
        if species.getId() == target_species_id:
            result = species.setInitialConcentration(0.0)
            result = species.setBoundaryCondition(True)
            if result == libsbml.LIBSBML_OPERATION_FAILED:
                print_log(
                    log_file, f"Error setting concentration for {species.getId()}"
                )

    return sbml_model


def knockout_species_via_reaction(
    sbml_model: libsbml.Model, target_species_id: str, log_file=None
) -> tuple:
    """
    Knockout the target species creating a new, fast reaction that consume the species.

    Parameters
    ----------
    sbml_model : libsbml.Model
        SBML model to use
    target_species_id : str
        Target species's id
    log_file : file, optional
        File to use to print informations

    Returns
    -------
    tuple
        The modified SBML model and the new reaction created
    """

    # Get the species obj from the model
    species = sbml_model.getSpecies(target_species_id)
    species_comp = species.getCompartment()

    # Creating the new kinetic constant
    # parameters = list(sbml_model.getListOfParameters())
    #
    # if len(parameters) > 0:
    #     max_p = np.max([p.getValue() for p in parameters]) * 100
    # else:
    #     max_p = 1e20

    # Create the new product species

    # p_species = sbml_model.createSpecies()
    # p_species.setId(f"ko_{target_species_id}")
    # p_species.setCompartment(species.getCompartment())
    # p_species.setInitialConcentration(0)
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
    react.setBoundaryCondition(species.getBoundaryCondition())

    # Create the product
    # prod = new_reaction.createProduct()
    # prod.setSpecies(p_species.getId())
    # prod.setStoichiometry(1)
    # prod.setConstant(False)

    # Create the kinetic law of the new reaction
    kl_deactivation = new_reaction.createKineticLaw()
    # Create the parameter Id
    ko_id = "k_ko"
    param_ko = kl_deactivation.createParameter()
    param_ko.setId(ko_id)
    param_ko.setValue(1e20)
    param_ko.setConstant(True)
    # Create MathMlL for the reaction
    math_ast = libsbml.parseL3Formula(f"{ko_id} * {species_comp} * {species.getId()}")
    kl_deactivation.setMath(math_ast)

    sbml_model.addReaction(new_reaction)

    return (sbml_model, new_reaction)


def knockin_species(
    sbml_model: libsbml.Model, species_id: str, new_val: float, log_file=None
) -> libsbml.Model:
    """
    Perform a knock-in operation on a species by setting it to a fixed concentration/amount.

    This function modifies the target species by:
    1. Setting its initial concentration or amount to the specified value
    2. Renaming it by appending '_KI' suffix to the original ID
    3. Making it constant (boundary condition = True, constant = True)

    The function automatically detects whether the species uses substance units (amounts)
    or concentration values and applies the new value accordingly.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the target species
    species_id : str
        The identifier of the species to knock-in
    new_val : float
        The new initial concentration or amount value to set
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    libsbml.Model
        The modified SBML model with the knocked-in species

    Raises
    ------
    SystemExit
        If the species doesn't exist in the model or if initial values cannot be accessed

    Notes
    -----
    The modified species becomes a constant boundary condition, meaning its value
    will remain fixed throughout simulations.
    """

    species = sbml_model.getSpecies(species_id)

    if species is not None:
        # Check if the species has concentration or amounts
        if species.getHasOnlySubstanceUnits() or species.isSetInitialAmount():
            print_log(log_file, f"Using amounts for {species_id}")
            # Setting the new amounts for the species
            species.setInitialAmount(new_val)
        elif species.isSetInitialConcentration():
            print_log(log_file, f"Using concentration for {species_id}")
            species.setInitialConcentration(new_val)
        else:
            raise exceptions.InvalidSpeciesError(
                species_id, sbml_model.getId(), "Cannot access species initial value"
            )

        species.setId(species_id + "_KI")

        # Make the species constant
        species.setBoundaryCondition(True)
        species.setConstant(True)

    else:
        raise exceptions.InvalidSpeciesError(
            species_id, sbml_model.getId(), "Species not presente in the model"
        )

    return sbml_model


# ============
# REACTIONS
# ============


# KEEP
def get_list_of_reactions(sbml_model: libsbml.Model, species_dict: dict) -> list:
    """
    Get a list of Reaction objects from the SBML model.

    Parameters
    ----------
    sbml_model : libsbml.Model
        SBML model object
    species_dict : dict
        Dictionary of Species objects indexed by ID

    Returns
    -------
    list
        List of Reaction objects
    """
    reactions = []
    for sbml_reaction in sbml_model.getListOfReactions():
        reaction = Reaction.from_sbml(sbml_reaction, species_dict)
        reactions.append(reaction)

    return reactions


def get_reactants_dict(reaction_objects: list) -> dict:
    """
    Creates a dictionary containing all the information about the reactants from a list of Reaction objects.

    Parameters
    ----------
    reaction_objects : list
        A list of Reaction objects

    Returns
    -------
    dict
        Dictionary with reaction information and their reactants
    """

    dict = {}

    for r_obj in reaction_objects:
        dict[r_obj.get_id()] = r_obj.get_reagents()

    return dict


def get_products_dict(reaction_objects: list) -> dict:
    """
    Creates a dictionary containing all the information about the products from a list of Reaction objects.

    Parameters
    ----------
    reaction_objects : list
        A list of Reaction objects

    Returns
    -------
    dict
        Dictionary with reaction information and their products
    """
    dict = {}

    for r_obj in reaction_objects:
        dict[r_obj.get_id()] = r_obj.get_products()

    return dict


def get_nodes_iterator(node: libsbml.ASTNode) -> Generator[libsbml.ASTNode]:
    """
    Generator function to recursively yield all nodes in an AST.

    Parameters
    ----------
    node : libsbml.ASTNode
        The root AST node to start the traversal from.
    Yields
    ------
    libsbml.ASTNode
        Each node in the AST, traversed in pre-order.
    """
    if node is None:
        return

    # Yield the current node immediately
    yield node

    # Yield from children
    for i in range(node.getNumChildren()):
        # 'yield from' seamlessly streams values from the recursive call
        yield from get_nodes_iterator(node.getChild(i))


def get_kinetic_type(
    sbml_model: libsbml.Model, kl_math: libsbml.ASTNode, log_file=None
) -> tuple:
    """
    Determine the kinetic rate law type of a reaction by analyzing its mathematical expression.

    This function recursively traverses the Abstract Syntax Tree (AST) of the kinetic law
    to identify the reaction kinetics type based on the presence of specific mathematical
    operations or function definitions.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing function definitions
    kl_math : libsbml.ASTNode
        The AST node representing the kinetic law mathematical expression
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    tuple of (int, str or None)
        A tuple containing:
        - Kinetic type code:
            * -1: No kinetic law found or error
            * 1: Explicit Mass Action kinetic
            * 2: Explicit Michaelis-Menten kinetic
        - Function name (str) if kinetics defined by a function, None otherwise

    Notes
    -----
    Detection logic:
    - If AST contains a FUNCTION node, recursively analyzes the function definition
    - If AST contains a DIVIDE node, assumes Michaelis-Menten kinetics
    - Otherwise, assumes Mass Action kinetics
    """

    if kl_math is None:
        raise exceptions.InvalidKineticLawError("Cannot get the type of the kinetic")

    for node in get_nodes_iterator(kl_math):
        # Get the node type
        type = node.getType()

        if type == libsbml.AST_FUNCTION:
            # Kinetic defined by a function
            fn = node.getName()

            # Retrieve the function
            react_func = sbml_model.getFunctionDefinition(fn)

            # Retrieve the function math
            fn_math = react_func.getMath()

            fn_kt, _ = get_kinetic_type(sbml_model, fn_math)

            return (fn_kt, fn)

        if type == libsbml.AST_DIVIDE:
            # Kinetic likely defined by Michaelis-Menten
            return (2, None)

    # Kientic likely defined by explicit Law of Mass Action
    return (1, None)


def reactions_to_dict(reaction_list: list) -> dict:
    """
    Convert a list of Reaction objects to a JSON-serializable dictionary.

    Parameters
    ----------
    reaction_list : list
        List of Reaction objects
    Returns
    -------
    dict
        Dictionary representation of the reactions
    """
    reactions_dict = {}

    if not reaction_list:
        return reactions_dict

    for reaction in reaction_list:
        reaction_id = reaction.getId()
        reactions_dict[reaction_id] = reaction.to_dict()

    return reactions_dict


def is_reversible(sbml_model: libsbml.Model, reaction: libsbml.Reaction, log_file=None):
    """
    Determine if a reaction is reversible based on its kinetic law.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the reaction.
    reaction : libsbml.Reaction
        The reaction to analyze.
    log_file : file, optional
        File object for logging operations, by default None.
    Returns
    -------
    bool
        True if the reaction is reversible, False otherwise.
    """

    kl = reaction.getKineticLaw()

    if kl is None:
        raise exceptions.InvalidKineticLawError(reaction.getId())

    kl_math = kl.getMath()

    kl_type, fn = get_kinetic_type(sbml_model, kl_math, log_file)

    if kl_type == -1:
        raise exceptions.InvalidKineticLawError(reaction.getId(), "No kinetic law found")

    if kl_type == 2:  # Is Michaelis-Menten
        return False

    if kl_type == 1:  # Is Law of Mass Action
        if fn is not None:
            function = sbml_model.getFunctionDefinition(fn)

            f_math = function.getMath()
            for node in get_nodes_iterator(f_math):
                if node.getType() == libsbml.AST_MINUS:
                    return True
        else:
            for node in get_nodes_iterator(kl_math):
                if node.getType() == libsbml.AST_MINUS:
                    return True

    else:
        return False


# KEEP
def split_all_reversible_reactions(
    model: libsbml.Model, log_file=None
) -> libsbml.Model:
    """
    Split all reversible reactions in an SBML model into separate forward and reverse reactions.

    This function identifies all reversible reactions in the model and converts each into
    two irreversible reactions (forward and reverse). The function handles reactions with
    kinetics defined either explicitly (Mass Action) or via function definitions.

    Parameters
    ----------
    model : libsbml.Model
        The SBML model containing reactions to split
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    libsbml.Model
        The modified SBML model with all reversible reactions split into forward and
        reverse irreversible reactions

    Notes
    -----
    The function processes reactions based on their kinetic law type:
    - Explicit kinetics: Uses `split_reversible_reaction_explicit()`
    - Function-based kinetics: Uses `split_reversible_reaction_function()`

    Original reversible reactions are removed from the model after successful splitting.
    New reactions are named with '_fwd' and '_rev' suffixes.
    Function definitions are also split when reactions use function-based kinetics.
    """

    # Get the model compartments
    model_comps = [c.getId() for c in model.getListOfCompartments()]

    model_params_dict = {}

    # Get the model parameters
    for p in model.getListOfParameters():
        model_params_dict[p.getId()] = p.getValue()

    # Get list of reversible reaction IDs (make a copy since we'll be modifying the model)
    reversible_reactions = []
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        if is_reversible(model, reaction, log_file):
            print_log(
                log_file, f"Found {reaction.getId()}, rev: {reaction.getReversible()}"
            )
            reversible_reactions.append(reaction)

    print_log(
        log_file, f"Found {len(reversible_reactions)} reversible reactions to split"
    )

    # Split each reversible reaction
    for reaction in reversible_reactions:
        kl = reaction.getKineticLaw()

        if kl is None:
            raise exceptions.InvalidKineticLawError(reaction.getId())

        kl_math = kl.getMath()

        klt, fn = get_kinetic_type(sbml_model=model, kl_math=kl_math, log_file=log_file)

        if fn is None:  # Kinetic described by explicit LMA
            forward_reaction, reverse_reaction = split_reversible_reaction_explicit(
                model, reaction.getId(), model_comps, model_params_dict, log_file
            )

            model.addReaction(forward_reaction)
            model.addReaction(reverse_reaction)

        else:  # Kinetic described by function
            forward_function, reverse_function, forward_reaction, reverse_reaction = (
                split_reversible_reaction_function(
                    model,
                    reaction.getId(),
                    fn,
                    model_comps,
                    model_params_dict,
                    log_file,
                )
            )

            if (
                forward_function
                and reverse_function
                and forward_reaction
                and reverse_reaction
            ):
                model.addFunctionDefinition(forward_function)
                model.addFunctionDefinition(reverse_function)

                model.addReaction(forward_reaction)
                model.addReaction(reverse_reaction)
            else:
                raise exceptions.InvalidKineticLawError(
                    reaction.getId(),
                    "Error during the splitting of the reversible reaction with function-based kinetics",
                )

    return model


def create_sbml_function(
    sbml_model: libsbml.Model,
    function_name: str,
    function_id: str,
    args: list,
    expression: str,
    log_file=None,
) -> libsbml.FunctionDefinition:
    """
    Create an SBML function definition using lambda notation.

    This function creates a new SBML function definition with the specified parameters
    and mathematical expression. The function is defined using SBML's lambda notation:
    lambda(arg1, arg2, ..., expression).

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model to which the function definition will be added
    function_name : str
        Human-readable name for the function
    function_id : str
        Unique identifier for the function in the SBML model
    args : list of str
        List of argument names for the function parameters
    expression : str
        Mathematical expression defining the function body in SBML formula syntax
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    libsbml.FunctionDefinition or None
        The created SBML function definition object, or None if creation failed

    Notes
    -----
    The function constructs a lambda expression in the form:
    lambda(arg1, arg2, ..., argN, expression)

    Returns None if:
    - The formula cannot be parsed
    - Setting the mathematical expression fails
    """

    function = sbml_model.createFunctionDefinition()

    if function_id:
        function.setId(function_id)

    if function_name:
        function.setName(function_name)

    # Creating the args string
    args_string = ",".join(args)

    full_formula = f"lambda({args_string}, {expression})"

    print_log(log_file, f"[CSBMLF] Constructed formula: {full_formula}")

    # Create the math
    math_ast = libsbml.parseL3Formula(full_formula)

    if math_ast is None:
        raise exceptions.InvalidFunctionDefinitionError(
            function_id, "Error during the creation of the AST for the function"
        )

    # Setting the math
    check = function.setMath(math_ast)

    if check != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise exceptions.InvalidFunctionDefinitionError(
            function_id, "Error during the setting of the math for the function"
        )

    return function


def create_sbml_reaction_LMA(
    sbml_model: libsbml.Model,
    reaction_name: str,
    reaction_id: str,
    reactants: list,
    products: list,
    modifiers: list,
    local_parameters: list,
    reaction_comps: list,
    kl_expr: str | None = None,
    function_id: str | None = None,
    function_args: list | None = None,
    log_file=None,
) -> libsbml.Reaction:
    """
    Create an SBML reaction with Law of Mass Action (LMA) kinetics.

    This function creates a new irreversible SBML reaction with either an explicit
    kinetic law expression or a function-based kinetic law. The kinetic law can be
    defined directly as a mathematical expression or via a reference to an SBML
    function definition.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model to which the reaction will be added
    reaction_name : str
        Human-readable name for the reaction
    reaction_id : str
        Unique identifier for the reaction in the SBML model
    reactants : list of tuple
        List of (species_id, stoichiometry) tuples for reactants
    products : list of tuple
        List of (species_id, stoichiometry) tuples for products
    modifiers : list of tuple
        List of (modifier_id, value) tuples for modifiers
    local_parameters : list of tuple
        List of (parameter_id, parameter_value) tuples for local parameters
    reaction_comps : list of str
        List of compartment IDs involved in the reaction
    kl_expr : str, optional
        Explicit kinetic law expression (used when function_id is None), by default None
    function_id : str, optional
        ID of the SBML function to use for kinetics, by default None
    function_args : list of str, optional
        Arguments to pass to the function (used with function_id), by default None
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    libsbml.Reaction
        The created SBML reaction object

    Notes
    -----
    The function supports two modes of kinetic law definition:

    1. Explicit expression mode (when function_id is None):
    Kinetic law = compartments * kl_expr

    2. Function-based mode (when function_id is provided):
    Kinetic law = compartments * function_id(function_args)

    The created reaction is always set as irreversible (reversible = False).
    """

    reaction = sbml_model.createReaction()
    reaction.setId(reaction_id)
    reaction.setName(reaction_name)
    reaction.setReversible(False)

    # Creating the reactants
    for r_id, stoichiometry in reactants:
        reactant = reaction.createReactant()
        reactant.setSpecies(r_id)
        reactant.setStoichiometry(stoichiometry)

    # Creating the products
    for p_id, stoichiometry in products:
        product = reaction.createProduct()
        product.setSpecies(p_id)
        product.setStoichiometry(stoichiometry)

    # Creating the modifiers
    for m_id in modifiers:
        modifier = reaction.createModifier()
        modifier.setSpecies(m_id)

    # Creating the kinetic Law
    kinetic_law = reaction.createKineticLaw()

    # Creating the local parameters
    for lp_id, lp_value in local_parameters:
        __import__("pprint").pprint(lp_id)
        __import__("pprint").pprint(lp_value)
        if sbml_model.getLevel() == 2:
            local_parameter = kinetic_law.createParameter()
        elif sbml_model.getLevel() == 3:
            local_parameter = kinetic_law.createLocalParameter()
        else:
            raise exceptions.ModelError()

        local_parameter.setId(lp_id)
        local_parameter.setValue(lp_value)

    # Construct the string
    if function_id is None or function_args is None:
        print_log(log_file, "[CSBMLR] Creating reaction without function!")

        # Assumes the order comps, parameters, reactants
        comps_str = ("*").join(reaction_comps)

        reaction_kl_formula = f"{comps_str}*({kl_expr})"

        reaction_kl_ast = libsbml.parseL3Formula(reaction_kl_formula)

        if reaction_kl_ast:
            kinetic_law.setMath(reaction_kl_ast)
        else:
            raise exceptions.InvalidKineticLawError(
                reaction_id, "Error during creation of the AST for the kinetic law"
            )

    else:
        # The reaction contains a functions
        comps = reaction_comps  # Assumes in this case is passed just the compartments

        comps_string = "*".join(comps)

        # Creating the function string
        function_args_string = ",".join(function_args)  # Assumed in order
        function_string = f"{function_id}({function_args_string})"

        full_string = f"{comps_string}*{function_string}"

        function_formula = libsbml.parseL3Formula(full_string)

        if function_formula:
            kinetic_law.setMath(function_formula)
        else:
            raise exceptions.InvalidKineticLawError(
                reaction_id, "Error during creation of the AST for the kinetic law with function"
            )

    return reaction


def split_kinetic_function(
    sbml_model: libsbml.Model, kinetic_math: libsbml.ASTNode, log_file=None
) -> tuple:
    """
    Split a kinetic function's mathematical expression into forward and reverse components.

    This function analyzes a lambda-notation kinetic function to separate it into
    forward and reverse reaction expressions. It expects the function to be in the
    form: lambda(args, forward_expr - reverse_expr).

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the kinetic function (currently not used in implementation)
    kinetic_math : libsbml.ASTNode
        The AST node representing the kinetic law mathematical expression in lambda notation
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    tuple of (str, str, list, list) or (None, None, None, None)
        A tuple containing:
        - forward_reaction (str): Mathematical expression for the forward reaction
        - reverse_reaction (str): Mathematical expression for the reverse reaction
        - fwd_args (list): List of argument names used in the forward expression
        - rev_args (list): List of argument names used in the reverse expression
        Returns (None, None, None, None) if the pattern doesn't match

    Notes
    -----
    The function expects kinetic expressions in the pattern: `function_name(arg1, arg2, ..., expr1 - expr2)`
    where the last argument contains the subtraction operation separating forward and reverse kinetics.
    Arguments are classified based on their presence in either the forward or reverse expression.
    """
    pattern = r"^(\w+)\((.*)\)$"

    kl_string = libsbml.formulaToL3String(kinetic_math).replace(" ", "")

    print_log(log_file, f"[SK] formula: {kl_string}")

    match = re.match(pattern, kl_string)  # Cover the pattern lambda(args)

    fwd_args = []
    rev_args = []

    if match:
        args_list = match.group(2).split(",")

        args = args_list[:-1]
        exp = args_list[-1].replace("(", "").replace(")", "")

        forward_reaction, reverse_reaction = exp.split("-")

        for a in args:
            if a in forward_reaction:
                fwd_args.append(a)
            elif a in reverse_reaction:
                rev_args.append(a)
            else:
                continue

        print_log(
            log_file,
            f"fwd formula: {forward_reaction} | rev formula: {reverse_reaction} | fwd params: {fwd_args} | rev params: {rev_args}",
        )

        return forward_reaction, reverse_reaction, fwd_args, rev_args

    else:
        return None, None, None, None


def split_reversible_reaction_explicit(
    sbml_model: libsbml.Model,
    reaction_id: str,
    model_compartments: list,
    model_parameters_dict: dict,
    log_file=None,
) -> tuple:
    """
    Split a reversible reaction with explicit Law of Mass Action kinetics into forward and reverse reactions.

    This function decomposes a reversible reaction into two separate irreversible reactions
    (forward and reverse) by parsing the kinetic law expression. It assumes the kinetic law
    follows the explicit Mass Action pattern: comps*(kforward*[reactants] - krev*[products]).

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the reaction to split
    reaction_id : str
        Unique identifier of the reversible reaction to modify
    model_compartments : list of str
        List of compartment IDs in the model
    model_parameters_dict : dict
        Dictionary mapping parameter IDs to their values
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    tuple of (libsbml.Reaction, libsbml.Reaction)
        A tuple containing:
        - fwd_reaction: The forward irreversible reaction
        - rev_reaction: The reverse irreversible reaction

    Raises
    ------
    SystemExit
        If the kinetic law doesn't match expected Law of Mass Action patterns or
        if reaction creation fails

    Notes
    -----
    The function supports two kinetic law patterns:
    1. `comp*(kf*r1 - kr*p1)` - Compartment multiplied by parenthesized expression
    2. `comp*kf*r1 - kr*p1` - Flattened expression without parentheses

    The original reversible reaction is removed from the model after successful splitting.
    New reactions are named with '_fwd' and '_rev' suffixes.

    Examples
    --------
    For a reaction with kinetic law `cell*(k1*A*B - k2*C)`:
    - Forward reaction: kinetic law = `cell*k1*A*B`
    - Reverse reaction: kinetic law = `cell*k2*C`
    """
    reaction = sbml_model.getReaction(reaction_id)

    kl = reaction.getKineticLaw()

    if kl is None:
        raise exceptions.InvalidKineticLawError(reaction.getId())

    reaction_parameters = [
        (lp.getId(), lp.getValue()) for lp in kl.getListOfLocalParameters()
    ]
    global_parameters = [
        (gp.getId(), gp.getValue()) for gp in sbml_model.getListOfParameters()
    ]

    kl_math = kl.getMath()

    kl_string = libsbml.formulaToL3String(kl_math)

    kl_string_clean = kl_string.replace(" ", "")

    reactants = reaction.getListOfReactants()
    products = reaction.getListOfProducts()
    modifiers = reaction.getListOfModifiers()
    # local_parameters = kl.getListOfLocalParameters()

    if sbml_model.getLevel() == 2:
        local_parameters = kl.getListOfParameters()
    elif sbml_model.getLevel() == 3:
        local_parameters = kl.getListOfLocalParameters()
    else:
        raise exceptions.ModelError

    rs = []
    ps = []
    ms = []
    lps = []  # TODO: Load the local parameters

    for r in reactants:
        r_tuple = (r.getSpecies(), r.getStoichiometry())
        rs.append(r_tuple)

    for p in products:
        p_tuple = (p.getSpecies(), p.getStoichiometry())
        ps.append(p_tuple)

    for m in modifiers:
        m_tuple = (m.getId(), m.getValue())
        ms.append(m_tuple)

    for lp in local_parameters:
        lp_id = lp.getName() if lp.getId() is None else lp.getId()
        lp_tuple = (lp_id, lp.getValue())
        lps.append(lp_tuple)

    pattern1 = r"^(.*)\*\((.*)\)$"  # match: comp*(kf*r1-kr*p1)
    pattern2 = r"^(\w+)\*(.*)$"  # match comp*kf*r1-kr*p1

    match1 = re.match(pattern1, kl_string_clean)
    match2 = re.match(pattern2, kl_string_clean)

    comps = []
    fwd_exp = ""
    rev_exp = ""

    fwd_reaction = None
    rev_reaction = None

    if match1:  # match: comp*(kf*r1-kr*p1)
        comps = match1.group(1).split("*")
        exp = match1.group(2)

        fwd_exp = exp.split("-")[0]
        rev_exp = exp.split("-")[1]

        # Create the new forward reaction
        fwd_reaction = create_sbml_reaction_LMA(
            sbml_model,
            f"{reaction.getId()}_fwd",
            f"{reaction.getId()}_fwd",
            rs,
            ps,
            ms,
            lps,
            comps,
            fwd_exp,
            None,
            None,
            log_file,
        )

        if fwd_reaction is None:
            raise exceptions.ModelError(f"Error during the creation of the forward reaction for {reaction.getId()}")

        # Create the new reverse reaction
        rev_reaction = create_sbml_reaction_LMA(
            sbml_model,
            f"{reaction.getId()}_rev",
            f"{reaction.getId()}_rev",
            ps,
            rs,
            ms,
            lps,
            comps,
            rev_exp,
            None,
            None,
            log_file,
        )

        if rev_reaction is None:
            raise exceptions.ModelError(f"Error during the creation of the reverse reaction for {reaction.getId()}")

    else:
        if match2:
            comps = match2.group(1).split("*")
            exp = match2.group(2)

            fwd_exp = exp.split("-")[0]
            rev_exp = exp.split("-")[1]

            # Check the intersection with the compartments
            fwd_exp_list = fwd_exp.split("*")

            for el in fwd_exp_list:
                if el in model_compartments:
                    comps.append(el)
                    fwd_exp_list.remove(el)

            # Unify the list again
            fwd_exp = ("*").join(fwd_exp_list)

            # Create the forward reaction
            fwd_reaction = create_sbml_reaction_LMA(
                sbml_model,
                reaction.getId(),
                reaction.getId(),
                rs,
                ps,
                ms,
                lps,
                comps,
                fwd_exp,
                None,
                None,
                log_file,
            )

            if fwd_reaction is None:
                raise exceptions.ModelError(f"Error during the creation of the forward reaction for {reaction.getId()}")

            # Creating the reverse reaction
            rev_reaction = create_sbml_reaction_LMA(
                sbml_model,
                reaction.getId(),
                reaction.getId(),
                ps,
                rs,
                ms,
                lps,
                comps,
                rev_exp,
                None,
                None,
                log_file,
            )

            if rev_reaction is None:
                raise exceptions.ModelError(f"Error during the creation of the reverse reaction for {reaction.getId()}")

        else:
            raise exceptions.InvalidKineticLawError(
                reaction.getId(),
                "Kinetic law doesn't match expected explicit Mass Action patterns",
            )

    if fwd_reaction is not None and rev_reaction is not None:
        # Delete the old reaction
        sbml_model.removeReaction(reaction.getId())

    return fwd_reaction, rev_reaction


def split_reversible_reaction_function(
    sbml_model: libsbml.Model,
    reaction_id: str,
    function_name: str,
    model_compartments: list,
    model_parameters_dict: dict,
    log_file=None,
) -> tuple:
    """
    Split a reversible reaction with function-based kinetics into forward and reverse reactions.

    This function decomposes a reversible reaction that uses a function definition for its
    kinetic law into two separate irreversible reactions (forward and reverse). It handles
    both compartment-prefixed and standalone function call patterns.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the reaction and function definitions
    reaction_id : str
        Unique identifier of the reversible reaction to split
    function_name : str
        Name of the function definition used in the reaction's kinetic law
    model_compartments : list of str
        List of compartment IDs in the model
    model_parameters_dict : dict
        Dictionary mapping parameter IDs to their values
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    tuple of (libsbml.FunctionDefinition, libsbml.FunctionDefinition, libsbml.Reaction, libsbml.Reaction) or (None, None, None, None)
        A tuple containing:
        - fwd_function: Forward reaction function definition
        - rev_function: Reverse reaction function definition
        - fwd_reaction: Forward irreversible reaction
        - rev_reaction: Reverse irreversible reaction
        Returns (None, None, None, None) if splitting fails

    Raises
    ------
    SystemExit
        If kinetic split information is unavailable or if reaction/function creation fails

    Notes
    -----
    The function supports two kinetic law patterns:
    1. `comp*function_name(params)` - Compartment multiplied by function call
    2. `function_name(params)` - Standalone function call

    The original reversible reaction and its function definition are removed from the
    model after successful splitting. New functions and reactions are named with '_fwd'
    and '_rev' suffixes.

    The function assumes that:
    - Lambda function body contains subtraction: `forward_expr - reverse_expr`
    - First parameter in actual_params is the forward rate constant
    - Second parameter in actual_params is the reverse rate constant

    Examples
    --------
    For a reaction with kinetic law `cell*Henri_Michaelis_Menten(k1, k2, A, B)`:
    - Forward function: `Henri_Michaelis_Menten_fwd(k1, A)`
    - Reverse function: `Henri_Michaelis_Menten_rev(k2, B)`
    - Forward reaction: kinetic law = `cell*Henri_Michaelis_Menten_fwd(k1, A)`
    - Reverse reaction: kinetic law = `cell*Henri_Michaelis_Menten_rev(k2, B)`
    """

    reaction = sbml_model.getReaction(reaction_id)

    kl = reaction.getKineticLaw()

    if kl is None:
        raise exceptions.InvalidKineticLawError(reaction_id)

    reaction_parameters = [
        (lp.getId(), lp.getValue()) for lp in kl.getListOfLocalParameters()
    ]
    global_parameters = [
        (gp.getId(), gp.getValue()) for gp in sbml_model.getListOfParameters()
    ]

    kl_math = kl.getMath()

    kl_string = libsbml.formulaToL3String(kl_math)

    __import__("pprint").pprint(kl_string)

    kl_string_clean = kl_string.replace(" ", "")

    reactants = reaction.getListOfReactants()
    products = reaction.getListOfProducts()
    modifiers = reaction.getListOfModifiers()

    rs = []
    ps = []
    ms = []
    lps = []

    for r in reactants:
        r_tuple = (r.getSpecies(), r.getStoichiometry())
        rs.append(r_tuple)

    for p in products:
        p_tuple = (p.getSpecies(), p.getStoichiometry())
        ps.append(p_tuple)

    for m in modifiers:
        m_tuple = (m.getId(), m.getValue())
        ms.append(m_tuple)

    pattern = r"^(\w+)\*(\w+)\((.*)\)$"  # Pattern = comp*func(parmas)

    comps_match = re.match(pattern, kl_string_clean)

    if comps_match:
        # Kinetic Law is likely comp*function(params)

        comp = comps_match.group(1)
        actual_function_params = comps_match.group(3).split(",")

        actual_params = []

        # Computing the intersection between parameters
        for afp in actual_function_params:
            for t in reaction_parameters:
                if afp == t[0]:
                    actual_params.append(t)

            for t in global_parameters:
                if afp == t[0]:
                    actual_params.append(t)

        lps = actual_params

        function_kl_math = sbml_model.getFunctionDefinition(function_name).getMath()

        kinetic_split_info = split_kinetic_function(
            sbml_model, function_kl_math, log_file
        )

        if None in kinetic_split_info:
            raise exceptions.InvalidKineticLawError(
                reaction_id,
                "Not available info for the kinetic split for reaction with function-based kinetics",
            )

        else:
            forward_formula, reverse_formula, forward_args, reverse_args = (
                kinetic_split_info
            )

            # Create the forward function
            fwd_function = create_sbml_function(
                sbml_model,
                f"{function_name}_fwd",
                f"{function_name}_fwd",
                forward_args,
                forward_formula,
                log_file=log_file,
            )

            if fwd_function is None:
                raise exceptions.InvalidFunctionDefinitionError(
                    f"{function_name}_fwd",
                    "Error during the creation of the forward function for reaction with function-based kinetics",
                )

            # Create the reverese function
            rev_function = create_sbml_function(
                sbml_model,
                f"{function_name}_rev",
                f"{function_name}_rev",
                reverse_args,
                reverse_formula,
                log_file=log_file,
            )

            if rev_function is None:
                raise exceptions.InvalidFunctionDefinitionError(
                    f"{function_name}_rev",
                    "Error during the creation of the reverse function for reaction with function-based kinetics",
                )

            # Creating the forward reaction

            # Creating the actual parameters string
            # Assumes the first parameter in actual_params is the forward constant

            fwd_actual_params = [actual_params[0][0]] + [
                r.getSpecies() for r in reactants
            ]

            fwd_reaction = create_sbml_reaction_LMA(
                sbml_model,
                f"{reaction.getName()}_fwd",
                f"{reaction.getId()}_fwd",
                rs,
                ps,
                ms,
                lps,
                [comp],
                None,
                f"{function_name}_fwd",
                fwd_actual_params,
                log_file=log_file,
            )

            # Creating the reverse reaction
            # Assumes the second parameter in actual_params is the reverse constant

            rev_actual_params = [actual_params[1][0]] + [
                p.getSpecies() for p in products
            ]

            rev_reaction = create_sbml_reaction_LMA(
                sbml_model,
                f"{reaction.getName()}_rev",
                f"{reaction.getId()}_rev",
                ps,
                rs,
                ms,
                lps,
                [comp],
                None,
                f"{function_name}_rev",
                rev_actual_params,
                log_file=log_file,
            )

            # Remove the old reaction
            sbml_model.removeReaction(reaction_id)

            # Remove the old function
            sbml_model.removeFunctionDefinition(function_name)

            return fwd_function, rev_function, fwd_reaction, rev_reaction

    else:
        # Kinetic Law is just function(params)
        function_kl_math = sbml_model.getFunctionDefinition(function_name).getMath()
        kinetic_split_info = split_kinetic_function(
            sbml_model, function_kl_math, log_file
        )

        if None in kinetic_split_info:
            raise exceptions.InvalidKineticLawError(
                reaction_id,
                "Not available info for the kinetic split for reaction with function-based kinetics",
            )

        else:  # Assumes that the arguments are ordered like comps, fwd_constant, reacts, rev_constant, prods
            reaction_pattern = r"^(\w+)\((.*)\)$"

            match = re.match(reaction_pattern, kl_string_clean)

            if match:
                fn = match.group(1)
                actual_function_params = match.group(2).split(",")

                actual_params = []

                # Computing the intersection between parameters
                for afp in actual_function_params:
                    for t in reaction_parameters:
                        if afp == t[0]:
                            actual_params.append(t)

                    for t in global_parameters:
                        if afp == t[0]:
                            actual_params.append(t)

                lps = actual_params

                if all([el is not None for el in kinetic_split_info]):
                    actual_comps = []

                    for c in model_compartments:
                        if c in actual_function_params:
                            actual_comps.append(c)

                    forward_formula, reverse_formula, forward_args, reverse_args = (
                        kinetic_split_info
                    )

                    # Remove the compartments from the formula and from the args
                    if forward_formula:
                        fwd_formula_array = forward_formula.split("*")
                        for i in range(len(actual_comps)):
                            fwd_formula_array.remove(fwd_formula_array[i])

                        # Rebuild the formula
                        forward_formula = "*".join(fwd_formula_array)

                    if forward_args:
                        for i in range(len(actual_comps)):
                            forward_args.remove(forward_args[i])

                    print_log(
                        log_file,
                        f"fn: {function_name} | fwd formula: {forward_formula} | rev formula: {reverse_formula} | fwd args: {forward_args} | rev args: {reverse_args}",
                    )

                    # Creating the forward function
                    fwd_function = create_sbml_function(
                        sbml_model,
                        f"{function_name}_fwd",
                        f"{function_name}_fwd",
                        forward_args,
                        forward_formula,
                        log_file,
                    )

                    if fwd_function is None:
                        raise exceptions.InvalidFunctionDefinitionError(
                            f"{function_name}_fwd",
                            "Error during the creation of the forward function in reaction with function-based kinetics",
                        )

                    # Creating the reverse reaction
                    rev_function = create_sbml_function(
                        sbml_model,
                        f"{function_name}_rev",
                        f"{function_name}_rev",
                        reverse_args,
                        reverse_formula,
                        log_file,
                    )

                    if rev_function is None:
                        raise exceptions.InvalidFunctionDefinitionError(
                            f"{function_name}_rev",
                            "Error during the creation of the reverse function in reaction with function-based kinetics",
                        )

                    # CREATION OF THE REACTIONS
                    # Creating the forward reaction

                    fwd_actual_params = [actual_params[0][0]] + [
                        r.getSpecies() for r in reactants
                    ]

                    (f"Function: {fn}, actual_params: {fwd_actual_params}")
                    fwd_reaction = create_sbml_reaction_LMA(
                        sbml_model,
                        f"{reaction.getName()}_fwd",
                        f"{reaction.getId()}_fwd",
                        rs,
                        ps,
                        ms,
                        lps,
                        actual_comps,
                        None,
                        f"{function_name}_fwd",
                        fwd_actual_params,
                        log_file=log_file,
                    )

                    if fwd_reaction is None:
                        raise exceptions.InvalidKineticLawError(    
                            reaction_id,
                            "Error during the creation of the forward reaction of reaction with function-based kinetics",
                        )

                    # Creating the reverse reaction

                    rev_actual_params = [actual_params[1][0]] + [
                        p.getSpecies() for p in products
                    ]

                    rev_reaction = create_sbml_reaction_LMA(
                        sbml_model,
                        f"{reaction.getName()}_rev",
                        f"{reaction.getId()}_rev",
                        ps,
                        rs,
                        ms,
                        lps,
                        actual_comps,
                        None,
                        f"{function_name}_rev",
                        rev_actual_params,
                        log_file=log_file,
                    )

                    if rev_reaction is None:
                        raise exceptions.InvalidKineticLawError(
                            reaction_id,
                            "Error during the creation of the reverse reaction of reaction with function-based kinetics",
                        )

                    # Remove the old reaction
                    if fwd_reaction is not None and rev_reaction is not None:
                        sbml_model.removeReaction(reaction_id)
                    # Remove the old function
                    if fwd_function is not None and rev_function is not None:
                        sbml_model.removeFunctionDefinition(function_name)

                    return fwd_function, rev_function, fwd_reaction, rev_reaction
            else:
                raise exceptions.InvalidKineticLawError(
                    reaction_id,
                    "Kinetic law doesn't match expected function call pattern for reaction with function-based kinetics",
                )
        return None, None, None, None


def split_reversible_reaction(
    sbml_model: libsbml.Model,
    reaction_id: str,
    model_compartments: list,
    model_parameters_dict: dict,
    log_file=None,
) -> tuple:
    """
    Split a reversible reaction into two irreversible reactions (forward and reverse).

    This is a legacy function that extracts compartments and parameters from the AST
    to construct forward and reverse reactions. It assumes the kinetic law has the form:
    k_forward * [reactants] - k_reverse * [products].

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the reaction to split
    reaction_id : str
        Unique identifier of the reversible reaction to split
    model_compartments : list of str
        List of compartment IDs in the model
    model_parameters_dict : dict
        Dictionary mapping parameter IDs to their values
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    tuple of (libsbml.Reaction, libsbml.Reaction)
        A tuple containing:
        - forward_reaction: The forward irreversible reaction
        - reverse_reaction: The reverse irreversible reaction

    Raises
    ------
    ValueError
        If the reaction is not found in the model, is already irreversible,
        or does not have a kinetic law defined

    Notes
    -----
    This function is deprecated. Use `split_reversible_reaction_explicit()` or
    `split_reversible_reaction_function()` instead for more robust splitting.

    The function assumes:
    - First parameter in the kinetic law is the forward rate constant
    - Second parameter is the reverse rate constant
    - Reactants and products can be directly swapped for the reverse reaction

    The original reversible reaction is removed from the model after splitting.
    New reactions are named with '_forward' and '_reverse' suffixes.
    """

    # Get the reaction from the model
    reaction = sbml_model.getReaction(reaction_id)

    if reaction is None:
        raise ValueError(f"Reaction with ID '{reaction_id}' not found in the model.")

    print_log(log_file, f"is reversible: {reaction.getReversible()}")

    if not is_reversible(sbml_model, reaction, log_file=log_file):
        raise ValueError(f"Reaction '{reaction_id}' is already irreversible.")

    # Get reactants, products, and modifiers
    reactants = reaction.getListOfReactants()
    products = reaction.getListOfProducts()
    modifiers = reaction.getListOfModifiers()

    # Get kinetic law
    kinetic_law = reaction.getKineticLaw()

    if kinetic_law is None:
        raise exceptions.InvalidKineticLawError(reaction_id)

    parameters = {}

    # Extract the nodes values of AST
    ast_nodes = kinetic_law.getMath().getListOfNodes()
    reaction_comps = []

    # Extract reaction parameters and compartments
    for i in range(ast_nodes.getSize()):
        # Get the AST node
        node = ast_nodes.get(i)

        if node.isName():
            # Check if the node is a parameter
            if node.getName() in model_parameters_dict.keys():
                parameters[node.getName()] = model_parameters_dict[node.getName()]
            else:
                for c in model_compartments:
                    # check if the node is a compartment
                    if c == node.getName():
                        reaction_comps.append(c)

    # Extract parameters from the kinetic law
    for i in range(kinetic_law.getNumParameters()):
        param = kinetic_law.getParameter(i)
        parameters[param.getId()] = param.getValue()

    print_log(log_file, f"{parameters}")

    # Creation of the compartments string
    reaction_comps_string = " * ".join([c for c in reaction_comps])

    # Creation of forward reaction
    forward_reaction = sbml_model.createReaction()
    forward_reaction.setId(f"{reaction_id}_forward")
    forward_reaction.setReversible(False)
    # forward_reaction.setFast(False)

    # Adding the reactants
    for reactant in reactants:
        sr = forward_reaction.createReactant()
        sr.setSpecies(reactant.getSpecies())
        sr.setStoichiometry(reactant.getStoichiometry())
        sr.setConstant(reactant.getConstant())
        # sr.setBoundaryCondition(reactant.getBoundaryCondition())

    # Adding the products
    for product in products:
        sr = forward_reaction.createProduct()
        sr.setSpecies(product.getSpecies())
        sr.setStoichiometry(product.getStoichiometry())
        sr.setConstant(product.getConstant())
        # sr.setBoundaryCondition(product.getBoundaryCondition())

    # Adding the modifiers
    for modifier in modifiers:
        sr = forward_reaction.createModifier()
        sr.setSpecies(modifier.getSpecies())

    # Define forward kinetic law
    kl_forward = forward_reaction.createKineticLaw()
    # Create MathML for forward rate: k_forward * [A] * [B]
    reactant_species = " * ".join([r.getSpecies() for r in reactants])

    k_forward_id = list(parameters.keys())[
        0
    ]  # Assuming the first parameter corresponds to the forward rate

    print_log(
        log_file,
        f"ast string: {reaction_comps_string}, {k_forward_id}, {reactant_species}",
    )

    parts = []
    if reaction_comps_string:
        parts.append(reaction_comps_string)
    if k_forward_id:
        parts.append(k_forward_id)
    if reactant_species:
        parts.append(reactant_species)

    expr = " * ".join(parts)
    math_ast_forward = libsbml.parseL3Formula(expr)

    kl_forward.setMath(math_ast_forward)

    # Add parameter k_forward
    param_k_forward = kl_forward.createParameter()
    param_k_forward.setId(k_forward_id)
    param_k_forward.setValue(parameters[k_forward_id])
    param_k_forward.setConstant(True)

    # Create reverse reaction
    reverse_reaction = sbml_model.createReaction()
    reverse_reaction.setId(f"{reaction_id}_reverse")
    reverse_reaction.setReversible(False)
    # reverse_reaction.setFast(False)

    # Create the reactants for reverse reaction
    for product in products:
        sr = reverse_reaction.createReactant()
        sr.setSpecies(product.getSpecies())
        sr.setStoichiometry(product.getStoichiometry())
        sr.setConstant(product.getConstant())
        # sr.setBoundaryCondition(product.getBoundaryCondition())

    # Create the products for reverse reaction
    for reactant in reactants:
        sr = reverse_reaction.createProduct()
        sr.setSpecies(reactant.getSpecies())
        sr.setStoichiometry(reactant.getStoichiometry())
        sr.setConstant(reactant.getConstant())
        # sr.setBoundaryCondition(reactant.getBoundaryCondition())

    for modifier in modifiers:
        sr = reverse_reaction.createModifier()
        sr.setSpecies(modifier.getSpecies())

    # Define reverse kinetic law
    kl_reverse = reverse_reaction.createKineticLaw()
    # Create MathML for reverse rate: k_reverse * [C]
    product_species = " * ".join([p.getSpecies() for p in products])
    k_reverse_id = list(parameters.keys())[
        1
    ]  # Assuming the second parameter corresponds to the reverse rate
    parts = []
    if reaction_comps_string:
        parts.append(reaction_comps_string)
    if k_reverse_id:
        parts.append(k_reverse_id)
    if product_species:
        parts.append(product_species)

    expr = " * ".join(parts)
    math_ast_reverse = libsbml.parseL3Formula(expr)

    kl_reverse.setMath(math_ast_reverse)

    # Add parameter k_reverse
    param_k_reverse = kl_reverse.createParameter()
    param_k_reverse.setId(k_reverse_id)
    param_k_reverse.setValue(parameters[k_reverse_id])
    param_k_reverse.setConstant(True)

    # Remove the original reversible reaction
    sbml_model.removeReaction(reaction_id)

    return (forward_reaction, reverse_reaction)


# KEEP
def knockout_reaction(
    sbml_model: libsbml.Model, target_reaction_id: str, log_file=None
) -> libsbml.Model:
    """
    Knockout a reaction by setting its kinetic law to zero.

    This function performs a knockout operation on a reaction by setting its kinetic law
    mathematical expression to 0, effectively disabling the reaction without removing it
    from the model structure.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the reaction to knockout
    target_reaction_id : str
        Unique identifier of the reaction to knockout
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    libsbml.Model
        The modified SBML model with the reaction knocked out

    Notes
    -----
    The function sets the kinetic law to 0 rather than removing the reaction entirely,
    preserving the model structure. If the reaction is not found, a warning is logged
    but no error is raised.

    If the reaction has no kinetic law defined, an error is logged and the program exits.
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
                print_log(
                    log_file, f"Successfully knocked out reaction {target_reaction_id}"
                )
            else:
                print_log(log_file, f"Error setting kinetic law: {result}")
        else:
            raise exceptions.InvalidKineticLawError(reaction.getId())
    else:
        print_log(
            log_file,
            f"[WARNING] Reaction not found in the model, no modification applied.",
        )

    return sbml_model


def knockin_reaction(
    sbml_model: libsbml.Model,
    target_reaction: libsbml.Reaction,
    new_vals: list,
    log_file=None,
) -> libsbml.Model:
    """
    Perform a knock-in operation on a reaction by replacing its reactants with constant species.

    This function creates new constant species with specified values for each reactant in the
    target reaction, then modifies the reaction to use these new constant species. The original
    reactants are replaced with new species having '_KI' suffix.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the target reaction
    target_reaction : libsbml.Reaction
        The reaction object to perform knock-in on
    new_vals : list of float
        List of new initial concentration/amount values for each reactant.
        Order must match the order of reactants in the reaction
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    libsbml.Model
        The modified SBML model with the knocked-in reaction

    Raises
    ------
    SystemExit
        If the reaction has no kinetic law or if species creation/modification fails

    Notes
    -----
    The function performs the following operations:
    1. Creates new constant species for each reactant with '_KI' suffix
    2. Sets new species to specified values (amounts or concentrations)
    3. Makes new species constant (boundary condition = True, constant = True)
    4. Clones the original reaction with '_KI' suffix
    5. Replaces reactants in the cloned reaction with the new constant species
    6. Updates the kinetic law to reference the new species
    7. Removes the original reaction and adds the modified one

    The function automatically detects whether species use substance units (amounts)
    or concentration values and applies the new values accordingly.

    The kinetic law is updated by token replacement, supporting both explicit
    kinetic laws and function-based kinetic laws.
    """
    # Retrieve the reactants, products, modifiers and local parameters
    rs = []
    ps = []
    ms = []
    lps = []

    for r in target_reaction.getListOfReactants():
        r_id = r.getSpecies()
        r_s = r.getStoichiometry()

        r_tuple = (r_id, r_s)
        rs.append(r_tuple)

    # Create the new species
    new_species_ids = []

    for i, t in enumerate(rs):
        r, _ = t
        original_species = sbml_model.getSpecies(r)  # Original species

        # Create unique ID for new species
        new_species_id = r + "_KI"
        new_species = sbml_model.createSpecies()
        new_species.setId(new_species_id)

        # Copy compartment (required for valid SBML)
        new_species.setCompartment(original_species.getCompartment())

        # Check if has Amounts or Concentrations
        if (
            original_species.isSetHasOnlySubstanceUnits()
            or original_species.isSetInitialAmount()
        ):
            # The species has Amounts
            new_species.setHasOnlySubstanceUnits(
                original_species.getHasOnlySubstanceUnits()
            )
            new_species.setInitialAmount(new_vals[i])
        elif original_species.isSetInitialConcentration():
            # The species has concentration
            new_species.setInitialConcentration(new_vals[i])
        else:
            raise exceptions.ModelError(
                f"Original species {r} must have either initial amount or initial concentration set."
            )

        # Make the new species constant
        new_species.setBoundaryCondition(True)
        new_species.setConstant(True)

        new_species_ids.append(new_species_id)

    # Creating the new reaction

    # Get the kinetic type
    kl = target_reaction.getKineticLaw()

    if kl is None:
        exceptions.InvalidKineticLawError(target_reaction.getId())

    kl_math = kl.getMath()
    kl_string = libsbml.formulaToL3String(kl_math).replace(" ", "")
    kin_type, fn = get_kinetic_type(sbml_model, kl_math, log_file)

    new_kl = ""
    tokens = []

    if fn is None:  # KL described by explicit Law
        if kin_type == 1 or kin_type == 2:  # KL is described using MM or LMA
            tokens = re.findall(r"\w+|[^\w\s]", kl_string)
        else:
            raise exceptions.InvalidKineticLawError(
                target_reaction.getId(),
                "Kinetic law type not supported for knock-in operation. " \
                "Only explicit kinetic laws (Mass Action or Michaelis-Menten) are supported.",
            )
    else:  # KL described by function
        fun_pattern = r"\d+(?:\.\d+)?|\w+|[^\w\s]"

        tokens = re.findall(fun_pattern, kl_string)

    for i, t in enumerate(rs):
        r_id, _ = t
        if r_id in tokens:
            # Get the index
            r_idx = tokens.index(r_id)

            # Remove the old reactant
            del tokens[r_idx]

            # Add the new reactant
            tokens.insert(r_idx, new_species_ids[i])

            # Build the new kinetic Law
            new_kl = ("").join(tokens)

    reaction_clone = target_reaction.clone()

    # Changing the name
    if reaction_clone.isSetId():
        reaction_clone.setId(f"{target_reaction.getId()}_KI")

    # Removing the original reactants
    for i in range(target_reaction.getNumReactants()):
        reaction_clone.removeReactant(0)

    # Adding the new reactants
    for i, t in enumerate(rs):
        r_id, r_stoc = t
        new_r = reaction_clone.createReactant()

        new_r.setId(new_species_ids[i])
        new_r.setStoichiometry(r_stoc)
        new_r.setConstant(True)

        reaction_clone.addReactant(new_r)

    # Change the kinetic law
    new_kl_math = libsbml.parseL3Formula(new_kl)

    reaction_clone.getKineticLaw().setMath(new_kl_math)

    # Remove the old reaction
    sbml_model.removeReaction(target_reaction.getId())

    # Add the new reaction
    sbml_model.addReaction(reaction_clone)

    return sbml_model


# ============
# FUNCTIONS
# ============
def get_functions_list(SBML_model: libsbml.Model) -> list:
    """
    Extract function definitions from an SBML model as Function objects.

    This function retrieves all function definitions from the SBML model and converts
    them into Function objects using the Function class's from_sbml() method.

    Parameters
    ----------
    SBML_model : libsbml.Model
        The SBML model containing function definitions to extract

    Returns
    -------
    list of Function
        List of Function objects created from SBML function definitions.
        Returns empty list if no function definitions exist in the model

    Notes
    -----
    Function definitions in SBML are typically lambda expressions that define
    reusable mathematical functions for kinetic laws or other model components.

    See Also
    --------
    create_sbml_function : Create a new SBML function definition
    split_kinetic_function : Split kinetic function into forward/reverse components
    """
    functions_list = []

    for sbml_func in SBML_model.getListOfFunctionDefinitions():
        function = Function.from_sbml(sbml_func)
        functions_list.append(function)

    return functions_list


# ============
# FUNCTIONS
# ============


def save_sbml_model(model: libsbml.Model, file_path: str, log_file=None) -> bool:
    """
    Save an SBML model to a file in XML format.

    This function accepts an SBML model in various formats (Model, XML string, or
    SBMLDocument) and writes it to the specified file path. It handles automatic
    conversion to SBMLDocument when necessary.

    Parameters
    ----------
    model : libsbml.Model, str, or libsbml.SBMLDocument
        The SBML model to save. Can be:
        - libsbml.Model: A model object (will be wrapped in SBMLDocument if needed)
        - str: An XML string representation of the model
        - libsbml.SBMLDocument: A complete SBML document
    file_path : str
        The absolute or relative path where the SBML file will be saved
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    bool
        True if the file was successfully written, False otherwise

    Notes
    -----
    If the model is a libsbml.Model without an associated SBMLDocument, a new
    SBMLDocument is created using the model's SBML level and version.

    Success and error messages are logged to log_file if provided.

    Examples
    --------
    Save a model object:
    >>> success = save_sbml_model(my_model, "output/model.xml", log_file)

    Save from XML string:
    >>> xml_str = "<sbml>...</sbml>"
    >>> success = save_sbml_model(xml_str, "output/model.xml")
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
        raise IOError(f"Failed to save SBML model to {file_path}. Check log for details.")

    return success


def get_sbml_as_xml(model: str, log_file=None) -> str:
    """
    Convert an SBML model to its XML string representation.

    This function accepts an SBML model in various formats and converts it to an XML
    string. It handles automatic conversion to SBMLDocument when necessary before
    serialization.

    Parameters
    ----------
    model : libsbml.Model, str, or libsbml.SBMLDocument
        The SBML model to convert. Can be:
        - libsbml.Model: A model object (will be wrapped in SBMLDocument if needed)
        - str: An XML string (returned as-is)
        - libsbml.SBMLDocument: A complete SBML document
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    str or None
        The XML string representation of the SBML model, or None if conversion fails

    Notes
    -----
    If the model is a libsbml.Model without an associated SBMLDocument, a new
    SBMLDocument is created using the model's SBML level and version.

    If the input is already a string, it is returned unchanged without validation.

    Error messages are logged to log_file if provided.

    See Also
    --------
    save_sbml_model : Save an SBML model to a file

    Examples
    --------
    Convert a model to XML string:
    >>> xml_string = get_sbml_as_xml(my_model)
    >>> print(xml_string[:100])  # Print first 100 characters

    Pass through an existing XML string:
    >>> xml_str = "<sbml>...</sbml>"
    >>> result = get_sbml_as_xml(xml_str)
    >>> assert result == xml_str
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
        raise ValueError("Failed to convert SBML model to XML string. Check log for details.")


# KEEP
def generate_species_random_combinations(
    sbml_model: libsbml.Model,
    target_species: list = [],
    n_samples: int = 5,
    variation: int = 20,
    log_file=None,
) -> list:
    """
    Generate random concentration/amount samples for specified species with percentage variation.

    This function creates random samples for each target species by varying their initial
    values within a specified percentage range. The samples are uniformly distributed
    around each species' initial concentration or amount.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the species to sample
    target_species : list of str, optional
        List of species IDs for which to generate samples, by default []
    n_samples : int, optional
        Number of random samples to generate for each species, by default 5
    variation : int, optional
        Percentage variation range around the initial value (e.g., 20 means ±20%),
        by default 20
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    list of list of float
        2D list where each sublist contains n_samples random values for one species.
        Order matches the order of target_species

    Raises
    ------
    SystemExit
        If a species is not found in the model or if initial values cannot be accessed

    Notes
    -----
    For each species, samples are generated as:
    - Normal case: uniform distribution in [initial * (1 - variation/100), initial * (1 + variation/100)]
    - Zero initial value: uniform distribution in [0, 1e-10] as a fallback

    The function automatically detects whether species use substance units (amounts)
    or concentration values.

    Random number generation uses numpy's default_rng for reproducibility.

    See Also
    --------
    get_fixed_combinations : Generate fixed variation combinations
    create_combinations : Generate all combinations from input samples

    Examples
    --------
    Generate 10 samples with ±30% variation for two species:
    >>> samples = generate_species_random_combinations(
    ...     model, ['A', 'B'], n_samples=10, variation=30, log_file=log
    ... )
    >>> len(samples)  # Number of species
    2
    >>> len(samples[0])  # Number of samples per species
    10
    """
    res = []

    for ts in target_species:
        species = sbml_model.getListOfSpecies().getElementBySId(ts)

        if species is None:
            raise exceptions.ModelError(f"Species {ts} not found in the model.")

        if species.getHasOnlySubstanceUnits() or species.isSetInitialAmount():
            print_log(log_file, f"Using amounts for {ts}")
            t0_val = species.getInitialAmount()
        elif species.isSetInitialConcentration():
            print_log(log_file, f"Using concentration for {ts}")
            t0_val = species.getInitialConcentration()
        else:
            raise exceptions.ModelError(
                f"Species {ts} must have either initial amount or initial concentration set."
            )
        tmp = []

        for i in range(n_samples):
            if t0_val == 0:
                print_log(
                    log_file,
                    f"[WARNING] initial value is 0 for species {ts}, using a small value",
                )
                # Use a small absolute range instead of percentage
                lower_bound = 0
                upper_bound = 1e-10
            else:
                lower_bound = t0_val * (1 - (variation / 100))
                upper_bound = t0_val * (1 + (variation / 100))

            rng = np.random.default_rng()

            sample = rng.uniform(lower_bound, np.nextafter(upper_bound, np.inf))
            tmp.append(sample)

        res.append(tmp)

    # print_log(log_file, f"[GENERATE SAMPLES]{res}")

    return res


# KEEP
def create_combinations(input_samples: list, log_file=None) -> Generator:
    """
    Generate all possible combinations from a 2D list of sample values using Cartesian product.

    This generator function creates all possible combinations by computing the Cartesian
    product of the input lists. It yields combinations one at a time, making it memory-efficient
    for large datasets.

    Parameters
    ----------
    input_samples : list of list
        2D list where each sublist contains sample values for one variable.
        For example: [[1, 2], [3, 4], [5, 6]] represents 2 values for var1,
        2 values for var2, and 2 values for var3
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Yields
    ------
    tuple
        Each combination as a tuple, where the i-th element comes from input_samples[i].
        Total number of combinations = product of lengths of all sublists

    Notes
    -----
    This function uses itertools.product for efficient Cartesian product computation.
    Memory usage is minimal as combinations are generated lazily.

    The order of elements in each yielded tuple corresponds to the order of sublists
    in input_samples.

    See Also
    --------
    generate_species_random_combinations : Generate random samples for species
    get_fixed_combinations : Generate fixed variation combinations

    Examples
    --------
    Generate all combinations from 2D samples:
    >>> samples = [[1, 2], [3, 4], [5, 6]]
    >>> combinations = list(create_combinations(samples))
    >>> len(combinations)
    8
    >>> combinations[0]
    (1, 3, 5)
    >>> combinations[-1]
    (2, 4, 6)

    Use as a generator for memory efficiency:
    >>> for combo in create_combinations([[1, 2], [3, 4]]):
    ...     print(combo)
    (1, 3)
    (1, 4)
    (2, 3)
    (2, 4)
    """
    # input_samples is a 2D array: e.g., [[1, 2], [3, 4], [5, 6]]
    for comb in itertools.product(*input_samples):
        yield comb


# KEEP
def get_fixed_combinations(
    sbml_model: libsbml.Model,
    input_species: list[str],
    fixed_variations: list[float] | None,
    log_file=None,
) -> list:
    """
    Generate fixed-percentage variation samples for specified species.

    This function creates samples for each target species by applying a set of fixed
    percentage variations to their initial values. Unlike random sampling, this produces
    deterministic samples based on the specified variation percentages.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the species to sample
    input_species : list of str
        List of species IDs for which to generate variation samples
    fixed_variations : list of float
        List of percentage variations to apply (e.g., [-20, 0, 20] for -20%, 0%, +20%)
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    list of list of float
        2D list where each sublist contains samples for one species.
        Each sublist has len(fixed_variations) values, sorted by variation percentage.
        Order matches the order of input_species

    Raises
    ------
    Exception
        If initial concentration/amount values cannot be accessed for any species

    Notes
    -----
    For each species, samples are generated as:
    - Normal case: initial_value * (1 + variation/100) for each variation percentage
    - Zero initial value: abs(variation) * 1e-10 as a fallback

    The function automatically detects whether species use substance units (amounts)
    or concentration values.

    Samples are sorted by variation percentage in ascending order.

    See Also
    --------
    generate_species_random_combinations : Generate random samples for species
    create_combinations : Generate Cartesian product of samples

    Examples
    --------
    Generate samples with fixed variations for two species:
    >>> samples = get_fixed_combinations(
    ...     model, ['A', 'B'], [-50, 0, 50], log_file=log
    ... )
    >>> len(samples)  # Number of species
    2
    >>> len(samples[0])  # Number of samples per species (3 variations)
    3

    For species 'A' with initial value 100:
    >>> # fixed_variations = [-50, 0, 50]
    >>> # samples[0] = [50.0, 100.0, 150.0]
    """
    samples = []

    for s_id in input_species:
        species = sbml_model.getListOfSpecies().getElementBySId(s_id)

        # Get initial concentration/amount using the same logic as generate_species_samples
        if species.getHasOnlySubstanceUnits() or species.isSetInitialAmount():
            print_log(log_file, f"Using amounts for {s_id}")
            t0_conc = species.getInitialAmount()
        elif species.isSetInitialConcentration():
            print_log(log_file, f"Using concentration for {s_id}")
            t0_conc = species.getInitialConcentration()
        else:
            raise exceptions.ModelError(
                f"Species {s_id} must have either initial amount or initial concentration set."
            )

        tmp = []

        # Handle zero initial concentration case (similar to generate_species_samples)
        if t0_conc == 0:
            print_log(
                log_file,
                f"[WARNING] initial value is 0 for species {s_id}, using small absolute values",
            )
            # For zero concentrations, use small absolute values based on variations
            for v in sorted(fixed_variations):
                # Use small absolute range instead of percentage
                sample = (
                    abs(v) * 1e-10
                )  # Small absolute value proportional to variation
                tmp.append(sample)
        else:
            # Apply percentage variations (same as before but with better logging)
            for v in sorted(fixed_variations):
                sample = t0_conc + (t0_conc * v / 100)
                # print_log(
                #     log_file,
                #     f"[FIXED_COMBINATIONS] Species: {s_id} | t0_value: {t0_conc} | variation: {v}% | sample: {sample}",
                # )
                tmp.append(sample)

        samples.append(tmp)

    return samples


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


def get_selections(
    sbml_model: libsbml.Model, rr_model, target_ids: list, log_file=None
) -> list:
    """
    Get selections for target species/reactions in the SBML model.

    This function retrieves the current selections from the roadrunner model and
    adds target species or reactions to the selection list based on their presence
    in the SBML model.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing species and reactions
    rr_model : roadrunner.RoadRunner
        The roadrunner model for accessing current selections
    target_ids : list of str
        List of target species or reaction IDs to check and add to selections
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    list of str
        Updated list of selections including target species/reactions
    Notes
    -----
    The function checks if each target ID is a reaction or species in the SBML model.
    If it's a reaction, it is added directly. If it's a species, it is added in the
    format "[species_id]" if not already present in the selections.
    """

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
