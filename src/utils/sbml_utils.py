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


def create_ki_model(
    target_id: str, new_val: float, sbml_model, sbml_str: str, log_file=None
) -> dict:
    """
    Create the knock-in model for the specified target ID.

    Parameters
    ----------
    target_id : str
        Target ID to knock-in.
    new_val : float
        New value to set for the knocked-in species.
    sbml_model : libsbml.Model
        The original SBML model.
    sbml_str : str
        The SBML model as a string.
    log_file : file, optional
        File to log information, by default None.

    Returns
    -------
    dict
        A dictionary with the target ID as key and its corresponding knock-in SBML model as value.
    """
    doc_copy = libsbml.readSBMLFromString(sbml_str)  # rebuild in memory
    model_copy = doc_copy.getModel()

    modified_model = knockin_species(model_copy, target_id, new_val, log_file)

    doc_copy.setModel(modified_model)

    return {target_id: libsbml.writeSBMLToString(doc_copy)}


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


# KEEP


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
                "Kinetic law type not supported for knock-in operation. "
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
        raise ValueError(
            "Failed to convert SBML model to XML string. Check log for details."
        )


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
