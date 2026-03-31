import libsbml
import re

from shapcrn.utils.utils import print_log
from shapcrn.utils.sbml.reactions import get_kinetic_type
from shapcrn import exceptions
from shapcrn.utils import species as species_ut


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
    for species in species_ut.get_list_of_species(sbml_model):
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

    print_log(log_file, f"Knocking in species {species_id} with new value {new_val}")

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

        #species.setId(species_id + "_KI")

        # Make the species constant
        species.setBoundaryCondition(True)
        species.setConstant(True)

    else:
        raise exceptions.InvalidSpeciesError(
            species_id, sbml_model.getId(), "Species not presente in the model"
        )

    return sbml_model


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
