import libsbml
import re

from src import exceptions
from src.utils.utils import print_log

from src.utils.sbml.helpers import get_nodes_iterator, get_list_of_reactions




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
        raise exceptions.InvalidKineticLawError(
            reaction.getId(), "No kinetic law found"
        )

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
                reaction_id,
                "Error during creation of the AST for the kinetic law with function",
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
            raise exceptions.ModelError(
                f"Error during the creation of the forward reaction for {reaction.getId()}"
            )

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
            raise exceptions.ModelError(
                f"Error during the creation of the reverse reaction for {reaction.getId()}"
            )

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
                raise exceptions.ModelError(
                    f"Error during the creation of the forward reaction for {reaction.getId()}"
                )

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
                raise exceptions.ModelError(
                    f"Error during the creation of the reverse reaction for {reaction.getId()}"
                )

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
