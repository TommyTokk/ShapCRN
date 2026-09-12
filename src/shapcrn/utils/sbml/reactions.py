"""SBML reaction queries and semantics-preserving mass-action decomposition.

Splitting uses kinetic inference (not the SBML reversible flag). Expressions
containing division retain the historical Michaelis–Menten exclusion. See SBML
L3V1 §§3.3, 4.3 and 4.11 for namespaces, functions and reaction semantics.
"""

import libsbml

from shapcrn import exceptions
from shapcrn.utils.sbml.helpers import get_nodes_iterator
from shapcrn.utils.sbml.validation import (
    check,
    fresh_metaids,
    reject_dependencies,
    require_fixed_references,
    require_free_id,
    transact,
)


def get_list_of_reactions(sbml_model: libsbml.Model) -> list:
    return sbml_model.getListOfReactions()


def get_list_of_reactions_ids(sbml_model: libsbml.Model) -> list:
    return [r.getId() for r in sbml_model.getListOfReactions()]


def _error(identifier, reason):
    return exceptions.InvalidKineticLawError(
        identifier, f"Reaction/function {identifier}: {reason}"
    )


def _replace_names(node, mapping):
    """Simultaneous AST substitution; actual arguments are never substituted twice."""
    if node.isName() and node.getName() in mapping:
        return mapping[node.getName()].deepCopy()
    result = node.deepCopy()
    for i in range(node.getNumChildren()):
        check(
            result.replaceChild(i, _replace_names(node.getChild(i), mapping)),
            "substitute function argument",
        )
    return result


def _expand(model, node, stack=()):
    if node is None:
        raise _error("unknown", "Missing kinetic math")
    if node.getType() == libsbml.AST_FUNCTION:
        identifier = node.getName()
        function = model.getFunctionDefinition(identifier)
        if function is None or function.getBody() is None or identifier in stack:
            raise _error(identifier, "Undefined or recursive function")
        if function.getNumArguments() != node.getNumChildren():
            raise _error(identifier, "Function argument count mismatch")
        arguments = {
            function.getArgument(i).getName(): _expand(model, node.getChild(i), stack)
            for i in range(node.getNumChildren())
        }
        body = _replace_names(function.getBody(), arguments)
        return _expand(model, body, stack + (identifier,))
    result = node.deepCopy()
    for i in range(node.getNumChildren()):
        check(
            result.replaceChild(i, _expand(model, node.getChild(i), stack)),
            "expand math",
        )
    return result


def get_kinetic_type(
    sbml_model: libsbml.Model, kl_math: libsbml.ASTNode, log_file=None
) -> tuple:
    expanded = _expand(sbml_model, kl_math)
    fn = next(
        (
            n.getName()
            for n in get_nodes_iterator(kl_math)
            if n.getType() == libsbml.AST_FUNCTION
        ),
        None,
    )
    return (
        2
        if any(n.getType() == libsbml.AST_DIVIDE for n in get_nodes_iterator(expanded))
        else 1,
        fn,
    )


def is_reversible(sbml_model: libsbml.Model, reaction: libsbml.Reaction, log_file=None):
    """Infer a split candidate from kinetics, retaining the historical MM exclusion.

    This is a project heuristic, not the declared SBML reversibility. Missing
    kinetics are unsupported for preparation even though SBML permits them.
    """
    law = reaction.getKineticLaw()
    if law is None or law.getMath() is None:
        raise _error(reaction.getId(), "Missing kinetic law")
    try:
        expanded = _expand(sbml_model, law.getMath())
    except exceptions.InvalidKineticLawError as exc:
        raise _error(reaction.getId(), str(exc)) from exc
    types = [n.getType() for n in get_nodes_iterator(expanded)]
    return libsbml.AST_DIVIDE not in types and libsbml.AST_MINUS in types


def _has_minus(model, node):
    return any(
        n.getType() == libsbml.AST_MINUS
        for n in get_nodes_iterator(_expand(model, node))
    )


def _decompose(model, node, pairs=None):
    """Split only a binary difference or a product with one separable factor."""
    kind = node.getType()
    if kind == libsbml.AST_MINUS and node.getNumChildren() == 2:
        return node.getChild(0).deepCopy(), node.getChild(1).deepCopy()
    if kind == libsbml.AST_TIMES:
        candidates = [
            i
            for i in range(node.getNumChildren())
            if _has_minus(model, node.getChild(i))
        ]
        if len(candidates) == 1:
            index = candidates[0]
            forward, reverse = _decompose(model, node.getChild(index), pairs)
            outputs = []
            for branch in (forward, reverse):
                result = node.deepCopy()
                check(result.replaceChild(index, branch), "split product")
                outputs.append(result)
            return tuple(outputs)
    if kind == libsbml.AST_FUNCTION:
        if pairs is None:
            return _decompose(model, _expand(model, node))
        identifier = node.getName()
        if identifier not in pairs:
            original = model.getFunctionDefinition(identifier)
            forward, reverse = _decompose(model, _expand(model, original.getBody()))
            definitions = []
            for suffix, body in zip(("_fwd", "_rev"), (forward, reverse)):
                new_id = identifier + suffix
                require_free_id(model, new_id)
                definition = original.clone()
                check(definition.setId(new_id), new_id)
                fresh_metaids(model, definition, suffix)
                math = original.getMath().deepCopy()
                check(math.replaceChild(math.getNumChildren() - 1, body), new_id)
                check(definition.setMath(math), new_id)
                check(model.addFunctionDefinition(definition), new_id)
                definitions.append(new_id)
            pairs[identifier] = definitions
        outputs = []
        for identifier in pairs[identifier]:
            call = node.deepCopy()
            check(call.setName(identifier), identifier)
            outputs.append(call)
        return tuple(outputs)
    raise _error(
        "kinetics",
        "Expected a binary forward-minus-reverse expression, optionally multiplied by common factors",
    )


def _split(model, identifier, suffixes=("_fwd", "_rev"), functions=False, pairs=None):
    reaction = model.getReaction(identifier)
    if reaction is None:
        raise exceptions.InvalidReactionError(identifier, model.getId())
    if reaction.getFast():
        raise _error(identifier, "fast=true splitting is unsupported")
    require_fixed_references(model, reaction)
    reject_dependencies(model, [identifier], identifier)
    law = reaction.getKineticLaw()
    if law is None or law.getMath() is None:
        raise _error(identifier, "Missing kinetic law")
    pairs = {} if pairs is None else pairs
    try:
        forward, reverse = _decompose(
            model, law.getMath(), pairs if functions else None
        )
    except exceptions.InvalidKineticLawError as exc:
        raise _error(identifier, str(exc)) from exc
    ids = []
    for suffix, math, backwards in zip(suffixes, (forward, reverse), (False, True)):
        new_id = identifier + suffix
        require_free_id(model, new_id)
        result = reaction.clone()
        check(result.setId(new_id), new_id)
        check(result.setName((reaction.getName() or identifier) + suffix), new_id)
        check(result.setReversible(False), new_id)
        if model.getLevel() == 2 or model.getVersion() == 1:
            check(result.setFast(False), new_id)
        if backwards:
            reactants = [s.clone() for s in result.getListOfReactants()]
            products = [s.clone() for s in result.getListOfProducts()]
            result.getListOfReactants().clear()
            result.getListOfProducts().clear()
            for ref in products:
                check(result.addReactant(ref), new_id)
            for ref in reactants:
                check(result.addProduct(ref), new_id)
        for ref in (
            list(result.getListOfReactants())
            + list(result.getListOfProducts())
            + list(result.getListOfModifiers())
        ):
            if ref.isSetId():
                ref_id = ref.getId() + suffix
                require_free_id(model, ref_id)
                check(ref.setId(ref_id), ref_id)
        fresh_metaids(model, result, suffix)
        check(result.getKineticLaw().setMath(math), new_id)
        check(model.addReaction(result), new_id)
        ids.append(new_id)
    model.removeReaction(identifier)
    return tuple(ids)


def split_all_reversible_reactions(
    model: libsbml.Model, log_file=None
) -> libsbml.Model:
    """Atomically split all inferred mass-action candidates into _fwd/_rev pairs."""

    def operation(staged):
        candidates = [
            r.getId()
            for r in staged.getListOfReactions()
            if is_reversible(staged, r, log_file)
        ]
        pairs = {}
        for identifier in candidates:
            _split(staged, identifier, functions=True, pairs=pairs)

    transact(model, operation, log_file)
    return model


def split_reversible_reaction_explicit(
    sbml_model, reaction_id, model_compartments, model_parameters_dict, log_file=None
):
    """Split explicit kinetics atomically; return the two attached reactions."""
    ids = transact(sbml_model, lambda m: _split(m, reaction_id), log_file)
    return tuple(sbml_model.getReaction(identifier) for identifier in ids)


def split_reversible_reaction_function(
    sbml_model,
    reaction_id,
    function_name,
    model_compartments,
    model_parameters_dict,
    log_file=None,
):
    """Split function kinetics, keeping lambda arguments and original definitions.

    Returns (forward_function, reverse_function, forward_reaction, reverse_reaction).
    """

    def operation(model):
        pairs = {}
        ids = _split(model, reaction_id, functions=True, pairs=pairs)
        if function_name not in pairs:
            raise _error(
                reaction_id,
                f"Function {function_name} is not the separable kinetic factor",
            )
        return (*pairs[function_name], *ids)

    ids = transact(sbml_model, operation, log_file)
    return (
        sbml_model.getFunctionDefinition(ids[0]),
        sbml_model.getFunctionDefinition(ids[1]),
        sbml_model.getReaction(ids[2]),
        sbml_model.getReaction(ids[3]),
    )


def split_reversible_reaction(
    sbml_model, reaction_id, model_compartments, model_parameters_dict, log_file=None
):
    """Legacy entry point, preserving _forward/_reverse suffixes."""
    ids = transact(
        sbml_model, lambda m: _split(m, reaction_id, ("_forward", "_reverse")), log_file
    )
    return tuple(sbml_model.getReaction(identifier) for identifier in ids)


def split_kinetic_function(sbml_model, kinetic_math, log_file=None):
    """Return branch formulas and ordered argument names, including shared names."""
    if kinetic_math is None or kinetic_math.getType() != libsbml.AST_LAMBDA:
        return None, None, None, None
    try:
        fwd, rev = _decompose(
            sbml_model, kinetic_math.getChild(kinetic_math.getNumChildren() - 1)
        )
    except exceptions.InvalidKineticLawError:
        return None, None, None, None
    args = [
        kinetic_math.getChild(i).getName()
        for i in range(kinetic_math.getNumChildren() - 1)
    ]
    names = [
        {n.getName() for n in get_nodes_iterator(branch) if n.isName()}
        for branch in (fwd, rev)
    ]
    return (
        libsbml.formulaToL3String(fwd),
        libsbml.formulaToL3String(rev),
        [a for a in args if a in names[0]],
        [a for a in args if a in names[1]],
    )


def create_sbml_function(
    sbml_model, function_name, function_id, args, expression, log_file=None
):
    """Create and attach a checked lambda function; failure leaves the model intact."""

    def operation(model):
        require_free_id(model, function_id)
        function = libsbml.FunctionDefinition(model.getSBMLNamespaces())
        check(function.setId(function_id), function_id)
        if function_name:
            check(function.setName(function_name), function_id)
        formula = "lambda(" + ",".join([*args, expression]) + ")"
        ast = libsbml.parseL3Formula(formula)
        if ast is None:
            raise exceptions.InvalidFunctionDefinitionError(
                function_id, f"Cannot parse {formula}"
            )
        check(function.setMath(ast), function_id)
        check(model.addFunctionDefinition(function), function_id)

    transact(sbml_model, operation, log_file)
    return sbml_model.getFunctionDefinition(function_id)


def create_sbml_reaction_LMA(
    sbml_model,
    reaction_name,
    reaction_id,
    reactants,
    products,
    modifiers,
    local_parameters,
    reaction_comps,
    kl_expr=None,
    function_id=None,
    function_args=None,
    log_file=None,
):
    """Create an irreversible reaction atomically.

    Reactants/products are (species_id, stoichiometry, constant) triples;
    modifiers are species IDs; local parameters are (id, value) pairs. The
    explicit expression/function call is multiplied by the supplied factors.
    """

    def operation(model):
        require_free_id(model, reaction_id)
        reaction = libsbml.Reaction(model.getSBMLNamespaces())
        check(reaction.setId(reaction_id), reaction_id)
        if reaction_name:
            check(reaction.setName(reaction_name), reaction_id)
        check(reaction.setReversible(False), reaction_id)
        if model.getLevel() == 2 or model.getVersion() == 1:
            check(reaction.setFast(False), reaction_id)
        for refs, create in (
            (reactants, reaction.createReactant),
            (products, reaction.createProduct),
        ):
            for identifier, stoichiometry, constant in refs:
                ref = create()
                check(ref.setSpecies(identifier), identifier)
                check(ref.setStoichiometry(stoichiometry), identifier)
                if model.getLevel() == 3:
                    check(ref.setConstant(constant), identifier)
        for identifier in modifiers:
            check(reaction.createModifier().setSpecies(identifier), identifier)
        law = reaction.createKineticLaw()
        for identifier, value in local_parameters:
            parameter = (
                law.createLocalParameter()
                if model.getLevel() == 3
                else law.createParameter()
            )
            check(parameter.setId(identifier), identifier)
            check(parameter.setValue(value), identifier)
        if function_id is not None:
            if function_args is None:
                raise _error(reaction_id, "Function arguments are required")
            expression = function_id + "(" + ",".join(function_args) + ")"
        else:
            expression = kl_expr
        if not expression:
            raise _error(reaction_id, "Kinetic expression is required")
        formula = "*".join([*(f"({c})" for c in reaction_comps), f"({expression})"])
        ast = libsbml.parseL3Formula(formula)
        if ast is None:
            raise _error(reaction_id, f"Cannot parse {formula}")
        check(law.setMath(ast), reaction_id)
        check(model.addReaction(reaction), reaction_id)

    transact(sbml_model, operation, log_file)
    return sbml_model.getReaction(reaction_id)
