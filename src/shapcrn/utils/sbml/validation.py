"""SBML validation and staged, checked Core transformations."""

import libsbml

from shapcrn.exceptions import InvalidModelFormatError, ModelModificationError
from shapcrn.utils.sbml.helpers import get_nodes_iterator
from shapcrn.utils.utils import print_log


def check(status, context):
    if status != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ModelModificationError(
            "modify SBML", context, libsbml.OperationReturnValue_toString(status)
        )


def as_document(value):
    if isinstance(value, str):
        return libsbml.readSBMLFromString(value)
    if isinstance(value, libsbml.SBMLDocument):
        return value.clone()
    if isinstance(value, libsbml.Model):
        doc = value.getSBMLDocument()
        if doc is not None:
            return doc.clone()
        doc = libsbml.SBMLDocument(value.getSBMLNamespaces())
        check(doc.setModel(value), "attach model")
        return doc
    raise TypeError("Expected SBML Model, SBMLDocument, or XML string")


def validate(value, context="SBML model", log_file=None):
    """Validate a copy and its XML round trip, leaving the caller's error log intact."""
    doc = as_document(value)
    diagnostics = []
    for candidate in (doc, libsbml.readSBMLFromString(libsbml.writeSBMLToString(doc))):
        candidate.checkConsistency()
        for i in range(candidate.getNumErrors()):
            error = candidate.getError(i)
            entry = (
                error.getSeverity(),
                error.getErrorId(),
                error.getMessage().strip(),
            )
            if entry not in diagnostics:
                diagnostics.append(entry)
    errors = [
        f"[{code}] {message}"
        for severity, code, message in diagnostics
        if severity >= libsbml.LIBSBML_SEV_ERROR
    ]
    if doc.getModel() is None:
        errors.append("No SBML model is present")
    if errors:
        raise InvalidModelFormatError(context, "\n".join(errors))
    if log_file is not None:
        for severity, code, message in diagnostics:
            if severity == libsbml.LIBSBML_SEV_WARNING:
                print_log(log_file, f"SBML warning [{code}]: {message}")
    return doc


def elements(model):
    return [model] + list(model.getListOfAllElements())


def require_core(model):
    if model.getLevel() not in (2, 3):
        raise ModelModificationError(
            "transform", model.getId(), "Only SBML Levels 2 and 3 are supported"
        )
    # libSBML attaches an empty legacy Layout plugin when reading ordinary L2
    # documents. It carries no package semantics and must not disable L2 support.
    plugins = (
        obj.getPlugin(i) for obj in elements(model) for i in range(obj.getNumPlugins())
    )
    if any(
        plugin.getPackageName() != "layout" or plugin.getListOfAllElements().getSize()
        for plugin in plugins
    ):
        raise ModelModificationError(
            "transform",
            model.getId(),
            "SBML package-dependent transformations are unsupported",
        )


def require_free_id(model, identifier):
    if (
        model.getElementBySId(identifier) is not None
        or model.getFunctionDefinition(identifier) is not None
    ):
        raise ModelModificationError("create", identifier, "Identifier already exists")


def math_names(node):
    return {n.getName() for n in get_nodes_iterator(node) if n.isName()}


def reject_dependencies(model, identifiers, context):
    identifiers = set(filter(None, identifiers))
    for obj in elements(model):
        # Local parameters shadow global identifiers inside their kinetic law.
        shadow = set()
        if isinstance(obj, libsbml.KineticLaw):
            params = (
                obj.getListOfLocalParameters()
                if model.getLevel() == 3
                else obj.getListOfParameters()
            )
            shadow = {p.getId() for p in params}
        if isinstance(obj, libsbml.FunctionDefinition):
            continue  # Lambda names have their own scope.
        if hasattr(obj, "getMath") and math_names(obj.getMath()) & (
            identifiers - shadow
        ):
            raise ModelModificationError(
                "transform",
                context,
                "Mathematical reference to a replaced reaction or species reference",
            )
        for accessor in ("getVariable", "getSymbol"):
            if hasattr(obj, accessor) and getattr(obj, accessor)() in identifiers:
                raise ModelModificationError(
                    "transform", context, "Assignment targets a replaced identifier"
                )


def reject_species_assignments(model, identifier):
    for obj in elements(model):
        if isinstance(obj, libsbml.AlgebraicRule) and identifier in math_names(
            obj.getMath()
        ):
            raise ModelModificationError(
                "fix species", identifier, "Species occurs in an algebraic rule"
            )
        for accessor in ("getVariable", "getSymbol"):
            if hasattr(obj, accessor) and getattr(obj, accessor)() == identifier:
                raise ModelModificationError(
                    "fix species",
                    identifier,
                    "Species is controlled by a rule, event, or initial assignment",
                )


def require_fixed_references(model, reaction):
    refs = list(reaction.getListOfReactants()) + list(reaction.getListOfProducts())
    for ref in refs:
        if (
            model.getLevel() == 3
            and (not ref.getConstant() or not ref.isSetStoichiometry())
        ) or ref.isSetStoichiometryMath():
            raise ModelModificationError(
                "transform",
                reaction.getId(),
                "Dynamic or unspecified stoichiometry is unsupported",
            )
    reject_dependencies(model, [r.getId() for r in refs], reaction.getId())


def fresh_metaids(model, root, suffix):
    """Copy metadata while keeping XML IDs and RDF fragment references unique."""
    used = {o.getMetaId() for o in elements(model) if o.isSetMetaId()}
    objects = [root] + list(root.getListOfAllElements())
    mapping = {}
    for obj in objects:
        if obj.isSetMetaId():
            old = obj.getMetaId()
            new = old + suffix
            while new in used:
                new += "_"
            used.add(new)
            mapping[old] = new
            check(obj.setMetaId(new), new)
    for obj in objects:
        if obj.isSetAnnotation():
            annotation = obj.getAnnotationString()
            for old, new in mapping.items():
                annotation = annotation.replace(f'"#{old}"', f'"#{new}"').replace(
                    f"'#{old}'", f"'#{new}'"
                )
            check(obj.setAnnotation(annotation), "copy annotation")


_LISTS = (
    "UnitDefinitions",
    "Species",
    "Reactions",
    "FunctionDefinitions",
    "Rules",
    "InitialAssignments",
    "Events",
)


def transact(model, operation, log_file=None):
    """Run operation on a copy, validate, then commit changed lists to the same Model."""
    doc = validate(model, model.getId(), log_file)
    staged = doc.getModel()
    require_core(staged)
    result = operation(staged)
    validate(doc, f"transformed {model.getId()}", log_file)
    changed = []
    for name in _LISTS:
        original = getattr(model, "getListOf" + name)()
        updated = getattr(staged, "getListOf" + name)()
        if original.toSBML() != updated.toSBML():
            changed.append((original, original.clone(), updated))
    try:
        for original, _, updated in changed:
            original.clear()
            for item in updated:
                check(original.append(item), "commit transformation")
    except Exception:
        for original, backup, _ in changed:
            original.clear()
            for item in backup:
                check(original.append(item), "restore transformation")
        raise
    return result
