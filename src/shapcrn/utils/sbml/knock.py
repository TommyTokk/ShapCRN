"""Checked SBML Core knock-in/knockout transformations."""

import libsbml

from shapcrn import exceptions
from shapcrn.utils.sbml import species as species_ut
from shapcrn.utils.sbml.reactions import _replace_names
from shapcrn.utils.sbml.validation import (
    check,
    elements,
    fresh_metaids,
    math_names,
    reject_dependencies,
    reject_species_assignments,
    require_fixed_references,
    require_free_id,
    transact,
)


def _species(model, identifier):
    species = model.getSpecies(identifier)
    if species is None:
        raise exceptions.InvalidSpeciesError(identifier, model.getId())
    return species


def _reaction(model, identifier):
    reaction = model.getReaction(identifier)
    if reaction is None:
        raise exceptions.InvalidReactionError(identifier, model.getId())
    if reaction.getFast():
        raise exceptions.ModelModificationError(
            "perturb", identifier, "fast=true reactions are unsupported"
        )
    return reaction


def _zero_reaction(model, identifier):
    reaction = _reaction(model, identifier)
    law = reaction.getKineticLaw()
    if law is None:
        raise exceptions.InvalidKineticLawError(identifier)
    check(law.setMath(libsbml.parseL3Formula("0")), identifier)


def knockout_reaction(sbml_model, target_reaction_id, log_file=None):
    """Set the rate to zero while preserving the reaction and its references."""
    transact(sbml_model, lambda m: _zero_reaction(m, target_reaction_id), log_file)
    return sbml_model


def knockout_species(sbml_model, target_species_id, log_file=None):
    """Pin species updates to zero and disable consuming/terminal reactions.

    Products are removed only when a valid reaction remains. Any removed product
    still used by the kinetic law is declared as a modifier (L3V1 §4.11.5).
    """

    def operation(model):
        species = _species(model, target_species_id)
        zero = libsbml.parseL3Formula("0")
        in_rules = False
        for rule in model.getListOfRules():
            if rule.isAlgebraic() and target_species_id in math_names(rule.getMath()):
                raise exceptions.ModelModificationError(
                    "knock out species",
                    target_species_id,
                    "Algebraic rule dependency is unsupported",
                )
            if rule.getVariable() == target_species_id:
                check(rule.setMath(zero), target_species_id)
                in_rules = True
        for obj in elements(model):
            if (
                isinstance(obj, libsbml.EventAssignment)
                and obj.getVariable() == target_species_id
            ):
                check(obj.setMath(zero), target_species_id)
            if (
                isinstance(obj, libsbml.InitialAssignment)
                and obj.getSymbol() == target_species_id
            ):
                check(obj.setMath(zero), target_species_id)
        if not in_rules:
            for reaction in model.getListOfReactions():
                if any(
                    r.getSpecies() == target_species_id
                    for r in reaction.getListOfReactants()
                ):
                    _zero_reaction(model, reaction.getId())
                    continue
                indices = [
                    i
                    for i, p in enumerate(reaction.getListOfProducts())
                    if p.getSpecies() == target_species_id
                ]
                if not indices:
                    continue
                _reaction(model, reaction.getId())
                require_fixed_references(model, reaction)
                if reaction.getNumReactants() + reaction.getNumProducts() == len(
                    indices
                ):
                    _zero_reaction(model, reaction.getId())
                    continue
                for i in reversed(indices):
                    reaction.removeProduct(i)
                if reaction.getNumProducts() == 0:
                    _zero_reaction(model, reaction.getId())
                law = reaction.getKineticLaw()
                if law is not None:
                    params = (
                        law.getListOfLocalParameters()
                        if model.getLevel() == 3
                        else law.getListOfParameters()
                    )
                    names = math_names(law.getMath()) - {p.getId() for p in params}
                    declared = {
                        r.getSpecies()
                        for r in list(reaction.getListOfReactants())
                        + list(reaction.getListOfProducts())
                        + list(reaction.getListOfModifiers())
                    }
                    for name in sorted(names - declared):
                        if model.getSpecies(name) is not None:
                            check(reaction.createModifier().setSpecies(name), name)
        species_ut.set_symbol_value(species, 0)
        check(species.setBoundaryCondition(True), target_species_id)

    transact(sbml_model, operation, log_file)
    return sbml_model


def knockin_species(sbml_model, species_id, new_val, log_file=None):
    """Fix a species at a value expressed in its SBML symbol units."""

    def operation(model):
        species = _species(model, species_id)
        reject_species_assignments(model, species_id)
        species_ut.set_symbol_value(species, new_val)
        check(species.setBoundaryCondition(True), species_id)
        check(species.setConstant(True), species_id)

    transact(sbml_model, operation, log_file)
    return sbml_model


def knockin_reaction(sbml_model, target_reaction, new_vals, log_file=None):
    """Replace reactants by fixed copies, retaining reference properties and units.

    Values follow reactant order and use SBML symbol units. Repeated references
    to the same species share one fixed copy and must supply the same value.
    """
    if target_reaction is None:
        raise exceptions.InvalidReactionError("None", sbml_model.getId())
    identifier = target_reaction.getId()

    def operation(model):
        reaction = _reaction(model, identifier)
        require_fixed_references(model, reaction)
        reject_dependencies(model, [identifier], identifier)
        refs = list(reaction.getListOfReactants())
        if len(new_vals) != len(refs):
            raise exceptions.ModelModificationError(
                "knock in reaction",
                identifier,
                "Expected one value per reactant reference",
            )
        new_id = identifier + "_KI"
        require_free_id(model, new_id)
        law = reaction.getKineticLaw()
        if law is None or law.getMath() is None:
            raise exceptions.InvalidKineticLawError(identifier)
        mapping, values = {}, {}
        for ref, value in zip(refs, new_vals):
            old_id = ref.getSpecies()
            if old_id in mapping:
                if values[old_id] != value:
                    raise exceptions.ModelModificationError(
                        "knock in reaction",
                        identifier,
                        "Repeated reactant species requires identical values",
                    )
                continue
            copy_id = old_id + "_KI"
            require_free_id(model, copy_id)
            clone = _species(model, old_id).clone()
            check(clone.setId(copy_id), copy_id)
            fresh_metaids(model, clone, "_KI")
            species_ut.set_symbol_value(clone, value)
            check(clone.setBoundaryCondition(True), copy_id)
            check(clone.setConstant(True), copy_id)
            check(model.addSpecies(clone), copy_id)
            mapping[old_id] = libsbml.parseL3Formula(copy_id)
            values[old_id] = value
        clone = reaction.clone()
        check(clone.setId(new_id), new_id)
        for ref in clone.getListOfReactants():
            check(ref.setSpecies(ref.getSpecies() + "_KI"), new_id)
        params = (
            law.getListOfLocalParameters()
            if model.getLevel() == 3
            else law.getListOfParameters()
        )
        for parameter in params:
            mapping.pop(parameter.getId(), None)
        check(
            clone.getKineticLaw().setMath(_replace_names(law.getMath(), mapping)),
            new_id,
        )
        fresh_metaids(model, clone, "_KI")
        model.removeReaction(identifier)
        check(model.addReaction(clone), new_id)

    transact(sbml_model, operation, log_file)
    return sbml_model


def knockout_species_via_reaction(sbml_model, target_species_id, log_file=None):
    """Add a rapid, ordinary irreversible sink with rate in reaction-extent/time."""
    identifier = "d_" + target_species_id

    def operation(model):
        species = _species(model, target_species_id)
        reject_species_assignments(model, target_species_id)
        if species.getBoundaryCondition() or species.getConstant():
            raise exceptions.ModelModificationError(
                "deactivate",
                target_species_id,
                "A reaction cannot consume a fixed or boundary species",
            )
        require_free_id(model, identifier)
        reaction = libsbml.Reaction(model.getSBMLNamespaces())
        check(reaction.setId(identifier), identifier)
        check(reaction.setName("deactivation_of_" + target_species_id), identifier)
        check(reaction.setReversible(False), identifier)
        if model.getLevel() == 2 or model.getVersion() == 1:
            check(reaction.setFast(False), identifier)
        ref = reaction.createReactant()
        check(ref.setSpecies(target_species_id), identifier)
        check(ref.setStoichiometry(1), identifier)
        if model.getLevel() == 3:
            check(ref.setConstant(True), identifier)
        law = reaction.createKineticLaw()
        parameter = (
            law.createLocalParameter()
            if model.getLevel() == 3
            else law.createParameter()
        )
        parameter_id = "k_ko"
        while model.getElementBySId(parameter_id) is not None:
            parameter_id += "_"
        check(parameter.setId(parameter_id), identifier)
        check(parameter.setValue(1e20), identifier)
        # Inverse model time units. Leave units unspecified if model time is unspecified.
        time_id = model.getTimeUnits() if model.getLevel() == 3 else "time"
        time_def = model.getUnitDefinition(time_id) if time_id else None
        if time_def is not None or time_id == "second" or model.getLevel() == 2:
            unit_id = identifier + "_per_time"
            require_free_id(model, unit_id)
            definition = (
                time_def.clone()
                if time_def is not None
                else libsbml.UnitDefinition(model.getSBMLNamespaces())
            )
            check(definition.setId(unit_id), unit_id)
            fresh_metaids(model, definition, "_inverse")
            if time_def is None:
                unit = definition.createUnit()
                check(unit.setKind(libsbml.UNIT_KIND_SECOND), unit_id)
                check(unit.setExponent(1), unit_id)
                check(unit.setScale(0), unit_id)
                check(unit.setMultiplier(1), unit_id)
            for unit in definition.getListOfUnits():
                check(unit.setExponent(-unit.getExponent()), unit_id)
            check(model.addUnitDefinition(definition), unit_id)
            check(parameter.setUnits(unit_id), unit_id)
        formula = f"{parameter_id}*{target_species_id}"
        if not species.getHasOnlySubstanceUnits():
            formula += "*" + species.getCompartment()
        factor = species.getConversionFactor() or model.getConversionFactor()
        if factor:
            formula = f"({formula})/{factor}"
        check(law.setMath(libsbml.parseL3Formula(formula)), identifier)
        check(model.addReaction(reaction), identifier)

    transact(sbml_model, operation, log_file)
    return sbml_model, sbml_model.getReaction(identifier)
