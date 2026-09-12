import libsbml
from typing import Optional


def get_list_of_species(sbml_model: libsbml.Model):
    """
    Get the list of species from the SBML model.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.

    Returns:
    list: A list of Species in the model.
    """

    return sbml_model.getListOfSpecies()


def get_list_of_species_ids(sbml_model: libsbml.Model) -> list:
    """
    Get the list of species IDs from the SBML model.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.

    Returns:
    list: A list containing all species IDs in the model.
    """
    return [species.getId() for species in get_list_of_species(sbml_model)]


def get_list_of_species_names(sbml_model: libsbml.Model) -> list:
    """
    Get the list of species names from the SBML model.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.

    Returns:
    list: A list containing all species names in the model.
    """
    return [species.getName() for species in get_list_of_species(sbml_model)]


def get_species_by_id(
    sbml_model: libsbml.Model, species_id: str
) -> Optional[libsbml.Species]:
    """
    Retrieve a species from the SBML model by its ID.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.
    species_id (str): The species ID.

    Returns:
    Optional[libsbml.Species]: The matching species, or None if not found.
    """
    return sbml_model.getSpecies(species_id)


def has_species(sbml_model: libsbml.Model, species_id: str) -> bool:
    """
    Check whether a species exists in the SBML model.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.
    species_id (str): The species ID.

    Returns:
    bool: True if the species exists, False otherwise.
    """
    return get_species_by_id(sbml_model, species_id) is not None


def get_num_species(sbml_model: libsbml.Model) -> int:
    """
    Get the number of species in the SBML model.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.

    Returns:
    int: The number of species in the model.
    """
    return sbml_model.getNumSpecies()


def symbol_selection(species):
    """RoadRunner selection for the quantity denoted by the SBML species symbol."""
    return (
        species.getId()
        if species.getHasOnlySubstanceUnits()
        else f"[{species.getId()}]"
    )


def initial_symbol_value(model, identifier):
    """Read a literal initial value in SBML symbol units (L3V1 §4.6).

    Initial assignments and assignment rules require evaluation and are explicitly
    unsupported here. Stored amount/concentration is independent of symbol units.
    """
    import math
    from shapcrn.exceptions import InvalidSpeciesError, ModelError
    from shapcrn.utils.sbml.validation import math_names

    species = model.getSpecies(identifier)
    if species is None:
        raise InvalidSpeciesError(identifier, model.getId())

    def literal(obj):
        sid = obj.getId()
        if (
            model.getInitialAssignmentBySymbol(sid) is not None
            or model.getAssignmentRuleByVariable(sid) is not None
        ):
            raise ModelError(f"{sid}: initialization is controlled by an assignment")
        if any(
            rule.isAlgebraic() and sid in math_names(rule.getMath())
            for rule in model.getListOfRules()
        ):
            raise ModelError(
                f"{sid}: initialization requires solving an algebraic rule"
            )

    literal(species)
    if species.isSetInitialAmount():
        value, stored_amount = species.getInitialAmount(), True
    elif species.isSetInitialConcentration():
        value, stored_amount = species.getInitialConcentration(), False
    else:
        raise ModelError(f"{identifier}: no resolved initial amount or concentration")
    if stored_amount != species.getHasOnlySubstanceUnits():
        compartment = model.getCompartment(species.getCompartment())
        literal(compartment)
        size = compartment.getSize()
        if not compartment.isSetSize() or not math.isfinite(size) or size <= 0:
            raise ModelError(
                f"{identifier}: positive, known compartment size is required for quantity conversion"
            )
        value = value * size if species.getHasOnlySubstanceUnits() else value / size
    if not math.isfinite(value):
        raise ModelError(f"{identifier}: initial quantity must be finite")
    return value


def set_symbol_value(species, value):
    """Set the initial value directly in the species symbol's units."""
    import math
    from shapcrn.exceptions import ModelError
    from shapcrn.utils.sbml.validation import check

    if not math.isfinite(value) or value < 0:
        raise ModelError(f"{species.getId()}: quantity must be finite and nonnegative")
    setter = (
        species.setInitialAmount
        if species.getHasOnlySubstanceUnits()
        else species.setInitialConcentration
    )
    check(setter(float(value)), species.getId())


def set_roadrunner_initial_values(runner, identifiers, values):
    """Apply sample values with explicit amount/concentration selections."""
    from shapcrn.exceptions import ModelError, InvalidSpeciesError
    from shapcrn.utils.sbml.validation import reject_species_assignments

    if len(identifiers) != len(values):
        raise ModelError("Sample value count does not match input species")
    doc = libsbml.readSBMLFromString(runner.getSBML())
    model = doc.getModel()
    for identifier, value in zip(identifiers, values):
        species = model.getSpecies(identifier)
        if species is None:
            raise InvalidSpeciesError(identifier, model.getId())
        reject_species_assignments(model, identifier)
        set_symbol_value(
            species, float(value)
        )  # Validate values before touching the runner.
    for identifier, value in zip(identifiers, values):
        selection = symbol_selection(model.getSpecies(identifier))
        runner.setValue(f"init({selection})", float(value))
