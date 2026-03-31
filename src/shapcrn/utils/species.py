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
