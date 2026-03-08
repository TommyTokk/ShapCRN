import libsbml
def get_list_of_species(sbml_model: libsbml.Model):
    """
    Get the list of species from the SBML model.

    Parameters:
    sbml_model (libsbml.Model): The SBML model object.

    Returns:
    list: A list of Species in the model.
    """
    
    return sbml_model.getListOfSpecies()