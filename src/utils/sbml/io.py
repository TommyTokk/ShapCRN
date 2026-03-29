import os
import libsbml

from src.utils.utils import print_log
from src.utils.sbml.helpers import get_sbml_as_xml
from src.utils.sbml import reactions as sbml_react


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


def load_and_prepare_model(
    model_file_path: str, split_reversible: bool = True, log_file=None
) -> tuple[libsbml.SBMLDocument, libsbml.Model]:
    """
    Load an SBML model and optionally split reversible reactions.

    Parameters
    ----------
    model_file_path : str
        Path to the SBML model file.
    split_reversible : bool, optional
        If True, split all reversible reactions into forward/reverse reactions.
    log_file : file-like, optional
        Optional log handle.

    Returns
    -------
    tuple
        (sbml_document, prepared_model)
    """
    sbml_doc = load_model(model_file_path)
    sbml_model = sbml_doc.getModel()

    if split_reversible:
        sbml_model = sbml_react.split_all_reversible_reactions(sbml_model, log_file)

    return sbml_doc, sbml_model


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


def save_sbml_model(
    model: libsbml.Model | libsbml.SBMLDocument | str, file_path: str, log_file=None
) -> bool:
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
        raise IOError(
            f"Failed to save SBML model to {file_path}. Check log for details."
        )

    return success
