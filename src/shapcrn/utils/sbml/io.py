import os
import libsbml

from shapcrn.exceptions import InvalidModelFormatError
from shapcrn.utils.utils import print_log
from shapcrn.utils.sbml.helpers import get_sbml_as_xml
from shapcrn.utils.sbml import reactions as sbml_react
from shapcrn.utils.sbml.validation import validate


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
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(f"SBML model does not exist: {model_file_path}")

    reader = libsbml.SBMLReader()

    document = reader.readSBMLFromFile(model_file_path)

    if document.getModel() is None:
        details = "; ".join(
            document.getError(index).getMessage()
            for index in range(document.getNumErrors())
        )
        raise InvalidModelFormatError(model_file_path, details or None)

    validate(document, model_file_path)
    return document


def load_model_from_string(
    xml_string: str, context: str = "<in-memory string>"
) -> libsbml.SBMLDocument:
    """
    Load and validate an SBML model directly from an XML string.

    Parameters
    ----------
    xml_string : str
        The XML string representation of the SBML model.
    context : str, optional
        Context label used for error reporting, by default "<in-memory string>".

    Returns
    -------
    libsbml.SBMLDocument
        The loaded and validated SBML document.

    Raises
    ------
    InvalidModelFormatError
        If the XML string cannot be parsed into a valid SBML model or fails validation.
    """
    reader = libsbml.SBMLReader()  # Initialize the libSBML reader instance
    document = reader.readSBMLFromString(
        xml_string
    )  # Parse XML directly from RAM string

    # Ensure a valid SBML model object was parsed successfully
    if document.getModel() is None:
        # Collect and format error messages reported by libSBML
        details = "; ".join(
            document.getError(index).getMessage()
            for index in range(document.getNumErrors())
        )
        raise InvalidModelFormatError(
            context, details or None
        )  # Raise error with string context

    validate(
        document, context=context
    )  # Perform round-trip consistency and error validation
    return document  # Return fully loaded SBMLDocument object


def load_model_from_bytes(
    sbml_bytes: bytes,
    encoding: str = "utf-8",
    context: str = "<in-memory bytes>",
) -> libsbml.SBMLDocument:
    """
    Load and validate an SBML model directly from raw bytes.

    Parameters
    ----------
    sbml_bytes : bytes
        The raw byte content of the SBML model.
    encoding : str, optional
        The character encoding used to decode the bytes, by default "utf-8".
    context : str, optional
        Context label used for error reporting, by default "<in-memory bytes>".

    Returns
    -------
    libsbml.SBMLDocument
        The loaded and validated SBML document.

    Raises
    ------
    InvalidModelFormatError
        If the byte stream cannot be decoded or parsed into a valid SBML model.
    """
    xml_string = sbml_bytes.decode(encoding)  # Decode raw byte stream into XML string
    return load_model_from_string(
        xml_string, context=context
    )  # Delegate parsing to string handler


def load_and_prepare_model_from_bytes(
    sbml_bytes: bytes,
    split_reversible: bool = True,
    log_file=None,
    encoding: str = "utf-8",
) -> tuple[libsbml.SBMLDocument, libsbml.Model]:
    """
    Load an SBML model from bytes and optionally split reversible reactions.

    Parameters
    ----------
    sbml_bytes : bytes
        The raw byte content of the SBML model.
    split_reversible : bool, optional
        If True, split all reversible reactions into forward/reverse reactions.
    log_file : file-like, optional
        Optional log handle.
    encoding : str, optional
        Character encoding used to decode bytes, by default "utf-8".

    Returns
    -------
    tuple
        (sbml_document, prepared_model)
    """
    context = "<in-memory bytes>"  # Define label for in-memory byte execution context
    sbml_doc = load_model_from_bytes(
        sbml_bytes, encoding=encoding, context=context
    )  # Parse SBML bytes
    sbml_model = sbml_doc.getModel()  # Extract model instance from document

    # Split reversible reactions into forward and backward steps if requested
    if split_reversible:
        sbml_model = sbml_react.split_all_reversible_reactions(sbml_model, log_file)

    validate(
        sbml_doc, context=context, log_file=log_file
    )  # Validate document post-processing
    return sbml_doc, sbml_model  # Return document and modified model tuple


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

    validate(sbml_doc, model_file_path, log_file)
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
    xml = get_sbml_as_xml(model, log_file)
    with open(file_path, "w", encoding="utf-8") as stream:
        stream.write(xml)
    print_log(log_file, f"Successfully saved SBML to: {file_path}")
    return True
