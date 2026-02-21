"""
Low-level helper utilities for SBML manipulation.

This module contains leaf functions that are shared between multiple sbml
sub-modules (e.g. utils, reactions, knock) and depend only on external
libraries (libsbml, etc.) — never on other local src modules.
This keeps the module free of circular-import risks.
"""

from typing import Generator
from enum import Enum

import libsbml


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

class Op(Enum):
    """AST operation types used for kinetic-law manipulation."""
    MINUS = libsbml.AST_MINUS
    PLUS = libsbml.AST_PLUS
    PROD = libsbml.AST_TIMES


def get_nodes_iterator(node: libsbml.ASTNode) -> Generator[libsbml.ASTNode, None, None]:
    """
    Generator function to recursively yield all nodes in an AST.

    Parameters
    ----------
    node : libsbml.ASTNode
        The root AST node to start the traversal from.
    Yields
    ------
    libsbml.ASTNode
        Each node in the AST, traversed in pre-order.
    """
    if node is None:
        return

    # Yield the current node immediately
    yield node

    # Yield from children
    for i in range(node.getNumChildren()):
        # 'yield from' seamlessly streams values from the recursive call
        yield from get_nodes_iterator(node.getChild(i))


# ---------------------------------------------------------------------------
# Model query helpers
# ---------------------------------------------------------------------------

def get_list_of_reactions(sbml_model: libsbml.Model) -> list:
    """
    Get a list of reaction IDs from the SBML model.

    Parameters
    ----------
    sbml_model : libsbml.Model
        SBML model object

    Returns
    -------
    list
        List of reaction IDs
    """
    return [r.getId() for r in sbml_model.getListOfReactions()]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def get_sbml_as_xml(model, log_file=None) -> str:
    """
    Convert an SBML model to its XML string representation.

    This function accepts an SBML model in various formats and converts it to an XML
    string. It handles automatic conversion to SBMLDocument when necessary before
    serialization.

    Parameters
    ----------
    model : libsbml.Model, str, or libsbml.SBMLDocument
        The SBML model to convert. Can be:
        - libsbml.Model: A model object (will be wrapped in SBMLDocument if needed)
        - str: An XML string (returned as-is)
        - libsbml.SBMLDocument: A complete SBML document
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    str or None
        The XML string representation of the SBML model, or None if conversion fails

    Notes
    -----
    If the model is a libsbml.Model without an associated SBMLDocument, a new
    SBMLDocument is created using the model's SBML level and version.

    If the input is already a string, it is returned unchanged without validation.

    Examples
    --------
    Convert a model to XML string:
    >>> xml_string = get_sbml_as_xml(my_model)
    >>> print(xml_string[:100])  # Print first 100 characters

    Pass through an existing XML string:
    >>> xml_str = "<sbml>...</sbml>"
    >>> result = get_sbml_as_xml(xml_str)
    >>> assert result == xml_str
    """
    # Check the type of the model and convert to SBMLDocument if necessary
    if isinstance(model, libsbml.Model):
        # If it's a Model, get the associated document
        doc = model.getSBMLDocument()
        if doc is None:
            # If there's no associated document, create a new one
            doc = libsbml.SBMLDocument(model.getLevel(), model.getVersion())
            doc.setModel(model)
        xml_string = libsbml.writeSBMLToString(doc)
    elif isinstance(model, str):
        # If it's already an XML string, return it
        return model
    else:
        # Otherwise, try to write it directly to string
        xml_string = libsbml.writeSBMLToString(model)

    if xml_string:
        return xml_string
    else:
        raise ValueError(
            "Failed to convert SBML model to XML string. Check log for details."
        )
