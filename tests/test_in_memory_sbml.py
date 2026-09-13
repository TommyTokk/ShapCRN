import pytest
import libsbml
from shapcrn.exceptions import InvalidModelFormatError
from shapcrn.utils.sbml.io import (
    load_model_from_string,
    load_model_from_bytes,
    load_and_prepare_model_from_bytes,
)

# Minimal valid SBML Level 3 Version 2 XML string for testing
VALID_SBML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="test_model" name="TestModel">
    <listOfCompartments>
      <compartment id="cytosol" size="1" constant="true"/>
    </listOfCompartments>
  </model>
</sbml>
"""

INVALID_SBML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <!-- Broken XML tag below -->
  <model id="test_model">
</sbml>
"""

NO_MODEL_SBML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
</sbml>
"""


def test_load_model_from_string_valid():
    """Test loading a valid SBML model directly from a string."""
    doc = load_model_from_string(VALID_SBML_XML)  # Load model from string
    assert isinstance(doc, libsbml.SBMLDocument)  # Check returned type
    assert doc.getModel() is not None  # Ensure model exists
    assert doc.getModel().getId() == "test_model"  # Verify model attributes


def test_load_model_from_bytes_valid():
    """Test loading a valid SBML model directly from raw bytes."""
    sbml_bytes = VALID_SBML_XML.encode("utf-8")  # Encode XML string to raw bytes
    doc = load_model_from_bytes(sbml_bytes)  # Load model from bytes
    assert isinstance(doc, libsbml.SBMLDocument)  # Check returned type
    assert doc.getModel() is not None  # Ensure model exists
    assert doc.getModel().getName() == "TestModel"  # Verify model attributes


def test_load_and_prepare_model_from_bytes_valid():
    """Test loading and preparing an SBML model from raw bytes."""
    sbml_bytes = VALID_SBML_XML.encode("utf-8")  # Convert string to bytes
    doc, model = load_and_prepare_model_from_bytes(
        sbml_bytes, split_reversible=True
    )  # Run load & prepare
    assert isinstance(doc, libsbml.SBMLDocument)  # Verify document instance
    assert isinstance(model, libsbml.Model)  # Verify model instance
    assert model.getId() == "test_model"  # Confirm model ID matches input


def test_load_model_from_string_invalid_xml():
    """Test that malformed XML raises InvalidModelFormatError."""
    with pytest.raises(InvalidModelFormatError) as exc_info:
        load_model_from_string(INVALID_SBML_XML)  # Should fail parsing
    assert "<in-memory string>" in str(exc_info.value)  # Check exception context label


def test_load_model_from_bytes_no_model():
    """Test that valid XML missing a <model> element raises InvalidModelFormatError."""
    sbml_bytes = NO_MODEL_SBML_XML.encode("utf-8")  # Encode SBML without model
    with pytest.raises(InvalidModelFormatError) as exc_info:
        load_model_from_bytes(sbml_bytes)  # Should fail validation
    assert "<in-memory bytes>" in str(exc_info.value)  # Verify context label in error
