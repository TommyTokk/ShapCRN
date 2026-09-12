"""Semantic and libSBML regression checks for Core transformations."""

from pathlib import Path

import libsbml
import numpy as np
import pytest
import roadrunner

from shapcrn.exceptions import (
    InvalidModelFormatError,
    ModelError,
    ModelModificationError,
    InvalidKineticLawError,
)
from shapcrn.utils.sbml import io, knock, reactions, species
from shapcrn.utils.sbml.helpers import get_sbml_as_xml
from shapcrn.utils.sbml.validation import validate
from shapcrn.utils.simulation import get_species_peak_value, simulate_samples
from shapcrn.utils.sbml.utils import (
    get_fixed_combinations,
    generate_species_random_combinations,
)
from shapcrn.utils.sensitivity import get_problem_parameters


LEVELS = [(2, 1), (2, 4), (3, 1), (3, 2)]


def model_doc(level=3, version=1, formula="cell*(kf*A-kr*B)", volume=2):
    doc = libsbml.SBMLDocument(level, version)
    model = doc.createModel()
    model.setId("test")
    if level == 3:
        model.setTimeUnits("second")
        model.setSubstanceUnits("mole")
        model.setExtentUnits("mole")
    compartment = model.createCompartment()
    compartment.setId("cell")
    compartment.setSize(volume)
    compartment.setConstant(True)
    compartment.setSpatialDimensions(3)
    compartment.setUnits("litre")
    for name, value in [("A", 4), ("B", 1), ("E", 2)]:
        s = model.createSpecies()
        s.setId(name)
        s.setCompartment("cell")
        s.setInitialConcentration(value)
        s.setHasOnlySubstanceUnits(False)
        s.setBoundaryCondition(False)
        s.setConstant(False)
        s.setSubstanceUnits("mole")
    unit_def = model.createUnitDefinition()
    unit_def.setId("per_second")
    u = unit_def.createUnit()
    u.setKind(libsbml.UNIT_KIND_SECOND)
    u.setExponent(-1)
    u.setScale(0)
    u.setMultiplier(1)
    for name, value in [("kf", 0.2), ("kr", 0.1)]:
        p = model.createParameter()
        p.setId(name)
        p.setValue(value)
        p.setUnits("per_second")
        p.setConstant(True)
    if formula in ("cell*kf*A-kr*B", "kf*A-kr*B"):
        volume_rate = unit_def.clone()
        volume_rate.setId("volume_per_second")
        unit = volume_rate.createUnit()
        unit.setKind(libsbml.UNIT_KIND_LITRE)
        unit.setExponent(1)
        unit.setScale(0)
        unit.setMultiplier(1)
        model.addUnitDefinition(volume_rate)
        model.getParameter("kr").setUnits("volume_per_second")
        if formula == "kf*A-kr*B":
            model.getParameter("kf").setUnits("volume_per_second")
    r = model.createReaction()
    r.setId("R")
    r.setReversible(True)
    if level == 2 or version == 1:
        r.setFast(False)
    for create, name in [(r.createReactant, "A"), (r.createProduct, "B")]:
        ref = create()
        ref.setSpecies(name)
        ref.setStoichiometry(1)
        if level == 3:
            ref.setConstant(True)
    law = r.createKineticLaw()
    law.setMath(libsbml.parseL3Formula(formula))
    return doc


def add_function(model, identifier="rate", formula="lambda(k,a,b,k*a-k*b)"):
    f = model.createFunctionDefinition()
    f.setId(identifier)
    f.setMath(libsbml.parseL3Formula(formula))
    return f


def assert_same_dynamics(before, after, forward="R_fwd", reverse="R_rev"):
    original = roadrunner.RoadRunner(libsbml.writeSBMLToString(before))
    split = roadrunner.RoadRunner(libsbml.writeSBMLToString(after))
    for a, b in [(1.0, 2.0), (7.0, 0.3), (0.0, 0.0)]:
        original["[A]"] = split["[A]"] = a
        original["[B]"] = split["[B]"] = b
        assert split[forward] - split[reverse] == pytest.approx(original["R"])
    original.resetAll()
    split.resetAll()
    np.testing.assert_allclose(
        original.simulate(0, 5, 31, selections=["time", "[A]", "[B]"]),
        split.simulate(0, 5, 31, selections=["time", "[A]", "[B]"]),
        rtol=2e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize("level,version", LEVELS)
@pytest.mark.parametrize(
    "formula",
    [
        "cell*(kf*A-kr*B)",
        "cell*kf*A-kr*B",
        "kf*A-kr*B",
        "cell*(kf*(A+A)-kr*B)",
        "cell*(1e-5*A-2e-6*B)",
    ],
)
def test_split_roundtrip_and_dynamics(level, version, formula):
    doc = model_doc(level, version, formula)
    original = doc.clone()
    model = doc.getModel()
    assert reactions.split_all_reversible_reactions(model) is model
    validate(doc)
    assert [r.id for r in model.getListOfReactions()] == ["R_fwd", "R_rev"]
    assert all(not r.getReversible() for r in model.getListOfReactions())
    for r in model.getListOfReactions():
        assert r.isSetFast() == (level == 2 or version == 1)
        for ref in list(r.getListOfReactants()) + list(r.getListOfProducts()):
            assert ref.isSetConstant() == (level == 3)
    assert_same_dynamics(original, doc)


@pytest.mark.parametrize("level,version", LEVELS)
def test_function_mapping_shared_arguments_and_definition(level, version):
    doc = model_doc(level, version, "cell*rate(kr,B,A)")
    model = doc.getModel()
    add_function(model)
    other = model.getReaction("R").clone()
    other.setId("other")
    other.getKineticLaw().setMath(libsbml.parseL3Formula("cell*rate(kf,A,B)"))
    model.addReaction(other)
    original = doc.clone()
    reactions.split_all_reversible_reactions(model)
    assert model.getNumFunctionDefinitions() == 3
    assert model.getFunctionDefinition("rate") is not None
    assert model.getFunctionDefinition("rate_rev").getNumArguments() == 3
    validate(doc)
    assert_same_dynamics(original, doc)


@pytest.mark.parametrize("level,version", LEVELS)
def test_clone_modifiers_and_local_parameter_properties(level, version):
    doc = model_doc(level, version, "cell*(kf*A-kr*B)")
    model = doc.getModel()
    reaction = model.getReaction("R")
    modifier = reaction.createModifier()
    modifier.setSpecies("E")
    reaction.setMetaId("reaction_meta")
    reaction.setAnnotation(
        '<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="#reaction_meta"/></rdf:RDF></annotation>'
    )
    law = reaction.getKineticLaw()
    parameter = law.createLocalParameter() if level == 3 else law.createParameter()
    parameter.setId("kf")
    parameter.setValue(0.7)
    parameter.setName("local forward constant")
    parameter.setUnits("per_second")
    original = doc.clone()
    reactions.split_all_reversible_reactions(model)
    for suffix in ("fwd", "rev"):
        r = model.getReaction("R_" + suffix)
        assert r.getModifier(0).getSpecies() == "E"
        p = (
            r.getKineticLaw().getLocalParameter("kf")
            if level == 3
            else r.getKineticLaw().getParameter("kf")
        )
        assert p.getValue() == 0.7
        assert p.getUnits() == "per_second"
        assert p.getName() == "local forward constant"
        assert r.getMetaId() == "reaction_meta_" + suffix
        assert "#reaction_meta_" + suffix in r.getAnnotationString()
    assert_same_dynamics(original, doc)


def test_inference_keeps_flag_independent_and_mm_exclusion():
    doc = model_doc()
    model = doc.getModel()
    model.getReaction("R").setReversible(False)
    assert reactions.is_reversible(model, model.getReaction("R"))
    model.getReaction("R").getKineticLaw().setMath(libsbml.parseL3Formula("kf*A/(1+B)"))
    model.getReaction("R").setReversible(True)
    assert not reactions.is_reversible(model, model.getReaction("R"))
    reactions.split_all_reversible_reactions(model)
    assert model.getNumReactions() == 1


@pytest.mark.parametrize(
    "case",
    [
        "unsupported",
        "collision",
        "dynamic",
        "initial_stoich",
        "reaction_reference",
        "fast",
    ],
)
def test_split_failure_is_atomic(case):
    doc = model_doc()
    model = doc.getModel()
    r = model.getReaction("R")
    if case == "unsupported":
        r.getKineticLaw().setMath(libsbml.parseL3Formula("kf*A+(-kr*B)"))
    elif case == "collision":
        p = model.createParameter()
        p.setId("R_rev")
        p.setValue(1)
        p.setConstant(True)
    elif case == "dynamic":
        r.getReactant(0).setConstant(False)
    elif case == "initial_stoich":
        r.getReactant(0).setId("nu")
        a = model.createInitialAssignment()
        a.setSymbol("nu")
        a.setMath(libsbml.parseL3Formula("2"))
    elif case == "reaction_reference":
        p = model.createParameter()
        p.setId("flux")
        p.setConstant(False)
        a = model.createAssignmentRule()
        a.setVariable("flux")
        a.setMath(libsbml.parseL3Formula("R"))
    elif case == "fast":
        r.setFast(True)
    before = libsbml.writeSBMLToString(doc)
    with pytest.raises(ModelError):
        reactions.split_all_reversible_reactions(model)
    assert libsbml.writeSBMLToString(doc) == before


def test_multi_reaction_failure_rolls_back_successful_first_split():
    doc = model_doc()
    m = doc.getModel()
    r = m.getReaction("R").clone()
    r.setId("bad")
    r.getKineticLaw().setMath(libsbml.parseL3Formula("kf*A+(-kr*B)"))
    m.addReaction(r)
    before = libsbml.writeSBMLToString(doc)
    with pytest.raises(InvalidKineticLawError, match="bad"):
        reactions.split_all_reversible_reactions(m)
    assert libsbml.writeSBMLToString(doc) == before


@pytest.mark.parametrize("level,version", LEVELS)
@pytest.mark.parametrize(
    "operation", ["ki_species", "ki_reaction", "ko_species", "ko_reaction", "sink"]
)
def test_perturbations_are_valid(level, version, operation):
    doc = model_doc(level, version)
    m = doc.getModel()
    if operation == "ki_species":
        knock.knockin_species(m, "A", 3)
        assert m.getSpecies("A").getInitialConcentration() == 3
    elif operation == "ki_reaction":
        knock.knockin_reaction(m, m.getReaction("R"), [3])
        r = m.getReaction("R_KI")
        assert r.getReactant(0).getSpecies() == "A_KI"
        assert m.getSpecies("A_KI").getSubstanceUnits() == "mole"
    elif operation == "ko_species":
        knock.knockout_species(m, "A")
    elif operation == "ko_reaction":
        knock.knockout_reaction(m, "R")
    else:
        _, sink = knock.knockout_species_via_reaction(m, "A")
        assert sink.getReactant(0).getStoichiometry() == 1
        assert not sink.getFast()
    validate(doc)
    roadrunner.RoadRunner(libsbml.writeSBMLToString(doc))


def test_knockin_repeated_occurrences_and_local_shadow():
    doc = model_doc(formula="cell*(kf*A*A-kr*B)")
    m = doc.getModel()
    r = m.getReaction("R")
    ref = r.createReactant()
    ref.setSpecies("A")
    ref.setStoichiometry(2)
    ref.setConstant(True)
    knock.knockin_reaction(m, r, [3, 3])
    assert m.getNumSpecies() == 4
    r = m.getReaction("R_KI")
    assert r.getNumReactants() == 2
    assert r.getReactant(1).getStoichiometry() == 2
    assert libsbml.formulaToL3String(r.getKineticLaw().getMath()).count("A_KI") == 2
    doc = model_doc(formula="cell*(kf*A-kr*B)")
    m = doc.getModel()
    r = m.getReaction("R")
    p = r.getKineticLaw().createLocalParameter()
    p.setId("A")
    p.setValue(9)
    knock.knockin_reaction(m, r, [3])
    assert "A_KI" not in libsbml.formulaToL3String(
        m.getReaction("R_KI").getKineticLaw().getMath()
    )


def test_knockout_source_and_modifier_dependency():
    doc = model_doc(formula="kf*B")
    m = doc.getModel()
    r = m.getReaction("R")
    r.getListOfReactants().clear()
    knock.knockout_species(m, "B")
    assert m.getReaction("R").getNumProducts() == 1
    assert (
        libsbml.formulaToL3String(m.getReaction("R").getKineticLaw().getMath()) == "0"
    )
    validate(doc)
    doc = model_doc(formula="cell*(kf*A-kr*B)")
    m = doc.getModel()
    r = m.getReaction("R")
    p = r.createProduct()
    p.setSpecies("E")
    p.setStoichiometry(1)
    p.setConstant(True)
    knock.knockout_species(m, "B")
    assert m.getReaction("R").getModifier(0).getSpecies() == "B"
    validate(doc)


@pytest.mark.parametrize("operation", ["ki", "ki_reaction", "sink"])
def test_perturbation_failure_leaves_original_untouched(operation):
    doc = model_doc()
    m = doc.getModel()
    if operation == "ki":
        a = m.createInitialAssignment()
        a.setSymbol("A")
        a.setMath(libsbml.parseL3Formula("2"))
    elif operation == "ki_reaction":
        p = m.createParameter()
        p.setId("A_KI")
        p.setValue(1)
        p.setConstant(True)
    else:
        m.getSpecies("A").setBoundaryCondition(True)
    before = libsbml.writeSBMLToString(doc)
    with pytest.raises(ModelModificationError):
        if operation == "ki":
            knock.knockin_species(m, "A", 5)
        elif operation == "ki_reaction":
            knock.knockin_reaction(m, m.getReaction("R"), [5])
        else:
            knock.knockout_species_via_reaction(m, "A")
    assert libsbml.writeSBMLToString(doc) == before


@pytest.mark.parametrize("amount_symbol", [False, True])
@pytest.mark.parametrize("stored_amount", [False, True])
def test_quantity_conversions_and_sampling(amount_symbol, stored_amount):
    doc = model_doc(formula="0", volume=4)
    m = doc.getModel()
    s = m.getSpecies("A")
    s.setHasOnlySubstanceUnits(amount_symbol)
    if stored_amount:
        s.setInitialAmount(12)
    else:
        s.setInitialConcentration(3)
    expected = 12 if amount_symbol else 3
    assert species.initial_symbol_value(m, "A") == expected
    assert get_fixed_combinations(m, ["A"], [0]) == [[expected]]
    assert get_problem_parameters(m, 1, ["A"], perturbation_range=10)["bounds"][
        0
    ] == pytest.approx([expected * 0.9, expected * 1.1])
    samples = generate_species_random_combinations(m, ["A"], n_samples=2, variation=0)
    np.testing.assert_allclose(samples, [[expected, expected]])
    assert get_species_peak_value(m, "A", sim_end_time=1) == pytest.approx(expected)
    runner = roadrunner.RoadRunner(libsbml.writeSBMLToString(doc))
    runner.timeCourseSelections = ["time", species.symbol_selection(s)]
    result, _, _ = simulate_samples(runner, [7], ["A"], end_time=1, output_rows=3)
    np.testing.assert_allclose(result[:, 1], 7)
    knock.knockin_species(m, "A", 5)
    assert species.initial_symbol_value(m, "A") == 5


def test_unresolved_quantity_reports_error():
    doc = model_doc()
    m = doc.getModel()
    m.getSpecies("A").setInitialAmount(4)
    m.getCompartment("cell").unsetSize()
    with pytest.raises(ModelError, match="compartment size"):
        species.initial_symbol_value(m, "A")


def test_validation_load_save_errors_warnings_and_no_overwrite(tmp_path):
    doc = model_doc()
    doc.getModel().getReaction("R").unsetFast()
    bad = tmp_path / "invalid.xml"
    bad.write_text(libsbml.writeSBMLToString(doc))
    with pytest.raises(InvalidModelFormatError, match="21110"):
        io.load_model(str(bad))
    target = tmp_path / "output.xml"
    target.write_text("keep me")
    with pytest.raises(InvalidModelFormatError):
        io.save_sbml_model(doc, str(target))
    assert target.read_text() == "keep me"
    good = model_doc()
    good.getModel().getParameter("kf").unsetUnits()
    log = tmp_path / "validation.log"
    get_sbml_as_xml(good, log)
    assert "warning" in log.read_text()


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).parents[1] / "models").glob("*.xml")),
    ids=lambda p: p.name,
)
def test_bundled_models_validate(path):
    io.load_model(str(path))


def test_public_split_helpers_and_constructors():
    doc = model_doc()
    m = doc.getModel()
    f, r = reactions.split_reversible_reaction(m, "R", [], {})
    assert (f.id, r.id) == ("R_forward", "R_reverse")
    doc = model_doc(formula="cell*rate(kf,A,B)")
    m = doc.getModel()
    add_function(m)
    result = reactions.split_reversible_reaction_function(m, "R", "rate", [], {})
    assert [o.id for o in result] == ["rate_fwd", "rate_rev", "R_fwd", "R_rev"]
    f = reactions.create_sbml_function(m, "new", "new", ["x"], "x")
    assert f.id == "new"
    r = reactions.create_sbml_reaction_LMA(
        m, "extra", "extra", [("A", 1, True)], [("B", 1, True)], ["E"], [], [], "kf*A"
    )
    assert r.id == "extra"
    validate(doc)


def test_function_actual_expressions_and_nested_calls():
    doc = model_doc(formula="cell*rate(kr, A+B, B)")
    model = doc.getModel()
    add_function(model, "multiply", "lambda(x,y,x*y)")
    add_function(model, "rate", "lambda(k,a,aa,multiply(k,a)-multiply(k,aa))")
    original = doc.clone()
    reactions.split_all_reversible_reactions(model)
    assert_same_dynamics(original, doc)
    validate(doc)


def test_global_dynamic_parameter_stays_global():
    doc = model_doc()
    model = doc.getModel()
    model.getParameter("kf").setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("kf")
    rule.setMath(libsbml.parseL3Formula("kr*(1+time)"))
    original = doc.clone()
    reactions.split_all_reversible_reactions(model)
    for r in model.getListOfReactions():
        assert r.getKineticLaw().getNumLocalParameters() == 0
    assert_same_dynamics(original, doc)


def test_named_fixed_references_and_unset_local_values_are_preserved():
    doc = model_doc()
    model = doc.getModel()
    r = model.getReaction("R")
    r.setCompartment("cell")
    r.getReactant(0).setId("nu")
    r.getReactant(0).setName("two_A")
    r.getReactant(0).setStoichiometry(2)
    p = r.getKineticLaw().createLocalParameter()
    p.setId("unused")
    original = doc.clone()
    reactions.split_all_reversible_reactions(model)
    for name, getter in [("R_fwd", "getReactant"), ("R_rev", "getProduct")]:
        r = model.getReaction(name)
        ref = getattr(r, getter)(0)
        assert ref.getStoichiometry() == 2
        assert ref.getName() == "two_A"
        assert r.getCompartment() == "cell"
        assert not r.getKineticLaw().getLocalParameter("unused").isSetValue()
    assert model.getReaction("R_rev").getProduct(0).id == "nu_rev"
    validate(doc)
    # Undefined unused local parameter is permitted; no numerical default is invented.
    assert (
        original.getModel()
        .getReaction("R")
        .getKineticLaw()
        .getLocalParameter("unused")
        .isSetValue()
        is False
    )


@pytest.mark.parametrize("amount_symbol", [False, True])
def test_sink_amount_concentration_and_conversion_factor(amount_symbol):
    doc = model_doc(formula="0", volume=4)
    model = doc.getModel()
    s = model.getSpecies("A")
    s.setHasOnlySubstanceUnits(amount_symbol)
    species.set_symbol_value(s, 3)
    factor = model.createParameter()
    factor.setId("conversion")
    factor.setValue(2)
    factor.setConstant(True)
    factor.setUnits("dimensionless")
    s.setConversionFactor("conversion")
    _, reaction = knock.knockout_species_via_reaction(model, "A")
    runner = roadrunner.RoadRunner(get_sbml_as_xml(doc))
    assert runner[reaction.id] == pytest.approx(
        1e20 * 3 * (1 if amount_symbol else 4) / 2
    )
    assert reaction.getReactant(0).getConstant()


def test_species_knockout_updates_rules_initial_assignments_and_events():
    doc = model_doc()
    model = doc.getModel()
    s = model.getSpecies("A")
    s.setBoundaryCondition(True)
    rule = model.createRateRule()
    rule.setVariable("A")
    rule.setMath(libsbml.parseL3Formula("1"))
    initial = model.createInitialAssignment()
    initial.setSymbol("A")
    initial.setMath(libsbml.parseL3Formula("2"))
    event = model.createEvent()
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setInitialValue(False)
    trigger.setPersistent(True)
    trigger.setMath(libsbml.parseL3Formula("time>1"))
    assignment = event.createEventAssignment()
    assignment.setVariable("A")
    assignment.setMath(libsbml.parseL3Formula("5"))
    knock.knockout_species(model, "A")
    validate(doc)
    runner = roadrunner.RoadRunner(get_sbml_as_xml(doc))
    result = runner.simulate(0, 3, 10, selections=["time", "[A]"])
    np.testing.assert_allclose(result[:, 1], 0)


def test_package_dependent_transform_rejected_atomically():
    doc = model_doc()
    doc.enablePackage(libsbml.LayoutExtension.getXmlnsL3V1V1(), "layout", True)
    doc.setPackageRequired("layout", False)
    layout = doc.getModel().getPlugin("layout").createLayout()
    layout.setId("diagram")
    dimensions = layout.getDimensions()
    dimensions.setWidth(10)
    dimensions.setHeight(10)
    dimensions.setDepth(0)
    layout.setDimensions(dimensions)
    before = libsbml.writeSBMLToString(doc)
    with pytest.raises(ModelModificationError, match="package"):
        reactions.split_all_reversible_reactions(doc.getModel())
    assert libsbml.writeSBMLToString(doc) == before


def test_missing_kinetics_is_valid_sbml_but_unsupported_preparation():
    doc = model_doc()
    doc.getModel().getReaction("R").unsetKineticLaw()
    validate(doc)
    with pytest.raises(InvalidKineticLawError, match="R"):
        reactions.split_all_reversible_reactions(doc.getModel())


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).parents[1] / "models").glob("*.xml")),
    ids=lambda p: p.name,
)
def test_bundled_models_prepare(path):
    doc, _ = io.load_and_prepare_model(str(path))
    validate(doc)
