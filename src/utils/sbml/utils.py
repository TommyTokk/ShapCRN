from typing import Generator


import libsbml

import numpy as np
import itertools

from src import exceptions
from src.classes.species import Species
from src.classes.function import Function
from src.utils.utils import print_log
import src.utils.sbml.knock as sk

# Re-exports from helpers for backward compatibility
from src.utils.sbml.helpers import get_nodes_iterator, Op, get_sbml_as_xml, get_list_of_reactions  # noqa: F401


def create_ki_models(
    target_ids: list, sbml_model, sbml_str: str, log_file=None
) -> dict:
    """
    Create the knock-in models for the specified target IDs.

    Parameters
    ----------
    target_ids : list
        List of target IDs to knock-in.
    sbml_model : libsbml.Model
        The original SBML model.
    sbml_str : str
        The SBML model as a string.
    log_file : file, optional
        File to log information, by default None.

    Returns
    -------
    dict
        A dictionary with target IDs as keys and their corresponding knock-in SBML models as values.
    """
    model_dict = {}
    for ids in target_ids:
        doc_copy = libsbml.readSBMLFromString(sbml_str)  
        model_copy = doc_copy.getModel()
        if ids in [s.getId() for s in sbml_model.getListOfSpecies()]:
            modified_model = sk.knockin_species(model_copy, ids, log_file)

        elif ids in [r.getId() for r in sbml_model.getListOfReactions()]:
            modified_model = sk.knockin_reaction(model_copy, ids, log_file)

        else:
            raise Exception("Id not present in the model")

        doc_copy.setModel(modified_model)
        model_dict[ids] = libsbml.writeSBMLToString(doc_copy)

    return model_dict


def create_ko_models(
    target_ids: list, sbml_model, sbml_str: str, log_file=None
) -> dict:
    """
    Create the knockout models for the specified target IDs.

    Parameters
    ----------
    target_ids : list
        List of target IDs to knockout.
    sbml_model : libsbml.Model
        The original SBML model.
    sbml_str : str
        The SBML model as a string.
    log_file : file, optional
        File to log information, by default None.

    Returns
    -------
    dict
        A dictionary with target IDs as keys and their corresponding knockout SBML models as values.
    """
    model_dict = {}
    for ids in target_ids:
        doc_copy = libsbml.readSBMLFromString(sbml_str)  # rebuild in memory
        model_copy = doc_copy.getModel()
        if ids in [s.getId() for s in sbml_model.getListOfSpecies()]:
            modified_model = sk.knockout_species(model_copy, ids, log_file)

        elif ids in [r.getId() for r in sbml_model.getListOfReactions()]:
            modified_model = sk.knockout_reaction(model_copy, ids, log_file)

        else:
            raise Exception("Id not present in the model")

        doc_copy.setModel(modified_model)
        model_dict[ids] = libsbml.writeSBMLToString(doc_copy)

    return model_dict


# KEEP


# ============
# SPECIES
# ============


def get_list_of_species(SBML_model: libsbml.Model) -> list:
    """
    Get a list of Species objects from the SBML model.

    Parameters
    ----------
    SBML_model : libsbml.Model
        SBML model object

    Returns
    -------
    list
        List of Species objects
    """
    species_list = []

    for sbml_species in SBML_model.getListOfSpecies():
        species = Species.from_sbml(sbml_species)
        species_list.append(species)

    return species_list


# KEEP
def get_species_dict(species_list: list) -> dict:
    """
    Create a dictionary of Species objects indexed by ID.

    Parameters
    ----------
    species_list : list
        List of Species objects

    Returns
    -------
    dict
        Dictionary of Species objects indexed by ID
    """
    dict = {}

    for s in species_list:
        dict[s.get_id()] = s.to_dict()

    return dict


def species_dict_list(species_list: list) -> list:
    """
    Convert a list of Species objects to a list of dictionaries.

    Parameters
    ----------
    species_list : list
        List of Species objects

    Returns
    -------
    list
        List of dictionaries representing the species
    """
    return [s.to_dict() for s in species_list]


# KEEP


# ============
# REACTIONS
# ============


# KEEP


def get_reactants_dict(reaction_objects: list) -> dict:
    """
    Creates a dictionary containing all the information about the reactants from a list of Reaction objects.

    Parameters
    ----------
    reaction_objects : list
        A list of Reaction objects

    Returns
    -------
    dict
        Dictionary with reaction information and their reactants
    """

    dict = {}

    for r_obj in reaction_objects:
        dict[r_obj.get_id()] = r_obj.get_reagents()

    return dict


def get_products_dict(reaction_objects: list) -> dict:
    """
    Creates a dictionary containing all the information about the products from a list of Reaction objects.

    Parameters
    ----------
    reaction_objects : list
        A list of Reaction objects

    Returns
    -------
    dict
        Dictionary with reaction information and their products
    """
    dict = {}

    for r_obj in reaction_objects:
        dict[r_obj.get_id()] = r_obj.get_products()

    return dict


def reactions_to_dict(reaction_list: list) -> dict:
    """
    Convert a list of Reaction objects to a JSON-serializable dictionary.

    Parameters
    ----------
    reaction_list : list
        List of Reaction objects
    Returns
    -------
    dict
        Dictionary representation of the reactions
    """
    reactions_dict = {}

    if not reaction_list:
        return reactions_dict

    for reaction in reaction_list:
        reaction_id = reaction.getId()
        reactions_dict[reaction_id] = reaction.to_dict()

    return reactions_dict


# ============
# FUNCTIONS
# ============
def get_functions_list(SBML_model: libsbml.Model) -> list:
    """
    Extract function definitions from an SBML model as Function objects.

    This function retrieves all function definitions from the SBML model and converts
    them into Function objects using the Function class's from_sbml() method.

    Parameters
    ----------
    SBML_model : libsbml.Model
        The SBML model containing function definitions to extract

    Returns
    -------
    list of Function
        List of Function objects created from SBML function definitions.
        Returns empty list if no function definitions exist in the model

    Notes
    -----
    Function definitions in SBML are typically lambda expressions that define
    reusable mathematical functions for kinetic laws or other model components.

    See Also
    --------
    create_sbml_function : Create a new SBML function definition
    split_kinetic_function : Split kinetic function into forward/reverse components
    """
    functions_list = []

    for sbml_func in SBML_model.getListOfFunctionDefinitions():
        function = Function.from_sbml(sbml_func)
        functions_list.append(function)

    return functions_list


# ============
# FUNCTIONS
# ============


# KEEP
def generate_species_random_combinations(
    sbml_model: libsbml.Model,
    target_species: list = [],
    n_samples: int = 5,
    variation: int = 20,
    log_file=None,
) -> list:
    """
    Generate random concentration/amount samples for specified species with percentage variation.

    This function creates random samples for each target species by varying their initial
    values within a specified percentage range. The samples are uniformly distributed
    around each species' initial concentration or amount.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the species to sample
    target_species : list of str, optional
        List of species IDs for which to generate samples, by default []
    n_samples : int, optional
        Number of random samples to generate for each species, by default 5
    variation : int, optional
        Percentage variation range around the initial value (e.g., 20 means ±20%),
        by default 20
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    list of list of float
        2D list where each sublist contains n_samples random values for one species.
        Order matches the order of target_species

    Raises
    ------
    SystemExit
        If a species is not found in the model or if initial values cannot be accessed

    Notes
    -----
    For each species, samples are generated as:
    - Normal case: uniform distribution in [initial * (1 - variation/100), initial * (1 + variation/100)]
    - Zero initial value: uniform distribution in [0, 1e-10] as a fallback

    The function automatically detects whether species use substance units (amounts)
    or concentration values.

    Random number generation uses numpy's default_rng for reproducibility.

    See Also
    --------
    get_fixed_combinations : Generate fixed variation combinations
    create_combinations : Generate all combinations from input samples

    Examples
    --------
    Generate 10 samples with ±30% variation for two species:
    >>> samples = generate_species_random_combinations(
    ...     model, ['A', 'B'], n_samples=10, variation=30, log_file=log
    ... )
    >>> len(samples)  # Number of species
    2
    >>> len(samples[0])  # Number of samples per species
    10
    """
    res = []

    for ts in target_species:
        species = sbml_model.getListOfSpecies().getElementBySId(ts)

        if species is None:
            raise exceptions.ModelError(f"Species {ts} not found in the model.")

        if species.getHasOnlySubstanceUnits() or species.isSetInitialAmount():
            print_log(log_file, f"Using amounts for {ts}")
            t0_val = species.getInitialAmount()
        elif species.isSetInitialConcentration():
            print_log(log_file, f"Using concentration for {ts}")
            t0_val = species.getInitialConcentration()
        else:
            raise exceptions.ModelError(
                f"Species {ts} must have either initial amount or initial concentration set."
            )
        tmp = []

        for i in range(n_samples):
            if t0_val == 0:
                print_log(
                    log_file,
                    f"[WARNING] initial value is 0 for species {ts}, using a small value",
                )
                # Use a small absolute range instead of percentage
                lower_bound = 0
                upper_bound = 1e-10
            else:
                lower_bound = t0_val * (1 - (variation / 100))
                upper_bound = t0_val * (1 + (variation / 100))

            rng = np.random.default_rng()

            sample = rng.uniform(lower_bound, np.nextafter(upper_bound, np.inf))
            tmp.append(sample)

        res.append(tmp)

    # print_log(log_file, f"[GENERATE SAMPLES]{res}")

    return res


# KEEP
def create_combinations(input_samples: list, log_file=None) -> Generator:
    """
    Generate all possible combinations from a 2D list of sample values using Cartesian product.

    This generator function creates all possible combinations by computing the Cartesian
    product of the input lists. It yields combinations one at a time, making it memory-efficient
    for large datasets.

    Parameters
    ----------
    input_samples : list of list
        2D list where each sublist contains sample values for one variable.
        For example: [[1, 2], [3, 4], [5, 6]] represents 2 values for var1,
        2 values for var2, and 2 values for var3
    log_file : file, optional
        File object for logging operations (currently unused), by default None

    Yields
    ------
    tuple
        Each combination as a tuple, where the i-th element comes from input_samples[i].
        Total number of combinations = product of lengths of all sublists

    Notes
    -----
    This function uses itertools.product for efficient Cartesian product computation.
    Memory usage is minimal as combinations are generated lazily.

    The order of elements in each yielded tuple corresponds to the order of sublists
    in input_samples.

    See Also
    --------
    generate_species_random_combinations : Generate random samples for species
    get_fixed_combinations : Generate fixed variation combinations

    Examples
    --------
    Generate all combinations from 2D samples:
    >>> samples = [[1, 2], [3, 4], [5, 6]]
    >>> combinations = list(create_combinations(samples))
    >>> len(combinations)
    8
    >>> combinations[0]
    (1, 3, 5)
    >>> combinations[-1]
    (2, 4, 6)

    Use as a generator for memory efficiency:
    >>> for combo in create_combinations([[1, 2], [3, 4]]):
    ...     print(combo)
    (1, 3)
    (1, 4)
    (2, 3)
    (2, 4)
    """
    # input_samples is a 2D array: e.g., [[1, 2], [3, 4], [5, 6]]
    for comb in itertools.product(*input_samples):
        yield comb


# KEEP
def get_fixed_combinations(
    sbml_model: libsbml.Model,
    input_species: list[str],
    fixed_variations: list[float] | None,
    log_file=None,
) -> list:
    """
    Generate fixed-percentage variation samples for specified species.

    This function creates samples for each target species by applying a set of fixed
    percentage variations to their initial values. Unlike random sampling, this produces
    deterministic samples based on the specified variation percentages.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing the species to sample
    input_species : list of str
        List of species IDs for which to generate variation samples
    fixed_variations : list of float
        List of percentage variations to apply (e.g., [-20, 0, 20] for -20%, 0%, +20%)
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    list of list of float
        2D list where each sublist contains samples for one species.
        Each sublist has len(fixed_variations) values, sorted by variation percentage.
        Order matches the order of input_species

    Raises
    ------
    Exception
        If initial concentration/amount values cannot be accessed for any species

    Notes
    -----
    For each species, samples are generated as:
    - Normal case: initial_value * (1 + variation/100) for each variation percentage
    - Zero initial value: abs(variation) * 1e-10 as a fallback

    The function automatically detects whether species use substance units (amounts)
    or concentration values.

    Samples are sorted by variation percentage in ascending order.

    See Also
    --------
    generate_species_random_combinations : Generate random samples for species
    create_combinations : Generate Cartesian product of samples

    Examples
    --------
    Generate samples with fixed variations for two species:
    >>> samples = get_fixed_combinations(
    ...     model, ['A', 'B'], [-50, 0, 50], log_file=log
    ... )
    >>> len(samples)  # Number of species
    2
    >>> len(samples[0])  # Number of samples per species (3 variations)
    3

    For species 'A' with initial value 100:
    >>> # fixed_variations = [-50, 0, 50]
    >>> # samples[0] = [50.0, 100.0, 150.0]
    """
    samples = []

    for s_id in input_species:
        species = sbml_model.getListOfSpecies().getElementBySId(s_id)

        # Get initial concentration/amount using the same logic as generate_species_samples
        if species.getHasOnlySubstanceUnits() or species.isSetInitialAmount():
            print_log(log_file, f"Using amounts for {s_id}")
            t0_conc = species.getInitialAmount()
        elif species.isSetInitialConcentration():
            print_log(log_file, f"Using concentration for {s_id}")
            t0_conc = species.getInitialConcentration()
        else:
            raise exceptions.ModelError(
                f"Species {s_id} must have either initial amount or initial concentration set."
            )

        tmp = []

        # Handle zero initial concentration case (similar to generate_species_samples)
        if t0_conc == 0:
            print_log(
                log_file,
                f"[WARNING] initial value is 0 for species {s_id}, using small absolute values",
            )
            # For zero concentrations, use small absolute values based on variations
            for v in sorted(fixed_variations):
                # Use small absolute range instead of percentage
                sample = (
                    abs(v) * 1e-10
                )  # Small absolute value proportional to variation
                tmp.append(sample)
        else:
            # Apply percentage variations (same as before but with better logging)
            for v in sorted(fixed_variations):
                sample = t0_conc + (t0_conc * v / 100)
                # print_log(
                #     log_file,
                #     f"[FIXED_COMBINATIONS] Species: {s_id} | t0_value: {t0_conc} | variation: {v}% | sample: {sample}",
                # )
                tmp.append(sample)

        samples.append(tmp)

    return samples


# FOR DEBUG ONLY
def check_for_duplicates(combinations, log_file=None):
    """
    Check if there are duplicate combinations in the list.

    Args:
        combinations: List of combinations to check
        log_file: Optional log file for output

    Returns:
        tuple: (has_duplicates, num_duplicates)
    """
    # Converto ogni combinazione in tupla se già non lo è
    # (le liste non sono hashable, le tuple sì)
    tuple_combinations = [
        tuple(combo) if not isinstance(combo, tuple) else combo
        for combo in combinations
    ]

    # Confronto lunghezze
    original_length = len(combinations)
    unique_combinations = set(tuple_combinations)
    unique_length = len(unique_combinations)

    if original_length == unique_length:
        print_log(log_file, f"No duplicates found in {original_length} combinations.")
        return False, 0
    else:
        # Ci sono duplicati
        num_duplicates = original_length - unique_length
        print_log(
            log_file,
            f"Found {num_duplicates} duplicates in {original_length} combinations.",
        )

        # Trova e stampa alcuni esempi di duplicati
        from collections import Counter

        counts = Counter(tuple_combinations)
        duplicates = {combo: count for combo, count in counts.items() if count > 1}

        print_log(log_file, "Examples of duplicates:")
        for i, (combo, count) in enumerate(duplicates.items()):
            print_log(log_file, f"  Combination {combo} appears {count} times")
            if i >= 2:  # Mostra al massimo 3 esempi
                break

        return True, num_duplicates


def get_selections(
    sbml_model: libsbml.Model, rr_model, target_ids: list, log_file=None
) -> list:
    """
    Get selections for target species/reactions in the SBML model.

    This function retrieves the current selections from the roadrunner model and
    adds target species or reactions to the selection list based on their presence
    in the SBML model.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model containing species and reactions
    rr_model : roadrunner.RoadRunner
        The roadrunner model for accessing current selections
    target_ids : list of str
        List of target species or reaction IDs to check and add to selections
    log_file : file, optional
        File object for logging operations, by default None

    Returns
    -------
    list of str
        Updated list of selections including target species/reactions
    Notes
    -----
    The function checks if each target ID is a reaction or species in the SBML model.
    If it's a reaction, it is added directly. If it's a species, it is added in the
    format "[species_id]" if not already present in the selections.
    """

    selections = rr_model.selections

    # Check if some target species are reaction
    for ts in target_ids:
        print_log(log_file, f"[GET SELECTIONS]{ts}")
        if ts in [r.getId() for r in sbml_model.getListOfReactions()]:
            selections = selections + [f"{ts}"]
        elif (
            ts in [s.getId() for s in sbml_model.getListOfSpecies()]
            and f"[{ts}]" not in selections
        ):
            selections = selections + [f"[{ts}]"]

    # print_log(log_file, f"[GET SELECTIONS]sel: {selections}")
    return selections


# FOR DEBUG ONLY
def check_presence(sbml_model, target_species_id, log_file):
    in_rules = False
    in_reactions = False

    for rule in sbml_model.getListOfRules():
        if rule.getVariable() == target_species_id:  # Found the updating rule
            in_rules = True

    for reaction in sbml_model.getListOfReactions():
        for i in range(reaction.getNumProducts()):
            product = reaction.getProduct(i)
            if product.getSpecies() == target_species_id:
                in_reactions = True

    print_log(
        log_file,
        f"species {target_species_id}, both in rules ({in_rules}) and reactions ({in_reactions})? {in_rules and in_reactions}",
    )
