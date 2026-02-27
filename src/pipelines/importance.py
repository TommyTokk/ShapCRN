import libsbml
import pandas as pd
import numpy as np

from src.utils import utils as ut
from src.utils.sbml import io as sbml_io
from src.utils.sbml import utils as sbml_ut
from src.utils.sbml import reactions as sbml_react
from src.utils import simulation as sim_ut



def parse_args(args):
    """
    Parses the command line arguments for the importance assessment pipeline.
    """

    # Model path
    input_path = args.input_path
    operation = args.operation

    # Species selection args
    input_species_ids = args.input_species
    knocked_species_ids = args.knocked
    target_ids = args.target_nodes
    preserve_inputs = args.preserve_inputs

    # Sampling and perturbations
    use_perturbations = args.use_perturbations
    use_fixed_perturbations = args.use_fixed_perturbations

    fixed_perturbations = None

    if use_fixed_perturbations:
        if args.fixed_perturbations is None:
            raise ValueError(
                "You need to specify the variations to use setting the --fixed-perturbations parameter"
            )
        fixed_perturbations = args.fixed_perturbations

    num_samples = args.num_samples
    variation_percentage = args.variation

    perturbations_importance = args.perturbations_importance
    random_perturbations_importance = args.random_perturbations_importance

    # Shapley arguments
    payoff_function = args.payoff_function

    # Simulation arguments
    sim_time = args.time
    sim_integrator = args.integrator
    use_steady_state = args.steady_state
    ss_max_time = args.max_time
    ss_sim_steps = args.sim_step
    ss_sim_points = args.points
    ss_threshold = args.threshold

    # Output arguments
    output_dir = args.output
    log_file = args.log

    #Pack args and return
    parsed_args = {
        "input_path": input_path,
        "operation": operation,
        "input_species_ids": input_species_ids,
        "knocked_species_ids": knocked_species_ids,
        "target_ids": target_ids,
        "preserve_inputs": preserve_inputs,
        "use_perturbations": use_perturbations,
        "use_fixed_perturbations": use_fixed_perturbations,
        "fixed_perturbations": fixed_perturbations,
        "num_samples": num_samples,
        "variation_percentage": variation_percentage,
        "perturbations_importance": perturbations_importance,
        "random_perturbations_importance": random_perturbations_importance,
        "payoff_function": payoff_function,
        "sim_time": sim_time,
        "sim_integrator": sim_integrator,
        "use_steady_state": use_steady_state,
        "ss_max_time": ss_max_time,
        "ss_sim_steps": ss_sim_steps,
        "ss_sim_points": ss_sim_points,
        "ss_threshold": ss_threshold,
        "output_dir": output_dir,
        "log_file": log_file
    }

    return parsed_args


def model_preparation(args):
    """
    Prepares the model for importance assessment by loading it and selecting the species to analyze.
    """


    log_file = args["log_file"]

    sbml_doc = sbml_io.load_model(args["input_path"])
    sbml_model = sbml_react.split_all_reversible_reactions(sbml_doc.getModel(), args['log_file'])

    species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

    # Validate the knocked list
    if args["knocked_species_ids"] is None:# If no knocked species are provided, all will be used
        knocked_ids = species_list
    else:# Otherwise only the ones passed
        knocked_ids = args["knocked_species_ids"]

    # Check for the input preservation
    if args["preserve_inputs"] is not None:
        knocked_ids = list(set(knocked_ids) - set(args["input_species_ids"]))

    # Sort the species
    id_to_idx = {}
    for indx, s in enumerate(species_list):
        id_to_idx[s] = indx

    knocked_ids.sort(key=lambda x: id_to_idx[x])

    return {
        'sbml_model': sbml_model,
        'knocked_ids': knocked_ids
    }

def generate_samples(sbml_model, args):

    samples = None

    use_perturbations = args["use_perturbations"]
    use_fixed_perturbations = args["use_fixed_perturbations"]

    if use_perturbations:
        if use_fixed_perturbations:
            samples = sbml_ut.get_fixed_combinations(
                sbml_model,
                args["input_species_ids"],
                args["fixed_perturbations"],
                args["log_file"]
            )
        else:
            samples = sbml_ut.generate_species_random_combinations(
                sbml_model,
                target_species=args["input_species_ids"],
                n_samples=args["num_samples"],
                variation = args["variation_percentage"]
            )
    return samples

def simulate_original_model(sbml_model:libsbml.Model, knocked_ids, samples,  args):

    species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

    # Load the roadrunner model
    rr = sim_ut.load_roadrunner_model(
        sbml_model=sbml_model,
        integrator=args["sim_integrator"],
        log_file=args["log_file"]
    )

    # fix the selections
    selections = rr.timeCourseSelections
    for s in species_list:
        if f"[{s}]" not in selections:
            selections.append(f"[{s}]")

    # Add also the reactions in the selections if target
    for ts in knocked_ids:
        if ts in sbml_model.getListOfReactions().getId():
            selections.append(f"{ts}")

    rr.timeCourseSelections = selections

    # Simulate 

    original_results, ss_time, colnames = sim_ut.simulate(
        rr,
        end_time=args["sim_time"],
        start_time=0,
        steady_state=args["use_steady_state"],
        max_end_time=args["ss_max_time"],
        log_file=args["log_file"]
    )

    original_df = pd.DataFrame(original_results[:, 1:], columns=colnames[1:])

    colnames_to_index = {}
    for i, el in enumerate(colnames):
        if el == "time":
            continue
        colnames_to_index[el] = i

    min_ss_time = (
        ss_time
        if ss_time is not None and ss_time <= args["ss_max_time"]
        else args["ss_max_time"]
    )

    original_data = None

    # Simulating the perturbations if required

    if args["use_perturbations"]:
        if args["use_fixed_perturbations"]:
            ut.print_log(args["log_file"], "[INFO] Simulating with fixed perturbations")
        else:
            ut.print_log(args["log_file"], "[INFO] Simulating with random perturbations")

        samples_simulations_results, _ = sim_ut.simulate_combinations(
            rr,
            sbml_ut.create_combinations(samples),
            args["input_species_ids"],
            min_ss_time,
            args["sim_time"],
            args["ss_max_time"],
            args["use_steady_state"],
            args["log_file"]
        )

    # Prepare the final data
    original_data = [original_df]

    if args["use_perturbations"]:
        for i in range(len(samples_simulations_results)):
            sim_res_i = samples_simulations_results[i]
            original_data.append(
                pd.DataFrame(sim_res_i[:, 1:], columns=colnames[1:])
            )

    return original_data, selections, min_ss_time

def simulate_knocked_data(sbml_model: libsbml.Model, knocked_ids, samples, selections, ss_min_time, args):
    
    operation = args["operation"]

    ut.print_log(args["log_file"], f"Operation: {operation}")

    # Create the models

    knocked_data = []

    sbml_str = libsbml.writeSBMLToString(sbml_model.getSBMLDocument())

    if operation == "knockin":
        models_dict = sbml_ut.create_ki_models(knocked_ids, sbml_model, sbml_str, args["log_file"])
    elif operation == "knockout":
        models_dict = sbml_ut.create_ko_models(knocked_ids, sbml_model, sbml_str, args["log_file"])

    # TODO: Complete the knockout 
    knocked_data = sim_ut.process_species_multiprocessing(
        knocked_ids,
        models_dict,
        samples,
        args["input_species_ids"],
        selections,
        args["sim_integrator"],
        start_time=0,
        end_time = args["sim_time"],
        steady_state=args["use_steady_state"],
        max_end_time=args["ss_max_time"],
        min_ss_time=ss_min_time,
        use_perturbations=args["use_perturbations"],
        preserve_input=args["preserve_inputs"],
        log_file=args["log_file"]
    )

    return knocked_data


def run_shap_analysis(original_data, knocked_data, n_combinations, n_input_ids, payoff = "last", log_file=None):
    #Get the dictionary of payoffs

    payoff_dict = sim_ut.get_payoff_vals(
        original_data,
        knocked_data,
        payoff,
        log_file=log_file
    )

    shap_values = sim_ut.get_shapley_values(
        payoff_dict,
        n_combinations,
        n_input_ids,
        log_file=log_file
    )
    



def importance_assessment(args):
    """
    Pipeline for importance assessment of reactions in a CRN.
    """

    # Arguent parsing
    parsed_args = parse_args(args)

    prep_res = model_preparation(parsed_args)

    sbml_model = prep_res["sbml_model"]
    knocked_ids = prep_res["knocked_ids"]

    # Handle the samples
    samples = generate_samples(sbml_model, parsed_args)

    ut.print_log(parsed_args["log_file"], f"{samples}")

    # Simulate original model
    original_simulation_data, selections, min_ss_time = simulate_original_model(sbml_model, knocked_ids, samples, parsed_args)


    knocked_data = simulate_knocked_data(sbml_model, knocked_ids, samples, selections, min_ss_time, parsed_args)

    

    # Analyse the results
    if parsed_args["use_perturbations"]:
        # TODO: Calculate the Shapley value
        n_combinations = np.power(parsed_args["num_samples"], len(parsed_args["input_species_ids"])) + 1
        shapley_values = run_shap_analysis(
            original_simulation_data, 
            knocked_data, 
            n_combinations, 
            len(parsed_args["input_species_ids"]),
            payoff=parsed_args["payoff_function"], 
            log_file=parsed_args["log_file"]
            )
        # TODO: Calculate the variations
        # TODO: Plot the heatmaps
        # TODO: Create the report
        pass