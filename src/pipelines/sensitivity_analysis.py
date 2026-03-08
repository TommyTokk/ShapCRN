import os 
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import libsbml

import src.utils.sensitivity as sens_ut
from src.utils import utils as ut
from src import exceptions as ex
from src.utils.sbml import io as sbml_io
from src.utils.sbml import reactions as sbml_react
from src.utils.sbml import utils as sbml_ut
from src.utils import species as sbml_species
from src.utils import simulation as sim_ut

N_VALUES = [64, 128, 256, 512, 1024]


def parse_args(args):
    model_path = args.input_path
    input_species_ids = args.input_species
    base_samples = args.base_samples
    perturbations_range = args.perturbation_range
    knock_operation = args.operation
    preserve_input = args.preserve_inputs
    target_species = args.target_species
    fixed_perturbations = args.fixed_perturbations
    check_convergence = args.check_convergence
    log_file = args.log if args.log else None

    parsed_args = {
        'model_path': model_path,
        'input_species_ids': input_species_ids,
        'base_samples': base_samples,
        'perturbation_range': perturbations_range,
        'knock_operation': knock_operation,
        'preserve_input': preserve_input,
        'target_species': target_species,
        'fixed_perturbations': fixed_perturbations,
        'check_convergence': check_convergence,
        'log_file': log_file
    }

    return parsed_args

def model_preparation(args):
    model_path = args['model_path']
    input_species_ids = args['input_species_ids']

    # Load the model
    sbml_doc = sbml_io.load_model(model_path)

    # Get the model from the document
    model = sbml_react.split_all_reversible_reactions(sbml_doc.getModel(), args['log_file'])

    # Check if the input species exist in the model
    for species_id in input_species_ids:
        if not model.getSpecies(species_id):
            raise ex.SpeciesNotFoundError(f"Species '{species_id}' not found in the model.")

    return sbml_doc, model


def run_convergence_analysis(rr_model, input_species, problem_specs, target_species, valid_idxs, out_dirs, log_file):
    convergence_results = {}

    valid_elements = list(valid_idxs.keys())
    
    for N in N_VALUES:
        params = saltelli.sample(problem_specs, N, calc_second_order=True)

        sim_results = sens_ut.run_simulation_with_params(
            rr_model,
            params,
            valid_elements,
            valid_idxs,
            input_species,
        )

        # Update the convergence results
        convergence_results[N] = {}

        for j, node in enumerate(valid_elements):
            node_data = sim_results[:, j]
            
            # Check for NaNs and Variance
            num_nans = np.isnan(node_data).sum()
            if num_nans < len(node_data):
                variance = np.var(node_data[~np.isnan(node_data)])
            else:
                variance = 0.0

            print(f"[DEBUG] N={N} | Node: {node} | NaNs dropped: {num_nans} | Variance: {variance:.6f}")

            valid_mask = ~np.isnan(node_data)

            # If even ONE NaN is dropped, SALib will fail due to length mismatch
            if valid_mask.sum() > 0:
                try:
                    Si = sobol.analyze(problem_specs, node_data[valid_mask], calc_second_order=True, print_to_console=False)
                    convergence_results[N][node] = Si
                except Exception as e:
                    ut.print_log(log_file, f"[WARNING] SALib failed for node '{node}' at N={N}. Error: {e}")
                    convergence_results[N][node] = None
            else:
                ut.print_log(log_file, f"[WARNING] All data is NaN for node '{node}' at N={N}.")
                convergence_results[N][node] = None

    convergence_info = sens_ut.check_convergence(convergence_results, valid_elements, tol_ci=0.10, min_consecutive=2, log_file=log_file)

    

    return convergence_results, convergence_info





def sensitivity_analysis(args, out_dirs):

    parsed_args = parse_args(args)

    sbml_doc, sbml_model = model_preparation(parsed_args)

    # Creation of the problem specifications for the sensitivity analysis
    problem_specs = sens_ut.get_problem_parameters(
        sbml_model=sbml_model,
        n_input_species=len(parsed_args['input_species_ids']),
        input_species_ids=parsed_args['input_species_ids'],
        perturbation_range=parsed_args['perturbation_range'],
        log_file=parsed_args['log_file']
    )

    ut.print_log(parsed_args["log_file"], f"{problem_specs}")

    # Getting all the species
    all_species = [s.getId() for s in sbml_species.get_list_of_species(sbml_model)]
    input_species = parsed_args['input_species_ids']

    internal_nodes = set(all_species) - set(input_species)

    if parsed_args['target_species'] is None:
        target_species = list(internal_nodes)
    else:
        target_species = parsed_args['target_species']
        for species_id in target_species:
            if species_id not in internal_nodes:
                raise ex.SpeciesNotFoundError(f"Target species '{species_id}' not found among internal nodes.")
            
    rr = sim_ut.load_roadrunner_model(sbml_model, log_file=parsed_args['log_file'])
            
    valid_idxs = {f"[{sp}]": rr.timeCourseSelections.index(f'[{sp}]') for sp in target_species}
            
    # Run the sensitivity analysis
    convergence_results, convergence_info = run_convergence_analysis(
        rr,
        input_species,
        problem_specs,
        target_species,
        valid_idxs,
        out_dirs,
        parsed_args['log_file']
    )

    valid_elements = list(valid_idxs.keys())

    converged_Ns = [info['converged_at'] for info in convergence_info.values() if info['converged_at'] is not None]
    if len(converged_Ns) == len(valid_elements):
        optimal_N = max(converged_Ns)
        print(f"\n>> All nodes converged. Optimal N found: {optimal_N}")
    else:
        optimal_N = max(N_VALUES)
        print(f"\n>> Not all nodes converged. Defaulting to maximum N: {optimal_N}")

    # Prepare the knocks

    # TODO: Create the knocked models 
    #   TODO: Create the new values for the knockin operation
    
     

    


