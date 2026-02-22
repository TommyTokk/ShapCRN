from src.utils import utils as ut



def parse_args(args):
    """
    Parses the command line arguments for the importance assessment pipeline.
    """

    # Model path
    input_path = args.input_path

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


    


def importance_assessment(args):
    """
    Pipeline for importance assessment of reactions in a CRN.
    """

    # Arguent parsing
    parsed_args = parse_args(args)

    log_file = parsed_args["log_file"]
