import os

from shapcrn.utils import utils as ut
from shapcrn.utils.sbml import io as sbml_io
from shapcrn.utils.sbml import knock as sbml_knock
import shapcrn.utils.simulation as sim_ut

def parse_args(args):
    
    model_path = args.input_path

    target_reaction = args.target_reaction_id

    model_dir = args.model_dir
    log_file = args.log

    parsed_args = {
        "model_path": model_path,
        "target_reaction": target_reaction,
        "model_dir": model_dir,
        "log_file": log_file
    }
    return parsed_args

def knockin_reaction(args, out_dirs):
    parsed_args = parse_args(args)

    _, sbml_model = sbml_io.load_and_prepare_model(
        parsed_args["model_path"], log_file=parsed_args["log_file"]
    )
    ut.print_log(parsed_args["log_file"], f"Model loaded and prepared: {parsed_args['model_path']}")

    if sbml_model.getReaction(parsed_args["target_reaction"]) is None:
        ut.print_log(parsed_args["log_file"], f"[WARNING]Target reaction '{parsed_args['target_reaction']}' not found in the model. \nNo changes will be made.")
        return sbml_model

    target_reaction = sbml_model.getReaction(parsed_args["target_reaction"])
    new_vals = sim_ut.get_reactants_peak_values(
        sbml_model,
        target_reaction,
        sim_end_time=60,
        log_file=parsed_args["log_file"],
    )

    # Apply the knockin
    modified_model = sbml_knock.knockin_reaction(
        sbml_model,
        target_reaction,
        new_vals,
        parsed_args["log_file"]
    )

    # Save the modified model
    model_dir = parsed_args["model_dir"]
    input_file = os.path.basename(parsed_args["model_path"])
    operation_name = f"ki_{parsed_args['target_reaction']}"

    ut.print_log(parsed_args["log_file"], operation_name)

    # Save the modified model
    sbml_io.save_file(
        input_file,
        operation_name,
        modified_model,
        True,
        save_path=model_dir,
        log_file=parsed_args["log_file"],
    )
