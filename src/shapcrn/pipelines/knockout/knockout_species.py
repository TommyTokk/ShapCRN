import os

from shapcrn.utils import utils as ut
from shapcrn.utils.sbml import io as sbml_io
from shapcrn.utils.sbml import knock as sbml_knock


def parse_args(args):
    
    # Model path
    input_path = args.input_path

    # Species to knockout
    target_species = args.species_id

    # Output parameters
    model_dir = args.model_dir
    log_file = args.log

    parsed_args = {
        "input_path": input_path,
        "target_species": target_species,
        "model_dir": model_dir,
        "log_file": log_file
    }

    return parsed_args

def knockout_species(args, out_dirs):
    parsed_args = parse_args(args)

    _, sbml_model = sbml_io.load_and_prepare_model(
        parsed_args["input_path"], log_file=parsed_args["log_file"]
    )
    ut.print_log(parsed_args["log_file"], f"Model loaded and prepared: {parsed_args['input_path']}")

    # Apply the knockout
    modified_model = sbml_knock.knockout_species(sbml_model, parsed_args["target_species"], parsed_args["log_file"])

    # Save the modified model
    model_dir = parsed_args["model_dir"]
    input_file = os.path.basename(parsed_args["input_path"])

    operation_name = f"ko_{parsed_args['target_species']}"

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
