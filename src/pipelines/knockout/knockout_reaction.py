import os

from src.utils import utils as ut
from src.utils.sbml import io as sbml_io
from src.utils.sbml import reactions as sbml_react
from src.utils.sbml import knock as sbml_knock


def parse_args(args):
    
    input_path = args.input_path

    target_reaction = args.reaction_id

    model_dir = args.model_dir

    log_file = args.log_file

    parsed_args = {
        "input_path": input_path,
        "target_reaction": target_reaction,
        "model_dir": model_dir,
        "log_file": log_file
    }
    return parsed_args

def model_preparation(args):
    sbml_doc = sbml_io.load_model(args["input_path"])
    sbml_model = sbml_react.split_all_reversible_reactions(sbml_doc.getModel(), args["log_file"])

    ut.print_log(args["log_file"], f"Model loaded and prepared: {args['input_path']}")

    return sbml_doc, sbml_model

def prepare_model_name(sbml_model, target_reaction=None):
    model_name = sbml_model.getName() if sbml_model.isSetName() else sbml_model.getId()

    complete_name = f"{model_name}_ko_{target_reaction}" if target_reaction is not None else f"{model_name}_edited"

    return complete_name


def knockout_reaction(args, out_dirs):
    parsed_args = parse_args(args)

    sbml_doc, sbml_model = model_preparation(parsed_args)

    # Aply the knockout
    modified_model = sbml_knock.knockout_reaction(sbml_model, parsed_args["target_reaction"], parsed_args["log_file"])

    # Save the modified model
    model_dir = parsed_args["model_dir"]
    input_file = os.path.basename(parsed_args["input_path"])
    operation_name = f"ko_{parsed_args['target_reaction']}"

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


    