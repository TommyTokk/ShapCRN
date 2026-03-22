import os
import pandas as pd

from src.utils import utils as ut
from src.utils.sbml import io as sbml_io
from src.utils.sbml import reactions as sbml_react
from src.utils.sbml import knock as sbml_knock
import src.utils.simulation as sim_ut

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

def model_preparation(args):
    sbml_doc = sbml_io.load_model(args["model_path"])
    sbml_model = sbml_react.split_all_reversible_reactions(sbml_doc.getModel(), args["log_file"])

    ut.print_log(args["log_file"], f"Model loaded and prepared: {args['model_path']}")

    return sbml_doc, sbml_model

def prepare_model_name(sbml_model, target_reaction=None):
    model_name = sbml_model.getName() if sbml_model.isSetName() else sbml_model.getId()

    complete_name = f"{model_name}_ki_{target_reaction}" if target_reaction is not None else f"{model_name}_edited"

    return complete_name

def create_new_vals(sbml_model, model_reaction, log_file = None):
    
    reactants = [r.getId() for r in model_reaction.getListOfReactants()]
    species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

    # Simulate the model to get the fluxes
    rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

    # Setting the selections
    selections = rr.timeCourseSelections
    for s in species_list:
        if f"[{s}]" not in selections:
            selections.append(f"[{s}]")

    rr.timeCourseSelections = selections

    res, _, colnames = sim_ut.simulate(rr, end_time=60, log_file=log_file)

    # Convert to dataframe
    res_df = pd.DataFrame(res, columns=colnames)

    # Retrieve the max values for the reactans
    reactants_new_vals = [res_df[f"[{r}]"].max() for r in reactants]

    return reactants_new_vals


def knockin_reaction(args, out_dirs):
    parsed_args = parse_args(args)

    sbml_doc, sbml_model = model_preparation(parsed_args)

    if sbml_model.getReaction(parsed_args["target_reaction"]) is None:
        ut.print_log(parsed_args["log_file"], f"[WARNING]Target reaction '{parsed_args['target_reaction']}' not found in the model. \nNo changes will be made.")
        return sbml_model

    complete_name = prepare_model_name(sbml_model, parsed_args["target_reaction"])

    new_vals = create_new_vals(sbml_model, sbml_model.getReaction(parsed_args["target_reaction"]), log_file=parsed_args["log_file"])

    # Apply the knockin
    modified_model = sbml_knock.knockin_reaction(
        sbml_model,
        parsed_args["target_reaction"],
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
