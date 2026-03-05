import os
import pandas as pd

from src.utils import utils as ut
from src.utils.sbml import io as sbml_io
from src.utils.sbml import reactions as sbml_react
from src.utils.sbml import knock as sbml_knock
import src.utils.simulation as sim_ut

def parse_args(args):
    
    parsed_args = {
        "model_path": args.input_path,
        "target_species": args.target_species_id,
        "model_dir": args.model_dir,
        "log_file": args.log
    }

    return parsed_args

def model_preparation(args):
    sbml_doc = sbml_io.load_model(args["model_path"])
    sbml_model = sbml_react.split_all_reversible_reactions(sbml_doc.getModel(), args["log_file"])
    ut.print_log(args["log_file"], f"Model loaded and prepared: {args['model_path']}")
    return sbml_doc, sbml_model

def prepare_model_name(sbml_model, target_species=None):
    model_name = sbml_model.getName() if sbml_model.isSetName() else sbml_model.getId()
    complete_name = f"{model_name}_ki_{target_species}" if target_species is not None else f"{model_name}_edited"
    return complete_name

def create_new_vals(sbml_model, model_species, log_file = None):
    
    species_list = [s.getId() for s in sbml_model.getListOfSpecies()]
    params = []

    rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)
    # Setting the selections
    selections = rr.timeCourseSelections
    for s in species_list:
        if f"[{s}]" not in selections:
            selections.append(f"[{s}]")

    rr.timeCourseSelections = selections

    # Simulate to take the maximum  value as auto value
    res, _, colnames = sim_ut.simulate(rr, end_time=60, log_file=log_file)

    res_df = pd.DataFrame(res, columns=colnames)

    # Retrieve the max value of the target species
    new_species_val = res_df[f"[{model_species}]"].max()

    return new_species_val

def knockin_species(args, out_dirs):
    
    parsed_args = parse_args(args)
    sbml_doc, sbml_model = model_preparation(parsed_args)
    model_name = prepare_model_name(sbml_model, parsed_args["target_species"])

    if parsed_args["target_species"] is None:
        ut.print_log(parsed_args["log_file"], f"Target species ({parsed_args['target_species']}) not found in the model.\n No changes will be made to the model.")
        return sbml_model
    
    model_species = sbml_model.getSpecies(parsed_args["target_species"]).getId()

    species_new_val = create_new_vals(sbml_model, model_species, log_file=parsed_args["log_file"])

    modified_model = sbml_knock.knockin_species(
        sbml_model,
        model_species,
        new_val=species_new_val,
        log_file=parsed_args["log_file"]
    )

    # Save the modified model
    # Save the modified model
    model_dir = parsed_args["model_dir"]
    input_file = os.path.basename(parsed_args["model_path"])
    operation_name = f"ki_{parsed_args['target_species']}"

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
