import os

from shapcrn.utils import utils as ut
from shapcrn.utils.sbml import io as sbml_io
from shapcrn.utils.sbml import knock as sbml_knock
import shapcrn.utils.simulation as sim_ut

def parse_args(args):
    
    parsed_args = {
        "model_path": args.input_path,
        "target_species": args.target_species_id,
        "model_dir": args.model_dir,
        "log_file": args.log
    }

    return parsed_args

def knockin_species(args, out_dirs):
    
    parsed_args = parse_args(args)
    _, sbml_model = sbml_io.load_and_prepare_model(
        parsed_args["model_path"], log_file=parsed_args["log_file"]
    )
    ut.print_log(parsed_args["log_file"], f"Model loaded and prepared: {parsed_args['model_path']}")

    if parsed_args["target_species"] is None:
        ut.print_log(parsed_args["log_file"], f"Target species ({parsed_args['target_species']}) not found in the model.\n No changes will be made to the model.")
        return sbml_model
    
    model_species = sbml_model.getSpecies(parsed_args["target_species"]).getId()

    species_new_val = sim_ut.get_species_peak_value(
        sbml_model, model_species, sim_end_time=60, log_file=parsed_args["log_file"]
    )

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
