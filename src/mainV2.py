#!/usr/bin/env python

import os
import roadrunner
import libsbml
import numpy as np
import pandas as pd

from src import exceptions as ex

from src.utils.sbml import utils as sbml_ut
from src.utils.sbml import io as sbml_io
from src.utils import simulation as sim_ut
from src.utils import plot as plt_ut
from src.utils.sbml import reactions as sbml_reactions
from src.utils import utils as ut

from src.pipelines import importance as imp
from src.pipelines.knockout import knockout_species as ko_species
from src.pipelines.knockout import knockout_reaction as ko_reaction
from src.pipelines.knockin import knockin_species as ki_species
from src.pipelines.knockin import knockin_reaction as ki_reaction




def main():
    # Parsing the arguments
    args = ut.parse_args()

    # Set the logger
    log_file = args.log if args.log else None

    file_path = os.path.abspath(args.input_path)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ex.FileNotFoundError(f"The file '{file_path}' does not exist.")
    

    # Extract the file name and extension
    file_name = os.path.basename(file_path)
    file_name, extension = os.path.splitext(file_name)

    command = args.command

    # Build standard output directory tree
    out_dirs = ut.setup_output_dirs(args.output, file_name)

    ut.print_log(log_file, out_dirs)

    try:
        if command == 'simulate':
            # Parsing the arguments for simulation
            simulation_time = args.time
            integrator = args.integrator
            use_steady_state = args.steady_state
            ss_max_time = args.max_time
            ss_sim_steps = args.sim_step
            ss_sim_points = args.points
            ss_threshold = args.threshold
            interactive = args.interactive

            # Load the model
            sbml_doc = sbml_io.load_model(file_path)

            # Get the model
            sbml_model = sbml_reactions.split_all_reversible_reactions(sbml_doc.getModel())

            # Check the integrator
            if integrator:
                rr = sim_ut.load_roadrunner_model(
                    sbml_model, integrator=integrator, log_file=log_file
                )
            else:
                rr = sim_ut.load_roadrunner_model(sbml_model, log_file=log_file)

            # Simulate the model
            ut.print_log(log_file, f"[INFO] Simulating model: {file_name}")
            res, ss_time, colnames = sim_ut.simulate(
                rr_model=rr,
                start_time=0,
                end_time=simulation_time,
                steady_state=use_steady_state,
                max_end_time=ss_max_time,
                sim_step=ss_sim_steps,
                threshold=ss_threshold,
                log_file=log_file,
            )

            res_df = pd.DataFrame(res, columns=colnames)
            # Saving the results
            res_df.to_csv(os.path.join(out_dirs['csv'], f"{file_name}_simulation.csv"), index=False)
            ut.print_log(log_file, f"[INFO] Simulation results saved to: {out_dirs['csv']}")

            # Plotting the results
            if interactive:
                plt_ut.plot_results_interactive(
                    res_df, model_name=file_name,
                    html_dir_path=out_dirs['images'], log_file=log_file,
                )

            else:
                plt_ut.plot_results(
                    res_df, img_dir_path=out_dirs['images'],
                    img_name=f"{file_name}.png", log_file=log_file,
                )
        elif command == 'importance_assessment':
            # Parsing the arguments for importance assessment 
            imp.importance_assessment(args, out_dirs)
        elif command == "knockout_species":
            ko_species.knockout_species(args, out_dirs)
        elif command == "knockout_reaction":
            ko_reaction.knockout_reaction(args, out_dirs)
        elif command == "knockin_species":
            ki_species.knockin_species(args, out_dirs)
        else:
            ki_reaction.knockin_reaction(args, out_dirs)
    except Exception as e:
        raise ex.ModelNotFoundError(f"Failed to load the model from '{file_path}': {str(e)}") from e



if __name__ == "__main__":
    main()
