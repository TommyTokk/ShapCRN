"""Command-line interface for ShapCRN."""

from __future__ import annotations

import sys
from pathlib import Path

from shapcrn import api, exceptions
from shapcrn.pipelines import network as network_pipeline
from shapcrn.utils import utils as common_utils


def _edited_model_path(args, operation: str, target: str) -> Path:
    source = Path(args.input_path)
    return Path(args.model_dir) / f"{source.stem}_{operation}_{target}{source.suffix}"


def _dispatch(args) -> None:
    log_file = args.log if args.log else None
    command = args.command

    if command == "simulate":
        api.simulate_model(
            args.input_path,
            end_time=args.time,
            integrator=args.integrator or "cvode",
            steady_state=args.steady_state,
            max_time=args.max_time,
            step=args.sim_step,
            points=args.points,
            threshold=args.threshold,
            output_dir=args.output,
            interactive=args.interactive,
            log_file=log_file,
        )
    elif command == "importance_assessment":
        fixed = args.fixed_perturbations if args.use_fixed_perturbations else None
        api.assess_importance(
            args.input_path,
            operation=args.operation,
            input_species=args.input_species,
            knocked_species=args.knocked,
            target_nodes=args.target_nodes,
            preserve_inputs=args.preserve_inputs,
            use_perturbations=args.use_perturbations,
            fixed_perturbations=fixed,
            num_samples=args.num_samples,
            variation=args.variation,
            max_combinations=args.max_combinations,
            payoff=args.payoff_function,
            end_time=args.time,
            integrator=args.integrator or "cvode",
            steady_state=args.steady_state,
            max_time=args.max_time,
            step=args.sim_step,
            points=args.points,
            threshold=args.threshold,
            n_jobs=args.n_jobs,
            seed=args.seed,
            output_dir=args.output,
            log_file=log_file,
        )
    elif command == "sensitivity_analysis":
        api.analyze_sensitivity(
            args.input_path,
            input_species=args.input_species or [],
            target_species=args.target_species,
            base_samples=args.base_samples,
            perturbation_range=args.perturbation_range,
            check_convergence=args.check_convergence,
            fixed_perturbations=args.fixed_perturbations,
            seed=args.seed,
            n_jobs=args.n_jobs,
            output_dir=args.output,
            log_file=log_file,
        )
    elif command == "knockout_species":
        api.knockout_species(
            args.input_path,
            args.species_id,
            output_path=_edited_model_path(args, "ko", args.species_id),
            log_file=log_file,
        )
    elif command == "knockout_reaction":
        api.knockout_reaction(
            args.input_path,
            args.reaction_id,
            output_path=_edited_model_path(args, "ko", args.reaction_id),
            log_file=log_file,
        )
    elif command == "knockin_species":
        api.knockin_species(
            args.input_path,
            args.target_species_id,
            output_path=_edited_model_path(args, "ki", args.target_species_id),
            log_file=log_file,
        )
    elif command == "knockin_reaction":
        api.knockin_reaction(
            args.input_path,
            args.target_reaction_id,
            output_path=_edited_model_path(args, "ki", args.target_reaction_id),
            log_file=log_file,
        )
    elif command == "create_network":
        path = Path(args.input_path)
        if not path.is_file():
            raise FileNotFoundError(f"SBML model does not exist: {path}")
        out_dirs = common_utils.setup_output_dirs(args.output, path.stem)
        network_pipeline.create_model_network(args, out_dirs)
    else:
        raise exceptions.InvalidCommandError(command)


def main(argv=None) -> int:
    """Run the CLI and return a process exit code."""
    args = common_utils.parse_args(argv)
    try:
        _dispatch(args)
    except (exceptions.KOShapleyError, OSError, ValueError) as error:
        print(f"shapcrn: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
