"""Compatibility adapter for the sensitivity-analysis pipeline."""

from __future__ import annotations


def parse_args(args) -> dict:
    """Convert the historical argparse namespace to public API arguments."""
    return {
        "model_path": args.input_path,
        "input_species": args.input_species or [],
        "base_samples": args.base_samples,
        "perturbation_range": args.perturbation_range,
        "target_species": args.target_species,
        "fixed_perturbations": args.fixed_perturbations,
        "check_convergence": args.check_convergence,
        "seed": getattr(args, "seed", None),
        "n_jobs": getattr(args, "n_jobs", None),
        "output_dir": args.output,
        "log_file": args.log if args.log else None,
    }


def sensitivity_analysis(args, out_dirs=None):
    """Run sensitivity analysis through :func:`shapcrn.api.analyze_sensitivity`."""
    from shapcrn.api import analyze_sensitivity

    parsed = parse_args(args)
    return analyze_sensitivity(**parsed)
