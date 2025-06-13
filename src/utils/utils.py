import datetime

import json


def parse_args():
    """
    Parse command line arguments using a subcommand structure.

    Returns:
        argparse.Namespace: Object containing all parsed arguments
    """
    import argparse

    # Create main parser
    parser = argparse.ArgumentParser(
        description="SBML model analysis and manipulation tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # === SIMULATE command ===
    simulate_parser = subparsers.add_parser("simulate", help="Simulate an SBML model")
    simulate_parser.add_argument("input_path", help="Path to the SBML model file")
    simulate_parser.add_argument(
        "-t", "--time", type=float, default=10, help="Simulation end time (default: 10)"
    )
    simulate_parser.add_argument(
        "-i",
        "--integrator",
        choices=["cvode", "gillespie", "rk4"],
        help="Integrator to use",
    )
    simulate_parser.add_argument(
        "-o",
        "--output",
        default="./imgs",
        help="Output directory for plots (default: ./imgs)",
    )
    # Steady state options for simulate
    simulate_parser.add_argument(
        "--steady-state",
        action="store_true",
        help="Simulate until steady state is reached",
    )
    simulate_parser.add_argument(
        "--max-time",
        type=float,
        default=1000,
        help="Maximum simulation time when seeking steady state (default: 1000)",
    )
    simulate_parser.add_argument(
        "--sim-step",
        type=float,
        default=5,
        help="Time step for steady state check (default: 5)",
    )
    simulate_parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of points in the final profile (default: 1000)",
    )
    simulate_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for steady state detection (default: 1e-6)",
    )

    # === SIMULATE_SAMPLES command ===
    simulate_samples_parser = subparsers.add_parser(
        "simulate_samples",
        help="Simulate model with different input species concentrations",
    )
    simulate_samples_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    simulate_samples_parser.add_argument(
        "-is",
        "--input_species",
        nargs="+",
        required=True,
        help="One or more species IDs to vary (e.g. -s ACEx GLCx P)",
    )
    simulate_samples_parser.add_argument(
        "-tids",
        "--target_ids",
        nargs="+",
        default=None,
        help="One or more IDs to check",
    )
    simulate_samples_parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples for each species (default: 5)",
    )
    simulate_samples_parser.add_argument(
        "-v",
        "--variation",
        type=float,
        default=20.0,
        help="Percentage variation around initial value (default: 20.0)",
    )
    simulate_samples_parser.add_argument(
        "-t", "--time", type=float, default=10, help="Simulation end time (default: 10)"
    )
    simulate_samples_parser.add_argument(
        "-i",
        "--integrator",
        choices=["cvode", "gillespie", "rk4"],
        help="Integrator to use",
    )
    # Steady state options for simulate_samples
    simulate_samples_parser.add_argument(
        "--steady-state",
        action="store_true",
        help="Simulate until steady state is reached",
    )
    simulate_samples_parser.add_argument(
        "--max-time",
        type=float,
        default=1000,
        help="Maximum simulation time when seeking steady state (default: 1000)",
    )
    simulate_samples_parser.add_argument(
        "--sim-step",
        type=float,
        default=5,
        help="Time step for steady state check (default: 5)",
    )
    simulate_samples_parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of points in the final profile (default: 1000)",
    )
    simulate_samples_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for steady state detection (default: 1e-6)",
    )

    simulate_samples_parser.add_argument(
        "-o",
        "--output",
        default="./imgs/samples",
        help="Output directory for plots (default: ./imgs/samples)",
    )

    # === KNOCKOUT_SPECIES command ===
    knockout_species_parser = subparsers.add_parser(
        "knockout_species", help="Knockout a species in the model"
    )
    knockout_species_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    knockout_species_parser.add_argument(
        "species_id", help="ID of the species to inhibit"
    )
    knockout_species_parser.add_argument(
        "-o",
        "--output",
        default="./imgs",
        help="Output directory for plots (default: ./imgs)",
    )

    # === KNOCKOUT_REACTION command ===
    knockout_reaction_parser = subparsers.add_parser(
        "knockout_reaction", help="Knockout a reaction in the model"
    )
    knockout_reaction_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    knockout_reaction_parser.add_argument(
        "reaction_id", help="ID of the reaction to inhibit"
    )
    knockout_reaction_parser.add_argument(
        "-o",
        "--output",
        default="./imgs",
        help="Output directory for plots (default: ./imgs)",
    )

    # === CREATE PETRINET command ==
    create_petrinet_parser = subparsers.add_parser(
        "create_petrinet", help="Create the PetriNet of the given model"
    )

    create_petrinet_parser.add_argument("input_path", help="Path to the SBML model")

    create_petrinet_parser.add_argument(
        "-o",
        "--output",
        default="./imgs/PetriNets",
        help="Output directory for plots (default: ./imgs/PetriNets)",
    )

    create_petrinet_parser.add_argument(
        "-tn", "--tests_number", default=10, help="Number of tests to perform"
    )

    create_petrinet_parser.add_argument("-iq", "--increase_quantity", default=5)

    # Common arguments for all commands
    for subparser in [
        simulate_parser,
        simulate_samples_parser,
        knockout_species_parser,
        knockout_reaction_parser,
    ]:
        subparser.add_argument("-l", "--log", help="Path to log file")

    return parser.parse_args()


def print_log(log_file, string):
    current_date = datetime.datetime.now()
    if log_file:
        with open(log_file, "a") as out:
            out.write(f"[{current_date}]: {string}\n")
    else:
        print(f"[{current_date}]: {string}")


def dict_pretty_print(dict_obj):
    """Pretty print a dictionary as formatted JSON.

    Args:
        dict_obj: Dictionary to be printed
    """

    json_formatted_str = json.dumps(dict_obj, indent=2)
    print(json_formatted_str)


def pretty_print_variations(variations_dict, precision=4, show_zero=False):
    """
    Pretty print the variations dictionary in a readable format.

    Args:
        variations_dict: Dict returned by get_simulations_variations
        precision: Number of decimal places to show (default: 4)
        show_zero: Whether to show variations that are zero (default: False)
    """
    if not variations_dict:
        print("No variations to display.")
        return

    print("=" * 80)
    print("SIMULATION VARIATIONS REPORT")
    print("=" * 80)

    for ko_species in sorted(variations_dict.keys()):
        print(f"\n🧬 KNOCKED OUT SPECIES: {ko_species}")
        print("-" * 60)

        combinations = variations_dict[ko_species]
        if not combinations:
            print("  No combinations found.")
            continue

        for combination in sorted(combinations.keys()):
            print(f"\n  📊 Combination: {combination}")
            print("  " + "─" * 50)

            species_variations = combinations[combination]
            if not species_variations:
                print("    No species variations found.")
                continue

            # Sort species by absolute variation value (largest first)
            sorted_species = sorted(
                species_variations.items(),
                key=lambda x: abs(x[1]["variation"]),
                reverse=True,
            )

            displayed_any = False
            for species, data in sorted_species:
                variation = data["variation"]

                # Skip zero variations if show_zero is False
                if not show_zero and abs(variation) < 1e-10:
                    continue

                # Format the variation with appropriate sign and color indicators
                if variation > 0:
                    sign_indicator = "↑"
                    variation_str = f"+{variation:.{precision}f}"
                elif variation < 0:
                    sign_indicator = "↓"
                    variation_str = f"{variation:.{precision}f}"
                else:
                    sign_indicator = "="
                    variation_str = f"{variation:.{precision}f}"

                print(f"    {sign_indicator} {species:<20} {variation_str:>12}")
                displayed_any = True

            if not displayed_any:
                print("    (All variations are zero)")

    print("\n" + "=" * 80)
    print("Legend: ↑ = Increase, ↓ = Decrease, = = No change")
    print("=" * 80)


# === DEBUG ===

# === NORMALIZATION ===


def normalize_variations_globally(heatmap_data, log_file=None):

    global_min = heatmap_data.min()
    global_max = heatmap_data.max()

    if global_max - global_min > 1e-10:
        heatmap_data = (heatmap_data - global_min) / (global_max - global_min)

    return heatmap_data
