import datetime
import numpy as np
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
    # Aggiunta delle opzioni per steady state
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
        help="Threshold for steady state detection on fluxes (default: 1e-6)",
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
        # required=True,
        help="One or more  IDs to check",
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
    simulate_samples_parser.add_argument(
        "-o",
        "--output",
        default="./imgs/samples",
        help="Output directory for plots (default: ./imgs/samples)",
    )

    # === INHIBIT_SPECIES command ===
    inhibit_species_parser = subparsers.add_parser(
        "inhibit_species", help="Inhibit a species in the model"
    )
    inhibit_species_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    inhibit_species_parser.add_argument(
        "species_id", help="ID of the species to inhibit"
    )
    inhibit_species_parser.add_argument(
        "-o",
        "--output",
        default="./imgs",
        help="Output directory for plots (default: ./imgs)",
    )

    # === INHIBIT_REACTION command ===
    inhibit_reaction_parser = subparsers.add_parser(
        "inhibit_reaction", help="Inhibit a reaction in the model"
    )
    inhibit_reaction_parser.add_argument(
        "input_path", help="Path to the SBML model file"
    )
    inhibit_reaction_parser.add_argument(
        "reaction_id", help="ID of the reaction to inhibit"
    )
    inhibit_reaction_parser.add_argument(
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
        inhibit_species_parser,
        inhibit_reaction_parser,
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


# === DEBUG ====
