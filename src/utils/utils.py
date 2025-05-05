import datetime

def parse_args():
    """
    Parse command line arguments using a subcommand structure.
    
    Returns:
        argparse.Namespace: Object containing all parsed arguments
    """
    import argparse
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description='SBML model analysis and manipulation tool',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True
    
    # === SIMULATE command ===
    simulate_parser = subparsers.add_parser('simulate', help='Simulate an SBML model')
    simulate_parser.add_argument('input_path', help='Path to the SBML model file')
    simulate_parser.add_argument('-t', '--time', type=float, default=10, 
                                help='Simulation end time (default: 10)')
    simulate_parser.add_argument('-i', '--integrator', choices=['cvode', 'gillespie', 'rk4'], 
                                help='Integrator to use')
    simulate_parser.add_argument('-o', '--output', default='./imgs',
                                help='Output directory for plots (default: ./imgs)')
    
    # === INHIBIT_SPECIES command ===
    inhibit_species_parser = subparsers.add_parser('inhibit_species', 
                                                  help='Inhibit a species in the model')
    inhibit_species_parser.add_argument('input_path', help='Path to the SBML model file')
    inhibit_species_parser.add_argument('species_id', help='ID of the species to inhibit')
    inhibit_species_parser.add_argument('-o', '--output', default='./imgs',
                                       help='Output directory for plots (default: ./imgs)')
    
    # === INHIBIT_REACTION command ===
    inhibit_reaction_parser = subparsers.add_parser('inhibit_reaction', 
                                                   help='Inhibit a reaction in the model')
    inhibit_reaction_parser.add_argument('input_path', help='Path to the SBML model file')
    inhibit_reaction_parser.add_argument('reaction_id', help='ID of the reaction to inhibit')
    inhibit_reaction_parser.add_argument('-o', '--output', default='./imgs',
                                       help='Output directory for plots (default: ./imgs)')
    
    # Common arguments for all commands
    for subparser in [simulate_parser, inhibit_species_parser, inhibit_reaction_parser]:
        subparser.add_argument('-l', '--log', help='Path to log file')
    
    return parser.parse_args()

def print_log(log_file, string):
    current_date = datetime.datetime.now()
    if log_file:
        with open(log_file, 'a') as out:
            out.write(f"[{current_date}]: {string}\n")
    else:
        print(f"[{current_date}]: {string}")

