import datetime

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        list: [model_path, operation_type (optional), target_id (optional), save_output (optional), log_file(optional)]
        
        operation_type:
            1 - Inhibit a species
            2 - Inhibit a reaction
        
        save_output:
            True/False - Whether to save the output files

        log_file:
            True/False - Whether to save the information prints in a log file
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process SBML models and perform operations such as species or reaction inhibition.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-f', '--file', type=str, required=True, 
                        help='Path to the SBML model file')
    parser.add_argument('-op', '--operation', type=int, choices=[1, 2],
                        help='Operation to perform:\n1 - Inhibit a species\n2 - Inhibit a reaction')
    parser.add_argument('-tn', '--target_id', type=str, required=True,
                        help='ID of the target species or reaction to inhibit')
    parser.add_argument('-so', '--save_output', action='store_true',
                        help='Save modified SBML models (default: False)')
    parser.add_argument('-lf', '--log_file', type=str,
                        help='Save the outputs on the specified log file')
    
    parsed_args = parser.parse_args()
    
    # Convert to the expected return format
    args = [parsed_args.file]
    
    # Add operation_type if provided
    if parsed_args.operation is not None:
        args.append(parsed_args.operation)
        
        # If operation is specified but target_id is not
        if parsed_args.target_id is None:
            parser.error("Error: When using --operation/-op, you must also specify --target_id/-tn")
        
        # Add target_id
        args.append(parsed_args.target_id)
    elif parsed_args.target_id is not None:
        # If target_id is specified but operation is not
        parser.error("Error: When using --target_id/-tn, you must also specify --operation/-op")
    
    # Add save_output flag at the end of the list
    args.append(parsed_args.save_output)
    
    # Add log_file to the list (may be None if not provided)
    args.append(parsed_args.log_file)
    
    return args

def print_log(log_file, string):
    current_date = datetime.datetime.now()
    if log_file:
        with open(log_file, 'a') as out:
            out.write(f"[{current_date}]: {string}\n")
    else:
        print(f"[{current_date}]: {string}")

