from shapcrn.utils import graph as graph_ut
from shapcrn.utils.sbml import io as sbml_io

def parse_args(args):
    """
    Parses the command-line arguments for the network pipeline.

    Parameters:
    - args: The command-line arguments to parse.

    Returns:
    - A namespace containing the parsed arguments.
    """

    input_path = args.input_path
    output_dir = args.output
    orientation = args.orientation
    layout = args.layout
    vertical_spacing = args.vertical_spacing
    horizontal_spacing = args.horizontal_spacing

    parsed_args = {
        "input_path": input_path,
        "output_dir": output_dir,
        "orientation": orientation,
        "layout": layout,
        "vertical_spacing": vertical_spacing,
        "horizontal_spacing": horizontal_spacing,
    }

    return parsed_args




def create_model_netwrok(args, out_dirs):
    """
    Creates a network representation of the model and saves it as an image.

    Parameters:
    - args: The command-line arguments containing the model file path and other options.
    - out_dirs: A dictionary containing the output directories for saving results.
    """

    parsed_args = parse_args(args)

    # Create the SBML model
    sbml_doc = sbml_io.load_model(parsed_args["input_path"])
    sbml_model = sbml_doc.getModel()

    # Get the network
    N = graph_ut.get_network_from_sbml(sbml_model)

    # Prefer the pipeline-managed images directory when available.
    image_output_dir = out_dirs.get("images", parsed_args["output_dir"])
    dot_output_dir = out_dirs.get("dot")

    print(f"Saving network image to: {image_output_dir}")

    # Plot the network
    graph_ut.plot_network(
        N,
        engine=parsed_args["layout"],
        orientation=parsed_args["orientation"],
        ranksep=parsed_args["vertical_spacing"],
        nodesep=parsed_args["horizontal_spacing"],
        img_dir_path=image_output_dir,
        img_name = "network",
        save_dot_dir=dot_output_dir,
    )


def create_model_network(args, out_dirs):
    """Backward-compatible alias with corrected function name."""
    return create_model_netwrok(args, out_dirs)
