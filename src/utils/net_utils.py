from itertools import product
from operator import indexOf
import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from networkx.algorithms.bipartite import color
import numpy as np
from pandas.core.groupby.generic import ScalarResult
from scipy.special import eval_sh_legendre

from src.utils import sbml_utils
from src.utils.utils import print_log


# ============
# NETWORK
# ============
# KEEP
def get_network_from_sbml(sbml_model, log_file=None):
    """
    Create a directed graph from the SBML model.

    Constructs a NetworkX directed graph where species and reactions are represented
    as nodes, with edges indicating reactant-to-reaction and reaction-to-product
    relationships. Edge weights represent stoichiometric coefficients.

    Parameters
    ----------
    sbml_model : libsbml.Model
        The SBML model to convert into a network representation.
    log_file : str or None, optional
        Path to log file for messages, by default None

    Returns
    -------
    nx.DiGraph
        A directed graph representing the reaction network with nodes for both
        species and reactions, and weighted edges for stoichiometry.
    """
    DG = nx.DiGraph()

    # Creates the species dict for nodes
    for s in sbml_model.getListOfSpecies():
        DG.add_node(s.getId(), species=s, label=s.getId(), type="species")

    # Creates the reaction dict for nodes
    for r in sbml_model.getListOfReactions():
        DG.add_node(r.getId(), reaction=r, label=r.getId(), type="reaction")

    weight_list = []
    # Add reaction nodes and edges
    for reaction in sbml_model.getListOfReactions():
        for reagent in reaction.getListOfReactants():
            DG.add_edge(
                reagent.getSpecies(),
                reaction.getId(),
                weight=reagent.getStoichiometry(),
                label=str(reagent.getStoichiometry()),  # <-- label per Graphviz
            )

        for modifier in reaction.getListOfModifiers():
            DG.add_edge(modifier.getSpecies(), reaction.getId(), weight=None, label="")

        for product in reaction.getListOfProducts():
            DG.add_edge(
                reaction.getId(),
                sbml_model.getSpecies(product.getSpecies()).getId(),
                weight=product.getStoichiometry(),
                label=str(product.getStoichiometry()),  # <-- label per Graphviz
            )

    # Add all weighted edges to the graph
    DG.add_weighted_edges_from(weight_list)

    return DG


# TODO: Test this version
def all_simple_paths_from_target(G: nx.DiGraph, target, cutoff=None):
    """
    Yield all simple paths starting at target and reaching any other node.

    Explores all reachable nodes from the target node and yields simple paths
    connecting the target to each reachable node in the network.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph to search for paths.
    target : node
        Starting node for all paths.
    cutoff : int, optional
        Maximum path length (depth limit), by default None

    Yields
    ------
    list
        Simple paths from target to reachable nodes, where each path is
        represented as a list of nodes.
    """
    if target not in G:
        return

    # Find all nodes reachable from target
    for node in G.nodes():
        if node == target:
            yield [target]
        else:
            # Find all simple paths from target to this node
            for path in nx.all_simple_paths(
                G, source=node, target=target, cutoff=cutoff
            ):
                yield path



# KEEP
def plot_network(
    graph: nx.DiGraph,
    img_dir_path:str="./imgs/PetriNets",
    save_dot_dir:str="./dots",
    img_name:str="network",
    log_file=None,
    engine:str="dot",
    orientation:str="TB",
    ranksep:str="0.5",
    nodesep:str="1",
    edge_label_distance:float=2.0,
):
    """
    Plot the reaction network using Petri Net notation with Graphviz layout.

    Creates a visualization where species are represented as circles and reactions
    as rectangles. Node sizes adapt automatically to label length, and the layout
    can be customized for orientation and spacing.

    Parameters
    ----------
    graph : nx.DiGraph
        NetworkX directed graph with node attribute "type" in {"species", "reaction"}.
    img_dir_path : str, optional
        Directory where to save the image, by default "./imgs/PetriNets"
    save_dot_dir : str, optional
        Directory where to save the DOT file, by default "./dots"
    img_name : str, optional
        Name of the image file without extension, by default "network"
    log_file : str or None, optional
        Path to log file for messages, by default None
    engine : str, optional
        Graphviz layout engine ("dot", "neato", "fdp", "sfdp"), by default "dot"
    orientation : str, optional
        Graph orientation ("TB" for top-bottom, "LR" for left-right, etc.), 
        by default "TB"
    ranksep : str, optional
        Vertical spacing between ranks, by default "0.5"
    nodesep : str, optional
        Horizontal spacing between nodes, by default "1"
    edge_label_distance : float, optional
        Distance of edge labels from edges, by default 2.0

    Returns
    -------
    None
        Saves the network plot as PNG and DOT files.
    """
    if img_dir_path is None:
        img_dir_path = "./imgs/PetriNets"

    if save_dot_dir is None:
        save_dot_dir = "./dots"

    if img_name is None:
        img_name = "network"

    # Ensure output directory exists
    os.makedirs(img_dir_path, exist_ok=True)
    os.makedirs(save_dot_dir, exist_ok=True)

    # Always save as PNG
    png_path = os.path.join(img_dir_path, f"{img_name}.png")
    dot_path = os.path.join(save_dot_dir, f"{img_name}.gv")
    try:
        # Convert to Graphviz AGraph
        A = nx.nx_agraph.to_agraph(graph)

        # Default node style
        A.node_attr.update(
            fontsize="12",
            fontname="Helvetica",
            shape="ellipse",  # overridden per node type
            style="filled",
            fillcolor="white",
            fixedsize="false",  # adaptive sizing
            width="0",
            height="0",
        )

        # Customize nodes based on type
        for n in graph.nodes():
            ntype = graph.nodes[n].get("type", "species")
            if ntype == "reaction":
                A.get_node(n).attr.update(shape="box", fillcolor="#ffcc99")
            else:  # species
                A.get_node(n).attr.update(shape="ellipse", fillcolor="#99ccff")

        # Customize edges to display labels if present
        for u, v, data in graph.edges(data=True):
            label = data.get("label", "")
            if label:
                A.get_edge(u, v).attr.update(
                    label=str(label),
                    fontsize="10",
                    fontcolor="black",
                    labeldistance=str(edge_label_distance),
                    labelangle="0",
                )

        # Graph-level layout tuning
        A.graph_attr.update(
            rankdir=orientation,  # "TB" or "LR"
            ranksep=str(ranksep),
            nodesep=str(nodesep),
            splines="true",
            size="30,30!",
        )

        # Apply layout and render as PNG
        A.layout(engine)
        A.draw(png_path, format="png")

        print_log(log_file, f"[INFO] Network plot saved at {png_path}")
        # Saving the dot file
        with open(dot_path, "w") as f:
            f.write(A.to_string())

        print_log(log_file, f"[INFO] Dot file saved at {dot_path}")

    except Exception as e:
        err_msg = f"[ERROR] Failed to plot network: {e}\n"
        if log_file:
            log_file.write(err_msg)
        else:
            print(err_msg)



# KEEP
def plot_interaction_graph(
    shap_values, input_nodes, sbml_model, target_node, log_file=None
):
    """
    Plot interaction graph with Shapley values colored by influence.

    Creates a network visualization where nodes are colored based on their Shapley
    value influence on the target node: green for positive influence, red for
    negative influence, and purple for the target node itself.

    Parameters
    ----------
    shap_values : pd.DataFrame
        DataFrame containing Shapley values with species as index and columns.
    input_nodes : list
        List of input node IDs.
    sbml_model : libsbml.Model
        SBML model to extract network structure.
    target_node : str
        ID of the target node to analyze influences for.
    log_file : str or None, optional
        Path to log file for messages, by default None

    Returns
    -------
    None
        Saves the interaction graph as a PNG file at ./imgs/test.png
    """
    all_species = shap_values.columns.str.strip("[]")

    N = get_network_from_sbml(sbml_model, log_file)
    GN = nx.nx_agraph.to_agraph(N)

    # Customize nodes based on type
    for n in N.nodes():
        ntype = N.nodes[n].get("type", "species")
        if ntype == "reaction":
            GN.get_node(n).attr.update(shape="box", style="filled", fillcolor="#ffcc99")
        else:  # species
            GN.get_node(n).attr.update(
                shape="ellipse", style="filled", fillcolor="#99ccff"
            )

    GN.get_node(target_node).attr.update(fillcolor="#a98aff")

    # Customize edges to display labels if present
    for u, v, data in N.edges(data=True):
        label = data.get("label", "")
        if label:
            GN.get_edge(u, v).attr.update(
                label=str(label),
                fontsize="10",
                fontcolor="black",
                labeldistance=str(2.0),
                labelangle="0",
            )

    stacked = shap_values.stack()
    # Retrieve only reaction nodes
    # reaction_nodes = [n for n, d in N.nodes(data=True) if d.get("type") == "reaction"]

    paths = list(all_simple_paths_from_target(N, target_node, None))
    __import__("pprint").pprint(paths)
    nodes_to_color = set()

    for p in paths:
        for node in p:
            if N.nodes[node]["type"] == "reaction":
                continue
            nodes_to_color.add(node)

    # Creating the colored graph
    for node in list(nodes_to_color):
        if node == target_node:
            continue

        shap = stacked[node][f"[{target_node}]"]
        if shap > 0:
            GN.get_node(node).attr.update(fillcolor="#5FA96D")
        elif shap < 0:
            GN.get_node(node).attr.update(fillcolor="#D9321C")
        else:
            continue

    layout = "dot"
    # if len(all_species) >= 10:
    #     layout = "sfdp"

    GN.layout(layout)
    GN.draw("./imgs/test.png", format="png")
