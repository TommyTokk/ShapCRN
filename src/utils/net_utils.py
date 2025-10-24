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
def get_network_from_sbml(sbml_model, log_file=None):
    # TODO: Add code to include modifiers in the graph
    """
    Create a directed graph from the SBML model using the class structure.

    Args:
       sbml_model: The SBML model to use to generate the network

    Returns:
        nx.DiGraph: A directed graph representing the reaction network
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
    Yield all simple paths in G that start at `target` and reach any other node in the network.
    Explores all reachable nodes from the target.

    Parameters:
      G: directed graph
      target: source node (starting point)
      cutoff: optional depth cutoff (max path length)

    Yields:
      Lists representing simple paths from target to reachable nodes
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


def network_to_sbml(net, rr_model, sbml):
    # #TODO: Creare il metodo per convertire la rete in un modello SBML per la simulazione
    pass


def inhibit_reaction_from_network(
    network, target_reaction_id, sbml_model, log_file=None
):
    # #TODO: Test if the model changes are correct
    """
    Removes the target reaction node and all its edges, and updates the SBML model

    Args:
        network: The networkx network to modify
        target_reaction_id: The id of the target reaction to remove
        sbml_model: SBML model to update

    Returns:
        tuple: (modified network, updated SBML model)
    """

    in_edges = list(network.in_edges(target_reaction_id))
    out_edges = list(network.out_edges(target_reaction_id))

    # Removing in_edges of the reaction
    network.remove_edges_from(in_edges)

    # removing out_edges of the reaction
    network.remove_edges_from(out_edges)

    new_model = sbml_utils.inhibit_reaction(sbml_model, target_reaction_id, log_file)

    return network, new_model


def inhibit_species_from_network(network, target_species_id, sbml_model, log_file=None):
    # #TODO: Test if the model changes are correct
    """
    Removes all the input edges to the specified species node and updates the SBML model

    Args:
        network: The networkx network to modify
        target_species_id: The id of the target species to isolate
        sbml_model: SBML model to update

    Returns:
        tuple: (modified network, updated SBML model)
    """

    # Get the input edges of the species node
    in_edges = list(network.in_edges(target_species_id))

    # Remove all the input edges
    network.remove_edges_from(in_edges)

    # Set the concentration to 0.0 in the node
    network[target_species_id].set_initial_concentration(0.0)

    # Update the SBML model
    new_model = sbml_utils.inhibit_species(sbml_model, target_species_id, log_file)

    return network, sbml_model


def plot_network(
    graph,
    img_dir_path="./imgs/PetriNets",
    save_dot_dir="./dots",
    img_name="network",
    log_file=None,
    engine="dot",
    orientation="TB",
    ranksep="0.5",
    nodesep="1",
    edge_label_distance=2.0,
):
    """
    Plot the reaction network using Petri Net notation with Graphviz layout.
    - Species = circles
    - Reactions = rectangles
    - Node size adapts automatically to label length
    - Supports tuning for orientation and spacing

    Args:
        graph: A networkx DiGraph with node attribute "type" in {"species", "reaction"}
        img_dir_path: Directory where to save the image (default: "./imgs")
        img_name: Name of the image file without extension (default: "network")
        log_file: Optional log file handler
        engine: Graphviz layout engine ("dot", "neato", "fdp", "sfdp", ...)
        orientation: Graph orientation, "TB" (top-bottom), "LR" (left-right), etc.
        ranksep: Vertical spacing between ranks (default: 0.5)
        nodesep: Horizontal spacing between nodes (default: 0.3)
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


# TODO: Add the color scheme for the target node
def plot_interaction_graph(
    shap_values, input_nodes, sbml_model, target_node, log_file=None
):
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
