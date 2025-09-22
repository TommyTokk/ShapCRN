from itertools import product
from operator import indexOf
import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from networkx.algorithms.bipartite import color
import numpy as np
from pandas.core.groupby.generic import ScalarResult

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
    nodesep="0.3",
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
            size="25,25!",
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


# TODO: Change graph creation counting also modifiers
def plot_interaction_graph(shap_values, input_nodes, sbml_model, log_file=None):
    all_species = shap_values.columns.str.strip("[]")

    nodes = []
    node_colors = []

    # Creating the nodes list
    for node in all_species:
        if node in input_nodes:
            nodes.append(node)
            node_colors.append("orange")
        else:
            nodes.append(node)
            node_colors.append("lightblue")

    IG = nx.DiGraph()
    IG.add_nodes_from(nodes)

    # Creating the edges
    edges = set()
    for reaction in sbml_model.getListOfReactions():
        reactants = [r.getSpecies() for r in reaction.getListOfReactants()]
        products = [p.getSpecies() for p in reaction.getListOfProducts()]
        modifiers = [m.getSpecies() for m in reaction.getListOfModifiers()]

        for r in reactants:
            for p in products:
                edges.add((r, p))

        for m in modifiers:

            for r in reactants:
                edges.add((m, r))
            for p in products:
                edges.add((m, p))

    IG.add_edges_from(list(edges))

    # Convert to Graphviz
    GG = nx.nx_agraph.to_agraph(IG)
    GG.node_attr.update(
        fontsize="12",
        fontname="Helvetica",
        shape="ellipse",  # overridden per node type
        style="filled",
        fillcolor="white",
        fixedsize="false",  # adaptive sizing
        width="0",
        height="0",
    )

    for node in IG.nodes():

        if node in input_nodes:
            GG.get_node(node).attr.update(shape="ellipse", fillcolor="#ffcc99")
        else:
            GG.get_node(node).attr.update(shape="ellipse", fillcolor="#99ccff")

    # Modifying the edges colors

    stacked = shap_values.stack()

    for edge in IG.edges():
        src = edge[0]
        dst = edge[1]
        shap = stacked[src][f"[{dst}]"]

        if shap < 0:
            GG.get_edge(src, dst).attr.update(color="#5FA96D", penwidth="2.5")
        elif shap > 0:
            GG.get_edge(src, dst).attr.update(color="#D9321C", penwidth="2.5")
        else:
            GG.get_edge(src, dst).attr.update(color="black", penwidth="2.5")

    GG.graph_attr.update(
        rankdir="LR",
        overlap="false",
        ranksep="3.0",
        # nodesep="0.7",
        size="30,30!",
        concentrate="false",
        sep="+15,15",  # Additional separation between components
        esep="+10,10",
        splines="true",
    )
    #
    layout = "dot"
    if len(all_species) >= 10:
        layout = "sfdp"

    GG.layout(layout)
    GG.draw("./imgs/test.png", format="png", args="-Gdpi=300")
