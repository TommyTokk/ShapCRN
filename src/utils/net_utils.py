import networkx as nx
import libsbml
import roadrunner
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils import sbml_utils
from src.utils.utils import print_log

# ============
# NETWORK
# ============

def get_network_from_sbml(list_of_reactions, list_of_species, log_file=None):
    # #TODO: Model also the reversible reactions adding the edges
    """
    Create a directed graph from the SBML model using the class structure.

    Args:
        list_of_reactions: List of Reaction objects
        list_of_species: List of Species objects

    Returns:
        nx.DiGraph: A directed graph representing the reaction network
    """
    DG = nx.DiGraph()

    s_list = []
    r_list = []

    #Creates the species dict for nodes
    for s in list_of_species:
        DG.add_node(s.get_id(), species = s, label = s.get_id(), type = "species")
        # s_dict = s.to_dict()
        # s_dict['label'] = (s.get_id(), s.get_initial_concentration())
        # s_list.append((s.get_id(), s))

    #Creates the reaction dict for nodes
    for r in list_of_reactions:
        DG.add_node(r.get_id(), reaction=r, label = r.get_id(), type = "reaction")

    weight_list = []
    # Add reaction nodes and edges
    for reaction in list_of_reactions:

        # Add edges from reagents to the reaction
        for reagent in reaction.get_reagents():
            weight_list.append((reagent.get_species()['id'], reaction.get_id(), reagent.get_stoichiometry()))
            #DG.add_edge(reagent.get_species()['id'], reaction.get_id(), stoichiometry=reagent.get_stoichiometry())
            
        # Add edges from the reaction to products
        for product in reaction.get_products():
            weight_list.append((reaction.get_id(), product.get_species()['id'], product.get_stoichiometry()))
            #DG.add_edge(reaction.get_id(), product.get_species()['id'], stoichiometry=product.get_stoichiometry())
            
    # Add all weighted edges to the graph
    DG.add_weighted_edges_from(weight_list)

    return DG


def network_to_sbml(net, rr_model, sbml):
    # #TODO: Creare il metodo per convertire la rete in un modello SBML per la simulazione
    pass





def inhibit_reaction_from_network(network, target_reaction_id, sbml_model, log_file=None):
    # #TODO: Test if the model changes are correct
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

    #Removing in_edges of the reaction
    network.remove_edges_from(in_edges)

    #removing out_edges of the reaction
    network.remove_edges_from(out_edges)

    new_model  = sbml_utils.inhibit_reaction(sbml_model, target_reaction_id, log_file)

    return network, new_model




def inhibit_species_from_network(network, target_species_id, sbml_model, log_file=None):
    # #TODO: Test if the model changes are correct
    """
    Removes all the input edges to the specified species node and updates the SBML model
    
    Args:
        network: The networkx network to modify
        target_species_id: The id of the target species to isolate
        sbml_model: SBML model to update
        
    Returns:
        tuple: (modified network, updated SBML model)
    """

    #Get the input edges of the species node
    in_edges = list(network.in_edges(target_species_id))

    #Remove all the input edges
    network.remove_edges_from(in_edges)

    #Set the concentration to 0.0 in the node
    network[target_species_id].set_initial_concentration(0.0)

    #Update the SBML model
    new_model = sbml_utils.inhibit_species(sbml_model, target_species_id, log_file)

    return network, sbml_model


def plot_network(graph, img_dir_path="./imgs", img_name="network", log_file=None):
    """
    Plot the reaction network using Petri Net notation with improved layout.
    
    Args:
        graph: A networkx DiGraph representing the reaction network
        img_dir_path: Directory where to save the image (default: "./imgs")
        img_name: Name of the image file without extension (default: "network")
        log_file: Optional log file handler
    """
    # Gestione dei valori None
    if img_dir_path is None:
        img_dir_path = "./imgs/PetriNets"
    
    if img_name is None:
        img_name = "network"
    
    plt.figure(figsize=(12, 10))
    
    # Create a better layout for the graph to minimize overlaps
    # Using Kamada-Kawai layout for better node distribution
    pos = nx.kamada_kawai_layout(graph)
    
    # Further adjust positions to prevent overlaps
    pos = adjust_positions(pos, spacing=1.0)
    
    # Separate nodes by type
    species_nodes = [node for node, data in graph.nodes(data=True) if data["type"] == "species"]
    reaction_nodes = [node for node, data in graph.nodes(data=True) if data["type"] == "reaction"]
    
    # Draw species as circles (places in Petri Net notation)
    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=species_nodes,
        node_color="skyblue", 
        node_size=700, 
        node_shape='o',  # Circle for species/places
        edgecolors="black", 
        linewidths=2
    )
    
    # Draw reactions as squares (transitions in Petri Net notation)
    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=reaction_nodes,
        node_color="lightcoral", 
        node_size=550, 
        node_shape='s',  # Square for reactions/transitions
        edgecolors="black", 
        linewidths=2
    )
    
    # Create custom edge styles based on Petri Net conventions
    edge_styles = []
    edge_weights = []
    edge_colors = []
    
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        edge_weights.append(weight)
        
        # Different colors for input and output arcs
        if graph.nodes[u]['type'] == 'species':
            edge_colors.append('navy')  # Input to reaction
        else:
            edge_colors.append('darkred')  # Output from reaction
            
        edge_styles.append('solid')
    
    # Draw the edges with arrows
    nx.draw_networkx_edges(
        graph, pos, 
        arrowstyle="-|>", 
        arrowsize=15, 
        edge_color=edge_colors,
        width=[max(1.0, min(2.5, w/2)) for w in edge_weights],
        style=edge_styles,
        connectionstyle='arc3,rad=0.1'  # Slightly curved edges to avoid overlap
    )
    
    # Draw the labels for nodes with different font sizes and positions
    species_labels = {n: graph.nodes[n]["label"] for n in species_nodes}
    reaction_labels = {n: graph.nodes[n]["label"] for n in reaction_nodes}
    
    nx.draw_networkx_labels(
        graph, pos, 
        labels=species_labels, 
        font_size=10, 
        font_color="black",
        font_weight='bold'
    )
    
    nx.draw_networkx_labels(
        graph, pos, 
        labels=reaction_labels, 
        font_size=9, 
        font_color="black",
        font_family='monospace'
    )
    
    # Create custom edge labels with weights
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        if weight != 1.0:  # Only show weight if not 1
            edge_labels[(u, v)] = f"{weight}"
    
    # Draw edge weight labels
    nx.draw_networkx_edge_labels(
        graph, pos, 
        edge_labels=edge_labels,
        font_size=9,
        font_color='black',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        rotate=False,
        label_pos=0.5
    )
    
    # Add a legend
    species_patch = mpatches.Patch(color='skyblue', label='Species (Places)', edgecolor='black')
    reaction_patch = mpatches.Patch(color='lightcoral', label='Reactions (Transitions)', edgecolor='black')
    plt.legend(handles=[species_patch, reaction_patch], loc='upper right')
    
    plt.title("Reaction Network (Petri Net Notation)")
    plt.axis("off")
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(img_dir_path, exist_ok=True)
    
    # Ensure img_name has .png extension
    if not img_name.endswith('.png'):
        img_name = f"{img_name}.png"
    
    # Combine directory path and filename
    img_file_path = os.path.join(img_dir_path, img_name)
    
    # Save the figure
    plt.savefig(img_file_path, dpi=300, bbox_inches='tight')
    print_log(log_file, f"Network plot saved to: {img_file_path}")


def adjust_positions(pos, spacing=1.0):
    """
    Adjust node positions to prevent overlaps.
    
    Args:
        pos: Dictionary of node positions
        spacing: Minimum distance between nodes
        
    Returns:
        Dictionary of adjusted node positions
    """
    import numpy as np
    
    # Convert positions dictionary to array for easier manipulation
    nodes = list(pos.keys())
    pos_array = np.array([pos[node] for node in nodes])
    
    # Simple repulsion algorithm
    for _ in range(50):  # Number of iterations
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                # Calculate distance between nodes
                delta = pos_array[i] - pos_array[j]
                distance = np.linalg.norm(delta)
                
                if distance < spacing:
                    # Calculate repulsion force
                    force = delta * (spacing - distance) / distance
                    
                    # Apply force (move both nodes away from each other)
                    pos_array[i] += force * 0.2
                    pos_array[j] -= force * 0.2
    
    # Convert back to dictionary
    new_pos = {nodes[i]: tuple(pos_array[i]) for i in range(len(nodes))}
    return new_pos