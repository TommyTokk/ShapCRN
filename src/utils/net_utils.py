import networkx as nx
import libsbml
import roadrunner

from src.utils import sbml_utils

# ============
# NETWORK
# ============

def get_network_from_sbml(list_of_reactions, list_of_species):
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

    # #TODO: Change to include the species obj as value of the node

    #Creates the species dict for nodes
    for s in list_of_species:
        DG.add_node(s.get_id(), species = s, label = s.get_id(), type = "species")
        # s_dict = s.to_dict()
        # s_dict['label'] = (s.get_id(), s.get_initial_concentration())
        # s_list.append((s.get_id(), s))


    # #TODO: Change to include the reaction obj as value of the node
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





def inhibit_reaction_from_network(network, target_reaction_id, sbml_model):
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

    new_model  = sbml_utils.inhibit_reaction(sbml_model, target_reaction_id)

    return network, new_model




def inhibit_species_from_network(network, target_species_id, sbml_model):
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
    new_model = sbml_utils.inhibit_species(target_species_id)

    return network, sbml_model


def plot_network(graph):
    """
    Plot the reaction network with some spacing between nodes and edge weights.

    Args:
        graph: A networkx DiGraph representing the reaction network
    """
    import matplotlib.pyplot as plt

    # Create a layout for the graph with spacing
    pos = nx.spring_layout(graph, k=0.5, iterations=50)

    # Draw the nodes with labels
    node_colors = ["lightblue" if data["type"] == "species" else "lightgreen" for _, data in graph.nodes(data=True)]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500, edgecolors="black")

    # Draw the edges
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=10, edge_color="gray")

    # Draw the labels for nodes
    labels = nx.get_node_attributes(graph, "label")
    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color="black")

    # Get edge weights and create edge labels
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in graph.edges(data=True)}
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(
        graph, 
        pos, 
        edge_labels=edge_labels,
        font_size=8,
        font_color='red',
        rotate=False
    )

    # Display the plot
    plt.title("Reaction Network")
    plt.axis("off")
    plt.show()