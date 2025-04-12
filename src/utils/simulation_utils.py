import roadrunner as rr
import networkx as nx

def load_roadrunner_model(file_path = None):
    if file_path is None:
        return rr.RoadRunner()
    else:
        return rr.RoadRunner(file_path)


def plot_results(simulation_data, species_names):
    """
    Visualize the simulation results.
    
    Args:
        simulation_data: NumPy array with simulation results
        species_names: Names of the species to display
    """
    import matplotlib.pyplot as plt
    
    time = simulation_data[:, 0]
    

    plt.figure(figsize=(10, 6))
    
    for i, species in enumerate(species_names):
        column_idx = i + 1
        if column_idx < simulation_data.shape[1]:
            plt.plot(time, simulation_data[:, column_idx], label=species)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('Simulation')
    plt.grid(True)
    plt.show()


def simulate(rr_model, start_time = 0, end_time = 10, output_rows = 100):

    result = rr_model.simulate(start_time, end_time, output_rows)

    return result


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

    #Creates the species dict for nodes
    for s in list_of_species:
        s_dict = s.to_dict()
        s_dict['label'] = s.get_id()
        s_list.append((s.get_id(), s_dict))

    #Creates the reaction dict for nodes

    for r in list_of_reactions:
        r_dict = r.to_dict()
        r_dict['label'] = r.get_id()
        r_list.append((r.get_id(), r_dict))
    

    # Add species nodes
    DG.add_nodes_from(s_list, type = "species")

    #Add reaction nodes
    DG.add_nodes_from(r_list, type = 'reaction')

    # Add reaction nodes and edges
    for reaction in list_of_reactions:

        # Add edges from reagents to the reaction
        for reagent in reaction.get_reagents():
            DG.add_edge(reagent.get_species()['id'], reaction.get_id(), stoichiometry=reagent.get_stoichiometry())

        # Add edges from the reaction to products
        for product in reaction.get_products():
            DG.add_edge(reaction.get_id(), product.get_species()['id'], stoichiometry=product.get_stoichiometry())

    return DG


def knockout_species_node(network, target_species) -> str:
    # #TODO: Tagliare tutti i rami in input di una determinata specie
    pass



def plot_network(graph):
    """
    Plot the reaction network with some spacing between nodes.

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

    # Draw the labels
    labels = nx.get_node_attributes(graph, "label")
    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color="black")

    # Display the plot
    plt.title("Reaction Network")
    plt.axis("off")
    plt.show()
