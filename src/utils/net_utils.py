import networkx as nx
import libsbml

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


def network_to_sbml(net, rr_model, sbml):

    # #TODO: Rendere il metodo scalabile per coprire anche reazioni con parametri e funzioni
    # #TODO: Rendere più efficiente il codice 

    
    #Create the species list
    s_ids = []

    r_ids = []

    
    species_id_map = {sid: sid for sid in s_ids}


    for node, data in net.nodes(data=True):
        #Update the species
        # #FIXED: Rimuovere le specie rimuove anche le reazioni
        if data['type'] == 'species':
            try:
                model_species_id = data.get('id')
                #DEBUG:
                # print(f"sid: {model_species_id}")
                # print(f"compartment:{data.get('compartment')}")
                # print(f"init_conc: {data.get('initial_concentration')}")

                if model_species_id:
                    rr_model.removeSpecies(model_species_id)

                rr_model.addSpeciesConcentration(model_species_id, data.get('compartment'), data.get('initial_concentration'))


            except Exception as e:
                print(f"Error during update of species {data['id']}: {e}")

    for reaction in sbml.getListOfReactions():
        r_ids.append(reaction.getId())
        
    # Create a mapping from network reaction IDs to SBML model reaction IDs
    reaction_id_map = {rid: rid for rid in r_ids}

    for node, data in net.nodes(data=True):
        #Update the reaction
        if data.get('type') == 'reaction':
            try:
                # Find the correct id for the reaction
                model_reaction_id = reaction_id_map.get(data['id'])
                
                if model_reaction_id:
                    #print(f"Updating reaction: {data['id']} (nel modello: {model_reaction_id})")

                    #rr_model.removeReaction(model_reaction_id)
                    
                    #Updating the reaction

                    #Preparing the needed information(reagents, products and kineticLaw)
                    reagents_ids = []
                    net_reagents = data['reagents']
                    for species in net_reagents:
                        s_info = species['species']
                        reagents_ids.append(s_info['id'])

                    products_ids = []
                    net_products = data['products']
                    for species in net_products:
                        s_info = species['species']
                        products_ids.append(s_info['id'])

                    reaction_kinetic_law = data['kinetic_law']

                    #DEBUG
                    # print(f"reagents:{reagents_ids}")
                    # print(f"products:{products_ids}")
                    # print(f"kinetic_law:{reaction_kinetic_law}")

                    rr_model.addReaction(model_reaction_id, reagents_ids, products_ids, reaction_kinetic_law, False)
                else:
                    print(f"Reaction {data['id']} not found in the model")
            except Exception as e:
                print(f"Error during update of reaction {data['id']}: {e}")
            

    

    rr_model.regenerateModel()

    return rr_model





def knockout_reaction_node(network, target_reaction_id):
    network.remove_node(target_reaction_id)

    return network



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