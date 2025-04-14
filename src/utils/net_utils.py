import networkx as nx
import libsbml
import roadrunner

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





def knockout_reaction_node(network, target_reaction_id, rr_model):
    """
    Removes the target reaction node and all its edges, and updates the SBML model
    
    Args:
        network: The networkx network to modify
        target_reaction_id: The id of the target reaction to remove
        rr_model: RoadRunner model to update
        
    Returns:
        tuple: (modified network, updated RoadRunner model)
    """

    rr_model.reset()
    # Check if the target reaction is in the graph
    if target_reaction_id not in network.nodes():
        print(f"Reaction: {target_reaction_id} not present in the graph")
        return network, rr_model
    
    # Getting the input/output edges of the target node
    in_edges = list(network.in_edges(target_reaction_id))
    out_edges = list(network.out_edges(target_reaction_id))
    
    # Removing the input/output edges of the target node
    network.remove_edges_from(in_edges)
    network.remove_edges_from(out_edges)
    
    # Extract the SBML document
    modified = False
    try:
        # Update the SBML model by removing the reaction
        rr_model.removeReaction(target_reaction_id)
        modified = True
    except Exception as e:
        print(f"Error removing reaction {target_reaction_id} from model: {e}")
    
    # If modifications were made, rebuild the model to ensure consistency
    if modified:
        try:
            # Get the current SBML after modifications
            updated_sbml = rr_model.getCurrentSBML()
            
            # Create a new RoadRunner instance with the updated SBML
            new_rr_model = roadrunner.RoadRunner(updated_sbml)
            
            # Reset the model to ensure initial values are correctly set
            #new_rr_model.reset()
            
            # Transfer integrator settings
            new_rr_model.getIntegrator().setValue('relative_tolerance', 
                                                rr_model.getIntegrator().getValue('relative_tolerance'))
            new_rr_model.getIntegrator().setValue('absolute_tolerance', 
                                                rr_model.getIntegrator().getValue('absolute_tolerance'))
            
            print(f"Successfully removed reaction {target_reaction_id} from SBML model")
            return network, new_rr_model
        except Exception as e:
            print(f"Error updating model after removing reaction: {e}")
    
    return network, rr_model


def knockout_species_node(network, target_species_id, rr_model):
    """
    Removes all the input edges to the specified species node and updates the SBML model
    
    Args:
        network: The networkx network to modify
        target_species_id: The id of the target species to isolate
        rr_model: RoadRunner model to update
        
    Returns:
        tuple: (modified network, updated RoadRunner model)
    """
    # Check if the target species is in the graph
    if target_species_id not in network.nodes():
        print(f"Species: {target_species_id} not present in the graph")
        return network, rr_model

    # Getting the input edges of the target node
    in_edges = list(network.in_edges(target_species_id))
    
    # Removing the input edges from the graph
    if in_edges:
        network.remove_edges_from(in_edges)
    
    # Setting the concentration of the target species to 0.0
    new_conc = 0.0
    network.nodes[target_species_id]['species'].set_initial_concentration(new_conc)
    
    # Updating the SBML model concentration
    rr_model[target_species_id] = new_conc
    
    # Extract the SBML document ONCE outside the loop
    modified = False
    sbml_doc = None
    sbml_model = None
    
    # Find reactions to update (only from producing reactions)
    reactions_to_update = {source for source, target in in_edges 
                          if network.nodes[source]['type'] == 'reaction'}
    
    if reactions_to_update:
        # Parse the SBML document only once
        sbml_doc = libsbml.readSBMLFromString(rr_model.getSBML())
        sbml_model = sbml_doc.getModel()
    
    # Update all reactions in the network data structure
    for reaction_id in reactions_to_update:
        reaction = network.nodes[reaction_id]['reaction']
        removed = reaction.remove_product(target_species_id)
        
        if removed:
            modified = True
            # Update the corresponding reaction in the SBML model
            sbml_reaction = sbml_model.getReaction(reaction_id)
            if sbml_reaction:
                # Remove all occurrences of this species as a product
                products_length = sbml_reaction.getNumProducts()
                for i in range(products_length-1, -1, -1):
                    product = sbml_reaction.getProduct(i)
                    if product.getSpecies() == target_species_id:
                        res = sbml_reaction.removeProduct(i)
                        if res == None:
                            exit("Removing the product has failed")
    
    # Create updated model ONCE if modifications were made
    if modified:
        try:
            updated_sbml = libsbml.writeSBMLToString(sbml_doc)
            new_rr_model = roadrunner.RoadRunner(updated_sbml)
            
            # Reset del modello per assicurarsi che i valori iniziali siano corretti
            new_rr_model.reset()
            
            # Trasferisci eventuali impostazioni dell'integratore
            new_rr_model.getIntegrator().setValue('relative_tolerance', 
                                                 rr_model.getIntegrator().getValue('relative_tolerance'))
            new_rr_model.getIntegrator().setValue('absolute_tolerance',
                                                 rr_model.getIntegrator().getValue('absolute_tolerance'))
            
            print("Successfully updated SBML model")
            return network, new_rr_model
        except Exception as e:
            print(f"Error updating model: {e}")
    
    return network, rr_model


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