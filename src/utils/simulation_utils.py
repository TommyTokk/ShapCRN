import roadrunner as rr

def load_roadrunner_model(file_path):
    return rr.RoadRunner(file_path)


def plot_results(simulation_data, species_names):
    """
    Visualize the simulation results.
    
    Args:
        simulation_data: NumPy array with simulation results
        species_names: Names of the species to display
    """
    import matplotlib.pyplot as plt
    
    # Estrai il tempo (prima colonna)
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