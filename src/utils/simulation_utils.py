import roadrunner as rr
import libsbml
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from src.utils.utils import print_log


def load_roadrunner_model(sbml_model):
    """
    Loads a SBML model into a RoadRunner instance and configures the integrator settings.
    
    Args:
        sbml_model: The SBML model (libsbml.Model or string)
        rel_tol: Relative tolerance for the integrator (default: 1e-6)
        abs_tol: Absolute tolerance for the integrator (default: 1e-8)
    
    Returns:
        roadrunner.RoadRunner: Configured RoadRunner instance
    """
    # Check if input is a libSBML model or a string
    if isinstance(sbml_model, libsbml.Model):
        sbml_doc = libsbml.SBMLWriter().writeSBMLToString(sbml_model.getSBMLDocument())
        rr_model = rr.RoadRunner(sbml_doc)
    else:
        # Assume it's a string representation or file path
        rr_model = rr.RoadRunner(sbml_model)
    
    # Configure integrator settings
    # rr_model.getIntegrator().setValue('relative_tolerance', rel_tol)
    # rr_model.getIntegrator().setValue('absolute_tolerance', abs_tol)
    
    return rr_model


def plot_results(simulation_data, img_dir_path="./imgs", img_name="simulation", log_file = None):
    """
    Visualize the simulation results and save the plot to a file.
    
    Args:
        simulation_data: NumPy array with simulation results
        img_dir_path: Directory where to save the image (default: "./imgs")
        img_name: Name of the image file without extension (default: "simulation")
    """
    species_names = simulation_data.colnames[1:]
    #print(species_names)

    # Create directory if it doesn't exist
    os.makedirs(img_dir_path, exist_ok=True)
    
    # Ensure img_name has .png extension
    if not img_name.endswith('.png'):
        img_name = f"{img_name}.png"
    
    # Combine directory path and filename
    img_file_path = os.path.join(img_dir_path, img_name)
    
    time = simulation_data[:,0]
    
    # Create figure with adjusted size to accommodate legend
    plt.figure(figsize=(12, 8))
    
    for i, species in enumerate(species_names):
        column_idx = i + 1
        if column_idx < simulation_data.shape[1]:
            plt.plot(time, simulation_data[:, column_idx], label=species)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title(f'Simulation: {img_name.replace(".png", "")}')
    plt.grid(True)
    
    # Calculate the optimal number of columns for the legend
    # based on the number of species to display
    num_species = len(species_names)
    if num_species <= 3:
        ncols = 1
    elif num_species <= 8:
        ncols = 2
    elif num_species <= 15:
        ncols = 3
    elif num_species <= 24:
        ncols = 4
    else:
        # For very large models, increase the number of columns
        ncols = math.ceil(num_species / 10)
    
    # Create a legend with multiple columns, positioned below the graph
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=ncols, fontsize='small', frameon=True, 
                       fancybox=True, shadow=True)
    
    # Dynamically adjust spacing to provide more room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 + 0.02 * math.ceil(num_species / ncols))
    
    # Save the figure with the complete path
    plt.savefig(img_file_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print_log(log_file, f"Plot saved to: {img_file_path}")


def simulate(rr_model, start_time = 0, end_time = 10, output_rows = 100):

    rr_model.setIntegrator('gillespie')
    result = rr_model.simulate(start_time, end_time, output_rows)

    return result


def pearson_correlation(simulation_data, correlation_threshold=0.5):
    """
        Calculate the Pearson correlation of the simulation results and identify important species
        
        Args:
            - simulation_data: Results of the roadrunner simulation
            - correlation_threshold: Threshold to consider a correlation as significant (default: 0.5)
    """
    # Take the columns names
    cols_names = simulation_data.colnames

    # Convert the simulation data into numpy array
    numpy_sim_data = np.array(simulation_data)

    species_data = numpy_sim_data[:, 1:]
    species_names = cols_names[1:]

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(species_data, rowvar=False)

    return correlation_matrix






