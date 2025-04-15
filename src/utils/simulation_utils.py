import roadrunner as rr
import libsbml


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


def plot_results(simulation_data, species_names, img_dir_path="./imgs", img_name="simulation"):
    """
    Visualize the simulation results and save the plot to a file.
    
    Args:
        simulation_data: NumPy array with simulation results
        species_names: Names of the species to display
        img_dir_path: Directory where to save the image (default: "./imgs")
        img_name: Name of the image file without extension (default: "simulation")
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(img_dir_path, exist_ok=True)
    
    # Ensure img_name has .png extension
    if not img_name.endswith('.png'):
        img_name = f"{img_name}.png"
    
    # Combine directory path and filename
    img_file_path = os.path.join(img_dir_path, img_name)
    
    time = simulation_data[:, 0]
    
    plt.figure(figsize=(10, 6))
    
    for i, species in enumerate(species_names):
        column_idx = i + 1
        if column_idx < simulation_data.shape[1]:
            plt.plot(time, simulation_data[:, column_idx], label=species)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title(f'Simulation: {img_name.replace(".png", "")}')
    plt.grid(True)
    
    # Save the figure with the complete path
    plt.savefig(img_file_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved to: {img_file_path}")


def simulate(rr_model, start_time = 0, end_time = 10, output_rows = 100):

    result = rr_model.simulate(start_time, end_time, output_rows)

    return result



