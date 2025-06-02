import matplotlib.pyplot as plt
import os
import math
from networkx.algorithms.bipartite import color
import numpy as np
import matplotlib.colors as colors

from utils.utils import print_log


def plot_results(
    simulation_data,
    colnames,
    img_dir_path="./imgs",
    img_name="simulation",
    log_file=None,
):
    """
    Visualize the simulation results and save the plot to a file.

    Args:
        simulation_data: NumPy array with simulation results
        img_dir_path: Directory where to save the image (default: "./imgs")
        img_name: Name of the image file without extension (default: "simulation")
    """
    species_names = colnames
    # print(species_names)

    # Create directory if it doesn't exist
    os.makedirs(img_dir_path, exist_ok=True)

    # Ensure img_name has .png extension
    if not img_name.endswith(".png"):
        img_name = f"{img_name}.png"

    # Combine directory path and filename
    img_file_path = os.path.join(img_dir_path, img_name)

    time = simulation_data[:, 0]

    # Create figure with adjusted size to accommodate legend
    plt.figure(figsize=(12, 8))

    for i, species in enumerate(species_names):
        column_idx = i + 1
        if column_idx < simulation_data.shape[1]:
            plt.plot(time, simulation_data[:, column_idx], label=species)

    plt.xlabel("Time")
    plt.ylabel("Concentration")
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
    legend = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=ncols,
        fontsize="small",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Dynamically adjust spacing to provide more room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 + 0.02 * math.ceil(num_species / ncols))

    # Save the figure with the complete path
    plt.savefig(img_file_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print_log(log_file, f"Plot saved to: {img_file_path}")


def plot_statistical_comparison(
    time,
    original_data,
    perturbed_array,
    avg_rms_variations_percentage,
    target_species,
    species_dir,
):
    """
    Plot the comparison between original data and simulation statistics with confidence intervals.

    Args:
        time: Time points array
        original_data: Original simulation data
        rms_variations
        target_species: Name of the species being analyzed
        species_dir: Directory to save the plot
    """
    max_traces = 125
    plt.figure(figsize=(12, 8))
    plt.plot(time, original_data, "b-", linewidth=2, label="Original")
    plt.fill_between(
        time,
        original_data - ((avg_rms_variations_percentage / 100) * original_data),
        original_data + ((avg_rms_variations_percentage / 100) * original_data),
        color="r",
        alpha=0.2,
        label="± avg_rms_variations",
    )

    min = np.min(perturbed_array, axis=0)
    max = np.max(perturbed_array, axis=0)

    plt.fill_between(time, min, max, color="grey", alpha=0.2, label="Min-Max")

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{target_species} - Original vs Perturbed (125 simulations)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(species_dir, f"{target_species}_statistical_comparison.png")
    )
    plt.close()


def plot_percentage_variation(
    time, mean_percent_variation, percent_variations, target_species, species_dir
):
    """
    Plot the percentage variation from original data.

    Args:
        time: Time points array
        mean_percent_variation: Mean percentage variation across simulations
        percent_variations: Array of percentage variations for all simulations
        target_species: Name of the species being analyzed
        species_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, mean_percent_variation, "g-", linewidth=2, label="RMS % variation")
    # Add standard deviation of the percentage variations
    std_percent = np.std(percent_variations, axis=0)
    plt.fill_between(
        time,
        mean_percent_variation - std_percent,
        mean_percent_variation + std_percent,
        color="g",
        alpha=0.2,
        label="±1 Std Dev",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Variation %")
    plt.title(f"{target_species} - Percentage Variation from Original")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(species_dir, f"{target_species}_percentage_variation.png"))
    plt.close()


def plot_variations_heatmap(
    percent_variations, perturbed_data_list, target_species, species_dir
):
    """
    Plot a heatmap of percentage variations across all simulations.

    Args:
        percent_variations: Array of percentage variations for all simulations
        perturbed_data_list: List of perturbed simulation data
        target_species: Name of the species being analyzed
        species_dir: Directory to save the plot
    """
    if len(perturbed_data_list) > 50:
        # Sample only 50 simulations for better visualization
        step = len(perturbed_data_list) // 50
        sampled_percent_variations = percent_variations[::step, :]
        plt.figure(figsize=(14, 8))
        plt.imshow(
            sampled_percent_variations, aspect="auto", cmap="RdBu_r", vmin=-50, vmax=50
        )  # Limit to ±50% for better visualization
        plt.colorbar(label="Variation %")
        plt.xlabel("Timepoints")
        plt.ylabel("Simulation #")
        plt.title(f"{target_species} - Variation Heatmap (Sample of simulations)")
    else:
        plt.figure(figsize=(14, 8))
        plt.imshow(percent_variations, aspect="auto", cmap="RdBu_r", vmin=-50, vmax=50)
        plt.colorbar(label="Variation %")
        plt.xlabel("Timepoints")
        plt.ylabel("Simulation #")
        plt.title(f"{target_species} - Variation Heatmap (All simulations)")

    plt.tight_layout()
    plt.savefig(os.path.join(species_dir, f"{target_species}_heatmap.png"))
    plt.close()


def plot_boxplot_distribution(
    perturbed_array,
    time,
    target_species,
    species_dir,
    num_timepoints=5,
    original_data=None,
):
    """
    Create boxplots showing the distribution of simulation values at selected timepoints.

    Args:
        perturbed_array: Array containing all simulation results for the species
        time: Time points array
        target_species: Name of the species being analyzed
        species_dir: Directory to save the plot
        num_timepoints: Number of timepoints to sample for the boxplots (default: 5)
        original_data: Optional array with original simulation data (default: None)
    """
    # Select timepoints evenly distributed across the simulation
    indices = np.linspace(0, len(time) - 1, num_timepoints, dtype=int)
    selected_times = time[indices]

    # Extract data at selected timepoints
    data_at_timepoints = [perturbed_array[:, idx] for idx in indices]

    plt.figure(figsize=(12, 8))

    # Create boxplot
    box = plt.boxplot(data_at_timepoints, patch_artist=True)

    # Customize boxplot colors
    for patch in box["boxes"]:
        patch.set_facecolor("lightblue")

    # Add a line for the legend entry (invisible, just for the legend)
    plt.plot([], [], "lightblue", linewidth=10, label="Simulation distribution")

    # Add original data points if provided
    if original_data is not None:
        original_values = original_data[indices]
        plt.scatter(
            range(1, num_timepoints + 1),
            original_values,
            color="blue",
            marker="o",
            s=80,
            label="Original value",
        )

    # Set labels for x-axis with selected time points
    plt.xticks(range(1, num_timepoints + 1), [f"t={t:.2f}" for t in selected_times])

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{target_species} - Distribution of Values Across Simulations")
    plt.grid(True, alpha=0.3)

    # Add simple legend like in plot_percentage_variation
    plt.legend()

    # Add a text with statistics in the corner
    stats_text = "Statistics per timepoint:\n"
    for i, data in enumerate(data_at_timepoints):
        stats_text += f"t={selected_times[i]:.2f}: mean={np.mean(data):.4f}, "
        stats_text += f"std={np.std(data):.4f}\n"

    plt.figtext(
        0.02,
        0.02,
        stats_text,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.5, boxstyle="round"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(species_dir, f"{target_species}_boxplot_distribution.png"))
    plt.close()


def plot_all_simulation_traces(
    time,
    original_data,
    perturbed_array,
    target_species,
    species_dir,
    max_traces=125,
    alpha=0.5,
):
    """
    Plot all simulation traces along with the original simulation for comparison.

    Args:
        time: Time points array
        original_data: Original simulation data for the target species
        perturbed_array: Array of all simulation results (shape: n_simulations, n_timepoints)
        target_species: Name of the species being analyzed
        species_dir: Directory to save the plot
        max_traces: Maximum number of traces to plot to avoid overcrowding (default: 125)
        alpha: Transparency level for the perturbed traces (default: 0.5)
    """
    plt.figure(figsize=(12, 8))

    # Limit the number of traces if there are too many
    n_simulations = perturbed_array.shape[0]
    if n_simulations > max_traces:
        # Sample evenly spaced traces
        indices = np.linspace(0, n_simulations - 1, max_traces, dtype=int)
        traces_to_plot = perturbed_array[indices]
        n_plotted = max_traces
    else:
        traces_to_plot = perturbed_array
        n_plotted = n_simulations

    cmap = plt.cm.viridis  # Altre opzioni: plasma, magma, inferno, cividis
    colors = [cmap(i / n_plotted) for i in range(n_plotted)]

    # Plot all perturbed traces with different colors
    for i in range(n_plotted):
        plt.plot(time, traces_to_plot[i], color=colors[i], alpha=alpha, linewidth=0.8)

    # Add a representative trace for legend
    plt.plot(
        [],
        [],
        color=colors[0],
        alpha=0.7,
        linewidth=1,
        label=f"Perturbed ({n_simulations} simulations)",
    )

    # Plot the original trace with emphasis
    plt.plot(time, original_data, "b-", linewidth=2, label="Original")

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{target_species} - All Simulation Traces")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(species_dir, f"{target_species}_all_traces.png"))
    plt.close()


def plot_knockdown_effect_heatmap(
    variations_dict,
    output_dir="./imgs",
    use_log_scale=True,
    cmap="viridis_r",  # "_r" per avere colori più scuri per valori maggiori
    log_file=None,
):
    """
    Crea una heatmap che visualizza l'impatto delle specie knockout sul modello.
    Maggiore è la variazione, più scuro è il colore.

    Args:
        variations_dict: Il dizionario delle variazioni da get_variations_dict
        output_dir: Directory dove salvare i grafici
        use_log_scale: Se usare scala logaritmica per visualizzare meglio variazioni di ordini diversi
        cmap: Colormap da usare (default: "viridis_r")
        log_file: File opzionale per il logging
    """
    os.makedirs(output_dir, exist_ok=True)

    # Estrai tutte le specie knocked out e le specie target
    knocked_species_list = list(variations_dict.keys())
    all_target_species = set()
    for knocked_species, combinations in variations_dict.items():
        for combination, species_dict in combinations.items():
            all_target_species.update(species_dict.keys())

    all_target_species = sorted(list(all_target_species))

    # Prepara la matrice di dati
    num_knocked = len(knocked_species_list)
    num_targets = len(all_target_species)

    # Inizializza la matrice con NaN per distinguere dati mancanti
    variation_matrix = np.full((num_knocked, num_targets), np.nan)

    # Per ogni specie knocked, calcola l'effetto medio su ogni specie target
    for i, knocked_species in enumerate(knocked_species_list):
        species_variations = {}

        # Raccogli tutte le variazioni per ogni specie target su tutte le combinazioni
        for combination, species_dict in variations_dict[knocked_species].items():
            for target_species, value in species_dict.items():
                if target_species not in species_variations:
                    species_variations[target_species] = []

                # Usa il valore assoluto per la magnitudine della variazione
                variation_value = np.abs(value["variation"])
                species_variations[target_species].append(variation_value)

        # Calcola la variazione media per ogni specie target
        for j, target_species in enumerate(all_target_species):
            if (
                target_species in species_variations
                and species_variations[target_species]
            ):
                variation_matrix[i, j] = np.nanmean(species_variations[target_species])

    # Crea la heatmap utilizzando matplotlib
    plt.figure(figsize=(max(12, num_targets / 2), max(8, num_knocked / 2)))

    # Maschera per i valori NaN
    masked_array = np.ma.array(variation_matrix, mask=np.isnan(variation_matrix))

    # Configura la normalizzazione dei colori
    if use_log_scale:
        # Gestisci zeri e negativi se usi scala logaritmica
        positive_values = masked_array[masked_array > 0]
        if len(positive_values) > 0:
            min_positive = np.min(positive_values)
            max_value = np.max(masked_array)
            norm = colors.LogNorm(vmin=min_positive, vmax=max_value)
        else:
            norm = None
    else:
        max_val = np.max(masked_array)
        if max_val > 0:
            norm = colors.Normalize(vmin=0, vmax=max_val)
        else:
            norm = None

    # Crea la heatmap con matplotlib
    plt.figure(figsize=(max(12, num_targets / 2), max(8, num_knocked / 2)))
    im = plt.imshow(masked_array, cmap=cmap, norm=norm, aspect="auto")

    # Aggiungi colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Magnitudine variazione media")

    # Aggiungi le etichette degli assi
    plt.xticks(np.arange(len(all_target_species)), all_target_species, rotation=90)
    plt.yticks(np.arange(len(knocked_species_list)), knocked_species_list)

    plt.title("Impatto delle specie knockout sui componenti del modello")
    plt.xlabel("Specie target")
    plt.ylabel("Specie knockout")

    # Aggiungi una griglia per migliorare la leggibilità
    plt.grid(False)

    plt.tight_layout()

    # Salva la figura
    output_path = os.path.join(output_dir, "knockdown_effects_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print_log(log_file, f"Salvata heatmap dell'effetto knockout in: {output_path}")

    # Crea una seconda heatmap con valori normalizzati per riga
    if not np.all(np.isnan(variation_matrix)):
        # Normalizza ogni riga (ogni specie knocked) per evidenziare cosa influenza maggiormente
        row_normalized = np.zeros_like(variation_matrix)
        for i in range(variation_matrix.shape[0]):
            row_data = variation_matrix[i, :]
            valid_data = row_data[~np.isnan(row_data)]
            if len(valid_data) > 0 and np.max(valid_data) > 0:
                row_normalized[i, ~np.isnan(row_data)] = valid_data / np.max(valid_data)

        # Maschera per i valori NaN nella matrice normalizzata
        masked_normalized = np.ma.array(row_normalized, mask=np.isnan(row_normalized))

        # Crea la heatmap normalizzata
        plt.figure(figsize=(max(12, num_targets / 2), max(8, num_knocked / 2)))
        im = plt.imshow(masked_normalized, cmap=cmap, aspect="auto")

        # Aggiungi colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Impatto normalizzato (0-1)")

        # Aggiungi le etichette degli assi
        plt.xticks(np.arange(len(all_target_species)), all_target_species, rotation=90)
        plt.yticks(np.arange(len(knocked_species_list)), knocked_species_list)

        plt.title("Impatto relativo delle specie knockout")
        plt.xlabel("Specie target")
        plt.ylabel("Specie knockout")

        plt.grid(False)
        plt.tight_layout()

        # Salva la figura normalizzata
        norm_output_path = os.path.join(
            output_dir, "knockdown_effects_normalized_heatmap.png"
        )
        plt.savefig(norm_output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print_log(
            log_file,
            f"Salvata heatmap normalizzata dell'effetto knockout in: {norm_output_path}",
        )
