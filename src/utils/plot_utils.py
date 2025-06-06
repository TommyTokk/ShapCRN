import matplotlib.pyplot as plt

import math
from networkx.algorithms.bipartite import color
import numpy as np
import matplotlib.colors as colors
import os  # needed for saving
from collections import defaultdict
from utils.simulation_utils import get_variations_mean
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


def plot_variations_heatmap(
    variations_dict,
    figsize=(12, 8),
    cmap="RdBu_r",
    save_path="./imgs",
    show_averages=True,
    title=None,
    log_file=None,
):
    """
    Create a heatmap visualization of species variations across knockouts,
    with visible grid lines between cells.

    Args:
        variations_dict: Dict returned by get_simulations_variations
        figsize: Figure size tuple (default: (12, 8))
        cmap: Colormap for heatmap (default: "RdBu_r")
        save_path: Path to save the plot (optional)
        show_averages: Whether to show average variations (default: True)
        title: Custom title for the plot (optional)
    """

    if not variations_dict:
        print("No variations data to plot.")
        return

    # Collect all species and knocked-out species
    all_species = set()
    ko_species_list = list(variations_dict.keys())

    heatmap_data, all_species = get_variations_mean(
        variations_dict, all_species, ko_species_list, log_file
    )
    # Create the figure
    plt.figure(figsize=figsize)

    # Color scale limits
    vmax = heatmap_data.max()
    vmin = 0  # minimum is zero (absolute values)

    # Draw the heatmap
    im = plt.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Set x- and y-axis labels
    plt.xticks(range(len(all_species)), all_species, rotation=45, ha="right")
    plt.yticks(range(len(ko_species_list)), ko_species_list)

    # Add separating lines between cells:
    ax = plt.gca()
    # Set minor ticks at -0.5, 0.5, 1.5, ..., up to the number of species
    ax.set_xticks(np.arange(-0.5, len(all_species), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ko_species_list), 1), minor=True)
    # Draw the grid using the minor ticks
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    # Remove minor tick markers (actual ticks are not needed)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("Mean Absolute Variation", rotation=270, labelpad=20)

    # Title
    if title is None:
        title = "Species Variations Heatmap\n(Mean absolute variation across all combinations)"
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.xlabel("Species", fontweight="bold")
    plt.ylabel("Knocked Out Species", fontweight="bold")

    plt.tight_layout()

    # Save if specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "Variation Heatmap.png"))
    plt.close()

    # Summary statistics
    print("\nHeatmap Summary:")
    print(f"Max variation: {heatmap_data.max():.4f}")
    print(f"Min variation: {heatmap_data.min():.4f}")
    print(f"Mean absolute variation: {heatmap_data.mean():.4f}")
    non_zero = np.count_nonzero(heatmap_data)
    total = heatmap_data.size
    print(f"Non-zero variations: {non_zero}/{total} ({non_zero/total*100:.1f}%)")
