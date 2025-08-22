import matplotlib.pyplot as plt

import math

import numpy as np

import pandas as pd

import os

from pandas.core.generic import pprint_thing  # needed for saving

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns

from utils.utils import print_log


def plot_results(
    simulation_data,
    colnames,
    img_dir_path="./imgs",
    img_name="simulation",
    log_file=None,
    ss_time=None,
):
    """
    Visualize the simulation results and save the plot to a file.

    Args:
        simulation_data: NumPy array with simulation results
        img_dir_path: Directory where to save the image (default: "./imgs")
        img_name: Name of the image file without extension (default: "simulation")
    """
    species_names = colnames[1:]
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
    plt.title("Model Simulation")
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

    # Creating the arrows for the states

    # Save the figure with the complete path
    plt.savefig(img_file_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print_log(log_file, f"Plot saved to: {img_file_path}")


def plot_results_interactive(
    simulation_data,
    colnames,
    html_dir_path="./plots",
    html_name="simulation",
    log_file=None,
    ss_time=None,
    show_plot=True,
    height=600,
    width=1000,
):
    """
    Create interactive visualization of simulation results using Plotly and save to HTML file.

    Args:
        simulation_data: NumPy array with simulation results
        colnames: List of column names (first should be time, rest are species)
        html_dir_path: Directory where to save the HTML file (default: "./plots")
        html_name: Name of the HTML file without extension (default: "simulation")
        log_file: Log file for messages (optional)
        ss_time: Steady state time marker (optional, adds vertical line)
        show_plot: Whether to display the plot in browser (default: True)
        height: Plot height in pixels (default: 600)
        width: Plot width in pixels (default: 1000)

    Returns:
        plotly.graph_objects.Figure: The created figure object
    """
    species_names = colnames[1:]

    # Create directory if it doesn't exist
    os.makedirs(html_dir_path, exist_ok=True)

    # Ensure html_name has .html extension
    if not html_name.endswith(".html"):
        html_name = f"{html_name}.html"

    # Combine directory path and filename
    html_file_path = os.path.join(html_dir_path, html_name)

    time = simulation_data[:, 0]

    # Create the interactive figure
    fig = go.Figure()

    # Add traces for each species
    for i, species in enumerate(species_names):
        column_idx = i + 1
        if column_idx < simulation_data.shape[1]:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=simulation_data[:, column_idx],
                    mode="lines",
                    name=species,
                    line=dict(width=2),
                    hovertemplate=f"<b>{species}</b><br>"
                    + "Time: %{x}<br>"
                    + "Concentration: %{y}<br>"
                    + "<extra></extra>",
                )
            )

    # Add steady state line if provided
    if ss_time is not None:
        fig.add_vline(
            x=ss_time,
            line_dash="dash",
            line_color="red",
            annotation_text="Steady State",
            annotation_position="top",
        )

    # Calculate legend positioning based on number of species
    num_species = len(species_names)

    # Determine legend layout and calculate required margins
    if num_species <= 10:
        legend_orientation = "v"  # vertical
        legend_x = 1.02
        legend_y = 1
        legend_xanchor = "left"
        legend_yanchor = "top"
        # Increase width to accommodate vertical legend
        adjusted_width = width + 150
        adjusted_height = height
        margin_updates = dict(r=180)  # Right margin for vertical legend
    else:
        legend_orientation = "h"  # horizontal
        legend_x = 0.5
        legend_y = -0.15
        legend_xanchor = "center"
        legend_yanchor = "top"
        adjusted_width = width
        # Calculate required height based on number of legend rows
        legend_rows = math.ceil(num_species / 4)  # Assume 4 items per row
        extra_height = max(80, legend_rows * 25)  # Minimum 80px, 25px per row
        adjusted_height = height + extra_height
        margin_updates = dict(
            b=extra_height + 20
        )  # Bottom margin for horizontal legend

    # Update layout with all settings at once
    fig.update_layout(
        title=dict(
            text=f'Interactive Simulation: {html_name.replace(".html", "")}',
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="Time",
        yaxis_title="Concentration",
        width=adjusted_width,
        height=adjusted_height,
        hovermode="x unified",
        legend=dict(
            orientation=legend_orientation,
            x=legend_x,
            y=legend_y,
            xanchor=legend_xanchor,
            yanchor=legend_yanchor,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=10),  # Smaller font for better fit
            itemsizing="constant",  # Consistent legend item sizing
        ),
        template="plotly_white",
        showlegend=True,
        margin=margin_updates,  # Apply calculated margins
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Save the interactive plot
    try:
        fig.write_html(html_file_path)
        success = True
    except Exception as e:
        success = False
        error_msg = f"Failed to save interactive plot: {str(e)}"
        try:
            print_log(log_file, error_msg)
        except NameError:
            print(error_msg)

    # Display the plot if requested and save was successful
    if show_plot and success:
        try:
            fig.show()
        except Exception as e:
            error_msg = f"Failed to display plot: {str(e)}"
            try:
                print_log(log_file, error_msg)
            except NameError:
                print(error_msg)

    # Log the save location
    if success:
        try:
            print_log(log_file, f"Interactive plot saved to: {html_file_path}")
        except NameError:
            print(f"Interactive plot saved to: {html_file_path}")

    return fig


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
    model_name,
    num_timepoints=5,
    save_path="./imgs",
    filename="boxplot_distribution",
):
    """
    Create boxplots showing the distribution of simulation values at selected timepoints.

    Args:
        perturbed_array: Array containing all simulation results for the species
        num_timepoints: Number of timepoints to sample for the boxplots (default: 5)
        save_path: Directory to save the plot (default: "./imgs")
        filename: Name of the saved file (default: "boxplot_distribution")
    """
    # Extract time and column names from the first simulation
    time = perturbed_array[0][:, 0]
    colnames = perturbed_array[0].colnames[1:]

    print(f"Time points shape: {time.shape}")
    print(f"Species columns: {colnames}")

    # Select timepoints evenly distributed across the simulation
    indices = np.linspace(0, len(time) - 1, num_timepoints, dtype=int)
    selected_times = time[indices]

    print(f"Selected time indices: {indices}")
    print(f"Selected times: {selected_times}")

    # Calculate means for each simulation across all species
    all_simulations_means = [np.mean(arr[:, 1:], axis=0) for arr in perturbed_array]

    # Create pandas DataFrame using colnames as column names and all_simulations_means as data
    df = pd.DataFrame(all_simulations_means, columns=colnames)

    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print("DataFrame head:")
    print(df.head())

    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Melt the DataFrame to long format for seaborn boxplot
    df_melted = df.melt(var_name="Species", value_name="Mean_Concentration")

    print(f"Melted DataFrame shape: {df_melted.shape}")
    print("Melted DataFrame head:")
    print(df_melted.head())

    # Create boxplot
    sns.boxplot(
        data=df_melted, x="Species", y="Mean_Concentration", ax=ax, palette="viridis"
    )

    # Customize the plot
    ax.set_title(
        "Distribution of Mean Concentrations Across Simulations",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Species", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Concentration", fontsize=12, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_path, model_name)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{filename}.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")

    print(f"Boxplot saved to: {save_file}")

    # Show the plot
    plt.show()
    plt.close()


def plot_all_simulation_traces(
    time,
    original_data,
    perturbed_array,
    target_species,
    species_dir="./imgs",
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


def plot_variations_heatmap_relative(
    heatmap_data,
    all_species,
    ko_species_list,
    figsize=(12, 8),
    cmap="PiYG_r",
    save_path="./imgs",
    show_averages=True,
    title=None,
    imgs_name=None,
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

    # Validate heatmap_data before proceeding
    if heatmap_data.size == 0:
        print("Error: Heatmap data is empty. No data to visualize.")
        if log_file:
            print_log(log_file, "Error: Heatmap data is empty. No data to visualize.")
        return

    # Check for all NaN values
    if np.all(np.isnan(heatmap_data)):
        print("Error: All heatmap data values are NaN. No valid data to visualize.")
        if log_file:
            print_log(
                log_file,
                "Error: All heatmap data values are NaN. No valid data to visualize.",
            )
        return

    # Check if we have valid data for visualization
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    if len(valid_data) == 0:
        print("Error: No valid (non-NaN) data points found.")
        if log_file:
            print_log(log_file, "Error: No valid (non-NaN) data points found.")
        return

    # Create the figure
    plt.figure(figsize=figsize)

    # Color scale limits - now safe to use .max() and .min()
    vmax = valid_data.max()
    vmin = 0

    # Log data range for debugging
    if log_file:
        print_log(log_file, f"Heatmap data shape: {heatmap_data.shape}")
        print_log(log_file, f"Valid data points: {len(valid_data)}/{heatmap_data.size}")
        print_log(log_file, f"Data range: {vmin:.10f} to {vmax:.10f}")

    masked_data = np.ma.masked_invalid(heatmap_data)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color="black")  # Optional: makes NaNs visible

    # Draw the heatmap
    im = plt.imshow(masked_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

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
    cbar.set_label(
        f"Mean relative Variation",
        rotation=270,
        labelpad=20,
    )

    # Title
    if title is None:
        title = f"Species Variations Heatmap\n(Mean relative variation across all combinations)"
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.xlabel("Species", fontweight="bold")
    plt.ylabel("Knocked Out Ids", fontweight="bold")

    plt.tight_layout()

    # Save if specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if imgs_name is None:
            plt.savefig(
                os.path.join(
                    save_path,
                    "Relative variations Heatmap.png",
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{imgs_name} Heatmap.png",
                )
            )

    plt.close()


def plot_variations_heatmap_absolute(
    heatmap_data,
    all_species,
    ko_species_list,
    figsize=(12, 8),
    cmap="PRGn_r",
    save_path="./imgs",
    show_averages=True,
    title=None,
    imgs_name=None,
    log_file=None,
):
    """
    Create a heatmap visualization of species variations across knockouts,
    with visible grid lines between cells.

    Args:
        variations_dict: Dict returned by get_simulations_variations
        figsize: Figure size tuple (default: (12, 8))
        cmap: Colormap for heatmap (default: "RdBu")
        save_path: Path to save the plot (optional)
        show_averages: Whether to show average variations (default: True)
        title: Custom title for the plot (optional)
    """

    # Validate heatmap_data before proceeding
    if heatmap_data.size == 0:
        print("Error: Heatmap data is empty. No data to visualize.")
        if log_file:
            print_log(log_file, "Error: Heatmap data is empty. No data to visualize.")
        return

    # Check for all NaN values
    if np.all(np.isnan(heatmap_data)):
        print("Error: All heatmap data values are NaN. No valid data to visualize.")
        if log_file:
            print_log(
                log_file,
                "Error: All heatmap data values are NaN. No valid data to visualize.",
            )
        return

    # Check if we have valid data for visualization
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    if len(valid_data) == 0:
        print("Error: No valid (non-NaN) data points found.")
        if log_file:
            print_log(log_file, "Error: No valid (non-NaN) data points found.")
        return

    # Create the figure
    plt.figure(figsize=figsize)

    # Color scale limits - now safe to use .max() and .min()
    vmax = valid_data.max()
    vmin = valid_data.min()

    # Log data range for debugging
    if log_file:
        print_log(log_file, f"Heatmap data shape: {heatmap_data.shape}")
        print_log(log_file, f"Valid data points: {len(valid_data)}/{heatmap_data.size}")
        print_log(log_file, f"Data range: {vmin:.10f} to {vmax:.10f}")

    masked_data = np.ma.masked_invalid(heatmap_data)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color="black")  # Optional: makes NaNs visible

    # Draw the heatmap
    im = plt.imshow(masked_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

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
    cbar.set_label(
        f"Mean Variation",
        rotation=270,
        labelpad=20,
    )

    # Title
    if title is None:
        title = f"Species Variations Heatmap\n(Mean variation across all combinations)"
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.xlabel("Species", fontweight="bold")
    plt.ylabel("Knocked Out Ids", fontweight="bold")

    plt.tight_layout()

    # Save if specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if imgs_name is None:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"Variations Heatmap.png",
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{imgs_name} Heatmap.png",
                )
            )

    plt.close()


def plot_matrix_table(
    matrix,
    row_headers,
    column_headers,
    show_index,
    show_columns,
    title,
    precision=20,
    log_file=None,
):
    # Validate dimensions
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional")

    rows, cols = matrix.shape

    if len(row_headers) != rows:
        raise ValueError(
            f"Number of row headers ({len(row_headers)}) must match matrix rows ({rows})"
        )

    if len(column_headers) != cols:
        raise ValueError(
            f"Number of column headers ({len(column_headers)}) must match matrix columns ({cols})"
        )

    # Create DataFrame
    df = pd.DataFrame(
        matrix,
        index=row_headers if show_index else None,
        columns=column_headers if show_columns else None,
    )

    # Format floating point numbers
    if np.issubdtype(matrix.dtype, np.floating):
        df = df.round(precision)

    # Display with title
    print_log(log_file, f"\n{title}")
    print_log(log_file, "=" * len(title))
    print_log(log_file, df.to_string())
    print_log(log_file, "")

    return df
