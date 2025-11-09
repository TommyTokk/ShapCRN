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
from matplotlib.colors import LinearSegmentedColormap

from utils.utils import print_log


# KEEP
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


# KEEP
def plot_results_interactive(
    simulation_data,
    colnames,
    model_name,
    html_dir_path="./imgs",
    html_name="interactive_model_simulation",
    log_file=None,
    ss_time=None,
    show_plot=False,
    height=600,
    width=1000,
):
    """
    Create interactive visualization of simulation results using Plotly and save to HTML file.

    Args:
        simulation_data: NumPy array with simulation results
        colnames: List of column names (first should be time, rest are species)
        html_dir_path: Directory where to save the HTML file (default: "./imgs")
        html_name: Name of the HTML file without extension (default: "interactive_model_simulation")
        log_file: Log file for messages (optional)
        ss_time: Steady state time marker (optional, adds vertical line)
        show_plot: Whether to display the plot in browser (default: False)
        height: Plot height in pixels (default: 600)
        width: Plot width in pixels (default: 1000)

    Returns:
        plotly.graph_objects.Figure: The created figure object
    """
    species_names = colnames[1:]

    # Ensure html_name has .html extension
    if not html_name.endswith(".html"):
        html_name = f"{html_name}.html"

    saving_path = os.path.join(html_dir_path, model_name)
    # Create directory if it doesn't exist
    os.makedirs(saving_path, exist_ok=True)

    # Combine directory path and filename
    html_file_path = os.path.join(saving_path, html_name)

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
            text=f"Interactive Simulation: {html_name.replace('.html', '')}",
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
        except NameError as ne:
            print_log(log_file, f"[WARNING] Error during saving the plot: {ne}")

    return fig


# KEEP
def plot_heatmap(data, y_labels, x_labels, colnames_to_index, **kwargs):
    """
    Simple heatmap plotting with customizable color mapping.

    Args:
        data: Pandas DataFrame
        y_labels (list): Labels for y-axis (rows)
        x_labels (list): Labels for x-axis (columns)
        **kwargs: Optional customization parameters

    Optional kwargs:
        title (str): Title for the heatmap. Default: "Heatmap"
        cmap (str): Colormap name. Default: "viridis"
        colors (list): Custom colors for colormap. Overrides cmap if provided
        figsize (tuple): Figure size. Default: (10, 6)
        save_path (str): Directory path to save figure. Default: None (don't save)
        img_name (str): Filename for saved image. Default: "heatmap.png"
        annot (bool): Show values in cells. Default: False

    Returns:
        fig, ax: matplotlib figure and axes objects
    """

    # Setting the NaN values

    # Default parameters
    defaults = {
        "title": "Heatmap",
        "cmap": "viridis",
        "figsize": (12, 8),
        "annot": False,
        "save_path": "./imgs",
        "img_name": "heatmap.png",
    }

    # Update defaults with user kwargs
    params = {**defaults, **kwargs}

    # Create custom colormap if colors provided
    if "colors" in kwargs:
        cmap = LinearSegmentedColormap.from_list("custom", kwargs["colors"])
    else:
        cmap = params["cmap"]

    # Set NaN values to be displayed as black
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap).copy()
    else:
        cmap = cmap.copy()
    cmap.set_bad(color="black")

    # Create plot
    fig, ax = plt.subplots(figsize=params["figsize"])

    # Create heatmap
    sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        annot=params["annot"],
        cmap=cmap,
        ax=ax,
    )

    # Set title and labels
    ax.set_title(params["title"], fontsize=14, pad=20)
    ax.set_xlabel("Species")
    ax.set_ylabel("Knocked-out Ids")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save if requested
    if params["save_path"]:
        # Create directory if it doesn't exist
        os.makedirs(params["save_path"], exist_ok=True)

        # Construct full file path
        full_file_path = os.path.join(params["save_path"], params["img_name"])

        # Debug logging
        print(f"Saving to directory: {params['save_path']}")
        print(f"File name: {params['img_name']}")
        print(f"Full file path: {full_file_path}")

        # Verify that the directory exists and the full path is not a directory
        # if os.path.isdir(full_file_path):
        #     raise ValueError(f"Cannot save file: '{full_file_path}' is a directory, not a file path")

        # Save the figure
        plt.savefig(full_file_path, bbox_inches="tight", dpi=300)
        print(f"Heatmap saved successfully to: {full_file_path}")

    return fig, ax
