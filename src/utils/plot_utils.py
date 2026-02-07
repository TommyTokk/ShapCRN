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
    simulation_data: np.ndarray,
    colnames: list,
    img_dir_path: str="./imgs",
    img_name: str="simulation",
    log_file=None,
    ss_time: float =None,
) -> None:
    """
    Visualize simulation results and save as a static PNG image.

    This function creates a matplotlib plot of time-series simulation data with automatic
    legend layout optimization based on the number of species. The plot is saved to disk
    and the figure is closed to free memory.

    Parameters
    ----------
    simulation_data : numpy.ndarray
        Structured array containing simulation results.
        Expected shape: (n_timepoints, n_species+1) where first column is time.
    colnames : list of str
        Column names from the simulation. First element should be 'time',
        remaining elements are species names that will be plotted.
    img_dir_path : str, optional
        Directory path where the image will be saved, by default "./imgs"
        Will be created if it doesn't exist.
    img_name : str, optional
        Filename for the saved image, by default "simulation"
        Extension .png will be added automatically if not present.
    log_file : file, optional
        File object for logging the save location, by default None
    ss_time : float, optional
        Steady-state time marker (currently unused but reserved for future use),
        by default None

    Returns
    -------
    None
        Plot is saved to disk and matplotlib figure is closed.

    Notes
    -----
    Legend layout is automatically optimized based on number of species:
    - ≤3 species: 1 column
    - 4-8 species: 2 columns
    - 9-15 species: 3 columns
    - 16-24 species: 4 columns
    - >24 species: ceil(n_species/10) columns

    The legend is positioned below the plot with dynamic spacing adjustment
    to prevent overlap with the graph.

    Examples
    --------
    Plot simulation results with default settings:
    >>> results, _, colnames = simulate(rr_model)
    >>> plot_results(results, colnames)

    Save to custom directory with specific name:
    >>> plot_results(
    ...     simulation_data=results,
    ...     colnames=colnames,
    ...     img_dir_path='output/plots',
    ...     img_name='model_timecourse',
    ...     log_file=log
    ... )

    See Also
    --------
    plot_results_interactive : Create interactive Plotly visualization
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
    simulation_data: np.ndarray,
    colnames: list,
    model_name: str,
    html_dir_path:str="./imgs",
    html_name:str="interactive_model_simulation",
    log_file=None,
    ss_time:float=None,
    show_plot:bool=False,
    height:int=600,
    width:int=1000,
):
    """
    Create interactive Plotly visualization of simulation results and save as HTML.

    This function generates an interactive time-series plot with hover tooltips, automatic
    legend layout optimization, and optional steady-state markers. The plot is saved as an
    HTML file and can optionally be displayed in a browser.

    Parameters
    ----------
    simulation_data : numpy.ndarray
        Structured array containing simulation results.
        Expected shape: (n_timepoints, n_species+1) where first column is time.
    colnames : list of str
        Column names from the simulation. First element should be 'time',
        remaining elements are species names that will be plotted.
    model_name : str
        Name of the model being visualized. Used to create a subdirectory
        within html_dir_path for organizing output files.
    html_dir_path : str, optional
        Base directory path where the HTML file will be saved, by default "./imgs"
        Final save location: {html_dir_path}/{model_name}/{html_name}
    html_name : str, optional
        Filename for the HTML output, by default "interactive_model_simulation"
        Extension .html will be added automatically if not present.
    log_file : file, optional
        File object for logging messages and errors, by default None
    ss_time : float, optional
        Steady-state time value. If provided, adds a vertical dashed red line
        with "Steady State" annotation, by default None
    show_plot : bool, optional
        If True, opens the plot in the default web browser after saving,
        by default False
    height : int, optional
        Plot height in pixels, by default 600
        May be automatically adjusted for horizontal legends.
    width : int, optional
        Plot width in pixels, by default 1000
        May be automatically adjusted for vertical legends.

    Returns
    -------
    plotly.graph_objects.Figure
        The created Plotly figure object, which can be further customized or displayed.

    Notes
    -----
    Legend layout is automatically optimized based on number of species:
    - ≤10 species: Vertical legend on the right (width increased by 150px)
    - >10 species: Horizontal legend below plot (height increased by 80-200px)

    The function handles errors gracefully:
    - Logs save/display failures without raising exceptions
    - Returns the figure object even if save/display fails

    Interactive features:
    - Hover tooltips show time and concentration for each species
    - Unified hover mode (vertical line across all traces)
    - Gridlines for easier value reading
    - Zoomable and pannable plot area

    Examples
    --------
    Create basic interactive plot:
    >>> results, _, colnames = simulate(rr_model)
    >>> fig = plot_results_interactive(results, colnames, 'MyModel')

    Include steady-state marker and display in browser:
    >>> fig = plot_results_interactive(
    ...     simulation_data=results,
    ...     colnames=colnames,
    ...     model_name='BIOMD0000000623',
    ...     html_dir_path='output/interactive',
    ...     html_name='ko_analysis',
    ...     ss_time=125.5,
    ...     show_plot=True,
    ...     log_file=log
    ... )

    Customize dimensions for large models:
    >>> fig = plot_results_interactive(
    ...     simulation_data=results,
    ...     colnames=colnames,
    ...     model_name='LargeModel',
    ...     height=800,
    ...     width=1400
    ... )

    See Also
    --------
    plot_results : Create static matplotlib PNG visualization
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
        raise IOError(f"Failed to save interactive plot to {html_file_path}: {str(e)}")

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
def plot_heatmap(data: pd.DataFrame, y_labels:list, x_labels:list, colnames_to_index:dict, **kwargs) -> tuple:
    """
    Create a customizable seaborn heatmap and optionally save to disk.

    This function generates a heatmap visualization using seaborn with support for custom
    colormaps, annotations, and automatic handling of NaN values (displayed as black).
    The plot can be saved as a high-resolution PNG file.

    Parameters
    ----------
    data : pandas.DataFrame
        2D data to visualize as a heatmap.
        Typically contains variation or distance metrics between species.
    y_labels : list of str
        Labels for the y-axis (rows), typically knockout species IDs.
    x_labels : list of str
        Labels for the x-axis (columns), typically affected species IDs.
    colnames_to_index : dict
        Mapping from column names to their indices.
        Currently unused but reserved for future functionality.
    **kwargs : dict
        Optional keyword arguments for customization.

    Other Parameters
    ----------------
    title : str, optional
        Title for the heatmap, by default "Heatmap"
    cmap : str, optional
        Matplotlib colormap name, by default "viridis"
        Ignored if 'colors' is provided.
    colors : list of str, optional
        Custom color list for creating a LinearSegmentedColormap.
        Overrides 'cmap' if provided.
    figsize : tuple of (float, float), optional
        Figure size in inches (width, height), by default (12, 8)
    save_path : str, optional
        Directory path where the image will be saved, by default "./imgs"
        If None or empty, the plot will not be saved.
    img_name : str, optional
        Filename for the saved image, by default "heatmap.png"
        Should include file extension (.png recommended).
    annot : bool, optional
        If True, display data values in each cell, by default False

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The created matplotlib figure and axes objects, which can be further customized.

    Notes
    -----
    - NaN values are automatically displayed as black in the heatmap
    - The plot uses tight_layout() for optimal spacing
    - X-axis labels are rotated 45° for better readability
    - Saved images use 300 DPI for high quality
    - Debug information is printed to console during save operations

    Examples
    --------
    Create basic heatmap with default settings:
    >>> df = pd.DataFrame(np.random.rand(5, 10))
    >>> fig, ax = plot_heatmap(df, ['S1', 'S2', 'S3', 'S4', 'S5'], 
    ...                        [f'T{i}' for i in range(10)], {})

    Create custom heatmap with annotations and custom colormap:
    >>> fig, ax = plot_heatmap(
    ...     data=variations_df,
    ...     y_labels=ko_species_list,
    ...     x_labels=all_species,
    ...     colnames_to_index={},
    ...     title='Knockout Impact Analysis',
    ...     colors=['blue', 'white', 'red'],
    ...     figsize=(15, 10),
    ...     annot=True,
    ...     save_path='output/figures',
    ...     img_name='ko_heatmap.png'
    ... )

    Use diverging colormap for signed data:
    >>> fig, ax = plot_heatmap(
    ...     data=relative_changes,
    ...     y_labels=ko_ids,
    ...     x_labels=species_ids,
    ...     colnames_to_index={},
    ...     cmap='RdBu_r',
    ...     title='Relative Concentration Changes'
    ... )

    See Also
    --------
    seaborn.heatmap : Underlying visualization function
    matplotlib.colors.LinearSegmentedColormap : Custom colormap creation
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


