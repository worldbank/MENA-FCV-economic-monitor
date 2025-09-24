import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_dual_metrics_by_country(
    data: pd.DataFrame,
    metrics_to_plot: list,
    metric_display_info: dict = None,
    sorting_metric: str = None,
    overall_title: str = 'Comparison of Metrics by Country',
    chart_subtitle: str = None,
    source_text: str = 'Source: ACLED. Accessed: 2024-10-01',
    figsize: tuple = (15, 8),
    subtitle: str = None  # Keep for backward compatibility
) -> plt.Figure:
    """
    Generates a dual horizontal bar chart for two specified metrics by country.

    Args:
        data (pd.DataFrame): The input DataFrame containing 'country' and metric columns.
        metrics_to_plot (list): A list of column names (metrics) to visualize.
                                Only the first two valid metrics will be plotted.
        metric_display_info (dict, optional): A dictionary mapping metric names to
            dictionaries with 'title' and 'color' keys.
        sorting_metric (str, optional): The name of the metric to use for sorting countries.
                                        If None, countries are not sorted.
        overall_title (str, optional): The main title for the concatenated chart.
        chart_subtitle (str, optional): Subtitle displayed below the main title.
        source_text (str, optional): Source information displayed at the bottom.
        figsize (tuple, optional): Figure size (width, height). Defaults to (15, 8).
        subtitle (str, optional): Deprecated - use source_text instead.

    Returns:
        plt.Figure: A matplotlib Figure object.
    """
    if metric_display_info is None:
        metric_display_info = {}

    df_plot = data.copy()

    # Sort the data if a sorting_metric is provided
    if sorting_metric and sorting_metric in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[sorting_metric]):
        df_plot = df_plot.sort_values(by=sorting_metric, ascending=True).reset_index(drop=True)
    elif sorting_metric:
        print(f"Warning: Sorting metric '{sorting_metric}' not found or not numeric. Data will not be sorted.")

    # Filter valid metrics and take only the first two for dual plot
    valid_metrics = [m for m in metrics_to_plot if m in df_plot.columns]
    # if len(valid_metrics) < 2:
    #     print("Error: At least two valid metrics are required for a dual bar plot.")
    #     return None
    
    metrics_for_plot = valid_metrics # Take only the first two metrics

    # Create figure and subplots for two metrics
    fig, axes = plt.subplots(1, len(metrics_for_plot), figsize=figsize, sharey=True)

    # Set overall style
    plt.style.use('default')
    fig.patch.set_facecolor('white')

    countries = df_plot['country'].values
    y_pos = np.arange(len(countries))

    for i, metric in enumerate(metrics_for_plot):
        ax = axes[i]
        
        # Get custom title and color, or use defaults
        display_info = metric_display_info.get(metric, {})
        metric_title = display_info.get('title', metric.replace("_", " ").title())
        metric_color = display_info.get('color', plt.cm.Paired(i)) # Use a colormap for default colors
        
        values = df_plot[metric].values

        # Create horizontal bar chart
        bars = ax.barh(y_pos, values, color=metric_color, alpha=0.8,
                       edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            # Position text slightly to the right of bar end
            ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{np.round(value,2):,}', ha='left', va='center', fontsize=9,
                            fontweight='bold', color='black')

        # Customize subplot
        ax.set_yticks(y_pos)
        ax.set_xlabel(f'', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_title}', fontsize=12, fontweight='bold', pad=15, loc='left')

        # Only show y-axis labels on leftmost subplot

        ax.set_yticklabels(countries, fontsize=10)
        #ax.set_ylabel('Country', fontsize=11, fontweight='bold')
       

        # Style the subplot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666')
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Set x-axis to start from 0 and add some padding
        ax.set_xlim(0, max(values) * 1.15)

    # Handle backward compatibility for subtitle parameter
    if subtitle and not source_text:
        source_text = subtitle
    
    # Add overall title
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.97, ha='left', x=0.05)
    
    # Add subtitle below title if provided
    if chart_subtitle:
        fig.text(0.05, 0.90, chart_subtitle, ha='left', va='top', fontsize=12, 
                color='#555555')
    
    # Add source text at the bottom if provided
    if source_text:
        fig.text(0.05, 0.02, source_text, ha='left', va='bottom', fontsize=10, 
                color='#666666', alpha=0.8)

    # Adjust layout with proper spacing
    plt.tight_layout()
    # Ensure enough space for title, subtitle, y-axis labels, and source
    # Increased gap between subtitle and subfigure titles
    top_margin = 0.80 if chart_subtitle else 0.88
    plt.subplots_adjust(left=0.15, top=top_margin, bottom=0.12)

    return fig
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import contextily as ctx

def plot_h3_maps_with_boundaries_and_quartiles(gdf, 
                                               category_column, 
                                               measure_column, 
                                               boundary_gdf, 
                                               cmap_name='Blues', 
                                               title='Geospatial Distribution of Conflict',
                                               chart_subtitle=None,
                                               source_text=""):
    """
    Creates a multi-panel plot of H3 hexagons with a boundary, one panel for each 
    unique category. The color scale is based on global quartiles of the 
    specified measure, and a unified legend is included.

    Parameters:
    -----------
    gdf : GeoDataFrame
        The H3 data to plot. Must have a defined CRS.
    category_column : str
        The name of the column to use for splitting data into subplots.
    measure_column : str
        The name of the numeric column to use for the color scale.
    boundary_gdf : GeoDataFrame
        A GeoDataFrame containing the boundaries to be plotted on each map.
        It will be reprojected to match the CRS of `gdf`.
    cmap_name : str, optional
        The name of the colormap to use (e.g., 'Blues'). Defaults to 'Blues'.
    """
    if gdf.crs is None:
        raise ValueError("The input GeoDataFrame must have a defined Coordinate Reference System (CRS).")

    unique_categories = gdf[category_column].dropna().unique()
    num_plots = len(unique_categories)


    if num_plots == 0:
        print("No data or unique categories to plot.")
        return

    # --- Robust Quartile Calculation ---
    data_to_bin = gdf[measure_column].dropna()
    if data_to_bin.empty:
        print("Warning: The measure column is empty. Cannot plot quartiles.")
        return
    
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    bin_edges = data_to_bin.quantile(quantiles).tolist()

    # Ensure bin edges are strictly increasing to prevent BoundaryNorm errors
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i-1]:
            bin_edges[i] = bin_edges[i-1] + 1e-9  # Add a tiny increment

    # Check if a valid range exists after fixing
    if bin_edges[0] == bin_edges[-1]:
        print("Warning: All data values are identical. Cannot create quartile bins.")
        return
    
    # --- Create Discrete Colormap and Normalization ---
    # Get the base colormap and create a new ListedColormap with 4 colors
    base_cmap = plt.get_cmap(cmap_name)
    cmap_discrete = mcolors.ListedColormap([base_cmap(i / 3.0) for i in range(4)])
    
    # Use the discrete colormap with BoundaryNorm for clear binning
    norm = mcolors.BoundaryNorm(bin_edges, cmap_discrete.N, extend='neither')

    # --- Plotting Setup ---
    ncols = min(num_plots, 3)
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4), squeeze=False)

    gdf_web_mercator = gdf.to_crs(epsg=3857)
    boundary_gdf_proj = boundary_gdf.to_crs(epsg=3857)

    # Reproject the boundary GDF once
    boundary_gdf_proj = boundary_gdf.to_crs(gdf_web_mercator.crs)

    for i, category in enumerate(unique_categories):
        ax = axes.flatten()[i]
        subset_gdf = gdf_web_mercator[gdf_web_mercator[category_column] == category]     
        
        # Plot H3 hexagons only if the subset is not empty
        if not subset_gdf.empty:
            subset_gdf.plot(
                ax=ax,
                column=measure_column,
                cmap=cmap_discrete,  # Use the discrete colormap
                norm=norm,
                legend=False,
                edgecolor='white',
                linewidth=0.5
            )
        
        # Plot boundary
        boundary_gdf_proj.plot(
            ax=ax, 
            facecolor='none', 
            edgecolor='grey', 
            linewidth=0.2
        )
        provider = ctx.providers.CartoDB.PositronNoLabels
        
        ctx.add_basemap(ax, source=provider, crs=gdf_web_mercator.crs)

        
        ax.set_title("1st January 2022 - 31st December 2024", loc='left')
        ax.set_axis_off()

    # Turn off any unused axes
    for j in range(num_plots, nrows * ncols):
        axes.flatten()[j].set_visible(False)
    
    # Create and position the legend
    legend_elements = [
        Patch(facecolor=cmap_discrete(i), edgecolor='none', alpha=0.8, 
              label=f'Quartile {i+1} ({bin_edges[i]:.2f} - {bin_edges[i+1]:.2f})')
        for i in range(4)
    ]
    fig.legend(handles=legend_elements, title=f"Conflict Intensity Index Quartiles", 
               loc='center right', ncol=1, bbox_to_anchor=(0.9, 0.2), frameon=False)
    
    # Add main title
    plt.suptitle(f"{title}", fontsize=16, fontweight='bold', y=0.97, ha='left', x=0.05)
    
    # Add subtitle below title if provided
    if chart_subtitle:
        fig.text(0.05, 0.88, chart_subtitle, ha='left', va='top', fontsize=12, 
                color='#555555')
    
    # Add source text at the bottom if provided
    if source_text:
        fig.text(0.12, -0.08, source_text, ha='left', va='bottom', fontsize=9, 
                color='#666666', alpha=0.8)

    plt.tight_layout()
    plt.show()