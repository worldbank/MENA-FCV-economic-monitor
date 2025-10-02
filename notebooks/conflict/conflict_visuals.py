import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import StrMethodFormatter

def plot_dual_metrics_by_country(
    data: pd.DataFrame,
    metrics_to_plot: list,
    category_column: str = 'country',
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

    countries = df_plot[category_column].values
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
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=1.02, ha='left', x=0.05)
    
    # Add subtitle below title if provided (even closer to the title)
    if chart_subtitle:
        fig.text(0.05, 0.98, chart_subtitle, ha='left', va='top', fontsize=12,
                color='#555555')
    
    # Add source text at the bottom if provided
    if source_text:
        # Bring source closer to the charts
        fig.text(0.05, 0.000001, source_text, ha='left', va='bottom', fontsize=10, 
                color='#666666', alpha=0.8)

    # Adjust layout with proper spacing
    plt.tight_layout()
    # Ensure enough space for title, subtitle, y-axis labels, and source
    # Reduce gap between subtitle and charts by moving subplot area up
    # Very slight increase in gap between subtitle and charts
    top_margin = 0.910 if chart_subtitle else 0.96
    # Reduce bottom margin so subplots sit closer to the source
    plt.subplots_adjust(left=0.15, top=top_margin, bottom=0.08)

    return fig


def _ensure_time_agg(
    df: pd.DataFrame,
    date_col: str,
    metrics: list,
    freq: str | None = None,
) -> pd.DataFrame:
    """Ensure dataframe is sorted by date and optionally aggregated to a time frequency.

    Args:
        df: Input DataFrame.
        date_col: Name of datetime column.
        metrics: Metric columns to keep/aggregate.
        freq: Pandas resample frequency like 'YS' (annual) or 'MS' (monthly). If None, no resample.

    Returns:
        DataFrame with [date_col] + metrics, optionally aggregated by freq.
    """
    df2 = df.copy()
    if date_col not in df2.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame")
    df2[date_col] = pd.to_datetime(df2[date_col])
    keep_metrics = [m for m in metrics if m in df2.columns]
    if not keep_metrics:
        raise ValueError("None of the requested metrics are in the DataFrame")
    df2 = df2[[date_col] + keep_metrics].sort_values(date_col)

    if not freq:
        return df2

    grouped = (
        df2.set_index(date_col)
           .resample(freq)[keep_metrics]
           .sum(min_count=1)
           .reset_index()
    )
    return grouped


def plot_dual_bars_by_date(
    data: pd.DataFrame,
    date_col: str = 'event_date',
    metrics_to_plot: list = ['nrEvents', 'nrFatalities'],
    metric_display_info: dict | None = None,
    overall_title: str = 'Conflict Events and Fatalities Over Time',
    chart_subtitle: str | None = None,
    source_text: str | None = None,
    freq: str | None = 'YS',
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Create a dual bar chart with time on x-axis and metrics on y-axis (one subplot per metric).

    Colors mirror plot_dual_metrics_by_country defaults: nrFatalities -> steelblue, nrEvents -> orange.

    Args:
        data: DataFrame containing at least [date_col] and metric columns.
        date_col: Name of the datetime column.
        metrics_to_plot: List of metric column names to plot (first N will be used).
        metric_display_info: Optional mapping {metric: {'title': str, 'color': str}}.
        overall_title: Suptitle for the figure.
        chart_subtitle: Optional subtitle under the title.
        source_text: Optional footer text.
        freq: Pandas resample frequency (e.g., 'YS' annual, 'MS' monthly). None to skip resampling.
        figsize: Figure size (w, h).

    Returns:
        Matplotlib Figure.
    """
    if metric_display_info is None:
        # Align colors with plot_dual_metrics_by_country usage
        metric_display_info = {
            'nrFatalities': {'title': 'Fatalities', 'color': 'steelblue'},
            'nrEvents': {'title': 'Events', 'color': 'orange'},
        }

    # Prepare data
    valid_metrics = [m for m in metrics_to_plot if m in data.columns]
    if not valid_metrics:
        raise ValueError("No valid metrics to plot found in DataFrame")
    dfp = _ensure_time_agg(data, date_col=date_col, metrics=valid_metrics, freq=freq)

    # Figure
    fig, axes = plt.subplots(1, len(valid_metrics), figsize=figsize, sharex=False)
    if len(valid_metrics) == 1:
        axes = [axes]

    plt.style.use('default')
    fig.patch.set_facecolor('white')

    # Determine bar width based on median spacing
    dates = pd.to_datetime(dfp[date_col].values)
    if len(dates) > 1:
        deltas = np.diff(dates).astype('timedelta64[D]').astype(float)
        delta_days = np.median(deltas) if len(deltas) else 30.0
        # Matplotlib supports numpy timedelta64 directly for width
        bar_width = np.timedelta64(int(max(1, round(delta_days * 0.8))), 'D')
    else:
        bar_width = np.timedelta64(20, 'D')

    for i, metric in enumerate(valid_metrics):
        ax = axes[i]
        info = metric_display_info.get(metric, {})
        color = info.get('color', plt.cm.Paired(i))
        title = info.get('title', metric.replace('_', ' ').title())

        ax.bar(dfp[date_col], dfp[metric], color=color, edgecolor='white', linewidth=0.5, width=bar_width)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=12)
        ax.set_ylabel('')
        # Remove x-axis label to declutter (no 'Date' label)
        ax.set_xlabel('')

        # Style similar to plot_dual_metrics_by_country
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666')
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Format y-axis tick labels with comma separators (e.g., 12,345)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # Titles and footer
    # Tighter title/subtitle spacing
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=1.1, ha='left', x=0.05)
    # Reduce gap between title and subtitle
    if chart_subtitle:
        fig.text(0.05, 1.05, chart_subtitle, ha='left', va='top', fontsize=12, color='#555555')
    if source_text:
        # Bring source closer to charts
        fig.text(0.05, 0.0001, source_text, ha='left', va='bottom', fontsize=10, color='#666666', alpha=0.8)

    plt.tight_layout()
    # Bring subplots closer to the titles and add a touch more space between columns
    # Very slight increase in gap between subtitle and charts
    top_margin = 0.945 if chart_subtitle else 0.975
    plt.subplots_adjust(left=0.08, top=top_margin, bottom=0.08, wspace=0.30)
    return fig


def plot_dual_bars_by_date_by_region(
    data: pd.DataFrame,
    region_col: str = 'wb_region',
    date_col: str = 'event_date',
    metrics_to_plot: list = ['nrFatalities', 'nrEvents'],
    metric_display_info: dict | None = None,
    overall_title: str = 'Conflict Events and Fatalities Over Time by Region',
    chart_subtitle: str | None = None,
    source_text: str | None = None,
    freq: str | None = 'YS',
    row_height: float = 3.4,
    col_width: float = 6.5,
    sharey_by_metric: bool = False,
) -> plt.Figure:
    """Facet the dual time-bar visuals by region: one row per region, one column per metric.

    Mirrors the style of plot_dual_bars_by_date for each region.

    Args:
        data: DataFrame with at least [region_col, date_col] and metric columns.
        region_col: Column to facet rows by (e.g., 'wb_region').
        date_col: Datetime column.
        metrics_to_plot: Metrics to show as separate columns per row.
        metric_display_info: Optional mapping {metric: {'title': str, 'color': str}}.
        overall_title: Figure-level title.
        chart_subtitle: Optional subtitle.
        source_text: Optional footer text.
        freq: Optional resample frequency (e.g., 'YS'); None to skip re-aggregation.
        row_height: Height per region row.
        col_width: Width per metric column.
        sharey_by_metric: If True, share y-axis per metric across all regions.

    Returns:
        Matplotlib Figure.
    """
    if region_col not in data.columns:
        raise ValueError(f"region_col '{region_col}' not in DataFrame")
    if date_col not in data.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame")

    # Display defaults consistent with other visuals
    if metric_display_info is None:
        metric_display_info = {
            'nrFatalities': {'title': 'Fatalities', 'color': 'steelblue'},
            'nrEvents': {'title': 'Events', 'color': 'orange'},
        }

    valid_metrics = [m for m in metrics_to_plot if m in data.columns]
    if not valid_metrics:
        raise ValueError("No valid metrics to plot found in DataFrame")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[region_col, date_col])

    regions = (
        df[region_col]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    regions = [r for r in regions if r and r.lower() != 'nan']
    regions.sort()
    if not regions:
        raise ValueError("No regions found to facet")

    nrows = len(regions)
    ncols = len(valid_metrics)

    # Optionally compute shared y-limits per metric across all regions
    shared_ylim = {}
    if sharey_by_metric:
        for m in valid_metrics:
            # aggregate (resample) before checking max if needed
            try:
                tmp = (
                    df[[region_col, date_col, m]]
                    .rename(columns={m: '__m'})
                )
                if freq:
                    tmp = (
                        tmp.set_index(date_col)
                           .groupby(region_col, group_keys=False)
                           .resample(freq)['__m']
                           .sum(min_count=1)
                           .reset_index()
                    )
                vmax = float(tmp['__m'].max()) if not tmp['__m'].empty else 0.0
            except Exception:
                vmax = float(df[m].max()) if m in df.columns else 0.0
            shared_ylim[m] = (0.0, vmax * 1.15 if vmax > 0 else 1.0)

    # Figure size based on requested per-row/column dimensions
    figsize = (max(10.0, ncols * col_width), max(4.0, nrows * row_height))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    plt.style.use('default')
    fig.patch.set_facecolor('white')

    for i, region in enumerate(regions):
        sub = df[df[region_col] == region]
        if sub.empty:
            for j in range(ncols):
                axes[i, j].set_visible(False)
            continue

        # Pre-aggregate once per region to avoid repeated work per metric
        sub_agg = _ensure_time_agg(sub, date_col=date_col, metrics=valid_metrics, freq=freq)

        # Determine a reasonable bar width (mirrors plot_dual_bars_by_date)
        dates = pd.to_datetime(sub_agg[date_col].values)
        if len(dates) > 1:
            deltas = np.diff(dates).astype('timedelta64[D]').astype(float)
            delta_days = np.median(deltas) if len(deltas) else 30.0
            bar_width = np.timedelta64(int(max(1, round(delta_days * 0.8))), 'D')
        else:
            bar_width = np.timedelta64(20, 'D')

        for j, metric in enumerate(valid_metrics):
            ax = axes[i, j]
            info = metric_display_info.get(metric, {})
            color = info.get('color', plt.cm.Paired(j))
            title = info.get('title', metric.replace('_', ' ').title())

            ax.bar(sub_agg[date_col], sub_agg[metric], color=color, edgecolor='white', linewidth=0.5, width=bar_width)

            # Titles: put region in the subplot title (top), alongside the metric
            title_text = f"{region} — {title}"
            ax.set_title(title_text, fontsize=11, fontweight='bold', loc='left', pad=8)
            ax.set_ylabel('')

            # Style similar to the base function
            # Remove x-axis label to declutter (no 'Date' label)
            ax.set_xlabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.tick_params(colors='#666666', labelsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

            # Shared y scaling per metric, if requested
            if sharey_by_metric and metric in shared_ylim:
                ax.set_ylim(shared_ylim[metric])

    # Titles and footer
    # Tighter title/subtitle spacing
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.99, ha='left', x=0.05)
    if chart_subtitle:
        fig.text(0.05, 0.965, chart_subtitle, ha='left', va='top', fontsize=12, color='#555555')
    if source_text:
        # Bring source closer to charts
        fig.text(0.1, 0.1, source_text, ha='left', va='bottom', fontsize=10, color='#666666', alpha=0.8)

    plt.tight_layout()
    # Reduce the gap between subtitle and charts; and between charts and source
    # Very slight increase in gap between subtitle and charts
    top_margin = 0.945 if chart_subtitle else 0.975
    plt.subplots_adjust(left=0.08, right=0.98, top=top_margin, bottom=0.08, wspace=0.25, hspace=0.35)
    return fig


def plot_annual_country_bars(
    data: pd.DataFrame,
    country_col: str = 'country',
    date_col: str = 'event_date',
    metrics: list = ['nrEvents', 'nrFatalities'],
    metric_display_info: dict | None = None,
    freq: str = 'YS',
    ncols: int = 4,
    figsize: tuple = (18, 12),
    overall_title: str = 'Annual Conflict Trends by Country',
    chart_subtitle: str | None = None,
    source_text: str | None = None,
    sharey: bool = False,
) -> plt.Figure:
    """Plot annual bar subplots for all countries.

    Each subplot shows grouped bars per year for the requested metrics (e.g., Events and Fatalities).

    Args:
        data: DataFrame with at least [country_col, date_col] and metric columns.
        country_col: Country column name.
        date_col: Datetime column name.
        metrics: Metric columns to show as grouped bars per year.
        metric_display_info: Optional mapping {metric: {'title': str, 'color': str}}.
        freq: Resample frequency for aggregation (default 'YS' annual). If None, assumes pre-aggregated.
        ncols: Number of subplot columns.
        figsize: Figure size (w, h).
        overall_title: Figure-level title.
        chart_subtitle: Optional subtitle below title.
        source_text: Optional footer text.
        sharey: Whether to share the y-axis across subplots.

    Returns:
        Matplotlib Figure
    """
    if metric_display_info is None:
        metric_display_info = {
            'nrFatalities': {'title': 'Fatalities', 'color': 'steelblue'},
            'nrEvents': {'title': 'Events', 'color': 'orange'},
        }

    # Validate metrics
    valid_metrics = [m for m in metrics if m in data.columns]
    if not valid_metrics:
        raise ValueError("No valid metrics found in DataFrame for plotting")

    df = data.copy()
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame")
    if country_col not in df.columns:
        raise ValueError(f"country_col '{country_col}' not found in DataFrame")

    # Coerce date column and country labels
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[country_col] = df[country_col].astype(str)
    keep_cols = [country_col, date_col] + valid_metrics
    df = df[keep_cols].dropna(subset=[country_col, date_col])

    # Aggregate to annual if requested
    if freq:
        try:
            df = (
                df.set_index(date_col)
                  .groupby(country_col, group_keys=False)
                  .resample(freq)[valid_metrics]
                  .sum(min_count=1)
                  .reset_index()
            )
        except Exception:
            # Fallback: build a year column then group
            df['__year'] = df[date_col].dt.to_period('Y').dt.to_timestamp()
            df = df.groupby([country_col, '__year'])[valid_metrics].sum(min_count=1).reset_index()
            df = df.rename(columns={'__year': date_col})
    else:
        # Assume pre-aggregated; if multiple rows per country-year, collapse
        df['__year'] = df[date_col].dt.to_period('Y').dt.to_timestamp()
        df = df.groupby([country_col, '__year'])[valid_metrics].sum(min_count=1).reset_index()
        df = df.rename(columns={'__year': date_col})
    df = df.sort_values([country_col, date_col])

    countries = df[country_col].dropna().unique().tolist()
    if not countries:
        raise ValueError("No countries found to plot")

    n = len(countries)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    plt.style.use('default')
    fig.patch.set_facecolor('white')

    # Legend handles (collect once)
    legend_handles = {}

    for i, country in enumerate(countries):
        ax = axes.flatten()[i]
        sub = df[df[country_col] == country]
        if sub.empty:
            ax.set_visible(False)
            continue

        dates = pd.to_datetime(sub[date_col].values)
        # Determine spacing for bar width and offsets
        if len(dates) > 1:
            deltas = np.diff(dates).astype('timedelta64[D]').astype(float)
            delta_days = np.median(deltas) if len(deltas) else 365.0
        else:
            delta_days = 365.0

        group_span = int(max(60, round(delta_days * 0.8)))  # total span for a year group
        single_width = int(max(20, round(group_span / max(1, len(valid_metrics) + 1))))
        offset_step = int(single_width)

        # Base positions as datetimes; create offsets for grouped bars
        base_pos = sub[date_col]
        positions = []
        for idx_m, metric in enumerate(valid_metrics):
            offset_days = (idx_m - (len(valid_metrics) - 1) / 2) * offset_step
            positions.append(base_pos + pd.to_timedelta(offset_days, unit='D'))

        # Plot each metric as bars
        for idx_m, metric in enumerate(valid_metrics):
            info = metric_display_info.get(metric, {})
            color = info.get('color', plt.cm.Paired(idx_m))
            title = info.get('title', metric.replace('_', ' ').title())

            ax.bar(
                positions[idx_m],
                sub[metric].values,
                color=color,
                edgecolor='white',
                linewidth=0.5,
                width=pd.to_timedelta(single_width, unit='D'),
                label=title,
                alpha=0.9,
            )

            # Save legend handle once
            if title not in legend_handles:
                legend_handles[title] = plt.Rectangle((0, 0), 1, 1, color=color)

        # Styling per subplot
        ax.set_title(str(country), fontsize=11, fontweight='bold', loc='left', pad=8)
        # Ensure no explicit x-axis label on subplots
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        # Improve x-axis: show years only
        try:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        except Exception:
            pass
        ax.xaxis.set_tick_params(rotation=0)

    # Hide any unused axes
    for j in range(n, nrows * ncols):
        axes.flatten()[j].set_visible(False)

    # Titles and footer
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.975, ha='left', x=0.05)
    if chart_subtitle:
        fig.text(0.05, 0.965, chart_subtitle, ha='left', va='top', fontsize=12, color='#555555')
    if source_text:
        # Bring source closer to charts
        fig.text(0.05, 0.05, source_text, ha='left', va='bottom', fontsize=10, color='#666666', alpha=0.8)

    # Single legend for the whole figure
    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
        )

    plt.tight_layout()
    # Bring subplots closer to the titles and slightly increase inter-subplot spacing
    # Very slight increase in gap between subtitle and charts
    top_margin = 0.945 if chart_subtitle else 0.975
    plt.subplots_adjust(left=0.06, right=0.98, top=top_margin, bottom=0.08, wspace=0.30, hspace=0.40)
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

        
        #ax.set_title("1st January 2022 - 31st December 2024", loc='left')
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


def plot_grouped_bars_by_region_eventtype(
    data: pd.DataFrame,
    region_col: str = 'region_code',
    type_col: str = 'event_type',
    value_col: str = 'fatalities',
    region_order: list | None = None,
    type_order: list | None = None,
    type_colors: dict | None = None,
    overall_title: str = 'Fatalities by Conflict Type and Region',
    chart_subtitle: str | None = None,
    source_text: str | None = None,
    figsize: tuple = (14, 7),
    rotate_xticks: int = 0,
    annotate: bool = True,
    nan_to_zero: bool = True,
    label_max_chars_per_line: int = 12,
    show_group_separators: bool = True,
    separator_color: str = '#DDDDDD',
    separator_linewidth: float = 0.6,
) -> plt.Figure:
    """Grouped bar chart of <value_col> by <type_col> within each <region_col>.

    Typical usage: pass a tidy DataFrame with columns [region_code, event_type, fatalities]
    (e.g., output of df.groupby(['region_code','event_type'])['fatalities'].sum().reset_index()).

    Args:
        data: Input DataFrame.
        region_col: Column representing regions (x-axis groups).
        type_col: Column representing conflict types (bars within each region group).
        value_col: Numeric column for bar heights (e.g., 'fatalities').
        region_order: Optional explicit order for regions. Defaults to descending total value.
        type_order: Optional explicit order for conflict types. Defaults to alphabetical.
        type_colors: Optional mapping {type -> color}. If None, a colormap is used.
        overall_title: Figure-level title.
        chart_subtitle: Optional subtitle.
        source_text: Optional footer text.
        figsize: Figure size (w, h).
        rotate_xticks: Degrees to rotate x tick labels.
        annotate: Whether to annotate bars with values.
        nan_to_zero: Fill missing combinations with 0.

    Returns:
        Matplotlib Figure.
    """
    if region_col not in data.columns:
        raise ValueError(f"region_col '{region_col}' not in DataFrame")
    if type_col not in data.columns:
        raise ValueError(f"type_col '{type_col}' not in DataFrame")
    if value_col not in data.columns:
        raise ValueError(f"value_col '{value_col}' not in DataFrame")

    df = data[[region_col, type_col, value_col]].copy()
    # Clean labels
    df[region_col] = df[region_col].astype(str)
    df[type_col] = df[type_col].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    # Aggregate in case there are duplicates
    df = df.groupby([region_col, type_col], as_index=False)[value_col].sum(min_count=1)

    # Orders
    if region_order is None:
        region_totals = df.groupby(region_col)[value_col].sum().sort_values(ascending=False)
        region_order = region_totals.index.tolist()
    if type_order is None:
        type_order = sorted(df[type_col].dropna().unique().tolist())

    # Complete grid of region x type
    full_idx = pd.MultiIndex.from_product([region_order, type_order], names=[region_col, type_col])
    df_full = df.set_index([region_col, type_col]).reindex(full_idx)
    if nan_to_zero:
        df_full[value_col] = df_full[value_col].fillna(0)
    df_full = df_full.reset_index()

    regions = region_order
    types = type_order
    n_regions = len(regions)
    n_types = len(types)

    # Colors
    if type_colors is None:
        cmap = plt.get_cmap('tab20') if n_types > 10 else plt.get_cmap('Paired')
        type_colors = {t: cmap(i % cmap.N) for i, t in enumerate(types)}
    else:
        # Ensure all types have a color
        fallback_cmap = plt.get_cmap('tab20')
        for i, t in enumerate(types):
            type_colors.setdefault(t, fallback_cmap(i % fallback_cmap.N))

    # X positions per region
    x = np.arange(n_regions)
    # Group width and individual bar width
    group_width = min(0.85, 0.15 * n_types + 0.35)  # heuristic to keep groups compact
    bar_width = group_width / max(1, n_types)
    offsets = (np.arange(n_types) - (n_types - 1) / 2) * bar_width

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.style.use('default')
    fig.patch.set_facecolor('white')

    # Plot bars per type
    legend_handles = []
    for i, t in enumerate(types):
        sub = df_full[df_full[type_col] == t]
        heights = sub[value_col].values
        bars = ax.bar(x + offsets[i], heights, width=bar_width * 0.95,
                      color=type_colors.get(t), edgecolor='white', linewidth=0.6,
                      label=t, alpha=0.9)
        if annotate:
            for b, v in zip(bars, heights):
                if pd.notna(v) and v > 0:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                            f"{int(round(v)):,}", ha='center', va='bottom', fontsize=8, color='#333333')

    # Helper: wrap region labels into two lines to prevent overflow
    def _wrap_two_lines(label: str, max_chars: int = 12) -> str:
        s = str(label)
        parts = s.split()
        if len(parts) <= 1:
            # Fallback: split mid-string if too long
            if len(s) <= max_chars:
                return s
            return s[:max_chars] + "\n" + s[max_chars:]
        line1 = []
        line2 = []
        cur = 0
        for w in parts:
            if cur + (len(w) if cur == 0 else len(w) + 1) <= max_chars:
                line1.append(w)
                cur += len(w) if cur == 0 else len(w) + 1
            else:
                line2.append(w)
        if not line1:
            # Ensure at least one word in line1
            line1 = [parts[0]]
            line2 = parts[1:]
        return " ".join(line1) + ("\n" + " ".join(line2) if line2 else "")

    wrapped_labels = [_wrap_two_lines(r, max_chars=label_max_chars_per_line) for r in regions]

    # Axes and labels
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center')
    ax.set_ylabel('Fatalities')

    # Style similar to other charts
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666')
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # Optional vertical separators between groups
    if show_group_separators and n_regions > 1:
        # draw lines between integer group centers at half-steps: 0.5, 1.5, ..., n_regions-0.5
        y_min, y_max = ax.get_ylim()
        for xi in np.arange(n_regions - 1) + 0.5:
            ax.vlines(xi, y_min, y_max, colors=separator_color, linestyles='-', linewidth=separator_linewidth, zorder=0, alpha=0.8)
        # Restore y-limits in case autoscale changed
        ax.set_ylim(y_min, y_max)

    # Titles and footer
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.98, ha='left', x=0.05)
    if chart_subtitle:
        fig.text(0.05, 0.95, chart_subtitle, ha='left', va='top', fontsize=12, color='#555555')
    if source_text:
        fig.text(0.05, 0.02, source_text, ha='left', va='bottom', fontsize=10, color='#666666', alpha=0.8)

    # Legend
    ax.legend(title='Conflict Type', ncol=min(4, n_types), frameon=False, loc='upper right')

    plt.tight_layout()
    # Slightly more bottom margin to accommodate wrapped labels
    plt.subplots_adjust(top=0.90, bottom=0.16, left=0.08, right=0.98)
    return fig