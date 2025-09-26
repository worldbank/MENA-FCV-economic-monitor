import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Optional

# Default colors aligned with plot_dual_metrics_by_country usage
DEFAULT_METRIC_COLORS: Dict[str, str] = {
    'nrFatalities': 'steelblue',
    'nrEvents': 'orange',
}


def _ensure_time_agg(
    df: pd.DataFrame,
    date_col: str,
    metrics: Tuple[str, str],
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensure the dataframe is aggregated on a time frequency if requested.
    - If freq is None, return df sorted by date_col without aggregation.
    - If freq provided (e.g., 'YS', 'MS'), resample/sum the metrics.
    """
    df2 = df.copy()
    if date_col not in df2.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame")

    df2[date_col] = pd.to_datetime(df2[date_col])
    df2 = df2.sort_values(date_col)

    metric_list = [m for m in metrics if m in df2.columns]
    if not metric_list:
        raise ValueError("None of the requested metrics are in the DataFrame")

    # If no frequency, return as-is (sorted)
    if not freq:
        return df2[[date_col] + metric_list]

    # Aggregate by frequency
    grouped = (
        df2.set_index(date_col)
           .resample(freq)[metric_list]
           .sum(min_count=1)
           .reset_index()
    )
    return grouped


def plot_dual_bars_by_date(
    data: pd.DataFrame,
    date_col: str = 'event_date',
    metrics: Tuple[str, str] = ('nrEvents', 'nrFatalities'),
    metric_display_info: Optional[Dict[str, Dict[str, str]]] = None,
    overall_title: str = 'Conflict Events and Fatalities Over Time',
    chart_subtitle: Optional[str] = None,
    source_text: Optional[str] = None,
    freq: Optional[str] = 'YS',
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Create a dual bar chart with time on the x-axis and one metric per subplot on the y-axis.
    Colors mirror the scheme used in plot_dual_metrics_by_country.

    Args:
        data: DataFrame containing at least [date_col, metrics...]
        date_col: Name of the datetime column (e.g., 'event_date')
        metrics: Tuple of two metric column names (e.g., ('nrEvents', 'nrFatalities'))
        metric_display_info: Optional mapping {metric: {title, color}}
        overall_title: Figure suptitle
        chart_subtitle: Optional subtitle rendered under the title
        source_text: Optional source footer text
        freq: Pandas resample frequency (e.g., 'YS' annual, 'MS' monthly). None to skip resample
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if metric_display_info is None:
        metric_display_info = {}

    # Prepare data (aggregate if requested)
    dfp = _ensure_time_agg(data, date_col=date_col, metrics=metrics, freq=freq)

    # Keep only requested metrics that exist
    plot_metrics = [m for m in metrics if m in dfp.columns]
    if len(plot_metrics) == 0:
        raise ValueError("No valid metrics to plot")

    # Build figure with one column per metric
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=figsize, sharex=False)
    if len(plot_metrics) == 1:
        axes = [axes]

    plt.style.use('default')
    fig.patch.set_facecolor('white')

    # Bar widths: try to infer a consistent width from date differences
    dates = pd.to_datetime(dfp[date_col].values)
    if len(dates) > 1:
        delta_days = np.median(np.diff(dates).astype('timedelta64[D]').astype(float))
        # Use 80% of the median spacing as width; matplotlib expects days for datetime x-axis
        width = np.timedelta64(int(max(1, round(delta_days * 0.8))), 'D')
    else:
        width = np.timedelta64(20, 'D')  # fallback

    for i, metric in enumerate(plot_metrics):
        ax = axes[i]
        info = metric_display_info.get(metric, {})
        color = info.get('color', DEFAULT_METRIC_COLORS.get(metric, plt.cm.Paired(i)))
        title = info.get('title', metric.replace('_', ' ').title())

        ax.bar(dfp[date_col], dfp[metric], color=color, edgecolor='white', linewidth=0.5, width=width)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=12)
        ax.set_ylabel('')
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')

        # Styling similar to plot_dual_metrics_by_country
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666')
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    # Titles and notes
    fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.97, ha='left', x=0.05)
    if chart_subtitle:
        fig.text(0.05, 0.90, chart_subtitle, ha='left', va='top', fontsize=12, color='#555555')
    if source_text:
        fig.text(0.05, 0.02, source_text, ha='left', va='bottom', fontsize=10, color='#666666', alpha=0.8)

    plt.tight_layout()
    top_margin = 0.80 if chart_subtitle else 0.88
    plt.subplots_adjust(left=0.08, top=top_margin, bottom=0.12)
    return fig


def plot_single_bar_by_date(
    data: pd.DataFrame,
    metric: str,
    date_col: str = 'event_date',
    color: Optional[str] = None,
    title: Optional[str] = None,
    chart_subtitle: Optional[str] = None,
    source_text: Optional[str] = None,
    freq: Optional[str] = 'YS',
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Convenience helper to plot a single metric by date with ACLED styling.
    """
    dfp = _ensure_time_agg(data, date_col=date_col, metrics=(metric,), freq=freq)

    if metric not in dfp.columns:
        raise ValueError(f"Metric '{metric}' not in DataFrame")

    if color is None:
        color = DEFAULT_METRIC_COLORS.get(metric, 'gray')
    if title is None:
        title = metric.replace('_', ' ').title()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    dates = pd.to_datetime(dfp[date_col].values)
    if len(dates) > 1:
        delta_days = np.median(np.diff(dates).astype('timedelta64[D]').astype(float))
        width = np.timedelta64(int(max(1, round(delta_days * 0.8))), 'D')
    else:
        width = np.timedelta64(20, 'D')

    ax.bar(dfp[date_col], dfp[metric], color=color, edgecolor='white', linewidth=0.5, width=width)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Date', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666')
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    if chart_subtitle:
        fig.text(0.05, 0.90, chart_subtitle, ha='left', va='top', fontsize=12, color='#555555')
    if source_text:
        fig.text(0.05, 0.02, source_text, ha='left', va='bottom', fontsize=10, color='#666666', alpha=0.8)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.88, bottom=0.12)
    return fig
