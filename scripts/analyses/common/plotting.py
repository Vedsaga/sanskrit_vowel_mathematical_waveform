"""
Plotting utilities and dark theme styling.

This module provides consistent styling for all visualizations across
the formant analysis scripts, aligned with the sound-topology dark mode.
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Color Scheme (aligned with sound-topology dark mode)
# ============================================================================

COLORS = {
    # Background colors
    'background': '#111111',
    'surface': '#1a1a1a',
    'border': '#333333',
    'grid': '#2a2a2a',
    
    # Text colors
    'text': '#eaeaea',
    'text_secondary': '#aaaaaa',
    
    # Accent colors
    'accent': '#e17100',  # Orange - use sparingly
    'primary': '#4ECDC4',  # Teal
    'secondary': '#FF6B6B',  # Coral/Red
    'tertiary': '#FFD93D',  # Yellow/Gold
    
    # Additional colors for multi-series plots
    'purple': '#A855F7',
    'blue': '#3B82F6',
    'green': '#10B981',
    'pink': '#EC4899',
}

# Color palette for categorical data
CATEGORICAL_PALETTE = [
    COLORS['primary'],
    COLORS['secondary'],
    COLORS['tertiary'],
    COLORS['purple'],
    COLORS['blue'],
    COLORS['green'],
    COLORS['pink'],
    COLORS['accent'],
]


def apply_dark_theme(ax, show_grid: bool = True, show_spines: bool = True):
    """
    Apply consistent dark theme styling to a matplotlib axis.
    
    Args:
        ax: Matplotlib axis object
        show_grid: Whether to show grid lines
        show_spines: Whether to show axis spines (borders)
    
    Usage:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        apply_dark_theme(ax)
    """
    ax.set_facecolor(COLORS['surface'])
    ax.tick_params(colors=COLORS['text'], labelsize=10)
    
    # Style spines
    if show_spines:
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.spines['left'].set_color(COLORS['border'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid
    if show_grid:
        ax.grid(True, alpha=0.15, color='white', linestyle='-', linewidth=0.5)
    
    # Label colors
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])
    ax.title.set_color(COLORS['text'])


def create_styled_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple = None,
    sharex: bool = False,
    sharey: bool = False
):
    """
    Create a figure with dark theme applied.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches
        sharex: Share x-axis between subplots
        sharey: Share y-axis between subplots
    
    Returns:
        (fig, axes): The figure and axes objects
    
    Usage:
        fig, axes = create_styled_figure(2, 2)
        for ax in axes.flat:
            ax.plot(x, y)
            apply_dark_theme(ax)
    """
    if figsize is None:
        # Auto-size based on subplot count
        width = 6 * ncols + 2
        height = 5 * nrows + 1
        figsize = (min(width, 20), min(height, 16))
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=figsize,
        sharex=sharex,
        sharey=sharey
    )
    fig.patch.set_facecolor(COLORS['background'])
    
    return fig, axes


def style_legend(ax, loc: str = 'best'):
    """
    Style a legend with dark theme colors.
    
    Args:
        ax: Matplotlib axis with an existing legend
        loc: Legend location
    
    Usage:
        ax.plot(x, y, label='Data')
        ax.legend()
        style_legend(ax)
    """
    legend = ax.legend(
        loc=loc,
        facecolor=COLORS['surface'],
        edgecolor=COLORS['border'],
        labelcolor=COLORS['text'],
        fontsize=9
    )
    return legend


def save_styled_figure(fig, filepath: str, dpi: int = 150):
    """
    Save a figure with consistent styling.
    
    Args:
        fig: Matplotlib figure
        filepath: Output path
        dpi: Resolution
    """
    fig.savefig(
        filepath,
        dpi=dpi,
        facecolor=COLORS['background'],
        edgecolor='none',
        bbox_inches='tight'
    )
    print(f"Saved: {filepath}")


def get_rainbow_colors(n: int) -> np.ndarray:
    """
    Get n colors from a rainbow colormap.
    
    Args:
        n: Number of colors needed
    
    Returns:
        Array of RGBA colors
    
    Usage:
        colors = get_rainbow_colors(len(categories))
        for i, cat in enumerate(categories):
            ax.bar(i, values[i], color=colors[i])
    """
    return plt.cm.rainbow(np.linspace(0, 1, n))


def create_colorbar(fig, ax, mappable, label: str = ''):
    """
    Add a styled colorbar to a figure.
    
    Args:
        fig: Matplotlib figure
        ax: Axis containing the mappable
        mappable: ScalarMappable (e.g., from imshow, scatter with c=)
        label: Colorbar label
    
    Returns:
        Colorbar object
    """
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
    cbar.outline.set_edgecolor(COLORS['border'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text'])
    if label:
        cbar.set_label(label, color=COLORS['text'])
    return cbar
