# plot/utils/color_utils.py

"""
This module provides utilities for generating and managing colors for
visualization of numerical methods, with a focus on creating eye-catching
Plotly visualizations.
"""

from typing import List, Optional, Tuple, Union, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_hex
import plotly.express as px
import plotly.colors as pc


def generate_colors(n_colors: int, palette: str = "viridis") -> List[str]:
    """
    Generate a list of colors from a specified palette.

    Args:
        n_colors: Number of colors to generate
        palette: Name of the color palette to use

    Returns:
        List of color strings in hexadecimal format
    """
    if n_colors <= 0:
        return []

    # Default palette if there's an issue
    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    # Check if the palette is a Plotly built-in sequential colorscale
    plotly_scales = [
        "Viridis",
        "Plasma",
        "Inferno",
        "Magma",
        "Cividis",
        "Turbo",
        "Blues",
        "Greens",
        "Reds",
        "YlOrRd",
        "Purples",
    ]
    plotly_qual_scales = [
        "Plotly3",
        "D3",
        "G10",
        "T10",
        "Alphabet",
        "Dark24",
        "Light24",
    ]

    # First check for Plotly qualitative color sequences
    if palette.lower() in [s.lower() for s in plotly_qual_scales]:
        # Get the Plotly qualitative colorscale
        for scale in plotly_qual_scales:
            if palette.lower() == scale.lower():
                try:
                    colors = getattr(pc.qualitative, scale)[:n_colors]
                    # Cycle through if needed
                    if n_colors > len(colors):
                        colors = [colors[i % len(colors)] for i in range(n_colors)]

                    # Validate that all colors are proper strings
                    for i, color in enumerate(colors):
                        if not isinstance(color, str) or not color.startswith(
                            ("#", "rgb", "rgba", "hsl", "hsla")
                        ):
                            colors[i] = default_colors[i % len(default_colors)]

                    return colors
                except Exception:
                    # Fall back to default colors on any error
                    return [
                        default_colors[i % len(default_colors)] for i in range(n_colors)
                    ]

    # Then check for Plotly sequential colorscales
    elif palette.lower() in [s.lower() for s in plotly_scales]:
        try:
            # Get the corresponding Plotly colorscale
            for scale in plotly_scales:
                if palette.lower() == scale.lower():
                    colorscale = getattr(pc.sequential, scale)
                    break
            else:
                colorscale = getattr(pc.sequential, "Viridis")

            # Generate evenly spaced colors from the colorscale
            colors = []
            for i in range(n_colors):
                t = i / max(1, n_colors - 1)

                # Simplify the color selection to avoid issues
                idx = min(int(t * (len(colorscale) - 1)), len(colorscale) - 1)
                color = colorscale[idx][1]

                # Ensure the color is a valid string format
                if not isinstance(color, str) or not color.startswith(
                    ("#", "rgb", "rgba", "hsl", "hsla")
                ):
                    color = default_colors[i % len(default_colors)]

                colors.append(color)

            return colors
        except Exception:
            # Fall back to default colors on any error
            return [default_colors[i % len(default_colors)] for i in range(n_colors)]

    # For categorical data, use categorical palettes or default to our backup colors
    return [default_colors[i % len(default_colors)] for i in range(n_colors)]


def create_color_map(
    start_color: str, end_color: str, n_colors: int = 256
) -> LinearSegmentedColormap:
    """
    Create a custom color map between two colors.

    Args:
        start_color: Starting color (hex or name)
        end_color: Ending color (hex or name)
        n_colors: Number of color levels

    Returns:
        LinearSegmentedColormap: Custom color map
    """
    # Convert colors to rgba
    start_rgba = to_rgba(start_color)
    end_rgba = to_rgba(end_color)

    # Create color map
    cmap = LinearSegmentedColormap.from_list(
        f"custom_{start_color}_{end_color}", [start_rgba, end_rgba], N=n_colors
    )
    return cmap


def get_method_colors(
    methods: List[str], palette: str = "D3", existing_colors: Optional[dict] = None
) -> dict:
    """
    Get colors for a list of methods, optimized for Plotly visualizations.

    Args:
        methods: List of method names
        palette: Color palette to use (default: D3 - a vibrant Plotly palette)
        existing_colors: Dictionary of existing method colors to maintain consistency

    Returns:
        Dictionary mapping method names to colors
    """
    # Initialize with existing colors
    method_colors = existing_colors or {}

    # Get methods that need new colors
    new_methods = [m for m in methods if m not in method_colors]

    if new_methods:
        # Generate new colors
        new_colors = generate_colors(len(new_methods), palette)

        # Assign colors to methods
        for method, color in zip(new_methods, new_colors):
            method_colors[method] = color

    return method_colors


def color_by_value(
    values: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Union[str, LinearSegmentedColormap] = "Viridis",
) -> List[Tuple[float, float, float, float]]:
    """
    Generate colors based on values using a colormap, optimized for Plotly.

    Args:
        values: Array of values to map to colors
        vmin: Minimum value for color mapping (default: min(values))
        vmax: Maximum value for color mapping (default: max(values))
        cmap: Colormap name or object to use

    Returns:
        List of RGBA color tuples or hex strings depending on the input cmap
    """
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    # Normalize values to [0, 1]
    norm_values = (
        (values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(values)
    )

    # Check if this is a Plotly colormap string
    plotly_cmaps = [
        "Viridis",
        "Plasma",
        "Inferno",
        "Magma",
        "Cividis",
        "Blues",
        "Greens",
        "Reds",
        "YlOrRd",
        "Purples",
    ]

    if isinstance(cmap, str) and cmap in plotly_cmaps:
        # Use Plotly's continuous color scale
        colorscale = getattr(pc.sequential, cmap)
        colors = []
        for val in norm_values:
            # Find the appropriate color in the colorscale
            for i in range(len(colorscale) - 1):
                if val >= colorscale[i][0] and val <= colorscale[i + 1][0]:
                    t = (val - colorscale[i][0]) / (
                        colorscale[i + 1][0] - colorscale[i][0]
                    )
                    color1 = to_rgba(colorscale[i][1])
                    color2 = to_rgba(colorscale[i + 1][1])
                    # Interpolate between the two colors
                    r = color1[0] + t * (color2[0] - color1[0])
                    g = color1[1] + t * (color2[1] - color1[1])
                    b = color1[2] + t * (color2[2] - color1[2])
                    a = color1[3] + t * (color2[3] - color1[3])
                    colors.append((r, g, b, a))
                    break
            else:
                # Default to the last color if we're out of range
                colors.append(to_rgba(colorscale[-1][1]))
        return colors
    else:
        # Use matplotlib colormap as fallback
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        # Map values to colors
        colors = [cmap(val) for val in norm_values]
        return colors


def get_plotly_colorscale(
    name: str = "Viridis", n_colors: int = None
) -> List[List[Union[float, str]]]:
    """
    Get a Plotly colorscale by name, optionally with a specified number of discrete colors.

    Args:
        name: Name of the Plotly colorscale
        n_colors: Number of discrete colors to extract (None for continuous scale)

    Returns:
        Plotly colorscale format: [[0, 'rgb(x,y,z)'], [1, 'rgb(a,b,c)']]
    """
    # First try to get a builtin colorscale
    try:
        if hasattr(pc.sequential, name):
            colorscale = getattr(pc.sequential, name)
        elif hasattr(pc.diverging, name):
            colorscale = getattr(pc.diverging, name)
        elif hasattr(pc.cyclical, name):
            colorscale = getattr(pc.cyclical, name)
        else:
            # Fall back to viridis
            colorscale = pc.sequential.Viridis
    except:
        # If not found, use a matplotlib colormap and convert
        try:
            mpl_cmap = plt.get_cmap(name)
            colorscale = []
            for i in range(11):  # 11 points for a smooth scale
                pos = i / 10
                rgba = mpl_cmap(pos)
                rgb = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
                colorscale.append([pos, rgb])
        except:
            # Last resort - use Viridis
            colorscale = pc.sequential.Viridis

    # If n_colors is specified, extract discrete colors
    if n_colors is not None and n_colors > 0:
        positions = np.linspace(0, 1, n_colors)
        discrete_scale = []

        for pos in positions:
            # Find the right segment in the colorscale
            for i in range(len(colorscale) - 1):
                if pos >= colorscale[i][0] and pos <= colorscale[i + 1][0]:
                    # Linear interpolation between colors
                    t = (pos - colorscale[i][0]) / (
                        colorscale[i + 1][0] - colorscale[i][0]
                    )
                    color1 = to_rgba(colorscale[i][1])
                    color2 = to_rgba(colorscale[i + 1][1])
                    r = int(color1[0] * 255 + t * (color2[0] * 255 - color1[0] * 255))
                    g = int(color1[1] * 255 + t * (color2[1] * 255 - color1[1] * 255))
                    b = int(color1[2] * 255 + t * (color2[2] * 255 - color1[2] * 255))
                    color = f"rgb({r},{g},{b})"
                    discrete_scale.append([pos, color])
                    break
            else:
                # Default to last color if needed
                discrete_scale.append([pos, colorscale[-1][1]])

        return discrete_scale

    return colorscale


def plotly_figure_layout(
    fig, title=None, colorscale="Viridis", template="plotly_white", show_colorbar=True
):
    """
    Apply consistent, eye-catching styling to a Plotly figure.

    Args:
        fig: A plotly figure object
        title: Title for the figure
        colorscale: The colorscale to use for the figure's color elements
        template: The plotly template to use
        show_colorbar: Whether to show colorbar when applicable

    Returns:
        The styled figure
    """
    # Use a template that looks good with the data
    fig.update_layout(
        template=template,
        title={
            "text": title,
            "font": {"size": 24, "family": "Arial, sans-serif"},
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font={"family": "Arial, sans-serif", "size": 14},
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
        legend={
            "font": {"size": 14},
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.2,
        },
        coloraxis={"colorscale": colorscale, "showscale": show_colorbar},
        # No hardcoded dimensions - let plotly take browser dimensions by default
    )

    # Add grid lines and improve axis styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(220, 220, 220, 0.5)",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        title_font={"size": 18},
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(220, 220, 220, 0.5)",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        title_font={"size": 18},
    )

    return fig
