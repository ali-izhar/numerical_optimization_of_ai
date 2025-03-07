# plot/utils/color_utils.py

"""
This module provides utilities for generating and managing colors for
visualization of numerical methods.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_hex


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

    # For categorical data, use categorical palettes
    if palette in ["Set1", "Set2", "Set3", "Paired", "Dark2", "Accent"]:
        # Avoid repeating colors by cycling through multiple palettes if needed
        if n_colors > 10:  # Most categorical palettes have around 8-10 colors
            palette_1 = sns.color_palette(palette, 10)
            palette_2 = sns.color_palette("Dark2", n_colors - 10)
            colors = palette_1 + palette_2
        else:
            colors = sns.color_palette(palette, n_colors)
    else:
        # For sequential or diverging palettes
        colors = sns.color_palette(palette, n_colors)

    # Convert to hex format using matplotlib's to_hex
    hex_colors = [to_hex(rgb) for rgb in colors]
    return hex_colors


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
    methods: List[str], palette: str = "viridis", existing_colors: Optional[dict] = None
) -> dict:
    """
    Get colors for a list of methods.

    Args:
        methods: List of method names
        palette: Color palette to use
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
    cmap: Union[str, LinearSegmentedColormap] = "viridis",
) -> List[Tuple[float, float, float, float]]:
    """
    Generate colors based on values using a colormap.

    Args:
        values: Array of values to map to colors
        vmin: Minimum value for color mapping (default: min(values))
        vmax: Maximum value for color mapping (default: max(values))
        cmap: Colormap name or object to use

    Returns:
        List of RGBA color tuples
    """
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    # Normalize values to [0, 1]
    norm_values = (
        (values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(values)
    )

    # Get colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Map values to colors
    colors = [cmap(val) for val in norm_values]
    return colors
