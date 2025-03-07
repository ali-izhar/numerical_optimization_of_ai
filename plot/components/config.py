# plot/components/config.py

"""
This module provides a standardized configuration for visualization settings
across different types of numerical method visualizations.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.

    This dataclass encapsulates all parameters related to visualization appearance,
    layout, and behavior for numerical methods visualization.

    Attributes:
        figsize: Figure size as (width, height) in inches
        show_top_right_plot: Whether to show top right plot (e.g., convergence)
        show_bottom_right_plot: Whether to show bottom right plot (e.g., error)
        show_convergence: Whether to show convergence plot (alias for show_top_right_plot)
        show_error: Whether to show error plot (alias for show_bottom_right_plot)
        show_contour: Whether to show contour for 2D functions
        style: Plot style (darkgrid, whitegrid, dark, white, ticks)
        context: Plot context (paper, notebook, talk, poster)
        palette: Color palette for different methods
        point_size: Size of points in scatter plots
        dpi: Dots per inch for saved figures
        show_legend: Whether to display the legend
        grid_alpha: Transparency of grid lines
        title: Main title for the visualization
        background_color: Background color for plots
        animation_duration: Duration for each animation frame (ms)
        animation_transition: Transition time between frames (ms)
        animation_interval: Interval between animation frames (ms)
        verbose: Enable verbose output
    """

    # Figure properties
    figsize: Tuple[int, int] = (15, 8)

    # Plot visibility
    show_top_right_plot: bool = True
    show_bottom_right_plot: bool = True
    show_convergence: bool = True  # Alias for top right plot
    show_error: bool = True  # Alias for bottom right plot
    show_contour: bool = True  # For 2D functions

    # Style properties
    style: str = "white"
    context: str = "talk"
    palette: str = "viridis"
    point_size: int = 100
    dpi: int = 100
    show_legend: bool = True
    grid_alpha: float = 0.3
    title: str = "Numerical Methods Comparison"
    background_color: str = "#FFFFFF"

    # Animation properties
    animation_duration: int = 800  # ms per frame
    animation_transition: int = 300  # ms for transition
    animation_interval: int = 1  # ms between frames

    # Debugging
    verbose: bool = False

    def __post_init__(self):
        """Ensure consistency between aliased properties."""
        # Sync aliases
        self.show_convergence = self.show_top_right_plot
        self.show_error = self.show_bottom_right_plot
