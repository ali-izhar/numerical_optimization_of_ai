# plot/components/config.py

"""
This module provides a standardized configuration for visualization settings
across different types of numerical method visualizations.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.

    This dataclass encapsulates all parameters related to visualization appearance,
    layout, and behavior for numerical methods visualization.

    Attributes:
        figsize: Figure size as (width, height) in inches for matplotlib (None for auto in Plotly)
        show_top_right_plot: Whether to show top right plot (e.g., convergence)
        show_bottom_right_plot: Whether to show bottom right plot (e.g., error)
        show_convergence: Whether to show convergence plot (alias for show_top_right_plot)
        show_error: Whether to show error plot (alias for show_bottom_right_plot)
        show_contour: Whether to show contour for 2D functions
        style: Plot style (darkgrid, whitegrid, dark, white, ticks) for matplotlib
        context: Plot context (paper, notebook, talk, poster) for matplotlib
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
        plotly_template: Plotly template for styling ('plotly', 'plotly_white', 'plotly_dark', etc.)
        plotly_colorscales: Dict mapping plot elements to colorscales for Plotly
        plotly_font: Font settings for Plotly visualizations
        plotly_margin: Margins for Plotly plots (None for auto)
        use_plotly_3d: Use 3D surface plots in Plotly for 2D functions
        plotly_line_width: Width of lines in Plotly visualizations
        plotly_marker_size: Size of markers in Plotly visualizations
        plotly_hover_mode: Hover mode for Plotly plots ('closest', 'x', 'y', 'x unified', 'y unified')
        verbose: Enable verbose output
    """

    # Figure properties - keeping for matplotlib compatibility but using None as default for Plotly
    figsize: Optional[Tuple[int, int]] = None

    # Plot visibility
    show_top_right_plot: bool = True
    show_bottom_right_plot: bool = True
    show_convergence: bool = True  # Alias for top right plot
    show_error: bool = True  # Alias for bottom right plot
    show_contour: bool = True  # For 2D functions

    # Style properties (mostly for matplotlib, but some shared)
    style: str = "white"  # For matplotlib
    context: str = "talk"  # For matplotlib
    palette: str = "turbo"  # Changed from viridis to turbo for more visual pop
    point_size: int = 120  # Increased for better visibility
    dpi: int = 100
    show_legend: bool = True
    grid_alpha: float = 0.2  # Slightly reduced for cleaner look
    title: str = "Numerical Methods Comparison"
    background_color: str = "rgba(255, 255, 255, 0.95)"  # Using rgba for transparency

    # Animation properties - adjusted for smoother animations
    animation_duration: int = 800  # ms per frame
    animation_transition: int = 300  # ms for transition
    animation_interval: int = 50  # ms between frames (increased for smoother playback)

    # Plotly-specific style properties
    plotly_template: str = "plotly_white"
    plotly_colorscales: Dict[str, str] = None  # Will be initialized in __post_init__
    plotly_font: Dict[str, Any] = None  # Will be initialized in __post_init__
    plotly_margin: Optional[Dict[str, int]] = None  # None for auto-sizing
    use_plotly_3d: bool = True  # Prefer 3D surface plots for 2D functions
    plotly_line_width: int = 3
    plotly_marker_size: int = 10
    plotly_hover_mode: str = "closest"

    # Debugging
    verbose: bool = False

    def __post_init__(self):
        """Ensure consistency between aliased properties and initialize complex defaults."""
        # Sync aliases
        self.show_convergence = self.show_top_right_plot
        self.show_error = self.show_bottom_right_plot

        # Initialize complex defaults
        if self.plotly_colorscales is None:
            self.plotly_colorscales = {
                "surface": "Viridis",  # For 3D surface plots
                "contour": "Viridis",  # For 2D contour plots
                "heatmap": "Inferno",  # For heatmaps
                "error": "Reds",  # For error plots
                "convergence": "Blues",  # For convergence plots
            }

        if self.plotly_font is None:
            self.plotly_font = {
                "family": "Arial, sans-serif",
                "size": 14,
                "color": "#333333",
            }

        if self.plotly_margin is None:
            # Auto margins that adapt to the display
            self.plotly_margin = {
                "l": 60,
                "r": 30,
                "t": 80,
                "b": 100,
                "pad": 4,
                "autoexpand": True,
            }
