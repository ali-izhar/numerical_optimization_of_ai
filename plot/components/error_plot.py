# plot/components/error_plot.py

"""
This module provides a component for visualizing error metrics during the
execution of numerical methods.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plot.utils.color_utils import generate_colors, get_method_colors


class ErrorPlot:
    """
    Component for visualizing error metrics during numerical method execution.

    This class provides methods for creating plots that show how error metrics
    change during the execution of one or more numerical methods, allowing
    comparison of their convergence properties.
    """

    def __init__(
        self,
        title: str = "Error Plot",
        xlabel: str = "Iteration",
        ylabel: str = "Error",
        color_palette: str = "Set1",
        log_scale: bool = True,
    ):
        """
        Initialize the error plot component.

        Args:
            title: Title for the plot
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            color_palette: Color palette for different methods
            log_scale: Whether to use logarithmic scale for the y-axis
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color_palette = color_palette
        self.log_scale = log_scale
        self.method_colors = {}

    def create_matplotlib_figure(
        self,
        data: Dict[str, List[float]],
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_grid: bool = True,
        include_markers: bool = True,
        existing_colors: Optional[Dict[str, str]] = None,
        legend_loc: str = "best",
        annotate_final: bool = False,
        tolerance_line: Optional[float] = None,
        display_iterations: bool = True,
        y_limit: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a matplotlib figure showing error metrics for multiple methods.

        Args:
            data: Dictionary mapping method names to lists of error values
            ax: Existing axes to plot on (creates new figure if None)
            figsize: Figure size for new figure
            show_grid: Whether to show grid lines
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors
            legend_loc: Location for the legend
            annotate_final: Whether to annotate final error values
            tolerance_line: Value for tolerance line to show
            display_iterations: Whether to display iteration numbers
            y_limit: Tuple of (min, max) for y-axis limits

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Plot each method's error values
        for method_name, error_values in data.items():
            color = self.method_colors.get(method_name, "gray")
            iterations = list(range(len(error_values)))

            # Handle any negative or zero values if using log scale
            if self.log_scale:
                # Replace zeros or negative values with a small positive number
                error_values = [max(1e-15, v) for v in error_values]

            # Line style
            line_style = "-"
            marker = "o" if include_markers else None

            # Plot line
            ax.plot(
                iterations,
                error_values,
                linestyle=line_style,
                marker=marker,
                markersize=5,
                linewidth=2,
                color=color,
                label=method_name,
            )

            # Annotate final value if requested
            if annotate_final and error_values:
                ax.annotate(
                    f"{error_values[-1]:.2e}",
                    xy=(len(error_values) - 1, error_values[-1]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        # Add tolerance line if provided
        if tolerance_line is not None:
            ax.axhline(
                y=tolerance_line,
                color="black",
                linestyle="--",
                alpha=0.5,
                label=f"Tolerance ({tolerance_line:.2e})",
            )

        # Set axis labels and title
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        # Set x-axis to show iteration numbers if requested
        if display_iterations:
            # Find the maximum number of iterations across all methods
            max_iterations = max(len(values) for values in data.values())
            ax.set_xticks(range(0, max_iterations, max(1, max_iterations // 10)))

        # Set log scale if requested
        if self.log_scale:
            ax.set_yscale("log")

        # Set y-axis limits if provided
        if y_limit:
            if y_limit[0] is not None:
                ax.set_ylim(bottom=y_limit[0])
            if y_limit[1] is not None:
                ax.set_ylim(top=y_limit[1])

        # Add grid
        if show_grid:
            ax.grid(alpha=0.3)

        # Add legend
        if data:
            ax.legend(loc=legend_loc)

        return fig, ax

    def create_plotly_figure(
        self,
        data: Dict[str, List[float]],
        height: int = 500,
        width: int = 800,
        include_markers: bool = True,
        existing_colors: Optional[Dict[str, str]] = None,
        annotate_final: bool = False,
        tolerance_line: Optional[float] = None,
        display_iterations: bool = True,
        y_limit: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> go.Figure:
        """
        Create a plotly figure showing error metrics for multiple methods.

        Args:
            data: Dictionary mapping method names to lists of error values
            height: Figure height
            width: Figure width
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors
            annotate_final: Whether to annotate final error values
            tolerance_line: Value for tolerance line to show
            display_iterations: Whether to display iteration numbers
            y_limit: Tuple of (min, max) for y-axis limits

        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()

        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Plot each method's error values
        for method_name, error_values in data.items():
            color = self.method_colors.get(method_name, "gray")
            iterations = list(range(len(error_values)))

            # Handle any negative or zero values if using log scale
            if self.log_scale:
                # Replace zeros or negative values with a small positive number
                error_values = [max(1e-15, v) for v in error_values]

            # Determine marker settings
            mode = "lines+markers" if include_markers else "lines"

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=error_values,
                    mode=mode,
                    name=method_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                )
            )

            # Add annotations for final values if requested
            if annotate_final and error_values:
                fig.add_annotation(
                    x=len(error_values) - 1,
                    y=error_values[-1],
                    text=f"{error_values[-1]:.2e}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=color,
                    ax=20,
                    ay=0,
                    font=dict(color=color, size=10),
                )

        # Add tolerance line if provided
        if tolerance_line is not None:
            # Get the maximum length of all error lists
            max_iterations = max(len(values) for values in data.values())

            fig.add_shape(
                type="line",
                x0=0,
                y0=tolerance_line,
                x1=max_iterations - 1,
                y1=tolerance_line,
                line=dict(color="black", width=1, dash="dash"),
            )

            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[tolerance_line],
                    mode="markers",
                    marker=dict(color="black", size=0),
                    name=f"Tolerance ({tolerance_line:.2e})",
                    hoverinfo="name",
                )
            )

        # Set axis titles and figure title
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            height=height,
            width=width,
            template="plotly_white",
            hovermode="closest",
        )

        # Set x-axis to show iteration numbers if requested
        if display_iterations:
            max_iterations = max(len(values) for values in data.values())
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(0, max_iterations, max(1, max_iterations // 10))),
            )

        # Set log scale if requested
        if self.log_scale:
            fig.update_yaxes(type="log")

        # Set y-axis limits if provided
        if y_limit:
            update_dict = {}
            if y_limit[0] is not None:
                update_dict["range"] = (
                    [np.log10(y_limit[0]), None]
                    if self.log_scale
                    else [y_limit[0], None]
                )
            if y_limit[1] is not None:
                if "range" in update_dict:
                    update_dict["range"][1] = (
                        np.log10(y_limit[1]) if self.log_scale else y_limit[1]
                    )
                else:
                    update_dict["range"] = [
                        None,
                        np.log10(y_limit[1]) if self.log_scale else y_limit[1],
                    ]

            if update_dict:
                fig.update_yaxes(**update_dict)

        return fig

    def create_convergence_rate_plot(
        self,
        data: Dict[str, List[float]],
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        existing_colors: Optional[Dict[str, str]] = None,
        legend_loc: str = "best",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a plot showing the convergence rate for each method.

        The convergence rate is estimated by computing the ratio of consecutive errors.

        Args:
            data: Dictionary mapping method names to lists of error values
            ax: Existing axes to plot on (creates new figure if None)
            figsize: Figure size for new figure
            existing_colors: Dictionary mapping method names to colors
            legend_loc: Location for the legend

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Plot convergence rates for each method
        for method_name, error_values in data.items():
            if len(error_values) < 3:
                continue  # Need at least 3 points to estimate convergence rate

            color = self.method_colors.get(method_name, "gray")

            # Calculate convergence rates (e_{i+1} / e_i)
            rates = []
            iterations = []

            for i in range(1, len(error_values)):
                if error_values[i - 1] == 0 or error_values[i] == 0:
                    continue  # Skip division by zero

                rate = error_values[i] / error_values[i - 1]
                if 0 < rate < 1.5:  # Filter out unrealistic rates
                    rates.append(rate)
                    iterations.append(i)

            if not rates:
                continue  # Skip if no valid rates

            # Plot rates
            ax.plot(
                iterations,
                rates,
                "o-",
                color=color,
                label=f"{method_name} (avg: {np.mean(rates):.3f})",
            )

        # Add horizontal line at y=1 (no convergence)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)

        # Add horizontal line at y=0.5 (quadratic convergence)
        ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5)

        # Set axis labels and title
        ax.set_title("Convergence Rate Analysis")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rate (e_{i+1} / e_i)")

        # Add grid
        ax.grid(alpha=0.3)

        # Add legend
        if data:
            ax.legend(loc=legend_loc)

        return fig, ax
