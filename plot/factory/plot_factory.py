# plot/factory/plot_factory.py

"""
This module provides factory classes for creating different types of plots
for numerical methods visualization.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from algorithms.convex.protocols import (
    BaseNumericalMethod,
    NumericalMethodConfig,
    MethodType,
)
from plot.components.config import VisualizationConfig
from plot.components.function_space import FunctionSpace
from plot.components.convergence_plot import ConvergencePlot
from plot.components.error_plot import ErrorPlot
from plot.components.animation import MethodAnimation
from plot.utils.color_utils import get_method_colors


class PlotFactory:
    """
    Factory class for creating different types of plots.

    This class provides methods for creating various types of plots for
    visualizing numerical methods, including function spaces, convergence
    plots, and error plots.
    """

    @staticmethod
    def create_comparison_plot(
        methods: List[BaseNumericalMethod],
        function_space: FunctionSpace,
        vis_config: Optional[VisualizationConfig] = None,
        method_colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        include_error_plot: bool = True,
        log_scale_error: bool = True,
        annotate_final: bool = True,
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """
        Create a comparison plot that includes function space, convergence, and error plots.

        Args:
            methods: List of numerical method instances
            function_space: Function space component
            vis_config: Visualization configuration
            method_colors: Dictionary mapping method names to colors
            figsize: Figure size
            include_error_plot: Whether to include error plot
            log_scale_error: Whether to use log scale for error plot
            annotate_final: Whether to annotate final values

        Returns:
            Tuple[plt.Figure, Dict[str, plt.Axes]]: Figure and dictionary of axes
        """
        # Check if there are any methods to plot
        if not methods:
            # Create a simple figure with a message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No methods to visualize",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Comparison Plot (No Data)")
            ax.axis("off")
            return fig, {"main": ax}

        # Create default visualization config if not provided
        if vis_config is None:
            vis_config = VisualizationConfig()

        # Determine subplot layout
        if include_error_plot:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            ax_func = axes[0, 0]
            ax_conv = axes[0, 1]
            ax_error = axes[1, 0]
            ax_summary = axes[1, 1]
            axes_dict = {
                "function": ax_func,
                "convergence": ax_conv,
                "error": ax_error,
                "summary": ax_summary,
            }
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            ax_func = axes[0, 0]
            ax_conv = axes[0, 1]
            ax_summary = axes[1, :]
            axes_dict = {
                "function": ax_func,
                "convergence": ax_conv,
                "summary": ax_summary,
            }

        # Set main title
        fig.suptitle(vis_config.title, fontsize=16)

        # Update method colors
        method_names = [method.name for method in methods]
        method_colors = get_method_colors(
            method_names, palette=vis_config.palette, existing_colors=method_colors
        )

        # Prepare data for convergence and error plots
        convergence_data = {}
        error_data = {}

        for method in methods:
            # Get iteration history
            history = method.get_iteration_history()

            if not history:
                continue

            # Extract convergence data
            convergence_data[method.name] = [h.x_new for h in history]

            # Extract error data
            error_data[method.name] = [h.error for h in history]

        # Create function space plot with paths
        method_paths = {}
        critical_points = []

        for method in methods:
            history = method.get_iteration_history()

            if not history:
                continue

            # Extract path for function space
            if function_space.is_2d:
                # For 2D function - path is list of (x, y) points
                try:
                    # Try to convert iteration history to 2D points
                    path = []
                    for h in history:
                        x_new = h.x_new
                        if isinstance(x_new, (list, np.ndarray)) and len(x_new) >= 2:
                            path.append((x_new[0], x_new[1]))
                        else:
                            # If not a 2D point, skip
                            continue
                    if path:  # Only add if we have valid points
                        method_paths[method.name] = path
                except (IndexError, TypeError, AttributeError):
                    # If we can't extract 2D points, skip this method
                    print(f"Warning: Could not extract 2D path for {method.name}")
            else:
                # For 1D function - path is list of x values
                method_paths[method.name] = [h.x_new for h in history]

            # Add final point to critical points if method converged
            if method.has_converged():
                x_final = method.get_current_x()
                if (
                    function_space.is_2d
                    and isinstance(x_final, (list, np.ndarray))
                    and len(x_final) >= 2
                ):
                    critical_points.append((x_final[0], x_final[1]))
                elif not function_space.is_2d:
                    critical_points.append(x_final)

        # Create function space plot
        function_space.create_matplotlib_figure(
            ax=ax_func,
            path=method_paths.get(method_names[0], []) if method_names else [],
            critical_points=critical_points,
        )

        # Customize function space plot
        ax_func.set_title("Function Space")

        # Create convergence plot
        convergence_plot = ConvergencePlot(
            title="Convergence Plot", color_palette=vis_config.palette
        )

        convergence_plot.create_matplotlib_figure(
            data=convergence_data,
            ax=ax_conv,
            existing_colors=method_colors,
            annotate_final=annotate_final,
        )

        # Create error plot if requested
        if include_error_plot:
            error_plot = ErrorPlot(
                title="Error Plot",
                color_palette=vis_config.palette,
                log_scale=log_scale_error,
            )

            error_plot.create_matplotlib_figure(
                data=error_data,
                ax=ax_error,
                existing_colors=method_colors,
                annotate_final=annotate_final,
            )

        # Create summary information
        PlotFactory._add_summary_info(methods, ax_summary)

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)  # Make room for suptitle

        return fig, axes_dict

    @staticmethod
    def create_interactive_comparison(
        methods: List[BaseNumericalMethod],
        function_space: FunctionSpace,
        vis_config: Optional[VisualizationConfig] = None,
        method_colors: Optional[Dict[str, str]] = None,
        include_error_plot: bool = True,
        log_scale_error: bool = True,
        height: int = 800,
        width: int = 1200,
        surface_plot: bool = False,
    ) -> go.Figure:
        """
        Create an interactive plotly comparison of numerical methods.

        Args:
            methods: List of numerical method instances
            function_space: Function space component
            vis_config: Visualization configuration
            method_colors: Dictionary mapping method names to colors
            include_error_plot: Whether to include error plot
            log_scale_error: Whether to use log scale for error plot
            height: Figure height
            width: Figure width
            surface_plot: Whether to use 3D surface plot for 2D functions

        Returns:
            go.Figure: Plotly figure object
        """
        # Check if there are any methods to plot
        if not methods:
            # Create a simple figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No methods to visualize",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18),
            )
            fig.update_layout(
                title="Comparison Plot (No Data)", height=height, width=width
            )
            return fig

        # Create default visualization config if not provided
        if vis_config is None:
            vis_config = VisualizationConfig()

        # Update method colors
        method_names = [method.name for method in methods]
        method_colors = get_method_colors(
            method_names, palette=vis_config.palette, existing_colors=method_colors
        )

        # Prepare data for convergence and error plots
        convergence_data = {}
        error_data = {}

        for method in methods:
            # Get iteration history
            history = method.get_iteration_history()

            if not history:
                continue

            # Extract convergence data
            convergence_data[method.name] = [h.x_new for h in history]

            # Extract error data
            error_data[method.name] = [h.error for h in history]

        # Determine critical points
        critical_points = []
        for method in methods:
            if method.has_converged():
                critical_points.append(method.get_current_x())

        # Create subplots
        if include_error_plot:
            specs = [[{}, {}], [{}, {}]]
            if surface_plot and function_space.is_2d:
                specs[0][0] = {"type": "scene"}

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Function Space",
                    "Convergence Plot",
                    "Error Plot",
                    "Summary",
                ],
                specs=specs,
                column_widths=[0.6, 0.4],
                row_heights=[0.6, 0.4],
            )
        else:
            specs = [[{}], [{}]]
            if surface_plot and function_space.is_2d:
                specs[0][0] = {"type": "scene"}

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=["Function Space", "Convergence Plot"],
                specs=specs,
                row_heights=[0.7, 0.3],
            )

        # Add function space
        function_plot = function_space.create_plotly_figure(
            critical_points=critical_points,
            plot_type="surface" if surface_plot and function_space.is_2d else "contour",
        )

        # Add all traces from function plot to the main figure
        for trace in function_plot.data:
            fig.add_trace(trace, row=1, col=1)

        # Add convergence plot
        convergence_plot = ConvergencePlot(
            title="Convergence Plot", color_palette=vis_config.palette
        )

        conv_fig = convergence_plot.create_plotly_figure(
            data=convergence_data, existing_colors=method_colors, annotate_final=True
        )

        # Add all traces from convergence plot to the main figure
        for trace in conv_fig.data:
            fig.add_trace(trace, row=1, col=2)

        # Add error plot if requested
        if include_error_plot:
            error_plot = ErrorPlot(
                title="Error Plot",
                color_palette=vis_config.palette,
                log_scale=log_scale_error,
            )

            error_fig = error_plot.create_plotly_figure(
                data=error_data, existing_colors=method_colors, annotate_final=True
            )

            # Add all traces from error plot to the main figure
            for trace in error_fig.data:
                fig.add_trace(trace, row=2, col=1)

            # Add summary information
            PlotFactory._add_plotly_summary(methods, fig, row=2, col=2)

        # Update layout
        fig.update_layout(
            title=vis_config.title, height=height, width=width, template="plotly_white"
        )

        return fig

    @staticmethod
    def _add_summary_info(methods: List[BaseNumericalMethod], ax: plt.Axes):
        """Add summary information to the provided axes."""
        ax.axis("off")  # Turn off axis

        # Create summary text
        summary_text = "Summary Information:\n\n"

        for method in methods:
            # Get iteration history
            history = method.get_iteration_history()
            iteration_count = len(history) if history else 0

            # Get final error
            final_error = history[-1].error if history else float("nan")

            summary_text += f"Method: {method.name}\n"
            summary_text += f"   Converged: {method.has_converged()}\n"
            summary_text += f"   Iterations: {iteration_count}\n"
            summary_text += f"   Final error: {final_error:.6g}\n"

            # Format the final value, handling arrays properly
            final_value = method.get_current_x()
            if isinstance(final_value, (list, tuple, np.ndarray)):
                try:
                    # For vector values, display a string representation
                    if isinstance(final_value, np.ndarray) and final_value.size > 1:
                        # For numpy arrays with multiple values, display the array
                        # with limited precision
                        final_value_str = np.array2string(
                            final_value, precision=4, separator=", "
                        )
                    else:
                        # For single value arrays
                        final_value_str = f"{float(final_value):.6g}"
                except (TypeError, ValueError):
                    # If conversion fails, use default string representation
                    final_value_str = str(final_value)
                summary_text += f"   Final value: {final_value_str}\n\n"
            else:
                # For scalar values, use standard formatting
                summary_text += f"   Final value: {final_value:.6g}\n\n"

        # Add text to axes
        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            family="monospace",
        )

    @staticmethod
    def _add_plotly_summary(
        methods: List[BaseNumericalMethod], fig: go.Figure, row: int = 1, col: int = 1
    ):
        """
        Add summary information to a plotly figure as an annotation.

        Args:
            methods: List of numerical methods
            fig: Plotly figure to add summary to
            row: Row index for subplot (1-based)
            col: Column index for subplot (1-based)
        """
        # Create summary text (HTML format for plotly)
        summary_text = "<b>Method Summary:</b><br><br>"

        for i, method in enumerate(methods):
            history = method.get_iteration_history()

            if not history:
                continue

            # Get convergence status
            converged = method.has_converged()
            status = "Converged" if converged else "Not converged"

            # Get iteration count
            iteration_count = len(history)

            # Get final error
            final_error = history[-1].error if history else float("nan")

            # Get final value
            final_value = method.get_current_x()

            # Add method info to summary
            summary_text += f"<b>{i+1}. {method.name}:</b><br>"
            summary_text += f"&nbsp;&nbsp;&nbsp;Status: {status}<br>"
            summary_text += f"&nbsp;&nbsp;&nbsp;Iterations: {iteration_count}<br>"
            summary_text += f"&nbsp;&nbsp;&nbsp;Final error: {final_error:.2e}<br>"

            # Format the final value, handling arrays properly
            if isinstance(final_value, (list, tuple, np.ndarray)):
                try:
                    # For vector values, display a string representation
                    if isinstance(final_value, np.ndarray) and final_value.size > 1:
                        # For numpy arrays with multiple values, display the array
                        # with limited precision
                        final_value_str = np.array2string(
                            final_value, precision=4, separator=", "
                        )
                    else:
                        # For single value arrays
                        final_value_str = f"{float(final_value):.6g}"
                except (TypeError, ValueError):
                    # If conversion fails, use default string representation
                    final_value_str = str(final_value)
                summary_text += (
                    f"&nbsp;&nbsp;&nbsp;Final value: {final_value_str}<br><br>"
                )
            else:
                # For scalar values, use standard formatting
                summary_text += (
                    f"&nbsp;&nbsp;&nbsp;Final value: {final_value:.6g}<br><br>"
                )

        # Add text to figure
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref=f"x{col}",
            yref=f"y{row}",
            text=summary_text,
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            borderpad=10,
            font=dict(family="Arial", size=10),
        )
