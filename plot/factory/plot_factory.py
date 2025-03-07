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
        height: int = None,
        width: int = None,
        surface_plot: bool = None,
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
            height: Figure height (None for auto-sizing)
            width: Figure width (None for auto-sizing)
            surface_plot: Whether to use 3D surface plot for 2D functions (None to use vis_config setting)

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
                font=dict(size=24, color="#333333", family="Arial, sans-serif"),
            )
            fig.update_layout(
                title={
                    "text": "<b>Comparison Plot (No Data)</b>",
                    "font": {"size": 28, "color": "#333333"},
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                height=height,
                width=width,
                autosize=True,
                template="plotly_white",
                paper_bgcolor="rgba(255, 255, 255, 0.95)",
            )
            return fig

        # Create default visualization config if not provided
        if vis_config is None:
            vis_config = VisualizationConfig()

        # Use config value for surface_plot if not explicitly specified
        if surface_plot is None and hasattr(vis_config, "use_plotly_3d"):
            surface_plot = vis_config.use_plotly_3d

        # Ensure surface_plot is a boolean
        surface_plot = bool(surface_plot)

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

        # Create subplots with explicit types to avoid NoneType errors
        if include_error_plot:
            # 2x2 grid for function, convergence, error, and summary
            if surface_plot and function_space.is_2d:
                # Use scene type for 3D surface plot
                specs = [
                    [{"type": "scene"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}],
                ]
            else:
                # Use xy type for 2D plots
                specs = [
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}],
                ]

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    f"<b>{function_space.title}</b>",
                    "<b>Convergence Progress</b>",
                    "<b>Error Reduction</b>",
                    "<b>Method Performance Summary</b>",
                ],
                specs=specs,
                column_widths=[0.6, 0.4],
                row_heights=[0.6, 0.4],
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )
        else:
            # 2x1 grid for function and convergence
            if surface_plot and function_space.is_2d:
                # Use scene type for 3D surface plot
                specs = [[{"type": "scene"}], [{"type": "xy"}]]
            else:
                # Use xy type for 2D plots
                specs = [[{"type": "xy"}], [{"type": "xy"}]]

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=[
                    f"<b>{function_space.title}</b>",
                    "<b>Convergence Progress</b>",
                ],
                specs=specs,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.12,
            )

        # Add function space
        plot_type = "surface" if surface_plot and function_space.is_2d else "contour"

        # Use colorscale from config if available
        colorscale = None
        if hasattr(vis_config, "plotly_colorscales"):
            if plot_type == "surface" and "surface" in vis_config.plotly_colorscales:
                colorscale = vis_config.plotly_colorscales["surface"]
            elif plot_type == "contour" and "contour" in vis_config.plotly_colorscales:
                colorscale = vis_config.plotly_colorscales["contour"]

        try:
            function_plot = function_space.create_plotly_figure(
                critical_points=critical_points,
                plot_type=plot_type,
                colorscale=colorscale,
            )

            # Add all traces from function plot to the main figure
            for trace in function_plot.data:
                fig.add_trace(trace, row=1, col=1)
        except Exception as e:
            print(f"Warning: Error creating function plot: {e}")
            # Add a fallback trace
            if plot_type == "surface":
                fig.add_trace(
                    go.Surface(
                        x=np.linspace(-1, 1, 10),
                        y=np.linspace(-1, 1, 10),
                        z=np.zeros((10, 10)),
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    row=1,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Contour(
                        x=np.linspace(-1, 1, 10),
                        y=np.linspace(-1, 1, 10),
                        z=np.zeros((10, 10)),
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    row=1,
                    col=1,
                )

        # Add convergence plot
        convergence_plot = ConvergencePlot(
            title="Convergence Plot", color_palette=vis_config.palette
        )

        try:
            conv_fig = convergence_plot.create_plotly_figure(
                data=convergence_data,
                existing_colors=method_colors,
                annotate_final=True,
            )

            # Add all traces from convergence plot to the main figure
            for trace in conv_fig.data:
                fig.add_trace(trace, row=1, col=2)
        except Exception as e:
            print(f"Warning: Error creating convergence plot: {e}")
            # Add a fallback trace
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(color="gray"),
                    name="No data",
                ),
                row=1,
                col=2,
            )

        # Add error plot if requested
        if include_error_plot:
            error_plot = ErrorPlot(
                title="Error Plot",
                color_palette=vis_config.palette,
                log_scale=log_scale_error,
            )

            try:
                error_fig = error_plot.create_plotly_figure(
                    data=error_data, existing_colors=method_colors, annotate_final=True
                )

                # Add all traces from error plot to the main figure
                for trace in error_fig.data:
                    fig.add_trace(trace, row=2, col=1)
            except Exception as e:
                print(f"Warning: Error creating error plot: {e}")
                # Add a fallback trace
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode="markers",
                        marker=dict(color="gray"),
                        name="No data",
                    ),
                    row=2,
                    col=1,
                )

            # Add summary information
            try:
                PlotFactory._add_plotly_summary(methods, fig, row=2, col=2)
            except Exception as e:
                print(f"Warning: Error creating summary: {e}")
                # Add a fallback annotation
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref=f"x2",
                    yref=f"y2",
                    text="Method summary unavailable",
                    showarrow=False,
                    font=dict(size=14),
                )

        # Get font settings from config if available
        font_settings = {"family": "Arial, sans-serif", "size": 14, "color": "#333333"}
        if hasattr(vis_config, "plotly_font") and vis_config.plotly_font:
            font_settings = vis_config.plotly_font

        # Get margin settings from config if available
        margin_settings = {"l": 60, "r": 30, "t": 80, "b": 100, "autoexpand": True}
        if hasattr(vis_config, "plotly_margin") and vis_config.plotly_margin:
            margin_settings = vis_config.plotly_margin

        # Get template from config if available
        template = "plotly_white"
        if hasattr(vis_config, "plotly_template"):
            template = vis_config.plotly_template

        # Update layout with enhanced styling
        fig.update_layout(
            title={
                "text": f"<b>{vis_config.title}</b>",
                "font": {"size": 24, "color": "#333333"},
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            height=height,  # None for auto-sizing
            width=width,  # None for auto-sizing
            autosize=True,  # Allow browser to set dimensions
            template=template,
            font=font_settings,
            margin=margin_settings,
            hoverlabel={"bgcolor": "white", "font_size": 12, "bordercolor": "#333333"},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.2 if include_error_plot else -0.15,
                "xanchor": "center",
                "x": 0.5,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "rgba(0, 0, 0, 0.2)",
                "borderwidth": 1,
                "font": {"size": 12},
            },
            paper_bgcolor=vis_config.background_color,
            plot_bgcolor="rgba(250, 250, 250, 0.95)",
        )

        # Update x and y axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor="rgba(0, 0, 0, 0.3)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0, 0, 0, 0.5)",
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor="rgba(0, 0, 0, 0.3)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0, 0, 0, 0.5)",
        )

        # Set 3D scene options for surface plot
        if surface_plot and function_space.is_2d:
            fig.update_scenes(
                aspectratio={"x": 1, "y": 1, "z": 0.8},
                camera={
                    "up": {"x": 0, "y": 0, "z": 1},
                    "center": {"x": 0, "y": 0, "z": 0},
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.2},
                },
                dragmode="turntable",
                xaxis_title={"text": "<b>x</b>", "font": font_settings},
                yaxis_title={"text": "<b>y</b>", "font": font_settings},
                zaxis_title={"text": "<b>f(x,y)</b>", "font": font_settings},
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
        summary_text = (
            "<b style='font-size:14px;'>Method Performance Summary:</b><br><br>"
        )

        for i, method in enumerate(methods):
            history = method.get_iteration_history()

            if not history:
                continue

            # Get convergence status
            converged = method.has_converged()
            status_color = (
                "#28a745" if converged else "#dc3545"
            )  # Green if converged, red if not
            status = "Converged" if converged else "Not converged"

            # Get iteration count
            iteration_count = len(history)

            # Get final error
            final_error = history[-1].error if history else float("nan")

            # Get final value
            final_value = method.get_current_x()

            # Add method info to summary
            summary_text += f"<b style='font-size:13px;'>{i+1}. {method.name}:</b><br>"
            summary_text += f"&nbsp;&nbsp;&nbsp;Status: <span style='color:{status_color};font-weight:bold;'>{status}</span><br>"
            summary_text += (
                f"&nbsp;&nbsp;&nbsp;Iterations: <b>{iteration_count}</b><br>"
            )
            summary_text += (
                f"&nbsp;&nbsp;&nbsp;Final error: <b>{final_error:.2e}</b><br>"
            )

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
                    f"&nbsp;&nbsp;&nbsp;Final value: <b>{final_value_str}</b><br><br>"
                )
            else:
                # For scalar values, use standard formatting
                summary_text += (
                    f"&nbsp;&nbsp;&nbsp;Final value: <b>{final_value:.6g}</b><br><br>"
                )

        # Add text to figure with enhanced styling
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
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            borderpad=15,
            font=dict(family="Arial, sans-serif", size=12),
            width=370,
            height=320,
            opacity=0.95,
        )
