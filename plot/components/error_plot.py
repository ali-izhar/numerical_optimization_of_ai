# plot/components/error_plot.py

"""
This module provides a component for visualizing error metrics during the
execution of numerical methods.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
        color_palette: str = "D3",
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
        include_markers: bool = True,
        existing_colors: Optional[Dict[str, str]] = None,
        annotate_final: bool = False,
        tolerance_line: Optional[float] = None,
        display_iterations: bool = True,
        y_limit: Optional[Tuple[Optional[float], Optional[float]]] = None,
        dash_patterns: Optional[List[str]] = None,
        marker_symbols: Optional[List[str]] = None,
        show_gradient: bool = False,
    ) -> go.Figure:
        """
        Create an eye-catching plotly figure showing error metrics for multiple methods.

        Args:
            data: Dictionary mapping method names to lists of error values
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors
            annotate_final: Whether to annotate final error values
            tolerance_line: Value for tolerance line to show
            display_iterations: Whether to display iteration numbers
            y_limit: Tuple of (min, max) for y-axis limits
            dash_patterns: Optional list of dash patterns to use for lines
            marker_symbols: Optional list of marker symbols to use
            show_gradient: Whether to show gradient along the path

        Returns:
            go.Figure: Plotly figure object with eye-catching styling
        """
        fig = go.Figure()

        # Default dash patterns if not provided
        if dash_patterns is None:
            dash_patterns = ["solid", "dash", "dot", "dashdot", "longdash"]

        # Default marker symbols if not provided
        if marker_symbols is None:
            marker_symbols = ["circle", "diamond", "square", "triangle-up", "x"]

        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Plot each method's error values
        for i, (method_name, error_values) in enumerate(data.items()):
            color = self.method_colors.get(method_name, "gray")
            iterations = list(range(len(error_values)))

            # Handle any negative or zero values if using log scale
            if self.log_scale:
                # Replace zeros or negative values with a small positive number
                error_values = [max(1e-15, v) for v in error_values]

            # Determine marker settings with enhanced styling
            mode = "lines+markers" if include_markers else "lines"
            dash = dash_patterns[i % len(dash_patterns)]
            symbol = marker_symbols[i % len(marker_symbols)]

            # Create gradient color effect along path if requested
            if show_gradient and len(error_values) > 10:
                color_scale = px.colors.sequential.Viridis
                gradient_colors = [
                    color_scale[
                        int(i * (len(color_scale) - 1) / (len(error_values) - 1))
                    ]
                    for i in range(len(error_values))
                ]

                # Add segments with gradient colors
                for j in range(len(error_values) - 1):
                    fig.add_trace(
                        go.Scatter(
                            x=iterations[j : j + 2],
                            y=error_values[j : j + 2],
                            mode="lines",
                            line=dict(
                                color=gradient_colors[j],
                                width=3,
                            ),
                            showlegend=j == 0,
                            name=method_name if j == 0 else None,
                            hoverinfo="skip",
                        )
                    )

                # Add markers separately if needed
                if include_markers:
                    fig.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=error_values,
                            mode="markers",
                            marker=dict(
                                size=8,
                                symbol=symbol,
                                color=color,
                                line=dict(width=1, color="white"),
                            ),
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=[
                                f"{method_name}<br>Iteration: {it}<br>Error: {err:.2e}"
                                for it, err in zip(iterations, error_values)
                            ],
                        )
                    )
            else:
                # Add regular trace with enhanced styling
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=error_values,
                        mode=mode,
                        name=method_name,
                        line=dict(
                            color=color,
                            width=3,
                            dash=dash,
                        ),
                        marker=dict(
                            size=8,
                            color=color,
                            symbol=symbol,
                            line=dict(width=1, color="white"),
                        ),
                        hoverinfo="text",
                        hovertext=[
                            f"{method_name}<br>Iteration: {it}<br>Error: {err:.2e}"
                            for it, err in zip(iterations, error_values)
                        ],
                    )
                )

            # Add annotations for final values if requested with enhanced styling
            if annotate_final and error_values:
                final_x = len(error_values) - 1
                final_y = error_values[-1]

                fig.add_annotation(
                    x=final_x,
                    y=final_y,
                    text=f"{final_y:.2e}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=30,
                    ay=0,
                    font=dict(
                        color=color,
                        size=12,
                        family="Arial, sans-serif",
                    ),
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                )

                # Add a special marker for the final point
                fig.add_trace(
                    go.Scatter(
                        x=[final_x],
                        y=[final_y],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=color,
                            symbol="star",
                            line=dict(width=2, color="white"),
                        ),
                        name=f"{method_name} (Final)",
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"{method_name}<br>Final Error: {final_y:.2e}",
                    )
                )

        # Add tolerance line if provided with enhanced styling
        if tolerance_line is not None:
            # Get the maximum length of all error lists
            max_iterations = max(len(values) for values in data.values())

            # Add dashed line with enhanced styling
            fig.add_shape(
                type="line",
                x0=0,
                y0=tolerance_line,
                x1=max_iterations - 1,
                y1=tolerance_line,
                line=dict(
                    color="rgba(0, 0, 0, 0.7)",
                    width=2,
                    dash="dash",
                ),
                name="Tolerance",
            )

            # Add label for tolerance line
            fig.add_annotation(
                x=max_iterations * 0.05,
                y=tolerance_line,
                text=f"Tolerance: {tolerance_line:.2e}",
                showarrow=False,
                yshift=10,
                font=dict(
                    size=12,
                    color="rgba(0, 0, 0, 0.7)",
                    family="Arial, sans-serif",
                ),
                bordercolor="rgba(0, 0, 0, 0.7)",
                borderwidth=1,
                borderpad=4,
                bgcolor="rgba(255, 255, 255, 0.8)",
            )

            # Add visible trace for legend entry
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(
                        color="rgba(0, 0, 0, 0.7)",
                        width=2,
                        dash="dash",
                    ),
                    name=f"Tolerance ({tolerance_line:.2e})",
                )
            )

        # Set axis titles and figure title with enhanced styling
        fig.update_layout(
            title=dict(
                text=self.title,
                font=dict(
                    size=24,
                    family="Arial, sans-serif",
                ),
                x=0.5,
                y=0.95,
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial, sans-serif",
            ),
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(
                title=dict(
                    text=self.xlabel,
                    font=dict(
                        size=18,
                        family="Arial, sans-serif",
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(220, 220, 220, 0.5)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
            ),
            yaxis=dict(
                title=dict(
                    text=self.ylabel,
                    font=dict(
                        size=18,
                        family="Arial, sans-serif",
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(220, 220, 220, 0.5)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
            ),
            legend=dict(
                font=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(l=80, r=40, t=100, b=80),
            # No height or width - let Plotly use browser dimensions
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

    def create_plotly_convergence_rate_figure(
        self,
        data: Dict[str, List[float]],
        existing_colors: Optional[Dict[str, str]] = None,
        include_reference_lines: bool = True,
    ) -> go.Figure:
        """
        Create a Plotly figure showing the convergence rate for each method.

        The convergence rate is estimated by computing the ratio of consecutive errors.

        Args:
            data: Dictionary mapping method names to lists of error values
            existing_colors: Dictionary mapping method names to colors
            include_reference_lines: Whether to include reference lines for convergence rates

        Returns:
            go.Figure: Plotly figure with convergence rate visualization
        """
        fig = go.Figure()

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

            avg_rate = np.mean(rates)

            # Add trace with enhanced styling
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=rates,
                    mode="lines+markers",
                    name=f"{method_name} (avg: {avg_rate:.3f})",
                    line=dict(
                        color=color,
                        width=2,
                    ),
                    marker=dict(
                        size=8,
                        color=color,
                        symbol="circle",
                        line=dict(width=1, color="white"),
                    ),
                    hoverinfo="text",
                    hovertext=[
                        f"{method_name}<br>Iteration {it}<br>Rate: {r:.4f}"
                        for it, r in zip(iterations, rates)
                    ],
                )
            )

            # Add a horizontal line for the average rate
            fig.add_shape(
                type="line",
                x0=min(iterations),
                y0=avg_rate,
                x1=max(iterations),
                y1=avg_rate,
                line=dict(
                    color=color,
                    width=1,
                    dash="dot",
                ),
            )

        # Add reference lines if requested
        if include_reference_lines:
            # Find min and max iterations across all methods for the reference lines
            all_iterations = []
            for method_name, error_values in data.items():
                if len(error_values) >= 3:
                    for i in range(1, len(error_values)):
                        all_iterations.append(i)

            if all_iterations:
                min_it = min(all_iterations)
                max_it = max(all_iterations)

                # Add horizontal line at y=1 (no convergence)
                fig.add_shape(
                    type="line",
                    x0=min_it,
                    y0=1.0,
                    x1=max_it,
                    y1=1.0,
                    line=dict(
                        color="rgba(0, 0, 0, 0.7)",
                        width=2,
                        dash="dash",
                    ),
                )

                fig.add_annotation(
                    x=min_it,
                    y=1.0,
                    text="No Convergence",
                    showarrow=False,
                    yshift=10,
                    xshift=5,
                    font=dict(
                        size=12,
                        color="rgba(0, 0, 0, 0.7)",
                    ),
                )

                # Add horizontal line at y=0.5 (quadratic convergence)
                fig.add_shape(
                    type="line",
                    x0=min_it,
                    y0=0.5,
                    x1=max_it,
                    y1=0.5,
                    line=dict(
                        color="rgba(0, 128, 0, 0.7)",
                        width=2,
                        dash="dash",
                    ),
                )

                fig.add_annotation(
                    x=min_it,
                    y=0.5,
                    text="Quadratic Convergence",
                    showarrow=False,
                    yshift=-15,
                    xshift=5,
                    font=dict(
                        size=12,
                        color="rgba(0, 128, 0, 0.7)",
                    ),
                )

                # Add reference line entries to legend
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(
                            color="rgba(0, 0, 0, 0.7)",
                            width=2,
                            dash="dash",
                        ),
                        name="No Convergence (r=1.0)",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(
                            color="rgba(0, 128, 0, 0.7)",
                            width=2,
                            dash="dash",
                        ),
                        name="Quadratic Convergence (r=0.5)",
                    )
                )

        # Enhanced styling for the figure
        fig.update_layout(
            title=dict(
                text="Convergence Rate Analysis",
                font=dict(
                    size=24,
                    family="Arial, sans-serif",
                ),
                x=0.5,
                y=0.95,
            ),
            xaxis=dict(
                title=dict(
                    text="Iteration",
                    font=dict(
                        size=18,
                        family="Arial, sans-serif",
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(220, 220, 220, 0.5)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
            ),
            yaxis=dict(
                title=dict(
                    text="Rate (e_{i+1} / e_i)",
                    font=dict(
                        size=18,
                        family="Arial, sans-serif",
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(220, 220, 220, 0.5)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
            ),
            legend=dict(
                font=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
            template="plotly_white",
            hovermode="closest",
            # No height or width - let Plotly use browser dimensions
        )

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
