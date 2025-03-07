# plot/components/convergence_plot.py

"""
This module provides a component for visualizing the convergence behavior of
numerical methods toward solutions.
"""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from plot.utils.color_utils import generate_colors, get_method_colors


class ConvergencePlot:
    """
    Component for visualizing how numerical methods converge to solutions.

    This class provides methods for creating plots that show the path of
    convergence for one or more numerical methods, allowing comparison
    of their efficiency and behavior.
    """

    def __init__(
        self,
        title: str = "Convergence Plot",
        xlabel: str = "Iteration",
        ylabel: str = "Value",
        color_palette: str = "D3",
        log_scale: bool = False,
    ):
        """
        Initialize the convergence plot component.

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
        target_value: Optional[float] = None,
        display_iterations: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a matplotlib figure showing the convergence of multiple methods.

        Args:
            data: Dictionary mapping method names to lists of values
            ax: Existing axes to plot on (creates new figure if None)
            figsize: Figure size for new figure
            show_grid: Whether to show grid lines
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors
            legend_loc: Location for the legend
            annotate_final: Whether to annotate final values
            target_value: Optional target value to show as horizontal line
            display_iterations: Whether to display iteration numbers

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Check if data is empty
        if not data or all(not values for values in data.values()):
            ax.set_title(f"{self.title} (No data available)")
            ax.text(
                0.5,
                0.5,
                "No convergence data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            return fig, ax

        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Plot each method's convergence path
        for method_name, values in data.items():
            if not values:  # Skip empty value lists
                continue

            color = self.method_colors.get(method_name, "gray")
            iterations = list(range(len(values)))

            # Line style
            line_style = "-"
            marker = "o" if include_markers else None

            # Plot line
            ax.plot(
                iterations,
                values,
                linestyle=line_style,
                marker=marker,
                markersize=5,
                linewidth=2,
                color=color,
                label=method_name,
            )

            # Annotate final value if requested
            if annotate_final and values:
                final_value = values[-1]
                # Handle numpy arrays by converting to scalar if needed
                if isinstance(final_value, (list, tuple, np.ndarray)):
                    try:
                        # For 2D points, use the norm
                        if isinstance(final_value, np.ndarray) and final_value.size > 1:
                            final_value = np.linalg.norm(final_value)
                        else:
                            # For single value arrays or other iterable types
                            final_value = float(final_value)
                    except (TypeError, ValueError):
                        # If conversion fails, use a default representation
                        final_value = str(final_value)

                ax.annotate(
                    (
                        f"{final_value:.6g}"
                        if isinstance(final_value, (int, float))
                        else final_value
                    ),
                    xy=(
                        len(values) - 1,
                        (
                            values[-1]
                            if not isinstance(values[-1], (list, tuple, np.ndarray))
                            else np.linalg.norm(values[-1])
                        ),
                    ),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        # Add target value line if provided
        if target_value is not None:
            ax.axhline(
                y=target_value, color="black", linestyle="--", alpha=0.5, label="Target"
            )

        # Set axis labels and title
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        # Set x-axis to show iteration numbers if requested
        if display_iterations and data and any(values for values in data.values()):
            # Find the maximum number of iterations across all methods
            max_iterations = max(len(values) for values in data.values() if values)
            ax.set_xticks(range(0, max_iterations, max(1, max_iterations // 10)))

        # Set log scale if requested
        if self.log_scale:
            ax.set_yscale("log")

        # Add grid
        if show_grid:
            ax.grid(alpha=0.3)

        # Add legend
        if data and any(values for values in data.values()):
            ax.legend(loc=legend_loc)

        return fig, ax

    def create_plotly_figure(
        self,
        data: Dict[str, List[float]],
        include_markers: bool = True,
        existing_colors: Optional[Dict[str, str]] = None,
        annotate_final: bool = False,
        target_value: Optional[float] = None,
        display_iterations: bool = True,
        dash_patterns: Optional[List[str]] = None,
        marker_symbols: Optional[List[str]] = None,
        show_gradient: bool = False,
    ) -> go.Figure:
        """
        Create an eye-catching plotly figure showing the convergence of multiple methods.

        Args:
            data: Dictionary mapping method names to lists of values
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors
            annotate_final: Whether to annotate final values
            target_value: Optional target value to show as horizontal line
            display_iterations: Whether to display iteration numbers
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

        # Check if data is empty
        if not data or all(not values for values in data.values()):
            fig.add_annotation(
                text="No convergence data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18, family="Arial, sans-serif", color="rgba(0,0,0,0.7)"),
                align="center",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                borderpad=4,
            )
            fig.update_layout(
                title=dict(
                    text=f"{self.title} (No data available)",
                    font=dict(size=24, family="Arial, sans-serif"),
                    x=0.5,
                    y=0.95,
                ),
                template="plotly_white",
            )
            return fig

        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Plot each method's convergence path
        for i, (method_name, values) in enumerate(data.items()):
            if not values:  # Skip empty value lists
                continue

            color = self.method_colors.get(method_name, "gray")
            iterations = list(range(len(values)))

            # Determine marker and line settings with enhanced styling
            mode = "lines+markers" if include_markers else "lines"
            dash = dash_patterns[i % len(dash_patterns)]
            symbol = marker_symbols[i % len(marker_symbols)]

            # Create gradient color effect along path if requested
            if show_gradient and len(values) > 10:
                # Generate gradient colors
                color_scale = px.colors.sequential.Viridis
                gradient_colors = [
                    color_scale[int(j * (len(color_scale) - 1) / (len(values) - 1))]
                    for j in range(len(values))
                ]

                # Add segments with gradient colors
                for j in range(len(values) - 1):
                    fig.add_trace(
                        go.Scatter(
                            x=iterations[j : j + 2],
                            y=values[j : j + 2],
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

                # Add markers separately
                if include_markers:
                    fig.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=values,
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
                                f"{method_name}<br>Iteration: {it}<br>Value: "
                                + (
                                    f"{v:.6g}"
                                    if isinstance(v, (int, float))
                                    else str(v)
                                )
                                for it, v in zip(iterations, values)
                            ],
                        )
                    )
            else:
                # Add trace with enhanced styling
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=values,
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
                            f"{method_name}<br>Iteration: {it}<br>Value: "
                            + (f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
                            for it, v in zip(iterations, values)
                        ],
                    )
                )

            # Add annotations for final values if requested with enhanced styling
            if annotate_final and values:
                final_value = values[-1]
                # Handle numpy arrays by converting to scalar if needed
                y_position = final_value
                if isinstance(final_value, (list, tuple, np.ndarray)):
                    try:
                        # For 2D points, use the norm
                        if isinstance(final_value, np.ndarray) and final_value.size > 1:
                            y_position = np.linalg.norm(final_value)
                            final_value = y_position
                        else:
                            # For single value arrays or other iterable types
                            final_value = float(final_value)
                            y_position = final_value
                    except (TypeError, ValueError):
                        # If conversion fails, use a default representation
                        final_value = str(final_value)
                        y_position = 0  # Default position

                # Add enhanced annotation
                fig.add_annotation(
                    x=len(values) - 1,
                    y=y_position,
                    text=(
                        f"{final_value:.6g}"
                        if isinstance(final_value, (int, float))
                        else final_value
                    ),
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
                        x=[len(values) - 1],
                        y=[y_position],
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
                        hovertext=(
                            f"{method_name}<br>Final Value: "
                            + (
                                f"{final_value:.6g}"
                                if isinstance(final_value, (int, float))
                                else str(final_value)
                            )
                        ),
                    )
                )

        # Add target value line if provided with enhanced styling
        if target_value is not None:
            # Get the maximum length of all value lists for line length
            max_iterations = max(len(values) for values in data.values() if values)

            # Add enhanced target line
            fig.add_shape(
                type="line",
                x0=0,
                y0=target_value,
                x1=max_iterations - 1,
                y1=target_value,
                line=dict(
                    color="rgba(0, 0, 0, 0.7)",
                    width=2,
                    dash="dash",
                ),
            )

            # Add label for target line
            fig.add_annotation(
                x=max_iterations * 0.05,
                y=target_value,
                text=f"Target: {target_value:.6g}",
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
                    name=f"Target ({target_value:.6g})",
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
            max_iterations = max(len(values) for values in data.values() if values)
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(0, max_iterations, max(1, max_iterations // 10))),
            )

        # Set log scale if requested
        if self.log_scale:
            fig.update_yaxes(type="log")

        return fig

    def create_animation_frames(
        self,
        data: Dict[str, List[float]],
        fps: int = 5,
        transition_duration: int = 500,
        include_markers: bool = True,
        existing_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """
        Create an animated plotly figure showing the convergence process with enhanced styling.

        Args:
            data: Dictionary mapping method names to lists of values
            fps: Frames per second
            transition_duration: Duration of transition between frames in ms
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors

        Returns:
            go.Figure: Animated plotly figure with eye-catching styling
        """
        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Determine the maximum number of iterations
        max_iterations = max(len(values) for values in data.values() if values)

        # Create base figure with enhanced styling
        fig = go.Figure()

        # Check if data is empty
        if not data or all(not values for values in data.values()):
            fig.add_annotation(
                text="No convergence data available for animation",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18, family="Arial, sans-serif", color="rgba(0,0,0,0.7)"),
                align="center",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                borderpad=4,
            )
            fig.update_layout(
                title=dict(
                    text=f"{self.title} (No data available)",
                    font=dict(size=24, family="Arial, sans-serif"),
                    x=0.5,
                    y=0.95,
                ),
                template="plotly_white",
            )
            return fig

        # Add empty traces for each method with enhanced styling
        for method_name, values in data.items():
            if not values:  # Skip empty value lists
                continue

            color = self.method_colors.get(method_name, "gray")
            mode = "lines+markers" if include_markers else "lines"

            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode=mode,
                    name=method_name,
                    line=dict(
                        color=color,
                        width=3,
                    ),
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=1, color="white"),
                        symbol="circle",
                    ),
                    hoverinfo="text",
                )
            )

        # Create animation frames with enhanced styling
        frames = []
        for frame_idx in range(1, max_iterations + 1):
            frame_data = []

            for method_idx, (method_name, values) in enumerate(data.items()):
                if not values:  # Skip empty value lists
                    continue

                # Ensure we don't go beyond the available data for each method
                valid_idx = min(frame_idx, len(values))
                x_values = list(range(valid_idx))
                y_values = values[:valid_idx]

                # Create hover text
                hover_text = [
                    f"{method_name}<br>Iteration: {it}<br>Value: "
                    + (f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
                    for it, v in zip(x_values, y_values)
                ]

                # Add data up to the current frame with enhanced styling
                frame_data.append(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        hoverinfo="text",
                        hovertext=hover_text,
                        line=dict(width=3),
                        marker=dict(
                            size=8,
                            line=dict(width=1, color="white"),
                        ),
                    )
                )

            frames.append(go.Frame(data=frame_data, name=f"frame{frame_idx}"))

        # Add frames to figure
        fig.frames = frames

        # Create animation controls with enhanced styling
        sliders = [
            {
                "steps": [
                    {
                        "args": [
                            [f"frame{k+1}"],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": transition_duration},
                            },
                        ],
                        "label": str(k + 1),
                        "method": "animate",
                    }
                    for k in range(0, max_iterations, max(1, max_iterations // 10))
                ],
                "active": 0,
                "currentvalue": {
                    "prefix": "Iteration: ",
                    "font": {"size": 14, "family": "Arial, sans-serif"},
                },
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "pad": {"t": 60, "b": 10},
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "rgba(0, 0, 0, 0.2)",
                "borderwidth": 1,
            }
        ]

        # Configure figure layout with enhanced styling
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
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 1000 // fps, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": transition_duration},
                                },
                            ],
                        },
                        {
                            "label": "❚❚ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                    "x": 0.1,
                    "y": 0,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "pad": {"t": 5, "r": 10},
                    "font": {"size": 14, "family": "Arial, sans-serif"},
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                    "bordercolor": "rgba(0, 0, 0, 0.2)",
                    "borderwidth": 1,
                }
            ],
            sliders=sliders,
            legend=dict(
                font=dict(
                    size=14,
                    family="Arial, sans-serif",
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            margin=dict(l=80, r=40, t=100, b=120),  # Extra bottom margin for controls
            # No height or width - let Plotly use browser dimensions
        )

        # Set log scale if requested
        if self.log_scale:
            fig.update_yaxes(type="log")

        return fig
