# plot/components/convergence_plot.py

"""
This module provides a component for visualizing the convergence behavior of
numerical methods toward solutions.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
        color_palette: str = "Set1",
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
        height: int = 500,
        width: int = 800,
        include_markers: bool = True,
        existing_colors: Optional[Dict[str, str]] = None,
        annotate_final: bool = False,
        target_value: Optional[float] = None,
        display_iterations: bool = True,
    ) -> go.Figure:
        """
        Create a plotly figure showing the convergence of multiple methods.

        Args:
            data: Dictionary mapping method names to lists of values
            height: Figure height
            width: Figure width
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors
            annotate_final: Whether to annotate final values
            target_value: Optional target value to show as horizontal line
            display_iterations: Whether to display iteration numbers

        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()

        # Check if data is empty
        if not data or all(not values for values in data.values()):
            fig.add_annotation(
                text="No convergence data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
            fig.update_layout(
                title=f"{self.title} (No data available)", height=height, width=width
            )
            return fig

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

            # Determine marker settings
            mode = "lines+markers" if include_markers else "lines"

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=values,
                    mode=mode,
                    name=method_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                )
            )

            # Add annotations for final values if requested
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
                    arrowwidth=1,
                    arrowcolor=color,
                    ax=20,
                    ay=0,
                    font=dict(color=color, size=10),
                )

        # Add target value line if provided
        if target_value is not None:
            fig.add_shape(
                type="line",
                x0=0,
                y0=target_value,
                x1=max(len(values) for values in data.values()),
                y1=target_value,
                line=dict(color="black", width=1, dash="dash"),
            )

            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[target_value],
                    mode="markers",
                    marker=dict(color="black", size=0),
                    name="Target",
                    hoverinfo="none",
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
        Create an animated plotly figure showing the convergence process.

        Args:
            data: Dictionary mapping method names to lists of values
            fps: Frames per second
            transition_duration: Duration of transition between frames in ms
            include_markers: Whether to include markers on lines
            existing_colors: Dictionary mapping method names to colors

        Returns:
            go.Figure: Animated plotly figure
        """
        # Update color dictionary with any existing colors
        self.method_colors = get_method_colors(
            list(data.keys()),
            palette=self.color_palette,
            existing_colors=existing_colors or self.method_colors,
        )

        # Determine the maximum number of iterations
        max_iterations = max(len(values) for values in data.values())

        # Create base figure
        fig = go.Figure()

        # Add empty traces for each method
        for method_name, values in data.items():
            color = self.method_colors.get(method_name, "gray")
            mode = "lines+markers" if include_markers else "lines"

            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode=mode,
                    name=method_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                )
            )

        # Create animation frames
        frames = []
        for frame_idx in range(1, max_iterations + 1):
            frame_data = []

            for method_idx, (method_name, values) in enumerate(data.items()):
                # Ensure we don't go beyond the available data for each method
                valid_idx = min(frame_idx, len(values))

                # Add data up to the current frame
                frame_data.append(
                    go.Scatter(x=list(range(valid_idx)), y=values[:valid_idx])
                )

            frames.append(go.Frame(data=frame_data, name=f"frame{frame_idx}"))

        # Add frames to figure
        fig.frames = frames

        # Create animation controls
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
                "currentvalue": {"prefix": "Iteration: "},
                "len": 0.9,
            }
        ]

        # Configure figure layout
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
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
                            "label": "Pause",
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
                }
            ],
            sliders=sliders,
        )

        # Set log scale if requested
        if self.log_scale:
            fig.update_yaxes(type="log")

        return fig
