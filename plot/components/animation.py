# plot/components/animation.py

"""This module provides components for creating animations of numerical methods."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot.utils.color_utils import generate_colors, get_method_colors
from plot.components.function_space import FunctionSpace


class MethodAnimation:
    """
    Component for creating animations of numerical methods.

    This class provides methods for creating animations that show the
    step-by-step execution of numerical methods, visualizing how they
    converge to solutions.
    """

    def __init__(
        self,
        function_space: FunctionSpace,
        title: str = "Method Animation",
        method_colors: Optional[Dict[str, str]] = None,
        color_palette: str = "Set1",
    ):
        """
        Initialize the method animation component.

        Args:
            function_space: FunctionSpace object for the function being optimized/analyzed
            title: Title for the animation
            method_colors: Dictionary mapping method names to colors
            color_palette: Color palette for methods if colors not provided
        """
        self.function_space = function_space
        self.title = title
        self.color_palette = color_palette
        self.method_colors = method_colors or {}

    def create_matplotlib_animation(
        self,
        method_paths: Dict[str, List[Union[float, Tuple[float, float]]]],
        error_data: Optional[Dict[str, List[float]]] = None,
        figsize: Tuple[int, int] = (12, 6),
        interval: int = 200,
        show_error_plot: bool = True,
        critical_points: Optional[List[Union[float, Tuple[float, float]]]] = None,
        save_path: Optional[str] = None,
        dpi: int = 100,
        fps: int = 10,
    ) -> animation.FuncAnimation:
        """
        Create a matplotlib animation of numerical methods.

        Args:
            method_paths: Dictionary mapping method names to lists of points
            error_data: Dictionary mapping method names to lists of error values
            figsize: Figure size
            interval: Interval between frames in milliseconds
            show_error_plot: Whether to show error plot alongside function space
            critical_points: Critical points to highlight
            save_path: Path to save animation (if None, animation is not saved)
            dpi: DPI for saved animation
            fps: Frames per second for saved animation

        Returns:
            animation.FuncAnimation: Matplotlib animation object
        """
        # Determine if we need a subplot for error
        if show_error_plot and error_data:
            fig, (ax_func, ax_error) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax_func = plt.subplots(figsize=figsize)
            ax_error = None

        # Update method colors
        self.method_colors = get_method_colors(
            list(method_paths.keys()),
            palette=self.color_palette,
            existing_colors=self.method_colors,
        )

        # Initialize paths and points
        paths = {}
        points = {}
        error_lines = {}
        error_points = {}

        # Set main title
        fig.suptitle(self.title, fontsize=14)

        # Initialize function space plot
        self.function_space.create_matplotlib_figure(
            ax=ax_func, critical_points=critical_points
        )

        # Create initial empty paths and points for each method
        for method_name, path in method_paths.items():
            color = self.method_colors.get(method_name, "gray")

            # For 2D functions
            if self.function_space.is_2d:
                # Initialize empty path line
                (paths[method_name],) = ax_func.plot(
                    [], [], "-", color=color, linewidth=1.5, label=method_name
                )

                # Initialize current point
                (points[method_name],) = ax_func.plot(
                    [], [], "o", color=color, markersize=8
                )
            else:
                # For 1D functions, we need to calculate y values
                (paths[method_name],) = ax_func.plot(
                    [], [], "-", color=color, linewidth=1.5, label=method_name
                )

                # Initialize current point
                (points[method_name],) = ax_func.plot(
                    [], [], "o", color=color, markersize=8
                )

        # Initialize error plot if needed
        if show_error_plot and error_data and ax_error is not None:
            ax_error.set_title("Error vs Iteration")
            ax_error.set_xlabel("Iteration")
            ax_error.set_ylabel("Error")
            ax_error.set_yscale("log")  # Log scale for error
            ax_error.grid(alpha=0.3)

            # Initialize error lines and points
            for method_name, errors in error_data.items():
                color = self.method_colors.get(method_name, "gray")

                # Initialize empty error line
                (error_lines[method_name],) = ax_error.plot(
                    [], [], "-", color=color, linewidth=1.5, label=method_name
                )

                # Initialize current error point
                (error_points[method_name],) = ax_error.plot(
                    [], [], "o", color=color, markersize=8
                )

            # Add legend to error plot
            ax_error.legend()

        # Add legend to function plot
        ax_func.legend()

        # Determine max number of frames
        max_path_length = max(len(path) for path in method_paths.values())

        # Function to initialize animation
        def init():
            artists = []
            for method_name in method_paths:
                # Reset path data
                paths[method_name].set_data([], [])
                artists.append(paths[method_name])

                # Reset point data
                points[method_name].set_data([], [])
                artists.append(points[method_name])

                # Reset error data if applicable
                if (
                    show_error_plot
                    and error_data
                    and ax_error is not None
                    and method_name in error_data
                ):
                    error_lines[method_name].set_data([], [])
                    artists.append(error_lines[method_name])

                    error_points[method_name].set_data([], [])
                    artists.append(error_points[method_name])

            return artists

        # Function to update animation at each frame
        def update(frame):
            artists = []

            for method_name, path in method_paths.items():
                # Skip if we've reached the end of this method's path
                if frame >= len(path):
                    artists.append(paths[method_name])
                    artists.append(points[method_name])
                    continue

                # Update paths based on dimensionality
                if self.function_space.is_2d:
                    # 2D function
                    x_path = [p[0] for p in path[: frame + 1]]
                    y_path = [p[1] for p in path[: frame + 1]]

                    # Update path
                    paths[method_name].set_data(x_path, y_path)

                    # Update current point
                    points[method_name].set_data([path[frame][0]], [path[frame][1]])
                else:
                    # 1D function
                    if isinstance(path[0], (int, float, np.number)):
                        # Path is a list of x values
                        x_path = path[: frame + 1]
                        y_path = [self.function_space.func(x) for x in x_path]
                    else:
                        # Path is a list of (x, y) tuples
                        x_path = [p[0] for p in path[: frame + 1]]
                        y_path = [
                            p[1] if len(p) > 1 else self.function_space.func(p[0])
                            for p in path[: frame + 1]
                        ]

                    # Update path
                    paths[method_name].set_data(x_path, y_path)

                    # Update current point
                    if isinstance(path[frame], (int, float, np.number)):
                        x = path[frame]
                        y = self.function_space.func(x)
                    else:
                        x = path[frame][0]
                        y = (
                            path[frame][1]
                            if len(path[frame]) > 1
                            else self.function_space.func(x)
                        )

                    points[method_name].set_data([x], [y])

                artists.append(paths[method_name])
                artists.append(points[method_name])

                # Update error plot if applicable
                if (
                    show_error_plot
                    and error_data
                    and ax_error is not None
                    and method_name in error_data
                ):
                    error_values = error_data[method_name]

                    # Skip if we've reached the end of this method's error data
                    if frame >= len(error_values):
                        artists.append(error_lines[method_name])
                        artists.append(error_points[method_name])
                        continue

                    # Update error line
                    iterations = list(range(frame + 1))
                    errors = error_values[: frame + 1]

                    # Ensure all error values are positive for log scale
                    errors = [max(1e-15, e) for e in errors]

                    error_lines[method_name].set_data(iterations, errors)

                    # Update current error point
                    error_points[method_name].set_data([frame], [errors[frame]])

                    artists.append(error_lines[method_name])
                    artists.append(error_points[method_name])

            return artists

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=max_path_length,
            init_func=init,
            interval=interval,
            blit=True,
        )

        # Save animation if path provided
        if save_path:
            anim.save(save_path, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)

        # Adjust layout
        plt.tight_layout()

        return anim

    def create_plotly_animation(
        self,
        method_paths: Dict[str, List[Union[float, Tuple[float, float]]]],
        error_data: Optional[Dict[str, List[float]]] = None,
        show_error_plot: bool = True,
        critical_points: Optional[List[Union[float, Tuple[float, float]]]] = None,
        height: int = None,
        width: int = None,
        duration: int = 800,
        transition_duration: int = 300,
        surface_plot: bool = False,
    ) -> go.Figure:
        """
        Create a plotly animation of numerical methods.

        Args:
            method_paths: Dictionary mapping method names to lists of points
            error_data: Dictionary mapping method names to lists of error values
            show_error_plot: Whether to show error plot alongside function space
            critical_points: Critical points to highlight
            height: Figure height (set to None for browser height)
            width: Figure width (set to None for browser width)
            duration: Duration of each frame in milliseconds
            transition_duration: Duration of transition between frames
            surface_plot: Whether to use 3D surface plot for 2D functions

        Returns:
            go.Figure: Plotly figure with animation
        """
        # Update method colors
        self.method_colors = get_method_colors(
            list(method_paths.keys()),
            palette=self.color_palette,
            existing_colors=self.method_colors,
        )

        # Determine subplot configuration
        if show_error_plot and error_data:
            # Create specs with explicit types to avoid NoneType errors
            if surface_plot and self.function_space.is_2d:
                specs = [[{"type": "scene"}, {"type": "xy"}]]
            else:
                specs = [[{"type": "xy"}, {"type": "xy"}]]

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[
                    f"<b>{self.function_space.title}</b>",
                    "<b>Convergence Progress</b>",
                ],
                column_widths=[0.7, 0.3],
                specs=specs,
                horizontal_spacing=0.08,
            )
        else:
            # Single plot, no need for subplots
            fig = go.Figure()

        # Determine max number of frames
        max_path_length = max(len(path) for path in method_paths.values())

        # Create base plot for function space
        if self.function_space.is_2d:
            # 2D function
            if surface_plot:
                # Surface plot
                fig.add_trace(
                    go.Surface(
                        x=self.function_space.X,
                        y=self.function_space.Y,
                        z=self.function_space.Z,
                        colorscale=self.function_space.colormap,
                        opacity=0.85,
                        name="Function",
                        showscale=True,
                        contours={
                            "z": {
                                "show": True,
                                "usecolormap": True,
                                "highlightcolor": "white",
                                "project": {"z": True},
                            }
                        },
                        lighting={
                            "ambient": 0.6,
                            "diffuse": 0.9,
                            "fresnel": 0.5,
                            "roughness": 0.5,
                            "specular": 1.0,
                        },
                    ),
                    row=1,
                    col=1,
                )
            else:
                # Contour plot
                fig.add_trace(
                    go.Contour(
                        x=np.linspace(
                            self.function_space.x_range[0],
                            self.function_space.x_range[1],
                            self.function_space.X.shape[1],
                        ),
                        y=np.linspace(
                            self.function_space.y_range[0],
                            self.function_space.y_range[1],
                            self.function_space.Y.shape[0],
                        ),
                        z=self.function_space.Z,
                        colorscale=self.function_space.colormap,
                        ncontours=25,
                        name="Function",
                        contours={
                            "showlabels": True,
                            "labelfont": {"size": 10, "color": "white"},
                        },
                        colorbar={"thickness": 20, "len": 0.8, "title": "f(x,y)"},
                    ),
                    row=1,
                    col=1,
                )
        else:
            # 1D function
            fig.add_trace(
                go.Scatter(
                    x=self.function_space.x,
                    y=self.function_space.y,
                    mode="lines",
                    line=dict(color="rgba(50, 50, 50, 0.8)", width=3, shape="spline"),
                    name="Function",
                    fill="tozeroy",
                    fillcolor="rgba(220, 220, 220, 0.3)",
                ),
                row=1,
                col=1,
            )

        # Add critical points if provided
        if critical_points:
            if self.function_space.is_2d:
                # 2D critical points
                cp_x = [p[0] for p in critical_points]
                cp_y = [p[1] for p in critical_points]

                if surface_plot:
                    # 3D critical points
                    cp_z = [
                        self.function_space.func(p[0], p[1]) for p in zip(cp_x, cp_y)
                    ]

                    fig.add_trace(
                        go.Scatter3d(
                            x=cp_x,
                            y=cp_y,
                            z=cp_z,
                            mode="markers",
                            marker=dict(
                                size=8,
                                color="red",
                                symbol="diamond",
                                line=dict(width=2, color="black"),
                            ),
                            name="Critical Points",
                            hoverinfo="name+text",
                            hovertext=["Critical point" for _ in cp_x],
                        ),
                        row=1,
                        col=1,
                    )
                else:
                    # 2D critical points
                    fig.add_trace(
                        go.Scatter(
                            x=cp_x,
                            y=cp_y,
                            mode="markers",
                            marker=dict(
                                size=12,
                                color="red",
                                symbol="star",
                                line=dict(width=2, color="black"),
                            ),
                            name="Critical Points",
                            hoverinfo="name+text",
                            hovertext=["Critical point" for _ in cp_x],
                        ),
                        row=1,
                        col=1,
                    )
            else:
                # 1D critical points
                if isinstance(critical_points[0], (int, float, np.number)):
                    cp_x = critical_points
                    cp_y = [self.function_space.func(x) for x in cp_x]
                else:
                    cp_x = [p[0] for p in critical_points]
                    cp_y = [p[1] for p in critical_points]

                fig.add_trace(
                    go.Scatter(
                        x=cp_x,
                        y=cp_y,
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="red",
                            symbol="star",
                            line=dict(width=2, color="black"),
                        ),
                        name="Critical Points",
                        hoverinfo="name+text",
                        hovertext=[f"Critical point at x={x:.4f}" for x in cp_x],
                    ),
                    row=1,
                    col=1,
                )

        # Add empty traces for each method
        for method_name, path in method_paths.items():
            color = self.method_colors.get(method_name, "gray")

            # Add trace for function space
            if self.function_space.is_2d:
                # 2D function
                if surface_plot:
                    # 3D path
                    fig.add_trace(
                        go.Scatter3d(
                            x=[],
                            y=[],
                            z=[],
                            mode="lines+markers",
                            line=dict(color=color, width=5, dash="solid"),
                            marker=dict(
                                size=8,
                                color=color,
                                symbol="circle",
                                line=dict(width=1, color="white"),
                            ),
                            name=method_name,
                            hoverinfo="name+text",
                            hovertext=[
                                f"{method_name} iteration" for _ in range(len(path))
                            ],
                        ),
                        row=1,
                        col=1,
                    )
                else:
                    # 2D path
                    fig.add_trace(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode="lines+markers",
                            line=dict(color=color, width=3, dash="solid"),
                            marker=dict(
                                size=10,
                                color=color,
                                symbol="circle",
                                line=dict(width=1, color="white"),
                            ),
                            name=method_name,
                            hoverinfo="name+text",
                            hovertext=[
                                f"{method_name} iteration" for _ in range(len(path))
                            ],
                        ),
                        row=1,
                        col=1,
                    )
            else:
                # 1D function
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="lines+markers",
                        line=dict(color=color, width=3, dash="solid"),
                        marker=dict(
                            size=10,
                            color=color,
                            symbol="circle",
                            line=dict(width=1, color="white"),
                        ),
                        name=method_name,
                        hoverinfo="name+text",
                        hovertext=[
                            f"{method_name} iteration" for _ in range(len(path))
                        ],
                    ),
                    row=1,
                    col=1,
                )

            # Add trace for error plot if applicable
            if show_error_plot and error_data and method_name in error_data:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="lines+markers",
                        line=dict(color=color, width=3, dash="solid"),
                        marker=dict(
                            size=8,
                            color=color,
                            symbol="circle",
                            line=dict(width=1, color="white"),
                        ),
                        name=method_name,
                        showlegend=False,
                        hoverinfo="name+text",
                        hovertext=[
                            f"{method_name} error"
                            for _ in range(len(error_data[method_name]))
                        ],
                    ),
                    row=1,
                    col=2,
                )

        # Create frames for animation
        frames = []
        for frame_idx in range(max_path_length):
            frame_data = []

            # Add frame data for each method
            for method_name, path in method_paths.items():
                # Skip if we've reached the end of this method's path
                if frame_idx >= len(path):
                    # Add empty data to maintain trace index consistency
                    if self.function_space.is_2d and surface_plot:
                        frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
                    else:
                        frame_data.append(go.Scatter(x=[], y=[]))

                    # Add empty data for error plot if applicable
                    if show_error_plot and error_data and method_name in error_data:
                        frame_data.append(go.Scatter(x=[], y=[]))

                    continue

                # Process path data based on dimensionality
                if self.function_space.is_2d:
                    # 2D function
                    x_path = [p[0] for p in path[: frame_idx + 1]]
                    y_path = [p[1] for p in path[: frame_idx + 1]]

                    if surface_plot:
                        # 3D path with z values
                        z_path = [
                            self.function_space.func(p[0], p[1])
                            for p in zip(x_path, y_path)
                        ]
                        frame_data.append(go.Scatter3d(x=x_path, y=y_path, z=z_path))
                    else:
                        # 2D path
                        frame_data.append(go.Scatter(x=x_path, y=y_path))
                else:
                    # 1D function
                    if isinstance(path[0], (int, float, np.number)):
                        # Path is a list of x values
                        x_path = path[: frame_idx + 1]
                        y_path = [self.function_space.func(x) for x in x_path]
                    else:
                        # Path is a list of (x, y) tuples
                        x_path = [p[0] for p in path[: frame_idx + 1]]
                        y_path = [
                            p[1] if len(p) > 1 else self.function_space.func(p[0])
                            for p in path[: frame_idx + 1]
                        ]

                    frame_data.append(go.Scatter(x=x_path, y=y_path))

                # Add error data if applicable
                if show_error_plot and error_data and method_name in error_data:
                    error_values = error_data[method_name]

                    # Skip if we've reached the end of this method's error data
                    if frame_idx >= len(error_values):
                        frame_data.append(go.Scatter(x=[], y=[]))
                        continue

                    # Process error data
                    iterations = list(range(frame_idx + 1))
                    errors = error_values[: frame_idx + 1]

                    # Ensure all error values are positive for log scale
                    errors = [max(1e-15, e) for e in errors]

                    frame_data.append(go.Scatter(x=iterations, y=errors))

            # Create frame
            frames.append(go.Frame(data=frame_data, name=f"frame{frame_idx}"))

        # Add frames to figure
        fig.frames = frames

        # Configure layout
        fig.update_layout(
            title={
                "text": f"<b>{self.title}</b>",
                "font": {"size": 24, "color": "#333333"},
                "y": 0.97,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            height=height,  # Set to None for browser dimensions
            width=width,  # Set to None for browser dimensions
            template="plotly_white",
            hovermode="closest",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.2,
                "xanchor": "center",
                "x": 0.5,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "rgba(0, 0, 0, 0.2)",
                "borderwidth": 1,
                "font": {"size": 12},
            },
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
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": transition_duration,
                                        "easing": "cubic-in-out",
                                    },
                                },
                            ],
                        },
                        {
                            "label": "⏸ Pause",
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
                        {
                            "label": "⏮ Reset",
                            "method": "animate",
                            "args": [
                                ["frame0"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
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
                    "pad": {"r": 10, "t": 10, "b": 10},
                    "bgcolor": "rgba(240, 240, 240, 0.8)",
                    "bordercolor": "rgba(0, 0, 0, 0.2)",
                    "borderwidth": 1,
                    "font": {"size": 13},
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame{k}"],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": transition_duration},
                                },
                            ],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k in range(
                            0, max_path_length, max(1, max_path_length // 10)
                        )
                    ],
                    "currentvalue": {
                        "prefix": "Frame: ",
                        "visible": True,
                        "font": {"size": 14},
                        "xanchor": "left",
                    },
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "pad": {"b": 10, "t": 50},
                    "bgcolor": "rgba(240, 240, 240, 0.3)",
                    "bordercolor": "rgba(0, 0, 0, 0.2)",
                    "borderwidth": 1,
                    "tickwidth": 1,
                    "tickcolor": "rgba(0, 0, 0, 0.4)",
                }
            ],
            margin={"l": 60, "r": 30, "t": 80, "b": 100},
            paper_bgcolor="rgba(255, 255, 255, 0.95)",
            plot_bgcolor="rgba(250, 250, 250, 0.95)",
            autosize=True,  # Allow autosize to use full browser dimensions
        )

        # Configure x and y axes for main plot
        fig.update_xaxes(
            title={
                "text": "<b>x</b>",
                "font": {"size": 14, "color": "#333333"},
            },
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor="rgba(0, 0, 0, 0.3)",
            row=1,
            col=1,
        )

        if self.function_space.is_2d:
            fig.update_yaxes(
                title={
                    "text": "<b>y</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(200, 200, 200, 0.3)",
                zeroline=True,
                zerolinewidth=1.5,
                zerolinecolor="rgba(0, 0, 0, 0.3)",
                row=1,
                col=1,
            )
        else:
            fig.update_yaxes(
                title={
                    "text": "<b>f(x)</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(200, 200, 200, 0.3)",
                zeroline=True,
                zerolinewidth=1.5,
                zerolinecolor="rgba(0, 0, 0, 0.3)",
                row=1,
                col=1,
            )

        # Configure error plot axes if applicable
        if show_error_plot and error_data:
            fig.update_xaxes(
                title={
                    "text": "<b>Iteration</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(200, 200, 200, 0.3)",
                row=1,
                col=2,
            )
            fig.update_yaxes(
                title={
                    "text": "<b>Error (log scale)</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                type="log",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(200, 200, 200, 0.3)",
                row=1,
                col=2,
            )

        # Set 3D scene options for surface plot
        if surface_plot and self.function_space.is_2d:
            fig.update_scenes(
                xaxis_title={
                    "text": "<b>x</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                yaxis_title={
                    "text": "<b>y</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                zaxis_title={
                    "text": "<b>f(x,y)</b>",
                    "font": {"size": 14, "color": "#333333"},
                },
                aspectratio={"x": 1, "y": 1, "z": 0.8},
                camera={
                    "up": {"x": 0, "y": 0, "z": 1},
                    "center": {"x": 0, "y": 0, "z": 0},
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.2},
                },
                dragmode="turntable",
            )

        return fig
