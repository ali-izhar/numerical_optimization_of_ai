# plot/components/function_space.py

"""This module provides components for visualizing function spaces in 1D and 2D."""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plot.utils.dimension_utils import (
    detect_function_dimensions,
    is_2d_function,
    prepare_grid_data,
)
from plot.utils.color_utils import generate_colors


class FunctionSpace:
    """
    Component for visualizing function spaces.

    This class provides methods for creating 1D and 2D function visualizations
    with support for highlighting critical points, paths, and other features.
    """

    def __init__(
        self,
        func: Callable,
        x_range: Tuple[float, float],
        y_range: Optional[Tuple[float, float]] = None,
        is_2d: Optional[bool] = None,
        title: str = "Function Visualization",
        xlabel: str = "x",
        ylabel: Optional[str] = None,
        zlabel: str = "f(x,y)",
        colormap: str = "viridis",
    ):
        """
        Initialize the function space visualizer.

        Args:
            func: The function to visualize
            x_range: Range of x values (min, max)
            y_range: Range of y values for 2D functions (min, max)
            is_2d: Explicitly specify if function is 2D (auto-detected if None)
            title: Title for the visualization
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis (auto-generated if None)
            zlabel: Label for the z-axis (for 2D functions)
            colormap: Colormap to use for 2D visualizations
        """
        self.func = func
        self.x_range = x_range
        self.y_range = y_range or x_range

        # Auto-detect dimensionality if not specified
        self.is_2d = is_2d if is_2d is not None else is_2d_function(func)

        # Set plot labels
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel or ("y" if self.is_2d else "f(x)")
        self.zlabel = zlabel
        self.colormap = colormap

        # Calculate function values
        self._calculate_function_values()

    def _calculate_function_values(self, num_points: int = 100):
        """
        Calculate function values for plotting.

        Args:
            num_points: Number of points to evaluate function at
        """
        if self.is_2d:
            # 2D function
            self.X, self.Y, self.Z = prepare_grid_data(
                self.func, self.x_range, self.y_range, num_points
            )
        else:
            # 1D function
            self.x = np.linspace(self.x_range[0], self.x_range[1], num_points)
            try:
                # Try calling with individual values
                self.y = np.array([self.func(xi) for xi in self.x])
            except:
                # Try calling with array
                self.y = self.func(self.x)

    def create_matplotlib_figure(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_grid: bool = True,
        include_points: Optional[List[Tuple[float, float]]] = None,
        critical_points: Optional[List[Union[float, Tuple[float, float]]]] = None,
        path: Optional[List[Union[float, Tuple[float, float]]]] = None,
        path_color: str = "red",
        point_colors: Optional[List[str]] = None,
        critical_point_color: str = "red",
        plot_type: str = "line",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a matplotlib figure of the function space.

        Args:
            ax: Existing axes to plot on (creates new figure if None)
            figsize: Figure size for new figure
            show_grid: Whether to show grid lines
            include_points: Additional points to highlight
            critical_points: Critical points to highlight
            path: Path to visualize (e.g., algorithm trajectory)
            path_color: Color for the path
            point_colors: Colors for include_points
            critical_point_color: Color for critical points
            plot_type: For 2D functions: 'contour', 'surface', or 'heatmap'

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot function
        if self.is_2d:
            # 2D function visualization
            if plot_type == "contour":
                # Contour plot
                contour = ax.contourf(
                    self.X, self.Y, self.Z, levels=20, cmap=self.colormap, alpha=0.7
                )
                fig.colorbar(contour, ax=ax, label=self.zlabel)

                # Add contour lines
                ax.contour(
                    self.X,
                    self.Y,
                    self.Z,
                    levels=20,
                    colors="k",
                    alpha=0.3,
                    linestyles="solid",
                    linewidths=0.5,
                )
            elif plot_type == "heatmap":
                # Heatmap
                heatmap = ax.imshow(
                    self.Z,
                    extent=[
                        self.x_range[0],
                        self.x_range[1],
                        self.y_range[0],
                        self.y_range[1],
                    ],
                    cmap=self.colormap,
                    origin="lower",
                    aspect="auto",
                )
                fig.colorbar(heatmap, ax=ax, label=self.zlabel)
            else:
                # Default to contour
                contour = ax.contourf(
                    self.X, self.Y, self.Z, levels=20, cmap=self.colormap
                )
                fig.colorbar(contour, ax=ax, label=self.zlabel)

            # Plot path if provided (for 2D)
            if path is not None and len(path) > 0:
                # For 2D paths, extract x and y coordinates
                try:
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    ax.plot(
                        path_x,
                        path_y,
                        "-o",
                        color=path_color,
                        markersize=4,
                        linewidth=1.5,
                    )

                    # Mark start and end with different markers (if we have enough points)
                    if len(path_x) > 0:
                        ax.plot(
                            path_x[0], path_y[0], "o", color=path_color, markersize=8
                        )
                    if len(path_x) > 1:
                        ax.plot(
                            path_x[-1], path_y[-1], "*", color=path_color, markersize=10
                        )
                except (IndexError, TypeError) as e:
                    # If we can't properly extract 2D coordinates, skip path visualization
                    print(f"Warning: Could not visualize 2D path: {e}")

            # Plot critical points for 2D
            if critical_points is not None:
                for point in critical_points:
                    ax.plot(
                        point[0],
                        point[1],
                        "X",
                        color=critical_point_color,
                        markersize=10,
                    )

        else:
            # 1D function visualization
            ax.plot(self.x, self.y, "-", linewidth=2, label=f"f(x)")

            # Plot path for 1D
            if path is not None and len(path) > 0:
                # Check if path is a list of scalars or a list of tuples
                is_scalar_path = isinstance(path[0], (int, float, np.number))

                if is_scalar_path:
                    # Path is a list of x values
                    path_x = np.array(path)
                    path_y = np.array([self.func(x) for x in path_x])
                else:
                    # Path is a list of (x, y) tuples
                    path_x = np.array([p[0] for p in path])
                    path_y = np.array(
                        [p[1] if len(p) > 1 else self.func(p[0]) for p in path]
                    )

                ax.plot(
                    path_x, path_y, "-o", color=path_color, markersize=6, linewidth=1.5
                )

                # Mark start and end with different markers
                ax.plot(path_x[0], path_y[0], "o", color=path_color, markersize=8)
                ax.plot(path_x[-1], path_y[-1], "*", color=path_color, markersize=10)

            # Plot critical points for 1D
            if critical_points is not None:
                if isinstance(critical_points[0], (int, float, np.number)):
                    cp_x = critical_points
                    cp_y = [self.func(x) for x in cp_x]
                else:
                    cp_x = [p[0] for p in critical_points]
                    cp_y = [p[1] for p in critical_points]

                ax.plot(cp_x, cp_y, "X", color=critical_point_color, markersize=10)

        # Add additional points if provided
        if include_points is not None:
            # Generate colors if not provided
            if point_colors is None:
                point_colors = generate_colors(len(include_points), "tab10")

            for i, point in enumerate(include_points):
                color = point_colors[i] if i < len(point_colors) else "black"

                if self.is_2d:
                    # 2D point
                    ax.plot(point[0], point[1], "o", color=color, markersize=8)
                else:
                    # 1D point
                    if isinstance(point, (int, float, np.number)):
                        x = point
                        y = self.func(x)
                    else:
                        x, y = point

                    ax.plot(x, y, "o", color=color, markersize=8)

        # Set labels and grid
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        if show_grid:
            ax.grid(alpha=0.3)

        return fig, ax

    def create_plotly_figure(
        self,
        include_points: Optional[List[Tuple[float, float]]] = None,
        critical_points: Optional[List[Union[float, Tuple[float, float]]]] = None,
        path: Optional[List[Union[float, Tuple[float, float]]]] = None,
        path_color: str = "red",
        point_colors: Optional[List[str]] = None,
        critical_point_color: str = "red",
        plot_type: str = "line",
        height: int = 600,
        width: int = 800,
    ) -> go.Figure:
        """
        Create a plotly figure of the function space.

        Args:
            include_points: Additional points to highlight
            critical_points: Critical points to highlight
            path: Path to visualize (e.g., algorithm trajectory)
            path_color: Color for the path
            point_colors: Colors for include_points
            critical_point_color: Color for critical points
            plot_type: For 2D functions: 'contour', 'surface', or 'heatmap'
            height: Figure height
            width: Figure width

        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()

        # Create figure based on dimensionality
        if self.is_2d:
            # 2D function visualization
            if plot_type == "surface":
                # 3D surface plot
                fig.add_trace(
                    go.Surface(
                        x=self.X,
                        y=self.Y,
                        z=self.Z,
                        colorscale=self.colormap,
                        colorbar=dict(title=self.zlabel),
                    )
                )

                # Set up 3D axes
                fig.update_layout(
                    scene=dict(
                        xaxis_title=self.xlabel,
                        yaxis_title=self.ylabel,
                        zaxis_title=self.zlabel,
                    ),
                    title=self.title,
                )
            else:
                # Contour plot (default)
                fig.add_trace(
                    go.Contour(
                        x=np.linspace(
                            self.x_range[0], self.x_range[1], self.X.shape[1]
                        ),
                        y=np.linspace(
                            self.y_range[0], self.y_range[1], self.Y.shape[0]
                        ),
                        z=self.Z,
                        colorscale=self.colormap,
                        colorbar=dict(title=self.zlabel),
                        ncontours=20,
                    )
                )

                # Set up 2D axes
                fig.update_layout(
                    xaxis_title=self.xlabel, yaxis_title=self.ylabel, title=self.title
                )

            # Plot path if provided (for 2D)
            if path is not None and len(path) > 0:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]

                if plot_type == "surface":
                    # 3D path with function values
                    path_z = [self.func([p[0], p[1]]) for p in zip(path_x, path_y)]

                    fig.add_trace(
                        go.Scatter3d(
                            x=path_x,
                            y=path_y,
                            z=path_z,
                            mode="lines+markers",
                            line=dict(color=path_color, width=4),
                            marker=dict(size=5, color=path_color),
                            name="Optimization Path",
                        )
                    )
                else:
                    # 2D path
                    fig.add_trace(
                        go.Scatter(
                            x=path_x,
                            y=path_y,
                            mode="lines+markers",
                            line=dict(color=path_color, width=2),
                            marker=dict(size=6, color=path_color),
                            name="Optimization Path",
                        )
                    )

            # Plot critical points for 2D
            if critical_points is not None:
                cp_x = [p[0] for p in critical_points]
                cp_y = [p[1] for p in critical_points]

                if plot_type == "surface":
                    # 3D critical points
                    cp_z = [self.func([p[0], p[1]]) for p in zip(cp_x, cp_y)]

                    fig.add_trace(
                        go.Scatter3d(
                            x=cp_x,
                            y=cp_y,
                            z=cp_z,
                            mode="markers",
                            marker=dict(size=8, color=critical_point_color, symbol="x"),
                            name="Critical Points",
                        )
                    )
                else:
                    # 2D critical points
                    fig.add_trace(
                        go.Scatter(
                            x=cp_x,
                            y=cp_y,
                            mode="markers",
                            marker=dict(
                                size=10, color=critical_point_color, symbol="x"
                            ),
                            name="Critical Points",
                        )
                    )
        else:
            # 1D function visualization
            fig.add_trace(
                go.Scatter(
                    x=self.x, y=self.y, mode="lines", line=dict(width=2), name="f(x)"
                )
            )

            # Plot path for 1D
            if path is not None and len(path) > 0:
                # Check if path is a list of scalars or a list of tuples
                is_scalar_path = isinstance(path[0], (int, float, np.number))

                if is_scalar_path:
                    # Path is a list of x values
                    path_x = path
                    path_y = [self.func(x) for x in path_x]
                else:
                    # Path is a list of (x, y) tuples
                    path_x = [p[0] for p in path]
                    path_y = [p[1] if len(p) > 1 else self.func(p[0]) for p in path]

                fig.add_trace(
                    go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode="lines+markers",
                        line=dict(color=path_color, width=2),
                        marker=dict(size=8, color=path_color),
                        name="Algorithm Path",
                    )
                )

            # Plot critical points for 1D
            if critical_points is not None:
                if isinstance(critical_points[0], (int, float, np.number)):
                    cp_x = critical_points
                    cp_y = [self.func(x) for x in cp_x]
                else:
                    cp_x = [p[0] for p in critical_points]
                    cp_y = [p[1] for p in critical_points]

                fig.add_trace(
                    go.Scatter(
                        x=cp_x,
                        y=cp_y,
                        mode="markers",
                        marker=dict(size=10, color=critical_point_color, symbol="x"),
                        name="Critical Points",
                    )
                )

        # Add additional points if provided
        if include_points is not None:
            # Generate colors if not provided
            if point_colors is None:
                point_colors = generate_colors(len(include_points), "tab10")

            for i, point in enumerate(include_points):
                color = point_colors[i] if i < len(point_colors) else "black"

                if self.is_2d:
                    if plot_type == "surface":
                        # 3D point
                        point_z = self.func([point[0], point[1]])

                        fig.add_trace(
                            go.Scatter3d(
                                x=[point[0]],
                                y=[point[1]],
                                z=[point_z],
                                mode="markers",
                                marker=dict(size=6, color=color),
                                name=f"Point {i+1}",
                            )
                        )
                    else:
                        # 2D point
                        fig.add_trace(
                            go.Scatter(
                                x=[point[0]],
                                y=[point[1]],
                                mode="markers",
                                marker=dict(size=10, color=color),
                                name=f"Point {i+1}",
                            )
                        )
                else:
                    # 1D point
                    if isinstance(point, (int, float, np.number)):
                        x = point
                        y = self.func(x)
                    else:
                        x, y = point

                    # Check if point is within our domain
                    if (
                        self.x_range[0] <= x <= self.x_range[1]
                        and self.y_range[0] <= y <= self.y_range[1]
                    ):
                        # Calculate z-value based on function
                        point_z = self.func([x, y])

                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers",
                            marker=dict(size=10, color=color),
                            name=f"Point {i+1}",
                        )
                    )

        # Set figure dimensions
        fig.update_layout(height=height, width=width, template="plotly_white")

        return fig
