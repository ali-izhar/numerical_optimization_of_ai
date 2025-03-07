# plot/components/function_space.py

"""This module provides components for visualizing function spaces in 1D and 2D."""

from typing import Callable, List, Optional, Tuple, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plot.utils.dimension_utils import (
    detect_function_dimensions,
    is_2d_function,
    prepare_grid_data,
    prepare_plotly_grid_data,
    create_plotly_surface,
    create_plotly_contour,
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
        path_color: str = "#FF4136",  # Vibrant red
        point_colors: Optional[List[str]] = None,
        critical_point_color: str = "#FFDC00",  # Vibrant yellow
        plot_type: str = "contour",
        colorscale: Optional[str] = None,  # Added parameter for colorscale
    ) -> go.Figure:
        """
        Create a plotly figure of the function space with eye-catching styling.

        Args:
            include_points: Additional points to highlight
            critical_points: Critical points to highlight
            path: Path to visualize (e.g., algorithm trajectory)
            path_color: Color for the path
            point_colors: Colors for include_points
            critical_point_color: Color for critical points
            plot_type: For 2D functions: 'contour', 'surface', or 'heatmap'
            colorscale: Optional colorscale to use instead of the default

        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()

        # Use provided colorscale if available, otherwise use the one from initialization
        actual_colorscale = colorscale if colorscale is not None else self.colormap

        # Calculate Plotly grid data if needed for 2D visualization
        if self.is_2d and not hasattr(self, "plotly_grid_data"):
            self.plotly_grid_data = {
                "x": self.X[0, :],  # x coordinates (first row of X)
                "y": self.Y[:, 0],  # y coordinates (first column of Y)
                "z": self.Z,  # z values (function values)
                "colorscale": actual_colorscale,
            }

        # Create figure based on dimensionality
        if self.is_2d:
            # 2D function visualization
            if plot_type == "surface":
                # 3D surface plot with enhanced styling
                fig.add_trace(
                    go.Surface(
                        x=self.X,
                        y=self.Y,
                        z=self.Z,
                        colorscale=actual_colorscale,
                        colorbar=dict(
                            title=dict(
                                text=self.zlabel, font=dict(size=14, family="Arial")
                            ),
                            thickness=20,
                            len=0.8,
                        ),
                        lighting=dict(
                            ambient=0.6,
                            diffuse=0.8,
                            fresnel=0.2,
                            roughness=0.4,
                            specular=1.0,
                        ),
                        contours={
                            "x": {
                                "show": True,
                                "width": 2,
                                "color": "rgba(255,255,255,0.3)",
                            },
                            "y": {
                                "show": True,
                                "width": 2,
                                "color": "rgba(255,255,255,0.3)",
                            },
                            "z": {
                                "show": True,
                                "width": 2,
                                "color": "rgba(255,255,255,0.3)",
                            },
                        },
                        opacity=0.85,
                        hoverinfo="all",
                        hoverlabel={"font": {"family": "Arial", "size": 14}},
                    )
                )

                # Set up 3D axes with enhanced styling
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            title=dict(
                                text=self.xlabel, font=dict(size=14, family="Arial")
                            ),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(220, 220, 220, 0.3)",
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor="rgba(0, 0, 0, 0.2)",
                        ),
                        yaxis=dict(
                            title=dict(
                                text=self.ylabel, font=dict(size=14, family="Arial")
                            ),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(220, 220, 220, 0.3)",
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor="rgba(0, 0, 0, 0.2)",
                        ),
                        zaxis=dict(
                            title=dict(
                                text=self.zlabel, font=dict(size=14, family="Arial")
                            ),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(220, 220, 220, 0.3)",
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor="rgba(0, 0, 0, 0.2)",
                        ),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                        aspectratio=dict(x=1, y=1, z=0.8),
                    ),
                )
            else:
                # Contour plot with enhanced styling
                fig.add_trace(
                    go.Contour(
                        x=np.linspace(
                            self.x_range[0], self.x_range[1], self.X.shape[1]
                        ),
                        y=np.linspace(
                            self.y_range[0], self.y_range[1], self.Y.shape[0]
                        ),
                        z=self.Z,
                        colorscale=actual_colorscale,
                        colorbar=dict(
                            title=dict(
                                text=self.zlabel, font=dict(size=14, family="Arial")
                            ),
                            thickness=20,
                            len=0.8,
                        ),
                        ncontours=25,
                        contours=dict(
                            coloring="heatmap",
                            showlabels=True,
                            labelfont=dict(family="Arial", size=12, color="white"),
                        ),
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                        hoverinfo="all",
                        hoverlabel=dict(font=dict(family="Arial", size=14)),
                    )
                )

            # Plot path if provided (for 2D)
            if path is not None and len(path) > 0:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]

                if plot_type == "surface":
                    # 3D path with function values
                    try:
                        # Try to compute z values based on function
                        path_z = []
                        for px, py in zip(path_x, path_y):
                            try:
                                pz = self.func([px, py])
                            except:
                                try:
                                    pz = self.func(px, py)
                                except:
                                    # If function call fails, interpolate from our precomputed grid
                                    idx_x = int(
                                        (px - self.x_range[0])
                                        / (self.x_range[1] - self.x_range[0])
                                        * (self.X.shape[1] - 1)
                                    )
                                    idx_y = int(
                                        (py - self.y_range[0])
                                        / (self.y_range[1] - self.y_range[0])
                                        * (self.Y.shape[0] - 1)
                                    )
                                    if (
                                        0 <= idx_x < self.Z.shape[1]
                                        and 0 <= idx_y < self.Z.shape[0]
                                    ):
                                        pz = self.Z[idx_y, idx_x]
                                    else:
                                        pz = 0
                            path_z.append(pz)

                        # Enhanced 3D path
                        fig.add_trace(
                            go.Scatter3d(
                                x=path_x,
                                y=path_y,
                                z=path_z,
                                mode="lines+markers",
                                line=dict(color=path_color, width=6),
                                marker=dict(
                                    size=6,
                                    color=path_color,
                                    line=dict(width=1, color="white"),
                                ),
                                name="Optimization Path",
                                hoverinfo="text",
                                hovertext=[
                                    f"Point {i+1}: ({x:.4f}, {y:.4f}, {z:.4f})"
                                    for i, (x, y, z) in enumerate(
                                        zip(path_x, path_y, path_z)
                                    )
                                ],
                            )
                        )

                        # Add a special marker for the final point
                        fig.add_trace(
                            go.Scatter3d(
                                x=[path_x[-1]],
                                y=[path_y[-1]],
                                z=[path_z[-1]],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=path_color,
                                    symbol="diamond",
                                    line=dict(width=2, color="white"),
                                ),
                                name="Final Point",
                            )
                        )
                    except Exception as e:
                        print(f"Warning: Could not visualize 3D path: {e}")
                else:
                    # Enhanced 2D path
                    fig.add_trace(
                        go.Scatter(
                            x=path_x,
                            y=path_y,
                            mode="lines+markers",
                            line=dict(
                                color=path_color,
                                width=3,
                                dash="solid",
                            ),
                            marker=dict(
                                size=8,
                                color=path_color,
                                line=dict(width=1, color="white"),
                                symbol="circle",
                            ),
                            name="Optimization Path",
                            hoverinfo="text",
                            hovertext=[
                                f"Point {i+1}: ({x:.4f}, {y:.4f})"
                                for i, (x, y) in enumerate(zip(path_x, path_y))
                            ],
                        )
                    )

                    # Add arrows to show direction
                    if len(path_x) > 1:
                        # Add arrows at regular intervals
                        arrow_indices = list(
                            range(0, len(path_x) - 1, max(1, len(path_x) // 8))
                        )
                        if len(path_x) - 2 not in arrow_indices:
                            arrow_indices.append(len(path_x) - 2)

                        for j in arrow_indices:
                            if j < len(path_x) - 1:
                                # Calculate arrow direction
                                dx = path_x[j + 1] - path_x[j]
                                dy = path_y[j + 1] - path_y[j]
                                # Normalize
                                magnitude = (dx**2 + dy**2) ** 0.5
                                if magnitude > 0:
                                    dx = dx / magnitude
                                    dy = dy / magnitude

                                fig.add_annotation(
                                    x=path_x[j + 1],
                                    y=path_y[j + 1],
                                    ax=path_x[j + 1] - dx * 10,
                                    ay=path_y[j + 1] - dy * 10,
                                    xref="x",
                                    yref="y",
                                    axref="x",
                                    ayref="y",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1.5,
                                    arrowwidth=2,
                                    arrowcolor=path_color,
                                    standoff=5,
                                )

                    # Add a special marker for the final point
                    fig.add_trace(
                        go.Scatter(
                            x=[path_x[-1]],
                            y=[path_y[-1]],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=path_color,
                                symbol="star",
                                line=dict(width=2, color="white"),
                            ),
                            name="Final Point",
                        )
                    )

            # Plot critical points for 2D with enhanced styling
            if critical_points is not None:
                cp_x = [p[0] for p in critical_points]
                cp_y = [p[1] for p in critical_points]

                if plot_type == "surface":
                    # 3D critical points
                    try:
                        # Try to compute z values
                        cp_z = []
                        for px, py in zip(cp_x, cp_y):
                            try:
                                pz = self.func([px, py])
                            except:
                                try:
                                    pz = self.func(px, py)
                                except:
                                    # If function call fails, interpolate from our precomputed grid
                                    idx_x = int(
                                        (px - self.x_range[0])
                                        / (self.x_range[1] - self.x_range[0])
                                        * (self.X.shape[1] - 1)
                                    )
                                    idx_y = int(
                                        (py - self.y_range[0])
                                        / (self.y_range[1] - self.y_range[0])
                                        * (self.Y.shape[0] - 1)
                                    )
                                    if (
                                        0 <= idx_x < self.Z.shape[1]
                                        and 0 <= idx_y < self.Z.shape[0]
                                    ):
                                        pz = self.Z[idx_y, idx_x]
                                    else:
                                        pz = 0
                            cp_z.append(pz)

                        fig.add_trace(
                            go.Scatter3d(
                                x=cp_x,
                                y=cp_y,
                                z=cp_z,
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=critical_point_color,
                                    symbol="diamond",
                                    line=dict(width=2, color="black"),
                                ),
                                name="Critical Points",
                                hoverinfo="text",
                                hovertext=[
                                    f"Critical Point: ({x:.4f}, {y:.4f}, {z:.4f})"
                                    for x, y, z in zip(cp_x, cp_y, cp_z)
                                ],
                            )
                        )
                    except Exception as e:
                        print(f"Warning: Could not visualize 3D critical points: {e}")
                else:
                    # Enhanced 2D critical points
                    fig.add_trace(
                        go.Scatter(
                            x=cp_x,
                            y=cp_y,
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=critical_point_color,
                                symbol="x",
                                line=dict(width=2, color="black"),
                            ),
                            name="Critical Points",
                            hoverinfo="text",
                            hovertext=[
                                f"Critical Point: ({x:.4f}, {y:.4f})"
                                for x, y in zip(cp_x, cp_y)
                            ],
                        )
                    )
        else:
            # Enhanced 1D function visualization
            fig.add_trace(
                go.Scatter(
                    x=self.x,
                    y=self.y,
                    mode="lines",
                    line=dict(
                        width=3,
                        color="#3D9970",  # Nice green
                        shape="spline",  # Smooth curve
                    ),
                    name="f(x)",
                    fill="tozeroy",
                    fillcolor="rgba(61, 153, 112, 0.1)",
                )
            )

            # Plot path for 1D with enhanced styling
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
                        line=dict(
                            color=path_color,
                            width=3,
                            dash="solid",
                        ),
                        marker=dict(
                            size=8,
                            color=path_color,
                            line=dict(width=1, color="white"),
                            symbol="circle",
                        ),
                        name="Algorithm Path",
                        hoverinfo="text",
                        hovertext=[
                            f"Iteration {i+1}: x={x:.4f}, f(x)={y:.4f}"
                            for i, (x, y) in enumerate(zip(path_x, path_y))
                        ],
                    )
                )

                # Add special markers for start and end points
                fig.add_trace(
                    go.Scatter(
                        x=[path_x[0]],
                        y=[path_y[0]],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=path_color,
                            symbol="circle-open",
                            line=dict(width=3, color=path_color),
                        ),
                        name="Start Point",
                        hoverinfo="text",
                        hovertext=f"Start: x={path_x[0]:.4f}, f(x)={path_y[0]:.4f}",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[path_x[-1]],
                        y=[path_y[-1]],
                        mode="markers",
                        marker=dict(
                            size=14,
                            color=path_color,
                            symbol="star",
                            line=dict(width=2, color="white"),
                        ),
                        name="Final Point",
                        hoverinfo="text",
                        hovertext=f"Final: x={path_x[-1]:.4f}, f(x)={path_y[-1]:.4f}",
                    )
                )

            # Plot critical points for 1D with enhanced styling
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
                        marker=dict(
                            size=12,
                            color=critical_point_color,
                            symbol="diamond",
                            line=dict(width=2, color="black"),
                        ),
                        name="Critical Points",
                        hoverinfo="text",
                        hovertext=[
                            f"Critical Point: x={x:.4f}, f(x)={y:.4f}"
                            for x, y in zip(cp_x, cp_y)
                        ],
                    )
                )

        # Add additional points with enhanced styling
        if include_points is not None:
            # Generate colors if not provided
            if point_colors is None:
                point_colors = generate_colors(len(include_points), "Plotly3")

            for i, point in enumerate(include_points):
                color = (
                    point_colors[i] if i < len(point_colors) else "#0074D9"
                )  # Default to blue

                if self.is_2d:
                    if plot_type == "surface":
                        # Enhanced 3D point
                        try:
                            # Try to compute z value
                            try:
                                point_z = self.func([point[0], point[1]])
                            except:
                                try:
                                    point_z = self.func(point[0], point[1])
                                except:
                                    # Fallback to interpolation
                                    idx_x = int(
                                        (point[0] - self.x_range[0])
                                        / (self.x_range[1] - self.x_range[0])
                                        * (self.X.shape[1] - 1)
                                    )
                                    idx_y = int(
                                        (point[1] - self.y_range[0])
                                        / (self.y_range[1] - self.y_range[0])
                                        * (self.Y.shape[0] - 1)
                                    )
                                    if (
                                        0 <= idx_x < self.Z.shape[1]
                                        and 0 <= idx_y < self.Z.shape[0]
                                    ):
                                        point_z = self.Z[idx_y, idx_x]
                                    else:
                                        point_z = 0

                            fig.add_trace(
                                go.Scatter3d(
                                    x=[point[0]],
                                    y=[point[1]],
                                    z=[point_z],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        color=color,
                                        line=dict(width=1, color="white"),
                                        symbol="circle",
                                    ),
                                    name=f"Point {i+1}",
                                    hoverinfo="text",
                                    hovertext=f"Point {i+1}: ({point[0]:.4f}, {point[1]:.4f}, {point_z:.4f})",
                                )
                            )
                        except Exception as e:
                            print(f"Warning: Could not visualize 3D point: {e}")
                    else:
                        # Enhanced 2D point
                        fig.add_trace(
                            go.Scatter(
                                x=[point[0]],
                                y=[point[1]],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=color,
                                    line=dict(width=1, color="white"),
                                    symbol="circle",
                                ),
                                name=f"Point {i+1}",
                                hoverinfo="text",
                                hovertext=f"Point {i+1}: ({point[0]:.4f}, {point[1]:.4f})",
                            )
                        )
                else:
                    # Enhanced 1D point
                    if isinstance(point, (int, float, np.number)):
                        x = point
                        y = self.func(x)
                    else:
                        x, y = point

                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=color,
                                line=dict(width=1, color="white"),
                                symbol="circle",
                            ),
                            name=f"Point {i+1}",
                            hoverinfo="text",
                            hovertext=f"Point {i+1}: x={x:.4f}, f(x)={y:.4f}",
                        )
                    )

        # Enhanced layout with modern styling and no hardcoded dimensions
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=self.title,
                font=dict(size=24, family="Arial, sans-serif"),
                y=0.95,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            xaxis=dict(
                title=dict(text=self.xlabel, font=dict(size=18, family="Arial")),
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(220, 220, 220, 0.5)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(size=14, family="Arial"),
            ),
            yaxis=dict(
                title=dict(text=self.ylabel, font=dict(size=18, family="Arial")),
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(220, 220, 220, 0.5)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(size=14, family="Arial"),
            ),
            font=dict(family="Arial, sans-serif", size=14),
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode="closest",
            legend=dict(
                font=dict(size=14, family="Arial"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                orientation="h" if not self.is_2d or plot_type != "surface" else "v",
                yanchor="top" if self.is_2d and plot_type == "surface" else "bottom",
                y=0.99 if self.is_2d and plot_type == "surface" else -0.15,
                xanchor="left" if self.is_2d and plot_type == "surface" else "center",
                x=0.01 if self.is_2d and plot_type == "surface" else 0.5,
            ),
            showlegend=True,
            # No height or width parameters - let it take browser dimensions by default
        )

        return fig
