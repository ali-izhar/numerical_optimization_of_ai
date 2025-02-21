# plot/optimizer_viz.py

"""
Visualization utilities for comparing optimization methods.

This module provides interactive and comparative visualizations for various
optimization algorithms. It supports both 1D and 2D optimization problems,
with features for visualizing:
- Function surfaces and contours
- Optimization paths
- Convergence behavior
- Error trends
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable

from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig


# Configuration for visualization options like figure size, animation speed, styles, etc.
@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    figsize: Tuple[int, int] = (15, 8)  # Default figure size
    animation_interval: int = 1  # Reduced from 50 to 1ms
    show_convergence: bool = True  # Default to show convergence plot
    show_error: bool = True  # Default to show error plot
    show_contour: bool = True  # Default to show contour for 2D
    style: str = "white"  # Options: darkgrid, whitegrid, dark, white, ticks
    context: str = "talk"  # Options: paper, notebook, talk, poster
    palette: str = "viridis"  # Default colormap
    point_size: int = 100  # Default point size
    dpi: int = 100  # Default DPI for figure resolution
    show_legend: bool = True  # Default to show legend
    grid_alpha: float = 0.3  # Default grid transparency
    title: str = "Optimization Methods Comparison"
    background_color: str = "#FFFFFF"  # White background
    verbose: bool = False  # Enable verbose output


class OptimizationVisualizer:
    """Visualizer for comparing optimization methods.

    Handles both 1D and 2D optimization problems with appropriate
    visualizations for each case.
    """

    def __init__(
        self,
        problem: NumericalMethodConfig,
        methods: List[BaseNumericalMethod],
        config: Optional[VisualizationConfig] = None,
    ):
        """Initialize the visualizer with optimization configuration.

        Args:
            problem: Configuration containing the objective function
            methods: List of optimization methods to compare
            config: Optional visualization configuration

        Note:
            Expects problem.method_type to be "optimize"
        """
        if problem.method_type != "optimize":
            raise ValueError("OptimizationVisualizer requires method_type='optimize'")

        self.problem = problem
        self.methods = methods
        self.config = config or VisualizationConfig()

        # Enhanced styling setup
        plt.style.use("seaborn-v0_8-white")
        sns.set_context("notebook", font_scale=1.2)

        # Custom style configurations
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.2,
                "axes.linewidth": 1.5,
                "axes.labelsize": 12,
                "lines.linewidth": 2.5,
                "lines.markersize": 8,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "legend.framealpha": 0.9,
                "legend.edgecolor": "0.8",
                "figure.titlesize": 16,
                "figure.titleweight": "bold",
                "figure.constrained_layout.use": True,  # Use constrained_layout instead
            }
        )

        # Enable interactive mode for live updating
        plt.ion()

        # Create main figure with enhanced size and DPI
        self.fig = plt.figure(
            figsize=self.config.figsize,
            dpi=120,  # Higher DPI for sharper rendering
            constrained_layout=True,  # Use constrained_layout here
        )

        self.is_2d = self._check_if_2d()
        self.setup_plots()
        self._initialize_visualization()

    def _initialize_visualization(self):
        """Initialize visualization state with enhanced styling."""
        # Get colors from a more vibrant palette
        n_colors = len(self.methods)
        self.colors = sns.color_palette("husl", n_colors)  # More vibrant colors

        self.method_states = {}
        self.error_lines = {}

        for i, method in enumerate(self.methods):
            method_id = id(method)
            color = self.colors[i]
            name = method.__class__.__name__

            # Enhanced line styling
            line_style = dict(
                linewidth=2.5,
                markersize=8,
                markeredgewidth=2,
                markeredgecolor="white",
                alpha=0.9,
                zorder=5,
            )

            if self.is_2d:
                main_line = self.ax_main.plot(
                    [], [], [], "o-", color=color, label=name, **line_style
                )[0]

                contour_line = (
                    self.ax_contour.plot([], [], "o-", color=color, **line_style)[0]
                    if self.config.show_contour
                    else None
                )
            else:
                main_line = self.ax_main.plot(
                    [], [], "o-", color=color, label=name, **line_style
                )[0]
                contour_line = None

            # Enhanced error line styling
            error_line = self.ax_error.plot(
                [], [], "-", color=color, label=name, linewidth=2.5, alpha=0.9
            )[0]

            self.method_states[method_id] = {
                "method": method,
                "line": main_line,
                "contour_line": contour_line,
                "color": color,
                "points": [],
                "errors": [],
            }
            self.error_lines[method_id] = error_line

        # Enhanced legend styling - only show in error plot
        if self.config.show_legend:
            legend_style = dict(
                framealpha=0.9, edgecolor="0.8", fancybox=True, shadow=True
            )
            # Remove legend from main plot
            # self.ax_main.legend(loc="upper right", **legend_style)
            self.ax_error.legend(loc="upper right", **legend_style)

        # Enhanced error plot styling
        self.ax_error.set_xlabel("Iteration")
        self.ax_error.set_ylabel("Error")
        self.ax_error.grid(True, alpha=0.2)
        self.ax_error.set_yscale("log")
        self.ax_error.set_xlim(-1, self.problem.max_iter)
        self.ax_error.set_ylim(1e-8, 1e2)

        # Plot function
        self.plot_function()

    def _get_colors(self, n_colors: int) -> List:
        """Get color scheme for methods."""
        if self.config.palette in plt.colormaps:
            cmap = plt.colormaps[self.config.palette]
            return [
                cmap(i / (n_colors - 1) if n_colors > 1 else 0.5)
                for i in range(n_colors)
            ]
        return sns.color_palette(self.config.palette, n_colors=n_colors)

    def _check_if_2d(self) -> bool:
        """Check if the optimization problem is 2D."""
        try:
            # Try calling with a 2D point
            test_point = np.array([0.0, 0.0])
            self.problem.func(test_point)
            # If it works and x0 is 2D, then it's a 2D problem
            return len(self.methods[0].get_current_x()) == 2
        except:
            return False

    def setup_plots(self):
        """Setup the subplot layout based on problem dimensionality."""
        if self.is_2d:
            # Create 2x2 GridSpec with equal sizing
            gs = GridSpec(2, 2, figure=self.fig)

            # Top left: 3D surface plot
            self.ax_main = self.fig.add_subplot(gs[0, 0], projection="3d")

            # Top right: Path progress plot
            self.ax_progress = self.fig.add_subplot(gs[0, 1])

            # Bottom left: Contour plot with gradient field
            self.ax_contour = self.fig.add_subplot(gs[1, 0])

            # Bottom right: Error convergence
            self.ax_error = self.fig.add_subplot(gs[1, 1])
        else:
            gs = GridSpec(2, 1, height_ratios=[2, 1], figure=self.fig)
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_error = self.fig.add_subplot(gs[1])

        # Remove the main title
        # self.fig.suptitle(self.config.title, y=0.95)

    def plot_function(self):
        """Plot the objective function landscape."""
        x = np.linspace(*self.problem.x_range, 200)
        if self.is_2d:
            self._plot_2d_function(x)
        else:
            self._plot_1d_function()

    def _plot_1d_function(self):
        """Plot 1D optimization landscape with enhanced styling."""
        x_min, x_max = self.problem.x_range
        x_range = x_max - x_min
        x_plot_min = x_min - 0.2 * x_range
        x_plot_max = x_max + 0.2 * x_range
        x = np.linspace(x_plot_min, x_plot_max, 1000)

        # Compute function values
        y = np.array([self.problem.func(np.array([xi])) for xi in x])

        # Create custom color gradient
        colors = ["#FF6B6B", "#4ECDC4"]  # Modern color scheme
        n_colors = 256
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_colors)

        # Plot the function with gradient fill
        self.ax_main.fill_between(x, y, alpha=0.2, color=colors[0], zorder=1)
        self.function_line = self.ax_main.plot(
            x,
            y,
            color=colors[1],
            alpha=0.8,
            label="f(x)",
            zorder=2,
            linewidth=2.5,
            path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()],
        )[0]

        # Enhanced styling
        self.ax_main.grid(True, linestyle="--", alpha=0.3)
        self.ax_main.set_xlabel("x", fontsize=12, fontweight="bold")
        self.ax_main.set_ylabel("f(x)", fontsize=12, fontweight="bold")

        # Add subtle background color
        self.ax_main.set_facecolor("#f8f9fa")

        # Fix scientific notation and scaling
        self.ax_main.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")

        # Set view limits with padding
        y_min, y_max = np.min(y), np.max(y)
        y_range = max(y_max - y_min, 1e-10)
        y_plot_min = y_min - 0.1 * y_range
        y_plot_max = y_max + 0.1 * y_range

        self.ax_main.set_xlim(x_plot_min, x_plot_max)
        self.ax_main.set_ylim(y_plot_min, y_plot_max)

        # Add optimal point annotation if known
        if hasattr(self.problem, "optimal_value"):
            opt_x = self.problem.optimal_point[0]
            opt_y = self.problem.optimal_value
            self.ax_main.plot(opt_x, opt_y, "r*", markersize=15, label="Global Minimum")
            self.ax_main.annotate(
                f"Minimum: ({opt_x:.2f}, {opt_y:.2f})",
                (opt_x, opt_y),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    def _plot_2d_function(self, x: np.ndarray):
        """Plot 2D objective function with enhanced professional visualization."""
        y = x
        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                [self.problem.func(np.array([xi, yi])) for xi, yi in zip(x_row, y_row)]
                for x_row, y_row in zip(X, Y)
            ]
        )

        # Calculate gradients for quiver plot
        dx = np.zeros_like(Z)
        dy = np.zeros_like(Z)
        grad_magnitude = np.zeros_like(Z)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                grad = self.problem.derivative(np.array([X[i, j], Y[i, j]]))
                dx[i, j] = grad[0]
                dy[i, j] = grad[1]
                grad_magnitude[i, j] = np.linalg.norm(grad)

        # Create professional colormap with perceptually uniform gradients
        colors = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff"]  # Blues
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

        # 1. Enhanced 3D surface plot (top left)
        self.surface = self.ax_main.plot_surface(
            X,
            Y,
            Z,
            cmap=custom_cmap,
            alpha=0.9,
            antialiased=True,
            linewidth=0.5,
            rcount=200,
            ccount=200,
        )
        self.ax_main.set_title("Surface Plot", fontsize=10, pad=5)

        # Add colorbar for surface plot
        colorbar = self.fig.colorbar(
            self.surface,
            ax=self.ax_main,
            pad=0.1,
            format=mtick.ScalarFormatter(useMathText=True),
        )
        colorbar.set_label("Function Value", fontsize=10)

        # 2. Path Progress Plot (top right)
        self.ax_progress.set_title("Function Value Progress", fontsize=10, pad=5)
        self.ax_progress.set_xlabel("Step", fontsize=10)
        self.ax_progress.set_ylabel("$f(x_k)$", fontsize=10)  # Mathematical notation
        self.ax_progress.grid(True, linestyle="--", alpha=0.2)
        self.ax_progress.set_yscale("log")

        # Add reference lines for better context
        if hasattr(self.problem, "optimal_value"):
            opt_val = self.problem.optimal_value
            self.ax_progress.axhline(
                y=opt_val,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Global Minimum: {opt_val:.2e}",
            )

        # 3. Enhanced Gradient Field (bottom left)
        if self.config.show_contour:
            # Add filled contours with better color scheme
            levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 20)
            contourf = self.ax_contour.contourf(
                X,
                Y,
                Z,
                levels=levels,
                cmap="Blues",
                alpha=0.2,
            )

            # Add contour lines
            contours = self.ax_contour.contour(
                X,
                Y,
                Z,
                levels=levels,
                colors="k",
                alpha=0.3,
                linewidths=0.5,
            )

            # Add gradient field with better visibility
            skip = 8
            magnitude_norm = plt.Normalize(grad_magnitude.min(), grad_magnitude.max())
            quiver = self.ax_contour.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                dx[::skip, ::skip],
                dy[::skip, ::skip],
                grad_magnitude[::skip, ::skip],
                cmap="Reds",
                alpha=0.8,
                norm=magnitude_norm,
                scale=50,
                width=0.003,
            )

            self.ax_contour.set_title("Gradient Field & Contours", fontsize=10, pad=5)
            self.ax_contour.set_xlabel("$x_1$", fontsize=10)
            self.ax_contour.set_ylabel("$x_2$", fontsize=10)
            self.ax_contour.set_aspect("equal")

            # Add colorbar for gradient magnitude
            quiver_cbar = self.fig.colorbar(
                quiver,
                ax=self.ax_contour,
                format=mtick.ScalarFormatter(useMathText=True),
            )
            quiver_cbar.set_label("Gradient Magnitude", fontsize=8)

        # 4. Error plot (bottom right)
        self.ax_error.set_title("Gradient Norm Convergence", fontsize=10, pad=5)
        self.ax_error.set_xlabel("Iteration", fontsize=10)
        self.ax_error.set_ylabel(
            "$\\| \\nabla f(x_k) \\|$", fontsize=10
        )  # Gradient norm notation
        self.ax_error.grid(True, linestyle="--", alpha=0.2)
        self.ax_error.set_yscale("log")

        # Add reference line for tolerance
        self.ax_error.axhline(
            y=self.problem.tol,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Tolerance: {self.problem.tol:.0e}",
        )

        # Common styling
        for ax in [self.ax_progress, self.ax_contour]:
            ax.grid(True, linestyle="--", alpha=0.2)

    def run_comparison(self):
        """Run and visualize the optimization methods."""
        plt.ion()

        # Reduce drawing frequency - only draw every N iterations
        draw_interval = 5  # Only draw every 5 iterations
        draw_counter = 0

        # Disable automatic drawing
        self.fig.canvas.draw_idle()
        plt.show(block=False)

        # Turn off automatic rendering
        self.ax_main.set_autoscale_on(False)
        if self.is_2d and self.config.show_contour:
            self.ax_contour.set_autoscale_on(False)
        self.ax_error.set_autoscale_on(False)

        # Generate test points and find global minimum
        if self.problem.is_2d:
            x = np.linspace(self.problem.x_range[0], self.problem.x_range[1], 100)
            y = np.linspace(self.problem.x_range[0], self.problem.x_range[1], 100)
            X, Y = np.meshgrid(x, y)
            x_test = np.vstack((X.ravel(), Y.ravel())).T
            f_min = np.min([self.problem.func(x) for x in x_test])
        else:
            x_test = np.linspace(self.problem.x_range[0], self.problem.x_range[1], 100)
            f_min = np.min([self.problem.func(np.array([x])) for x in x_test])

        for iteration in range(self.problem.max_iter):
            any_updated = False

            for method_id, state in self.method_states.items():
                method = state["method"]
                if not method.has_converged():
                    any_updated = True
                    x_new = method.step()
                    f_new = self.problem.func(x_new)

                    # Update progress plot
                    state["points"].append(x_new)
                    state["f_values"] = state.get("f_values", []) + [f_new]
                    iterations = range(len(state["f_values"]))
                    self.ax_progress.plot(
                        iterations,
                        state["f_values"],
                        color=state["color"],
                        label=method.__class__.__name__ if iteration == 0 else "",
                    )

                    # Get error as scalar (gradient norm for optimization)
                    error = float(np.linalg.norm(method.derivative(x_new)))

                    state["errors"].append(error)

                    # Update optimization path
                    points = np.array(state["points"])
                    values = np.array([self.problem.func(p) for p in points])

                    if self.is_2d:
                        # Update 3D path and contour
                        state["line"].set_data(points[:, 0], points[:, 1])
                        state["line"].set_3d_properties(values)
                        if state["contour_line"]:
                            state["contour_line"].set_data(points[:, 0], points[:, 1])
                    else:
                        # Update 1D path
                        state["line"].set_data(points.ravel(), values)

                    # Update error plot with iteration numbers
                    errors = np.array(state["errors"])
                    self.error_lines[method_id].set_data(iterations, errors)

            if any_updated:
                draw_counter += 1
                if draw_counter >= draw_interval:
                    # Batch update all artists
                    if self.is_2d:
                        self.ax_main.draw_artist(self.surface)
                        for state in self.method_states.values():
                            self.ax_main.draw_artist(state["line"])
                            if state["contour_line"]:
                                self.ax_contour.draw_artist(state["contour_line"])
                    else:
                        self.ax_main.draw_artist(self.function_line)
                        for state in self.method_states.values():
                            self.ax_main.draw_artist(state["line"])

                    # Update error lines
                    for line in self.error_lines.values():
                        self.ax_error.draw_artist(line)

                    # Fast update of the canvas
                    self.fig.canvas.flush_events()
                    draw_counter = 0

                # Update progress plot
                self.ax_progress.relim()
                self.ax_progress.autoscale_view()
                # Remove legend from progress plot
                # if iteration == 0:
                #     self.ax_progress.legend()

            if all(method.has_converged() for method in self.methods):
                break

        # Final update
        self.fig.canvas.draw()
        plt.show(block=True)

    def _update_plot_limits(self):
        """Update axis limits to show all data."""
        if self.is_2d:
            all_points = np.concatenate(
                [
                    state["points"]
                    for state in self.method_states.values()
                    if len(state["points"]) > 0
                ]
            )
            all_values = [
                v for state in self.method_states.values() for v in state["errors"]
            ]

            self.ax_main.set_xlim(np.min(all_points[:, 0]), np.max(all_points[:, 0]))
            self.ax_main.set_ylim(np.min(all_points[:, 1]), np.max(all_points[:, 1]))
            self.ax_main.set_zlim(np.min(all_values), np.max(all_values))

            if self.config.show_contour:
                self.ax_contour.set_xlim(
                    np.min(all_points[:, 0]), np.max(all_points[:, 0])
                )
                self.ax_contour.set_ylim(
                    np.min(all_points[:, 1]), np.max(all_points[:, 1])
                )
        else:
            # For 1D case, keep x limits fixed as set in _plot_1d_function
            # Only update error plot limits
            if any(len(state["errors"]) > 0 for state in self.method_states.values()):
                self.ax_error.relim()
                self.ax_error.autoscale_view()
