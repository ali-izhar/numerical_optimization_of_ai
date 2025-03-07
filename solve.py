#!/usr/bin/env python3

"""
Unified tool for solving numerical problems.

This script provides a unified interface for both root-finding and optimization methods,
allowing for easy comparison and visualization of different algorithms.
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import importlib
import time

from algorithms.convex.protocols import (
    BaseNumericalMethod,
    NumericalMethodConfig,
    MethodType,
    IterationData,
)

# Import our new visualization components
from plot import (
    VisualizationConfig,
    FunctionSpace,
    ConvergencePlot,
    ErrorPlot,
    MethodAnimation,
    VisualizerFactory,
    PlotFactory,
    extract_iteration_data,
    prepare_method_comparison_data,
    prepare_animation_data,
)

# Root-Finding Methods
ROOT_FINDING_METHODS = {
    "bisection": "algorithms.convex.bisection.BisectionMethod",
    "regula_falsi": "algorithms.convex.regula_falsi.RegulaFalsiMethod",
    "secant": "algorithms.convex.secant.SecantMethod",
    "newton": "algorithms.convex.newton.NewtonMethod",
}

# Optimization Methods
OPTIMIZATION_METHODS = {
    "golden_section": "algorithms.convex.golden_section.GoldenSectionMethod",
    "fibonacci": "algorithms.convex.fibonacci.FibonacciMethod",
    "steepest_descent": "algorithms.convex.steepest_descent.SteepestDescentMethod",
    "newton_opt": "algorithms.convex.newton.NewtonMethod",
    "newton_hessian": "algorithms.convex.newton_hessian.NewtonHessianMethod",
    "quasi_newton": "algorithms.convex.quasi_newton.BFGSMethod",
    "nelder_mead": "algorithms.convex.nelder_mead.NelderMeadMethod",
    "powell": "algorithms.convex.powell_quadratic.PowellMethod",
    "powell_conjugate": "algorithms.convex.powell_conjugate.PowellConjugateMethod",
}

# All available methods (combined)
ALL_METHODS = {**ROOT_FINDING_METHODS, **OPTIMIZATION_METHODS}

# Test Functions for Root-Finding
ROOT_FUNCTIONS = {
    "simple_quadratic": lambda x: x**2 - 4,
    "cubic": lambda x: x**3 - 2 * x**2 - 5 * x + 6,
    "trigonometric": lambda x: np.sin(x) - 0.5,
    "exponential": lambda x: np.exp(x) - 5,
    "logarithmic": lambda x: np.log(x) - 1,
    "compound": lambda x: np.sin(x) * np.exp(-0.1 * x) - 0.2,
    "discontinuous": lambda x: 1 / x if x != 0 else float("inf"),
    "stiff": lambda x: 1e6 * (x - 1) + 1e-6 * np.sin(1000 * x),
}

# Test Functions for Optimization
OPTIMIZATION_FUNCTIONS = {
    "quadratic": lambda x: x**2 + 2 * x + 1,
    "rosenbrock": lambda x: (
        (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else (1 - x) ** 2
    ),
    "himmelblau": lambda x: (
        (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else (x**2 + 2 - 11) ** 2 + (x + 4 - 7) ** 2
    ),
    "rastrigin": lambda x: (
        20
        + x[0] ** 2
        - 10 * np.cos(2 * np.pi * x[0])
        + x[1] ** 2
        - 10 * np.cos(2 * np.pi * x[1])
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else 10 + x**2 - 10 * np.cos(2 * np.pi * x)
    ),
    "beale": lambda x: (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else (1.5 - x + x * 0.5) ** 2
    ),
    "booth": lambda x: (
        (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else (x + 2 * 0.5 - 7) ** 2
    ),
    "matyas": lambda x: (
        0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else 0.26 * (x**2 + 0.5**2) - 0.48 * x * 0.5
    ),
    "sphere": lambda x: (
        sum(xi**2 for xi in x)
        if isinstance(x, (list, np.ndarray)) and hasattr(x, "__len__") and len(x) > 1
        else x**2
    ),
}

# Step Length Methods
STEP_LENGTH_METHODS = {
    "fixed": "fixed",
    "backtracking": "backtracking",
    "exact": "exact",
    "wolfe": "wolfe",
    "strong_wolfe": "strong_wolfe",
}

# Descent Direction Methods
DESCENT_DIRECTION_METHODS = {
    "gradient": "steepest_descent",
    "newton": "newton",
    "bfgs": "bfgs",
    "conjugate": "conjugate_gradient",
}


def load_config_file(config_path: Path) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def determine_x_range(
    function_name: str,
    x0_values: List[float],
    method_type: MethodType,
    specified_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Determine an appropriate x-range for visualization based on the function and initial points.

    Args:
        function_name: Name of the function
        x0_values: Initial points
        method_type: Type of method (root-finding or optimization)
        specified_range: User-specified range (if provided)

    Returns:
        Tuple[float, float]: Appropriate x-range for visualization
    """
    if specified_range is not None:
        return specified_range

    # Define default ranges for known functions
    default_ranges = {
        # Root-finding function ranges
        "simple_quadratic": (-3, 3),
        "cubic": (-2, 4),
        "trigonometric": (0, 2 * np.pi),
        "exponential": (0, 2),
        "logarithmic": (0.1, 5),
        "compound": (-2, 8),
        "discontinuous": (-5, 5),
        "stiff": (0.5, 1.5),
        # Optimization function ranges
        "quadratic": (-5, 3),
        "rosenbrock": (-2, 2),
        "himmelblau": (-5, 5),
        "rastrigin": (-5.12, 5.12),
        "beale": (-4.5, 4.5),
        "booth": (-10, 10),
        "matyas": (-10, 10),
        "sphere": (-5, 5),
    }

    if function_name in default_ranges:
        return default_ranges[function_name]

    # If no default range exists, use the initial points to determine a range
    min_x0 = min(x0_values)
    max_x0 = max(x0_values)

    # Ensure the range has a minimum width
    width = max(max_x0 - min_x0, 1.0)

    # Add padding
    padding = width * 0.5

    return (min_x0 - padding, max_x0 + padding)


def create_method(
    method_name: str,
    config: NumericalMethodConfig,
    x0: Union[float, np.ndarray],
    x1: Optional[Union[float, np.ndarray]] = None,
) -> BaseNumericalMethod:
    """
    Create a numerical method instance based on the method name and configuration.

    Args:
        method_name: Name of the method to create
        config: Numerical method configuration
        x0: Initial point
        x1: Second initial point (for methods that require it)

    Returns:
        BaseNumericalMethod: Instantiated numerical method
    """
    if method_name not in ALL_METHODS:
        raise ValueError(f"Unknown method: {method_name}")

    # Import the method class
    module_path, class_name = ALL_METHODS[method_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    method_class = getattr(module, class_name)

    # Methods that require an interval (a, b)
    interval_methods = ["bisection", "regula_falsi", "golden_section", "fibonacci"]

    # Special handling based on method type
    if method_name in interval_methods:
        # These methods need two initial points (an interval)
        if x1 is None:
            # If only one point is provided, create an interval around it
            if method_name in ["bisection", "regula_falsi"]:
                # For root-finding, we need to bracket the root
                # Try to find an interval that brackets the root
                f_x0 = config.func(x0)

                # Start with a positive and negative point
                if x0 >= 0:
                    a, b = -abs(x0) - 1, x0
                else:
                    a, b = x0, abs(x0) + 1

                # Check if this interval works
                f_a, f_b = config.func(a), config.func(b)

                # If we don't have a sign change, try to expand the interval
                if f_a * f_b > 0:  # Same sign
                    # Try expanding the interval
                    for _ in range(10):  # Try a few times
                        a = a * 2  # Double the interval size
                        b = b * 2
                        f_a, f_b = config.func(a), config.func(b)
                        if f_a * f_b <= 0:  # Opposite sign or one is zero
                            break
                    else:
                        # We couldn't find a bracketing interval
                        raise ValueError(
                            f"Could not find interval that brackets the root for {method_name}"
                        )
            else:
                # For optimization methods, we just need a reasonable interval
                a, b = x0 - 2, x0 + 2

        else:
            # User provided both points, use them
            a, b = min(x0, x1), max(x0, x1)

            # For root-finding methods, verify the interval brackets the root
            if method_name in ["bisection", "regula_falsi"]:
                f_a, f_b = config.func(a), config.func(b)
                if f_a * f_b > 0:  # Same sign
                    raise ValueError(
                        f"Interval [{a}, {b}] does not bracket a root. f({a}) = {f_a}, f({b}) = {f_b}"
                    )

        # Create method with interval
        method = method_class(config, a, b)

    elif method_name == "secant":
        # Secant method needs two points
        if x1 is None:
            # For secant method, use a default value for x1 if not provided
            x1 = x0 + 0.1
        method = method_class(config, x0, x1)

    elif method_name == "newton_opt":
        # Newton optimization method needs a second derivative
        if config.method_type == "optimize":
            # Debug print statements
            print(f"Creating Newton optimization method with second derivative")
            # Create method with second derivative
            if config.hessian is None:
                raise ValueError(
                    "Newton's method requires second derivative for optimization"
                )
            method = method_class(config, x0, second_derivative=config.hessian)
        else:
            # For root-finding, just use the regular constructor
            method = method_class(config, x0)

    elif method_name == "powell":
        # Powell method requires a and b parameters
        if x1 is None:
            # If only one point provided, create reasonable bounds around it
            a, b = x0 - 2, x0 + 2
        else:
            # Use user-provided bounds
            a, b = min(x0, x1), max(x0, x1)

        method = method_class(config, a, b)

    else:
        # For other methods, just use the standard constructor
        method = method_class(config, x0)

    return method


def run_methods(
    function_name: str,
    method_names: List[str],
    x0_values: List[float],
    method_type: str = "root",
    tol: float = 1e-6,
    max_iter: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
    step_length_method: Optional[str] = None,
    step_length_params: Optional[Dict[str, Any]] = None,
    descent_direction_method: Optional[str] = None,
    descent_direction_params: Optional[Dict[str, Any]] = None,
    visualize: bool = True,
    save_viz: Optional[str] = None,
    viz_format: str = "html",
    viz_3d: bool = False,
    save_data: Optional[Path] = None,
    is_2d: bool = False,
) -> Tuple[List[BaseNumericalMethod], pd.DataFrame]:
    """
    Run specified numerical methods on a function and optionally visualize the results.

    Args:
        function_name: Name of the function to analyze
        method_names: List of methods to run
        x0_values: List of initial points
        method_type: Type of method (root-finding or optimization)
        tol: Error tolerance
        max_iter: Maximum number of iterations
        x_range: Range for x-axis visualization (min, max)
        step_length_method: Step length method for optimization
        step_length_params: Parameters for step length method
        descent_direction_method: Descent direction method for optimization
        descent_direction_params: Parameters for descent direction method
        visualize: Whether to visualize the results
        save_viz: Path to save visualization (None for no saving)
        viz_format: Format for saved visualization ("html", "png", etc.)
        viz_3d: Whether to create 3D visualization
        save_data: Path to save iteration history data
        is_2d: Whether the function is 2D

    Returns:
        Tuple[List[BaseNumericalMethod], pd.DataFrame]: List of method instances and results table
    """
    # Get the function
    if method_type == "root":
        if function_name not in ROOT_FUNCTIONS:
            raise ValueError(f"Unknown root-finding function: {function_name}")
        func = ROOT_FUNCTIONS[function_name]
    else:  # method_type == "optimize"
        if function_name not in OPTIMIZATION_FUNCTIONS:
            raise ValueError(f"Unknown optimization function: {function_name}")
        func = OPTIMIZATION_FUNCTIONS[function_name]

    # Filter methods based on method type
    valid_methods = {}
    if method_type == "root":
        valid_methods = ROOT_FINDING_METHODS
    else:  # method_type == "optimize"
        valid_methods = OPTIMIZATION_METHODS

    filtered_methods = [m for m in method_names if m in valid_methods]
    if not filtered_methods:
        raise ValueError(f"No valid {method_type} methods specified")

    # Determine x range for visualization
    if x_range is None:
        x_range = determine_x_range(function_name, x0_values, method_type)

    # Create function configuration
    config = NumericalMethodConfig(
        func=func,
        method_type=method_type,
        x_range=x_range,
        tol=tol,
        max_iter=max_iter,
        is_2d=is_2d,
    )

    # Add derivative if available
    if function_name == "simple_quadratic":
        config.derivative = lambda x: 2 * x
    elif function_name == "cubic":
        config.derivative = lambda x: 3 * x**2 - 4 * x - 5
    elif function_name == "quadratic":
        config.derivative = lambda x: 2 * x + 2
    elif function_name == "rosenbrock":
        # For scalar or 1D case
        if not is_2d:
            config.derivative = lambda x: -2 + 2 * x  # derivative of (1-x)^2
        else:
            # For 2D case - gradient of Rosenbrock function
            config.derivative = lambda x: np.array(
                [
                    -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                    200 * (x[1] - x[0] ** 2),
                ]
            )

            # Hessian matrix for Newton method
            config.hessian = lambda x: np.array(
                [[2 + 1200 * x[0] ** 2 - 400 * x[1], -400 * x[0]], [-400 * x[0], 200]]
            )

    # Debug print statements
    print(f"Function type: {function_name}")
    print(f"Method type: {method_type}")
    print(f"is_2d: {is_2d}")
    print(f"Has derivative: {config.derivative is not None}")
    print(f"Has hessian: {config.hessian is not None}")

    # Add step length method if specified
    if step_length_method is not None:
        if step_length_method not in STEP_LENGTH_METHODS:
            raise ValueError(f"Unknown step length method: {step_length_method}")
        config.step_length_method = STEP_LENGTH_METHODS[step_length_method]
        config.step_length_params = step_length_params or {}

    # Add descent direction method if specified
    if descent_direction_method is not None:
        if descent_direction_method not in DESCENT_DIRECTION_METHODS:
            raise ValueError(
                f"Unknown descent direction method: {descent_direction_method}"
            )
        config.descent_direction_method = DESCENT_DIRECTION_METHODS[
            descent_direction_method
        ]
        config.descent_direction_params = descent_direction_params or {}

    # Create methods
    methods = []
    for method_name in filtered_methods:
        if is_2d:
            # For 2D optimization, make sure we have at least two values
            if len(x0_values) < 2:
                x0_values = x0_values + [0.0] * (2 - len(x0_values))
            # Create a 2D array from the first two values
            x0 = np.array(x0_values[:2])
        else:
            # For 1D, just use the first value
            x0 = x0_values[0]

        try:
            # Create method
            method = create_method(method_name, config, x0)
            # Add to list
            methods.append(method)
        except Exception as e:
            print(f"Error creating method {method_name}: {e}")

    # Run methods
    results = []
    for method in methods:
        print(f"Running {method.name}...")
        start_time = time.time()

        # Run method until convergence or max iterations
        while not method.has_converged():
            method.step()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get results
        result = {
            "Method": method.name,
            "Converged": method.has_converged(),
            "Iterations": len(method.get_iteration_history()),
            "Final Value": method.get_current_x(),
            "Final Error": method.get_error(),
            "Time (s)": elapsed_time,
        }
        results.append(result)

    # Create results table
    results_df = pd.DataFrame(results)

    # Print results
    if results_df is not None and not results_df.empty:
        print("\nResults:")
        # Reset display options to show all information
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.expand_frame_repr", False)
        print(results_df)
    elif not methods:
        print("\nNo methods could be successfully created.")
    else:
        print("\nNo results available.")

    # Save iteration history if requested
    if save_data:
        save_iteration_history(methods, function_name, save_data)

    # Visualize results if requested
    if visualize and methods:
        visualize_results(
            methods,
            config,
            function_name,
            save_viz=save_viz,
            viz_format=viz_format,
            viz_3d=viz_3d,
            method_type=method_type,
        )

    return methods, results_df


def save_iteration_history(
    methods: List[BaseNumericalMethod], function_name: str, save_dir: Path
):
    """
    Save iteration history data to CSV files.

    Args:
        methods: List of method instances
        function_name: Name of the function
        save_dir: Directory to save data
    """
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        history = method.get_iteration_history()
        if not history:
            continue

        # Create DataFrame
        data = []
        for h in history:
            row = {
                "iteration": h.iteration,
                "x_old": h.x_old,
                "x_new": h.x_new,
                "f_old": h.f_old,
                "f_new": h.f_new,
                "error": h.error,
            }

            # Add details
            for k, v in h.details.items():
                row[f"detail_{k}"] = v

            data.append(row)

        df = pd.DataFrame(data)

        # Save to CSV
        filename = f"{function_name}_{method.name}.csv"
        filepath = save_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved iteration history to {filepath}")


def visualize_results(
    methods: List[BaseNumericalMethod],
    config: NumericalMethodConfig,
    function_name: str,
    save_viz: Optional[str] = None,
    viz_format: str = "html",
    viz_3d: bool = False,
    method_type: str = "root",
):
    """
    Visualize the results of numerical methods using Plotly for eye-catching visualizations.

    Args:
        methods: List of method instances
        config: Numerical method configuration
        function_name: Name of the function
        save_viz: Path to save visualization (None for no saving)
        viz_format: Format for saved visualization ("html", "png", etc.)
        viz_3d: Whether to create 3D visualization
        method_type: Type of method (root-finding or optimization)
    """
    # Check if we have any valid methods to visualize
    if not methods:
        print("No valid methods to visualize.")
        return

    # Create enhanced visualization configuration with improved styling
    vis_config = VisualizationConfig(
        title=f"{method_type.capitalize()} Methods for {function_name}",
        palette="turbo",  # More visually appealing color palette
        plotly_template="plotly_white",
        background_color="rgba(255, 255, 255, 0.95)",
        use_plotly_3d=viz_3d and config.is_2d,  # Use 3D when appropriate
    )

    # Create function space with enhanced title
    title_prefix = "Root Finding" if method_type == "root" else "Optimization"
    function_space = FunctionSpace(
        func=config.func,
        x_range=config.x_range,
        title=f"{title_prefix} for {function_name}",
        is_2d=config.is_2d,
        # Use colorscale from config
        colormap=(
            vis_config.plotly_colorscales["surface"]
            if hasattr(vis_config, "plotly_colorscales") and viz_3d and config.is_2d
            else "Viridis"
        ),
    )

    # Prepare data for visualization
    method_data = extract_iteration_data(methods, is_2d=config.is_2d)
    comparison_data = prepare_method_comparison_data(methods)
    animation_data = prepare_animation_data(methods, is_2d=config.is_2d)

    # Create interactive Plotly visualization as the primary visualization
    # Let it use the config for dimensions - no hardcoded values
    interactive_fig = PlotFactory.create_interactive_comparison(
        methods=methods,
        function_space=function_space,
        vis_config=vis_config,  # Pass the enhanced config
        include_error_plot=True,
        log_scale_error=True,
        surface_plot=vis_config.use_plotly_3d,  # Use the config setting
        # No height/width parameters - let it be responsive
    )

    # Show the interactive plot - no need for additional update_layout as
    # the enhanced PlotFactory handles styling
    interactive_fig.show()

    # Save the visualization if requested
    if save_viz:
        # Create directory if it doesn't exist
        save_path = Path(save_viz)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the figure
        if viz_format == "html":
            interactive_fig.write_html(
                save_path.with_suffix(".html"),
                full_html=True,
                include_plotlyjs="cdn",
                include_mathjax="cdn",
                config={"responsive": True},  # Make responsive in HTML output
            )
            print(f"Saved visualization to {save_path.with_suffix('.html')}")
        elif viz_format in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
            # For static images, set a reasonable size with higher resolution
            interactive_fig.write_image(
                save_path.with_suffix(f".{viz_format}"), width=1200, height=800, scale=2
            )
            print(f"Saved visualization to {save_path.with_suffix(f'.{viz_format}')}")

    # Create Plotly animation with enhanced styling
    plotly_animation = MethodAnimation(
        function_space=function_space,
        title=f"{method_type.capitalize()} Methods Animation",
        color_palette=vis_config.palette,  # Use the config palette
    )

    # Create Plotly animation - no hardcoded dimensions
    anim_fig = plotly_animation.create_plotly_animation(
        method_paths=animation_data["method_paths"],
        error_data=animation_data["error_data"],
        critical_points=animation_data["critical_points"],
        surface_plot=vis_config.use_plotly_3d,  # Use the config setting
        # No height/width parameters - let it be responsive
        duration=vis_config.animation_duration,
        transition_duration=vis_config.animation_transition,
    )

    # Show the animation - no need for additional update_layout
    # as the enhanced MethodAnimation handles styling
    anim_fig.show()

    # Save the animation if requested
    if save_viz:
        if viz_format == "html":
            anim_fig.write_html(
                save_path.with_suffix("_animation.html"),
                full_html=True,
                include_plotlyjs="cdn",
                config={"responsive": True},  # Make responsive in HTML output
            )
            print(f"Saved animation to {save_path.with_suffix('_animation.html')}")
        elif viz_format == "mp4" and hasattr(anim_fig, "write_video"):
            try:
                # For video, set a reasonable size and framerate
                anim_fig.write_video(
                    save_path.with_suffix(".mp4"),
                    width=1200,
                    height=800,
                    fps=15,  # Smoother framerate
                )
                print(f"Saved animation to {save_path.with_suffix('.mp4')}")
            except Exception as e:
                print(f"Could not save animation as MP4: {e}")
                print("Falling back to HTML format for animation.")
                anim_fig.write_html(
                    save_path.with_suffix("_animation.html"),
                    config={"responsive": True},
                )
                print(f"Saved animation to {save_path.with_suffix('_animation.html')}")

    # Only fallback to matplotlib if explicitly requested or if Plotly is not available
    if viz_format == "matplotlib":
        # Create matplotlib plots for backward compatibility
        fig, axes = PlotFactory.create_comparison_plot(
            methods=methods,
            function_space=function_space,
            vis_config=vis_config,
            include_error_plot=True,
        )
        plt.show()

        # Create matplotlib animation
        anim = plotly_animation.create_matplotlib_animation(
            method_paths=animation_data["method_paths"],
            error_data=animation_data["error_data"],
            critical_points=animation_data["critical_points"],
        )
        plt.show()


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Solve numerical problems using various methods"
    )

    # Add subparsers for root-finding and optimization
    subparsers = parser.add_subparsers(
        dest="problem_type", help="Type of problem to solve"
    )

    # Root-finding subparser
    root_parser = subparsers.add_parser("root", help="Find roots of functions")
    root_parser.add_argument(
        "-f",
        "--function",
        choices=ROOT_FUNCTIONS.keys(),
        default="simple_quadratic",
        help="Function to find roots of",
    )
    root_parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        choices=ROOT_FINDING_METHODS.keys(),
        default=["bisection", "newton"],
        help="Methods to use",
    )

    # Optimization subparser
    opt_parser = subparsers.add_parser("optimize", help="Minimize functions")
    opt_parser.add_argument(
        "-f",
        "--function",
        choices=OPTIMIZATION_FUNCTIONS.keys(),
        default="quadratic",
        help="Function to minimize",
    )
    opt_parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        choices=OPTIMIZATION_METHODS.keys(),
        default=["golden_section", "newton_opt"],
        help="Methods to use",
    )
    opt_parser.add_argument(
        "--step-length",
        choices=STEP_LENGTH_METHODS.keys(),
        help="Step length method for gradient-based optimization",
    )
    opt_parser.add_argument(
        "--descent-direction",
        choices=DESCENT_DIRECTION_METHODS.keys(),
        help="Descent direction method for gradient-based optimization",
    )
    opt_parser.add_argument(
        "--2d",
        dest="is_2d",
        action="store_true",
        help="Use 2D optimization (for methods that support it)",
    )

    # Common parameters
    for p in [root_parser, opt_parser]:
        p.add_argument(
            "-x0",
            "--initial-points",
            type=float,
            nargs="+",
            default=[1.0],
            help="Initial point(s) for methods",
        )
        p.add_argument(
            "-t", "--tolerance", type=float, default=1e-6, help="Error tolerance"
        )
        p.add_argument(
            "-i",
            "--max-iterations",
            type=int,
            default=100,
            help="Maximum number of iterations",
        )
        p.add_argument(
            "-r",
            "--range",
            type=float,
            nargs=2,
            help="Range for x-axis visualization (min max)",
        )
        p.add_argument(
            "-c", "--config", type=Path, help="Path to configuration JSON file"
        )
        p.add_argument("--no-viz", action="store_true", help="Disable visualization")
        p.add_argument("--save-viz", type=str, help="Path to save visualization")
        p.add_argument(
            "--viz-format",
            choices=["html", "png", "jpg", "mp4"],
            default="html",
            help="Format for saved visualization",
        )
        p.add_argument(
            "--viz-3d",
            action="store_true",
            help="Create 3D visualization (for 2D functions)",
        )
        p.add_argument(
            "--save-data", type=Path, help="Path to save iteration history data"
        )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Default to root-finding if no problem type specified
    if args.problem_type is None:
        args.problem_type = "root"
        args.function = "simple_quadratic"
        args.methods = ["bisection", "newton"]
        args.initial_points = [1.0]

    # Determine method type
    method_type = args.problem_type

    # Load configuration file if provided
    config = {}
    if hasattr(args, "config") and args.config:
        config = load_config_file(args.config)

    # Set is_2d flag
    is_2d = False
    if hasattr(args, "is_2d"):
        is_2d = args.is_2d

    # Run methods
    methods, results = run_methods(
        function_name=args.function,
        method_names=args.methods,
        x0_values=args.initial_points,
        method_type=method_type,
        tol=args.tolerance,
        max_iter=args.max_iterations,
        x_range=args.range,
        step_length_method=args.step_length if hasattr(args, "step_length") else None,
        descent_direction_method=(
            args.descent_direction if hasattr(args, "descent_direction") else None
        ),
        visualize=not args.no_viz,
        save_viz=args.save_viz,
        viz_format=args.viz_format,
        viz_3d=args.viz_3d,
        save_data=args.save_data,
        is_2d=is_2d,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
