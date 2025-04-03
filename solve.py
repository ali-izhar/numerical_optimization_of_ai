#!/usr/bin/env python3

"""
Unified tool for solving numerical problems.

This script provides a unified interface for both root-finding and optimization methods,
allowing for easy comparison and visualization of different algorithms.
"""

import sys
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

# Import from utils.cli to avoid duplication
from utils.cli import (
    ROOT_FUNCTIONS,
    OPTIMIZATION_FUNCTIONS,
    ROOT_FINDING_METHODS,
    OPTIMIZATION_METHODS,
    ALL_METHODS,
    STEP_LENGTH_METHODS,
    DESCENT_DIRECTION_METHODS,
    parse_args,
)

# Import from utils.file_manager for file operations
from utils.file_manager import (
    load_config_file,
    save_iteration_history,
    save_visualization,
    save_animation,
)

# Import from utils.funcs for function-related utilities
from utils.funcs import determine_x_range, get_function, list_function_categories

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

    # If two points weren't provided but are needed, get appropriate initial points
    if x1 is None and (method_name in interval_methods or method_name == "secant"):
        try:
            # Use midpoint utility to get safe initial points
            from utils.midpoint import get_safe_initial_points

            x0, x1 = get_safe_initial_points(
                config.func, config.x_range, method_name, x0
            )
        except Exception as e:
            print(f"Warning: Could not automatically find initial points: {e}")
            # Fallback to basic logic if midpoint utility fails
            if method_name in ["bisection", "regula_falsi"]:
                # Try to bracket a root with a simple approach
                try:
                    from utils.midpoint import find_bracket_points

                    a, b = find_bracket_points(config.func, config.x_range)
                    x0, x1 = a, b
                except ValueError:
                    # If that also fails, try expanding around x0
                    if x0 >= 0:
                        a, b = -abs(x0) - 1, x0
                    else:
                        a, b = x0, abs(x0) + 1

                    # Verify interval brackets a root
                    f_a, f_b = config.func(a), config.func(b)
                    if f_a * f_b > 0:  # Same sign, no root bracketed
                        raise ValueError(
                            f"Could not find interval that brackets a root for {method_name}. "
                            f"Try providing explicit initial points that bracket a root."
                        )
                    x0, x1 = a, b
            elif method_name == "secant":
                # For secant, just create a point nearby
                x1 = x0 + 0.1
            else:
                # For optimization methods, create a reasonable interval
                x1 = x0 + 2

    # Create the method based on its type
    if method_name in interval_methods:
        # Methods that need an interval (a, b)
        a, b = min(x0, x1), max(x0, x1)

        # For root-finding methods, verify the interval brackets the root
        if method_name in ["bisection", "regula_falsi"]:
            f_a, f_b = config.func(a), config.func(b)
            if f_a * f_b > 0:  # Same sign, no root bracketed
                raise ValueError(
                    f"Interval [{a}, {b}] does not bracket a root. f({a}) = {f_a}, f({b}) = {f_b}"
                )

        # Create method with interval
        method = method_class(config, a, b)

    elif method_name == "secant":
        # Secant method needs two points
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
    show_animation: bool = True,
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
        show_animation: Whether to generate and display animation

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
        x_range = determine_x_range(
            function_name, x0_values, method_type, specified_range=None
        )

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
        config.derivative = lambda x: 2 * x
    elif function_name == "rosenbrock":
        # Get the function object directly
        func_obj = get_function(function_name)
        # Use its df and d2f methods
        config.derivative = func_obj.df
        config.hessian = func_obj.d2f
    elif function_name == "diagonal_quadratic":
        # Get the function object directly
        func_obj = get_function(function_name)
        # Use its df and d2f methods
        config.derivative = func_obj.df
        config.hessian = func_obj.d2f

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
            show_animation=show_animation,
        )

    return methods, results_df


def visualize_results(
    methods: List[BaseNumericalMethod],
    config: NumericalMethodConfig,
    function_name: str,
    save_viz: Optional[str] = None,
    viz_format: str = "html",
    viz_3d: bool = False,
    method_type: str = "root",
    show_animation: bool = True,
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
        show_animation: Whether to include animation controls in the visualization
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
    interactive_fig = PlotFactory.create_interactive_comparison(
        methods=methods,
        function_space=function_space,
        vis_config=vis_config,
        include_error_plot=True,
        log_scale_error=True,
        surface_plot=vis_config.use_plotly_3d,
        include_animation=show_animation,  # Include animation controls within the main visualization
        animation_data=animation_data if show_animation else None,
    )

    # Show the interactive plot
    interactive_fig.show()

    # Save the visualization if requested
    if save_viz:
        # Create save path
        save_path = Path(save_viz)
        # Save the figure using the file manager utility
        save_visualization(interactive_fig, save_path, viz_format)

        # If animation is enabled and separate animation file is needed
        if show_animation and viz_format == "mp4":
            # Create Plotly animation for video export
            plotly_animation = MethodAnimation(
                function_space=function_space,
                title=f"{method_type.capitalize()} Methods Animation",
                color_palette=vis_config.palette,
            )

            # Create animation figure for saving
            anim_fig = plotly_animation.create_plotly_animation(
                method_paths=animation_data["method_paths"],
                error_data=animation_data["error_data"],
                critical_points=animation_data["critical_points"],
                surface_plot=vis_config.use_plotly_3d,
                duration=vis_config.animation_duration,
                transition_duration=vis_config.animation_transition,
            )

            # Save animation to file without showing it
            save_animation(anim_fig, save_path, viz_format)

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

        # Create matplotlib animation only if show_animation is True
        if show_animation:
            # Create the animation object
            plotly_animation = MethodAnimation(
                function_space=function_space,
                title=f"{method_type.capitalize()} Methods Animation",
                color_palette=vis_config.palette,
            )

            anim = plotly_animation.create_matplotlib_animation(
                method_paths=animation_data["method_paths"],
                error_data=animation_data["error_data"],
                critical_points=animation_data["critical_points"],
            )
            plt.show()


def main():
    """Main function that handles CLI commands and runs the appropriate methods."""
    args = parse_args()

    # Handle list command
    if args.problem_type == "list":
        # Get function categories
        categories = list_function_categories()

        if args.category != "All":
            # Filter by selected category
            if args.category in categories:
                functions_to_show = {args.category: categories[args.category]}
            else:
                print(f"No functions found in category: {args.category}")
                return 0
        else:
            functions_to_show = categories

        # Display functions by category
        print("\nAvailable Functions by Category:")
        print("===============================\n")

        for category, func_names in functions_to_show.items():
            print(f"{category}:")
            for name in sorted(func_names):
                func = get_function(name)
                if args.details:
                    print(f"  - {name}: {func.description}")
                    if func.known_roots:
                        if isinstance(func.known_roots[0], (list, tuple, np.ndarray)):
                            # For multidimensional roots
                            roots_str = ", ".join(str(r) for r in func.known_roots)
                        else:
                            # For scalar roots
                            roots_str = ", ".join(f"{r:.4f}" for r in func.known_roots)
                        print(f"    Known roots/minima: {roots_str}")
                    print(f"    Recommended visualization range: {func.x_range}")
                    print("")
                else:
                    print(f"  - {name}")
            print("")

        return 0

    # Default to root-finding if no problem type specified
    if args.problem_type is None:
        args.problem_type = "root"
        args.function = "quadratic"
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
        show_animation=not args.no_animation,
    )

    return 0


# Use the main function from this module if this script is run directly
if __name__ == "__main__":
    sys.exit(main())
