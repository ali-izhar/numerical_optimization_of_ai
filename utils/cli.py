#!/usr/bin/env python3

"""
Command-line interface for numerical optimization tools.

This module provides command-line argument parsing and the main entry point
for the numerical optimization utility.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Import needed functions from other modules
from utils.file_manager import load_config_file

# Define functions mapping as in solve.py
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
    opt_parser = subparsers.add_parser("optimize", help="Optimize functions")
    opt_parser.add_argument(
        "-f",
        "--function",
        choices=OPTIMIZATION_FUNCTIONS.keys(),
        default="quadratic",
        help="Function to optimize",
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
        default="backtracking",
        help="Step length method for gradient descent",
    )
    opt_parser.add_argument(
        "--descent-direction",
        choices=DESCENT_DIRECTION_METHODS.keys(),
        default="gradient",
        help="Descent direction method for optimization",
    )
    opt_parser.add_argument(
        "--2d", action="store_true", dest="is_2d", help="Use 2D version of function"
    )

    # Common arguments for both subparsers
    for p in [root_parser, opt_parser]:
        p.add_argument(
            "-x0",
            "--initial-points",
            type=float,
            nargs="+",
            default=[1.0],
            help="Initial points for the method",
        )
        p.add_argument(
            "-t",
            "--tolerance",
            type=float,
            default=1e-6,
            help="Error tolerance",
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
        p.add_argument(
            "--no-viz", action="store_true", help="Disable all visualization"
        )
        p.add_argument(
            "--no-animation",
            action="store_true",
            help="Disable animation generation (static plots only)",
        )
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
    # Import run_methods here to avoid circular imports
    from solve import run_methods

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
        show_animation=not args.no_animation,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
