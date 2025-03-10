#!/usr/bin/env python3

"""Command-line interface for numerical optimization tools."""

import argparse
from pathlib import Path

from utils.funcs import FUNCTION_REGISTRY

# Get all function objects from the registry
AVAILABLE_FUNCTIONS = list(FUNCTION_REGISTRY.keys())

# Create dictionaries of functions for root-finding and optimization
ROOT_FUNCTIONS = {name: func.f for name, func in FUNCTION_REGISTRY.items()}
OPTIMIZATION_FUNCTIONS = {name: func.f for name, func in FUNCTION_REGISTRY.items()}

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
        choices=AVAILABLE_FUNCTIONS,
        default="quadratic",
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
        choices=AVAILABLE_FUNCTIONS,
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

    # List functions subparser
    list_parser = subparsers.add_parser(
        "list", help="List available functions by category"
    )
    list_parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed information for each function",
    )
    list_parser.add_argument(
        "--category",
        choices=["Polynomial", "Transcendental", "Trigonometric", "All"],
        default="All",
        help="Filter functions by category",
    )

    # Common arguments for both root-finding and optimization subparsers
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
