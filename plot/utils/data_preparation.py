# plot/utils/data_preparation.py

"""
This module provides utility functions for preparing data for visualization
of numerical methods.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from algorithms.convex.protocols import BaseNumericalMethod, IterationData

from plot.utils.dimension_utils import is_2d_function


def extract_iteration_data(
    methods: List[BaseNumericalMethod], is_2d: Optional[bool] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract iteration data from numerical methods for visualization.

    Args:
        methods: List of numerical methods to extract data from
        is_2d: Whether the function is 2D (auto-detected if None)

    Returns:
        Dictionary containing extracted data for visualization
    """
    method_data = {}

    for method in methods:
        history = method.get_iteration_history()

        if not history:
            continue

        # Get method name
        method_name = method.name

        # Extract x values (convergence data)
        x_values = [h.x_new for h in history]

        # Extract error values
        error_values = [h.error for h in history]

        # Extract function values
        f_values = [h.f_new for h in history]

        # Determine if function is 2D
        method_is_2d = (
            is_2d
            if is_2d is not None
            else (
                is_2d_function(method.config.func)
                if hasattr(method, "config")
                else False
            )
        )

        # Extract path data
        if method_is_2d:
            # For 2D optimization methods, we need to extract both x and y coordinates
            # This assumes the x_new contains both coordinates as a tuple or array
            try:
                path = [
                    (x[0], x[1]) if hasattr(x, "__getitem__") else (x, 0)
                    for x in x_values
                ]
            except (IndexError, TypeError):
                # Fall back to treating as 1D if extraction fails
                path = [(x, 0) for x in x_values]
        else:
            # For 1D methods, path is just the x values
            path = x_values

        # Extract additional details
        details = [h.details for h in history]

        # Store all data for this method
        method_data[method_name] = {
            "x_values": x_values,
            "error_values": error_values,
            "f_values": f_values,
            "path": path,
            "details": details,
            "iterations": list(range(len(history))),
            "converged": method.has_converged(),
            "final_x": method.get_current_x(),
            "final_error": error_values[-1] if error_values else float("nan"),
            "is_2d": method_is_2d,
        }

    return method_data


def prepare_method_comparison_data(
    methods: List[BaseNumericalMethod],
) -> Dict[str, Dict[str, List]]:
    """
    Prepare data for method comparison visualizations.

    Args:
        methods: List of numerical methods to compare

    Returns:
        Dictionary containing data for convergence and error plots
    """
    comparison_data = {"convergence": {}, "error": {}, "function_values": {}}

    for method in methods:
        history = method.get_iteration_history()

        if not history:
            continue

        # Get method name
        method_name = method.name

        # Extract x values for convergence plot
        comparison_data["convergence"][method_name] = [h.x_new for h in history]

        # Extract error values for error plot
        comparison_data["error"][method_name] = [h.error for h in history]

        # Extract function values
        comparison_data["function_values"][method_name] = [h.f_new for h in history]

    return comparison_data


def calculate_convergence_rates(
    error_data: List[float],
) -> Tuple[List[float], List[int]]:
    """
    Calculate convergence rates from error data.

    The convergence rate is estimated by computing the ratio of consecutive errors.

    Args:
        error_data: List of error values

    Returns:
        Tuple containing list of convergence rates and corresponding iterations
    """
    if len(error_data) < 3:
        return [], []

    rates = []
    iterations = []

    for i in range(1, len(error_data)):
        if error_data[i - 1] == 0 or error_data[i] == 0:
            continue  # Skip division by zero

        rate = error_data[i] / error_data[i - 1]
        if 0 < rate < 1.5:  # Filter out unrealistic rates
            rates.append(rate)
            iterations.append(i)

    return rates, iterations


def prepare_animation_data(
    methods: List[BaseNumericalMethod], is_2d: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Prepare data for method animation.

    Args:
        methods: List of numerical methods to animate
        is_2d: Whether the function is 2D (auto-detected if None)

    Returns:
        Dictionary containing data for animation
    """
    # Extract method data
    method_data = extract_iteration_data(methods, is_2d)

    # Prepare animation data
    animation_data = {
        "method_paths": {},
        "error_data": {},
        "function_values": {},
        "critical_points": [],
    }

    for method_name, data in method_data.items():
        # Add path data
        animation_data["method_paths"][method_name] = data["path"]

        # Add error data
        animation_data["error_data"][method_name] = data["error_values"]

        # Add function values
        animation_data["function_values"][method_name] = data["f_values"]

        # Add critical point if method converged
        if data["converged"]:
            animation_data["critical_points"].append(data["final_x"])

    return animation_data


def prepare_summary_data(
    methods: List[BaseNumericalMethod],
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare summary data for numerical methods.

    Args:
        methods: List of numerical methods

    Returns:
        Dictionary containing summary data for each method
    """
    summary_data = {}

    for method in methods:
        history = method.get_iteration_history()

        if not history:
            continue

        # Get method name
        method_name = method.name

        # Get convergence status
        converged = method.has_converged()

        # Get iteration count
        iteration_count = len(history)

        # Get final error
        final_error = history[-1].error if history else float("nan")

        # Get final value
        final_value = method.get_current_x()

        # Calculate average convergence rate if enough iterations
        error_values = [h.error for h in history]
        if len(error_values) >= 3:
            rates, _ = calculate_convergence_rates(error_values)
            avg_rate = np.mean(rates) if rates else float("nan")
        else:
            avg_rate = float("nan")

        # Store summary data
        summary_data[method_name] = {
            "converged": converged,
            "iterations": iteration_count,
            "final_error": final_error,
            "final_value": final_value,
            "avg_rate": avg_rate,
        }

    return summary_data
