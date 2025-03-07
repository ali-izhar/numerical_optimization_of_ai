# plot/utils/dimension_utils.py

"""
This module provides utilities for detecting and handling dimensions in
numerical method visualizations.
"""

from typing import Callable, List, Optional, Tuple
import inspect
import numpy as np


def detect_function_dimensions(func: Callable) -> int:
    """
    Detect the dimensionality of a function based on its signature and behavior.

    This function tries to determine if a function accepts a scalar or vector input
    by examining its signature and testing its behavior with sample inputs.

    Args:
        func: The function to analyze

    Returns:
        int: The number of dimensions (1 for scalar input, N for N-dimensional vector input)
    """
    # Try to determine dimensions from signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if not params:
        # No parameters, assume 1D
        return 1

    # Check if function accepts multiple scalar arguments
    if len(params) > 1:
        # Multiple scalar parameters indicate multi-dimensional function
        return len(params)

    # Function has one parameter, check if it's vectorized
    try:
        # Test with scalar input
        scalar_result = func(1.0)

        # Test with 1D vector input
        vector_result = func(np.array([1.0]))

        # Test with 2D vector input
        vector_2d_result = func(np.array([1.0, 2.0]))

        # Analyze results
        if isinstance(vector_2d_result, (float, int)):
            # Function accepts vectors and returns scalar
            return len(vector_2d_result) if hasattr(vector_2d_result, "__len__") else 2
        elif hasattr(vector_2d_result, "__len__") and len(vector_2d_result) == 2:
            # Function seems to return a vector for each input point
            return 2
        else:
            # Function accepts scalar or 1D vector
            return 1
    except (TypeError, ValueError, IndexError):
        # Function doesn't work with vector input, assume 1D
        return 1
    except Exception as e:
        # Any other error, default to 1D
        print(f"Warning: Could not determine function dimensions: {e}")
        return 1


def is_2d_function(func: Callable) -> bool:
    """
    Check if a function is 2D (takes 2D input or returns 2D output).

    Args:
        func: The function to check

    Returns:
        bool: True if the function is 2D, False otherwise
    """
    dims = detect_function_dimensions(func)
    return dims > 1


def get_function_signature(func: Callable) -> List[str]:
    """
    Get the parameter names of a function.

    Args:
        func: The function to analyze

    Returns:
        List[str]: List of parameter names
    """
    sig = inspect.signature(func)
    return list(sig.parameters.keys())


def prepare_grid_data(
    func: Callable,
    x_range: Tuple[float, float],
    y_range: Optional[Tuple[float, float]] = None,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare grid data for 2D function visualization.

    Args:
        func: The function to visualize
        x_range: Range of x values (min, max)
        y_range: Range of y values (min, max), defaults to x_range if None
        num_points: Number of points along each dimension

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: X grid, Y grid, Z values
    """
    if y_range is None:
        y_range = x_range

    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    # Check function signature to determine how to call it
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if len(params) >= 2:
            # Function takes separate x, y arguments
            Z = np.array(
                [
                    [func(xi, yi) for xi, yi in zip(x_row, y_row)]
                    for x_row, y_row in zip(X, Y)
                ]
            )
        else:
            # Function takes array or tuple input
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        # Try passing as array
                        Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
                    except:
                        # Try passing as tuple
                        Z[i, j] = func((X[i, j], Y[i, j]))
    except Exception as e:
        # Fallback method - this might work for more general cases
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = func(X[i, j], Y[i, j])
                except:
                    try:
                        Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
                    except:
                        Z[i, j] = np.nan

    return X, Y, Z
