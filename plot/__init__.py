# plot/__init__.py

"""
Visualization Package for Numerical Methods.

This package provides tools for visualizing numerical methods, including:
- Function visualization
- Optimization method visualization
- Root-finding method visualization
"""

# Import components
from .components import (
    VisualizationConfig,
    FunctionSpace,
    ConvergencePlot,
    ErrorPlot,
    MethodAnimation,
)

# Import factories
from .factory import VisualizerFactory, VisualizerType, PlotFactory

# Import utilities
from .utils import (
    # Color utilities
    generate_colors,
    create_color_map,
    get_method_colors,
    color_by_value,
    # Dimension utilities
    detect_function_dimensions,
    is_2d_function,
    get_function_signature,
    prepare_grid_data,
    # Data preparation utilities
    extract_iteration_data,
    prepare_method_comparison_data,
    calculate_convergence_rates,
    prepare_animation_data,
    prepare_summary_data,
)

__all__ = [
    # Components
    "VisualizationConfig",
    "FunctionSpace",
    "ConvergencePlot",
    "ErrorPlot",
    "MethodAnimation",
    # Factories
    "VisualizerFactory",
    "VisualizerType",
    "PlotFactory",
    # Utilities
    "generate_colors",
    "create_color_map",
    "get_method_colors",
    "color_by_value",
    "detect_function_dimensions",
    "is_2d_function",
    "get_function_signature",
    "prepare_grid_data",
    "extract_iteration_data",
    "prepare_method_comparison_data",
    "calculate_convergence_rates",
    "prepare_animation_data",
    "prepare_summary_data",
]
