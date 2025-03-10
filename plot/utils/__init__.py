# plot/utils/__init__.py


from .color_utils import (
    generate_colors,
    create_color_map,
    get_method_colors,
    color_by_value,
)

from .dimension_utils import (
    detect_function_dimensions,
    is_2d_function,
    get_function_signature,
    prepare_grid_data,
)

from .data_preparation import (
    extract_iteration_data,
    prepare_method_comparison_data,
    calculate_convergence_rates,
    prepare_animation_data,
    prepare_summary_data,
)

__all__ = [
    # Color utilities
    "generate_colors",
    "create_color_map",
    "get_method_colors",
    "color_by_value",
    # Dimension utilities
    "detect_function_dimensions",
    "is_2d_function",
    "get_function_signature",
    "prepare_grid_data",
    # Data preparation utilities
    "extract_iteration_data",
    "prepare_method_comparison_data",
    "calculate_convergence_rates",
    "prepare_animation_data",
    "prepare_summary_data",
]
