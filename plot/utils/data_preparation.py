# plot/utils/data_preparation.py

"""
This module provides utility functions for preparing data for visualization
of numerical methods, with enhanced support for eye-catching Plotly visualizations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from algorithms.convex.protocols import BaseNumericalMethod, IterationData
import plotly.graph_objects as go
import plotly.express as px
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


# New Plotly-specific functions for enhanced visualizations


def create_plotly_error_comparison(
    method_data: Dict[str, Dict[str, Any]],
    log_scale: bool = True,
    colorscale: str = "D3",
    title: str = "Error Comparison",
    line_width: int = 3,
    dash_sequences: Optional[List[str]] = None,
    show_markers: bool = True,
) -> go.Figure:
    """
    Create a Plotly figure for comparing error convergence between methods.

    Args:
        method_data: Method data from extract_iteration_data
        log_scale: Whether to use logarithmic scale for y-axis
        colorscale: Color palette to use
        title: Plot title
        line_width: Width of plot lines
        dash_sequences: Optional list of dash patterns for lines
        show_markers: Whether to show markers on the lines

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Get a color for each method
    methods = list(method_data.keys())
    if colorscale.lower() == "d3":
        colors = px.colors.qualitative.D3[: len(methods)]
    elif colorscale.lower() == "g10":
        colors = px.colors.qualitative.G10[: len(methods)]
    elif colorscale.lower() == "plotly":
        colors = px.colors.qualitative.Plotly[: len(methods)]
    else:
        colors = px.colors.qualitative.D3[: len(methods)]

    # Default dash sequences if not provided
    if dash_sequences is None:
        dash_sequences = ["solid", "dash", "dot", "dashdot", "longdash"]

    # Add traces for each method
    for i, (method_name, data) in enumerate(method_data.items()):
        error_values = data["error_values"]
        iterations = data["iterations"]

        # Use modulo to cycle through dash patterns
        dash_pattern = dash_sequences[i % len(dash_sequences)]

        # Add trace with styling
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=error_values,
                mode="lines+markers" if show_markers else "lines",
                name=method_name,
                line=dict(width=line_width, color=colors[i], dash=dash_pattern),
                marker=(
                    dict(
                        size=8,
                        symbol="circle",
                        line=dict(width=1, color="white"),
                    )
                    if show_markers
                    else None
                ),
                hovertemplate="Iteration: %{x}<br>Error: %{y:.6e}<extra></extra>",
            )
        )

    # Style the layout
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 24, "family": "Arial, sans-serif"},
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Iteration",
        yaxis_title="Error",
        xaxis=dict(
            title_font=dict(size=18, family="Arial"),
            tickfont=dict(size=14, family="Arial"),
            gridcolor="rgba(200, 200, 200, 0.2)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0, 0, 0, 0.4)",
        ),
        yaxis=dict(
            title_font=dict(size=18, family="Arial"),
            tickfont=dict(size=14, family="Arial"),
            gridcolor="rgba(200, 200, 200, 0.2)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0, 0, 0, 0.4)",
            type="log" if log_scale else "linear",
        ),
        font=dict(family="Arial, sans-serif"),
        legend=dict(
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        hovermode="closest",
        # No hardcoded dimensions - let plotly use browser dimensions
    )

    return fig


def create_plotly_convergence_visualization(
    animation_data: Dict[str, Any],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    colorscale: str = "D3",
    show_critical_points: bool = True,
    show_path_arrows: bool = True,
    arrow_size: float = 10,
) -> go.Figure:
    """
    Create a 2D Plotly visualization of optimization path convergence.

    Args:
        animation_data: Animation data from prepare_animation_data
        x_range: Optional x-axis range
        y_range: Optional y-axis range
        colorscale: Color palette to use
        show_critical_points: Whether to show the critical points
        show_path_arrows: Whether to show arrows along the path
        arrow_size: Size of the arrows

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Get a color for each method
    methods = list(animation_data["method_paths"].keys())
    if colorscale.lower() == "d3":
        colors = px.colors.qualitative.D3[: len(methods)]
    elif colorscale.lower() == "g10":
        colors = px.colors.qualitative.G10[: len(methods)]
    elif colorscale.lower() == "plotly":
        colors = px.colors.qualitative.Plotly[: len(methods)]
    else:
        colors = px.colors.qualitative.D3[: len(methods)]

    # Add traces for each method's path
    for i, (method_name, path) in enumerate(animation_data["method_paths"].items()):
        # Extract x and y coordinates
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]

        # Add path trace
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines+markers",
                name=method_name,
                line=dict(
                    width=3,
                    color=colors[i],
                ),
                marker=dict(
                    size=8,
                    symbol="circle",
                    color=colors[i],
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<extra>"
                + method_name
                + "</extra>",
            )
        )

        # Add arrows to show direction if requested
        if show_path_arrows and len(x_coords) > 1:
            # Add arrows at regular intervals
            arrow_indices = list(
                range(0, len(x_coords) - 1, max(1, len(x_coords) // 10))
            )
            if len(x_coords) - 1 not in arrow_indices:
                arrow_indices.append(len(x_coords) - 1)

            for j in arrow_indices:
                if j < len(x_coords) - 1:
                    # Calculate arrow direction
                    dx = x_coords[j + 1] - x_coords[j]
                    dy = y_coords[j + 1] - y_coords[j]
                    # Normalize
                    magnitude = (dx**2 + dy**2) ** 0.5
                    if magnitude > 0:
                        dx = dx / magnitude
                        dy = dy / magnitude

                    fig.add_annotation(
                        x=x_coords[j],
                        y=y_coords[j],
                        ax=x_coords[j] - dx * arrow_size,
                        ay=y_coords[j] - dy * arrow_size,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=colors[i],
                        standoff=5,
                    )

    # Add critical points if requested
    if show_critical_points and animation_data["critical_points"]:
        critical_xs = [p[0] for p in animation_data["critical_points"]]
        critical_ys = [p[1] for p in animation_data["critical_points"]]

        fig.add_trace(
            go.Scatter(
                x=critical_xs,
                y=critical_ys,
                mode="markers",
                name="Critical Points",
                marker=dict(
                    size=12,
                    symbol="star",
                    color="gold",
                    line=dict(width=2, color="black"),
                ),
                hovertemplate="Critical Point<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
            )
        )

    # Set axis ranges if provided
    x_axis_range = x_range if x_range is not None else None
    y_axis_range = y_range if y_range is not None else None

    # Style the layout
    fig.update_layout(
        title={
            "text": "Optimization Path Convergence",
            "font": {"size": 24, "family": "Arial, sans-serif"},
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis=dict(
            title="x",
            title_font=dict(size=18, family="Arial"),
            tickfont=dict(size=14, family="Arial"),
            gridcolor="rgba(200, 200, 200, 0.2)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0, 0, 0, 0.4)",
            range=x_axis_range,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="rgba(0, 0, 0, 0.2)",
        ),
        yaxis=dict(
            title="y",
            title_font=dict(size=18, family="Arial"),
            tickfont=dict(size=14, family="Arial"),
            gridcolor="rgba(200, 200, 200, 0.2)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0, 0, 0, 0.4)",
            range=y_axis_range,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="rgba(0, 0, 0, 0.2)",
        ),
        font=dict(family="Arial, sans-serif"),
        legend=dict(
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        template="plotly_white",
        hovermode="closest",
        # No hardcoded dimensions - let plotly use browser dimensions
    )

    return fig


def create_plotly_summary_table(summary_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create a Plotly table visualization for method comparison summary.

    Args:
        summary_data: Summary data from prepare_summary_data

    Returns:
        Plotly figure object with styled table
    """
    # Prepare headers and data for table
    headers = ["Method", "Converged", "Iterations", "Final Error", "Avg. Rate"]
    table_data = []

    for method_name, data in summary_data.items():
        row = [
            method_name,
            "✓" if data["converged"] else "✗",
            data["iterations"],
            f"{data['final_error']:.6e}",
            f"{data['avg_rate']:.4f}" if not np.isnan(data["avg_rate"]) else "N/A",
        ]
        table_data.append(row)

    # Sort by iterations (faster methods first)
    table_data.sort(key=lambda x: x[2])

    # Transpose to get columns for plotly table
    columns = list(zip(*table_data))
    if not columns:  # Handle empty data case
        columns = [[] for _ in range(len(headers))]

    # Create figure
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="rgba(65, 105, 225, 0.8)",
                    font=dict(color="white", size=16, family="Arial"),
                    align="center",
                    height=40,
                ),
                cells=dict(
                    values=columns,
                    fill_color=[
                        "rgba(240, 240, 240, 0.8)",  # Method column
                        [
                            (
                                "rgba(200, 255, 200, 0.8)"
                                if row[1] == "✓"
                                else "rgba(255, 200, 200, 0.8)"
                            )
                            for row in table_data
                        ],  # Converged column with conditional coloring
                        "rgba(240, 240, 240, 0.8)",  # Iterations column
                        "rgba(240, 240, 240, 0.8)",  # Final Error column
                        "rgba(240, 240, 240, 0.8)",  # Avg. Rate column
                    ],
                    font=dict(size=14, family="Arial"),
                    align=["left", "center", "center", "right", "right"],
                    height=30,
                    format=[None, None, None, None, None],
                ),
            )
        ]
    )

    # Style the layout
    fig.update_layout(
        title={
            "text": "Optimization Methods Summary",
            "font": {"size": 24, "family": "Arial, sans-serif"},
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(t=80, b=40, l=20, r=20),
        height=None,  # Let height be determined by content
        template="plotly_white",
        # No hardcoded dimensions - let plotly use browser dimensions
    )

    return fig
