"""
Minimum Surface Problem Visualization

This module provides visualizations and solver for the minimum surface problem,
which aims to find the surface of minimum area that interpolates a prescribed
function on the boundary of a unit square.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def create_grid(q, boundary_func=None):
    """
    Create a (q+1)×(q+1) grid for the minimum surface problem.

    Parameters:
    -----------
    q : int
        Number of intervals along each edge
    boundary_func : callable, optional
        Function that takes (x,y) coordinates and returns z value for the boundary

    Returns:
    --------
    X, Y : 2D arrays
        Meshgrid of x and y coordinates
    Z : 2D array
        Initial heights with boundary conditions set
    mask : 2D boolean array
        True for interior points (variables to optimize)
    """
    # Create grid points
    x = np.linspace(0, 1, q + 1)
    y = np.linspace(0, 1, q + 1)
    X, Y = np.meshgrid(x, y)

    # Initialize Z with zeros
    Z = np.zeros((q + 1, q + 1))

    # Create mask for interior points
    mask = np.ones((q + 1, q + 1), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False

    # Set boundary values
    if boundary_func is None:
        # Example boundary function: z = sin(πx) * sin(πy) on the boundary
        boundary_func = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

    # Top and bottom boundaries
    Z[0, :] = boundary_func(X[0, :], Y[0, :])
    Z[-1, :] = boundary_func(X[-1, :], Y[-1, :])

    # Left and right boundaries
    Z[:, 0] = boundary_func(X[:, 0], Y[:, 0])
    Z[:, -1] = boundary_func(X[:, -1], Y[:, -1])

    return X, Y, Z, mask


def compute_surface_area(Z, q):
    """
    Compute the total surface area using finite difference approximation.

    Parameters:
    -----------
    Z : 2D array
        Heights of the surface at each grid point
    q : int
        Number of intervals along each edge

    Returns:
    --------
    float
        Total surface area
    """
    # Calculate step size
    h = 1.0 / q

    # Initialize area
    area = 0.0

    # Loop over each sub-square
    for i in range(q):
        for j in range(q):
            # Get corner heights
            z_j = Z[i, j]
            z_j1 = Z[i, j + 1]
            z_jq = Z[i + 1, j]
            z_jq1 = Z[i + 1, j + 1]

            # Finite difference approximations of derivatives
            dz_dx = (z_j1 - z_j + z_jq1 - z_jq) / (2 * h)
            dz_dy = (z_jq - z_j + z_jq1 - z_j1) / (2 * h)

            # Area of this sub-square using the formula
            sub_area = h**2 * np.sqrt(1 + dz_dx**2 + dz_dy**2)

            # For visualization: equivalence to the formula in the problem
            # sub_area_alt = (h**2) * np.sqrt(1 + (q**2/2) * ((z_j - z_jq1)**2 + (z_j1 - z_jq)**2))

            area += sub_area

    return area


def objective_function(z_flat, Z_fixed, mask, q):
    """
    Objective function for optimization: total surface area.

    Parameters:
    -----------
    z_flat : 1D array
        Flattened array of variables (interior points)
    Z_fixed : 2D array
        Fixed grid with boundary values
    mask : 2D boolean array
        True for interior points
    q : int
        Number of intervals along each edge

    Returns:
    --------
    float
        Total surface area
    """
    # Create a copy of the fixed grid
    Z = Z_fixed.copy()

    # Fill in the interior points with the variables
    Z[mask] = z_flat

    # Compute the surface area
    return compute_surface_area(Z, q)


def compute_minimum_surface(q, boundary_func=None):
    """
    Compute the minimum surface given boundary conditions.

    Parameters:
    -----------
    q : int
        Number of intervals along each edge
    boundary_func : callable, optional
        Function that takes (x,y) coordinates and returns z value for the boundary

    Returns:
    --------
    X, Y, Z : 2D arrays
        Coordinates and heights of the minimum surface
    """
    # Create grid
    X, Y, Z_fixed, mask = create_grid(q, boundary_func)

    # Initial guess: average of neighboring boundary points
    Z_init = Z_fixed.copy()
    for _ in range(q):  # Iterate to propagate boundary values inward
        Z_new = Z_init.copy()
        for i in range(1, q):
            for j in range(1, q):
                if mask[i, j]:
                    Z_new[i, j] = 0.25 * (
                        Z_init[i - 1, j]
                        + Z_init[i + 1, j]
                        + Z_init[i, j - 1]
                        + Z_init[i, j + 1]
                    )
        Z_init = Z_new.copy()

    # Extract initial values for interior points
    z0 = Z_init[mask]

    # Run optimization
    result = minimize(
        objective_function,
        z0,
        args=(Z_fixed, mask, q),
        method="L-BFGS-B",
        options={"maxiter": 500},
    )

    # Update solution
    Z_solution = Z_fixed.copy()
    Z_solution[mask] = result.x

    return X, Y, Z_solution


def visualize_grid_indexing(q):
    """
    Visualize the grid indexing scheme for the minimum surface problem.

    Parameters:
    -----------
    q : int
        Number of intervals along each edge

    Returns:
    --------
    plotly.graph_objects.Figure
        Figure showing the grid indexing
    """
    # Create grid points
    x = np.linspace(0, 1, q + 1)
    y = np.linspace(0, 1, q + 1)
    X, Y = np.meshgrid(x, y)

    # Create indices
    indices = np.arange(1, (q + 1) ** 2 + 1).reshape(q + 1, q + 1)

    # Create figure
    fig = go.Figure()

    # Add scatter points for grid
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y.flatten(),
            mode="markers",
            marker=dict(
                size=8,
                color="blue",
            ),
            showlegend=False,
        )
    )

    # Add line segments for grid
    for i in range(q + 1):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[y[i]] * len(x),
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x[i]] * len(y),
                y=y,
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False,
            )
        )

    # Add index labels
    for i in range(q + 1):
        for j in range(q + 1):
            is_boundary = i == 0 or i == q or j == 0 or j == q
            fig.add_annotation(
                x=X[i, j],
                y=Y[i, j],
                text=f"x<sub>{indices[i, j]}</sub>",
                showarrow=False,
                font=dict(size=10, color="red" if is_boundary else "black"),
                yshift=10,
            )

    # Highlight a typical sub-square
    i, j = q // 2, q // 2
    sub_x = [X[i, j], X[i, j + 1], X[i + 1, j + 1], X[i + 1, j], X[i, j]]
    sub_y = [Y[i, j], Y[i, j + 1], Y[i + 1, j + 1], Y[i + 1, j], Y[i, j]]

    fig.add_trace(
        go.Scatter(
            x=sub_x,
            y=sub_y,
            mode="lines",
            line=dict(color="red", width=2),
            name="Typical Sub-square",
        )
    )

    # Add annotations for the typical sub-square
    fig.add_annotation(
        x=(X[i, j] + X[i, j + 1] + X[i + 1, j + 1] + X[i + 1, j]) / 4,
        y=(Y[i, j] + Y[i, j + 1] + Y[i + 1, j + 1] + Y[i + 1, j]) / 4,
        text="A<sub>j</sub>",
        showarrow=False,
        font=dict(size=12, color="red"),
    )

    fig.update_layout(
        title=f"Grid Indexing for Minimum Surface Problem (q={q})",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def visualize_subsquare_area(q=2):
    """
    Visualize the area calculation for a single sub-square.

    Parameters:
    -----------
    q : int
        Number of intervals for demonstration

    Returns:
    --------
    plotly.graph_objects.Figure
        Figure showing the sub-square area calculation
    """
    # Create a simple 3x3 grid for demonstration
    x = np.linspace(0, 1, q + 1)
    y = np.linspace(0, 1, q + 1)
    X, Y = np.meshgrid(x, y)

    # Create a simple surface
    Z = np.ones((q + 1, q + 1)) * 0.5
    Z[0, 0] = 0.0
    Z[0, 1] = 0.3
    Z[1, 0] = 0.7
    Z[1, 1] = 1.0

    # Calculate finite differences
    h = 1.0 / q
    dz_dx = (Z[0, 1] - Z[0, 0] + Z[1, 1] - Z[1, 0]) / (2 * h)
    dz_dy = (Z[1, 0] - Z[0, 0] + Z[1, 1] - Z[0, 1]) / (2 * h)

    # Calculate area using the formula
    area = h**2 * np.sqrt(1 + dz_dx**2 + dz_dy**2)

    # Verify the equivalence to the formula in the problem
    diag1 = Z[0, 0] - Z[1, 1]
    diag2 = Z[0, 1] - Z[1, 0]
    area_alt = (h**2) * np.sqrt(1 + (q**2 / 2) * (diag1**2 + diag2**2))

    # Create a 3D surface for this sub-square
    fig = go.Figure(
        data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False)]
    )

    # Add lines for the edges
    for i in range(2):
        for j in range(2):
            fig.add_trace(
                go.Scatter3d(
                    x=[X[i, j]],
                    y=[Y[i, j]],
                    z=[Z[i, j]],
                    mode="markers+text",
                    marker=dict(size=6, color="red"),
                    text=[f"z={Z[i, j]:.1f}"],
                    textposition="top center",
                    showlegend=False,
                )
            )

    # Add diagonal lines to show the formula derivation
    fig.add_trace(
        go.Scatter3d(
            x=[X[0, 0], X[1, 1]],
            y=[Y[0, 0], Y[1, 1]],
            z=[Z[0, 0], Z[1, 1]],
            mode="lines",
            line=dict(color="red", width=5, dash="dash"),
            name=f"Diagonal 1: ({Z[0, 0]:.1f} - {Z[1, 1]:.1f})²",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[X[0, 1], X[1, 0]],
            y=[Y[0, 1], Y[1, 0]],
            z=[Z[0, 1], Z[1, 0]],
            mode="lines",
            line=dict(color="blue", width=5, dash="dash"),
            name=f"Diagonal 2: ({Z[0, 1]:.1f} - {Z[1, 0]:.1f})²",
        )
    )

    # Update layout
    fig.update_layout(
        title=(
            f"Area Calculation for a Sub-square<br>"
            f"Area = {area:.4f} (Using derivatives)<br>"
            f"Area = {area_alt:.4f} (Using diagonal formula)"
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        width=800,
        height=600,
    )

    return fig


def visualize_minimum_surface(q=10, boundary_func=None):
    """
    Visualize the minimum surface problem solution.

    Parameters:
    -----------
    q : int
        Number of intervals along each edge
    boundary_func : callable, optional
        Function that takes (x,y) coordinates and returns z value for the boundary

    Returns:
    --------
    plotly.graph_objects.Figure
        Figure showing the minimum surface
    """
    # Compute minimum surface
    X, Y, Z = compute_minimum_surface(q, boundary_func)

    # Create 3D surface plot
    fig = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Height (z)"),
            )
        ]
    )

    # Add boundary points with different color
    boundary_x = np.concatenate([X[0, :], X[-1, :], X[1:-1, 0], X[1:-1, -1]])
    boundary_y = np.concatenate([Y[0, :], Y[-1, :], Y[1:-1, 0], Y[1:-1, -1]])
    boundary_z = np.concatenate([Z[0, :], Z[-1, :], Z[1:-1, 0], Z[1:-1, -1]])

    fig.add_trace(
        go.Scatter3d(
            x=boundary_x,
            y=boundary_y,
            z=boundary_z,
            mode="markers",
            marker=dict(
                size=4,
                color="red",
            ),
            name="Boundary Points",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Minimum Surface Solution (q={q})",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        width=800,
        height=700,
    )

    return fig


def visualize_finite_difference(q=10):
    """
    Visualize the finite difference approximation used in the problem.

    Returns:
    --------
    plotly.graph_objects.Figure
        Figure showing the finite difference approximation
    """
    # Create a simple example for demonstration
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["Surface Sub-square", "Finite Difference Approximation"],
    )

    # Create a simple 2x2 grid for demonstration
    x = np.linspace(0, 1 / q, 2)
    y = np.linspace(0, 1 / q, 2)
    X, Y = np.meshgrid(x, y)

    # Create a simple surface
    Z = np.array([[0.0, 0.3], [0.7, 1.0]])

    # Add the surface
    fig.add_trace(
        go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False), row=1, col=1
    )

    # Calculate derivatives
    h = 1.0 / q
    dz_dx = (Z[0, 1] - Z[0, 0] + Z[1, 1] - Z[1, 0]) / (2 * h)
    dz_dy = (Z[1, 0] - Z[0, 0] + Z[1, 1] - Z[0, 1]) / (2 * h)

    # Create a tangent plane
    x_plane = np.linspace(0, 1 / q, 10)
    y_plane = np.linspace(0, 1 / q, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)

    # Calculate the tangent plane z = z0 + dz_dx*(x-x0) + dz_dy*(y-y0)
    x0, y0 = 1 / (2 * q), 1 / (2 * q)  # center of the sub-square
    z0 = (Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1]) / 4  # average height
    Z_plane = z0 + dz_dx * (X_plane - x0) + dz_dy * (Y_plane - y0)

    # Add the tangent plane
    fig.add_trace(
        go.Surface(
            z=Z_plane,
            x=X_plane,
            y=Y_plane,
            colorscale="Reds",
            opacity=0.7,
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Add points for the corners
    for i in range(2):
        for j in range(2):
            fig.add_trace(
                go.Scatter3d(
                    x=[X[i, j]],
                    y=[Y[i, j]],
                    z=[Z[i, j]],
                    mode="markers+text",
                    marker=dict(size=6, color="red"),
                    text=[f"z={Z[i, j]:.1f}"],
                    textposition="top center",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[X[i, j]],
                    y=[Y[i, j]],
                    z=[Z[i, j]],
                    mode="markers",
                    marker=dict(size=6, color="red"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    # Add a vector for the gradient
    fig.add_trace(
        go.Scatter3d(
            x=[x0, x0 + 0.05 * dz_dx],
            y=[y0, y0],
            z=[z0, z0],
            mode="lines+text",
            line=dict(color="green", width=5),
            text=["", f"∂z/∂x = {dz_dx:.2f}"],
            textposition="top center",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter3d(
            x=[x0, x0],
            y=[y0, y0 + 0.05 * dz_dy],
            z=[z0, z0],
            mode="lines+text",
            line=dict(color="blue", width=5),
            text=["", f"∂z/∂y = {dz_dy:.2f}"],
            textposition="top center",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Update layout
    for i in range(1, 3):
        fig.update_scenes(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            row=1,
            col=i,
        )

    fig.update_layout(
        title=dict(text="Finite Difference Approximation for Surface Area", x=0.5),
        margin=dict(l=0, r=0, t=70, b=0),
        width=1200,
        height=600,
    )

    return fig


def verify_area_formula():
    """
    Verify the equivalence of the two formulas for the surface area.

    Returns:
    --------
    matplotlib.figure.Figure
        Figure showing the verification
    """
    # Create a range of values for demonstration
    q_values = [5, 10, 20, 40]
    delta_values = np.linspace(0, 2, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    for q in q_values:
        h = 1.0 / q

        # Calculate areas using both formulas
        areas_formula1 = []
        areas_formula2 = []

        for delta in delta_values:
            # Create a simple configuration where we vary the diagonals
            z_j = 0
            z_j1 = delta
            z_jq = delta
            z_jq1 = 0

            # Formula 1: using derivatives
            dz_dx = (z_j1 - z_j + z_jq1 - z_jq) / (2 * h)
            dz_dy = (z_jq - z_j + z_jq1 - z_j1) / (2 * h)
            area1 = h**2 * np.sqrt(1 + dz_dx**2 + dz_dy**2)

            # Formula 2: using diagonals
            diag1 = z_j - z_jq1
            diag2 = z_j1 - z_jq
            area2 = h**2 * np.sqrt(1 + (q**2 / 2) * (diag1**2 + diag2**2))

            areas_formula1.append(area1)
            areas_formula2.append(area2)

        # Plot both formulas
        ax.plot(
            delta_values,
            areas_formula1,
            "o-",
            label=f"Derivative (q={q})",
            markersize=3,
            markevery=10,
        )
        ax.plot(
            delta_values,
            areas_formula2,
            "x--",
            label=f"Diagonal (q={q})",
            markersize=5,
            markevery=10,
        )

    ax.set_xlabel("Diagonal Height Difference")
    ax.set_ylabel("Sub-square Surface Area")
    ax.set_title("Equivalence of Surface Area Formulas")
    ax.legend()
    ax.grid(True)

    return fig


def demonstrate_derivation():
    """
    Create a figure demonstrating the derivation of the formula for surface area.

    Returns:
    --------
    plotly.graph_objects.Figure
        Figure showing the derivation
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["3D Sub-square", "Tangent Plane Approximation", "Diagonals"],
    )

    # Create a simple 2x2 grid
    x = np.linspace(0, 1, 2)
    y = np.linspace(0, 1, 2)
    X, Y = np.meshgrid(x, y)

    # Create a surface with varying heights
    Z = np.array([[0.0, 0.5], [0.5, 0.0]])

    # Calculate derivatives and diagonals
    h = 1.0
    dz_dx = (Z[0, 1] - Z[0, 0] + Z[1, 1] - Z[1, 0]) / (2 * h)
    dz_dy = (Z[1, 0] - Z[0, 0] + Z[1, 1] - Z[0, 1]) / (2 * h)

    diag1 = Z[0, 0] - Z[1, 1]  # z_j - z_j+q+1
    diag2 = Z[0, 1] - Z[1, 0]  # z_j+1 - z_j+q

    # Create surface plots
    for col in range(1, 4):
        fig.add_trace(
            go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False),
            row=1,
            col=col,
        )

    # Add diagonal lines
    fig.add_trace(
        go.Scatter3d(
            x=[X[0, 0], X[1, 1]],
            y=[Y[0, 0], Y[1, 1]],
            z=[Z[0, 0], Z[1, 1]],
            mode="lines",
            line=dict(color="red", width=5),
            name=f"Diag1: {diag1:.1f}",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter3d(
            x=[X[0, 1], X[1, 0]],
            y=[Y[0, 1], Y[1, 0]],
            z=[Z[0, 1], Z[1, 0]],
            mode="lines",
            line=dict(color="blue", width=5),
            name=f"Diag2: {diag2:.1f}",
        ),
        row=1,
        col=3,
    )

    # Add tangent plane
    x_plane = np.linspace(0, 1, 10)
    y_plane = np.linspace(0, 1, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)

    x0, y0 = 0.5, 0.5  # center of the square
    z0 = 0.25  # average height
    Z_plane = z0 + dz_dx * (X_plane - x0) + dz_dy * (Y_plane - y0)

    fig.add_trace(
        go.Surface(
            z=Z_plane,
            x=X_plane,
            y=Y_plane,
            colorscale="Reds",
            opacity=0.7,
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Add equations as annotations for the second subplot
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=f"∂z/∂x ≈ {dz_dx:.2f}<br>∂z/∂y ≈ {dz_dy:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-100,
        xanchor="center",
        yanchor="top",
    )

    # Update layout
    for i in range(1, 4):
        fig.update_scenes(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            row=1,
            col=i,
        )

    fig.update_layout(
        title=dict(
            text=(
                "Derivation of the Surface Area Formula<br>"
                f"Area = (1/q²)·√[1 + (q²/2)·((z<sub>j</sub>-z<sub>j+q+1</sub>)² + (z<sub>j+1</sub>-z<sub>j+q</sub>)²)]<br>"
                f"     = (1/q²)·√[1 + (q²/2)·({diag1:.1f}² + {diag2:.1f}²)]"
            ),
            x=0.5,
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        width=1200,
        height=600,
    )

    return fig


def save_all_figures(output_dir="plots"):
    """
    Save all the visualization figures to files.

    Parameters:
    -----------
    output_dir : str
        Directory to save the figures
    """
    import os

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create and save all figures
    grid_fig = visualize_grid_indexing(q=4)
    grid_fig.write_html(os.path.join(output_dir, "grid_indexing.html"))

    subsquare_fig = visualize_subsquare_area()
    subsquare_fig.write_html(os.path.join(output_dir, "subsquare_area.html"))

    finite_diff_fig = visualize_finite_difference()
    finite_diff_fig.write_html(os.path.join(output_dir, "finite_difference.html"))

    derivation_fig = demonstrate_derivation()
    derivation_fig.write_html(os.path.join(output_dir, "derivation.html"))

    # Verification figure (matplotlib)
    verification_fig = verify_area_formula()
    verification_fig.savefig(
        os.path.join(output_dir, "formula_verification.png"), dpi=300
    )

    # Minimum surface figures for different boundary conditions

    # Standard sine function boundary
    def boundary_sine(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    # Tent-like boundary
    def boundary_tent(x, y):
        return 1.0 - 2.0 * np.maximum(np.abs(x - 0.5), np.abs(y - 0.5))

    # Saddle boundary
    def boundary_saddle(x, y):
        return 0.5 * (x**2 - y**2)

    min_surface_fig1 = visualize_minimum_surface(q=20, boundary_func=boundary_sine)
    min_surface_fig1.write_html(os.path.join(output_dir, "min_surface_sine.html"))

    min_surface_fig2 = visualize_minimum_surface(q=20, boundary_func=boundary_tent)
    min_surface_fig2.write_html(os.path.join(output_dir, "min_surface_tent.html"))

    min_surface_fig3 = visualize_minimum_surface(q=20, boundary_func=boundary_saddle)
    min_surface_fig3.write_html(os.path.join(output_dir, "min_surface_saddle.html"))

    print(f"All figures saved to {output_dir} directory")


if __name__ == "__main__":
    # Create directory if it doesn't exist
    import os

    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate and save specific visualizations
    print("Generating grid indexing visualization...")
    grid_fig = visualize_grid_indexing(q=4)
    grid_fig.write_html(os.path.join(output_dir, "grid_indexing.html"))

    print("Generating subsquare area visualization...")
    subsquare_fig = visualize_subsquare_area()
    subsquare_fig.write_html(os.path.join(output_dir, "subsquare_area.html"))

    print("Generating finite difference visualization...")
    finite_diff_fig = visualize_finite_difference()
    finite_diff_fig.write_html(os.path.join(output_dir, "finite_difference.html"))

    print("Generating formula derivation visualization...")
    derivation_fig = demonstrate_derivation()
    derivation_fig.write_html(os.path.join(output_dir, "derivation.html"))

    print("Generating minimum surface visualization...")

    # Standard sine function boundary
    def boundary_sine(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    min_surface_fig = visualize_minimum_surface(q=10, boundary_func=boundary_sine)
    min_surface_fig.write_html(os.path.join(output_dir, "min_surface_sine.html"))

    print(f"All visualizations saved to {output_dir} directory")
