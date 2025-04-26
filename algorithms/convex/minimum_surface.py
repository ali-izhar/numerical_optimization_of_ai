"""Minimum Surface Problem using BFGS"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import time


def minimal_area_surface():
    # domain
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    N = 31  # grid points per direction
    Nv = (N - 2) ** 2  # number of unknowns (interior)
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    h = x[1] - x[0]  # grid spacing

    print(f"Grid: {N}x{N} points (h = {h:.6f})")
    print(f"Number of variables: {Nv}")

    # initialize full surface Z and impose boundary conditions
    Z = np.zeros((N, N))
    Z[0, :] = 0.0  # x = 0
    Z[-1, :] = 0.0  # x = 1
    Z[:, 0] = x * (1.0 - x)  # y = 0
    Z[:, -1] = x * (1.0 - x)  # y = 1

    print("Boundary conditions:")
    print(f"  z(0,y) = z(1,y) = 0 for y ∈ [0,1]")
    print(f"  z(x,0) = z(x,1) = x(1-x) for x ∈ [0,1]")

    # Calculate initial surface area
    X, Y = np.meshgrid(x, y)

    # initial guess (zero) for interior values
    z0 = np.zeros(Nv)

    # Function to calculate surface area with current values
    def calc_total_area(Z_full):
        total = 0.0
        for i in range(N - 1):
            for j in range(N - 1):
                px = (
                    (Z_full[i + 1, j] - Z_full[i, j])
                    + (Z_full[i + 1, j + 1] - Z_full[i, j + 1])
                ) / (2 * h)
                py = (
                    (Z_full[i, j + 1] - Z_full[i, j])
                    + (Z_full[i + 1, j + 1] - Z_full[i + 1, j])
                ) / (2 * h)
                r = 1.0 + px * px + py * py
                total += np.sqrt(r) * h**2
        return total

    # Calculate initial area
    Z_init = Z.copy()
    initial_area = calc_total_area(Z_init)
    print(f"Initial surface area: {initial_area:.8f}")

    # Track function evaluations
    num_evals = [0]

    def objective(v):
        num_evals[0] += 1

        # reshape into interior grid and re‐insert into Z
        V = v.reshape((N - 2, N - 2))
        Z_full = Z.copy()
        Z_full[1:-1, 1:-1] = V

        f = 0.0
        grad = np.zeros_like(Z_full)

        # loop over cells
        for i in range(N - 1):
            for j in range(N - 1):
                px = (
                    (Z_full[i + 1, j] - Z_full[i, j])
                    + (Z_full[i + 1, j + 1] - Z_full[i, j + 1])
                ) / (2 * h)
                py = (
                    (Z_full[i, j + 1] - Z_full[i, j])
                    + (Z_full[i + 1, j + 1] - Z_full[i + 1, j])
                ) / (2 * h)
                r = 1.0 + px * px + py * py

                f += np.sqrt(r)

                # gradient contributions
                d = 1.0 / (2 * h * np.sqrt(r))
                grad[i, j] += d * (-px - py)
                grad[i, j + 1] += d * (-px + py)
                grad[i + 1, j] += d * (px - py)
                grad[i + 1, j + 1] += d * (px + py)

        # scale by cell area and extract interior gradient
        f *= h**2
        grad *= h**2
        grad_i = grad[1:-1, 1:-1].ravel()

        return f, grad_i

    # Progress callback
    iteration = [0]

    def callback(xk):
        iteration[0] += 1
        if iteration[0] % 10 == 0:
            f_val = objective(xk)[0]
            print(f"Iteration {iteration[0]}: f(z) = {f_val:.8f}")

    print("\nStarting BFGS optimization...")
    start_time = time()

    # call scipy.optimize.minimize with BFGS and analytic gradient
    result = minimize(
        lambda v: objective(v)[0],
        z0,
        jac=lambda v: objective(v)[1],
        method="BFGS",
        tol=1e-6,
        callback=callback,
        options={"gtol": 1e-6, "maxiter": 1000},
    )

    end_time = time()

    # Print optimization results
    print("\nOptimization Results:")
    print(f"Converged: {result.success}")
    print(f"Function evaluations: {num_evals[0]}")
    print(f"Iterations: {result.nit}")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Minimum surface area: {result.fun:.8f}")
    if initial_area > 0:
        print(f"Reduction: {(1 - result.fun/initial_area)*100:.2f}%")

    # reconstruct full surface and plot
    z_est = result.x.reshape((N - 2, N - 2))
    Z[1:-1, 1:-1] = z_est

    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", antialiased=True)

    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Height (z)")

    # Add title with optimization results
    title = (
        f"Minimal Surface (Area: {result.fun:.6f})\n"
        f"Grid: {N}×{N}, Iterations: {result.nit}, Evals: {num_evals[0]}"
    )
    ax.set_title(title)

    # Add boundary condition annotations
    ax.text(0, 0.5, 0, "z(0,y)=0", color="red")
    ax.text(1, 0.5, 0, "z(1,y)=0", color="red")
    ax.text(0.5, 0, 0.125, "z(x,0)=x(1-x)", color="red")
    ax.text(0.5, 1, 0.125, "z(x,1)=x(1-x)", color="red")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

    return Z, result


if __name__ == "__main__":
    minimal_area_surface()
