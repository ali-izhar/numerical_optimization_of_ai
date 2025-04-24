import numpy as np
from scipy.optimize import line_search


# ----------------------------------------
# BFGS Optimization Algorithm
# ----------------------------------------


def bfgs(func, grad, x0, tol=1e-5, max_iter=1000, track_path=False):
    """
    BFGS optimization algorithm implementation.

    Follows the mathematical formulation:
    1. Start with initial point x₀ and inverse Hessian approximation H₀
    2. While ||∇f_k|| > tol:
       - Compute search direction p_k = -H_k ∇f_k
       - Find step size α_k using line search to satisfy Wolfe conditions
       - Update position x_{k+1} = x_k + α_k p_k
       - Compute s_k = x_{k+1} - x_k and y_k = ∇f_{k+1} - ∇f_k
       - Update inverse Hessian approximation H_{k+1} using BFGS formula

    Parameters:
    -----------
    func : callable
        Objective function to minimize, f(x)
    grad : callable
        Gradient of the objective function, ∇f(x)
    x0 : ndarray
        Initial point x₀
    tol : float, optional
        Convergence tolerance for ||∇f|| > ε
    max_iter : int, optional
        Maximum number of iterations
    track_path : bool, optional
        Whether to track and return the optimization path

    Returns:
    --------
    x : ndarray
        The solution vector
    f_val : float
        The function value at the solution
    n_iter : int
        Number of iterations
    success : bool
        Whether the algorithm converged
    path : list, optional
        List of points visited during optimization (if track_path=True)
    """
    # Initialize
    x = np.asarray(x0).flatten()  # x₀ (starting point)
    n = len(x)

    # Track optimization path if requested
    path = []
    if track_path:
        path.append((x.copy(), func(x)))

    # Initial inverse Hessian approximation (identity matrix) H₀
    H = np.eye(n)

    # Evaluate function and gradient at starting point
    f_val = func(x)
    g = grad(x)  # ∇f₀ (gradient at starting point)

    # Main loop (while ||∇f_k|| > ε)
    k = 0
    while np.linalg.norm(g) > tol and k < max_iter:
        # Compute search direction: p_k = -H_k ∇f_k
        p = -H.dot(g)

        # Line search to satisfy Wolfe conditions to find step size α_k
        alpha, _, _, _, _, _ = line_search(func, grad, x, p)

        # Handle failed line search
        if alpha is None:
            alpha = 0.001  # Use a small step if line search fails

        # Update position: x_{k+1} = x_k + α_k p_k
        x_new = x + alpha * p

        # Compute new gradient ∇f_{k+1}
        g_new = grad(x_new)

        # Define s_k = x_{k+1} - x_k (position difference)
        s = x_new - x

        # Define y_k = ∇f_{k+1} - ∇f_k (gradient difference)
        y = g_new - g

        # Update only if s and y satisfy curvature condition (y_k^T s_k > 0)
        sy = np.dot(s, y)
        if sy > 0:
            # BFGS update formula for inverse Hessian approximation H_{k+1} (formula 6.17)
            # H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
            # where ρ_k = 1 / (y_k^T s_k)
            rho = 1.0 / sy
            v = np.eye(n) - rho * np.outer(s, y)
            H = v.dot(H).dot(v.T) + rho * np.outer(s, s)

        # Update for next iteration
        x = x_new
        g = g_new
        f_val = func(x)

        # Track path if requested
        if track_path:
            path.append((x.copy(), f_val))

        k += 1

    # Check if converged (||∇f_k|| ≤ ε)
    success = np.linalg.norm(g) <= tol

    if track_path:
        return x, f_val, k, success, path
    else:
        return x, f_val, k, success


def wolfe_line_search(func, grad, x, p, c1=1e-4, c2=0.9, alpha_max=1.0, max_iter=25):
    """
    Line search that satisfies strong Wolfe conditions.

    Wolfe conditions:
    1. Armijo (sufficient decrease): f(x_k + α_k p_k) ≤ f(x_k) + c₁ α_k ∇f_k^T p_k
    2. Curvature: |∇f(x_k + α_k p_k)^T p_k| ≤ c₂ |∇f_k^T p_k|

    Parameters:
    -----------
    func : callable
        Objective function
    grad : callable
        Gradient function
    x : ndarray
        Current point x_k
    p : ndarray
        Search direction p_k
    c1 : float
        Parameter for Armijo condition (typically small, 10⁻⁴)
    c2 : float
        Parameter for curvature condition (typically 0.9 for BFGS)
    alpha_max : float
        Maximum step size
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    alpha : float
        Step size α_k that satisfies Wolfe conditions
    """

    def phi(alpha):
        return func(x + alpha * p)

    def dphi(alpha):
        return np.dot(grad(x + alpha * p), p)

    # Initial values
    phi_0 = phi(0)
    dphi_0 = dphi(0)

    alpha_prev = 0
    alpha = alpha_max / 2
    phi_prev = phi_0

    for i in range(max_iter):
        phi_curr = phi(alpha)

        # Check Armijo condition (sufficient decrease)
        if phi_curr > phi_0 + c1 * alpha * dphi_0 or (i > 0 and phi_curr >= phi_prev):
            return zoom(
                alpha_prev, alpha, phi_0, dphi_0, phi_prev, phi_curr, phi, dphi, c1, c2
            )

        dphi_curr = dphi(alpha)

        # Check curvature condition
        if abs(dphi_curr) <= -c2 * dphi_0:
            return alpha

        # If derivative is positive, search in opposite direction
        if dphi_curr >= 0:
            return zoom(
                alpha, alpha_prev, phi_0, dphi_0, phi_curr, phi_prev, phi, dphi, c1, c2
            )

        # Update for next iteration
        alpha_prev = alpha
        alpha = min(alpha * 2, alpha_max)
        phi_prev = phi_curr

    # If we reach here, return the last alpha
    return alpha


def zoom(
    alpha_lo, alpha_hi, phi_0, dphi_0, phi_lo, phi_hi, phi, dphi, c1, c2, max_iter=10
):
    """
    Zoom helper function for Wolfe line search.

    Used to find a step length that satisfies the Wolfe conditions by
    repeatedly refining an interval that contains a suitable step length.
    """
    for i in range(max_iter):
        # Interpolate to find a trial step between alpha_lo and alpha_hi
        alpha = 0.5 * (alpha_lo + alpha_hi)

        phi_trial = phi(alpha)

        # Check Armijo condition
        if phi_trial > phi_0 + c1 * alpha * dphi_0 or phi_trial >= phi_lo:
            alpha_hi = alpha
            phi_hi = phi_trial
        else:
            dphi_trial = dphi(alpha)

            # Check curvature condition
            if abs(dphi_trial) <= -c2 * dphi_0:
                return alpha

            # Update interval based on derivative sign
            if dphi_trial * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                phi_hi = phi_lo

            alpha_lo = alpha
            phi_lo = phi_trial

    # If we reach here, return the best alpha found
    return alpha_lo


# ----------------------------------------
# Example with Rosenbrock Function
# ----------------------------------------


def main():
    """
    Test BFGS on the extended Rosenbrock function with varying parameters:
    f(x) = Σ[α(x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2] for i=1 to n/2
    """
    import time
    import matplotlib.pyplot as plt

    def rosenbrock(x, alpha=100):
        """Extended Rosenbrock function"""
        n = len(x)
        f = 0.0
        for i in range(n // 2):
            f += alpha * (x[2 * i + 1] - x[2 * i] ** 2) ** 2 + (1.0 - x[2 * i]) ** 2
        return f

    def rosenbrock_grad(x, alpha=100):
        """Gradient of extended Rosenbrock function"""
        n = len(x)
        grad = np.zeros_like(x)
        for i in range(n // 2):
            grad[2 * i] = -4.0 * alpha * (x[2 * i + 1] - x[2 * i] ** 2) * x[
                2 * i
            ] - 2.0 * (1.0 - x[2 * i])
            grad[2 * i + 1] = 2.0 * alpha * (x[2 * i + 1] - x[2 * i] ** 2)
        return grad

    # Problem dimensions
    n = 100  # Larger problem size

    # Test with different alpha values
    alphas = [1, 10, 100]

    # Store results for comparison
    results = []

    for alpha in alphas:
        print(f"\nTesting with α={alpha}, n={n}")

        # Create wrapper functions with fixed alpha
        def f(x):
            return rosenbrock(x, alpha=alpha)

        def grad_f(x):
            return rosenbrock_grad(x, alpha=alpha)

        # Starting point
        x0 = np.full(n, -1.0)

        # Track optimization path
        path = []

        # Time the optimization
        start_time = time.time()
        x_opt, f_opt, iterations, success, path = bfgs(
            f, grad_f, x0, tol=1e-6, max_iter=2000, track_path=True
        )
        elapsed = time.time() - start_time

        # Calculate error from known solution (all ones)
        x_star = np.ones_like(x0)
        error = np.linalg.norm(x_opt - x_star)

        # Extract function values and iterations for plotting
        f_vals = [p[1] for p in path]
        iterations_list = list(range(len(path)))

        # Calculate gradient norms
        grad_norms = [np.linalg.norm(grad_f(p[0])) for p in path]

        # Store and display results
        result = {
            "alpha": alpha,
            "iterations": iterations,
            "f_opt": f_opt,
            "error": error,
            "time": elapsed,
            "success": success,
            "f_vals": f_vals,
            "grad_norms": grad_norms,
            "iterations_list": iterations_list,
        }
        results.append(result)

        print(f"  Iterations: {iterations}")
        print(f"  Final f(x): {f_opt:.8e}")
        print(f"  Error: {error:.8e}")
        print(f"  Success: {success}")
        print(f"  Time: {elapsed:.4f} seconds")

    # Plot convergence for different alpha values
    plt.figure(figsize=(12, 10))

    # Plot function values
    plt.subplot(2, 1, 1)
    for result in results:
        plt.semilogy(
            result["iterations_list"],
            result["f_vals"],
            label=f"α={result['alpha']}",
        )
    plt.title(f"Convergence of BFGS on Rosenbrock Function (n={n})")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.grid(True)

    # Plot gradient norms
    plt.subplot(2, 1, 2)
    for result in results:
        plt.semilogy(
            result["iterations_list"],
            result["grad_norms"],
            label=f"α={result['alpha']}",
        )
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm (log scale)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"bfgs_rosenbrock_n{n}.png")
    plt.show()


if __name__ == "__main__":
    main()
