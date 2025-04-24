"""L-BFGS algorithm implementation"""

import numpy as np
from typing import Callable, Tuple, List, Optional


def two_loop_recursion(
    grad: np.ndarray,
    s_history: List[np.ndarray],
    y_history: List[np.ndarray],
    rho_history: List[float],
    H0: np.ndarray,
) -> np.ndarray:
    """
    L-BFGS two-loop recursion

    Parameters:
    -----------
    grad : np.ndarray
        Current gradient vector ∇f_k
    s_history : List[np.ndarray]
        List of s vectors (x_{k+1} - x_k)
    y_history : List[np.ndarray]
        List of y vectors (∇f_{k+1} - ∇f_k)
    rho_history : List[float]
        List of ρ values (1 / (y_i^T s_i))
    H0 : np.ndarray
        Initial Hessian approximation H_k^0

    Returns:
    --------
    np.ndarray
        Direction vector p_k = -H_k∇f_k
    """
    q = np.copy(grad)
    m = len(s_history)
    alphas = np.zeros(m)

    # First loop (backward)
    for i in range(m - 1, -1, -1):
        alphas[i] = rho_history[i] * np.dot(s_history[i], q)
        q = q - alphas[i] * y_history[i]

    # Apply initial Hessian approximation
    r = np.dot(H0, q)

    # Second loop (forward)
    for i in range(m):
        beta = rho_history[i] * np.dot(y_history[i], r)
        r = r + s_history[i] * (alphas[i] - beta)

    return -r  # Return -H_k∇f_k as the search direction


def wolfe_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    xk: np.ndarray,
    pk: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
) -> float:
    """
    Backtracking line search to find step size that satisfies strong Wolfe conditions

    Parameters:
    -----------
    f : Callable
        Objective function
    grad_f : Callable
        Gradient function
    xk : np.ndarray
        Current point
    pk : np.ndarray
        Search direction
    c1 : float
        Parameter for Armijo condition (sufficient decrease)
    c2 : float
        Parameter for curvature condition
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    float
        Step size that satisfies Wolfe conditions
    """
    alpha = 1.0
    alpha_prev = 0.0
    f_prev = f(xk)
    grad_prev = grad_f(xk)
    f_curr = f_prev

    # Initial directional derivative
    dg_init = np.dot(grad_prev, pk)
    if dg_init >= 0:
        return 0.0  # Not a descent direction

    for i in range(max_iter):
        x_curr = xk + alpha * pk
        f_curr = f(x_curr)

        # Check Armijo condition (sufficient decrease)
        if f_curr > f_prev + c1 * alpha * dg_init:
            # Zoom to find alpha that satisfies both conditions
            return _zoom(f, grad_f, xk, pk, alpha_prev, alpha, f_prev, dg_init, c1, c2)

        grad_curr = grad_f(x_curr)
        dg_curr = np.dot(grad_curr, pk)

        # Check curvature condition
        if abs(dg_curr) <= -c2 * dg_init:
            return alpha

        # Check if directional derivative is positive
        if dg_curr >= 0:
            return _zoom(f, grad_f, xk, pk, alpha, alpha_prev, f_prev, dg_init, c1, c2)

        alpha_prev = alpha
        alpha *= 2.0  # Increase step size

    return alpha


def _zoom(
    f: Callable,
    grad_f: Callable,
    xk: np.ndarray,
    pk: np.ndarray,
    alpha_lo: float,
    alpha_hi: float,
    f_k: float,
    dg_init: float,
    c1: float,
    c2: float,
    max_iter: int = 10,
) -> float:
    """Helper function for Wolfe line search"""
    for i in range(max_iter):
        # Bisection
        alpha = 0.5 * (alpha_lo + alpha_hi)
        x_curr = xk + alpha * pk
        f_curr = f(x_curr)

        if f_curr > f_k + c1 * alpha * dg_init:
            alpha_hi = alpha
        else:
            grad_curr = grad_f(x_curr)
            dg_curr = np.dot(grad_curr, pk)

            if abs(dg_curr) <= -c2 * dg_init:
                return alpha

            if dg_curr * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            alpha_lo = alpha

    return alpha


def lbfgs(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    m: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-6,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
) -> Tuple[np.ndarray, float, int, bool]:
    """
    Limited-memory BFGS optimization algorithm (Algorithm 7.5)

    Parameters:
    -----------
    f : Callable
        Objective function f(x) -> scalar
    grad_f : Callable
        Gradient function ∇f(x) -> vector
    x0 : np.ndarray
        Starting point
    m : int
        Number of correction vectors to store
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance on gradient norm
    callback : Callable, optional
        Function called after each iteration with current point and iteration count

    Returns:
    --------
    x : np.ndarray
        Solution vector
    f_val : float
        Final function value
    iter_count : int
        Number of iterations
    success : bool
        Whether optimization succeeded
    """
    n = len(x0)
    xk = np.copy(x0)

    # Storage for s and y vectors
    s_history = []  # s_i = x_{i+1} - x_i
    y_history = []  # y_i = ∇f_{i+1} - ∇f_i
    rho_history = []  # ρ_i = 1 / (y_i^T s_i)

    # Initial function value and gradient
    fk = f(xk)
    gk = grad_f(xk)

    # Identity matrix scaled by some factor for initial Hessian approximation
    k = 0

    for k in range(max_iter):
        # Check convergence
        grad_norm = np.linalg.norm(gk)
        if grad_norm < tol:
            if callback is not None:
                callback(xk, k)
            return xk, fk, k, True

        # Call the callback if provided
        if callback is not None:
            callback(xk, k)

        # Choose initial Hessian approximation H_k^0
        if k > 0 and len(y_history) > 0:
            # Use scaling as suggested in the original L-BFGS paper
            s = s_history[-1]
            y = y_history[-1]
            H0_diag = np.dot(s, y) / np.dot(y, y)
        else:
            H0_diag = 1.0

        H0 = np.eye(n) * H0_diag

        # Compute search direction p_k = -H_k∇f_k
        pk = two_loop_recursion(gk, s_history, y_history, rho_history, H0)

        # Line search to find step size alpha_k that satisfies Wolfe conditions
        alpha_k = wolfe_line_search(f, grad_f, xk, pk)

        # Update current point
        xk_new = xk + alpha_k * pk
        gk_new = grad_f(xk_new)

        # Compute and save s_k and y_k
        sk = xk_new - xk
        yk = gk_new - gk

        # Skip update if curvature condition not satisfied
        ys = np.dot(yk, sk)
        if ys > 1e-10:
            # Discard oldest vector pair if memory limit reached
            if len(s_history) == m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

            # Add new vectors to history
            s_history.append(sk)
            y_history.append(yk)
            rho_history.append(1.0 / ys)

        # Update function value and gradient for next iteration
        xk = xk_new
        fk = f(xk)
        gk = gk_new

    # If we reached max_iter, optimization didn't converge
    return xk, fk, k, False


def main():
    """
    Test L-BFGS on the extended Rosenbrock function with varying parameters:
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

    # Callback to track convergence
    history = {"f_vals": [], "grad_norms": [], "iterations": []}

    def callback(xk, iteration):
        f_val = rosenbrock(xk, alpha=alpha)
        grad_norm = np.linalg.norm(rosenbrock_grad(xk, alpha=alpha))
        history["f_vals"].append(f_val)
        history["grad_norms"].append(grad_norm)
        history["iterations"].append(iteration)

    # Test with different alpha and memory parameters
    alphas = [1, 10, 100]
    memory_params = [3, 5, 10, 20]

    # Store results for comparison
    results = []

    for alpha in alphas:
        for m in memory_params:
            print(f"\nTesting with α={alpha}, m={m}, n={n}")

            # Reset callback history
            history = {"f_vals": [], "grad_norms": [], "iterations": []}

            # Create wrapper functions with fixed alpha
            def f(x):
                return rosenbrock(x, alpha=alpha)

            def grad_f(x):
                return rosenbrock_grad(x, alpha=alpha)

            # Starting point
            x0 = np.full(n, -1.0)

            # Time the optimization
            start_time = time.time()
            x_opt, f_opt, iterations, success = lbfgs(
                f, grad_f, x0, m=m, max_iter=2000, tol=1e-6, callback=callback
            )
            elapsed = time.time() - start_time

            # Calculate error from known solution (all ones)
            x_star = np.ones_like(x0)
            error = np.linalg.norm(x_opt - x_star)

            # Store and display results
            result = {
                "alpha": alpha,
                "m": m,
                "iterations": iterations,
                "f_opt": f_opt,
                "error": error,
                "time": elapsed,
                "success": success,
                "history": history.copy(),
            }
            results.append(result)

            print(f"  Iterations: {iterations}")
            print(f"  Final f(x): {f_opt:.8e}")
            print(f"  Error: {error:.8e}")
            print(f"  Success: {success}")
            print(f"  Time: {elapsed:.4f} seconds")

    # Plot convergence for different memory parameters (for the last alpha value)
    plt.figure(figsize=(12, 10))

    # Plot function values
    plt.subplot(2, 1, 1)
    for result in [r for r in results if r["alpha"] == alphas[-1]]:
        plt.semilogy(
            result["history"]["iterations"],
            result["history"]["f_vals"],
            label=f"m={result['m']}",
        )
    plt.title(f"Convergence of L-BFGS on Rosenbrock Function (α={alphas[-1]}, n={n})")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.grid(True)

    # Plot gradient norms
    plt.subplot(2, 1, 2)
    for result in [r for r in results if r["alpha"] == alphas[-1]]:
        plt.semilogy(
            result["history"]["iterations"],
            result["history"]["grad_norms"],
            label=f"m={result['m']}",
        )
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm (log scale)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"lbfgs_rosenbrock_alpha{alphas[-1]}_n{n}.png")
    plt.show()

    # Compare iterations across different alphas and memory parameters
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        x_values = []
        y_values = []
        for result in [r for r in results if r["alpha"] == alpha]:
            x_values.append(result["m"])
            y_values.append(result["iterations"])
        plt.plot(x_values, y_values, "o-", label=f"α={alpha}")

    plt.title(f"L-BFGS Iterations vs Memory Parameter (n={n})")
    plt.xlabel("Memory Parameter (m)")
    plt.ylabel("Iterations to Converge")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"lbfgs_memory_comparison_n{n}.png")
    plt.show()


if __name__ == "__main__":
    main()
