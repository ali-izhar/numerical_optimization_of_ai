"""Newton-Conjugate Gradient method implementation"""

import numpy as np
from typing import Callable, Tuple, Optional


def hessian_vector_product(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    h: float = 1e-7,
) -> np.ndarray:
    """
    Approximate Hessian-vector product using finite difference (formula 7.10)
    ∇²f(x)d ≈ (∇f(x+hd) - ∇f(x))/h

    Parameters:
    -----------
    grad_f : Callable
        Gradient function ∇f(x)
    x : np.ndarray
        Current point
    d : np.ndarray
        Direction vector
    h : float
        Step size for finite difference

    Returns:
    --------
    np.ndarray
        Approximation of Hessian-vector product ∇²f(x)d
    """
    return (grad_f(x + h * d) - grad_f(x)) / h


def conjugate_gradient_solver(
    A_op: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Conjugate Gradient method to solve linear system Ax = b
    where A_op is an operator that computes A*x

    Parameters:
    -----------
    A_op : Callable
        Function that computes the matrix-vector product Ax
    b : np.ndarray
        Right-hand side vector
    x0 : np.ndarray, optional
        Initial guess (default: zeros)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    np.ndarray
        Approximate solution to Ax = b
    """
    n = len(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    # Initial residual r = b - Ax
    r = b - A_op(x)
    p = r.copy()

    # Initialize values
    rsold = np.dot(r, r)

    for i in range(max_iter):
        # Matrix-vector product
        Ap = A_op(p)

        # Step size
        alpha = rsold / np.dot(p, Ap)

        # Update solution and residual
        x += alpha * p
        r -= alpha * Ap

        # Check convergence
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break

        # Update direction
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x


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
    Line search to find step size that satisfies strong Wolfe conditions

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
        Parameter for Armijo condition
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


def newton_cg(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    h: float = 1e-7,
    cg_tol: float = 1e-5,
    cg_max_iter: int = 100,
    tol: float = 1e-6,
    max_iter: int = 100,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
) -> Tuple[np.ndarray, float, int, bool]:
    """
    Inexact Newton-Conjugate Gradient method using finite difference for Hessian-vector products

    Parameters:
    -----------
    f : Callable
        Objective function f(x) -> scalar
    grad_f : Callable
        Gradient function ∇f(x) -> vector
    x0 : np.ndarray
        Starting point
    h : float
        Step size for finite difference approximation
    cg_tol : float
        Tolerance for CG inner loop
    cg_max_iter : int
        Maximum iterations for CG inner loop
    tol : float
        Convergence tolerance on gradient norm
    max_iter : int
        Maximum number of iterations
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
    xk = np.copy(x0)
    n = len(x0)

    for k in range(max_iter):
        # Compute gradient
        gk = grad_f(xk)
        grad_norm = np.linalg.norm(gk)

        # Check convergence
        if grad_norm < tol:
            if callback is not None:
                callback(xk, k)
            return xk, f(xk), k, True

        # Call the callback if provided
        if callback is not None:
            callback(xk, k)

        # Define Hessian-vector product function using finite difference
        def hvp(d):
            return hessian_vector_product(grad_f, xk, d, h)

        # Solve the Newton system approximately using CG
        # We're solving H_k p_k = -g_k, so b = -gk
        pk = conjugate_gradient_solver(hvp, -gk, None, cg_tol, cg_max_iter)

        # Ensure descent direction
        if np.dot(gk, pk) > 0:
            pk = -gk  # Fall back to steepest descent if not a descent direction

        # Line search to find step size
        alpha_k = wolfe_line_search(f, grad_f, xk, pk)

        # Update current point
        xk = xk + alpha_k * pk

    # If reached max_iter, optimization didn't converge
    return xk, f(xk), max_iter, False


def main():
    """
    Test Newton-CG on the extended Rosenbrock function with varying parameters:
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
    n = 100

    # Callback to track convergence
    history = {"f_vals": [], "grad_norms": [], "iterations": []}

    def callback(xk, iteration):
        f_val = rosenbrock(xk, alpha=alpha)
        grad_norm = np.linalg.norm(rosenbrock_grad(xk, alpha=alpha))
        history["f_vals"].append(f_val)
        history["grad_norms"].append(grad_norm)
        history["iterations"].append(iteration)

    # Test with different alpha parameters
    alphas = [1, 10, 100]

    # Store results for comparison
    results = []

    for alpha in alphas:
        print(f"\nTesting with α={alpha}, n={n}")

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
        x_opt, f_opt, iterations, success = newton_cg(
            f,
            grad_f,
            x0,
            h=1e-7,
            cg_tol=1e-5,
            cg_max_iter=n // 2,
            tol=1e-6,
            max_iter=100,
            callback=callback,
        )
        elapsed = time.time() - start_time

        # Calculate error from known solution (all ones)
        x_star = np.ones_like(x0)
        error = np.linalg.norm(x_opt - x_star)

        # Store and display results
        result = {
            "alpha": alpha,
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

    # Plot convergence for different alpha values
    plt.figure(figsize=(12, 10))

    # Plot function values
    plt.subplot(2, 1, 1)
    for result in results:
        plt.semilogy(
            result["history"]["iterations"],
            result["history"]["f_vals"],
            label=f"α={result['alpha']}",
        )
    plt.title(f"Convergence of Newton-CG on Rosenbrock Function (n={n})")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.grid(True)

    # Plot gradient norms
    plt.subplot(2, 1, 2)
    for result in results:
        plt.semilogy(
            result["history"]["iterations"],
            result["history"]["grad_norms"],
            label=f"α={result['alpha']}",
        )
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm (log scale)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"newton_cg_rosenbrock_n{n}.png")
    plt.show()


if __name__ == "__main__":
    main()
