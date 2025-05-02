import numpy as np


def cg(A, b, x0=None, tol=1e-6, max_iter=None):
    """
    Conjugate Gradient method for solving Ax = b where A is symmetric positive definite.

    Parameters:
    -----------
    A : callable or ndarray
        If callable, should take a vector x and return the matrix-vector product Ax.
        If ndarray, represents the coefficient matrix.
    b : ndarray
        The right-hand side vector.
    x0 : ndarray, optional
        Initial guess for the solution. If None, a zero vector is used.
    tol : float, optional
        Tolerance for convergence based on the norm of the residual.
    max_iter : int, optional
        Maximum number of iterations. If None, it's set to the size of b.

    Returns:
    --------
    x : ndarray
        The solution vector.
    residual_norms : list
        The residual norms at each iteration.
    """
    # Make sure b is a numpy array
    b = np.asarray(b)
    n = len(b)

    # Set default values
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    # Initial setup
    x = x0.copy()

    # Handle matrix-vector product based on A's type
    if callable(A):
        matvec = A
    else:
        A = np.asarray(A)
        matvec = lambda v: A @ v

    # Following Algorithm 5.2
    r = matvec(x) - b  # r0 ← Ax0 - b
    p = -r.copy()  # p0 ← -r0
    k = 0

    residual_norms = [np.linalg.norm(r)]

    while np.linalg.norm(r) > tol and k < max_iter:
        Ap = matvec(p)

        # Compute step size αk ← (rk^T rk) / (pk^T A pk)
        r_dot_r = r.dot(r)
        alpha = r_dot_r / p.dot(Ap)

        # Update solution: xk+1 ← xk + αk pk
        x = x + alpha * p

        # Update residual: rk+1 ← rk + αk A pk
        r_next = r + alpha * Ap

        # Compute βk+1 ← (rk+1^T rk+1) / (rk^T rk)
        r_next_dot_r_next = r_next.dot(r_next)
        beta = r_next_dot_r_next / r_dot_r

        # Update direction: pk+1 ← -rk+1 + βk+1 pk
        p = -r_next + beta * p

        # Update residual for next iteration
        r = r_next

        # Track residual norm
        residual_norms.append(np.linalg.norm(r))

        k += 1

    return x, residual_norms


if __name__ == "__main__":
    # Test the CG algorithm with Hilbert matrices of different dimensions
    def hilbert_matrix(n):
        """Generate an n x n Hilbert matrix."""
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)
        return H

    # Test with dimensions n=5, 8, 12, 20
    dimensions = [5, 8, 12, 20]
    tol = 1e-6
    max_iter = 1000  # Set a much higher max_iter to focus on tolerance

    print("Conjugate Gradient method for solving Hilbert systems:")
    print("Hilbert matrix elements: A_{i,j} = 1/(i+j-1)")
    print("Right-hand side: b = (1, 1, ..., 1)^T")
    print("Initial point: x_0 = (0, 0, ..., 0)^T")
    print(f"Tolerance: {tol}")
    print(f"Maximum iterations: {max_iter}")
    print("-" * 50)

    for n in dimensions:
        # Create the Hilbert matrix
        A = hilbert_matrix(n)

        # Create the right-hand side vector b = (1, 1, ..., 1)^T
        b = np.ones(n)

        # Initial point x_0 = (0, 0, ..., 0)^T will be set by default in cg()

        # Solve the system using CG with high max_iter
        solution, residual_norms = cg(A, b, tol=tol, max_iter=max_iter)

        # Report the number of iterations and convergence status
        print(f"Dimension n={n}:")
        print(f"  Number of iterations: {len(residual_norms) - 1}")
        print(f"  Final residual norm: {residual_norms[-1]:.2e}")
        print(f"  Reached tolerance: {'Yes' if residual_norms[-1] <= tol else 'No'}")

        # Verify the solution
        actual_residual_norm = np.linalg.norm(A @ solution - b)
        print(f"  Actual residual norm: {actual_residual_norm:.2e}")
        print("-" * 50)
