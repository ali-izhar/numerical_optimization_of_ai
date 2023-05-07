import numpy as np

__all__ = ['gauss_seidel']

def gauss_seidel(A, b, x0, delta, max_it):
    """
    A program implementing the Gauss-Seidel iteration method to solve
    the linear system Ax=b.

    Input:
        A: square coefficient matrix
        b: right side vector
        x0: initial guess
        delta: error tolerance for the relative difference between
               two consecutive iterates
        max_it: maximum number of iterations to be allowed

    Output:
        x: numerical solution vector
        iflag: 1 if a numerical solution satisfying the error
                 tolerance is found within max_it iterations
               -1 if the program fails to produce a numerical
                  solution in max_it iterations
        itnum: the number of iterations used to compute x
    """

    n = len(b)
    iflag = 1
    k = 0
    x = x0.copy()
    prev_err = np.linalg.norm(x-x0, np.inf) / (np.linalg.norm(x, np.inf) + np.finfo(float).eps)

    while k < max_it:
        k += 1

        # update x(1), the first component of the solution
        x[0] = (b[0] - np.dot(A[0,1:], x[1:])) / A[0,0]
        for i in range(1, n):
            if i < n - 1:
                # Update x(i), the ith component of the solution
                x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
            else:
                # Update x(n), the last component of the solution
                x[n-1] = (b[n-1] - np.dot(A[n-1,:n-1], x[:n-1])) / A[n-1,n-1]

        # compute relative error
        relerr = np.linalg.norm(x-x0, np.inf) / (np.linalg.norm(x, np.inf) + np.finfo(float).eps)
        if relerr < delta:
            break
        elif relerr > prev_err:
            # error has increased, break out of loop and return current solution
            iflag = -1
            break
        prev_err = relerr
        x0 = x.copy()
        
    itnum = k
    if itnum == max_it:
        iflag = -1
    
    if iflag == -1:
        print('Gauss-Seidel failed to converge in %d iterations.' % max_it)
        print('The last computed solution is:', x)
        return x, iflag, itnum
    else:
        print('Gauss-Seidel converged in %d iterations.' % itnum)
        return x, iflag, itnum