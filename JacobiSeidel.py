import numpy as np
import pandas as pd

def MatJacobiSeid(A, b, x0, Tol, niter, method):
    """
Jacobi and Gauss-Seidel iterative methods to find solutions for systems of equations using infinite norm to calculate error.
    Parameters:
    - A: 2D numpy array, the coefficient matrix.
    - b: 2D numpy array, the independent vector.
    - x0: 2D numpy array, the initial guess for the solution.
    - Tol: float, the tolerance for the solution.
    - niter: int, the maximum number of iterations.
    - method: int (0 or 1), 0 for Jacobi method and 1 for Gauss-Seidel method.

    Prints the approximation solution and the intermediate values for the solution.

    Returns:
    DataFrame with all x-values and Error when calculating
    """

    n = 0  # Initialize counter
    error = Tol + 1  # Initialize error
    x = x0.copy()
    diag = np.diag(A)  # Get diagonal
    D = np.diag(diag)  # Create matrix with diagonal
    L = -np.tril(A, -1)  # Lower triangular
    U = -np.triu(A, 1)  # Upper triangular

    df = pd.DataFrame(columns=['x' + str(i) for i in range(1, len(x0) + 1)] + ['Error'])

    while error > Tol and n < niter:
        if method == 0:  # Jacobi
            T = np.linalg.inv(D).dot(L + U)
            C = np.linalg.inv(D).dot(b)
            x1 = T.dot(x) + C
        elif method == 1:  # Gauss-Seidel
            T = np.linalg.inv(D - L).dot(U)
            C = np.linalg.inv(D - L).dot(b)
            x1 = T.dot(x) + C
        else:
            raise ValueError("Invalid method. Use 0 for Jacobi or 1 for Gauss-Seidel.")

        error = np.linalg.norm(x1 - x, np.inf)  # Calculate error
        itemsol = [item for sublist in x.tolist() for item in sublist]  # Convert numpy array to list
        itemsol.append(error)  # Append error
        df.loc[n] = itemsol  # Append to DataFrame
        n += 1
        x = x1.copy()  # Update solution

    if error < Tol:
        print([item for sublist in x.tolist() for item in sublist], "is an approximation with a tolerance of", error)
    else:
        print("Failed within", n, "iterations")

    print("The intermediate values for the solution are:")
    print(df)

A = np.array(([(1/1160)+1, 0, -1/1160, 1/1160],
              [1.225*18.5, -1.225*15.165*0.0292*185+1e-6, 0, 0],
              [1/1160, 0, 1, -1/1160],
              [1, 0, 0, -1-1e-6])) #Define matrix A with syntax = ([a11, a12, a13], [a21, a22, a23], ...)
b = np.array(([300], [5], [185], [133])) #Define matrix b with syntax = ([b11], [b21], [b31])
x0 = np.array(([1], [1], [1], [1])) #Define initial condition with syntax = ([11], [21], [31])

MatJacobiSeid(A, b, x0, 0.5e-5, 100, 0)  # Using Jacobi method

