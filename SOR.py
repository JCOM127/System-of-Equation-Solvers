import numpy as np
import pandas as pd

def SOR(A, b, x0, Tol, niter, w):
    """
    Solve a system of equations using the Successive Over-Relaxation (SOR) method using infinite norm to calculate error.

    Parameters:
    - A: 2D numpy array, the coefficient matrix.
    - b: 2D numpy array, the independent vector.
    - x0: 2D numpy array, the initial guess for the solution.
    - Tol: float, the tolerance for the solution.
    - niter: int, the maximum number of iterations.
    - w: float, the relaxation parameter.

    Prints the approximation solution and the intermediate values for the solution.

    Returns:
    None
    """

    n = 0  # Initialize counter
    error = Tol + 1  # Initialize error
    x = x0.copy()
    D = np.diag(np.diag(A))  # Create matrix with diagonal elements of A
    L = -np.tril(A, -1)  # Lower triangular part of A
    U = -np.triu(A, 1)  # Upper triangular part of A

    df = pd.DataFrame(columns=['x' + str(i) for i in range(1, len(x0) + 1)] + ['Error'])

    while error > Tol and n < niter:
        T = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)
        C = w * np.linalg.inv(D - w * L) @ b
        x1 = T @ x + C
        error = np.linalg.norm(x1 - x, np.inf)
        itemsol = [item for sublist in x1.tolist() for item in sublist]
        itemsol.append(error)
        df.loc[n] = itemsol
        n += 1
        x = x1.copy()

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

SOR(A, b, x0, 0.5e-5, 100, 1)
