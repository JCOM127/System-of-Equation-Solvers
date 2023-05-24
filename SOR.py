import numpy as np
import pandas as pd

def SOR(A, b, x0, Tol, niter, w):
    """
    Solve a system of equations using the Successive Over-Relaxation (SOR) method.

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
        error = np.linalg.norm(x1 - x)
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

A = np.array([[45, 13, 4, 8], [-5, -28, 4, -14], [9, 15, 63, -7], [2, 3, -8, -42]])
b = np.array([[-25], [82], [75], [-43]])
x0 = np.array([[2], [2], [2], [2]])

SOR(A, b, x0, 1e-6, 100, 1)
