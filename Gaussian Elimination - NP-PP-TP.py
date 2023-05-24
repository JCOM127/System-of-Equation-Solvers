import numpy as np

def gaussian_elimination_no_pivot(A, b):
    """
    Solve a system of linear equations using Gaussian Elimination without pivoting.

    Parameters:
    - A: 2D numpy array, the coefficient matrix of the linear system.
    - b: 1D numpy array, the right-hand side vector of the linear system.

    Returns:
    - x: 1D numpy array, the solution vector of the linear system.
    """
    
    n = len(b)
    
    # Elimination phase
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j,i]/A[i,i]
            A[j,i:] = A[j,i:] - factor*A[i,i:]
            b[j] = b[j] - factor*b[i]
            
    # Back substitution phase
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:]))/A[i,i]
        
    return x
def gaussian_elimination_partial_pivot(A, b):
    """
    Solve a system of linear equations using Gaussian Elimination with partial pivoting.

    Parameters:
    - A: 2D numpy array, the coefficient matrix of the linear system.
    - b: 1D numpy array, the right-hand side vector of the linear system.

    Returns:
    - x: 1D numpy array, the solution vector of the linear system.
    """
    
    n = len(b)
    
    # Elimination phase
    for i in range(n):
        pivot_row = i
        for j in range(i+1, n):
            if abs(A[j,i]) > abs(A[pivot_row,i]):
                pivot_row = j
        if pivot_row != i:
            A[[i,pivot_row]] = A[[pivot_row,i]]
            b[i], b[pivot_row] = b[pivot_row], b[i]
            
        for j in range(i+1, n):
            factor = A[j,i]/A[i,i]
            A[j,i:] = A[j,i:] - factor*A[i,i:]
            b[j] = b[j] - factor*b[i]
            
    # Back substitution phase
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:]))/A[i,i]
        
    return x

def gaussian_elimination_total_pivot(A, b):
    """
    Solve a system of linear equations using Gaussian Elimination with total pivoting.

    Parameters:
    - A: 2D numpy array, the coefficient matrix of the linear system.
    - b: 1D numpy array, the right-hand side vector of the linear system.

    Returns:
    - x: 1D numpy array, the solution vector of the linear system.
    - p: 1D numpy array, the permutation vector representing the pivoting.
    """
    
    n = len(b)
    p = np.arange(n)
    
    # Elimination phase
    for i in range(n):
        pivot = np.unravel_index(np.argmax(np.abs(A[i:,i:])), (n-i,n-i))
        pivot_row = pivot[0] + i
        pivot_col = pivot[1] + i
        
        if pivot_row != i:
            A[[i,pivot_row]] = A[[pivot_row,i]]
            b[i], b[pivot_row] = b[pivot_row], b[i]
        if pivot_col != i:
            A[:,[i,pivot_col]] = A[:,[pivot_col,i]]
            p[[i,pivot_col]] = p[[pivot_col,i]]
            
        for j in range(i+1, n):
            factor = A[j,i]/A[i,i]
            A[j,i:] = A[j,i:] - factor*A[i,i:]
            b[j] = b[j] - factor*b[i]
            
    # Back substitution phase
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:]))/A[i,i]
        
    return x, p
