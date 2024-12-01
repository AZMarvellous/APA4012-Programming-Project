import numpy as np


# simply run the script with
# you may change the list here to alter the size of random testing matrix.
TEST_SIZE = [3, 4, 10]

def gauss(A, b):
    """
    This function solves Ax = b using Gaussian elimination without pivoting.

    Parameters:
    matrix A (list of lists or numpy.ndarray): n by n Coefficient matrix.
    vector b (list or numpy.ndarray): vector of size n.

    Returns:
    vector x (numpy.ndarray): Solution vector of size n.

    Raises:
    ValueError: If a zero pivot is encountered.
    """
    # Initialize
    A = A.astype(float)  
    b = b.astype(float)  
    n = len(b)

    # Forward Elimination
    for k in range(n-1):
        if A[k, k] == 0:
            raise ValueError(f"Zero pivot encountered at row {k}.")
        
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

    if A[n-1, n-1] == 0:
        raise ValueError(f"Zero pivot encountered at row {n-1}.")

    # Back Substitution
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_ax = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_ax) / A[i, i]

    return x

def test_gauss():
    for n in TEST_SIZE:
        print(f"\nTesting system of size {n}x{n} with random matrix:")
        A = np.random.rand(n, n)
        b = np.random.rand(n)

        print("Coefficient matrix A:")
        print(A)
        print("Right-hand side vector b:")
        print(b)

        try:
            x = gauss(A.copy(), b.copy())
            print("Solution vector x:")
            print(x)

            # compute residuals
            residual = np.dot(A, x) - b
            print("Residual (Ax - b):")
            print(residual)
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    test_gauss()
