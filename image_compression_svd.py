import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

def power_iteration(M, num_simulations: int):
    """
    Performs power iteration to find the dominant eigenvalue and eigenvector of matrix M.

    Parameters:
    M (numpy.ndarray): Symmetric matrix.
    num_simulations (int): Number of iterations.

    Returns:
    eigenvalue (float): Dominant eigenvalue.
    eigenvector (numpy.ndarray): Corresponding eigenvector.
    """
    b_k = np.random.rand(M.shape[1])
    for _ in range(num_simulations):
        # Calculate the matrix-by-vector product M * b_k
        b_k1 = np.dot(M, b_k)
        # Normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    # Rayleigh quotient for eigenvalue
    eigenvalue = np.dot(b_k.T, np.dot(M, b_k))
    return eigenvalue, b_k

def compute_eigenpairs(M, num_values, num_iterations):
    """
    Computes the largest num_values eigenvalues and eigenvectors of matrix M.

    Parameters:
    M (numpy.ndarray): Symmetric matrix.
    num_values (int): Number of eigenvalues and eigenvectors to compute.
    num_iterations (int): Number of iterations for power iteration.

    Returns:
    eigenvalues (list): List of eigenvalues.
    eigenvectors (list): List of eigenvectors.
    """
    eigenvalues = []
    eigenvectors = []
    M_copy = M.copy()

    for _ in range(num_values):
        eigenvalue, eigenvector = power_iteration(M_copy, num_iterations)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        # Deflate the matrix
        M_copy = M_copy - eigenvalue * np.outer(eigenvector, eigenvector)
    
    return eigenvalues, eigenvectors

def SVD_matrix(A, num_values, num_iterations=1000):
    """
    Computes the Singular Value Decomposition of matrix A without using library functions.

    Parameters:
    A (numpy.ndarray): Original matrix.
    num_values (int): Number of singular values and vectors to compute.
    num_iterations (int): Number of iterations for power iteration.

    Returns:
    U (numpy.ndarray): Left singular vectors.
    S (numpy.ndarray): Singular values (as a diagonal matrix).
    Vt (numpy.ndarray): Right singular vectors (transposed).
    """
    # Compute A^T A and A A^T
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues_V, eigenvectors_V = compute_eigenpairs(ATA, num_values, num_iterations)
    singular_values = np.sqrt(np.abs(eigenvalues_V))
    V = np.array(eigenvectors_V).T

    # Compute U
    U = np.zeros((A.shape[0], num_values))
    for i in range(num_values):
        if singular_values[i] > 1e-10:
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(A.shape[0])

    # Construct S
    S = np.diag(singular_values)

    return U, S, V.T

def SVD_image(image_path):
    """
    Reads a gray-scale image, converts it to a matrix, and performs custom SVD.

    Parameters:
    image_path (str): Path to the gray-scale image (.jpg format).

    Returns:
    U (numpy.ndarray): Left singular vectors.
    S (numpy.ndarray): Singular values (as a diagonal matrix).
    Vt (numpy.ndarray): Right singular vectors (transposed).
    A (numpy.ndarray): Original image matrix.
    """
    image = Image.open(image_path).convert('L')
    A = np.asarray(image, dtype=np.float64)

    # Determine number of singular values to compute
    num_values = min(A.shape)  # You can adjust this value for performance
    print(f"Computing {num_values} singular values and vectors...")

    U, S, Vt = SVD_matrix(A, num_values)

    return U, S, Vt, A

def low_rank_approximations(U, S, Vt, rates):
    """
    Computes low-rank approximations of the image matrix based on given rates.

    Parameters:
    U (numpy.ndarray): Left singular vectors.
    S (numpy.ndarray): Singular values (as a diagonal matrix).
    Vt (numpy.ndarray): Right singular vectors (transposed).
    rates (list of float): Rates of singular values' summation for approximation.

    Returns:
    approximations (list of numpy.ndarray): List of approximated image matrices.
    """
    total_sum = np.sum(np.diag(S))
    singular_values = np.diag(S)
    cumulative_sum = np.cumsum(singular_values)
    approximations = []

    for rate in rates:
        # Determine the rank k for the given rate
        k = np.searchsorted(cumulative_sum / total_sum, rate) + 1
        print(f"Rate: {rate}, Rank k: {k}")

        # Truncate U, S, Vt to rank k
        U_k = U[:, :k]
        S_k = S[:k, :k]
        Vt_k = Vt[:k, :]

        # Compute the approximated image matrix
        A_k = np.dot(U_k, np.dot(S_k, Vt_k))
        approximations.append(A_k)
    
    return approximations

def compare_images(original, approximations, rates):
    """
    Displays the original and approximated images side by side for comparison.

    Parameters:
    original (numpy.ndarray): Original image matrix.
    approximations (list of numpy.ndarray): Approximated image matrices.
    rates (list of float): Rates corresponding to the approximations.
    """
    num_images = len(approximations) + 1
    plt.figure(figsize=(15, 5))

    # Display the original image
    plt.subplot(1, num_images, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Display approximated images
    for i, (A_k, rate) in enumerate(zip(approximations, rates), start=2):
        plt.subplot(1, num_images, i)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'Rate: {rate:.2f}')
        plt.axis('off')

    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python image_svd_compression.py <image-path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        U, S, Vt, A = SVD_image(image_path)
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        sys.exit(1)

    # Rates of singular values' summation
    rates = [0.5, 0.7, 0.9, 0.95, 0.99]
    approximations = low_rank_approximations(U, S, Vt, rates)

    compare_images(A, approximations, rates)

if __name__ == '__main__':
    main()