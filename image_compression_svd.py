import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# !pip install numpy pillow matplotlib

def SVD(image_path):
    """
    Reads a gray-scale image, converts it to a matrix, and performs SVD.

    Parameters:
    image_path (str): Path to the gray-scale image (.jpg format).

    Returns:
    U (numpy.ndarray): Left singular vectors.
    S (numpy.ndarray): Singular values (as a diagonal matrix).
    V (numpy.ndarray): Right singular vectors (transposed).
    A (numpy.ndarray): Original image matrix.
    """


    # Load the image and convert it to gray-scale
    image = Image.open(image_path).convert('L')
    A = np.asarray(image, dtype=np.float64)

    # Perform Singular Value Decomposition
    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

    # Convert singular values to a diagonal matrix
    S = np.diag(singular_values)

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
    import numpy as np

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



def SVD(image_path):
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert('L')
    A = np.asarray(image, dtype=np.float64)

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(singular_values)

    return U, S, Vt, A

def low_rank_approximations(U, S, Vt, rates):
    import numpy as np

    total_sum = np.sum(np.diag(S))
    singular_values = np.diag(S)
    cumulative_sum = np.cumsum(singular_values)
    approximations = []

    for rate in rates:
        k = np.searchsorted(cumulative_sum / total_sum, rate) + 1
        print(f"Rate: {rate}, Rank k: {k}")

        U_k = U[:, :k]
        S_k = S[:k, :k]
        Vt_k = Vt[:k, :]

        A_k = np.dot(U_k, np.dot(S_k, Vt_k))
        approximations.append(A_k)

    return approximations

def compare_images(original, approximations, rates):
    import matplotlib.pyplot as plt

    num_images = len(approximations) + 1
    plt.figure(figsize=(15, 5))

    plt.subplot(1, num_images, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    for i, (A_k, rate) in enumerate(zip(approximations, rates), start=2):
        plt.subplot(1, num_images, i)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'Rate: {rate:.2f}')
        plt.axis('off')

    plt.show()

def main():
    image_path = 'your_image.jpg'  # Replace with your image file
    U, S, Vt, A = SVD(image_path)

    # Rates of singular values' summation
    rates = [0.5, 0.7, 0.9, 0.95, 0.99]
    approximations = low_rank_approximations(U, S, Vt, rates)

    compare_images(A, approximations, rates)

if __name__ == '__main__':
    main()
