import numpy as np
import matplotlib.pyplot as plt
import cv2

def dft_naive(x):
    """Compute the DFT of a 1D array x using the naive approach."""
    N = len(x)
    X = np.zeros(N, dtype=complex)  # Output array to store DFT results
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft_naive(X):
    """Compute the Inverse DFT of a 1D array X using the naive approach."""
    N = len(X)
    x = np.zeros(N, dtype=complex)  # Output array to store the inverse DFT results
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N  # Normalize by dividing by N
    return x

def fft_combined(x):
    """
    A combined recursive implementation of the 1D Cooley-Tukey FFT.
    This function assumes the input length is a power of 2.
    """
    N = len(x)
    
    # Base case: If there's only one element, return it as is
    if N == 1:
        return x
    
    # Recursive calls to compute FFT of even and odd parts
    X_even = fft_combined(x[::2])
    X_odd = fft_combined(x[1::2])
    
    # Precompute the twiddle factors
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # Use the first half of the factor for "addition" and the second half for "subtraction" effects
    # This achieves the same as explicit addition and subtraction in the Cooley-Tukey FFT
    X = np.concatenate([
        X_even + factor[:N // 2] * X_odd,    # First half with positive twiddle factors
        X_even + factor[N // 2:] * X_odd     # Second half with "negative" twiddle factors (mirrored phase)
    ])
    
    return X

def dft_2d(matrix):
    """Compute the 2D Discrete Fourier Transform (DFT) of a 2D matrix."""
    # Step 1: Apply the 1D DFT to each row
    rows_transformed = np.zeros_like(matrix, dtype=complex)
    for i in range(matrix.shape[0]):
        rows_transformed[i, :] = fft_combined(matrix[i, :])

    # Step 2: Apply the 1D DFT to each column of the result
    cols_transformed = np.zeros_like(rows_transformed, dtype=complex)
    for j in range(rows_transformed.shape[1]):
        cols_transformed[:, j] = fft_combined(rows_transformed[:, j])

    return cols_transformed


def main():
    return 0
    
if __name__ == '__main__':
    main()