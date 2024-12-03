import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import os
import time
import argparse

class DiscreteFourierTransform:
    def __init__(self, image):
        self.original_image = image
        self.resized_image = self.pad_image(image)
    
    def pad_image(self, image):
        row, col = image.shape
        
        # Pad the image to the next power of 2
        next_power_of_2 = lambda x: 2**int(np.ceil(np.log2(x)))
        resized_row, resized_col = next_power_of_2(row), next_power_of_2(col)
        
        padded_image = np.zeros((resized_row, resized_col))
        padded_image[:row, :col] = image
        
        return padded_image 
    
    def dft(self, signal):
        N = len(signal)
        X = np.zeros(N, dtype=np.complex128)
        
        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
                
        return X
    
    def idft(self, signal):
        N = len(signal)
        x = np.zeros(N, dtype=np.complex128)
        
        for n in range(N):
            for k in range(N):
                x[n] += signal[k] * np.exp(2j * np.pi * k * n / N)
                
            x[n] /= N
                
        return x
    
    def fft(self, signal, size_threshold=8):
        N = len(signal)
        
        # If the signal is small enough, use DFT
        if N <= size_threshold:
            return self.dft(signal)
        
        # Split the signal into even and odd parts (divide and conquer)
        even = self.fft(signal[0::2])
        odd = self.fft(signal[1::2])
        
        # Precompute the coefficients
        coeff = np.exp(-2j * np.pi * np.arange(N) / N)
        
        # Combine the results using symmetry properties of the DFT
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd])  
    
    def ifft(self, signal, size_threshold=8):
        N = len(signal)
        
        # If the signal is small enough, use IDFT
        if N <= size_threshold:
            return self.idft(signal)
        
        # Split the signal into even and odd parts (divide and conquer)
        even = self.ifft(signal[0::2])
        odd = self.ifft(signal[1::2])
        
        # Precompute the coefficients
        coeff = np.exp(2j * np.pi * np.arange(N) / N)
        
        # Combine the results using symmetry properties of the IDFT
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd]) / 2

    def dft_2d(self, array):
        row_transformed_image = np.apply_along_axis(self.dft, axis=1, arr=array)
        
        col_transformed_image = np.apply_along_axis(self.dft, axis=0, arr=row_transformed_image)
        
        return col_transformed_image

    
    def fft_2d(self):
        row, col = self.resized_image.shape
        
        # Perform 1D FFT on rows
        row_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for i in range(row):
            row_transformed_image[i, :] = self.fft(self.resized_image[i, :])
        
        # Perform 1D FFT on columns of the transformed image
        col_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for j in range(col):
            col_transformed_image[:, j] = self.fft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def ifft_2d(self, transformed_image):
        row, col = transformed_image.shape
        
        # Perform 1D IFFT on rows
        row_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for i in range(row):
            row_transformed_image[i, :] = self.ifft(transformed_image[i, :])
        
        # Perform 1D IFFT on columns of the transformed image
        col_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for j in range(col):
            col_transformed_image[:, j] = self.ifft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def plot_fft(self):
        frequency_domain = self.fft_2d()
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        # Plot the Fourier Transform; take the absolute value to get the magnitude; log-scaled
        axs[1].imshow(np.abs(frequency_domain)[:self.original_image.shape[0], :self.original_image.shape[1]], norm=LogNorm())
        axs[1].set_title("Fourier Transform")
        
        plt.show()
    
    def plot_denoise_high_frequency(self, low_frequency_threshold=0.1):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape
        
        # determine boundaries for the low frequency components
        low_frq_row_start, low_frq_row_end = int(row*low_frequency_threshold), int(row*(1-low_frequency_threshold))
        low_frq_col_start, low_frq_col_end = int(col*low_frequency_threshold), int(col*(1-low_frequency_threshold))
        
        # set the high frequency components to zero, representing a cross-like shape in the center of the frequency domain
        # We are left with the low frequency components in the corners of the frequency domain
        frequency_domain[low_frq_row_start:low_frq_row_end, :] = 0
        frequency_domain[:, low_frq_col_start:low_frq_col_end] = 0
        
        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = row * col
        print(f'Non-zero coefficients: {non_zero_coefficients}')
        print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')
        
        # Perform the inverse Fourier Transform to get the denoised image
        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        # Crop the image to the original size for beter comparison
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        
        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("High-Frequency Denoised Image")
        
        plt.show()
        
    def plot_denoise_low_frequency(self, high_frequency_threshold=0.015):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape
        
        # determine boundaries for the high frequency components
        high_frq_row_start, high_frq_row_end = int(row*high_frequency_threshold), int(row*(1-high_frequency_threshold))
        high_frq_col_start, high_frq_col_end = int(col*high_frequency_threshold), int(col*(1-high_frequency_threshold))
        
        # set the low frequency components to zero, representing the corners of the frequency domain
        # We are left with the high frequency components representing a cross-like shape in the center of the frequency domain
        frequency_domain[:high_frq_row_start, :high_frq_col_start] = 0
        frequency_domain[:high_frq_row_start, high_frq_col_end:] = 0
        frequency_domain[high_frq_row_end:, :high_frq_col_start] = 0
        frequency_domain[high_frq_row_end:, high_frq_col_end:] = 0 
        
        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = row * col
        print(f'Non-zero coefficients: {non_zero_coefficients}')
        print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')
        
        # Perform the inverse Fourier Transform to get the denoised image
        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        
        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("Low-Frequency Denoised Image")
        
        plt.show()
        
    def plot_compression(self):
        frequency_domain = self.fft_2d()
        magnitude = np.abs(frequency_domain)
        row, col = frequency_domain.shape
        compression_levels = [0, 50, 90, 95, 99, 99.9]
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()
        
        total_coefficients = row * col
        
        # Compress the image for different compression levels
        for i, compression_level in enumerate(compression_levels):
            print(f'Compression level: {compression_level}%')
            
            # Threshold the magnitude of the frequency domain
            threshold = np.percentile(magnitude, compression_level)
            mask = magnitude >= threshold
            # Multiply the frequency domain with the mask to set the high frequency components to zero
            frequency_domain_compressed = frequency_domain * mask
            
            non_zero_coefficients = np.count_nonzero(frequency_domain)
            total_coefficients = row * col
            print(f'Non-zero coefficients: {non_zero_coefficients}')
            print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')
            
            # Perform the inverse Fourier Transform to get the compressed image
            compressed_image = self.ifft_2d(frequency_domain_compressed)
            compressed_image = np.real(compressed_image)
            compressed_image = compressed_image[:self.original_image.shape[0], :self.original_image.shape[1]]
            
            # Save the compressed image to determine the file size
            file_name = f'compressed_image_{compression_level}.png'
            cv2.imwrite(file_name, compressed_image)
            file_size = os.path.getsize(file_name)
            print(f"Size of {file_name}: {file_size} bytes")

            axs[i].imshow(compressed_image, cmap='gray')
            axs[i].set_title(f'{compression_level}% Compression')
        
        plt.tight_layout()
        plt.show()
    
    def plot_compression_alternative(self):
        frequency_domain = self.fft_2d()
        magnitude = np.abs(frequency_domain)
        row, col = frequency_domain.shape
        compression_levels = [0, 50, 90, 95, 99, 99.9]

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()

        total_coefficients = row * col

        # Compress the image for different compression levels
        for i, compression_level in enumerate(compression_levels):
            print(f'Compression level: {compression_level}%')

            # Threshold the magnitude of the frequency domain
            high_threshold = np.percentile(magnitude, compression_level)
            high_mask = magnitude >= high_threshold

            low_frequency_threshold = 0.05  # Define a threshold for "low frequencies"
            
            # Determine the boundaries for the low frequency components
            low_frq_row_start, low_frq_row_end = int(row * low_frequency_threshold), int(row * (1 - low_frequency_threshold))
            low_frq_col_start, low_frq_col_end = int(col * low_frequency_threshold), int(col * (1 - low_frequency_threshold))
            low_mask = np.zeros_like(magnitude, dtype=bool)
            low_mask[low_frq_row_start:low_frq_row_end, low_frq_col_start:low_frq_col_end] = True

            # Combine the high and low frequency masks
            combined_mask = high_mask | low_mask
            
            # Multiply the frequency domain with the mask to set the high frequency components to zero
            frequency_domain_compressed = frequency_domain * combined_mask
            
            non_zero_coefficients = np.count_nonzero(frequency_domain)
            total_coefficients = row * col
            print(f'Non-zero coefficients: {non_zero_coefficients}')
            print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')

            # Perform the inverse Fourier Transform to get the compressed image
            compressed_image = self.ifft_2d(frequency_domain_compressed)
            compressed_image = np.real(compressed_image)
            compressed_image = compressed_image[:self.original_image.shape[0], :self.original_image.shape[1]]

            # Save the compressed image to determine the file size
            file_name = f'compressed_image_{compression_level}.png'
            cv2.imwrite(file_name, compressed_image)
            file_size = os.path.getsize(file_name)
            print(f"Size of {file_name}: {file_size} bytes")

            axs[i].imshow(compressed_image, cmap='gray')
            axs[i].set_title(f'{compression_level}% Compression')

        plt.tight_layout()
        plt.show()
        
    def plot_runtime(self):
        sizes = [2 ** i for i in range(5, 10)]
        repetitions = 10  # Number of repetitions to calculate mean and standard deviation
        mean_naive_runtimes = []  # Mean runtimes for naive DFT
        mean_fft_runtimes = []  # Mean runtimes for FFT
        naive_errors = []  # Standard deviation for naive DFT
        fft_errors = []  # Standard deviation for FFT

        print(f"Sizes list: {sizes}")

        for size in sizes:
            print(f"Running for size {size}x{size}")

            array = np.random.rand(size, size)
            naive_time = []
            for _ in range(repetitions):
                print(f"Repetition {_}")
                start = time.time()
                self.dft_2d(array)
                end = time.time()
                naive_time.append(end - start)

            mean_naive_runtimes.append(np.mean(naive_time))
            naive_errors.append(np.std(naive_time))

            # Measure runtimes for FFT
            fft_time = []
            for _ in range(repetitions):
                start = time.time()
                self.fft_2d()
                end = time.time()
                fft_time.append(end - start)

            mean_fft_runtimes.append(np.mean(fft_time))
            fft_errors.append(np.std(fft_time))

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            sizes, mean_naive_runtimes, yerr=2 * np.array(naive_errors),
            fmt='-o', label="Naïve DFT (97% Confidence Interval)", capsize=5
        )
        ax.errorbar(
            sizes, mean_fft_runtimes, yerr=2 * np.array(fft_errors),
            fmt='-o', label="FFT (97% Confidence Interval)", capsize=5
        )
        ax.set_title("Runtime Comparison: Naïve DFT vs. FFT")
        ax.set_xlabel("Array Size (N x N)")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{size}x{size}" for size in sizes])
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
    def validate_fft(self):
        # Custom FFT
        frequency_domain = self.fft_2d()
        # Numpy FFT
        frequency_domain_numpy = np.fft.fft2(self.resized_image)
        
        # Compare the results
        fig, axs = plt.subplots(1, 3)

        # Plot custom FFT log-scaled
        axs[0].imshow(np.abs(frequency_domain), norm = LogNorm())
        axs[0].set_title("Custom FFT")

        # Plot Numpy FFT log-scaled
        axs[1].imshow(np.abs(frequency_domain_numpy), norm = LogNorm())
        axs[1].set_title("Numpy FFT")
        
        # Plot the difference
        difference = frequency_domain - frequency_domain_numpy
        axs[2].imshow(np.abs(difference), cmap='hot')
        axs[2].set_title("Difference")

        plt.show()
        
    def validate_ifft(self):
        # Compute the FFT using NumPy for consistency
        frequency_domain = np.fft.fft2(self.resized_image)

        # Custom IFFT
        custom_image = self.ifft_2d(frequency_domain)
        # Numpy IFFT
        numpy_image = np.fft.ifft2(frequency_domain)

        # Convert to real components and crop to original image size
        custom_image = np.real(custom_image)[:self.original_image.shape[0], :self.original_image.shape[1]]
        numpy_image = np.real(numpy_image)[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        # Calculate the difference
        difference = np.abs(custom_image - numpy_image)

        # Compare the results
        fig, axs = plt.subplots(1, 3)

        # Plot custom IFFT result
        axs[0].imshow(custom_image, cmap='gray')
        axs[0].set_title("Custom IFFT")

        # Plot Numpy IFFT result
        axs[1].imshow(numpy_image, cmap='gray')
        axs[1].set_title("Numpy IFFT Result")

        # Plot the difference
        axs[2].imshow(difference, cmap='hot')
        axs[2].set_title("Difference")

        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, required=False, help='Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime', default=1)
    parser.add_argument('-i', type=str, required=False, help='Filename of the image for the DFT', default='moonlanding.png')
    
    args = parser.parse_args()
    mode = args.m
    filename = args.i
    
    # Read the image in grayscale
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    dft = DiscreteFourierTransform(image)
    
    # Perform requested mode
    if mode == 1:
        dft.plot_fft()
    elif mode == 2:
        dft.plot_denoise_high_frequency()
    elif mode == 3:
        dft.plot_compression()
    elif mode == 4:
        dft.plot_runtime()
    elif mode == 5:
        dft.plot_denoise_low_frequency()
    elif mode == 6:
        dft.plot_compression_alternative()
    elif mode == 7:
        dft.validate_fft()
    elif mode == 8:
        dft.validate_ifft()
    else:
        print('''Invalid mode. Mode: 1 for Fast mode, 2 for High Frequency Denoise, 3 for Compress, 4 for Plot runtime, 
                5 for Low Frequency Denoise, 6 for Alternative Compression, 7 for Fast mode validation, 
                8 for Inverse Fast mode validation''')
        exit()
    
if __name__ == '__main__':
    main()
