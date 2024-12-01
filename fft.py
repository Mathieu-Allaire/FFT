import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import argparse
import os
import time

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
        
        if N <= size_threshold:
            return self.dft(signal)
        
        even = self.fft(signal[0::2])
        odd = self.fft(signal[1::2])
        
        coeff = np.exp(-2j * np.pi * np.arange(N) / N)
        
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd])  
    
    def ifft(self, signal, size_threshold=8):
        N = len(signal)
        
        if N <= size_threshold:
            return self.idft(signal)
        
        even = self.ifft(signal[0::2])
        odd = self.ifft(signal[1::2])
        
        coeff = np.exp(2j * np.pi * np.arange(N) / N)
        
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd])
    
    def fft_2d(self):
        row, col = self.resized_image.shape
        
        row_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for i in range(row):
            row_transformed_image[i, :] = self.fft(self.resized_image[i, :])
        
        col_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for j in range(col):
            col_transformed_image[:, j] = self.fft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def ifft_2d(self, transformed_image):
        row, col = transformed_image.shape
        
        row_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for i in range(row):
            row_transformed_image[i, :] = self.ifft(transformed_image[i, :])
        
        col_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for j in range(col):
            col_transformed_image[:, j] = self.ifft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def plot_fft(self):
        frequency_domain = self.fft_2d()
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(np.abs(frequency_domain)[:self.original_image.shape[0], :self.original_image.shape[1]], norm=LogNorm())
        axs[1].set_title("Fourier Transform")
        
        plt.show()

    
    def plot_denoise(self, low_frequency_threshold=0.1):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape
        
        low_frq_row_start, low_frq_row_end = int(row*low_frequency_threshold), int(row*(1-low_frequency_threshold))
        low_frq_col_start, low_frq_col_end = int(col*low_frequency_threshold), int(col*(1-low_frequency_threshold))
        
        frequency_domain[low_frq_row_start:low_frq_row_end, :] = 0
        frequency_domain[:, low_frq_col_start:low_frq_col_end] = 0

        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = frequency_domain.size
        fraction_non_zero = non_zero_coefficients / total_coefficients

        print(f"Low-frequency denoise: {non_zero_coefficients} non-zero coefficients")
        print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}")

        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        
        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("Denoised Image")
        
        plt.show()

    def plot_denoise_alternative(self, high_frequency_threshold=0.05):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape

        high_frq_row_start, high_frq_row_end = int(row * high_frequency_threshold), int(row * (1 - high_frequency_threshold))
        high_frq_col_start, high_frq_col_end = int(col * high_frequency_threshold), int(col * (1 - high_frequency_threshold))

        frequency_domain[:high_frq_row_start, :] = 0
        frequency_domain[high_frq_row_end:, :] = 0
        frequency_domain[:, :high_frq_col_start] = 0
        frequency_domain[:, high_frq_col_end:] = 0

        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = frequency_domain.size
        fraction_non_zero = non_zero_coefficients / total_coefficients

        print(f"Low-frequency denoise: {non_zero_coefficients} non-zero coefficients")
        print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}")


        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("High-Frequency Denoised Image")

        plt.show()

    def plot_combined_denoise(self, low_frequency_threshold=0.001, high_frequency_threshold=0.1):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape

        if low_frequency_threshold is not None:
            low_frq_row_start, low_frq_row_end = int(row * low_frequency_threshold), int(row * (1 - low_frequency_threshold))
            low_frq_col_start, low_frq_col_end = int(col * low_frequency_threshold), int(col * (1 - low_frequency_threshold))

            frequency_domain[low_frq_row_start:low_frq_row_end, :] = 0
            frequency_domain[:, low_frq_col_start:low_frq_col_end] = 0

        if high_frequency_threshold is not None:
            high_frq_row_start, high_frq_row_end = int(row * high_frequency_threshold), int(row * (1 - high_frequency_threshold))
            high_frq_col_start, high_frq_col_end = int(col * high_frequency_threshold), int(col * (1 - high_frequency_threshold))

            frequency_domain[:high_frq_row_start, :] = 0
            frequency_domain[high_frq_row_end:, :] = 0
            frequency_domain[:, :high_frq_col_start] = 0
            frequency_domain[:, high_frq_col_end:] = 0

        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = frequency_domain.size
        fraction_non_zero = non_zero_coefficients / total_coefficients

        print(f"Low-frequency denoise: {non_zero_coefficients} non-zero coefficients")
        print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}")

        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(denoised_image, cmap='gray')

        # Dynamically set the title based on thresholds
        if low_frequency_threshold and high_frequency_threshold:
            axs[1].set_title("Low + High-Frequency Denoised Image")
        elif low_frequency_threshold:
            axs[1].set_title("Low-Frequency Denoised Image")
        elif high_frequency_threshold:
            axs[1].set_title("High-Frequency Denoised Image")

        plt.show()

    def plot_compression(self):
        frequency_domain = self.fft_2d()
        magnitude = np.abs(frequency_domain)
        row, col = frequency_domain.shape
        compression_levels = [0, 50, 90, 95, 99, 99.9]
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()
        
        total_coefficients = row * col
        
        for i, compression_level in enumerate(compression_levels):
            print(f'Compression level: {compression_level}%')


            threshold = np.percentile(magnitude, compression_level)
            mask = magnitude >= threshold
            frequency_domain_compressed = frequency_domain * mask
            
            compressed_image = self.ifft_2d(frequency_domain_compressed)
            compressed_image = np.real(compressed_image)
            compressed_image = compressed_image[:self.original_image.shape[0], :self.original_image.shape[1]]

            non_zero_coefficients = np.count_nonzero(frequency_domain_compressed)
            fraction_non_zero = non_zero_coefficients / total_coefficients
            print(f"Compression of {compression_level}%' has {non_zero_coefficients} non-zero coefficients")
            print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}")

            file_name = f'compressed_image_{compression_level}.png'
            cv2.imwrite(file_name, compressed_image)
            file_size = os.path.getsize(file_name)
            print(f"Size of the saved compressed image file: {file_size} bytes")


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

        for i, compression_level in enumerate(compression_levels):
            print(f'Compression level: {compression_level}%')

            high_threshold = np.percentile(magnitude, compression_level)
            high_mask = magnitude >= high_threshold

            low_frequency_threshold = 0.05  # Define a threshold for "low frequencies"
            low_frq_row_start, low_frq_row_end = int(row * low_frequency_threshold), int(row * (1 - low_frequency_threshold))
            low_frq_col_start, low_frq_col_end = int(col * low_frequency_threshold), int(col * (1 - low_frequency_threshold))
            low_mask = np.zeros_like(magnitude, dtype=bool)
            low_mask[low_frq_row_start:low_frq_row_end, low_frq_col_start:low_frq_col_end] = True

            combined_mask = high_mask | low_mask
            frequency_domain_compressed = frequency_domain * combined_mask

            compressed_image = self.ifft_2d(frequency_domain_compressed)
            compressed_image = np.real(compressed_image)
            compressed_image = compressed_image[:self.original_image.shape[0], :self.original_image.shape[1]]

            non_zero_coefficients = np.count_nonzero(frequency_domain_compressed)
            fraction_non_zero = non_zero_coefficients / total_coefficients
            print(f"Compression of {compression_level}%' has {non_zero_coefficients} non-zero coefficients")
            print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}")

            file_name = f'compressed_image_filtered{compression_level}.png'
            cv2.imwrite(file_name, compressed_image)
            file_size = os.path.getsize(file_name)
            print(f"Size of the saved compressed image file: {file_size} bytes")

            axs[i].imshow(compressed_image, cmap='gray')
            axs[i].set_title(f'{compression_level}% Compression')

        plt.tight_layout()
        plt.show()

    def plot_runtime(self):
        sizes = [2 ** i for i in range(5, 10)]
        repetitions = 10  # Number of repetitions to calculate mean and standard deviation
        naive_runtimes = []  # Mean runtimes for naive DFT
        fft_runtimes = []  # Mean runtimes for FFT
        naive_errors = []  # Standard deviation for naive DFT
        fft_errors = []  # Standard deviation for FFT

        print(f"Sizes list: {sizes}")

        for size in sizes:
            print(f"Running for size {size}x{size}")


            array = np.random.rand(size, size)


            naive_time = []
            for _ in range(repetitions):
                start = time.time()
                _ = np.apply_along_axis(self.dft, axis=0, arr=array)  # Column-wise DFT
                _ = np.apply_along_axis(self.dft, axis=1, arr=array)  # Row-wise DFT
                naive_time.append(time.time() - start)

            naive_runtimes.append(np.mean(naive_time))
            naive_errors.append(np.std(naive_time))

            # Measure runtimes for FFT
            fft_time = []
            for _ in range(repetitions):
                start = time.time()
                _ = self.fft_2d()
                fft_time.append(time.time() - start)

            fft_runtimes.append(np.mean(fft_time))
            fft_errors.append(np.std(fft_time))

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            sizes, naive_runtimes, yerr=2 * np.array(naive_errors),
            fmt='-o', label="Naïve DFT (97% CI)", capsize=5
        )
        ax.errorbar(
            sizes, fft_runtimes, yerr=2 * np.array(fft_errors),
            fmt='-o', label="FFT (97% CI)", capsize=5
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
        dft.plot_denoise()
    elif mode == 3:
        dft.plot_compression()
    elif mode == 4:
        dft.plot_runtime()
    elif mode == 5:
        dft.plot_denoise_alternative()
    elif mode == 6:
        dft.plot_combined_denoise()
    elif mode == 7:
        dft.plot_compression_alternative()
    else:
        print('Invalid mode. Mode: 1 for Fast mode, 2 for Low Frequency Denoise, 3 for Compress, 4 for Plot runtime, '
              '5 for High Frequency Denoise, 6 For Both High and Low Frequency Denoise, 7 for Compression and Frequency filtering')
        exit()
    
if __name__ == '__main__':
    main()