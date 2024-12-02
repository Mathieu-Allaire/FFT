import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
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

        axs[1].imshow(np.abs(frequency_domain), norm=LogNorm())
        axs[1].set_title("Fourier Transform")
        
        plt.show()
    
    def plot_denoise_high_frequency(self, low_frequency_threshold=0.1):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape
        
        low_frq_row_start, low_frq_row_end = int(row*low_frequency_threshold), int(row*(1-low_frequency_threshold))
        low_frq_col_start, low_frq_col_end = int(col*low_frequency_threshold), int(col*(1-low_frequency_threshold))
        
        frequency_domain[low_frq_row_start:low_frq_row_end, :] = 0
        frequency_domain[:, low_frq_col_start:low_frq_col_end] = 0
        
        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = row * col
        print(f'Non-zero coefficients: {non_zero_coefficients}')
        print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')
        
        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        
        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("Low-Frequency Denoised Image")
        
        plt.show()
        
    def plot_denoise_low_frequency(self, high_frequency_threshold=0.015):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape
        
        high_frq_row_start, high_frq_row_end = int(row*high_frequency_threshold), int(row*(1-high_frequency_threshold))
        high_frq_col_start, high_frq_col_end = int(col*high_frequency_threshold), int(col*(1-high_frequency_threshold))
        
        frequency_domain[:high_frq_row_start, :high_frq_col_start] = 0
        frequency_domain[:high_frq_row_start, high_frq_col_end:] = 0
        frequency_domain[high_frq_row_end:, :high_frq_col_start] = 0
        frequency_domain[high_frq_row_end:, high_frq_col_end:] = 0 
        
        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = row * col
        print(f'Non-zero coefficients: {non_zero_coefficients}')
        print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')
        
        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        
        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("High-Frequency Denoised Image")
        
        plt.show()
        
    def plot_denoise_thresholding(self, threshold=0.0009):
        frequency_domain = self.fft_2d()
        row, col = frequency_domain.shape

        magnitude = np.abs(frequency_domain)
        threshold = np.quantile(magnitude, 1 - threshold)
        frequency_domain[magnitude > threshold] = 0

        non_zero_coefficients = np.count_nonzero(frequency_domain)
        total_coefficients = row * col
        print(f'Non-zero coefficients: {non_zero_coefficients}')
        print(f'Fraction of non-zero coefficients: {non_zero_coefficients/total_coefficients}')

        denoised_image = self.ifft_2d(frequency_domain)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title(f"Thresholding Denoised Image")

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

            axs[i].imshow(compressed_image, cmap='gray')
            axs[i].set_title(f'{compression_level}% Compression')
        
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
        dft.plot_denoise_high_frequency()
    elif mode == 3:
        dft.plot_compression()
    elif mode == 4:
        dft.plot_runtime()
    elif mode == 5:
        dft.plot_denoise_low_frequency()
    elif mode == 6:
        dft.plot_denoise_thresholding()
    else:
        print('Invalid mode. Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime')
        exit()
    
if __name__ == '__main__':
    main()