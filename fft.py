import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import argparse

class DiscreteFourierTransform:
    def __init__(self, image):
        self.original_image = image
        self.resized_image = self.resize_image(image)
        
    def resize_image(self, image):
        row, col = image.shape
        resized_row, resized_col = 2**int(np.ceil(np.log2(row))), 2**int(np.ceil(np.log2(col)))
        
        padded_image = np.zeros((resized_row, resized_col), dtype=np.uint8)
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
    
    def plot_compression(self):
        return 0
    
    def plot_denoise(self):
        frequency_domaine = self.fft_2d()

        magnitude = np.abs(frequency_domaine)
        frequency_domaine[magnitude > np.quantile(magnitude, 1 - 0.00099)] = 0

        denoised_image = self.ifft_2d(frequency_domaine)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("FFT Denoised Image")

        plt.show()
    
    def plot_fft(self):
        frequency_domaine = self.fft_2d()
        magnitude = np.abs(frequency_domaine)
        
        fft_image = magnitude[:self.original_image.shape[0], :self.original_image.shape[1]]


        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        # Custom Fourier Transform (log scaled)
        axs[1].imshow(fft_image, norm=LogNorm(), cmap='gray')
        axs[1].set_title("FFT Image")
        
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
        dft.display_fft()
    elif mode == 2:
        dft.plot_denoise()
    elif mode == 3:
        dft.plot_compression()
    elif mode == 4:
        dft.plot_runtime()
    else:
        print('Invalid mode. Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime')
        exit()
    
if __name__ == '__main__':
    main()