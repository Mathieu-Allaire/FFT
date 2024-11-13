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
        print("Resized image dimensions:", (resized_row, resized_col))
        
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
    
    def denoise(self, threshold_ratio=0.0009):
        # Perform 2D FFT on the resized image
        fft_data = self.fft_2d()
        
        # Set high frequencies to zero
        magnitude = np.abs(fft_data)
        threshold = np.quantile(magnitude, 1 - threshold_ratio)
        fft_data[magnitude > threshold] = 0

        # Inverse FFT to get the denoised image
        denoised_image = self.ifft_2d(fft_data)
        
        # Take only the real part for visualization
        denoised_image = np.real(denoised_image)
        
        # Resize denoised image back to original dimensions for display
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]

        # OpenCV denoising for comparison
        denoised_cv2 = cv2.fastNlMeansDenoising(self.original_image, None, h=10)

        # Display the original, custom FFT denoised, and OpenCV denoised images side by side
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("Custom FFT Denoised Image")
        axs[1].axis('off')

        axs[2].imshow(denoised_cv2, cmap='gray')
        axs[2].set_title("OpenCV Denoised Image")
        axs[2].axis('off')

        plt.show()


    
    def display_fft(self):
        # Custom FFT result on the resized image
        fft_result = self.fft_2d()
        # Crop the FFT result to the original image dimensions
        magnitude_spectrum_custom = np.abs(fft_result)[:self.original_image.shape[0], :self.original_image.shape[1]]

        # Built-in FFT result using np.fft.fft2 for comparison
        fft_builtin = np.fft.fft2(self.resized_image)
        magnitude_spectrum_builtin = np.abs(fft_builtin)[:self.original_image.shape[0], :self.original_image.shape[1]]

        # Plot the original, custom FFT, and built-in FFT images
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Original image (full size)
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Custom Fourier Transform (log scaled)
        axs[1].imshow(magnitude_spectrum_custom, norm=LogNorm(), cmap='gray')
        axs[1].set_title("Custom Fourier Transform (Log Scale)")
        axs[1].axis('off')

        # Built-in Fourier Transform (log scaled)
        axs[2].imshow(magnitude_spectrum_builtin, norm=LogNorm(), cmap='gray')
        axs[2].set_title("Built-in Fourier Transform (Log Scale)")
        axs[2].axis('off')

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
        dft.denoise()
    elif mode == 3:
        dft.compress()
    elif mode == 4:
        dft.plot_runtime()
    else:
        print('Invalid mode. Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime')
        exit()
    
if __name__ == '__main__':
    main()
