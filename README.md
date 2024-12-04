# Discrete Fourier Transform Tool

This Python script processes images using DFT and FFT algorithms. It supports Fourier transforms, denoising, compression, runtime analysis, and validation.

## Usage

Run the script with the following command:

### Arguments:

- **-m mode**: Operation mode (default: 1):
  1. Fourier Transform Visualization
  2. High-Frequency Denoise
  3. Compression
  4. Runtime Analysis
  5. Low-Frequency Denoise
  6. Alternative Compression
  7. FFT Validation
  8. IFFT Validation

- **-i image_file**: Path to the image file (default: `moonlanding.png`).

- Example: python fft.py -m 6 -i moonlanding.png

**Note**: If `moonlanding.png` is missing and no image is provided, the script will not work.

**Note**: Python Version used: 3.12

**Note**: Npm Install (If you do not already have them) the following packages for the script to work:
- numpy
- matplotlib
- opencv
