# Import libraries
import os
import gdown
import zipfile
import glob
import scipy.stats as stats
import numpy as np
import PIL.Image as Image

def power_spectrum_1d(image):
    pixel_count = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)
    k_frequencies = np.fft.fftfreq(pixel_count) * pixel_count
    k_frequencies2D = np.meshgrid(k_frequencies, k_frequencies)
    k_norm = np.sqrt(k_frequencies2D[0]**2 + k_frequencies2D[1]**2)
    k_bins = np.arange(0.5, pixel_count // 2 + 1, 1.)
    a_bins, _, _ = stats.binned_statistic(k_norm.flatten(), (fourier_amplitudes**2).flatten(),
                                          statistic="mean", bins=k_bins)
    a_bins *= np.pi * (k_bins[1:]**2 - k_bins[:-1]**2)
    return a_bins

def main():
    # Define file paths
    url = 'https://drive.google.com/uc?export=download&id=1oK2DczGhBMhvJaLe8sUZUOQQy9WGBW2D'
    output_zip = 'AIA171_Miniset_BW.zip'
    extract_path = 'AIA171_Miniset_BW'
    output_npz = 'aia171_miniset_pow_spect.npz'

    # Check if the final .npz file already exists
    if os.path.exists(output_npz):
        print(f"{output_npz} already exists.")

    # Download the ZIP file if it doesn't exist
    if not os.path.exists(output_zip):
        print(f"Downloading {output_zip}...")
        gdown.download(url, output_zip, quiet=False)
    
    # Extract the ZIP file
    if not os.path.exists(extract_path):
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                print(f"Extracting {output_zip}...")
                zip_ref.extractall()

    # Create 1D Power Spectra
    print("Creating 1D Power Spectra...")
    images = []
    pow_spect = []
    image_paths = glob.glob('AIA171_Miniset_BW/**/*.jpg', recursive=True)

    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        image = image.astype(np.float32) / 255
        x = power_spectrum_1d(image)
        if x.any():
            images.append(image)
            pow_spect.append(x)

    images = np.array(images)
    pow_spect = np.array(pow_spect)
    pow_spect = pow_spect / pow_spect.max(axis=1, keepdims=True)

    # Save the arrays into an .npz file
    np.savez(output_npz, pow_spect=pow_spect, images=images, )

if __name__ == '__main__':
    main()
