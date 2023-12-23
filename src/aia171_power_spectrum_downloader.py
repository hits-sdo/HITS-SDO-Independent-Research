import glob
import numpy as np
import os
import scipy.stats as stats
from PIL import Image

def power_spectrum_1d(image):

    # Get pixel count
    pixel_count = image.shape[0]

    # Convert into fourier transform
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)

    #Calculate 1D power spectrum
    k_frequencies = np.fft.fftfreq(pixel_count) * pixel_count
    k_frequencies2D = np.meshgrid(k_frequencies, k_frequencies)
    k_norm = np.sqrt(k_frequencies2D[0] ** 2 + k_frequencies2D[1] ** 2)
    k_bins = np.arange(0.5, pixel_count // 2 + 1, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    a_bins, _, _ = stats.binned_statistic(k_norm.flatten(),
                                        (fourier_amplitudes ** 2).flatten(),
                                        statistic = "mean", bins = k_bins)
    a_bins *= np.pi * (k_bins[1:] ** 2 - k_bins[:-1] ** 2)

    return a_bins

def main():
    # Collect all images in AIA171_Miniset_BW and represent them as 1D power specturm numpy arrays
    x = []
    image_paths = glob.glob('./../data/AIA171_Miniset_BW/**/*.jpg', recursive = True)
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        image = image.astype(float) / 255
        pow_spect = power_spectrum_1d(image)
        if pow_spect.any():
            x.append(pow_spect)
    x = np.array(x)   

    # Save the dataset as a CSV file
    if not os.path.exists('./../data'):
        os.makedirs('./../data')
    np.savetxt('./../data/power_spectrum_dataset.csv', x, delimiter = ',')


if __name__ == '__main__':
    main()