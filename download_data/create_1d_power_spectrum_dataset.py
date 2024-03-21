import glob
import numpy as np
import PIL.Image
import scipy.stats as stats

# Calculates 1d Power Spectrum
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

def power_spectrum_1d_dataset(image_directory, save_directory, data_stride):
    pow_spect_dataset = []
    image_paths = glob.glob(image_directory + '/**/*.jpg', recursive = True)

    if data_stride > 1:
        image_paths = image_paths[::data_stride]

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}")
        image = PIL.Image.open(image_path).convert('L')
        image = np.array(image)
        image = image.astype(np.float32) / 255
        pow_spect = power_spectrum_1d(image)
        if pow_spect.any():
            pow_spect_dataset.append(pow_spect)

    print("Saving data...")
    pow_spect_dataset = np.array(pow_spect_dataset)
    np.save(save_directory, pow_spect_dataset)
    print("Data saved successfully!")

def wasserstein(x, y):
    return stats.wasserstein_distance(np.arange(len(x)), np.arange(len(y)), x, y)

if __name__ == "__main__":
    image_directory = './AIA_171_Images'
    save_directory = './1d_power_spectrum_dataset'
    data_stride = 1
    power_spectrum_1d_dataset(image_directory, save_directory, data_stride)