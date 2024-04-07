import glob
import multiprocessing
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

def process_image(image_path):
    print(f"Processing image: {image_path}")
    try:
        image = PIL.Image.open(image_path).convert('L')
    except OSError as e:
        print(f"Skipping file {image_path}, OSError: {e}")
        return None
    
    image = np.array(image)
    image = image.astype(np.float32) / 255
    pow_spect = power_spectrum_1d(image)
    return pow_spect

def power_spectrum_1d_dataset(image_directory, save_directory, data_stride):
    pow_spect_dataset = []
    image_paths = glob.glob(image_directory + '/**/*.jpg', recursive=True)
    print(f"Found {len(image_paths)} images.")

    if data_stride > 1:
        image_paths = image_paths[::data_stride]
        print(f"Using every {data_stride}th image, resulting in {len(image_paths)} images.")

    # Define a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for parallel processing.")
    pool = multiprocessing.Pool(processes=num_processes)

    try:
        # Map the processing function to image paths using parallel processing
        pow_spect_dataset = pool.map(process_image, image_paths)
    finally:
        pool.close()
        pool.join()

    # Filter out empty results
    pow_spect_dataset = [ps for ps in pow_spect_dataset if ps is not None and ps.any()]
    print(f"Finished processing. Total successful power spectra calculated: {len(pow_spect_dataset)}")

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