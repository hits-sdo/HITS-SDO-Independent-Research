# Import libraries
import random
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from augmentation_test import read_image
from augmentation_list import AugmentationList
from augmentation import Augmentations

def apply_random_augmentations(image: np.ndarray) -> np.ndarray:
    augmentation_list = AugmentationList('euv')
    augmentation = Augmentations(image, augmentation_list.randomize())
    augmented_image, _ = augmentation.perform_augmentations()
    return augmented_image

def create_power_spectrum(fourier_image: np.ndarray) -> np.ndarray | np.ndarray:
    pixel_count = fourier_image.shape[0]
    fourier_amplitudes = (np.abs(fourier_image) ** 2).flatten()

    # Find k_vals
    k_frequencies = np.fft.fftfreq(pixel_count) * pixel_count
    k_frequencies2D = np.meshgrid(k_frequencies, k_frequencies)
    k_norm = np.sqrt(k_frequencies2D[0] ** 2 + k_frequencies2D[1] ** 2)
    k_bins = np.arange(0.5, pixel_count // 2 + 1, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Create 1D power spectrum
    a_bins, _, _ = stats.binned_statistic(k_norm.flatten(),
                                        fourier_amplitudes,
                                        statistic = "mean", bins = k_bins)
    a_bins *= np.pi * (k_bins[1:] ** 2 - k_bins[:-1] ** 2)

    return k_vals, a_bins

def main():

    # Collect all pickle files and select a random one
    pkl_paths = glob.glob('./sdo_augmentation/data/euv/tiles/*.p')
    pkl_path = random.choice(pkl_paths)

    # Read image and show it
    image = pickle.load(imfile := open(pkl_path, 'rb'))
    image = image.astype(float) / 255
    imfile.close()
    plt.figure("Original Image")
    plt.imshow(image, cmap = 'gray')
    plt.title('Original Image')
    plt.show()

    # Apply set of random augmentations and show
    augmented_image = apply_random_augmentations(image)
    plt.figure("Augmented Image")
    plt.imshow(image, cmap = 'gray')
    plt.title('Augmented Image')
    plt.show()

    # Fast fourier for augmented image and plot it
    fourier_augmented_image = np.fft.fftn(augmented_image)
    fourier_augmented_amplitudes = np.abs(fourier_augmented_image)
    fourier_augmented_phases = np.angle(fourier_augmented_image)
    
    fig, axs = plt.subplots(2, figsize = (5, 6))
    fig.canvas.manager.set_window_title('Fourier Amplitudes and Phases') 
    axs[0].plot(np.log(fourier_augmented_amplitudes.flatten()))
    axs[0].set_title('Amplitudes')
    axs[0].set_ylabel('Amplitude')
    axs[1].plot(np.log(fourier_augmented_phases.flatten()))
    axs[1].set_title('Phases')
    axs[1].set_ylabel('Phase')
    plt.show()

    # Create 1D power spectrum and plot
    k_vals, a_bins = create_power_spectrum(fourier_augmented_image)
    plt.figure("1D Power Spectrum")
    plt.loglog(k_vals, a_bins)
    plt.title('1D Power Spectrum')
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.show()



if __name__ == '__main__':
    main()