import csv
import glob
import multiprocessing
import numpy as np
import os
import PIL.Image
import scipy.stats as stats
from datetime import datetime

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

def parse_filename(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    date_str = parts[0]
    time_str = parts[1]
    date_time = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    return date_time

def check_for_solar_flare(image_date, flare_data):
    for flare_entry in flare_data:
        start_time = datetime.strptime(flare_entry['start_time'], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(flare_entry['end_time'], '%Y-%m-%d %H:%M:%S')
        if start_time <= image_date <= end_time:
            return 1
    return 0

def process_image(image_path, flare_data, counter, total_images):
    current_count = counter.value
    counter.value += 1
    print(f"Processing image {current_count + 1}/{total_images}: {image_path}")

    try:
        # Parse the datetime from the filename
        image_date = parse_filename(image_path)
        year = image_date.strftime('%Y')
        month = image_date.strftime('%m')
        day = image_date.strftime('%d')
        file_basename = os.path.basename(image_path)
        image_url = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{year}/{month}/{day}/{file_basename}"

        image = PIL.Image.open(image_path).convert('L')
        image = np.array(image).astype(np.float32) / 255
        pow_spect = power_spectrum_1d(image)
        flare_label = check_for_solar_flare(image_date, flare_data)
        return (pow_spect, flare_label, image_url)
    except Exception as e:
        print(f"Skipping file {image_path}, Error: {e}")
        return None

def power_spectrum_1d_dataset(image_directory, flare_csv_file, save_directory, data_stride):
    image_paths = glob.glob(image_directory + '/**/*.jpg', recursive=True)
    if data_stride > 1:
        image_paths = image_paths[::data_stride]
    total_images = len(image_paths)
    flare_data = []
    with open(flare_csv_file, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            flare_data.append(row)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    pool = multiprocessing.Pool(max(multiprocessing.cpu_count() - 1, 1))

    try:
        # Use starmap for clarity with argument passing
        result = pool.starmap(process_image, [(path, flare_data, counter, total_images) for path in image_paths])
    finally:
        pool.close()
        pool.join()

    dataset = [item for item in result if item is not None and np.any(item[0])]

    print("Saving data...")
    if os.path.exists(save_directory):
        with np.load(save_directory, allow_pickle=True) as data:
            existing_data = data['pow_spect']
            existing_labels = data['flare_label']
            existing_urls = data['image_url']
        pow_spect_dataset = np.concatenate((existing_data, [np.array(item[0]) for item in dataset]))
        flare_labels_dataset = np.concatenate((existing_labels, [item[1] for item in dataset]))
        image_urls_dataset = np.concatenate((existing_urls, [item[2] for item in dataset]))
    else:
        pow_spect_dataset = np.array([item[0] for item in dataset])
        flare_labels_dataset = np.array([item[1] for item in dataset])
        image_urls_dataset = np.array([item[2] for item in dataset])

    
    np.savez_compressed(save_directory, pow_spect=pow_spect_dataset, flare_label=flare_labels_dataset, image_url=image_urls_dataset)

if __name__ == "__main__":
    image_directory = './AIA_171_Images'
    flare_csv_file = './ISEP_Data/FLARE/sdo_goes_flares.csv'
    save_directory = './1d_power_spectrum_dataset.npz'
    data_stride = 1
    power_spectrum_1d_dataset(image_directory, flare_csv_file, save_directory, data_stride)
