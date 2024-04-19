import bs4
import glob
import multiprocessing
import numpy as np
import os
import pandas as pd
import re
import requests
import time
import urllib.parse
from multiprocessing import Pool
from PIL import Image

def remove_bad_data(image_path):
    try:
        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) != 3:
            os.remove(image_path)
            return 1
        return 0 
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return -1

def fetch_image_list(url):
    retry_delay = 5  # Seconds to wait before retrying
    
    with requests.Session() as session:
        while True:
            try:
                response = session.get(url)
                if response.status_code == 200:
                    soup = bs4.BeautifulSoup(response.text, 'html.parser')
                    images = soup.find_all('a', href=True)
                    image_list = [urllib.parse.urljoin(url, img['href']) for img in images if img['href'].endswith('.jpg')]
                    return image_list
                else:
                    print(f"Failed to fetch image list for {url}. Status code: {response.status_code}")
                    return []
            except requests.ConnectionError:
                print(f"Connection error for {url}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

def download_image(image_url, image_path):
    retry_delay = 5  # Seconds to wait before retrying
    
    with requests.Session() as session:
        while True:
            try:
                response = session.get(image_url)
                if response.status_code == 200:
                    with open(image_path, "wb") as f:
                        f.write(response.content)
                    return
                else:
                    print(f"Failed to download {image_url}. Status code: {response.status_code}")
                    return
            except requests.ConnectionError:
                print(f"Connection error when downloading {image_url}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

def download_images(image_urls):
    for image_url, image_path in image_urls:
        download_image(image_url, image_path)
        print(f"Downloaded {image_url}")

def download_aia_images(start_date, end_date, resolution, aia_type, save_directory, n, thread_count):
    base_url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    regex_pattern = re.compile(fr"(\d{{4}})?(\d{{4}})_\d{{6}}_{resolution}_{aia_type}\.jpg")

    for date in date_range:
        year, month, day = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
        url = f"{base_url}{year}/{month}/{day}/"
        print(f"Fetching image list for {url}")
        image_list = fetch_image_list(url)

        if not image_list:
            continue

        download_urls = []
        valid_image_count = 0
        for image_url in image_list:
            if regex_pattern.match(image_url.split('/')[-1]):
                valid_image_count += 1
                if valid_image_count % n == 0:
                    file_name = image_url.split('/')[-1]
                    image_path = os.path.join(save_directory, year, month, day, file_name)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    if not os.path.exists(image_path):
                        download_urls.append((image_url, image_path))

        with Pool(processes=thread_count) as pool:
            print(f"Downloading images for {year}/{month}/{day}")
            pool.map(download_images, [download_urls[i:i+thread_count] for i in range(0, len(download_urls), thread_count)])
            print(f"Downloaded images for {year}/{month}/{day}")

if __name__ == "__main__":
    start_date = "2010-05-01"
    end_date = "2015-12-31"
    resolution = "4096"
    aia_type = "0171"
    save_directory = "./AIA_171_Images"
    n = 1
    thread_count = max(multiprocessing.cpu_count() - 1, 1)

    start_time = time.time()
    download_aia_images(start_date, end_date, resolution, aia_type, save_directory, n, thread_count)

    print("Removing Bad Data...")

    image_paths = glob.glob(save_directory + "/**/*.jpg", recursive=True)
    total_images = len(image_paths)
    removed_images = 0

    print(f"Total images to process: {total_images}")

    with multiprocessing.Pool(processes=thread_count) as pool:
            results = pool.map(remove_bad_data, image_paths)

    removed_images = sum(results)
    remaining_images = total_images - removed_images
    print("Bad data removal completed.")
    print(f"Total images processed: {total_images}")
    print(f"Total images removed: {removed_images}")
    print(f"Total images remaining: {total_images - removed_images}")
    
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    
    print(f"Script executed in {elapsed_time_formatted}.")
