import bs4
import multiprocessing
import os
import pandas as pd
import re
import requests
import time
import urllib.parse
from multiprocessing import Pool

def fetch_image_list(url):
    with requests.Session() as session:
        response = session.get(url)
    if response.status_code == 200:
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('a', href=True)
        image_list = [urllib.parse.urljoin(url, img['href']) for img in images if img['href'].endswith('.jpg')]
        return image_list
    else:
        print(f"Failed to fetch image list for {url}.")
        return []

def download_image(image_url, image_path):
    with requests.Session() as session:
        response = session.get(image_url)
        with open(image_path, "wb") as f:
            f.write(response.content)

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
    start_date = "2013-04-29"
    end_date = "2018-07-06"
    resolution = "1024"
    aia_type = "0171"
    save_directory = "./AIA_171_Images"
    n = 1
    thread_count = multiprocessing.cpu_count()

    start_time = time.time()
    download_aia_images(start_date, end_date, resolution, aia_type, save_directory, n, thread_count)
    end_time = time.time()
    
    elapsed_time_seconds = end_time - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    
    print(f"Script executed in {elapsed_time_formatted}.")
