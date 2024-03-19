import bs4
import os
import pandas as pd
import re
import requests
import time
import urllib.parse

def fetch_image_list(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('a', href=True)
        image_list = [urllib.parse.urljoin(url, img['href']) for img in images if img['href'].endswith('.jpg')]
        return image_list
    else:
        print(f"Failed to fetch image list for {url}.")
        return []

def download_aia_images(start_date, end_date, resolution, aia_type, save_directory, n):
    base_url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    image_counter = 0

    for date in date_range:
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        url = f"{base_url}{year}/{month}/{day}/"
        print(f"Fetching image list for {url}")
        image_list = fetch_image_list(url)

        if len(image_list) == 0:
            continue

        for image_url in image_list:
            if not re.match(fr"(\d{{4}})?(\d{{4}})_\d{{6}}_{resolution}_{aia_type}\.jpg", image_url.split('/')[-1]):
                continue

             # Extract file name from URL
            file_name = image_url.split('/')[-1] 
            image_path = os.path.join(save_directory, year, month, day)

            # Create directory if it doesn't exist
            os.makedirs(image_path, exist_ok=True)  
            image_path = os.path.join(image_path, file_name)

            # Check if the file already exists
            if os.path.exists(image_path):
                print(f"Skipping {image_url} as it already exists.")
                image_counter += 1
                continue

            if image_counter % n == 0:
                print(f"Downloading {image_url}")
                response = requests.get(image_url)
                with open(image_path, "wb") as f:
                    f.write(response.content)

            image_counter += 1

if __name__ == "__main__":
    start_date = "2010-05-01"
    end_date = "2018-07-06"
    resolution = "4096"
    aia_type = "0171"
    save_directory = "./../AIA_171_Images"
    n = 10 

    start_time = time.time()
    download_aia_images(start_date, end_date, resolution, aia_type, save_directory, n)
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time_seconds = end_time - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    
    print(f"Script executed in {elapsed_time_formatted}.")
