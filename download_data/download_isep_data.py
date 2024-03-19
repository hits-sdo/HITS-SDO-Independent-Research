import os
import requests
import tarfile
import time

def download_and_extract(url, download_dir):
    # Extract filename from URL
    file_name = url.split("/")[-1]

    # Check if tar file already exists
    file_path = os.path.join(download_dir, file_name)
    if os.path.exists(file_path):
        print("Tar file already exists. Skipping download.")
    else:
        print("Downloading file...")
        # Download file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download completed.")

    # Create directory for extraction
    extract_dir = os.path.join(download_dir, file_name.split('.')[0])
    if os.path.exists(extract_dir):
        print("Extracted directory already exists.")
    else:
        print("Creating directory for extraction...")
        os.makedirs(extract_dir)
        print("Directory created.")

    # Extract contents
    print("Extracting files...")
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print("Extraction completed.")

if __name__ == "__main__":
    url = "https://dmlab.cs.gsu.edu/solar/data/ISEP_Data.tar.gz"
    download_dir = "./.."

    start_time = time.time()
    download_and_extract(url, download_dir)
    end_time = time.time()

    elapsed_time_seconds = end_time - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    print(f"Task completed in {elapsed_time_formatted}.")
