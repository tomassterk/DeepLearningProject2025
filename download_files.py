## Run this file only on the HPC computer

import ais_to_parquet
import calendar
import requests
from tqdm import tqdm
import os
import zipfile

def get_zip_names(query_year, query_months, url):
    file_names = list()

    for query_month in query_months:

        month_str = f"{query_month:02d}"
        monthly_url = f"{url}/{query_year}/aisdk-{query_year}-{month_str}.zip"
        response = requests.head(monthly_url)

        if response.status_code == 200:
            # Monthly file exists
            return [monthly_url]

        # Otherwise: create daily files
        query_days = calendar.monthrange(query_year, query_month)[1]
        file_names_month = [
            f"{url}/{query_year}/aisdk-{query_year}-{month_str}-{day:02d}.zip"
            for day in range(1, query_days + 1)
        ]
        file_names.extend(file_names_month)

    return file_names

def download_files(file_urls, target_folder_name="ship_data"):
    target_folder = f"/dtu/blackhole/08/223112/{target_folder_name}"
    os.makedirs(target_folder, exist_ok=True)

    for file_url in file_urls:

        try:
            # Extract the filename from the URL
            file_name = os.path.basename(file_url)
            save_path = os.path.join(target_folder, file_name)
            print(save_path)



            # Download with streaming
            response = requests.get(file_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            # Progress bar
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=file_name
            )

            # Write file in chunks
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))

            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(target_folder)
            os.remove(save_path)





            progress.close()
            print(f"Processed: {save_path}")
        except:

            print(f"Download of file: {file_url} failed!")



query_year = 2024
query_months = [3, 5, 7]
url = 'http://aisdata.ais.dk'
save_folder = "/dtu/blackhole/08/223112"
zip_urls = get_zip_names(query_year, query_months, url)

download_files(zip_urls)
