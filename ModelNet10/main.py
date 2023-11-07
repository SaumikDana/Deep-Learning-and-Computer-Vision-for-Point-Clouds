__author__ = "Saumik"
__date__ = "11/06/2023"
__source__ = "ModelNet10"

import sys
import os
import gdown
from PointCloudExtractor import PointCloudExtractor

url = 'https://drive.google.com/uc?id=13qtk2agwtZzKgB-0O3-V9zEF-yUDm8Iu'

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <action> ")
        sys.exit(1)

    action = sys.argv[1]

    # Local directory to save the downloaded .zip file
    data_dir = 'data'
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    # Local file path to save the downloaded .zip file
    zip_file_path = os.path.join(data_dir, 'ModelNet10.zip')

    # Download the .mat file from Google Drive if it does not exist
    if not os.path.exists(zip_file_path):
        print("Downloading .zip file...")
        gdown.download(url, zip_file_path, quiet=False)
    else:
        print(".zip file already exists. Skipping download.")

    if action == "extract":
        extractor = PointCloudExtractor(zip_file_path)
        extractor.extract_point_clouds()
    else:
        print("Invalid action. Use 'extract'.")

if __name__ == "__main__":
    main()
