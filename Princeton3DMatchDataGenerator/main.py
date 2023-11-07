__author__ = "Saumik"
__date__ = "11/03/2023"
__source__ = "https://3dmatch.cs.princeton.edu/"

# This script is used for extracting and registering point clouds using the
# datasets provided by the 3DMatch project (https://3dmatch.cs.princeton.edu/).
# It includes functionalities to download a .mat file from Google Drive,
# extract point clouds, and run a registration pipeline.

import sys
import os
import gdown
from RegistrationPipeline import RegistrationPipeline
from PointCloudExtractor import PointCloudExtractor

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <action> <number of point clouds>")
        sys.exit(1)

    action = sys.argv[1]
    num = int(sys.argv[2])

    # Google Drive URL for the .mat file
    url = 'https://drive.google.com/uc?id=1ba9Styp7qjCpJlrZnvCCHtgYV1Mv255E'
    # Local directory to save the downloaded .mat file
    data_dir = 'data'
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    # Local file path to save the downloaded .mat file
    mat_file_path = os.path.join(data_dir, 'validation-set.mat')

    # Download the .mat file from Google Drive if it does not exist
    if not os.path.exists(mat_file_path):
        print("Downloading .mat file...")
        gdown.download(url, mat_file_path, quiet=False)
    else:
        print(".mat file already exists. Skipping download.")

    if action == "extract":
        extractor = PointCloudExtractor(mat_file_path)
        extractor.extract_point_clouds(limit=num)
    elif action == "register":
        pipeline = RegistrationPipeline(mat_file_path)
        pipeline.run(limit=num)
        pipeline.save_to_hdf5()
    else:
        print("Invalid action. Use 'extract' or 'register'.")

if __name__ == "__main__":
    main()
