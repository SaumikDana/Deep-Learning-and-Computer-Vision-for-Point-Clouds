__author__ = "Saumik"
__date__ = "11/08/2023"

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Function to visualize the histogram of features
def visualize_fpfh(fpfh):
    # The FPFH feature is a 33-bin histogram for each point
    labels = []
    for i in range(fpfh.shape[1]):
        labels.append(f"Bin {i}")
    for i in range(fpfh.shape[0]):
        plt.bar(labels, fpfh[i, :])
        plt.title(f"FPFH Histogram for Point {i}")
        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.show()

# Read the .ply file
pcd = o3d.io.read_point_cloud("pc-20230425-132107084_Bstar_3_I-(0,1024)w-29fps-RGB.ply")

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Prepare the FPFH computation class
radius_feature = 0.1
fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
)

# Convert FPFH features to numpy array for visualization
fpfh_data = np.asarray(fpfh.data).T

# Visualize the first few FPFH feature histograms
visualize_fpfh(fpfh_data[:5, :])  # Visualize the first 5 histograms

# For a full visualization, you would call visualize_fpfh(fpfh_data)
