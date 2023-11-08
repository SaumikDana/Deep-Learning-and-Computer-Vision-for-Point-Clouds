import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_fpfh(fpfh):
    labels = list(range(fpfh.shape[1]))
    for i in range(fpfh.shape[0]):
        plt.bar(labels, fpfh[i, :])
        plt.title(f"FPFH Histogram for Point {i}")
        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.show()

# Get the directory in which the script file resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Now construct the absolute path to the PLY file
file_path = os.path.join(script_dir, "pc-20230425-132107084_Bstar_3_I-(0,1024)w-29fps-RGB.ply")

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File {file_path} does not exist.")
else:
    # Read the .ply file
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if the point cloud is empty
    if pcd.is_empty():
        print("The point cloud is empty.")
    else:
        # Estimate normals if they don't exist
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Check again if normals were estimated
        if not pcd.has_normals():
            print("Normal estimation failed.")
        else:
            # Compute FPFH features
            radius_feature = 0.1
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

            # Convert FPFH features to numpy array for visualization
            fpfh_data = np.asarray(fpfh.data).T

            # Visualize the first few FPFH feature histograms
            visualize_fpfh(fpfh_data[:5, :])  # Visualize the first 5 histograms
