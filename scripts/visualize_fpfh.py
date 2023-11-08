import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import copy

def visualize_fpfh(fpfh, feature_index):
    labels = list(range(fpfh.shape[1]))
    plt.bar(labels, fpfh[feature_index, :])
    plt.title(f"FPFH Histogram for Feature Index {feature_index}")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()

def color_point_cloud_by_fpfh(pcd, fpfh, feature_index):
    # Convert to numpy array if not already
    fpfh_values = np.asarray(fpfh.data).T  # Transpose to make it points x features
    # Normalize each feature across all points
    max_values = np.max(fpfh_values, axis=0)
    min_values = np.min(fpfh_values, axis=0)
    normalized_fpfh = (fpfh_values - min_values) / (max_values - min_values)

    # Apply the colors to the point cloud based on the feature importance
    cmap = plt.get_cmap("inferno")
    colors = cmap(normalized_fpfh[:, feature_index])[:, :3]  # Take only RGB, not alpha
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a coordinate frame (axis)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

    # Define the rotation angle around the Z-axis (180 degrees clockwise)
    angle_z = np.pi  # 180 degrees in radians

    # Create a numpy array combining the axis and the angle
    axis_angle = np.array([0, 0, -angle_z])  # Negative for clockwise rotation
    rotation_matrix_z = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)

    # Apply the rotation to the point cloud and the coordinate frame
    pcd.rotate(rotation_matrix_z, center=(0, 0, 0))
    coordinate_frame.rotate(rotation_matrix_z, center=(0, 0, 0))

    # Define the rotation angle around the X-axis (45 degrees cclockwise)
    angle_x = np.pi/4  

    # Create a numpy array combining the axis and the angle
    axis_angle = np.array([angle_x, 0, 0])  # 
    rotation_matrix_x = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)

    # Apply the rotation to the point cloud and the coordinate frame
    pcd.rotate(rotation_matrix_x, center=(0, 0, 0))
    coordinate_frame.rotate(rotation_matrix_x, center=(0, 0, 0))

    # Visualize the point cloud with coordinate axes
    o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                      window_name=f"Point Cloud with FPFH Feature {feature_index}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FPFH features on a point cloud.")
    parser.add_argument("file_path", help="Path to the PLY file")
    parser.add_argument("--feature_index", type=int, default=0, help="Index of the FPFH feature to visualize")
    
    args = parser.parse_args()

    # Ensure the PLY file exists
    if not os.path.isfile(args.file_path):
        print(f"File {args.file_path} does not exist.")
        exit()

    # Read the .ply file
    pcd = o3d.io.read_point_cloud(args.file_path)

    # Estimate normals if they don't exist
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute FPFH features
    radius_feature = 0.1
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    # Visualize the point cloud colored by the specified FPFH feature
    color_point_cloud_by_fpfh(pcd, fpfh, args.feature_index)

