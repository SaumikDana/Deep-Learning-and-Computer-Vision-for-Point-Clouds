__author__ = "Saumik"
__date__ = "11/03/2023"

import open3d as o3d  # Open3D for handling point clouds and performing ICP
import numpy as np  # NumPy for handling numerical operations

# Define the TransformationEstimator class
class TransformationEstimator:
    """
    A class for estimating the rigid transformation matrix that aligns a source point cloud to a target point cloud.
    """
    
    # Initialize the class with source and target point clouds
    def __init__(self, source_points, target_points):
        """
        Initializes the TransformationEstimator object.
        
        :param source_points: Nx3 NumPy array representing the source point cloud.
        :param target_points: Nx3 NumPy array representing the target point cloud.
        """
        self.source_points = source_points  # Store the source point cloud as a NumPy array
        self.target_points = target_points  # Store the target point cloud as a NumPy array

    # Method to compute the transformation matrix that aligns the source point cloud to the target point cloud
    def compute_transformation(self):
        """
        Computes the rigid transformation matrix that aligns the source point cloud to the target point cloud.
        
        :return: 4x4 NumPy array representing the transformation matrix.
        """
        # Convert numpy arrays to Open3D point clouds
        source_cloud = o3d.geometry.PointCloud()  # Create an Open3D point cloud object for the source points
        target_cloud = o3d.geometry.PointCloud()  # Create an Open3D point cloud object for the target points
        source_cloud.points = o3d.utility.Vector3dVector(self.source_points)  # Set the points of the source cloud
        target_cloud.points = o3d.utility.Vector3dVector(self.target_points)  # Set the points of the target cloud

        # Apply ICP (Iterative Closest Point) registration
        threshold = 0.02  # Set an appropriate threshold value for the ICP algorithm. This value determines the maximum distance between matched points.
        trans_init = np.asarray([[1, 0, 0, 0],   # Initial transformation matrix (identity matrix)
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        # Perform the ICP registration to align the source point cloud to the target point cloud
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # Get the transformation matrix resulting from the ICP registration
        transformation_matrix = reg_p2p.transformation
        return transformation_matrix  # Return the transformation matrix



