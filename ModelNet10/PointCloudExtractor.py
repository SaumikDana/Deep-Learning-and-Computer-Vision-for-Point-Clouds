__author__ = "Saumik"
__date__ = "11/03/2023"

import zipfile
import numpy as np
import open3d as o3d
import os

# Define the PointCloudExtractor class
class PointCloudExtractor:
    """
    A class for extracting point clouds from a .mat file and saving them as PLY files.
    """
    
    # Initialize the class with the file path to the .mat file and an optional output directory
    def __init__(self, file_path, output_dir='output_ModelNet10'):
        """
        Initializes the PointCloudExtractor object.
        
        :param file_path: Path to the .mat file containing the point cloud data.
        :param output_dir: Directory where the extracted point clouds will be saved. Defaults to 'output'.
        """
        self.file_path = file_path  # Store the file path
        self.output_dir = output_dir  # Store the output directory
        
        # Create the output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # Method to save a point cloud to a PLY file
    def save_point_cloud_to_ply(self, points, filename):
        """
        Saves a point cloud to a PLY file.
        
        :param points: Nx3 array of 3D points.
        :param filename: Name of the output PLY file.
        """
        # # Debugging code to check the points variable
        # print("Points type:", type(points))
        # print("Points dtype:", points.dtype)
        # print("Points shape:", points.shape)

        # Ensure points is a numpy array with the correct shape and type
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            points = np.array(points).reshape(-1, 3).astype(np.float64)

        # Check for NaN or inf values
        if np.isnan(points).any() or np.isinf(points).any():
            print("Warning: NaN or inf values found in points data.")
            # Handle NaN or inf values here, for example by removing them
            points = points[~np.isnan(points).any(axis=1)]
            points = points[~np.isinf(points).any(axis=1)]

        # Try setting the points and catch any exceptions
        try:
            point_cloud = o3d.geometry.PointCloud()  # Create an Open3D point cloud object
            point_cloud.points = o3d.utility.Vector3dVector(points)  # Set the points of the point cloud
        except Exception as e:
            print("An error occurred while setting points:", e)
            # Handle the error appropriately
            return

        output_path = os.path.join(self.output_dir, filename)  # Create the output file path
        o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=True)  # Save the point cloud to a PLY file
        print(f"Point cloud saved to {output_path}")  # Print a message indicating where the point cloud was saved

    def extract_point_clouds(self, limit=None):
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)  # Extract all files into the output directory

            # Now, iterate over the extracted files and convert them to PLY
            count = 0
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file.endswith('.off') and (limit is None or count < limit):
                        file_path = os.path.join(root, file)
                        mesh = o3d.io.read_triangle_mesh(file_path)
                        if not mesh.has_vertex_normals():
                            mesh.compute_vertex_normals()
                        points = np.asarray(mesh.vertices)
                        # You may want to add additional processing here if needed
                        self.save_point_cloud_to_ply(points, file.replace('.off', '.ply'))
                        count += 1
                        if limit is not None and count >= limit:
                            break
