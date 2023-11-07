__author__ = "Saumik"
__date__ = "11/03/2023"

import h5py
import numpy as np
import open3d as o3d
import os

# Define the PointCloudExtractor class
class PointCloudExtractor:
    """
    A class for extracting point clouds from a .mat file and saving them as PLY files.
    """
    
    # Initialize the class with the file path to the .mat file and an optional output directory
    def __init__(self, file_path, output_dir='output'):
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

    # Static method to dereference HDF5 references
    @staticmethod
    def dereference_ref(ref, h5_file):
        """
        Dereferences an HDF5 reference.
        
        :param ref: HDF5 reference or data.
        :param h5_file: HDF5 file object.
        :return: Dereferenced data or the input data if it's not an HDF5 reference.
        """
        # If the input is an HDF5 reference, return the dereferenced data
        if isinstance(ref, h5py.Reference):
            return h5_file[ref]
        # If not, return the input as is
        return ref


    @staticmethod
    def depth_to_point_cloud(depth_patch, camK):
        """
        Converts a depth patch and camera intrinsics to a point cloud.
        
        :param depth_patch: 2D array of depth values.
        :param camK: Camera intrinsics matrix.
        :return: Nx3 array of 3D points, or None if invalid values are found.
        """
        # Check for NaN or Inf values in the depth patch
        if np.isnan(depth_patch).any() or np.isinf(depth_patch).any():
            print("Invalid values (NaN or Inf) found in depth patch")
            return None

        # Check if the camera matrix is singular
        if np.isclose(np.linalg.det(camK), 0):
            print("Camera matrix is singular or near singular")
            return None

        # Invert the camera matrix
        try:
            K_inv = np.linalg.inv(camK)
        except np.linalg.LinAlgError:
            print("Error inverting camera matrix")
            return None

        # Check for NaN or Inf values in the inverted camera matrix
        if np.isnan(K_inv).any() or np.isinf(K_inv).any():
            print("Invalid values (NaN or Inf) found in inverted camera matrix")
            return None

        H, W = depth_patch.shape  # Get the height and width of the depth patch
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))  # Create a grid of x and y coordinates
        zs = depth_patch  # The z coordinates are the depth values
        points = np.vstack((xs.ravel(), ys.ravel(), zs.ravel()))  # Stack the coordinates
        
        # Apply the inverse camera matrix to transform to world coordinates
        points = K_inv @ points
        points = points / points[2, :]  # Normalize by the depth
        
        # Check for NaN or Inf values in the computed 3D points
        if np.isnan(points).any() or np.isinf(points).any():
            print("Invalid values (NaN or Inf) found in computed 3D points")
            return None
        
        return points.T  # Return the points as a Nx3 array

    # Method to save a point cloud to a PLY file
    def save_point_cloud_to_ply(self, points, filename):
        """
        Saves a point cloud to a PLY file.
        
        :param points: Nx3 array of 3D points.
        :param filename: Name of the output PLY file.
        """
        point_cloud = o3d.geometry.PointCloud()  # Create an Open3D point cloud object
        point_cloud.points = o3d.utility.Vector3dVector(points)  # Set the points of the point cloud
        output_path = os.path.join(self.output_dir, filename)  # Create the output file path
        o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=True)  # Save the point cloud to a PLY file
        print(f"Point cloud saved to {output_path}")  # Print a message indicating where the point cloud was saved

    # Method to extract point clouds from the .mat file
    def extract_point_clouds(self, limit=None):
        """
        Extracts point clouds from the .mat file and saves them as PLY files.
        
        :param limit: Optional limit on the number of point cloud pairs to extract.
        :return: List of tuples, each containing two point clouds.
        """
        point_cloud_pairs = []  # Initialize a list to store pairs of point clouds
        
        # Open the .mat file
        with h5py.File(self.file_path, 'r') as file:
            data = file['data']  # Access the 'data' dataset
            num_pairs = data.shape[1]  # Get the number of point cloud pairs
            
            # Loop through each pair of point clouds
            for i in range(num_pairs):
                interest_point_refs = data[:, i]  # Get the HDF5 references for the current pair
                
                # Dereference the HDF5 references to get the actual data
                interest_point_data = [self.dereference_ref(ref, file) for ref in interest_point_refs]
                
                point_clouds = []  # Initialize a list to store the point clouds for the current pair
                skip_pair = False  # Flag to indicate whether to skip the current pair
                
                # Loop through each interest point in the current pair
                for j, interest_point in enumerate(interest_point_data):
                    depth_patch = interest_point['depthPatch'][:]  # Get the depth patch
                    camK = interest_point['camK'][:]  # Get the camera intrinsics
                    point_cloud = self.depth_to_point_cloud(depth_patch, camK)  # Convert to a point cloud
                    
                    # If the point cloud is None, set the flag and break the inner loop
                    if point_cloud is None:
                        print(f"Skipping point cloud pair {i+1} due to invalid values in point cloud {j+1}.")
                        skip_pair = True
                        break
                    
                    point_clouds.append(point_cloud)  # Add the point cloud to the list
                
                # If the flag is set, skip the rest of the current iteration of the outer loop
                if skip_pair:
                    continue
                
                # If both point clouds in the pair are valid, save them and add the pair to the list
                for j, point_cloud in enumerate(point_clouds):
                    filename = f"point_cloud_{i+1}_{j+1}.ply"  # Create a filename for the point cloud
                    self.save_point_cloud_to_ply(point_cloud, filename)  # Save the point cloud to a PLY file
                
                point_cloud_pairs.append(tuple(point_clouds))  # Add the pair of point clouds to the list
                
                # If a limit on the number of pairs is set and reached, break out of the loop
                if limit and len(point_cloud_pairs) >= limit:
                    break
                    
        return point_cloud_pairs  # Return the list of point cloud pairs
