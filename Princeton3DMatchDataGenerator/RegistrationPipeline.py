__author__ = "Saumik"
__date__ = "11/03/2023"

import h5py  # Library to interact with HDF5 files
import os  # Library for operating system dependent functionality
from PointCloudExtractor import PointCloudExtractor  # Import the PointCloudExtractor class
from TransformationEstimator import TransformationEstimator  # Import the TransformationEstimator class

# Define the RegistrationPipeline class
class RegistrationPipeline:
    """
    A class to run a registration pipeline that extracts point clouds from a .mat file,
    computes transformation matrices to align them, and saves the results to an HDF5 file.
    """
    
    # Initialize the class with the file path to the .mat file and an optional output directory
    def __init__(self, file_path, output_dir='output'):
        """
        Initializes the RegistrationPipeline object.
        
        :param file_path: Path to the .mat file containing the data.
        :param output_dir: Directory where the output files will be saved.
        """
        self.file_path = file_path  # Store the file path
        self.output_dir = output_dir  # Store the output directory
        # Create an instance of PointCloudExtractor with the given file path and output directory
        self.point_cloud_extractor = PointCloudExtractor(file_path, output_dir)
        self.transformation_matrices = []  # Initialize a list to store transformation matrices
        self.point_cloud_pairs = []  # Initialize a list to store pairs of point clouds
        # Create the output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # Method to run the registration pipeline
    def run(self, limit=None):
        """
        Runs the registration pipeline to extract point clouds, compute transformations, 
        and align the point clouds.
        
        :param limit: Optional limit on the number of point cloud pairs to process.
        :return: List of transformation matrices.
        """
        # Extract point cloud pairs using the PointCloudExtractor
        self.point_cloud_pairs = self.point_cloud_extractor.extract_point_clouds(limit)
        # Loop through each pair of point clouds
        for source_points, target_points in self.point_cloud_pairs:
            # Create an instance of TransformationEstimator with the current pair of point clouds
            transformation_estimator = TransformationEstimator(source_points, target_points)
            # Compute the transformation matrix for the current pair of point clouds
            transformation_matrix = transformation_estimator.compute_transformation()
            # Add the transformation matrix to the list
            self.transformation_matrices.append(transformation_matrix)
        # Return the list of transformation matrices
        return self.transformation_matrices

    # Method to save the point cloud pairs and transformation matrices to an HDF5 file
    def save_to_hdf5(self, filename='output_data.h5'):
        """
        Saves the point cloud pairs and their corresponding transformation matrices to an HDF5 file.
        
        :param filename: Name of the HDF5 file to save the data.
        """
        # Create or open the HDF5 file
        with h5py.File(os.path.join(self.output_dir, filename), 'w') as f:
            # Loop through each pair of point clouds and the corresponding transformation matrix
            for i, ((source_points, target_points), transformation_matrix) in enumerate(zip(self.point_cloud_pairs, self.transformation_matrices)):
                # Create a group for the current pair
                grp = f.create_group(f'pair_{i}')
                # Save the source points, target points, and transformation matrix to the group
                grp.create_dataset('source_points', data=source_points)
                grp.create_dataset('target_points', data=target_points)
                grp.create_dataset('transformation_matrix', data=transformation_matrix)
        # Print a message indicating where the data was saved
        print(f"Data saved to {os.path.join(self.output_dir, filename)}")
