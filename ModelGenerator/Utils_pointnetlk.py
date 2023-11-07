__author__ = "Saumik"
__date__ = "11/03/2023"

import h5py
import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset

class CustomPointCloudDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        """
        Custom dataset for loading point cloud data from an HDF5 file.

        :param h5_file: Path to the HDF5 file containing the dataset.
        :param transform: Optional transform to be applied on a sample.
        """
        self.h5_file = h5_file
        self.transform = transform

        # Open the HDF5 file to determine the number of samples.
        with h5py.File(self.h5_file, 'r') as f:
            self.num_samples = len(f.keys())

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: Number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: Index of the sample to retrieve.
        :return: A dictionary containing the point cloud and transformation matrix.
        """
        # Open the HDF5 file.
        with h5py.File(self.h5_file, 'r') as f:
            # Access the group corresponding to the specified index.
            group_name = f'pair_{idx}'
            group = f[group_name]
            
            # Load the point cloud and transformation matrix from the group.
            source_points = torch.tensor(group['source_points'][:], dtype=torch.float32)
            target_points = torch.tensor(group['target_points'][:], dtype=torch.float32)
            transformation_matrix = torch.tensor(group['transformation_matrix'][:], dtype=torch.float32)

            # Check for nan or inf values in the source points, target points, and transformation matrix
            if torch.isnan(source_points).any() or torch.isinf(source_points).any():
                print(f"Invalid values found in source_points for pair {idx}")
            if torch.isnan(target_points).any() or torch.isinf(target_points).any():
                print(f"Invalid values found in target_points for pair {idx}")
            if torch.isnan(transformation_matrix).any() or torch.isinf(transformation_matrix).any():
                print(f"Invalid values found in transformation_matrix for pair {idx}")
        
        # Create a dictionary to hold the sample data.
        sample = {
            'source_points': source_points, 
            'target_points': target_points, 
            'transformation_matrix': transformation_matrix
        }
        
        # If a transform function is provided, apply it to the sample.
        if self.transform:
            sample = self.transform(sample)
            
        # Return the sample.
        return sample

def check_data(group):
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):  # Check if the item is a dataset
            data = item[:]
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Dataset {key} in group {group.name} contains nan or inf values.")
        elif isinstance(item, h5py.Group):  # If the item is a group, check its datasets recursively
            check_data(item)

def get_datasets(args):
    """
    Prepare and return the training and testing datasets.
    
    :param args: Command line arguments.
    :return: A tuple of (train_dataset, test_dataset).
    """
    # Define a series of transformations to be applied to the data.
    transform = torchvision.transforms.Compose([
        # Add any required transformations here.
        # Example transformations could include normalization, resampling, etc.
    ])

    # Initialize the custom point cloud dataset with the provided h5 file and transformations.
    dataset = CustomPointCloudDataset(h5_file=args.h5_file, transform=transform)

    # Check for nan or inf values in the dataset
    with h5py.File(args.h5_file, 'r') as f:
        check_data(f)

    # Calculate the sizes of the training and testing datasets (80% training, 20% testing).
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Randomly split the dataset into training and testing sets.
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def check_dimension_sanity(trainset, testset, trainset_number, testset_number):
    """
    Sanity dimension check.
    """

    if len(trainset) > 0:
        train_data_example = trainset[trainset_number]
        print(f"Dimension of source pcd for training point # {trainset_number}: {train_data_example['source_points'].shape}")
        print(f"Dimension of target pcd for training point # {trainset_number}: {train_data_example['target_points'].shape}")
    else:
        print("Training set is empty.")

    if len(testset) > 0:
        test_data_example = testset[testset_number]
        print(f"Dimension of source pcd for test point # {testset_number}: {test_data_example['source_points'].shape}")
        print(f"Dimension of source pcd for test point # {testset_number}: {test_data_example['target_points'].shape}")
    else:
        print("Testing set is empty.")

