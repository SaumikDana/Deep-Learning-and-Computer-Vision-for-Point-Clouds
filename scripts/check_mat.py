__author__ = "Saumik"
__purpose__ = "Interrogate the Princeton 3D Match dataset"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# Construct the path to the .mat file
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
mat_file_path = os.path.join(parent_dir, 'data', 'validation-set.mat')

# Load .mat file
with h5py.File(mat_file_path, 'r') as f:
    # Dereference data and labels
    data = f['data']
    labels = f['labels']

    # Example: Visualize the RGB and Depth patches of the first pair in the dataset
    # Assuming that the patches are stored in 'colorPatch' and 'depthPatch' fields
    # You might need to adjust the indices and field names based on the actual structure of your .mat file
    first_pair_ref = data[0, 0]
    first_pair = f[first_pair_ref]
    rgb_patch_ref = first_pair['colorPatch'][0, 0]
    depth_patch_ref = first_pair['depthPatch'][0, 0]

    rgb_patch = f[rgb_patch_ref][()]
    depth_patch = f[depth_patch_ref][()]

# Function to visualize RGB and Depth patches
def visualize_patches(rgb_patch, depth_patch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Visualize RGB patch
    axes[0].imshow(rgb_patch)
    axes[0].set_title('RGB Patch')
    axes[0].axis('off')
    
    # Visualize Depth patch
    # You might want to normalize the depth values for better visualization
    normalized_depth = (depth_patch - np.min(depth_patch)) / (np.max(depth_patch) - np.min(depth_patch))
    axes[1].imshow(normalized_depth, cmap='gray')
    axes[1].set_title('Depth Patch')
    axes[1].axis('off')
    
    plt.show()

visualize_patches(rgb_patch, depth_patch)
