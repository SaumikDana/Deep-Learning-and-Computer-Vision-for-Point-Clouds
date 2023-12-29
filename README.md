## Directory Structure

```
DL_CV_Images/
│
├── .vscode/ - Configuration files for Visual Studio Code.
│   └── launch.json - Defines configuration for launching and debugging the project.
│
├── ModelGenerator/ - Scripts and modules for generating and training models.
│   ├── Action_pointnetlk.py - Script related to actions for the PointNetLK model.
│   ├── PointNetLK/ - Contains the implementation of the PointNetLK architecture.
│   │   ├── __init__.py - Initializes the PointNetLK package.
│   │   ├── invmat.py - Inverse matrix computations for PointNetLK.
│   │   ├── pointnet.py - Core PointNet model definitions.
│   │   ├── pointnet_classifier.py - PointNet model for classification tasks.
│   │   ├── pointnet_segmenter.py - PointNet model for segmentation tasks.
│   │   ├── pointnetlk.py - PointNetLK model implementation.
│   │   ├── se3.py - Script for SE(3) transformations.
│   │   ├── sinc.py - Sinc function implementation.
│   │   ├── so3.py - Script for SO(3) rotations.
│   │   └── __pycache__/ - Compiled Python files for faster loading.
│   ├── Settings_pointnetlk.py - Configuration settings for the PointNetLK model.
│   ├── Trainer_pointnetlk.py - Training script for the PointNetLK model.
│   ├── Utils_pointnetlk.py - Utility functions for the PointNetLK model.
│   ├── train_pointnetlk.py - Main training script for PointNetLK.
│   ├── train_pointnetsegmenter.py - Training script for PointNet segmenter.
│   └── __pycache__/ - Compiled Python files for faster loading.
│
├── ModelNet10/ - Scripts related to the ModelNet10 dataset.
│   ├── PointCloudExtractor.py - Extracts point clouds from the dataset.
│   ├── main.py - Main script for processing ModelNet10 data.
│   └── __pycache__/ - Compiled Python files for faster loading.
│
├── Princeton3DMatchDataGenerator/ - Scripts for generating 3D match data.
│   ├── PointCloudExtractor.py - Extracts point clouds for matching.
│   ├── RegistrationPipeline.py - Pipeline for registering 3D point clouds.
│   ├── TransformationEstimator.py - Estimates transformations between point clouds.
│   ├── main.py - Main script for the 3D match data generation process.
│   └── __pycache__/ - Compiled Python files for faster loading.
│
├── SimpleITK/ - Scripts using the SimpleITK library for registration.
│   ├── non-rigid_registration.py - Performs non-rigid registration.
│   └── rigid_registration.py - Performs rigid registration.
│
├── scripts/ - Various utility and testing scripts.
│   ├── check_mat.py - Checks matrix files for errors or issues.
│   ├── check_npz.py - Verifies the integrity of NPZ files.
│   ├── data/MNIST/raw/ - Raw MNIST data for testing and examples.
│   ├── encoding_patterns.py - Scripts related to encoding patterns.
│   ├── failure_cnn_1.py - Script simulating a failure case for CNNs.
│   ├── failure_cnn_2.py - Another script for CNN failure simulation.
│   ├── generate_call_graphs.py - Generates call graphs for analysis.
│   ├── generate_call_stacks.py - Produces call stacks for debugging.
│   ├── marching_patterns.py - Related to generating marching patterns.
│   ├── pareto_chart.py - Script for creating Pareto charts.
│   ├── print_keys.py - Prints keys from data structures or files.
│   ├── show_and_tell.py - Script for visualization and presentation.
│   ├── test_npz.py - Tests NPZ files for integrity and content.
│   ├── visualize_deformation_patterns.py - Visualizes deformation patterns.
│   └── visualize_fpfh.py - Visualizes Fast Point Feature Histograms.
│
├── .gitattributes - Defines attributes for paths in the Git repository.
├── Approach.png - An image illustrating the project's approach or architecture.
├── README.md - The repository's readme file with an overview and instructions.
├── __pycache__/ - Compiled Python files for faster loading.
├── poetry.lock - Lock file for Poetry, specifying exact versions of dependencies.
└── pyproject.toml - Configuration file for Poetry and project metadata.
```

## PointNet architecture

![pointnet](https://github.com/SaumikDana/Deep-Learning-and-Computer-Vision-for-Point-Clouds/assets/9474631/eeb20b61-33b5-498a-a5c8-a5275f4507a7)

## PointNetLK based registration

PointNetLK (PointNet Lucas-Kanade) is an adaptation and combination of the PointNet architecture with the Lucas-Kanade algorithm for the task of 3D point cloud registration. PointNet is a deep neural network designed to process point clouds (sets of points in a 3D space), and the Lucas-Kanade method is a classical algorithm for image registration, typically used for aligning images and tracking motion. When adapted for 3D point cloud registration, the goal is to align two sets of 3D points (point clouds) from different perspectives or times. Here's an overall algorithmic framework for PointNetLK applied to 3D point cloud registration:

- Input Preparation:

    Point Clouds: Obtain two point clouds, a source and a target, that you want to align.
    Preprocessing: Preprocess the point clouds if necessary (e.g., downsampling, denoising).

- Feature Extraction with PointNet:

    Source and Target Features: Pass both the source and target point clouds through a PointNet architecture to extract features. PointNet processes each point individually and uses a symmetric function (like max pooling) to ensure invariance to permutations of the points.
    Feature Representation: Obtain a global feature representation for each point cloud, capturing the distribution of points and their spatial relationships.

- Lucas-Kanade Iterative Alignment:

    Initial Parameters: Start with an initial guess of the transformation (e.g., identity if no prior knowledge).
    Iterative Process:
        Warping: Apply the current estimate of the transformation to the source point cloud to align it with the target.
        Error Computation: Compute the difference between the warped source and the target in the feature space provided by PointNet. This difference is an error metric representing how well the two point clouds are aligned.
        Parameter Update: Use the Lucas-Kanade method to update the transformation parameters to minimize this error. This typically involves solving a linear system where the solution gives the best update to the parameters under the least squares criterion.

- Convergence Check:

    Termination Criteria: Check if the transformation parameters have converged (e.g., changes are below a certain threshold) or if a maximum number of iterations has been reached.
    Output: If converged, return the final transformation parameters that best align the source to the target.

- Transformation Application:

    Apply Final Transformation: Use the final estimated transformation to warp the source point cloud fully into the coordinate system of the target point cloud.

```
Function PointNetLK(PointCloud_source, PointCloud_target):
    // PointCloud_source: The source point cloud to be aligned
    // PointCloud_target: The target point cloud

    // Step 1: Feature Extraction
    Features_source := PointNet(PointCloud_source)
    Features_target := PointNet(PointCloud_target)

    Initialize transformation_parameters to an identity transformation

    // Step 2: Iterative Alignment
    While not converged and within iteration limits:
        // Warp the source point cloud with the current estimate of the transformation
        Warped_PointCloud_source := Apply_Transformation(PointCloud_source, transformation_parameters)

        // Extract features of the warped source point cloud
        Features_warped_source := PointNet(Warped_PointCloud_source)

        // Step 3: Compute the error in feature space
        Error := Features_target - Features_warped_source

        // Step 4: Update the transformation parameters using the Lucas-Kanade method
        transformation_parameters := Update_Parameters_LK(Features_warped_source, Error, transformation_parameters)

        // Check for convergence
        If parameters have converged:
            break

    // Step 5: Apply the final transformation to the source point cloud
    Final_aligned_source := Apply_Transformation(PointCloud_source, transformation_parameters)

    Return Final_aligned_source, transformation_parameters
```

Notes:

    PointNet Architecture: The specific architecture of PointNet can vary based on the version used and the specific task requirements. It typically involves several layers of point-wise MLPs (multi-layer perceptrons), a max pooling layer for feature aggregation, and fully connected layers for further processing.
    Transformation Model: The choice of transformation model (e.g., rigid, affine) will affect the form of the Apply_Transformation and Update_Parameters_LK functions. For 3D registration, a rigid or affine transformation is commonly used.
    Convergence Criteria: Common criteria include small changes in the error metric or transformation parameters and reaching a maximum number of iterations.
    
### Generating synthetic data 

``python .\Princeton3DMatchDataGenerator\main.py register N # N is the number of point cloud pairs with corresponding grouth tranformation matrices``

### Training the PointNetLK on synthetic data

``python .\ModelGenerator\train_pointnetlk.py --h5-file .\output\output_data.h5 --outfile output --num-points 1024 --epochs 100 --batch-size 1`` 

## PointNet based segmentation

### Generating synthetic data

``python .\Stanford3DSemanticDataGenerator\main.py extract N # N -- number of point clouds with corresponding labels``

