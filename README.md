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

```
Function Lucas_Kanade_Registration(I_fixed, I_moving, transformation_model):
    // I_fixed: The fixed image that remains constant
    // I_moving: The moving image that needs to be registered to the fixed image
    // transformation_model: The model of transformation (e.g., affine, projective)

    Initialize parameters for the transformation_model

    // Create image pyramids for a multi-resolution approach
    pyramid_fixed := Create_Image_Pyramid(I_fixed)
    pyramid_moving := Create_Image_Pyramid(I_moving)

    // Iterate from the coarsest to the finest resolution
    For level from max_level down to 1:
        // Warp the moving image at the current level using the current estimate of parameters
        warped_moving := Warp_Image(pyramid_moving[level], parameters)

        // Compute the difference between the fixed image and the warped moving image
        difference := pyramid_fixed[level] - warped_moving

        // Use Lucas-Kanade to estimate the parameters that minimize this difference
        // Update the parameters based on the estimated motion
        parameters := Update_Parameters_Lucas_Kanade(difference, parameters, level)

    // After all levels are processed, the parameters should align the moving image to the fixed image
    Return Warp_Image(I_moving, parameters)

Function Create_Image_Pyramid(I, max_levels):
    // I: Input image
    // max_levels: Number of pyramid levels to create

    Initialize pyramid as an empty list
    current_level_image := I

    // Generate each level of the pyramid by repeatedly smoothing and downsampling the image
    For level from 1 to max_levels:
        pyramid.append(current_level_image)
        current_level_image := Downsample(Smooth(current_level_image))

    Return pyramid

Function Warp_Image(I, parameters):
    // I: Image to be warped
    // parameters: Parameters of the transformation model

    // Apply the transformation to each pixel in the image according to the parameters
    // This could be a complex operation depending on the transformation model
    Return transformed_image

Function Update_Parameters_Lucas_Kanade(difference, parameters, level):
    // difference: Difference between the fixed image and the warped moving image
    // parameters: Current estimate of the parameters
    // level: Current level of the pyramid

    // Calculate the gradient of the difference image
    gradient := Calculate_Gradient(difference)

    // Estimate the motion and update the parameters accordingly
    // This could involve solving a system of equations or optimization problem
    delta_parameters := Estimate_Motion(gradient, difference)

    // Update parameters with the estimated change
    new_parameters := parameters + delta_parameters

    Return new_parameters

Function Estimate_Motion(gradient, difference):
    // gradient: Gradient of the difference image
    // difference: Difference between the fixed image and the warped moving image

    // Implement the core of Lucas-Kanade here to estimate the motion
    // This might involve setting up and solving a least squares problem
    // The exact implementation will depend on the chosen transformation model

    Return estimated_motion

// Note: Smooth and Downsample functions are assumed to be standard image processing functions.
```

### Generating synthetic data 

``python .\Princeton3DMatchDataGenerator\main.py register N # N is the number of point cloud pairs with corresponding grouth tranformation matrices``

### Training the PointNetLK on synthetic data

``python .\ModelGenerator\train_pointnetlk.py --h5-file .\output\output_data.h5 --outfile output --num-points 1024 --epochs 100 --batch-size 1`` 

## PointNet based segmentation

### Generating synthetic data

``python .\Stanford3DSemanticDataGenerator\main.py extract N # N -- number of point clouds with corresponding labels``

