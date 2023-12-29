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
Function Lucas_Kanade(I1, I2, window_size):
    // I1: First image frame (at time t)
    // I2: Second image frame (at time t+1)
    // window_size: Size of the window to consider around each pixel

    Initialize flow_vectors as an empty list or matrix to store the flow vectors for each pixel

    // Iterate through each pixel in the first image
    For each pixel (x, y) in I1:
        // Define a small window around the current pixel in the first image
        W := window centered at (x, y) of size window_size in I1

        // Compute the image gradients within the window:
        // Ix and Iy are the spatial gradients (change of image intensity in x and y directions)
        // It is the temporal gradient (change of intensity over time, or between frames)
        Ix := gradient of W in the x-direction
        Iy := gradient of W in the y-direction
        It := difference between the window W in I1 and the same window in I2

        // Construct matrices A and b from the gradients for the least squares solution:
        // A is composed of the gradients Ix and Iy for all points in the window,
        // representing the change in image intensity for each point in x and y directions.
        // b is composed of the negative temporal gradient -It for all points,
        // representing the change in image intensity over time.
        A := [Ix, Iy]  // A 2-column matrix where each row is [Ix_i, Iy_i] for each pixel i in the window
        b := [-It]     // b is a column vector where each element is -It_i for each pixel i in the window

        // Solve for the velocity (v_x, v_y) using the least squares method:
        // The equation A'Av = A'b arises from minimizing the error between the observed
        // intensities and the intensities predicted by the flow velocities.
        // This gives us a 2D vector representing the apparent motion at (x, y)
        // v_x is the horizontal motion, and v_y is the vertical motion.
        v := Inverse(A' * A) * A' * b

        // Store the computed flow vector for the current pixel
        // The flow vector represents the estimated motion of the pixel between the two frames.
        flow_vectors[x, y] := v

    // Return the matrix of flow vectors, representing the estimated motion for each pixel
    Return flow_vectors
```

### Generating synthetic data 

``python .\Princeton3DMatchDataGenerator\main.py register N # N is the number of point cloud pairs with corresponding grouth tranformation matrices``

### Training the PointNetLK on synthetic data

``python .\ModelGenerator\train_pointnetlk.py --h5-file .\output\output_data.h5 --outfile output --num-points 1024 --epochs 100 --batch-size 1`` 

## PointNet based segmentation

### Generating synthetic data

``python .\Stanford3DSemanticDataGenerator\main.py extract N # N -- number of point clouds with corresponding labels``

