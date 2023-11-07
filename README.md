## Usage

### Generating synthetic data for PointNetLK based registration
python .\Princeton3DMatchDataGenerator\main.py register N

N -- number of point cloud pairs with corresponding grouth tranformation matrices

### Training the PointNetLk on synthetic data
python .\ModelGenerator\train_pointnetlk.py --h5-file .\output\output_data.h5 --outfile output --num-points 1024 --epochs 100 --batch-size 1 

### Generating synthetic data for PointNet based segmentation
python .\Stanford3DSemanticDataGenerator\main.py extract N

N -- number of point clouds with corresponding labels

# Lucas-Kanade Optical Flow Algorithm

## Overview
The Lucas-Kanade method is a well-known optical flow estimation technique. It assumes that the flow is essentially constant in a local neighborhood of the pixel under consideration, and solves the basic optical flow equations for all pixels in that neighborhood by the least squares criterion.

## Algorithm Pseudocode

```plaintext
Lucas-Kanade Algorithm Pseudocode:

Given two frames I1 and I2 of a video sequence:

For each point p in the first frame I1:
    1. Take a window W around the point p.
    2. Compute the image gradients Ix, Iy, and It within W across I1 and I2.
    3. Formulate the matrix A and vector b from Ix, Iy, and It:
        A = [ Ix1 Iy1 ]
            [ Ix2 Iy2 ]
            [ ... ... ]
            [ Ixn Iyn ]
        b = [ It1 ]
            [ It2 ]
            [ ... ]
            [ Itn ]
    4. Solve for v in the equation A^T * A * v = A^T * b, where v = [vx, vy]^T is the velocity (optical flow) at p.
    5. If the system has no solution or is underdetermined, discard the point p or use regularization.

Repeat the above steps for all points of interest in I1.

The result is a vector field where each vector represents the estimated motion of a point from frame I1 to frame I2.

```

# PointNet with Lucas-Kanade Optical Flow Estimation

## Overview
This project extends the capabilities of PointNet, a deep neural network for processing point clouds, by integrating the Lucas-Kanade optical flow algorithm. This combination allows for the estimation of motion in a sequence of 3D point cloud frames, providing insights into the dynamics of the observed scene.

## Lucas-Kanade Algorithm Adaptation for Point Clouds

The Lucas-Kanade method, traditionally used for 2D optical flow estimation in image sequences, has been adapted here for use with 3D point clouds processed by PointNet. The algorithm assumes that the flow is constant in a local neighborhood around each point and solves for the flow vectors using a least squares approach.

## Algorithm Pseudocode

```plaintext
Lucas-Kanade for Point Clouds Pseudocode:

Given two consecutive point cloud frames P1 and P2:

For each point p in frame P1:
    1. Define a local neighborhood N around point p.
    2. Compute the local gradients with respect to x, y, z, and time (t) within N for P1 and P2.
    3. Assemble the matrix A and vector b using the computed gradients:
        A = [ dx1 dy1 dz1 ]
            [ dx2 dy2 dz2 ]
            [ ... ... ... ]
            [ dxn dyn dzn ]
        b = [ dt1 ]
            [ dt2 ]
            [ ... ]
            [ dtn ]
    4. Solve for the velocity vector v in the equation A^T * A * v = A^T * b, where v = [vx, vy, vz]^T.
    5. If the system is underdetermined or has no solution, apply regularization or discard point p.

Repeat for all points in P1 to obtain a flow field for the point cloud.

The result is a 3D vector field where each vector represents the estimated motion of a point from P1 to P2.

```

## RANSAC Process Comparison: With and Without FPFH

### 1. Random Sampling

#### Without FPFH
- Points are randomly selected directly from the point clouds.
- Selection is often based on spatial proximity or other simple heuristics.
- The selected points might not represent distinctive features of the point clouds, leading to potential inaccuracies.

#### With FPFH
- Random sampling involves selecting matches based on FPFH feature correspondences.
- FPFH vectors represent the local geometric properties around each point, leading to more meaningful selections.
- This approach is more likely to include points that represent significant features of the objects, improving the robustness of the correspondences.

### 2. Estimate Transformation

#### Without FPFH
- The transformation is estimated to align the randomly selected pairs of points based on their coordinates.
- This approach can be sensitive to noise and outliers, as it relies solely on the spatial arrangement of points.

#### With FPFH
- The transformation is estimated based on the alignment of feature correspondences (FPFH vectors).
- This method is more robust against noise and outliers since it considers the local geometric context of each point.

### 3. Apply Transformation and Measure Consensus

#### Without FPFH
- The transformation is applied to one of the point clouds.
- The quality of alignment is assessed by measuring the distances between corresponding points in the two clouds.
- Misalignments can occur if the randomly selected points are not representative of the overall shapes of the objects.

#### With FPFH
- Similarly, the transformation is applied, but the quality of alignment is assessed based on the agreement of the FPFH-based correspondences.
- This method typically leads to more accurate alignments since it accounts for the shape and geometry of the surfaces, not just point locations.

### 4. Iterate and Optimize

#### Without FPFH
- The process is repeated with different sets of randomly selected points.
- The best transformation is the one that maximizes the alignment of the point clouds.
- The result might vary significantly with each iteration due to the reliance on random point selections.

#### With FPFH
- Iteration involves selecting different sets of FPFH-based correspondences.
- The optimal transformation is the one that achieves the best overall alignment according to the FPFH features.
- This approach tends to be more consistent and reliable across iterations.

### Summary
- **Without FPFH:** Relying solely on point coordinates, this approach is simpler but can be less accurate and more susceptible to noise and outliers.
- **With FPFH:** By utilizing feature-based correspondences, this method is more robust, accurate, and better suited for complex or noisy environments. The geometric context provided by FPFH leads to more reliable correspondences and, consequently, a more accurate final transformation.
