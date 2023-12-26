## Directory Structure

```
DL_CV_Images/
│
├── .vscode/
│   └── launch.json
│
├── ModelGenerator/
│   ├── Action_pointnetlk.py
│   ├── PointNetLK/
│   │   ├── __init__.py
│   │   ├── invmat.py
│   │   ├── pointnet.py
│   │   ├── pointnet_classifier.py
│   │   ├── pointnet_segmenter.py
│   │   ├── pointnetlk.py
│   │   ├── se3.py
│   │   ├── sinc.py
│   │   ├── so3.py
│   │   └── __pycache__/
│   ├── Settings_pointnetlk.py
│   ├── Trainer_pointnetlk.py
│   ├── Utils_pointnetlk.py
│   ├── train_pointnetlk.py
│   ├── train_pointnetsegmenter.py
│   └── __pycache__/
│
├── ModelNet10/
│   ├── PointCloudExtractor.py
│   ├── main.py
│   └── __pycache__/
│
├── Princeton3DMatchDataGenerator/
│   ├── PointCloudExtractor.py
│   ├── RegistrationPipeline.py
│   ├── TransformationEstimator.py
│   ├── main.py
│   └── __pycache__/
│
├── SimpleITK/
│   ├── non-rigid_registration.py
│   └── rigid_registration.py
│
├── scripts/
│   ├── check_mat.py
│   ├── check_npz.py
│   ├── data/
│   │   └── MNIST/raw/
│   ├── encoding_patterns.py
│   ├── failure_cnn_1.py
│   ├── failure_cnn_2.py
│   ├── generate_call_graphs.py
│   ├── generate_call_stacks.py
│   ├── marching_patterns.py
│   ├── pareto_chart.py
│   ├── print_keys.py
│   ├── show_and_tell.py
│   ├── test_npz.py
│   ├── visualize_deformation_patterns.py
│   └── visualize_fpfh.py
│
├── .gitattributes
├── Approach.png
├── README.md
├── __pycache__/
├── poetry.lock
└── pyproject.toml
```

## Generating synthetic data for PointNetLK based registration

``python .\Princeton3DMatchDataGenerator\main.py register N``

N -- number of point cloud pairs with corresponding grouth tranformation matrices

### Training the PointNetLk on synthetic data

``python .\ModelGenerator\train_pointnetlk.py --h5-file .\output\output_data.h5 --outfile output --num-points 1024 --epochs 100 --batch-size 1`` 

## Generating synthetic data for PointNet based segmentation

``python .\Stanford3DSemanticDataGenerator\main.py extract N``

N -- number of point clouds with corresponding labels

