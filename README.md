## Generating synthetic data for PointNetLK based registration
python .\Princeton3DMatchDataGenerator\main.py register N

N -- number of point cloud pairs with corresponding grouth tranformation matrices

### Training the PointNetLk on synthetic data
python .\ModelGenerator\train_pointnetlk.py --h5-file .\output\output_data.h5 --outfile output --num-points 1024 --epochs 100 --batch-size 1 

## Generating synthetic data for PointNet based segmentation
python .\Stanford3DSemanticDataGenerator\main.py extract N

N -- number of point clouds with corresponding labels

