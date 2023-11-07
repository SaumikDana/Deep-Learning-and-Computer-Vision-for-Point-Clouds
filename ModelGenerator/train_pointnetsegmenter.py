__author__ = "Saumik"
__date__ = "11/07/2023"
__purpose__ = "Test out the PointNet segmenter on ModelNet10 Dataset with feature vector visualizer"
__notes__ = "Segmenter is the sum total of what PointNet has to offer"

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from your_dataset_class import YourDatasetClass  # You need to implement this

# Parameters
num_classes = 10  # Update with the number of classes in ModelNet10
batch_size = 32
learning_rate = 0.001
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
transform = transforms.Compose([
    # Add any required transforms here
])
train_dataset = YourDatasetClass(root_dir='path_to_modelnet10', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model initialization
pointnet_feat = PointNet_features(dim_k=1024, use_tnet=True)
pointnet_segmenter = PointNet_segmenter(num_c=num_classes, ptfeat=pointnet_feat, dim_k=1024).to(device)

# Loss function and optimizer
optimizer = torch.optim.Adam(pointnet_segmenter.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    pointnet_segmenter.train()
    total_loss = 0
    for data in train_loader:
        points, target = data
        points, target = points.to(device), target.to(device)

        # Forward pass
        pred = pointnet_segmenter(points)
        trans_feat = pointnet_feat.t_out_t2  # Assuming you want to use the second T-Net's transformation matrix
        loss = pointnet_segmenter.loss(pred, target, trans_feat)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Optional: Evaluation on validation set

# Save the model
torch.save(pointnet_segmenter.state_dict(), 'pointnet_segmenter.pth')
