import torch
from pointnet import mlp_layers

class PointNet_segmenter(torch.nn.Module):
    """
    PointNet_segmenter is a PyTorch module for performing semantic segmentation on point cloud data.
    """

    def __init__(self, num_c, ptfeat, dim_k):
        """
        Initialize the PointNet_segmenter module.

        """
        super().__init__()
        self.features = ptfeat

        # Define the segmentation network layers using the mlp_layers function
        # Since we are dealing with point cloud data, we set b_shared to True to use Conv1d layers
        segmentation_layers = mlp_layers(dim_k, [dim_k, dim_k // 2, num_c], b_shared=True)
        self.segmentation_net = torch.nn.Sequential(*segmentation_layers)

    def forward(self, points):
        """
        Forward pass of the PointNet_segmenter.

        Parameters:
        points (tensor): The input point cloud data.

        Returns:
        out (tensor): The class scores for each point in the point cloud.
        """
        # Extract features for each point
        feat = self.features(points)

        # Perform segmentation on the features
        # The segmentation network will output a score for each class for each point
        out = self.segmentation_net(feat)

        # Transpose the output to have the same shape as the input points
        # This is necessary because the convolution layers expect the feature dimension to be the second dimension
        out = out.transpose(2, 1)

        return out

    def loss(self, pred, target, trans_feat, weight=0.001):
        """
        Compute the loss for the segmentation task.

        Parameters:
        pred (tensor): The predicted class scores for each point.
        target (tensor): The ground truth labels for each point.
        trans_feat (tensor): The transformation matrix from the T-Net.
        weight (float): The weight for the regularization loss.

        Returns:
        loss (tensor): The computed loss value.
        """
        # Calculate the segmentation loss using cross-entropy loss
        loss = torch.nn.functional.cross_entropy(pred, target)

        # Regularization term for the transformation matrix to encourage orthogonality
        mat_diff = torch.bmm(trans_feat, trans_feat.transpose(1, 2)) - torch.eye(trans_feat.size(1)).to(pred.device)
        reg_loss = torch.norm(mat_diff) * weight

        # Combine the losses
        total_loss = loss + reg_loss

        return total_loss
