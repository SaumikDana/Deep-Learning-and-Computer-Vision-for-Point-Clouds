import torch
from pointnet import mlp_layers

class PointNet_classifier(torch.nn.Module):
    # The PointNet_classifier is a neural network module designed for classifying point cloud data.

    def __init__(self, num_c, ptfeat, dim_k):
        # Initialize the PointNet_classifier module.
        super().__init__()
        # 'num_c' is the number of classes for classification.
        # 'ptfeat' is an instance of the PointNet_features class, which will be used to extract features from the point cloud.
        # 'dim_k' is the dimension of the output feature vector from the PointNet_features module.

        # Store the PointNet_features module.
        self.features = ptfeat

        # Create a list of layers for the classifier using the helper function 'mlp_layers'.
        # This will create a Multi-Layer Perceptron (MLP) with layers of size 'dim_k', 512, and 256.
        # 'b_shared' is set to False, meaning the layers do not share weights across the point cloud.
        # 'bn_momentum' sets the momentum for the batch normalization layers.
        # 'dropout' is set to 0.0, meaning no dropout is applied.
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        
        # Append a final Linear layer that maps the 256-dimensional features to the number of classes 'num_c'.
        list_layers.append(torch.nn.Linear(256, num_c))
        
        # Combine the list of layers into a Sequential module to create the classifier.
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        # The forward pass of the classifier.

        # Extract features from the input point cloud using the PointNet_features module.
        feat = self.features(points)
        
        # Pass the extracted features through the classifier to get the output logits for each class.
        out = self.classifier(feat)
        
        # Return the output logits.
        return out

    def loss(self, out, target, w=0.001):
        # Calculate the loss for the classifier.

        # Calculate the classification loss using negative log likelihood loss.
        # 'log_softmax' is applied to the output logits to obtain log probabilities, which are used with 'nll_loss'.
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)

        # Get the transformation matrix from the T-Net (if used) in the PointNet_features module.
        t2 = self.features.t_out_t2
        
        # If the T-Net is not used or the regularization weight 'w' is 0, return the classification loss only.
        if (t2 is None) or (w == 0):
            return loss_c

        # Calculate the regularization loss for the transformation matrix.
        # This encourages the transformation matrix to be close to an orthogonal matrix, which helps prevent overfitting.
        batch = t2.size(0)
        K = t2.size(1)  # [B, K, K]
        
        # Create an identity matrix of size K, repeated for each item in the batch.
        I = torch.eye(K).repeat(batch, 1, 1).to(t2)
        
        # Multiply the transformation matrix by its transpose to get matrix A.
        A = t2.bmm(t2.transpose(1, 2))
        
        # Calculate the mean squared error between matrix A and the identity matrix.
        # This represents how far the transformation matrix is from being orthogonal.
        loss_m = torch.nn.functional.mse_loss(A, I, size_average=False)
        
        # Combine the classification loss and the regularization loss into the total loss.
        loss = loss_c + w * loss_m
        
        # Return the total loss.
        return loss
