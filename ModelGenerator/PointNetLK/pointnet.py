""" PointNet
    References.
        .. [1] Charles R. Qi, Hao Su, Kaichun Mo and Leonidas J. Guibas,
        "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation",
        CVPR (2017)
"""

import torch

def symfn_max(x):
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a

def symfn_avg(x):
    a = torch.nn.functional.avg_pool1d(x, x.size(-1))
    return a

def flatten(x):
    return x.view(x.size(0), -1)

class PointNet_features(torch.nn.Module):
    """
    PointNet_features is a PyTorch module for extracting features from point cloud data.
    It can optionally include T-Net modules for spatial transformer networks.
    """

    def __init__(self, dim_k=32, use_tnet=False, sym_fn=symfn_max, scale=1):
        """
        Initialize the PointNet_features module.

        Parameters:
        dim_k (int): The dimension of the output feature vector for each point.
        use_tnet (bool): Flag to determine if T-Net modules should be used for input and feature transformations.
        sym_fn (callable): The symmetry function to be used for aggregating point features.
        scale (float): A scaling factor for the number of neurons in each layer.
        """
        super().__init__()
        # Print the status of T-Net usage
        print("use_tnet is set to:", use_tnet)  

        # Define the number of output channels for the first MLP layer, scaled by the 'scale' factor.
        mlp_h1 = [int(8/scale)]  

        # Define the number of output channels for the second MLP layer, including the final output dimension 'dim_k'.
        mlp_h2 = [int(16/scale), int(dim_k/scale)]  

        # The first set of layers in the network, consisting of a convolution, batch normalization, and ReLU activation.
        # This set processes the input point cloud data, applying shared weights across all points (hence, Conv1d).
        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers

        # The second set of layers in the network, which further processes the data from the first set of layers.
        # It includes two sets of convolution, batch normalization, and ReLU activation layers.
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers

        # The symmetry function for aggregating features from all points into a global feature vector.
        # Commonly, a max pooling operation is used as the symmetry function.
        self.sy = sym_fn

        # Optional T-Net modules for aligning the input point cloud and the features.
        # T-Net modules are small PointNet networks that regress to spatial transformation matrices.
        self.tnet1 = TNet(3) if use_tnet else None  # T-Net for input transformation
        self.tnet2 = TNet(mlp_h1[-1]) if use_tnet else None  # T-Net for feature transformation

        # Placeholders for the output transformations of the T-Nets.
        # These will store the transformation matrices produced by the T-Nets if they are used.
        self.t_out_t2 = None  # Transformation matrix from the second T-Net
        self.t_out_h1 = None  # Transformation matrix from the first T-Net

    def forward(self, points):
        # Transpose the input points to match the expected input dimensions for Conv1d layers.
        # The expected format is (batch_size, channels, number_of_points).
        x = points.transpose(1, 2)  # [B, 3, N]

        # If the T-Net for input transformation is used, apply it to the input points.
        if self.tnet1:
            # The T-Net returns a transformation matrix 't1' for the input points.
            t1 = self.tnet1(x)
            # Apply the transformation matrix to the input points using batch matrix multiplication.
            x = t1.bmm(x)

        # Pass the (possibly transformed) points through the first set of MLP layers.
        x = self.h1(x)

        # If the T-Net for feature transformation is used, apply it to the features after the first MLP.
        if self.tnet2:
            # The T-Net returns a transformation matrix 't2' for the features.
            t2 = self.tnet2(x)
            # Store the transformation matrix for later use or analysis.
            self.t_out_t2 = t2
            # Apply the transformation matrix to the features using batch matrix multiplication.
            x = t2.bmm(x)

        # Store the local features after the first MLP and optional T-Net transformation for later use or analysis.
        self.t_out_h1 = x  # local features

        # Pass the features through the second set of MLP layers.
        x = self.h2(x)

        # Apply the symmetry function to the features across all points to get a global feature vector.
        # This is typically a max pooling operation that aggregates features to be invariant to input permutations.
        x = flatten(self.sy(x))

        # Return the global feature vector which can be used for further processing or as input to a classifier.
        return x
    
class TNet(torch.nn.Module):
    """
    TNet (Transformation Network) module in PyTorch.
    
    This module is a mini-PointNet that regresses a spatial transformation matrix
    for aligning points in a point cloud. It is used within the larger PointNet
    architecture to learn an optimal spatial transformer for the input data.
    
    The TNet outputs a [B, K, K] tensor representing a batch of transformation
    matrices, where B is the batch size and K is the spatial dimension of the
    input data (e.g., K=3 for 3D point clouds).
    """

    def __init__(self, K):
        """
        Initialize the TNet module.

        Parameters:
        K (int): The size of the spatial dimension for the input and output transformation matrices.
        """
        super().__init__()  # Initialize the superclass (Module)

        # Define the first multi-layer perceptron (MLP) which operates on shared features.
        # This MLP transforms the input data [B, K, N] to a higher-dimensional space [B, 1024, N].
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True))
        
        # Define the second MLP which operates on global features.
        # This MLP further processes the data to [B, 256].
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False))
        
        # A final fully connected layer that outputs [B, K*K] elements, which will be reshaped
        # into a [B, K, K] transformation matrix.
        self.lin = torch.nn.Linear(256, K*K)

        # Initialize the weights of the MLPs and the linear layer to zero.
        # This is a peculiar choice and may be changed based on the specific requirements
        # or empirical results.
        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp):
        """
        Forward pass of the TNet.

        Parameters:
        inp (tensor): The input tensor of shape [B, K, N], where B is the batch size,
                      K is the spatial dimension, and N is the number of points.

        Returns:
        x (tensor): The output tensor of shape [B, K, K], representing a batch of
                    transformation matrices.
        """
        # Get the spatial dimension (K) and the number of points (N) from the input tensor.
        K = inp.size(1)
        N = inp.size(2)
        
        # Create an identity matrix of shape [1, K, K] and replicate it for the whole batch.
        # This will be added to the output to initialize the transformation as an identity matrix.
        eye = torch.eye(K).unsqueeze(0).to(inp.device) # [1, K, K]

        # Pass the input through the first MLP.
        x = self.mlp1(inp)
        
        # Apply max pooling across the points to get a global feature vector.
        x = torch.nn.functional.max_pool1d(x, N)
        
        # Flatten the output for the next fully connected layers.
        x = x.view(-1, 1024)
        
        # Pass the global feature vector through the second MLP.
        x = self.mlp2(x)
        
        # Pass the output through the final linear layer to get the elements of the transformation matrix.
        x = self.lin(x)
        
        # Reshape the output to a batch of [B, K, K] transformation matrices.
        x = x.view(-1, K, K)
        
        # Add the identity matrix to the output, which initializes the transformation as an identity
        # transformation with learned adjustments.
        x = x + eye
        
        # Return the final transformation matrix.
        return x

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return flatten(x)

def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """
    Create a list of layers for a multi-layer perceptron (MLP) network.

    Parameters:
    nch_input (int): Number of input channels.
    nch_layers (list of int): List of output channels for each layer in the MLP.
    b_shared (bool): If True, use shared MLPs (Conv1d), otherwise use fully connected layers (Linear).
    bn_momentum (float): Momentum for the batch normalization layers.
    dropout (float): Dropout rate for dropout layers. Only applied if b_shared is False.

    Returns:
    layers (list of nn.Module): List of layers comprising the MLP.
    """
    # Initialize an empty list to store the layers.
    layers = []
    
    # 'last' keeps track of the number of output channels from the previous layer.
    last = nch_input
    
    # Iterate over the number of output channels for each layer specified in 'nch_layers'.
    for i, outp in enumerate(nch_layers):
        # If b_shared is True, use Conv1d for shared MLPs across the point cloud.
        if b_shared:
            # Conv1d layer with kernel size 1 (acts as a per-point fully connected layer).
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            # Fully connected layer (Linear) for non-shared MLPs.
            weights = torch.nn.Linear(last, outp)
        
        # Add the created layer (Conv1d or Linear) to the layers list.
        layers.append(weights)
        
        # Add a BatchNorm1d layer to normalize the activations from the previous layer.
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        
        # Add a ReLU activation function to introduce non-linearity.
        layers.append(torch.nn.ReLU())
        
        # If b_shared is False and dropout is specified, add a Dropout layer.
        if not b_shared and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        
        # Update 'last' to the number of output channels of the current layer.
        last = outp
    
    # Return the list of layers.
    return layers

class MLPNet(torch.nn.Module):
    """
    Multi-layer perceptron (MLP) network module in PyTorch.
    
    This module can be used to create an MLP that operates on either point cloud data
    (where the MLP is shared across points) or on standard vector data.
    
    The MLP consists of a sequence of layers, each comprising a linear transformation
    (fully connected layer or 1D convolution), batch normalization, and a ReLU activation.
    Optionally, dropout can be added after the ReLU activation for regularization.
    """

    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        """
        Initialize the MLPNet module.

        Parameters:
        nch_input (int): Number of input channels.
        nch_layers (list of int): List specifying the number of output channels for each layer.
        b_shared (bool): If True, use shared MLPs (Conv1d), otherwise use fully connected layers (Linear).
        bn_momentum (float): Momentum for the batch normalization layers.
        dropout (float): Dropout rate for dropout layers. Only applied if b_shared is False.
        """
        super().__init__()  # Initialize the superclass (Module)

        # Create the list of layers using the utility function 'mlp_layers'
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        
        # The Sequential container will process the input through all layers in order
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        """
        Forward pass of the MLPNet.

        Parameters:
        inp (tensor): The input tensor. Shape [B, Cin, N] for point cloud data or [B, Cin] for vector data.

        Returns:
        out (tensor): The output tensor after passing through the MLP. Shape [B, Cout, N] or [B, Cout].
        """
        # Pass the input through the sequential container of layers
        out = self.layers(inp)
        
        # Return the output which is the transformed input after going through the MLP
        return out

