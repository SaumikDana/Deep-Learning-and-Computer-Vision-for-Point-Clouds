__author__ = "Saumik"
__date__ = "11/03/2023"

import PointNetLK as ptlk
import torch

class Action:
    """
    The Action class defines the operations for training and evaluating a PointNet-LK model for point cloud registration.
    """
    def __init__(self, args):
        """
        Initializes the Action object with configuration settings.

        :param args: A namespace or dictionary containing configuration settings.
        """
        # Dimension of the feature vector produced by PointNet.
        self.dim_k = args.dim_k

        # The symmetric function to be used in PointNet for aggregating features from all points.
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg

        # The step size for the approximate Jacobian in the Lucas-Kanade method.
        self.delta = args.delta

        # A flag indicating whether the step size delta should be learned during training.
        self.learn_delta = args.learn_delta

        # The maximum number of iterations for the Lucas-Kanade optimization loop.
        self.max_iter = args.max_iter

        # A tolerance value for the Lucas-Kanade optimization.
        self.xtol = 1.0e-7

        # Flags indicating whether to subtract the mean from the point clouds before processing.
        self.p0_zero_mean = True
        self.p1_zero_mean = True

        # A variable to determine the type of loss function used during training.
        self._loss_type = 1

    def create_model(self):
        """
        Creates and returns a PointNet-LK model.

        :return: A PointNet-LK model.
        """
        # Create the PointNet features model.
        ptnet = ptlk.pointnet.PointNet_features(self.dim_k, use_tnet=False, sym_fn=self.sym_fn)
        
        # Create and return the PointNet-LK model using the PointNet features.
        ptnetlk = ptlk.pointnetlk.PointNetLK(ptnet, self.delta, self.learn_delta)

        return ptnetlk

    def train(self, model, trainloader, optimizer, device):
        """
        Trains the model for one epoch.

        :param model: The PointNet-LK model.
        :param trainloader: DataLoader for the training dataset.
        :param optimizer: The optimizer used for training.
        :param device: The device (CPU or GPU) to run the training on.
        :return: The average loss and geometric loss for the epoch.
        """
        print("Setting model to training mode...")
        model.train()
        
        # Initialize variables to accumulate losses.
        vloss = 0.0
        gloss = 0.0
        count = 0
        
        print("Starting training epoch...")
        # Iterate over batches of data.
        for i, data in enumerate(trainloader):
            print(f'===========================================')
            print(f"Processing batch {i + 1}/{len(trainloader)}")
            print(f'===========================================')
            
            # Compute the loss for the current batch.
            loss, loss_g = self.compute_loss(model, data, device)
            print(f"Batch {i + 1} - Loss: {loss.item()}, Geometric Loss: {loss_g.item()}")
            
            # Zero the gradients before backpropagation.
            optimizer.zero_grad()
            
            # Backpropagate the loss.
            loss.backward()

            # Apply gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update the model parameters.
            optimizer.step()
            
            # Accumulate the losses.
            vloss += loss.item()
            gloss += loss_g.item()
            count += 1
        
        # Calculate the average losses.
        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count
        
        print(f"Epoch completed. Average Loss: {ave_vloss}, Average Geometric Loss: {ave_gloss}\n")
        return ave_vloss, ave_gloss

    def eval(self, model, testloader, device):
        """
        Evaluates the model on the test dataset.

        :param model: The PointNet-LK model.
        :param testloader: DataLoader for the test dataset.
        :param device: The device (CPU or GPU) to run the evaluation on.
        :return: The average loss and geometric loss for the evaluation.
        """
        # Set the model to evaluation mode.
        model.eval()
        
        # Initialize variables to accumulate losses.
        vloss = 0.0
        gloss = 0.0
        count = 0
        
        # Disable gradient computation for evaluation.
        with torch.no_grad():
            # Iterate over batches of data.
            for i, data in enumerate(testloader):
                # Compute the loss for the current batch.
                loss, loss_g = self.compute_loss(model, data, device)
                
                # Accumulate the losses.
                vloss += loss.item()
                gloss += loss_g.item()
                count += 1
        
        # Calculate the average losses.
        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count
        
        return ave_vloss, ave_gloss

    def compute_loss(self, model, data, device):
        """
        Computes the loss for a batch of data using the PointNet-LK model.

        Parameters:
        model: The PointNet-LK model to be used for computing the loss.
        data: A dictionary containing the batch of data with source points, target points, and the ground truth transformation matrix.
        device: The device (CPU or GPU) on which to perform the computations.

        Returns:
        The total loss and geometric loss for the batch.
        """
        # Log the process of extracting data
        print("Extracting data and moving to device...")
        # Extract source points from the data and move them to the specified device (CPU/GPU)
        source_points = data['source_points'].to(device)
        # Extract target points from the data and move them to the specified device (CPU/GPU)
        target_points = data['target_points'].to(device)
        # Extract the ground truth transformation matrix from the data and move it to the specified device (CPU/GPU)
        transformation_matrix = data['transformation_matrix'].to(device)
        
        # Log the forward pass process
        print("Performing forward pass...")
        # Perform the forward pass of the model to align the source points to the target points
        # and get the residual vector 'r' which indicates the difference between aligned and target points
        r = ptlk.pointnetlk.PointNetLK.do_forward(model, source_points, target_points, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean)
        
        # Log the computation of the estimated transformation matrix
        print("Computing estimated transformation matrix...")
        # Retrieve the estimated transformation matrix from the model after the forward pass
        est_g = model.g
        
        # Log the computation of geometric loss
        print("Computing geometric loss...")
        # Compute the geometric loss which measures the discrepancy between the estimated transformation matrix and the ground truth
        loss_g = ptlk.pointnetlk.PointNetLK.comp(est_g, transformation_matrix)
        # Log the geometric loss value
        print(f"Geometric Loss: {loss_g.item()}")
        
        # Log the computation of total loss
        print("Computing total loss...")
        # Compute the total loss based on the specified loss type
        if self._loss_type == 0:
            # If loss type is 0, use only the residual loss
            loss_r = ptlk.pointnetlk.PointNetLK.rsq(r)
            loss = loss_r
        elif self._loss_type == 1:
            # If loss type is 1, use the sum of residual loss and geometric loss
            loss_r = ptlk.pointnetlk.PointNetLK.rsq(r)
            loss = loss_r + loss_g
        elif self._loss_type == 2:
            # If loss type is 2, use the change in residual plus geometric loss
            # This involves computing the difference between the current and previous residuals if the previous residual exists
            pr = model.prev_r
            loss_r = ptlk.pointnetlk.PointNetLK.rsq(r - pr) if pr is not None else ptlk.pointnetlk.PointNetLK.rsq(r)
            loss = loss_r + loss_g
        else:
            # Otherwise, use only the geometric loss
            loss = loss_g
        
        # Log the total loss value
        print(f"Total Loss: {loss.item()}")

        # Return the total loss and the geometric loss
        return loss, loss_g
