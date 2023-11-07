""" PointLK ver. 2018.07.06.
    using approximated Jacobian by backward-difference.
"""
import torch
from . import se3, invmat

class PointNetLK(torch.nn.Module):
    """
    PointLK module for point cloud alignment using PointNet features and the Lucas & Kanade algorithm.
    """

    def __init__(self, ptnet, delta=1.0e-2, learn_delta=False):
        """
        Initialize the PointLK module.

        Parameters:
        ptnet (torch.nn.Module): The PointNet model used for feature extraction.
        delta (float): The initial step size for the SE(3) transformation parameters.
        learn_delta (bool): Flag to determine if delta should be a learnable parameter.
        """
        super().__init__()
        # The PointNet model for extracting features from point cloud data.
        self.ptnet = ptnet

        # Function to compute the inverse of a matrix, used for transformations.
        self.inverse = invmat.InvMatrix.apply

        # Function to compute the matrix exponential of a twist vector, which represents
        # an SE(3) transformation (rotation and translation in 3D space).
        self.exp = se3.Exp

        # Function to apply an SE(3) transformation to a point cloud.
        # It takes a batch of transformation matrices and a batch of point clouds and
        # applies the transformation to each point cloud.
        self.transform = se3.transform

        # Initialize the twist vector with the delta value for each parameter.
        # The twist vector represents the infinitesimal transformation of the point cloud.
        w1 = delta
        w2 = delta
        w3 = delta
        v1 = delta
        v2 = delta
        v3 = delta
        twist = torch.Tensor([w1, w2, w3, v1, v2, v3])

        # Define the twist vector as a learnable parameter if learn_delta is True.
        # This allows the network to learn the optimal step size for the transformation
        # parameters during training.
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

        # Placeholder for the last error computed during the alignment process.
        # This can be used to monitor convergence or for debugging purposes.
        self.last_err = None

        # Placeholder for storing the series of transformations computed during the
        # iterative process. This is mainly for debugging purposes to track the
        # transformations applied at each iteration.
        self.g_series = None

        # Placeholder for the previous rotation matrix between iterations.
        # This can be used to check for convergence or to implement certain optimization
        # strategies that depend on the previous state.
        self.prev_r = None

        # Placeholder for the final estimated transformation matrix after the alignment
        # process. This will hold the result of the point cloud registration.
        self.g = None

        # An iteration counter to keep track of the number of iterations performed
        # during the alignment process. This can be used for debugging or to enforce
        # a maximum number of iterations.
        self.itr = 0

    @staticmethod
    def rsq(r):
        """
        Compute the mean squared error of the residual tensor r against a tensor of zeros.
        
        The residual tensor r should ideally contain near-zero values if the predictions are accurate.
        This function computes the MSE as a measure of the residuals' magnitude, where a lower MSE
        indicates better performance (i.e., the predictions are closer to the actual values).
        
        Args:
        r (Tensor): The residual tensor, representing the difference between predictions and actual values.
        
        Returns:
        Tensor: The mean squared error of the residuals.
        """
        # Create a tensor of zeros with the same shape and device as the residual tensor r
        z = torch.zeros_like(r)
        
        # Compute the mean squared error between the residual tensor r and the zero tensor z
        # The parameter 'size_average=False' is deprecated and should be replaced with 'reduction='sum''
        # to compute the sum of squared errors without averaging.
        mse = torch.nn.functional.mse_loss(r, z, reduction='sum')
        
        # Return the computed MSE
        return mse

    @staticmethod
    def comp(g, igt):
        """
        Compute the mean squared error between the product of g and igt, and the identity matrix.
        
        Args:
        g (Tensor): A batch of transformation matrices.
        igt (Tensor): A batch of inverse transformation matrices.
        
        Returns:
        Tensor: The mean squared error.
        """
        # Ensure that the dimensions of g and igt are compatible for matrix multiplication
        assert g.size(0) == igt.size(0), "Batch sizes of g and igt must be equal."
        assert g.size(1) == igt.size(1) and g.size(1) == 4, "g and igt must be 4x4 matrices (second dimension)."
        assert g.size(2) == igt.size(2) and g.size(2) == 4, "g and igt must be 4x4 matrices (third dimension)."
        
        # Compute the product of g and igt
        A = g.matmul(igt)
        
        # Create a batch of identity matrices with the same batch size and device as A
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        
        # Compute the mean squared error between A and I
        mse = torch.nn.functional.mse_loss(A, I, reduction='sum')
        
        # No need to scale the MSE by the number of elements in the identity matrix
        # if we are using reduction='sum', as it already represents the total error.
        return mse

    @staticmethod
    def do_forward(net, p0, p1, maxiter=10, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        """
        Perform the forward pass for the PointNet-LK algorithm.

        Parameters:
        net (PointNetLK): The PointNet-LK network instance.
        p0 (torch.Tensor): The source point cloud batch data.
        p1 (torch.Tensor): The target point cloud batch data.
        maxiter (int): Maximum number of iterations for the iterative closest point algorithm.
        xtol (float): Tolerance for stopping criteria.
        p0_zero_mean (bool): If True, zero-centers the source point cloud.
        p1_zero_mean (bool): If True, zero-centers the target point cloud.

        Returns:
        r (torch.Tensor): The residual vector after alignment.
        """
        # Initialize identity transformation matrices for each batch element.
        # These will be used to zero-center the point clouds if required.
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0)  # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1)  # [B, 4, 4]
        
        # If zero-centering for p0 is enabled, compute the mean and adjust the points.
        if p0_zero_mean:
            print("Zero-centering p0...")
            p0_m = p0.mean(dim=1)  # Compute the mean of each batch of points [B, N, 3] -> [B, 3]
            a0 = a0.clone()  # Clone to avoid in-place operations which can cause issues with autograd
            a0[:, 0:3, 3] = p0_m  # Set the translation part of the transformation matrix
            q0 = p0 - p0_m.unsqueeze(1)  # Subtract the mean from the points to zero-center them
        else:
            q0 = p0  # If not zero-centering, use the original points

        # If zero-centering for p1 is enabled, compute the mean and adjust the points.
        if p1_zero_mean:
            print("Zero-centering p1...")
            p1_m = p1.mean(dim=1)  # Compute the mean of each batch of points [B, N, 3] -> [B, 3]
            a1 = a1.clone()  # Clone to avoid in-place operations
            a1[:, 0:3, 3] = -p1_m  # Set the translation part of the transformation matrix to negative mean
            q1 = p1 - p1_m.unsqueeze(1)  # Subtract the mean from the points to zero-center them
        else:
            q1 = p1  # If not zero-centering, use the original points

        # Perform the forward pass of the network to align the zero-centered source points to the target points
        print("Performing forward pass...")
        r = net(q0, q1, maxiter=maxiter, xtol=xtol)

        # If either point cloud was zero-centered, adjust the estimated transformation matrix accordingly.
        if p0_zero_mean or p1_zero_mean:
            print("Adjusting estimated transformation matrix...")
            est_g = net.g  # Get the estimated transformation matrix from the network
            # If the source points were zero-centered, pre-multiply the transformation matrix by the zero-centering matrix
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            # If the target points were zero-centered, post-multiply the transformation matrix by the inverse zero-centering matrix
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            net.g = est_g  # Update the network's estimated transformation matrix

            print("Adjusting transformation matrix series...")
            est_gs = net.g_series  # Get the series of estimated transformation matrices from the network
            # Adjust the series of transformation matrices if zero-centering was applied
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            net.g_series = est_gs  # Update the network's series of transformation matrices

        print("Forward pass complete.")
        return r  # Return the residual vector after alignment

    def forward(self, p0, p1, maxiter=10, xtol=1.0e-7):
        """
        Forward pass of the PointNetLK network.

        Parameters:
        p0 (torch.Tensor): The source point cloud batch data.
        p1 (torch.Tensor): The target point cloud batch data.
        maxiter (int): Maximum number of iterations for the iterative closest point algorithm.
        xtol (float): Tolerance for stopping criteria.

        Returns:
        r (torch.Tensor): The residual vector after alignment.
        """
        # Log the start of the forward method
        print("Entering forward method")
        
        # Initialize the transformation matrix g0 as an identity matrix for each batch element
        # and make it contiguous in memory for efficient computation.
        g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
        # Uncomment the following line to print the initial transformation matrix.
        # print("Initial transformation matrix:", g0)
        
        # Call the iterative closest point-like method (iclk) with the initial transformation matrix g0,
        # the source point cloud p0, and the target point cloud p1, along with the maximum number of iterations
        # and the tolerance for stopping criteria.
        r, g, itr = self.iclk(g0, p0, p1, maxiter, xtol)
        # Log the result from the iclk method, which includes the residual vector, the final transformation matrix,
        # and the number of iterations performed.
        print("Result from iclk:", r, g, itr)

        # Store the final transformation matrix in the class instance variable self.g.
        self.g = g
        # Store the number of iterations performed in the class instance variable self.itr.
        self.itr = itr
        # Return the residual vector, which represents the alignment error between the source and target point clouds.
        return r

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    def approx_Jic(self, p0, f0, dt, epsilon=1e-8):
        """
        Approximate the Jacobian matrix required for the ICP-like algorithm.

        Parameters:
        p0 (torch.Tensor): Source point cloud [B, N, 3], where B is the batch size, N is the number of points.
        f0 (torch.Tensor): Feature vectors corresponding to p0 [B, K], where K is the number of features.
        dt (torch.Tensor): Small changes in transformation parameters [B, 6].

        Returns:
        J (torch.Tensor): Approximated Jacobian matrix.
        """
        # Get the batch size and number of points from the source point cloud
        batch_size = p0.size(0)
        num_points = p0.size(1)

        # Initialize a tensor to store the transformation matrices
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)

        # Compute the transformation matrices for each element in the batch
        for b in range(batch_size):
            d = torch.diag(dt[b, :])  # Create a diagonal matrix from the dt vector
            D = self.exp(-d)  # Compute the matrix exponential to get the transformation matrix
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  # Reshape for broadcasting [B, 6, 1, 4, 4]

        # Apply the transformations to the source point cloud
        p = self.transform(transf, p0.unsqueeze(1))  # [B, 1, N, 3] -> [B, 6, N, 3]

        # Unsqueeze the feature vector to match the dimensions for broadcasting
        f0 = f0.unsqueeze(-1)  # [B, K, 1]

        # Compute the feature vectors for the transformed point clouds and reshape
        f = self.ptnet(p.view(-1, num_points, 3)).view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]

        # Calculate the difference in feature vectors
        df = f0 - f  # [B, K, 6]

        # Before division, add a small epsilon to avoid division by zero
        dt_safe = dt + epsilon * (dt == 0).float()

        # Divide the difference by the small changes to approximate the Jacobian
        # Use dt_safe to avoid division by zero
        J = df / dt_safe.unsqueeze(1)  # [B, K, 6]

        # Check for NaNs in the output and raise an error or handle it
        if torch.isnan(J).any():
            raise ValueError("NaN encountered in Jacobian computation.")

        # Return the approximated Jacobian matrix
        return J

    def iclk(self, g0, p0, p1, maxiter, xtol):
        """
        Iterative Closest Point-like method for aligning point clouds.

        Parameters:
        g0 (torch.Tensor): Initial guess for the transformation matrix.
        p0 (torch.Tensor): Source point cloud.
        p1 (torch.Tensor): Target point cloud.
        maxiter (int): Maximum number of iterations to run.
        xtol (float): Tolerance for stopping the iteration.

        Returns:
        r (torch.Tensor): Residuals after alignment.
        g (torch.Tensor): Final transformation matrix.
        itr (int): Number of iterations run.
        """
        # Log entry into the iclk method
        print("Entering iclk method")
        
        # Store the current training state of the network and set it to evaluation mode.
        # This is important to ensure that BatchNorm layers behave consistently during inference.
        training = self.ptnet.training
        self.ptnet.eval()
        print("Network set to evaluation mode.")
        
        # Get the batch size from the source point cloud
        batch_size = p0.size(0)

        # Initialize the transformation matrix 'g' and a series to track its evolution
        g = g0
        self.g_series = torch.zeros(maxiter+1, *g0.size(), dtype=g0.dtype, device=g0.device)
        self.g_series[0] = g0.clone()

        # Calculate the feature vectors for the source point cloud using the PointNet model
        print("Calculating feature vectors for p0...")
        f0 = self.ptnet(p0)

        # Calculate the approximate Jacobian matrix for the ICP-like process
        print("Calculating approximate Jacobian...")
        dt = self.dt.to(p0).expand(batch_size, 6)  # Expand the twist vector to match the batch size
        print("Calculating dt and preparing to call approx_Jic...")
        J = self.approx_Jic(p0, f0, dt)  # Approximate the Jacobian using finite differences
        print("Calculated Jacobian:", J)

        # Initialize the last error to None and the iteration counter to -1
        self.last_err = None
        itr = -1

        # Attempt to compute the pseudo-inverse of the Jacobian matrix
        try:
            print("Calculating pseudo-inverse of J...")
            # Transpose the Jacobian matrix to prepare for matrix multiplication
            Jt = J.transpose(1, 2)  # [B, K, 6] -> [B, 6, K]
            
            # Compute the Hessian matrix, which is the square matrix of second-order partial derivatives
            # This is done by matrix-multiplying the transpose of the Jacobian with the Jacobian itself
            H = Jt.bmm(J)  # [B, 6, K] x [B, K, 6] -> [B, 6, 6]
            
            # Compute the inverse of the Hessian matrix using a custom inverse function
            # This step is crucial for solving the system of linear equations in the least squares sense
            B = self.inverse(H)  # [B, 6, 6]
            
            # Compute the pseudo-inverse of the Jacobian matrix
            # The pseudo-inverse is used when the system does not have a unique solution or has many solutions
            pinv = B.bmm(Jt)  # [B, 6, 6] x [B, 6, K] -> [B, 6, K]
        except RuntimeError as err:
            # If a runtime error occurs, it is likely due to the Hessian being singular and not invertible
            # This can happen if the Jacobian does not have full rank
            self.last_err = err
            print("Runtime error during pseudo-inverse calculation:", err)
            
            # Calculate the feature vectors for the target point cloud
            # This is necessary to compute the residual vector, which measures the difference between
            # the current transformed source point cloud and the target point cloud
            f1 = self.ptnet(p1)  # [B, N, 3] -> [B, K]
            
            # Compute the residual vector, which is the difference between the feature vectors
            # of the target and the transformed source point clouds
            r = f1 - f0  # [B, K]
            
            # Reset the network to its original training state
            # This is important to ensure that the network's behavior is consistent with its state
            # before entering this function
            self.ptnet.train(training)
            
            # Return the current residuals, transformation matrix, and iteration count
            # These values are used to determine the convergence of the algorithm and to update
            # the transformation matrix in the next iteration
            return r, g, itr

        # Begin the iterative process
        itr = 0
        r = None
        for itr in range(maxiter):
            print(f"Iteration {itr + 1}/{maxiter}")
            self.prev_r = r  # Store the previous residuals
            p = self.transform(g.unsqueeze(1), p1)  # Apply the current transformation to the target point cloud
            f = self.ptnet(p)  # Calculate the feature vectors for the transformed target point cloud
            r = f - f0  # Compute the new residuals

            # Compute the change in transformation parameters using the pseudo-inverse
            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            # Check if the norm of the parameter change is below the tolerance
            check = dx.norm(p=2, dim=1, keepdim=True).max()
            print("Max norm of dx:", float(check))
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0  # No update was necessary
                break  # If the change is below the tolerance, stop iterating

            # Update the transformation matrix with the computed change
            g = self.update(g, dx)
            self.g_series[itr + 1] = g.clone()  # Store the updated transformation matrix in the series

        # If the iteration stopped early, fill the remaining series with the last transformation matrix
        rep = len(range(itr, maxiter))
        self.g_series[(itr + 1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        # Reset the network to its original training state
        self.ptnet.train(training)
        print("Exiting iclk method")
        
        # Return the final residuals, transformation matrix, and iteration count
        return r, g, (itr + 1)
