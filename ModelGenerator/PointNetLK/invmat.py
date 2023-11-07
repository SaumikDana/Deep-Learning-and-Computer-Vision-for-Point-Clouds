""" inverse matrix """

import torch

def batch_inverse(x):
    """
    Compute the inverse of each matrix in a batch.
    
    Args:
    x (Tensor): A batch of square matrices of shape (batch_size, n, n).
    
    Returns:
    Tensor: A batch of inverted matrices.
    """
    batch_size, h, w = x.size()
    assert h == w  # Ensure the matrices are square.
    y = torch.zeros_like(x)  # Initialize a tensor for the inverses.
    for i in range(batch_size):  # Loop over the batch.
        y[i, :, :] = x[i, :, :].inverse()  # Invert each matrix individually.
    return y

def batch_inverse_dx(y):
    """
    Compute the derivative of the inverse operation for a batch of matrices.
    
    Args:
    y (Tensor): A batch of inverted matrices.
    
    Returns:
    Tensor: The derivative of the inverse operation.
    """
    batch_size, h, w = y.size()
    assert h == w  # Ensure the matrices are square.
    # Compute the derivative of the inverse with respect to the original matrix.
    # This involves a Kronecker product and a matrix multiplication.
    yl = y.repeat(1, 1, h).view(batch_size*h*h, h, 1)  # Prepare left multiplicand for batch matrix multiplication.
    yr = y.transpose(1, 2).repeat(1, h, 1).view(batch_size*h*h, 1, h)  # Prepare right multiplicand for batch matrix multiplication.
    dy = - yl.bmm(yr).view(batch_size, h, h, h, h)  # Perform batch matrix multiplication and reshape.

    return dy

def batch_pinv_dx(x):
    """
    Compute the pseudo-inverse of each matrix in a batch and its derivative.
    
    Args:
    x (Tensor): A batch of matrices of shape (batch_size, n, m).
    
    Returns:
    Tuple[Tensor, Tensor]: A tuple containing the batch of pseudo-inverses and their derivatives.
    """
    batch_size, h, w = x.size()
    xt = x.transpose(1, 2)  # Transpose each matrix in the batch.
    s = xt.bmm(x)  # Compute the Gram matrix (x^T * x).
    b = batch_inverse(s)  # Invert the Gram matrix.
    y = b.bmm(xt)  # Multiply the inverse Gram matrix by x^T to get the pseudo-inverse.

    # Compute the derivative of the pseudo-inverse with respect to the original matrix.
    ex = torch.eye(h*w).to(x).unsqueeze(0).view(1, h, w, h, w)  # Create an identity matrix for each element in the batch.
    ex1 = ex.view(1, h, w*h*w)  # Reshape for batch matrix multiplication.
    dx1 = x.transpose(1, 2).matmul(ex1).view(batch_size, w, w, h, w)  # Compute the derivative of the Gram matrix.
    ds_dx = dx1.transpose(1, 2) + dx1  # Sum the derivatives to get the final derivative of the Gram matrix.
    db_ds = batch_inverse_dx(b)  # Compute the derivative of the inverse Gram matrix.
    db1 = db_ds.view(batch_size, w*w, w*w).bmm(ds_dx.view(batch_size, w*w, h*w))  # Multiply derivatives to get the derivative of the inverse Gram matrix with respect to the original matrix.
    db_dx = db1.view(batch_size, w, w, h, w)  # Reshape to the original dimensions.
    dy1 = db_dx.transpose(1, 2).contiguous().view(batch_size, w, w*h*w)  # Prepare for final batch matrix multiplication.
    dy1 = x.matmul(dy1).view(batch_size, h, w, h, w)  # Compute the first part of the derivative of the pseudo-inverse.
    ext = ex.transpose(1, 2).contiguous().view(1, w, h*h*w)  # Prepare the identity matrix for the second part of the derivative computation.
    dy2 = b.matmul(ext).view(batch_size, w, h, h, w)  # Compute the second part of the derivative of the pseudo-inverse.
    dy_dx = dy1.transpose(1, 2) + dy2  # Combine the two parts to get the final derivative of the pseudo-inverse.

    return y, dy_dx

class InvMatrix(torch.autograd.Function):
    """
    A custom autograd Function to compute the inverse of a matrix and its gradient.
    """
    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass, compute the inverse of each matrix in a batch.
        
        Args:
        x (Tensor): A batch of square matrices.
        
        Returns:
        Tensor: A batch of inverted matrices.
        """
        y = batch_inverse(x)  # Compute the inverse.
        ctx.save_for_backward(y)  # Save for use in the backward pass.
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, compute the gradient of the loss with respect to the input of the forward pass.
        
        Args:
        grad_output (Tensor): The gradient of the loss with respect to the output of the forward pass.
        
        Returns:
        Tensor: The gradient of the loss with respect to the input of the forward pass.
        """
        y, = ctx.saved_tensors  # Retrieve saved tensors from the forward pass.
        batch_size, h, w = y.size()
        assert h == w  # Ensure the matrices are square.

        # Compute the gradient of the inverse operation.
        dy = batch_inverse_dx(y)  # Compute the derivative of the inverse.
        go = grad_output.contiguous().view(batch_size, 1, h*h)  # Reshape the gradient output for batch matrix multiplication.
        ym = dy.view(batch_size, h*h, h*h)  # Reshape the derivative for batch matrix multiplication.
        r = go.bmm(ym)  # Perform batch matrix multiplication.
        grad_input = r.view(batch_size, h, h)  # Reshape to the original dimensions.

        return grad_input
