import torch
# Assuming sinc1, sinc2, sinc3, and so3 are modules that are defined elsewhere and imported here.
from .sinc import sinc1, sinc2, sinc3
from . import so3

# Function to compute the product of two twists in the Lie algebra se(3).
def twist_prod(x, y):
    # Reshape the input vectors to ensure they are two-dimensional with 6 components each.
    x_ = x.view(-1, 6)
    y_ = y.view(-1, 6)

    # Split the twists into their rotational (w) and translational (v) parts.
    xw, xv = x_[:, 0:3], x_[:, 3:6]
    yw, yv = y_[:, 0:3], y_[:, 3:6]

    # Compute the cross product of the rotational parts.
    zw = so3.cross_prod(xw, yw)
    # Compute the cross product of the rotational part of x with the translational part of y and vice versa.
    zv = so3.cross_prod(xw, yv) + so3.cross_prod(xv, yw)

    # Concatenate the rotational and translational parts to form the product twist.
    z = torch.cat((zw, zv), dim=1)

    # Reshape the result to have the same shape as the input.
    return z.view_as(x)

# Function to compute the Lie bracket, which is equivalent to the twist product for se(3).
def liebracket(x, y):
    return twist_prod(x, y)

# Function to convert a twist vector into its corresponding matrix representation in SE(3).
def mat(x):
    # Reshape the input vector to ensure it is two-dimensional with 6 components.
    x_ = x.view(-1, 6)
    # Extract the components of the twist vector.
    w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
    v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
    # Create a tensor of zeros with the same type and device as the input.
    O = torch.zeros_like(w1)

    # Construct the matrix representation of the twist.
    X = torch.stack((
        torch.stack((  O, -w3,  w2, v1), dim=1),
        torch.stack(( w3,   O, -w1, v2), dim=1),
        torch.stack((-w2,  w1,   O, v3), dim=1),
        torch.stack((  O,   O,   O,  O), dim=1)), dim=1)
    # Reshape the result to have the appropriate dimensions.
    return X.view(*(x.size()[0:-1]), 4, 4)

# Function to convert a matrix representation of an element in SE(3) back into a twist vector.
def vec(X):
    # Reshape the input matrix to ensure it is three-dimensional.
    X_ = X.view(-1, 4, 4)
    # Extract the components of the twist vector from the matrix representation.
    w1, w2, w3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    v1, v2, v3 = X_[:, 0, 3], X_[:, 1, 3], X_[:, 2, 3]
    # Concatenate the extracted components to form the twist vector.
    x = torch.stack((w1, w2, w3, v1, v2, v3), dim=1)
    # Reshape the result to have the same shape as the input matrix.
    return x.view(*X.size()[0:-2], 6)

# Function to generate the identity element of the Lie algebra se(3).
def genvec():
    # Return the 6x6 identity matrix.
    return torch.eye(6)

# Function to convert the generator vector into its matrix representation.
def genmat():
    # Convert the identity element of the Lie algebra into its matrix representation.
    return mat(genvec())

# Function to compute the exponential map from se(3) to SE(3).
def exp(x):
    # Reshape the input vector to ensure it is two-dimensional with 6 components.
    x_ = x.view(-1, 6)
    # Split the twist into its rotational (w) and translational (v) parts.
    w, v = x_[:, 0:3], x_[:, 3:6]
    # Compute the norm of the rotational part.
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    # Compute the matrix representation of the rotational part.
    W = so3.mat(w)
    # Compute the square of the matrix representation of the rotational part.
    S = W.bmm(W)
    # Create an identity matrix with the same type and device as the input.
    I = torch.eye(3).to(w)

    # Compute the rotation matrix using Rodrigues' rotation formula.
    R = I + sinc1(t)*W + sinc2(t)*S

    # Compute the matrix V used for the translational part.
    V = I + sinc2(t)*W + sinc3(t)*S

    # Compute the translational part of the transformation matrix.
    p = V.bmm(v.contiguous().view(-1, 3, 1))

    # Create a tensor representing the last row of the transformation matrix.
    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    # Concatenate R and p to form the upper part of the transformation matrix.
    Rp = torch.cat((R, p), dim=2)
    # Concatenate Rp and z to form the full transformation matrix.
    g = torch.cat((Rp, z), dim=1)

    # Reshape the result to have the same shape as the input vector.
    return g.view(*(x.size()[0:-1]), 4, 4)

# Function to compute the inverse of a transformation matrix in SE(3).
def inverse(g):
    # Reshape the input matrix to ensure it is three-dimensional.
    g_ = g.view(-1, 4, 4)
    # Extract the rotation matrix R and the translational part p from the transformation matrix.
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]
    # Compute the transpose of the rotation matrix, which is its inverse since R is orthogonal.
    Q = R.transpose(1, 2)
    # Compute the inverse translational part.
    q = -Q.matmul(p.unsqueeze(-1))

    # Create a tensor representing the last row of the inverse transformation matrix.
    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(g_.size(0), 1, 1).to(g)
    # Concatenate Q and q to form the upper part of the inverse transformation matrix.
    Qq = torch.cat((Q, q), dim=2)
    # Concatenate Qq and z to form the full inverse transformation matrix.
    ig = torch.cat((Qq, z), dim=1)

    # Reshape the result to have the same shape as the input matrix.
    return ig.view(*(g.size()[0:-2]), 4, 4)

# Function to compute the logarithmic map from SE(3) to se(3).
def log(g):
    # Reshape the input matrix to ensure it is three-dimensional.
    g_ = g.view(-1, 4, 4)
    # Extract the rotation matrix R and the translational part p from the transformation matrix.
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]

    # Compute the logarithm of the rotation matrix, which gives the rotational part of the twist.
    w = so3.log(R)
    # Compute the matrix H used for the translational part of the twist.
    H = so3.inv_vecs_Xg_ig(w)
    # Compute the translational part of the twist.
    v = H.bmm(p.contiguous().view(-1, 3, 1)).view(-1, 3)

    # Concatenate the rotational and translational parts to form the twist.
    x = torch.cat((w, v), dim=1)
    # Reshape the result to have the same shape as the input matrix.
    return x.view(*(g.size()[0:-2]), 6)

# Function to apply a transformation matrix to a set of points.
def transform(g, a):
    # g : SE(3),  * x 4 x 4
    # a : R^3,    * x 3[x N]
    # Reshape the input matrix to ensure it is three-dimensional.
    g_ = g.view(-1, 4, 4)
    # Extract the rotation matrix R and the translational part p from the transformation matrix.
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    # Apply the rotation and translation to the points.
    if len(g.size()) == len(a.size()):
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    # Return the transformed points.
    return b

# Function to compute the product of two transformation matrices, which corresponds to the composition of transformations.
def group_prod(g, h):
    # g, h : SE(3)
    # Multiply the two matrices to get the product.
    g1 = g.matmul(h)
    # Return the product matrix.
    return g1

# Class that defines a custom PyTorch autograd function for the exponential map.
class ExpMap(torch.autograd.Function):
    """ Exp: se(3) -> SE(3)
    """
    @staticmethod
    def forward(ctx, x):
        """ Exp: R^6 -> M(4),
            size: [B, 6] -> [B, 4, 4],
              or  [B, 1, 6] -> [B, 1, 4, 4]
        """
        # Save the input for use in the backward pass.
        ctx.save_for_backward(x)
        # Compute the exponential map.
        g = exp(x)
        # Return the result.
        return g

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input.
        x, = ctx.saved_tensors
        # Compute the exponential map of the input.
        g = exp(x)
        # Get the generator matrix for the Lie algebra.
        gen_k = genmat().to(x)

        # Compute the gradient of the output with respect to the input.
        # This involves multiplying the gradient of the output with respect to the transformation matrix
        # by the derivative of the transformation matrix with respect to the input twist vector.

        # Multiply the generator matrix with the exponential map of the input.
        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # Ensure the result has the same type and device as the gradient output.
        dg = dg.to(grad_output)

        # Reshape the gradient output to match the dimensions of dg.
        go = grad_output.contiguous().view(-1, 1, 4, 4)
        # Element-wise multiplication of the gradient output with dg.
        dd = go * dg
        # Sum over the appropriate dimensions to get the gradient input.
        grad_input = dd.sum(-1).sum(-1)

        # Return the gradient input.
        return grad_input

# The Exp function is an alias for the forward pass of the ExpMap class.
Exp = ExpMap.apply



#EOF
