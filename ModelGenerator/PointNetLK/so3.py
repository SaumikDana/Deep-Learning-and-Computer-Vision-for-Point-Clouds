""" 3-d rotation group and corresponding Lie algebra """
import torch
from . import sinc
from .sinc import sinc1, sinc2, sinc3

def cross_prod(x, y):
    z = torch.cross(x.view(-1, 3), y.view(-1, 3), dim=1).view_as(x)
    return z

def liebracket(x, y):
    return cross_prod(x, y)

def mat(x):
    # size: [*, 3] -> [*, 3, 3]
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)

    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)

def vec(X):
    X_ = X.view(-1, 3, 3)
    x1, x2, x3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    x = torch.stack((x1, x2, x3), dim=1)
    return x.view(*X.size()[0:-2], 3)

def genvec():
    return torch.eye(3)

def genmat():
    return mat(genvec())

def RodriguesRotation(x):
    # for autograd
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    R = I + sinc.Sinc1(t)*W + sinc.Sinc2(t)*S

    return R.view(*(x.size()[0:-1]), 3, 3)

def exp(x):
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    R = I + sinc1(t)*W + sinc2(t)*S

    return R.view(*(x.size()[0:-1]), 3, 3)

def inverse(g):
    R = g.view(-1, 3, 3)
    Rt = R.transpose(1, 2)
    return Rt.view_as(g)

def btrace(X):
    # batch-trace: [B, N, N] -> [B]
    n = X.size(-1)
    X_ = X.view(-1, n, n)
    tr = torch.zeros(X_.size(0)).to(X)
    for i in range(tr.size(0)):
        m = X_[i, :, :]
        tr[i] = torch.trace(m)
    return tr.view(*(X.size()[0:-2]))

def log(g):
    eps = 1.0e-7
    R = g.view(-1, 3, 3)
    tr = btrace(R)
    c = (tr - 1) / 2
    t = torch.acos(c)
    sc = sinc1(t)
    idx0 = (torch.abs(sc) <= eps)
    idx1 = (torch.abs(sc) > eps)
    sc = sc.view(-1, 1, 1)

    X = torch.zeros_like(R)
    if idx1.any():
        X[idx1] = (R[idx1] - R[idx1].transpose(1, 2)) / (2*sc[idx1])

    if idx0.any():
        # t[idx0] == math.pi
        t2 = t[idx0] ** 2
        A = (R[idx0] + torch.eye(3).type_as(R).unsqueeze(0)) * t2.view(-1, 1, 1) / 2
        aw1 = torch.sqrt(A[:, 0, 0])
        aw2 = torch.sqrt(A[:, 1, 1])
        aw3 = torch.sqrt(A[:, 2, 2])
        sgn_3 = torch.sign(A[:, 0, 2])
        sgn_3[sgn_3 == 0] = 1
        sgn_23 = torch.sign(A[:, 1, 2])
        sgn_23[sgn_23 == 0] = 1
        sgn_2 = sgn_23 * sgn_3
        w1 = aw1
        w2 = aw2 * sgn_2
        w3 = aw3 * sgn_3
        w = torch.stack((w1, w2, w3), dim=-1)
        W = mat(w)
        X[idx0] = W

    x = vec(X.view_as(g))
    return x

def transform(g, a):
    # g in SO(3):  * x 3 x 3
    # a in R^3:    * x 3[x N]
    if len(g.size()) == len(a.size()):
        b = g.matmul(a)
    else:
        b = g.matmul(a.unsqueeze(-1)).squeeze(-1)
    return b

def group_prod(g, h):
    # g, h : SO(3)
    g1 = g.matmul(h)
    return g1

def vecs_Xg_ig(x):
    # Vi = vec(dg/dxi * inv(g)), where g = exp(x)
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat(x)
    S = X.bmm(X)
    I = torch.eye(3).to(X)

    V = I + sinc2(t)*X + sinc3(t)*S

    return V.view(*(x.size()[0:-1]), 3, 3)

def inv_vecs_Xg_ig(x):
    # H = inv(vecs_Xg_ig(x))
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat(x)
    S = X.bmm(X)
    I = torch.eye(3).to(x)

    e = 1e-10  # Smaller threshold for numerical stability
    eta = torch.zeros_like(t)
    s = (t < e)
    c = ~s  # Use bitwise NOT for complement
    t2 = t[s] ** 2
    # Use more terms in the Taylor series if needed for higher accuracy
    eta[s] = ((t2/40 + 1)*t2/42 + 1)*t2/720 + 1/12  # Taylor series O(t^8)
    # Use the cotangent identity with a stable implementation
    eta[c] = (1 - (t[c]/2) / torch.tan(t[c]/2)) / (t[c]**2)

    H = I - 1/2*X + eta.view(-1, 1, 1)*S  # Ensure proper broadcasting
    return H.view(*(x.size()[0:-1]), 3, 3)

class ExpMap(torch.autograd.Function):
    """
    This class implements the exponential map for the Lie algebra so(3) to the Lie group SO(3).
    The exponential map is a map from the Lie algebra (tangent space at the identity element)
    to the Lie group, which in this case is the group of 3D rotations.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the exponential map.

        Args:
            ctx: Context object that can be used to stash information for backward computation.
            x: A tensor of shape [B, 3] or [B, 1, 3] representing a batch of 3D vectors.

        Returns:
            A tensor of shape [B, 3, 3] or [B, 1, 3, 3] representing a batch of rotation matrices.
        """
        # Save the input for use in the backward pass.
        ctx.save_for_backward(x)

        # Compute the exponential map, which converts a vector in so(3) to a matrix in SO(3).
        g = exp(x)

        # Return the resulting rotation matrix.
        return g

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the exponential map.

        Args:
            ctx: Context object with saved input.
            grad_output: Gradient of the loss function with respect to the output of the forward pass.

        Returns:
            Gradient of the loss function with respect to the input of the forward pass.
        """
        # Retrieve the saved input.
        x, = ctx.saved_tensors

        # Recompute the exponential map from the saved input.
        g = exp(x)

        # Get the generator matrices for so(3), which form a basis for the tangent space.
        gen_k = genmat().to(x)

        # Compute the derivative of the exponential map.
        # This involves matrix-multiplying the generator matrices with the rotation matrix.
        dg = gen_k.matmul(g.view(-1, 1, 3, 3))

        # Ensure the gradient has the correct shape.
        dg = dg.to(grad_output)

        # Multiply the gradient of the loss with respect to the output (grad_output)
        # with the derivative of the exponential map (dg).
        go = grad_output.contiguous().view(-1, 1, 3, 3)
        dd = go * dg

        # Sum over the last two dimensions to get the gradient with respect to the input.
        grad_input = dd.sum(-1).sum(-1)

        # Return the computed gradient.
        return grad_input

# Alias the apply method of ExpMap for ease of use.
Exp = ExpMap.apply
