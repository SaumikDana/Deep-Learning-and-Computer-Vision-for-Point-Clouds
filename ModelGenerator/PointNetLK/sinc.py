import torch
from torch import sin, cos

# Define the sinc function, which is sin(t)/t.
def sinc1(t):
    """ sinc1: Computes the sinc function, sin(t)/t, with a stable implementation for small t. """
    e = 1e-10  # A small threshold to avoid division by zero.
    r = torch.zeros_like(t)  # Initialize the result tensor with zeros.
    a = torch.abs(t)  # Take the absolute value of t.

    # Use a Taylor series expansion for small values of t to avoid numerical instability.
    s = a < e  # Boolean mask for where t is small.
    c = ~s  # Boolean mask for where t is not small.
    t2 = t[s] ** 2  # Compute t squared for small t.
    # Taylor series expansion for sinc around 0.
    r[s] = 1 - t2/6*(1 - t2/20*(1 - t2/42))  # Use the Taylor series for small t.
    r[c] = sin(t[c]) / t[c]  # Use the standard sinc definition for other t.

    return r

# Define the derivative of the sinc function.
def sinc1_dt(t):
    """ sinc1_dt: Computes the derivative of sinc1 with respect to t. """
    # The implementation follows a similar pattern to sinc1.
    e = 1e-10
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = ~s
    t2 = t ** 2
    # Taylor series expansion for the derivative of sinc around 0.
    r[s] = -t[s]/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))
    r[c] = cos(t[c])/t[c] - sin(t[c])/t2[c]

    return r

# Define the derivative of the sinc function divided by t.
def sinc1_dt_rt(t):
    """ sinc1_dt_rt: Computes the derivative of sinc1 divided by t. """
    # The implementation follows a similar pattern to sinc1.
    e = 1e-10
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = ~s
    t2 = t ** 2
    # Taylor series expansion for the derivative of sinc divided by t around 0.
    r[s] = -1/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))
    r[c] = (cos(t[c]) / t[c] - sin(t[c]) / t2[c]) / t[c]

    return r

# Define the reciprocal of the sinc function.
def rsinc1(t):
    """ rsinc1: Computes the reciprocal of the sinc function, t/sin(t). """
    # The implementation follows a similar pattern to sinc1.
    e = 1e-10
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = ~s
    t2 = t[s] ** 2
    # Taylor series expansion for the reciprocal of sinc around 0.
    r[s] = (((31*t2)/42 + 7)*t2/60 + 1)*t2/6 + 1
    r[c] = t[c] / sin(t[c])

    return r

# Define the derivative of the reciprocal of the sinc function.
def rsinc1_dt(t):
    """ rsinc1_dt: Computes the derivative of the reciprocal of sinc1. """
    # The implementation follows a similar pattern to sinc1.
    e = 1e-10
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = ~s
    t2 = t[s] ** 2
    # Taylor series expansion for the derivative of the reciprocal of sinc around 0.
    r[s] = ((((127*t2)/30 + 31)*t2/28 + 7)*t2/30 + 1)*t[s]/3
    r[c] = 1/sin(t[c]) - (t[c]*cos(t[c]))/(sin(t[c])**2)

    return r

# Define the derivative of the reciprocal of the sinc function divided by sin(t).
def rsinc1_dt_csc(t):
    """ rsinc1_dt_csc: Computes the derivative of the reciprocal of sinc1 divided by sin(t). """
    # The implementation follows a similar pattern to sinc1.
    e = 1e-10
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = ~s
    t2 = t[s] ** 2
    # Taylor series expansion for the derivative of the reciprocal of sinc divided by sin(t) around 0.
    r[s] = t2*(t2*((4*t2)/675 + 2/63) + 2/15) + 1/3
    r[c] = (1/sin(t[c]) - (t[c]*cos(t[c]))/(sin(t[c])**2)) / sin(t[c])

    return r

# The following functions sinc2, sinc2_dt, sinc3, sinc3_dt, and sinc4 follow the same pattern as sinc1.
# They compute higher-order variations of the sinc function and their derivatives.
# Each function includes a Taylor series expansion for small values of t to maintain numerical stability.
# The comments for sinc1 can be applied to these functions as well, adjusting for the specific variation of sinc being computed.

def sinc2(t):
    """
    sinc2: Computes the second-order sinc function, (1 - cos(t)) / (t^2).
    
    This function is used to calculate the second-order sinc function, which is a variation of the sinc function
    that arises in various mathematical contexts. It is particularly useful in signal processing and physics.
    
    Parameters:
    t (Tensor): The input tensor containing values at which to evaluate sinc2.
    
    Returns:
    Tensor: The computed second-order sinc function values.
    """
    e = 1e-10  # A small threshold to avoid division by zero when t is very small.
    r = torch.zeros_like(t)  # Initialize the result tensor with zeros.
    a = torch.abs(t)  # Take the absolute value of t to handle negative values symmetrically.

    s = a < e  # Boolean mask for where t is smaller than the threshold.
    c = ~s  # Boolean mask for where t is not smaller than the threshold.
    t2 = t ** 2  # Compute t squared for all values of t.
    
    # For small values of t, use a Taylor series expansion to approximate the function.
    # This avoids numerical instability that would occur due to the division by a very small t.
    r[s] = 1/2*(1 - t2[s]/12*(1 - t2[s]/30*(1 - t2[s]/56)))  # Taylor series approximation for small t.
    
    # For values of t that are not too small, use the standard definition of sinc2.
    r[c] = (1 - cos(t[c])) / t2[c]  # Standard calculation for larger t.

    return r

def sinc2_dt(t):
    """
    sinc2_dt: Computes the derivative of the second-order sinc function with respect to t.
    
    This function calculates the derivative of sinc2, which is useful when one needs to understand
    the rate of change of the second-order sinc function with respect to its input.
    
    Parameters:
    t (Tensor): The input tensor containing values at which to evaluate the derivative of sinc2.
    
    Returns:
    Tensor: The computed derivative values of the second-order sinc function.
    """
    e = 1e-10  # A small threshold to avoid division by zero for small t values.
    r = torch.zeros_like(t)  # Initialize the result tensor with zeros.
    a = torch.abs(t)  # Take the absolute value of t.

    s = a < e  # Boolean mask for where t is smaller than the threshold.
    c = ~s  # Boolean mask for where t is not smaller than the threshold.
    t2 = t ** 2  # Compute t squared for all values of t.
    
    # For small values of t, use a Taylor series expansion to approximate the derivative.
    r[s] = -t[s]/12*(1 - t2[s]/20*(1 - t2[s]/42*(1 - t2[s]/72)))  # Taylor series approximation for small t.
    
    # For values of t that are not too small, use the standard definition of the derivative of sinc2.
    r[c] = sin(t[c])/t2[c] - 2*(1 - cos(t[c])) / (t2[c] * t[c])  # Standard calculation for larger t.

    return r

def sinc3(t):
    """
    sinc3: Computes the third-order sinc function, (t - sin(t)) / (t^3).
    
    This function is another variation of the sinc function that is used in various mathematical
    and engineering fields. It is particularly useful when dealing with series expansions in
    theoretical physics and signal processing.
    
    Parameters:
    t (Tensor): The input tensor containing values at which to evaluate sinc3.
    
    Returns:
    Tensor: The computed third-order sinc function values.
    """
    e = 1e-10  # A small threshold to avoid division by zero when t is very small.
    r = torch.zeros_like(t)  # Initialize the result tensor with zeros.
    a = torch.abs(t)  # Take the absolute value of t to handle negative values symmetrically.

    s = a < e  # Boolean mask for where t is smaller than the threshold.
    c = ~s  # Boolean mask for where t is not smaller than the threshold.
    t2 = t[s] ** 2  # Compute t squared for small values of t.
    
    # For small values of t, use a Taylor series expansion to approximate the function.
    # This avoids numerical instability that would occur due to the division by a very small t.
    r[s] = 1/6*(1 - t2/20*(1 - t2/42*(1 - t2/72)))  # Taylor series approximation for small t.
    
    # For values of t that are not too small, use the standard definition of sinc3.
    r[c] = (t[c] - sin(t[c])) / (t[c]**3)  # Standard calculation for larger t.

    return r

def sinc3_dt(t):
    """
    sinc3_dt: Computes the derivative of the third-order sinc function with respect to t.
    
    This function calculates the derivative of sinc3, which is useful when one needs to understand
    the rate of change of the third-order sinc function with respect to its input.
    
    Parameters:
    t (Tensor): The input tensor containing values at which to evaluate the derivative of sinc3.
    
    Returns:
    Tensor: The computed derivative values of the third-order sinc function.
    """
    e = 1e-10  # A small threshold to avoid division by zero for small t values.
    r = torch.zeros_like(t)  # Initialize the result tensor with zeros.
    a = torch.abs(t)  # Take the absolute value of t.

    s = a < e  # Boolean mask for where t is smaller than the threshold.
    c = ~s  # Boolean mask for where t is not smaller than the threshold.
    t2 = t[s] ** 2  # Compute t squared for small values of t.
    
    # For small values of t, use a Taylor series expansion to approximate the derivative.
    r[s] = -t[s]/60*(1 - t2/21*(1 - t2/24*(1 - t2/165)))  # Taylor series approximation for small t.
    
    # For values of t that are not too small, use the standard definition of the derivative of sinc3.
    r[c] = (3 * sin(t[c]) - t[c] * (cos(t[c]) + 2)) / (t[c]**4)  # Standard calculation for larger t.

    return r

def sinc4(t):
    """
    sinc4: Computes a fourth-order sinc-like function, which is a ratio involving the second-order sinc function.
    
    This function is a higher-order variation of the sinc function that can be used in advanced mathematical
    applications, such as signal processing, wave propagation, and other areas where higher-order terms
    in series expansions are necessary.
    
    Parameters:
    t (Tensor): The input tensor containing values at which to evaluate sinc4.
    
    Returns:
    Tensor: The computed fourth-order sinc-like function values.
    """
    e = 1e-10  # A small threshold to avoid division by zero when t is very small.
    r = torch.zeros_like(t)  # Initialize the result tensor with zeros.
    a = torch.abs(t)  # Take the absolute value of t to handle negative values symmetrically.

    s = a < e  # Boolean mask for where t is smaller than the threshold.
    c = ~s  # Boolean mask for where t is not smaller than the threshold.
    t2 = t ** 2  # Compute t squared for all values of t.
    
    # For small values of t, use a Taylor series expansion to approximate the function.
    # This avoids numerical instability that would occur due to the division by a very small t.
    # The Taylor series is expanded to the order of t^8 to ensure a high degree of accuracy for small t.
    r[s] = 1/24*(1 - t2[s]/30*(1 - t2[s]/56*(1 - t2[s]/90)))  # Taylor series approximation for small t.
    
    # For values of t that are not too small, use the standard definition of sinc4.
    # This involves subtracting the second-order sinc function from 1/2 and then dividing by t squared.
    r[c] = (0.5 - (1 - cos(t[c])) / t2[c]) / t2[c]  # Standard calculation for larger t.

    return r

# Custom autograd functions for each sinc function.
class Sinc1_autograd(torch.autograd.Function):
    """
    Custom autograd Function for the first-order sinc function (sinc1).
    """
    @staticmethod
    def forward(ctx, theta):
        # Forward pass definition...
        ctx.save_for_backward(theta)
        return sinc1(theta)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass (gradient computation) definition...
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc1_dt(theta).to(grad_output)
        return grad_theta

# Create a callable function for sinc1 using the custom autograd function.
Sinc1 = Sinc1_autograd.apply

class RSinc1_autograd(torch.autograd.Function):
    """
    Custom autograd Function for the reciprocal of the first-order sinc function (rsinc1).
    """
    @staticmethod
    def forward(ctx, theta):
        # Forward pass definition...
        ctx.save_for_backward(theta)
        return rsinc1(theta)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass (gradient computation) definition...
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * rsinc1_dt(theta).to(grad_output)
        return grad_theta

# Create a callable function for rsinc1 using the custom autograd function.
RSinc1 = RSinc1_autograd.apply

class Sinc2_autograd(torch.autograd.Function):
    """
    Custom autograd Function for the second-order sinc function (sinc2).
    """
    @staticmethod
    def forward(ctx, theta):
        # Forward pass definition...
        ctx.save_for_backward(theta)
        return sinc2(theta)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass (gradient computation) definition...
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc2_dt(theta).to(grad_output)
        return grad_theta

# Create a callable function for sinc2 using the custom autograd function.
Sinc2 = Sinc2_autograd.apply

class Sinc3_autograd(torch.autograd.Function):
    """
    Custom autograd Function for the third-order sinc function (sinc3).
    """
    @staticmethod
    def forward(ctx, theta):
        # Forward pass definition...
        ctx.save_for_backward(theta)
        return sinc3(theta)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass (gradient computation) definition...
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc3_dt(theta).to(grad_output)
        return grad_theta

# Create a callable function for sinc3 using the custom autograd function.
Sinc3 = Sinc3_autograd.apply

#EOF
