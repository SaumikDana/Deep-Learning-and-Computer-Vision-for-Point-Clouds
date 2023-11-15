import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_binary_pattern(rows, cols):
    """Generate a basic binary alternating pattern."""
    pattern = np.zeros((rows, cols))
    for i in range(cols):
        if i % 2 == 0:
            pattern[:, i] = 1  # White stripe
    return pattern

def deform_pattern_on_sphere(pattern, radius=1, sphere_resolution=100):
    """Simulate the deformation of the pattern on a spherical surface."""
    rows, cols = pattern.shape
    x = np.linspace(-radius, radius, cols)
    y = np.linspace(-radius, radius, rows)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(radius**2 - X**2 - Y**2)

    # Mask to handle NaN values (outside of the sphere)
    mask = ~np.isnan(Z)
    Z[~mask] = 0  # Set points outside the sphere to 0

    # Convert binary pattern to RGB for coloring
    pattern_rgb = np.repeat(pattern[:, :, np.newaxis], 3, axis=2)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the original pattern
    ax1.imshow(pattern_rgb, cmap='gray')
    ax1.set_title("Original Binary Pattern")
    ax1.axis('off')

    # Plot the deformed pattern on the sphere
    ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=pattern_rgb, shade=False)
    ax2.set_title("Pattern Deformed on Sphere")
    ax2.set_xlim([-radius, radius])
    ax2.set_ylim([-radius, radius])
    ax2.set_zlim([0, radius])
    ax2.axis('off')

    plt.show()

# Parameters
rows, cols = 20, 20

# Generate and deform the pattern
pattern = generate_binary_pattern(rows, cols)
deform_pattern_on_sphere(pattern)


