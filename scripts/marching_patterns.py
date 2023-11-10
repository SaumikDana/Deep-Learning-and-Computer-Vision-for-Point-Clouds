import cv2
import numpy as np

# Load the Lenna image
lenna_img = cv2.imread('Scan.png', cv2.IMREAD_GRAYSCALE)
if lenna_img is None:
    raise ValueError("Image not found or path is incorrect")

# Function to create a sinusoidal pattern with a given phase shift
def create_sinusoidal_pattern(width, height, phase_shift):
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Sinusoidal pattern generation with increased frequency
    frequency = 2 * np.pi / 100 
    pattern = np.sin(frequency * X + phase_shift)
    pattern_normalized = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
    return pattern_normalized

# Number of patterns with different phase shifts
num_patterns = 14

# Generate an array of phase shifts evenly spaced between 0 and 2*pi
phase_steps = np.linspace(0, 2 * np.pi, num_patterns, endpoint=False)

# Offset for each subsequent image
offset_x, offset_y = 30, 30  # Adjust as needed

# Scale factor for each image
scale_factor = 0.05  # Reduce the size of each image

# Calculate the size of the final composite image
scaled_width = int(lenna_img.shape[1] * scale_factor)
scaled_height = int(lenna_img.shape[0] * scale_factor)
final_width = scaled_width + offset_x * (num_patterns - 1)
final_height = scaled_height + offset_y * (num_patterns - 1)

# Create a transparent composite image (RGBA)
composite_image = np.zeros((final_height, final_width, 4), dtype=np.uint8)

# Place each pattern-applied image in the composite image
for i, phase_shift in enumerate(phase_steps):
    pattern = create_sinusoidal_pattern(lenna_img.shape[1], lenna_img.shape[0], phase_shift)
    pattern_enhanced = cv2.equalizeHist(pattern)
    applied_pattern = cv2.bitwise_and(lenna_img, pattern_enhanced)

    # Scale down the image
    scaled_image = cv2.resize(applied_pattern, (scaled_width, scaled_height))

    start_x = i * offset_x
    start_y = i * offset_y
    end_x = start_x + scaled_width
    end_y = start_y + scaled_height

    # Place the scaled image in the composite image and set the alpha channel to opaque for this region
    composite_image[start_y:end_y, start_x:end_x, :3] = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
    composite_image[start_y:end_y, start_x:end_x, 3] = 255  # Alpha channel set to opaque

# Save the final composite image
cv2.imwrite('Scan_marching_patterns_collage_labeled.png', composite_image)
