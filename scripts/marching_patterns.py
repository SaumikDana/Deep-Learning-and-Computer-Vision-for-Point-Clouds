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

    # Sinusoidal pattern generation
    frequency = 2 * np.pi / 20  # Adjust frequency as needed
    pattern = np.sin(frequency * X + phase_shift)
    pattern_normalized = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
    return pattern_normalized

# Number of patterns with different phase shifts
num_patterns = 8

# Generate and apply phase-shifted patterns
patterns_applied = []
phase_steps = np.linspace(0, 2 * np.pi, num_patterns, endpoint=False)

for phase_shift in phase_steps:
    pattern = create_sinusoidal_pattern(lenna_img.shape[1], lenna_img.shape[0], phase_shift)
    applied_pattern = cv2.bitwise_and(lenna_img, pattern)
    patterns_applied.append(applied_pattern)

# Define spacing between images
spacing = 10  # pixels

# Determine the size of the collage with spacing
num_images = len(patterns_applied)
collage_rows = int(np.sqrt(num_images))
collage_cols = int(np.ceil(num_images / collage_rows))

# Create a blank collage image with spacing
background_color = 127  # Medium gray
collage_height = collage_rows * (lenna_img.shape[0] + spacing) - spacing
collage_width = collage_cols * (lenna_img.shape[1] + spacing) - spacing
collage_image = np.full((collage_height, collage_width), background_color, dtype=np.uint8)

# Define font for the label
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 1

# Convert the grayscale collage to a BGR image for labeling
collage_image_bgr = cv2.cvtColor(collage_image, cv2.COLOR_GRAY2BGR)

# Place each pattern-applied image into the collage image with spacing and label it
for i, img in enumerate(patterns_applied):
    row = i // collage_cols
    col = i % collage_cols
    start_row = row * (lenna_img.shape[0] + spacing)
    start_col = col * (lenna_img.shape[1] + spacing)
    
    # Insert the image with spacing
    collage_image_bgr[start_row:start_row+lenna_img.shape[0], start_col:start_col+lenna_img.shape[1]] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Label the image
    label = f'Phase {i+1}'
    label_pos = (start_col + 5, start_row + lenna_img.shape[0] - 5)
    cv2.putText(collage_image_bgr, label, label_pos, font, font_scale, font_color, font_thickness)

# Save the collage image
cv2.imwrite('Scan_patterns_collage_labeled.png', collage_image_bgr)
