import cv2
import numpy as np

# Load the Lenna image in color
lenna_img = cv2.imread('Approach.png', cv2.IMREAD_COLOR)
if lenna_img is None:
    raise ValueError("Image not found or path is incorrect")

# Function to create a binary pattern with 3 channels
def create_pattern(width, height, num_bands):
    pattern = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for color
    if num_bands == 1:  # Bright image
        pattern.fill(255)
    elif num_bands > 1:  # Band patterns
        band_width = width // num_bands
        for i in range(height):
            for j in range(width):
                if (j // band_width) % 2 == 0:
                    pattern[i, j] = [255, 255, 255]  # Bright band
    # Dark image is handled by initialization with zeros
    return pattern

# Apply patterns to the Lenna image
patterns_applied = []
patterns = [1, 0] + [2**i for i in range(1,8)]

for num_bands in patterns:
    pattern = create_pattern(lenna_img.shape[1], lenna_img.shape[0], num_bands)
    applied_pattern = cv2.bitwise_and(lenna_img, pattern)
    patterns_applied.append(applied_pattern)
    if num_bands > 1:
        inverse_pattern = cv2.bitwise_and(lenna_img, cv2.bitwise_not(pattern))
        patterns_applied.append(inverse_pattern)

# Define spacing between images and collage settings
spacing = 100
num_images = len(patterns_applied)
collage_rows = int(np.sqrt(num_images))
collage_cols = int(np.ceil(num_images / collage_rows))
background_color = [127, 127, 127]  # For BGR

collage_height = collage_rows * (lenna_img.shape[0] + spacing) - spacing
collage_width = collage_cols * (lenna_img.shape[1] + spacing) - spacing
collage_image = np.full((collage_height, collage_width, 3), background_color, dtype=np.uint8)

# Font settings for labeling
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 1

# Place each pattern-applied image into the collage with labeling
for i, img in enumerate(patterns_applied):
    row = i // collage_cols
    col = i % collage_cols
    start_row = row * (lenna_img.shape[0] + spacing)
    start_col = col * (lenna_img.shape[1] + spacing)
    
    collage_image[start_row:start_row+lenna_img.shape[0], start_col:start_col+lenna_img.shape[1]] = img

    label = f'Pattern {i+1}' if i % 2 == 0 else f'Inv {i}'
    label_pos = (start_col + 5, start_row + lenna_img.shape[0] - 5)
    cv2.putText(collage_image, label, label_pos, font, font_scale, font_color, font_thickness)

# Save the collage image
cv2.imwrite('Scan_encoding_patterns_collage_color_labeled.png', collage_image)
