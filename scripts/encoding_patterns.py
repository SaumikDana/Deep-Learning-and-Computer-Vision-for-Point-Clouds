import cv2
import numpy as np

# Load the Lenna image
lenna_img = cv2.imread('Scan.png', cv2.IMREAD_GRAYSCALE)
if lenna_img is None:
    raise ValueError("Image not found or path is incorrect")

# Function to create a binary pattern
def create_pattern(width, height, num_bands):
    pattern = np.zeros((height, width), dtype=np.uint8)
    if num_bands == 1:  # Bright image
        pattern.fill(255)
    elif num_bands > 1:  # Band patterns
        band_width = width // num_bands
        for i in range(height):
            for j in range(width):
                if (j // band_width) % 2 == 0:
                    pattern[i, j] = 255  # Bright band
    # Dark image is handled by initialization with zeros
    return pattern

# Apply patterns to the Lenna image
patterns_applied = []
# Start with a bright image, then a dark image, then increasing band patterns
patterns = [1, 0] + [2**i for i in range(1,8)]  # 1, 0, 2, 4, 8, 16, 32, 64, 128

for num_bands in patterns:
    pattern = create_pattern(lenna_img.shape[1], lenna_img.shape[0], num_bands)
    applied_pattern = cv2.bitwise_and(lenna_img, pattern)
    patterns_applied.append(applied_pattern)
    if num_bands > 1:  # Add inverse pattern for band patterns only
        inverse_pattern = cv2.bitwise_and(lenna_img, cv2.bitwise_not(pattern))
        patterns_applied.append(inverse_pattern)

# Define spacing between images
spacing = 10  # pixels

# Determine the size of the collage with spacing
num_images = len(patterns_applied)
collage_rows = int(np.sqrt(num_images))
collage_cols = int(np.ceil(num_images / collage_rows))

# Create a blank collage image with spacing
# Define your desired background color (e.g., medium gray)
background_color = 127

collage_height = collage_rows * (lenna_img.shape[0] + spacing) - spacing
collage_width = collage_cols * (lenna_img.shape[1] + spacing) - spacing
collage_image = np.full((collage_height, collage_width), background_color, dtype=np.uint8)

# Define font for the label
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 1

# Convert the grayscale collage to a BGR image
collage_image_bgr = cv2.cvtColor(collage_image, cv2.COLOR_GRAY2BGR)

set1 = [3,4,7,8,11,12,15]

# Place each pattern-applied image into the collage image with spacing and label it
for i, img in enumerate(patterns_applied):
    row = i // collage_cols
    col = i % collage_cols
    start_row = row * (lenna_img.shape[0] + spacing)
    start_col = col * (lenna_img.shape[1] + spacing)
    
    # Insert the image with spacing
    collage_image[start_row:start_row+lenna_img.shape[0], start_col:start_col+lenna_img.shape[1]] = img

    # Convert the grayscale pattern image to BGR before inserting into the collage
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    collage_image_bgr[start_row:start_row+lenna_img.shape[0], start_col:start_col+lenna_img.shape[1]] = img_bgr

    # Draw a diagonal red strikethrough line for the 4th and 5th images
    if i in set1: 
        line_start = (start_col, start_row)
        line_end = (start_col + lenna_img.shape[1], start_row + lenna_img.shape[0])
        cv2.line(collage_image_bgr, line_start, line_end, (0, 0, 255), thickness=5)  # Red line
    
    # Label the image
    label = f'Pattern {i+1}' if i % 2 == 0 else f'Inv {i}'
    label_pos = (start_col + 5, start_row + lenna_img.shape[0] - 5)
    cv2.putText(collage_image, label, label_pos, font, font_scale, font_color, font_thickness)

# Save the collage image
cv2.imwrite('Scan_encoding_patterns_collage_labeled.png', collage_image_bgr)
