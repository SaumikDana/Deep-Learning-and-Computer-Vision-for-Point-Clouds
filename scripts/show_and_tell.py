import cv2
import numpy as np

# Load the knee model image
knee_image_path = 'Approach.png'
knee_image = cv2.imread(knee_image_path, cv2.IMREAD_UNCHANGED)
if knee_image is None:
    raise ValueError("Knee image not found or path is incorrect")

# Load the LCD scanner image
lcd_image_path = '/Users/saumikdana/DL_CV_Images/Visie-Why-Visie-e1697704233555.png'
lcd_image = cv2.imread(lcd_image_path, cv2.IMREAD_UNCHANGED)
if lcd_image is None:
    raise ValueError("LCD image not found or path is incorrect")

# Resize the LCD image if necessary
scale_factor = 0.5  # Adjust scale factor as needed
lcd_image_resized = cv2.resize(lcd_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Define the position where the LCD will be placed over the knee
x_offset = (knee_image.shape[1] - lcd_image_resized.shape[1]) // 2
y_offset = 30  # For example, 30 pixels from the top

# Create a light projection effect
# This will be a semi-transparent white area that simulates the light
projection_effect = np.zeros_like(knee_image, dtype=np.uint8)
projection_height = 200  # Change as needed
cv2.rectangle(projection_effect, (0, y_offset + lcd_image_resized.shape[0]), (knee_image.shape[1], y_offset + lcd_image_resized.shape[0] + projection_height), (255, 255, 255), -1)

# Blend the projection effect with the knee image
alpha = 0.2  # Transparency factor for the projection effect
blended_projection = cv2.addWeighted(knee_image, 1, projection_effect, alpha, 0)

# Overlay the LCD image onto the knee image with projection
knee_with_lcd = blended_projection.copy()
lcd_alpha_channel = lcd_image_resized[:, :, 3] / 255.0  # Normalize the alpha channel
inverse_alpha = 1.0 - lcd_alpha_channel
for c in range(3):
    knee_with_lcd[y_offset:y_offset+lcd_image_resized.shape[0], x_offset:x_offset+lcd_image_resized.shape[1], c] = \
        (inverse_alpha * knee_with_lcd[y_offset:y_offset+lcd_image_resized.shape[0], x_offset:x_offset+lcd_image_resized.shape[1], c]) + \
        (lcd_alpha_channel * lcd_image_resized[:, :, c])

# Save the final image
final_image_path = 'knee_with_lcd_projection.png'
cv2.imwrite(final_image_path, knee_with_lcd)
