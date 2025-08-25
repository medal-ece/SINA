import cv2
import numpy as np
import os

# Define directories
current_dir = os.getcwd()
PARENT_PATH = os.path.dirname(current_dir)  # Parent path is one level up
PROJECT_PATH = os.path.dirname(PARENT_PATH)  # Project path is two levels up

# Define paths
IMAGE_FOLDER = os.path.join(PARENT_PATH, 'Model test', 'Test images')
OUTPUT_FOLDER = os.path.join(PARENT_PATH, 'Model test', 'Text Removed images')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to process each image
def remove_text_from_image(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for **Red** (Red has two ranges due to hue wrapping)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])

    # Define HSV range for **Green**
    lower_green, upper_green = np.array([35, 100, 100]), np.array([85, 255, 255])

    # Create masks for red and green
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks (Keep only red & green)
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_green)

    # Create a new image with only red and green components
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert non-red & non-green areas to white
    background = np.full_like(image, 255)  # White background
    final_result = np.where(mask[:, :, None] == 0, background, filtered_image)

    # Save the processed image
    cv2.imwrite(output_path, final_result)

# Process all images in the folder
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    input_path = os.path.join(IMAGE_FOLDER, image_file)
    output_path = os.path.join(OUTPUT_FOLDER, image_file)

    remove_text_from_image(input_path, output_path)
    print(f"Processed: {image_file} -> Saved to: {OUTPUT_FOLDER}")

print("\nAll images have been processed and saved in 'Text Removed images'!")

# Optionally display the last processed image
if image_files:
    last_image_path = os.path.join(OUTPUT_FOLDER, image_files[-1])
    last_image = cv2.imread(last_image_path)
