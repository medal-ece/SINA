import os
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------
# Refine the bounding box using contour extraction
# -------------------------------------------------------
def refine_bbox_by_contour(image, bbox, expand_pixels=0):
    """
    A color-based approach for schematics that have black or very dark outlines.
    Converts to HSV and thresholds for dark values, then picks the largest contour.
    """
    x_min, y_min, x_max, y_max = bbox  # Bounding box from YOLO

    roi = image[y_min:y_max, x_min:x_max]  # Crop the region of interest (ROI)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Convert ROI to HSV color space

    lower_dark = np.array([0, 0, 0])      # HSV lower bound for dark
    upper_dark = np.array([180, 255, 50]) # HSV upper bound for dark (low V)

    mask = cv2.inRange(hsv, lower_dark, upper_dark)  # Mask dark pixels

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Structuring kernel for morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Close small holes

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    if not contours:
        return bbox  # If no contour is found, return original

    c = max(contours, key=cv2.contourArea)  # Choose the largest contour
    x, y, w, h = cv2.boundingRect(c)  # Get bounding box around it

    refined_bbox = (x_min + x, y_min + y, x_min + x + w, y_min + y + h)  # Adjust to global coords
    return refined_bbox

# -------------------------------------------------------
# Detect pins along the edges of a given bounding box region
# -------------------------------------------------------
def detect_pins(image, bbox, expand_pixels=5):
    """
    Detect pins by scanning along each edge of the specified bounding box region.
    The region is expanded by expand_pixels (for better coverage during pin detection).
    """
    x_min, y_min, x_max, y_max = bbox  # Unpack the bounding box

    # Expand the region to include more border area
    x_min = max(0, x_min - expand_pixels)
    y_min = max(0, y_min - expand_pixels)
    x_max = min(image.shape[1], x_max + expand_pixels)
    y_max = min(image.shape[0], y_max + expand_pixels)

    component = image[y_min:y_max, x_min:x_max]  # Crop the component region

    gray = cv2.cvtColor(component, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur to reduce noise

    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)  # Threshold the image

    top_pins = []
    bottom_pins = []
    left_pins = []
    right_pins = []

    # Top edge: scan first 3 rows
    for x in range(binary.shape[1]):
        for y in range(3):
            if binary[y, x] == 255:
                top_pins.append((x_min + x, y_min + y))
                break

    # Bottom edge: scan last 3 rows
    for x in range(binary.shape[1]):
        for y in range(binary.shape[0] - 3, binary.shape[0]):
            if binary[y, x] == 255:
                bottom_pins.append((x_min + x, y_min + y))
                break

    # Left edge: scan first 3 columns
    for y in range(binary.shape[0]):
        for x in range(3):
            if binary[y, x] == 255:
                left_pins.append((x_min + x, y_min + y))
                break

    # Right edge: scan last 3 columns
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1] - 3, binary.shape[1]):
            if binary[y, x] == 255:
                right_pins.append((x_min + x, y_min + y))
                break

    return top_pins, bottom_pins, left_pins, right_pins

# -------------------------------------------------------
# Main processing function: combine refined bbox with pin detection
# -------------------------------------------------------
def process_all_images():
    current_dir = os.getcwd()  # Get the current working directory
    PARENT_PATH = os.path.dirname(current_dir)  # One level up
    PROJECT_PATH = os.path.dirname(os.path.dirname(current_dir))  # Two levels up

    text_removed_folder = os.path.join(PARENT_PATH, 'Model test/Text Removed images')  # Preprocessed inputs
    test_images_folder = os.path.join(PARENT_PATH, 'Model test/Test images')  # Original test images
    output_folder = os.path.join(PARENT_PATH, 'Model test/Model test results')  # Where to save output
    os.makedirs(output_folder, exist_ok=True)

    pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')  # Folder with pose models
    train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]  # List of training runs
    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the pose directory.")

    def extract_suffix(folder_name):
        return 0 if folder_name == "train" else int(folder_name[5:])  # Extract training number

    latest_train_folder = max(train_folders, key=extract_suffix)  # Latest training run
    latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')  # Path to model weights
    print(f"Loading YOLO model from: {latest_train_path}")
    model = YOLO(latest_train_path)  # Load model

    image_files = [f for f in os.listdir(text_removed_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # Image list

    for image_file in image_files:
        image_path = os.path.join(text_removed_folder, image_file)  # Path to input
        img = cv2.imread(image_path)  # Read the cleaned image
        results = model(image_path)[0]  # YOLO inference

        original_image_path = os.path.join(test_images_folder, image_file)  # Matching original
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            print(f"Warning: Could not load {original_image_path}. Skipping.")
            continue

        for bbox in results.boxes.xyxy.cpu().numpy():  # Iterate over each detection
            x_min, y_min, x_max, y_max = map(int, bbox)  # Convert bbox coordinates to integers

            refined_bbox = refine_bbox_by_contour(img, (x_min, y_min, x_max, y_max), expand_pixels=0)  # Refine box
            top_pins, bottom_pins, left_pins, right_pins = detect_pins(img, refined_bbox, expand_pixels=8)  # Detect pins
            pins = top_pins + bottom_pins + left_pins + right_pins  # All pins

            # Draw the refined bounding box in black
            cv2.rectangle(original_img,
                          (refined_bbox[0], refined_bbox[1]),
                          (refined_bbox[2], refined_bbox[3]),
                          (0, 0, 0), 2)

            # Draw pins in green
            for pin in pins:
                cv2.circle(original_img, pin, radius=4, color=(0, 255, 0), thickness=-1)

        output_path = os.path.join(output_folder, image_file)  # Path to save
        cv2.imwrite(output_path, original_img)  # Save the result
        print(f"Processed image saved to: {output_path}")

# -------------------------------------------------------
# Main entry point
# -------------------------------------------------------
if __name__ == '__main__':
    process_all_images()  # Run the full pipeline
