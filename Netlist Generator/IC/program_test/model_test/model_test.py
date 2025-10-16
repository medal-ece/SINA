import os
import cv2
from ultralytics import YOLO
# Get the current working directory and define the project root
current_dir = os.getcwd()
PROJECT_PATH = os.path.dirname(os.path.dirname(current_dir))  # Project path is two levels up

# Uses YOLO's built-in visualization (results.plot) – quick, simple, limited control.
def process_all_images():
    # Find the latest training folder dynamically
    model_folder = os.path.join(PROJECT_PATH, 'current_trained_model')
    train_folders = [folder for folder in os.listdir(model_folder) if folder.startswith('train')]

    # Check if there are train folders
    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the directory.")

    # Determine the latest train folder
    def extract_suffix(folder_name):
        if folder_name == "train":
            return 0
        else:
            return int(folder_name[5:])

    latest_train_folder = max(train_folders, key=extract_suffix)
    latest_model_path = os.path.join(model_folder, latest_train_folder, 'weights', 'last.pt')

    # Define the image folder and output folder paths (updated to match the second version)
    image_folder = os.path.join(current_dir, 'inputs', 'test_images')
    output_folder = os.path.join(current_dir, 'outputs', 'test_results')
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from: {latest_model_path}")
    model = YOLO(latest_model_path)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process and save each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # Run inference
        results = model(image_path, conf=0.1)[0]

        # Generate visualization using YOLO's built-in plotting (simple rendering)
        processed_img = results.plot(
            line_width=1,
            font_size=2,
            labels=True,
            conf=True
        )

        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, processed_img)

        print(f"Processed image saved to: {output_image_path}")

# Uses manual OpenCV drawing – more control, more code, fully customizable.
def process_all_images_flexible():
    # Find the latest training folder dynamically
    model_folder = os.path.join(PROJECT_PATH, 'current_trained_model/')
    train_folders = [folder for folder in os.listdir(model_folder) if folder.startswith('train')]

    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the directory.")

    def extract_suffix(folder_name):
        if folder_name == "train":
            return 0
        else:
            return int(folder_name[5:])

    latest_train_folder = max(train_folders, key=extract_suffix)
    latest_model_path = os.path.join(model_folder, latest_train_folder, 'weights', 'last.pt')

    image_folder = os.path.join(current_dir, 'inputs', 'test_images')
    output_folder = os.path.join(current_dir, 'outputs', "test_results")
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from: {latest_model_path}")
    model = YOLO(latest_model_path)

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        results = model(image_path, conf=0.1)[0]
        img = results.orig_img.copy()

        # Loop through boxes and draw manually
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()

            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box.astype(int)
                conf = boxes.conf[i]
                cls = int(boxes.cls[i])
                label = f"{results.names[cls]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw label with small font
                # font_scale = 0.3  # smaller text
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # thickness = 1
                # (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                # cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), (255, 0, 0), -1)
                # cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw keypoints (pose) if available
        if results.keypoints is not None:
            for kpt in results.keypoints.xy:
                for x, y in kpt.cpu().numpy():
                    if x > 0 and y > 0:
                        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, img)
        print(f"Processed image saved to: {output_image_path}")

# Main function with user selection
def main():
    #  Choose between process_all_images() and process_all_images_flexible() which ever you want.
    process_all_images()
    # process_all_images_flexible()

if __name__ == '__main__':
    main()
