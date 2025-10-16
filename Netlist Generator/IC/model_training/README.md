# **model_training**

This folder contains all the essential resources needed to train the YOLO model used in the Netlist Generator project. It includes datasets, preprocessing utilities, training scripts, configuration helpers, and annotation format conversion tools.

## **Directory Structure**

### **1. `train_model.py`**

-   Python script for training the YOLO model using Ultralytics.
-   Handles dataset paths, configurations, and model training loops.

### **2. `data/`**

-   Contains the datasets used for training and validation.
-   **Contents**:
    -   `images/`: Circuit schematic images used during training and validation.
    -   `labels/`: Corresponding annotation files in COCO Keypoints format.

### **3. `cvat_to_coco_keypoints/`**

-   Utilities to convert annotation files from CVAT XML format to COCO Keypoints format, which is compatible with YOLO training.
-  **Contents**:
    - **`inputs/`**: `annotations.xml` - the annotation file exported from CVAT.
    - **`cvat_to_coco_keypoints.py`**: Converts `annotations.xml` from CVAT to YOLO-compatible `.txt` files.
    - **`copy_lables_to_data.py`**: Copies labels into correct subfolders after splitting the dataset:
        - For each image in data/images/train, its label will be copied to data/labels/train/
        - For each image in data/images/val, its label will be copied to data/labels/val/ 

### **4. `helper_files/`**

-    Contains utility scripts for dynamically generating configuration files.
  
---

## **How to Train the Model**

1. Prepare the Dataset:
    - Place training images in: `data/images/train/`
    - Place validation images in: `data/images/val/`
    - Place matching label files in: `data/labels/train/` and `data/labels/val/`
2. Complete the Environment Setup:
    - Follow the setup guide in the root `README.md` to configure Python, virtual environments, and install dependencies.
3. Start Training:
    - Run the training script `train_model.py`.

