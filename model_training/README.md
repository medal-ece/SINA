# **model_training**

This folder contains all the necessary resources to train the machine learning model used in the **Netlist Generator** project. It includes datasets, training scripts, configuration files, and tools to support the training workflow.

## **Directory Contents**

### **1. `train_model.py`**

-   A Scirpt that contains the code to train the YOLO model.

### **2. `data/` Folder**

-   Contains datasets required for training and validation.
-   **Contents**:
-   `images/`: Images used for training and validation.
-   `labels/`: Corresponding label files in the COCO keypoints format.

### **3. `cvat_to_coco_keypoints/` Folder**

-   Contains utility scripts for converting annotation files from **CVAT format** to **COCO Keypoints** format.
-   **Contents**:
    -   **cvat_to_coco_keypoints.py**: Use this script to convert annotations from CVAT XML format to YOLO-compatible COCO Keypoints `.txt` files.

    -   **copy_lables_to_data.py**: After splitting the dataset into training and validation sets, this script copies the corresponding .txt label files into the appropriate subdirectories:
        - For each image in data/images/train, its label will be copied to data/labels/train/
        - For each image in data/images/val, its label will be copied to data/labels/val/

    -   **inputs**: `annotations.xml` CVAT XML file.

### **4. `helper_files/` Folder**

-    Contains utility scripts used to set up configuration files.
-    These functions are called within `train_model.py` during the training setup.
  
---


## **How to Train the Model**

1. Ensure your dataset is correctly organized under the data/ folder:
    - Images should be placed in `data/images/train/` and `data/images/val/`
    - Labels should be placed in `data/labels/train/` and `data/labels/val/`
2. Follow the setup instructions provided in the root directory.
3. Once setup is complete, run `train_model.py` to begin training the YOLO model.

