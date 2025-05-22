# **model_test**

The **model_test** folder provides the tools and resources required to test the performance of the trained object-keypoint detection model. This folder allows users to evaluate how well the model performs on sample circuit schematic images by visualizing bounding boxes and keypoints.

## **Folder Structure**

### **1. `model_test.ipynb`**

-   A Jupyter Notebook for testing the trained model on the images provided in the `test_images` folder.
-   **Functionality**:
    -   Processes all images in the `test_images` folder.
    -   Outputs visualized results:
        -   **Blue Boxes**: Drawn around detected components in the schematic.
        -   **Green Dots**: Marked at the keypoints representing the connection points of each component.

### **2. inputs**

-   A folder containing sample images for testing the trained model.
---

## **How to Use**

1. Place your test_images in the `test_images` folder.
2. run the `model_test.py` file.

---
