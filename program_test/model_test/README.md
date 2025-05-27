# **model_test**

The model_test folder contains tools and resources for evaluating the performance of the trained YOLO model. It enables users to test the model on sample circuit schematic images and visualize detection results, including bounding boxes and confidence scores.

## **Directory Structure**

### **1. `model_test.py`**

-    This script is used to evaluate the trained model using images located in the `inputs/test_images/` directory. It supports two visualization methods:
    -    OpenCV-based – customizable display of bounding boxes, labels, and confidence scores.
    -    Ultralytics built-in – quick visualization using results.plot().

### **2. `model_validation.py`**

-    This script is used to validate the trained model using images located in the `inputs/validate_images/` directory. It generates a performance report that provides an overview of the model’s overall accuracy as well as class-wise performance metrics. This helps identify underperforming classes, enabling targeted improvements to enhance model accuracy.

### **3. `inputs/test_images/`**

-   Contains sample test images used to evaluate model performance.
---

## **How to Use**

1. Place your test_images in the `inputs/test_images/` folder.
2. run the `model_test.py` file.

---
