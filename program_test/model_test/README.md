# **model_test**

The model_test folder contains tools and resources for evaluating the performance of the trained YOLO model. It enables users to test the model on sample circuit schematic images and visualize detection results, including bounding boxes and confidence scores.

## **Directory Structure**

### **1. `model_test.py`**

-    model_test.py is the main script for evaluating the trained model on images in the test_images/ folder. It supports two visualization methods:
    -    OpenCV-based – customizable display of bounding boxes, labels, and confidence scores.
    -    Ultralytics built-in – quick visualization using results.plot().

### **2. `inputs/test_images/`**

-   Contains sample test images used to evaluate model performance.
---

## **How to Use**

1. Place your test_images in the `inputs/test_images/` folder.
2. run the `model_test.py` file.

---
