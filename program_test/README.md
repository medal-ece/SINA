# **program_test**

Provides scripts and tools to validate, test, and debug both the trained YOLO model and the netlist generation pipeline. This serves as the project's central hub for quality assurance, ensuring that both detection and netlist creation perform reliably and accurately.

## **Directory Structure**

### **1. `model_test/`**

-   Contains resources for evaluating the performance of the YOLO model.
-  **Contents**:
    -   Scripts to run inference on test datasets
    -   Sample input images for model testing
    -   Debugging output visualizations (e.g., bounding boxes, keypoints, confidence scores)

### **2. `netlist_generator_algorithm_test/`**

-   Contains tools to test and verify the accuracy of the netlist generation process based on model outputs.
-   **Contents**:
    -   Scripts for generating netlists from the model's inference data
    -   Utilities for comparing generated netlists against ground truth/reference netlists
    -   Debugging output for analyzing connectivity, component detection, and errors

---
