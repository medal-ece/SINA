# **Current Trained Model**

This folder contains resources related to the latest trained machine learning model used for analyzing and interpreting circuit schematic images. The model is designed to detect circuit components in order to generate a netlist.

## **Directory Structure**

### **1. `train/`**
- Contains the currently deployed trained model, developed using Ultralytics YOLO.
- Model weights are located in `train/weights/` directory (last.pt for latest, best.pt for best performing).
- This model is used in the testing and evaluation phases found in the program_test directory.

---
