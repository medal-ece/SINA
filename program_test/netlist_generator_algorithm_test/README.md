# **netlist_generator_algorithm_test**

Contains tools to test, validate, and debug the netlist generation algorithm. It allows you to generate netlists from processed circuit schematics and compare them against known ground-truth netlists.

## **Directory Structure**

### **1. `netlist_generator.py`**
- Main script for generating netlists from circuit schematic images.
- Takes model output and converts it into a structured netlist format.

### **2. `netlist_graph_matcher.py`**
- Compares the generated netlists with true/reference netlists.
- Helps assess the accuracy of the netlist generation algorithm by identifying mismatches.

### **3. `inputs/`**
-   `test_images/`:  Sample schematic images to be processed.
- `true_netlists/`: Ground-truth netlist files used for comparison.

### **4. `helper_files/`**
-  `process_images.py`:
        -   Runs the trained model on test images.
        -   Extracts components, connections, and positional data required for netlist generation.
- `generate_netlists.py` Takes processed data from process_images.py and constructs the actual netlist.

---

## **How to Use**

1. **`netlist_generator.py`**:
- Ensure:
  - The trained model is available in `current_trained_model/`.
  - Test images are in `inputs/test_images/`.
- run `netlist_generator.py`

2. **`netlist_graph_matcher.py`**:
- Ensure:
   - Generated netlists are available in `outputs/test_results/`.
   - True netlists are located in `inputs/true_netlists/`.
- run `netlist_graph_matcher.py`

---
