# **Netlist Generator**

Welcome to the **Netlist Generator** repository! This project is designed to convert an image of a circuit schematic into a netlist, which is outputted as a text file. The netlist represents the electrical connections between circuit components, serving as a textual representation of the circuit.

## **Directory Structure**

### **1. model_training**

-   Resources for training the YOLO model.
-   Contents:
    -   Datasets
    -   Data preprocessing scripts
    -   Training pipeline and configuration
    -   CVAT-to-COCO format conversion tool

### **2. program_test**

-   Tools and scripts to evaluate model performance and test netlist generation.
-   Contents:
    -   Model testing scripts
    -   Netlist generation and comparison scripts
    -   Debugging tools

### **3. current_trained_model**

-   Contains the latest trained YOLO model used for inference.

### **4. .gitignore**

-   Specifies ignored files/directories for Git.

---

## **Project Setup Guide**

## 1) Install Python

-   Download **Python 3.12** (version 3.13 may work, but revert to 3.12 if you encounter issues).
-   [Python Downloads Page](https://www.python.org/downloads/)

## 2) Set Up Python Virtual Environment

-   Navigate to your project folder.

-   Create a new Python virtual environment. Replace `env_netlist` with your preferred name if desired:
    -  Windows:
    ```bash
    py -m venv env_netlist
    ```
    -  macOS/Linux:
    ```bash
    python3 -m venv env_netlist
    ```
### Activate the Virtual Environment

-   To activate the virtual environment, run:
    -   Windows:
    ```bash
    env_netlist\Scripts\activate
    ```
    -   macOS/Linux:
    ```bash
    source env_netlist/bin/activate
    ```

**Note:** You’ll know the environment is activated when you see the environment name in parentheses before the command prompt:

```bash
(env_netlist) C:\your\path\here>
```

## 3) Upgrade Pip

- Run the following command to upgrade `pip`:

    ```bash
    py -m pip install --upgrade pip
    ```

## 4) Install PyTorch

- Go to the PyTorch website (https://pytorch.org/get-started/locally/) and select the options that fit your machine for example mine were:
   - **Stable** always pick stable
   - **Windows** my machine
   - **Pip** 
   - **Python** pick python
   - **CUDA 12.4** (for NVIDIA GPUs)
   - Copy the provided install command and run it.
   - **Example command:**
     
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

## 5) Install Ultralytics

- Install the `ultralytics` package (for YOLO):

    ```bash
    pip install ultralytics
    ```
---

## **How to run files**

- This project is written entirely in Python, so all files can be executed using the same method from the command line or terminal.
   - For windows: `py .\file_name.py` or `python .\file_name.py`
   - For Mac and Linux: `python3 ./file_name.py`

---
