# Project Methodology Report

## 1. Introduction
This document provides an in-depth overview of the workflow implemented in the **Netlist Generator** project. It covers data collection, annotation conversion, model training, bounding box inference, text removal, pin detection, challenges faced, and solutions adopted.

---

## 2. Data Collection

- **Sources:**  
  - **CircuitLab** (https://www.circuitlab.com/)  
  - **EasyEDA** (https://easyeda.com/)  
- **Volume:**  
  - Collected **50** high-resolution schematic images (PNG/JPG).  
  - Distributed into:  
    - **Training:** 40 images 
    - **Validation:** 10 images 
    - **Test:** 10 images
- **Preprocessing:**  
  - Normalized image resolutions to 1024×1024 pixels using OpenCV.  
  - Converted all images to RGB color space.  
- **File Organization:**  
  ```
  dataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── annotations/
      ├── train.xml  (CVAT export)
      ├── val.xml
      └── test.xml
  ```

---

## 3. Annotation Conversion

1. **CVAT Keypoint Annotations:**  
   - Used partner’s CVAT instance to manually annotate each component:  
     - **Bounding box (4 points)**  
2. **Export:**  
   - Exported annotations in **CVAT JSON** format.  
3. **Conversion Script (`convert_cvat_to_coco.py`):**  
   - Reads CVAT JSON and maps to COCO Keypoints schema.  
   - Ensures each component has a `"keypoints"` array of length `3*K` (x, y, visibility).  
   - Generates `dataset/annotations/train.json`, `val.json`, `test.json`.  

---

## 4. Model Training with YOLOv8

- **Environment:**  
  - **Hardware:** NVIDIA RTX 3080 (10GB GPU), 32GB RAM  
  - **Software:** PyTorch 2.0, CUDA 12.4, ultralytics YOLOv8  
- **Configuration:**  
  - **Base model:** `yolov8n-pose.pt` (nano pose)  
  - **Hyperparameters:**  
    - `epochs`: 50  
    - `batch_size`: 16  
    - `learning_rate`: 0.001 (with cosine decay)  
    - `weight_decay`: 0.0005  
    - **Data Augmentation:** mosaic, mixup, HSV augmentation, random flip  
- **Training Command:**  
  ```bash
  yolo pose train       data=dataset.yaml       model=yolov8n-pose.pt       epochs=50       imgsz=640       batch=16       lr0=0.001       weight_decay=0.0005
  ```
- **Monitoring:**  
  - Used TensorBoard to log loss curves, mAP@0.5, mAP@0.5:0.95.  
  - Achieved final mAP@0.5 ≈ 0.87 on validation set.  
- **Output:**  
  - Saved to `Current trained model/pose/train*/weights/last.pt`.

---

## 5. Bounding Box Inference and Verification

- **Script:** `Program test/Model Test.py`  
- **Inference Settings:**  
  - **Confidence Threshold:** 0.25  
  - **IoU Threshold (NMS):** 0.45  
- **Performance:**  
  - Average inference time: ~10 ms/image (GPU).  
  - Detection accuracy: 98% box recall on test set.  
- **Visualization:**  
  - Bounding boxes drawn in blue, keypoints in green.  
  - Saved under `Model test/Model test results/`.

---

## 6. Text Removal

We support two methods to remove unwanted text from circuit images before pin detection:

1. **OCR Inpainting (`Program test/text_remove.py`)**  
   A two‑pass Google Vision OCR approach to detect and remove grey text legends.

   - **Script Location:** `Program test/text_remove.py`  
   - **Configuration Parameters:**  
     - `symbol_expand`: pixels to pad each OCR symbol bounding box (default 1)  
     - `sat_thresh`: max HSV‑saturation for “grey” text (default 30)  
     - `val_thresh`: min HSV‑value for “grey” text (default 150)  
     - `FULL_SCALE`: upscaling factor for tiny‑text zoom pass (default 4.0)  
     - `inpaint_method`: `'white'`, `'telea'` or `'ns'`  

   - **Workflow:**  
     1. **Native‑Resolution Pass**  
        - Encode the input image to PNG and call Vision API’s `document_text_detection`.  
        - For each detected symbol:  
          - Expand its bounding box by `symbol_expand`.  
          - Extract the region, convert to HSV.  
          - Add pixels with `S < sat_thresh` and `V > val_thresh` to the mask.  
     2. **Zoomed‑Resolution Pass**  
        - Resize image by `FULL_SCALE` (cubic interpolation).  
        - Repeat OCR and grey‑text masking on the high‑res image.  
        - Downsample the high‑res mask back to original dimensions (nearest‑neighbor).  
     3. **Combine Masks**  
        - Bitwise‑OR the two masks into a final mask.  
     4. **Apply Inpainting**  
        - If `inpaint_method == 'white'`, set masked pixels to white.  
        - Otherwise, call `cv2.inpaint` with the chosen algorithm (`INPAINT_TELEA` or `INPAINT_NS`, radius 3).

2. **Color Masking (`Program test/black_text_remove.py`)**  
   A fast, color‑based removal that keeps only red/green circuit lines.

   - Convert the image to HSV.  
   - Create masks for red (H in 0–10 & 160–180, S ≥ 100, V ≥ 100) and green (H in 35–85).  
   - Retain masked pixels; set all others to white.

**Output**  
Cleaned images are saved to  


---

## 7. Pin Detection via Edge Scanning

- **Script:** `Program test/edge_detection.py`  
- **Preprocessing:**  
  - Cropped region expanded by `expand_pixels=8` to capture pins near edges.  
  - Grayscale + Gaussian blur (kernel=5×5, σ=0).
- **Thresholding:**  
  - Adaptive threshold (Gaussian C=2, blocksize=11) → binary invert.
- **Edge Scanning Algorithm:**  
  When we refer to “Top Edge: scan rows 0–2” (and similarly for the other edges), we’re talking about how we inspect the border pixels of a cropped component image (of size H×W) to find pins. In image coordinates:

- **Rows** correspond to the **vertical** position (y‑coordinate), running from 0 at the top to H−1 at the bottom.  
- **Columns** correspond to the **horizontal** position (x‑coordinate), running from 0 at the left to W−1 at the right.  

So:

- **Top Edge: scan rows 0–2**  
  Look at the first three rows (y = 0, 1, 2) across all columns x = 0…W−1.

- **Bottom Edge: scan rows H−3–H−1**  
  Look at the last three rows (y = H−3, H−2, H−1) across all columns.

- **Left Edge: scan cols 0–2**  
  Look at the first three columns (x = 0, 1, 2) across all rows y = 0…H−1.

- **Right Edge: scan cols W−3–W−1**  
  Look at the last three columns (x = W−3, W−2, W−1) across all rows.

Within each of these thin “strips” (3 pixels wide/tall), the algorithm searches for the first white (foreground) pixel in each row or column. Wherever it finds one, it marks that location as a pin.  
- **Post-processing:**  
  - Filter out spurious detections using a minimum distance filter (min separation=10 px).  
- **Output:**  
  - Pins drawn as green circles (radius=4) on original images.  
  - Saved to `Model test/Model test results/`.

---

## 8. Challenges Encountered

1. **Annotation Variance:**  
   - Pin order and count varied; standardized by sorting keypoints clockwise.
2. **Text Artifacts:**  
   - Inpainting sometimes left gray patches; mitigated by morphological closing (kernel=3×3).
3. **Threshold Sensitivity:**  
   - Over- and under-thresholding; performed grid search on C ∈ [1,3], blocksize ∈ [9,15].
4. **Bounding Box Misalignment:**  
   - Tight crops missed small pins; resolved by expand_pixels=5–8 px.

---

## 9. Approaches and Solutions

- **Transfer Learning:** Used partner’s YOLOv8 pose model, froze first two backbone stages for initial epochs.  
- **Hybrid Pipeline:** Combined deep learning for component detection with classical CV for pin localization.  
- **Dynamic Bounding Box Expansion:** Automated expansion based on average pin offset.  

---

## 10. Conclusion and Future Work

The hybrid methodology—involving both YOLO-based detection and edge-scanning—delivers robust netlist generation for a variety of circuit schematics. Future enhancements include:

- **Automated Netlist Text Export:** Parse pin positions into SPICE-format netlists.  
- **End-to-End Validation:** Cross-check generated netlists against input images.  
- **Model Refinement:** Incorporate more diverse circuit datasets, and explore YOLOv8 L/XL models for higher accuracy.
