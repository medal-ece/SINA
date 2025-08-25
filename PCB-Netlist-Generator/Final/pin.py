#!/usr/bin/env python3
"""
pin.py

Detect/refine YOLO boxes and find pin locations from schematics.
- Credentials are HARD-CODED to 'credentials.json'.
- Pass both text_dir (text-removed images) and data_dir (original images) from index.py.
- Outputs annotated images to output_dir and pin coords to output_dir/positions/<name>.txt.
"""
import os
import math
import shutil
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from ultralytics import YOLO
from skimage.morphology import skeletonize
from google.cloud import vision
from google.oauth2 import service_account

__all__ = ["Settings", "process_all_images"]

# ──────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────
CREDENTIALS_PATH = "credentials.json"  # hard-coded

# morphological kernels
K3   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
KH_H = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))

# drawing colors/thickness
LINE_COLOR     = (0, 0, 0)
YOLO_COLOR     = (0, 0, 255)
PIN_COLOR      = (0, 255, 0)
PIN_RADIUS     = 4

# HSV thresholds for colored wires (red/green)
SAT, VAL = 50, 50
LOW_R1 = np.array([  0, SAT, VAL]); HI_R1 = np.array([ 20,255,255])
LOW_R2 = np.array([160, SAT, VAL]); HI_R2 = np.array([180,255,255])
LOW_G  = np.array([ 35, SAT, VAL]); HI_G  = np.array([ 85,255,255])

# how far outside the box boundary to look for skeleton pixels
PIN_DIST = 3

# ──────────────────────────────────────────────────────────────────────────
# SETTINGS (paths come from index.py)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    text_dir: str                  # input: text-removed images (from text_removal)
    data_dir: str                  # original images path (same as index --input)
    output_dir: str = "pin"     # output folder
    model_weights: str = "model/best.pt"
    clean_output: bool = False
    conf_threshold: float = 0.7

# ──────────────────────────────────────────────────────────────────────────
# CLIENTS
# ──────────────────────────────────────────────────────────────────────────
def make_vision_client() -> vision.ImageAnnotatorClient:
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return vision.ImageAnnotatorClient(credentials=creds)

# ──────────────────────────────────────────────────────────────────────────
# OCR TEXT MASK
# ──────────────────────────────────────────────────────────────────────────
def get_text_mask(img: np.ndarray, vclient: vision.ImageAnnotatorClient) -> np.ndarray:
    h, w = img.shape[:2]
    _, encoded = cv2.imencode('.png', img)
    request = vision.Image(content=encoded.tobytes())
    response = vclient.text_detection(image=request)

    mask = np.zeros((h, w), dtype=np.uint8)
    for ta in response.text_annotations[1:]:  # [0] is the full text
        vertices = [(v.x, v.y) for v in ta.bounding_poly.vertices]
        pts = np.array(vertices, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return cv2.dilate(mask, K3, iterations=1)

# ──────────────────────────────────────────────────────────────────────────
# BOX REFINEMENT
# ──────────────────────────────────────────────────────────────────────────
def refine_bbox_by_contour(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return bbox

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x_min + x, y_min + y, x_min + x + w, y_min + y + h)

# ──────────────────────────────────────────────────────────────────────────
# PIN LOGIC
# ──────────────────────────────────────────────────────────────────────────
def get_pins_for_bbox(skeleton: np.ndarray, bbox: Tuple[int, int, int, int], text_mask: np.ndarray | None = None
                      ) -> List[Tuple[int, int]]:
    x0, y0, x1, y1 = bbox

    box_mask = np.zeros_like(skeleton, dtype=np.uint8)
    cv2.rectangle(box_mask, (x0, y0), (x1, y1), 255, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PIN_DIST + 1, 2 * PIN_DIST + 1))
    dilated = cv2.dilate(box_mask, kernel, iterations=1)

    candidate = cv2.bitwise_and(skeleton, dilated)
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(box_mask))

    if text_mask is not None:
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(text_mask))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    pin_centers: List[Tuple[int, int]] = []
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] == 0:
            continue
        cx, cy = centroids[lbl]
        pin_centers.append((int(round(cx)), int(round(cy))))
    return pin_centers

def detect_image_pins(img: np.ndarray, refined_bboxes: List[Tuple[int, int, int, int]],
                      vclient: vision.ImageAnnotatorClient) -> List[Tuple[int, int]]:
    text_mask = get_text_mask(img, vclient)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, LOW_R1, HI_R1)
    m2 = cv2.inRange(hsv, LOW_R2, HI_R2)
    m3 = cv2.inRange(hsv, LOW_G,  HI_G)
    wire_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    wire_mask[text_mask > 0] = 0
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, KH_H, iterations=1)
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, K3,   iterations=2)

    sk = skeletonize((wire_mask > 0)).astype(np.uint8) * 255
    for (x0, y0, x1, y1) in refined_bboxes:
        sk[y0:y1+1, x0:x1+1] = 0

    pins: List[Tuple[int, int]] = []
    for bbox in refined_bboxes:
        pins.extend(get_pins_for_bbox(sk, bbox, text_mask=text_mask))
    return pins

# ──────────────────────────────────────────────────────────────────────────
# ENHANCER (unchanged)
# ──────────────────────────────────────────────────────────────────────────
def enhance_image(img: np.ndarray) -> np.ndarray:
    den = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    boosted = cv2.merge((cl, a, b))
    boosted = cv2.cvtColor(boosted, cv2.COLOR_LAB2BGR)
    tophat = cv2.morphologyEx(boosted, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    boosted = cv2.addWeighted(boosted, 1.0, tophat, 0.5, 0)
    gauss = cv2.GaussianBlur(boosted, (0, 0), sigmaX=1.5)
    sharp = cv2.addWeighted(boosted, 1.8, gauss, -0.8, 0)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(sharp, 1.0, edges_colored, 0.3, 0)
    return out

# ──────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────
def process_all_images(s: Settings) -> None:
    if s.clean_output and os.path.exists(s.output_dir):
        shutil.rmtree(s.output_dir)
    os.makedirs(s.output_dir, exist_ok=True)

    positions_dir = os.path.join(s.output_dir, "positions")
    os.makedirs(positions_dir, exist_ok=True)

    yolo = YOLO(s.model_weights)
    vclient = make_vision_client()

    image_files = [
        f for f in os.listdir(s.text_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    for fname in sorted(image_files):
        text_removed_path = os.path.join(s.text_dir, fname)
        img_text_removed = cv2.imread(text_removed_path)
        if img_text_removed is None:
            print(f"⚠ Skipping {fname}: failed to load text-removed image.")
            continue
        img_text_removed = enhance_image(img_text_removed)

        # YOLO on text-removed
        results = yolo(text_removed_path)[0]
        confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else np.array([])
        xyxy  = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else np.empty((0,4))
        raw_bboxes: List[Tuple[int, int, int, int]] = [
            tuple(map(int, box)) for box, c in zip(xyxy, confs) if c >= s.conf_threshold
        ]

        # Refine bboxes
        refined_bboxes: List[Tuple[int, int, int, int]] = [
            refine_bbox_by_contour(img_text_removed, b) for b in raw_bboxes
        ]

        # Build skeleton from text-removed image (remove any remaining text)
        text_mask_removed = get_text_mask(img_text_removed, vclient)
        hsv = cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, LOW_R1, HI_R1)
        m2 = cv2.inRange(hsv, LOW_R2, HI_R2)
        m3 = cv2.inRange(hsv, LOW_G,  HI_G)
        wire_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        wire_mask[text_mask_removed > 0] = 0
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, KH_H, iterations=1)
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, K3,   iterations=2)
        sk = skeletonize((wire_mask > 0)).astype(np.uint8) * 255

        for (x0, y0, x1, y1) in refined_bboxes:
            sk[y0:y1+1, x0:x1+1] = 0

        # Load original from the SAME base input path sent from index.py
        orig_path = os.path.join(s.data_dir, fname)
        orig_img = cv2.imread(orig_path)
        if orig_img is None:
            print(f"⚠ Skipping {fname}: failed to load original image.")
            continue
        orig_img = enhance_image(orig_img)

        # Draw YOLO bboxes and refined boxes
        for (x0, y0, x1, y1) in raw_bboxes:
            cv2.rectangle(orig_img, (x0, y0), (x1, y1), YOLO_COLOR, thickness=2)
        for (x0, y0, x1, y1) in refined_bboxes:
            cv2.rectangle(orig_img, (x0, y0), (x1, y1), LINE_COLOR, thickness=2)

        # Detect pins on both images
        pins_tr   = detect_image_pins(img_text_removed, refined_bboxes, vclient)
        pins_data = detect_image_pins(orig_img,           refined_bboxes, vclient)

        # Combine & dedupe
        all_pins = pins_tr + pins_data
        clustered: List[Tuple[int, int]] = []
        for (px, py) in all_pins:
            if not any(math.hypot(px - cx, py - cy) <= 2 * PIN_RADIUS for cx, cy in clustered):
                clustered.append((px, py))

        # Save positions
        base, _ = os.path.splitext(fname)
        pos_path = os.path.join(positions_dir, f"{base}.txt")
        with open(pos_path, "w", encoding="utf-8") as f:
            for (cx, cy) in clustered:
                f.write(f"{cx},{cy}\n")

        # Draw pins & save annotated image
        for (cx, cy) in clustered:
            cv2.circle(orig_img, (cx, cy), radius=PIN_RADIUS, color=PIN_COLOR, thickness=-1)

        out_path = os.path.join(s.output_dir, fname)
        cv2.imwrite(out_path, orig_img)
        print(f"✔ {fname}: {len(refined_bboxes)} boxes, {len(clustered)} pins drawn")

    print("✅ All done – check", s.output_dir)

# Optional standalone run (we’ll use index.py normally)
if __name__ == "__main__":
    settings = Settings(
        text_dir="text removed",
        data_dir="data",
        output_dir="pin",
        model_weights="model/best.pt",
        clean_output=False,
        conf_threshold=0.7,
    )
    process_all_images(settings)