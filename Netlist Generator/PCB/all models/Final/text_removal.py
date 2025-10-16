#!/usr/bin/env python3
"""
text_removal.py

Module for removing grey text from schematic images using Google Vision OCR.
- Credentials are HARD-CODED to 'credentials.json' in this module.
- Provide input/output paths and optional params via the Settings dataclass.
- Call process_folder(Settings(...)) from your orchestrator (index.py).
"""
import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import shutil
from google.cloud import vision
from google.oauth2 import service_account

# --------------------------- Constants ---------------------------
CREDENTIALS_PATH = "credentials.json"  # <- hard-coded as requested

# --------------------------- Config ---------------------------
@dataclass
class Settings:
    # Paths (provided by index.py or your caller)
    image_dir: str                   # input folder with images
    output_dir: str                  # output folder for cleaned images

    # Algorithm params (defaults preserved from your original script)
    symbol_expand: int = 1           # pixels to expand around each OCR symbol bbox
    sat_thresh: int = 30             # max saturation to be considered grey
    val_thresh: int = 150            # min brightness to be considered grey
    full_scale: float = 4.0          # upscaling factor for tiny text pass
    inpaint_method: Literal["white", "telea", "ns"] = "white"  # how to remove text

# --------------------- Vision client factory ------------------
def make_vision_client() -> vision.ImageAnnotatorClient:
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return vision.ImageAnnotatorClient(credentials=creds)

# --------------------------- Core OCR -------------------------
def ocr_grey_mask(img: np.ndarray, s: Settings, client: vision.ImageAnnotatorClient) -> np.ndarray:
    """
    Run OCR at original resolution and return a binary mask marking grey text.
    """
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    _, enc = cv2.imencode(".png", img)
    gv = vision.Image(content=enc.tobytes())
    ann = client.document_text_detection(image=gv).full_text_annotation

    for page in ann.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    for sym in word.symbols:
                        xs = [v.x for v in sym.bounding_box.vertices]
                        ys = [v.y for v in sym.bounding_box.vertices]
                        x1 = max(0, min(xs) - s.symbol_expand)
                        y1 = max(0, min(ys) - s.symbol_expand)
                        x2 = min(W, max(xs) + s.symbol_expand)
                        y2 = min(H, max(ys) + s.symbol_expand)
                        roi = img[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        grey = (hsv[:, :, 1] < s.sat_thresh) & (hsv[:, :, 2] > s.val_thresh)
                        mask[y1:y2, x1:x2][grey] = 255
    return mask

def ocr_grey_mask_zoom(img: np.ndarray, s: Settings, client: vision.ImageAnnotatorClient) -> np.ndarray:
    """
    Upscale the image, run OCR for tiny text, return downsampled binary mask.
    """
    H, W = img.shape[:2]
    hi, hj = int(W * s.full_scale), int(H * s.full_scale)

    img_hi = cv2.resize(img, (hi, hj), interpolation=cv2.INTER_CUBIC)

    _, enc = cv2.imencode(".png", img_hi)
    gv = vision.Image(content=enc.tobytes())
    ann = client.document_text_detection(image=gv).full_text_annotation

    mask_hi = np.zeros((hj, hi), dtype=np.uint8)

    for page in ann.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    for sym in word.symbols:
                        xs = [v.x for v in sym.bounding_box.vertices]
                        ys = [v.y for v in sym.bounding_box.vertices]
                        x1 = max(0, min(xs) - s.symbol_expand)
                        y1 = max(0, min(ys) - s.symbol_expand)
                        x2 = min(hi, max(xs) + s.symbol_expand)
                        y2 = min(hj, max(ys) + s.symbol_expand)
                        roi_hi = img_hi[y1:y2, x1:x2]
                        if roi_hi.size == 0:
                            continue
                        hsv = cv2.cvtColor(roi_hi, cv2.COLOR_BGR2HSV)
                        grey = (hsv[:, :, 1] < s.sat_thresh) & (hsv[:, :, 2] > s.val_thresh)
                        mask_hi[y1:y2, x1:x2][grey] = 255

    mask = cv2.resize(mask_hi, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask

# ----------------------- Folder processor ---------------------
def process_folder(s: Settings) -> None:
    """
    Processes all images in s.image_dir and writes cleaned images to s.output_dir.
    Equivalent to the original main loop, just parameterized.
    """
    if os.path.exists(s.output_dir):
        shutil.rmtree(s.output_dir)
    os.makedirs(s.output_dir, exist_ok=True)
    client = make_vision_client()

    for fn in sorted(os.listdir(s.image_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        src = os.path.join(s.image_dir, fn)
        img = cv2.imread(src)
        if img is None:
            print("Skipping unreadable:", fn)
            continue

        m1 = ocr_grey_mask(img, s, client)
        m2 = ocr_grey_mask_zoom(img, s, client)
        mask = cv2.bitwise_or(m1, m2)

        if s.inpaint_method == "white":
            out = img.copy()
            out[mask == 255] = (255, 255, 255)
        else:
            flag = cv2.INPAINT_TELEA if s.inpaint_method == "telea" else cv2.INPAINT_NS
            out = cv2.inpaint(img, mask, inpaintRadius=3, flags=flag)

        dst = os.path.join(s.output_dir, fn)
        cv2.imwrite(dst, out)
        print(fn, "→ ", s.output_dir)

    print("All done. Check:", s.output_dir)

# Optional: allow direct execution (kept minimal; index.py is preferred)
if __name__ == "__main__":
    print(
        "This module is intended to be called from index.py. "
        "Import Settings and call process_folder(Settings(image_dir=..., output_dir=...))."
    )