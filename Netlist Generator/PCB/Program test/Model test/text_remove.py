#!/usr/bin/env python3
import os
import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account

# ── CONFIG ───────────────────────────────────────────────────────────
symbol_expand  = 1       # symbol_expand : pixels to expand around each OCR symbol bounding box
sat_thresh     = 30      # sat_thresh : max saturation value to classify a pixel as grey
val_thresh     = 150     # val_thresh : min brightness to classify a pixel as grey
FULL_SCALE     = 4.0     # FULL_SCALE : scaling factor for upsampling tiny text
inpaint_method = 'white' # inpaint_method : how to remove text - options: 'white', 'telea', 'ns'

# ── PATHS ───────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))       # SCRIPT_DIR : current script path
IMAGE_DIR   = os.path.join(SCRIPT_DIR, 'Test images')          # IMAGE_DIR : input folder for original images
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'Text Removed images')  # OUTPUT_DIR : output folder for cleaned images
os.makedirs(OUTPUT_DIR, exist_ok=True)                         # create OUTPUT_DIR if it doesn't exist
CREDS_PATH  = os.path.join(SCRIPT_DIR, 'credentials.json')     # CREDS_PATH : path to Vision API credentials

# ── GOOGLE VISION CLIENT ───────────────────────────────────────────
creds = service_account.Credentials.from_service_account_file(CREDS_PATH)  # creds : loaded credentials
vision_client = vision.ImageAnnotatorClient(credentials=creds)             # vision_client : Google OCR client

def ocr_grey_mask(img):
    """
    Run OCR at original resolution and return a binary mask marking grey text.
    """
    H, W = img.shape[:2]                              # H : image height, W : image width
    mask = np.zeros((H, W), dtype=np.uint8)           # mask : blank binary mask

    _, enc = cv2.imencode('.png', img)                # enc : encoded image bytes
    gv = vision.Image(content=enc.tobytes())          # gv : image for Vision API
    ann = vision_client.document_text_detection(image=gv).full_text_annotation  # ann : OCR results

    for page in ann.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    for sym in word.symbols:
                        xs = [v.x for v in sym.bounding_box.vertices]         # xs : list of x-coords
                        ys = [v.y for v in sym.bounding_box.vertices]         # ys : list of y-coords
                        x1 = max(0, min(xs) - symbol_expand)                  # x1 : left bound
                        y1 = max(0, min(ys) - symbol_expand)                  # y1 : top bound
                        x2 = min(W, max(xs) + symbol_expand)                  # x2 : right bound
                        y2 = min(H, max(ys) + symbol_expand)                  # y2 : bottom bound
                        roi = img[y1:y2, x1:x2]                               # roi : cropped region of interest
                        if roi.size == 0:
                            continue
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)           # hsv : ROI in HSV color space
                        grey = (hsv[:,:,1] < sat_thresh) & (hsv[:,:,2] > val_thresh)  # grey : mask of grey pixels
                        mask[y1:y2, x1:x2][grey] = 255                        # mark grey pixels in mask
    return mask

def ocr_grey_mask_zoom(img):
    """
    Upscale the image, run OCR for tiny text, return downsampled binary mask.
    """
    H, W = img.shape[:2]                                 # H : original height, W : original width
    hi, hj = int(W * FULL_SCALE), int(H * FULL_SCALE)    # hi : scaled width, hj : scaled height

    img_hi = cv2.resize(img, (hi, hj), interpolation=cv2.INTER_CUBIC)  # img_hi : upscaled image

    _, enc = cv2.imencode('.png', img_hi)                     # enc : encoded high-res image
    gv = vision.Image(content=enc.tobytes())                  # gv : Vision-compatible image
    ann = vision_client.document_text_detection(image=gv).full_text_annotation  # ann : OCR result

    mask_hi = np.zeros((hj, hi), dtype=np.uint8)              # mask_hi : high-res binary mask

    for page in ann.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    for sym in word.symbols:
                        xs = [v.x for v in sym.bounding_box.vertices]         # xs : x-coordinates
                        ys = [v.y for v in sym.bounding_box.vertices]         # ys : y-coordinates
                        x1 = max(0, min(xs) - symbol_expand)                  # x1 : left bound
                        y1 = max(0, min(ys) - symbol_expand)                  # y1 : top bound
                        x2 = min(hi, max(xs) + symbol_expand)                 # x2 : right bound
                        y2 = min(hj, max(ys) + symbol_expand)                 # y2 : bottom bound
                        roi_hi = img_hi[y1:y2, x1:x2]                         # roi_hi : cropped region in upscaled image
                        if roi_hi.size == 0:
                            continue
                        hsv = cv2.cvtColor(roi_hi, cv2.COLOR_BGR2HSV)        # hsv : HSV representation of ROI
                        grey = (hsv[:,:,1] < sat_thresh) & (hsv[:,:,2] > val_thresh)  # grey : mask of grey pixels
                        mask_hi[y1:y2, x1:x2][grey] = 255                    # update mask

    mask = cv2.resize(mask_hi, (W, H), interpolation=cv2.INTER_NEAREST)     # mask : final downsampled mask
    return mask

# Main processing loop for all images in IMAGE_DIR
for fn in sorted(os.listdir(IMAGE_DIR)):              # fn : filename of each image
    if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    src = os.path.join(IMAGE_DIR, fn)                 # src : full path to image
    img = cv2.imread(src)                             # img : loaded input image
    if img is None:
        print("Skipping unreadable:", fn)
        continue

    m1 = ocr_grey_mask(img)                           # m1 : text mask at native resolution
    m2 = ocr_grey_mask_zoom(img)                      # m2 : text mask from upscaled image
    mask = cv2.bitwise_or(m1, m2)                      # mask : combined binary mask

    if inpaint_method == 'white':
        out = img.copy()                              # out : output image copy
        out[mask == 255] = (255, 255, 255)            # fill grey-text areas with white
    else:
        flag = cv2.INPAINT_TELEA if inpaint_method == 'telea' else cv2.INPAINT_NS  # flag : inpaint mode
        out = cv2.inpaint(img, mask, inpaintRadius=3, flags=flag)                 # out : inpainted image

    dst = os.path.join(OUTPUT_DIR, fn)                # dst : output image path
    cv2.imwrite(dst, out)                             # write output image to disk
    print(fn, "→ text removed")

print("All done. Check Text Removed images/")
