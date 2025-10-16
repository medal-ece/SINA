#!/usr/bin/env python3
import os
import shutil
import json
import math
import re
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from google.cloud import vision
from google.oauth2 import service_account
from collections import defaultdict
from dataclasses import dataclass

# ────────────────────────────────────────────────────────────────
# CONFIG (constants)
# ────────────────────────────────────────────────────────────────
CREDENTIALS_PATH = "credentials.json"   # hard-coded

# morphological kernels & params
K3       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
KH_H     = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
PIN_DIST = 3

# HSV thresholds for colored wires (red/green)
SAT, VAL = 50, 50
LOW_R1   = np.array([  0, SAT, VAL]);  HI_R1 = np.array([ 20,255,255])
LOW_R2   = np.array([160, SAT, VAL]);  HI_R2 = np.array([180,255,255])
LOW_G    = np.array([ 35, SAT, VAL]);  HI_G  = np.array([ 85,255,255])

# fallback search radius for nearest-word
FALLBACK_RADIUS = 30

# globals set at runtime
vision_client = None
yolo = None

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    image_dir: str                 # originals (same as index.py --input)
    text_dir: str                  # text-removed images (index.py --output)
    output_dir: str = "netlists_json"
    model_weights: str = "model/best.pt"

# ────────────────────────────────────────────────────────────────
# OCR + helpers
# ────────────────────────────────────────────────────────────────
def get_text_annotations(img):
    """Return list of (polygon, word) from Google Vision OCR."""
    _, buf = cv2.imencode('.png', img)
    resp = vision_client.text_detection(image=vision.Image(content=buf.tobytes()))
    anns = []
    for ta in resp.text_annotations[1:]:
        pts = np.array([(v.x, v.y) for v in ta.bounding_poly.vertices], dtype=np.int32).reshape(-1,1,2)
        anns.append((pts, ta.description.strip()))
    return anns

def refine_bbox_by_contour(img, bbox):
    """Tighten a YOLO bbox to the largest dark contour inside it."""
    x0,y0,x1,y1 = bbox
    roi = img[y0:y1, x0:x1]
    if roi is None or roi.size == 0:
        return bbox
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,50]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return bbox
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (x0+x, y0+y, x0+x+w, y0+y+h)

def compute_wire_skeleton(img):
    """Build a 1-px skeleton of the red+green wires in `img`."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, LOW_R1, HI_R1)
    m2   = cv2.inRange(hsv, LOW_R2, HI_R2)
    m3   = cv2.inRange(hsv, LOW_G,  HI_G)
    wire = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    wire = cv2.morphologyEx(wire, cv2.MORPH_CLOSE, KH_H, 1)
    wire = cv2.morphologyEx(wire, cv2.MORPH_CLOSE, K3,   2)
    return skeletonize(wire > 0).astype(np.uint8) * 255

def get_pins_for_bbox(sk, bbox):
    """From skeleton `sk` and one refined bbox, return pin centroids touching dilated bbox."""
    x0,y0,x1,y1 = bbox
    box_mask = np.zeros_like(sk)
    cv2.rectangle(box_mask, (x0,y0), (x1,y1), 255, -1)
    dilated = cv2.dilate(box_mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*PIN_DIST+1, 2*PIN_DIST+1)), 1)
    cand = cv2.bitwise_and(sk, dilated)
    cand = cv2.bitwise_and(cand, cv2.bitwise_not(box_mask))
    nl, labels, stats, cn = cv2.connectedComponentsWithStats(cand, 8, cv2.CV_32S)
    pins = []
    for lbl in range(1, nl):
        if stats[lbl, cv2.CC_STAT_AREA] > 0:
            cx, cy = cn[lbl]
            pins.append((int(round(cx)), int(round(cy))))
    return pins

def wire_direction_at(pin, sk, radius=4):
    """Estimate local wire direction at pin by averaging neighbor offsets."""
    px, py = pin
    h, w = sk.shape
    x0, x1 = max(0, px-radius), min(w, px+radius+1)
    y0, y1 = max(0, py-radius), min(h, py+radius+1)
    ys, xs = np.where(sk[y0:y1, x0:x1] > 0)
    pts = np.vstack([xs + x0, ys + y0]).T
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    vecs = pts - np.array([px, py])
    avg  = vecs.sum(axis=0)
    n = np.linalg.norm(avg)
    if n < 1e-3:
        vec = vecs[0]; n = np.linalg.norm(vec) or 1.0
        return vec / n
    return avg / n

def is_valid_label(txt):
    return bool(re.search(r'[A-Za-z]', txt))

def label_along_wire(pin, sk, ocr):
    px, py = pin
    dx, dy = wire_direction_at(pin, sk)
    candidates = set()
    for sign in (+1, -1):
        for t in range(3, 60, 3):
            x = int(round(px + sign*dx*t)); y = int(round(py + sign*dy*t))
            for ox, oy in [(0,0),(5,0),(-5,0),(0,5),(0,-5)]:
                xx, yy = x+ox, y+oy
                for poly, txt in ocr:
                    if cv2.pointPolygonTest(poly, (xx, yy), False) >= 0 and is_valid_label(txt):
                        candidates.add(txt)
    if candidates:
        return max(candidates, key=len)
    best_txt, best_d = "", FALLBACK_RADIUS + 1
    for poly, txt in ocr:
        if not is_valid_label(txt): continue
        cx = poly[:,0,0].mean(); cy = poly[:,0,1].mean()
        d  = math.hypot(cx - px, cy - py)
        if d < best_d:
            best_txt, best_d = txt, d
    return best_txt

def get_text_mask(img):
    _, encoded = cv2.imencode('.png', img)
    resp = vision_client.text_detection(image=vision.Image(content=encoded.tobytes()))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ta in resp.text_annotations[1:]:
        pts = np.array([(v.x, v.y) for v in ta.bounding_poly.vertices], np.int32).reshape(-1,1,2)
        cv2.fillPoly(mask, [pts], 255)
    return cv2.dilate(mask, K3, iterations=1)

def enhance_image(img):
    den = cv2.bilateralFilter(img, 9, 75, 75)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(5.0, (4,4)).apply(l)
    boosted = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    toph = cv2.morphologyEx(boosted, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    boosted = cv2.addWeighted(boosted, 1.0, toph, 0.5, 0)
    ga = cv2.GaussianBlur(boosted, (0,0), 1.5)
    sharp = cv2.addWeighted(boosted, 1.8, ga, -0.8, 0)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.addWeighted(sharp, 1.0, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

def detect_image_pins(img, refined_bboxes):
    text_mask = get_text_mask(img)
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
    pins = []
    for bbox in refined_bboxes:
        pins.extend(get_pins_for_bbox(sk, bbox))
    return pins

# ────────────────────────────────────────────────────────────────
# Core runner
# ────────────────────────────────────────────────────────────────
def generate_netlists(s: Settings) -> None:
    # clean output
    if os.path.exists(s.output_dir):
        shutil.rmtree(s.output_dir)
    os.makedirs(s.output_dir, exist_ok=True)

    # init clients/models
    global vision_client, yolo
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    yolo = YOLO(s.model_weights)

    files = [f for f in sorted(os.listdir(s.text_dir))
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    for fn in files:
        base, _ = os.path.splitext(fn)

        tr_path   = os.path.join(s.text_dir, fn)
        data_path = os.path.join(s.image_dir, fn)

        tr = cv2.imread(tr_path)
        if tr is None:
            print(f"⚠ could not load {tr_path}")
            continue

        if not os.path.isfile(data_path):
            print(f"⚠ missing original for {fn} in {s.image_dir}; skipping")
            continue

        orig = cv2.imread(data_path)
        if orig is None:
            print(f"⚠ could not load {data_path}")
            continue

        # skeleton from text-removed
        sk = compute_wire_skeleton(tr)

        # YOLO boxes on text-removed
        res = yolo(tr_path)[0]
        try:
            xyxy = res.boxes.xyxy.cpu().numpy() if getattr(res.boxes, "xyxy", None) is not None else np.empty((0,4))
        except Exception:
            xyxy = np.empty((0,4))
        raw_boxes = [tuple(map(int, b)) for b in xyxy]

        # refine & mask inside boxes
        refined = [refine_bbox_by_contour(tr, b) for b in raw_boxes]
        for (x0, y0, x1, y1) in refined:
            sk[y0:y1+1, x0:x1+1] = 0

        # OCR on original (for labels)
        ocr = get_text_annotations(orig)

        # detect pins on both (enhanced) images
        pins_tr   = detect_image_pins(enhance_image(tr),   refined)
        pins_data = detect_image_pins(enhance_image(orig), refined)
        all_pins  = pins_tr + pins_data

        # de-duplicate pins
        clustered = []
        for px, py in all_pins:
            if not any(math.hypot(px - cx, py - cy) <= 2 * PIN_DIST for cx, cy in clustered):
                clustered.append((px, py))

        # build components + labels
        components = []
        for idx, bbox in enumerate(refined, start=1):
            x0, y0, x1, y1 = bbox
            comp_pins = []
            for px, py in clustered:
                if x0 - PIN_DIST <= px <= x1 + PIN_DIST and y0 - PIN_DIST <= py <= y1 + PIN_DIST:
                    lbl = label_along_wire((px, py), sk, ocr)
                    comp_pins.append({"x": px, "y": py, "label": lbl})
            components.append({
                "component_id": idx,
                "bbox": [x0, y0, x1, y1],
                "pin_count": len(comp_pins),
                "pins": comp_pins
            })

        # connectivity via skeleton component ids
        num_nets, net_labels = cv2.connectedComponents(sk, connectivity=8)
        net_to_components = defaultdict(set)
        for comp in components:
            for pin in comp["pins"]:
                n = net_labels[pin["y"], pin["x"]]
                net_to_components[n].add(comp["component_id"])

        for comp in components:
            cid = comp["component_id"]
            conns = set()
            for pin in comp["pins"]:
                n = net_labels[pin["y"], pin["x"]]
                for other in net_to_components[n]:
                    if other != cid:
                        conns.add(other)
            comp["connected"] = sorted(conns)

        # write JSON
        out = {"image": fn, "components": components}
        with open(os.path.join(s.output_dir, f"{base}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"✔ {fn}: wrote {len(components)} components to JSON")

    print(f"✅ All done – JSON files in {s.output_dir}")

# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="Build JSON netlists from text-removed + original images.")
    ap.add_argument("--input",    "-i", required=True, help="Original images folder (same as index.py --input)")
    ap.add_argument("--text-dir", "-t", required=True, help="Text-removed images folder (output of text removal)")
    ap.add_argument("--out-dir",  "-o", default="netlists_json", help="Output folder for JSON (default: netlists_json)")
    ap.add_argument("--weights",  "-w", default="model/best.pt", help="YOLO weights path")
    return ap.parse_args()

def main():
    args = _parse_args()
    if not os.path.isdir(args.input):
        raise SystemExit(f"Input folder not found: {args.input}")
    if not os.path.isdir(args.text_dir):
        raise SystemExit(f"Text-removed folder not found: {args.text_dir}")
    s = Settings(
        image_dir=args.input,
        text_dir=args.text_dir,
        output_dir=args.out_dir,
        model_weights=args.weights,
    )
    generate_netlists(s)

if __name__ == "__main__":
    main()