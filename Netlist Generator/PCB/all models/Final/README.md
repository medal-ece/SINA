# PCB Netlist Generator

Turn your schematic images into JSON netlists and SPICE `.cir` files with a four-stage pipeline:

1. **Text removal** (Google Vision OCR)
2. **Component & pin detection** (YOLO + wire skeletons)
3. **JSON netlist** export (components, pins, connectivity)
4. **SPICE** export (placeholder `.cir` files)

Works with `.png`, `.jpg`, `.jpeg`, `.webp`.

---

## TL;DR — One command

```bash
# from the project root (where index.py lives)
python index.py \
  --input "/path/to/input_images" \
  --output "/path/to/text_removed_out" \
  --pin-output "/path/to/pin_out" \
  --netlist-output "/path/to/json_out" \
  --spice-output "/path/to/spice_out" \
  --weights "/path/to/model/best.pt"
```

This writes:

* **Text-removed images** → your `--output` folder
* **Annotated images + pin coordinate files** → your `--pin-output` folder (with `positions/*.txt`)
* **Per-image JSON netlists** → your `--netlist-output` folder
* **SPICE `.cir` files** → your `--spice-output` folder

> Outputs are **auto-cleaned** before each run. If you want to keep previous results, point each stage to a **different folder**.

---

## Requirements

* **Python 3.9+** (3.10/3.11 recommended)
* Install packages:

  ```bash
  pip install opencv-python numpy ultralytics scikit-image google-cloud-vision google-auth
  ```
* **YOLO weights** (e.g., `model/best.pt`) — pass the path via `--weights`
* **Google Cloud Vision credentials** saved as `credentials.json` in the project root
  (how to create this is below)

> GPU is optional. Ultralytics will use CUDA automatically if available.

---

## Set up Google Cloud Vision (your own `credentials.json`)

You do **not** commit credentials. Each user creates their own and places them at the project root as **`credentials.json`** (the scripts read this exact path).

1. **Create/select a GCP project**
   [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. **Enable the Vision API**
   Console → **APIs & Services → Library** → search **“Vision API”** → **Enable**.
3. **Create a service account**
   Console → **IAM & Admin → Service Accounts** → **Create service account**
   Give it a name → **Create** → grant a role allowed to call Vision API
   (for testing: **Project → Editor**; for production: least-privilege Vision role).
4. **Create & download a key (JSON)**
   Service account → **Keys** → **Add key** → **Create new key** → **JSON** → download.
5. **Place the key file**
   Save/rename it in your repo root as:

   ```
   credentials.json
   ```

That’s it. No env vars required; the code reads this file directly.

---

## Suggested project layout

```
.
├─ index.py
├─ text_removal.py
├─ pin.py
├─ netlists_json.py
├─ json_to_spice.py
├─ model/
│  └─ best.pt                  # your YOLO weights (or pass path via --weights)
├─ credentials.json            # your Vision API service account key
├─ <your input folder>         # images you want to process
├─ <your text-removed out>     # Stage 1 output (auto-created/cleaned)
├─ <your pin out>              # Stage 2 output (auto-created/cleaned)
│  └─ positions/
├─ <your json out>             # Stage 3 output (auto-created/cleaned)
└─ <your spice out>            # Stage 4 output (auto-created/cleaned)
```

---

## The pipeline (`index.py`)

`index.py` orchestrates all four stages:

1. **Text removal** (`text_removal.py`)
   Finds grey text using OCR and removes (white-fill or inpaint).

   * Output: `--output` (your text-removed folder)

2. **Pin detection** (`pin.py`)
   YOLO detects components; colored wires are skeletonized; pins found near boxes.

   * Output: `--pin-output` (annotated images + `positions/*.txt`)

3. **JSON netlists** (`netlists_json.py`)
   Builds per-image JSON: components, pins (labels), connectivity.

   * Output: `--netlist-output`

4. **SPICE** (`json_to_spice.py`)
   Converts JSON to `.cir` (R for 2-pin, X for multi-pin).

   * Output: `--spice-output`

### CLI options

```bash
python index.py \
  --input </path/to/input_images> \
  --output </path/to/text_removed_out> \
  --pin-output </path/to/pin_out> \
  --netlist-output </path/to/json_out> \
  --spice-output </path/to/spice_out> \
  [--weights </path/to/model/best.pt>] \
  [--yolo-conf 0.7] \
  [--spice-json-dir </path/to/json_src>] \
  [--expand 1 --sat 30 --val 150 --scale 4.0 --method white]
```

* `--weights` — YOLO weights path (default is `model/best.pt` if present)
* `--yolo-conf` — YOLO confidence threshold (default `0.7`)
* **Text removal tuning**:

  * `--expand` — pixels around OCR symbol bbox
  * `--sat` — max saturation classed as “grey”
  * `--val` — min brightness classed as “grey”
  * `--scale` — OCR upsample factor for tiny text
  * `--method` — `white` | `telea` | `ns`
* **SPICE source**:

  * `--spice-json-dir` — where to read JSONs from (defaults to `--netlist-output`)

> Safety: `index.py` **pre-cleans** the folders passed to `--output` and `--pin-output`.
> Stage 3 & 4 **auto-clean** their outputs. Use unique folders if you want to keep old results.

---

## Running individual stages

> Each stage **cleans its output folder** when run. If you want to keep older results, choose a different output path.

### Stage 1 — Text removal

```python
from text_removal import Settings, process_folder

process_folder(Settings(
    image_dir="/path/to/input_images",
    output_dir="/path/to/text_removed_out",
    symbol_expand=1, sat_thresh=30, val_thresh=150,
    full_scale=4.0, inpaint_method="white"  # or "telea"/"ns"
))
```

### Stage 2 — Pin detection

```python
from pin import Settings, process_all_images

process_all_images(Settings(
    text_dir="/path/to/text_removed_out",  # Stage 1 output
    data_dir="/path/to/input_images",      # same images as --input
    output_dir="/path/to/pin_out",         # annotated + positions/
    model_weights="/path/to/model/best.pt",
    conf_threshold=0.7
))
```

### Stage 3 — JSON netlists

```bash
python netlists_json.py \
  --input "/path/to/input_images" \
  --text-dir "/path/to/text_removed_out" \
  --out-dir "/path/to/json_out" \
  --weights "/path/to/model/best.pt"
```

### Stage 4 — SPICE

```bash
python json_to_spice.py \
  --json-dir "/path/to/json_out" \
  --out-dir "/path/to/spice_out"
```

---

## New runs vs. continuing runs

* **Fresh run on new images**
  Place your images in an input folder (e.g., `/path/to/input_images`) and run the full pipeline with your chosen output folders.

* **Continue from a middle stage**

  * If you already have **text-removed images**, you can run Stage 2–4 only.
  * If you already have **pin outputs**, you can skip to JSON/SPICE.

    > By design, `netlists_json.py` recomputes pins internally. If you want it to read `positions/*.txt` verbatim, that script can be adapted.

**Tip:** To keep older results, pick new output folder names (e.g., `.../pin_v2`, `.../json_v2`, …).

---

## Output formats

### JSON (from Stage 3)

```json
{
  "image": "your_image.png",
  "components": [
    {
      "component_id": 1,
      "bbox": [x0, y0, x1, y1],
      "pin_count": 2,
      "pins": [
        {"x": 123, "y": 456, "label": "VCC"},
        {"x": 234, "y": 456, "label": ""}
      ],
      "connected": [2, 5]
    }
  ]
}
```

### SPICE (from Stage 4)

```
* SPICE netlist for your_image.png
R1 VCC N1 1k
X2 N2 N3 N4 SUBCKT
.end
```

* 2-pin components → `R<id> <n1> <n2> 1k` (placeholder)
* N-pin components → `X<id> <nodes...> SUBCKT`
* Unlabeled pins are auto-named `N1`, `N2`, …

---

## Troubleshooting

* **“Missing image” at later stages**
  Make sure filenames in your text-removed folder match filenames in your input folder.

* **Vision API auth/permission errors**
  Check that `credentials.json` exists in the repo root, your project has Vision API enabled, and the service account has an appropriate role.

* **Too few/many detections**
  Try a different `--yolo-conf` (e.g., `0.5` for more boxes) and verify your weights path.

* **OCR is slow**
  Text removal calls Vision twice per image (native + upsampled). Lower `--scale` or process fewer images per run.

---

## Notes & assumptions

* Filenames in the text-removed folder should match those in the input folder.
* Wire colors assumed **red/green**; HSV thresholds live in `pin.py` / `netlists_json.py`.
* Pin proximity radius is `PIN_DIST = 3`.
* Credentials file path is hard-coded: `credentials.json`.

---

## Credits

* **Ultralytics YOLO**, **OpenCV**, **scikit-image**, **Google Cloud Vision**
* Use in accordance with each dependency’s license and your GCP billing/quotas.
