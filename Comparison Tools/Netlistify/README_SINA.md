# Netlistify — as run for the SINA comparison

Source: https://github.com/NYCU-AI-EDA/Netlistify (MLCAD 2025). The authors' code is unchanged; `README.md` is theirs.

**Our decisions**
- **Component detector = SINA's YOLO.** The released Netlistify model covers wires, connectivity and orientation, but has no component detector that works on our images. `run_netlistify.py` therefore feeds it SINA's YOLO boxes, so both tools start from the same detections. Everything after detection is Netlistify's own code.
- **Class mapping** (Netlistify has fewer classes):
  - inductor is run as a resistor (same 2-pin handling) and renamed `l` afterwards
  - BJT is run as npn and MOSFET as nmos; if the circuit is PNP/PMOS (read from SINA's netlist for that image, `--sina-netlists`), the device is renamed and its pins reordered afterwards
  - op-amps and diodes are not supported and are dropped
- **Only addition:** `run_netlistify.py`.

**Setup** (Windows, Python 3.12, CUDA 12.6)
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements_used.txt --extra-index-url https://download.pytorch.org/whl/cu126
```
Download the pretrained model (https://drive.google.com/file/d/1Jlx9HNfrTIXrjIOyIL3zyy_rDrcfKKzZ) and unzip it here, so that `runs/` and `bubble_orientation/` sit next to `inference.py`.

**Run**
```bash
python run_netlistify.py <image or folder> --yolo "<SINA>/Netlist Generator/IC/current_trained_model/train/weights/best.pt" --sina-netlists <folder of SINA netlists> --out <output folder>
```
Netlist per image: `<out>/<image>/output/<image>_final.sp`
