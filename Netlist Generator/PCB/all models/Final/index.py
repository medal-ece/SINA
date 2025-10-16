#!/usr/bin/env python3
"""
index.py

Four-stage orchestrator:
  1) text_removal.process_folder(...)     -> writes cleaned images to --output  (cleaned here if exists)
  2) pin.process_all_images(...)          -> reads from --output and writes pins to --pin-output (cleaned here if exists)
  3) netlists_json.generate_netlists(...) -> reads originals (--input) + text-removed (--output) and writes JSON
  4) spice (convert_one loop)     -> reads JSON (default: netlists_json) and writes SPICE .cir files

Notes:
- Google Vision credentials are hard-coded inside the modules to 'credentials.json'.
- YOLO weights default to 'model/best.pt' but can be overridden with --weights.
"""
import argparse
import os
import shutil

# stage 1: text removal
from text_removal import Settings as TextSettings, process_folder as run_text_clean
# stage 2: pin detection
from pin import Settings as PinSettings, process_all_images as run_pins
# stage 3: JSON netlist assembly
from jsonnetlists import Settings as NetSettings, generate_netlists as run_netlists
# stage 4: JSON -> SPICE converter (we'll call convert_one in a loop)
from spice import convert_one as convert_json_to_spice


def parse_args():
    ap = argparse.ArgumentParser(
        description="Schematic → text-removed → pin detection → JSON → SPICE pipeline"
    )
    # ---- text-removal inputs/outputs ----
    ap.add_argument("--input",  "-i", required=True, help="Input folder with original images")
    ap.add_argument("--output", "-o", required=True, help="Output folder for cleaned (text-removed) images")

    # text-removal tuning (match defaults in text_removal.Settings)
    ap.add_argument("--expand", type=int, default=1, help="Pixels to expand around OCR symbol bbox")
    ap.add_argument("--sat",    type=int, default=30, help="Max saturation to be considered grey")
    ap.add_argument("--val",    type=int, default=150, help="Min brightness to be considered grey")
    ap.add_argument("--scale",  type=float, default=4.0, help="Upscale factor for tiny text OCR")
    ap.add_argument("--method", choices=["white", "telea", "ns"], default="white",
                    help="How to remove text: 'white' fill, or inpaint 'telea'/'ns'")

    # ---- pin-detection options ----
    ap.add_argument("--pin-output", "-pout", default="pin",
                    help="Output folder for pin detection results (annotated images + positions)")
    ap.add_argument("--weights", "-w", default="model/best.pt",
                    help="Path to YOLO model weights for symbol detection")
    ap.add_argument("--yolo-conf", type=float, default=0.7,
                    help="YOLO confidence threshold (default 0.7)")

    # ---- netlist assembly (JSON) ----
    ap.add_argument("--netlist-output", "-nout", default="netlists_json",
                    help="Output folder for JSON netlists (default: netlists_json)")

    # ---- SPICE conversion (JSON -> .cir) ----
    ap.add_argument("--spice-json-dir", default=None,
                    help="Folder containing JSON netlists to convert (default: use --netlist-output)")
    ap.add_argument("--spice-output", "-sout", default="spice_netlists",
                    help="Output folder for SPICE .cir files (default: spice_netlists)")

    return ap.parse_args()


def main():
    args = parse_args()

    # Basic checks
    if not os.path.isdir(args.input):
        raise SystemExit(f"Input folder not found: {args.input}")
    if os.path.abspath(args.output) == os.path.abspath(args.input):
        raise SystemExit("--output must be different from --input to avoid deleting your originals.")

    # -------------------------------
    # Pre-clean Stage 1 & Stage 2 outputs
    # -------------------------------
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    if os.path.exists(args.pin_output):
        shutil.rmtree(args.pin_output)

    # -------------------------------
    # Stage 1: Text removal
    # -------------------------------
    print("==> [1/4] Removing grey text…")
    tsettings = TextSettings(
        image_dir=args.input,
        output_dir=args.output,
        symbol_expand=args.expand,
        sat_thresh=args.sat,
        val_thresh=args.val,
        full_scale=args.scale,
        inpaint_method=args.method,
    )
    run_text_clean(tsettings)
    print(f"==> Text removal complete. Clean images in: {args.output}")

    # -------------------------------
    # Stage 2: Pin detection
    # -------------------------------
    print("==> [2/4] Running pin detection…")
    psettings = PinSettings(
        text_dir=args.output,        # text-removed output from Stage 1
        data_dir=args.input,         # originals
        output_dir=args.pin_output,  # annotated images + positions/
        model_weights=args.weights,
        conf_threshold=args.yolo_conf,
    )
    run_pins(psettings)
    print(f"==> Pin detection complete. Results in: {args.pin_output}")

    # -------------------------------
    # Stage 3: JSON netlist assembly
    # -------------------------------
    print("==> [3/4] Building JSON netlists…")
    nsettings = NetSettings(
        image_dir=args.input,            # originals
        text_dir=args.output,            # text-removed
        output_dir=args.netlist_output,  # JSON output
        model_weights=args.weights,
    )
    run_netlists(nsettings)
    print(f"==> Netlists written to: {args.netlist_output}")

    # -------------------------------
    # Stage 4: JSON → SPICE conversion
    # -------------------------------
    spice_json_dir = args.spice_json_dir or args.netlist_output  # default to netlists_json
    print(f"==> [4/4] Converting JSON → SPICE… (from: {spice_json_dir})")

    if not os.path.isdir(spice_json_dir):
        raise SystemExit(f"JSON input folder not found: {spice_json_dir}")

    # auto-clean SPICE output
    if os.path.exists(args.spice_output):
        shutil.rmtree(args.spice_output)
    os.makedirs(args.spice_output, exist_ok=True)

    json_files = [f for f in sorted(os.listdir(spice_json_dir)) if f.lower().endswith(".json")]
    if not json_files:
        print(f"⚠ No .json files found in {spice_json_dir}")
    else:
        for fn in json_files:
            convert_json_to_spice(os.path.join(spice_json_dir, fn), args.spice_output)

    print(f"==> SPICE netlists written to: {args.spice_output}")


if __name__ == "__main__":
    main()
