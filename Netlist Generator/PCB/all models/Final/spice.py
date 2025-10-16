#!/usr/bin/env python3
"""
spice.py

Convert per-image JSON netlists (from netlists_json.py) into simple SPICE .cir files.

Defaults:
  --json-dir  netlists_json
  --out-dir   spice_netlists

Behavior:
  - Auto-cleans the output directory at the start of each run.
  - For each component:
      * If pin_count == 2 → writes a resistor placeholder: R<id> <n1> <n2> 1k
      * Else → writes a generic subckt placeholder: X<id> <n1> <n2> ... SUBCKT
  - Any unlabeled pin gets a unique net name N1, N2, ...
"""

import os
import json
import argparse
import shutil


def convert_one(json_path: str, out_dir: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We'll assign a unique net name to any unlabeled pin, per-file
    next_net = 1

    def get_net(label: str) -> str:
        nonlocal next_net
        lbl = (label or "").strip()
        if lbl:
            return lbl
        name = f"N{next_net}"
        next_net += 1
        return name

    components = data.get("components", [])
    comp_nets = {}
    for comp in components:
        cid = comp.get("component_id")
        pins = comp.get("pins", [])
        nets = [get_net(p.get("label", "")) for p in pins]
        comp_nets[cid] = nets

    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(out_dir, base + ".cir")

    with open(out_path, "w", encoding="utf-8") as of:
        of.write(f"* SPICE netlist for {data.get('image', base)}\n")

        for comp in components:
            cid = comp.get("component_id")
            nets = comp_nets.get(cid, [])
            pin_count = comp.get("pin_count", len(nets))

            if pin_count == 2:
                # two-pin device → resistor placeholder (adjust later as needed)
                if len(nets) < 2:
                    # pad with auto nets if JSON pin array was shorter than pin_count
                    while len(nets) < 2:
                        nets.append(get_net(""))
                line = f"R{cid} {nets[0]} {nets[1]} 1k"
            else:
                node_list = " ".join(nets)
                line = f"X{cid} {node_list} SUBCKT"

            of.write(line + "\n")

        of.write(".end\n")

    print(f"✔ Converted {os.path.basename(json_path)} → {os.path.basename(out_path)}")


def main():
    ap = argparse.ArgumentParser(description="Convert JSON netlists to SPICE .cir files.")
    ap.add_argument("--json-dir", "-j", default="netlists_json",
                    help="Folder containing JSON netlists (default: netlists_json)")
    ap.add_argument("--out-dir", "-o", default="spice_netlists",
                    help="Output folder for .cir files (default: spice_netlists)")
    args = ap.parse_args()

    if not os.path.isdir(args.json_dir):
        raise SystemExit(f"JSON input folder not found: {args.json_dir}")

    # Auto-clean output directory
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    json_files = [f for f in sorted(os.listdir(args.json_dir)) if f.lower().endswith(".json")]
    if not json_files:
        print(f"⚠ No .json files found in {args.json_dir}")
        return

    for fn in json_files:
        convert_one(os.path.join(args.json_dir, fn), args.out_dir)

    print(f"✅ All done – SPICE files in {args.out_dir}")


if __name__ == "__main__":
    main()
