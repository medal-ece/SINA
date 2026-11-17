"""Run Netlistify on images, using the SINA YOLO model as the component detector.

Usage:
    python run_netlistify.py <image_or_folder> --yolo <SINA best.pt> [--sina-netlists <folder>] [--out runs_yolo] [--conf 0.25]

Classes Netlistify's released (REAL-config) model lacks are mapped to geometrically
equivalent supported classes, then relabeled in the netlist afterwards:
    Inductor          -> resistor  (same 2-pin split: util.py cut_box groups them in split_rcl)
    Transistor_BJT    -> npn       (pnp only reorders pins c,b,e -> e,b,c)
    Transistor_MOSFET -> nmos      (pmos only reorders pins d,g,s,s -> s,g,d,d)
Transistor polarity is taken from SINA's netlist for the same image (.model cards).

Outputs go to <out>/<image_stem>/
    images/<image>            copy of the input
    labels/<stem>.txt         YOLO labels in Netlistify's class scheme
    labels/<stem>_meta.json   original SINA class + mapping per label line
    output/<stem>_output.sp   raw Netlistify netlist (as produced)
    output/<stem>_final.sp    netlist after relabeling unsupported classes
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE                 # this script sits in the Netlistify repository folder
YOLO_WEIGHTS = None         # set from --yolo
SINA_NETLISTS = None        # set from --sina-netlists (optional)
OUT_ROOT = HERE / "runs_yolo"  # set from --out

# SINA class id -> Netlistify REAL-config class id
# SINA:       0 Resistor, 1 Capacitor, 2 Inductor, 3 Transistor_BJT, 4 Transistor_MOSFET,
#             5 Voltage_src, 6 Current_src, 7 GND, 8 Op-Amp, 9 Diode
# Netlistify: 0 gnd, 1 pmos, 2 nmos, 3 pnp, 4 npn, 5 resistor, 6 capacitor, 7 voltage,
#             8 current, 9 text, 10 node, 11 crossing
CLASS_MAP = {0: 5, 1: 6, 2: 5, 3: 4, 4: 2, 5: 7, 6: 8, 7: 0}
SURROGATE = {2: "inductor->resistor", 3: "bjt->npn", 4: "mosfet->nmos"}
UNSUPPORTED = {8: "Op-Amp", 9: "Diode"}

# Netlistify instance-name prefix per Netlistify class id (inference.py type_name)
NL_PREFIX = {0: "gnd", 1: "m", 2: "m", 3: "q", 4: "q", 5: "r", 6: "c", 7: "v", 8: "i"}

# Part numbers SINA may emit instead of a .model card
KNOWN_PARTS = {"BS170": "NMOS", "2N7000": "NMOS", "2N3904": "NPN", "2N2222": "NPN",
               "2N3906": "PNP", "BS250": "PMOS"}


def sina_polarity(stem):
    """Return ({'bjt': 'NPN'|'PNP'|None, 'mos': 'NMOS'|'PMOS'|None}, notes) from SINA's netlist."""
    pol, notes = {"bjt": None, "mos": None}, []
    if SINA_NETLISTS is None:
        return pol, ["no --sina-netlists given; transistors stay N-type"]
    f = SINA_NETLISTS / f"{stem}.txt"
    if not f.exists():
        return pol, [f"no SINA netlist at {f}"]
    lines = f.read_text().splitlines()
    models = {}
    for l in lines:
        m = re.match(r"\.model\s+(\S+)\s+([A-Za-z]+)", l, re.I)
        if m:
            models[m.group(1).upper()] = m.group(2).upper()
    found = {"bjt": set(), "mos": set()}
    in_control = False
    for l in lines:
        tok = l.split()
        if not tok:
            continue
        low = tok[0].lower()
        if low == ".control":
            in_control = True
        elif low == ".endc":
            in_control = False
        if in_control or tok[0].startswith(".") or tok[0][0].upper() not in "QM" or len(tok) < 4:
            continue
        fam = "bjt" if tok[0][0].upper() == "Q" else "mos"
        mname = tok[-1].upper()
        p = models.get(mname) or KNOWN_PARTS.get(mname)
        if p:
            found[fam].add(p)
        else:
            notes.append(f"unknown model '{tok[-1]}' for {tok[0]}")
    for fam, s in found.items():
        if len(s) == 1:
            pol[fam] = s.pop()
        elif len(s) > 1:
            notes.append(f"MIXED {fam} polarity in SINA netlist {sorted(s)}; defaulting to N-type")
    return pol, notes


_yolo = None


def make_labels(img_path: Path, work: Path, conf: float):
    global _yolo
    from ultralytics import YOLO

    if _yolo is None:
        _yolo = YOLO(str(YOLO_WEIGHTS))
    res = _yolo.predict(str(img_path), conf=conf, verbose=False)[0]
    lines, meta, skipped = [], [], []
    for cls, xywhn, c in zip(res.boxes.cls.tolist(), res.boxes.xywhn.tolist(), res.boxes.conf.tolist()):
        cls = int(cls)
        name = _yolo.names[cls]
        if cls in CLASS_MAP:
            x, y, w, h = xywhn
            nl = CLASS_MAP[cls]
            lines.append(f"{nl} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            meta.append({"index": len(meta), "sina_class": name, "sina_id": cls, "netlistify_id": nl,
                         "surrogate": SURROGATE.get(cls), "conf": round(c, 3)})
            tag = f" (surrogate {SURROGATE[cls]})" if cls in SURROGATE else ""
            print(f"  detected {name:<18} conf={c:.2f} -> netlistify class {nl}{tag}")
        else:
            skipped.append(f"{UNSUPPORTED.get(cls, name)} (conf={c:.2f})")
    for s in skipped:
        print(f"  WARNING: skipped unsupported detection: {s}")
    (work / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
    (work / "labels" / f"{img_path.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {len(lines)} boxes")
    return meta, skipped


def relabel(raw_sp: Path, final_sp: Path, meta, pol):
    """Turn surrogate components back into their real types."""
    by_index = {m["index"]: m for m in meta}
    out, changes = [], []
    for line in raw_sp.read_text().splitlines():
        tok = line.split()
        m = re.match(r"^([a-z]+)(\d+)$", tok[0]) if tok else None
        if not m or line.startswith("."):
            out.append(line)
            continue
        idx = int(m.group(2))
        info = by_index.get(idx)
        sur = info and info["surrogate"]
        name, pins, model = tok[0], tok[1:-1], tok[-1]
        if sur == "inductor->resistor":
            name, model = f"l{idx}", "l"
            changes.append(f"{tok[0]} -> {name} (inductor)")
        elif sur == "bjt->npn" and pol["bjt"] == "PNP":
            pins, model = [pins[2], pins[1], pins[0]], "pnp"   # (c,b,e) -> (e,b,c)
            changes.append(f"{name} npn -> pnp (pins reordered)")
        elif sur == "mosfet->nmos" and pol["mos"] == "PMOS":
            pins, model = [pins[2], pins[1], pins[0], pins[0]], "pmos4"  # (d,g,s,s) -> (s,g,d,d)
            changes.append(f"{name} nmos -> pmos (pins reordered)")
        out.append(" ".join([name, *pins, model]))
    final_sp.write_text("\n".join(out) + "\n")
    return changes


_inference = None


def run_netlistify(img_path: Path, out_dir: Path):
    # inference.py runs its batch loop (which wipes ./results) at module level,
    # so execute only the definitions above that loop (once).
    global _inference
    if _inference is None:
        src = (REPO / "inference.py").read_text(encoding="utf-8")
        src = src.split("is_code_ocean = False")[0]
        os.chdir(REPO)  # weights are loaded via relative paths
        sys.path[:0] = [str(REPO), str(REPO / "Visualizer")]
        ns = {"__name__": "netlistify_inference", "__file__": str(REPO / "inference.py")}
        exec(compile(src, str(REPO / "inference.py"), "exec"), ns)
        _inference = ns["inference"]
    return _inference(str(img_path), output_folder=str(out_dir) + "/")


def process(src_img: Path, conf: float):
    work = OUT_ROOT / src_img.stem
    # overwrite in place (folders may be open in Explorer, so don't delete them)
    out = work / "output"
    for sub in ("images", "labels", "output"):
        (work / sub).mkdir(parents=True, exist_ok=True)
        for f in (work / sub).iterdir():
            if f.is_file():
                f.unlink()
    img = work / "images" / src_img.name
    shutil.copy2(src_img, img)

    print(f"\n######## {src_img.name} ########")
    print("[1/3] YOLO detection")
    meta, skipped = make_labels(img, work, conf)
    print("[2/3] Netlistify connectivity")
    run_netlistify(img, out)
    print("[3/3] Relabel surrogate classes")
    pol, notes = sina_polarity(src_img.stem)
    needs_pol = any(m["surrogate"] in ("bjt->npn", "mosfet->nmos") for m in meta)
    if needs_pol:
        print(f"  polarity from SINA: BJT={pol['bjt']} MOSFET={pol['mos']}")
        for n in notes:
            print(f"  WARNING: {n}")
    changes = relabel(out / f"{src_img.stem}_output.sp", out / f"{src_img.stem}_final.sp", meta, pol)
    for c in changes:
        print(f"  relabeled {c}")
    print("===== final netlist =====")
    print((out / f"{src_img.stem}_final.sp").read_text())
    (work / "run_meta.json").write_text(json.dumps({
        "image": src_img.name, "skipped_unsupported": skipped, "polarity": pol,
        "polarity_notes": notes, "relabel_changes": changes,
        "surrogates": [m["surrogate"] for m in meta if m["surrogate"]],
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="an image file or a folder of images")
    ap.add_argument("--yolo", required=True,
                    help="SINA YOLO weights (Netlist Generator/IC/current_trained_model/train/weights/best.pt)")
    ap.add_argument("--sina-netlists", default=None,
                    help="folder of SINA netlists (<image_stem>.txt) used only for BJT/MOSFET polarity")
    ap.add_argument("--out", default=str(HERE / "runs_yolo"), help="output folder")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    global YOLO_WEIGHTS, SINA_NETLISTS, OUT_ROOT
    YOLO_WEIGHTS = Path(args.yolo).resolve()
    SINA_NETLISTS = Path(args.sina_netlists).resolve() if args.sina_netlists else None
    OUT_ROOT = Path(args.out).resolve()

    src = Path(args.input).resolve()
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        imgs = [src]
    for p in imgs:
        try:
            process(p, args.conf)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"FAILED: {p.name}")


if __name__ == "__main__":
    main()
