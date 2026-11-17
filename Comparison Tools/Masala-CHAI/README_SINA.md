# Masala-CHAI — as run for the SINA comparison

Source: https://github.com/jitendra-bhandari/Masala-CHAI (arXiv 2411.14299). `README.md` is the authors'.

**Our decisions**
- Run end to end as released: its own YOLOv8 detector, Hough-based wire detection and GPT-4o.
- **One code fix:** `utils/yolov8.py` loaded the detector from the authors' machine (`/home/jb7410/...`). It now loads the checkpoint shipped in the repo (`./trained_checkpoints/yolov8_best.pt`).
- Its GPT-4o call has no fixed temperature, so the final netlist can vary between runs.

**Setup** (Linux or WSL2, conda, Python 3.9)
```bash
conda env create -f environment.yml   # the pip step fails on opencv-python-headless ("no RECORD file"); continue below
conda activate autospice_env
pip install -r requirements_pip.txt   # environment.yml's pip packages without the two opencv lines
pip install --no-deps --ignore-installed opencv-python==4.10.0.84 opencv-python-headless==4.10.0.84
pip install --no-deps numpy==1.23.0
```
Download `hough/` (https://drive.google.com/file/d/1mTwWWSMsYwhJW-GfKKVm21Lm5lVtMJ1y) into `hough/` so that `hough/checkpoint.pth` exists. The first run also builds `vote_index_128_128_3_1.mat` (1.4 GB, a few minutes).

**Run** (from this folder)
```bash
python run.py --src <image folder>/ --tgt <output folder> --api_key <OpenAI key>
```
Netlist per image: `<out>/<image name + extension>/spice.txt` (e.g. `circuit_0001jpg/spice.txt`)
