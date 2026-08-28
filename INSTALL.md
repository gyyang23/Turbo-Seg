## Installation

### Requirements
- Linux with Python ≥ 3.10
- PyTorch ≥ 2.4.1 is recommended and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: install from local source in editable mode (version 0.6): `pip install -e detectron2/`.
- open_clip: install from local source in editable mode (version 2.10.1): `pip install -e open_clip/`.
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

An example of installation is shown below:

```
git clone https://github.com/gyyang23/Turbo-Seg.git
cd Turbo-Seg
conda create -n turboseg python=3.10.18
conda activate turboseg
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e detectron2/ --no-build-isolation
pip install -e open_clip/ --no-build-isolation
```
