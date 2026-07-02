# Requirements note

## Recommended default install

```bash
pip install -r requirements.txt
```

`requirements.txt` uses `opencv-contrib-python` instead of `opencv-python` so that future OpenCV contrib features are available in the same environment. Do not install `opencv-python` and `opencv-contrib-python` together, because both provide the same `cv2` namespace.

## Kornia is optional

The current pipeline is deterministic and implemented with NumPy + OpenCV remap. Kornia is not required for the current modules.

Use Kornia only for later research/prototype work such as differentiable pose refinement, differentiable projection-map optimization, or PyTorch/GPU batch warping:

```bash
pip install -r requirements_optional_kornia.txt
```

Kornia is PyTorch-based, so installing it will also involve the PyTorch stack, which is much heavier than the default OpenCV/NumPy dependency set.
