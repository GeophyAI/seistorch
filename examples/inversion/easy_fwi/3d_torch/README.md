# 3D FWI By PyTorch
This is a simple code for illustrating how to perform 3DFWI(acoustic) by pytorch with automatic differentiation.

# Forward modeling
It's really easy to modify the 2d inversion codes to 3d case. The only thing you need to do is to modify the forward modeling function (change `conv2d` to `conv3d`).

# Usage
```bash
# 1. Generate the ground truth and initial model
python generate_model.py
# 2. Perform inversion
python python fwi.py
```

*Note*: due to the limitation of my GPU memory, I only use a tiny model and 1 shot to perform the inversion. You can change the model size in `configures.py` to a larger one.