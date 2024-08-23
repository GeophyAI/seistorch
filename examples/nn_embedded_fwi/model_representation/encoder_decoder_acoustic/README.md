# Usage

## Full band data
Step 1. Generate the models
```bash
python generate_model.py
```
Step 2. Simulate the observed data
```bash
python forward.py
```
Step 3. Run inversion
```bash
python ifwi.py
```

## Data with missing low frequencies
Step 1. Generate the models
```bash
python generate_model.py
```
Step 2. Simulate the observed data
```bash
python forward.py
```
Step 3. High-pass filter the observed data and wavelet
```bash
python bandpass.py
```
Step 4. Run inversion
```bash
python ifwi_cycleskipping.py
```