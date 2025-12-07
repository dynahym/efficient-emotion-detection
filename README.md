# Lightweight CNN Benchmark for Emotion Recognition

*A comparative evaluation of MobileNet (V1/V2/V3), ShuffleNetV2, SqueezeNet, and ShiftNet on facial emotion recognition with FLOPs, parameters, inference time, and carbon footprint.*

## Overview

This project benchmarks several lightweight convolutional architectures on a facial emotion recognition dataset.
The goal is to evaluate **accuracy**, **computational cost**, and **environmental impact** to identify the most efficient models for real-time deployment.

The models implemented and tested:

* **MobileNetV1**
* **MobileNetV2**
* **MobileNetV3 Small**
* **ShuffleNetV2 (0.5x–2.0x)**
* **SqueezeNet**
* **ShiftNet**

All models are implemented **from scratch in PyTorch**, except MobileNetV3 which uses torchvision with input/classifier patches.

---

## Project Structure

```
├── models/
│   ├── mobilenet_v1.py
│   ├── mobilenet_v2.py
│   ├── mobilenet_v3.py
│   ├── shufflenet_v2.py
│   ├── squeezenet.py
│   ├── shiftnet.py
│
├── notebooks/
│   ├── 03_model_evaluation.ipynb      # Full evaluation pipeline
│   ├── 04_analysis.ipynb              # Evaluation summarization + plots
│
├── results.json                       # Saved metrics after evaluation
├── README.md
└── requirements.txt
```

---

## Implemented Architectures

### **🔹 MobileNetV1**

Depthwise-separable convolutions for lightweight inference.

### **🔹 MobileNetV2**

Inverted residuals + linear bottlenecks.

### **🔹 MobileNetV3 (Small)**

Torchvision implementation adapted for grayscale input and 7 classes.

### **🔹 ShuffleNetV2**

Channel shuffle + split branches for efficient memory access.

### **🔹 SqueezeNet**

Fire modules to reduce parameters (≈ 50× smaller than AlexNet).

### **🔹 ShiftNet**

Implements shift-based spatial operations instead of standard convolution.

---

## 📈 Evaluation Metrics

The evaluation notebook computes:

| Metric                     | Description                               |
| -------------------------- | ----------------------------------------- |
| **Accuracy**               | Emotion classification accuracy           |
| **Inference time (total)** | Total runtime over the test set           |
| **Inference speed**        | Time per sample                           |
| **FLOPs**                  | Total number of floating-point operations |
| **Parameter count**        | Model size                                |
| **Energy consumed (kWh)**  | Converted from GPU power logs             |
| **CO₂ emissions (kg)**     | Carbon footprint estimation               |

Results are stored in `results.json`

---

## Analysis Notebook (`04_analysis.ipynb`)

This notebook:

✔ Loads `results.json`
✔ Builds a Pandas summary table
✔ Plots accuracy bars
✔ Prints FLOPs / params / carbon impact for all models

---

## Results (Summary)

After evaluation on the FER dataset:

* Lightweight CNNs can achieve strong accuracy (close to MobileNetV3) with **2–10× fewer FLOPs**.
* Some classical lightweight models (ShuffleNetV2, MobileNetV2) achieve the **lowest carbon footprint**.
* ShiftNet is extremely fast and efficient due to removal of spatial convolution.

(Detailed numbers appear automatically in generated plots and tables.)

---

## 🔬 Research Motivation

As ML models grow, **energy and carbon costs** increase. This project shows that:

* Efficient architectures remain competitive
* FLOPs alone do not predict real energy usage
* Carbon metrics should be included in reproducible benchmarks
