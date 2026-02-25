# Neuro-Edge ReRAM Simulator

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://reram-simulation-hfnk8pm6bnjdhgxfp2hpag.streamlit.app/)
[![Tests](https://img.shields.io/badge/Tests-11%2F11_Passing-00C853?style=for-the-badge)](#tests)
[![Hardware Target](https://img.shields.io/badge/Versal-FPGA_Ready-orange?style=for-the-badge&logo=xilinx)](./verilog)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](./LICENSE)

---

## Overview

**Neuro-Edge** is a hardware-accurate simulator for Resistive RAM (ReRAM) crossbar arrays, purpose-built for neuromorphic computing research. It models the full pipeline from analog in-memory computing to spiking neural network inference.

### Key Features

- ⚡ **Crossbar Simulation** — Ideal (Ohm's law) and non-ideal (noise, IR drop, variability, quantization).
- 🧠 **Spiking Neural Network (SNN)** — Poisson spike encoding, LIF neurons, and rate-coded surrogate-gradient training.
- 🎯 **MNIST Classification** — Achieved **~80–85% accuracy** on a single-layer crossbar (10 epochs).
- 🔬 **Silicon Lab Dashboard** — Interactive Streamlit console with live heatmaps, spike rasters, and power profiling.
- 🏗️ **Synthesisable RTL** — SystemVerilog modules targeting Xilinx Versal ACAP (spike encoder, crossbar controller, accumulator).
- ⚙️ **Energy Estimation** — Physics-based model: E = V² × G × t.

---

## 🚀 Live Demo

👉 **[Launch Silicon Lab Console](https://reram-simulation-hfnk8pm6bnjdhgxfp2hpag.streamlit.app/)**

---

## Installation

```bash
git clone https://github.com/anykrver/reram-simulation.git
cd reram-simulation

python -m venv venv
# Windows: venv\Scripts\activate
# Unix:    source venv/bin/activate

pip install -r requirements.txt
```

---

## How to Run

### 1. Dashboard (Streamlit)
```bash
streamlit run dashboard/app.py
```
Visualize crossbar conductances, spike rasters, power curves, and run MNIST inference.

### 2. Training (MNIST)
```bash
python experiments/train_mnist.py --epochs 10 --batch-size 64
```
Trained weights are saved to `experiments/trained_weights.npy`.

### 3. CLI Simulations
```bash
# Ideal crossbar
python src/main.py --mode ideal

# Non-ideal (noise + IR drop + variability)
python src/main.py --config configs/non_ideal.yaml --mode both
```

### 4. Tests
```bash
pytest tests/ -v
```

---

## Project Structure

```
reram-simulation/
├── src/
│   ├── crossbar/        # Crossbar physics (ideal, IR drop, variability, quantization)
│   ├── snn/             # SNN engine (LIF neurons, Poisson encoder, trainer)
│   ├── hardware/        # Energy estimator, accelerator model, controller
│   └── utils/           # Config loader, logger, metrics, visualization, weight I/O
├── dashboard/           # Streamlit Silicon Lab Console
├── configs/             # Simulation YAML configs (ideal, non-ideal, SNN)
├── experiments/         # Training scripts & Jupyter notebooks
├── tests/               # Unit tests (11/11 passing)
├── verilog/             # Synthesisable RTL for Xilinx Versal ACAP
├── docs/                # Architecture, energy model, fabrication notes
├── .streamlit/          # Streamlit Cloud config
├── .github/workflows/   # CI pipeline
├── pyproject.toml       # Package metadata & dependencies
├── requirements.txt     # Pip dependencies
└── README.md
```

---

## Results

| Configuration | MNIST Accuracy | Energy/Op |
|---|---|---|
| Random Weights | ~10–15% | ~15 µJ |
| **Trained SNN (10 epochs)** | **~80–85%** | **~15 µJ** |

---

## Hardware (Verilog)

The `verilog/` directory contains synthesisable SystemVerilog for the **Xilinx Versal ACAP** (`xcvc1902`):

| Module | Description |
|---|---|
| `spike_encoder.sv` | LFSR-based Poisson spike generator |
| `crossbar_controller.sv` | Multi-cycle VMM integration FSM |
| `accumulator.sv` | Bitline spike counter (32-bit) |
| `top_neuro_edge.sv` | Top-level pipeline wrapper |
| `tb_neuro_edge.sv` | Testbench |

**Estimated resources:** ~450 LUT, ~320 FF, 0 DSP @ 250 MHz per 32×10 core.

---

## Future Work

- Multi-layer SNN architectures with backpropagation.
- FPGA-in-the-loop simulation using the Verilog RTL.
- High-fidelity fab-calibrated IR drop models.

---

## License

[MIT](./LICENSE)
