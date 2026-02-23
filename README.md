# Neuro-Edge ReRAM Simulator

Research-grade simulation framework for **in-memory computing** using ReRAM crossbars: ideal and non-ideal models, SNN integration, energy estimation, and a Streamlit-Wasm dashboard.

[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-00F0FF?style=for-the-badge&logo=vercel)](https://reram-simulation-main.vercel.app)
[![Hardware Target](https://img.shields.io/badge/Versal-Ready-orange?style=for-the-badge&logo=xilinx)](./verilog)

---

## Overview

Neuro-Edge ReRAM Simulator models resistive crossbar arrays used for matrix-vector multiplication in place: voltages \(V\) on rows, conductance \(G\) at cells, currents \(I = V \times G\) on columns. It supports:

- **🚀 Live View**: Web-based [Silicon Lab Console](https://reram-simulation-main.vercel.app) (Zero-install via stlite).
- **Ideal crossbar**: Pure matrix-vector multiplication.
- **Non-ideal modeling**: Accounts for noise, IR drop, variability, and quantization.
- **SNN Integration**: Rate-coded spiking neural networks using Poisson encoding and LIF neurons.
- **Training**: Surrogate-gradient based training achieved **~80% MNIST accuracy** on a single-layer crossbar.
- **Hardware (Xilinx Versal)**: Synthesisable SystemVerilog modules with a [Top-Level Wrapper](./verilog/top_neuro_edge.sv).
- **Energy Model**: Theoretical estimation based on \(E = V^2 \times G \times t\).

---

## Installation (Local Tooling)

```bash
git clone https://github.com/anykrver/reram-simulation.git
cd reram-simulation

python -m venv venv
# Windows: venv\Scripts\activate
# Unix:    source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## How to Run

### 1. Training (MNIST)
To train the crossbar weights for meaningful classification:
```bash
python experiments/train_mnist.py --epochs 10 --batch-size 64
```
Weights will be saved to `experiments/trained_weights.npy`.

### 2. Dashboard (Streamlit)
```bash
streamlit run dashboard/app.py
```
- Visualize conductances, firing rasters, and power curves.
- **Run MNIST test**: Toggle between random weights and trained weights.

### 3. CLI Simulations
```bash
# Ideal mode
python src/main.py --mode ideal
# Ideal vs Non-Ideal comparison
python src/main.py --config configs/non_ideal.yaml --mode both
```

### 4. Tests
```bash
pytest tests/ -v
```

---

## Project Layout

```
neuro-edge-reram-simulator/
├── configs/          # Simulation parameters (YAML)
├── dashboard/       # Streamlit visualization dashboard
├── experiments/      # Research notebooks & training scripts
├── src/
│   ├── crossbar/     # Crossbar physics (Ideal, IR Drop, Noise, etc.)
│   ├── snn/          # SNN logic (Neurons, Encoders, Trainers)
│   ├── hardware/     # Energy & Accelerator modeling
│   └── utils/        # Persistance (Weight I/O) & Metrics
├── verilog/         # Functional RTL (Spike Encoder, Controller, Accumulator)
└── tests/           # Full unit test suite (11/11 passing)
```

---

## Results

| Mode | MNIST Accuracy (Test) | Energy/Op (Avg) |
|---|---|---|
| **Random Weights** | ~10-15% | ~15 µJ |
| **Trained SNN** | **~80-85%** | ~15 µJ |

*(Note: Accuracy achieved on a single-layer dense crossbar after 10 epochs.)*

---

## Future Work

- Deeper multi-layer SNN architectures.
- FPGA-in-the-loop simulation using the provided Verilog RTL.
- High-fidelity fab-calibrated IR drop models.

---

## License

MIT.

# reram-simulation
