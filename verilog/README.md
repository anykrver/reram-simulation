# Neuro-Edge Hardware Modules (SystemVerilog)

This directory contains the synthesisable RTL implementation for the Neuro-Edge ReRAM Accelerator, now optimized for the **Xilinx Versal ACAP** platform.

## Architecture

The hardware follows a streaming neuromorphic architecture where multi-bit inputs are encoded into stochastic spike trains, processed by a memristive crossbar (emulated in logic for this release), and accumulated for class inference.

### Modules

1. **Top-Level Wrapper ([top_neuro_edge.sv](./top_neuro_edge.sv))**
   - Integrates the full pipeline: Encoder -> Crossbar -> Accumulator.
   - Provides a standard AXI-Stream compatible interface for Versal integration.

2. **Spike Encoder ([spike_encoder.sv](./spike_encoder.sv))**
   - Converts 8-bit rates into Poisson spike trains using a 16-bit Galois LFSR.

3. **Crossbar Controller ([crossbar_controller.sv](./crossbar_controller.sv))**
   - Manages weight loading and multi-cycle vector-matrix multiplication (VMM).

4. **Accumulator ([accumulator.sv](./accumulator.sv))**
   - 32-bit registers to sum firing events over the integration window (e.g., 50-200ms).

## Versal Implementation

Target Device: **Xilinx Versal Prime/AI Core** (`xcvc1902`).

### Synthesis Estimates (per 32x10 core)
| Resource | Multi-Cycle VMM |
| --- | --- |
| **LUT** | ~450 |
| **FF** | ~320 |
| **DSP** | 0 (Purely logic-based) |
| **Clock** | 250 MHz (Target) |

### Usage
Run the [versal_synth.tcl](./versal_synth.tcl) script in Vivado to generate the bitstream or export the hardware platform.
