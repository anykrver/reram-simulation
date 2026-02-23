# Neuro-Edge Hardware Modules (Verilog)

This directory contains the RTL implementation of the digital hardware surrounding the ReRAM crossbar.

## Modules

### 1. Spike Encoder (`spike_encoder.v`)
- **Function**: Converts multi-bit input rates (weights/activations) into a Poisson spike train.
- **Implementation**: Uses a 16-bit Galois LFSR as a pseudo-random number generator. A spike is generated in a cycle if the LFSR value is less than the target `rate`.
- **Hardware Mapping**: Maps to the row-driver circuitry that applies voltage pulses to the crossbar.

### 2. Crossbar Controller (`crossbar_controller.v`)
- **Function**: Orchestrates the multi-cycle integration process.
- **FSM States**:
  - `S_IDLE`: Waiting for `start` signal.
  - `S_CLEAR`: Resets the bitline accumulators.
  - `S_INTEGRATE`: Drives the rows for `TIMESTEPS` cycles.
  - `S_SAMPLE`: Triggers the final readout sampling.
  - `S_DONE`: Signals completion.

### 3. Accumulator (`accumulator.v`)
- **Function**: Sums the current pulses (or digitised spikes) on each bitline over the integration period.
- **Parameters**: 
  - `ACC_WIDTH`: Internal bit-width to prevent overflow (e.g., 16-bit for 50-200 timesteps).
  - `COLS`: Number of parallel bitlines.
- **Hardware Mapping**: Represents the integrate-and-fire or ADC-based readout at the bottom of the crossbar columns.

## Simulation & Synthesis
These modules are written in synthesisable Verilog-2001 and are designed for FPGA or ASIC implementation.

| Module | Resource Estimate |
| --- | --- |
| `spike_encoder` | ~20 LUTs / 17 FFs |
| `crossbar_controller` | ~15 LUTs / 12 FFs |
| `accumulator` | ~10 LUTs per bitline |
