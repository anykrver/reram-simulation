# Fabrication Notes

Short notes on variability, IR drop, and process assumptions.

## Variability

- **Device-to-device**: Conductance varies across cells (e.g. Gaussian around nominal). Modeled in `src/crossbar/variability.py` (`add_conductance_variability`, `add_variability_percent`).
- **Cycle-to-cycle**: Same device can vary between reads; same noise model can be applied per read.

## IR drop

- Resistive interconnects cause voltage drop along rows and columns. Modeled in `src/crossbar/ir_drop_model.py` (`apply_ir_drop`) with row/column resistance and iterative update.

## Process assumptions

- Conductance non-negative (ReRAM in conductance regime).
- No detailed PDK or fabrication data; parameters (noise std, resistance) are configurable in `configs/non_ideal.yaml`.
