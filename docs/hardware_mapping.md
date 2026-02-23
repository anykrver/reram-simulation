# Hardware Mapping

## Mapping weights to crossbar

- Each **synaptic weight** is mapped to a **conductance** value \(G_{i,j}\) at row \(i\), column \(j\).
- Positive weights: mapped to conductance in a defined range (e.g. after normalization).
- Negative weights: handled by differential pairs or signed encoding (not implemented in the minimal stub).

## Tile / array assumptions

- **`src/hardware/accelerator_model.py`**: Abstract accelerator with `num_tiles`, `rows_per_crossbar`, `cols_per_crossbar`.
- One tile = one crossbar; multiple tiles allow parallel matrix-vector ops.
- Memory movement cost (words read/written) is a placeholder for off-chip traffic.
