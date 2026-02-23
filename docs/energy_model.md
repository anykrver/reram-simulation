# Energy Model

Energy per crossbar read: \(E = V^2 \times G \times t\) (sum over cells).

## Units

- \(V\): Volts (V)
- \(G\): Siemens (S)
- \(t\): seconds (s)
- \(E\): Joules (J)

Per cell: \(E_{cell} = V^2 \times G \times t\). Total energy is the sum over all cells for the applied voltage pattern and conductance matrix.

## Where it is used in code

- **`src/hardware/energy_estimator.py`**: `EnergyEstimator.energy_crossbar(V, G, t_s)` computes \(E = \sum_{i,j} V_{i,j}^2 G_{i,j} \times t\).
- **`src/main.py`**: Calls the estimator after an ideal crossbar run to log energy.
- **`dashboard/app.py`**: Uses the estimator for the power curve placeholder.
- **`experiments/energy_comparison.ipynb`**: Compares energy across crossbar sizes.
