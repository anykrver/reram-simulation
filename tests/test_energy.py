"""Tests: energy formula and estimator consistency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.hardware.energy_estimator import EnergyEstimator


def test_energy_crossbar_positive():
    est = EnergyEstimator(voltage=1.0, timestep_s=1e-6)
    V = np.ones(4, dtype=np.float64) * 0.5
    G = np.ones((4, 4), dtype=np.float64) * 0.1
    E = est.energy_crossbar(V, G)
    assert E >= 0
    assert np.isfinite(E)


def test_energy_formula():
    # E = sum(V^2 * G) * t; for V uniform 1, G uniform 0.1, 4x4: E = 16 * 1 * 0.1 * t
    est = EnergyEstimator(voltage=1.0, timestep_s=1.0)
    V = np.ones(4, dtype=np.float64)
    G = np.ones((4, 4), dtype=np.float64) * 0.1
    E = est.energy_crossbar(V, G)
    expected = 16 * 1.0 * 0.1 * 1.0
    np.testing.assert_allclose(E, expected, rtol=1e-5)


def test_latency_cycles():
    est = EnergyEstimator()
    c = est.latency_cycles(rows=8, cols=8, cycles_per_op=1)
    assert c == 64
