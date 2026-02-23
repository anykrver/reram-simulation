"""Tests: ideal I = V*G; non-ideal sanity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.crossbar import IdealCrossbar, NonIdealCrossbar

def test_ideal_shape():
    cb = IdealCrossbar(8, 8)
    G = np.random.rand(8, 8).astype(np.float32) * 0.1
    cb.set_conductance(G)
    V = np.random.rand(8).astype(np.float32) * 0.5
    I = cb.run(V)
    assert I.shape == (8,)

def test_ideal_math():
    cb = IdealCrossbar(4, 4)
    cb.set_conductance(np.eye(4, dtype=np.float32))
    V = np.array([1.0, 0, 0, 0], dtype=np.float32)
    I = cb.run(V)
    np.testing.assert_allclose(I, V, rtol=1e-5)

def test_non_ideal_shape():
    np.random.seed(42)
    cb = NonIdealCrossbar(8, 8, noise_std=0.01, seed=42)
    G = np.random.rand(8, 8).astype(np.float32) * 0.1
    cb.set_conductance(G)
    I = cb.run(np.random.rand(8).astype(np.float32) * 0.5)
    assert I.shape == (8,) and np.all(np.isfinite(I))
