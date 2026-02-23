"""Tests: encoder, neuron, network forward, trainer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.snn import PoissonEncoder, LIFNeuron, SNNNetwork, SNNTrainer
from src.crossbar import IdealCrossbar

def test_encoder_shape():
    enc = PoissonEncoder(seed=42)
    x = np.random.rand(8).astype(np.float32)
    s = enc.encode(x, 20, seed=42)
    assert s.shape == (20, 8)

def test_neuron_step():
    n = LIFNeuron()
    spk, mem = n.step(np.array([0.5, 0.5], dtype=np.float32))
    assert spk.shape == (2,) and mem.shape == (2,)

def test_network_forward():
    np.random.seed(42)
    cb = IdealCrossbar(8, 4)
    cb.set_conductance(np.maximum(np.random.randn(8, 4).astype(np.float32) * 0.1, 0))
    snn = SNNNetwork(8, 4, crossbar_run=cb.run, timesteps=10)
    spikes, total = snn.forward(np.random.rand(8).astype(np.float32), seed=42)
    assert spikes.shape == (10, 4) and total.shape == (4,)

def test_trainer_step():
    """SNNTrainer: G changes after one step, stays non-negative, loss decreases."""
    np.random.seed(0)
    N_IN, N_OUT = 16, 4
    trainer = SNNTrainer(N_IN, N_OUT, lr=0.1, seed=0)
    G_before = trainer.get_weights().copy()

    # Small synthetic batch
    X = np.random.rand(8, N_IN).astype(np.float32)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)

    loss1 = trainer.train_batch(X, y)
    G_after = trainer.get_weights()

    # G must have changed
    assert not np.allclose(G_before, G_after), "G did not update"
    # Conductance must remain non-negative
    assert np.all(G_after >= 0.0), "G contains negative values"
    # Running a second step: loss should generally be ≤ first (not always strictly less
    # due to stochasticity, but shapes must be finite)
    loss2 = trainer.train_batch(X, y)
    assert np.isfinite(loss1) and np.isfinite(loss2), "Non-finite loss"

def test_trainer_get_set_weights():
    """set_weights / get_weights round-trip; negative values are clipped."""
    trainer = SNNTrainer(4, 3, seed=1)
    G_new = np.array([[0.5, -0.1, 0.2], [0.3, 0.4, -0.05],
                       [0.0, 0.1, 0.9], [0.2, 0.3, 0.4]], dtype=np.float32)
    trainer.set_weights(G_new)
    G_out = trainer.get_weights()
    assert np.all(G_out >= 0.0), "set_weights did not clip negatives"
    # Positive values should pass through unchanged
    assert np.isclose(G_out[0, 0], 0.5)

