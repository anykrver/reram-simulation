"""
Test SNN on MNIST using kagglehub dataset (hojjatk/mnist-dataset).
Download dataset, load samples, run through crossbar-backed SNN, report accuracy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import kagglehub
from src.crossbar import IdealCrossbar
from src.snn import PoissonEncoder, SNNNetwork
from src.utils.mnist_loader import get_mnist_path, load_mnist_from_path
from src.utils.metrics import compute_accuracy
from src.utils.weight_io import load_weights


def main():
    SEED = 42
    np.random.seed(SEED)
    N_CLASSES = 10
    TIMESTEPS = 50
    MAX_TEST = 100

    # Paths
    WEIGHTS_PATH = Path(__file__).resolve().parent / "trained_weights.npy"

    print("Downloading MNIST via kagglehub...")
    path = get_mnist_path()
    print("Path to dataset files:", path)

    print("Loading MNIST samples...")
    X_train, y_train, X_test, y_test, n_pixels = load_mnist_from_path(
        path, max_test=MAX_TEST
    )
    print(f"Test size: {X_test.shape}, pixels: {n_pixels}")

    # SNN Setup
    n_in, n_out = n_pixels, N_CLASSES
    
    if WEIGHTS_PATH.exists():
        print(f"Loading trained weights from: {WEIGHTS_PATH.name}")
        W = load_weights(str(WEIGHTS_PATH))
        weight_type = "trained"
    else:
        print("Trained weights not found. Using random...")
        W = np.maximum(np.random.randn(n_in, n_out).astype(np.float32) * 0.01, 0.0)
        weight_type = "random"

    crossbar = IdealCrossbar(n_in, n_out)
    crossbar.set_conductance(W)

    encoder = PoissonEncoder(max_rate=100.0, seed=SEED)
    snn = SNNNetwork(
        n_in, n_out,
        crossbar_run=crossbar.run,
        encoder=encoder,
        timesteps=TIMESTEPS,
    )

    print(f"Running SNN inference on {len(X_test)} samples ({weight_type} weights)...")
    logits = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)
    for i in range(len(X_test)):
        _, total = snn.forward(X_test[i], seed=SEED + i)
        logits[i] = total

    acc = compute_accuracy(logits, y_test, top_k=1)
    print("-" * 40)
    print(f"Test accuracy ({weight_type}): {acc:.4f}")
    print("-" * 40)
    print("Done.")


if __name__ == "__main__":
    main()
