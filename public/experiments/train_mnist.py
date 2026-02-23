"""
Train crossbar SNN weights on MNIST using rate-coded surrogate gradients.

Usage
-----
    python experiments/train_mnist.py [options]

Options
-------
    --epochs        INT    Training epochs      (default: 10)
    --batch-size    INT    Mini-batch size      (default: 64)
    --lr            FLOAT  Learning rate        (default: 0.01)
    --max-train     INT    Max training samples (default: 5000)
    --max-test      INT    Max test samples     (default: 1000)
    --timesteps     INT    SNN timesteps        (default: 50)
    --out           PATH   Weight output path   (default: experiments/trained_weights.npy)
    --seed          INT    Random seed          (default: 42)

On completion, trained weights are saved to --out and can be loaded by
the Streamlit dashboard for inference with trained (non-random) weights.
"""

import argparse
import sys
from pathlib import Path

# Make project root importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.snn import SNNTrainer
from src.utils.mnist_loader import get_mnist_path, load_mnist_from_path
from src.utils.weight_io import save_weights
from src.utils.metrics import compute_accuracy


def evaluate(trainer: SNNTrainer, X: np.ndarray, y: np.ndarray) -> float:
    """Compute accuracy for the full (or large) set using vectorised forward."""
    logits, _ = trainer.forward(X)
    return compute_accuracy(logits, y, top_k=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train SNN on MNIST (rate-coded surrogate gradient)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--max-train",  type=int,   default=5000)
    parser.add_argument("--max-test",   type=int,   default=1000)
    parser.add_argument("--timesteps",  type=int,   default=50)
    parser.add_argument("--out",        type=str,
                        default=str(_ROOT / "experiments" / "trained_weights.npy"))
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # -- Download / load MNIST ------------------------------------------
    print("Downloading MNIST via kagglehub ...")
    path = get_mnist_path()
    print(f"Dataset path: {path}")

    print(f"Loading MNIST (train<={args.max_train}, test<={args.max_test}) ...")
    X_train, y_train, X_test, y_test, n_pixels = load_mnist_from_path(
        path, max_train=args.max_train, max_test=args.max_test
    )
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}   Pixels: {n_pixels}")

    N_CLASSES = int(y_train.max()) + 1   # 10 for MNIST

    # -- Create trainer ------------------------------------------------
    trainer = SNNTrainer(
        n_in=n_pixels,
        n_out=N_CLASSES,
        lr=args.lr,
        max_rate=100.0,
        timestep_ms=1.0,
        timesteps=args.timesteps,
        G_max=1.0,
        seed=args.seed,
    )

    # Baseline accuracy with random weights
    base_acc = evaluate(trainer, X_test, y_test)
    print(f"\nBaseline test accuracy (random weights): {base_acc:.4f}")

    # -- Training loop -------------------------------------------------
    n_train = X_train.shape[0]
    rng = np.random.default_rng(args.seed)

    print(f"\nTraining for {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Shuffle
        idx = rng.permutation(n_train)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            Xb = X_shuf[start : start + args.batch_size]
            yb = y_shuf[start : start + args.batch_size]
            loss = trainer.train_batch(Xb, yb)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = evaluate(trainer, X_train, y_train)
        test_acc  = evaluate(trainer, X_test,  y_test)
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"test_acc={test_acc:.4f}"
        )

    # -- Save weights --------------------------------------------------
    G_final = trainer.get_weights()
    save_weights(G_final, args.out)
    print(f"\nWeights saved -> {args.out}  shape={G_final.shape}")
    print(f"Final test accuracy: {evaluate(trainer, X_test, y_test):.4f}")
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
