"""Rate-coded surrogate-gradient trainer for the crossbar SNN.

Training intuition
------------------
In the rate approximation the expected total output spikes for sample x is:

    total_spikes ≈ mean_rate_in(x) @ G      (linear in G)

where mean_rate_in(x) = x * max_rate * dt is the expected input spike rate.

We minimise cross-entropy loss on softmax(total_spikes / T) with
standard SGD and project G to the non-negative orthant after each step
(conductance cannot be negative).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .spike_encoder import PoissonEncoder


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels (N,) to one-hot (N, n_classes)."""
    oh = np.zeros((y.size, n_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh


class SNNTrainer:
    """
    Trains crossbar conductances G via a rate-coded surrogate gradient.

    The forward pass computes expected spike rates analytically
    (mean_rate_in = x * max_rate * dt) and uses I = mean_rate_in @ G as
    the logit.  The gradient flows exactly through this linear map.

    Parameters
    ----------
    n_in:       Number of input neurons (e.g. 784 for MNIST).
    n_out:      Number of output classes (e.g. 10).
    lr:         Learning rate for SGD.
    max_rate:   PoissonEncoder max firing rate (Hz). Used only to scale
                the analytic mean-rate; must match the encoder used at
                inference time.
    timestep_ms: Encoder timestep (ms).
    timesteps:  Number of SNN timesteps (used to normalise logits).
    G_max:      Upper bound for conductance clipping (S).
    seed:       Optional RNG seed for weight initialisation.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        lr: float = 0.01,
        max_rate: float = 100.0,
        timestep_ms: float = 1.0,
        timesteps: int = 50,
        G_max: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.lr = lr
        self.max_rate = max_rate
        self.timestep_ms = timestep_ms
        self.timesteps = timesteps
        self.G_max = G_max
        self._dt = max_rate * timestep_ms / 1000.0  # firing prob per timestep per unit input

        rng = np.random.default_rng(seed)
        # Small positive initialisation (conductances ≥ 0)
        self.G = rng.uniform(0.0, 0.01, (n_in, n_out)).astype(np.float32)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        """Return current conductance matrix G (n_in, n_out)."""
        return self.G.copy()

    def set_weights(self, G: np.ndarray) -> None:
        """Set conductance matrix; projected to [0, G_max]."""
        G = np.asarray(G, dtype=np.float32)
        if G.shape != (self.n_in, self.n_out):
            raise ValueError(f"Expected G shape ({self.n_in}, {self.n_out}), got {G.shape}")
        self.G = np.clip(G, 0.0, self.G_max)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def mean_rate_in(self, X: np.ndarray) -> np.ndarray:
        """
        Compute analytic mean input spike rate for a batch.

        Parameters
        ----------
        X : (N, n_in) float32 in [0, 1].

        Returns
        -------
        rates : (N, n_in) float32.
        """
        X = np.asarray(X, dtype=np.float32)
        return np.clip(X, 0.0, 1.0) * self._dt * self.timesteps

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rate-coded forward pass.

        Parameters
        ----------
        X : (N, n_in) batch of inputs in [0, 1].

        Returns
        -------
        logits       : (N, n_out) — I = mean_rate_in @ G.
        mean_rates   : (N, n_in) — cached for gradient.
        """
        rates = self.mean_rate_in(X)
        logits = rates @ self.G          # (N, n_out)
        return logits, rates

    # ------------------------------------------------------------------
    # Loss and gradient
    # ------------------------------------------------------------------

    def loss_and_grad(
        self,
        logits: np.ndarray,
        mean_rates: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Cross-entropy loss and gradient dL/dG.

        Parameters
        ----------
        logits      : (N, n_out).
        mean_rates  : (N, n_in).
        y           : (N,) integer labels.

        Returns
        -------
        loss : scalar float.
        dG   : (n_in, n_out) gradient.
        """
        N = logits.shape[0]
        probs = _softmax(logits)                     # (N, n_out)
        oh = _one_hot(y, self.n_out)                 # (N, n_out)
        # Cross-entropy
        eps = 1e-9
        loss = -float(np.mean(np.sum(oh * np.log(probs + eps), axis=1)))
        delta = (probs - oh) / N                     # (N, n_out)
        dG = mean_rates.T @ delta                    # (n_in, n_out)
        return loss, dG.astype(np.float32)

    # ------------------------------------------------------------------
    # Optimiser step
    # ------------------------------------------------------------------

    def step(self, dG: np.ndarray) -> None:
        """
        SGD update + non-negativity + G_max projection.

        Parameters
        ----------
        dG : (n_in, n_out) gradient from loss_and_grad.
        """
        self.G -= self.lr * dG
        np.clip(self.G, 0.0, self.G_max, out=self.G)

    # ------------------------------------------------------------------
    # Convenience: one full training step
    # ------------------------------------------------------------------

    def train_batch(
        self, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Forward + loss + backward + weight update for one mini-batch.

        Parameters
        ----------
        X : (N, n_in) inputs in [0, 1].
        y : (N,) integer labels.

        Returns
        -------
        loss : scalar cross-entropy loss.
        """
        logits, rates = self.forward(X)
        loss, dG = self.loss_and_grad(logits, rates, y)
        self.step(dG)
        return loss
