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
    Trains multi-layer crossbar conductances G via rate-coded surrogate gradients and backprop.

    The forward pass computes expected spike rates analytically. For hidden layers,
    we use a ReLU activation as a surrogate for the LIF rate-response function.
    The final layer output (logits) is used for cross-entropy loss.

    Parameters
    ----------
    layer_sizes: List of layer dimensions, e.g., [784, 256, 128, 10].
    lr:          Learning rate for SGD.
    max_rate:    PoissonEncoder max firing rate (Hz).
    timestep_ms: Encoder timestep (ms).
    timesteps:   Number of SNN timesteps (used to normalise logits).
    G_max:       Upper bound for conductance clipping (S).
    seed:        Optional RNG seed for weight initialisation.
    """

    def __init__(
        self,
        layer_sizes: list[int] | int = 784,
        n_out: Optional[int] = None,  # For backward compatibility
        lr: float = 0.01,
        max_rate: float = 100.0,
        timestep_ms: float = 1.0,
        timesteps: int = 50,
        G_max: float = 1.0,
        seed: Optional[int] = None,
    ):
        # Handle backward compatibility: if n_in and n_out are passed as ints
        if isinstance(layer_sizes, int) and n_out is not None:
            self.layer_sizes = [layer_sizes, n_out]
        elif isinstance(layer_sizes, list):
            self.layer_sizes = layer_sizes
        else:
            raise ValueError("layer_sizes must be a list of ints or (n_in, n_out) must be provided")

        self.lr = lr
        self.max_rate = max_rate
        self.timestep_ms = timestep_ms
        self.timesteps = timesteps
        self.G_max = G_max
        self._dt = max_rate * timestep_ms / 1000.0

        rng = np.random.default_rng(seed)
        self.weights = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            # Kaiming-ish initialisation for ReLU layers
            std = np.sqrt(2.0 / n_in)
            W = rng.normal(0.01, std * 0.1, (n_in, n_out)).astype(np.float32)
            self.weights.append(np.clip(W, 0.0, G_max))

    def get_weights(self) -> list[np.ndarray] | np.ndarray:
        """Return conductance matrices. Returns a single array if only 1 layer exists."""
        if len(self.weights) == 1:
            return self.weights[0].copy()
        return [W.copy() for W in self.weights]

    def set_weights(self, weights: list[np.ndarray] | np.ndarray) -> None:
        """Set weights; projected to [0, G_max]."""
        if isinstance(weights, np.ndarray):
            weights = [weights]
        
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} layers, got {len(weights)}")
        
        for i, W in enumerate(weights):
            if W.shape != self.weights[i].shape:
                raise ValueError(f"Layer {i} shape mismatch: {W.shape} vs {self.weights[i].shape}")
            self.weights[i] = np.clip(W, 0.0, self.G_max).astype(np.float32)

    def mean_rate_in(self, X: np.ndarray) -> np.ndarray:
        """Compute analytic mean input spike rate for the first layer."""
        X = np.asarray(X, dtype=np.float32)
        return np.clip(X, 0.0, 1.0) * self._dt * self.timesteps

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Multi-layer forward pass with ReLU activations (surrogate).
        Returns final logits and activations for all layers.
        """
        activations = [self.mean_rate_in(X)]
        curr = activations[0]
        
        for i, W in enumerate(self.weights):
            curr = curr @ W
            if i < len(self.weights) - 1:
                # ReLU surrogate for hidden layers
                curr = np.maximum(curr, 0.0)
            activations.append(curr)
            
        return activations[-1], activations

    def loss_and_grad(
        self,
        logits: np.ndarray,
        activations: list[np.ndarray],
        y: np.ndarray,
    ) -> tuple[float, list[np.ndarray]]:
        """Multi-layer cross-entropy loss and backpropagation."""
        N = logits.shape[0]
        probs = _softmax(logits)
        oh = _one_hot(y, self.layer_sizes[-1])
        
        # Cross-entropy loss
        eps = 1e-9
        loss = -float(np.mean(np.sum(oh * np.log(probs + eps), axis=1)))
        
        # Backprop
        grads = []
        delta = (probs - oh) / N  # Error at output
        
        for i in range(len(self.weights) - 1, -1, -1):
            W = self.weights[i]
            A_prev = activations[i]
            
            # dL/dW = A_prev^T @ delta
            dW = A_prev.T @ delta
            grads.insert(0, dW.astype(np.float32))
            
            if i > 0:
                # delta_prev = (delta @ W^T) * derivative_of_activation(A_prev)
                # derivative of ReLU is 1 if A > 0 else 0
                delta = (delta @ W.T) * (A_prev > 0)
                
        return loss, grads

    def step(self, dWs: list[np.ndarray]) -> None:
        """SGD update + non-negativity + G_max projection."""
        for i, dW in enumerate(dWs):
            self.weights[i] -= self.lr * dW
            np.clip(self.weights[i], 0.0, self.G_max, out=self.weights[i])

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Full forward + backward + update cycle."""
        logits, activations = self.forward(X)
        loss, grads = self.loss_and_grad(logits, activations, y)
        self.step(grads)
        return loss
