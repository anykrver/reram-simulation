"""SNN network: layers, timesteps, uses crossbar for weight multiply."""

from typing import Optional, Callable

import numpy as np

from .spike_encoder import PoissonEncoder
from .neuron import LIFNeuron


class SNNNetwork:
    """Single-layer SNN that uses a crossbar for dense weight multiplication."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        crossbar_run: Callable[[np.ndarray], np.ndarray],
        encoder: Optional[PoissonEncoder] = None,
        neuron: Optional[LIFNeuron] = None,
        timesteps: int = 100,
    ):
        """
        Args:
            input_size: Number of input units.
            output_size: Number of output (LIF) neurons.
            crossbar_run: Function V -> I (voltages to currents); expects V (timesteps, input_size) or (input_size,).
            encoder: Spike encoder; default PoissonEncoder().
            neuron: LIF layer; default LIFNeuron().
            timesteps: Simulation timesteps.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.crossbar_run = crossbar_run
        self.encoder = encoder or PoissonEncoder()
        self.neuron = neuron or LIFNeuron()
        self.timesteps = timesteps

    def forward(
        self,
        x: np.ndarray,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run SNN: encode x -> spikes, crossbar(V) -> I, LIF -> output spikes.

        Args:
            x: Input (input_size,) normalized to [0,1].
            seed: Encoder RNG seed.

        Returns:
            output_spikes: (timesteps, output_size).
            total_spikes: (output_size,) per-neuron spike counts.
        """
        x = np.asarray(x).ravel()
        if x.size != self.input_size:
            raise ValueError(f"x size {x.size} != input_size {self.input_size}")
        spikes_in = self.encoder.encode(x, self.timesteps, seed=seed)
        output_spikes = np.zeros((self.timesteps, self.output_size), dtype=np.float32)
        membrane = None
        for t in range(self.timesteps):
            V = spikes_in[t]
            I = self.crossbar_run(V)
            I = np.asarray(I).ravel()
            if I.size != self.output_size:
                I = np.resize(I, self.output_size)
            spk, membrane = self.neuron.step(I, membrane)
            output_spikes[t] = spk
        total_spikes = output_spikes.sum(axis=0)
        return output_spikes, total_spikes
