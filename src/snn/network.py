"""SNN network: layers, timesteps, uses crossbar for weight multiply."""

from typing import Optional, Callable

import numpy as np

from .spike_encoder import PoissonEncoder
from .neuron import LIFNeuron


class SNNNetwork:
    """Multi-layer SNN that uses crossbars for dense weight multiplication."""

    def __init__(
        self,
        layer_sizes: list[int],
        crossbar_runs: list[Callable[[np.ndarray], np.ndarray]],
        encoder: Optional[PoissonEncoder] = None,
        neuron_factory: Optional[Callable[[], LIFNeuron]] = None,
        timesteps: int = 100,
    ):
        """
        Args:
            layer_sizes: List of layer dimensions, e.g. [784, 256, 128, 10].
            crossbar_runs: List of functions V -> I for each weight matrix.
            encoder: Spike encoder; default PoissonEncoder().
            neuron_factory: Function that returns a new LIFNeuron; default LIFNeuron().
            timesteps: Simulation timesteps.
        """
        if len(crossbar_runs) != len(layer_sizes) - 1:
            raise ValueError(f"Expected {len(layer_sizes)-1} crossbar_runs, got {len(crossbar_runs)}")

        self.layer_sizes = layer_sizes
        self.crossbar_runs = crossbar_runs
        self.encoder = encoder or PoissonEncoder()
        
        # Create a layer of neurons for each output stage
        nf = neuron_factory or (lambda: LIFNeuron())
        self.neurons = [nf() for _ in range(len(layer_sizes) - 1)]
        self.timesteps = timesteps

    def forward(
        self,
        x: np.ndarray,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run SNN: encode x -> spikes, pass through multiple stages of crossbars and LIF neurons.

        Args:
            x: Input (layer_sizes[0],) normalized to [0,1].
            seed: Encoder RNG seed.

        Returns:
            output_spikes: (timesteps, layer_sizes[-1]).
            total_spikes: (layer_sizes[-1],) per-neuron spike counts.
        """
        x = np.asarray(x).ravel()
        if x.size != self.layer_sizes[0]:
            raise ValueError(f"x size {x.size} != input_size {self.layer_sizes[0]}")
        
        spikes_in = self.encoder.encode(x, self.timesteps, seed=seed)
        output_spikes = np.zeros((self.timesteps, self.layer_sizes[-1]), dtype=np.float32)
        
        # States for each neuron layer
        membranes = [None] * len(self.neurons)
        
        for t in range(self.timesteps):
            curr_spikes = spikes_in[t]
            
            for i, (run_cb, neuron) in enumerate(zip(self.crossbar_runs, self.neurons)):
                V = curr_spikes
                I = run_cb(V)
                I = np.asarray(I).ravel()
                
                # Resize if crossbar output shape is different (e.g. non-ideal effects)
                target_size = self.layer_sizes[i + 1]
                if I.size != target_size:
                    I = np.resize(I, target_size)
                
                spk, membrane = neuron.step(I, membranes[i])
                membranes[i] = membrane
                curr_spikes = spk
            
            output_spikes[t] = curr_spikes
            
        total_spikes = output_spikes.sum(axis=0)
        return output_spikes, total_spikes
