"""Abstract accelerator: tiles, crossbars, memory."""


class AcceleratorModel:
    def __init__(self, num_tiles: int = 1, rows_per_crossbar: int = 64, cols_per_crossbar: int = 64,
                 memory_energy_per_word_pJ: float = 1.0):
        self.num_tiles = num_tiles
        self.rows_per_crossbar = rows_per_crossbar
        self.cols_per_crossbar = cols_per_crossbar
        self.memory_energy_per_word_pJ = memory_energy_per_word_pJ

    def total_crossbar_elements(self) -> int:
        return self.num_tiles * self.rows_per_crossbar * self.cols_per_crossbar

    def memory_energy(self, words_read: int = 0, words_written: int = 0) -> float:
        return (words_read + words_written) * self.memory_energy_per_word_pJ
