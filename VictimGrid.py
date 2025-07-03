import numpy as np

class VictimGrid:
    def __init__(self, size):
        """
        Initializes a grid representing the belief over victim locations.

        :param size: Number of regions or states in the workspace.
        """
        self.grid = np.zeros(size, dtype=np.float64)

    def add_victim(self, index, intensity=1.0):
        """
        Simulates the detection of a victim at a specific location.

        :param index: Index of the grid (region) where a victim is added.
        :param intensity: Strength of belief update.
        """
        self.grid[index] += intensity

    def get_region(self, region_indices):
        """
        Returns the belief over the specified region indices without normalization.
        """
        return self.grid[region_indices].copy()

    def get_global_grid(self):
        """
        Returns the full victim grid (unnormalized).
        """
        return self.grid.copy()

    def __str__(self):
        return str(np.round(self.grid, 3))
