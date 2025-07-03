import numpy as np

class Belief:
    def __init__(self, belief_vector):
        """
        Initialize a belief vector (scalar probabilities per cell).

        Args:
            belief_vector (array-like): List or numpy array of probabilities (0 to 1).
        """
        self.vector = np.array(belief_vector, dtype=np.float64)
        # No normalization needed here

    def log(self):
        """
        Returns the log of the belief vector, with clipping to avoid log(0).
        """
        return np.log(np.clip(self.vector, 1e-10, 1.0))

    def exp(self):
        """
        Returns exponentiated belief vector (useful after log-sum ops).
        """
        return np.exp(self.vector)

    def update(self, new_vector):
        """
        Update the belief vector without normalization.
        """
        self.vector = np.array(new_vector, dtype=np.float64)

    def get(self):
        """
        Returns the belief vector.
        """
        return self.vector

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, idx):
        return self.vector[idx]

    def __str__(self):
        return str(self.vector)

    def as_numpy(self):
        return self.vector.copy()
