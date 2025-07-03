import numpy as np

class HellingerDistance:
    def __init__(self):
        pass

    def compute(self, p, q):
        """
        Compute the Hellinger distance between two PMFs p and q
        :param p: np.array (PMF 1)
        :param q: np.array (PMF 2)
        :return: float distance ∈ [0, 1]
        """
        p = np.asarray(p)
        q = np.asarray(q)

        bc = np.sum(np.sqrt(p * q))  # Bhattacharyya coefficient
        return np.sqrt(1 - bc)
