import numpy as np

class OccupancyVector:
    def __init__(self, l_bar=0.95):
        """
        :param l_bar: confidence threshold (typically 0.95)
        """
        self.l_bar = l_bar

    def compute(self, fused_belief, nominal_pmf):
        """
        :param fused_belief: np.array of fused PMF
        :param nominal_pmf: np.array of nominal/reference PMF
        :return: np.array of binary occupancy vector θ_cher
        """
        fused_belief = np.asarray(fused_belief)
        nominal_pmf = np.asarray(nominal_pmf)

        occupancy = np.where(fused_belief > nominal_pmf, self.l_bar, 1 - self.l_bar)
        return occupancy
    
    def compute(self, fused_belief, nominal_pmf):
        """
        Soft occupancy vector using logistic function on log-likelihood ratio.
        :param fused_belief: np.array of fused PMF
        :param nominal_pmf: np.array of nominal/reference PMF
        :return: np.array of soft occupancy values in [0, 1]
        """
        fused_belief = np.asarray(fused_belief)
        nominal_pmf = np.asarray(nominal_pmf)

        # Avoid log(0) by adding a small constant
        log_likelihood_ratio = np.log(fused_belief + 1e-10) - np.log(nominal_pmf + 1e-10)

        # Threshold as log-odds
        threshold = np.log(self.l_bar / (1 - self.l_bar))

        # Soft occupancy: use sigmoid on (LLR - threshold)
        occupancy = 1 / (1 + np.exp(-(log_likelihood_ratio - threshold)))

        return occupancy

