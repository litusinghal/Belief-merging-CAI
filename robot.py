import numpy as np
from belief import Belief
from fusion import FusionRule
from occupancy import OccupancyVector
from HellingerDistance import HellingerDistance

class Robot:
    def __init__(self, robot_id, region_indices, initial_belief, nominal_belief, victim_grid, l_bar=0.95, observation_range=0):
        self.id = robot_id
        self.region_indices = region_indices
        self.observation_range = observation_range

        self.belief = Belief(initial_belief)
        self.nominal_belief = np.array(nominal_belief)
        self.occupancy = OccupancyVector(l_bar)
        self.hellinger = HellingerDistance()
        self.fusion = FusionRule()
        self.victim_grid = victim_grid

        self.current_occupancy = self.occupancy.compute(self.belief.get(), self.nominal_belief)

    def get_observation_indices(self):
        grid_size = len(self.belief.get())
        if isinstance(self.region_indices, slice):
            start, stop = self.region_indices.start, self.region_indices.stop
        else:
            start, stop = min(self.region_indices), max(self.region_indices) + 1

        obs_start = max(0, start - self.observation_range)
        obs_stop = min(grid_size, stop + self.observation_range)
        return slice(obs_start, obs_stop)

    def communicate_and_fuse(self, other_robot, omega=0.5):
        other_belief = other_robot.belief.get()
        new_fused = self.fusion.chernoff_fusion(self.belief.get(), other_belief, omega)
        self.belief.update(new_fused)
        self.current_occupancy = self.occupancy.compute(new_fused, self.nominal_belief)

    def fuse_region_beliefs(self, region_robots, omega=0.5):
        fused_belief = self.belief.get().copy()
        for robot in region_robots:
            if robot.id != self.id:
                fused_belief = self.fusion.chernoff_fusion(fused_belief, robot.belief.get(), omega)
        return fused_belief

    def observe_and_chernoff_update(self):
        """
        Intra-robot belief update using Chernoff fusion (log-domain method).
        """
        obs_indices = self.get_observation_indices()
        region_obs = self.victim_grid.get_region(obs_indices)

        current_belief = self.belief.get().copy()
        belief_region = current_belief[obs_indices]

        fused_region = self.fusion.chernoff_fusion(belief_region, region_obs, omega=0.5)
        current_belief[obs_indices] = fused_region

        self.belief.update(current_belief)
        self.current_occupancy = self.occupancy.compute(current_belief, self.nominal_belief)

    def observe_and_bayes_update(self):
        """
        Intra-robot belief update using Bayes' Theorem (probabilistic inference).
        """
        obs_indices = self.get_observation_indices()
        observation = self.victim_grid.get_region(obs_indices)
        current_belief = self.belief.get().copy()

        for i in range(obs_indices.start, obs_indices.stop):
            z_i = observation[i - obs_indices.start]
            prior = current_belief[i]

            # Gaussian sensor model
            p_z_given_H = 0.8     # if victim present
            p_z_given_not_H = 0.1 # if no victim

            numerator = p_z_given_H * prior
            denominator = numerator + p_z_given_not_H * (1 - prior)
            posterior = numerator / (denominator + 1e-10)  # epsilon to avoid zero division

            current_belief[i] = posterior

        self.belief.update(current_belief)
        self.current_occupancy = self.occupancy.compute(current_belief, self.nominal_belief)

    @staticmethod
    def gaussian(x, mu, sigma):
        coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
        return coeff * np.exp(exponent)

    def distance_to_reference(self):
        return self.hellinger.compute(self.belief.get(), self.nominal_belief)

    def __repr__(self):
        return f"Robot {self.id} | Belief: {np.round(self.belief.get(), 3)}"
