import numpy as np

class FusionRule:
    def __init__(self, reference=None):
        self.reference = reference  # Optional reference PMF for comparisons

    def compute_metropolis_weight(self, id_a, id_b):
        """
        Dummy weight calculation.
        If needed, replace with actual Metropolis-Hastings rule or degrees.
        """
        return 0.5 if id_a != id_b else 1.0
    
    def chernoff_fusion(self, belief_a, belief_b, omega=None):
        """
        Perform Chernoff fusion between two belief vectors (element-wise).
        :param belief_a: np.array of robot a's belief (probabilities per cell)
        :param belief_b: np.array of robot b's belief (probabilities per cell)
        :param omega: weight for robot a; if None, default to 0.5
        :return: np.array of fused belief (probabilities per cell)
        """
        belief_a = np.asarray(belief_a)
        belief_b = np.asarray(belief_b)

        if omega is None:
            omega = 0.5

        # Element-wise fusion in log domain
        log_fused = omega * np.log(belief_a + 1e-10) + (1 - omega) * np.log(belief_b + 1e-10)
        fused = np.exp(log_fused)
        
        # No normalization across all cells, as these are independent probabilities
        # Clamp fused probabilities to [0,1] just in case of numerical errors
        fused = np.clip(fused, 0.0, 1.0)

        return fused
    
    def chernoff_fusion_n(self, belief_list, omega_list):
        """
        Generalized Chernoff fusion for N robots using weighted log averaging.
        :param belief_list: List of belief vectors from N robots
        :param omega_list: Corresponding weights for each robot (sum should be 1)
        :return: fused belief vector
        """
        belief_list = [np.asarray(b) for b in belief_list]
        omega_list = np.asarray(omega_list)

        # Weighted sum of logs
        log_beliefs = np.array([w * np.log(b + 1e-10) for b, w in zip(belief_list, omega_list)])
        log_fused = np.sum(log_beliefs, axis=0)
        fused = np.exp(log_fused)

        return np.clip(fused, 0.0, 1.0)

    def compute_omega_weights(self, robots, current_time, 
                              alpha_time=0.25, alpha_conf=0.25, alpha_degrade=0.25, alpha_sensor=0.25):
        """
        Compute omega for each robot using:
        - Time since last update (smaller gap = higher score)
        - Confidence = 1 - occupancy
        - Belief degradation (1 - Hellinger distance)
        - Sensor quality (static attribute per robot)
        """
        time_scores = []
        conf_scores = []
        degrade_scores = []
        sensor_scores = []

        for r in robots:
            # Time score (more recent = higher)
            time_score = 1.0 / (1 + max(1, current_time - getattr(r, "last_update_time", 0)))
            time_scores.append(time_score)

            # Confidence score: 1 - avg occupancy
            conf_score = 1.0 - np.mean(getattr(r, "current_occupancy", np.zeros_like(r.belief.get())))
            conf_scores.append(conf_score)

            # Degradation: 1 - Hellinger distance
            h_dist = self.hellinger.compute(r.belief.get(), r.nominal_belief)
            degrade_scores.append(1.0 - h_dist)

            # Sensor quality (default = 0.7 if not set)
            sensor_score = getattr(r, "sensor_quality", 0.7)
            sensor_scores.append(sensor_score)

        # Normalize individual components
        def normalize(vec):
            arr = np.array(vec)
            return arr / (np.sum(arr) + 1e-10)

        time_scores = normalize(time_scores)
        conf_scores = normalize(conf_scores)
        degrade_scores = normalize(degrade_scores)
        sensor_scores = normalize(sensor_scores)

        # Weighted sum of all factors
        omega = (
            alpha_time * time_scores +
            alpha_conf * conf_scores +
            alpha_degrade * degrade_scores +
            alpha_sensor * sensor_scores
        )

        return normalize(omega).tolist()


    