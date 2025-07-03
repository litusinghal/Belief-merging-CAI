import numpy as np
from collections import defaultdict

class MainCentralizer:
    def __init__(self, grid_size, num_regions, num_robots, fusion):
        self.grid_size = grid_size
        self.num_regions = num_regions
        self.num_robots = num_robots
        self.region_assignments = defaultdict(list)
        self.region_centralizers = {}
        self.fusion = fusion  # FusionRule instance
        self.global_belief = np.full(grid_size, 0)
        self.global_belief_history = []

    def divide_grid(self):
        region_size = self.grid_size // self.num_regions
        return [slice(i * region_size, (i + 1) * region_size) for i in range(self.num_regions)]

    def assign_robots_to_regions(self, robots, victim_grid):
        regions = self.divide_grid()
        region_scores = [np.sum(victim_grid.grid[r]) for r in regions]
        ranked_regions = np.argsort(region_scores)[::-1]

        robots_per_region = len(robots) // self.num_regions
        extra = len(robots) % self.num_regions
        robot_idx = 0

        for count, region_idx in enumerate(ranked_regions):
            num_assigned = robots_per_region + (1 if count < extra else 0)
            assigned_robots = robots[robot_idx:robot_idx + num_assigned]

            for r in assigned_robots:
                r.region_indices = regions[region_idx]
                self.region_assignments[region_idx].append(r)

            if assigned_robots:
                self.region_centralizers[region_idx] = assigned_robots[0]

            robot_idx += num_assigned

    def step(self, victim_grid):
        # Step 1: Each robot observes and updates its belief
        for region_robots in self.region_assignments.values():
            for robot in region_robots:
                robot.observe_and_update()

        # Step 2: Region centralizers collect and merge regional beliefs
        region_beliefs = {}
        for region_idx, centralizer in self.region_centralizers.items():
            region_robots = self.region_assignments[region_idx]
            beliefs = [r.belief.get() for r in region_robots]

            fused = beliefs[0].copy()
            for b in beliefs[1:]:
                fused = self.fusion.chernoff_fusion(fused, b, omega=0.5)

            region_beliefs[region_idx] = fused

        # Step 3: Main centralizer merges regional beliefs
        fused_global = list(region_beliefs.values())[0].copy()
        for rb in list(region_beliefs.values())[1:]:
            fused_global = self.fusion.chernoff_fusion(fused_global, rb, omega=0.5)

        self.global_belief = fused_global

    def get_global_belief(self):
        return self.global_belief
    
    def global_fuse(self, region_beliefs_dict):
        """
        Fuse regional beliefs into a single global belief using Chernoff fusion.
        :param region_beliefs_dict: dict {region_idx: belief_vector}
        :param nominal_pmf: reference PMF (not used in fusion directly but could be)
        :return: fused global belief vector
        """
        region_beliefs = list(region_beliefs_dict.values())
        if not region_beliefs:
            return np.full(self.grid_size, 0.0)

        fused_global = region_beliefs[0].copy()
        for rb in region_beliefs[1:]:
            fused_global = self.fusion.chernoff_fusion(fused_global, rb, omega=0.5)

        self.global_belief = fused_global
        return fused_global

