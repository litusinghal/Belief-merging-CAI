from robot import Robot
from VictimGrid import VictimGrid
from centralizer import MainCentralizer
from fusion import FusionRule
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors

# --- Simulation Parameters ---
T = 5
GRID_SIZE = 100
NUM_REGIONS = 5
NUM_ROBOTS = 10
l_bar = 0.95
OBSERVATION_RANGE = 2

# --- Initial Beliefs and Nominal PMF ---
np.random.seed(42)
initial_beliefs = [np.full(GRID_SIZE, 0.5) for _ in range(NUM_ROBOTS)]
nominal_pmf = np.full(GRID_SIZE, 0.5)

# --- Victim Grid ---
victim_grid = VictimGrid(size=GRID_SIZE)
victim_grid.grid[:] = 0

# --- Victim Schedule (Many Victims) ---
victim_schedule = {
    1: [(i, np.random.uniform(0.2, 0.9)) for i in np.random.choice(GRID_SIZE, 10, replace=False)],
    2: [(i, np.random.uniform(0.2, 0.9)) for i in np.random.choice(GRID_SIZE, 10, replace=False)],
    3: [(i, np.random.uniform(0.2, 0.9)) for i in np.random.choice(GRID_SIZE, 10, replace=False)],
    4: [(i, np.random.uniform(0.2, 0.9)) for i in np.random.choice(GRID_SIZE, 10, replace=False)],
    5: [(i, np.random.uniform(0.2, 0.9)) for i in np.random.choice(GRID_SIZE, 10, replace=False)],
}

# --- Color Map for Belief Ranges ---
thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
colors = ['blue', 'green', 'yellow', 'orange', 'red']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(thresholds, cmap.N)

# To store histogram of belief bins at each time step
belief_bins_history = []

# --- Instantiate Robots ---
robots = [
    Robot(
        robot_id=chr(ord('A') + i),
        initial_belief=initial_beliefs[i],
        nominal_belief=nominal_pmf,
        l_bar=l_bar,
        victim_grid=victim_grid,
        region_indices=slice(0, 1),
        observation_range=OBSERVATION_RANGE
    )
    for i in range(NUM_ROBOTS)
]

# --- Centralizer Setup ---
fusion_rule = FusionRule()
centralizer = MainCentralizer(
    grid_size=GRID_SIZE,
    num_regions=NUM_REGIONS,
    num_robots=NUM_ROBOTS,
    fusion=fusion_rule
)
centralizer.assign_robots_to_regions(robots, victim_grid)
assignments = centralizer.region_assignments
leaders = centralizer.region_centralizers
regions = centralizer.divide_grid()

for region_idx, assigned_robots in assignments.items():
    for robot in assigned_robots:
        robot.region_indices = regions[region_idx]

# --- Visualize Assignments ---
G = nx.Graph()
for region_idx, robots_in_region in assignments.items():
    leader = leaders[region_idx]
    for r in robots_in_region:
        G.add_edge(f"Region {region_idx}", r.id, color='red' if r.id == leader.id else 'blue')

colors = [G[u][v]['color'] for u, v in G.edges()]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, edge_color=colors, node_size=700)
plt.title("Robot Assignments to Regions with Leaders")
plt.show()

# === Simulation Loop ===
print("\n=== Simulation Loop ===")
for t in range(1, T + 1):
    print(f"\n--- Time Step {t} ---")

    if t in victim_schedule:
        print(f"[{len(victim_schedule[t])} Victims added at t = {t}]")
        for idx, intensity in victim_schedule[t]:
            victim_grid.add_victim(index=idx, intensity=intensity)
            print(f"  → Victim at index {idx} with intensity {round(intensity, 2)}")

    # Intra-robot update
    for robot in robots:
        robot.observe_and_bayes_update()

    # Regional fusion
    region_fused = {}
    for region_idx, leader in leaders.items():
        fused = leader.fuse_region_beliefs(assignments[region_idx])
        region_fused[region_idx] = fused

    # Global fusion
    global_belief = centralizer.global_fuse(region_fused)
    centralizer.global_belief_history.append(global_belief)

    # Print full grid
    belief_reshaped = global_belief.reshape(10, 10)
    print("\nGlobal Belief Grid (10x10):")
    print(np.round(belief_reshaped, 3))

    # Belief distribution histogram
    bin_counts = np.histogram(global_belief, bins=thresholds)[0]
    belief_bins_history.append(bin_counts)

    # Heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(belief_reshaped, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7, 0.9], label="Belief")
    plt.title(f"Global Belief Map at Time {t}")
    plt.tight_layout()
    plt.show()

print("\n=== Simulation Complete ===")

# === Plot Stacked Histogram Over Time ===
belief_bins_history = np.array(belief_bins_history)
time_steps = np.arange(1, len(belief_bins_history) + 1)

plt.figure(figsize=(10, 6))
bottom = np.zeros(len(time_steps))

for i, color in enumerate(cmap.colors):
    plt.bar(time_steps, belief_bins_history[:, i], bottom=bottom, color=color,
            label=f"{thresholds[i]}–{thresholds[i+1]}")
    bottom += belief_bins_history[:, i]

plt.xlabel("Time Step")
plt.ylabel("Number of Grid Cells")
plt.title("Distribution of Global Belief Values Across Grid Cells")
plt.legend(title="Belief Range")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
