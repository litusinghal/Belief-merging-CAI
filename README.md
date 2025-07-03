# 🤖 Belief Merging in Semi-Decentralized Multi-Robot Systems using Chernoff Fusion

This project implements a **semi-decentralized belief fusion system** for multi-robot victim detection using a **generalized Chernoff fusion** technique. The system is designed for use in **search-and-rescue** or **disaster response** environments where robots must collaboratively detect victims under uncertainty, while balancing autonomy and hierarchical coordination.

---

## 🧭 Problem Overview

In high-stakes environments such as collapsed buildings or disaster zones, **multi-robot systems** must operate with **partial observability**, **noisy sensors**, and **limited communication**. Centralized control may be infeasible, but full decentralization often lacks global coordination. This project adopts a **hybrid (semi-decentralized)** architecture to address both scalability and consistency.

The core challenges addressed include:

1. **Accurate local belief updates** from noisy observations
2. **Efficient regional fusion** of robot beliefs
3. **Global-level coordination** and robot reassignment based on a fused belief map

---

## 🏗️ System Architecture

The simulation is based on a **10×10 discretized grid world** and involves the following hierarchy:

### 🔹 Robots
- Collect observations from their assigned subregions
- Perform **Bayesian belief updates** using a probabilistic sensor model
- Do **not communicate with each other** directly
- Send updated beliefs to their **regional centralizer**

### 🔹 Regional Centralizers
- Each region has its own central unit managing multiple robots
- Collects updated beliefs from robots in its region
- Merges them using **generalized Chernoff fusion**
- Forwards the fused regional belief to the **main centralizer**

### 🔹 Main Centralizer
- Collects regional belief maps
- Performs **global belief fusion** using Chernoff fusion
- Assigns or re-allocates robots to subregions based on **belief strength (probability of finding victims)**

### 🔁 Continuous Operation Loop
1. Robots observe and update beliefs
2. Regional centralizers merge local beliefs
3. Main centralizer builds global belief
4. Robots are reassigned to new regions
5. Repeat

---



## 🔬 Key Techniques

### 🧠 Generalized Chernoff Fusion
- Beliefs are fused in the **log domain** using **Chernoff information**
- Our version uses **dynamic fusion weights (ω)** based on:
  - Sensor reliability
  - Confidence from Hellinger distance
  - Belief degradation due to staleness
  - Time of observation

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `main.py` | Orchestrates the simulation loop |
| `robot.py` | Robot logic: movement, observation, belief update |
| `centralizer.py` | Main centralizer managing global belief and assignments |
| `fusion.py` | Implements Chernoff fusion logic |
| `belief.py` | Handles belief vectors and Bayesian updates |
| `occupancy.py` | Defines grid structure and victim placements |
| `HellingerDistance.py` | Computes inter-distribution similarity |
| `VictimGrid.py` | Generates and updates the victim ground truth |
| `__pycache__/` | Ignored (compiled files) |

---

## 🧪 Simulation Details

- **10 robots**, divided across **5 regions**
- Each region has a **regional centralizer**
- Victims are added **dynamically over time**
- Beliefs evolve with each iteration
- **Visualization** (e.g., heatmaps) tracks belief convergence and victim detection

---

## 🧑‍🔬 Evaluation Metrics

- **Accuracy** of victim detection
- **Time to convergence** across belief grids
- **Efficiency** of robot allocation and reallocation
- **Scalability** under communication and observation constraints

---

## 🔧 How to Run

### 1. Install dependencies
```bash
pip install numpy matplotlib
