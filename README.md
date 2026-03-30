# Computational Mathematics

This repository contains numerical simulations and interactive applications for dynamical systems, focusing on opinion consensus and vehicular traffic flows. The models are solved using classical numerical integration methods such as Explicit Euler and Heun's method.

---

## Opinion Dynamics

This section simulates a bounded confidence model (based on the Hegselmann-Krause dynamics) where a population of interacting agents updates their opinions over time. Agents only influence each other if their opinions are closer than a certain confidence radius, $\epsilon$.

### Mathematical Model
The evolution of the opinion $x_i(t)$ of the $i$-th agent is governed by the following system of ordinary differential equations:

$$\frac{dx_i}{dt} = \sum_{j=1}^{N} a_{ij}(X) (x_j - x_i)$$

where the interaction weights $a_{ij}$ depend on the distance between opinions: $a_{ij} > 0$ if $|x_i - x_j| < \epsilon$, and $a_{ij} = 0$ otherwise. This is solved numerically using **Heun's method** to ensure better stability.

### Interpretation of the Results
The simulation tracks the ensemble metrics of the population to identify steady states and consensus:
1.  **First Moment (Mean):** Remains constant throughout the simulation. This reflects the conservation of the system's "mass" (the average global opinion does not change).
2.  **Second Moment:** Stabilizes as the agents group together.
3.  **Variance:** The variance $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ shrinks but **does not necessarily go to zero**. It only reaches exactly $0$ in the case of a *Global Consensus* (all agents merging into a single cluster). If the system polarizes into multiple distinct clusters, the variance settles on a positive constant, representing the spatial distance between these isolated clusters.
4.  **Multiplicity (Clusters):** Calculated by finding the geometric multiplicity of the eigenvalue $\lambda = 1$ of the stochastic adjacency matrix $A$. It physically represents the final number of isolated opinion clusters.

---

## Traffic Flows

This section models traffic dynamics on a ring road using the **Bando Follow-the-Leader (FTL) Model**, which is a microscopic optimal velocity model. 

### Mathematical Model
Each vehicle adjusts its acceleration to match an "optimal velocity" that depends exclusively on the distance (headway) to the vehicle immediately ahead. The dynamics are described by:

$$\frac{dx_i}{dt} = v_i$$

$$\frac{dv_i}{dt} = \beta \frac{\Delta v_i (t)}{(\Delta x_i (i))^2} \alpha \left( V(\Delta x_i) - v_i \right)$$

where $x_i$ and $v_i$ are the position and velocity of vehicle $i$, $a$ is the sensitivity parameter, $\Delta x_i = x_{i+1} - x_i$ is the headway, and $V(\cdot)$ is the non-linear optimal velocity function.

### Simulation Setup & Stop-and-Go Waves
* **Initial Conditions:** The simulation starts from a steady-state equilibrium. Vehicles are equally spaced with headway $\Delta x_i = \frac{L}{N}$ (where $L$ is the track length and $N$ the number of vehicles) and travel at the identical constant velocity $v_i = V\left(\frac{L}{N}\right)$.
* **Perturbation:** A small perturbation is introduced to a vehicle's trajectory. Due to string instability at certain densities, this small disturbance amplifies over time, spontaneously generating backward-propagating **stop-and-go waves** (traffic jams without bottlenecks).
* **Numerical Methods:** The system is solved and compared using both **Explicit Euler** and **Heun's** methods.

### Autonomous Vehicle (AV) Integration
To stabilize the traffic flow, an Autonomous Vehicle is introduced into the ring. Unlike human drivers, the AV looks ahead at multiple neighbors and calculates its target speed based on the average of the optimal velocities computed from the distances of these neighbors:

$$\frac{dv_{AV}}{dt} = \beta \frac{\Delta v_{AV}}{(\Delta x_{AV})^2} \alpha \left( \frac{1}{m} \sum_{j=1}^{m} V(\Delta x_{AV, j}) - v_{AV} \right)$$

This non-local information allows the AV to anticipate braking and dampen the stop-and-go waves, smoothing the overall traffic flow.

---

## Linear Advection Equation
*In progress*

---

### Requirements
To run the simulations and the web applications, you will need the following Python libraries installed:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `plotly`
- `streamlit`

*(You can install them via pip: `pip install numpy scipy matplotlib pandas plotly streamlit`)*

---

### Repository Structure & How to Run

* **Core Modules:** `TrafficFlowsModule.py` and `OpinionDynamicsModule.py` contain the core mathematical functions and numerical integration logic.
* **Interactive Apps:** The files prefixed with `App-` are interactive web applications built with the Streamlit framework. To launch an app, open your terminal and run:
  ```bash
  streamlit run AppName.py

* Test Scripts: The other Python scripts in the repository are standalone tests designed to visualize numerical results, debug, and experiment with the variables.
* `Note`: The Applications are made with the help of Google Gemini just for this scope.
