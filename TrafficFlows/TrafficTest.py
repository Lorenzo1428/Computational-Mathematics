import matplotlib.pyplot as plt
import numpy as np
from TrafficFlowsModule import V_func, traffic_flow

N = 90
L = 1500
T = 700
dt = 0.05
alpha = 1
beta = 100
isheun = True
isauto = False
pert_time = 100
ni = 0.1

pos = np.linspace(0, L, N, endpoint=False)
vel = V_func(L / N) * np.ones(N)

X, V = traffic_flow(pos, vel, dt, T, L, N, alpha, beta, ni, pert_time, isheun, isauto)

Nt = int(np.floor(T / dt))
time = np.linspace(0, T, Nt + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(time, V[:, 43], color="blue", linewidth=1.5)
ax1.set_title("Velocity of the First Vehicle Over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Velocity (m/s)")
ax1.grid(True)

for i in range(N):
    ax2.plot(time, X[:, i], color="black", alpha=0.3, linewidth=0.8)

ax2.set_title("Vehicle Trajectories (Space-Time Diagram)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Absolute Position (m)")
ax2.grid(True)

plt.tight_layout()
plt.show()
