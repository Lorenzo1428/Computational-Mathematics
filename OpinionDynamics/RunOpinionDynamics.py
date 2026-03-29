import matplotlib.pyplot as plt
import OpinionDynamicsModule

epsi = 0.2
issym = False
dt = 0.1
T = 5
dim = 10
init_distribution = 2
x = OpinionDynamicsModule.get_init_distributions(10, 2)
X, mean = OpinionDynamicsModule.heun_step(x, dt, T, epsi, issym)
plt.plot(X)
plt.xlabel("Time steps")
plt.ylabel("Opinion Value")
plt.title(
    f"Opinion Evolution (Epsilon={epsi}, Symmetric={issym})",
    fontsize=14,
)
plt.show()
