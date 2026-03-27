import matplotlib.pyplot as plt
import numpy as np
import Opinion

epsi = 0.1
x = Opinion.get_init_distributions(10, 1)
M = Opinion.get_neighborhood(x, epsi)

print(x, "\n", M)
