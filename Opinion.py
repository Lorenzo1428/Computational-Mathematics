import numpy as np


def get_init_distributions(dim, choose):
    if choose == 1:
        return np.random.uniform(0, 1, dim)
    elif choose == 2:
        return np.random.normal(0.5, 0.3, dim)
    elif choose == 3:
        return np.random.normal(0.3, 0.15, dim)
    else:
        raise ValueError("Invalid choice")


def get_neighborhood(x, eps):
    dim = len(x)
    Z = np.ones((dim, dim))
    M = np.abs((x * Z).T - (x * Z))
    return M < eps


def get_interaction_matrix(dim, eps):
    A = np.zeros((dim, dim))
    return A


def heun_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + dt * k1)
    return x + 0.5 * dt * (k1 + k2)
