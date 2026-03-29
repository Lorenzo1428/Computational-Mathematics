import numpy as np
from scipy.stats import pearson3


def get_init_distributions(dim, choose):
    if choose == 1:
        return np.random.uniform(0, 1, dim) * 2 - 1
    elif choose == 2:
        return np.random.normal(0.5, 0.3, dim)
    elif choose == 3:
        return pearson3.rvs(skew=1, loc=0.3, scale=0.15, size=dim)
    else:
        raise ValueError("Invalid choice")


def get_neighborhood(x, epsi):
    dim = len(x)
    Z = np.ones((dim, dim))
    M = np.abs((x * Z).T - (x * Z))
    return M < epsi


def get_interaction_matrix(x, dim, epsi, issym):
    Nij = get_neighborhood(x, epsi)
    sumij = np.sum(Nij, axis=0)
    if not issym:
        sigma = sumij
    else:
        sigma = dim * np.ones(dim)
    A = np.zeros((dim, dim))
    for i in range(dim):
        A[i, Nij[i, :] > 0] = 1 / sigma[i]
    L = (1 / sigma) * sumij * np.eye(dim)
    K = A - L
    if not issym:
        mul = findEigMul(A)
    else:
        mul = 0
    return (K, mul)


def heun_step(x, dt, T, epsi, issym):
    dim = len(x)
    Nt = int(np.floor(T / dt) + 1)
    dt = T / Nt
    mean = np.zeros((Nt + 1, 4))
    Y = np.zeros((Nt + 1, dim))
    Y[0, :] = x
    mean[0, 0] = (1 / dim) * np.sum(x)
    mean[0, 1] = (1 / dim) * np.sum(x**2)
    for t in range(0, Nt):
        K, mean[t + 1, 3] = get_interaction_matrix(Y[t, :], dim, epsi, issym)
        y1 = Y[t, :] + dt * K @ Y[t, :]
        K1, _ = get_interaction_matrix(y1, dim, epsi, issym)
        Y[t + 1, :] = Y[t, :] + 0.5 * dt * (K @ Y[t, :] + K1 @ y1)
        mean[t + 1, 0] = (1 / dim) * np.sum(Y[t + 1, :])
        mean[t + 1, 1] = (1 / dim) * np.sum(Y[t + 1, :] ** 2)
    mean[:, 2] = mean[:, 1] - mean[:, 0] ** 2
    return Y, mean


def findEigMul(K):
    eigvals, eigvecs = np.linalg.eig(K)
    alg_mul = np.where(np.abs(eigvals - eigvals[-1]) < 1e-6)[0]
    return np.linalg.matrix_rank(eigvecs[:, alg_mul])
