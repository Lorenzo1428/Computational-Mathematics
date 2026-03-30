import numpy as np


def get_headway(pos, vel, L, N):
    p = np.ravel(pos)
    v = np.ravel(vel)

    Dx = np.zeros(N)
    Dv = np.zeros(N)

    Dx[:-1] = p[1:] - p[:-1]
    Dx[-1] = np.mod(p[0] - p[-1], L)

    Dv[:-1] = v[1:] - v[:-1]
    Dv[-1] = v[0] - v[-1]

    return Dx, Dv


def eul_esp(pos, vel, dt, Dx, Dv, V, a, b):
    y = pos + vel * dt
    v = vel + dt * (b * (Dv / (Dx**2 + 1e-6)) + a * (V - vel))
    return y, v


def heun(pos, vel, vel1, dt, Dx, Dv, Dx1, Dv1, V, V1, a, b):
    y = pos + (vel + vel1) * dt / 2
    ft = b * (Dv / (Dx**2 + 1e-6)) + a * (V - vel)
    ft1 = b * (Dv1 / (Dx1**2 + 1e-6)) + a * (V1 - vel1)
    v = vel + dt * (ft + ft1) / 2
    return y, v


def V_func(dx):
    Vs = 6.75 + 7.91 * (0.13 * (dx - 5) - 1.57)
    return np.maximum(0, Vs)


def traffic_flow(pos, vel, dt, T, L, N, a, b, ni, pert_time, is_heun, isauto):
    Nt = int(np.floor(T / dt))
    X = np.zeros((Nt + 1, N))
    V = np.zeros((Nt + 1, N))

    x = np.ravel(pos)
    v = np.ravel(vel)

    X[0, :] = x
    V[0, :] = v

    k = 10
    M = 15

    mask_not_k = np.ones(N, dtype=bool)
    mask_not_k[k] = False

    for i in range(Nt):
        Dx, Dv = get_headway(x, v, L, N)
        V0 = V_func(Dx)

        if not is_heun:
            if not isauto or (i * dt) < 400:
                x, v = eul_esp(x, v, dt, Dx, Dv, V0, a, b)
            else:
                x[mask_not_k], v[mask_not_k] = eul_esp(
                    x[mask_not_k],
                    v[mask_not_k],
                    dt,
                    Dx[mask_not_k],
                    Dv[mask_not_k],
                    V0[mask_not_k],
                    a,
                    b,
                )
                mean_dx = np.mean(Dx[k : min(k + M, N)])
                V2 = V_func(mean_dx)
                x[k], v[k] = eul_esp(x[k], v[k], dt, Dx[k], Dv[k], V2, a, b)

        else:
            if not isauto or (i * dt) < 400:
                x_pred, v_pred = eul_esp(x, v, dt, Dx, Dv, V0, a, b)
                Dx1, Dv1 = get_headway(x_pred, v_pred, L, N)
                V1 = V_func(Dx1)
                x, v = heun(x, v, v_pred, dt, Dx, Dv, Dx1, Dv1, V0, V1, a, b)
            else:
                x_pred = np.copy(x)
                v_pred = np.copy(v)

                x_pred[mask_not_k], v_pred[mask_not_k] = eul_esp(
                    x[mask_not_k],
                    v[mask_not_k],
                    dt,
                    Dx[mask_not_k],
                    Dv[mask_not_k],
                    V0[mask_not_k],
                    a,
                    b,
                )
                mean_dx = np.mean(Dx[k : min(k + M, N)])
                V2 = V_func(mean_dx)
                x_pred[k], v_pred[k] = eul_esp(x[k], v[k], dt, Dx[k], Dv[k], V2, a, b)

                Dx1, Dv1 = get_headway(x_pred, v_pred, L, N)
                V1 = V_func(Dx1)
                mean_dx2 = np.mean(Dx1[k : min(k + M, N)])
                V3 = V_func(mean_dx2)

                x[mask_not_k], v[mask_not_k] = heun(
                    x[mask_not_k],
                    v[mask_not_k],
                    v_pred[mask_not_k],
                    dt,
                    Dx[mask_not_k],
                    Dv[mask_not_k],
                    Dx1[mask_not_k],
                    Dv1[mask_not_k],
                    V0[mask_not_k],
                    V1[mask_not_k],
                    a,
                    b,
                )
                x[k], v[k] = heun(
                    x[k],
                    v[k],
                    v_pred[k],
                    dt,
                    Dx[k],
                    Dv[k],
                    Dx1[k],
                    Dv1[k],
                    V2,
                    V3,
                    a,
                    b,
                )

        if np.abs(i * dt - pert_time) < 1e-6:
            v[N // 2 - 2] = ni * v[N // 2 - 2]

        X[i + 1, :] = x
        V[i + 1, :] = v

    return X, V
