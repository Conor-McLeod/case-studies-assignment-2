import numpy as np
from scipy.linalg import solve_triangular


def apply_givens_rotation(h, cs, sn):
    v1 = cs * h[0] + sn * h[1]
    v2 = -sn * h[0] + cs * h[1]
    return v1, v2


def generate_givens_rotation(v1, v2):
    if v2 == 0:
        return 1.0, 0.0
    elif v1 == 0:
        return 0.0, 1.0
    else:
        hypot = np.hypot(v1, v2)
        cs = v1 / hypot
        sn = v2 / hypot
        return cs, sn


def gmres_monitor(A, b, x0=None, tol=1e-8, max_iter=100):
    """
    GMRES implementation that monitors the residual norm using the
    Givens rotation g vector without building the approximate solution.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - (A @ x)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1.0

    rel_res = np.linalg.norm(r) / b_norm
    res_hist = [rel_res]
    if rel_res < tol:
        return x, 0, res_hist

    V = [r / np.linalg.norm(r)]
    H = np.zeros((max_iter + 1, max_iter))

    cs = np.zeros(max_iter)
    sn = np.zeros(max_iter)

    beta = np.linalg.norm(r)
    g = np.zeros(max_iter + 1)
    g[0] = beta

    for j in range(max_iter):
        w = A @ V[j]

        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = np.vdot(V[i], w)
            w = w - H[i, j] * V[i]

        h_next = np.linalg.norm(w)
        H[j + 1, j] = h_next

        # Apply previous Givens rotations to the new column of H
        for i in range(j):
            H[i, j], H[i + 1, j] = apply_givens_rotation(H[i : i + 2, j], cs[i], sn[i])

        # Generate new Givens rotation
        cs[j], sn[j] = generate_givens_rotation(H[j, j], H[j + 1, j])

        # Apply it to the new column of H
        H[j, j], H[j + 1, j] = apply_givens_rotation(H[j : j + 2, j], cs[j], sn[j])

        # Apply it to g
        g[j], g[j + 1] = apply_givens_rotation(g[j : j + 2], cs[j], sn[j])

        # Monitor residual norm approximation without building x
        rel_res = np.abs(g[j + 1]) / b_norm
        res_hist.append(rel_res)

        if rel_res < tol or h_next < 1e-14:
            y = solve_triangular(H[: j + 1, : j + 1], g[: j + 1])
            V_mat = np.column_stack(V)
            x_approx = x + V_mat @ y
            return x_approx, j + 1, res_hist

        if h_next > 1e-14:
            V.append(w / h_next)
        else:
            V.append(np.zeros_like(w))

    # Max iterations reached, build final approximate solution
    y = solve_triangular(H[:max_iter, :max_iter], g[:max_iter])
    V_mat = np.column_stack(V[:-1] if len(V) > max_iter else V)
    x_approx = x + V_mat @ y

    return x_approx, max_iter, res_hist
