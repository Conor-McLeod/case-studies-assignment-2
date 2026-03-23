import numpy as np


def gmres_basic(A, b, x0=None, tol=1e-8, max_iter=100):
    """
    Basic GMRES implementation that solves the least squares problem at each step
    using np.linalg.lstsq.
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
    H = []

    beta = np.linalg.norm(r)

    for j in range(max_iter):
        w = A @ V[j]

        # Modified Gram-Schmidt
        H_col = []
        for i in range(j + 1):
            h = np.vdot(V[i], w)
            H_col.append(h)
            w = w - h * V[i]

        h_next = np.linalg.norm(w)
        H_col.append(h_next)
        H.append(H_col)

        # Build dense H matrix for lstsq
        H_mat = np.zeros((j + 2, j + 1))
        for col in range(j + 1):
            for row in range(col + 2):
                H_mat[row, col] = H[col][row]

        e1 = np.zeros(j + 2)
        e1[0] = beta

        # Solve least squares
        y, _, _, _ = np.linalg.lstsq(H_mat, e1, rcond=None)

        # Build approximate solution
        V_mat = np.column_stack(V)
        x_approx = x + V_mat @ y

        # Check true residual
        r_approx = b - (A @ x_approx)
        rel_res = np.linalg.norm(r_approx) / b_norm
        res_hist.append(rel_res)

        if rel_res < tol or h_next < 1e-14:
            return x_approx, j + 1, res_hist

        # Append next basis vector
        if h_next > 1e-14:
            V.append(w / h_next)
        else:
            # Breakdown
            V.append(np.zeros_like(w))

    return x_approx, max_iter, res_hist
