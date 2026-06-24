import numpy as np
from scipy.optimize import minimize


def Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)


def Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                      [-1j * s, c]], dtype=complex)


def R(alpha, beta, gamma):
    return Rz(alpha) @ Rx(beta) @ Rz(gamma)


def CZ_gate():
    return np.diag([1, 1, 1, -1]).astype(complex)


def embed_2qubit(gate_2q, q1, n_qubits):
    I = np.eye(2, dtype=complex)
    result = np.eye(1, dtype=complex)
    i = 0
    while i < n_qubits:
        if i == q1:
            result = np.kron(result, gate_2q)
            i += 2
        else:
            result = np.kron(result, I)
            i += 1
    return result


def build_unitary(n_qubits, params, n_layers=2):
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=complex)
    idx = 0
    for layer in range(n_layers):
        rot_layer = np.eye(1, dtype=complex)
        for q in range(n_qubits):
            a, b, g = params[idx], params[idx + 1], params[idx + 2]
            rot_layer = np.kron(rot_layer, R(a, b, g))
            idx += 3
        U = rot_layer @ U
        start = 0 if layer % 2 == 0 else 1
        for q in range(start, n_qubits - 1, 2):
            CZ = embed_2qubit(CZ_gate(), q, n_qubits)
            U = CZ @ U
    final_layer = np.eye(1, dtype=complex)
    for q in range(n_qubits):
        a, b, g = params[idx], params[idx + 1], params[idx + 2]
        final_layer = np.kron(final_layer, R(a, b, g))
        idx += 3
    U = final_layer @ U
    return U


def param_count(n_qubits, n_layers):
    return (6 * n_layers + 3) * n_qubits


def apply_qsvd(state, n_A, n_B, params_A, params_B, n_layers):
    U_A = build_unitary(n_A, params_A, n_layers)
    V_B = build_unitary(n_B, params_B, n_layers)
    return np.kron(U_A, V_B) @ state


def cost_function(params, state, n_A, n_B, n_layers):
    n_pA = param_count(n_A, n_layers)
    psi = apply_qsvd(state, n_A, n_B, params[:n_pA], params[n_pA:], n_layers)
    dim_A, dim_B = 2 ** n_A, 2 ** n_B
    n_min = min(dim_A, dim_B)
    C = psi.reshape(dim_A, dim_B)
    rho = np.abs(C) ** 2
    diag_prob = sum(rho[i, i] for i in range(n_min))
    return 1.0 - diag_prob / np.sum(rho)


def train_qsvd(state, n_A, n_B, n_layers=2, max_iter=300, n_restarts=3, seed=42):
    np.random.seed(seed)
    n_pA = param_count(n_A, n_layers)
    n_pB = param_count(n_B, n_layers)
    n_total = n_pA + n_pB
    best_x, best_cost = None, np.inf
    for _ in range(n_restarts):
        x0 = np.random.uniform(0, 2 * np.pi, n_total)
        res = minimize(cost_function, x0, args=(state, n_A, n_B, n_layers),
                        method="L-BFGS-B",
                        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
        if res.fun < best_cost:
            best_cost, best_x = res.fun, res.x
    return best_x, best_cost, n_pA, n_pB


def next_pow2(x):
    return 1 << (x - 1).bit_length()


def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def fold(M, mode, shape):
    full_shape = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(M.reshape(full_shape), 0, mode)


def mode_n_product(T, F, mode):
    T_unf = unfold(T, mode)
    new_unf = F @ T_unf
    new_shape = list(T.shape)
    new_shape[mode] = F.shape[0]
    return fold(new_unf, mode, new_shape)


def qsvd_factor(M, rank, n_layers=2, n_restarts=3, max_iter=300, seed=42):
    n_rows, n_cols = M.shape
    assert rank <= min(n_rows, n_cols)
    r2, c2 = next_pow2(n_rows), next_pow2(n_cols)
    M_pad = np.zeros((r2, c2), dtype=complex)
    M_pad[:n_rows, :n_cols] = M
    vec = M_pad.flatten()
    norm_val = np.linalg.norm(vec)
    if norm_val < 1e-15:
        raise ValueError("Unfolded matrix is all zeros.")
    state = vec / norm_val
    n_A, n_B = int(np.log2(r2)), int(np.log2(c2))
    params, cost, n_pA, n_pB = train_qsvd(state, n_A, n_B, n_layers, max_iter, n_restarts, seed)
    params_A, params_B = params[:n_pA], params[n_pA:]
    psi = apply_qsvd(state, n_A, n_B, params_A, params_B, n_layers)
    dim_A, dim_B = 2 ** n_A, 2 ** n_B
    n_min = min(dim_A, dim_B)
    C = psi.reshape(dim_A, dim_B)
    diag_mag = np.abs(np.diag(C[:n_min, :n_min]))
    order = np.argsort(diag_mag)[::-1][:rank]
    sigma = norm_val * diag_mag[order]
    U_A = build_unitary(n_A, params_A, n_layers)
    A_full = U_A.conj().T[:n_rows, :]
    factor = np.real(A_full[:, order])
    factor, _ = np.linalg.qr(factor)
    factor = factor[:, :rank]
    return factor, sigma, cost


def quantum_tucker_decomposition(T, ranks, n_layers=2, n_restarts=3, max_iter=300, seed=42):
    ndim = T.ndim
    if len(ranks) != ndim:
        raise ValueError("ranks must have one entry per tensor mode.")
    factors, sigmas, costs = [], [], []
    for mode in range(ndim):
        Tk = unfold(T, mode)
        Uk, sigma_k, cost_k = qsvd_factor(Tk, ranks[mode], n_layers, n_restarts, max_iter, seed)
        factors.append(Uk)
        sigmas.append(sigma_k)
        costs.append(cost_k)

    G = T.astype(complex)
    for mode in range(ndim):
        G = mode_n_product(G, factors[mode].conj().T, mode)

    T_rec = G
    for mode in range(ndim):
        T_rec = mode_n_product(T_rec, factors[mode], mode)
    if np.isrealobj(T):
        T_rec = np.real(T_rec)

    return {
        "factors": factors,
        "singular_values": sigmas,
        "costs": costs,
        "core": G,
        "reconstruction": T_rec,
    }


if __name__ == "__main__":
    np.random.seed(0)
    T = np.random.rand(4, 4, 4)
    ranks = [2, 2, 2]
    result = quantum_tucker_decomposition(T, ranks, n_layers=3, n_restarts=6, max_iter=300)
    rel_err = np.linalg.norm(T - result["reconstruction"]) / np.linalg.norm(T)
    print("Per-mode QSVD costs:", result["costs"])
    for k, Uk in enumerate(result["factors"]):
        print(f"\nFactor matrix U({k+1}), shape {Uk.shape}:")
        print(Uk)
    print("\nCore tensor G, shape", result["core"].shape, ":")
    print(np.real(result["core"]))
    print("\nRelative reconstruction error:", rel_err)
