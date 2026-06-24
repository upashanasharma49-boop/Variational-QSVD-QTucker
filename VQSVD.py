import numpy as np
from scipy.optimize import minimize
import time

#Rotation and entangling gates

def Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0,  np.exp( 1j * theta / 2)]])


def Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s, c]])


def R(alpha, beta, gamma):
    return Rz(alpha) @ Rx(beta) @ Rz(gamma)


def CZ_gate():
    return np.diag([1, 1, 1, -1]).astype(complex)


def embed_2qubit(gate_2q, q1, q2, n_qubits):
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




#Unitary

def build_unitary(n_qubits, params, n_layers=2):
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=complex)
    idx = 0

    for layer in range(n_layers):

        #Rotation layer
        
        rot_layer = np.eye(1, dtype=complex)
        for q in range(n_qubits):
            a, b, g = params[idx], params[idx+1], params[idx+2]
            rot_layer = np.kron(rot_layer, R(a, b, g))
            idx += 3
        U = rot_layer @ U

        #Brick-wall entangling layer
        # Even layer: CZ on (0,1), (2,3), (4,5),..
        # Odd layer:  CZ on (1,2), (3,4), (5,6),..
        start = 0 if (layer % 2 == 0) else 1
        for q in range(start, n_qubits - 1, 2):
            CZ = embed_2qubit(CZ_gate(), q, q + 1, n_qubits)
            U  = CZ @ U

    #Final rotation layer
    final_layer = np.eye(1, dtype=complex)
    for q in range(n_qubits):
        a, b, g = params[idx], params[idx+1], params[idx+2]
        final_layer = np.kron(final_layer, R(a, b, g))
        idx += 3
    U = final_layer @ U

    return U


def param_count(n_qubits, n_layers):
    return (6 * n_layers + 3) * n_qubits


#Cost function

def apply_qsvd(state, n_A, n_B, params_A, params_B, n_layers):
    U_A = build_unitary(n_A, params_A, n_layers)
    V_B = build_unitary(n_B, params_B, n_layers)
    UV  = np.kron(U_A, V_B)
    return UV @ state


def cost_function(params, state, n_A, n_B, n_layers):
    n_pA = param_count(n_A, n_layers)
    params_A = params[:n_pA]
    params_B = params[n_pA:]

    psi = apply_qsvd(state, n_A, n_B, params_A, params_B, n_layers)
    dim_A = 2 ** n_A
    dim_B = 2 ** n_B
    n_min = min(dim_A, dim_B)

    C   = psi.reshape(dim_A, dim_B)
    rho = np.abs(C) ** 2  

    diag_prob = sum(rho[i, i] for i in range(n_min))
    total_prob = np.sum(rho)

    return 1.0 - (diag_prob / total_prob)


# Training

def train_qsvd(state, n_A, n_B, n_layers=2, max_iter=300, n_restarts=3, seed=42):
    np.random.seed(seed)
    n_pA = param_count(n_A, n_layers)
    n_pB = param_count(n_B, n_layers)
    n_total = n_pA + n_pB

    best_result = None
    best_cost = np.inf
    cost_history = []

    print(f"\n{'='*55}")
    print(f"  QSVD Training")
    print(f"  Subsystem A: {n_A} qubits | Subsystem B: {n_B} qubits")
    print(f"  Layers: {n_layers} | Total parameters: {n_total}")
    print(f"{'='*55}")

    for restart in range(n_restarts):
        x0 = np.random.uniform(0, 2 * np.pi, n_total)
        history = []

        def cb(xk):
            c = cost_function(xk, state, n_A, n_B, n_layers)
            history.append(c)

        res = minimize(
            cost_function,
            x0,
            args=(state, n_A, n_B, n_layers),
            method="L-BFGS-B",
            callback=cb,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8}
        )

        print(f"  Restart {restart+1}/{n_restarts} | "
              f"Cost = {res.fun:.6f} | Iters = {res.nit}")

        if res.fun < best_cost:
            best_cost    = res.fun
            best_result  = res
            cost_history = history

    print(f"\n  Best cost : {best_cost:.8f}")
    print(f"  (Cost = 0 means perfect SVD)")

    return {
        "optimal_params": best_result.x,
        "cost_history"  : cost_history,
        "final_cost"    : best_cost,
        "n_pA"          : n_pA,
        "n_pB"          : n_pB,
    }


#Post-processing

def get_von_neumann_entropy(lambdas):
    """S = -Σ λ_i² log₂(λ_i²)"""
    lsq = lambdas ** 2
    lsq = lsq[lsq > 1e-15]
    return float(-np.sum(lsq * np.log2(lsq)))


def get_schmidt_coefficients(state, n_A, n_B, optimal_params, n_layers):
    n_pA = param_count(n_A, n_layers)
    params_A = optimal_params[:n_pA]
    params_B = optimal_params[n_pA:]

    psi = apply_qsvd(state, n_A, n_B, params_A, params_B, n_layers)
    dim_A = 2 ** n_A
    dim_B = 2 ** n_B
    C = psi.reshape(dim_A, dim_B)
    n_min = min(dim_A, dim_B)

    lambdas = np.linalg.svd(C, compute_uv=False)[:n_min]
    norm = np.linalg.norm(lambdas)
    return lambdas / (norm + 1e-12)

def get_eigenvectors(n_A, n_B, optimal_params, n_layers):
    n_pA = param_count(n_A, n_layers)
    params_A = optimal_params[:n_pA]
    params_B = optimal_params[n_pA:]

    U_A = build_unitary(n_A, params_A, n_layers)
    V_B = build_unitary(n_B, params_B, n_layers)
    U_A_dag = U_A.conj().T
    V_B_dag = V_B.conj().T

    n_min = min(2**n_A, 2**n_B)
    U_vecs, V_vecs = [], []

    for i in range(n_min):
        e_i = np.zeros(2**n_A, dtype=complex); e_i[i] = 1.0
        U_vecs.append(U_A_dag @ e_i)
        e_i = np.zeros(2**n_B, dtype=complex); e_i[i] = 1.0
        V_vecs.append(V_B_dag @ e_i)

    return np.column_stack(U_vecs), np.column_stack(V_vecs)


#Classical SVD via eigendecomposition of MtM

def classical_svd(state, n_A, n_B, epsilon=1e-12):
    
    matrix = state.reshape(2**n_A, 2**n_B)
    transpose_matrix = matrix.conj().T
    transpose_A_A = np.dot(transpose_matrix, matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(transpose_A_A)

    sorted_indices = np.argsort(-np.abs(eigenvalues))
    sorted_eigenvalues = np.abs(eigenvalues[sorted_indices])
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    V = sorted_eigenvectors
    V_transpose = np.transpose(V)

    singular_values = np.sqrt(sorted_eigenvalues)
    Sigma = np.diag(singular_values)
    Sigma_inv = np.diag(1.0 / (singular_values + epsilon))

    U = np.dot(matrix, np.dot(V, Sigma_inv))

    n_min = min(2**n_A, 2**n_B)
    U2 = U[:, :n_min]
    S2 = singular_values[:n_min]
    VT2 = V_transpose[:n_min, :]

    norm = np.linalg.norm(S2)
    S2_norm = S2 / (norm + epsilon)

    entropy = get_von_neumann_entropy(S2_norm)

    return {
        "lambdas" : S2_norm,
        "U"       : U2,
        "V"       : V[:, :n_min],
        "VT"      : VT2,
        "Sigma"   : Sigma,
        "entropy" : entropy,
    }


#State preparation

def state_from_random(n_A, n_B, seed=42):
    np.random.seed(seed)
    dim = 2 ** (n_A + n_B)
    raw = np.random.randn(dim) + 1j * np.random.randn(dim)
    return raw / np.linalg.norm(raw)


def state_from_custom_matrix(M):
    
    M = np.array(M, dtype=complex)
    rows, cols = M.shape

    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    if not is_power_of_two(rows):
        raise ValueError(f"Matrix rows ({rows}) must be a power of 2.")
    if not is_power_of_two(cols):
        raise ValueError(f"Matrix cols ({cols}) must be a power of 2.")

    n_A   = int(np.log2(rows))
    n_B   = int(np.log2(cols))
    vec   = M.flatten()
    norm  = np.linalg.norm(vec)

    if norm < 1e-15:
        raise ValueError("Custom matrix is all zeros; cannot normalise.")

    state = vec / norm
    print(f"  Custom matrix accepted: {rows}×{cols}  →  n_A={n_A}, n_B={n_B}")
    return state, n_A, n_B


#Main

def run():
    n_layers = 2

    choice = input(
        "\nDo you want a random or custom matrix? (random/custom): "
    ).strip().lower()

    if choice == "random":
        n_A = int(input("Enter number of qubits in subsystem A: "))
        n_B = int(input("Enter number of qubits in subsystem B: "))
        print(f"\n[Mode] Random state  (n_A={n_A}, n_B={n_B})")
        state = state_from_random(n_A, n_B)

    elif choice == "custom":
        rows = int(input("Enter number of rows (must be a power of 2): "))
        cols = int(input("Enter number of cols (must be a power of 2): "))
        print(f"\nEnter elements for {rows}×{cols} matrix.")
        print("Complex numbers: use Python notation, e.g.  1+2j  or  0.5-1j\n")
        M = np.empty((rows, cols), dtype=complex)
        for i in range(rows):
            for j in range(cols):
                val      = input(f"  Element ({i+1},{j+1}): ")
                M[i, j]  = complex(val)
        state, n_A, n_B = state_from_custom_matrix(M)

    else:
        print("Invalid choice. Please enter 'random' or 'custom'.")
        return


#Classical SVD
    start_classical = time.perf_counter()
    ref = classical_svd(state, n_A, n_B)
    end_classical = time.perf_counter()
    classical_time = end_classical - start_classical

    print ("Time taken to run classical SVD = ", classical_time)

    
    print("\nClassical SVD (eigendecomposition of M†M):")
    for i, lam in enumerate(ref["lambdas"]):
        print(f"  σ_{i} = {lam:.6f}   (prob = {lam**2:.6f})")
    print(f"  Entropy S = {ref['entropy']:.6f} bits")

#QSVD training
    start_quantum = time.perf_counter()
    result = train_qsvd(state, n_A, n_B, n_layers=n_layers,
                        max_iter=300, n_restarts=5)
    end_quantum = time.perf_counter()
    quantum_time = end_quantum - start_quantum
    print ("Time taken to run quantum SVD = ", quantum_time)

#Comparison
    
    qsvd_lam = get_schmidt_coefficients(state, n_A, n_B,
                                              result["optimal_params"], n_layers)
    qsvd_S = get_von_neumann_entropy(qsvd_lam)
    U_vecs, V_vecs = get_eigenvectors(n_A, n_B,
                                      result["optimal_params"], n_layers)

    print(f"\nQSVD Results:")
    print(f"  {'σ_i (QSVD)':<18} {'σ_i (classical)':<18} {'|error|'}")
    print(f"  {'-'*55}")
    for i, (q, c) in enumerate(zip(qsvd_lam, ref["lambdas"])):
        print(f"  {q:<18.6f} {c:<18.6f} {abs(q-c):.6f}")

    print(f"\n  QSVD entropy      S = {qsvd_S:.6f} bits")
    print(f"  Classical entropy S = {ref['entropy']:.6f} bits")
    ent_err = abs(qsvd_S - ref["entropy"]) / (ref["entropy"] + 1e-12)
    print(f"  Relative entropy error = {ent_err:.4%}")

    return result, ref





if __name__ == "__main__":
    run()
