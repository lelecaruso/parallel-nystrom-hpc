import numpy as np
from mpi4py import MPI
import sys

"""
Parallel Randomized NYSTROM Low-Rank Approximation using MPI
Computes C = AΩ and B = Ω^T A Ω in parallel
where A is a kernel matrix computed on-the-fly
"""


def load_svmlight_text(filepath, n_samples=None, n_features=None):
    if n_samples is None:
        with open(filepath) as f:
            n_samples = sum(1 for _ in f)

    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)

    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            parts = line.strip().split()
            y[i] = int(parts[0])
            for item in parts[1:]:
                idx, val = item.split(":")
                X[i, int(idx) - 1] = float(val)
    return X, y


class ParallelNystrom:
    def __init__(self, comm):
        """
        Initializes the processor grid. Assumes P is a perfect square for 2D distribution.
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.P = comm.Get_size()
        self.sqrt_P = int(np.sqrt(self.P))
        self.my_i = self.rank // self.sqrt_P
        self.my_j = self.rank % self.sqrt_P

    def load_and_distribute_data(self, filepath, n_samples, d):
        """
        2D DATA DISTRIBUTION STRATEGY:
        To compute the kernel matrix A in a 2D block-cyclic fashion without
        storing the full dataset on every node:
        1. The dataset X is partitioned into sqrt(P) horizontal blocks .
        2. Each processor in the virtual grid receives TWO of these blocks:
           - X_row
           - X_col
        """
        n = n_samples
        n_local = n // self.sqrt_P  # Size of each macro-block of rows

        if self.rank == 0:
            print(f"\nLoading data from {filepath}...")
            X_full, _ = load_svmlight_text(filepath, n_samples=n_samples, n_features=d)

            #  Slice and distribute data across the P = sqrt_P x sqrt_P grid
            for p in range(self.P):
                # Map the linear MPI rank to 2D grid coordinates (pi, pj)
                pi, pj = p // self.sqrt_P, p % self.sqrt_P

                if p == 0:
                    # Root process keeps the first blocks for its local grid position (0,0)
                    self.X_row = X_full[0:n_local, :]
                    self.X_col = X_full[0:n_local, :]
                else:
                    # Send the horizontal slice pi to be used as 'rows' in the local kernel
                    self.comm.send(
                        X_full[pi * n_local : (pi + 1) * n_local, :], dest=p, tag=0
                    )
                    # Send the horizontal slice pj to be used as 'columns' in the local kernel
                    self.comm.send(
                        X_full[pj * n_local : (pj + 1) * n_local, :], dest=p, tag=1
                    )
        else:
            # Non-root processes receive their specific row/column blocks
            # This allows each node to compute A_ij = K(X_pi, X_pj) independently
            self.X_row = self.comm.recv(source=0, tag=0)
            self.X_col = self.comm.recv(source=0, tag=1)

        self.n, self.n_local, self.d = n, n_local, d

    def generate_omega(self, l, seed=42):
        """
        Generates the sketching matrix Omega distributed across processors.
        """
        seed = self.comm.bcast(seed if self.rank == 0 else None, root=0)
        np.random.seed(seed + self.my_i * 1000)
        self.Omega_i = np.random.randn(self.n_local, l)
        np.random.seed(seed + self.my_j * 1000)
        self.Omega_j = np.random.randn(self.n_local, l)
        self.l = l

    def compute_C_local(self, c_squared):
        """
        Computes local contribution to C = A * Omega using the RBF kernel.
        Each processor (i,j) computes the similarity between row-block 'i'
        and row-block 'j' of the original dataset.
        This step implements the kernel matrix computation on-the-fly.
        """
        sq_row = np.sum(self.X_row**2, axis=1)[:, np.newaxis]
        sq_col = np.sum(self.X_col**2, axis=1)
        dists = sq_row + sq_col - 2 * self.X_row @ self.X_col.T
        # the matrix A_ij is computed on-the-fly locally on each processor using the block of data
        A_ij = np.exp(-dists / c_squared)
        self.C_ij = A_ij @ self.Omega_j

    def reduce_C(self):
        """
        Performs row-wise reduction to aggregate contributions to C_i = sum_j (A_ij * Omega_j).
        """
        row_comm = self.comm.Split(self.my_i, self.my_j)
        self.C_i = np.zeros((self.n_local, self.l))
        row_comm.Reduce(self.C_ij, self.C_i, op=MPI.SUM, root=0)
        row_comm.Free()

    def allreduce_B(self):
        """
        Computes the core matrix B = Omega^T * A * Omega.
        This is done by aggregating local contributions: B = sum(Omega_i^T * A_ij * Omega_j).
        Since B is a small (l x l) matrix, we use Allreduce to make it available
        to all processors for the final Nystrom decomposition.
        """
        # Calcola B_ij = Omega_i.T @ C_ij
        B_local = self.Omega_i.T @ self.C_ij
        self.B_global = np.zeros((self.l, self.l))
        # Somma globale su TUTTI i processori
        self.comm.Allreduce(B_local, self.B_global, op=MPI.SUM)

    def gather_C(self):
        """
        Gathers blocks of C to the root processor for final reconstruction. [cite: 31]
        """
        if self.rank == 0:
            self.C = np.zeros((self.n, self.l))
            self.C[0 : self.n_local, :] = self.C_i
            for i in range(1, self.sqrt_P):
                C_temp = self.comm.recv(source=i * self.sqrt_P, tag=200)
                self.C[i * self.n_local : (i + 1) * self.n_local, :] = C_temp
        elif self.my_j == 0:
            self.comm.send(self.C_i, dest=0, tag=200)

    def run_nystrom(self, filepath, n_samples, d, l, c_squared):
        self.load_and_distribute_data(filepath, n_samples, d)
        self.generate_omega(l)
        self.compute_C_local(c_squared)
        self.reduce_C()
        self.allreduce_B()  # B è già sommato correttamente qui
        self.gather_C()
        return (self.C, self.B_global) if self.rank == 0 else (None, None)


def compute_nystrom_approximation(C, B, k):
    """
    Given C and B from the Nystrom method, compute the low-rank approximation
    A ≈ U_hat Σ_k U_hat^T
    where U_hat are the approximated eigenvectors and Σ_k the approximated eigenvalues.
    """
    # 1. Eigen-decomposition of the core matrix B
    evals, evecs = np.linalg.eigh(B)

    # 2. Sort in descending order to handle the top-k approximation
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # 3. Numerical stability: filter out near-zero eigenvalues
    threshold = 1e-12
    mask = evals > threshold

    # Sigma contains the approximated eigenvalues of A
    # Since B = Omega^T A Omega, its eigenvalues relate to the spectrum of A
    Sigma_k = evals[mask]

    # 4. Compute U_hat (Approximated Eigenvectors)
    # Formula: U_hat = C * U_B * Sigma_B^-1
    # We use a scale factor to normalize the columns
    scale = np.zeros_like(evals)
    scale[mask] = 1.0 / evals[mask]
    U_hat = C @ evecs @ np.diag(scale)

    # Return exactly what the main expects:
    # U_hat_k (eigenvectors) and Sigma_k_sq (eigenvalues)
    return U_hat[:, :k], Sigma_k[:k]


import matplotlib.pyplot as plt


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nystrom = ParallelNystrom(comm)

    filepath = "dataset/YearPredictionMSD.t"
    n_samples = 1000
    d = 90
    c_squared = 1e8

    # Definiamo i parametri da testare
    l_values = [400, 700, 1000]
    k_values = [25, 50, 100, 200, 400]

    if rank == 0:
        plt.figure(figsize=(12, 7))
        X_full, _ = load_svmlight_text(filepath, n_samples=n_samples, n_features=d)
        sq_norms = np.sum(X_full**2, axis=1)[:, np.newaxis]
        A_full = np.exp(-(sq_norms + sq_norms.T - 2 * X_full @ X_full.T) / c_squared)
        norm_A = np.linalg.norm(A_full, ord="nuc")

    for l in l_values:
        errors_for_l = []
        # Calcolo parallelo una sola volta per ogni l
        C, B = nystrom.run_nystrom(filepath, n_samples, d, l, c_squared)

        for k in k_values:
            # Salta se il rango richiesto è maggiore dello sketch disponibile
            if k > l:
                continue

            if rank == 0:
                # Approssimazione di rango k partendo dallo sketch l
                U_hat_k, Sigma_k = compute_nystrom_approximation(C, B, k)
                A_nyst = U_hat_k @ np.diag(Sigma_k) @ U_hat_k.T

                error_nuc = np.linalg.norm(A_full - A_nyst, ord="nuc") / norm_A
                errors_for_l.append(error_nuc)
                print(f"l={l:4d}, k={k:4d} | Error: {error_nuc:.6e}")

        if rank == 0:
            plt.plot(
                k_values[: len(errors_for_l)],
                errors_for_l,
                marker="o",
                label=f"Sketch l={l}",
            )

    if rank == 0:
        plt.title(
            "Nyström Convergence: Impact of Sketch Size (l) and Rank (k)", fontsize=14
        )
        plt.xlabel("Target Rank (k)", fontsize=12)
        plt.ylabel("Relative Nuclear Norm Error", fontsize=12)
        plt.yscale(
            "log"
        )  # La scala logaritmica aiuta a vedere bene le differenze piccole
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.savefig("nystrom_l_vs_k_comparison.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
