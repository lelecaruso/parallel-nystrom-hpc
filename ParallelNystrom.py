"""
        Parallel Randomized NYSTROM Low-Rank Approximation using MPI
        Computes C = AΩ and B = Ω^T A Ω in parallel


                DAVIDE VILLANI - EMANUELE CARUSO

"""
import numpy as np
from mpi4py import MPI
from utils import load_svmlight_text

class ParallelNystrom:
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.P = comm.Get_size()
        self.sqrt_P = int(np.sqrt(self.P))
        # Determine my position in the PxP process grid
        self.my_i = self.rank // self.sqrt_P
        self.my_j = self.rank % self.sqrt_P


    def DistributeData(self, filepath, n_samples, d):
        n = n_samples
        n_local = n // self.sqrt_P  
        if self.rank == 0:
            X_full, _ = load_svmlight_text(filepath, n_samples=n_samples, n_features=d)
            for p in range(self.P):
                pi, pj = p // self.sqrt_P, p % self.sqrt_P
                X_row = X_full[pi * n_local:(pi + 1) * n_local, :]
                X_col = X_full[pj * n_local:(pj + 1) * n_local, :]
                if p == 0:
                    self.X_row = X_row
                    self.X_col = X_col
                else:
                    self.comm.send(X_row, dest=p, tag=0)
                    self.comm.send(X_col, dest=p, tag=1)
        else:
            self.X_row = self.comm.recv(source=0, tag=0)
            self.X_col = self.comm.recv(source=0, tag=1)
        self.n, self.n_local, self.d = n, n_local, d


    def GenSketchMtx(self, l, seed=42):
        np.random.seed(seed + self.my_i )
        self.Omega_i = np.random.randn(self.n_local, l) / np.sqrt(l)
        np.random.seed(seed + self.my_j )
        self.Omega_j = np.random.randn(self.n_local, l) / np.sqrt(l)
        self.l = l


    def ComputeCijBij(self, c_squared):
        # distances between local rows and GLOBAL landmarks
        sq_row = np.sum(self.X_row**2, axis=1)[:, None]
        sq_col = np.sum(self.X_col**2, axis=1)[None, :]
        dists = sq_row + sq_col - 2 * self.X_row @ self.X_col.T
        K = np.exp(-dists / c_squared)
        self.C_ij = K @ self.Omega_j
        self.B_ij = self.Omega_i.T @ self.C_ij


    def ReduceB(self):
        self.B_global = np.zeros((self.l, self.l))
        self.comm.Reduce(self.B_ij, self.B_global, op=MPI.SUM, root=0)


    def ReduceC(self):
        # Split groups all processes with the same row index (my_i) into a new communicator,
        # ordered by col index (my_j). This allows us to do a Sum-Reduce over the rows.
        row_comm = self.comm.Split(self.my_i, self.my_j)
        self.C_i = np.zeros((self.n_local, self.l))
        row_comm.Reduce(self.C_ij, self.C_i, op=MPI.SUM, root=0)
        row_comm.Free()


    def GatherC(self):
        # Gather C_i from all row leaders to the root process
        if self.rank == 0:
            self.C = np.zeros((self.n, self.l))
            self.C[0 : self.n_local, :] = self.C_i
            for i in range(1, self.sqrt_P):
                C_temp = self.comm.recv(source=i * self.sqrt_P, tag=200)
                self.C[i * self.n_local : (i + 1) * self.n_local, :] = C_temp
        elif self.my_j == 0:
            self.comm.send(self.C_i, dest=0, tag=200)


    def Nystrom(self, filepath, n_samples, d, l, c_squared):
        if self.rank == 0:
            print(f"Running Parallel Nystrom with l={l}...")
        self.DistributeData(filepath, n_samples, d)
        self.GenSketchMtx(l)
        self.ComputeCijBij(c_squared)
        self.ReduceC()
        self.GatherC()
        self.ReduceB()  
        if self.rank == 0:
            print("Nystrom computation completed.\n")
        return (self.C, self.B_global) if self.rank == 0 else (None, None)




def KRankApprox(C, B, k):
    print("Computing K-Rank Approximation...")
    # Eigen-decomposition of the core matrix B
    evals, evecs = np.linalg.eigh(B)

    # Sort in descending order to handle the top-k approximation
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Sigma contains the approximated eigenvalues of A
    # Since B = Omega^T A Omega, its eigenvalues relate to the spectrum of A
    Sigma_k = evals

    # U_hat = C * U_B * Sigma_B^-1
    # We use a scale factor to normalize the columns
    scale = 1.0 / evals
    U_hat = C @ evecs @ np.diag(scale)

    print("K-Rank Approximation computed.\n")
    # Return exactly what the main expects:
    # U_hat_k (eigenvectors) and Sigma_k_sq (eigenvalues)
    return U_hat[:, :k], Sigma_k[:k]





