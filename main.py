"""
        Parallel Randomized NYSTROM Low-Rank Approximation using MPI
        Computes C = AΩ and B = Ω^T A Ω in parallel


                DAVIDE VILLANI - EMANUELE CARUSO

"""

import numpy as np
from mpi4py import MPI
from ParallelNystrom import KRankApprox, ParallelNystrom
from utils import load_svmlight_text


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #IMPORTANT ASSUMPTION: P is a perfect square
    P = comm.Get_size()
    sP = np.sqrt(P)
    nystrom = ParallelNystrom(comm)

    filepath = ""
    n_samples = 0
    d = 90
    c_squared = 1e10

    #testing param
    l_values = []
    k_values = []

    # Precompute full A and its nuclear norm for error computation
    if rank == 0:
        X_full, _ = load_svmlight_text(filepath, n_samples=n_samples, n_features=d)
        sq_norms = np.sum(X_full**2, axis=1)[:, np.newaxis]
        A_full = np.exp(-(sq_norms + sq_norms.T - 2 * X_full @ X_full.T) / c_squared)
        norm_A = np.linalg.norm(A_full, ord="nuc")

    for l in l_values:
        errors_for_l = []
        # Run Parallel NYSTROM
        C, B = nystrom.Nystrom(filepath, n_samples, d, l, c_squared)
        for k in k_values:
            # only compute for meaningful k <= l
            if k > l:
                continue
            if rank == 0:
                #compute the actual k rank approximation 
                U_hat_k, Sigma_k = KRankApprox(C, B, k)
                # reconstruct the Nystrom approximation to compute error
                A_nyst = U_hat_k @ np.diag(Sigma_k) @ U_hat_k.T
                # compute and print the relative nuclear norm error
                error_nuc = np.linalg.norm(A_full - A_nyst, ord="nuc") / norm_A
                errors_for_l.append(error_nuc)
                print(f"l={l:4d}, k={k:4d} | Error: {error_nuc:.6e}")




if __name__ == "__main__":
    main()
