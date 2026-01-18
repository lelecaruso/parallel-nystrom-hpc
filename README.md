# Parallel Nystrom Approximation  
# Davide Villani - Emanuele Caruso - Sorbonne Université

This project implements the **Nystrom approximation** for large kernel matrices and supports **parallel execution** using MPI.

## Requirements

- Python 3.8+
- `mpi4py`
- `numpy`
- `scipy`

Install dependencies with:

```bash
pip install numpy scipy mpi4py
```

Run The project:
After specifying your filepath, number of samples and features in the main function
```bash
mpirun -n [perfect_square] python3 main.py
```
