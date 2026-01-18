import numpy as np


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
