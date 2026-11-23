import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        dw = np.dot(X.T, (p - y)) / n_samples
        db = np.sum(p - y) / n_samples

        w -= lr * dw
        b -= lr * db

    return w, b