import numpy as np

# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.sum((y_true - y_predicted) ** 2) / len(y_true)

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    return 1 - np.sum((y_predicted - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X = np.hstack((np.ones(X.shape[0])[:, np.newaxis], X))
        self.weights = np.linalg.inv(X.T @ X).T @ (X.T @ y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        return X @ self.weights[1:] + self.weights[0]
    
# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.alpha = alpha
        self.l = l
        self.iterations = iterations
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.c_[X, np.ones(X.shape[0])]
        n, f = X.shape
        self.weights = np.zeros(f)
        for i in range(self.iterations):
            y_pred = X.dot(self.weights)
            dw = (1 / n) * X.T.dot(y_pred - y) + self.l * np.sign(self.weights)
            self.weights -= self.alpha * dw

    def predict(self, X: np.ndarray):
        X = np.c_[X, np.ones(X.shape[0])]
        return X.dot(self.weights)
# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[:-1])

def get_most_important_features(linear_regression):
    return np.argsort(np.abs(linear_regression.weights[:-1]))[::-1]