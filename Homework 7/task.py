import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        n_samples, n_features = X.shape
        P = matrix((y[:, None] * X) @ (y[:, None] * X).T)
        q = matrix(- np.ones(n_samples))
        G = matrix(np.vstack((- np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.full(n_samples, self.C))))
        A = matrix(y.astype(float), (1, n_samples))
        b = matrix(0.)
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x']).flatten()
        support = alpha > 1e-6
        self.support = np.where(support)[0]
        self.w = np.sum(alpha[support, None] * y[support, None] * X[support], axis=0)
        self.b = np.mean(y[support] - X[support] @ self.w)
        
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
    
# Task 2

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"
    def polynomial_kernel(x, y, c=c, power=power):
        return (x @ y.T + c) ** power
    return polynomial_kernel

def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффициентом sigma"
    def gaussian_kernel(x, y, sigma=sigma):
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        result = np.exp(- sigma * (np.sum(x**2, axis=1)[:, np.newaxis] + np.sum(y**2, axis=1) - 2 * x @ y.T))
        if x.shape[0] == 1 or y.shape[0] == 1:
            result = np.squeeze(result)
        return result
    return gaussian_kernel

# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        n_samples, n_features = X.shape
        P = matrix(np.outer(y, y) * self.kernel(X, X))
        q = matrix(- np.ones(n_samples))
        G = matrix(np.vstack((- np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.full(n_samples, self.C))))
        A = matrix(y.astype(float), (1, n_samples))
        b = matrix(0.)
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x']).flatten()
        support = alpha > 1e-6
        self.support = np.where(support)[0]
        self.alpha = alpha[support]
        self.X = X[support]
        self.y = y[support]
        K = self.kernel(self.X, self.X)
        self.b = np.mean(self.y - np.sum(self.alpha * self.y * K, axis=1))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        K = self.kernel(X, self.X)
        return np.sum(self.alpha * self.y * K, axis=1) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))