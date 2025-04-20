import numpy as np
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iterations = iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        n = len(X)
        X = np.hstack((np.ones(n)[:, np.newaxis], X))
        y = np.where(y > 0, 1, -1)
        self.w = np.zeros(X.shape[1])
        for _ in range(self.iterations):
            y_predicted = np.sign(X @ self.w)
            if np.sum(y != y_predicted) == 0:
                break
            i = np.random.choice(np.argwhere(y != y_predicted).flatten())
            self.w += y[i] * X[i]
                    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.hstack((np.ones(len(X))[:, np.newaxis], X))
        y_predicted = X @ self.w
        return np.where(y_predicted > 0, 1, 0)
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.best_w = None
        self.iterations = iterations
        self.X = None
        self.y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        n = len(X)
        self.X = np.hstack((np.ones(n)[:, np.newaxis], X))
        self.y = np.where(y == y[0], 1, -1)
        self.w = np.zeros(self.X.shape[1])
        self.best_w = np.copy(self.w)
        accuracy = 0
        for _ in range(self.iterations - 10000):
            y_predicted = np.sign(self.X @ self.w)
            current_accuracy = np.sum(y_predicted == self.y)
            if current_accuracy > accuracy:
                self.best_w = np.copy(self.w)
                accuracy = current_accuracy
            self.w += self.X.T @ (self.y - y_predicted)
        y_predicted = np.sign(self.X @ self.w)
        current_accuracy = np.sum(y_predicted == self.y)
        if current_accuracy > accuracy:
            self.best_w = np.copy(self.w)
            accuracy = current_accuracy
        self.w += self.X.T @ (self.y - y_predicted)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.hstack((np.ones(len(X))[:, np.newaxis], X))
        y_predicted = X @ self.best_w
        return np.where(y_predicted > 0, 1, 0)
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    height, width = images.shape[1:]
    threshold = images.max()
    def max_p(image, threshold):
        b = image >= threshold
        diff = np.diff(np.concatenate(([0], b.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts
        return np.max(lengths) if lengths.size > 0 else 0
    v = []
    for image in images:
        v.append(int(max(np.apply_along_axis(max_p, axis=0, arr=image, threshold=threshold)) > height // 2))
    up = [max_p(image[0], threshold=images[images>0].mean()) for image in images]
    result = np.stack((v, up), axis=1)
    return result