from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    nums = np.unique(x, return_counts=True)[1]
    return 1 - np.sum((nums / np.sum(nums)) ** 2)
    
def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    nums = np.unique(x, return_counts=True)[1]
    return - np.sum((nums / np.sum(nums)) * np.log2(nums / np.sum(nums)))

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    n_left = len(left_y)
    n_right = len(right_y)
    n = n_left + n_right
    return criterion(np.concatenate((left_y, right_y))) - n_left / n * criterion(left_y) - n_right / n * criterion(right_y)


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        self.counts = np.bincount(ys)
        self.y = np.argmax(self.counts)

class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion : str = "gini", 
                 max_depth : Optional[int] = None, 
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.labels = np.unique(y)
        y = np.searchsorted(self.labels, y)
        self.root = self.build(X, y)
        
    def build(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return DecisionTreeLeaf(y)
        
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_value = None
        best_left_indices = None
        best_right_indices = None
        
        for i in range(n_features):
            values = np.unique(X[:, i])
            if len(values) > 20:
                values = np.percentile(values, np.linspace(0, 100, 10))
            
            for value in values:
                left_mask = X[:, i] < value
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_y, right_y = y[left_mask], y[right_mask]
                inf_gain = gain(left_y, right_y, self.criterion)
                if inf_gain > best_gain:
                    best_gain = inf_gain
                    best_feature = i
                    best_value = value
                    best_left_indices = left_mask
                    best_right_indices = right_mask
        
        if best_feature is None:
            return DecisionTreeLeaf(y)
        
        left_tree = self.build(X[best_left_indices], y[best_left_indices], depth + 1)
        right_tree = self.build(X[best_right_indices], y[best_right_indices], depth + 1)
        
        return DecisionTreeNode(best_feature, best_value, left_tree, right_tree)
    
    def predict_element(self, x, node):
        
        while not isinstance(node, DecisionTreeLeaf):
            if x[node.split_dim] < node.split_value:
                node = node.left
            else:
                node = node.right
        return {self.labels[label]: count / np.sum(node.counts) for label, count in enumerate(node.counts)}
    
    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        
        return [self.predict_element(x, self.root) for x in X]
    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4
task4_dtc = DecisionTreeClassifier(max_depth=20, min_samples_leaf=30, criterion='entropy')

