from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 0

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


# Task 1
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
    def __init__(self, split_dim: int, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.X = X
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.labels = np.unique(y)
        self.y = np.searchsorted(self.labels, y)
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = int(np.sqrt(self.n_features)) if max_features == "auto" else int(max_features)
        self.root = None
        self.bag_indices = np.random.choice(self.n_samples, size=self.n_samples, replace=True)
        self.out_of_bag = np.setdiff1d(np.arange(self.n_samples), np.unique(self.bag_indices))
        self.X_oob = X[self.out_of_bag]
        self.y_oob = y[self.out_of_bag]
        self.fit()

    def fit(self):
        X_bag, y_bag = self.X[self.bag_indices], self.y[self.bag_indices]
        self.root = self.build(X_bag, y_bag)

    def build(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_leaf:
            return DecisionTreeLeaf(y)
        
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        
        feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
        
        for i in feature_indices:
            left_mask = X[:, i] == 0
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            left_y, right_y = y[left_mask], y[right_mask]
            inf_gain = gain(left_y, right_y, self.criterion)
            if inf_gain > best_gain:
                best_gain = inf_gain
                best_feature = i
                best_left_indices = left_mask
                best_right_indices = right_mask
        
        if best_feature is None:
            return DecisionTreeLeaf(y)
        
        left_tree = self.build(X[best_left_indices], y[best_left_indices], depth + 1)
        right_tree = self.build(X[best_right_indices], y[best_right_indices], depth + 1)
        
        return DecisionTreeNode(best_feature, left_tree, right_tree)

    def predict_element(self, x, node):
        while not isinstance(node, DecisionTreeLeaf):
            if x[node.split_dim] == 0:
                node = node.left
            else:
                node = node.right
        return node.y
        
    def predict(self, X):
        return [self.labels[self.predict_element(x, self.root)] for x in X]
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.labels = None
    
    def fit(self, X, y):
        self.labels = np.unique(y)
        n_samples, n_features = X.shape
        self.max_features = int(np.sqrt(n_features)) if self.max_features == "auto" else int(self.max_features)
        for _ in range(self.n_estimators):
            tree = DecisionTree(X, y, criterion=self.criterion, max_depth=self.max_depth, 
                                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.searchsorted(self.labels, np.array([tree.predict(X) for tree in self.trees]))
        res_predictions = np.array([np.bincount(predictions[:, i], minlength=len(self.labels)).argmax() for i in range(X.shape[0])])
        return np.array([self.labels[i] for i in res_predictions])
    
# Task 3

def feature_importance(rfc):
    n_features = rfc.trees[0].X.shape[1]
    importances = np.zeros(n_features)
    for tree in rfc.trees:
        indices = tree.out_of_bag
        if len(indices) == 0:
            continue
        y_pred = tree.predict(tree.X_oob)
        error = np.mean(y_pred != tree.y_oob)
        for j in range(n_features):
            X_shuffle = tree.X_oob.copy()
            np.random.shuffle(X_shuffle[:, j])
            y_shuffle = tree.predict(X_shuffle)
            error_j = np.mean(y_shuffle != tree.y_oob)
            importances[j] += (error_j - error)
    importances /= len(rfc.trees)
    return importances

def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]

# Task 4

rfc_age = RandomForestClassifier(n_estimators=100, min_samples_leaf=100, max_depth=4, max_features=20)
rfc_gender = RandomForestClassifier(n_estimators=100, min_samples_leaf=100, max_depth=3)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = CatBoostClassifier(n_estimators=200, min_data_in_leaf=20, max_depth=10, loss_function='MultiClass')
catboost_rfc_gender = CatBoostClassifier(n_estimators=200, min_data_in_leaf=30, max_depth=7, loss_function='MultiClass')
catboost_rfc_age.load_model(__file__[:-7] + '/' + 'model_age')
catboost_rfc_gender.load_model(__file__[:-7] + '/' + 'model_sex')