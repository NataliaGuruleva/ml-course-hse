from sklearn.neighbors import KDTree
import numpy as np
import random
import copy
from collections import deque
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", max_iter: int = 300):
        """
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из X,
            3. k-means++ --- центроиды кластеров инициализируются при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X: np.ndarray, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn.
        """
        if self.init == 'sample':
            indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
            centroids = X[indices]
        elif self.init == 'random':
            centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), (self.n_clusters, X.shape[1]))
        elif self.init == 'k-means++':
            centroids = [X[np.random.randint(len(X))]]
            for _ in range(1, self.n_clusters):
                distances = np.array([min(np.linalg.norm(x - centroid) for centroid in centroids) for x in X])
                probabilities = distances ** 2 / np.sum(distances ** 2)
                new_centroid_idx = np.random.choice(len(X), p=probabilities)
                centroids.append(X[new_centroid_idx])
            centroids = np.array(centroids)
        it = 0
        diff = float('inf')
        while it < self.max_iter and diff > 0.001:
            it += 1
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            clusters = np.argmin(distances, axis=1)
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X[clusters == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(X[np.random.randint(len(X))])
            new_centroids = np.array(new_centroids)
            diff = np.sum(np.linalg.norm(centroids - new_centroids, axis=1))
            centroids = new_centroids
        self.centroids = centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Для каждого элемента из X возвращает номер кластера, к которому относится данный элемент.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для элементов которого находятся ближайшие кластеры.

        Return
        ------
        labels : np.ndarray
            Вектор индексов ближайших кластеров (по одному индексу для каждого элемента из X).
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        all_neighbours = tree.query_radius(X, r=self.eps)
        
        def find_query(point):
            return all_neighbours[point]
    
        def expand_cluster(point, neighbours):
            clusters[point] = c
            while neighbours:
                x = neighbours.pop()
                if not visited[x]:
                    visited[x] = 1
                    neighbours_x = find_query(x)
                    if len(neighbours_x) >= self.min_samples:
                        neighbours.extend([y for y in neighbours_x if not visited[y]])
                if clusters[x] == -1:
                    clusters[x] = c
    
        c = -1
        n = len(X)
        visited = np.zeros(n, dtype=bool)
        clusters = np.full(n, -1, dtype=int)
        for i in range(len(X)):
            if visited[i]:
                continue
            visited[i] = 1
            neighbours = deque(find_query(i))
            if len(neighbours) >= self.min_samples:
                c += 1
                expand_cluster(i, neighbours)
        return clusters
# Task 3

class AgglomertiveClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        pass
    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        pass
