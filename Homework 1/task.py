import numpy as np
import pandas
from typing import NoReturn, Tuple, List

# Task 1
def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    data = pandas.read_csv(path_to_csv)
    data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)
    X = data.iloc[:, 1:].to_numpy()
    y = data.iloc[:,0].to_numpy().ravel()
    y = y == 'M'
    y = y.astype(int)
    
    return (X, y)

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    data = pandas.read_csv(path_to_csv)
    data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:,-1].to_numpy().ravel()
    
    return (X, y)
    
# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    train_size = int(len(X) * ratio)
    test_size = len(X) - train_size
    X_train = X[: train_size]
    X_test = X[-test_size :]
    y_train = y[: train_size]
    y_test = y[-test_size :]
    
    return (X_train, y_train, X_test, y_test)
    
# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    labels = np.unique(y_true)
    precision = np.zeros(len(labels))
    recall = np.zeros(len(labels))
    true_positive = np.zeros(len(labels))
    false_positive = np.zeros(len(labels))
    false_negative = np.zeros(len(labels))
    true_negative = np.zeros(len(labels))

    for i, label in enumerate(labels):
        true_positive[i] = np.sum((y_true == label) & (y_pred == label))
        false_positive[i] = np.sum((y_true != label) & (y_pred == label))
        false_negative[i] = np.sum((y_true == label) & (y_pred != label))
        true_negative[i] = np.sum((y_true != label) & (y_pred != label))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    precision = np.nan_to_num(precision, nan=0.0)
    recall = np.nan_to_num(recall, nan=0.0)
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return precision, recall, accuracy
    
# Task 4

class Heap():
    def __init__(self):
        self.heap = []
        self.size = 0

    def add(self, x):
        self.size += 1
        self.heap.append(x)
        i = len(self.heap) - 1
        while i > 0 and self.heap[(i - 1) // 2][0] > self.heap[i][0]:
            self.heap[(i - 1) // 2], self.heap[i] = self.heap[i], self.heap[(i - 1) // 2]
            i = (i - 1) // 2

    def pop(self):
        if not self.heap:
            return None
        self.size -= 1
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        i = 0
        while 2 * i + 1 < len(self.heap):
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest == i:
                break

            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

        return root

    def peek(self):
        return self.heap[0] if self.heap else None

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        self.X = X
        self.leaf_size = leaf_size
    
    def distance(self, x, y):
        return np.linalg.norm(np.array(x) - np.array(y))
    
    def build_tree(self, points, leaf_size, depth=0):
        n = len(points)
        if not n:
            return
        k = len(points[0][1])
        if n // 2 - 1 + n % 2 < leaf_size or n == leaf_size:
            return points
        axis = depth % k
        points_np = np.array([point[1] for point in points])
        sorted_indices = np.argsort(points_np[:, axis])
        points_sorted = [points[i] for i in sorted_indices]
        median_idx = n // 2
        return {
            'point': points_sorted[median_idx],
            'left': self.build_tree(points_sorted[:median_idx], leaf_size, depth=depth + 1),
            'right': self.build_tree(points_sorted[median_idx + 1:], leaf_size, depth=depth + 1)
        }
    
    def kdtree_closest_points(self, root, points, point_index, k, dim, neighbors=None, depth=0):
        if neighbors is None:
            neighbors = Heap()

        if root is None:
            return neighbors

        point = points[point_index][1]

        if 'point' in root:
            axis = depth % dim
            dist = self.distance(root['point'][1], point)
            if neighbors.size < k:
                neighbors.add((-dist, -root['point'][0]))
            elif (-dist, -root['point'][0]) > neighbors.heap[0]:
                neighbors.pop()
                neighbors.add((-dist, -root['point'][0]))

            next_branch = None
            opposite_branch = None
            if point[axis] <= root['point'][1][axis]:
                next_branch = root['left']
                opposite_branch = root['right']
            else:
                next_branch = root['right']
                opposite_branch = root['left']

            self.kdtree_closest_points(next_branch, points, point_index, k, dim, neighbors=neighbors, depth=depth + 1)

            diff = point[axis] - root['point'][1][axis]
            if neighbors.size < k or abs(diff) <= -neighbors.peek()[0]:
                self.kdtree_closest_points(opposite_branch, points, point_index, k, dim, neighbors=neighbors, depth=depth + 1)
        else:
            coord = np.array([i[1] for i in root])
            distances = np.linalg.norm(coord - point, axis=1)
            roots = np.array([(-distances[i], -int(root[i][0])) for i in range(len(distances))])
            root = sorted(roots, key=lambda element: (-element[0], -element[1]))
            for x in root:
                if neighbors.size < k:
                    neighbors.add(tuple(x))
                elif tuple(x) > neighbors.heap[0]:
                    poped = neighbors.pop()
                    neighbors.add(tuple(x))
                else:
                    break

        return neighbors.heap
    
    def query(self, X: np.array, k: int = 1) -> List[List]:
        points_test = [(i, X[i]) for i in range(len(X))]
        points_tr = [(i, self.X[i]) for i in range(len(self.X))]
        if len(X) == 0 or len(self.X) == 0:
            return
        dim = len(X[0])
        answer = []
        tree = self.build_tree(points_tr, self.leaf_size)
        n = len(points_test)
        if k > len(self.X):
            return
        for i in range(n):
            k_neighbors = self.kdtree_closest_points(tree, points_test, i, k, dim)
            k_neighbors.sort(key=lambda x: x[0])
            answer.append([-int(i[1]) for i in k_neighbors][::-1])
        return answer


# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.kdtree = None
        self.tree = None
        self.labels = None  
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        points_tr = [(i, X[i]) for i in range(len(X))]
        self.kdtree = KDTree(X, leaf_size=self.leaf_size)
        self.tree = self.kdtree.build_tree(points_tr, self.leaf_size)
        self.labels = y
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
        dim = len(X[0])
        answer = []
        n = len(X)
        classes = np.unique(self.labels)
        n_classes = len(classes)
        k_neighbors = self.kdtree.query(X, k=self.n_neighbors)
        for i in range(n):
            neighbor_labels = [self.labels[j] for j in k_neighbors[i]]

            class_counts = [0] * n_classes
            for label in neighbor_labels:
                class_counts[label] += 1

            total_neighbors = sum(class_counts)
            answer.append([count / total_neighbors for count in class_counts])
            
        return np.array(answer)
    
    def predict(self, X: np.array):
        probs = self.predict_proba(X)
        
        return np.argmax(probs, axis=1)
