import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass
    
    
class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(out_features, in_features) * (1 / np.sqrt(in_features + out_features))
        self.b = np.random.randn(out_features) * (1 / np.sqrt(in_features + out_features))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
    
        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.x = x
        self.y = x @ self.W.T + self.b
        return self.y
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.grad_x = d @ self.W
        if self.x.ndim == 1:
            self.grad_W = d[np.newaxis, :].T @ self.x[np.newaxis, :]
        else:
            self.grad_W = d.T @ self.x
        self.grad_b = np.sum(d, axis=0)
        return self.grad_x
        
    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * self.grad_W
        self.b -= alpha * self.grad_b
    

class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x
        self.y = np.maximum(0, x)
        return self.y
        
    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        grad_x = self.x > 0
        return grad_x * d
    

# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        
        def criterion(out, y):
            return - np.sum(y * out, axis=1) + np.log(np.sum(np.exp(out), axis=1))
        n_classes = np.max(y) + 1
        y_train = np.zeros((y.shape[0], n_classes))
        y_train[np.arange(y.shape[0]), y] = 1
        in_features = X.shape[1]
        if self.modules[0].in_features != in_features and self.modules[-1].__class__.__name__ == 'Linear':
            self.modules[0] = Linear(in_features=in_features, out_features=self.modules[0].out_features)
        if self.modules[-1].out_features != n_classes and self.modules[-1].__class__.__name__ == 'Linear':
            self.modules[-1] = Linear(in_features=self.modules[-1].in_features, out_features=n_classes)
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                x_batch = X[i : i + self.batch_size]
                y_batch = y_train[i : i + self.batch_size]
                outp = x_batch
                for i, module in enumerate(self.modules):
                    outp = module.forward(outp)
                loss = criterion(outp, y_batch)
                y_pred = np.exp(outp) / np.sum(np.exp(outp), axis=1)[:, np.newaxis]
                grad = y_pred - y_batch
                for module in reversed(self.modules):
                    grad = module.backward(grad)
                    if module.__class__.__name__ == 'Linear':
                        module.update(self.alpha)
        self.y_pred = y_pred
            
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        outp = X
        for module in self.modules:
            outp = module.forward(outp)
        y_pred = np.exp(outp) / np.sum(np.exp(outp), axis=1)[:, np.newaxis]
        return y_pred
        
    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
    
# Task 3

classifier_moons = MLPClassifier([Linear(0, 10), ReLU(), Linear(10, 6), ReLU(), Linear(6, 6), ReLU(), Linear(6, 0)]) # Нужно указать гиперпараметры
classifier_blobs = MLPClassifier([Linear(0, 5), ReLU(), Linear(5, 4), ReLU(), Linear(4, 0)]) # Нужно указать гиперпараметры


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(in_channels=10, out_channels=6, kernel_size=3, padding=1)
        self.fc = nn.Linear(6 * 32 * 32, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
    
    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] + "model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        model_path = __file__[:-7] + "model.pth"
        self.load_state_dict(torch.load(model_path))
    
    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        model_path = __file__[:-7] + "model.pth"
        torch.save(self.state_dict(), model_path)
        
def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    out = model(X)
    loss = F.cross_entropy(out, y)
    return loss