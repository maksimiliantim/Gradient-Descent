from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.

    Parameters
    ----------
    lambda_ : float, optional
        Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional
        Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional
        Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional
        Текущая итерация. По умолчанию 0.

    Methods
    -------
    __call__()
        Вычисляет скорость обучения на текущей итерации.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Вычисляет скорость обучения по формуле lambda * (s0 / (s0 + t))^p.

        Returns
        -------
        float
            Скорость обучения на текущем шаге.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.

    Attributes
    ----------
    MSE : auto
        Среднеквадратическая ошибка.
    MAE : auto
        Средняя абсолютная ошибка.
    LogCosh : auto
        Логарифм гиперболического косинуса от ошибки.
    Huber : auto
        Функция потерь Хьюбера.
    """
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    Базовый класс для всех методов градиентного спуска.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional
        Функция потерь, которая будет оптимизироваться. По умолчанию MSE.

    Attributes
    ----------
    w : np.ndarray
        Вектор весов модели.
    lr : LearningRate
        Скорость обучения.
    loss_function : LossFunction
        Функция потерь.

    Methods
    -------
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray
        Вычисление прогнозов на основе признаков x.
    """


    def fit(self, x: np.ndarray, y: np.ndarray, max_iter: int = 10, tolerance: float = 1e-4):
        """
        Обучение модели с использованием градиентного спуска.
        
        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.
        max_iter : int, optional
            Максимальное количество итераций, по умолчанию 100.
        tolerance : float, optional
            Порог для сходимости, по умолчанию 1e-4.
        """
        self.loss_history = []
        for iteration in range(max_iter):
            gradient = self.calc_gradient(x, y)
            self.update_weights(gradient)
            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)
            print(f"Iteration {iteration}: Loss = {loss:.6f}, ||grad|| = {np.linalg.norm(gradient):.4f}")
            if loss > 1e5:
                print(f"STOP: loss exploded at iteration {iteration}")
                break
            if loss < tolerance:
                print(f"сходимость достигнута на {iteration+1} итерации")
                break


    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация базового класса для градиентного спуска.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Функция потерь, которая будет оптимизирована.

        Attributes
        ----------
        w : np.ndarray
            Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate
            Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction
            Выбранная функция потерь.
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_, s0=1, p=0.5)
        self.loss_function: LossFunction = loss_function
        self.loss_history: list = []

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами.
        """

        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для вычисления градиента функции потерь по весам. Должен быть переопределен в подклассах.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с использованием текущих весов.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        y_pred = self.predict(x)
        if self.loss_function == LossFunction.MSE:
            return np.mean((y_pred - y) ** 2)
        elif self.loss_function == LossFunction.MAE:
            return np.mean(np.abs(y_pred - y))
        elif self.loss_function == LossFunction.LogCosh:
            error = y_pred - y
            error = np.clip(error, -20, 20)
            return np.mean(np.log(np.cosh(error)))
        elif self.loss_function == LossFunction.Huber:
            delta = 1.0
            error = y_pred - y
            is_small_error = np.abs(error) <= delta
            squared_loss = 0.5 * error ** 2
            linear_loss = delta * (np.abs(error) - 0.5 * delta)
            return np.mean(np.where(is_small_error, squared_loss, linear_loss))
        else:
            raise ValueError("ne pon")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
        x = np.array(x, dtype=float)
        self.w = np.array(self.w, dtype=float)
        if x.shape[1] != self.w.shape[0]:
            raise ValueError(f"размерности x: {x.shape[1]} и w: {self.w.shape[0]} несовместимы.")
        return np.dot(x, self.w)


class VanillaGradientDescent(BaseDescent):
    """
    Класс полного градиентного спуска.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам.
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.w -= self.lr() * gradient
        return self.w

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.
        """
        y_pred = self.predict(x)
        error = y_pred - y
        if self.loss_function == LossFunction.MSE:
            return -2 * np.dot(x.T, (y - y_pred)) / len(y)
        elif self.loss_function == LossFunction.LogCosh:
            error = np.clip(error, -20, 20)
            return np.dot(x.T, np.tanh(error)) / len(y)
        else:
            raise ValueError(f"неподдерживаемая функция потерь: {self.loss_function}")
                


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска.

    Parameters
    ----------
    batch_size : int, optional
        Размер мини-пакета. По умолчанию 50.

    Attributes
    ----------
    batch_size : int
        Размер мини-пакета.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по мини-пакетам.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):

        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по мини-пакетам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам, вычисленный по мини-пакету.
        """
        indices = np.random.randint(0, len(y), self.batch_size)
        x_batch = x[indices]
        y_batch = y[indices]
        y_pred = self.predict(x_batch)
        return -2 * np.dot(x_batch.T, (y_batch - y_pred)) / len(y_batch)


class MomentumDescent(VanillaGradientDescent):
    """
    Класс градиентного спуска с моментом.

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    alpha : float
        Коэффициент момента.
    h : np.ndarray
        Вектор момента для весов.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием момента.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса градиентного спуска с моментом.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием момента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.h = self.alpha * self.h + (1 - self.alpha) * gradient
        self.w -= self.lr() * self.h
        return self.w


class Adam(VanillaGradientDescent):
    """
    Класс градиентного спуска с адаптивной оценкой моментов (Adam).

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    eps : float
        Малая добавка для предотвращения деления на ноль.
    m : np.ndarray
        Векторы первого момента.
    v : np.ndarray
        Векторы второго момента.
    beta_1 : float
        Коэффициент распада для первого момента.
    beta_2 : float
        Коэффициент распада для второго момента.
    iteration : int
        Счетчик итераций.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием адаптивной оценки моментов.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса Adam.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.iteration += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)
        self.w -= self.lr() * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.w



class BaseDescentReg(BaseDescent):
    """
    Базовый класс для градиентного спуска с регуляризацией.

    Параметры
    ----------
    *args : tuple
        Аргументы, передаваемые в базовый класс.
    mu : float, optional
        Коэффициент регуляризации. По умолчанию равен 0.
    **kwargs : dict
        Ключевые аргументы, передаваемые в базовый класс.

    Атрибуты
    ----------
    mu : float
        Коэффициент регуляризации.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь с учетом L2 регуляризации по весам.
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        Инициализация базового класса для градиентного спуска с регуляризацией.
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации по весам.
        """
        grad = super().calc_gradient(x, y)
        l2_gradient = 2 * self.w
        return grad + self.mu * l2_gradient


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    """
    def fit(self, x: np.ndarray, y: np.ndarray, max_iter: int = 30, tolerance: float = 1e-4):
        """
        Обучение модели с регуляризацией с использованием градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.
        max_iter : int, optional
            Максимальное количество итераций для обучения. По умолчанию 1000.
        tolerance : float, optional
            Порог сходимости для ошибки. По умолчанию 1e-6.

        """
        for iteration in range(max_iter):
            prev_w = np.copy(self.w)  
            self.step(x, y)
            if np.linalg.norm(self.w - prev_w) < tolerance:
                print(f"Конвергенция получилась на {iteration + 1} итерации")
                break


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    """


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1}
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'надо это: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))