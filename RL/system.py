import numpy as np

class System:
    def __init__(self, name='simple_system', n_dim=2, dt=0.01, x0=np.array([0, 0]), 
                 controller=None):
        """
        Параметры:
        -----------
        name : str
            Имя системы
        n_dim : int
            Размерность состояния
        dt : float
            Шаг времени
        x0 : np.array
            Начальное состояние [позиция, скорость]
        controller : Controller или None
            Контроллер для управления ускорением. Если None, ускорение = 0
        """
        self.name = name
        self.n_dim = n_dim
        self.dt = dt
        self.x0 = x0.copy()
        self.x = x0.copy()
        self.history = [x0.copy()]
        self.time = [0.0]
        self.controller = controller

    def x_dot(self, x, a):
        """Производная состояния: [скорость, ускорение]"""
        return np.array([x[1], a])

    def get_acceleration(self):
        """
        Получить ускорение для текущего шага от контроллера.
        
        Возвращает:
        -----------
        float : ускорение
        """
        if self.controller is not None:
            return self.controller.compute(self.time[-1], self.x)
        else:
            return 0.0

    def step(self, x, a):
        """Один шаг метода Рунге-Кутта 4-го порядка"""
        k1 = self.x_dot(x, a)
        k2 = self.x_dot(x + self.dt * k1 / 2, a)
        k3 = self.x_dot(x + self.dt * k2 / 2, a)
        k4 = self.x_dot(x + self.dt * k3, a)
        return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def update(self):
        """
        Обновить состояние системы на один шаг.
        Ускорение вычисляется контроллером (если задан).
        """
        acceleration = self.get_acceleration()
        self.x = self.step(self.x, acceleration)
        self.history.append(self.x.copy())
        self.time.append(self.time[-1] + self.dt)

    def get_history(self):
        """Получить историю состояний"""
        return np.array(self.history)
    
    def get_time_history(self):
        """Получить историю времени"""
        return np.array(self.time)
    
    def set_controller(self, controller):
        """
        Установить контроллер для управления ускорением.
        
        Параметры:
        -----------
        controller : Controller
            Объект контроллера
        """
        self.controller = controller
    
    def reset(self, x0=None):
        """Сбросить систему в начальное состояние"""
        if x0 is not None:
            self.x = x0.copy()
            self.history = [x0.copy()]
        else:
            self.x = self.history[0].copy()
            self.history = [self.history[0].copy()]
        self.time = [0.0]
        # Сбросить контроллер, если он есть
        if self.controller is not None:
            self.controller.reset()

