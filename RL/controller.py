import numpy as np


class Controller:
    """
    Базовый класс контроллера для управления ускорением системы.
    Все контроллеры должны наследоваться от этого класса и реализовывать метод compute().
    """
    
    def __init__(self, name='controller'):
        """
        Параметры:
        -----------
        name : str
            Имя контроллера
        """
        self.name = name
    
    def compute(self, t, x):
        """
        Вычислить ускорение на основе времени и состояния.
        
        Параметры:
        -----------
        t : float
            Текущее время
        x : np.array
            Текущее состояние [позиция, скорость]
        
        Возвращает:
        -----------
        float : ускорение
        """
        raise NotImplementedError("Метод compute() должен быть реализован в подклассе")
    
    def reset(self):
        """Сбросить внутреннее состояние контроллера (если есть)"""
        pass


class ConstantController(Controller):
    """Контроллер с постоянным ускорением"""
    
    def __init__(self, acceleration=0.0, name='constant_controller'):
        """
        Параметры:
        -----------
        acceleration : float
            Постоянное значение ускорения
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.acceleration = acceleration
    
    def compute(self, t, x):
        return self.acceleration


class TimeBasedController(Controller):
    """Контроллер, ускорение которого зависит только от времени"""
    
    def __init__(self, func, name='time_based_controller'):
        """
        Параметры:
        -----------
        func : callable
            Функция от времени: func(t) -> acceleration
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.func = func
    
    def compute(self, t, x):
        return self.func(t)


class PDController(Controller):
    """
    PD-контроллер (Proportional-Derivative) для стабилизации системы.
    Ускорение = -kp * position - kd * velocity
    """
    
    def __init__(self, kp=1.0, kd=0.5, target=0.0, name='pd_controller'):
        """
        Параметры:
        -----------
        kp : float
            Коэффициент пропорциональной составляющей
        kd : float
            Коэффициент дифференциальной составляющей
        target : float
            Целевая позиция
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.kp = kp
        self.kd = kd
        self.target = target
    
    def compute(self, t, x):
        position, velocity = x[0], x[1]
        error = position - self.target
        return -self.kp * error - self.kd * velocity


class PIDController(Controller):
    """
    PID-контроллер (Proportional-Integral-Derivative) для стабилизации системы.
    """
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.5, target=0.0, name='pid_controller'):
        """
        Параметры:
        -----------
        kp : float
            Коэффициент пропорциональной составляющей
        ki : float
            Коэффициент интегральной составляющей
        kd : float
            Коэффициент дифференциальной составляющей
        target : float
            Целевая позиция
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.last_error = 0.0
    
    def compute(self, t, x):
        position, velocity = x[0], x[1]
        error = position - self.target
        
        # Интегральная составляющая
        self.integral += error
        
        # Пропорциональная, интегральная и дифференциальная составляющие
        p_term = -self.kp * error
        i_term = -self.ki * self.integral
        d_term = -self.kd * velocity
        
        return p_term + i_term + d_term
    
    def reset(self):
        """Сбросить интегральную составляющую"""
        self.integral = 0.0
        self.last_error = 0.0


class StateBasedController(Controller):
    """Контроллер, ускорение которого зависит от состояния системы"""
    
    def __init__(self, func, name='state_based_controller'):
        """
        Параметры:
        -----------
        func : callable
            Функция от состояния: func(x) -> acceleration
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.func = func
    
    def compute(self, t, x):
        return self.func(x)


class FullStateController(Controller):
    """Контроллер, ускорение которого зависит от времени и состояния"""
    
    def __init__(self, func, name='full_state_controller'):
        """
        Параметры:
        -----------
        func : callable
            Функция от времени и состояния: func(t, x) -> acceleration
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.func = func
    
    def compute(self, t, x):
        return self.func(t, x)


class CompositeController(Controller):
    """Композитный контроллер, объединяющий несколько контроллеров"""
    
    def __init__(self, controllers, name='composite_controller'):
        """
        Параметры:
        -----------
        controllers : list of Controller
            Список контроллеров, ускорения которых суммируются
        name : str
            Имя контроллера
        """
        super().__init__(name)
        self.controllers = controllers
    
    def compute(self, t, x):
        total_acceleration = 0.0
        for controller in self.controllers:
            total_acceleration += controller.compute(t, x)
        return total_acceleration
    
    def reset(self):
        """Сбросить все контроллеры"""
        for controller in self.controllers:
            controller.reset()







