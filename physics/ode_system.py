"""
Система ОДУ и численные методы решения
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

class ODESolver:
    """Решатель системы ОДУ для задачи о колебаниях"""
    
    def __init__(self, params: dict):
        self.params = params
        self.g = params['g']
        
    def ode_system(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Система ОДУ движения груза
        z = [x, vx, y, vy]
        """
        x, vx, y, vy = z
        m, mu1, k1, k2 = self.params['m'], self.params['mu1'], self.params['k1'], self.params['k2']
        
        # Уравнения движения
        dxdt = vx
        dvxdt = -(k2/m) * x - (mu1/m) * vx
        dydt = vy
        dvydt = -(k1/m) * y - (mu1/m) * vy - self.g
        
        return np.array([dxdt, dvxdt, dydt, dvydt])
    
    def get_initial_conditions(self) -> np.ndarray:
        """Вычисление начальных условий"""
        m, k1, gamma, v0 = self.params['m'], self.params['k1'], self.params['gamma'], self.params['v0']
        
        # Установившееся положение
        y_star = -m * self.g / k1
        
        # Начальные условия с учетом коэффициента рассогласования
        x0 = 0
        y0 = gamma * y_star
        vx0 = v0
        vy0 = 0
        
        return np.array([x0, vx0, y0, vy0])
    
    def solve_numeric(self, t_span: Tuple[float, float] = (0, 10), 
                     n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Численное решение методом Рунге-Кутты 4-5 порядка"""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        initial_conditions = self.get_initial_conditions()
        
        solution = solve_ivp(self.ode_system, t_span, initial_conditions,
                           t_eval=t_eval, method='RK45', rtol=1e-8)
        
        return solution.t, solution.y
    
    def calculate_energy(self, t: np.ndarray, solution: np.ndarray) -> dict:
        """Вычисление энергии системы"""
        x, vx, y, vy = solution
        m, k1, k2 = self.params['m'], self.params['k1'], self.params['k2']
        
        kinetic = 0.5 * m * (vx**2 + vy**2)
        potential = 0.5 * k1 * y**2 + 0.5 * k2 * x**2 + m * self.g * y
        
        return {
            'time': t,
            'kinetic': kinetic,
            'potential': potential,
            'total': kinetic + potential
        }