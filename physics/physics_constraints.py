"""
Физические ограничения и проверки
"""

import torch
import numpy as np
from typing import Dict

class PhysicsConstraints:
    """Класс для применения физических ограничений"""
    
    def __init__(self):
        self.g = 9.81
    
    def check_energy_conservation(self, solution: np.ndarray, params: Dict, 
                                tolerance: float = 1e-2) -> bool:
        """Проверка сохранения энергии (с учетом диссипации)"""
        m, mu1 = params['m'], params['mu1']
        x, vx, y, vy = solution
        
        # Энергия должна уменьшаться из-за демпфирования
        initial_energy = self.calculate_total_energy(solution[:, 0], params)
        final_energy = self.calculate_total_energy(solution[:, -1], params)
        
        energy_loss = initial_energy - final_energy
        return energy_loss >= -tolerance  # Допускаем небольшую погрешность
    
    def calculate_total_energy(self, state: np.ndarray, params: Dict) -> float:
        """Вычисление полной энергии системы"""
        m, k1, k2 = params['m'], params['k1'], params['k2']
        x, vx, y, vy = state
        
        kinetic = 0.5 * m * (vx**2 + vy**2)
        potential = 0.5 * k1 * y**2 + 0.5 * k2 * x**2 + m * self.g * y
        
        return kinetic + potential
    
    def apply_bounds(self, solution: np.ndarray, bounds: Dict) -> np.ndarray:
        """Применение граничных условий"""
        x, vx, y, vy = solution
        
        # Ограничение координат
        if 'x_bounds' in bounds:
            x_min, x_max = bounds['x_bounds']
            x = np.clip(x, x_min, x_max)
        
        if 'y_bounds' in bounds:
            y_min, y_max = bounds['y_bounds']
            y = np.clip(y, y_min, y_max)
        
        return np.array([x, vx, y, vy])
    
    def verify_initial_conditions(self, initial_state: np.ndarray, 
                                expected_state: np.ndarray, 
                                tolerance: float = 1e-6) -> bool:
        """Проверка начальных условий"""
        return np.allclose(initial_state, expected_state, atol=tolerance)
    
    def get_physical_bounds(self, params: Dict) -> Dict:
        """Получение физически обоснованных границ"""
        m, k1, k2 = params['m'], params['k1'], params['k2']
        
        # Максимальное отклонение основано на энергии
        max_potential = 0.5 * m * params['v0']**2  # Начальная кинетическая энергия
        max_x = np.sqrt(2 * max_potential / k2) if k2 > 0 else 10.0
        max_y = np.sqrt(2 * max_potential / k1) if k1 > 0 else 10.0
        
        return {
            'x_bounds': (-max_x * 1.5, max_x * 1.5),
            'y_bounds': (-max_y * 1.5, max_y * 1.5),
            'v_bounds': (-params['v0'] * 2, params['v0'] * 2)
        }