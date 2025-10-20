"""
Метрики для оценки качества моделей
"""

import numpy as np
from typing import Dict, Tuple

class MetricsCalculator:
    """Калькулятор метрик для оценки моделей"""
    
    @staticmethod
    def calculate_mse(true: np.ndarray, pred: np.ndarray) -> float:
        """Среднеквадратичная ошибка"""
        return np.mean((true - pred) ** 2)
    
    @staticmethod
    def calculate_mae(true: np.ndarray, pred: np.ndarray) -> float:
        """Средняя абсолютная ошибка"""
        return np.mean(np.abs(true - pred))
    
    @staticmethod
    def calculate_rmse(true: np.ndarray, pred: np.ndarray) -> float:
        """Среднеквадратичная ошибка (корень)"""
        return np.sqrt(MetricsCalculator.calculate_mse(true, pred))
    
    @staticmethod
    def calculate_relative_error(true: np.ndarray, pred: np.ndarray) -> float:
        """Относительная ошибка"""
        return np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8)))
    
    @staticmethod
    def calculate_trajectory_error(true_traj: np.ndarray, pred_traj: np.ndarray) -> Dict:
        """Ошибки для траектории"""
        errors = {}
        
        # Ошибки по координатам
        for i, coord in enumerate(['x', 'vx', 'y', 'vy']):
            errors[f'mse_{coord}'] = MetricsCalculator.calculate_mse(true_traj[i], pred_traj[i])
            errors[f'mae_{coord}'] = MetricsCalculator.calculate_mae(true_traj[i], pred_traj[i])
            errors[f'rmse_{coord}'] = MetricsCalculator.calculate_rmse(true_traj[i], pred_traj[i])
        
        # Общая ошибка траектории
        errors['trajectory_mse'] = MetricsCalculator.calculate_mse(true_traj, pred_traj)
        errors['trajectory_mae'] = MetricsCalculator.calculate_mae(true_traj, pred_traj)
        
        return errors
    
    @staticmethod
    def calculate_energy_error(true_energy: Dict, pred_energy: Dict) -> Dict:
        """Ошибки энергии"""
        errors = {}
        
        for energy_type in ['kinetic', 'potential', 'total']:
            true = true_energy[energy_type]
            pred = pred_energy[energy_type]
            
            errors[f'{energy_type}_mse'] = MetricsCalculator.calculate_mse(true, pred)
            errors[f'{energy_type}_mae'] = MetricsCalculator.calculate_mae(true, pred)
            errors[f'{energy_type}_relative'] = MetricsCalculator.calculate_relative_error(true, pred)
        
        return errors
    
    @staticmethod
    def print_metrics(metrics: Dict, title: str = "Метрики качества"):
        """Красивый вывод метрик"""
        print(f"\n=== {title} ===")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.6f}")
            else:
                print(f"{key:20}: {value}")