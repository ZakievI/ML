"""
Функции для визуализации результатов
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class ResultVisualizer:
    """Класс для визуализации результатов"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
    def plot_trajectory_comparison(self, t_numeric: np.ndarray, solution_numeric: np.ndarray,
                                 t_ml: np.ndarray, solution_ml: np.ndarray,
                                 title: str = "Сравнение траекторий"):
        """Сравнение численного решения и ML предсказания"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Траектория
        axes[0, 0].plot(solution_numeric[0], solution_numeric[2], 
                       'b-', label='Численный', linewidth=2)
        axes[0, 0].plot(solution_ml[:, 0], solution_ml[:, 2], 
                       'r--', label='ML', linewidth=2)
        axes[0, 0].set_xlabel('x, м')
        axes[0, 0].set_ylabel('y, м')
        axes[0, 0].set_title('Траектория движения')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Координаты по времени
        for i, coord_name in enumerate(['x', 'y']):
            col = i % 2
            if col == 0:
                row = 1
            else:
                row = 0
            axes[row, col].plot(t_numeric, solution_numeric[i*2], 
                              'b-', label='Численный', linewidth=2)
            axes[row, col].plot(t_ml, solution_ml[:,i*2], 
                              'r--', label='ML', linewidth=2)
            axes[row, col].set_xlabel('Время, с')
            axes[row, col].set_ylabel(f'{coord_name}, м')
            axes[row, col].set_title(f'{coord_name}(t)')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        # Скорости
        axes[1, 1].plot(t_numeric, np.sqrt(solution_numeric[1]**2 + solution_numeric[3]**2),
                       'b-', label='Численный', linewidth=2)
        axes[1, 1].plot(t_ml, np.sqrt(solution_ml[:,1]**2 + solution_ml[:, 3]**2),
                       'r--', label='ML', linewidth=2)
        axes[1, 1].set_xlabel('Время, с')
        axes[1, 1].set_ylabel('|v|, м/с')
        axes[1, 1].set_title('Модуль скорости')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_energy_analysis(self, energy_data: Dict[str, np.ndarray]):
        """Визуализация энергетического анализа"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(energy_data['time'], energy_data['kinetic'], 
               'r-', label='Кинетическая энергия', linewidth=2)
        ax.plot(energy_data['time'], energy_data['potential'], 
               'b-', label='Потенциальная энергия', linewidth=2)
        ax.plot(energy_data['time'], energy_data['total'], 
               'g-', label='Полная энергия', linewidth=2)
        
        ax.set_xlabel('Время, с')
        ax.set_ylabel('Энергия, Дж')
        ax.set_title('Энергия системы во времени')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_sensitivity_analysis(self, sensitivities: Dict[str, float]):
        """Визуализация анализа чувствительности"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        param_names = list(sensitivities.keys())
        values = list(sensitivities.values())
        
        bars = ax.bar(param_names, values, color=self.colors)
        ax.set_xlabel('Параметры')
        ax.set_ylabel('Чувствительность')
        ax.set_title('Анализ чувствительности к параметрам')
        ax.grid(True, axis='y')
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
        
        return fig