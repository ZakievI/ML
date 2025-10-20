"""
Генерация тренировочных данных
"""

import numpy as np
from physics.ode_system import ODESolver
from utils.config import DEFAULT_PARAMS

class DataGenerator:
    """Генератор данных для обучения ML моделей"""
    
    def __init__(self):
        self.base_params = DEFAULT_PARAMS
        
    def generate_parameter_variations(self, n_samples: int) -> list:
        """Генерация вариаций параметров"""
        variations = []
        
        for i in range(n_samples):
            params = self.base_params.copy()
            # Случайные вариации параметров (±30%)
            params['m'] *= np.random.uniform(0.7, 1.3)
            params['mu1'] *= np.random.uniform(0.7, 1.3)
            params['k1'] *= np.random.uniform(0.7, 1.3)
            params['k2'] *= np.random.uniform(0.7, 1.3)
            params['gamma'] *= np.random.uniform(0.7, 1.3)
            variations.append(params)
            
        return variations
    
    def solve_for_parameters(self, params: dict, n_points: int = 50) -> tuple:
        """Решение ОДУ для заданных параметров"""
        solver = ODESolver(params)
        t, solution = solver.solve_numeric(n_points=n_points)
        return t, solution
    
    def generate_dataset(self, n_samples: int = 1000) -> tuple:
        """Генерация полного набора данных"""
        X_data = []
        Y_data = []
        
        param_variations = self.generate_parameter_variations(n_samples)
        
        for i, params in enumerate(param_variations):
            if i % 100 == 0:
                print(f"Генерация данных: {i}/{n_samples}")
                
            t, solution = self.solve_for_parameters(params)
            
            # Создаем признаки: время + параметры
            for j, time_point in enumerate(t):
                features = np.concatenate([[time_point], list(params.values())])
                X_data.append(features)
                Y_data.append(solution[:, j])
        
        return np.array(X_data), np.array(Y_data)
    
    def generate_sequence_data(self, n_sequences: int = 500, seq_length: int = 20):
        """Генерация последовательных данных для LSTM"""
        sequences = []
        targets = []
        
        param_variations = self.generate_parameter_variations(n_sequences)
        
        for params in param_variations:
            t, solution = self.solve_for_parameters(params, n_points=100)
            
            # Создаем последовательности
            for i in range(seq_length, len(solution[0])):
                seq = solution[:, i-seq_length:i].T  # Форма: (seq_length, 4)
                target = solution[:, i]              # Следующее состояние
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)