"""
Базовый пример использования библиотеки
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.ode_system import ODESolver
from models.pinn_model import PhysicsInformedNN
from models.ml_predictor import MLPredictorTrainer
from utils.visualization import ResultVisualizer
from utils.config import DEFAULT_PARAMS

def run_basic_example(params: dict = None):
    """Запуск базового примера"""
    if params is None:
        params = DEFAULT_PARAMS
    
    print("=== Базовый пример: Свободные колебания груза ===")
    print("Параметры системы:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # 1. Численное решение
    print("\n1. Численное решение ОДУ...")
    solver = ODESolver(params)
    t_numeric, solution_numeric = solver.solve_numeric()
    
    # 2. Решение с помощью PINN
    print("2. Решение с помощью Physics-Informed NN...")
    pinn = PhysicsInformedNN()
    pinn_losses = pinn.train(params, epochs=500)
    solution_pinn = pinn.predict(t_numeric, params)
    
    # 3. Решение с помощью ML
    print("3. Решение с помощью ML модели...")
    ml_trainer = MLPredictorTrainer()
    ml_model, ml_losses = ml_trainer.train(epochs=1000)
    solution_ml = ml_trainer.predict(t_numeric, params)
    
    # 4. Визуализация
    print("4. Визуализация результатов...")
    visualizer = ResultVisualizer()
    
    # Сравнение методов
    fig = visualizer.plot_trajectory_comparison(
        t_numeric, solution_numeric, t_numeric, solution_ml,
        "Сравнение методов решения"
    )
    
    # Энергетический анализ
    energy_data = solver.calculate_energy(t_numeric, solution_numeric)
    energy_fig = visualizer.plot_energy_analysis(energy_data)
    
    # Вывод результатов
    print("\nРезультаты:")
    final_pos_numeric = solution_numeric[:, -1]
    final_pos_ml = solution_ml[-1]
    
    print(f"Численное решение - конечное положение: x={final_pos_numeric[0]:.4f} м, y={final_pos_numeric[2]:.4f} м")
    print(f"ML решение - конечное положение: x={final_pos_ml[0]:.4f} м, y={final_pos_ml[2]:.4f} м")
    
    error = np.linalg.norm(final_pos_numeric - final_pos_ml)
    print(f"Ошибка предсказания: {error:.6f}")
    
    plt.show()

if __name__ == "__main__":
    run_basic_example()