"""
Пример сравнения различных методов
"""

import numpy as np
import matplotlib.pyplot as plt
from physics.ode_system import ODESolver
from models.pinn_model import PhysicsInformedNN
from models.ml_predictor import MLPredictorTrainer
from utils.metrics import MetricsCalculator
from utils.visualization import ResultVisualizer
from utils.config import DEFAULT_PARAMS

def run_comparison(params: dict = None, args = None):
    """Сравнение различных методов решения"""
    if params is None:
        params = DEFAULT_PARAMS
    print("=== Сравнение методов решения ОДУ ===")
    
    # Создание решателей
    ode_solver = ODESolver(params)
    pinn = PhysicsInformedNN()
    ml_trainer = MLPredictorTrainer()
    
    # Обучение моделей
    print("Обучение PINN...")
    pinn_losses = pinn.train(params, epochs=args.epochs)
    
    print("Обучение ML модели...")
    ml_model, ml_losses = ml_trainer.train(epochs=args.epochs)
    
    # Численное решение (эталон)
    print("Численное решение...")
    t_numeric, solution_numeric = ode_solver.solve_numeric(n_points=100)
    
    # Предсказания моделей
    print("Предсказание PINN...")
    solution_pinn = pinn.predict(t_numeric, params)
    
    print("Предсказание ML...")
    solution_ml = ml_trainer.predict(t_numeric, params)
    
    # Транспонирование для единого формата
    solution_pinn = solution_pinn.T
    solution_ml = solution_ml.T
    
    # Вычисление метрик
    metrics_pinn = MetricsCalculator.calculate_trajectory_error(solution_numeric, solution_pinn)
    metrics_ml = MetricsCalculator.calculate_trajectory_error(solution_numeric, solution_ml)
    
    print("\nМетрики качества:")
    print("PINN:")
    MetricsCalculator.print_metrics(metrics_pinn, "PINN Метрики")
    
    print("\nML модель:")
    MetricsCalculator.print_metrics(metrics_ml, "ML Метрики")
    
    # Визуализация сравнения
    visualizer = ResultVisualizer()
    
    # Сравнение траекторий
    fig1 = visualizer.plot_trajectory_comparison(
        t_numeric, solution_numeric, t_numeric, solution_ml,
        "Сравнение численного решения и ML"
    )
    
    # Графики потерь при обучении
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(pinn_losses)
    ax1.set_title('Обучение PINN')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потеря')
    ax1.grid(True)
    
    ax2.plot(ml_losses)
    ax2.set_title('Обучение ML модели')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потеря')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Сравнение конечных положений
    final_numeric = solution_numeric[:, -1]
    final_pinn = solution_pinn[:, -1]
    final_ml = solution_ml[:, -1]
    
    print("\nКонечные положения:")
    print(f"Численный: x={final_numeric[0]:.4f}, y={final_numeric[2]:.4f}")
    print(f"PINN:      x={final_pinn[0]:.4f}, y={final_pinn[2]:.4f}")
    print(f"ML:        x={final_ml[0]:.4f}, y={final_ml[2]:.4f}")

if __name__ == "__main__":
    run_comparison()