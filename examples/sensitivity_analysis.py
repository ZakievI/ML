"""
Анализ чувствительности к параметрам системы
"""

import numpy as np
import matplotlib.pyplot as plt
from physics.ode_system import ODESolver
from utils.metrics import MetricsCalculator
from utils.visualization import ResultVisualizer
from utils.config import DEFAULT_PARAMS

def run_sensitivity_analysis(params: dict = None):
    """Анализ чувствительности системы к параметрам"""
    if params is None:
        params = DEFAULT_PARAMS
    
    print("=== Анализ чувствительности системы ===")
    
    # Базовое решение
    base_solver = ODESolver(params)
    t_base, solution_base = base_solver.solve_numeric()
    final_base = solution_base[:, -1]
    
    # Параметры для анализа
    param_names = ['m', 'mu1', 'k1', 'k2', 'gamma']
    param_variations = [0.8, 0.9, 1.0, 1.1, 1.2]  # ±20%
    
    sensitivities = {}
    
    for param_name in param_names:
        print(f"\nАнализ чувствительности к параметру: {param_name}")
        variations_results = []
        
        for variation in param_variations:
            # Модифицируем параметр
            modified_params = params.copy()
            modified_params[param_name] = params[param_name] * variation
            
            # Решаем ОДУ
            solver = ODESolver(modified_params)
            t_modified, solution_modified = solver.solve_numeric()
            final_modified = solution_modified[:, -1]
            
            # Вычисляем изменение
            delta = np.linalg.norm(final_modified - final_base)
            variations_results.append(delta)
            
            print(f"  {variation:4.1f} × {param_name}: изменение = {delta:.6f}")
        
        sensitivities[param_name] = {
            'variations': param_variations,
            'changes': variations_results,
            'sensitivity': max(variations_results) - min(variations_results)
        }
    
    # Визуализация
    visualizer = ResultVisualizer()
    
    # Графики чувствительности для каждого параметра
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    sensitivity_scores = {}
    
    for i, param_name in enumerate(param_names):
        if i < len(axes):
            data = sensitivities[param_name]
            axes[i].plot(data['variations'], data['changes'], 'o-', linewidth=2, markersize=8)
            axes[i].set_xlabel(f'Множитель {param_name}')
            axes[i].set_ylabel('Изменение конечного положения')
            axes[i].set_title(f'Чувствительность к {param_name}')
            axes[i].grid(True)
            
            sensitivity_scores[param_name] = data['sensitivity']
    
    # Убираем лишние subplots
    for i in range(len(param_names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Сводная таблица чувствительности
    print("\n=== Сводная таблица чувствительности ===")
    print("Параметр    | Чувствительность")
    print("-" * 30)
    
    for param_name, score in sorted(sensitivity_scores.items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"{param_name:10} | {score:.6f}")
    
    # Визуализация сводной чувствительности
    fig2 = plt.figure(figsize=(10, 6))
    param_names_sorted = [x[0] for x in sorted(sensitivity_scores.items(), 
                                             key=lambda x: x[1], reverse=True)]
    scores_sorted = [sensitivity_scores[name] for name in param_names_sorted]
    
    bars = plt.bar(param_names_sorted, scores_sorted, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(param_names_sorted))))
    
    plt.xlabel('Параметры')
    plt.ylabel('Чувствительность')
    plt.title('Относительная чувствительность системы к параметрам')
    plt.grid(True, axis='y')
    
    # Добавляем значения на столбцы
    for bar, score in zip(bars, scores_sorted):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return sensitivity_scores

if __name__ == "__main__":
    sensitivities = run_sensitivity_analysis()