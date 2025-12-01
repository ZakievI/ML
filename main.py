"""
Основной файл проекта: Свободные колебания груза с ML подходом
"""

import argparse
# import torch
# import numpy as np
from models.pinn_model import PhysicsInformedNN
from models.ml_predictor import MLPredictorTrainer
# from physics.ode_system import ODESolver
# from utils.visualization import ResultVisualizer
from utils.config import DEFAULT_PARAMS

def main():
    parser = argparse.ArgumentParser(description='ML решение задачи о колебаниях груза')
    parser.add_argument('--mode', type=str, default='predict', 
                       choices=['train', 'predict', 'compare', 'analyze'],
                       help='Режим работы')
    parser.add_argument('--model', type=str, default='ml',
                       choices=['pinn', 'ml', 'lstm', 'all'])
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--visualize', action='store_true', default=True)
    
    args = parser.parse_args()
    
    print("=== ML решение задачи о колебаниях груза с боковыми амортизаторами ===")
    print(f"Режим: {args.mode}, Модель: {args.model}")
    
    # Параметры из задачи
    params = DEFAULT_PARAMS
    
    if args.mode == 'train':
        train_models(params, args)
    elif args.mode == 'predict':
        make_predictions(params, args)
    elif args.mode == 'compare':
        compare_methods(params, args)
    elif args.mode == 'analyze':
        analyze_system(params)

def train_models(params, args):
    """Обучение различных моделей"""
    
    if args.model in ['pinn', 'all']:
        print("\n1. Обучение Physics-Informed Neural Network...")
        pinn_trainer = PhysicsInformedNN()
        pinn_losses = pinn_trainer.train(params, epochs=args.epochs)
        
        if args.visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(pinn_losses)
            plt.title('Обучение PINN')
            plt.xlabel('Эпоха')
            plt.ylabel('Потеря')
            plt.grid(True)
            plt.show()
    
    if args.model in ['ml', 'all']:
        print("\n2. Обучение ML предсказателя...")
        ml_trainer = MLPredictorTrainer()
        ml_model, losses = ml_trainer.train(epochs=args.epochs)
        
        if args.visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(losses)
            plt.title('Обучение ML модели')
            plt.xlabel('Эпоха')
            plt.ylabel('Потеря')
            plt.grid(True)
            plt.show()

def make_predictions(params, args):
    """Создание предсказаний"""
    from examples.basic_example import run_basic_example
    run_basic_example(params, args)

def compare_methods(params, args):
    """Сравнение различных методов"""
    from examples.comparison_example import run_comparison
    run_comparison(params, args)

def analyze_system(params):
    """Анализ системы"""
    from examples.sensitivity_analysis import run_sensitivity_analysis
    run_sensitivity_analysis(params)

if __name__ == "__main__":
    main()