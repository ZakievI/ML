"""
Модуль машинного обучения - различные модели для решения ОДУ
"""

from .pinn_model import PhysicsInformedNN, ODENet
from .ml_predictor import MLPredictor, MLPredictorTrainer
from .parameter_optimizer import ParameterOptimizer
from .lstm_predictor import LSTM_Predictor, LSTMTrainer

__all__ = [
    'PhysicsInformedNN',
    'ODENet',
    'MLPredictor', 
    'MLPredictorTrainer',
    'ParameterOptimizer',
    'LSTM_Predictor',
    'LSTMTrainer'
]