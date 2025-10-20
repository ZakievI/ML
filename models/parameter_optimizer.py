"""
Оптимизация параметров системы с помощью нейросетей
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ParameterOptimizer(nn.Module):
    """Нейросеть для оптимизации параметров системы по траектории"""
    
    def __init__(self, input_dim: int = 400, hidden_dim: int = 128, output_dim: int = 5):
        super(ParameterOptimizer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Прямой проход: траектория -> параметры"""
        return self.encoder(trajectory)

class ParameterEstimator:
    """Оценщик параметров системы"""
    
    def __init__(self):
        self.model = ParameterOptimizer()
        self.scaler_traj = MinMaxScaler()
        self.scaler_params = MinMaxScaler()
        
    def prepare_training_data(self, trajectories: np.ndarray, true_params: np.ndarray) -> tuple:
        """Подготовка данных для обучения"""
        # Нормализация траекторий и параметров
        trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
        
        trajectories_scaled = self.scaler_traj.fit_transform(trajectories_flat)
        params_scaled = self.scaler_params.fit_transform(true_params)
        
        return trajectories_scaled, params_scaled
    
    def train(self, trajectories: np.ndarray, true_params: np.ndarray, 
              epochs: int = 1000, learning_rate: float = 0.001) -> list:
        """Обучение модели"""
        
        # Подготовка данных
        X_scaled, y_scaled = self.prepare_training_data(trajectories, true_params)
        
        # Преобразование в тензоры
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')
        
        return losses
    
    def estimate_parameters(self, trajectory: np.ndarray) -> np.ndarray:
        """Оценка параметров по траектории"""
        self.model.eval()
        
        # Подготовка траектории
        trajectory_flat = trajectory.reshape(1, -1)
        trajectory_scaled = self.scaler_traj.transform(trajectory_flat)
        
        with torch.no_grad():
            params_scaled = self.model(torch.FloatTensor(trajectory_scaled))
            params = self.scaler_params.inverse_transform(params_scaled.numpy())
        
        return params[0]