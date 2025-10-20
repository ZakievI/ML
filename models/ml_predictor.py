"""
Традиционные ML модели для предсказания траекторий
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from data.generate_data import DataGenerator

class MLPredictor(nn.Module):
    """ML модель для предсказания траектории"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, output_dim: int = 4):
        super(MLPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MLPredictorTrainer:
    """Тренер ML модели"""
    
    def __init__(self, input_dim: int = 8):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.loss_history = []
        
    def prepare_data(self, n_samples: int) -> tuple:
        """Подготовка тренировочных данных"""
        generator = DataGenerator()
        X, Y = generator.generate_dataset(n_samples)
        
        # Разделение на train/test
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        # Нормализация
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        Y_train_scaled = self.scaler_y.fit_transform(Y_train)
        Y_test_scaled = self.scaler_y.transform(Y_test)
        
        return (X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled)
    
    def train(self, epochs: int, batch_size: int = 32) -> tuple:
        """Обучение модели"""
        # Подготовка данных
        X_train, X_test, y_train, y_test = self.prepare_data(n_samples=epochs)
        
        # Создание модели
        self.model = MLPredictor()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Преобразование в тензоры
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        losses = []
        
        for epoch in range(epochs):
            # Случайный батч
            indices = torch.randperm(X_train_tensor.size(0))[:batch_size]
            X_batch = X_train_tensor[indices]
            y_batch = y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                # Оценка на тестовых данных
                with torch.no_grad():
                    test_outputs = self.model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor)
                print(f'Epoch {epoch}/{epochs}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')
        
        self.loss_history = losses
        return self.model, losses
    
    def predict(self, t: np.ndarray, params: dict) -> np.ndarray:
        """Предсказание для новых данных"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
            
        # Подготовка входных данных
        n_points = len(t)
        X_pred = np.column_stack([t, np.tile(list(params.values()), (n_points, 1))])
        X_pred_scaled = self.scaler_X.transform(X_pred)
        
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(torch.FloatTensor(X_pred_scaled)).numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred