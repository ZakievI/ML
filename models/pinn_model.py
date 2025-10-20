"""
Physics-Informed Neural Network для решения ОДУ
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

class ODENet(nn.Module):
    """Нейросетевая модель для решения системы ОДУ"""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 4):
        super(ODENet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        # t shape: (batch_size,) -> (batch_size, 1)
        # params shape: (5,) -> (batch_size, 5)
        
        # Проверяем и изменяем размерности
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)
        
        # Повторяем параметры для каждого временного шага
        if params.dim() == 1:
            params = params.unsqueeze(0).repeat(t.shape[0], 1)  # (batch_size, 5)
        
        # Объединяем время и параметры
        input_tensor = torch.cat([t, params], dim=1)  # (batch_size, 6)
        
        return self.network(input_tensor)

class PhysicsInformedNN:
    """Физически информированная нейронная сеть"""
    
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.001):
        self.model = ODENet(hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_history = []
        
    def physics_constraints(self, t: torch.Tensor, predictions: torch.Tensor, 
                          params: torch.Tensor) -> torch.Tensor:
        """Физические ограничения - удовлетворение ОДУ"""
        m, mu1, k1, k2, gamma = params
        
        # Извлекаем переменные
        x, vx, y, vy = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        
        # Вычисляем производные через autograd

        # x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), 
        #                         create_graph=True)[0]
        # y_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), 
        #                         create_graph=True)[0]
        
        vx_t = torch.autograd.grad(vx, t, grad_outputs=torch.ones_like(vx), 
                                 create_graph=True)[0]
        vy_t = torch.autograd.grad(vy, t, grad_outputs=torch.ones_like(vy), 
                                 create_graph=True)[0]
        
        # Уравнения ОДУ
        g = 9.81
        eq1 = vx_t + (k2/m)*x + (mu1/m)*vx
        eq2 = vy_t + (k1/m)*y + (mu1/m)*vy + g
        
        # Начальные условия
        y_star = -m * g / k1
        ic_loss = (x[0]**2 + (vx[0] - 0.9)**2 + 
                  (y[0] - gamma*y_star)**2 + vy[0]**2)
        
        return torch.mean(eq1**2 + eq2**2) + 0.1 * ic_loss
    
    def train(self, params: dict, epochs: int = 1000, t_span: tuple = (0, 10)) -> List[float]:
        """Обучение модели"""
        # Преобразование параметров в тензор
        param_tensor = torch.tensor([
            params['m'], params['mu1'], params['k1'], params['k2'], params['gamma']
        ], dtype=torch.float32)
        
        print(f"Параметры модели: {param_tensor}")
        print(f"Размерность параметров: {param_tensor.shape}")
        
        # Временная сетка
        t = torch.linspace(t_span[0], t_span[1], 100, requires_grad=True)
        print(f"Размерность времени: {t.shape}")
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            try:
                # Предсказание
                predictions = self.model(t, param_tensor)
                # print(f"Размерность предсказаний: {predictions.shape}")
                
                # Физическая потеря
                loss = self.physics_constraints(t, predictions, param_tensor)
                
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')
                    
            except Exception as e:
                print(f"Ошибка на эпохе {epoch}: {e}")
                print(f"t shape: {t.shape}")
                print(f"params shape: {param_tensor.shape}")
                break
        
        self.loss_history = losses
        return losses
    
    def predict(self, t: np.ndarray, params: dict) -> np.ndarray:
        """Предсказание траектории"""
        self.model.eval()
        
        # Преобразование параметров
        param_tensor = torch.tensor([
            params['m'], params['mu1'], params['k1'], params['k2'], params['gamma']
        ], dtype=torch.float32)
        
        t_tensor = torch.FloatTensor(t)
        
        with torch.no_grad():
            predictions = self.model(t_tensor, param_tensor)
        
        return predictions.numpy()