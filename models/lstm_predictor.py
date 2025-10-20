"""
LSTM модели для предсказания временных рядов
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class LSTM_Predictor(nn.Module):
    """LSTM модель для предсказания динамики системы"""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 4):
        super(LSTM_Predictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)
        
        # Берем только последний выход последовательности
        last_output = lstm_out[:, -1, :]
        
        return self.fc(last_output)

class LSTMTrainer:
    """Тренер для LSTM модели"""
    
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.model = LSTM_Predictor()
        self.loss_history = []
        
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка последовательностей для LSTM"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            seq = data[i-self.sequence_length:i]  # последовательность
            target = data[i]                      # целевое значение
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, sequences: np.ndarray, targets: np.ndarray, 
              epochs: int = 500, batch_size: int = 32) -> list:
        """Обучение LSTM модели"""
        
        # Преобразование в тензоры
        sequences_tensor = torch.FloatTensor(sequences)
        targets_tensor = torch.FloatTensor(targets)
        
        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            # Случайный батч
            indices = torch.randperm(sequences_tensor.size(0))[:batch_size]
            batch_seq = sequences_tensor[indices]
            batch_target = targets_tensor[indices]
            
            optimizer.zero_grad()
            outputs = self.model(batch_seq)
            loss = criterion(outputs, batch_target)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                print(f'LSTM Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')
        
        self.loss_history = losses
        return losses
    
    def predict_sequence(self, initial_sequence: np.ndarray, 
                        steps: int = 50) -> np.ndarray:
        """Предсказание последовательности"""
        self.model.eval()
        
        current_sequence = torch.FloatTensor(initial_sequence).unsqueeze(0)
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps):
                # Предсказываем следующий шаг
                next_step = self.model(current_sequence)
                predictions.append(next_step.numpy()[0])
                
                # Обновляем последовательность
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :], 
                    next_step.unsqueeze(1)
                ], dim=1)
        
        return np.array(predictions)