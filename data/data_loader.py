"""
Загрузка и предобработка данных
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

class DataLoader:
    """Класс для загрузки и управления данными"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, filename: str) -> None:
        """Сохранение набора данных"""
        filepath = self.data_dir / filename
        np.savez(filepath, X=X, y=y)
        print(f"Данные сохранены в {filepath}")
    
    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка набора данных"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Файл {filepath} не найден")
        
        data = np.load(filepath)
        return data['X'], data['y']
    
    def create_dataframe(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: Optional[list] = None,
                        target_names: Optional[list] = None) -> pd.DataFrame:
        """Создание DataFrame из данных"""
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if target_names is None:
            target_names = [f'target_{i}' for i in range(y.shape[1])]
        
        # Объединяем признаки и цели
        data = np.column_stack([X, y])
        columns = feature_names + target_names
        
        return pd.DataFrame(data, columns=columns)
    
    def split_sequences(self, sequences: np.ndarray, targets: np.ndarray, 
                       train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
        """Разделение последовательных данных на train/val/test"""
        
        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        indices = np.random.permutation(n_total)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        return (sequences[train_idx], targets[train_idx],
                sequences[val_idx], targets[val_idx], 
                sequences[test_idx], targets[test_idx])