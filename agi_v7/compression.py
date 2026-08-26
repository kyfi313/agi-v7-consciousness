# -*- coding: utf-8 -*-
"""
МОДУЛЬ СЖАТИЯ ДАННЫХ
Реализует автоэнкодероподобное сжатие сенсорных данных.
Уменьшает размерность, сохраняя важные паттерны.
Поддерживает: сжатие изображений, звука, проприоцепции.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


class CompressionModule:
    """
    Сжимает сенсорные данные в компактное латентное представление.
    Использует PCA-подобное преобразование (адаптивное).
    """
    
    def __init__(self, latent_dim: int = 32, history_len: int = 100):
        self.latent_dim = latent_dim
        self.history_len = history_len
        
        # Адаптивные статистики
        self.mean = None
        self.std = None
        self.n_samples = 0
        
        # Буфер для накопления данных
        self.data_buffer = deque(maxlen=history_len)
        
        # Проекционная матрица (обучается адаптивно)
        self.projection = None
        self.is_trained = False
        
    def compress(self, data: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Сжимает входной вектор в латентное представление.
        
        Args:
            data: Входной вектор (1D или 2D)
            train: Обновлять статистики и проекцию
        
        Returns:
            Сжатый вектор размерности latent_dim
        """
        # Преобразуем в 1D
        if len(data.shape) > 1:
            data_flat = data.flatten()
        else:
            data_flat = data.copy()
        
        # Нормализация
        data_norm = self._normalize(data_flat)
        
        # Если проекция не обучена, используем случайную
        if not self.is_trained:
            self._initialize_projection(data_norm.shape[0])
        
        # Применяем проекцию
        latent = np.dot(data_norm, self.projection)
        
        # Активация (ReLU)
        latent = np.maximum(0, latent)
        
        # Ограничение размера
        if len(latent) > self.latent_dim:
            latent = latent[:self.latent_dim]
        elif len(latent) < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - len(latent)))
        
        # Обучение на новых данных
        if train:
            self._update_statistics(data_flat)
            self.data_buffer.append(data_flat)
            if len(self.data_buffer) >= 10:
                self._train_projection()
        
        return latent.astype(np.float32)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Нормализует данные к нулевому среднему и единичной дисперсии."""
        if self.mean is not None and self.std is not None:
            return (data - self.mean[:len(data)]) / (self.std[:len(data)] + 1e-8)
        return data / (np.max(np.abs(data)) + 1e-8)
    
    def _update_statistics(self, data: np.ndarray) -> None:
        """Обновляет скользящие среднее и дисперсию."""
        if self.mean is None:
            self.mean = np.zeros(len(data))
            self.std = np.ones(len(data))
        
        # Обновляем размер
        if len(self.mean) != len(data):
            # Изменение размера
            new_mean = np.zeros(len(data))
            new_std = np.ones(len(data))
            min_len = min(len(self.mean), len(data))
            new_mean[:min_len] = self.mean[:min_len]
            new_std[:min_len] = self.std[:min_len]
            self.mean = new_mean
            self.std = new_std
        
        # Обновление с экспоненциальным сглаживанием
        alpha = 0.1
        self.mean = (1 - alpha) * self.mean + alpha * data
        self.std = (1 - alpha) * self.std + alpha * np.abs(data - self.mean)
        self.std = np.maximum(self.std, 0.01)  # предотвращаем деление на ноль
        self.n_samples += 1
    
    def _initialize_projection(self, dim: int) -> None:
        """Инициализирует случайную проекционную матрицу."""
        self.projection = np.random.randn(dim, self.latent_dim) * 0.1
        # Ортогонализация (Грам-Шмидт)
        self.projection = self._gram_schmidt(self.projection)
        self.is_trained = True
    
    def _train_projection(self) -> None:
        """Обновляет проекционную матрицу на основе накопленных данных."""
        if len(self.data_buffer) < 2:
            return
        
        # Формируем матрицу данных
        data_matrix = np.array(self.data_buffer)
        
        # Центрирование
        data_centered = data_matrix - np.mean(data_matrix, axis=0)
        
        # Вычисляем ковариационную матрицу
        cov = np.cov(data_centered.T)
        
        # Вычисляем собственные векторы
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Берём top-k собственных векторов
        idx = np.argsort(eigenvalues)[::-1][:self.latent_dim]
        new_projection = eigenvectors[:, idx]
        
        # Смешиваем со старой проекцией (инерция)
        if self.projection is not None:
            alpha = 0.3  # скорость обучения
            self.projection = (1 - alpha) * self.projection + alpha * new_projection
        else:
            self.projection = new_projection
        
        # Ортогонализация
        self.projection = self._gram_schmidt(self.projection)
        self.is_trained = True
    
    def _gram_schmidt(self, matrix: np.ndarray) -> np.ndarray:
        """Ортогонализирует столбцы матрицы (Грам-Шмидт)."""
        n_cols = matrix.shape[1]
        ortho = np.zeros_like(matrix)
        
        for i in range(n_cols):
            vec = matrix[:, i].copy()
            for j in range(i):
                vec -= np.dot(ortho[:, j], matrix[:, i]) * ortho[:, j]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                ortho[:, i] = vec / norm
            else:
                # Если выродился, добавляем случайный вектор
                ortho[:, i] = np.random.randn(matrix.shape[0])
                ortho[:, i] /= np.linalg.norm(ortho[:, i])
        
        return ortho
    
    def decompress(self, latent: np.ndarray, original_shape: Optional[Tuple] = None) -> np.ndarray:
        """
        Восстанавливает данные из латентного представления.
        
        Args:
            latent: Сжатый вектор
            original_shape: Исходная форма для восстановления
        
        Returns:
            Восстановленные данные
        """
        if self.projection is None:
            return latent
        
        # Восстановление
        reconstructed = np.dot(latent, self.projection.T)
        
        # Денормализация
        if self.mean is not None and self.std is not None:
            reconstructed = reconstructed * (self.std[:len(reconstructed)] + 1e-8) + self.mean[:len(reconstructed)]
        
        # Восстановление формы
        if original_shape is not None:
            reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed
    
    def get_latent_dim(self) -> int:
        """Возвращает размерность латентного пространства."""
        return self.latent_dim
    
    def get_compression_ratio(self, original_dim: int) -> float:
        """Возвращает коэффициент сжатия."""
        return original_dim / self.latent_dim
