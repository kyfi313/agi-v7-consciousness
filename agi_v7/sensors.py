# -*- coding: utf-8 -*-
"""
СЕНСОРНЫЙ МОДУЛЬ
Преобразует сырые данные среды в нейронные сигналы.
Поддерживает: зрение (2D), слух (1D), проприоцепцию (координаты), боль.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class SensoryInput:
    """Структура сенсорных данных."""
    visual: Optional[np.ndarray] = None      # 2D изображение (H, W) или (H, W, C)
    auditory: Optional[np.ndarray] = None    # 1D звуковой сигнал
    position: Tuple[float, float] = (0.0, 0.0)  # координаты в среде
    energy: float = 0.5                      # уровень энергии (0-1)
    health: float = 1.0                      # здоровье (0-1)
    pain: float = 0.0                        # боль (0-1)
    reward: float = 0.0                      # внешняя награда (дофамин)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'visual': self.visual,
            'auditory': self.auditory,
            'position': self.position,
            'energy': self.energy,
            'health': self.health,
            'pain': self.pain,
            'reward': self.reward
        }


class SensorModule:
    """
    Обрабатывает сырые данные среды и преобразует их в нейронные сигналы.
    Использует нормализацию, фильтрацию и выделение признаков.
    """
    
    def __init__(self, visual_size: Tuple[int, int] = (64, 64)):
        self.visual_size = visual_size
        self.last_input = None
        self.feature_buffer = []  # история признаков для временной интеграции
        
    def process(self, raw_data: Dict[str, Any]) -> SensoryInput:
        """
        Принимает сырые данные и возвращает структурированный сенсорный вход.
        """
        # Извлечение и нормализация
        visual = self._process_visual(raw_data.get('visual'))
        auditory = self._process_auditory(raw_data.get('auditory'))
        position = raw_data.get('position', (0.0, 0.0))
        energy = np.clip(raw_data.get('energy', 0.5), 0.0, 1.0)
        health = np.clip(raw_data.get('health', 1.0), 0.0, 1.0)
        pain = np.clip(raw_data.get('pain', 0.0), 0.0, 1.0)
        reward = raw_data.get('reward', 0.0)
        
        # Сохраняем для истории
        self.last_input = SensoryInput(
            visual=visual,
            auditory=auditory,
            position=position,
            energy=energy,
            health=health,
            pain=pain,
            reward=reward
        )
        
        return self.last_input
    
    def _process_visual(self, raw_visual: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Обрабатывает визуальные данные: изменение размера, нормализация, серый."""
        if raw_visual is None:
            return None
        
        # Если пришло изображение
        if isinstance(raw_visual, np.ndarray):
            # Приводим к float32 и нормализуем
            if raw_visual.dtype != np.float32:
                raw_visual = raw_visual.astype(np.float32) / 255.0
            
            # Изменяем размер до visual_size
            if raw_visual.shape[:2] != self.visual_size:
                # Простое изменение размера через интерполяцию
                h, w = raw_visual.shape[:2]
                target_h, target_w = self.visual_size
                if len(raw_visual.shape) == 3:
                    # Цветное -> серый + resize
                    raw_visual = np.mean(raw_visual, axis=-1)
                # Resize (бикубическая интерполяция упрощённо)
                raw_visual = self._resize_2d(raw_visual, target_h, target_w)
            
            # Нормализация к [0, 1]
            raw_visual = np.clip(raw_visual, 0.0, 1.0)
            
            return raw_visual
        
        return None
    
    def _resize_2d(self, arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        """Упрощённый resize 2D массива (ближайший сосед)."""
        h, w = arr.shape
        h_ratio = h / new_h
        w_ratio = w / new_w
        
        h_indices = np.floor(np.arange(new_h) * h_ratio).astype(int)
        w_indices = np.floor(np.arange(new_w) * w_ratio).astype(int)
        
        return arr[h_indices][:, w_indices]
    
    def _process_auditory(self, raw_auditory: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Обрабатывает звуковые данные: нормализация, обрезание."""
        if raw_auditory is None:
            return None
        
        if isinstance(raw_auditory, np.ndarray):
            # Нормализация
            if np.max(raw_auditory) > 1.0:
                raw_auditory = raw_auditory / (np.max(raw_auditory) + 1e-8)
            # Ограничение длины (до 1024)
            if len(raw_auditory) > 1024:
                raw_auditory = raw_auditory[:1024]
            return raw_auditory.astype(np.float32)
        
        return None
    
    def get_features(self, sensory: SensoryInput) -> np.ndarray:
        """Извлекает компактные признаки из сенсорных данных."""
        features = []
        
        # Визуальные признаки (если есть)
        if sensory.visual is not None:
            # Усреднение по блокам (простое сжатие)
            h, w = sensory.visual.shape
            block_h, block_w = max(1, h // 8), max(1, w // 8)
            pooled = sensory.visual.reshape(block_h, h//block_h, block_w, w//block_w).mean(axis=(1, 3))
            features.append(pooled.flatten())
        
        # Аудиальные признаки
        if sensory.auditory is not None:
            # Спектрограмма - усреднение
            aud = sensory.auditory
            if len(aud) > 0:
                aud_features = np.array([
                    np.mean(aud),
                    np.std(aud),
                    np.max(aud),
                    np.min(aud),
                    np.percentile(aud, 50),
                    np.percentile(aud, 90)
                ])
                features.append(aud_features)
        
        # Проприоцептивные признаки
        pos = sensory.position
        features.append(np.array([pos[0], pos[1], sensory.energy, sensory.health, sensory.pain, sensory.reward]))
        
        # Объединение
        if features:
            return np.concatenate(features)
        else:
            # Если нет данных, возвращаем нулевой вектор
            return np.zeros(16, dtype=np.float32)
