# -*- coding: utf-8 -*-
"""
Глобальное состояние для модулей AGI v7.0
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class GlobalState:
    """Глобальное состояние, доступное всем модулям."""
    
    # Базовые параметры
    energy: float = 1.0
    health: float = 1.0
    food: int = 0
    age: int = 0
    alive: bool = True
    
    # Эмоции
    emotions: Dict[str, float] = field(default_factory=lambda: {
        'dopamine': 0.0,
        'serotonin': 0.0,
        'norepinephrine': 0.0,
        'acetylcholine': 0.0,
        'oxytocin': 0.0,
        'cortisol': 0.0,
        'endorphin': 0.0,
        'melatonin': 0.0
    })
    
    # Позиция в среде
    position: tuple = (0, 0)
    
    # Внутренние состояния
    memory: Dict[str, Any] = field(default_factory=dict)
    attention: Dict[str, float] = field(default_factory=dict)
    
    # Системные флаги
    is_system1_active: bool = True
    is_system2_active: bool = False
    consciousness_level: float = 0.5
    
    def update_emotion(self, key: str, value: float) -> None:
        """Обновить эмоцию."""
        if key in self.emotions:
            self.emotions[key] = max(-1.0, min(1.0, value))
            
    def get_emotion(self, key: str) -> float:
        """Получить значение эмоции."""
        return self.emotions.get(key, 0.0)
        
    def __repr__(self) -> str:
        return f"GlobalState(energy={self.energy:.2f}, health={self.health:.2f}, alive={self.alive})"
