# -*- coding: utf-8 -*-
"""
СИСТЕМА АВТОНОМНЫХ ЦЕЛЕЙ (нейронно-подобная динамика)
Не упрощённые правила, а динамические нейронные поля,
которые генерируют цели на основе состояния.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import time


class DesireSystem:
    """
    Система автономных целей.
    Работает как нейронное поле: разные желания конкурируют,
    и побеждает то, которое имеет максимальную активацию.
    """
    
    def __init__(self, num_desires: int = 8):
        # Нейронные поля для каждого желания
        self.num_desires = num_desires
        self.desire_names = [
            'выжить', 'поесть', 'отдохнуть', 'исследовать',
            'общаться', 'учиться', 'защищаться', 'размножаться'
        ]
        
        # Активация желаний (нейронная динамика)
        self.activations = np.array([0.1] * num_desires, dtype=np.float32)
        
        # Веса связей между желаниями (конкуренция и взаимопомощь)
        self.weights = np.random.randn(num_desires, num_desires) * 0.1
        np.fill_diagonal(self.weights, 0)  # нет связей с самим собой
        
        # Пороги активации (как у нейронов)
        self.thresholds = np.array([0.3] * num_desires, dtype=np.float32)
        
        # Усталость желаний (чтобы не зацикливались)
        self.fatigue = np.zeros(num_desires, dtype=np.float32)
        self.fatigue_decay = 0.05
        self.fatigue_increment = 0.1
        
        # История активаций
        self.activation_history = deque(maxlen=50)
        self.current_winner = None
        self.current_winner_name = None
        
        # Внутреннее состояние
        self.homeostasis = 0.5
        self.drive = 0.5
        
    def update(self, state: Dict[str, Any]) -> None:
        """
        Обновляет нейронные поля желаний на основе состояния.
        """
        energy = state.get('energy', 0.5)
        fear = state.get('fear', 0.0)
        curiosity = state.get('curiosity', 0.3)
        hunger = state.get('hunger', 0.3)
        fatigue = state.get('fatigue', 0.2)
        social = state.get('social_need', 0.2)
        
        # --- ВХОДНЫЕ СИГНАЛЫ (как сенсорные входы в нейроны) ---
        inputs = np.array([
            1.0 - energy,  # выжить — если мало энергии
            hunger,         # поесть
            fatigue,        # отдохнуть
            0.5 + 0.5 * curiosity,  # исследовать
            social,         # общаться
            1.0 - energy * 0.5 + curiosity * 0.5,  # учиться
            fear,           # защищаться
            0.3 + 0.1 * energy  # размножаться
        ], dtype=np.float32)
        
        # --- НЕЙРОННАЯ ДИНАМИКА (конкуренция) ---
        # 1. Суммируем входы и связи
        activation_input = inputs * 1.5  # усиление входов
        
        # 2. Взаимодействие между желаниями (конкуренция и поддержка)
        interaction = np.dot(self.activations, self.weights) * 0.3
        
        # 3. Общий вход в нейроны
        net_input = activation_input + interaction
        
        # 4. Применяем порог (как у нейронов)
        net_input = net_input - self.thresholds
        
        # 5. Применяем усталость (ингибирование активных нейронов)
        net_input = net_input - self.fatigue * 0.5
        
        # 6. Активация через функцию (сигмоидная, как у нейронов)
        new_activations = 1.0 / (1.0 + np.exp(-net_input * 2.0))
        
        # 7. Нормализация (конкуренция за ресурсы)
        total = np.sum(new_activations) + 0.001
        self.activations = new_activations / total * 0.8 + self.activations * 0.2
        
        # 8. Обновляем усталость
        self.fatigue += self.fatigue_increment * self.activations
        self.fatigue *= (1.0 - self.fatigue_decay)
        self.fatigue = np.clip(self.fatigue, 0.0, 1.0)
        
        # 9. Определяем победителя (конкуренция)
        winner_idx = np.argmax(self.activations)
        winner_activation = self.activations[winner_idx]
        
        if winner_activation > 0.2:
            self.current_winner = winner_idx
            self.current_winner_name = self.desire_names[winner_idx]
        else:
            self.current_winner = None
            self.current_winner_name = None
        
        # 10. Сохраняем историю
        self.activation_history.append({
            'activations': self.activations.copy(),
            'winner': self.current_winner_name,
            'homeostasis': self.homeostasis,
            'time': time.time()
        })
        
        # 11. Обновляем гомеостаз (внутреннее равновесие)
        self.homeostasis = 0.9 * self.homeostasis + 0.1 * (1.0 - np.std(self.activations))
        
    def get_desire(self) -> Optional[Tuple[str, float]]:
        """
        Возвращает текущее доминирующее желание.
        """
        if self.current_winner is None:
            return None
        return (self.current_winner_name, self.activations[self.current_winner])
    
    def get_all_desires(self) -> List[Tuple[str, float]]:
        """
        Возвращает все желания с их активациями.
        """
        return [(self.desire_names[i], float(self.activations[i])) 
                for i in range(self.num_desires)]
    
    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает состояние системы.
        """
        return {
            'activations': self.activations.tolist(),
            'desire_names': self.desire_names,
            'current_winner': self.current_winner_name,
            'current_activation': self.activations[self.current_winner] if self.current_winner is not None else 0.0,
            'homeostasis': self.homeostasis,
            'drive': self.drive,
            'fatigue': self.fatigue.tolist()
        }
    
    def set_state(self, energy: float = 0.5, fear: float = 0.0, 
                  curiosity: float = 0.3, hunger: float = 0.3,
                  fatigue: float = 0.2, social_need: float = 0.2):
        """
        Устанавливает состояние системы извне.
        """
        self.update({
            'energy': energy,
            'fear': fear,
            'curiosity': curiosity,
            'hunger': hunger,
            'fatigue': fatigue,
            'social_need': social_need
        })
        
    def inhibit_desire(self, desire_name: str, inhibition: float = 0.5):
        """
        Ингибирует конкретное желание (например, при страхе).
        """
        if desire_name in self.desire_names:
            idx = self.desire_names.index(desire_name)
            self.activations[idx] *= (1.0 - inhibition)
            self.activations[idx] = max(0.0, self.activations[idx])
