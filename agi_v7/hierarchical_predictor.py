# -*- coding: utf-8 -*-
"""
ИЕРАРХИЧЕСКИЙ ПРЕДИКТОР (HierarchicalPredictor)

Три уровня прогнозов:
- Быстрый (0.1с): мгновенные изменения
- Средний (1с): ближайшие действия
- Долгий (10с): долгосрочные последствия

Агент использует прогнозы для выбора действий, которые ведут к лучшему будущему.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Prediction:
    """Прогноз на одном уровне."""
    horizon: float           # горизонт прогноза (0.1, 1, 10)
    hunger: float            # прогнозируемый голод
    energy: float            # прогнозируемая энергия
    health: float            # прогнозируемое здоровье
    pain: float              # прогнозируемая боль
    food_nearby: bool        # будет ли еда рядом
    danger_nearby: bool      # будет ли опасность рядом
    confidence: float = 0.5  # уверенность в прогнозе


class HierarchicalPredictor:
    """
    Иерархический предиктор.
    
    Три уровня прогнозов:
    1. Быстрый (0.1с) — реакция на мгновенные изменения
    2. Средний (1с) — ближайшие действия
    3. Долгий (10с) — долгосрочные последствия
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
        # История состояний для прогнозов
        self.history: List[Dict[str, float]] = []
        
        # Статистика по действиям
        self.action_outcomes: Dict[str, List[Dict[str, float]]] = {}
        
        # Последние прогнозы
        self.last_predictions: Dict[str, Prediction] = {}
        
        # Порог уверенности для использования прогнозов
        self.confidence_threshold = 0.6
        
        print("🔮 Иерархический предиктор инициализирован")
    
    def update(self, state: Dict[str, float]):
        """Обновляет историю состояний."""
        self.history.append(state.copy())
        if len(self.history) > self.window_size * 3:
            self.history.pop(0)
    
    def record_action_outcome(self, action: str, state_before: Dict[str, float], state_after: Dict[str, float], reward: float):
        """Записывает результат действия для обучения прогнозов."""
        if action not in self.action_outcomes:
            self.action_outcomes[action] = []
        
        outcome = {
            'before': state_before.copy(),
            'after': state_after.copy(),
            'reward': reward
        }
        self.action_outcomes[action].append(outcome)
        
        # Ограничиваем историю
        if len(self.action_outcomes[action]) > self.window_size:
            self.action_outcomes[action].pop(0)
    
    def predict(self, current_state: Dict[str, float], horizon: float) -> Prediction:
        """
        Делает прогноз на заданный горизонт.
        
        Параметры:
        - current_state: текущее состояние
        - horizon: горизонт прогноза (0.1, 1, 10)
        
        Возвращает:
        - Prediction: прогнозируемое состояние
        """
        # Базовые значения — экстраполяция текущего состояния
        hunger = current_state.get('hunger', 0.0)
        energy = current_state.get('energy', 0.5)
        health = current_state.get('health', 1.0)
        pain = current_state.get('pain', 0.0)
        
        # Прогноз изменений в зависимости от горизонта
        if horizon == 0.1:
            # Быстрый горизонт — почти без изменений
            hunger_change = 0.01 * random.uniform(0.5, 1.5)
            energy_change = -0.005 * random.uniform(0.5, 1.5)
            confidence = 0.8
        elif horizon == 1.0:
            # Средний горизонт — заметные изменения
            hunger_change = 0.1 * random.uniform(0.5, 1.5)
            energy_change = -0.05 * random.uniform(0.5, 1.5)
            confidence = 0.6
        else:
            # Долгий горизонт — значительные изменения
            hunger_change = 0.3 * random.uniform(0.5, 1.5)
            energy_change = -0.15 * random.uniform(0.5, 1.5)
            confidence = 0.4
        
        # Учитываем историю для улучшения прогноза
        if len(self.history) > 5:
            # Средняя скорость изменения
            avg_hunger_change = 0.0
            avg_energy_change = 0.0
            for i in range(1, min(5, len(self.history))):
                prev = self.history[-i-1]
                curr = self.history[-i]
                avg_hunger_change += curr.get('hunger', 0.0) - prev.get('hunger', 0.0)
                avg_energy_change += curr.get('energy', 0.5) - prev.get('energy', 0.5)
            avg_hunger_change /= min(5, len(self.history) - 1)
            avg_energy_change /= min(5, len(self.history) - 1)
            
            # Смешиваем с текущим прогнозом
            if horizon == 0.1:
                hunger_change = avg_hunger_change * 0.2 + hunger_change * 0.8
                energy_change = avg_energy_change * 0.2 + energy_change * 0.8
            elif horizon == 1.0:
                hunger_change = avg_hunger_change * 0.5 + hunger_change * 0.5
                energy_change = avg_energy_change * 0.5 + energy_change * 0.5
            else:
                hunger_change = avg_hunger_change * 0.8 + hunger_change * 0.2
                energy_change = avg_energy_change * 0.8 + energy_change * 0.2
            
            confidence = min(0.9, confidence + 0.1)
        
        # Применяем изменения
        new_hunger = max(0.0, min(1.0, hunger + hunger_change))
        new_energy = max(0.0, min(1.0, energy + energy_change))
        
        # Здоровье зависит от голода и энергии
        health_change = -0.01 * new_hunger - 0.005 * (1.0 - new_energy)
        new_health = max(0.0, min(1.0, health + health_change))
        
        # Боль зависит от здоровья
        new_pain = max(0.0, min(1.0, pain + 0.01 * (1.0 - new_health)))
        
        # Еда и опасность рядом
        food_nearby = new_hunger < 0.3 and random.random() < 0.3
        danger_nearby = new_health < 0.3 and random.random() < 0.2
        
        # Создаём прогноз
        prediction = Prediction(
            horizon=horizon,
            hunger=new_hunger,
            energy=new_energy,
            health=new_health,
            pain=new_pain,
            food_nearby=food_nearby,
            danger_nearby=danger_nearby,
            confidence=confidence
        )
        
        # Сохраняем
        self.last_predictions[f"{horizon:.1f}"] = prediction
        
        return prediction
    
    def predict_action(self, action: str, current_state: Dict[str, float]) -> float:
        """
        Предсказывает результат действия.
        
        Возвращает ожидаемую награду от действия.
        """
        if action not in self.action_outcomes or not self.action_outcomes[action]:
            # Нет данных — даём нейтральную оценку
            return 0.0
        
        # Ищем похожие состояния в истории
        similar_outcomes = []
        for outcome in self.action_outcomes[action]:
            before = outcome['before']
            # Сравниваем ключевые параметры
            hunger_diff = abs(before.get('hunger', 0.0) - current_state.get('hunger', 0.0))
            energy_diff = abs(before.get('energy', 0.5) - current_state.get('energy', 0.5))
            pain_diff = abs(before.get('pain', 0.0) - current_state.get('pain', 0.0))
            
            similarity = 1.0 - (hunger_diff + energy_diff + pain_diff) / 3.0
            if similarity > 0.5:
                similar_outcomes.append((similarity, outcome))
        
        if not similar_outcomes:
            return 0.0
        
        # Взвешенная средняя награда
        total_weight = 0.0
        weighted_reward = 0.0
        for similarity, outcome in similar_outcomes:
            weight = similarity * similarity  # квадрат для усиления похожести
            total_weight += weight
            weighted_reward += weight * outcome.get('reward', 0.0)
        
        if total_weight == 0:
            return 0.0
        
        return weighted_reward / total_weight
    
    def get_best_action(self, actions: List[str], current_state: Dict[str, float]) -> Tuple[str, float]:
        """
        Выбирает лучшее действие на основе прогнозов.
        
        Возвращает (действие, ожидаемая награда).
        """
        best_action = actions[0] if actions else 'move_right'
        best_reward = -999.0
        
        for action in actions:
            reward = self.predict_action(action, current_state)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        # Если нет данных, выбираем случайное
        if best_reward == -999.0:
            best_action = random.choice(actions) if actions else 'move_right'
            best_reward = 0.0
        
        return best_action, best_reward
    
    def get_forecast(self, current_state: Dict[str, float]) -> Dict[str, Prediction]:
        """
        Возвращает прогнозы на всех трёх уровнях.
        """
        return {
            'fast': self.predict(current_state, 0.1),
            'medium': self.predict(current_state, 1.0),
            'long': self.predict(current_state, 10.0)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику предиктора."""
        return {
            'history_length': len(self.history),
            'action_count': len(self.action_outcomes),
            'total_outcomes': sum(len(v) for v in self.action_outcomes.values()),
            'confidence': self.confidence_threshold
        }
