# -*- coding: utf-8 -*-
"""
АКТИВНОЕ ВНИМАНИЕ
Выбирает наиболее релевантные сигналы и подавляет остальные.

Принцип работы:
1. Оценивает важность каждого входного сигнала на основе текущей цели и эмоций
2. Выбирает 3-5 наиболее важных сигналов
3. Остальные сигналы подавляются (уменьшается их вес)
4. Может переключаться, если цель меняется
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple, List


class ActiveAttention:
    """
    Активное внимание с выбором и подавлением.
    
    Вместо простого усиления сигнала (салиенция), этот модуль
    активно выбирает, на что направить ресурсы, и подавляет всё остальное.
    """
    
    def __init__(self, num_signals=10, focus_size=5):
        """
        Args:
            num_signals: Количество входных сигналов
            focus_size: Количество сигналов в фокусе внимания
        """
    def __init__(self, focus_duration: float = 5.0, num_signals: int = 10, focus_size: int = 3):
        self.focus_duration = focus_duration
        self.num_signals = num_signals
        self.focus_size = focus_size
        
        # История внимания для обучения
        self.attention_history = deque(maxlen=100)
        
        # Веса важности для каждого сигнала (обучаемые)
        self.signal_weights = np.ones(num_signals) / num_signals
        
        # Текущий фокус
        self.current_focus = []  # индексы сигналов в фокусе
        self.current_salience = np.zeros(num_signals)
        
        # Связь сигналов с целями
        self.signal_goal_relevance = defaultdict(lambda: defaultdict(float))
        
        # Параметры
        self.learning_rate = 0.1
        self.inhibition_strength = 0.5  # Насколько сильно подавлять не-фокус
        
        # Поля для совместимости с LivingAgent
        self.current_focus_obj = None
        self.saliency_map = {}
        
        print(f"👁️ Активное внимание инициализировано (фокус: {focus_size} из {num_signals})")
    
    def update(self, perception: Dict[str, Any], internal_state: Dict[str, float]) -> Optional[Any]:
        """
        Обновляет фокус внимания на основе восприятия и внутреннего состояния.
        
        Параметры:
        - perception: данные восприятия
        - internal_state: внутреннее состояние (голод, энергия, эмоции)
        
        Возвращает:
        - AttentionFocus: новый фокус внимания или None
        """
        # Создаём фокус на основе восприятия и состояния
        hunger = internal_state.get('hunger', 0.0)
        energy = internal_state.get('energy', 0.5)
        pain = internal_state.get('pain', 0.0)
        
        # Определяем, на что смотреть
        if hunger > 0.6 and perception.get('visible_food'):
            target = 'food'
            nearest = perception.get('nearest_food')
            if nearest:
                pos = (nearest[0], nearest[1])
                priority = min(1.0, hunger * 1.5)
                self.current_focus_obj = type('AttentionFocus', (), {
                    'target': target,
                    'position': pos,
                    'priority': priority,
                    'intensity': 0.8,
                    'duration': 0.0
                })()
                return self.current_focus_obj
        elif pain > 0.4:
            target = 'danger'
            nearest = perception.get('nearest_danger')
            if nearest:
                pos = (nearest[0], nearest[1])
                priority = min(1.0, pain * 1.5)
                self.current_focus_obj = type('AttentionFocus', (), {
                    'target': target,
                    'position': pos,
                    'priority': priority,
                    'intensity': 0.8,
                    'duration': 0.0
                })()
                return self.current_focus_obj
        elif energy > 0.6:
            target = 'explore'
            # Случайная точка для исследования
            import random
            x, y = perception.get('position', (0, 0))
            pos = (x + random.uniform(-3, 3), y + random.uniform(-3, 3))
            self.current_focus_obj = type('AttentionFocus', (), {
                'target': target,
                'position': pos,
                'priority': 0.5,
                'intensity': 0.6,
                'duration': 0.0
            })()
            return self.current_focus_obj
        
        # Если ничего не привлекло внимание, сбрасываем фокус
        self.current_focus_obj = None
        return None
    
    def focus(self, signals, goal=None, emotion=None):
        """
        Применяет активное внимание к входным сигналам.
        
        Args:
            signals: Входной вектор сигналов
            goal: Текущая цель (для оценки релевантности)
            emotion: Текущее эмоциональное состояние
        
        Returns:
            dict: {
                'focused': массив с усиленными сигналами,
                'suppressed': массив с подавленными сигналами,
                'focus_indices': индексы в фокусе,
                'salience': салиенция каждого сигнала
            }
        """
        signals = np.array(signals)
        if len(signals) != self.num_signals:
            # Адаптируем размер
            if len(signals) > self.num_signals:
                signals = signals[:self.num_signals]
            else:
                signals = np.pad(signals, (0, self.num_signals - len(signals)))
        
        # 1. Оцениваем важность каждого сигнала
        importance = self._compute_importance(signals, goal, emotion)
        
        # 2. Выбираем фокус (top-k по важности)
        focus_indices = np.argsort(importance)[-self.focus_size:]
        self.current_focus = focus_indices.tolist()
        
        # 3. Строим маску внимания
        attention_mask = np.ones(self.num_signals) * self.inhibition_strength
        attention_mask[focus_indices] = 1.0
        
        # 4. Применяем маску
        focused = signals * attention_mask
        suppressed = signals * (1.0 - attention_mask)
        
        # 5. Сохраняем историю
        self.attention_history.append({
            'signals': signals.tolist(),
            'focus': focus_indices.tolist(),
            'importance': importance.tolist()
        })
        
        self.current_salience = importance
        
        return {
            'focused': focused.tolist(),
            'suppressed': suppressed.tolist(),
            'focus_indices': focus_indices.tolist(),
            'salience': importance.tolist(),
            'attention_mask': attention_mask.tolist()
        }
    
    def _compute_importance(self, signals, goal=None, emotion=None):
        """Вычисляет важность каждого сигнала"""
        importance = np.zeros(self.num_signals)
        
        # 1. Базовая важность от силы сигнала (нормализованная)
        signal_strength = np.abs(signals) / (np.max(np.abs(signals)) + 1e-6)
        importance += 0.3 * signal_strength
        
        # 2. Важность от цели (если есть)
        if goal is not None and goal in self.signal_goal_relevance:
            goal_relevance = np.array([
                self.signal_goal_relevance[goal][i] 
                for i in range(self.num_signals)
            ])
            importance += 0.4 * goal_relevance
        
        # 3. Важность от эмоций
        if emotion is not None:
            # Страх усиливает сигналы опасности
            if emotion.get('fear', 0) > 0.5:
                # Предполагаем, что сигналы опасности находятся в определённых позициях
                danger_indices = self._get_danger_indices()
                importance[danger_indices] += 0.3 * emotion['fear']
            
            # Любопытство усиливает новые сигналы
            if emotion.get('curiosity', 0) > 0.5:
                # Сигналы с низкой историей внимания
                history_weights = self._get_history_weights()
                importance += 0.2 * emotion['curiosity'] * history_weights
        
        # 4. Обновляем веса на основе истории
        self._update_weights(importance)
        
        return importance
    
    def _get_danger_indices(self):
        """Возвращает индексы сигналов, связанных с опасностью"""
        # По умолчанию: сигналы 2, 4 (danger_nearby, min_danger_dist)
        return [2, 4]
    
    def _get_history_weights(self):
        """Вычисляет веса на основе истории внимания"""
        if len(self.attention_history) < 10:
            return np.ones(self.num_signals) / self.num_signals
        
        # Считаем, какие сигналы редко попадали в фокус
        focus_counts = np.zeros(self.num_signals)
        for entry in self.attention_history:
            for idx in entry['focus']:
                focus_counts[idx] += 1
        
        # Нормализуем (чем реже, тем выше вес)
        max_count = max(1, np.max(focus_counts))
        novelty = 1.0 - focus_counts / max_count
        return novelty
    
    def _update_weights(self, importance):
        """Обновляет веса сигналов на основе важности"""
        self.signal_weights = 0.9 * self.signal_weights + 0.1 * importance
        self.signal_weights = self.signal_weights / (np.sum(self.signal_weights) + 1e-6)
    
    def update_goal_relevance(self, goal, signal_idx, relevance):
        """Обновляет связь сигнала с целью"""
        self.signal_goal_relevance[goal][signal_idx] = relevance
    
    def get_focus_summary(self):
        """Возвращает краткое описание текущего фокуса"""
        if not self.current_focus:
            return "Нет фокуса"
        return f"Фокус на сигналах: {self.current_focus}"
    
    def get_stats(self):
        """Возвращает статистику внимания"""
        return {
            'focus_size': len(self.current_focus),
            'history_size': len(self.attention_history),
            'signal_weights': self.signal_weights.tolist(),
            'avg_focus_stability': self._compute_stability()
        }
    
    def _compute_stability(self):
        """Вычисляет стабильность фокуса (насколько часто меняется)"""
        if len(self.attention_history) < 2:
            return 1.0
        
        changes = 0
        for i in range(1, len(self.attention_history)):
            prev = set(self.attention_history[i-1]['focus'])
            curr = set(self.attention_history[i]['focus'])
            if prev != curr:
                changes += 1
        
        return 1.0 - changes / len(self.attention_history)
