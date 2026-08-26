# -*- coding: utf-8 -*-
"""
Модуль кристаллизации привычек (HabitCrystallizer)
Превращает успешные длинные цепочки действий в автоматические привычки
"""

import numpy as np
from collections import defaultdict, deque


class HabitCrystallizer:
    """
    Кристаллизует успешные последовательности действий в привычки.
    
    Принцип работы:
    1. Отслеживает состояния и действия, которые привели к положительной награде
    2. Если одно и то же состояние + действие повторяются с успехом >= threshold,
       запоминает их как привычку
    3. Привычка может быть использована автоматически, минуя кору
    """
    
    def __init__(self, threshold=3, decay=0.95, max_habits=100):
        """
        Args:
            threshold: Количество успешных повторений для кристаллизации
            decay: Скорость забывания старых успехов (для адаптации)
            max_habits: Максимальное число привычек
        """
        self.threshold = threshold
        self.decay = decay
        self.max_habits = max_habits
        
        # Счётчики успехов: (state_tuple, action) -> count
        self.success_counts = defaultdict(int)
        
        # Кэш привычек: state_tuple -> action
        self.habits = {}
        
        # Для адаптации: храним последние 100 событий
        self.history = deque(maxlen=100)
        
        # Статистика использования привычек
        self.habit_usage = {}
        self.formation_events = []
    
    def update(self, state, action, reward):
        """
        Обновляет кристаллизатор на основе опыта.
        
        Args:
            state: Вектор состояния или словарь (преобразуется в tuple)
            action: Название действия
            reward: Полученная награда
        
        Returns:
            bool: True, если сформирована новая привычка
        """
        if reward <= 0.1:
            # Только успешные действия кристаллизуются
            return False
        
        # Преобразуем состояние в кортеж для хеширования
        state_key = self._state_to_key(state)
        event_key = (state_key, action)
        
        # Увеличиваем счётчик успеха
        self.success_counts[event_key] += 1
        
        # Записываем в историю
        self.history.append((state_key, action, reward))
        
        # Проверяем, можно ли кристаллизовать привычку
        if self.success_counts[event_key] >= self.threshold:
            # Проверяем, не превышен ли лимит привычек
            if len(self.habits) >= self.max_habits:
                self._prune_habits()
            
            if state_key not in self.habits:
                self.habits[state_key] = action
                self.habit_usage[state_key] = 0
                self.formation_events.append((state_key, action, self.success_counts[event_key]))
                print(f"🔧 Привычка сформирована: {action} для состояния {state_key[:3]}... (повторений: {self.success_counts[event_key]})")
                return True
        
        # Применяем затухание для старых счётчиков
        if len(self.history) % 10 == 0:
            self._decay_counts()
        
        return False
    
    def get_habit(self, state):
        """
        Возвращает привычку для данного состояния, если она есть.
        
        Args:
            state: Вектор состояния или словарь
        
        Returns:
            str или None: Название действия, если привычка есть
        """
        state_key = self._state_to_key(state)
        if state_key in self.habits:
            self.habit_usage[state_key] += 1
            return self.habits[state_key]
        return None
    
    def is_habit(self, state):
        """Проверяет, есть ли привычка для данного состояния"""
        state_key = self._state_to_key(state)
        return state_key in self.habits
    
    def get_confidence(self, state, action):
        """
        Возвращает уверенность в том, что действие является привычкой.
        
        Returns:
            float: 0.0-1.0, уверенность
        """
        state_key = self._state_to_key(state)
        event_key = (state_key, action)
        count = self.success_counts.get(event_key, 0)
        confidence = min(1.0, count / self.threshold)
        return confidence
    
    def _state_to_key(self, state):
        """Преобразует состояние в хешируемый кортеж"""
        if isinstance(state, dict):
            # Если словарь, используем только ключи, которые есть
            sorted_items = sorted(state.items())
            return tuple((k, v) for k, v in sorted_items if isinstance(v, (int, float, bool, str)))
        elif isinstance(state, (list, tuple, np.ndarray)):
            # Если вектор, преобразуем в кортеж с округлением для устойчивости
            if isinstance(state, np.ndarray):
                state = state.tolist()
            # Округляем до 2 знаков для группировки
            rounded = [round(x, 2) if isinstance(x, float) else x for x in state]
            return tuple(rounded)
        else:
            return (str(state),)
    
    def _decay_counts(self):
        """Применяет затухание к старым счётчикам успехов"""
        # Уменьшаем счётчики, которые не использовались недавно
        recent_states = set()
        for state_key, action, _ in list(self.history)[-20:]:
            recent_states.add((state_key, action))
        
        for key in list(self.success_counts.keys()):
            if key not in recent_states:
                self.success_counts[key] *= self.decay
                # Если счётчик стал слишком маленьким, удаляем
                if self.success_counts[key] < 0.5:
                    del self.success_counts[key]
                    # Если это была привычка, удаляем её
                    state_key, action = key
                    if state_key in self.habits and self.habits[state_key] == action:
                        del self.habits[state_key]
    
    def _prune_habits(self):
        """Удаляет наименее используемые привычки"""
        if len(self.habits) < self.max_habits:
            return
        
        # Сортируем по частоте использования
        sorted_habits = sorted(
            self.habit_usage.items(),
            key=lambda x: x[1]
        )
        
        # Удаляем половину наименее используемых
        to_remove = [state for state, _ in sorted_habits[:len(sorted_habits) // 2]]
        for state in to_remove:
            if state in self.habits:
                del self.habits[state]
            if state in self.habit_usage:
                del self.habit_usage[state]
        
        print(f"🧹 Привычки обрезаны: удалено {len(to_remove)} старых привычек")
    
    def get_stats(self):
        """Возвращает статистику кристаллизатора"""
        return {
            'total_habits': len(self.habits),
            'total_successes': len(self.success_counts),
            'history_size': len(self.history),
            'formation_events': len(self.formation_events),
            'habits': list(self.habits.keys())[:10]
        }
    
    def reset(self):
        """Сбрасывает кристаллизатор"""
        self.success_counts.clear()
        self.habits.clear()
        self.history.clear()
        self.habit_usage.clear()
        self.formation_events.clear()
        print("🔄 Кристаллизатор привычек сброшен")
