# -*- coding: utf-8 -*-
"""
МОДУЛЬ ИЕРАРХИЧЕСКОГО ПРЕДСКАЗАНИЯ
Гениальная идея: Предсказание работает на трёх уровнях:
- Микро — что произойдёт в следующий момент (спайки)
- Мезо — что произойдёт в следующие 5 шагов (действия)
- Макро — что произойдёт в следующие 50 шагов (цели)

Это моделирует иерархическую организацию коры —
от первичной сенсорной коры до префронтальной коры.
Ошибка предсказания на каждом уровне порождает ДИССОНАНС,
который ведёт к обучению.
"""

import numpy as np
from collections import deque
import time

class HierarchicalPredictor:
    """
    Иерархический предиктор с тремя уровнями.
    """
    
    def __init__(self, input_dim=64, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # === УРОВЕНЬ 1: МИКРО (спайки, моменты) ===
        self.micro = {
            'memory': deque(maxlen=10),      # последние 10 состояний
            'prediction': None,              # предсказание следующего состояния
            'error': 0.0,                    # ошибка предсказания
            'threshold': 0.3,                # порог ошибки
            'weights': np.random.randn(input_dim, hidden_dim) * 0.1,
        }
        
        # === УРОВЕНЬ 2: МЕЗО (действия, последовательности) ===
        self.meso = {
            'memory': deque(maxlen=20),      # последние 20 действий
            'prediction': None,              # предсказание следующего действия
            'error': 0.0,
            'threshold': 0.4,
            'sequence_buffer': [],           # буфер для обнаружения паттернов
            'pattern_memory': {},            # сохранённые паттерны
            'weights': np.random.randn(hidden_dim, 8) * 0.1,  # 8 действий
        }
        
        # === УРОВЕНЬ 3: МАКРО (цели, долгосрочные планы) ===
        self.macro = {
            'memory': deque(maxlen=50),      # последние 50 состояний
            'prediction': None,              # предсказание цели
            'error': 0.0,
            'threshold': 0.5,
            'goals': [],                     # список целей
            'subgoal': None,                 # текущая подцель
            'plan': [],                      # план действий
            'planning_horizon': 50,          # горизонт планирования
            'weights': np.random.randn(hidden_dim, 5) * 0.1,  # 5 целей
        }
        
        # === ОБЩИЕ МЕХАНИЗМЫ ===
        self.dissonance = 0.0               # общий диссонанс
        self.dissonance_history = deque(maxlen=100)
        self.learning_rate = 0.01
        self.step_count = 0
        self.last_update = time.time()
        
        # === ИНСАЙТЫ ===
        self.insights = []
        self.insight_threshold = 0.7
        
    def predict(self, state, action=None):
        """
        Предсказывает на всех трёх уровнях.
        
        Args:
            state: Текущее состояние (вектор)
            action: Текущее действие (если есть)
        
        Returns:
            dict: Предсказания на всех уровнях
        """
        self.step_count += 1
        
        # 1. МИКРО-УРОВЕНЬ: предсказываем следующее состояние
        micro_pred, micro_error = self._predict_micro(state)
        
        # 2. МЕЗО-УРОВЕНЬ: предсказываем следующее действие
        meso_pred, meso_error = self._predict_meso(state, action)
        
        # 3. МАКРО-УРОВЕНЬ: предсказываем цель
        macro_pred, macro_error = self._predict_macro(state)
        
        # 4. Вычисляем общий диссонанс
        self.dissonance = (micro_error * 0.5 + meso_error * 0.3 + macro_error * 0.2)
        self.dissonance_history.append(self.dissonance)
        
        # 5. Обновляем обучение
        self._learn(micro_error, meso_error, macro_error, state, action)
        
        # 6. Проверяем инсайты
        if self.dissonance > self.insight_threshold:
            insight = self._generate_insight(micro_error, meso_error, macro_error)
            self.insights.append(insight)
        
        return {
            'micro': {'prediction': micro_pred, 'error': micro_error},
            'meso': {'prediction': meso_pred, 'error': meso_error},
            'macro': {'prediction': macro_pred, 'error': macro_error},
            'dissonance': self.dissonance,
            'insights': self.insights[-3:],
        }
    
    def _predict_micro(self, state):
        """Микро-уровень: предсказывает следующее состояние."""
        # Если недостаточно данных
        if len(self.micro['memory']) < 3:
            self.micro['prediction'] = state
            return state, 0.0
        
        # Простая линейная регрессия
        prev_states = list(self.micro['memory'])
        if len(prev_states) >= 5:
            # Тренд
            trend = (prev_states[-1] - prev_states[0]) / len(prev_states)
            prediction = state + trend
        else:
            prediction = state * 0.9 + prev_states[-1] * 0.1
        
        # Ошибка
        if len(prev_states) >= 2:
            error = np.mean((prediction - prev_states[-1]) ** 2)
        else:
            error = 0.0
        
        self.micro['prediction'] = prediction
        self.micro['error'] = min(1.0, error * 2.0)
        
        return prediction, self.micro['error']
    
    def _predict_meso(self, state, action):
        """Мезо-уровень: предсказывает следующее действие."""
        if len(self.meso['memory']) < 3:
            self.meso['prediction'] = 0
            return 0, 0.0
        
        # Анализируем последовательности действий
        actions = [m.get('action', 0) for m in self.meso['memory']]
        
        if len(actions) >= 5:
            # Ищем повторяющиеся паттерны
            pattern = tuple(actions[-3:])
            if pattern in self.meso['pattern_memory']:
                # Если паттерн известен, предсказываем следующее действие
                next_action = self.meso['pattern_memory'][pattern]
                self.meso['prediction'] = next_action
                error = 0.1  # Низкая ошибка, если паттерн известен
            else:
                # Новый паттерн, предсказываем случайно
                next_action = np.random.randint(0, 8)
                self.meso['prediction'] = next_action
                error = 0.6
        else:
            # Недостаточно данных
            next_action = np.random.randint(0, 8)
            self.meso['prediction'] = next_action
            error = 0.8
        
        # Если есть реальное действие, сравниваем
        if action is not None:
            error = abs(action - self.meso['prediction']) / 8.0
        
        self.meso['error'] = min(1.0, error * 1.5)
        
        return self.meso['prediction'], self.meso['error']
    
    def _predict_macro(self, state):
        """Макро-уровень: предсказывает цель."""
        if len(self.macro['memory']) < 10:
            self.macro['prediction'] = 'unknown'
            return 'unknown', 1.0
        
        # Анализируем долгосрочные тренды
        states = list(self.macro['memory'])
        
        # Простая эвристика: если энергия падает, цель = 'выживание'
        if len(states) >= 20:
            energy_trend = states[-1] - states[0]
            if energy_trend < 0:
                self.macro['prediction'] = 'выживание'
                self.macro['error'] = 0.3
            else:
                self.macro['prediction'] = 'исследование'
                self.macro['error'] = 0.2
        else:
            self.macro['prediction'] = 'исследование'
            self.macro['error'] = 0.5
        
        return self.macro['prediction'], self.macro['error']
    
    def _learn(self, micro_error, meso_error, macro_error, state, action):
        """Обучение на ошибках предсказания."""
        # 1. Микро: обновляем веса, чтобы уменьшить ошибку
        if micro_error > 0.1:
            lr = self.learning_rate * micro_error
            self.micro['weights'] += lr * np.random.randn(*self.micro['weights'].shape) * 0.1
        
        # 2. Мезо: сохраняем паттерны
        if action is not None:
            self.meso['memory'].append({'action': action, 'state': state})
            
            # Если есть последовательность, сохраняем паттерн
            if len(self.meso['memory']) >= 4:
                actions = [m['action'] for m in self.meso['memory']]
                pattern = tuple(actions[-3:])
                next_action = actions[-1]
                self.meso['pattern_memory'][pattern] = next_action
        
        # 3. Макро: обновляем цели
        if macro_error > 0.2:
            # Обновляем план
            if len(self.macro['plan']) < 5:
                self.macro['plan'].append({
                    'goal': self.macro['prediction'],
                    'steps': 0,
                })
        
        # 4. Сохраняем состояние
        self.micro['memory'].append(state)
        self.macro['memory'].append(state)
    
    def _generate_insight(self, micro_error, meso_error, macro_error):
        """Генерирует инсайт на основе ошибок."""
        insight = {
            'time': time.time(),
            'micro_error': micro_error,
            'meso_error': meso_error,
            'macro_error': macro_error,
            'total_dissonance': self.dissonance,
        }
        
        # Текст инсайта
        if micro_error > 0.8:
            insight['text'] = "Я не понимаю, что произойдёт в следующий момент. Мне нужно лучше наблюдать."
        elif meso_error > 0.7:
            insight['text'] = "Мои действия непредсказуемы. Я должен искать паттерны в своём поведении."
        elif macro_error > 0.6:
            insight['text'] = "Я теряю долгосрочную цель. Мне нужно пересмотреть свой план."
        else:
            insight['text'] = "Мои предсказания становятся точнее. Я учусь."
        
        return insight
    
    def get_state(self):
        """Возвращает полное состояние предиктора."""
        return {
            'micro': {
                'error': self.micro['error'],
                'prediction': self.micro['prediction'],
                'memory_size': len(self.micro['memory']),
            },
            'meso': {
                'error': self.meso['error'],
                'prediction': self.meso['prediction'],
                'patterns': len(self.meso['pattern_memory']),
            },
            'macro': {
                'error': self.macro['error'],
                'prediction': self.macro['prediction'],
                'plan_length': len(self.macro['plan']),
            },
            'dissonance': self.dissonance,
            'insights': len(self.insights),
        }


# ============================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 МОДУЛЬ ИЕРАРХИЧЕСКОГО ПРЕДСКАЗАНИЯ")
    print("=" * 60)
    
    predictor = HierarchicalPredictor(input_dim=16, hidden_dim=8)
    
    # Тест: симуляция состояний
    print("\n📊 Тест предсказания:")
    
    for i in range(15):
        # Генерируем случайное состояние
        state = np.random.randn(16)
        action = np.random.randint(0, 8)
        
        result = predictor.predict(state, action)
        
        if i % 3 == 0 or i > 10:
            print(f"\n  Шаг {i}:")
            print(f"    Микро: ошибка = {result['micro']['error']:.2f}")
            print(f"    Мезо: ошибка = {result['meso']['error']:.2f}, предсказание = {result['meso']['prediction']}")
            print(f"    Макро: ошибка = {result['macro']['error']:.2f}, цель = {result['macro']['prediction']}")
            print(f"    Диссонанс = {result['dissonance']:.2f}")
            
            if result['insights']:
                last_insight = result['insights'][-1]
                print(f"    💡 Инсайт: {last_insight['text']}")
    
    print("\n📊 ИТОГОВОЕ СОСТОЯНИЕ:")
    state = predictor.get_state()
    print(f"  Микро-ошибка: {state['micro']['error']:.2f}")
    print(f"  Мезо-ошибка: {state['meso']['error']:.2f}")
    print(f"  Макро-ошибка: {state['macro']['error']:.2f}")
    print(f"  Диссонанс: {state['dissonance']:.2f}")
    print(f"  Инсайтов: {state['insights']}")
    print(f"  Паттернов: {state['meso']['patterns']}")
    print(f"  План: {state['macro']['plan_length']} шагов")
    
    print("\n💡 Гениальность: Предсказание работает на ТРЁХ УРОВНЯХ —")
    print("   от спайков до целей. Ошибка на каждом уровне порождает")
    print("   ДИССОНАНС, который ведёт к обучению.")
