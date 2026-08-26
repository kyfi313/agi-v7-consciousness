# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ДИСКОМФОРТ КАК ДРАЙВЕР РАЗВИТИЯ
Гениальность: Дискомфорт — это не враг, а сигнал к росту.
Именно дискомфорт заставляет мозг менять паттерны и искать новые пути.

Без дискомфорта нет развития. Комфорт — это застой.

Реализовано: DiscomfortSignal — сигнал дискомфорта,
GrowthDrive — драйвер роста,
DiscomfortGrowthSystem — система дискомфорта и развития.
"""

import numpy as np
from collections import deque
import time
import random

class DiscomfortSignal:
    """Сигнал дискомфорта."""
    def __init__(self, source, intensity=0.5, duration=1.0):
        self.source = source  # 'physical', 'cognitive', 'social', 'existential'
        self.intensity = intensity
        self.duration = duration
        self.timestamp = time.time()
        self.resolved = False
        self.resolution_time = None

    def resolve(self):
        """Разрешает сигнал дискомфорта."""
        self.resolved = True
        self.resolution_time = time.time()

    def get_state(self):
        return {
            'source': self.source,
            'intensity': self.intensity,
            'duration': self.duration,
            'resolved': self.resolved,
            'age': time.time() - self.timestamp
        }

class GrowthDrive:
    """Драйвер роста, активируемый дискомфортом."""
    def __init__(self):
        self.discomfort_signals = deque(maxlen=20)
        self.growth_opportunities = []
        self.growth_history = deque(maxlen=30)
        self.adaptation_rate = 0.3
        self.current_growth_urge = 0.0

    def add_discomfort(self, discomfort):
        """Добавляет сигнал дискомфорта."""
        self.discomfort_signals.append(discomfort)
        # Чем интенсивнее дискомфорт, тем сильнее побуждение к росту
        self.current_growth_urge = min(1.0, self.current_growth_urge + discomfort.intensity * 0.2)

    def seek_growth(self):
        """Ищет возможности для роста на основе дискомфорта."""
        if not self.discomfort_signals:
            return None
        # Находим самый сильный сигнал дискомфорта
        strongest = max(self.discomfort_signals, key=lambda d: d.intensity)
        if strongest.intensity > 0.5:
            # Генерируем возможность роста
            opportunity = {
                'source': strongest.source,
                'intensity': strongest.intensity,
                'action': self._suggest_action(strongest.source),
                'timestamp': time.time()
            }
            self.growth_opportunities.append(opportunity)
            self.growth_history.append(opportunity)
            return opportunity
        return None

    def _suggest_action(self, source):
        """Предлагает действие для роста."""
        actions = {
            'physical': ['упражнение', 'растяжка', 'отдых'],
            'cognitive': ['изучение', 'решение_задачи', 'медитация'],
            'social': ['общение', 'выражение_чувств', 'просьба_о_помощи'],
            'existential': ['рефлексия', 'смена_перспективы', 'принятие']
        }
        return random.choice(actions.get(source, ['адаптация']))

    def apply_growth(self, opportunity):
        """Применяет рост на основе возможности."""
        if opportunity['intensity'] > 0.7:
            # Сильный дискомфорт даёт большой рост
            growth_amount = opportunity['intensity'] * self.adaptation_rate
        else:
            growth_amount = opportunity['intensity'] * self.adaptation_rate * 0.5
        # Уменьшаем дискомфорт после роста
        self.current_growth_urge = max(0.0, self.current_growth_urge - growth_amount * 0.5)
        return growth_amount

    def get_state(self):
        return {
            'signal_count': len(self.discomfort_signals),
            'growth_urge': self.current_growth_urge,
            'opportunities': len(self.growth_opportunities),
            'adaptation_rate': self.adaptation_rate
        }

class DiscomfortGrowthSystem:
    """Полная система дискомфорта и развития."""
    def __init__(self):
        self.growth_drive = GrowthDrive()
        self.discomfort_tolerance = 0.5
        self.growth_level = 0.2
        self.history = deque(maxlen=30)

    def experience_discomfort(self, source, intensity=0.5, duration=1.0):
        """Испытывает дискомфорт."""
        # Проверяем, превышает ли дискомфорт допустимый уровень
        if intensity > self.discomfort_tolerance:
            # Сильный дискомфорт запускает процесс роста
            discomfort = DiscomfortSignal(source, intensity, duration)
            self.growth_drive.add_discomfort(discomfort)
            self.history.append({
                'time': time.time(),
                'source': source,
                'intensity': intensity,
                'action': 'growth_initiated'
            })
            return True
        else:
            # Терпимый дискомфорт
            self.history.append({
                'time': time.time(),
                'source': source,
                'intensity': intensity,
                'action': 'tolerated'
            })
            return False

    def grow(self):
        """Запускает процесс роста."""
        opportunity = self.growth_drive.seek_growth()
        if opportunity:
            growth_amount = self.growth_drive.apply_growth(opportunity)
            self.growth_level = min(1.0, self.growth_level + growth_amount)
            # Увеличиваем толерантность к дискомфорту
            self.discomfort_tolerance = min(1.0, self.discomfort_tolerance + growth_amount * 0.2)
            self.history.append({
                'time': time.time(),
                'growth': growth_amount,
                'opportunity': opportunity['source'],
                'action': opportunity['action']
            })
            return opportunity
        return None

    def get_state(self):
        return {
            'growth_drive': self.growth_drive.get_state(),
            'tolerance': self.discomfort_tolerance,
            'growth_level': self.growth_level,
            'history_size': len(self.history)
        }

if __name__ == "__main__":
    print("="*60)
    print("🔥 ДИСКОМФОРТ КАК ДРАЙВЕР РАЗВИТИЯ")
    print("="*60)
    system = DiscomfortGrowthSystem()
    sources = ['physical', 'cognitive', 'social', 'existential', 'physical', 'cognitive']
    intensities = [0.3, 0.6, 0.8, 0.4, 0.7, 0.9]
    for i, (source, intensity) in enumerate(zip(sources, intensities)):
        print(f"\nШаг {i}: {source} дискомфорт (интенсивность={intensity:.1f})")
        triggered = system.experience_discomfort(source, intensity)
        if triggered:
            opportunity = system.grow()
            if opportunity:
                print(f"  Рост: {opportunity['action']} (источник: {opportunity['source']})")
    state = system.get_state()
    print(f"\nУровень роста: {state['growth_level']:.2f}")
    print(f"Толерантность к дискомфорту: {state['tolerance']:.2f}")
    print("\n💡 Гениальность: Дискомфорт — это не враг, а СИГНАЛ К РОСТУ.")
    print("   Без дискомфорта нет развития. Комфорт — это застой.")
