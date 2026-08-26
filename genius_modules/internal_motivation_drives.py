# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ВНУТРЕННЯЯ МОТИВАЦИЯ ВМЕСТО ВНЕШНЕЙ НАГРАДЫ
Гениальность: Агент действует не ради +1, а потому что НЕ МОЖЕТ ИНАЧЕ.
Голод — это императив, а не награда.

Внутренние драйверы:
- Голод (Body.hunger) — поиск еды
- Страх (LimbicSystem.fear) — избегание опасности
- Любопытство (Curiosity) — исследование новизны
- Боль (PainModule) — долговременное избегание

Это делает поведение органичным, а не оппортунистическим.
"""

import numpy as np
from collections import deque
import time

class InternalDrive:
    """Базовый класс для внутренних драйверов."""
    def __init__(self, name, initial=0.0, decay=0.01, threshold=0.5):
        self.name = name
        self.value = initial
        self.decay = decay
        self.threshold = threshold
        self.history = deque(maxlen=100)
        self.last_update = time.time()

    def update(self, delta=0.0):
        """Обновляет значение драйвера."""
        dt = time.time() - self.last_update
        self.value = min(1.0, max(0.0, self.value + delta - self.decay * dt))
        self.history.append(self.value)
        self.last_update = time.time()
        return self.value

    def is_urgent(self):
        """Проверяет, превышает ли драйвер порог."""
        return self.value > self.threshold

    def get_state(self):
        return {'name': self.name, 'value': self.value, 'threshold': self.threshold, 'urgent': self.is_urgent()}

class HungerDrive(InternalDrive):
    """Голод — императив поиска еды."""
    def __init__(self, initial=0.2, decay=0.005, threshold=0.6):
        super().__init__('hunger', initial, decay, threshold)
        self.food_seen = 0
        self.last_meal_time = time.time()

    def eat(self, amount=0.3):
        """Утоляет голод."""
        self.value = max(0.0, self.value - amount)
        self.last_meal_time = time.time()
        self.food_seen += 1

    def update(self, delta=0.0):
        """Голод растёт со временем и усиливается при активности."""
        dt = time.time() - self.last_update
        base_increase = self.decay * dt
        activity_bonus = delta * 0.01
        self.value = min(1.0, self.value + base_increase + activity_bonus)
        self.history.append(self.value)
        self.last_update = time.time()
        return self.value

class FearDrive(InternalDrive):
    """Страх — избегание опасности."""
    def __init__(self, initial=0.1, decay=0.02, threshold=0.5):
        super().__init__('fear', initial, decay, threshold)
        self.threat_encounters = 0
        self.quick_fear = 0.0      # быстрый страх (амигдала)
        self.medium_fear = 0.0     # среднесрочный (гипоталамус)
        self.long_fear = 0.0       # долгосрочный (гиппокамп+кора)

    def encounter_threat(self, intensity=0.5):
        """Встреча с угрозой."""
        self.quick_fear = min(1.0, self.quick_fear + intensity * 0.8)
        self.medium_fear = min(1.0, self.medium_fear + intensity * 0.4)
        self.long_fear = min(1.0, self.long_fear + intensity * 0.2)
        self.threat_encounters += 1
        self.update()

    def update(self, delta=0.0):
        """Страх затухает с разной скоростью."""
        dt = time.time() - self.last_update
        # Быстрый страх затухает быстро
        self.quick_fear = max(0.0, self.quick_fear - 0.1 * dt)
        # Среднесрочный — медленнее
        self.medium_fear = max(0.0, self.medium_fear - 0.02 * dt)
        # Долгосрочный — очень медленно
        self.long_fear = max(0.0, self.long_fear - 0.005 * dt)
        # Общий страх — сумма
        self.value = min(1.0, self.quick_fear * 0.5 + self.medium_fear * 0.3 + self.long_fear * 0.2)
        self.history.append(self.value)
        self.last_update = time.time()
        return self.value

class CuriosityDrive(InternalDrive):
    """Любопытство — исследование новизны."""
    def __init__(self, initial=0.3, decay=0.01, threshold=0.4):
        super().__init__('curiosity', initial, decay, threshold)
        self.novelty_seen = 0
        self.exploration_urge = 0.5
        self.boredom = 0.0

    def encounter_novelty(self, novelty_level=0.5):
        """Встреча с новизной."""
        self.novelty_seen += 1
        self.value = min(1.0, self.value + novelty_level * 0.3)
        self.exploration_urge = min(1.0, self.exploration_urge + novelty_level * 0.1)
        self.boredom = max(0.0, self.boredom - novelty_level * 0.2)

    def update(self, delta=0.0):
        """Любопытство затухает при отсутствии новизны, растёт от скуки."""
        dt = time.time() - self.last_update
        # Если нет новизны, любопытство падает, а скука растёт
        if self.novelty_seen < 3:
            self.boredom = min(1.0, self.boredom + 0.01 * dt)
            self.value = max(0.0, self.value - self.decay * dt * 0.5)
        else:
            self.boredom = max(0.0, self.boredom - 0.01 * dt)
            self.value = min(1.0, self.value + self.boredom * 0.1 * dt)
        # Со временем любопытство восстанавливается
        self.value = min(1.0, self.value + 0.005 * dt)
        self.history.append(self.value)
        self.last_update = time.time()
        return self.value

class PainDrive(InternalDrive):
    """Боль — долговременное избегание."""
    def __init__(self, initial=0.0, decay=0.01, threshold=0.4):
        super().__init__('pain', initial, decay, threshold)
        self.pain_events = []
        self.pain_memory = deque(maxlen=20)

    def experience_pain(self, intensity=0.5, duration=1.0):
        """Опыт боли."""
        self.value = min(1.0, self.value + intensity * 0.3)
        self.pain_events.append({'intensity': intensity, 'time': time.time(), 'duration': duration})
        self.pain_memory.append((intensity, time.time()))

    def update(self, delta=0.0):
        """Боль медленно затухает."""
        dt = time.time() - self.last_update
        # Боль затухает, но медленно — защита от повторных травм
        self.value = max(0.0, self.value - self.decay * dt * 0.3)
        # Если были недавние боли, остаётся повышенная чувствительность
        recent_pain = sum(1 for _, t in self.pain_memory if time.time() - t < 10)
        if recent_pain > 0:
            self.value = min(1.0, self.value + recent_pain * 0.05)
        self.history.append(self.value)
        self.last_update = time.time()
        return self.value

class InternalMotivationSystem:
    """Система внутренней мотивации — все драйверы вместе."""
    def __init__(self):
        self.hunger = HungerDrive()
        self.fear = FearDrive()
        self.curiosity = CuriosityDrive()
        self.pain = PainDrive()
        self.drives = [self.hunger, self.fear, self.curiosity, self.pain]
        self.priority_drive = None
        self.action_history = deque(maxlen=50)

    def update_all(self, delta=0.0):
        """Обновляет все драйверы."""
        for drive in self.drives:
            drive.update(delta)
        self._determine_priority()
        return self.get_state()

    def _determine_priority(self):
        """Определяет, какой драйвер сейчас самый сильный."""
        urgent_drives = [d for d in self.drives if d.is_urgent()]
        if urgent_drives:
            # Самый сильный драйвер
            self.priority_drive = max(urgent_drives, key=lambda d: d.value)
        else:
            # Самый активный драйвер (любопытство по умолчанию)
            self.priority_drive = max(self.drives, key=lambda d: d.value)
        return self.priority_drive

    def get_action_urge(self):
        """Возвращает побуждение к действию на основе драйверов."""
        if self.priority_drive:
            return {
                'drive': self.priority_drive.name,
                'intensity': self.priority_drive.value,
                'urgent': self.priority_drive.is_urgent()
            }
        return {'drive': 'none', 'intensity': 0.0, 'urgent': False}

    def get_state(self):
        """Возвращает состояние системы."""
        return {
            'hunger': self.hunger.get_state(),
            'fear': self.fear.get_state(),
            'curiosity': self.curiosity.get_state(),
            'pain': self.pain.get_state(),
            'priority_drive': self.priority_drive.name if self.priority_drive else None,
            'action_urge': self.get_action_urge()
        }

if __name__ == "__main__":
    print("="*60)
    print("🍽️ ВНУТРЕННЯЯ МОТИВАЦИЯ")
    print("="*60)
    mot = InternalMotivationSystem()
    # Симуляция
    for i in range(10):
        if i == 2:
            mot.hunger.eat(-0.3)  # голод растёт
        if i == 4:
            mot.fear.encounter_threat(0.8)  # страх
        if i == 6:
            mot.curiosity.encounter_novelty(0.9)  # новизна
        if i == 8:
            mot.pain.experience_pain(0.6)  # боль
        state = mot.update_all()
        urge = mot.get_action_urge()
        print(f"Шаг {i}: приоритет={urge['drive']}, интенсивность={urge['intensity']:.2f}")
    print("\n💡 Гениальность: Агент действует не ради +1, а потому что НЕ МОЖЕТ ИНАЧЕ.")
    print("   Голод — это императив, а не награда.")
