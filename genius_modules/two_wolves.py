# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ДВА ВОЛКА — ХАРАКТЕР КАК БОРЬБА
Гениальность: Внутри каждого — два волка: один добрый, один злой.
Побеждает тот, которого кормят. Характер — это не сущность, а процесс выбора.

Реализовано: Wolf — внутренняя субличность,
WolfPack — группа волков,
CharacterBuilder — система формирования характера через кормление.
"""

import numpy as np
from collections import deque
import time
import random

class Wolf:
    """Внутренний волк — субличность."""
    def __init__(self, name, nature='neutral', strength=0.5):
        self.name = name
        self.nature = nature  # 'good', 'evil', 'neutral', 'wise', 'foolish'
        self.strength = strength
        self.fed_count = 0
        self.last_fed = 0.0
        self.hunger = 0.3
        self.influence = 0.0
        self.history = deque(maxlen=20)

    def feed(self, amount=0.1):
        """Кормит волка — усиливает его влияние."""
        self.strength = min(1.0, self.strength + amount)
        self.fed_count += 1
        self.last_fed = time.time()
        self.hunger = max(0.0, self.hunger - amount * 0.5)
        self.history.append({'action': 'feed', 'time': time.time(), 'strength': self.strength})
        return self.strength

    def starve(self, amount=0.05):
        """Морит волка голодом — ослабляет его."""
        self.strength = max(0.0, self.strength - amount)
        self.hunger = min(1.0, self.hunger + amount * 0.3)
        self.history.append({'action': 'starve', 'time': time.time(), 'strength': self.strength})
        return self.strength

    def get_urge(self):
        """Побуждение от волка."""
        # Чем сильнее волк, тем больше его влияние
        self.influence = self.strength * (1 + self.hunger * 0.2)
        return self.influence

    def get_state(self):
        return {
            'name': self.name,
            'nature': self.nature,
            'strength': self.strength,
            'hunger': self.hunger,
            'influence': self.influence,
            'fed_count': self.fed_count
        }

class WolfPack:
    """Стая волков — все внутренние субличности."""
    def __init__(self):
        self.wolves = {}
        self.dominant_wolf = None
        self.pack_history = deque(maxlen=30)

    def add_wolf(self, name, nature='neutral', strength=0.5):
        """Добавляет волка в стаю."""
        wolf = Wolf(name, nature, strength)
        self.wolves[name] = wolf
        return wolf

    def feed_wolf(self, name, amount=0.1):
        """Кормит конкретного волка."""
        if name in self.wolves:
            self.wolves[name].feed(amount)
            self._update_dominant()
            return True
        return False

    def starve_wolf(self, name, amount=0.05):
        """Морит волка голодом."""
        if name in self.wolves:
            self.wolves[name].starve(amount)
            self._update_dominant()
            return True
        return False

    def _update_dominant(self):
        """Обновляет доминирующего волка."""
        if not self.wolves:
            return
        # Доминирует волк с наибольшей силой
        dominant = max(self.wolves.values(), key=lambda w: w.strength)
        self.dominant_wolf = dominant.name
        self.pack_history.append({
            'time': time.time(),
            'dominant': self.dominant_wolf,
            'strength': dominant.strength
        })

    def get_urges(self):
        """Возвращает побуждения всех волков."""
        return {name: wolf.get_urge() for name, wolf in self.wolves.items()}

    def get_state(self):
        return {
            'wolves': {name: wolf.get_state() for name, wolf in self.wolves.items()},
            'dominant': self.dominant_wolf,
            'pack_size': len(self.wolves)
        }

class CharacterBuilder:
    """Система формирования характера через кормление волков."""
    def __init__(self):
        self.pack = WolfPack()
        self.character_traits = {}
        self.decision_history = deque(maxlen=50)
        self.initialize_wolves()

    def initialize_wolves(self):
        """Инициализирует базовую стаю волков."""
        wolves = [
            ('добрый', 'good', 0.5),
            ('злой', 'evil', 0.5),
            ('мудрый', 'wise', 0.4),
            ('глупый', 'foolish', 0.4),
            ('смелый', 'brave', 0.5),
            ('трусливый', 'cowardly', 0.5)
        ]
        for name, nature, strength in wolves:
            self.pack.add_wolf(name, nature, strength)

    def make_choice(self, situation, choice):
        """Принимает решение, которое кормит определённых волков."""
        # Выбор кормит соответствующих волков
        if choice == 'good':
            self.pack.feed_wolf('добрый', 0.15)
            self.pack.starve_wolf('злой', 0.1)
            self.pack.feed_wolf('мудрый', 0.05)
            self.pack.starve_wolf('глупый', 0.05)
        elif choice == 'evil':
            self.pack.feed_wolf('злой', 0.15)
            self.pack.starve_wolf('добрый', 0.1)
            self.pack.feed_wolf('глупый', 0.05)
            self.pack.starve_wolf('мудрый', 0.05)
        elif choice == 'brave':
            self.pack.feed_wolf('смелый', 0.15)
            self.pack.starve_wolf('трусливый', 0.1)
        elif choice == 'cowardly':
            self.pack.feed_wolf('трусливый', 0.15)
            self.pack.starve_wolf('смелый', 0.1)
        elif choice == 'wise':
            self.pack.feed_wolf('мудрый', 0.15)
            self.pack.starve_wolf('глупый', 0.1)

        self.decision_history.append({
            'time': time.time(),
            'situation': situation,
            'choice': choice,
            'dominant': self.pack.dominant_wolf
        })

        # Обновляем черты характера
        self._update_traits()
        return self.pack.dominant_wolf

    def _update_traits(self):
        """Обновляет черты характера на основе силы волков."""
        for name, wolf in self.pack.wolves.items():
            self.character_traits[name] = wolf.strength

    def get_character(self):
        """Возвращает текущий характер."""
        # Характер определяется доминирующим волком
        if self.pack.dominant_wolf:
            return {
                'dominant': self.pack.dominant_wolf,
                'traits': self.character_traits,
                'strength': self.pack.wolves[self.pack.dominant_wolf].strength
            }
        return {'dominant': None, 'traits': self.character_traits}

    def get_state(self):
        return {
            'pack': self.pack.get_state(),
            'traits': self.character_traits,
            'decisions': len(self.decision_history),
            'character': self.get_character()
        }

if __name__ == "__main__":
    print("="*60)
    print("🐺 ДВА ВОЛКА — ХАРАКТЕР КАК БОРЬБА")
    print("="*60)
    builder = CharacterBuilder()
    print("Начальный характер:", builder.get_character()['dominant'])
    # Симуляция выбора
    choices = ['good', 'good', 'brave', 'wise', 'good', 'evil', 'wise', 'good', 'brave', 'wise']
    for i, choice in enumerate(choices):
        situation = f"ситуация_{i}"
        dominant = builder.make_choice(situation, choice)
        print(f"Шаг {i}: выбор={choice}, доминирует={dominant}")
    print("\nКонечный характер:", builder.get_character()['dominant'])
    print("Черты:")
    for trait, strength in builder.character_traits.items():
        print(f"  {trait}: {strength:.2f}")
    print("\n💡 Гениальность: Внутри каждого — два волка. Побеждает тот, которого кормят.")
    print("   Характер — это не сущность, а процесс выбора.")
