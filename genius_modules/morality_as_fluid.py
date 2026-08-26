# -*- coding: utf-8 -*-
"""
МОДУЛЬ: МОРАЛЬ КАК НЬЮТОНОВСКАЯ ЖИДКОСТЬ
Гениальность: Мораль — это не набор правил, а динамическая система,
которая течёт, как жидкость, под давлением обстоятельств.

Мораль имеет вязкость (сопротивление изменениям) и пластичность.
Она может быть твёрдой (ригидная) или жидкой (адаптивная).

Реализовано: MoralFluid — мораль как жидкость с вязкостью,
MoralContainer — контейнер моральных принципов,
MoralFlowSystem — система моральной динамики.
"""

import numpy as np
from collections import deque
import time
import random

class MoralPrinciple:
    """Один моральный принцип."""
    def __init__(self, name, base_strength=0.5, viscosity=0.3):
        self.name = name
        self.base_strength = base_strength
        self.current_strength = base_strength
        self.viscosity = viscosity  # сопротивление изменению
        self.history = deque(maxlen=30)
        self.temperature = 0.5  # влияние эмоций
        self.last_update = time.time()

    def apply_pressure(self, force, direction='positive'):
        """Применяет давление к принципу."""
        dt = time.time() - self.last_update
        # Принцип меняется пропорционально силе и обратно пропорционально вязкости
        change = force / (self.viscosity + 0.01) * dt * 0.1
        if direction == 'positive':
            self.current_strength = min(1.0, self.current_strength + change)
        else:
            self.current_strength = max(0.0, self.current_strength - change)
        self.history.append(self.current_strength)
        self.last_update = time.time()
        return self.current_strength

    def get_state(self):
        return {
            'name': self.name,
            'strength': self.current_strength,
            'base': self.base_strength,
            'viscosity': self.viscosity,
            'temperature': self.temperature
        }

class MoralFluid:
    """Мораль как жидкость с вязкостью."""
    def __init__(self):
        self.principles = {}
        self.fluid_temperature = 0.5
        self.pressure_history = deque(maxlen=20)
        self.fluidity = 0.5  # 0 = твёрдая, 1 = жидкая

    def add_principle(self, name, base_strength=0.5, viscosity=0.3):
        """Добавляет моральный принцип."""
        self.principles[name] = MoralPrinciple(name, base_strength, viscosity)
        return self.principles[name]

    def apply_social_pressure(self, principle_name, force, direction='positive'):
        """Применяет социальное давление к принципу."""
        if principle_name not in self.principles:
            return None
        principle = self.principles[principle_name]
        # Давление ослабляется вязкостью и усиливается температурой
        effective_force = force * (1 + self.fluid_temperature * 0.5)
        result = principle.apply_pressure(effective_force, direction)
        self.pressure_history.append({
            'time': time.time(),
            'principle': principle_name,
            'force': effective_force,
            'result': result
        })
        return result

    def set_temperature(self, temperature):
        """Устанавливает температуру моральной жидкости."""
        self.fluid_temperature = max(0.0, min(1.0, temperature))
        self.fluidity = self.fluid_temperature

    def get_state(self):
        return {
            'principles': {name: p.get_state() for name, p in self.principles.items()},
            'temperature': self.fluid_temperature,
            'fluidity': self.fluidity,
            'pressure_count': len(self.pressure_history)
        }

class MoralContainer:
    """Контейнер моральных принципов."""
    def __init__(self):
        self.principles = {}
        self.conflicts = deque(maxlen=20)
        self.resolution_history = deque(maxlen=20)

    def add_principle(self, name, strength=0.5):
        self.principles[name] = {'strength': strength, 'active': True}

    def detect_conflict(self, p1, p2):
        """Обнаруживает конфликт между принципами."""
        if p1 not in self.principles or p2 not in self.principles:
            return 0.0
        # Конфликт = оба активны и имеют высокую силу
        if self.principles[p1]['active'] and self.principles[p2]['active']:
            conflict = (self.principles[p1]['strength'] + self.principles[p2]['strength']) / 2
            if conflict > 0.6:
                self.conflicts.append((p1, p2, time.time()))
                return conflict
        return 0.0

    def resolve_conflict(self, p1, p2, resolution='compromise'):
        """Разрешает конфликт между принципами."""
        if resolution == 'compromise':
            # Компромисс
            avg = (self.principles[p1]['strength'] + self.principles[p2]['strength']) / 2
            self.principles[p1]['strength'] = avg * 0.8
            self.principles[p2]['strength'] = avg * 0.8
        elif resolution == 'dominance':
            # Доминирование одного
            if self.principles[p1]['strength'] > self.principles[p2]['strength']:
                self.principles[p2]['active'] = False
            else:
                self.principles[p1]['active'] = False
        self.resolution_history.append((p1, p2, resolution, time.time()))

    def get_state(self):
        return {
            'principles': self.principles,
            'conflicts': len(self.conflicts),
            'resolutions': len(self.resolution_history)
        }

class MoralFlowSystem:
    """Полная система моральной динамики."""
    def __init__(self):
        self.fluid = MoralFluid()
        self.container = MoralContainer()
        self.initialize_principles()

    def initialize_principles(self):
        """Инициализирует базовые моральные принципы."""
        principles = [
            ('честность', 0.7, 0.2),
            ('справедливость', 0.6, 0.3),
            ('сострадание', 0.5, 0.4),
            ('лояльность', 0.5, 0.5),
            ('свобода', 0.4, 0.6),
            ('безопасность', 0.6, 0.3)
        ]
        for name, strength, viscosity in principles:
            self.fluid.add_principle(name, strength, viscosity)
            self.container.add_principle(name, strength)

    def process_situation(self, situation):
        """Обрабатывает ситуацию, применяя моральную динамику."""
        # 1. Определяем, какие принципы затронуты
        affected = [p for p in self.fluid.principles if p in situation.get('affected', [])]
        # 2. Применяем давление
        for p in affected:
            force = situation.get('force', 0.3)
            direction = situation.get('direction', 'positive')
            self.fluid.apply_social_pressure(p, force, direction)
        # 3. Обнаруживаем конфликты
        conflicts = []
        for p1 in affected:
            for p2 in affected:
                if p1 != p2:
                    conflict = self.container.detect_conflict(p1, p2)
                    if conflict > 0.5:
                        conflicts.append((p1, p2))
        # 4. Разрешаем конфликты
        for p1, p2 in conflicts:
            self.container.resolve_conflict(p1, p2, 'compromise')
        # 5. Обновляем температуру
        self.fluid.set_temperature(0.3 + len(conflicts) * 0.1)
        return {
            'affected': affected,
            'conflicts': len(conflicts),
            'temperature': self.fluid.fluid_temperature
        }

    def get_state(self):
        return {
            'fluid': self.fluid.get_state(),
            'container': self.container.get_state()
        }

if __name__ == "__main__":
    print("="*60)
    print("🌊 МОРАЛЬ КАК НЬЮТОНОВСКАЯ ЖИДКОСТЬ")
    print("="*60)
    system = MoralFlowSystem()
    print("Начальное состояние:")
    for name, p in system.fluid.principles.items():
        print(f"  {name}: {p.current_strength:.2f} (вязкость={p.viscosity:.2f})")
    # Ситуации
    situations = [
        {'affected': ['честность', 'справедливость'], 'force': 0.6, 'direction': 'positive'},
        {'affected': ['сострадание', 'лояльность'], 'force': 0.5, 'direction': 'positive'},
        {'affected': ['свобода', 'безопасность'], 'force': 0.7, 'direction': 'negative'},
        {'affected': ['честность', 'лояльность'], 'force': 0.4, 'direction': 'negative'}
    ]
    for i, situation in enumerate(situations):
        result = system.process_situation(situation)
        print(f"\nСитуация {i+1}: конфликтов={result['conflicts']}, температура={result['temperature']:.2f}")
    print("\nКонечное состояние:")
    for name, p in system.fluid.principles.items():
        print(f"  {name}: {p.current_strength:.2f}")
    print("\n💡 Гениальность: Мораль — это не набор правил, а ДИНАМИЧЕСКАЯ СИСТЕМА.")
    print("   Она течёт, как жидкость, под давлением обстоятельств.")
