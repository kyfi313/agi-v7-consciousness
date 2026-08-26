# -*- coding: utf-8 -*-
"""
МОДУЛЬ: МОЗГ КАК ГЕНЕРАТОР ПАТТЕРНОВ
Гениальность: Мозг — это не калькулятор, а генератор паттернов.
Он не вычисляет ответы, а создаёт шаблоны поведения и мышления.

Эти паттерны могут быть:
- Простые (рефлексы)
- Сложные (социальные стратегии)
- Абстрактные (математика, философия)

Реализовано: PatternGenerator — генерирует паттерны на основе прошлого опыта,
PatternRecognizer — распознаёт паттерны во входных данных,
PatternAssociator — связывает паттерны между собой.
"""

import numpy as np
from collections import deque
import random
import time

class Pattern:
    """Один паттерн — шаблон поведения или мысли."""
    def __init__(self, name, vector, strength=0.5, abstraction_level=0.3):
        self.name = name
        self.vector = np.array(vector)
        self.strength = strength
        self.abstraction_level = abstraction_level  # 0 = конкретный, 1 = абстрактный
        self.activation = 0.0
        self.associations = {}  # pattern_name -> weight
        self.usage_count = 0
        self.last_used = time.time()
        self.creativity_score = random.random() * 0.5

    def activate(self, input_vector):
        """Вычисляет активацию паттерна на основе входа."""
        similarity = np.dot(input_vector, self.vector) / (np.linalg.norm(input_vector) * np.linalg.norm(self.vector) + 1e-8)
        self.activation = similarity * self.strength
        self.usage_count += 1
        self.last_used = time.time()
        return self.activation

    def associate(self, other_pattern, weight=0.1):
        """Связывает паттерн с другим."""
        self.associations[other_pattern.name] = weight

    def get_state(self):
        return {
            'name': self.name,
            'strength': self.strength,
            'abstraction': self.abstraction_level,
            'activation': self.activation,
            'associations': len(self.associations),
            'usage': self.usage_count
        }

class PatternGenerator:
    """Генерирует паттерны на основе прошлого опыта."""
    def __init__(self):
        self.patterns = []
        self.generation_history = deque(maxlen=50)
        self.creativity = 0.5
        self.temperature = 0.5

    def generate(self, context_vector=None, existing_patterns=None):
        """Генерирует новый паттерн."""
        if context_vector is None:
            context_vector = np.random.randn(5) * 0.5

        # Генерация нового паттерна на основе контекста
        if existing_patterns and len(existing_patterns) > 0:
            # Комбинируем существующие паттерны
            base_pattern = random.choice(existing_patterns)
            new_vector = base_pattern.vector.copy()
            # Добавляем шум и креативность
            noise = np.random.randn(len(new_vector)) * (1 - self.creativity) * 0.3
            new_vector += noise
            # Нормализация
            new_vector = new_vector / (np.linalg.norm(new_vector) + 1e-8)
            abstraction = min(1.0, base_pattern.abstraction_level + random.random() * 0.2 - 0.1)
            strength = max(0.1, base_pattern.strength + random.random() * 0.2 - 0.1)
            name = f"generated_{len(self.patterns)}_{int(time.time())}"
        else:
            # Полностью новый паттерн
            new_vector = context_vector + np.random.randn(len(context_vector)) * 0.5
            new_vector = new_vector / (np.linalg.norm(new_vector) + 1e-8)
            abstraction = random.random()
            strength = random.random() * 0.8 + 0.2
            name = f"novel_{len(self.patterns)}_{int(time.time())}"

        pattern = Pattern(name, new_vector, strength, abstraction)
        self.patterns.append(pattern)
        self.generation_history.append({
            'time': time.time(),
            'name': pattern.name,
            'abstraction': abstraction
        })
        return pattern

    def mutate_pattern(self, pattern):
        """Мутирует существующий паттерн (креативная модификация)."""
        if pattern not in self.patterns:
            return None
        mutation_strength = self.creativity * 0.3 + 0.1
        new_vector = pattern.vector + np.random.randn(len(pattern.vector)) * mutation_strength
        new_vector = new_vector / (np.linalg.norm(new_vector) + 1e-8)
        # Создаём новый паттерн на основе мутации
        mutated = Pattern(
            f"mutated_{pattern.name}_{int(time.time())}",
            new_vector,
            min(1.0, pattern.strength * 1.1),
            min(1.0, pattern.abstraction_level + 0.1)
        )
        self.patterns.append(mutated)
        return mutated

    def get_state(self):
        return {
            'pattern_count': len(self.patterns),
            'creativity': self.creativity,
            'generation_count': len(self.generation_history),
            'avg_abstraction': np.mean([p.abstraction_level for p in self.patterns]) if self.patterns else 0
        }

class PatternRecognizer:
    """Распознаёт паттерны во входных данных."""
    def __init__(self):
        self.pattern_library = {}
        self.recognition_history = deque(maxlen=30)
        self.threshold = 0.3

    def add_pattern(self, pattern):
        """Добавляет паттерн в библиотеку."""
        self.pattern_library[pattern.name] = pattern

    def recognize(self, input_vector):
        """Распознаёт паттерны во входном векторе."""
        recognized = []
        for name, pattern in self.pattern_library.items():
            activation = pattern.activate(input_vector)
            if activation > self.threshold:
                recognized.append({
                    'name': name,
                    'activation': activation,
                    'abstraction': pattern.abstraction_level,
                    'strength': pattern.strength
                })
        # Сортируем по активации
        recognized.sort(key=lambda x: x['activation'], reverse=True)
        self.recognition_history.append({
            'time': time.time(),
            'found': len(recognized),
            'top': recognized[0]['name'] if recognized else None
        })
        return recognized

    def get_state(self):
        return {
            'library_size': len(self.pattern_library),
            'threshold': self.threshold,
            'history_size': len(self.recognition_history),
            'last_recognition': self.recognition_history[-1] if self.recognition_history else None
        }

class PatternAssociator:
    """Связывает паттерны между собой."""
    def __init__(self):
        self.associations = {}  # (p1_name, p2_name) -> weight
        self.association_history = deque(maxlen=30)
        self.strength = 0.5

    def associate(self, pattern1, pattern2, weight=None):
        """Создаёт ассоциацию между двумя паттернами."""
        if weight is None:
            # Вес зависит от силы паттернов
            weight = (pattern1.strength + pattern2.strength) / 2 * 0.3
        key = tuple(sorted([pattern1.name, pattern2.name]))
        self.associations[key] = weight
        # Также сохраняем ассоциацию в самих паттернах
        pattern1.associate(pattern2, weight)
        pattern2.associate(pattern1, weight)
        self.association_history.append({
            'time': time.time(),
            'p1': pattern1.name,
            'p2': pattern2.name,
            'weight': weight
        })

    def get_related(self, pattern, max_related=5):
        """Возвращает паттерны, связанные с данным."""
        related = []
        for (p1, p2), weight in self.associations.items():
            if p1 == pattern.name:
                related.append((p2, weight))
            elif p2 == pattern.name:
                related.append((p1, weight))
        # Сортируем по весу
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_related]

    def get_state(self):
        return {
            'association_count': len(self.associations),
            'strength': self.strength,
            'history_size': len(self.association_history)
        }

class BrainPatternSystem:
    """Полная система генерации, распознавания и ассоциации паттернов."""
    def __init__(self):
        self.generator = PatternGenerator()
        self.recognizer = PatternRecognizer()
        self.associator = PatternAssociator()
        self.initialize_seed_patterns()

    def initialize_seed_patterns(self):
        """Создаёт начальные паттерны."""
        seed_patterns = [
            ("базовый_безопасности", [1.0, 0.0, 0.0, 0.0, 0.0], 0.7, 0.1),
            ("базовый_голода", [0.0, 1.0, 0.0, 0.0, 0.0], 0.6, 0.1),
            ("базовый_социальный", [0.0, 0.0, 1.0, 0.0, 0.0], 0.5, 0.2),
            ("базовый_исследования", [0.0, 0.0, 0.0, 1.0, 0.0], 0.4, 0.3),
            ("базовый_творчества", [0.0, 0.0, 0.0, 0.0, 1.0], 0.3, 0.5)
        ]
        for name, vector, strength, abstraction in seed_patterns:
            pattern = Pattern(name, vector, strength, abstraction)
            self.generator.patterns.append(pattern)
            self.recognizer.add_pattern(pattern)

    def process_input(self, input_vector):
        """Обрабатывает входные данные и генерирует новые паттерны."""
        # 1. Распознаём паттерны
        recognized = self.recognizer.recognize(input_vector)
        # 2. Если ничего не распознано — генерируем новый паттерн
        if not recognized:
            new_pattern = self.generator.generate(input_vector, self.generator.patterns)
            self.recognizer.add_pattern(new_pattern)
            return {'action': 'generated', 'pattern': new_pattern.name}
        # 3. Если распознано — активируем ассоциации
        else:
            top_pattern_name = recognized[0]['name']
            top_pattern = next((p for p in self.generator.patterns if p.name == top_pattern_name), None)
            if top_pattern:
                related = self.associator.get_related(top_pattern)
                if related:
                    return {
                        'action': 'associated',
                        'top_pattern': top_pattern_name,
                        'related': related[:3]
                    }
                # Создаём ассоциацию с похожими паттернами
                for other in self.generator.patterns:
                    if other.name != top_pattern_name:
                        similarity = np.dot(top_pattern.vector, other.vector) / (
                            np.linalg.norm(top_pattern.vector) * np.linalg.norm(other.vector) + 1e-8
                        )
                        if similarity > 0.5:
                            self.associator.associate(top_pattern, other, similarity * 0.1)
                return {'action': 'associated_created', 'top_pattern': top_pattern_name}
        return {'action': 'no_pattern'}

    def get_state(self):
        return {
            'generator': self.generator.get_state(),
            'recognizer': self.recognizer.get_state(),
            'associator': self.associator.get_state(),
            'total_patterns': len(self.generator.patterns)
        }

if __name__ == "__main__":
    print("="*60)
    print("🧠 МОЗГ КАК ГЕНЕРАТОР ПАТТЕРНОВ")
    print("="*60)
    brain = BrainPatternSystem()
    # Симуляция
    for i in range(10):
        input_vector = np.random.randn(5) * 0.8
        result = brain.process_input(input_vector)
        print(f"Шаг {i}: {result}")
        # Иногда мутируем паттерны
        if i % 3 == 0 and brain.generator.patterns:
            pattern = random.choice(brain.generator.patterns)
            brain.generator.mutate_pattern(pattern)
    state = brain.get_state()
    print(f"\nВсего паттернов: {state['total_patterns']}")
    print(f"Ассоциаций: {state['associator']['association_count']}")
    print("\n💡 Гениальность: Мозг — это не калькулятор, а ГЕНЕРАТОР ПАТТЕРНОВ.")
    print("   Он создаёт шаблоны поведения и мышления, а не вычисляет ответы.")
