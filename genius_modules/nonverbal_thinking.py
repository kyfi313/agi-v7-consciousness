# -*- coding: utf-8 -*-
"""
МОДУЛЬ: НЕВЕРБАЛЬНОЕ МЫШЛЕНИЕ
Гениальность: Мышление не сводится к языку. Есть мышление образами,
интуицией, паттернами, чувствами. Вербальное мышление — лишь надстройка.

Реализовано: ImageThinking — мышление образами,
IntuitiveThinking — интуитивное мышление,
PatternThinking — мышление паттернами,
NonverbalSystem — интеграция невербальных форм мышления.
"""

import numpy as np
from collections import deque
import time
import random

class ImageThinking:
    """Мышление образами — визуальное/пространственное."""
    def __init__(self, image_size=64):
        self.image_size = image_size
        self.mental_images = deque(maxlen=20)
        self.image_working_memory = deque(maxlen=5)
        self.image_manipulation_skill = 0.3

    def create_image(self, description=None):
        """Создаёт ментальный образ."""
        if description is None:
            # Генерируем случайный образ
            image = np.random.randn(self.image_size, self.image_size) * 0.5
        else:
            # Создаём образ на основе описания (заглушка)
            image = np.random.randn(self.image_size, self.image_size) * 0.3
            # Добавляем структуру в зависимости от описания
            if 'круг' in str(description).lower():
                center = self.image_size // 2
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        if (i - center)**2 + (j - center)**2 < (self.image_size//4)**2:
                            image[i, j] = 1.0
        self.mental_images.append(image)
        self.image_working_memory.append(image)
        return image

    def rotate_image(self, angle=90):
        """Поворачивает образ в воображении."""
        if not self.image_working_memory:
            return None
        current = self.image_working_memory[-1]
        # Простая симуляция поворота
        if angle == 90:
            rotated = np.rot90(current)
        elif angle == 180:
            rotated = np.rot90(np.rot90(current))
        else:
            rotated = current
        self.image_working_memory.append(rotated)
        self.mental_images.append(rotated)
        self.image_manipulation_skill = min(1.0, self.image_manipulation_skill + 0.01)
        return rotated

    def get_state(self):
        return {
            'image_count': len(self.mental_images),
            'wm_size': len(self.image_working_memory),
            'manipulation_skill': self.image_manipulation_skill
        }

class IntuitiveThinking:
    """Интуитивное мышление — чувство, предчувствие."""
    def __init__(self):
        self.intuitions = deque(maxlen=20)
        self.intuition_accuracy = 0.5
        self.intuition_strength = 0.3
        self.experience_count = 0

    def sense(self, context):
        """Интуитивное восприятие ситуации."""
        # Интуиция основана на опыте и контексте
        noise = random.random() * 0.3
        confidence = min(1.0, self.intuition_accuracy + noise)
        # Генерируем интуитивный сигнал
        signal = {
            'context': context[:10] if len(context) > 10 else context,
            'confidence': confidence,
            'feeling': 'right' if confidence > 0.5 else 'wrong',
            'strength': self.intuition_strength
        }
        self.intuitions.append(signal)
        self.experience_count += 1
        # Интуиция улучшается с опытом
        self.intuition_accuracy = min(1.0, self.intuition_accuracy + 0.01)
        return signal

    def trust(self):
        """Доверие к интуиции."""
        if self.intuitions:
            last = self.intuitions[-1]
            return last['confidence'] * self.intuition_strength
        return 0.3

    def get_state(self):
        return {
            'accuracy': self.intuition_accuracy,
            'strength': self.intuition_strength,
            'experience': self.experience_count,
            'intuitions': len(self.intuitions)
        }

class PatternThinking:
    """Мышление паттернами — распознавание и создание структур."""
    def __init__(self):
        self.patterns = {}
        self.pattern_history = deque(maxlen=30)
        self.pattern_recognition_skill = 0.4

    def add_pattern(self, name, pattern):
        """Добавляет паттерн."""
        self.patterns[name] = pattern
        self.pattern_history.append({'action': 'add', 'name': name, 'time': time.time()})

    def recognize(self, input_vector):
        """Распознаёт паттерн во входе."""
        recognized = []
        for name, pattern in self.patterns.items():
            # Проверяем, соответствует ли вход паттерну
            similarity = np.dot(input_vector, pattern) / (np.linalg.norm(input_vector) * np.linalg.norm(pattern) + 1e-8)
            if similarity > 0.5:
                recognized.append((name, similarity))
        recognized.sort(key=lambda x: x[1], reverse=True)
        self.pattern_history.append({'action': 'recognize', 'found': len(recognized), 'time': time.time()})
        return recognized

    def get_state(self):
        return {
            'pattern_count': len(self.patterns),
            'recognition_skill': self.pattern_recognition_skill,
            'history_size': len(self.pattern_history)
        }

class NonverbalSystem:
    """Интеграция невербальных форм мышления."""
    def __init__(self):
        self.image = ImageThinking()
        self.intuitive = IntuitiveThinking()
        self.pattern = PatternThinking()
        self.nonverbal_consciousness = 0.0
        self.history = deque(maxlen=30)

    def think(self, input_data):
        """Невербальное мышление на основе входных данных."""
        results = {}
        # 1. Образное мышление
        if isinstance(input_data, str):
            image = self.image.create_image(input_data)
        else:
            image = self.image.create_image()
        results['image'] = image.shape
        # 2. Интуитивное мышление
        intuition = self.intuitive.sense(input_data[:10] if hasattr(input_data, '__len__') else str(input_data))
        results['intuition'] = intuition
        # 3. Паттерное мышление
        if isinstance(input_data, np.ndarray):
            patterns = self.pattern.recognize(input_data)
            results['patterns'] = patterns
        # 4. Интеграция
        self.nonverbal_consciousness = min(1.0, self.nonverbal_consciousness + 0.05)
        results['consciousness_level'] = self.nonverbal_consciousness
        self.history.append({'time': time.time(), 'results': len(results)})
        return results

    def get_state(self):
        return {
            'image': self.image.get_state(),
            'intuitive': self.intuitive.get_state(),
            'pattern': self.pattern.get_state(),
            'consciousness': self.nonverbal_consciousness,
            'history': len(self.history)
        }

if __name__ == "__main__":
    print("="*60)
    print("🧠 НЕВЕРБАЛЬНОЕ МЫШЛЕНИЕ")
    print("="*60)
    system = NonverbalSystem()
    # Тестируем
    test_inputs = ["круг", np.random.randn(5), "треугольник", np.random.randn(8)]
    for i, inp in enumerate(test_inputs):
        result = system.think(inp)
        print(f"Шаг {i}: результаты={result.keys()}")
    state = system.get_state()
    print(f"\nУровень невербального сознания: {state['consciousness']:.2f}")
    print("\n💡 Гениальность: Мышление не сводится к языку.")
    print("   Есть мышление образами, интуицией, паттернами, чувствами.")
    print("   Вербальное мышление — лишь надстройка.")
