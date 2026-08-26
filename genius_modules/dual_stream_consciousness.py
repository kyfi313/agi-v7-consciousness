# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ДВУХПОТОЧНОЕ СОЗНАНИЕ
Гениальность: Сознание имеет два потока — фокус внимания и периферическое восприятие.
Фокус — узкий, яркий, детальный. Периферия — широкий, тусклый, контекстуальный.

Эти два потока взаимодействуют и дополняют друг друга.

Реализовано: FocusStream — поток фокуса внимания,
PeripheralStream — периферический поток,
DualStreamConsciousness — интеграция двух потоков.
"""

import numpy as np
from collections import deque
import time
import random

class FocusStream:
    """Поток фокуса внимания — узкий, яркий, детальный."""
    def __init__(self, capacity=3):
        self.capacity = capacity
        self.current_focus = None
        self.focus_history = deque(maxlen=20)
        self.attention_span = 0.5
        self.detail_level = 0.8
        self.focus_objects = deque(maxlen=capacity)

    def set_focus(self, object_id, detail=0.8):
        """Устанавливает объект в фокус внимания."""
        if len(self.focus_objects) >= self.capacity:
            # Вытесняем самый старый объект
            self.focus_objects.popleft()
        self.focus_objects.append({'id': object_id, 'detail': detail, 'time': time.time()})
        self.current_focus = object_id
        self.detail_level = detail
        self.focus_history.append({'object': object_id, 'time': time.time()})
        return True

    def get_focus(self):
        """Возвращает текущий объект в фокусе."""
        return self.current_focus

    def shift_focus(self, direction='next'):
        """Переключает внимание на следующий объект."""
        if len(self.focus_objects) > 1:
            if direction == 'next':
                # Циклический сдвиг
                self.focus_objects.rotate(-1)
            elif direction == 'prev':
                self.focus_objects.rotate(1)
            self.current_focus = self.focus_objects[-1]['id']
            self.detail_level = self.focus_objects[-1]['detail']
            return True
        return False

    def get_state(self):
        return {
            'capacity': self.capacity,
            'current_focus': self.current_focus,
            'detail_level': self.detail_level,
            'objects_in_focus': len(self.focus_objects),
            'history_size': len(self.focus_history)
        }

class PeripheralStream:
    """Периферический поток — широкий, тусклый, контекстуальный."""
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.peripheral_objects = deque(maxlen=capacity)
        self.contextual_binding = {}
        self.awareness_level = 0.3
        self.background_signals = deque(maxlen=30)

    def add_peripheral(self, object_id, relevance=0.3):
        """Добавляет объект в периферию."""
        if len(self.peripheral_objects) >= self.capacity:
            self.peripheral_objects.popleft()
        self.peripheral_objects.append({'id': object_id, 'relevance': relevance, 'time': time.time()})
        self.contextual_binding[object_id] = relevance
        self.background_signals.append({'object': object_id, 'relevance': relevance, 'time': time.time()})

    def get_peripheral(self, min_relevance=0.2):
        """Возвращает периферические объекты с достаточной релевантностью."""
        return [obj for obj in self.peripheral_objects if obj['relevance'] > min_relevance]

    def update_awareness(self, stimulus):
        """Обновляет уровень осознания периферии."""
        # Внешний стимул повышает осознание
        self.awareness_level = min(1.0, self.awareness_level + stimulus * 0.1)
        # Со временем осознание затухает
        self.awareness_level = max(0.1, self.awareness_level - 0.01)
        return self.awareness_level

    def get_state(self):
        return {
            'capacity': self.capacity,
            'object_count': len(self.peripheral_objects),
            'awareness_level': self.awareness_level,
            'contextual_bindings': len(self.contextual_binding)
        }

class DualStreamConsciousness:
    """Интеграция двух потоков сознания."""
    def __init__(self):
        self.focus = FocusStream()
        self.peripheral = PeripheralStream()
        self.integration_level = 0.5
        self.consciousness_history = deque(maxlen=30)
        self.attention_cycle = 0.0

    def process_input(self, inputs):
        """Обрабатывает входные данные через оба потока."""
        # 1. Определяем, что попадает в фокус
        if len(inputs) > 0:
            # Сначала определяем самый важный объект
            main_input = max(inputs, key=lambda x: x.get('importance', 0.5))
            self.focus.set_focus(main_input.get('id', 'default'), main_input.get('detail', 0.8))

        # 2. Всё остальное — в периферию
        for inp in inputs:
            if inp.get('id') != self.focus.current_focus:
                self.peripheral.add_peripheral(
                    inp.get('id', 'unknown'),
                    inp.get('relevance', 0.3)
                )

        # 3. Обновляем осознание периферии
        if self.focus.detail_level > 0.5:
            # Когда фокус яркий, периферия ослабевает
            self.peripheral.awareness_level = max(0.1, self.peripheral.awareness_level - 0.02)
        else:
            # Когда фокус слабый, периферия усиливается
            self.peripheral.awareness_level = min(1.0, self.peripheral.awareness_level + 0.02)

        # 4. Интеграция
        self.integration_level = (self.focus.detail_level + self.peripheral.awareness_level) / 2
        self.consciousness_history.append({
            'time': time.time(),
            'focus': self.focus.current_focus,
            'integration': self.integration_level
        })
        self.attention_cycle += 0.1

        return self.get_state()

    def shift_attention(self):
        """Переключает внимание между объектами."""
        # Если есть объекты в периферии, перемещаем их в фокус
        peripheral_objects = self.peripheral.get_peripheral(0.3)
        if peripheral_objects:
            # Берём самый релевантный объект из периферии
            best = max(peripheral_objects, key=lambda x: x['relevance'])
            self.focus.set_focus(best['id'], best['relevance'])
            # Удаляем из периферии
            self.peripheral.peripheral_objects = deque(
                [obj for obj in self.peripheral.peripheral_objects if obj['id'] != best['id']],
                maxlen=self.peripheral.capacity
            )
            return True
        return False

    def get_state(self):
        return {
            'focus': self.focus.get_state(),
            'peripheral': self.peripheral.get_state(),
            'integration': self.integration_level,
            'attention_cycle': self.attention_cycle,
            'history_size': len(self.consciousness_history)
        }

if __name__ == "__main__":
    print("="*60)
    print("🔦 ДВУХПОТОЧНОЕ СОЗНАНИЕ")
    print("="*60)
    dual = DualStreamConsciousness()
    # Симуляция
    for i in range(8):
        inputs = []
        for j in range(3 + i % 2):
            inputs.append({
                'id': f'obj_{j}',
                'importance': random.random() * 0.5 + 0.3,
                'detail': random.random() * 0.5 + 0.3,
                'relevance': random.random() * 0.5 + 0.2
            })
        # Делаем один объект более важным
        if i % 3 == 2:
            inputs[0]['importance'] = 0.9
            inputs[0]['detail'] = 0.9
        state = dual.process_input(inputs)
        print(f"Шаг {i}: фокус={state['focus']['current_focus']}, "
              f"периферия={state['peripheral']['object_count']}, "
              f"интеграция={state['integration']:.2f}")
        if i % 4 == 3:
            dual.shift_attention()
            print("  ▶ Переключение внимания")
    print("\n💡 Гениальность: Сознание имеет два потока — ФОКУС и ПЕРИФЕРИЮ.")
    print("   Они взаимодействуют и дополняют друг друга.")
