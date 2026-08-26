# -*- coding: utf-8 -*-
"""
МОДУЛЬ: СОЗНАНИЕ КАК ДИСПЕТЧЕР РЕСУРСОВ
Гениальность: Сознание — это не центр управления, а диспетчер,
который распределяет ограниченные ресурсы внимания между конкурирующими процессами.

Сознание имеет пропускную способность (bandwidth) и решает,
какие процессы получат доступ.

Реализовано: ConsciousnessDispatcher с bandwidth, phenomenal field,
self-model и recursive обновлением.
"""

import numpy as np
from collections import deque
import time
import random

class ConsciousnessDispatcher:
    """Сознание как диспетчер ресурсов."""
    def __init__(self, bandwidth=10.0, num_processes=5):
        self.bandwidth = bandwidth
        self.num_processes = num_processes
        self.processes = []
        self.phenomenal_field = deque(maxlen=20)
        self.self_model = {}
        self.allocation_history = deque(maxlen=50)
        self.current_load = 0.0
        self.consciousness_level = 0.5

    def register_process(self, name, priority=0.5, resource_demand=1.0):
        """Регистрирует процесс, который требует ресурсов сознания."""
        process = {
            'name': name,
            'priority': priority,
            'resource_demand': resource_demand,
            'active': False,
            'allocation_time': 0.0,
            'urgency': 0.5,
            'last_active': time.time()
        }
        self.processes.append(process)
        return process

    def request_access(self, process_name, urgency=0.5):
        """Запрос доступа к сознанию."""
        for p in self.processes:
            if p['name'] == process_name:
                p['urgency'] = urgency
                return self._allocate(p)
        return None

    def _allocate(self, process):
        """Распределяет ресурсы сознания между процессами."""
        # Проверка, есть ли свободный ресурс
        if self.current_load + process['resource_demand'] <= self.bandwidth:
            # Выделяем ресурс
            process['active'] = True
            process['allocation_time'] = time.time()
            self.current_load += process['resource_demand']
            self.phenomenal_field.append({
                'process': process['name'],
                'time': time.time(),
                'load': self.current_load
            })
            self.allocation_history.append({
                'process': process['name'],
                'allocated': True,
                'load': self.current_load
            })
            return True
        else:
            # Конфликт — нужно выбрать, что вытеснить
            return self._preempt(process)

    def _preempt(self, new_process):
        """Вытесняет процесс с низким приоритетом."""
        # Находим процесс с наименьшим приоритетом
        active_processes = [p for p in self.processes if p['active']]
        if not active_processes:
            return False
        # Сортируем по приоритету и срочности
        sorted_processes = sorted(active_processes,
                                  key=lambda p: (p['priority'], -p['urgency']))
        # Вытесняем самый низкоприоритетный
        victim = sorted_processes[0]
        if victim['priority'] < new_process['priority']:
            victim['active'] = False
            self.current_load -= victim['resource_demand']
            # Выделяем ресурс новому процессу
            new_process['active'] = True
            new_process['allocation_time'] = time.time()
            self.current_load += new_process['resource_demand']
            self.phenomenal_field.append({
                'process': new_process['name'],
                'time': time.time(),
                'preempted': victim['name'],
                'load': self.current_load
            })
            self.allocation_history.append({
                'process': new_process['name'],
                'allocated': True,
                'preempted': victim['name'],
                'load': self.current_load
            })
            return True
        return False

    def update_self_model(self):
        """Обновляет модель себя на основе феноменального поля."""
        # Вычисляем статистику по феноменальному полю
        if len(self.phenomenal_field) > 0:
            recent = list(self.phenomenal_field)[-10:]
            # Кто чаще всего появлялся в сознании?
            processes = [item['process'] for item in recent]
            if processes:
                main_process = max(set(processes), key=processes.count)
                self.self_model['dominant_process'] = main_process
                self.self_model['self_awareness'] = len(set(processes)) / max(1, len(processes))
                self.self_model['load_avg'] = np.mean([item.get('load', 0) for item in recent])
                self.self_model['last_update'] = time.time()
        return self.self_model

    def get_current_content(self):
        """Возвращает текущее содержимое сознания."""
        if self.phenomenal_field:
            last = self.phenomenal_field[-1]
            return {
                'process': last.get('process'),
                'time': last.get('time'),
                'load': self.current_load,
                'bandwidth_usage': self.current_load / self.bandwidth
            }
        return None

    def get_state(self):
        """Возвращает состояние диспетчера."""
        return {
            'bandwidth': self.bandwidth,
            'load': self.current_load,
            'usage_ratio': self.current_load / self.bandwidth,
            'processes': self.processes,
            'phenomenal_field_size': len(self.phenomenal_field),
            'self_model': self.self_model,
            'consciousness_level': self.consciousness_level
        }

if __name__ == "__main__":
    print("="*60)
    print("🧠 СОЗНАНИЕ КАК ДИСПЕТЧЕР РЕСУРСОВ")
    print("="*60)
    dispatcher = ConsciousnessDispatcher(bandwidth=5.0, num_processes=3)
    # Регистрируем процессы
    p1 = dispatcher.register_process('восприятие', priority=0.8, resource_demand=1.0)
    p2 = dispatcher.register_process('речь', priority=0.6, resource_demand=2.0)
    p3 = dispatcher.register_process('планирование', priority=0.7, resource_demand=1.5)
    # Запросы доступа
    for i in range(10):
        if i % 2 == 0:
            dispatcher.request_access('восприятие', urgency=0.5 + i * 0.05)
        elif i % 3 == 0:
            dispatcher.request_access('речь', urgency=0.3 + i * 0.03)
        else:
            dispatcher.request_access('планирование', urgency=0.7 + i * 0.02)
        dispatcher.update_self_model()
        content = dispatcher.get_current_content()
        if content:
            print(f"Шаг {i}: в сознании — {content['process']}, загрузка={content['load']:.2f}/{dispatcher.bandwidth}")
    print("\n💡 Гениальность: Сознание — это диспетчер с ограниченной пропускной способностью.")
    print("   Оно решает, какие процессы получают доступ к вниманию.")
