# -*- coding: utf-8 -*-
"""
МОДУЛЬ СОЗНАНИЯ - ДИСПЕТЧЕР
Гениальная идея: Сознание — это не функция, а диспетчер,
который решает, что допустить в сознание, а что оставить в подсознании.
Имеет пропускную способность (access_consciousness),
феноменальное поле (phenomenal_field) — то, что переживается,
и модель себя (self_model), которая обновляется через рекурсию.

Это не «сознание как флаг», а динамический процесс,
где сознание — это узкое горлышко,
через которое проходит только самое важное.
"""

import numpy as np
from collections import deque
import time

class ConsciousnessDispatcher:
    """
    Диспетчер сознания — решает, что попадает в сознание.
    """
    
    def __init__(self, bandwidth=7.0, phenomenal_dim=64):
        # Пропускная способность сознания (бит/сек) — аналог узкого горлышка
        self.bandwidth = bandwidth
        self.available_bandwidth = bandwidth
        
        # Феноменальное поле — содержимое сознания
        self.phenomenal_field = {
            'current': None,          # Что сейчас в сознании
            'history': deque(maxlen=100),  # История содержания сознания
            'intensity': 0.0,         # Интенсивность переживания
            'clarity': 0.0,           # Ясность (чем выше, тем более осознанно)
        }
        
        # Модель себя — рекурсивно обновляется
        self.self_model = {
            'core': None,             # Ядро личности
            'narrative': '',          # Повествование о себе
            'self_awareness': 0.0,    # Уровень самосознания
            'reflection_depth': 0,    # Глубина рефлексии
            'update_counter': 0,
        }
        
        # Входящий буфер (то, что ждёт доступа в сознание)
        self.incoming_buffer = deque(maxlen=50)
        
        # Подсознание (то, что не попало в сознание)
        self.subconscious = deque(maxlen=200)
        
        # Статистика
        self.access_count = 0
        self.rejected_count = 0
        self.last_update = time.time()
        
    def request_access(self, content, priority=0.5, source='unknown'):
        """
        Запрос на доступ в сознание.
        
        Args:
            content: Содержание, которое хочет попасть в сознание
            priority: Приоритет (0-1). Выше = больше шансов попасть
            source: Источник (сенсоры, память, внутренний диалог, эмоции)
        
        Returns:
            bool: Попало ли в сознание
        """
        # Оцениваем, нужно ли это в сознании
        importance = self._evaluate_importance(content, priority, source)
        
        # Проверяем, есть ли место в сознании
        if self.phenomenal_field['current'] is not None:
            # Если сознание занято, оцениваем, стоит ли вытеснять текущее
            current_importance = self._evaluate_importance(
                self.phenomenal_field['current'],
                self.phenomenal_field.get('priority', 0.5),
                'current'
            )
            if importance > current_importance:
                # Вытесняем текущее в подсознание
                self._evict_current()
            else:
                # Отклоняем запрос
                self.rejected_count += 1
                self.subconscious.append({
                    'content': content,
                    'priority': priority,
                    'source': source,
                    'time': time.time()
                })
                return False
        
        # Пропускаем в сознание
        self._admit(content, priority, source)
        self.access_count += 1
        return True
    
    def _evaluate_importance(self, content, priority, source):
        """Оценивает важность содержания для сознания."""
        # Базовая важность от приоритета
        importance = priority
        
        # Бонус за источник
        source_bonus = {
            'threat': 1.0,
            'emotion_intense': 0.9,
            'novelty': 0.7,
            'social': 0.6,
            'memory': 0.5,
            'inner_speech': 0.4,
            'default': 0.3
        }
        importance += source_bonus.get(source, source_bonus['default'])
        
        # Бонус за новизну (если это новый паттерн)
        if isinstance(content, dict) and content.get('is_novel', False):
            importance += 0.5
        
        return min(1.0, importance)
    
    def _admit(self, content, priority, source):
        """Пропускает содержание в сознание."""
        # Если есть текущее содержание, отправляем его в историю
        if self.phenomenal_field['current'] is not None:
            self.phenomenal_field['history'].append({
                'content': self.phenomenal_field['current'],
                'priority': self.phenomenal_field.get('priority', 0.5),
                'timestamp': time.time()
            })
        
        # Обновляем феноменальное поле
        self.phenomenal_field['current'] = content
        self.phenomenal_field['priority'] = priority
        self.phenomenal_field['intensity'] = min(1.0, priority * 1.5)
        self.phenomenal_field['clarity'] = min(1.0, priority * 0.8 + 0.2)
        self.phenomenal_field['source'] = source
        self.phenomenal_field['timestamp'] = time.time()
        
        # Обновляем модель себя (рекурсивно)
        self._update_self_model(content)
    
    def _evict_current(self):
        """Вытесняет текущее содержание из сознания в подсознание."""
        if self.phenomenal_field['current'] is not None:
            self.subconscious.append({
                'content': self.phenomenal_field['current'],
                'priority': self.phenomenal_field.get('priority', 0.5),
                'source': self.phenomenal_field.get('source', 'unknown'),
                'time': time.time(),
                'evicted': True
            })
            self.phenomenal_field['current'] = None
            self.phenomenal_field['clarity'] = 0.0
            self.phenomenal_field['intensity'] = 0.0
    
    def _update_self_model(self, content):
        """Обновляет модель себя на основе содержания сознания."""
        self.self_model['update_counter'] += 1
        
        # Если содержание содержит информацию о себе
        if isinstance(content, dict):
            if 'self_reference' in content:
                self.self_model['core'] = content['self_reference']
                self.self_model['self_awareness'] = min(1.0, 
                    self.self_model['self_awareness'] + 0.05)
            
            if 'narrative' in content:
                self.self_model['narrative'] = content['narrative']
        
        # Рефлексия: каждые 10 обновлений — углубляемся
        if self.self_model['update_counter'] % 10 == 0:
            self.self_model['reflection_depth'] += 1
            self.self_model['self_awareness'] = min(1.0, 
                self.self_model['self_awareness'] + 0.02)
    
    def get_current_content(self):
        """Возвращает текущее содержание сознания."""
        return self.phenomenal_field['current']
    
    def get_consciousness_state(self):
        """Возвращает полное состояние сознания."""
        return {
            'phenomenal_field': self.phenomenal_field,
            'self_model': self.self_model,
            'bandwidth': self.bandwidth,
            'available_bandwidth': self.available_bandwidth,
            'access_count': self.access_count,
            'rejected_count': self.rejected_count,
            'subconscious_size': len(self.subconscious),
            'incoming_buffer_size': len(self.incoming_buffer),
        }
    
    def step(self, dt=1.0):
        """Шаг обновления сознания."""
        # Восстановление пропускной способности
        self.available_bandwidth = min(self.bandwidth, 
            self.available_bandwidth + 0.1 * dt)
        
        # Затухание интенсивности
        self.phenomenal_field['intensity'] *= 0.99
        
        # Обновление времени
        self.last_update = time.time()


# ============================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 МОДУЛЬ СОЗНАНИЯ - ДИСПЕТЧЕР")
    print("=" * 60)
    
    disp = ConsciousnessDispatcher(bandwidth=5.0)
    
    # Тест: запросы в сознание
    test_contents = [
        ("Впереди опасность!", 0.9, 'threat'),
        ("Я хочу есть", 0.7, 'emotion_intense'),
        ("Этот паттерн новый", 0.6, 'novelty'),
        ("Кто-то смотрит на меня", 0.8, 'social'),
        ("Я помню, как было раньше", 0.5, 'memory'),
        ("Почему я здесь?", 0.4, 'inner_speech'),
    ]
    
    for content, priority, source in test_contents:
        admitted = disp.request_access(content, priority, source)
        print(f"  Запрос: {content} (приоритет {priority:.1f}, источник {source})")
        print(f"    → {'ДОПУЩЕН' if admitted else 'ОТКЛОНЁН'}")
        if admitted:
            print(f"    → Текущее сознание: {disp.get_current_content()}")
        print()
    
    print("\n📊 СОСТОЯНИЕ СОЗНАНИЯ:")
    state = disp.get_consciousness_state()
    print(f"  Пропускная способность: {state['bandwidth']}")
    print(f"  Доступная: {state['available_bandwidth']:.2f}")
    print(f"  Допущено: {state['access_count']}")
    print(f"  Отклонено: {state['rejected_count']}")
    print(f"  Размер подсознания: {state['subconscious_size']}")
    print(f"  Самосознание: {state['self_model']['self_awareness']:.2f}")
    print(f"  Глубина рефлексии: {state['self_model']['reflection_depth']}")
    
    print("\n💡 Гениальность: Сознание — это не флаг, а ДИНАМИЧЕСКИЙ ПРОЦЕСС,")
    print("   где сознание — это узкое горлышко, через которое проходит")
    print("   только самое важное. Модель себя обновляется РЕКУРСИВНО.")
