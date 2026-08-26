# -*- coding: utf-8 -*-
"""
МОДУЛЬ АКТИВНОГО ВНИМАНИЯ
Гениальная идея: Внимание — это активный процесс, а не пассивный фильтр.
Агент:
- focus — фокусируется на объекте
- orient — ориентируется на новизну
- disengage — отпускает объект, когда он становится неважным

Это моделирует ориентировочный рефлекс —
то, что позволяет агенту не застревать на одном объекте,
а переключаться между важными вещами.
"""

import numpy as np
from collections import deque
import time
import math

class ActiveAttention:
    """
    Активное внимание с фокусировкой, ориентацией и переключением.
    """
    
    def __init__(self, capacity=5, novelty_threshold=0.3):
        # Вместимость внимания (сколько объектов может удерживать)
        self.capacity = capacity
        self.novelty_threshold = novelty_threshold
        
        # Текущий фокус внимания
        self.focus = {
            'object': None,        # Объект в фокусе
            'intensity': 0.0,      # Интенсивность фокуса
            'duration': 0,         # Длительность фокусировки
            'type': None,          # Тип объекта
        }
        
        # Поле внимания (объекты, которые находятся под вниманием)
        self.attention_field = {
            'objects': [],         # Список объектов в поле внимания
            'saliency': {},        # Салиентность каждого объекта
            'novelty': {},         # Новизна каждого объекта
        }
        
        # Переключение внимания
        self.switch_history = deque(maxlen=20)
        self.switch_count = 0
        self.last_switch = time.time()
        
        # Ориентировочный рефлекс
        self.orienting_response = 0.0  # 0-1, сила ориентировочной реакции
        self.habituation = 0.0         # Привыкание к повторяющимся стимулам
        
        # Внутреннее состояние
        self.boredom = 0.0
        self.distraction = 0.0
        self.engagement = 0.5
        
        # Сенсорный буфер
        self.sensory_buffer = deque(maxlen=10)
        
    def update(self, objects, internal_state=None):
        """
        Обновляет поле внимания на основе списка объектов.
        
        Args:
            objects: Список объектов в восприятии
            internal_state: Внутреннее состояние агента (эмоции, цели)
        
        Returns:
            dict: Обновлённое состояние внимания
        """
        # 1. Вычисляем салиентность каждого объекта
        self._compute_saliency(objects, internal_state)
        
        # 2. Вычисляем новизну
        self._compute_novelty(objects)
        
        # 3. Обновляем ориентировочный рефлекс
        self._update_orienting_response(objects)
        
        # 4. Выбираем объект для фокуса
        self._select_focus(objects, internal_state)
        
        # 5. Проверяем, нужно ли переключиться
        self._check_switch(objects, internal_state)
        
        # 6. Обновляем внутреннее состояние
        self._update_internal_state(objects)
        
        return self.get_state()
    
    def _compute_saliency(self, objects, internal_state):
        """Вычисляет салиентность каждого объекта."""
        self.attention_field['objects'] = objects
        
        for obj in objects:
            # Базовая салиентность
            saliency = 0.3
            
            # Если объект представляет угрозу
            if hasattr(obj, 'type') and obj.type == 'threat':
                saliency += 0.5
            
            # Если объект — ресурс
            if hasattr(obj, 'type') and obj.type == 'resource':
                saliency += 0.3
            
            # Если объект — социальный
            if hasattr(obj, 'type') and obj.type == 'social':
                saliency += 0.2
            
            # Если объект близко
            if hasattr(obj, 'distance') and obj.distance < 3:
                saliency += 0.2
            
            # Если объект соответствует цели
            if internal_state and internal_state.get('goal') and hasattr(obj, 'type'):
                if obj.type == internal_state.get('goal'):
                    saliency += 0.4
            
            # Добавляем шум
            saliency += np.random.uniform(-0.05, 0.05)
            
            # Нормализация
            self.attention_field['saliency'][id(obj)] = max(0.0, min(1.0, saliency))
    
    def _compute_novelty(self, objects):
        """Вычисляет новизну каждого объекта."""
        for obj in objects:
            # Проверяем, видели ли мы этот объект раньше
            obj_id = id(obj)
            if obj_id not in self.attention_field['novelty']:
                # Новый объект
                self.attention_field['novelty'][obj_id] = 1.0
            else:
                # Привыкание
                self.attention_field['novelty'][obj_id] *= 0.95
            
            # Если объект изменился
            if hasattr(obj, 'changed') and obj.changed:
                self.attention_field['novelty'][obj_id] = min(1.0, 
                    self.attention_field['novelty'][obj_id] + 0.3)
    
    def _update_orienting_response(self, objects):
        """Обновляет ориентировочный рефлекс."""
        # Сила рефлекса зависит от новизны и салиентности
        max_novelty = max(self.attention_field['novelty'].values()) if self.attention_field['novelty'] else 0.0
        max_saliency = max(self.attention_field['saliency'].values()) if self.attention_field['saliency'] else 0.0
        
        # Ориентировочная реакция
        self.orienting_response = (max_novelty * 0.5 + max_saliency * 0.5)
        
        # Привыкание
        self.habituation = min(1.0, self.habituation + 0.01)
        
        # Если есть новый объект, реакция сильнее
        if max_novelty > self.novelty_threshold:
            self.orienting_response = min(1.0, self.orienting_response * 1.3)
            self.habituation = max(0.0, self.habituation - 0.1)
        
        # Если всё повторяется, реакция слабее
        if max_novelty < 0.1:
            self.habituation = min(1.0, self.habituation + 0.05)
            self.orienting_response = max(0.0, self.orienting_response - 0.02)
    
    def _select_focus(self, objects, internal_state):
        """Выбирает объект для фокуса внимания."""
        if not objects:
            self.focus['object'] = None
            self.focus['intensity'] = 0.0
            self.focus['duration'] = 0
            return
        
        # Если фокус уже есть и он всё ещё важен
        if self.focus['object'] is not None:
            obj_id = id(self.focus['object'])
            saliency = self.attention_field['saliency'].get(obj_id, 0.0)
            novelty = self.attention_field['novelty'].get(obj_id, 0.0)
            
            # Продолжаем фокусироваться, если объект всё ещё важен
            if saliency + novelty > 0.4:
                self.focus['duration'] += 1
                self.focus['intensity'] = min(1.0, 
                    self.focus['intensity'] + 0.01 * (1.0 + novelty))
                return
        
        # Ищем лучший объект для фокуса
        best_obj = None
        best_score = -1.0
        
        for obj in objects:
            obj_id = id(obj)
            saliency = self.attention_field['saliency'].get(obj_id, 0.0)
            novelty = self.attention_field['novelty'].get(obj_id, 0.0)
            
            # Оценка объекта для фокуса
            score = saliency * 0.6 + novelty * 0.4
            
            # Если объект — угроза, приоритет выше
            if hasattr(obj, 'type') and obj.type == 'threat':
                score += 0.2
            
            # Если объект соответствует цели
            if internal_state and internal_state.get('goal') and hasattr(obj, 'type'):
                if obj.type == internal_state.get('goal'):
                    score += 0.3
            
            if score > best_score:
                best_score = score
                best_obj = obj
        
        # Переключаем фокус
        if best_obj and best_score > 0.3:
            self._switch_focus(best_obj, best_score)
    
    def _switch_focus(self, obj, score):
        """Переключает фокус на новый объект."""
        # Записываем переключение
        self.switch_history.append({
            'from': self.focus['object'],
            'to': obj,
            'score': score,
            'time': time.time()
        })
        self.switch_count += 1
        self.last_switch = time.time()
        
        # Обновляем фокус
        self.focus['object'] = obj
        self.focus['intensity'] = min(1.0, score * 1.2)
        self.focus['duration'] = 0
        if hasattr(obj, 'type'):
            self.focus['type'] = obj.type
        else:
            self.focus['type'] = None
    
    def _check_switch(self, objects, internal_state):
        """Проверяет, нужно ли переключить внимание."""
        if self.focus['object'] is None:
            return
        
        obj_id = id(self.focus['object'])
        saliency = self.attention_field['saliency'].get(obj_id, 0.0)
        novelty = self.attention_field['novelty'].get(obj_id, 0.0)
        
        # Если объект перестал быть важным
        if saliency + novelty < 0.2:
            # Дисенгейдж (отпускаем)
            self._disengage_focus()
            return
        
        # Если объект в фокусе слишком долго
        if self.focus['duration'] > 20 and self.focus['intensity'] > 0.8:
            # Может возникнуть усталость внимания
            self.distraction = min(1.0, self.distraction + 0.02)
            if self.distraction > 0.5:
                self._disengage_focus()
                return
        
        # Если появился более важный объект
        if objects:
            for obj in objects:
                if id(obj) == obj_id:
                    continue
                new_score = self.attention_field['saliency'].get(id(obj), 0.0) * 0.6 + \
                           self.attention_field['novelty'].get(id(obj), 0.0) * 0.4
                
                current_score = saliency * 0.6 + novelty * 0.4
                if new_score > current_score + 0.3:
                    self._switch_focus(obj, new_score)
                    return
    
    def _disengage_focus(self):
        """Отпускает текущий фокус."""
        self.switch_history.append({
            'from': self.focus['object'],
            'to': None,
            'score': 0,
            'time': time.time(),
            'disengage': True
        })
        self.focus['object'] = None
        self.focus['intensity'] = 0.0
        self.focus['duration'] = 0
        self.focus['type'] = None
        self.distraction = max(0.0, self.distraction - 0.1)
    
    def _update_internal_state(self, objects):
        """Обновляет внутреннее состояние внимания."""
        # Скука
        if len(objects) == 0:
            self.boredom = min(1.0, self.boredom + 0.02)
        else:
            # Проверяем, есть ли новизна
            max_novelty = max(self.attention_field['novelty'].values()) if self.attention_field['novelty'] else 0.0
            if max_novelty < 0.1:
                self.boredom = min(1.0, self.boredom + 0.01)
            else:
                self.boredom = max(0.0, self.boredom - 0.02)
        
        # Вовлечённость
        if self.focus['object'] is not None:
            self.engagement = min(1.0, 
                self.engagement + 0.01 * self.focus['intensity'])
        else:
            self.engagement = max(0.0, self.engagement - 0.01)
        
        # Отвлекаемость
        if self.focus['object'] is None:
            self.distraction = min(1.0, self.distraction + 0.01)
        else:
            self.distraction = max(0.0, self.distraction - 0.01)
    
    def get_state(self):
        """Возвращает полное состояние внимания."""
        return {
            'focus': self.focus.copy(),
            'attention_field': {
                'objects': self.attention_field['objects'],
                'saliency': self.attention_field['saliency'].copy(),
                'novelty': self.attention_field['novelty'].copy(),
            },
            'orienting_response': self.orienting_response,
            'habituation': self.habituation,
            'boredom': self.boredom,
            'distraction': self.distraction,
            'engagement': self.engagement,
            'switch_count': self.switch_count,
            'switch_history': list(self.switch_history),
            'capacity': self.capacity,
        }


# ============================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("👁️ МОДУЛЬ АКТИВНОГО ВНИМАНИЯ")
    print("=" * 60)
    
    attention = ActiveAttention(capacity=5)
    
    # Тест: разные объекты
    class TestObject:
        def __init__(self, name, type, distance=5, changed=False):
            self.name = name
            self.type = type
            self.distance = distance
            self.changed = changed
    
    test_scenarios = [
        # Сценарий 1: разные объекты
        [
            TestObject("еда", "resource", 2),
            TestObject("камень", "neutral", 5),
            TestObject("зверь", "threat", 8),
            TestObject("друг", "social", 3),
        ],
        # Сценарий 2: новый объект
        [
            TestObject("еда", "resource", 2),
            TestObject("камень", "neutral", 5),
            TestObject("зверь", "threat", 8),
            TestObject("новый друг", "social", 2, changed=True),
        ],
        # Сценарий 3: ничего нового
        [
            TestObject("еда", "resource", 2),
            TestObject("камень", "neutral", 5),
        ],
    ]
    
    for i, objects in enumerate(test_scenarios):
        print(f"\n🎯 Сценарий {i+1}:")
        print(f"   Объекты: {[o.name for o in objects]}")
        
        internal_state = {'goal': 'resource'}
        state = attention.update(objects, internal_state)
        
        focus = state['focus']
        if focus['object']:
            print(f"   👁️ Фокус: {focus['object'].name} (интенсивность {focus['intensity']:.2f}, длительность {focus['duration']})")
        else:
            print(f"   👁️ Фокус: НЕТ")
        
        print(f"   🆕 Ориентировочная реакция: {state['orienting_response']:.2f}")
        print(f"   🔄 Привыкание: {state['habituation']:.2f}")
        print(f"   😑 Скука: {state['boredom']:.2f}")
        print(f"   🎯 Вовлечённость: {state['engagement']:.2f}")
        print(f"   🌀 Переключений: {state['switch_count']}")
        
        # Выводим салиентность
        print(f"   📊 Салиентность:")
        for obj in objects:
            obj_id = id(obj)
            saliency = state['attention_field']['saliency'].get(obj_id, 0.0)
            novelty = state['attention_field']['novelty'].get(obj_id, 0.0)
            print(f"      {obj.name}: салиентность={saliency:.2f}, новизна={novelty:.2f}")
    
    print("\n💡 Гениальность: Внимание — это АКТИВНЫЙ ПРОЦЕСС,")
    print("   а не пассивный фильтр. Агент не застревает на одном объекте,")
    print("   а ПЕРЕКЛЮЧАЕТСЯ между важными вещами через ориентировочный рефлекс.")
