# -*- coding: utf-8 -*-
"""
Сенсорный вход — модель восприятия окружающего мира
Генерирует искусственные данные для всех модальностей
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from ..core.base import BaseModule
from ..core.state import GlobalState


@dataclass
class SensoryData:
    """Структура всех сенсорных данных за один шаг"""
    visual: Dict[str, Any] = field(default_factory=dict)      # зрение
    auditory: Dict[str, Any] = field(default_factory=dict)    # слух
    tactile: Dict[str, Any] = field(default_factory=dict)     # осязание
    proprioceptive: Dict[str, Any] = field(default_factory=dict) # положение тела
    interoceptive: Dict[str, Any] = field(default_factory=dict)  # внутреннее состояние
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'visual': self.visual,
            'auditory': self.auditory,
            'tactile': self.tactile,
            'proprioceptive': self.proprioceptive,
            'interoceptive': self.interoceptive,
        }


class SensorModule(BaseModule):
    """
    Генератор сенсорных данных для AGI
    В реальной системе здесь будет подключение к камере, микрофону и т.д.
    """
    name = "sensors"
    
    def __init__(self):
        self.step = 0
        self.world_state = {
            'objects': [],           # список объектов в поле зрения
            'sounds': [],            # список звуков
            'touch': None,           # тактильный контакт
            'body_position': [0, 0, 0],  # x, y, z
            'body_rotation': [0, 0, 0],  # pitch, yaw, roll
            'internal_temp': 37.0,   # температура тела
            'hunger': 0.3,           # голод (0-1)
            'thirst': 0.2,           # жажда (0-1)
            'pain': 0.0,             # боль (0-1)
            'energy': 1.0,           # энергия (0-1)
        }
        # История сенсорных данных для обнаружения изменений
        self.history = []
        
    def update(self, state: GlobalState) -> GlobalState:
        """Генерирует новые сенсорные данные на основе состояния мира"""
        self.step += 1
        
        # Эмулируем динамику мира
        self._update_world_state(state)
        
        # Генерируем сенсорные данные
        sensory = self._generate_sensory_data(state)
        
        # Сохраняем в состояние
        state.sensory = sensory.to_dict()
        state.perception['sensory_time'] = self.step
        
        # Обновляем perception на основе сенсоров
        self._update_perception(state, sensory)
        
        # Сохраняем историю для обнаружения новизны
        self.history.append(sensory.to_dict())
        if len(self.history) > 100:
            self.history.pop(0)
        
        # Вычисляем новизну (изменение сенсорного входа)
        state.perception['novelty'] = self._compute_novelty(sensory)
        
        return state
    
    def _update_world_state(self, state: GlobalState):
        """Эмулирует изменения в мире"""
        # Случайные события
        if random.random() < 0.05:
            # Новый объект появляется
            new_obj = {
                'type': random.choice(['food', 'threat', 'social', 'object', 'resource']),
                'position': [random.uniform(-5, 5), 0, random.uniform(-5, 5)],
                'size': random.uniform(0.5, 2.0),
                'color': random.choice(['red', 'green', 'blue', 'yellow', 'white']),
                'active': True,
            }
            self.world_state['objects'].append(new_obj)
            if len(self.world_state['objects']) > 10:
                self.world_state['objects'].pop(0)
        
        # Случайные звуки
        if random.random() < 0.03:
            sound = {
                'type': random.choice(['speech', 'footstep', 'explosion', 'music', 'alarm']),
                'volume': random.uniform(0.2, 1.0),
                'direction': random.uniform(-180, 180),
            }
            self.world_state['sounds'].append(sound)
            if len(self.world_state['sounds']) > 5:
                self.world_state['sounds'].pop(0)
        
        # Изменение внутреннего состояния
        self.world_state['hunger'] = min(1.0, self.world_state['hunger'] + random.uniform(-0.02, 0.01))
        self.world_state['thirst'] = min(1.0, self.world_state['thirst'] + random.uniform(-0.02, 0.01))
        self.world_state['energy'] = max(0.0, self.world_state['energy'] - random.uniform(0.001, 0.005))
        
        # Боль может возникнуть при столкновении
        if random.random() < 0.01:
            self.world_state['pain'] = min(1.0, self.world_state['pain'] + random.uniform(0.1, 0.3))
        else:
            self.world_state['pain'] = max(0.0, self.world_state['pain'] - random.uniform(0.01, 0.05))
    
    def _generate_sensory_data(self, state: GlobalState) -> SensoryData:
        """Генерирует сенсорные данные из состояния мира"""
        # Визуальные данные (из объектов в поле зрения)
        visual_data = {
            'objects': self.world_state['objects'][:5],  # до 5 объектов
            'num_objects': len(self.world_state['objects']),
            'brightness': random.uniform(0.3, 0.9),
            'motion': random.uniform(0.0, 0.5),
            'color_dominance': random.choice(['warm', 'cool', 'neutral']),
        }
        
        # Слуховые данные
        auditory_data = {
            'sounds': self.world_state['sounds'][:3],
            'num_sounds': len(self.world_state['sounds']),
            'loudness': sum(s.get('volume', 0) for s in self.world_state['sounds']) if self.world_state['sounds'] else 0,
            'speech_detected': any(s.get('type') == 'speech' for s in self.world_state['sounds']),
        }
        
        # Тактильные данные
        tactile_data = {
            'contact': random.choice([True, False]) if random.random() < 0.2 else False,
            'pressure': random.uniform(0, 0.5) if random.random() < 0.2 else 0,
            'texture': random.choice(['smooth', 'rough', 'soft', 'hard']) if random.random() < 0.2 else None,
            'temperature': random.uniform(20, 40) if random.random() < 0.2 else None,
        }
        
        # Проприоцептивные данные
        proprioceptive_data = {
            'position': self.world_state['body_position'].copy(),
            'rotation': self.world_state['body_rotation'].copy(),
            'velocity': [random.uniform(-0.5, 0.5) for _ in range(3)],
            'balance': random.uniform(0.7, 1.0),
            'joint_angles': [random.uniform(-90, 90) for _ in range(6)],
        }
        
        # Интероцептивные данные
        interoceptive_data = {
            'temperature': self.world_state['internal_temp'] + random.uniform(-0.5, 0.5),
            'hunger': self.world_state['hunger'],
            'thirst': self.world_state['thirst'],
            'pain': self.world_state['pain'],
            'energy': self.world_state['energy'],
            'heart_rate': random.uniform(60, 90),
            'stress': random.uniform(0, 0.5),
        }
        
        return SensoryData(
            visual=visual_data,
            auditory=auditory_data,
            tactile=tactile_data,
            proprioceptive=proprioceptive_data,
            interoceptive=interoceptive_data,
        )
    
    def _update_perception(self, state: GlobalState, sensory: SensoryData):
        """Обновляет perception на основе сенсорных данных"""
        # Восприятие объектов
        if sensory.visual.get('objects'):
            state.objects = sensory.visual['objects']
        
        # Восприятие опасности (из звуков и объектов)
        danger = 0.0
        for obj in sensory.visual.get('objects', []):
            if obj.get('type') == 'threat':
                danger += 0.3
        for sound in sensory.auditory.get('sounds', []):
            if sound.get('type') == 'explosion':
                danger += 0.5
            if sound.get('type') == 'alarm':
                danger += 0.4
        state.perception['danger'] = min(1.0, danger)
        
        # Восприятие награды (еда, ресурсы)
        reward = 0.0
        for obj in sensory.visual.get('objects', []):
            if obj.get('type') == 'food':
                reward += 0.2
            if obj.get('type') == 'resource':
                reward += 0.1
        if sensory.tactile.get('contact') and sensory.tactile.get('texture') == 'soft':
            reward += 0.1
        state.perception['reward'] = min(1.0, reward)
        
        # Восприятие социального взаимодействия
        if sensory.auditory.get('speech_detected'):
            state.perception['social'] = min(1.0, state.perception.get('social', 0.0) + 0.1)
        else:
            state.perception['social'] = max(0.0, state.perception.get('social', 0.0) - 0.02)
    
    def _compute_novelty(self, sensory: SensoryData) -> float:
        """Вычисляет новизну сенсорного входа"""
        if not self.history:
            return 0.5  # первое восприятие
        
        last = self.history[-1]
        score = 0.0
        
        # Сравниваем количество объектов
        if 'visual' in last and 'visual' in sensory.to_dict():
            last_objects = last['visual'].get('num_objects', 0)
            curr_objects = sensory.visual.get('num_objects', 0)
            if last_objects != curr_objects:
                score += 0.2
        
        # Сравниваем звуки
        if 'auditory' in last and 'auditory' in sensory.to_dict():
            last_sounds = last['auditory'].get('num_sounds', 0)
            curr_sounds = sensory.auditory.get('num_sounds', 0)
            if last_sounds != curr_sounds:
                score += 0.2
        
        # Сравниваем боль
        if 'interoceptive' in last and 'interoceptive' in sensory.to_dict():
            last_pain = last['interoceptive'].get('pain', 0)
            curr_pain = sensory.interoceptive.get('pain', 0)
            if abs(last_pain - curr_pain) > 0.1:
                score += 0.2
        
        return min(1.0, score)
    
    def get_sensory_data(self) -> dict:
        """Получить текущие сенсорные данные"""
        return self._generate_sensory_data(None).to_dict()
