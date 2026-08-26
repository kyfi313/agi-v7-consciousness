# -*- coding: utf-8 -*-
"""
Таламус — сенсорное реле, фильтрует и направляет сигналы в кору
"""

import numpy as np
from ..core.base import BaseModule
from ..core.state import GlobalState


class ThalamusModule(BaseModule):
    name = "thalamus"

    def __init__(self):
        self.gain = 0.7  # общий коэффициент усиления
        self.noise_threshold = 0.05  # порог шума
        self.attention_gain = 1.0
        self.last_visual = None
        self.last_auditory = None
        self.last_tactile = None

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем сырые сенсорные данные
        sensory = state.sensory if hasattr(state, 'sensory') else {}
        
        # Фильтруем и направляем в кору
        filtered = self._filter_sensory(sensory)
        
        # Сохраняем отфильтрованные сигналы в состоянии
        state.perception['thalamus_filtered'] = filtered
        state.perception['thalamus_gain'] = self.gain
        state.perception['thalamus_attention'] = self.attention_gain
        
        # Модулируем усиление в зависимости от внимания
        if state.attention_focus is not None:
            self.attention_gain = 1.5
        else:
            self.attention_gain = 1.0

        # Таламус также регулирует бодрствование
        state.consciousness['level'] = min(1.0, state.consciousness['level'] + 0.01)
        
        # Обновляем историю для отслеживания изменений
        if 'visual' in sensory:
            self.last_visual = sensory['visual']
        if 'auditory' in sensory:
            self.last_auditory = sensory['auditory']
        if 'tactile' in sensory:
            self.last_tactile = sensory['tactile']

        return state
    
    def _filter_sensory(self, sensory: dict) -> dict:
        """Фильтрует сенсорные данные и подготавливает их для коры"""
        filtered = {}
        
        # Визуальный поток
        if 'visual' in sensory and sensory['visual']:
            visual = sensory['visual']
            filtered['visual'] = {
                'objects': visual.get('objects', []),
                'num_objects': visual.get('num_objects', 0),
                'brightness': self._filter_value(visual.get('brightness', 0.5)),
                'motion': self._filter_value(visual.get('motion', 0.0)),
                'color_dominance': visual.get('color_dominance', 'neutral'),
                'salience': self._compute_salience(visual),
            }
        
        # Слуховой поток
        if 'auditory' in sensory and sensory['auditory']:
            auditory = sensory['auditory']
            filtered['auditory'] = {
                'sounds': auditory.get('sounds', []),
                'num_sounds': auditory.get('num_sounds', 0),
                'loudness': self._filter_value(auditory.get('loudness', 0.0)),
                'speech_detected': auditory.get('speech_detected', False),
                'salience': self._compute_salience(auditory),
            }
        
        # Тактильный поток
        if 'tactile' in sensory and sensory['tactile']:
            tactile = sensory['tactile']
            filtered['tactile'] = {
                'contact': tactile.get('contact', False),
                'pressure': self._filter_value(tactile.get('pressure', 0.0)),
                'texture': tactile.get('texture', None),
                'temperature': self._filter_value(tactile.get('temperature', 25.0)),
                'salience': self._compute_salience(tactile),
            }
        
        # Проприоцептивный поток
        if 'proprioceptive' in sensory and sensory['proprioceptive']:
            proprio = sensory['proprioceptive']
            filtered['proprioceptive'] = {
                'position': proprio.get('position', [0, 0, 0]),
                'rotation': proprio.get('rotation', [0, 0, 0]),
                'velocity': proprio.get('velocity', [0, 0, 0]),
                'balance': self._filter_value(proprio.get('balance', 1.0)),
                'salience': self._compute_salience(proprio),
            }
        
        # Интероцептивный поток (тело)
        if 'interoceptive' in sensory and sensory['interoceptive']:
            intero = sensory['interoceptive']
            filtered['interoceptive'] = {
                'temperature': self._filter_value(intero.get('temperature', 37.0)),
                'hunger': self._filter_value(intero.get('hunger', 0.0)),
                'thirst': self._filter_value(intero.get('thirst', 0.0)),
                'pain': self._filter_value(intero.get('pain', 0.0)),
                'energy': self._filter_value(intero.get('energy', 1.0)),
                'heart_rate': self._filter_value(intero.get('heart_rate', 70.0)),
                'stress': self._filter_value(intero.get('stress', 0.0)),
                'salience': self._compute_salience(intero),
            }
        
        return filtered
    
    def _filter_value(self, value: float) -> float:
        """Применяет порог шума и усиление к числовому значению"""
        if abs(value) < self.noise_threshold:
            return 0.0
        return value * self.gain * self.attention_gain
    
    def _compute_salience(self, data: dict) -> float:
        """Вычисляет салиентность (заметность) потока"""
        salience = 0.0
        
        # Для числовых значений — чем больше амплитуда, тем выше салиентность
        for key, value in data.items():
            if isinstance(value, (int, float)) and key != 'salience':
                salience += abs(value) * 0.1
        
        # Для списков — количество элементов
        if 'objects' in data and isinstance(data['objects'], list):
            salience += len(data['objects']) * 0.05
        if 'sounds' in data and isinstance(data['sounds'], list):
            salience += len(data['sounds']) * 0.05
        
        return min(1.0, salience)
