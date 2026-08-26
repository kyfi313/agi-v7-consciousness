# -*- coding: utf-8 -*-
"""
Слуховая кора — обработка звуков, распознавание речи и паттернов
"""

import numpy as np
from collections import deque
from ...core.base import BaseModule
from ...core.state import GlobalState


class AuditoryCortexModule(BaseModule):
    name = "auditory_cortex"
    
    def __init__(self):
        # Память звуков (последние 20)
        self.sound_memory = deque(maxlen=20)
        # Карта распознанных звуков
        self.sound_map = {}
        # Обнаруженные паттерны
        self.patterns = {}
        # Речь — временный буфер
        self.speech_buffer = []
        # Пороги для разных типов звуков
        self.thresholds = {
            'speech': 0.3,
            'explosion': 0.4,
            'alarm': 0.3,
            'footstep': 0.2,
            'music': 0.2,
        }
        
    def update(self, state: GlobalState) -> GlobalState:
        # Получаем слуховые данные из таламуса
        thalamus_filtered = state.perception.get('thalamus_filtered', {})
        auditory = thalamus_filtered.get('auditory', {})
        
        # Если нет данных, возвращаем
        if not auditory:
            return state
        
        # Обрабатываем звуки
        processed = self._process_auditory(auditory)
        
        # Сохраняем в состояние
        state.perception['auditory_processed'] = processed
        state.perception['speech_detected'] = processed.get('speech', False)
        state.perception['sound_salience'] = processed.get('salience', 0.0)
        
        # Обновляем внимание
        if processed.get('speech', False):
            # Речь привлекает внимание
            state.attention_focus = 'speech'
            state.attention_salience = processed.get('salience', 0.0)
        elif processed.get('danger', False):
            # Опасные звуки привлекают внимание
            state.attention_focus = 'danger_sound'
            state.attention_salience = processed.get('salience', 0.0)
        
        # Обновляем социальное восприятие
        if processed.get('speech', False):
            state.perception['social'] = min(1.0, state.perception.get('social', 0.0) + 0.1)
        
        return state
    
    def _process_auditory(self, auditory: dict) -> dict:
        """Обрабатывает слуховые данные"""
        sounds = auditory.get('sounds', [])
        num_sounds = auditory.get('num_sounds', 0)
        loudness = auditory.get('loudness', 0.0)
        speech_detected = auditory.get('speech_detected', False)
        salience = auditory.get('salience', 0.0)
        
        # Запоминаем звуки
        for sound in sounds:
            self.sound_memory.append(sound)
            sound_type = sound.get('type', 'unknown')
            # Считаем частоту звуков
            self.sound_map[sound_type] = self.sound_map.get(sound_type, 0) + 1
        
        # Обнаруживаем паттерны (повторяющиеся звуки)
        patterns = self._detect_patterns(sounds)
        
        # Распознаём речь
        speech = self._process_speech(auditory)
        
        # Вычисляем опасность
        danger = self._detect_danger(sounds)
        
        return {
            'num_sounds': num_sounds,
            'loudness': loudness,
            'speech': speech,
            'danger': danger,
            'salience': salience,
            'patterns': patterns,
            'sound_types': list(self.sound_map.keys()),
            'sound_counts': self.sound_map,
        }
    
    def _process_speech(self, auditory: dict) -> dict:
        """Обрабатывает речь"""
        speech_detected = auditory.get('speech_detected', False)
        if not speech_detected:
            return {'detected': False, 'text': None, 'confidence': 0.0}
        
        # Эмуляция распознавания речи
        # В реальной системе здесь был бы STT (Speech-to-Text)
        sample_phrases = [
            'hello', 'help', 'food', 'danger', 'follow', 'stay',
            'yes', 'no', 'come here', 'go there', 'stop', 'run'
        ]
        
        confidence = min(1.0, 0.3 + np.random.random() * 0.5)
        if confidence > 0.5:
            text = np.random.choice(sample_phrases)
            self.speech_buffer.append(text)
            if len(self.speech_buffer) > 10:
                self.speech_buffer.pop(0)
            return {
                'detected': True,
                'text': text,
                'confidence': confidence,
                'buffer': self.speech_buffer[-5:],
            }
        else:
            return {
                'detected': True,
                'text': None,
                'confidence': confidence,
                'buffer': self.speech_buffer[-5:],
            }
    
    def _detect_patterns(self, sounds: list) -> list:
        """Обнаруживает повторяющиеся паттерны звуков"""
        patterns = []
        # Проверяем, есть ли повторяющиеся типы звуков
        sound_types = [s.get('type') for s in sounds if s.get('type')]
        for st in set(sound_types):
            count = sound_types.count(st)
            if count >= 2:
                patterns.append({'type': st, 'count': count})
        return patterns
    
    def _detect_danger(self, sounds: list) -> dict:
        """Обнаруживает опасные звуки"""
        danger = False
        threat_type = None
        threat_level = 0.0
        
        for sound in sounds:
            s_type = sound.get('type', 'unknown')
            volume = sound.get('volume', 0.0)
            
            if s_type == 'explosion':
                danger = True
                threat_type = 'explosion'
                threat_level = min(1.0, volume * 1.5)
            elif s_type == 'alarm':
                danger = True
                threat_type = 'alarm'
                threat_level = min(1.0, volume * 1.2)
            elif s_type == 'footstep' and volume > 0.7:
                danger = True
                threat_type = 'footstep'
                threat_level = volume
        
        return {
            'danger': danger,
            'threat_type': threat_type,
            'threat_level': threat_level,
        }
    
    def reset(self):
        self.sound_memory.clear()
        self.sound_map = {}
        self.patterns = {}
        self.speech_buffer = []
