# -*- coding: utf-8 -*-
"""
Темпоральная доля — обработка речи, слуховой памяти и временных последовательностей
"""

import numpy as np
from collections import deque
from ...core.base import BaseModule
from ...core.state import GlobalState


class TemporalLobeModule(BaseModule):
    name = "temporal_lobe"

    def __init__(self):
        # Речевой буфер (последние 10 фраз)
        self.speech_history = deque(maxlen=10)
        # Семантическая память (значение слов)
        self.semantic_memory = {
            'hello': {'meaning': 'greeting', 'valence': 0.7},
            'help': {'meaning': 'request', 'valence': -0.2},
            'food': {'meaning': 'resource', 'valence': 0.8},
            'danger': {'meaning': 'threat', 'valence': -0.9},
            'follow': {'meaning': 'action', 'valence': 0.3},
            'stay': {'meaning': 'action', 'valence': 0.2},
            'yes': {'meaning': 'affirmation', 'valence': 0.5},
            'no': {'meaning': 'negation', 'valence': -0.3},
            'come here': {'meaning': 'directive', 'valence': 0.3},
            'go there': {'meaning': 'directive', 'valence': 0.2},
            'stop': {'meaning': 'directive', 'valence': -0.1},
            'run': {'meaning': 'action', 'valence': -0.4},
        }
        # Временная память (последовательности событий)
        self.temporal_memory = deque(maxlen=50)
        # Ассоциации
        self.associations = {}

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем слуховую информацию из коры
        auditory_processed = state.perception.get('auditory_processed', {})
        speech = auditory_processed.get('speech', {})

        if speech.get('detected', False):
            # Распознаём речь
            text = speech.get('text')
            if text:
                self._process_speech(text, state)

        # Сохраняем временную последовательность
        temporal_event = self._encode_temporal_event(state)
        self.temporal_memory.append(temporal_event)

        # Сохраняем в состояние
        state.perception['temporal_events'] = list(self.temporal_memory)[-20:]
        state.perception['speech_history'] = list(self.speech_history)
        state.perception['semantic_context'] = self._get_semantic_context(state)

        # Влияние на эмоции
        if self.speech_history:
            last_speech = self.speech_history[-1]
            meaning = last_speech.get('meaning', '')
            if meaning in ['threat', 'negation']:
                state.emotions['valence'] = max(0.0, state.emotions.get('valence', 0.5) - 0.1)
            elif meaning in ['resource', 'affirmation']:
                state.emotions['valence'] = min(1.0, state.emotions.get('valence', 0.5) + 0.1)

        return state

    def _process_speech(self, text: str, state: GlobalState):
        """Обрабатывает распознанную речь"""
        # Ищем в семантической памяти
        meaning = self.semantic_memory.get(text, {'meaning': 'unknown', 'valence': 0.0})

        # Сохраняем в историю
        entry = {
            'text': text,
            'meaning': meaning.get('meaning', 'unknown'),
            'valence': meaning.get('valence', 0.0),
            'step': state.step,
        }
        self.speech_history.append(entry)

        # Обновляем ассоциации
        if 'objects' in state.perception:
            obj_types = [obj.get('type') for obj in state.perception.get('objects', []) if isinstance(obj, dict)]
            for obj_type in obj_types:
                key = (text, obj_type)
                self.associations[key] = self.associations.get(key, 0) + 1

        # Речевой ответ (эмуляция)
        state.perception['speech_response'] = self._generate_response(text)

    def _generate_response(self, text: str) -> dict:
        """Генерирует речевой ответ"""
        responses = {
            'hello': 'Hello! How are you?',
            'help': 'I will try to help.',
            'food': 'Where is the food?',
            'danger': 'I will be careful.',
            'follow': 'I will follow you.',
            'stay': 'I will stay here.',
            'yes': 'Understood.',
            'no': 'Understood.',
            'come here': 'I am coming.',
            'go there': 'I will go there.',
            'stop': 'Stopping.',
            'run': 'Running!',
        }
        response = responses.get(text, f'What do you mean by "{text}"?')
        return {
            'text': response,
            'confidence': np.random.uniform(0.6, 0.9),
        }

    def _encode_temporal_event(self, state: GlobalState) -> dict:
        """Кодирует временное событие"""
        return {
            'step': state.step,
            'action': state.final_action,
            'valence': state.emotions.get('valence', 0.5),
            'speech': self.speech_history[-1].get('text') if self.speech_history else None,
            'objects': state.objects[:3] if state.objects else [],
        }

    def _get_semantic_context(self, state: GlobalState) -> dict:
        """Получает семантический контекст"""
        if not self.speech_history:
            return {'context': 'silent', 'valence': 0.0}

        # Анализируем последние 3 фразы
        recent = list(self.speech_history)[-3:]
        meanings = [e.get('meaning', 'unknown') for e in recent]

        # Определяем контекст
        if 'threat' in meanings:
            context = 'dangerous'
        elif 'resource' in meanings:
            context = 'opportunity'
        elif 'directive' in meanings:
            context = 'guided'
        elif 'greeting' in meanings:
            context = 'social'
        else:
            context = 'neutral'

        # Средняя валентность
        avg_valence = sum(e.get('valence', 0.0) for e in recent) / len(recent)

        return {
            'context': context,
            'valence': avg_valence,
            'recent_meanings': meanings,
        }

    def reset(self):
        self.speech_history.clear()
        self.temporal_memory.clear()
        self.associations = {}
