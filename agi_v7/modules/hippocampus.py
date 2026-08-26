# -*- coding: utf-8 -*-
"""
Гиппокамп — эпизодическая память, запоминание событий и последовательностей
"""

import numpy as np
from collections import deque
from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class HippocampusModule(BaseModule):
    name = "hippocampus"

    def __init__(self):
        self.capacity = CONFIG.get('MEMORY_CAPACITY', 10000)
        # Используем обычный список вместо deque, чтобы избежать проблем со слайсами
        self.episodic_buffer = []
        self.patterns = {}  # ключ: паттерн, значение: частота
        self.sequence_memory = {}  # ключ: последовательность, значение: частота

    def update(self, state: GlobalState) -> GlobalState:
        # Кодируем текущее состояние в эпизод
        episode = self._encode_episode(state)

        # Сохраняем в буфер
        self.episodic_buffer.append(episode)
        # Ограничиваем размер буфера
        if len(self.episodic_buffer) > self.capacity:
            self.episodic_buffer = self.episodic_buffer[-self.capacity:]

        # Обновляем паттерны (извлечение значимых шаблонов)
        self._update_patterns(episode)

        # Запоминаем последовательности (последние 5 шагов)
        if len(self.episodic_buffer) >= 5:
            last_five = self.episodic_buffer[-5:]
            # Преобразуем словари в хешируемые кортежи для использования в качестве ключа
            try:
                seq_hash = tuple(
                    tuple(sorted(ep.items())) if isinstance(ep, dict) else ep
                    for ep in last_five
                )
                # Используем только если seq_hash хешируемый (все элементы - кортежи)
                if all(isinstance(item, tuple) for item in seq_hash):
                    self.sequence_memory[seq_hash] = self.sequence_memory.get(seq_hash, 0) + 1
            except (TypeError, AttributeError):
                # Если не удалось хешировать, пропускаем
                pass

        # Формируем извлечённые воспоминания
        state.memory['episodic'] = self.episodic_buffer[-100:] if self.episodic_buffer else []
        state.memory['patterns'] = self.patterns
        state.memory['sequences'] = self.sequence_memory

        # Гиппокамп активируется при новизне
        novelty = state.perception.get('novelty', 0.0)
        if novelty > 0.5:
            state.emotions['curiosity'] = min(1.0, state.emotions.get('curiosity', 0.0) + 0.1)

        return state

    def _encode_episode(self, state: GlobalState) -> dict:
        # Безопасное извлечение объектов
        objects = []
        if hasattr(state, 'objects') and state.objects is not None:
            try:
                # Если это список или кортеж
                if isinstance(state.objects, (list, tuple)):
                    objects = state.objects[:5]
                # Если это словарь — берём значения
                elif isinstance(state.objects, dict):
                    objects = list(state.objects.values())[:5]
                # Если это массив NumPy
                elif hasattr(state.objects, '__array__'):
                    objects = state.objects.tolist()[:5]
                else:
                    objects = []
            except (TypeError, IndexError, AttributeError):
                objects = []
        
        return {
            'step': state.step,
            'energy': state.get_energy(),
            'valence': state.emotions.get('valence', 0.5),
            'action': state.final_action,
            'reward': state.learning.get('reward', 0.0),
            'objects': objects,
        }

    def _update_patterns(self, episode: dict):
        key = (episode.get('action'), round(episode.get('valence', 0.5), 2))
        self.patterns[key] = self.patterns.get(key, 0) + 1

    def recall(self, cue: dict) -> list:
        """Извлечение похожих эпизодов"""
        results = []
        for ep in self.episodic_buffer:
            score = 0
            for key, value in cue.items():
                if key in ep and ep[key] == value:
                    score += 1
            if score > 0:
                results.append((score, ep))
        results.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in results[:10]]
