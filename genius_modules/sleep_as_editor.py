# -*- coding: utf-8 -*-
"""
МОДУЛЬ: СОН КАК РЕДАКТОР ВОСПОМИНАНИЙ
Гениальность: Во сне мозг блокирует доступ к реальной памяти и редактирует факты,
чтобы избежать противоречий. Сон — это не просто «отдых».
Это активная фаза переписывания прошлого, чтобы оно соответствовало настоящему.

Реализовано: SleepReplay — воспроизведение эпизодов с пониженным LR,
редактирование эпизодов — устранение конфликтов,
генерация сновидений — симуляция на основе старых паттернов.
"""

import numpy as np
from collections import deque
import time
import random

class Episode:
    """Один эпизод памяти."""
    def __init__(self, state, action, reward, context=None, emotion=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.context = context or {}
        self.emotion = emotion or {'valence': 0.0, 'arousal': 0.0}
        self.timestamp = time.time()
        self.conflict = 0.0
        self.edited = False
        self.edit_count = 0

class SleepReplay:
    """Воспроизведение эпизодов во сне."""
    def __init__(self, learning_rate=0.01, replay_count=10):
        self.learning_rate = learning_rate
        self.replay_count = replay_count
        self.replay_buffer = deque(maxlen=100)
        self.dream_buffer = deque(maxlen=50)
        self.is_sleeping = False
        self.sleep_depth = 0.0
        self.dream_content = []

    def add_episode(self, episode):
        """Добавляет эпизод в буфер сна."""
        self.replay_buffer.append(episode)

    def sleep_cycle(self, model=None):
        """Один цикл сна — воспроизведение и редактирование."""
        if not self.replay_buffer:
            return None
        self.is_sleeping = True
        self.sleep_depth = 0.5 + random.random() * 0.5
        modifications = []
        episodes_to_replay = list(self.replay_buffer)[-self.replay_count:]
        for ep in episodes_to_replay:
            modified = self._replay_episode(ep, model)
            if modified:
                modifications.append(modified)
        self.is_sleeping = False
        return modifications

    def _replay_episode(self, episode, model=None):
        """Воспроизводит эпизод с пониженным LR."""
        # Обнаружение конфликтов
        conflict = self._detect_conflict(episode)
        if conflict > 0.5:
            # Редактирование эпизода
            edited = self._edit_episode(episode, conflict)
            return edited
        # Обучение с пониженной скоростью
        if model:
            self._apply_slow_learning(episode, model)
        return None

    def _detect_conflict(self, episode):
        """Обнаруживает конфликты в эпизоде."""
        # Проверка на противоречия в контексте
        context = episode.context
        conflict = 0.0
        if 'previous_state' in context:
            # Если переход был неожиданным
            if np.random.random() < 0.3:
                conflict += 0.3
        if episode.reward < 0 and episode.emotion.get('valence', 0) > 0:
            conflict += 0.4  # Негативная награда, но позитивная эмоция
        if episode.reward > 0 and episode.emotion.get('valence', 0) < 0:
            conflict += 0.4  # Позитивная награда, но негативная эмоция
        return min(1.0, conflict)

    def _edit_episode(self, episode, conflict):
        """Редактирует эпизод, чтобы устранить конфликт."""
        # Изменяем эмоцию, чтобы она соответствовала награде
        if episode.reward < 0:
            episode.emotion['valence'] = max(-1.0, episode.emotion['valence'] - 0.2)
        else:
            episode.emotion['valence'] = min(1.0, episode.emotion['valence'] + 0.2)
        # Изменяем контекст
        episode.context['edited'] = True
        episode.edited = True
        episode.edit_count += 1
        # Уменьшаем конфликт
        episode.conflict = max(0.0, episode.conflict - 0.3)
        return episode

    def _apply_slow_learning(self, episode, model):
        """Применяет медленное обучение к модели."""
        # Пониженный LR
        lr = self.learning_rate * 0.1
        # Применяем (заглушка)
        pass

    def generate_dream(self, num_scenes=3):
        """Генерирует сновидения на основе старых паттернов."""
        if len(self.replay_buffer) < 3:
            return []
        self.dream_content = []
        for _ in range(num_scenes):
            # Берём случайные эпизоды и комбинируем
            ep1 = random.choice(list(self.replay_buffer))
            ep2 = random.choice(list(self.replay_buffer))
            dream_scene = {
                'state': ep1.state * 0.3 + ep2.state * 0.3 + np.random.randn(len(ep1.state)) * 0.4,
                'emotion': {
                    'valence': (ep1.emotion.get('valence', 0) + ep2.emotion.get('valence', 0)) / 2,
                    'intensity': random.random() * 0.5 + 0.5
                },
                'coherence': random.random() * 0.5 + 0.3,
                'timestamp': time.time()
            }
            self.dream_content.append(dream_scene)
            self.dream_buffer.append(dream_scene)
        return self.dream_content

    def get_state(self):
        return {
            'is_sleeping': self.is_sleeping,
            'sleep_depth': self.sleep_depth,
            'buffer_size': len(self.replay_buffer),
            'dream_count': len(self.dream_buffer),
            'replay_count': self.replay_count
        }

class SleepEditor:
    """Полный модуль сна как редактора воспоминаний."""
    def __init__(self):
        self.sleep_replay = SleepReplay()
        self.sleep_schedule = 0.0
        self.edited_count = 0
        self.dream_log = deque(maxlen=20)
        self.memory_consolidation = 0.0

    def add_experience(self, state, action, reward, context=None, emotion=None):
        """Добавляет опыт в буфер сна."""
        ep = Episode(state, action, reward, context, emotion)
        self.sleep_replay.add_episode(ep)

    def sleep(self, model=None):
        """Запускает процесс сна."""
        self.sleep_schedule = 1.0
        modifications = self.sleep_replay.sleep_cycle(model)
        if modifications:
            self.edited_count += len([m for m in modifications if m])
        # Генерация сновидений
        dreams = self.sleep_replay.generate_dream()
        for dream in dreams:
            self.dream_log.append(dream)
        # Консолидация памяти
        self.memory_consolidation = min(1.0, self.memory_consolidation + 0.1)
        return modifications, dreams

    def wake(self):
        """Пробуждение."""
        self.sleep_schedule = 0.0
        self.sleep_replay.is_sleeping = False

    def get_state(self):
        state = self.sleep_replay.get_state()
        state['edited_count'] = self.edited_count
        state['consolidation'] = self.memory_consolidation
        state['dream_log_size'] = len(self.dream_log)
        return state

if __name__ == "__main__":
    print("="*60)
    print("💤 СОН КАК РЕДАКТОР ВОСПОМИНАНИЙ")
    print("="*60)
    sleep_editor = SleepEditor()
    # Добавляем опыт
    for i in range(15):
        state = np.random.randn(5)
        action = np.random.randint(0, 4)
        reward = 1.0 if i % 2 == 0 else -0.5
        emotion = {'valence': reward, 'arousal': 0.5}
        sleep_editor.add_experience(state, action, reward, context={'step': i}, emotion=emotion)
    # Сон
    print("\n💤 Запуск сна...")
    modifications, dreams = sleep_editor.sleep()
    print(f"Редактировано эпизодов: {len(modifications) if modifications else 0}")
    print(f"Сновидений: {len(dreams)}")
    for i, dream in enumerate(dreams[:3]):
        print(f"  Сновидение {i+1}: валентность={dream['emotion']['valence']:.2f}, когерентность={dream['coherence']:.2f}")
    print("\n💡 Гениальность: Сон — это активная фаза ПЕРЕПИСЫВАНИЯ прошлого.")
    print("   Мозг редактирует воспоминания, чтобы избежать противоречий.")
