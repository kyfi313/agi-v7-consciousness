# -*- coding: utf-8 -*-
"""
Средний мозг (midbrain) — зрительные и слуховые рефлексы, ориентация в пространстве
"""

import numpy as np
from ..core.base import BaseModule
from ..core.state import GlobalState


class MidbrainModule(BaseModule):
    name = "midbrain"

    def __init__(self):
        self.orienting_gain = 1.0
        self.startle_threshold = 0.7
        self.reflex_speed = 0.8
        self.visual_tectum = np.array([0.5, 0.5])  # карта визуального пространства
        self.auditory_tectum = np.array([0.5, 0.5])  # карта слухового пространства
        self.last_orientation = np.array([0.5, 0.5])

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем визуальные и слуховые сигналы
        visual = state.perception.get('visual', {})
        auditory = state.perception.get('auditory', {})

        # Визуальный рефлекс: если есть движение — поворачиваем голову
        motion = visual.get('motion', 0.0)
        if motion > 0.3:
            # Вычисляем направление движения
            motion_dir = visual.get('motion_direction', [0.5, 0.5])
            if isinstance(motion_dir, list):
                motion_dir = np.array(motion_dir)
            self.visual_tectum = 0.7 * self.visual_tectum + 0.3 * motion_dir
            state.motor['orient_visual'] = self.visual_tectum.tolist()
            state.motor['reflex_visual'] = True

        # Слуховой рефлекс: громкий звук → поворот
        loudness = auditory.get('loudness', 0.0)
        if loudness > self.startle_threshold:
            # Испуганный рефлекс — поворот к источнику звука
            sound_dir = auditory.get('sound_direction', [0.5, 0.5])
            if isinstance(sound_dir, list):
                sound_dir = np.array(sound_dir)
            self.auditory_tectum = 0.7 * self.auditory_tectum + 0.3 * sound_dir
            state.motor['orient_auditory'] = self.auditory_tectum.tolist()
            state.motor['reflex_auditory'] = True
            # Испуганный рефлекс влияет на эмоции
            state.emotions['fear'] = min(1.0, state.emotions.get('fear', 0.0) + 0.15)

        # Обновляем ориентацию (среднее между зрительной и слуховой)
        self.last_orientation = 0.5 * self.visual_tectum + 0.5 * self.auditory_tectum
        state.perception['orientation'] = self.last_orientation.tolist()

        # Рефлекторная скорость (влияет на время реакции)
        state.motor['reflex_speed'] = self.reflex_speed
        if motion > 0.3 or loudness > self.startle_threshold:
            state.motor['reaction_time'] = 0.05  # очень быстро
        else:
            state.motor['reaction_time'] = 0.2

        return state
