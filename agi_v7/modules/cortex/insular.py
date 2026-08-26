# -*- coding: utf-8 -*-
"""
Островковая кора — интерцепция (восприятие внутреннего состояния тела)
"""

import numpy as np
from ...core.base import BaseModule
from ...core.state import GlobalState


class InsularCortexModule(BaseModule):
    name = "insular"

    def __init__(self):
        self.sensitivity = 0.8
        self.body_awareness = 0.5
        self.pain_threshold = 0.7

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем сигналы от тела
        energy = state.get_energy()
        hunger = state.body.get('hunger', 0.0)
        fatigue = state.body.get('fatigue', 0.0)
        pain = state.body.get('pain', 0.0)

        # Интегрируем внутренние ощущения
        interoceptive_signal = (
            (1.0 - energy / 100.0) * 0.4 +
            (hunger / 100.0) * 0.3 +
            (fatigue / 100.0) * 0.2 +
            pain * 0.1
        )

        # Обновляем осознание тела
        self.body_awareness = np.clip(
            self.body_awareness + 0.1 * (interoceptive_signal - self.body_awareness),
            0, 1
        )

        # Влияние на эмоции (островковая кора связана с миндалевиной)
        if hunger > 70:
            state.emotions['frustration'] = min(1.0, state.emotions['frustration'] + 0.02)
            state.emotions['valence'] = max(0, state.emotions['valence'] - 0.02)

        if fatigue > 80:
            state.emotions['arousal'] = max(0, state.emotions['arousal'] - 0.03)

        if pain > self.pain_threshold:
            state.emotions['fear'] = min(1.0, state.emotions['fear'] + 0.1)
            state.emotions['valence'] = max(0, state.emotions['valence'] - 0.1)

        # Передаём осознание тела в сознание
        state.consciousness['body_awareness'] = self.body_awareness

        # Сохраняем интерцептивный сигнал
        state.perception['interoceptive'] = interoceptive_signal

        return state
