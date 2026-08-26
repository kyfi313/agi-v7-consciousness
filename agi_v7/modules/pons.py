# -*- coding: utf-8 -*-
"""
Варолиев мост (pons) — связь между мозжечком и корой, координация движений
"""

import numpy as np
from ..core.base import BaseModule
from ..core.state import GlobalState


class PonsModule(BaseModule):
    name = "pons"

    def __init__(self):
        self.cerebellum_bridge = 0.5  # сила связи с мозжечком
        self.cortex_bridge = 0.5      # сила связи с корой
        self.signal_gain = 1.0
        self.crossing_delay = 3       # задержка перекрёстной передачи (в шагах)
        self.buffer = []

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем сигналы от мозжечка (если есть)
        cerebellum_signal = state.motor.get('cerebellum_correction', 0.0)

        # Получаем сигналы от коры (если есть)
        cortex_signal = state.consciousness.get('executive_signal', 0.0)

        # Комбинируем сигналы
        combined = (
            cerebellum_signal * self.cerebellum_bridge +
            cortex_signal * self.cortex_bridge
        ) * self.signal_gain

        # Добавляем задержку (перекрёстная передача)
        self.buffer.append(combined)
        if len(self.buffer) > self.crossing_delay:
            delayed_signal = self.buffer.pop(0)
        else:
            delayed_signal = combined

        # Мост также регулирует баланс между быстрыми и медленными движениями
        state.motor['pons_correction'] = delayed_signal * 0.3
        state.motor['balance_coef'] = 0.5 + 0.3 * np.tanh(delayed_signal)

        # Передаём в сознание
        state.consciousness['pons_activity'] = abs(combined)

        return state
