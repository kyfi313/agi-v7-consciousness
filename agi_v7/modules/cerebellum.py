# -*- coding: utf-8 -*-
"""
Мозжечок — двигательное обучение, координация и автоматизация движений
"""

import numpy as np
from collections import defaultdict
from ..core.base import BaseModule
from ..core.state import GlobalState


class CerebellumModule(BaseModule):
    name = "cerebellum"

    def __init__(self):
        self.learning_rate = 0.1
        self.memory = defaultdict(float)  # ключ: состояние+действие, значение: предсказание
        self.error_history = []
        self.last_motor_command = None

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем текущее состояние и действие
        action = state.final_action
        reward = state.learning.get('reward', 0.0)

        if action is not None:
            # Кодируем состояние
            state_key = self._encode_state(state)
            action_key = str(action)
            key = (state_key, action_key)

            # Предсказываем результат
            if key in self.memory:
                predicted_reward = self.memory[key]
                # Обновляем предсказание на основе ошибки
                error = reward - predicted_reward
                self.memory[key] += self.learning_rate * error
                self.error_history.append(error)
            else:
                self.memory[key] = reward

            # Мозжечок предлагает корректировки движений
            if self.last_motor_command is not None:
                correction = self._compute_correction(state, action)
                state.motor['calibration'] = {'correction': correction}

            # Сохраняем для будущих корректировок
            self.last_motor_command = action

            # Добавляем кандидата на действие
            state.candidates['cerebellum'] = [f'cerebellum_{action}']

        return state

    def _encode_state(self, state: GlobalState) -> str:
        return f"{state.get_energy():.1f}_{state.emotions.get('valence', 0.5):.2f}"

    def _compute_correction(self, state: GlobalState, action: str) -> float:
        """Вычисляет поправку для действия на основе прошлых ошибок"""
        if len(self.error_history) < 5:
            return 0.0
        return np.mean(self.error_history[-5:]) * 0.1
