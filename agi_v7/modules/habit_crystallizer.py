# -*- coding: utf-8 -*-
"""
Модуль кристаллизации привычек — превращает повторяющиеся действия в привычки
"""

from collections import defaultdict

from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class HabitCrystallizer(BaseModule):
    name = "habit_crystallizer"

    def __init__(self):
        self.repetition_counter = defaultdict(int)
        self.strength = defaultdict(float)
        self.threshold = CONFIG['HABIT_STRENGTH_THRESHOLD']
        self.min_repetitions = CONFIG['HABIT_REPETITIONS']

    def update(self, state: GlobalState) -> GlobalState:
        action = state.final_action
        reward = state.learning.get('reward', 0.0)

        if action is not None:
            key = str(action)
            if reward > 0.1:
                self.repetition_counter[key] += 1
                self.strength[key] = min(1.0, self.strength[key] + 0.1)
            else:
                self.strength[key] = max(0.0, self.strength[key] - 0.02)

            if (self.repetition_counter[key] >= self.min_repetitions and
                    self.strength[key] >= self.threshold and
                    key not in state.habits['crystallized']):
                state.habits['crystallized'].append(key)
                state.habits['strength'][key] = self.strength[key]

        return state
