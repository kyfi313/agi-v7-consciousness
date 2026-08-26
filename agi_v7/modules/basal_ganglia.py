# -*- coding: utf-8 -*-
"""
Модуль базальных ганглиев — Q-learning и выбор действий
"""

import numpy as np
import random
from collections import defaultdict

from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class BasalGangliaModule(BaseModule):
    name = "basal_ganglia"

    def __init__(self, state_size: int = 10, action_size: int = 10):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.learning_rate = CONFIG.get('LEARNING_RATE', 0.1)
        self.discount = CONFIG.get('GAMMA', 0.95)
        self.epsilon = CONFIG.get('EXPLORATION_RATE', 0.1)
        self.actions = ['explore', 'eat', 'rest', 'socialize', 'avoid', 'attack',
                        'investigate', 'flee', 'approach', 'wait']
        self._prev_state_key = None
        self._prev_action = None

    def update(self, state: GlobalState) -> GlobalState:
        state_vec = self._get_state_vector(state)
        state_key = tuple(state_vec[:self.state_size])
        reward = state.learning.get('reward', 0.0)

        if self._prev_state_key is not None and self._prev_action is not None:
            if state_key in self.q_table:
                current_q = self.q_table[state_key]
            else:
                current_q = np.zeros(self.action_size)
                self.q_table[state_key] = current_q
            best_next_q = np.max(current_q)
            td_target = reward + self.discount * best_next_q
            td_error = td_target - self.q_table[self._prev_state_key][self._prev_action]
            self.q_table[self._prev_state_key][self._prev_action] += self.learning_rate * td_error
            state.learning['td_error'] = td_error

        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            if state_key in self.q_table:
                action_idx = np.argmax(self.q_table[state_key])
            else:
                action_idx = random.randint(0, self.action_size - 1)

        action = self.actions[action_idx] if action_idx < len(self.actions) else 'explore'
        state.candidates['basal_ganglia'] = [action]

        self._prev_state_key = state_key
        self._prev_action = action_idx

        return state

    def _get_state_vector(self, state: GlobalState) -> np.ndarray:
        vec = []
        vec.append(state.get_energy() / 100.0)
        vec.append(state.body.get('hunger', 0) / 100.0)
        vec.append(state.body.get('fatigue', 0) / 100.0)
        vec.append(state.emotions.get('valence', 0.5))
        vec.append(state.emotions.get('arousal', 0.3))
        vec.append(state.emotions.get('fear', 0.1))
        vec.append(state.neuromodulators.get('dopamine', 0.5))
        vec.append(state.neuromodulators.get('serotonin', 0.5))
        vec.append(1.0 if state.perception.get('danger', False) else 0.0)
        vec.append(len(state.objects) / 10.0)
        return np.array(vec, dtype=np.float32)
