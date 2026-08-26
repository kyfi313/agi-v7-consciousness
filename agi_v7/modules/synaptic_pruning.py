# -*- coding: utf-8 -*-
"""
Модуль синаптического прунинга — удаляет редко используемые связи и привычки
"""

from collections import defaultdict

from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class SynapticPruning(BaseModule):
    name = "synaptic_pruning"

    def __init__(self):
        self.last_prune = 0
        self.interval = CONFIG['PRUNE_INTERVAL']
        self.threshold = CONFIG['PRUNE_THRESHOLD']
        self.usage = defaultdict(int)

    def update(self, state: GlobalState) -> GlobalState:
        action = state.final_action
        if action is not None:
            self.usage[str(action)] += 1

        if state.step - self.last_prune > self.interval:
            self._prune(state)
            self.last_prune = state.step

        return state

    def _prune(self, state: GlobalState) -> None:
        to_remove = []
        for key, count in self.usage.items():
            if count < self.threshold * self.interval:
                to_remove.append(key)
        for key in to_remove:
            del self.usage[key]
            if key in state.habits['crystallized']:
                state.habits['crystallized'].remove(key)
