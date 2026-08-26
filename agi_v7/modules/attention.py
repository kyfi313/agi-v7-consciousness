# -*- coding: utf-8 -*-
"""
Модуль внимания — выделяет наиболее важные объекты в сцене
"""

import numpy as np
from typing import Any

from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class AttentionModule(BaseModule):
    name = "attention"

    def __init__(self):
        self.capacity = CONFIG['ATTENTION_CAPACITY']
        self.focus = None

    def update(self, state: GlobalState) -> GlobalState:
        objects = state.objects
        if not objects:
            return state

        scored = []
        for obj in objects:
            score = self._compute_saliency(obj, state)
            scored.append((obj, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:self.capacity]

        if selected:
            state.attention_focus = selected[0][0]
            state.visual_saliency = np.array([s for _, s in selected])

        return state

    def _compute_saliency(self, obj: Any, state: GlobalState) -> float:
        base = 0.5
        goal = state.planning.get('current_goal')
        if goal is not None and str(goal) in str(obj):
            base += 0.3
        if hasattr(obj, 'novelty') and obj.novelty > 0.5:
            base += 0.2
        return base
