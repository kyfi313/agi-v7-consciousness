# -*- coding: utf-8 -*-
"""
Чёрная субстанция (substantia nigra) — дофаминовая регуляция, модуляция базальных ганглиев
"""

import numpy as np
from ..core.base import BaseModule
from ..core.state import GlobalState


class SubstantiaNigraModule(BaseModule):
    name = "substantia_nigra"

    def __init__(self):
        self.dopamine_level = 0.5
        self.dopamine_decay = 0.01
        self.reward_sensitivity = 0.8
        self.punishment_sensitivity = 0.6
        self.target_dopamine = 0.5
        self.last_reward = 0.0
        self.dopamine_history = []

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем текущую награду
        reward = state.learning.get('reward', 0.0)

        # Предсказание ошибки награды (как в дофаминовой теории)
        dopamine_change = (reward - self.last_reward) * self.reward_sensitivity

        # Если награда отрицательная — снижаем дофамин
        if reward < -0.1:
            dopamine_change -= abs(reward) * self.punishment_sensitivity

        # Обновляем дофамин с учётом распада
        self.dopamine_level = np.clip(
            self.dopamine_level * (1 - self.dopamine_decay) + dopamine_change * 0.1,
            0.0, 1.0
        )

        # Целевой уровень (гомеостаз)
        self.dopamine_level = 0.9 * self.dopamine_level + 0.1 * self.target_dopamine
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)

        # Сохраняем историю
        self.dopamine_history.append(self.dopamine_level)
        if len(self.dopamine_history) > 50:
            self.dopamine_history = self.dopamine_history[-50:]

        # Дофамин модулирует базальные ганглии (через состояние)
        state.neuromodulators['dopamine'] = self.dopamine_level
        state.basal_ganglia['dopamine'] = self.dopamine_level

        # Дофамин влияет на мотивацию и энергию
        state.motivation['drive'] = 0.5 + 0.5 * self.dopamine_level
        state.motivation['energy'] = 0.6 + 0.4 * self.dopamine_level

        # Дофамин влияет на обучение
        state.learning['dopamine'] = self.dopamine_level
        state.learning['learning_rate_mod'] = 0.5 + 0.5 * self.dopamine_level

        # Сохраняем для следующего шага
        self.last_reward = reward

        return state

    def get_dopamine_trend(self) -> str:
        """Возвращает тренд дофамина (для мета-познания)"""
        if len(self.dopamine_history) < 5:
            return "стабильный"
        recent = np.mean(self.dopamine_history[-5:])
        older = np.mean(self.dopamine_history[-10:-5]) if len(self.dopamine_history) >= 10 else recent
        if recent > older + 0.1:
            return "растёт"
        elif recent < older - 0.1:
            return "падает"
        else:
            return "стабильный"
