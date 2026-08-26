# -*- coding: utf-8 -*-
"""
Поясная кора — внимание, эмоциональный контроль, мониторинг ошибок
"""

import numpy as np
from ...core.base import BaseModule
from ...core.state import GlobalState


class CingulateCortexModule(BaseModule):
    name = "cingulate"

    def __init__(self):
        self.attention_span = 5
        self.error_monitor = []
        self.conflict_detected = False

    def update(self, state: GlobalState) -> GlobalState:
        # Мониторинг ошибок
        td_error = state.learning.get('td_error', 0.0)
        if abs(td_error) > 0.5:
            self.error_monitor.append(td_error)
            if len(self.error_monitor) > 10:
                self.error_monitor.pop(0)

        # Обнаружение конфликтов (противоречивые сигналы)
        self.conflict_detected = self._detect_conflict(state)

        # Регулировка внимания
        if self.conflict_detected:
            state.zone_modes['prefrontal_cortex'] = 'focused'
            state.zone_modes['parietal_lobe'] = 'focused'
            # Увеличиваем уровень сознания при конфликте
            state.consciousness['level'] = min(1.0, state.consciousness['level'] + 0.05)
        else:
            state.consciousness['level'] = max(0.1, state.consciousness['level'] - 0.01)

        # Влияние на эмоции (поясная кора связана с миндалевиной)
        if len(self.error_monitor) > 3 and np.mean(self.error_monitor) > 0.3:
            state.emotions['frustration'] = min(1.0, state.emotions['frustration'] + 0.05)
            state.emotions['arousal'] = min(1.0, state.emotions['arousal'] + 0.05)

        return state

    def _detect_conflict(self, state: GlobalState) -> bool:
        """Обнаружение конфликта между целями и текущими действиями"""
        goal = state.planning.get('current_goal')
        action = state.final_action

        if goal is None or action is None:
            return False

        # Простые конфликты
        if goal == 'найти_еду' and action == 'rest':
            return True
        if goal == 'избежать_опасности' and action == 'explore':
            return True
        if goal == 'исследовать' and state.get_energy() < 30:
            return True

        # Конфликт на основе эмоций
        if state.emotions.get('fear', 0) > 0.5 and state.emotions.get('curiosity', 0) > 0.5:
            return True

        return False
