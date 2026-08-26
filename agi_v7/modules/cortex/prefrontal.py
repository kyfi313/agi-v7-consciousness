# -*- coding: utf-8 -*-
"""
Префронтальная кора — планирование, контроль импульсов, принятие решений
"""

import numpy as np
from ...core.base import BaseModule
from ...core.state import GlobalState


class PrefrontalCortexModule(BaseModule):
    name = "prefrontal"

    def __init__(self):
        self.planning_depth = 3
        self.inhibition_strength = 0.5
        self.goal_stack = []

    def update(self, state: GlobalState) -> GlobalState:
        # Проверяем, есть ли цель
        current_goal = state.planning.get('current_goal')

        # Если цели нет, создаём её
        if current_goal is None and state.get_energy() > 50:
            state.planning['current_goal'] = self._generate_goal(state)

        # Планирование: разбиваем цель на подцели
        if current_goal is not None:
            subgoals = self._decompose_goal(current_goal, state)
            state.planning['subgoals'] = subgoals
            if len(subgoals) > 0:
                state.planning['step'] += 1
                if state.planning['step'] >= len(subgoals):
                    state.planning['step'] = 0
                    state.planning['subgoals'] = []
                    state.planning['current_goal'] = None

        # Торможение импульсов (проверка, стоит ли действовать)
        if state.final_action is not None:
            if not self._should_act(state):
                state.final_action = 'wait'

        # Обработка обратной связи от лимбической системы
        if state.emotions.get('frustration', 0) > 0.5:
            self.inhibition_strength = max(0.1, self.inhibition_strength - 0.1)
        if state.emotions.get('satisfaction', 0) > 0.5:
            self.inhibition_strength = min(1.0, self.inhibition_strength + 0.1)

        return state

    def _generate_goal(self, state: GlobalState) -> str:
        """Генерирует цель на основе текущего состояния"""
        if state.body.get('hunger', 0) > 70:
            return 'найти_еду'
        if state.get_energy() < 30:
            return 'восстановить_энергию'
        if state.perception.get('danger', False):
            return 'избежать_опасности'
        if state.emotions.get('curiosity', 0) > 0.7:
            return 'исследовать'
        return 'выжить'

    def _decompose_goal(self, goal: str, state: GlobalState) -> list:
        """Разбивает цель на подцели"""
        if goal == 'найти_еду':
            return ['осмотреться', 'найти_источник_еды', 'приблизиться', 'съесть']
        elif goal == 'восстановить_энергию':
            return ['отдыхать', 'восстановиться']
        elif goal == 'избежать_опасности':
            return ['оценить_угрозу', 'отступить', 'спрятаться']
        elif goal == 'исследовать':
            return ['выбрать_направление', 'двигаться', 'анализировать']
        else:
            return ['наблюдать', 'анализировать']

    def _should_act(self, state: GlobalState) -> bool:
        """Проверяет, стоит ли действовать (торможение импульсов)"""
        # Если энергия низкая — не действуем
        if state.get_energy() < 20:
            return False
        # Если страх слишком высокий — избегаем действий
        if state.emotions.get('fear', 0) > 0.8:
            return False
        # Если есть опасность — действуем осторожно
        if state.perception.get('danger', False):
            return self.inhibition_strength > 0.3
        return True
