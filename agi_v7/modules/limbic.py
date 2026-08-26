# -*- coding: utf-8 -*-
"""
Лимбический модуль — управляет эмоциями, каскадом, нейромодуляторами
"""

import numpy as np

from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class LimbicModule(BaseModule):
    name = "limbic"

    def __init__(self):
        self.cascade_stages = CONFIG['CASCADE_STAGES']
        self.cascade_duration = CONFIG['CASCADE_DURATION']
        self.stage_counter = 0
        self.current_stage_index = -1

    def update(self, state: GlobalState) -> GlobalState:
        reward = state.learning.get('reward', 0.0)
        td_error = state.learning.get('td_error', 0.0)

        state.emotions['valence'] = np.clip(
            state.emotions['valence'] + 0.1 * (reward - state.emotions['valence']), 0, 1)
        state.emotions['arousal'] = np.clip(
            state.emotions['arousal'] + 0.05 * (abs(td_error) - state.emotions['arousal']), 0, 1)

        if state.perception.get('danger', False):
            state.emotions['fear'] = min(1.0, state.emotions['fear'] + 0.2)
        else:
            state.emotions['fear'] = max(0, state.emotions['fear'] - 0.01)

        novelty = state.perception.get('novelty', 0.0)
        state.emotions['curiosity'] = np.clip(
            state.emotions['curiosity'] + 0.02 * (novelty - state.emotions['curiosity']), 0, 1)

        if reward > 0.5:
            state.emotions['satisfaction'] = min(1.0, state.emotions['satisfaction'] + 0.05)
        else:
            state.emotions['satisfaction'] = max(0, state.emotions['satisfaction'] - 0.01)

        if reward < -0.3:
            state.emotions['frustration'] = min(1.0, state.emotions['frustration'] + 0.1)
        else:
            state.emotions['frustration'] = max(0, state.emotions['frustration'] - 0.01)

        state = self._update_cascade(state)
        state = self._update_neuromodulators(state)
        return state

    def _update_cascade(self, state: GlobalState) -> GlobalState:
        cascade = state.emotional_cascade

        if cascade['stage'] == 'idle':
            if state.emotions.get('ambition', 0) > 0.5 and state.get_energy() > 50:
                cascade['stage'] = 'ambition'
                cascade['progress'] = 0
                cascade['history'].append({'stage': 'ambition', 'time': state.time})
                state.emotions['ambition'] = 0.7
        elif cascade['stage'] != 'idle':
            cascade['progress'] += 1
            if cascade['progress'] >= self.cascade_duration:
                current_index = self.cascade_stages.index(cascade['stage'])
                can_progress = self._check_cascade_condition(cascade['stage'], state)
                if can_progress and current_index < len(self.cascade_stages) - 1:
                    next_stage = self.cascade_stages[current_index + 1]
                    cascade['stage'] = next_stage
                    cascade['progress'] = 0
                    cascade['history'].append({'stage': next_stage, 'time': state.time})
                    self._apply_stage_effects(next_stage, state)
                elif current_index >= len(self.cascade_stages) - 1:
                    cascade['stage'] = 'idle'
                    cascade['progress'] = 0
                    state.emotions['satisfaction'] = min(1.0, state.emotions['satisfaction'] + 0.1)

        return state

    def _check_cascade_condition(self, stage: str, state: GlobalState) -> bool:
        if stage == 'ambition':
            return state.planning.get('current_goal') is not None
        elif stage == 'focus':
            return state.learning.get('reward') is not None
        elif stage == 'frustration':
            return state.emotions.get('frustration', 0) > 0.3
        elif stage == 'analysis':
            return state.consciousness.get('reflection', '') != ''
        elif stage == 'reappraisal':
            return state.final_action is not None
        else:
            return True

    def _apply_stage_effects(self, stage: str, state: GlobalState) -> None:
        if stage == 'ambition':
            state.neuromodulators['dopamine'] = min(1.0, state.neuromodulators['dopamine'] + 0.1)
        elif stage == 'focus':
            state.neuromodulators['norepinephrine'] = min(1.0, state.neuromodulators['norepinephrine'] + 0.2)
            state.zone_modes['prefrontal_cortex'] = 'focused'
        elif stage == 'frustration':
            state.neuromodulators['norepinephrine'] = min(1.0, state.neuromodulators['norepinephrine'] + 0.15)
            state.emotions['frustration'] = min(1.0, state.emotions['frustration'] + 0.1)
        elif stage == 'analysis':
            state.neuromodulators['acetylcholine'] = min(1.0, state.neuromodulators['acetylcholine'] + 0.2)
            state.zone_modes['prefrontal_cortex'] = 'creative'
        elif stage == 'reappraisal':
            state.neuromodulators['serotonin'] = min(1.0, state.neuromodulators['serotonin'] + 0.15)
            state.emotions['satisfaction'] = min(1.0, state.emotions['satisfaction'] + 0.05)
        elif stage == 'resolution':
            state.neuromodulators['serotonin'] = min(1.0, state.neuromodulators['serotonin'] + 0.1)
            state.zone_modes['prefrontal_cortex'] = 'default'

    def _update_neuromodulators(self, state: GlobalState) -> GlobalState:
        state.neuromodulators['dopamine'] = np.clip(
            0.3 + 0.4 * state.emotions['valence'] + 0.3 * state.emotions['curiosity'], 0, 1)
        state.neuromodulators['serotonin'] = np.clip(
            0.3 + 0.5 * state.emotions['satisfaction'] + 0.2 * (1 - state.emotions['frustration']), 0, 1)
        state.neuromodulators['norepinephrine'] = np.clip(
            0.2 + 0.3 * state.emotions['fear'] + 0.3 * state.emotions['arousal'] + 0.2 * state.emotions['frustration'], 0, 1)
        state.neuromodulators['acetylcholine'] = np.clip(
            0.2 + 0.5 * state.emotions['curiosity'] + 0.3 * state.perception.get('novelty', 0), 0, 1)
        return state
