# -*- coding: utf-8 -*-
"""
Модуль тела — управляет энергетикой, голодом, усталостью
"""

from ..core.base import BaseModule
from ..core.state import GlobalState
from ..config import CONFIG


class BodyModule(BaseModule):
    name = "body"

    def __init__(self):
        self.drain_rate = CONFIG['ENERGY_DRAIN_DEFAULT']
        self.regen_rate = CONFIG['ENERGY_REGEN_RATE']

    def update(self, state: GlobalState) -> GlobalState:
        total_drain = CONFIG['ENERGY_DRAIN_DEFAULT']
        for zone, mode in state.zone_modes.items():
            if mode == 'focused':
                total_drain += CONFIG['ENERGY_DRAIN_FOCUSED'] * 0.1
            elif mode == 'creative':
                total_drain += CONFIG['ENERGY_DRAIN_CREATIVE'] * 0.1

        state.body['energy'] = max(0, state.body['energy'] - total_drain)
        if state.body['energy'] < 100:
            state.body['energy'] = min(100, state.body['energy'] + self.regen_rate)

        state.body['hunger'] = min(100, state.body['hunger'] + 0.02)
        if state.final_action is not None:
            state.body['fatigue'] = min(100, state.body['fatigue'] + 0.01)
        if state.body['hunger'] > 80:
            state.body['energy'] = max(0, state.body['energy'] - 0.1)
        return state
