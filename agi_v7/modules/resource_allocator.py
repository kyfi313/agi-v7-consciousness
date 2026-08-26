# -*- coding: utf-8 -*-
"""
Модуль распределения ресурсов — управляет режимами зон и аллокацией
"""

from ..core.base import BaseModule
from ..core.state import GlobalState


class ResourceAllocator(BaseModule):
    name = "resource_allocator"

    def __init__(self):
        self.zones = [
            'visual_cortex', 'auditory_cortex', 'prefrontal_cortex',
            'motor_cortex', 'temporal_lobe', 'parietal_lobe',
            'hippocampus', 'amygdala', 'thalamus',
            'basal_ganglia', 'cerebellum'
        ]

    def update(self, state: GlobalState) -> GlobalState:
        energy = state.get_energy()
        neuromodulators = state.neuromodulators
        emotions = state.emotions

        alloc = {'perception': 0.2, 'planning': 0.2, 'memory': 0.2,
                 'social': 0.1, 'motor': 0.2, 'consciousness': 0.1}

        if neuromodulators.get('norepinephrine', 0) > 0.6:
            alloc['perception'] += 0.2
            alloc['planning'] -= 0.1
            for zone in ['visual_cortex', 'auditory_cortex', 'thalamus']:
                state.set_mode(zone, 'focused')

        if neuromodulators.get('acetylcholine', 0) > 0.6:
            alloc['memory'] += 0.15
            alloc['planning'] += 0.1
            alloc['perception'] -= 0.05
            for zone in ['prefrontal_cortex', 'temporal_lobe', 'hippocampus']:
                state.set_mode(zone, 'creative')

        if neuromodulators.get('dopamine', 0) > 0.6:
            alloc['planning'] += 0.15
            alloc['motor'] += 0.05
            state.set_mode('prefrontal_cortex', 'focused')

        if neuromodulators.get('serotonin', 0) > 0.6:
            alloc['social'] += 0.15
            alloc['perception'] -= 0.05
            state.set_mode('temporal_lobe', 'default')

        if emotions.get('fear', 0) > 0.5:
            alloc['perception'] += 0.2
            alloc['planning'] -= 0.1
            for zone in ['visual_cortex', 'amygdala']:
                state.set_mode(zone, 'focused')

        if emotions.get('curiosity', 0) > 0.6:
            alloc['perception'] += 0.1
            alloc['planning'] += 0.1
            state.set_mode('prefrontal_cortex', 'creative')

        if energy < 30:
            for zone in state.zone_modes:
                if state.get_mode(zone) != 'default':
                    state.set_mode(zone, 'default')
            alloc['perception'] = max(0.1, alloc['perception'] - 0.05)
            alloc['planning'] = max(0.05, alloc['planning'] - 0.1)
            alloc['memory'] = max(0.05, alloc['memory'] - 0.05)

        if energy > 70:
            if emotions.get('curiosity', 0) > 0.5:
                state.set_mode('prefrontal_cortex', 'creative')
            if emotions.get('ambition', 0) > 0.5:
                state.set_mode('motor_cortex', 'focused')

        total = sum(alloc.values())
        if total > 0:
            for key in alloc:
                alloc[key] /= total

        state.allocation = alloc
        state.consciousness['level'] = min(1.0, (
            alloc.get('perception', 0) * 0.3 +
            alloc.get('planning', 0) * 0.4 +
            alloc.get('consciousness', 0) * 0.3
        ))

        return state
