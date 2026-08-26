# -*- coding: utf-8 -*-
"""
Модули AGI v7
"""

from .body import BodyModule
from .resource_allocator import ResourceAllocator
from .limbic import LimbicModule
from .basal_ganglia import BasalGangliaModule
from .habit_crystallizer import HabitCrystallizer
from .synaptic_pruning import SynapticPruning
from .attention import AttentionModule
from .thalamus import ThalamusModule
from .hippocampus import HippocampusModule
from .cerebellum import CerebellumModule
from .sensors import SensorModule
from .semantic_memory import SemanticMemory
from .memory_consolidator import MemoryConsolidator
from .value_system import ValueSystem
from .cortex import (
    PrefrontalCortexModule,
    VisualCortexModule,
    AuditoryCortexModule,
    MotorCortexModule,
    TemporalLobeModule,
    ParietalLobeModule,
    CingulateCortexModule,
    InsularCortexModule,
)
from .thalamus import ThalamusModule
from .hippocampus import HippocampusModule
from .cerebellum import CerebellumModule
from .cortex import (
    PrefrontalCortexModule,
    VisualCortexModule,
    CingulateCortexModule,
    InsularCortexModule,
)

__all__ = [
    'BodyModule',
    'ResourceAllocator',
    'LimbicModule',
    'BasalGangliaModule',
    'HabitCrystallizer',
    'SynapticPruning',
    'AttentionModule',
    'ThalamusModule',
    'HippocampusModule',
    'CerebellumModule',
    'PrefrontalCortexModule',
    'VisualCortexModule',
    'CingulateCortexModule',
    'SemanticMemory',
    'MemoryConsolidator',
    'InsularCortexModule',
    'ValueSystem',
]
