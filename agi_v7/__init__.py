# -*- coding: utf-8 -*-
"""
AGI v7.0 — Модульная архитектура (основные модули)
"""

from .brain_module import NeuronNetwork, RealNeuron, MemoryCompressor
from .orchestrator import CognitiveOrchestrator
from .terminal_agent import TerminalAgent

__all__ = [
    'NeuronNetwork',
    'RealNeuron',
    'MemoryCompressor',
    'CognitiveOrchestrator',
    'TerminalAgent',
]
