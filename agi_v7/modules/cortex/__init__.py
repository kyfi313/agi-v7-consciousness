# -*- coding: utf-8 -*-
"""
Кора головного мозга — неокортекс и связанные структуры
"""

from .prefrontal import PrefrontalCortexModule
from .visual import VisualCortexModule
from .auditory import AuditoryCortexModule
from .motor import MotorCortexModule
from .temporal import TemporalLobeModule
from .parietal import ParietalLobeModule
from .cingulate import CingulateCortexModule
from .insular import InsularCortexModule

__all__ = [
    'PrefrontalCortexModule',
    'VisualCortexModule',
    'CingulateCortexModule',
    'InsularCortexModule',
]
