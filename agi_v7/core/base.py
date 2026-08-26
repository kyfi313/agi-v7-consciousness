# -*- coding: utf-8 -*-
"""
Базовый класс для всех модулей AGI v7.0
"""

from typing import Dict, Any, Optional

class BaseModule:
    """Базовый класс для всех модулей нейробиологической архитектуры."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.activated = False
        
    def activate(self) -> None:
        """Активировать модуль."""
        self.activated = True
        
    def deactivate(self) -> None:
        """Деактивировать модуль."""
        self.activated = False
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Обработать входные данные. Должен быть переопределён в наследниках."""
        return {}
        
    def reset(self) -> None:
        """Сбросить состояние модуля."""
        pass
