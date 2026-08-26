# -*- coding: utf-8 -*-
"""
Поток сознания (Stream of Consciousness)
Непрерывный внутренний монолог, где агент осознаёт себя.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time


@dataclass
class Thought:
    """Единица мышления."""
    content: str
    source: str  # 'perception', 'memory', 'emotion', 'planning', 'reflection'
    timestamp: float = field(default_factory=time.time)
    intensity: float = 1.0  # насколько сильно осознаётся
    emotional_valence: float = 0.0  # -1...1
    connected_thoughts: List[str] = field(default_factory=list)


class ConsciousnessStream:
    """
    Поток сознания — непрерывный внутренний диалог.
    Память — это продолжение мышления, мгновенный доступ.
    """
    
    def __init__(self, buffer_size: int = 100):
        self.thoughts: deque = deque(maxlen=buffer_size)
        self.current_thought: Optional[Thought] = None
        self.attention_focus: Optional[str] = None  # на что направлено внимание
        self.self_awareness: Dict[str, Any] = {
            'name': 'Я',
            'feeling': None,
            'wanting': None,
            'remembering': None,
            'planning': None,
        }
        self.stream_active = True
        self.reflection_depth = 0
        
    def add_thought(self, content: str, source: str, 
                    emotional_valence: float = 0.0,
                    intensity: float = 1.0):
        """Добавляет мысль в поток сознания."""
        thought = Thought(
            content=content,
            source=source,
            emotional_valence=emotional_valence,
            intensity=intensity
        )
        self.thoughts.append(thought)
        self.current_thought = thought
        
    def get_latest_thoughts(self, n: int = 5) -> List[Thought]:
        """Возвращает последние n мыслей."""
        return list(self.thoughts)[-n:]
    
    def get_conscious_summary(self) -> str:
        """Получить краткое описание текущего состояния сознания."""
        if not self.thoughts:
            return "Сознание пусто..."
        latest = list(self.thoughts)[-3:]
        summary = " ".join([f"[{t.source}] {t.content}" for t in latest])
        return summary
    
    def update_self_awareness(self, feeling: str = None, wanting: str = None,
                              remembering: str = None, planning: str = None):
        """Обновляет самосознание агента."""
        if feeling is not None:
            self.self_awareness['feeling'] = feeling
        if wanting is not None:
            self.self_awareness['wanting'] = wanting
        if remembering is not None:
            self.self_awareness['remembering'] = remembering
        if planning is not None:
            self.self_awareness['planning'] = planning
            
    def reflect(self) -> str:
        """Рефлексия — осознание собственного мышления."""
        self.reflection_depth += 1
        if self.current_thought:
            return f"Я осознаю: {self.current_thought.content} (глубина {self.reflection_depth})"
        return "Я думаю о своём мышлении..."
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'thoughts_count': len(self.thoughts),
            'current_thought': self.current_thought.content if self.current_thought else None,
            'attention_focus': self.attention_focus,
            'self_awareness': self.self_awareness,
            'reflection_depth': self.reflection_depth,
            'stream_active': self.stream_active
        }


class MindIntegration:
    """
    Интегратор всех модулей в единый мыслительный процесс.
    Позволяет агенту мыслить с помощью модулей.
    """
    
    def __init__(self):
        self.modules = {}  # название модуля -> ссылка
        self.module_memory = {}  # название модуля -> последний результат
        self.integration_context = {}
        self.processing_depth = 0
        
    def register_module(self, name: str, module):
        """Регистрирует модуль для использования в мышлении."""
        self.modules[name] = module
        self.module_memory[name] = None
        
    def think_with_module(self, module_name: str, input_data: Any) -> Any:
        """
        Мыслит с использованием конкретного модуля.
        Агент обращается к модулю как к части своего мышления.
        """
        if module_name not in self.modules:
            return None
        
        module = self.modules[module_name]
        self.processing_depth += 1
        
        # Если у модуля есть метод process или update
        if hasattr(module, 'process'):
            result = module.process(input_data)
        elif hasattr(module, 'update'):
            result = module.update(input_data)
        else:
            result = None
            
        self.module_memory[module_name] = result
        self.processing_depth -= 1
        
        return result
    
    def recall_from_memory(self, query: Any) -> Any:
        """
        Мгновенное обращение к памяти (как продолжение мышления).
        Не запрос, а естественное извлечение.
        """
        if 'hippocampus' in self.modules:
            hippocampus = self.modules['hippocampus']
            if hasattr(hippocampus, 'recall'):
                return hippocampus.recall(query)
        return None
    
    def integrate(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Интегрирует сигналы от всех модулей в единое состояние.
        """
        integrated = {
            'perception': {},
            'memory': {},
            'emotion': {},
            'planning': {},
            'action': None,
            'consciousness': {}
        }
        
        # Собираем данные из всех модулей
        for name, module in self.modules.items():
            if hasattr(module, 'get_state'):
                try:
                    state = module.get_state()
                    if name == 'hippocampus':
                        integrated['memory'] = state
                    elif name == 'limbic':
                        integrated['emotion'] = state
                    elif name == 'planner':
                        integrated['planning'] = state
                    elif name == 'world_model':
                        integrated['perception'] = state
                    elif name == 'consciousness_dispatcher':
                        integrated['consciousness'] = state
                except Exception:
                    pass
                    
        # Добавляем контекст интеграции
        integrated['integration_depth'] = self.processing_depth
        integrated['timestamp'] = time.time()
        
        return integrated
