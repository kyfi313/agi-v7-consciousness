# -*- coding: utf-8 -*-
"""
Единый протокол обмена данными между нейронами, сознанием и модулями AGI v7.
Все компоненты говорят на одном языке.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import hashlib
import time


class SignalType(Enum):
    """Типы сигналов в системе."""
    # Сенсорные
    PERCEPTION = "perception"
    SENSORY = "sensory"
    VISUAL = "visual"
    AUDITORY = "auditory"
    
    # Когнитивные
    THOUGHT = "thought"
    PLAN = "plan"
    INTENTION = "intention"
    DECISION = "decision"
    
    # Эмоциональные
    EMOTION = "emotion"
    FEAR = "fear"
    CURIOSITY = "curiosity"
    SURPRISE = "surprise"
    
    # Нейронные
    SPIKE = "spike"
    FIRING_RATE = "firing_rate"
    SYNAPSE = "synapse"
    
    # Социальные
    SOCIAL = "social"
    COMMUNICATION = "communication"
    
    # Системные
    MEMORY = "memory"
    LEARNING = "learning"
    SLEEP = "sleep"
    
    # Контрольные
    CONTROL = "control"
    COMMAND = "command"
    STATUS = "status"
    ERROR = "error"


@dataclass
class NeuroSignal:
    """
    Универсальный сигнал для обмена между всеми компонентами.
    
    Примеры:
        - Сенсорный сигнал: NeuroSignal(type=SignalType.PERCEPTION, payload={"position": (5,10), "color": (255,0,0)})
        - Нейронный спайк: NeuroSignal(type=SignalType.SPIKE, payload={"neuron_id": 42, "time": 0.123})
        - План: NeuroSignal(type=SignalType.PLAN, payload={"action": "move", "target": (3,7), "priority": 0.9})
        - Эмоция: NeuroSignal(type=SignalType.EMOTION, payload={"fear": 0.7, "curiosity": 0.3, "surprise": 0.5})
    """
    
    type: SignalType
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    target: str = ""
    strength: float = 0.5
    priority: int = 0
    hemisphere: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.type.value}{self.source}{self.target}{str(self.payload)[:50]}{self.timestamp}".encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует сигнал в словарь для сериализации."""
        return {
            'type': self.type.value,
            'payload': self.payload,
            'source': self.source,
            'target': self.target,
            'strength': self.strength,
            'priority': self.priority,
            'hemisphere': self.hemisphere,
            'timestamp': self.timestamp,
            'id': self.id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuroSignal':
        """Создаёт сигнал из словаря."""
        return cls(
            type=SignalType(data.get('type', 'perception')),
            payload=data.get('payload', {}),
            source=data.get('source', ''),
            target=data.get('target', ''),
            strength=data.get('strength', 0.5),
            priority=data.get('priority', 0),
            hemisphere=data.get('hemisphere'),
            timestamp=data.get('timestamp', time.time()),
            id=data.get('id', ''),
        )
    
    def to_legacy_signal(self):
        """Преобразует в Signal из consciousness_dispatcher (обратная совместимость)."""
        from agi_v7.consciousness_dispatcher import Signal as LegacySignal
        return LegacySignal(
            source=self.source,
            target=self.target or "dispatcher",
            data=self.to_dict(),
            strength=self.strength,
            priority=self.priority,
            hemisphere=self.hemisphere,
        )
    
    def __repr__(self):
        return f"NeuroSignal({self.type.value} from {self.source}->{self.target}, s={self.strength:.2f})"


class NeuroProtocol:
    """
    Единый протокол обмена данными.
    Преобразует данные между разными форматами.
    """
    
    @staticmethod
    def from_spikes(spikes: List[float], source: str = "brain") -> NeuroSignal:
        """Создаёт сигнал из спайков нейронов."""
        return NeuroSignal(
            type=SignalType.SPIKE,
            payload={
                'spikes': spikes,
                'count': len(spikes),
                'mean': float(np.mean(spikes)) if spikes else 0.0,
                'max': float(np.max(spikes)) if spikes else 0.0,
                'sum': float(np.sum(spikes)) if spikes else 0.0,
            },
            source=source,
            strength=0.5 + 0.5 * (float(np.mean(spikes)) if spikes else 0.0),
        )
    
    @staticmethod
    def from_firing_rates(rates: List[float], source: str = "brain") -> NeuroSignal:
        """Создаёт сигнал из частот срабатывания нейронов."""
        return NeuroSignal(
            type=SignalType.FIRING_RATE,
            payload={
                'rates': rates,
                'mean': float(np.mean(rates)) if rates else 0.0,
                'std': float(np.std(rates)) if rates else 0.0,
                'max': float(np.max(rates)) if rates else 0.0,
            },
            source=source,
            strength=0.5 + 0.5 * (float(np.mean(rates)) if rates else 0.0),
        )
    
    @staticmethod
    def from_emotion(fear: float, curiosity: float = 0.5, surprise: float = 0.2) -> NeuroSignal:
        """Создаёт эмоциональный сигнал."""
        return NeuroSignal(
            type=SignalType.EMOTION,
            payload={
                'fear': min(1.0, fear),
                'curiosity': min(1.0, curiosity),
                'surprise': min(1.0, surprise),
            },
            source="emotion",
            strength=0.3 + 0.7 * max(fear, curiosity, surprise),
        )
    
    @staticmethod
    def from_perception(data: Dict[str, Any]) -> NeuroSignal:
        """Создаёт сенсорный сигнал."""
        return NeuroSignal(
            type=SignalType.PERCEPTION,
            payload=data,
            source="perception",
            strength=0.6,
        )
    
    @staticmethod
    def from_plan(action: str, target: Any, priority: float = 0.5) -> NeuroSignal:
        """Создаёт сигнал плана."""
        return NeuroSignal(
            type=SignalType.PLAN,
            payload={'action': action, 'target': target, 'priority': priority},
            source="planner",
            strength=0.7 + 0.3 * priority,
            priority=int(priority * 3),
        )
    
    @staticmethod
    def to_spikes(signal: NeuroSignal) -> List[float]:
        """Извлекает спайки из сигнала."""
        if signal.type == SignalType.SPIKE:
            return signal.payload.get('spikes', [])
        elif signal.type == SignalType.FIRING_RATE:
            # Преобразуем частоты в спайки с вероятностью
            rates = signal.payload.get('rates', [])
            return [1.0 if np.random.random() < r else 0.0 for r in rates]
        elif isinstance(signal.payload, dict) and 'spikes' in signal.payload:
            return signal.payload['spikes']
        return []
    
    @staticmethod
    def to_emotion(signal: NeuroSignal) -> Dict[str, float]:
        """Извлекает эмоции из сигнала."""
        if signal.type == SignalType.EMOTION:
            return {
                'fear': signal.payload.get('fear', 0.0),
                'curiosity': signal.payload.get('curiosity', 0.5),
                'surprise': signal.payload.get('surprise', 0.2),
            }
        if isinstance(signal.payload, dict):
            return {
                'fear': signal.payload.get('fear', 0.0),
                'curiosity': signal.payload.get('curiosity', 0.5),
                'surprise': signal.payload.get('surprise', 0.2),
            }
        return {'fear': 0.0, 'curiosity': 0.5, 'surprise': 0.2}
    
    @staticmethod
    def to_action(signal: NeuroSignal) -> Optional[str]:
        """Извлекает действие из сигнала."""
        if signal.type == SignalType.PLAN:
            return signal.payload.get('action')
        if signal.type == SignalType.DECISION:
            return signal.payload.get('action')
        if isinstance(signal.payload, dict):
            return signal.payload.get('action')
        return None
    
    @staticmethod
    def extract_data(signal: NeuroSignal) -> Dict[str, Any]:
        """Универсальное извлечение данных из сигнала."""
        return signal.payload.copy()


# ============================================================
# АДАПТЕР ДЛЯ BRAIN_INTERFACE
# ============================================================

class BrainInterfaceAdapter:
    """
    Адаптер между BrainInterface и NeuroProtocol.
    Позволяет нейронам и сознанию говорить на одном языке.
    """
    
    def __init__(self, brain_interface):
        self.brain = brain_interface
        self.last_signal = None
        self.history = []
    
    def step(self, signal: Optional[NeuroSignal] = None) -> NeuroSignal:
        """
        Один шаг через адаптер.
        
        Args:
            signal: Входной сигнал (опционально). Если не передан, мозг работает автономно.
            
        Returns:
            NeuroSignal: Выходной сигнал мозга.
        """
        # Преобразуем входной сигнал в список для нейронов
        if signal:
            external_input = self._signal_to_input(signal)
            result = self.brain.step(external_input)
        else:
            result = self.brain.step()
        
        # Преобразуем результат в NeuroSignal
        output = self._result_to_signal(result)
        self.last_signal = output
        self.history.append(output)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return output
    
    def _signal_to_input(self, signal: NeuroSignal) -> List[float]:
        """Преобразует NeuroSignal в список float для нейронов."""
        # Получаем данные из сигнала
        data = NeuroProtocol.extract_data(signal)
        
        # Преобразуем в список чисел
        if 'spikes' in data:
            return data['spikes']
        elif 'rates' in data:
            return data['rates']
        elif 'values' in data:
            return data['values']
        else:
            # Пытаемся извлечь числа из словаря
            values = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    values.extend(value)
            return values[:self.brain.max_neurons] if values else [signal.strength] * min(10, self.brain.max_neurons)
    
    def _result_to_signal(self, result: Dict[str, Any]) -> NeuroSignal:
        """Преобразует результат мозга в NeuroSignal."""
        # Определяем тип сигнала
        if 'fear' in result and 'curiosity' in result:
            return NeuroProtocol.from_emotion(
                fear=result.get('fear', 0.0),
                curiosity=result.get('curiosity', 0.5),
                surprise=result.get('surprise', 0.2),
            )
        elif 'spikes' in result:
            return NeuroProtocol.from_spikes(
                spikes=result.get('spikes', []),
                source='brain'
            )
        elif 'firing_rates' in result:
            return NeuroProtocol.from_firing_rates(
                rates=result.get('firing_rates', []),
                source='brain'
            )
        else:
            # Общий случай
            return NeuroSignal(
                type=SignalType.STATUS,
                payload=result,
                source='brain',
                strength=0.5,
            )
    
    def get_state(self) -> NeuroSignal:
        """Возвращает состояние мозга в виде NeuroSignal."""
        state = self.brain.get_state()
        return NeuroSignal(
            type=SignalType.STATUS,
            payload=state,
            source='brain',
            strength=0.5,
        )


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

if __name__ == "__main__":
    print("🧠 Единый протокол обмена данными AGI v7")
    print("=" * 50)
    
    # Создаём сигналы разных типов
    spike_signal = NeuroProtocol.from_spikes([0.1, 0.8, 0.3, 0.9, 0.2])
    emotion_signal = NeuroProtocol.from_emotion(0.8, 0.3, 0.5)
    plan_signal = NeuroProtocol.from_plan("move", (5, 10), 0.8)
    perception_signal = NeuroProtocol.from_perception({"position": (3, 4), "food": True})
    
    print("\n📡 Сигналы:")
    print(f"  {spike_signal}")
    print(f"  {emotion_signal}")
    print(f"  {plan_signal}")
    print(f"  {perception_signal}")
    
    print("\n📊 Преобразования:")
    # Извлечение спайков
    spikes = NeuroProtocol.to_spikes(spike_signal)
    print(f"  Спайки: {spikes}")
    
    # Извлечение эмоций
    emotions = NeuroProtocol.to_emotion(emotion_signal)
    print(f"  Эмоции: {emotions}")
    
    # Извлечение действия
    action = NeuroProtocol.to_action(plan_signal)
    print(f"  Действие: {action}")
    
    print("\n✅ Единый протокол работает.")
    print("\n🧭 Да пребудет с вами лес.")
