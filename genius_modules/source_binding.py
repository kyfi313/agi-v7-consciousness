# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ИСТОЧНИКОВАЯ ПРИВЯЗКА СИГНАЛОВ
Гениальность: Сигналы должны быть привязаны к своему источнику.
Без привязки — это просто шум. Привязка даёт смысл.

Реализовано: SignalSource — источник сигнала,
SignalBinding — привязка сигнала к источнику,
SourceBindingSystem — система управления привязками.
"""

import numpy as np
from collections import deque
import time
import hashlib

class SignalSource:
    """Источник сигнала."""
    def __init__(self, name, source_type='external', reliability=0.7):
        self.name = name
        self.source_type = source_type  # 'external', 'internal', 'memory'
        self.reliability = reliability
        self.signals_emitted = 0
        self.last_emission = 0.0
        self.pattern = None  # характерный паттерн источника
        self.id = hashlib.md5(name.encode()).hexdigest()[:8]

    def emit_signal(self, data, strength=0.5):
        """Излучает сигнал."""
        self.signals_emitted += 1
        self.last_emission = time.time()
        return Signal(data, source=self, strength=strength, timestamp=time.time())

    def get_state(self):
        return {
            'name': self.name,
            'type': self.source_type,
            'reliability': self.reliability,
            'emissions': self.signals_emitted,
            'last': self.last_emission
        }

class Signal:
    """Сигнал с привязкой к источнику."""
    def __init__(self, data, source, strength=0.5, timestamp=None):
        self.data = data
        self.source = source
        self.strength = strength
        self.timestamp = timestamp or time.time()
        self.id = hashlib.md5(f"{source.name}{self.timestamp}".encode()).hexdigest()[:8]
        self.binding_strength = 0.3
        self.processing_history = deque(maxlen=10)

    def bind_to_source(self):
        """Укрепляет привязку к источнику."""
        self.binding_strength = min(1.0, self.binding_strength + 0.1)
        self.processing_history.append('bind')

    def weaken_binding(self, amount=0.05):
        """Ослабляет привязку."""
        self.binding_strength = max(0.0, self.binding_strength - amount)
        self.processing_history.append('weaken')

    def is_confident(self):
        """Уверенность в источнике сигнала."""
        return self.binding_strength > 0.5

    def get_state(self):
        return {
            'id': self.id,
            'source': self.source.name,
            'strength': self.strength,
            'binding': self.binding_strength,
            'confident': self.is_confident(),
            'timestamp': self.timestamp
        }

class SourceBindingSystem:
    """Система управления привязками сигналов."""
    def __init__(self):
        self.sources = {}
        self.signals = deque(maxlen=100)
        self.bindings = {}  # signal_id -> binding_strength
        self.binding_history = deque(maxlen=50)
        self.conflict_threshold = 0.3

    def register_source(self, name, source_type='external', reliability=0.7):
        """Регистрирует источник сигналов."""
        source = SignalSource(name, source_type, reliability)
        self.sources[name] = source
        return source

    def receive_signal(self, data, source_name, strength=0.5):
        """Принимает сигнал от источника."""
        if source_name not in self.sources:
            # Авторегистрация неизвестного источника
            self.register_source(source_name, 'unknown', 0.3)
        source = self.sources[source_name]
        signal = source.emit_signal(data, strength)
        # Базовая привязка
        signal.binding_strength = source.reliability * 0.6
        self.signals.append(signal)
        self.bindings[signal.id] = signal.binding_strength
        self.binding_history.append({
            'time': time.time(),
            'source': source_name,
            'binding': signal.binding_strength
        })
        return signal

    def strengthen_binding(self, signal_id, amount=0.1):
        """Укрепляет привязку сигнала."""
        for signal in self.signals:
            if signal.id == signal_id:
                signal.bind_to_source()
                self.bindings[signal_id] = signal.binding_strength
                return True
        return False

    def detect_conflict(self, signal1, signal2):
        """Обнаруживает конфликт между сигналами."""
        # Конфликт = разные источники, разные данные, но похожие по времени
        if signal1.source.name == signal2.source.name:
            return 0.0  # Один источник
        if abs(signal1.timestamp - signal2.timestamp) > 2.0:
            return 0.0  # Слишком далеко во времени
        # Проверяем данные
        data1 = np.array(signal1.data) if isinstance(signal1.data, (list, np.ndarray)) else np.array([signal1.data])
        data2 = np.array(signal2.data) if isinstance(signal2.data, (list, np.ndarray)) else np.array([signal2.data])
        # Приводим к одинаковой длине
        if len(data1) > len(data2):
            data2 = np.pad(data2, (0, len(data1) - len(data2)), 'constant')
        elif len(data2) > len(data1):
            data1 = np.pad(data1, (0, len(data2) - len(data1)), 'constant')
        similarity = np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2) + 1e-8)
        # Если данные похожи, но источники разные — конфликт
        if similarity > 0.7:
            return min(1.0, 1.0 - similarity + 0.3)
        return 0.0

    def resolve_conflicts(self):
        """Разрешает конфликты между сигналами."""
        resolved = []
        signals_list = list(self.signals)
        for i in range(len(signals_list)):
            for j in range(i+1, len(signals_list)):
                conflict = self.detect_conflict(signals_list[i], signals_list[j])
                if conflict > self.conflict_threshold:
                    # Разрешаем в пользу сигнала с более сильной привязкой
                    if signals_list[i].binding_strength > signals_list[j].binding_strength:
                        signals_list[j].weaken_binding(conflict * 0.2)
                        resolved.append({'winner': signals_list[i].id, 'loser': signals_list[j].id})
                    else:
                        signals_list[i].weaken_binding(conflict * 0.2)
                        resolved.append({'winner': signals_list[j].id, 'loser': signals_list[i].id})
        return resolved

    def get_trusted_signals(self, min_binding=0.5):
        """Возвращает сигналы с привязкой выше порога."""
        return [s for s in self.signals if s.binding_strength > min_binding]

    def get_state(self):
        return {
            'sources': {name: src.get_state() for name, src in self.sources.items()},
            'signal_count': len(self.signals),
            'binding_count': len(self.bindings),
            'trusted_count': len(self.get_trusted_signals()),
            'history_size': len(self.binding_history)
        }

if __name__ == "__main__":
    print("="*60)
    print("🔗 ИСТОЧНИКОВАЯ ПРИВЯЗКА СИГНАЛОВ")
    print("="*60)
    system = SourceBindingSystem()
    # Регистрируем источники
    system.register_source('камера', 'external', 0.8)
    system.register_source('микрофон', 'external', 0.6)
    system.register_source('внутренний_голос', 'internal', 0.9)
    # Получаем сигналы
    signal1 = system.receive_signal([1.0, 0.0, 0.5], 'камера', 0.7)
    signal2 = system.receive_signal([1.1, 0.2, 0.4], 'микрофон', 0.5)
    signal3 = system.receive_signal([0.0, 1.0, 0.0], 'внутренний_голос', 0.9)
    # Укрепляем привязку
    system.strengthen_binding(signal1.id, 0.2)
    system.strengthen_binding(signal3.id, 0.3)
    # Разрешаем конфликты
    conflicts = system.resolve_conflicts()
    print(f"Конфликтов разрешено: {len(conflicts)}")
    # Доверенные сигналы
    trusted = system.get_trusted_signals(0.4)
    print(f"Доверенных сигналов: {len(trusted)}")
    for sig in trusted:
        print(f"  {sig.id} от {sig.source.name}, привязка={sig.binding_strength:.2f}")
    print("\n💡 Гениальность: Без привязки к источнику — сигнал это просто шум.")
    print("   Привязка даёт смысл и доверие.")
