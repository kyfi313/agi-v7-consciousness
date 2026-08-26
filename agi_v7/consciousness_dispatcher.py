# -*- coding: utf-8 -*-
"""
Модуль сознания как диспетчера с биологической архитектурой.
Реализует модель с разделением на левое и правое полушарие,
двунаправленными маршрутами через мозолистое тело,
асимметрией весов и межполушарными задержками.
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import random


# ============================================================
# БИОЛОГИЧЕСКИЕ КОНСТАНТЫ
# ============================================================

LEFT_HEMISPHERE = "left"
RIGHT_HEMISPHERE = "right"

LEFT_MODULES = {
    'planner': 'планирование и стратегия',
    'theory_of_mind': 'социальное познание',
    'communication': 'язык и общение',
    'metacognition': 'саморефлексия'
}

RIGHT_MODULES = {
    'world_model': 'модель мира',
    'brain': 'сенсорная обработка',
    'emotion': 'эмоции',
    'habits': 'привычки'
}

CORPUS_CALLOSUM_LR = {
    'planner': {'world_model': 0.7, 'brain': 0.6, 'emotion': 0.5},
    'theory_of_mind': {'emotion': 0.7, 'world_model': 0.6},
    'communication': {'brain': 0.6, 'emotion': 0.5},
    'metacognition': {'brain': 0.5, 'emotion': 0.4},
}

CORPUS_CALLOSUM_RL = {
    'world_model': {'planner': 0.3, 'theory_of_mind': 0.3, 'metacognition': 0.2},
    'brain': {'planner': 0.3, 'communication': 0.3, 'metacognition': 0.2},
    'emotion': {'theory_of_mind': 0.3, 'metacognition': 0.2},
    'habits': {'planner': 0.2, 'communication': 0.2},
}

RIGHT_DELAY = 0.7
LEFT_DELAY = 1.0


class Signal:
    PRIORITY_CRITICAL = 3
    PRIORITY_HIGH = 2
    PRIORITY_MEDIUM = 1
    PRIORITY_LOW = 0

    def __init__(self, source: str, target: str, data: Any, strength: float = 0.5, priority: int = 0,
                 hemisphere: str = None, delay: float = 0.0):
        self.source = source
        self.target = target
        self.data = data
        self.strength = strength
        self.priority = priority
        self.hemisphere = hemisphere
        self.delay = delay
        self.timestamp = 0
        self.id = hashlib.md5(f"{source}{target}{str(data)}".encode()).hexdigest()[:8]

    def __repr__(self):
        return f"Signal({self.source}->{self.target}, s={self.strength:.2f}, p={self.priority}, h={self.hemisphere})"

    def __lt__(self, other):
        return self.priority < other.priority


class Stream:
    def __init__(self, source: str, target: str, weight: float = 0.5, hemisphere: str = None, delay: float = 0.0):
        self.source = source
        self.target = target
        self.weight = weight
        self.hemisphere = hemisphere
        self.delay = delay
        self.history = deque(maxlen=100)
        self.throughput = 0.0
        self.latency = 0.0
        self.stdp_trace = deque(maxlen=50)

    def push(self, signal: Signal):
        self.history.append(signal)
        self.throughput = 0.9 * self.throughput + 0.1 * 1.0
        self.weight = min(1.0, self.weight + 0.005)
        self.stdp_trace.append(signal.strength)



class StreamNetwork:
    """Сеть потоков между модулями."""
    
    def __init__(self):
        self.streams: Dict[str, Stream] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.module_activity: Dict[str, float] = defaultdict(float)
    
    def add_stream(self, source: str, target: str, weight: float = 0.5):
        key = f"{source}->{target}"
        if key not in self.streams:
            self.streams[key] = Stream(source, target, weight)
            self.adjacency[source].append(target)
    
    def get_streams_from(self, source: str) -> List[Stream]:
        return [self.streams[f"{source}->{t}"] for t in self.adjacency[source] 
                if f"{source}->{t}" in self.streams]
    
    def get_streams_to(self, target: str) -> List[Stream]:
        result = []
        for key, stream in self.streams.items():
            if stream.target == target:
                result.append(stream)
        return result
    
    def propagate(self, signal: Signal) -> List[Signal]:
        """Распространяет сигнал по сети с учётом приоритета."""
        if signal.source not in self.adjacency:
            return []
        
        propagated = []
        # Сортируем потоки по весу (более сильные потоки обрабатываются первыми)
        streams = sorted(self.get_streams_from(signal.source), key=lambda s: s.weight, reverse=True)
        
        for stream in streams:
            # Сила потока влияет на ослабление сигнала
            # Приоритет усиливает сигнал
            priority_boost = 1.0 + 0.2 * signal.priority
            new_strength = signal.strength * stream.weight * 0.9 * priority_boost
            
            # Критические сигналы не ослабляются
            if signal.priority >= Signal.PRIORITY_CRITICAL:
                new_strength = min(1.0, new_strength * 1.5)
            
            if new_strength > 0.01:
                new_signal = Signal(
                    source=signal.source,
                    target=stream.target,
                    data=signal.data,
                    strength=new_strength,
                    priority=signal.priority
                )
                stream.push(new_signal)
                propagated.append(new_signal)
        
        return propagated
    
    def update_activity(self, module: str, activity: float):
        self.module_activity[module] = 0.9 * self.module_activity[module] + 0.1 * activity


class ImportanceEvaluator:
    """
    Оценщик важности сигналов.
    Определяет, насколько сигнал важен для агента в текущем контексте.
    """
    
    def __init__(self):
        self.importance_history = defaultdict(lambda: deque(maxlen=10))
        self.base_importance = {
            'danger': 0.9,
            'hunger': 0.8,
            'pain': 0.8,
            'thirst': 0.7,
            'fatigue': 0.6,
            'curiosity': 0.5,
            'explore': 0.3,
            'rest': 0.4,
            'collect': 0.6,
            'flee': 0.9,
        }
    
    def evaluate(self, signal: Signal, context: Dict[str, Any] = None) -> float:
        """
        Оценивает важность сигнала (0.0–1.0).
        
        Args:
            signal: Сигнал для оценки
            context: Текущий контекст (энергия, голод, опасность и т.д.)
        
        Returns:
            Важность сигнала (0.0–1.0)
        """
        context = context or {}
        importance = 0.0
        
        # 1. Базовая важность от источника
        if signal.source in self.base_importance:
            importance = self.base_importance[signal.source]
        elif 'perception' in signal.source:
            importance = 0.3
        elif 'brain' in signal.source:
            importance = 0.5
        else:
            importance = 0.2
        
        # 2. Модификация на основе данных сигнала
        if isinstance(signal.data, dict):
            if 'danger' in signal.data or 'fear' in signal.data:
                importance = max(importance, 0.9)
            if 'hunger' in signal.data and signal.data.get('hunger', 0) > 0.7:
                importance = max(importance, 0.85)
            if 'energy' in signal.data and signal.data.get('energy', 1.0) < 0.3:
                importance = max(importance, 0.8)
        
        # 3. Модификация на основе контекста
        if context:
            # Голод важнее, когда энергия низкая
            if context.get('energy_pct', 1.0) < 0.3 and 'hunger' in str(signal.data):
                importance = min(1.0, importance + 0.2)
            # Опасность важнее, когда страх высок
            if context.get('fear', 0) > 0.7 and 'danger' in str(signal.data):
                importance = min(1.0, importance + 0.15)
            # Исследование важнее, когда энергии много
            if context.get('energy_pct', 1.0) > 0.7 and 'explore' in str(signal.data):
                importance = min(1.0, importance + 0.1)
        
        # 4. Учёт истории (повторяющиеся сигналы становятся менее важными)
        signal_key = f"{signal.source}:{str(signal.data)[:20]}"
        history = self.importance_history[signal_key]
        if history:
            # Если сигнал часто повторяется, снижаем важность
            repetition_factor = min(1.0, len(history) / 5.0)
            importance = importance * (1.0 - 0.1 * repetition_factor)
        
        # 5. Ограничение
        importance = max(0.0, min(1.0, importance))
        
        # Запоминаем важность
        self.importance_history[signal_key].append(importance)
        
        return importance
    
    def get_importance_report(self) -> Dict[str, Any]:
        """Возвращает отчёт об оценках важности."""
        return {
            'history_size': sum(len(h) for h in self.importance_history.values()),
            'base_importance': self.base_importance,
            'top_signals': sorted(
                [(k, list(v)) for k, v in self.importance_history.items()],
                key=lambda x: x[1][-1] if x[1] else 0,
                reverse=True
            )[:5]
        }


class ConsciousnessDispatcher:
    """
    Глобальное рабочее пространство (Global Workspace Theory).
    Реализует биологические механизмы сознания:
    - Конкуренция за внимание
    - Широковещательная рассылка
    - Ворота внимания (энергия + эмоции)
    - Нейромедиаторная модуляция
    - Интеграция сигналов в единую картину
    """
    
    def __init__(self, memory_size: int = 1000, min_modules: int = 3):
        self.stream_network = StreamNetwork()
        self.memory = deque(maxlen=memory_size)
        self.short_term_memory = deque(maxlen=50)
        self.routes = {}  # маршруты перенаправления
        self.active = False
        self.min_modules = min_modules
        self.module_count = 0
        self.consciousness_level = 0.0
        
        # Оценщик важности
        self.importance_evaluator = ImportanceEvaluator()
        
        # Статистика
        self.total_signals_processed = 0
        self.signals_routed = 0
        self.memory_hits = 0
        self.bottlenecks = []
        
        # Текущий контекст
        self.current_context = {}
        
        # --- НОВОЕ: Онлайн-обучение сознания ---
        self.recent_actions = []  # (action, reward, success)
        self.learning_rate = 0.05  # скорость адаптации весов
        self.last_action = None
        self.last_reward = 0.0
        
        # --- БИОЛОГИЧЕСКИЕ МЕХАНИЗМЫ GWT ---
        # Глобальное рабочее пространство (содержимое сознания)
        self.global_workspace = deque(maxlen=10)  # сигналы в сознании
        self.workspace_content = []  # текущее содержимое сознания
        self.workspace_attention = 0.0  # уровень внимания (0-1)
        
        # Ворота внимания
        self.attention_gate = 0.3  # порог входа в сознание
        self.attention_energy_factor = 1.0  # множитель от энергии
        self.attention_emotion_factor = 1.0  # множитель от эмоций
        
        # Конкуренция за внимание
        self.competition_pool = deque(maxlen=50)  # ожидающие сигналы
        self.competition_threshold = 0.5  # минимальная сила для входа
        self.broadcast_active = False  # идёт ли широковещательная рассылка
        
        # Нейромедиаторы (единый пул)
        self.neurotransmitters = {
            'dopamine': 0.3,       # награда, мотивация
            'noradrenaline': 0.3,  # внимание, бодрствование
            'serotonin': 0.5,      # настроение, социальное
            'acetylcholine': 0.4,  # обучение, внимание
            'glutamate': 0.5,      # возбуждение
            'gaba': 0.5,           # торможение
        }
        
        # Эмоциональный фон
        self.emotional_background = {
            'fear': 0.0,
            'curiosity': 0.5,
            'surprise': 0.2,
            'pleasure': 0.3,
            'anger': 0.0,
            'sadness': 0.1,
        }
        
        # Энергетический баланс
        self.energy_level = 0.7  # 0-1
        self.energy_consumption = 0.01
        
        # История сознания (для непрерывности)
        self.consciousness_history = deque(maxlen=30)
        
        # Метакогниция о работе GWT
        self.gwt_awareness = {
            'content_count': 0,
            'dominant_signal': None,
            'attention_shift_count': 0,
            'broadcast_count': 0,
        }
    
    def register_module(self, module_name: str, capabilities: List[str] = None):
        """Регистрирует модуль в системе."""
        if module_name not in self.stream_network.adjacency:
            self.stream_network.adjacency[module_name] = []
            self.module_count += 1
            self.routes[module_name] = []
        
        # Проверяем, достаточно ли развиты модули для сознания
        if self.module_count >= self.min_modules:
            self.active = True
    
    def add_route(self, source: str, target: str, weight: float = 0.5):
        """Добавляет маршрут для перенаправления потоков."""
        self.stream_network.add_stream(source, target, weight)
        self.routes.setdefault(source, []).append(target)
        
        # Обновляем уровень сознания
        self._update_consciousness_level()
    
    def route_signal(self, source: str, data: Any, strength: float = 0.5, priority: int = 0, context: Dict[str, Any] = None) -> List[Signal]:
        """
        Получает сигнал от модуля и перенаправляет его.
        Частично запоминает.
        """
        self.total_signals_processed += 1
        
        # Создаём сигнал
        signal = Signal(source, "dispatcher", data, strength, priority)
        
        # Оцениваем важность сигнала
        importance = self.importance_evaluator.evaluate(signal, context or self.current_context)
        
        # Важность влияет на силу сигнала
        signal.strength = min(1.0, signal.strength * (0.7 + 0.3 * importance))
        
        # Частично запоминаем
        self._remember(signal)
        
        # Если сознание активно, перенаправляем
        if self.active:
            return self._dispatch(signal)
        else:
            return []
    
    def _dispatch(self, signal: Signal) -> List[Signal]:
        """Перенаправляет сигнал по маршрутам с учётом приоритета."""
        propagated = []
        
        # Определяем маршруты для этого сигнала
        if signal.source in self.routes:
            # Сортируем цели по важности (критические модули первыми)
            targets = sorted(
                self.routes[signal.source],
                key=lambda t: 1 if t in ['brain', 'self_model', 'emotion'] else 0,
                reverse=True
            )
            
            for target in targets:
                if target in self.stream_network.adjacency:
                    # Для критических сигналов усиление
                    strength_multiplier = 1.0
                    if signal.priority >= Signal.PRIORITY_CRITICAL:
                        strength_multiplier = 1.3
                    elif signal.priority >= Signal.PRIORITY_HIGH:
                        strength_multiplier = 1.1
                    
                    new_strength = min(1.0, signal.strength * 0.8 * strength_multiplier)
                    
                    new_signal = Signal(
                        source=signal.source,
                        target=target,
                        data=signal.data,
                        strength=new_strength,
                        priority=signal.priority
                    )
                    # Отправляем по сети
                    network_signals = self.stream_network.propagate(new_signal)
                    propagated.extend(network_signals)
                    self.signals_routed += 1
        
        return propagated
    
    def _remember(self, signal: Signal):
        """Частично запоминает сигнал."""
        # Проверяем, было ли что-то подобное
        memory_key = f"{signal.source}:{str(signal.data)[:50]}"
        for mem in self.memory:
            if hasattr(mem, 'data') and str(mem.data)[:50] == str(signal.data)[:50]:
                self.memory_hits += 1
                # Усиливаем воспоминание
                mem.strength = min(1.0, mem.strength + 0.05)
                break
        else:
            self.memory.append(signal)
        
        # Добавляем в краткосрочную память
        self.short_term_memory.append(signal)
    
    def _update_consciousness_level(self):
        """Обновляет уровень сознания на основе развитости системы."""
        # Сознание зависит от:
        # 1. Количества модулей (развитость)
        # 2. Количества маршрутов (связанность)
        # 3. Объёма памяти (непрерывность)
        # 4. Активности в сети
        # 5. Наличия иерархии (приоритетов)
        
        module_factor = min(1.0, self.module_count / 10.0)
        route_factor = min(1.0, sum(len(r) for r in self.routes.values()) / 20.0)
        memory_factor = min(1.0, len(self.memory) / 200.0)
        activity_factor = sum(self.stream_network.module_activity.values()) / max(1, self.module_count)
        
        # Иерархия: проверяем наличие маршрутов к высшим модулям
        has_hierarchy = any(t in ['brain', 'self_model', 'emotion'] for targets in self.routes.values() for t in targets)
        hierarchy_factor = 1.0 if has_hierarchy else 0.0
        
        self.consciousness_level = (
            0.35 * module_factor +
            0.25 * route_factor +
            0.15 * memory_factor +
            0.1 * activity_factor +
            0.15 * hierarchy_factor
        )
        
        # Сознание появляется только при достаточной развитости и иерархии
        if module_factor < 0.3 or not has_hierarchy:
            self.consciousness_level = 0.0
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Возвращает отчёт о состоянии сознания."""
        return {
            'active': self.active,
            'consciousness_level': self.consciousness_level,
            'module_count': self.module_count,
            'route_count': sum(len(r) for r in self.routes.values()),
            'memory_size': len(self.memory),
            'short_term_size': len(self.short_term_memory),
            'signals_processed': self.total_signals_processed,
            'signals_routed': self.signals_routed,
            'memory_hits': self.memory_hits,
            'module_activity': dict(self.stream_network.module_activity),
        }
    
    def step(self, signals: List[Tuple[str, Any, float]] = None):
        """Один шаг диспетчера."""
        if signals:
            for source, data, strength in signals:
                self.route_signal(source, data, strength)
        
        # Обновляем уровень сознания
        self._update_consciousness_level()
        
        return self.get_consciousness_report()
    
    def get_recent_memories(self, n: int = 5) -> List[Signal]:
        """Возвращает последние n воспоминаний."""
        return list(self.memory)[-n:]
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Возвращает статус сети потоков."""
        return {
            'total_streams': len(self.stream_network.streams),
            'streams': {
                k: {
                    'weight': v.weight,
                    'throughput': v.throughput,
                    'history_len': len(v.history)
                }
                for k, v in self.stream_network.streams.items()
            },
            'adjacency': dict(self.stream_network.adjacency)
        }
    
    def process_consciousness(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Основной цикл глобального рабочего пространства.
        - Конкуренция за внимание между сигналами
        - Выбор победителя (самый сильный/важный)
        - Широковещательная рассылка всем модулям
        - Обновление состояния сознания
        """
        context = context or self.current_context
        
        if not self.active or not self.competition_pool:
            return {'broadcasted': False, 'winner': None}
        
        # 1. Конкуренция за внимание
        candidates = list(self.competition_pool)
        self.competition_pool.clear()
        
        scored = []
        for signal in candidates:
            # Базовая сила
            strength = signal.strength
            
            # Эмоциональная модуляция
            emotion_boost = 1.0
            if isinstance(signal.data, dict):
                if 'fear' in signal.data and signal.data['fear'] > 0.5:
                    emotion_boost = 1.5
                if 'curiosity' in signal.data and signal.data['curiosity'] > 0.7:
                    emotion_boost = 1.3
                if 'danger' in signal.data:
                    emotion_boost = 1.8
                if 'reward' in signal.data and signal.data['reward'] > 0.5:
                    emotion_boost *= 1.2
            
            # Нейромедиаторная модуляция
            noradrenaline = self.neurotransmitters.get('noradrenaline', 0.3)
            dopamine = self.neurotransmitters.get('dopamine', 0.3)
            attention_mod = 0.5 + 0.5 * (noradrenaline * 0.7 + dopamine * 0.3)
            
            # Энергетический фактор
            energy_mod = 0.5 + 0.5 * self.energy_level
            
            # Итоговая сила
            final_strength = strength * emotion_boost * attention_mod * energy_mod
            scored.append((signal, final_strength))
        
        if not scored:
            return {'broadcasted': False, 'winner': None}
        
        # 2. Выбор победителя
        scored.sort(key=lambda x: x[1], reverse=True)
        winner, winner_strength = scored[0]
        
        # 3. Ворота внимания
        threshold = self.attention_gate * (0.7 + 0.3 * self.energy_level)
        threshold *= (0.8 + 0.4 * self.neurotransmitters.get('noradrenaline', 0.3))
        
        if winner_strength < threshold:
            return {'broadcasted': False, 'winner': None, 'reason': 'below_threshold'}
        
        # 4. Широковещательная рассылка
        self.broadcast_active = True
        self.workspace_content.append(winner)
        if len(self.workspace_content) > 5:
            self.workspace_content.pop(0)
        
        self.global_workspace.append(winner)
        self.consciousness_history.append({
            'signal': winner,
            'strength': winner_strength,
            'emotions': self.emotional_background.copy()
        })
        
        # 5. Рассылка всем модулям
        broadcast_signals = []
        for module in self.routes.keys():
            if module != winner.source:
                broadcast_signal = Signal(
                    source='consciousness',
                    target=module,
                    data=winner.data,
                    strength=winner_strength * 0.7,
                    priority=winner.priority,
                    hemisphere=winner.hemisphere
                )
                broadcast_signals.append(broadcast_signal)
                self._remember(broadcast_signal)
        
        # 6. Обновление метрик
        self.gwt_awareness['content_count'] = len(self.global_workspace)
        self.gwt_awareness['dominant_signal'] = winner.source
        self.gwt_awareness['broadcast_count'] += 1
        self.broadcast_active = False
        
        return {
            'broadcasted': True,
            'winner': winner,
            'winner_strength': winner_strength,
            'broadcast_count': len(broadcast_signals),
            'threshold': threshold,
            'attention_level': self.workspace_attention,
            'energy': self.energy_level,
        }
    
    def update_neurotransmitters(self, emotions: Dict[str, float]):
        """
        Обновляет нейромедиаторы на основе эмоционального состояния.
        Связывает эмоции с биохимией мозга.
        """
        self.emotional_background.update(emotions)
        
        # Страх → норадреналин ↑
        self.neurotransmitters['noradrenaline'] = 0.3 + 0.7 * self.emotional_background.get('fear', 0.0)
        
        # Любопытство → дофамин ↑
        self.neurotransmitters['dopamine'] = 0.3 + 0.5 * self.emotional_background.get('curiosity', 0.0)
        
        # Удовольствие → серотонин ↑
        self.neurotransmitters['serotonin'] = 0.5 + 0.5 * self.emotional_background.get('pleasure', 0.0)
        
        # Удивление → ацетилхолин ↑ (обучение)
        self.neurotransmitters['acetylcholine'] = 0.4 + 0.6 * self.emotional_background.get('surprise', 0.0)
        
        # Гнев → глутамат ↑ (возбуждение)
        self.neurotransmitters['glutamate'] = 0.5 + 0.5 * self.emotional_background.get('anger', 0.0)
        
        # Печаль → ГАМК ↑ (торможение)
        self.neurotransmitters['gaba'] = 0.5 + 0.5 * self.emotional_background.get('sadness', 0.0)
        
        # Ограничение
        for key in self.neurotransmitters:
            self.neurotransmitters[key] = max(0.0, min(1.0, self.neurotransmitters[key]))

    def sleep(self, steps: int = 10):
        """
        Механизм консолидации памяти (сон).
        Проигрывает последние steps воспоминаний и укрепляет важные маршруты.
        """
        # Берём последние воспоминания
        memories = list(self.memory)[-steps:]
        if not memories:
            return
        
        # Считаем частоту сигналов
        signal_counts = defaultdict(int)
        for mem in memories:
            key = f"{mem.source}->{mem.target}"
            signal_counts[key] += 1
        
        # Укрепляем маршруты, которые часто встречаются
        for key, count in signal_counts.items():
            if count >= 2:  # частота больше 1
                if key in self.stream_network.streams:
                    # Укрепляем на 10% за каждое повторение
                    boost = min(0.3, count * 0.05)
                    new_weight = min(1.0, self.stream_network.streams[key].weight + boost)
                    self.stream_network.streams[key].weight = new_weight
        
        # Очищаем краткосрочную память (имитация очистки во сне)
        self.short_term_memory.clear()
        
        # Обновляем уровень сознания
        self._update_consciousness_level()
    
    def learn_from_step(self, action: str, reward: float, success: bool, context: Dict[str, Any] = None):
        """
        Онлайн-обучение сознания на основе каждого шага с эмоциональной модуляцией.
        Обновляет веса маршрутов, которые привели к действию.
        
        Эмоции влияют на обучение:
        - Дофамин (dopamine): ускоряет обучение (learning_rate * 1.5)
        - Норадреналин (noradrenaline): сужает фокус (только критические маршруты)
        - Серотонин (serotonin): увеличивает горизонт планирования (добавляет долгосрочные маршруты)
        """
        context = context or {}
        
        # Извлекаем эмоциональные параметры
        dopamine = context.get('dopamine', 0.0)
        noradrenaline = context.get('noradrenaline', 0.0)
        serotonin = context.get('serotonin', 0.0)
        
        # Сохраняем результат
        self.recent_actions.append((action, reward, success))
        if len(self.recent_actions) > 100:
            self.recent_actions.pop(0)
        
        # Если награда значимая, усиливаем маршруты, связанные с этим действием
        if abs(reward) > 0.1:
            # Дофамин ускоряет обучение
            effective_lr = self.learning_rate * (1.0 + 0.5 * dopamine)
            
            for key, stream in self.stream_network.streams.items():
                # Норадреналин сужает фокус: только критические маршруты
                if noradrenaline > 0.5:
                    if not any(c in key for c in ['brain', 'self_model', 'emotion']):
                        continue
                
                # Ищем маршруты, которые могли привести к этому действию
                if 'brain' in key or 'planning' in key or 'reasoning' in key:
                    # Усиливаем, если успех, ослабляем, если неудача
                    delta = effective_lr * reward * (1.0 if success else -0.5)
                    new_weight = max(0.0, min(1.0, stream.weight + delta))
                    stream.weight = new_weight
        
        # Серотонин: добавляем долгосрочные маршруты (если их нет)
        if serotonin > 0.5:
            # Проверяем наличие маршрутов к 'memory' и 'prediction'
            if 'memory' not in self.routes.get('brain', []):
                self.add_route('brain', 'memory', weight=0.3 + 0.2 * serotonin)
            if 'prediction' not in self.routes.get('memory', []):
                self.add_route('memory', 'prediction', weight=0.3 + 0.2 * serotonin)
        
        # Сохраняем последнее действие
        self.last_action = action
        self.last_reward = reward


# ============================================================
# ИНТЕГРАЦИЯ С СУЩЕСТВУЮЩЕЙ АРХИТЕКТУРОЙ
# ============================================================

class EvolvableDispatcher(ConsciousnessDispatcher):
    """
    Эволюционирующий диспетчер сознания.
    Веса маршрутов мутируют и оцениваются по fitness.
    """
    
    def __init__(self):
        super().__init__()
        self.fitness_score = 0.0
        self.generation = 0
        self.mutation_rate = 0.1
        self.best_weights = {}
        self.best_fitness = -float('inf')
        
        # Регистрируем модули
        self._register_core_modules()
        
        # Сохраняем начальные веса
        self._save_weights()
    
    def _register_core_modules(self):
        """Регистрирует ядро модулей для возникновения сознания."""
        core_modules = [
            'brain', 'perception', 'memory', 'emotion', 
            'attention', 'prediction', 'self_model', 'habits',
            'planning', 'reasoning'
        ]
        
        for module in core_modules:
            self.register_module(module)
        
        # Создаём маршруты между модулями
        routes = [
            ('perception', 'attention'),
            ('perception', 'brain'),
            ('attention', 'brain'),
            ('brain', 'memory'),
            ('brain', 'emotion'),
            ('emotion', 'self_model'),
            ('memory', 'prediction'),
            ('prediction', 'planning'),
            ('planning', 'brain'),
            ('self_model', 'brain'),
            ('habits', 'brain'),
            ('brain', 'reasoning'),
            ('reasoning', 'self_model'),
        ]
        
        for source, target in routes:
            self.add_route(source, target, weight=0.5)
    
    def _save_weights(self):
        """Сохраняет текущие веса маршрутов."""
        self.best_weights = {}
        for key, stream in self.stream_network.streams.items():
            self.best_weights[key] = stream.weight
    
    def mutate(self):
        """Мутирует веса маршрутов."""
        self.generation += 1
        
        for key, stream in self.stream_network.streams.items():
            if np.random.random() < self.mutation_rate:
                # Мутация: изменение веса в пределах ±0.2
                delta = np.random.normal(0, 0.1)
                new_weight = max(0.0, min(1.0, stream.weight + delta))
                stream.weight = new_weight
    
    def evaluate_fitness(self, food_collected: float, energy: float, survival: float) -> float:
        """
        Оценивает fitness диспетчера на основе результатов.
        
        Args:
            food_collected: сколько еды собрано
            energy: уровень энергии
            survival: выжил ли агент (0 или 1)
        """
        # fitness = еда * 2 + энергия * 0.1 + выживание * 10
        self.fitness_score = (food_collected * 2.0) + (energy * 0.1) + (survival * 10.0)
        
        # Если fitness лучше сохранённого — сохраняем веса
        if self.fitness_score > self.best_fitness:
            self.best_fitness = self.fitness_score
            self._save_weights()
        
        return self.fitness_score
    
    def restore_best(self):
        """Восстанавливает лучшие веса."""
        for key, weight in self.best_weights.items():
            if key in self.stream_network.streams:
                self.stream_network.streams[key].weight = weight
    
    def get_weights(self) -> Dict[str, float]:
        """Возвращает текущие веса маршрутов."""
        return {key: stream.weight for key, stream in self.stream_network.streams.items()}


class ConsciousAgent:
    """
    Агент с сознанием-диспетчером.
    Интегрирует все модули через диспетчер сознания.
    """
    
    def __init__(self, use_evolution: bool = False):
        if use_evolution:
            self.dispatcher = EvolvableDispatcher()
        else:
            self.dispatcher = ConsciousnessDispatcher()
            # Регистрируем модули для обычного диспетчера
            self._register_core_modules()
        
        self.modules = {}
        self.module_connections = {}
        self.use_evolution = use_evolution
    
    def _register_core_modules(self):
        """Регистрирует ядро модулей для возникновения сознания."""
        core_modules = [
            'brain', 'perception', 'memory', 'emotion', 
            'attention', 'prediction', 'self_model', 'habits',
            'planning', 'reasoning'
        ]
        
        for module in core_modules:
            self.dispatcher.register_module(module)
        
        # Создаём маршруты между модулями
        routes = [
            ('perception', 'attention'),
            ('perception', 'brain'),
            ('attention', 'brain'),
            ('brain', 'memory'),
            ('brain', 'emotion'),
            ('emotion', 'self_model'),
            ('memory', 'prediction'),
            ('prediction', 'planning'),
            ('planning', 'brain'),
            ('self_model', 'brain'),
            ('habits', 'brain'),
            ('brain', 'reasoning'),
            ('reasoning', 'self_model'),
        ]
        
        for source, target in routes:
            self.dispatcher.add_route(source, target, weight=0.5)
    
    def decide(self, grid, position, other_agents=None):
        """Принимает решение на основе текущего состояния."""
        # Преобразуем входные данные в сигналы
        input_data = {
            'grid': grid,
            'position': position,
            'other_agents': other_agents or []
        }
        
        # Обрабатываем через диспетчер
        report = self.process(input_data)
        
        # Извлекаем действие из отчёта
        action = report.get('action', 'explore')
        thought = report.get('thought', 'Действую по интуиции')
        
        return action, thought
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Обрабатывает входящие данные через сеть сознания."""
        # Отправляем сигналы от модулей
        signals = [
            ('perception', input_data, 0.6),
            ('brain', input_data, 0.5),
        ]
        
        # Диспетчер обрабатывает сигналы
        report = self.dispatcher.step(signals)
        
        return report
    
    def get_consciousness_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние сознания."""
        return {
            'level': self.dispatcher.consciousness_level,
            'active': self.dispatcher.active,
            'memory': len(self.dispatcher.memory),
            'routes': sum(len(r) for r in self.dispatcher.routes.values()),
        }


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

if __name__ == "__main__":
    # Создаём агента с сознанием-диспетчером
    agent = ConsciousAgent()
    
    print("🧠 СОЗНАНИЕ-ДИСПЕТЧЕР")
    print("=" * 50)
    
    # Проверяем начальное состояние
    state = agent.get_consciousness_state()
    print(f"\n📊 Начальное состояние:")
    print(f"   Уровень сознания: {state['level']:.3f}")
    print(f"   Активно: {state['active']}")
    print(f"   Память: {state['memory']} сигналов")
    print(f"   Маршрутов: {state['routes']}")
    
    # Обрабатываем несколько сигналов
    print("\n🔄 Обработка сигналов...")
    for i in range(20):
        input_data = f"сигнал_{i}"
        report = agent.process(input_data)
        
        if i % 5 == 0:
            print(f"   Шаг {i}: уровень сознания = {report['consciousness_level']:.3f}")
    
    # Финальное состояние
    final_state = agent.get_consciousness_state()
    print(f"\n📊 Финальное состояние:")
    print(f"   Уровень сознания: {final_state['level']:.3f}")
    print(f"   Активно: {final_state['active']}")
    print(f"   Память: {final_state['memory']} сигналов")
    print(f"   Маршрутов: {final_state['routes']}")
    
    # Получаем подробный отчёт
    report = agent.dispatcher.get_consciousness_report()
    print(f"\n📋 Подробный отчёт:")
    for key, value in report.items():
        if key not in ['module_activity']:
            print(f"   {key}: {value}")
    
    # Потоки
    streams = agent.dispatcher.get_stream_status()
    print(f"\n🌊 Сеть потоков:")
    print(f"   Всего потоков: {streams['total_streams']}")
    print(f"   Активные маршруты: {len(streams['streams'])}")
    
    print("\n✅ Сознание-диспетчер работает.")
    print("\n🧭 Да пребудет с вами лес.")
