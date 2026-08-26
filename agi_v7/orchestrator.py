# -*- coding: utf-8 -*-
"""
ОРКЕСТРАТОР — главный управляющий модуль AGI

Интегрирует:
1. Нейронную сеть (brain_module.NeuronNetwork)
2. Компрессор памяти (brain_module.MemoryCompressor)
3. Энергетический менеджмент
4. Три системы (страх, любопытство, удивление)
5. Сон и консолидацию
6. Внимание и предсказание
7. Метакогницию
8. Навыки (обучение и воспроизведение)
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# Импортируем мозговой модуль
from agi_v7.brain_module import NeuronNetwork, MemoryCompressor, RealNeuron
# Импортируем кристаллизатор привычек
from agi_v7.habits import HabitCrystallizer
# Импортируем самомодель и иерархический предиктор
from agi_v7.self_model import SelfModel
from agi_v7.hierarchical_predictor import HierarchicalPredictor
# Импортируем активное внимание
from agi_v7.active_attention import ActiveAttention


class CognitiveOrchestrator:
    """Оркестратор, управляющий всеми когнитивными функциями"""
    
    def __init__(
        self,
        num_neurons: int = 200,
        connectivity: float = 0.1,
        input_dim: int = 100
    ):
        # --- НЕЙРОННАЯ СЕТЬ ---
        self.brain = NeuronNetwork(
            num_neurons=num_neurons,
            connectivity=connectivity
        )
        
        # --- КОМПРЕССОР ПАМЯТИ ---
        self.compressor = MemoryCompressor(
            input_dim=input_dim,
            code_dim=20
        )
        
        # --- КРИСТАЛЛИЗАТОР ПРИВЫЧЕК ---
        self.habit_crystallizer = HabitCrystallizer(threshold=4, decay=0.9)
        
        # --- САМОМОДЕЛЬ ---
        self.self_model = SelfModel()
        
        # --- ИЕРАРХИЧЕСКИЙ ПРЕДИКТОР ---
        self.predictor = HierarchicalPredictor(state_dim=7, action_dim=4)
        
        # --- АКТИВНОЕ ВНИМАНИЕ ---
        self.attention = ActiveAttention(num_signals=7, focus_size=3)
        
        # --- СОСТОЯНИЕ ---
        self.step_count = 0
        self.time = 0.0
        
        # --- ИСТОРИЯ ---
        self.history = {
            'spikes': [],
            'fear': [],
            'curiosity': [],
            'surprise': [],
            'energy': [],
            'latency': []
        }
        
        # --- НАВЫКИ ---
        self.skills = {}
        self.skill_sequences = {}
        
        # --- МЕТАДАННЫЕ ---
        self.metadata = {
            'birth_time': time.time(),
            'total_steps': 0,
            'total_spikes': 0,
            'neurogenesis_count': 0,
            'prune_count': 0,
            'sleep_count': 0,
        }
        
        print(f"🧠 Оркестратор инициализирован")
        print(f"   Нейронов: {len(self.brain.neurons)}")
        print(f"   Синапсов: {self.brain._count_synapses()}")
        print(f"   Компрессор: {self.compressor.input_dim} -> {self.compressor.code_dim}")
    
    # ============================================================
    # ОСНОВНОЙ ЦИКЛ
    # ============================================================
    
    def step(self, external_input: Optional[List[float]] = None) -> Dict[str, Any]:
        """Один шаг работы оркестратора"""
        self.step_count += 1
        self.time += 0.1
        
        # --- ПОДГОТОВКА ВХОДА ---
        if external_input is not None:
            # Нормализуем вход
            max_val = max(external_input) if external_input else 1.0
            if max_val > 0:
                external_input = [x / max_val for x in external_input]
            
            # Обрезаем или дополняем до размера сети
            n_neurons = len(self.brain.neurons)
            if len(external_input) > n_neurons:
                external_input = external_input[:n_neurons]
            elif len(external_input) < n_neurons:
                external_input = external_input + [0.0] * (n_neurons - len(external_input))
        else:
            # Если вход не подан, создаём случайный
            external_input = [random.random() for _ in range(len(self.brain.neurons))]
        
        # --- ШАГ СЕТИ ---
        result = self.brain.step(external_input)
        spikes = result['spikes']
        latency = result.get('latency', 0.0)
        
        # --- СОХРАНЯЕМ ПАТТЕРН В ПАМЯТЬ ---
        if spikes and sum(spikes) > 3:
            pattern = spikes[:self.compressor.input_dim]
            while len(pattern) < self.compressor.input_dim:
                pattern.append(0.0)
            label = f"step_{self.step_count}"
            self.compressor.store(pattern, label=label)
        
        # --- ОБНОВЛЕНИЕ ИСТОРИИ ---
        self.history['spikes'].append(sum(spikes))
        self.history['fear'].append(self.brain.fear)
        self.history['curiosity'].append(self.brain.curiosity)
        self.history['surprise'].append(self.brain.surprise)
        self.history['energy'].append(self.brain.network_energy)
        self.history['latency'].append(latency)
        
        # Ограничиваем историю
        for key in self.history:
            if len(self.history[key]) > 1000:
                self.history[key] = self.history[key][-1000:]
        
        # --- МЕТАДАННЫЕ ---
        self.metadata['total_steps'] += 1
        self.metadata['total_spikes'] += sum(spikes)
        
        # --- ВОЗВРАТ ---
        return {
            'spikes': spikes,
            'spike_count': sum(spikes),
            'fear': self.brain.fear,
            'curiosity': self.brain.curiosity,
            'surprise': self.brain.surprise,
            'energy': self.brain.network_energy,
            'latency': latency,
            'step': self.step_count,
            'time': self.time,
        }
    
    # ============================================================
    # НАВЫКИ
    # ============================================================
    
    def learn_skill(self, skill_name: str, sequence: List[float]) -> bool:
        """Обучает навык (последовательность действий)"""
        if not sequence:
            return False
        
        # Сохраняем последовательность
        self.skill_sequences[skill_name] = sequence
        
        # Пропускаем через сеть для закрепления
        n_neurons = len(self.brain.neurons)
        for value in sequence:
            # Создаём входной сигнал с паттерном, масштабированным значением
            input_vector = [value * (0.5 + 0.5 * random.random()) for _ in range(n_neurons)]
            self.brain.step(input_vector)
        
        # Сохраняем в памяти
        self.skills[skill_name] = {
            'sequence': sequence,
            'length': len(sequence),
            'learned_at': self.time,
            'activation_count': 0,
            'strength': 1.0,
        }
        
        print(f"📚 Навык '{skill_name}' выучен (длина: {len(sequence)})")
        return True
    
    def recall_skill(self, skill_name: str) -> Optional[List[float]]:
        """Воспроизводит навык"""
        if skill_name not in self.skills:
            return None
        
        skill = self.skills[skill_name]
        skill['activation_count'] += 1
        
        # Укрепляем навык при воспроизведении
        for value in skill['sequence']:
            input_vector = [value] * len(self.brain.neurons)
            self.brain.step(input_vector)
        
        return skill['sequence']
    
    def get_skill_strength(self, skill_name: str) -> float:
        """Возвращает силу навыка"""
        if skill_name not in self.skills:
            return 0.0
        return self.skills[skill_name]['strength']
    
    # ============================================================
    # ПАМЯТЬ
    # ============================================================
    
    def store_memory(self, pattern: List[float], label: str = "") -> int:
        """Сохраняет паттерн в память"""
        return self.compressor.store(pattern, label=label)
    
    def recall_memory(self, index: int) -> Optional[List[float]]:
        """Восстанавливает паттерн из памяти"""
        return self.compressor.recall(index)
    
    def recall_by_label(self, label: str) -> Optional[List[float]]:
        """Восстанавливает паттерн по метке"""
        return self.compressor.recall_by_label(label)
    
    def find_similar(self, pattern: List[float], top_k: int = 3) -> List[Tuple[int, float]]:
        """Находит похожие паттерны"""
        return self.compressor.find_similar(pattern, top_k=top_k)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Статистика памяти"""
        return self.compressor.get_stats()
    
    # ============================================================
    # ЭНЕРГИЯ
    # ============================================================
    
    def get_energy_state(self) -> Dict[str, Any]:
        """Состояние энергии"""
        return {
            'current': self.brain.network_energy,
            'critical_threshold': self.brain.energy_critical_threshold,
            'low_threshold': self.brain.energy_low_threshold,
            'high_threshold': self.brain.energy_high_threshold,
            'depletion_rate': self.brain.energy_depletion_rate,
            'recovery_rate': self.brain.energy_recovery_rate,
        }
    
    def get_chains_energy(self) -> Dict[str, Any]:
        """Какие цепи можно запустить"""
        return self.brain.get_energy_for_chains()
    
    # ============================================================
    # ПРЕДСКАЗАНИЕ
    # ============================================================
    
    def predict_next(self, current_state: List[float]) -> List[float]:
        """Предсказывает следующее состояние"""
        return self.brain.predict_next(current_state)
    
    # ============================================================
    # МЕТАКОГНИЦИЯ
    # ============================================================
    
    def get_metacognition(self) -> Dict[str, Any]:
        """Метакогнитивное состояние"""
        return self.brain.get_metacognition()
    
    # ============================================================
    # УПРАВЛЕНИЕ СНОМ
    # ============================================================
    
    def sleep(self) -> Dict[str, Any]:
        """Запускает консолидацию во сне"""
        self.brain.sleep_consolidation()
        self.metadata['sleep_count'] += 1
        return {
            'sleep_count': self.metadata['sleep_count'],
            'energy': self.brain.network_energy,
            'patterns': len(self.brain.memory_compressor.patterns),
        }
    
    # ============================================================
    # НЕЙРОГЕНЕЗИС
    # ============================================================
    
    def trigger_neurogenesis(self, count: int = 1) -> int:
        """Принудительно запускает нейрогенезис"""
        self.brain._neurogenesis()
        self.metadata['neurogenesis_count'] += 1
        return len(self.brain.neurons)
    
    # ============================================================
    # СТАТИСТИКА
    # ============================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Полная статистика"""
        return {
            'step': self.step_count,
            'time': self.time,
            'neurons': len(self.brain.neurons),
            'synapses': self.brain._count_synapses(),
            'energy': self.brain.network_energy,
            'fear': self.brain.fear,
            'curiosity': self.brain.curiosity,
            'surprise': self.brain.surprise,
            'skills': len(self.skills),
            'patterns': len(self.compressor.patterns),
            'long_term': len(self.brain.long_term_memory),
            'working_memory': len(self.brain.working_memory),
            'metacognition': self.brain.get_metacognition(),
            'metadata': self.metadata.copy(),
        }
    
    def get_history(self, key: str, limit: int = 100) -> List[float]:
        """Возвращает историю по ключу"""
        if key not in self.history:
            return []
        return self.history[key][-limit:]
    
    # ============================================================
    # ВИЗУАЛИЗАЦИЯ (краткая)
    # ============================================================
    
    def print_status(self):
        """Выводит текущее состояние"""
        stats = self.get_stats()
        print("=" * 50)
        print(f"🧠 СОСТОЯНИЕ ОРКЕСТРАТОРА")
        print(f"   Шаг: {stats['step']}")
        print(f"   Нейроны: {stats['neurons']} | Синапсы: {stats['synapses']}")
        print(f"   Энергия: {stats['energy']:.1f}")
        print(f"   Страх: {stats['fear']:.2f} | Любопытство: {stats['curiosity']:.2f} | Удивление: {stats['surprise']:.2f}")
        print(f"   Навыки: {stats['skills']} | Паттерны: {stats['patterns']}")
        print(f"   Осознанность: {stats['metacognition']['awareness_level']:.2f}")
        print(f"   Уверенность: {stats['metacognition']['self_confidence']:.2f}")
        print("=" * 50)
    
    def print_memory(self):
        """Выводит состояние памяти"""
        stats = self.get_memory_stats()
        print("=" * 50)
        print(f"💾 ПАМЯТЬ")
        print(f"   Паттернов: {stats['total_patterns']}")
        print(f"   Сохранено: {stats['total_saved']}")
        print(f"   Размер кода: {stats['code_dim']}")
        print(f"   Коэффициент сжатия: {stats['compression_ratio']:.2f}")
        print(f"   PCA обучена: {stats['pca_fitted']}")
        print("=" * 50)
    
    def print_metacognition(self):
        """Выводит метакогнитивное состояние"""
        meta = self.get_metacognition()
        print("=" * 50)
        print(f"🧠 МЕТАКОГНИЦИЯ")
        print(f"   Осознанность: {meta['awareness_level']:.2f}")
        print(f"   Уверенность: {meta['self_confidence']:.2f}")
        print(f"   Ошибка: {meta['error_rate']:.2f}")
        print(f"   Точность предсказаний: {meta['prediction_accuracy']:.2f}")
        print(f"   Инсайтов: {meta['insight_count']}")
        print(f"   Глубина рефлексии: {meta['reflection_depth']}")
        print("=" * 50)


# ============================================================
# ТЕСТИРОВАНИЕ
# ============================================================

def test_orchestrator():
    """Тестирует оркестратор"""
    print("🧪 ТЕСТ ОРКЕСТРАТОРА")
    print("=" * 50)
    
    # Создаём оркестратор
    orchestrator = CognitiveOrchestrator(num_neurons=100, connectivity=0.08)
    
    # Несколько шагов
    for i in range(20):
        # Генерируем случайный вход
        input_signal = [random.uniform(0, 1) for _ in range(100)]
        result = orchestrator.step(input_signal)
        
        if i % 5 == 0:
            print(f"Шаг {i}: спайков={result['spike_count']}, энергия={result['energy']:.1f}, страх={result['fear']:.2f}")
    
    # Обучаем навык
    skill_sequence = [0.5, 0.7, 0.3, 0.9, 0.1, 0.6, 0.8]
    orchestrator.learn_skill("test_skill", skill_sequence)
    
    # Воспроизводим навык
    recalled = orchestrator.recall_skill("test_skill")
    if recalled:
        print(f"✅ Навык воспроизведён: {recalled[:5]}...")
    
    # Сохраняем паттерн
    pattern = [random.uniform(0, 1) for _ in range(100)]
    idx = orchestrator.store_memory(pattern, label="test_pattern")
    print(f"💾 Паттерн сохранён: {idx}")
    
    # Восстанавливаем
    recalled_pattern = orchestrator.recall_memory(idx)
    if recalled_pattern:
        print(f"📂 Паттерн восстановлен: {recalled_pattern[:5]}...")
    
    # Предсказание
    predicted = orchestrator.predict_next(pattern)
    print(f"🔮 Предсказание: {predicted[:5]}...")
    
    # Статистика
    stats = orchestrator.get_stats()
    print(f"📊 Статистика: нейронов={stats['neurons']}, синапсов={stats['synapses']}")
    
    # Метакогниция
    orchestrator.print_metacognition()
    
    # Состояние
    orchestrator.print_status()
    
    print("✅ Тест завершён")
    return orchestrator


if __name__ == "__main__":
    test_orchestrator()
