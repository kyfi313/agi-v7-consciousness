# -*- coding: utf-8 -*-
"""
Интерфейс между AGI v7.0 и нейронным движком
AGI v7.0 → нейроны → AGI v7.0
"""

import sys
import os
import random
import numpy as np
from typing import Dict, Any, List, Optional

# Добавляем путь к neuron_engine из соседней папки
neuron_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Новая папка (2)", "agi_v7")
if neuron_path not in sys.path:
    sys.path.insert(0, neuron_path)

from neuron_engine import NeuronNetwork, RealNeuron
from agi_v7.neuro_protocol import NeuroSignal, SignalType, NeuroProtocol


class BrainInterface:
    """
    Подключает нейронный движок к AGI
    """
    
    def __init__(self, num_neurons: int = 200):
        # Создаём нейронную сеть
        self.brain = NeuronNetwork(num_neurons=num_neurons, connectivity=0.12)
        
        # Включаем нейрогенезис
        self.brain.neurogenesis_active = True
        self.brain.neurogenesis_rate = 0.02
        
        # Три системы как состояния сети
        self.fear_level = 0.0
        self.curiosity_level = 0.5
        self.surprise_level = 0.2
        
        # Память навыков
        self.skills = []
        
        print(f"🧠 Мозговой интерфейс инициализирован: {num_neurons} нейронов")
    
    def step(self, external_input: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Один шаг мозга
        """
        # Если нет входных данных, используем текущее состояние как вход
        if external_input is None:
            external_input = [
                self.fear_level,
                self.curiosity_level,
                self.surprise_level,
                random.random() * 0.5,  # шум
            ] * (self.brain.max_neurons // 4)
            # Обрезаем до нужного размера
            external_input = external_input[:self.brain.max_neurons]
        
        # Запускаем сеть
        spikes = self.brain.step(external_input)
        
        # Извлекаем состояния из активности сети
        firing_rates = self.brain.get_firing_rates()
        
        # Обновляем три системы на основе активности нейронов
        if len(firing_rates) > 10:
            # Страх = активность в нейронах с высоким порогом
            self.fear_level = min(1.0, np.mean(firing_rates[:20]) * 1.5)
            
            # Любопытство = активность в нейронах с низким порогом
            self.curiosity_level = min(1.0, np.mean(firing_rates[20:40]) * 1.5 + 0.3)
            
            # Удивление = изменение активности
            if len(self.brain.activity_history) > 1:
                prev_spikes = self.brain.activity_history[-2]
                current_spikes = self.brain.activity_history[-1]
                if prev_spikes and current_spikes:
                    diff = sum(abs(a - b) for a, b in zip(prev_spikes, current_spikes))
                    self.surprise_level = min(1.0, diff / len(current_spikes) * 2.0)
        
        return {
            'spikes': spikes,
            'firing_rates': firing_rates,
            'fear': self.fear_level,
            'curiosity': self.curiosity_level,
            'surprise': self.surprise_level,
            'neurons': len(self.brain.neurons),
            'synapses': sum(len(n.synapses) for n in self.brain.neurons),
            'energy': self.brain.network_energy,
        }
    
    def step_with_signal(self, signal: Optional[NeuroSignal] = None) -> NeuroSignal:
        """
        Один шаг мозга с использованием NeuroSignal.
        
        Args:
            signal: Входной сигнал (опционально). Если не передан, мозг работает автономно.
            
        Returns:
            NeuroSignal: Выходной сигнал мозга.
        """
        # Преобразуем NeuroSignal в список для нейронов
        if signal:
            external_input = self._signal_to_input(signal)
            result = self.step(external_input)
        else:
            result = self.step()
        
        # Преобразуем результат в NeuroSignal
        return self._result_to_signal(result)
    
    def _signal_to_input(self, signal: NeuroSignal) -> List[float]:
        """Преобразует NeuroSignal в список float для нейронов."""
        data = NeuroProtocol.extract_data(signal)
        
        if 'spikes' in data:
            return data['spikes']
        elif 'rates' in data:
            return data['rates']
        elif 'values' in data:
            return data['values']
        else:
            # Извлекаем числа из словаря
            values = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    values.extend(value)
            return values[:self.brain.max_neurons] if values else [signal.strength] * min(10, self.brain.max_neurons)
    
    def _result_to_signal(self, result: Dict[str, Any]) -> NeuroSignal:
        """Преобразует результат мозга в NeuroSignal."""
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
            return NeuroSignal(
                type=SignalType.STATUS,
                payload=result,
                source='brain',
                strength=0.5,
            )
    
    def learn_skill(self, pattern: List[float], label: str = "") -> Dict[str, Any]:
        """
        Обучает мозг новому навыку
        """
        # Hebbian обучение через сеть
        self.brain.add_skill(pattern)
        
        # Сохраняем навык в память
        skill = {
            'id': len(self.skills),
            'label': label or f"skill_{len(self.skills)}",
            'pattern': pattern[:10],  # храним только часть для демонстрации
            'time': self.brain.time,
        }
        self.skills.append(skill)
        
        # Укрепляем связи в мозге
        for _ in range(3):
            self.brain.step(pattern)
        
        print(f"📚 Выучен навык: {skill['label']}")
        return skill
    
    def recall_skill(self, skill_id: int) -> List[float]:
        """
        Воспроизводит навык по ID
        """
        if skill_id >= len(self.skills):
            print(f"⚠️ Навык {skill_id} не найден")
            return []
        
        skill = self.skills[skill_id]
        pattern = skill['pattern']
        
        # Воспроизводим через сеть
        spikes = self.brain.recall_skill(pattern)
        
        print(f"🔁 Воспроизведён навык: {skill['label']}")
        return spikes
    
    def recall_skill_as_signal(self, skill_id: int) -> Optional[NeuroSignal]:
        """
        Воспроизводит навык по ID и возвращает NeuroSignal.
        """
        spikes = self.recall_skill(skill_id)
        if spikes:
            return NeuroProtocol.from_spikes(spikes, source='brain')
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает состояние мозга
        """
        brain_state = self.brain.get_state()
        return {
            'neurons': brain_state['neurons_count'],
            'synapses': brain_state['total_synapses'],
            'energy': brain_state['network_energy'],
            'fear': self.fear_level,
            'curiosity': self.curiosity_level,
            'surprise': self.surprise_level,
            'skills': len(self.skills),
            'activity': brain_state['activity_spikes'],
        }
    
    def get_state_as_signal(self) -> NeuroSignal:
        """Возвращает состояние мозга в виде NeuroSignal."""
        state = self.get_state()
        return NeuroSignal(
            type=SignalType.STATUS,
            payload=state,
            source='brain',
            strength=0.5,
        )
    
    def trigger_neurogenesis(self):
        """Принудительный нейрогенез"""
        self.brain.neurogenesis_rate = 0.1
        for _ in range(5):
            self.brain._neurogenesis()
        self.brain.neurogenesis_rate = 0.02
        print("🧬 Нейрогенезис активирован!")


def test_brain_interface():
    """Тестирование мозгового интерфейса"""
    print("🧠 ТЕСТ МОЗГОВОГО ИНТЕРФЕЙСА")
    print("=" * 60)
    
    brain = BrainInterface(num_neurons=100)
    
    # Шаг 1: просто работа мозга
    print("\n1. Запуск мозга...")
    state = brain.step()
    print(f"   Нейронов: {state['neurons']}")
    print(f"   Синапсов: {state['synapses']}")
    print(f"   Страх: {state['fear']:.2f}")
    print(f"   Любопытство: {state['curiosity']:.2f}")
    print(f"   Удивление: {state['surprise']:.2f}")
    
    # Шаг 2: обучение навыку
    print("\n2. Обучение навыку...")
    pattern = [random.uniform(0, 1) for _ in range(100)]
    skill = brain.learn_skill(pattern, "тестовый_навык")
    print(f"   ID навыка: {skill['id']}")
    print(f"   Метка: {skill['label']}")
    
    # Шаг 3: воспроизведение навыка
    print("\n3. Воспроизведение навыка...")
    result = brain.recall_skill(skill['id'])
    print(f"   Результат: {len(result)} спайков")
    
    # Шаг 4: нейрогенезис
    print("\n4. Нейрогенезис...")
    brain.trigger_neurogenesis()
    state = brain.get_state()
    print(f"   Нейронов: {state['neurons']}")
    print(f"   Синапсов: {state['synapses']}")
    
    # Шаг 5: итоговое состояние
    print("\n5. Итоговое состояние:")
    state = brain.get_state()
    print(f"   Страх: {state['fear']:.2f}")
    print(f"   Любопытство: {state['curiosity']:.2f}")
    print(f"   Удивление: {state['surprise']:.2f}")
    print(f"   Навыков: {state['skills']}")
    
    print("\n✅ Тест завершён")


if __name__ == "__main__":
    test_brain_interface()
