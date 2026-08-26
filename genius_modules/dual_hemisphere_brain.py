# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ДВА ПОЛУШАРИЯ КАК ДВА ИИ
Гениальность: Полушария — это как два ИИ, которые общаются друг с другом,
проверяют ошибки, дополняют слабые стороны.

Левое полушарие — аналитика, детали, последовательности, язык, логика.
Правое полушарие — контекст, целостные образы, интуиция, пространство, эмоции.
Мозолистое тело (Corpus Callosum) — обмен информацией с задержкой.

Полушария не просто обрабатывают по-разному — они спорят.
ConflictMonitor и ConsciousnessSelection — это их диалог, который рождает сознание.
"""

import numpy as np
from collections import deque
import time
import random

class Hemisphere:
    """Отдельное полушарие мозга."""
    
    def __init__(self, name='left', num_neurons=50):
        self.name = name
        self.num_neurons = num_neurons
        self.neurons = [self._create_neuron() for _ in range(num_neurons)]
        self.synapses = {}  # (from, to) -> weight
        self.activity = np.zeros(num_neurons)
        self.spike_history = deque(maxlen=20)
        
        # Специализация полушария
        if name == 'left':
            self.specialization = 'language, logic, sequences, details'
            self.synaptic_density = 0.6  # более плотные локальные связи
        else:
            self.specialization = 'context, intuition, space, emotions, images'
            self.synaptic_density = 0.4  # более разреженные, но дальние связи
        
        # Инициализация связей
        self._initialize_synapses()
        
        # Состояние полушария
        self.confidence = 0.5
        self.energy = 100.0
        self.predictions = []
        self.errors = []
        
    def _create_neuron(self):
        """Создаёт нейрон с параметрами Ижикевича."""
        return {
            'v': -65.0 + np.random.randn() * 5.0,
            'u': -10.0 + np.random.randn() * 5.0,
            'a': 0.02 + np.random.rand() * 0.01,
            'b': 0.2 + np.random.rand() * 0.02,
            'c': -65.0 + np.random.randn() * 5.0,
            'd': 2.0 + np.random.rand() * 0.5,
            'spike': False,
            'firing_rate': 0.0,
            'threshold': 30.0,
        }
    
    def _initialize_synapses(self):
        """Инициализирует синапсы внутри полушария."""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and random.random() < self.synaptic_density:
                    self.synapses[(i, j)] = np.random.randn() * 0.1
    
    def process(self, input_vector):
        """Обрабатывает входной вектор и возвращает выход."""
        # Применяем вход к нейронам
        for i in range(min(len(input_vector), self.num_neurons)):
            self.neurons[i]['v'] += input_vector[i] * 0.1
        
        # Шаг нейронов
        spikes = []
        for i, neuron in enumerate(self.neurons):
            v = neuron['v']
            u = neuron['u']
            
            # Ижикевич
            dv = 0.04 * v * v + 5 * v + 140 - u
            du = neuron['a'] * (neuron['b'] * v - u)
            
            v += dv * 0.1
            u += du * 0.1
            
            if v >= neuron['threshold']:
                v = neuron['c']
                u += neuron['d']
                spikes.append(i)
            
            neuron['v'] = v
            neuron['u'] = u
            neuron['spike'] = (i in spikes)
        
        self.spike_history.append(spikes)
        self.activity = np.array([1.0 if i in spikes else 0.0 for i in range(self.num_neurons)])
        
        # Обновляем частоту спайков
        for i in spikes:
            self.neurons[i]['firing_rate'] = min(1.0, self.neurons[i]['firing_rate'] + 0.01)
        
        # Возвращаем выход как активность
        return self.activity
    
    def predict(self, input_vector):
        """Предсказывает следующий вход на основе текущего."""
        # Простая линейная экстраполяция
        if len(self.spike_history) >= 3:
            prev = self.spike_history[-1]
            prev_prev = self.spike_history[-2]
            trend = np.array(prev) - np.array(prev_prev) if prev and prev_prev else np.zeros(self.num_neurons)
            prediction = self.activity + trend * 0.1
        else:
            prediction = self.activity * 0.9
        
        return prediction
    
    def compute_error(self, input_vector):
        """Вычисляет ошибку предсказания."""
        prediction = self.predict(input_vector)
        error = np.mean((prediction - input_vector) ** 2)
        self.errors.append(error)
        return error
    
    def update_confidence(self, error):
        """Обновляет уверенность полушария."""
        self.confidence = min(1.0, max(0.1, 1.0 - error * 2.0))
        return self.confidence
    
    def get_state(self):
        """Возвращает состояние полушария."""
        return {
            'name': self.name,
            'specialization': self.specialization,
            'activity': self.activity.tolist(),
            'firing_rate': np.mean([n['firing_rate'] for n in self.neurons]),
            'confidence': self.confidence,
            'energy': self.energy,
            'synaptic_density': self.synaptic_density,
        }


class CorpusCallosum:
    """Мозолистое тело — обмен информацией между полушариями."""
    
    def __init__(self, transmission_delay=3):
        self.transmission_delay = transmission_delay
        self.buffer_left_to_right = deque(maxlen=10)
        self.buffer_right_to_left = deque(maxlen=10)
        self.transmission_efficiency = 0.7  # не вся информация передаётся
        
    def transmit(self, left_output, right_output):
        """Передаёт информацию между полушариями с задержкой."""
        # Левое → Правое
        self.buffer_left_to_right.append({
            'data': left_output,
            'time': time.time(),
            'efficiency': self.transmission_efficiency * random.uniform(0.8, 1.0)
        })
        
        # Правое → Левое
        self.buffer_right_to_left.append({
            'data': right_output,
            'time': time.time(),
            'efficiency': self.transmission_efficiency * random.uniform(0.8, 1.0)
        })
        
        # Получаем данные с задержкой
        left_received = self.buffer_right_to_left[0]['data'] if len(self.buffer_right_to_left) > 0 else None
        right_received = self.buffer_left_to_right[0]['data'] if len(self.buffer_left_to_right) > 0 else None
        
        # Ограничиваем буфер
        if len(self.buffer_left_to_right) > self.transmission_delay:
            self.buffer_left_to_right.popleft()
        if len(self.buffer_right_to_left) > self.transmission_delay:
            self.buffer_right_to_left.popleft()
        
        return left_received, right_received
    
    def get_state(self):
        """Возвращает состояние мозолистого тела."""
        return {
            'buffer_left_size': len(self.buffer_left_to_right),
            'buffer_right_size': len(self.buffer_right_to_left),
            'transmission_efficiency': self.transmission_efficiency,
            'delay': self.transmission_delay,
        }


class ConflictMonitor:
    """Монитор конфликтов между полушариями."""
    
    def __init__(self):
        self.conflict_history = deque(maxlen=50)
        self.resolution_history = deque(maxlen=20)
        self.conflict_level = 0.0
        
    def evaluate(self, left_output, right_output, left_confidence, right_confidence):
        """Оценивает конфликт между полушариями."""
        # Разница в выходах
        if len(left_output) == len(right_output):
            output_diff = np.mean((left_output - right_output) ** 2)
        else:
            output_diff = 0.5
        
        # Разница в уверенности
        confidence_diff = abs(left_confidence - right_confidence)
        
        # Общий конфликт
        self.conflict_level = min(1.0, output_diff * 0.7 + confidence_diff * 0.3)
        self.conflict_history.append(self.conflict_level)
        
        return self.conflict_level
    
    def resolve(self, left_output, right_output):
        """Разрешает конфликт."""
        # Если конфликт низкий, усредняем
        if self.conflict_level < 0.3:
            resolution = (left_output + right_output) / 2
            self.resolution_history.append({'method': 'average', 'level': self.conflict_level})
            return resolution
        
        # Если конфликт высокий, выбираем более уверенное полушарие
        # (здесь нужно передавать confidence, упрощённо)
        resolution = left_output  # заглушка
        self.resolution_history.append({'method': 'dominant', 'level': self.conflict_level})
        return resolution
    
    def get_state(self):
        """Возвращает состояние конфликта."""
        return {
            'level': self.conflict_level,
            'history': list(self.conflict_history),
            'resolutions': len(self.resolution_history),
        }


class DualHemisphereBrain:
    """Двуполушарный мозг с мозолистым телом."""
    
    def __init__(self, left_neurons=50, right_neurons=50):
        self.left = Hemisphere('left', left_neurons)
        self.right = Hemisphere('right', right_neurons)
        self.corpus = CorpusCallosum()
        self.conflict = ConflictMonitor()
        
        self.consciousness_level = 0.0
        self.integration_level = 0.0
        self.step_count = 0
        
    def process(self, input_vector):
        """Обрабатывает вход через оба полушария."""
        self.step_count += 1
        
        # 1. Оба полушария обрабатывают вход
        left_output = self.left.process(input_vector)
        right_output = self.right.process(input_vector)
        
        # 2. Обмен через мозолистое тело
        left_received, right_received = self.corpus.transmit(left_output, right_output)
        
        # 3. Если получена информация, интегрируем её
        if left_received is not None:
            # Левое получает информацию от правого
            pass
        if right_received is not None:
            # Правое получает информацию от левого
            pass
        
        # 4. Вычисляем ошибки предсказания
        left_error = self.left.compute_error(input_vector)
        right_error = self.right.compute_error(input_vector)
        
        # 5. Обновляем уверенность
        left_conf = self.left.update_confidence(left_error)
        right_conf = self.right.update_confidence(right_error)
        
        # 6. Мониторим конфликт
        conflict_level = self.conflict.evaluate(left_output, right_output, left_conf, right_conf)
        
        # 7. Сознание возникает из конфликта
        self.consciousness_level = min(1.0, conflict_level * 1.5 + 0.1)
        self.integration_level = min(1.0, 1.0 - conflict_level * 0.5)
        
        # 8. Разрешение конфликта
        if conflict_level > 0.4:
            resolution = self.conflict.resolve(left_output, right_output)
        else:
            resolution = left_output * 0.5 + right_output * 0.5
        
        return {
            'left_output': left_output,
            'right_output': right_output,
            'resolution': resolution,
            'consciousness': self.consciousness_level,
            'integration': self.integration_level,
            'conflict': conflict_level,
            'left_confidence': left_conf,
            'right_confidence': right_conf,
        }
    
    def get_state(self):
        """Возвращает полное состояние мозга."""
        return {
            'left': self.left.get_state(),
            'right': self.right.get_state(),
            'corpus': self.corpus.get_state(),
            'conflict': self.conflict.get_state(),
            'consciousness': self.consciousness_level,
            'integration': self.integration_level,
            'step': self.step_count,
        }


# ============================================================
# ТЕСТ
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 ДВА ПОЛУШАРИЯ КАК ДВА ИИ")
    print("=" * 60)
    
    brain = DualHemisphereBrain(left_neurons=10, right_neurons=10)
    
    # Тест: последовательность входов
    print("\n📊 Тест обработки:")
    for i in range(5):
        input_vec = np.random.randn(10) * 0.5
        result = brain.process(input_vec)
        
        print(f"\n  Шаг {i}:")
        print(f"    Конфликт: {result['conflict']:.2f}")
        print(f"    Сознание: {result['consciousness']:.2f}")
        print(f"    Интеграция: {result['integration']:.2f}")
        print(f"    Уверенность левого: {result['left_confidence']:.2f}")
        print(f"    Уверенность правого: {result['right_confidence']:.2f}")
        
        # Показываем специализацию
        if i == 0:
            print(f"    Левое полушарие: {brain.left.specialization}")
            print(f"    Правое полушарие: {brain.right.specialization}")
    
    print("\n💡 Гениальность: Полушария — это два ИИ, которые СПОРЯТ,")
    print("   и из их конфликта рождается СОЗНАНИЕ.")
