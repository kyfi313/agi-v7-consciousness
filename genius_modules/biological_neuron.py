# -*- coding: utf-8 -*-
"""
МОДУЛЬ: БИОЛОГИЧЕСКИЙ НЕЙРОН
Гениальность: Реализация нейрона как интегратора с мембранным потенциалом,
потенциалом действия, рефрактерным периодом и энергетическими затратами.

Основан на модели Ходжкина-Хаксли в упрощённой форме.
"""

import numpy as np
from collections import deque
import time
import random


class BiologicalNeuron:
 """
 Нейрон, моделирующий реальный биологический процесс.

 Ключевые параметры:
 - membrane_potential: мембранный потенциал (мВ), покой -70 мВ
 - threshold: порог возбуждения (обычно -50 мВ)
 - refractory_period: рефрактерный период (мс)
 - energy_cost: энергозатраты на спайк
 - fatigue: утомляемость
 """
 def __init__(self, neuron_id=0):
 self.id = neuron_id

 # Мембранные параметры (в мВ)
 self.resting_potential = -70.0
 self.membrane_potential = -70.0
 self.threshold = -50.0
 self.reset_potential = -75.0

 # Временные параметры (в мс)
 self.refractory_period = 2.0  # абсолютная рефрактерность
 self.relative_refractory_period = 5.0  # относительная
 self.time_since_spike = 0.0
 self.firing_rate = 0.0

 # Энергетические параметры
 self.energy_level = 1.0  # 0.0 - 1.0
 self.energy_consumption_per_spike = 0.05
 self.energy_recovery_rate = 0.001
 self.fatigue = 0.0  # 0.0 - 1.0

 # Синаптические входы
 self.synaptic_inputs = deque(maxlen=100)
 self.excitatory_inputs = []
 self.inhibitory_inputs = []

 # Выход
 self.spikes = deque(maxlen=50)
 self.last_spike_time = 0.0
 self.spike_count = 0

 # Мембранные константы
 self.membrane_time_constant = 20.0  # мс
 self.leak_conductance = 0.1

 def add_synapse(self, synapse):
 """Добавляет синапс к нейрону."""
 if synapse.type == 'excitatory':
 self.excitatory_inputs.append(synapse)
 else:
 self.inhibitory_inputs.append(synapse)

 def receive_input(self, current, dt=1.0):
 """Получает входной ток (в мкА)."""
 if self.refractory_period > 0:
 return

 # Интеграция мембранного потенциала (упрощённая модель)
 dv = (-self.leak_conductance * (self.membrane_potential - self.resting_potential) + current) / self.membrane_time_constant
 self.membrane_potential += dv * dt

 def update(self, dt=1.0):
 """Обновляет состояние нейрона."""
 self.time_since_spike += dt

 # Рефрактерный период
 if self.time_since_spike < self.refractory_period:
 self.membrane_potential = self.reset_potential
 return None

 if self.time_since_spike < self.relative_refractory_period:
 refractory_factor = 1.0 - (self.time_since_spike - self.refractory_period) / (self.relative_refractory_period - self.refractory_period)
 self.threshold = -50.0 + 10.0 * refractory_factor
 else:
 self.threshold = -50.0

 # Суммация входов
 total_current = 0.0
 for syn in self.excitatory_inputs:
 total_current += syn.get_current()
 for syn in self.inhibitory_inputs:
 total_current -= syn.get_current()

 # Утомляемость
 self.fatigue = max(0.0, self.fatigue - 0.001)
 total_current *= (1.0 - self.fatigue * 0.5)

 # Интеграция
 if self.energy_level > 0.1:
 self.receive_input(total_current, dt)

 # Проверка на спайк
 if self.membrane_potential >= self.threshold:
 return self.fire()

 return None

 def fire(self):
 """Генерирует спайк."""
 self.spike_count += 1
 self.spikes.append({'time': time.time(), 'potential': self.membrane_potential})
 self.last_spike_time = self.membrane_potential

 # Сброс мембранного потенциала
 self.membrane_potential = self.reset_potential
 self.time_since_spike = 0.0

 # Энергозатраты
 self.energy_level -= self.energy_consumption_per_spike
 self.energy_level = max(0.0, self.energy_level)
 self.fatigue = min(1.0, self.fatigue + 0.01)

 return 1.0

 def recover_energy(self):
 """Восстанавливает энергию."""
 self.energy_level = min(1.0, self.energy_level + self.energy_recovery_rate)
 return self.energy_level

 def get_state(self):
 return {
 'id': self.id,
 'membrane_potential': self.membrane_potential,
 'threshold': self.threshold,
 'firing_rate': self.firing_rate,
 'energy': self.energy_level,
 'fatigue': self.fatigue,
 'spike_count': self.spike_count,
 'excitatory_inputs': len(self.excitatory_inputs),
 'inhibitory_inputs': len(self.inhibitory_inputs)
 }


class Synapse:
 """
 Биологический синапс с пластичностью.
 """
 def __init__(self, from_neuron, to_neuron, weight=0.5, synapse_type='excitatory'):
 self.from_neuron = from_neuron
 self.to_neuron = to_neuron
 self.weight = weight
 self.type = synapse_type
 self.delay = 0.5  # мс
 self.transmitter_release_prob = 0.8
 self.transmitter_depletion = 0.0
 self.depression_rate = 0.02
 self.recovery_rate = 0.01

 self.ltp = 0.0  # долговременная потенциация
 self.ltd = 0.0  # долговременная депрессия

 self.spike_times_pre = deque(maxlen=20)
 self.spike_times_post = deque(maxlen=20)

 def get_current(self):
 """Вычисляет ток, который синапс передаёт на постсинаптический нейрон."""
 if random.random() > self.transmitter_release_prob:
 return 0.0

 # Депрессия
 effective_weight = self.weight * (1.0 - self.transmitter_depletion)
 # Пластичность
 effective_weight *= (1.0 + self.ltp - self.ltd)

 return effective_weight * 0.1

 def update_plasticity(self, pre_spike_time, post_spike_time):
 """Обновляет синаптическую пластичность на основе времени спайков."""
 # STDP: пре-пост
 dt = post_spike_time - pre_spike_time
 if dt > 0:  # пре-пост (потенциация)
 self.ltp += 0.01 * np.exp(-dt / 20.0)
 elif dt < 0:  # пост-пре (депрессия)
 self.ltd += 0.01 * np.exp(dt / 20.0)

 def depress(self):
 """Депрессия синапса после использования."""
 self.transmitter_depletion = min(1.0, self.transmitter_depletion + self.depression_rate)

 def recover(self):
 """Восстановление синапса."""
 self.transmitter_depletion = max(0.0, self.transmitter_depletion - self.recovery_rate)

 def get_state(self):
 return {
 'weight': self.weight,
 'type': self.type,
 'ltp': self.ltp,
 'ltd': self.ltd,
 'depletion': self.transmitter_depletion
 }


# ============================================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================================

if __name__ == "__main__":
 print("="*70)
 print("🧠 БИОЛОГИЧЕСКИЙ НЕЙРОН — ТЕСТ")
 print("="*70)

 # Создаём два нейрона
 neuron1 = BiologicalNeuron(1)
 neuron2 = BiologicalNeuron(2)

 # Создаём синапс между ними
 synapse = Synapse(neuron1, neuron2, weight=0.8, synapse_type='excitatory')

 # Добавляем синапс к нейрону 2
 neuron2.add_synapse(synapse)

 # Симулируем активность
 print("\nСимуляция 100 шагов:")
 for i in range(100):
 # Стимулируем нейрон 1 случайным током
 current = random.random() * 0.5
 neuron1.receive_input(current, dt=1.0)
 spike = neuron1.update(dt=1.0)

 if spike:
 # Запоминаем время спайка для пластичности
 synapse.spike_times_pre.append(i)
 # Передаём сигнал на постсинаптический нейрон
 neuron2.receive_input(synapse.get_current(), dt=1.0)

 # Обновляем нейрон 2
 spike2 = neuron2.update(dt=1.0)
 if spike2:
 synapse.spike_times_post.append(i)
 # Обновляем пластичность
 if len(synapse.spike_times_pre) > 0 and len(synapse.spike_times_post) > 0:
 synapse.update_plasticity(synapse.spike_times_pre[-1], synapse.spike_times_post[-1])

 # Восстанавливаем энергию
 neuron1.recover_energy()
 neuron2.recover_energy()
 synapse.recover()

 # Выводим состояние
 if i % 20 == 0:
 print(f"Шаг {i}: Нейрон1 энергия={neuron1.energy_level:.2f}, спайков={neuron1.spike_count}")
 print(f"Шаг {i}: Нейрон2 энергия={neuron2.energy_level:.2f}, спайков={neuron2.spike_count}")

 print("\nСостояние синапса:")
 print(f"  Вес={synapse.weight:.2f}, LTP={synapse.ltp:.3f}, LTD={synapse.ltd:.3f}")
 print(f"  Депрессия={synapse.transmitter_depletion:.2f}")

 print("\n✅ Биологический нейрон работает!")
