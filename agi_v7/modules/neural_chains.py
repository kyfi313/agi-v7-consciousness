# -*- coding: utf-8 -*-
"""
Нейронные цепочки — реальная симуляция нейронов с разной длиной и энергопотреблением
"""

import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple, Optional


class Neuron:
    """Один нейрон с мембранным потенциалом и спайками"""
    
    def __init__(self, threshold: float = 0.5, resting: float = 0.0):
        self.membrane_potential = resting
        self.resting = resting
        self.threshold = threshold
        self.spike_history = deque(maxlen=100)  # последние 100 спайков
        self.last_spike_time = 0
        self.energy_consumed = 0.0
        
    def update(self, input_current: float, time_step: float = 0.1) -> bool:
        """Обновляет мембранный потенциал и генерирует спайк"""
        # Интегрируем входной ток
        self.membrane_potential += input_current * time_step
        
        # Проверяем порог
        if self.membrane_potential >= self.threshold:
            # Спайк!
            self.membrane_potential = self.resting  # сброс
            self.spike_history.append(1)
            self.last_spike_time = len(self.spike_history)
            self.energy_consumed += 0.1  # энергия на спайк
            return True
        else:
            # Утечка (мембранный потенциал возвращается к покою)
            self.membrane_potential -= 0.02 * time_step
            self.spike_history.append(0)
            return False
    
    def get_firing_rate(self, window: int = 20) -> float:
        """Частота спайков за последние N шагов"""
        if len(self.spike_history) < window:
            return 0.0
        recent = list(self.spike_history)[-window:]
        return sum(recent) / window
    
    def reset(self):
        self.membrane_potential = self.resting
        self.spike_history.clear()
        self.energy_consumed = 0.0


class NeuralChain:
    """
    Цепочка нейронов с последовательной передачей сигнала.
    Длина цепочки определяет сложность обработки и энергопотребление.
    """
    
    def __init__(self, chain_id: int, length: int, energy_cost: float = 0.1):
        self.id = chain_id
        self.length = length
        self.neurons = [Neuron(threshold=0.3 + i * 0.05) for i in range(length)]
        self.energy_cost = energy_cost
        self.activation = 0.0  # текущая активность (0-1)
        self.total_energy_used = 0.0
        self.spike_count = 0
        self.cache = None  # сохранённый результат (для быстрого доступа)
        self.cached_result = None
        
    def process(self, input_signal: float, time_step: float = 0.1, use_cache: bool = False) -> float:
        """
        Пропускает сигнал через цепочку нейронов.
        Возвращает выходной сигнал (частота спайков последнего нейрона).
        """
        # Если используем кеш — возвращаем сохранённый результат
        if use_cache and self.cached_result is not None:
            self.activation = 0.1  # низкая активность (экономия энергии)
            return self.cached_result
        
        # Симуляция нейронов
        current = input_signal
        spike_count = 0
        
        for i, neuron in enumerate(self.neurons):
            # Передаём сигнал с усилением или ослаблением
            if i > 0:
                # Синаптическая задержка и ослабление
                current = current * random.uniform(0.7, 1.3)
            
            # Обновляем нейрон
            spiked = neuron.update(current, time_step)
            if spiked:
                spike_count += 1
            
            # Считываем выход нейрона как частоту спайков
            current = neuron.get_firing_rate(window=10)
        
        # Выходной сигнал = частота спайков последнего нейрона
        output = self.neurons[-1].get_firing_rate(window=10)
        
        # Обновляем статистику
        self.spike_count = spike_count
        self.activation = min(1.0, spike_count / self.length)
        self.total_energy_used += self.energy_cost * spike_count
        
        # Сохраняем результат в кеш (если активация высокая)
        if self.activation > 0.7:
            self.cached_result = output
        
        return output
    
    def get_energy(self) -> float:
        """Возвращает общую энергию, потраченную цепочкой"""
        return self.total_energy_used
    
    def get_accuracy(self) -> float:
        """Чем длиннее цепочка, тем выше потенциальная точность"""
        return min(1.0, self.length / 20.0)
    
    def get_efficiency(self) -> float:
        """Эффективность = точность / энергия"""
        energy = max(0.1, self.total_energy_used)
        return self.get_accuracy() / energy
    
    def reset(self):
        for neuron in self.neurons:
            neuron.reset()
        self.activation = 0.0
        self.total_energy_used = 0.0
        self.spike_count = 0
        self.cached_result = None


class ChainPool:
    """
    Пул цепочек нейронов разной длины.
    Мозг активирует цепочки в зависимости от доступной энергии и внимания.
    """
    
    def __init__(self, num_short: int = 100, num_medium: int = 50, num_long: int = 20):
        self.chains = []
        
        # Короткие цепочки (дёшево, быстро, но неточно)
        for i in range(num_short):
            length = random.randint(2, 5)
            cost = 0.1 + random.uniform(-0.02, 0.02)
            self.chains.append(NeuralChain(i, length, cost))
        
        # Средние цепочки (баланс)
        for i in range(num_short, num_short + num_medium):
            length = random.randint(6, 10)
            cost = 0.2 + random.uniform(-0.03, 0.03)
            self.chains.append(NeuralChain(i, length, cost))
        
        # Длинные цепочки (дорого, точно)
        for i in range(num_short + num_medium, num_short + num_medium + num_long):
            length = random.randint(11, 20)
            cost = 0.4 + random.uniform(-0.05, 0.05)
            self.chains.append(NeuralChain(i, length, cost))
        
        self.total_chains = len(self.chains)
        print(f"🧬 Создано {self.total_chains} цепочек: {num_short} коротких, {num_medium} средних, {num_long} длинных")
    
    def process(self, input_signal: float, energy_available: float, attention_intensity: float, 
                novelty: float = 0.0, time_step: float = 0.1) -> Tuple[float, Dict]:
        """
        Обрабатывает сигнал через цепочки с учётом энергии и внимания.
        
        Возвращает:
        - output: усреднённый выходной сигнал
        - stats: статистика использования цепочек
        """
        # Определяем, сколько цепочек активировать
        # Чем выше энергия и внимание, тем больше цепочек и длиннее
        
        # Базовый уровень активации
        base_activation_ratio = 0.3  # 30% цепочек всегда активны
        
        # Дополнительная активация от внимания и новизны
        extra_ratio = attention_intensity * 0.4 + novelty * 0.3
        
        # Общий коэффициент активации (0.3 - 1.0)
        activation_ratio = min(1.0, base_activation_ratio + extra_ratio)
        
        # Корректировка по энергии: если энергии мало, снижаем активацию
        if energy_available < 0.3:
            activation_ratio = max(0.2, activation_ratio * 0.5)
        elif energy_available < 0.6:
            activation_ratio = max(0.3, activation_ratio * 0.8)
        
        # Определяем, сколько цепочек активировать
        num_active = int(self.total_chains * activation_ratio)
        
        # Выбираем цепочки: предпочтение длинным при высоком внимании
        if attention_intensity > 0.7:
            # Фокус → больше длинных цепочек
            sorted_chains = sorted(self.chains, key=lambda c: c.length, reverse=True)
        elif novelty > 0.5:
            # Новизна → больше длинных цепочек (исследование)
            sorted_chains = sorted(self.chains, key=lambda c: c.length, reverse=True)
        else:
            # Обычный режим → все цепочки равны
            sorted_chains = self.chains.copy()
            random.shuffle(sorted_chains)
        
        # Берём первые num_active цепочек
        active_chains = sorted_chains[:num_active]
        
        # Обрабатываем сигнал через каждую активную цепочку
        outputs = []
        total_energy_used = 0.0
        chain_stats = {
            'num_active': num_active,
            'activation_ratio': activation_ratio,
            'total_chains': self.total_chains,
            'short_used': 0,
            'medium_used': 0,
            'long_used': 0,
            'avg_length': 0,
            'avg_accuracy': 0,
        }
        
        for chain in active_chains:
            # Решаем, использовать ли кеш (экономия энергии)
            use_cache = (energy_available < 0.4 and chain.cached_result is not None)
            
            # Обрабатываем
            output = chain.process(input_signal, time_step, use_cache=use_cache)
            outputs.append(output)
            total_energy_used += chain.get_energy()
            
            # Статистика
            if chain.length <= 5:
                chain_stats['short_used'] += 1
            elif chain.length <= 10:
                chain_stats['medium_used'] += 1
            else:
                chain_stats['long_used'] += 1
            chain_stats['avg_length'] += chain.length
            chain_stats['avg_accuracy'] += chain.get_accuracy()
        
        # Усредняем статистику
        if chain_stats['num_active'] > 0:
            chain_stats['avg_length'] /= chain_stats['num_active']
            chain_stats['avg_accuracy'] /= chain_stats['num_active']
        
        # Итоговый выход = среднее по активным цепочкам
        output = sum(outputs) / len(outputs) if outputs else 0.0
        
        # Возвращаем результат и статистику
        return output, {
            'output': output,
            'energy_used': total_energy_used,
            'stats': chain_stats,
            'activation_ratio': activation_ratio,
            'num_active': num_active,
        }
    
    def get_summary(self) -> Dict:
        """Возвращает сводку по пулу цепочек"""
        total_energy = sum(c.get_energy() for c in self.chains)
        avg_accuracy = sum(c.get_accuracy() for c in self.chains) / len(self.chains)
        avg_length = sum(c.length for c in self.chains) / len(self.chains)
        
        return {
            'total_chains': self.total_chains,
            'total_energy_used': total_energy,
            'avg_accuracy': avg_accuracy,
            'avg_length': avg_length,
        }
    
    def reset(self):
        for chain in self.chains:
            chain.reset()
