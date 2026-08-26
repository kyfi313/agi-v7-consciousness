# -*- coding: utf-8 -*-
"""
Зрительная кора — обработка визуальной информации с использованием нейронных цепочек
Реализует реальную симуляцию нейронов с разной длиной цепочек и непрерывной регуляцией мощности
"""

import numpy as np
from collections import deque
from ...core.base import BaseModule
from ...core.state import GlobalState
from ..neural_chains import ChainPool


class VisualCortexModule(BaseModule):
    name = "visual"

    def __init__(self):
        # Создаём пул нейронных цепочек с разной длиной
        # Короткие (2-5 нейронов) — дёшево, быстро, но грубо
        # Средние (6-10) — баланс точности и энергии
        # Длинные (11-20) — дорого, точно, требуют много энергии
        self.chain_pool = ChainPool(num_short=100, num_medium=50, num_long=20)
        
        # Визуальная память
        self.visual_memory = deque(maxlen=20)
        # Обнаруженные объекты
        self.detected_objects = []
        # Статистика обработки
        self.processing_stats = {}
        # История активации цепочек
        self.chain_history = deque(maxlen=50)
        
        # Веса для обучения (сохраняем для совместимости)
        self.weights = {}

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем визуальный вход из perception
        visual_input = state.perception.get('visual', None)
        
        # Если нет данных, возвращаем
        if visual_input is None:
            return state
        
        # Извлекаем параметры состояния для регуляции мощности
        energy_available = state.energy
        attention_intensity = state.attention_salience if hasattr(state, 'attention_salience') else 0.5
        novelty = state.perception.get('novelty', 0.0)
        
        # Обрабатываем визуальные данные через нейронные цепочки
        processed, chain_stats = self._process_with_chains(
            visual_input, 
            energy_available, 
            attention_intensity, 
            novelty,
            state
        )
        
        # Сохраняем статистику
        self.processing_stats = chain_stats
        self.chain_history.append(chain_stats)
        
        # ДЕТАЛЬНЫЙ ВЫВОД ПО ЦЕПОЧКАМ
        print(f"🔗 ЦЕПОЧКИ: энергия={energy_available:.2f}, внимание={attention_intensity:.2f}, новизна={novelty:.2f}")
        print(f"   Активировано: {chain_stats.get('num_active', 0)}/{chain_stats.get('total_chains', 0)} цепочек")
        print(f"   Средняя длина: {chain_stats.get('stats', {}).get('avg_length', 0):.1f} нейронов")
        print(f"   Энергозатраты: {chain_stats.get('energy_used', 0.0):.3f}")
        
        # Показываем распределение по типам цепочек
        if 'chains_activated' in chain_stats:
            short = chain_stats['chains_activated'].get('short', 0)
            medium = chain_stats['chains_activated'].get('medium', 0)
            long = chain_stats['chains_activated'].get('long', 0)
            total_activated = chain_stats.get('num_active', 0)
            if total_activated > 0:
                print(f"   Распределение: короткие {short} ({short/total_activated*100:.0f}%), "
                      f"средние {medium} ({medium/total_activated*100:.0f}%), "
                      f"длинные {long} ({long/total_activated*100:.0f}%)")
        print()
        
        # Сохраняем в состояние
        state.visual_latent = processed.get('latent', np.zeros(64))
        state.perception['visual_processed'] = True
        state.perception['visual_salience'] = processed.get('salience', 0.0)
        state.perception['chain_stats'] = chain_stats
        
        # Обновляем внимание
        if processed.get('salience', 0.0) > 0.5:
            state.attention_focus = 'visual'
            state.attention_salience = processed.get('salience', 0.0)
        
        # Распознаём объекты
        objects = self._recognize_objects(processed, state)
        if objects:
            state.objects = objects
        
        # Тратим энергию на обработку
        state.energy = max(0.0, state.energy - chain_stats.get('energy_used', 0.0) * 0.05)
        
        return state
    
    def _process_with_chains(self, visual_input, energy: float, attention: float, novelty: float, state: GlobalState) -> tuple:
        """
        Обрабатывает визуальные данные через нейронные цепочки.
        Возвращает (обработанные_данные, статистика_цепочек).
        """
        # Преобразуем визуальный вход в числовой сигнал
        if isinstance(visual_input, np.ndarray):
            # Если массив — усредняем до скаляра
            input_signal = float(np.mean(visual_input))
        elif isinstance(visual_input, dict):
            # Если словарь — извлекаем ключевые параметры
            num_objects = visual_input.get('num_objects', 0)
            brightness = visual_input.get('brightness', 0.5)
            motion = visual_input.get('motion', 0.0)
            input_signal = (brightness * 0.5 + motion * 0.3 + min(1.0, num_objects / 10) * 0.2)
        else:
            input_signal = 0.5  # значение по умолчанию
        
        # Пропускаем через цепочки с непрерывной регуляцией мощности
        output, chain_stats = self.chain_pool.process(
            input_signal=input_signal,
            energy_available=energy,
            attention_intensity=attention,
            novelty=novelty,
            time_step=0.1
        )
        
        # Формируем латентное представление
        latent = np.array([input_signal, output, attention, novelty, energy * 0.1])
        
        # Обновляем память
        self.visual_memory.append({
            'input': input_signal,
            'output': output,
            'attention': attention,
            'novelty': novelty,
            'energy': energy,
            'step': state.step,
        })
        
        # Возвращаем обработанные данные
        processed = {
            'latent': latent,
            'salience': output,
            'input_signal': input_signal,
            'chain_activation': chain_stats,
            'num_chains_active': chain_stats.get('num_active', 0),
            'avg_chain_length': chain_stats.get('stats', {}).get('avg_length', 0),
        }
        
        return processed, chain_stats
    
    def _recognize_objects(self, processed: dict, state: GlobalState) -> list:
        """Распознавание объектов на основе обработанного сигнала"""
        objects = []
        salience = processed.get('salience', 0.0)
        
        # Если салиентность выше порога — распознаём объекты
        if salience > 0.3:
            # Количество объектов зависит от салиентности и длины цепочек
            num_objects = min(5, int(salience * 10) + 1)
            avg_length = processed.get('avg_chain_length', 0)
            
            # Чем длиннее цепочки, тем точнее распознавание
            accuracy = min(1.0, avg_length / 15.0)
            
            for i in range(num_objects):
                obj = {
                    'id': i,
                    'type': f'object_{i}',
                    'novelty': float(np.random.random() * 0.3 * (1 - accuracy)),
                    'position': (float(i * 10 + np.random.random() * 5), float(i * 20 + np.random.random() * 5)),
                    'confidence': accuracy * (0.7 + np.random.random() * 0.3),
                }
                objects.append(obj)
            
            self.detected_objects = objects
        
        return objects
    
    def get_chain_summary(self) -> dict:
        """Возвращает сводку по цепочкам"""
        return self.chain_pool.get_summary()
    
    def reset(self):
        self.visual_memory.clear()
        self.detected_objects = []
        self.processing_stats = {}
        self.chain_history.clear()
        self.chain_pool.reset()
