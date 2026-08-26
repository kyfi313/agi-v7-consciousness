# -*- coding: utf-8 -*-
"""
Мозговой модуль AGI v7
Реализует все когнитивные функции на основе биологически-реалистичных нейронов
"""

import numpy as np
import random
import time
from collections import deque

# --- ИМПОРТ ВЫЧИСЛИТЕЛЬНОГО БЭКЕНДА ---
try:
    from compute import get_xp, to_numpy, to_device, get_backend, is_gpu_available
    XP = get_xp()
    BACKEND = get_backend()
    GPU_AVAILABLE = is_gpu_available()
    print(f"🧠 Brain module использует бэкенд: {BACKEND}")
except ImportError:
    XP = np
    BACKEND = 'numpy'
    GPU_AVAILABLE = False
    print("🧠 Brain module: бэкенд не найден, используем NumPy")

class MemoryCompressor:
    """Сжатие памяти с помощью SVD (NumPy) и нейронных кодов"""
    
    def __init__(self, input_dim=50, code_dim=10):
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.patterns = []  # (code, label, raw, valence, arousal, importance)
        self.fitted = False
        self.buffer = []
        self.components = None
        self.mean = None
        
    def compress(self, pattern):
        """Сжимает паттерн в код"""
        if len(pattern) != self.input_dim:
            pattern = pattern[:self.input_dim]
            if len(pattern) < self.input_dim:
                pattern += [0.0] * (self.input_dim - len(pattern))
        
        pattern = np.array(pattern)
        
        if not self.fitted:
            self.buffer.append(pattern)
            if len(self.buffer) >= max(10, self.input_dim):
                self._fit_pca()
                self.fitted = True
                return np.zeros(self.code_dim).tolist()
            return np.zeros(self.code_dim).tolist()
        
        try:
            # Центрируем
            centered = pattern - self.mean
            # Проецируем на компоненты
            code = np.dot(centered, self.components.T)
            return code.tolist()
        except:
            return np.zeros(self.code_dim).tolist()
    
    def _fit_pca(self):
        """Вычисляет PCA через SVD"""
        data = np.array(self.buffer)
        self.mean = np.mean(data, axis=0)
        centered = data - self.mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components = Vt[:self.code_dim]
        self.buffer = []
    
    def store(self, pattern, label=None, valence=0.0, arousal=0.0):
        """
        Сохраняет паттерн в память с эмоциональной валентностью.
        
        Args:
            pattern: Входной паттерн
            label: Метка для поиска
            valence: Эмоциональная валентность (-1.0 до 1.0)
            arousal: Уровень возбуждения (0.0 до 1.0)
        """
        code = self.compress(pattern)
        idx = len(self.patterns)
        # Важность = комбинация абсолютной валентности и возбуждения
        importance = 0.3 + 0.7 * (abs(valence) * 0.5 + arousal * 0.5)
        self.patterns.append((code, label, pattern, valence, arousal, importance))
        return idx
    
    def recall(self, idx):
        """Восстанавливает паттерн по индексу"""
        if 0 <= idx < len(self.patterns):
            return self.patterns[idx][2]
        return None
    
    def recall_with_emotion(self, idx):
        """Восстанавливает паттерн с эмоциональными данными"""
        if 0 <= idx < len(self.patterns):
            code, label, pattern, valence, arousal, importance = self.patterns[idx]
            return pattern, valence, arousal, importance
        return None, 0.0, 0.0, 0.0
    
    def recall_by_label(self, label):
        """Восстанавливает паттерн по метке"""
        for code, lbl, pattern, valence, arousal, importance in self.patterns:
            if lbl == label:
                return pattern
        return None
    
    def get_code(self, idx):
        """Возвращает код паттерна"""
        if 0 <= idx < len(self.patterns):
            return self.patterns[idx][0]
        return None
    
    def find_similar(self, pattern, top_n=3, min_importance=0.0):
        """
        Находит похожие паттерны с учётом важности.
        
        Args:
            pattern: Входной паттерн
            top_n: Количество результатов
            min_importance: Минимальная важность для включения
        """
        code = self.compress(pattern)
        if not code or len(code) == 0:
            return []
        
        similarities = []
        for i, (c, lbl, p, valence, arousal, importance) in enumerate(self.patterns):
            if importance < min_importance:
                continue
            if len(c) != len(code):
                continue
            sim = np.corrcoef(code, c)[0, 1]
            if np.isnan(sim):
                sim = 0.0
            # Взвешиваем сходство на важность
            weighted_sim = sim * (0.5 + 0.5 * importance)
            similarities.append((i, weighted_sim, importance))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_stats(self):
        return {
            'total_patterns': len(self.patterns),
            'input_dim': self.input_dim,
            'code_dim': self.code_dim,
            'fitted': self.fitted
        }
    
    def _update_pca(self):
        """Обновляет PCA если есть новые данные"""
        if len(self.buffer) >= max(10, self.input_dim * 2):
            self.pca.fit(np.array(self.buffer))
            self.fitted = True
            self.buffer = []


class RealNeuron:
    """Биологически-реалистичный нейрон с динамикой Ижикевича и STDP"""
    
    def __init__(self, neuron_type='regular', idx=0):
        self.idx = idx
        self.type = neuron_type
        
        # Параметры Ижикевича для разных типов нейронов
        params = {
            'regular': (0.02, 0.2, -65.0, 8.0),
            'fast': (0.1, 0.2, -65.0, 2.0),
            'burst': (0.02, 0.25, -55.0, 4.0),
            'chattering': (0.02, 0.2, -50.0, 2.0),
            'low_threshold': (0.02, 0.25, -65.0, 7.0),
            'high_threshold': (0.02, 0.2, -60.0, 10.0)
        }
        self.a, self.b, self.c, self.d = params.get(neuron_type, params['regular'])
        
        # Состояние
        self.v = -65.0  # мембранный потенциал
        self.u = 0.0    # восстановление
        self.spiked = False
        self.last_spike_time = -100.0
        
        # Синапсы: список (целевой_нейрон, вес)
        self.synapses = []
        
        # STDP параметры
        self.trace = 0.0
        self.A_plus = 0.1
        self.A_minus = 0.12
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        
        # История изменений весов для метапластичности
        self.weight_change_history = deque(maxlen=20)
        self.last_stdp_change = 0.0
        
    def step(self, I_ext, dt=0.5, time_step=0):
        """Один шаг динамики Ижикевича"""
        # Обновление по Ижикевичу
        dv = (0.04 * self.v**2 + 5.0 * self.v + 140.0 - self.u + I_ext) * dt
        du = (self.a * (0.2 * self.v - self.u)) * dt
        
        self.v += dv
        self.u += du
        
        # Проверка спайка
        if self.v >= 30.0:
            self.v = self.c
            self.u += self.d
            self.spiked = True
            self.last_spike_time = time_step
        else:
            self.spiked = False
        
        return self.spiked
    
    def fire(self, dt=0.5, time_step=0):
        """Генерирует спайк"""
        self.v = self.c
        self.u += self.d
        self.spiked = True
        self.last_spike_time = time_step
        return True
    
    def update_synapses(self, pre_spike_time, post_spike_time, dt, modulation=None):
        """Обновляет веса синапсов по STDP с учётом эмоциональной модуляции"""
        if modulation is None:
            modulation = {'ltp_rate': 1.0, 'ltd_rate': 1.0, 'plasticity': 1.0}
        
        delta_t = post_spike_time - pre_spike_time
        if delta_t == 0:
            return
        
        # STDP: если пре-синаптический спайк предшествует пост-синаптическому
        ltp_rate = modulation.get('ltp_rate', 1.0)
        ltd_rate = modulation.get('ltd_rate', 1.0)
        plasticity = modulation.get('plasticity', 1.0)
        
        if delta_t > 0 and delta_t < 50.0:
            weight_change = self.A_plus * np.exp(-delta_t / self.tau_plus) * ltp_rate * plasticity
        elif delta_t < 0 and delta_t > -50.0:
            weight_change = -self.A_minus * np.exp(delta_t / self.tau_minus) * ltd_rate * plasticity
        else:
            weight_change = 0.0
        
        # Сохраняем изменение для метапластичности
        self.last_stdp_change = weight_change
        if abs(weight_change) > 0.001:
            self.weight_change_history.append(abs(weight_change))
        
        # Применяем изменения к весам
        for idx, (target_idx, weight) in enumerate(self.synapses):
            new_weight = weight + weight_change * 0.01
            new_weight = max(0.0, min(2.0, new_weight))
            self.synapses[idx] = (target_idx, new_weight)
    
    def add_synapse(self, target_idx, weight=0.5):
        """Добавляет синапс"""
        self.synapses.append((target_idx, weight))
    
    def get_state(self):
        return {
            'v': self.v,
            'u': self.u,
            'spiked': self.spiked,
            'last_spike': self.last_spike_time,
            'type': self.type,
            'synapses': len(self.synapses)
        }


class NeuronNetwork:
    """Нейронная сеть с энергетическим управлением, нейрогенезом, рекуррентными связями, WTA и консолидацией"""
    
    def __init__(self, num_neurons=100, connectivity=0.08, recurrent_strength=0.3, wta_radius=0.2):
        self.num_neurons = num_neurons
        self.connectivity = connectivity
        self.recurrent_strength = recurrent_strength  # Сила рекуррентных связей
        self.wta_radius = wta_radius  # Радиус латерального торможения (доля нейронов)
        self.neurons = []
        self.step_count = 0
        
        # Энергетика
        self.network_energy = 100.0
        self.max_energy = 100.0
        self.energy_depletion = 0.01
        self.energy_recovery = 0.02
        
        # Три системы
        self.fear = 0.0
        self.curiosity = 0.3
        self.surprise = 0.2
        self.surprise_threshold = 0.15
        self.crisis_threshold = 0.3
        self.crisis_mode = False
        
        # --- ЭМОЦИОНАЛЬНЫЙ КАСКАД (модуляция пластичности) ---
        # Каждая эмоция модулирует скорость обучения (LTP/LTD)
        self.emotion_ltp_modulation = {
            'fear': 1.5,       # Страх ускоряет обучение (выживание)
            'curiosity': 1.2,  # Любопытство ускоряет исследование
            'surprise': 1.0,   # Удивление — нейтрально
            'sadness': 0.7,    # Печаль замедляет обучение
            'pleasure': 1.3,   # Удовольствие укрепляет успешные паттерны
            'anger': 1.4,      # Гнев ускоряет изменение стратегии
            'boredom': 0.5,    # Скука замедляет обучение
        }
        self.current_dominant_emotion = 'curiosity'  # По умолчанию
        
        # --- СОСТОЯНИЕ КАСКАДА ---
        self.cascade_stage = 0  # 0-5
        self.cascade_timer = 0
        self.cascade_duration = 20
        self.cascade_active = False
        
        # --- ЯКОРЬ РЕАЛЬНОСТИ ---
        self.reality_anchor = 1.0  # 1.0 = реальность, 0.0 = воображение
        self.reality_history = deque(maxlen=100)
        
        # --- АССОЦИАТИВНАЯ ПАМЯТЬ И ПРЕДСКАЗАНИЕ (ВНУТРИ МОЗГА) ---
        self.memory_episodes = []          # (input_vec, action, reward)
        self.memory_activations = []       # Активированные воспоминания на текущем шаге
        self.prediction = None             # Предсказание на текущем шаге
        self.prediction_error = 0.0        # Ошибка предсказания
        self.recent_inputs = deque(maxlen=10)  # Для предсказания паттернов
        self.association_strength = {}     # Сила ассоциаций между паттернами
        
        # Метакогниция
        self.awareness_level = 0.1
        self.self_confidence = 0.5
        self.prediction_accuracy = 0.5
        self.insight_count = 0
        self.reflection_depth = 0.0
        self.insight_buffer = deque(maxlen=10)
        
        # --- ЭНЕРГЕТИЧЕСКОЕ ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ МЫШЛЕНИЯ ---
        self.thinking_mode = 'normal'  # 'fast', 'normal', 'deep'
        self.mode_switch_thresholds = {
            'fast': 0.4,   # энергия < 40% -> быстрый режим
            'normal': 0.7, # энергия < 70% -> нормальный режим
            'deep': 1.0    # энергия >= 70% -> глубокий режим
        }
        self.chain_length_multiplier = 1.0  # множитель длины цепочек нейронов
        self.recurrent_gain = 1.0  # усиление рекуррентных связей
        self.depth_penetration = 1.0  # глубина распространения сигнала
        self.focus_level = 0.0  # 0-1, концентрация
        self.thought_complexity = 0.5  # сложность мышления
        
        # Память
        self.working_memory = deque(maxlen=9)  # 7±2
        self.long_term_memory = []
        self.prediction_memory = []
        self.prediction_window = 20
        
        # Навыки
        self.skills = {}  # name -> (sequence, strength)
        self.skill_strength = {}
        
        # Нейрогенез
        self.birth_rate = 0.015
        self.max_neurons = 1000
        self.pruning_threshold = 0.05
        self.pruning_interval = 50
        
        # Гомеостатическое масштабирование
        self.homeostatic_interval = 20  # Каждые 20 шагов
        self.target_activity = 0.3  # Целевая активность нейрона
        self.scaling_factor = 0.95  # Масштабирование на 5%
        self.activity_history = {}  # id -> [активности]
        
        # Выживаемость новых нейронов
        self.survival_window = 50  # Шагов на адаптацию
        self.new_neurons = {}  # id -> (нейрон, счётчик)
        
        # --- РЕКУРРЕНТНЫЕ СВЯЗИ ---
        self.recurrent_weights = {}  # neuron_idx -> {neuron_idx: weight}
        self.recurrent_delay = 3  # Количество шагов задержки (аналог синаптической задержки)
        self.recurrent_buffer = []  # Буфер для задержанных сигналов
        
        # --- WINNER-TAKE-ALL (латеральное торможение) ---
        self.wta_active = True
        self.lateral_inhibition_strength = 0.3
        self.wta_clusters = 5  # Количество кластеров для конкуренции
        
        # --- СОН И КОНСОЛИДАЦИЯ ---
        self.sleep_counter = 0
        self.sleep_interval = 100  # Сон каждые 100 шагов
        self.sleep_duration = 20  # Длительность сна в шагах
        self.is_sleeping = False
        self.consolidation_buffer = []  # Паттерны для консолидации
        self.synaptic_strength_distribution = []  # Для анализа связей
        
        # Инициализация нейронов
        self._init_neurons()
        
    def _init_neurons(self):
        """Инициализирует нейроны с синапсами"""
        types = ['regular', 'fast', 'burst', 'chattering', 'low_threshold', 'high_threshold']
        
        for i in range(self.num_neurons):
            neuron_type = random.choice(types)
            neuron = RealNeuron(neuron_type, i)
            self.neurons.append(neuron)
        
        # Создаём синапсы
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and random.random() < self.connectivity:
                    weight = random.uniform(0.1, 0.8)
                    self.neurons[i].add_synapse(j, weight)
    
    def step(self, input_signal, reality_anchor=None):
        """Основной шаг сети с рекуррентными связями, WTA и сном"""
        self.step_count += 1
        
        # --- ЯКОРЬ РЕАЛЬНОСТИ ---
        if reality_anchor is not None:
            self.reality_anchor = reality_anchor
        else:
            self.reality_anchor = 1.0  # По умолчанию реальность
        self.reality_history.append(self.reality_anchor)
        
        # --- ФАЗА СНА ---
        if self.is_sleeping:
            return self._process_sleep()
        
        # Проверка, нужно ли уснуть
        self.sleep_counter += 1
        if self.sleep_counter >= self.sleep_interval:
            self.sleep_counter = 0
            self.is_sleeping = True
            self.sleep_duration = 20
            print(f"💤 Сон начался на шаге {self.step_count}")
            return self._process_sleep()
        
        # Энергия
        self._update_energy()
        
        # --- ПЕРЕКЛЮЧЕНИЕ РЕЖИМА МЫШЛЕНИЯ НА ОСНОВЕ ЭНЕРГИИ ---
        self._update_thinking_mode()
        
        # --- АКТИВАЦИЯ ДЛИННЫХ ЦЕПОЧЕК ПРИ КОНЦЕНТРАЦИИ ---
        # Чем выше энергия, тем длиннее цепочки и глубже мышление
        energy_ratio = self.network_energy / self.max_energy
        if energy_ratio >= 0.7:
            self.chain_length_multiplier = 2.0 + (energy_ratio - 0.7) * 3.0  # 2.0-3.5
            self.recurrent_gain = 1.5 + (energy_ratio - 0.7) * 2.0  # 1.5-2.5
            self.depth_penetration = 2.0 + (energy_ratio - 0.7) * 3.0  # 2.0-3.5
            self.focus_level = (energy_ratio - 0.7) / 0.3  # 0-1
        elif energy_ratio >= 0.4:
            self.chain_length_multiplier = 1.0 + (energy_ratio - 0.4) * 3.0  # 1.0-2.0
            self.recurrent_gain = 1.0 + (energy_ratio - 0.4) * 2.0  # 1.0-1.5
            self.depth_penetration = 1.0 + (energy_ratio - 0.4) * 3.0  # 1.0-2.0
            self.focus_level = (energy_ratio - 0.4) / 0.3  # 0-1
        else:
            self.chain_length_multiplier = 0.5 + energy_ratio * 1.25  # 0.5-1.0
            self.recurrent_gain = 0.5 + energy_ratio * 1.25  # 0.5-1.0
            self.depth_penetration = 0.5 + energy_ratio * 1.25  # 0.5-1.0
            self.focus_level = 0.0
        
        # Ограничения
        self.chain_length_multiplier = max(0.3, min(4.0, self.chain_length_multiplier))
        self.recurrent_gain = max(0.3, min(3.0, self.recurrent_gain))
        self.depth_penetration = max(0.3, min(4.0, self.depth_penetration))
        
        # Входной сигнал — приводим к размеру сети
        n_neurons = len(self.neurons)
        if len(input_signal) > n_neurons:
            input_signal = input_signal[:n_neurons]
        elif len(input_signal) < n_neurons:
            input_signal = input_signal + [0.0] * (n_neurons - len(input_signal))
        
        # Обновление систем
        self._update_systems(input_signal)
        
        # --- РЕКУРРЕНТНЫЕ СИГНАЛЫ ---
        # Добавляем сигналы из рекуррентного буфера (задержанные)
        recurrent_input = np.zeros(n_neurons)
        if self.recurrent_buffer:
            delayed = self.recurrent_buffer.pop(0)
            for idx, val in delayed.items():
                if idx < n_neurons:
                    recurrent_input[idx] += val * self.recurrent_strength
        
        # Внимание (салиенция)
        salience = self.fear * 0.4 + self.curiosity * 0.4 + self.surprise * 0.2
        
        # Шаг нейронов
        spikes = []
        firing_rates = []
        for i, neuron in enumerate(self.neurons):
            I_ext = input_signal[i] * (1.0 + salience * 0.5)
            I_ext += recurrent_input[i]  # Рекуррентный сигнал
            # Добавляем шум
            I_ext += random.uniform(-0.1, 0.1)
            
            spiked = neuron.step(I_ext, dt=0.5, time_step=self.step_count)
            if spiked:
                spikes.append(i)
                firing_rates.append(1.0)
            else:
                firing_rates.append(0.0)
        
        # --- WINNER-TAKE-ALL (латеральное торможение) ---
        if self.wta_active and spikes:
            # Находим самый активный кластер
            spike_indices = np.array(spikes)
            if len(spike_indices) > 0:
                # Группируем спайки по кластерам
                cluster_size = max(1, n_neurons // self.wta_clusters)
                cluster_spike_counts = {}
                for idx in spike_indices:
                    cluster = idx // cluster_size
                    cluster_spike_counts[cluster] = cluster_spike_counts.get(cluster, 0) + 1
                
                # Выбираем кластер-победитель
                if cluster_spike_counts:
                    winner_cluster = max(cluster_spike_counts, key=cluster_spike_counts.get)
                    # Тормозим нейроны вне кластера-победителя
                    for i in range(n_neurons):
                        cluster = i // cluster_size
                        if cluster != winner_cluster:
                            # Снижаем активность нейронов вне победителя
                            self.neurons[i].v -= self.lateral_inhibition_strength * 2.0
        
        # Обновление синапсов по STDP
        for i in spikes:
            for target_idx, weight in self.neurons[i].synapses:
                if target_idx < len(self.neurons):
                    post_spike = self.neurons[target_idx].last_spike_time
                    if post_spike > 0:
                        self.neurons[i].update_synapses(
                            self.neurons[i].last_spike_time,
                            post_spike,
                            0.5
                        )
        
        # --- РЕКУРРЕНТНЫЙ БУФЕР ---
        # Сохраняем текущие спайки в буфер с задержкой
        if spikes:
            spike_dict = {i: 1.0 for i in spikes}
            self.recurrent_buffer.append(spike_dict)
            if len(self.recurrent_buffer) > self.recurrent_delay:
                self.recurrent_buffer.pop(0)
        
        # Добавляем спайки в рабочую память
        if spikes:
            self.working_memory.append(spikes)
            # Сохраняем паттерн для консолидации
            if len(self.consolidation_buffer) < 100:
                self.consolidation_buffer.append(spikes)
        
        # Обновление предсказания
        self._update_prediction(input_signal)
        
        # Обновление метакогниции
        self._update_metacognition()
        
        # Нейрогенез
        if self.step_count % 10 == 0:
            self._neurogenesis()
        
        # Проверка выживаемости новых нейронов
        if self.step_count % 5 == 0 and self.new_neurons:
            self._check_survival()
        
        # Гомеостатическое масштабирование
        if self.step_count % self.homeostatic_interval == 0:
            self._homeostatic_scaling()
        
        # Метапластичность (каждые 30 шагов)
        if self.step_count % 30 == 0:
            self._metaplasticity()
        
        # Обрезка
        if self.step_count % self.pruning_interval == 0:
            self._prune_synapses()
            self._prune_chains()
        
        # Сжатие памяти
        if self.step_count % 20 == 0 and len(self.working_memory) > 5:
            self._compact_memory()
        
        return {
            'spikes': spikes,
            'spike_count': len(spikes),
            'energy': self.network_energy,
            'fear': self.fear,
            'curiosity': self.curiosity,
            'surprise': self.surprise,
            'salience': salience,
            'awareness': self.awareness_level,
            'neurons': len(self.neurons)
        }
    
    def _update_energy(self):
        """Обновляет энергию сети"""
        # Расход энергии от активности (уменьшен)
        active_count = sum(1 for n in self.neurons if n.spiked)
        depletion = self.energy_depletion * (0.5 + active_count / len(self.neurons) * 0.5)
        
        self.network_energy -= depletion
        
        # Восстановление (увеличено)
        if not self.crisis_mode:
            self.network_energy += self.energy_recovery * 1.5
        else:
            # В кризисе восстановление замедлено
            self.network_energy += self.energy_recovery * 0.5
        
        self.network_energy = max(0.0, min(self.max_energy, self.network_energy))
        
        # Проверка кризиса
        if self.network_energy < self.max_energy * self.crisis_threshold:
            self.crisis_mode = True
            self._existential_crisis()
        elif self.network_energy > self.max_energy * 0.6:
            self.crisis_mode = False
    
    def _update_systems(self, input_signal):
        """Обновляет страх, любопытство и удивление"""
        # Страх зависит от энергии
        energy_ratio = self.network_energy / self.max_energy
        self.fear = max(0.0, min(1.0, 1.0 - energy_ratio + 0.2 * self.surprise))
        
        # Любопытство зависит от новизны
        novelty = self._compute_novelty(input_signal)
        self.curiosity = min(1.0, 0.3 + 0.4 * novelty + 0.2 * (1.0 - self.fear))
        
        # Удивление от неожиданных паттернов
        if len(self.prediction_memory) > 5:
            last_pred = self.prediction_memory[-1]
            if len(last_pred) == len(input_signal):
                error = np.mean((np.array(last_pred) - np.array(input_signal))**2)
                self.surprise = min(1.0, error * 2.0)
            else:
                self.surprise = 0.2
        else:
            self.surprise = 0.2
        
        # Если удивление превышает порог, генерируем инсайт
        if self.surprise > self.surprise_threshold:
            self.insight_count += 1
            self.insight_buffer.append(f"Инсайт {self.insight_count}: удивление {self.surprise:.2f}")
            self.reflection_depth += 0.05
        
        # --- ЭМОЦИОНАЛЬНЫЙ КАСКАД ---
        self._update_cascade()
    
    def _update_cascade(self):
        """Управляет эмоциональным каскадом: амбиции → фокус → фрустрация → анализ → переоценка → разрешение"""
        # Запуск каскада при сильном удивлении
        if not self.cascade_active and self.surprise > 0.4:
            self.cascade_active = True
            self.cascade_stage = 0
            self.cascade_timer = 0
            self.insight_buffer.append("🌀 Каскад начат: амбиции")
        
        if self.cascade_active:
            self.cascade_timer += 1
            
            # Переход к следующей стадии по таймеру
            if self.cascade_timer >= self.cascade_duration:
                self.cascade_timer = 0
                self.cascade_stage += 1
                
                if self.cascade_stage >= 6:
                    self.cascade_active = False
                    self.cascade_stage = 0
                    self.insight_buffer.append("✅ Каскад завершён")
                    return
                
                # Логируем переход
                stage_names = ["амбиции", "фокус", "фрустрация", "анализ", "переоценка", "разрешение"]
                self.insight_buffer.append(f"➡️ Каскад: {stage_names[self.cascade_stage]}")
            
            # Принудительная установка эмоций в зависимости от стадии
            if self.cascade_stage == 0:   # амбиции → высокий дофамин
                self.dopamine = 0.8
                self.fear = 0.1
                self.current_dominant_emotion = 'curiosity'
            elif self.cascade_stage == 1: # фокус → высокий норадреналин
                self.norepinephrine = 0.7
                self.current_dominant_emotion = 'curiosity'
            elif self.cascade_stage == 2: # фрустрация → высокий кортизол (стресс)
                self.fear = 0.5
                self.surprise = 0.3
                self.current_dominant_emotion = 'fear'
            elif self.cascade_stage == 3: # анализ → высокий ацетилхолин
                self.acetylcholine = 0.7
                self.current_dominant_emotion = 'surprise'
            elif self.cascade_stage == 4: # переоценка → высокий серотонин
                self.serotonin = 0.8
                self.current_dominant_emotion = 'pleasure'
            elif self.cascade_stage == 5: # разрешение → баланс
                self.dopamine = 0.5
                self.serotonin = 0.5
                self.norepinephrine = 0.3
                self.acetylcholine = 0.3
                self.current_dominant_emotion = 'pleasure'
    
    def _metaplasticity(self):
        """Адаптирует скорость обучения в зависимости от истории изменений"""
        # Для каждого нейрона отслеживаем среднее изменение весов
        for neuron in self.neurons:
            # Убеждаемся, что атрибут существует
            if not hasattr(neuron, 'weight_change_history'):
                neuron.weight_change_history = deque(maxlen=20)
            
            # Если есть история изменений, корректируем скорость обучения
            if len(neuron.weight_change_history) >= 10:
                avg_change = np.mean(neuron.weight_change_history)
                if avg_change > 0.1:
                    # Увеличиваем пластичность
                    neuron.A_plus = min(0.2, neuron.A_plus * 1.05)
                    neuron.A_minus = min(0.2, neuron.A_minus * 1.05)
                elif avg_change < 0.02:
                    # Уменьшаем пластичность (ригидность)
                    neuron.A_plus = max(0.05, neuron.A_plus * 0.95)
                    neuron.A_minus = max(0.05, neuron.A_minus * 0.95)
    
    def _compute_novelty(self, input_signal):
        """Вычисляет новизну входного сигнала"""
        if len(self.working_memory) < 2:
            return 0.5
        
        # Сравниваем с последними паттернами
        similarities = []
        for mem in list(self.working_memory)[-5:]:
            # Преобразуем mem в список, если это массив или список
            if hasattr(mem, 'tolist'):
                mem = mem.tolist()
            elif not isinstance(mem, list):
                mem = list(mem)
            
            # Приводим к одинаковой длине
            if len(mem) != len(input_signal):
                # Если разная длина, используем минимальную
                min_len = min(len(mem), len(input_signal))
                mem = mem[:min_len]
                input_signal_trimmed = input_signal[:min_len]
            else:
                input_signal_trimmed = input_signal
            
            try:
                sim = np.corrcoef(mem, input_signal_trimmed)[0, 1]
                if not np.isnan(sim):
                    similarities.append(sim)
            except:
                continue
        
        if similarities:
            avg_sim = np.mean(similarities)
            return 1.0 - avg_sim
        return 0.5
    
    def _existential_crisis(self):
        """Режим экзистенциального кризиса"""
        self.birth_rate = 1.0  # Ускоренный нейрогенез
        self.fear = min(1.0, self.fear + 0.1)
        self.curiosity = min(1.0, self.curiosity + 0.1)
        self.surprise = min(1.0, self.surprise + 0.1)
        
        # Генерируем инсайт о кризисе
        if self.step_count % 5 == 0:
            self.insight_count += 1
            self.insight_buffer.append(f"КРИЗИС: энергия {self.network_energy:.1f}")
    
    def _neurogenesis(self):
        """Создаёт новые нейроны с механизмом выживаемости"""
        if len(self.neurons) >= self.max_neurons:
            return
        
        # Вероятность рождения зависит от удивления и кризиса
        birth_prob = self.birth_rate * (1.0 + 2.0 * self.surprise)
        if self.crisis_mode:
            birth_prob *= 2.0
        
        if random.random() < birth_prob:
            # Создаём новый нейрон
            types = ['regular', 'fast', 'burst', 'chattering', 'low_threshold', 'high_threshold']
            neuron_type = random.choice(types)
            new_neuron = RealNeuron(neuron_type, len(self.neurons))
            
            # Добавляем случайные связи (изначально слабые)
            for i in range(len(self.neurons)):
                if random.random() < self.connectivity * 0.5:
                    weight = random.uniform(0.05, 0.2)  # Слабые связи
                    new_neuron.add_synapse(i, weight)
                if random.random() < self.connectivity * 0.5:
                    weight = random.uniform(0.05, 0.2)
                    self.neurons[i].add_synapse(len(self.neurons), weight)
            
            # Добавляем нейрон в сеть и в список новых нейронов для проверки выживаемости
            new_idx = len(self.neurons)
            self.neurons.append(new_neuron)
            self.new_neurons[new_idx] = (new_neuron, 0)  # (нейрон, шаги_выживания)
            
            if len(self.neurons) % 20 == 0:
                self.insight_buffer.append(f"Нейрогенез: {len(self.neurons)} нейронов")
    
    def _check_survival(self):
        """Проверяет выживаемость новых нейронов"""
        to_remove = []
        for idx, (neuron, steps) in self.new_neurons.items():
            if steps >= self.survival_window:
                # Проверяем, интегрировался ли нейрон
                if len(neuron.synapses) < 3 or neuron.last_spike_time < self.step_count - 20:
                    # Нейрон не выжил — удаляем
                    to_remove.append(idx)
                else:
                    # Нейрон выжил — убираем из списка новых
                    self.new_neurons.pop(idx, None)
            else:
                # Увеличиваем счётчик шагов
                self.new_neurons[idx] = (neuron, steps + 1)
        
        # Удаляем не выжившие нейроны
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.neurons):
                # Удаляем синапсы, указывающие на этот нейрон
                for neuron in self.neurons:
                    neuron.synapses = [(t, w) for t, w in neuron.synapses if t != idx]
                # Удаляем нейрон
                self.neurons.pop(idx)
                self.new_neurons.pop(idx, None)
                
                # Переиндексация
                for i, neuron in enumerate(self.neurons):
                    neuron.idx = i
                self.insight_buffer.append(f"Нейрон {idx} не выжил")
    
    def _homeostatic_scaling(self):
        """Масштабирует синапсы нейронов для поддержания баланса"""
        # Собираем статистику активности
        for i, neuron in enumerate(self.neurons):
            if i not in self.activity_history:
                self.activity_history[i] = []
            # Запоминаем, был ли спайк
            self.activity_history[i].append(1 if neuron.spiked else 0)
            # Ограничиваем историю
            if len(self.activity_history[i]) > 20:
                self.activity_history[i] = self.activity_history[i][-20:]
        
        # Масштабирование для каждого нейрона
        for i, neuron in enumerate(self.neurons):
            history = self.activity_history.get(i, [])
            if len(history) >= 10:
                activity = np.mean(history)
                # Если активность слишком высокая — ослабляем синапсы
                if activity > self.target_activity * 1.2:
                    scaling = self.scaling_factor
                # Если активность слишком низкая — усиливаем синапсы
                elif activity < self.target_activity * 0.8 and activity > 0.01:
                    scaling = 1.0 / self.scaling_factor
                else:
                    scaling = 1.0
                
                # Применяем масштабирование ко всем синапсам
                if scaling != 1.0:
                    new_synapses = []
                    for target_idx, weight in neuron.synapses:
                        new_weight = weight * scaling
                        new_weight = max(0.01, min(2.0, new_weight))
                        new_synapses.append((target_idx, new_weight))
                    neuron.synapses = new_synapses
    
    def _prune_synapses(self):
        """Удаляет слабые синапсы"""
        pruned = 0
        for neuron in self.neurons:
            new_synapses = []
            for target_idx, weight in neuron.synapses:
                if weight > self.pruning_threshold:
                    new_synapses.append((target_idx, weight))
                else:
                    pruned += 1
            neuron.synapses = new_synapses
        
        if pruned > 0 and self.step_count % 100 == 0:
            self.insight_buffer.append(f"Обрезано {pruned} синапсов")
    
    def _prune_chains(self):
        """Удаляет слабые цепочки нейронов"""
        # Удаляем нейроны с малым количеством синапсов
        new_neurons = []
        removed = 0
        for neuron in self.neurons:
            if len(neuron.synapses) >= 2:
                new_neurons.append(neuron)
            else:
                removed += 1
        
        if removed > 0 and len(new_neurons) >= 10:
            self.neurons = new_neurons
            # Переиндексация
            for i, neuron in enumerate(self.neurons):
                neuron.idx = i
    
    def _compact_memory(self):
        """Сжимает рабочую память в долгосрочную"""
        if len(self.working_memory) > 5:
            # Берём средний паттерн
            patterns = list(self.working_memory)
            if patterns and len(patterns[0]) > 0:
                # Усредняем спайки
                avg_pattern = []
                max_len = max(len(p) for p in patterns)
                for i in range(max_len):
                    vals = [p[i] if i < len(p) else 0 for p in patterns]
                    avg_pattern.append(np.mean(vals))
                
                self.long_term_memory.append(avg_pattern)
                
                # Ограничиваем долгосрочную память
                if len(self.long_term_memory) > 100:
                    self.long_term_memory = self.long_term_memory[-100:]
    
    def _update_prediction(self, input_signal):
        """Обновляет предсказание"""
        self.prediction_memory.append(input_signal)
        if len(self.prediction_memory) > self.prediction_window:
            self.prediction_memory = self.prediction_memory[-self.prediction_window:]
    
    def _update_metacognition(self):
        """Обновляет метакогнитивные показатели"""
        # Осознанность растёт с инсайтами
        self.awareness_level = min(1.0, 0.1 + 0.05 * self.insight_count / max(1, self.step_count / 100))
        
        # Уверенность зависит от стабильности
        if len(self.prediction_memory) > 5:
            self.self_confidence = min(1.0, 0.3 + 0.5 * (1.0 - self.surprise))
        
        # Точность предсказаний
        if len(self.prediction_memory) > 2:
            errors = []
            for i in range(1, min(5, len(self.prediction_memory))):
                p1 = self.prediction_memory[-i]
                p2 = self.prediction_memory[-i-1]
                if len(p1) == len(p2):
                    err = np.mean((np.array(p1) - np.array(p2))**2)
                    errors.append(err)
            if errors:
                avg_error = np.mean(errors)
                self.prediction_accuracy = min(1.0, max(0.0, 1.0 - avg_error))
    
    def predict_next(self, current):
        """Предсказывает следующий шаг"""
        if len(self.prediction_memory) < 3:
            return current
        
        # Простая экстраполяция
        if len(self.prediction_memory) >= 3:
            p1 = np.array(self.prediction_memory[-1])
            p2 = np.array(self.prediction_memory[-2])
            if len(p1) == len(p2) == len(current):
                diff = p1 - p2
                predicted = p1 + diff * 0.5
                return predicted.tolist()
        return current
    
    def add_skill(self, name, sequence):
        """Добавляет навык"""
        self.skills[name] = (sequence, 0.5)
        self.skill_strength[name] = 0.5
        
        # Подкрепляем через сеть
        for step in sequence:
            if isinstance(step, (int, float)):
                self.step([step] * len(self.neurons))
    
    def recall_skill(self, name):
        """Воспроизводит навык"""
        if name in self.skills:
            sequence, strength = self.skills[name]
            # Усиливаем навык
            new_strength = min(1.0, strength + 0.1)
            self.skills[name] = (sequence, new_strength)
            self.skill_strength[name] = new_strength
            return sequence
        return None
    
    def get_energy_for_chains(self):
        """Определяет, на какие цепочки хватает энергии"""
        energy_per_chain = 10.0
        max_chains = int(self.network_energy / energy_per_chain)
        
        # Вычисляем цепочки (активные нейроны)
        active_chains = 0
        for neuron in self.neurons:
            if neuron.spiked and len(neuron.synapses) > 0:
                active_chains += 1
        
        return {
            'energy': self.network_energy,
            'max_chains': max_chains,
            'active_chains': active_chains,
            'can_run_more': active_chains < max_chains
        }
    
    def _process_sleep(self):
        """
        Обрабатывает фазу сна — консолидация памяти.
        
        Во время сна происходит:
        1. Реплей паттернов из буфера консолидации (гиппокамп → кора)
        2. Ослабление синапсов (downscaling) для предотвращения переобучения
        3. Укрепление повторяющихся паттернов (формирование долговременной памяти)
        4. Обновление метакогниции (рефлексия о том, что было усвоено)
        """
        self.sleep_duration -= 1
        
        # --- РЕПЛЕЙ ПАТТЕРНОВ ---
        if self.consolidation_buffer and self.sleep_duration % 3 == 0:
            # Выбираем случайный паттерн из буфера
            pattern = random.choice(self.consolidation_buffer)
            if pattern:
                # Реплей: активируем нейроны в том же паттерне
                for idx in pattern:
                    if idx < len(self.neurons):
                        self.neurons[idx].fire(dt=0.5, time_step=self.step_count)
                
                # Укрепляем синапсы, связанные с этим паттерном
                for idx in pattern:
                    if idx < len(self.neurons):
                        for target_idx, weight in self.neurons[idx].synapses:
                            if target_idx < len(self.neurons) and target_idx in pattern:
                                new_weight = weight * 1.02
                                new_weight = min(2.0, new_weight)
                                for i, (t, w) in enumerate(self.neurons[idx].synapses):
                                    if t == target_idx:
                                        self.neurons[idx].synapses[i] = (t, new_weight)
                                        break
        
        # --- ОСЛАБЛЕНИЕ СИНАПСОВ (downscaling) ---
        if self.sleep_duration % 5 == 0:
            for neuron in self.neurons:
                new_synapses = []
                for target_idx, weight in neuron.synapses:
                    new_weight = weight * 0.98
                    if new_weight > 0.01:
                        new_synapses.append((target_idx, new_weight))
                neuron.synapses = new_synapses
        
        # --- УКРЕПЛЕНИЕ ПОВТОРЯЮЩИХСЯ ПАТТЕРНОВ ---
        if self.consolidation_buffer and len(self.consolidation_buffer) > 5:
            from collections import Counter
            pattern_strings = [str(p) for p in self.consolidation_buffer]
            counter = Counter(pattern_strings)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] > 3:
                try:
                    pattern = eval(most_common[0][0])
                    for idx in pattern:
                        if idx < len(self.neurons):
                            for target_idx, weight in self.neurons[idx].synapses:
                                if target_idx < len(self.neurons) and target_idx in pattern:
                                    new_weight = weight * 1.05
                                    new_weight = min(2.0, new_weight)
                                    for i, (t, w) in enumerate(self.neurons[idx].synapses):
                                        if t == target_idx:
                                            self.neurons[idx].synapses[i] = (t, new_weight)
                                            break
                except:
                    pass
        
        # --- ОБНОВЛЕНИЕ МЕТАКОГНИЦИИ ---
        if self.sleep_duration % 5 == 0:
            self.reflection_depth = min(1.0, self.reflection_depth + 0.02)
            self.awareness_level = min(1.0, self.awareness_level + 0.01)
            
            if random.random() < 0.05:
                self.insight_count += 1
                self.insight_buffer.append(f"Сон {self.step_count}: консолидация {len(self.consolidation_buffer)} паттернов")
        
        # --- ЗАВЕРШЕНИЕ СНА ---
        if self.sleep_duration <= 0:
            self.is_sleeping = False
            self.consolidation_buffer = []
            self.sleep_counter = 0
            print(f"💤 Сон закончился на шаге {self.step_count}")
            print(f"   Инсайтов: {self.insight_count}, Осознанность: {self.awareness_level:.2f}")
            return {'spikes': [], 'spike_count': 0, 'energy': self.network_energy,
                    'fear': self.fear, 'curiosity': self.curiosity, 'surprise': self.surprise,
                    'salience': 0, 'awareness': self.awareness_level, 'neurons': len(self.neurons),
                    'sleep_consolidated': True, 'insights': self.insight_count}
        
        return {'spikes': [], 'spike_count': 0, 'energy': self.network_energy,
                'fear': self.fear, 'curiosity': self.curiosity, 'surprise': self.surprise,
                'salience': 0, 'awareness': self.awareness_level, 'neurons': len(self.neurons),
                'sleep_consolidated': False}
    
    def sleep_consolidation(self):
        """Консолидация памяти во время сна с реплеем (устаревший метод, используйте _process_sleep)"""
        # --- 1. РЕПЛЕЙ (воспроизведение в обратном порядке) ---
        replay_count = 0
        if len(self.long_term_memory) > 5:
            patterns_to_replay = self.long_term_memory[-10:]
            for pattern in reversed(patterns_to_replay):
                if len(pattern) == len(self.neurons):
                    self.step(pattern)
                    replay_count += 1
        
        # --- 2. УСИЛЕНИЕ НАВЫКОВ ---
        for name in self.skills:
            sequence, strength = self.skills[name]
            new_strength = min(1.0, strength + 0.05)
            self.skills[name] = (sequence, new_strength)
            self.skill_strength[name] = new_strength
        
        # --- 3. ВОССТАНОВЛЕНИЕ ЭНЕРГИИ ---
        self.network_energy = min(self.max_energy, self.network_energy + 20.0)
        self.fear = max(0.0, self.fear - 0.1)
        
        # --- 4. ГЕНЕРАЦИЯ ИНСАЙТА О СНЕ ---
        if replay_count > 0:
            self.insight_count += 1
            self.insight_buffer.append(f"Сон: реплей {replay_count} паттернов")
        
        return {
            'consolidated': len(self.long_term_memory),
            'replay_count': replay_count,
            'energy': self.network_energy,
            'skills': len(self.skills)
        }
    
    def attention(self, signal):
        """Салиенция: усиление сигнала на основе страха, любопытства и удивления"""
        salience = self.fear * 0.4 + self.curiosity * 0.4 + self.surprise * 0.2
        amplified = signal * (1.0 + salience * 0.5)
        return amplified
    
    def get_metacognition(self):
        """Возвращает метакогнитивные показатели"""
        return {
            'awareness_level': self.awareness_level,
            'self_confidence': self.self_confidence,
            'prediction_accuracy': self.prediction_accuracy,
            'insight_count': self.insight_count,
            'reflection_depth': self.reflection_depth,
            'insight_buffer': list(self.insight_buffer)
        }
    
    def get_stats(self):
        """Возвращает статистику сети"""
        total_synapses = sum(len(n.synapses) for n in self.neurons)
        return {
            'neurons': len(self.neurons),
            'synapses': total_synapses,
            'energy': self.network_energy,
            'fear': self.fear,
            'curiosity': self.curiosity,
            'surprise': self.surprise,
            'crisis': self.crisis_mode,
            'skills': len(self.skills),
            'patterns': len(self.long_term_memory),
            'awareness': self.awareness_level,
            'insights': self.insight_count
        }
    
    def _count_synapses(self):
        """Подсчёт общего числа синапсов"""
        return sum(len(n.synapses) for n in self.neurons)
