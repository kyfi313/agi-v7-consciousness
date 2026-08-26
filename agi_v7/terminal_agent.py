# -*- coding: utf-8 -*-
"""
Терминальный агент для визуализации.
Наследует простую версию CognitiveOrchestrator.
Использует существующие механизмы мозга для принятия решений.
Никаких новых механизмов обучения — только интерпретация.
"""

from agi_v7.orchestrator import CognitiveOrchestrator
from agi_v7.consciousness import ConsciousnessModule
from agi_v7.consciousness_dispatcher import ConsciousnessDispatcher
from agi_v7.relay_evolution import RelayEvolution
from typing import Dict, Any, Tuple
import random


class TerminalAgent(CognitiveOrchestrator):
    """
    Агент с интерфейсом для терминальной среды.
    Интерпретирует активность мозга (спайки, страх, любопытство, удивление)
    в действия и мысли.
    """

    def __init__(self, num_neurons: int = 80, connectivity: float = 0.06, input_dim: int = 7):
        super().__init__(num_neurons=num_neurons, connectivity=connectivity, input_dim=input_dim)
        self.action_history = []
        self.thoughts = []
        
        # --- ВНУТРЕННИЕ ПОТРЕБНОСТИ (гомеостатическая мотивация) ---
        self.hunger = 0.0          # 0.0 - сыт, 1.0 - голоден
        self.fatigue = 0.0         # 0.0 - отдохнул, 1.0 - устал
        self.pain = 0.0            # 0.0 - нет боли, 1.0 - сильная боль
        self.thirst = 0.0          # 0.0 - не хочет пить, 1.0 - хочет пить
        self.hunger_rate = 0.012   # Скорость роста голода
        self.fatigue_rate = 0.008  # Скорость роста усталости
        self.pain_decay = 0.01     # Скорость заживления
        self.thirst_rate = 0.005   # Скорость роста жажды
        
        # --- СОЗНАНИЕ ---
        self.consciousness = ConsciousnessModule()
        
        # --- ДИСПЕТЧЕР СОЗНАНИЯ ---
        self.dispatcher = ConsciousnessDispatcher()
        
        # Регистрируем модули в диспетчере
        self.dispatcher.register_module('brain')
        self.dispatcher.register_module('perception')
        self.dispatcher.register_module('memory')
        self.dispatcher.register_module('emotion')
        self.dispatcher.register_module('attention')
        self.dispatcher.register_module('prediction')
        self.dispatcher.register_module('self_model')
        self.dispatcher.register_module('habits')
        
        # Создаём маршруты между модулями
        routes = [
            ('perception', 'attention'),
            ('attention', 'brain'),
            ('brain', 'memory'),
            ('brain', 'emotion'),
            ('emotion', 'self_model'),
            ('memory', 'prediction'),
            ('prediction', 'brain'),
            ('self_model', 'brain'),
        ]
        for source, target in routes:
            self.dispatcher.add_route(source, target, weight=0.5)
        
        # --- МОТОРНАЯ КОРА (последние 4 нейрона) ---
        self.motor_neurons = list(range(num_neurons - 4, num_neurons))
        
        # Паттерны действий (эталоны, к которым мозг стремится)
        self.action_patterns = {
            'explore':  [0, 0, 0, 1],
            'collect':  [0, 0, 1, 0],
            'flee':     [0, 1, 0, 0],
            'rest':     [1, 0, 0, 0],
            'interact': [1, 1, 0, 0],
        }
        
        # --- ОБУЧЕНИЕ С ПОДКРЕПЛЕНИЕМ ---
        self.reward_history = []
        self.selected_action = None
        
        # Q-значения для каждого действия (средняя ожидаемая награда)
        self.action_values = {action: 0.0 for action in self.action_patterns.keys()}
        self.epsilon = 0.2  # Вероятность исследования (снижена для более целенаправленного поведения)
        self.alpha = 0.15   # Скорость обучения (увеличена для более быстрой адаптации)
        self.steps_count = 0  # Счётчик шагов для управления исследованием

    def learn_from_outcome(self, reward: float, action: str):
        """
        Получает награду за выполненное действие и обновляет внутреннее состояние.
        
        Args:
            reward: Числовая награда (положительная или отрицательная)
            action: Выполненное действие
        """
        self.reward_history.append((action, reward))
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        # --- ВНУТРЕННЯЯ ГОМЕОСТАТИЧЕСКАЯ НАГРАДА ---
        homeostatic_reward = 0.0
        
        # Действия влияют на внутренние потребности
        if action == 'eat':
            self.hunger = max(0.0, self.hunger - 0.3)
            homeostatic_reward += 0.25
        elif action == 'collect':
            self.hunger = max(0.0, self.hunger - 0.15)
            homeostatic_reward += 0.15
        elif action == 'rest':
            self.fatigue = max(0.0, self.fatigue - 0.3)
            self.pain = max(0.0, self.pain - 0.1)
            homeostatic_reward += 0.2
        elif action == 'flee':
            self.fatigue = min(1.0, self.fatigue + 0.05)
            self.pain = min(1.0, self.pain + 0.02)
            homeostatic_reward -= 0.05
        elif action == 'explore':
            self.fatigue = min(1.0, self.fatigue + 0.03)
            self.thirst = min(1.0, self.thirst + 0.02)
            homeostatic_reward += 0.05
        elif action == 'interact':
            self.hunger = min(1.0, self.hunger + 0.02)
            self.fatigue = min(1.0, self.fatigue + 0.02)
            homeostatic_reward -= 0.02
        
        # Награда за поддержание баланса
        balance_penalty = (self.hunger ** 2 + self.fatigue ** 2 + self.pain ** 2) * 0.1
        homeostatic_reward -= balance_penalty
        
        # Общая награда
        total_reward = reward + 0.3 * homeostatic_reward
        
        # 1. Обновляем Q-значение для выполненного действия
        old_value = self.action_values.get(action, 0.0)
        self.action_values[action] = old_value + self.alpha * (total_reward - old_value)
        
        # 2. Модулируем страх и любопытство на основе награды
        if total_reward > 0:
            # Положительная награда → снижаем страх, повышаем любопытство
            self.brain.fear = max(0.0, self.brain.fear - total_reward * 0.1)
            self.brain.curiosity = min(1.0, self.brain.curiosity + total_reward * 0.05)
        elif total_reward < 0:
            # Отрицательная награда → повышаем страх, снижаем любопытство
            self.brain.fear = min(1.0, self.brain.fear + abs(total_reward) * 0.1)
            self.brain.curiosity = max(0.0, self.brain.curiosity - abs(total_reward) * 0.05)
        
        # 3. Связь с энергией мозга
        if action in ['eat', 'collect']:
            self.brain.network_energy = min(100.0, self.brain.network_energy + 15.0)
        elif action == 'rest':
            self.brain.network_energy = min(100.0, self.brain.network_energy + 25.0)
        elif action == 'flee':
            self.brain.network_energy = max(0.0, self.brain.network_energy - 5.0)
        
        # Автоматическое восстановление энергии (базовый метаболизм)
        if self.brain.network_energy < 50.0 and action in ['rest', 'eat', 'collect']:
            self.brain.network_energy = min(100.0, self.brain.network_energy + 2.0)

    def decide(self, perception: Dict[str, Any]) -> Tuple[str, str]:
        """
        Принимает решение на основе активности мозга.
        
        1. Формирует входной сигнал из восприятия
        2. Запускает мозг (self.step) — это вызывает brain_module
        3. Интерпретирует спайки и эмоциональное состояние
        4. Возвращает действие и мысль
        """
        # --- 1. ВОСПРИЯТИЕ → ВХОДНОЙ СИГНАЛ ---
        input_vec = [
            perception.get('energy', 100) / 100.0,
            1.0 if perception.get('food_nearby', False) else 0.0,
            1.0 if perception.get('danger_nearby', False) else 0.0,
            perception.get('min_food_dist', 10) / 10.0,
            perception.get('min_danger_dist', 10) / 10.0,
            perception.get('food_count', 0) / 10.0,
            perception.get('danger_count', 0) / 10.0,
        ]
        
        # --- ОБНОВЛЕНИЕ ВНУТРЕННИХ ПОТРЕБНОСТЕЙ ---
        self.hunger = min(1.0, self.hunger + self.hunger_rate)
        self.fatigue = min(1.0, self.fatigue + self.fatigue_rate)
        self.thirst = min(1.0, self.thirst + self.thirst_rate)
        self.pain = max(0.0, self.pain - self.pain_decay)
        
        # Голод и усталость влияют на энергию мозга
        if self.hunger > 0.7:
            self.brain.network_energy = max(0.0, self.brain.network_energy - 0.2)
        if self.fatigue > 0.7:
            self.brain.network_energy = max(0.0, self.brain.network_energy - 0.15)
        
        # --- 2. МОЗГ ДУМАЕТ ---
        # self.step() вызывает self.brain.step() — это единственный мозг
        result = self.step(input_vec)
        
        # --- 3. ИЗВЛЕКАЕМ СОСТОЯНИЕ МОЗГА ---
        spikes = result.get('spikes', [])
        fear = result.get('fear', 0.0)
        curiosity = result.get('curiosity', 0.0)
        surprise = result.get('surprise', 0.0)
        energy = result.get('energy', 100.0)
        spike_count = result.get('spike_count', 0)
        
        # 🔥 УСИЛИВАЕМ СТРАХ ЕСЛИ ОПАСНОСТЬ РЯДОМ
        if perception.get('danger_nearby', False):
            fear = max(fear, 0.6)  # Принудительно поднимаем страх
        
        # --- УПРАВЛЕНИЕ ПОВЕДЕНИЕМ НА ОСНОВЕ ВОСПРИЯТИЯ ---
        # Если рядом еда и энергии мало → собирать
        # Если рядом опасность и страх высок → убегать
        # Иначе использовать сознание
        
        food_nearby = perception.get('food_nearby', False)
        danger_nearby = perception.get('danger_nearby', False)
        energy_pct = perception.get('energy', 100) / 100.0
        min_food_dist = perception.get('min_food_dist', 10)
        
        # 🔥 ЭКСТРЕННЫЙ ПРИОРИТЕТ: выживание важнее всего
        if self.hunger > 0.6 and food_nearby:
            # Голод и есть еда → собирать
            # Отправляем сигнал в диспетчер
            self.dispatcher.route_signal('brain', {'action': 'collect', 'hunger': self.hunger}, strength=0.9)
            return 'collect', self._generate_thought('collect', fear, curiosity, surprise, energy, spike_count)
        elif danger_nearby and fear > 0.4:
            # Опасность рядом → бежать
            self.dispatcher.route_signal('emotion', {'fear': fear, 'action': 'flee'}, strength=0.8)
            return 'flee', self._generate_thought('flee', fear, curiosity, surprise, energy, spike_count)
        elif energy_pct < 0.15:
            # Критически мало энергии → отдыхать
            self.dispatcher.route_signal('brain', {'energy': energy_pct, 'action': 'rest'}, strength=0.9)
            return 'rest', self._generate_thought('rest', fear, curiosity, surprise, energy, spike_count)
        
        # --- 4. САМОМОДЕЛЬ И ПРОГНОЗИРОВАНИЕ ---
        # Используем самомодель для оценки текущего состояния
        risk_tolerance = self.self_model.get_risk_tolerance()
        exploration_bias = self.self_model.get_exploration_bias()
        
        # Используем предиктор
        try:
            state_vec = [
                energy_pct,
                self.hunger,
                self.fatigue,
                1.0 if food_nearby else 0.0,
                1.0 if danger_nearby else 0.0,
                perception.get('food_count', 0) / 10.0,
                perception.get('danger_count', 0) / 10.0
            ]
            predictions = self.predictor.predict(state_vec, horizon=5)
            uncertainty = self.predictor.get_uncertainty(state_vec, horizon=5)
        except:
            predictions = {}
            uncertainty = 0.5
        
        # --- 5. АКТИВНОЕ ВНИМАНИЕ ---
        # Фокусируемся на важных сигналах
        attention_result = self.attention.focus(
            state_vec,
            goal='survival' if energy_pct < 0.3 else 'explore',
            emotion={'fear': fear, 'curiosity': curiosity}
        )
        focus_indices = attention_result.get('focus_indices', [])
        
        # --- 6. ДИСПЕТЧЕР СОЗНАНИЯ ---
        # Отправляем все сигналы в диспетчер с приоритетами
        # Опасность → критический приоритет
        if danger_nearby:
            self.dispatcher.route_signal(
                'perception',
                {'danger': True, 'fear': fear, 'position': perception.get('position', (0,0))},
                strength=1.0,
                priority=3  # CRITICAL
            )
        
        # Голод → высокий приоритет
        if self.hunger > 0.6:
            self.dispatcher.route_signal(
                'brain',
                {'hunger': self.hunger, 'food_nearby': food_nearby},
                strength=0.9,
                priority=2  # HIGH
            )
        
        # Обычные сигналы → средний/низкий приоритет
        self.dispatcher.route_signal(
            'perception',
            perception,
            strength=0.5,
            priority=1
        )
        self.dispatcher.route_signal(
            'brain',
            {'spikes': spike_count, 'fear': fear, 'curiosity': curiosity, 'surprise': surprise},
            strength=0.6,
            priority=1
        )
        self.dispatcher.route_signal(
            'self_model',
            {'risk_tolerance': risk_tolerance, 'exploration_bias': exploration_bias, 'self_esteem': self.self_model.self_esteem},
            strength=0.5,
            priority=1
        )
        
        # Получаем отчёт о сознании
        consciousness_report = self.dispatcher.step()
        
        # --- 7. ПОЛУЧАЕМ РЕШЕНИЕ ОТ ДИСПЕТЧЕРА (СОЗНАНИЕ) ---
        # Диспетчер собирает все сигналы, сортирует по приоритету
        # и перенаправляет в высшие модули для обработки
        
        # Формируем сигналы для сознания
        signals = {
            'brain': {
                'fear': fear,
                'curiosity': curiosity,
                'surprise': surprise,
                'spikes': spikes,
                'spike_count': spike_count
            },
            'perception': perception,
            'energy': energy,
            'memory': self.action_history[-20:] if self.action_history else [],
            'action_history': self.action_history,
            'q_values': self.action_values,
            'self_model': {
                'self_esteem': self.self_model.self_esteem,
                'risk_tolerance': risk_tolerance,
                'exploration_bias': exploration_bias,
                'traits': self.self_model.traits
            },
            'predictions': predictions,
            'uncertainty': uncertainty,
            'attention': {
                'focus': focus_indices,
                'salience': attention_result.get('salience', [])
            },
            'consciousness': consciousness_report
        }
        
        # Сознание принимает решение на основе всех сигналов
        action, thought = self.consciousness.think(signals)
        
        # Прогнозируем будущее (горизонт 5 шагов)
        current_state = {
            'energy': energy_pct,
            'hunger': self.hunger,
            'fatigue': self.fatigue,
            'food_nearby': 1.0 if food_nearby else 0.0,
            'danger_nearby': 1.0 if danger_nearby else 0.0,
            'food_count': perception.get('food_count', 0) / 10.0,
            'danger_count': perception.get('danger_count', 0) / 10.0
        }
        
        # --- 5. ДИСПЕТЧЕР СОЗНАНИЯ ---
        # Отправляем все сигналы в диспетчер
        self.dispatcher.route_signal('perception', perception, strength=0.5)
        self.dispatcher.route_signal('brain', {'spikes': spike_count, 'fear': fear, 'curiosity': curiosity}, strength=0.6)
        self.dispatcher.route_signal('self_model', {'risk_tolerance': risk_tolerance, 'exploration_bias': exploration_bias}, strength=0.5)
        
        # Получаем отчёт о сознании
        consciousness_report = self.dispatcher.step()
        
        # Используем предиктор
        try:
            state_vec = [
                energy_pct,
                self.hunger,
                self.fatigue,
                1.0 if food_nearby else 0.0,
                1.0 if danger_nearby else 0.0,
                perception.get('food_count', 0) / 10.0,
                perception.get('danger_count', 0) / 10.0
            ]
            predictions = self.predictor.predict(state_vec, horizon=5)
            uncertainty = self.predictor.get_uncertainty(state_vec, horizon=5)
        except:
            predictions = {}
            uncertainty = 0.5
        
        # --- 5. АКТИВНОЕ ВНИМАНИЕ ---
        # Фокусируемся на важных сигналах
        attention_result = self.attention.focus(
            state_vec,
            goal='survival' if energy_pct < 0.3 else 'explore',
            emotion={'fear': fear, 'curiosity': curiosity}
        )
        focus_indices = attention_result.get('focus_indices', [])
        
        # --- 6. СОЗНАНИЕ ПРИНИМАЕТ РЕШЕНИЕ С УЧЁТОМ САМОМОДЕЛИ ---
        signals = {
            'brain': {
                'fear': fear,
                'curiosity': curiosity,
                'surprise': surprise,
                'spikes': spikes,
                'spike_count': spike_count
            },
            'perception': perception,
            'energy': energy,
            'memory': self.action_history[-20:] if self.action_history else [],
            'action_history': self.action_history,
            'q_values': self.action_values,
            'self_model': {
                'self_esteem': self.self_model.self_esteem,
                'risk_tolerance': risk_tolerance,
                'exploration_bias': exploration_bias,
                'traits': self.self_model.traits
            },
            'predictions': predictions,
            'uncertainty': uncertainty,
            'attention': {
                'focus': focus_indices,
                'salience': attention_result.get('salience', [])
            }
        }
        
        action, thought = self.consciousness.think(signals)
        
        self.action_history.append(action)
        self.thoughts.append(thought)
        if len(self.thoughts) > 10:
            self.thoughts.pop(0)
        
        return action, thought

    def _get_motor_spikes(self, spikes: list) -> list:
        """Извлекает спайки из моторной коры (последние 4 нейрона)."""
        motor = [0.0, 0.0, 0.0, 0.0]
        for i, neuron_idx in enumerate(self.motor_neurons):
            if neuron_idx < len(spikes) and spikes[neuron_idx] > 0:
                motor[i] = 1.0
        return motor

    def _action_from_spikes(self, motor_pattern: list) -> tuple:
        """
        Определяет действие по паттерну спайков в моторной коре.
        Возвращает (действие, уверенность).
        """
        best_action = 'explore'
        best_similarity = -1.0
        
        for action, pattern in self.action_patterns.items():
            if len(motor_pattern) == len(pattern):
                dot = sum(a * b for a, b in zip(motor_pattern, pattern))
                norm1 = sum(a * a for a in motor_pattern) ** 0.5
                norm2 = sum(b * b for b in pattern) ** 0.5
                if norm1 > 0 and norm2 > 0:
                    similarity = dot / (norm1 * norm2)
                else:
                    similarity = 0.0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_action = action
        
        # Если паттерн не распознан, используем случайное действие
        if best_similarity < 0.1:
            import random
            best_action = random.choice(list(self.action_patterns.keys()))
            best_similarity = 0.1
        
        return best_action, best_similarity

    def _interpret_spikes(self, spikes: list, fear: float, curiosity: float, 
                          energy: float, perception: Dict[str, Any]) -> str:
        """
        Интерпретирует спайки в действие, используя моторную кору и Q-learning.
        """
        # --- ПРИНУДИТЕЛЬНОЕ ПОВЕДЕНИЕ (приоритет выживания) ---
        food_nearby = perception.get('food_nearby', False)
        danger_nearby = perception.get('danger_nearby', False)
        energy_pct = perception.get('energy', 100) / 100.0
        
        if danger_nearby and fear > 0.3:
            return 'flee'
        elif food_nearby and energy_pct < 0.7:
            return 'collect'
        elif energy_pct < 0.2:
            return 'rest'
        
        # 1. Пытаемся распознать паттерн в моторной коре
        motor_pattern = self._get_motor_spikes(spikes)
        action, confidence = self._action_from_spikes(motor_pattern)
        
        # 2. Если паттерн распознан с высокой уверенностью — используем его
        if confidence > 0.5:
            return action
        
        # 3. Иначе выбираем на основе Q-значений (обучение с подкреплением)
        import random
        
        # Случайное исследование (epsilon-greedy)
        if random.random() < self.epsilon:
            # Исследование: пробуем случайное действие
            return random.choice(list(self.action_patterns.keys()))
        
        # Использование: выбираем действие с максимальной ожидаемой наградой
        best_action = max(self.action_values, key=self.action_values.get)
        
        # Но если все Q-значения равны (начальное состояние), используем внутреннюю мотивацию
        if all(v == 0.0 for v in self.action_values.values()):
            if fear > 0.6:
                return 'flee'
            elif curiosity > 0.6:
                return 'explore'
            elif energy < 30:
                return 'rest'
            else:
                return 'explore'
        
        return best_action

    def _generate_thought(self, action: str, fear: float, curiosity: float,
                          surprise: float, energy: float, spike_count: int) -> str:
        """
        Генерирует мысль на основе состояния мозга.
        Использует существующие эмоциональные показатели.
        """
        # Эмоциональная окраска из существующих показателей
        if fear > 0.7:
            emotion = "😰 страх"
        elif surprise > 0.5:
            emotion = "😲 удивление"
        elif curiosity > 0.6:
            emotion = "🤔 любопытство"
        elif energy < 30:
            emotion = "😴 усталость"
        else:
            emotion = "😌 спокойствие"
        
        # Эмодзи действия
        action_emojis = {
            'explore': '🧭',
            'collect': '🍎',
            'flee': '🏃',
            'rest': '😴',
            'interact': '🤝'
        }
        emoji = action_emojis.get(action, '❓')
        
        # Формируем мысль
        if fear > 0.7:
            return f"{emoji} {action} — {emotion}, опасность!"
        elif curiosity > 0.6:
            return f"{emoji} {action} — {emotion}, интересно!"
        elif energy < 30:
            return f"{emoji} {action} — {emotion}, силы кончаются"
        else:
            return f"{emoji} {action} — {emotion}"

    def get_latest_thought(self) -> str:
        """Возвращает последнюю мысль."""
        return self.thoughts[-1] if self.thoughts else "..."

    def get_stats(self) -> Dict[str, Any]:
        """Дополнительная статистика для отображения."""
        base_stats = super().get_stats()
        base_stats['action_history_length'] = len(self.action_history)
        return base_stats
