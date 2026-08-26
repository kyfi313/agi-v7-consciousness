# -*- coding: utf-8 -*-
"""
Модуль эстафетной эволюции.
Каждый модуль эволюционирует во время работы, передавая улучшения следующему.

Принцип:
1. Каждый модуль — это эволюционируемый агент.
2. Модули выстроены в цепочку (эстафету).
3. Каждый модуль обрабатывает данные и передаёт их дальше.
4. Каждый модуль эволюционирует на своих данных (улучшает свою конфигурацию).
5. Улучшения передаются по цепочке как эстафетная палочка.
"""

import random
from collections import deque
from typing import Dict, Any, List
from abc import ABC, abstractmethod


class EvolvableModule(ABC):
    """
    Базовый класс для эволюционируемых модулей.
    Каждый модуль наследует этот класс и реализует свои тесты.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.best_config = config or {}
        self.best_fitness = -float('inf')
        self.fitness_history = deque(maxlen=20)
        self.evolution_step = 0
        self.mutation_rate = 0.1
        self.population = [self.config.copy()]
        self.generation = 0

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Обрабатывает данные."""
        pass

    @abstractmethod
    def test(self, config: Dict[str, Any]) -> float:
        """
        Тестирует конфигурацию модуля.
        Возвращает пригодность (0.0–1.0).
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Возвращает текущую конфигурацию модуля."""
        pass

    @abstractmethod
    def set_config(self, config: Dict[str, Any]):
        """Устанавливает конфигурацию модуля."""
        pass

    def evolve(self, data: Any, output: Any) -> bool:
        """
        Эволюционирует модуль на основе входных и выходных данных.
        Возвращает True, если найдена лучшая конфигурация.
        """
        self.evolution_step += 1

        current_fitness = self.test(self.config)
        self.fitness_history.append(current_fitness)

        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_config = self.config.copy()
            return True

        if self.evolution_step % 5 == 0:
            new_config = self._mutate(self.config)
            new_fitness = self.test(new_config)

            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_config = new_config.copy()
                self.config = new_config.copy()
                return True
            else:
                self.config = self.best_config.copy()

        return False

    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Мутирует конфигурацию."""
        new_config = config.copy()

        for key, value in new_config.items():
            if random.random() < self.mutation_rate:
                if isinstance(value, (int, float)):
                    noise = random.uniform(-0.2, 0.2) * abs(value) if value != 0 else random.uniform(-0.1, 0.1)
                    new_config[key] = max(0.0, value + noise)
                elif isinstance(value, bool):
                    new_config[key] = not value
                elif isinstance(value, str):
                    if key == 'activation':
                        options = ['relu', 'sigmoid', 'tanh', 'linear']
                        new_config[key] = random.choice(options)

        return new_config

    def is_better(self) -> bool:
        """Проверяет, лучше ли текущая конфигурация, чем предыдущая лучшая."""
        current_fitness = self.test(self.config)
        return current_fitness > self.best_fitness

    def get_best_fitness(self) -> float:
        """Возвращает лучшую пригодность."""
        return self.best_fitness


class VisionModule(EvolvableModule):
    """Модуль зрения. Эволюционирует, чтобы лучше обрабатывать визуальные данные."""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'resolution': 8,
            'fovea_size': 3,
            'contrast_threshold': 0.3,
            'edge_detection': True,
        }
        super().__init__(config or default_config)
        self.memory = deque(maxlen=10)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        features = {
            'food_positions': [],
            'danger_positions': [],
            'agent_position': data.get('position', (0, 0)),
            'grid': data.get('grid', []),
            'detected_objects': [],
        }

        grid = data.get('grid', [])
        if grid:
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell == '🍎':
                        features['food_positions'].append((x, y))
                    elif cell == '⚠️':
                        features['danger_positions'].append((x, y))
                    elif cell == '🤖':
                        features['agent_position'] = (x, y)

        self.memory.append(features)
        return features

    def test(self, config: Dict[str, Any]) -> float:
        test_grid = [
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '🍎', '.'],
            ['.', '.', '🤖', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '🍎', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '⚠️', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '🍎', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '🍎', '.', '.', '⚠️'],
        ]

        data = {'grid': test_grid, 'position': (2, 2)}
        result = self.process(data)

        expected_food = {(6, 1), (3, 3), (1, 6), (4, 7)}
        expected_danger = {(5, 4), (7, 7)}

        found_food = set(result.get('food_positions', []))
        found_danger = set(result.get('danger_positions', []))

        food_accuracy = len(found_food & expected_food) / max(1, len(expected_food))
        danger_accuracy = len(found_danger & expected_danger) / max(1, len(expected_danger))

        false_food_penalty = len(found_food - expected_food) * 0.1
        false_danger_penalty = len(found_danger - expected_danger) * 0.1

        accuracy = (food_accuracy + danger_accuracy) / 2 - false_food_penalty - false_danger_penalty
        return max(0.0, min(1.0, accuracy))

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]):
        self.config = config.copy()


class ConsciousnessModuleEvolver(EvolvableModule):
    """Эволюционируемая версия модуля сознания. Оценивает важность сигналов."""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'fear_threshold': 0.4,
            'curiosity_threshold': 0.5,
            'surprise_threshold': 0.5,
            'attention_span': 5,
            'decision_temperature': 0.7,
        }
        super().__init__(config or default_config)
        self.importance_history = deque(maxlen=50)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        signals = data.get('signals', [])
        context = data.get('context', {})

        ranked_signals = []
        for signal in signals:
            importance = self._evaluate_importance(signal, context)
            ranked_signals.append({
                'signal': signal,
                'importance': importance,
                'priority': self._get_priority(importance),
            })

        ranked_signals.sort(key=lambda x: x['importance'], reverse=True)
        top_signals = ranked_signals[:int(self.config.get('attention_span', 5))]

        return {
            'ranked_signals': ranked_signals,
            'top_signals': top_signals,
            'dominant_emotion': self._get_dominant_emotion(ranked_signals),
        }

    def _evaluate_importance(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        source = signal.get('source', 'unknown')
        importance = {
            'danger': 0.9,
            'food': 0.7,
            'energy': 0.6,
            'curiosity': 0.4,
        }.get(source, 0.3)

        if context.get('energy', 1.0) < 0.3 and source == 'food':
            importance = min(1.0, importance + 0.2)
        if context.get('fear', 0) > 0.7 and source == 'danger':
            importance = min(1.0, importance + 0.15)

        return importance

    def _get_priority(self, importance: float) -> str:
        if importance > 0.8:
            return 'critical'
        elif importance > 0.5:
            return 'high'
        elif importance > 0.3:
            return 'medium'
        return 'low'

    def _get_dominant_emotion(self, ranked_signals: List[Dict]) -> str:
        if not ranked_signals:
            return 'boredom'

        source = ranked_signals[0]['signal'].get('source', 'unknown')
        if source == 'danger':
            return 'fear'
        elif source == 'food':
            return 'pleasure'
        elif source == 'curiosity':
            return 'curiosity'
        elif source == 'energy' and ranked_signals[0]['signal'].get('value', 1.0) < 0.3:
            return 'sadness'
        return 'boredom'

    def test(self, config: Dict[str, Any]) -> float:
        self.config = config
        test_signals = [
            {'source': 'danger', 'data': {'level': 0.8}},
            {'source': 'food', 'data': {'distance': 2}},
            {'source': 'curiosity', 'data': {'novelty': 0.5}},
            {'source': 'noise', 'data': {'level': 0.1}},
            {'source': 'energy', 'data': {'value': 0.2}},
        ]
        context = {'energy': 0.2, 'fear': 0.3}
        result = self.process({'signals': test_signals, 'context': context})

        ranked = result.get('ranked_signals', [])
        if len(ranked) < 3:
            return 0.0

        expected_order = ['danger', 'food', 'energy', 'curiosity', 'noise']
        actual_order = [s['signal']['source'] for s in ranked]

        correct = 0
        for i, expected in enumerate(expected_order):
            if i < len(actual_order) and actual_order[i] == expected:
                correct += 1
            else:
                break

        accuracy = correct / len(expected_order)
        if result.get('dominant_emotion') == 'fear':
            accuracy += 0.1

        return min(1.0, accuracy)

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]):
        self.config = config.copy()


class MemoryModuleEvolver(EvolvableModule):
    """Эволюционируемая версия модуля памяти. Учится сохранять и восстанавливать информацию."""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'capacity': 100,
            'compression': 0.5,
            'decay_rate': 0.01,
            'importance_threshold': 0.3,
        }
        super().__init__(config or default_config)
        self.memory = {}
        self.importance_scores = {}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action = data.get('action', 'store')
        key = data.get('key', 'default')
        value = data.get('value', {})
        importance = data.get('importance', 0.5)

        if action == 'store':
            if importance > self.config.get('importance_threshold', 0.3):
                self.memory[key] = self._compress(value)
                self.importance_scores[key] = importance

                if len(self.memory) > self.config.get('capacity', 100):
                    min_key = min(self.importance_scores, key=self.importance_scores.get)
                    del self.memory[min_key]
                    del self.importance_scores[min_key]
            return data

        elif action == 'recall':
            if key in self.memory:
                return {'value': self._decompress(self.memory[key])}
            return {'value': {}}

        return data

    def _compress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        compressed = {}
        rate = self.config.get('compression', 0.5)
        for k, v in data.items():
            if isinstance(v, (int, float)):
                compressed[k] = round(v * rate) / rate
            elif isinstance(v, list):
                compressed[k] = v[:int(len(v) * rate) + 1]
            else:
                compressed[k] = v
        return compressed

    def _decompress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def test(self, config: Dict[str, Any]) -> float:
        self.config = config
        test_data = {
            'key': 'test',
            'value': {
                'energy': 100,
                'hunger': 0.2,
                'position': [3, 5],
                'inventory': ['apple', 'berry', 'stick'],
                'history': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            'importance': 0.8,
            'action': 'store'
        }

        self.process(test_data)
        recovered = self.process({'key': 'test', 'action': 'recall'})

        if not recovered or 'value' not in recovered:
            return 0.0

        original = test_data['value']
        recovered_value = recovered.get('value', {})

        total_fields = len(original)
        correct_fields = 0

        for key, value in original.items():
            if key in recovered_value:
                if isinstance(value, (int, float)):
                    if abs(value - recovered_value[key]) / max(1, abs(value)) < 0.1:
                        correct_fields += 1
                elif isinstance(value, list):
                    if len(recovered_value[key]) > 0:
                        correct_fields += 1
                else:
                    if recovered_value[key] == value:
                        correct_fields += 1

        return correct_fields / max(1, total_fields)

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]):
        self.config = config.copy()


class PredictorModuleEvolver(EvolvableModule):
    """Эволюционируемая версия модуля предсказания. Учится точно предсказывать будущее."""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'horizon': 5,
            'learning_rate': 0.1,
            'memory_length': 10,
            'confidence_threshold': 0.6,
        }
        super().__init__(config or default_config)
        # Сохраняем значения как целые числа
        self._horizon = int(self.config.get('horizon', 5))
        self._memory_length = int(self.config.get('memory_length', 10))
        self.history = deque(maxlen=self._memory_length)
        self.prediction_errors = deque(maxlen=20)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        state = data.get('state', {})
        action = data.get('action', 'predict')

        if action == 'update':
            self.history.append(state)
            if len(self.history) > 2:
                self._learn()
            return data

        elif action == 'predict':
            return {'predictions': self._predict(state)}

        return data

    def _learn(self):
        if len(self.history) < 2:
            return
        for i in range(len(self.history) - 1):
            current = self.history[i]
            next_state = self.history[i + 1]

    def _predict(self, state: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        predictions = {}
        horizon = int(self.config.get('horizon', 5))

        for h in range(1, horizon + 1):
            pred = state.copy()
            if len(self.history) > 1:
                changes = []
                for i in range(len(self.history) - 1):
                    current = self.history[i]
                    next_val = self.history[i + 1]
                    for key in state.keys():
                        if key in current and key in next_val:
                            if isinstance(current[key], (int, float)) and isinstance(next_val[key], (int, float)):
                                changes.append(next_val[key] - current[key])

                if changes:
                    avg_change = sum(changes) / len(changes)
                    for key in state.keys():
                        if isinstance(state[key], (int, float)):
                            pred[key] = state[key] + avg_change * h

            predictions[h] = pred

        return predictions

    def test(self, config: Dict[str, Any]) -> float:
        self.config = config
        test_sequence = [
            {'energy': 100, 'hunger': 0.1, 'food': 0},
            {'energy': 95, 'hunger': 0.15, 'food': 0},
            {'energy': 90, 'hunger': 0.2, 'food': 0},
            {'energy': 85, 'hunger': 0.25, 'food': 1},
            {'energy': 80, 'hunger': 0.3, 'food': 1},
            {'energy': 75, 'hunger': 0.35, 'food': 1},
        ]

        for state in test_sequence:
            self.process({'state': state, 'action': 'update'})

        current = {'energy': 70, 'hunger': 0.4, 'food': 1}
        predictions = self._predict(current)

        if 1 not in predictions:
            return 0.0

        pred = predictions[1]
        expected = {'energy': 65, 'hunger': 0.45, 'food': 1}

        errors = []
        for key in expected.keys():
            if key in pred and isinstance(expected[key], (int, float)) and isinstance(pred[key], (int, float)):
                errors.append(abs(expected[key] - pred[key]) / max(1, abs(expected[key])))

        if not errors:
            return 0.0

        return max(0.0, 1.0 - min(1.0, sum(errors) / len(errors)))

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]):
        self.config = config.copy()


class ActionModuleEvolver(EvolvableModule):
    """Эволюционируемая версия модуля действий. Учится выбирать правильные действия."""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'epsilon': 0.2,
            'learning_rate': 0.15,
            'discount_factor': 0.9,
            'exploration_decay': 0.99,
        }
        super().__init__(config or default_config)
        self.action_values = {}
        self.action_history = deque(maxlen=20)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        state = data.get('state', {})
        available_actions = data.get('actions', ['explore', 'collect', 'flee', 'rest'])
        action = data.get('action', None)
        reward = data.get('reward', 0.0)

        if action is not None:
            self._learn(state, action, reward)
            return {'action': action, 'reward': reward}
        else:
            return {'action': self._choose_action(state, available_actions)}

    def _choose_action(self, state: Dict[str, Any], actions: List[str]) -> str:
        state_key = self._get_state_key(state)

        for action in actions:
            key = f"{state_key}_{action}"
            if key not in self.action_values:
                self.action_values[key] = 0.0

        if random.random() < self.config.get('epsilon', 0.2):
            return random.choice(actions)

        values = {}
        for action in actions:
            key = f"{state_key}_{action}"
            values[action] = self.action_values.get(key, 0.0)
        return max(values, key=values.get)

    def _learn(self, state: Dict[str, Any], action: str, reward: float):
        state_key = self._get_state_key(state)
        key = f"{state_key}_{action}"
        old_value = self.action_values.get(key, 0.0)
        lr = self.config.get('learning_rate', 0.15)

        new_value = old_value + lr * (reward - old_value)
        self.action_values[key] = new_value
        self.action_history.append((state_key, action, reward))

    def _get_state_key(self, state: Dict[str, Any]) -> str:
        if not state:
            return 'default'

        key_parts = []
        for k in ['energy', 'hunger', 'food_nearby', 'danger_nearby']:
            if k in state:
                val = state[k]
                if isinstance(val, (int, float)):
                    key_parts.append(f"{k}_{round(val / 0.2) * 0.2}")
                else:
                    key_parts.append(f"{k}_{val}")

        return '_'.join(key_parts) if key_parts else 'default'

    def test(self, config: Dict[str, Any]) -> float:
        self.config = config
        test_states = [
            {'energy': 90, 'hunger': 0.1, 'food_nearby': False, 'danger_nearby': False},
            {'energy': 30, 'hunger': 0.8, 'food_nearby': True, 'danger_nearby': False},
            {'energy': 50, 'hunger': 0.3, 'food_nearby': False, 'danger_nearby': True},
            {'energy': 10, 'hunger': 0.9, 'food_nearby': False, 'danger_nearby': False},
            {'energy': 70, 'hunger': 0.2, 'food_nearby': True, 'danger_nearby': False},
        ]
        expected_actions = ['explore', 'collect', 'flee', 'rest', 'collect']

        correct = 0
        for i, state in enumerate(test_states):
            if i > 0:
                self.process({'state': state, 'action': expected_actions[i-1], 'reward': 1.0})

            result = self.process({'state': state, 'actions': ['explore', 'collect', 'flee', 'rest']})
            chosen = result.get('action', 'explore')

            if chosen == expected_actions[i]:
                correct += 1

        return correct / len(test_states)

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]):
        self.config = config.copy()


class RelayEvolution:
    """
    Диспетчер эстафетной эволюции.
    Управляет цепочкой модулей и передаёт между ними данные.
    """

    def __init__(self):
        """Инициализирует цепочку модулей."""
        self.modules = []
        self.module_names = []
        self.current_data = None
        self.evolution_count = 0
        self.improvements = []
        self._build_chain()

    def _build_chain(self):
        """Строит цепочку модулей: зрение → сознание → память → предиктор → действия."""
        self.modules = [
            VisionModule(),
            ConsciousnessModuleEvolver(),
            MemoryModuleEvolver(),
            PredictorModuleEvolver(),
            ActionModuleEvolver(),
        ]
        self.module_names = ['Vision', 'Consciousness', 'Memory', 'Predictor', 'Action']

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные через всю цепочку модулей.
        Каждый модуль эволюционирует на основе своих входных и выходных данных.
        """
        self.current_data = data
        result = data.copy()

        for i, (module, name) in enumerate(zip(self.modules, self.module_names)):
            # Сохраняем входные данные для эволюции
            input_data = result.copy()

            # Обрабатываем модулем
            output = module.process(result)

            # Эволюционируем модуль
            improved = module.evolve(input_data, output)

            if improved:
                self.improvements.append({
                    'module': name,
                    'step': self.evolution_count,
                    'fitness': module.get_best_fitness(),
                    'config': module.get_config()
                })

            # Передаём результат дальше
            if isinstance(output, dict):
                result.update(output)
            else:
                result = output

        self.evolution_count += 1
        return result

    def get_module(self, name: str):
        """Возвращает модуль по имени."""
        for module, n in zip(self.modules, self.module_names):
            if n.lower() == name.lower():
                return module
        return None

    def get_best_configs(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает лучшие конфигурации всех модулей."""
        configs = {}
        for module, name in zip(self.modules, self.module_names):
            configs[name] = module.best_config.copy()
        return configs

    def get_improvements(self) -> List[Dict[str, Any]]:
        """Возвращает историю улучшений."""
        return self.improvements.copy()

    def reset(self):
        """Сбрасывает состояние эволюции."""
        self.improvements = []
        self.evolution_count = 0
        for module in self.modules:
            module.best_fitness = -float('inf')
            module.best_config = module.config.copy()
            module.fitness_history.clear()
            module.evolution_step = 0

    def __repr__(self) -> str:
        status = []
        for module, name in zip(self.modules, self.module_names):
            fitness = module.get_best_fitness()
            status.append(f"{name}: {fitness:.3f}")
        return f"RelayEvolution({', '.join(status)})"
