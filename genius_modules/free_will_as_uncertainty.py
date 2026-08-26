# -*- coding: utf-8 -*-
"""
МОДУЛЬ: СВОБОДА ВОЛИ КАК НЕОПРЕДЕЛЁННОСТЬ
Гениальность: Свобода воли — это не иллюзия, а результат фундаментальной
неопределённости в принятии решений. Когда детерминизм даёт сбой,
возникает пространство выбора.

Реализовано: DeterministicPath — жёсткий путь (рефлексы),
ProbabilisticPath — вероятностный путь (выбор),
FreeWillSystem — баланс между детерминизмом и свободой.

Свобода = 1 / (детерминизм + epsilon).
"""

import numpy as np
from collections import deque
import random
import math

class DeterministicPath:
    """Детерминированный путь — жёсткие рефлексы."""
    def __init__(self):
        self.rules = []
        self.strength = 0.7
        self.rule_count = 0

    def add_rule(self, condition, action):
        self.rules.append((condition, action))
        self.rule_count += 1

    def decide(self, state):
        """Принимает детерминированное решение."""
        for condition, action in self.rules:
            if condition(state):
                return action
        return None

    def get_state(self):
        return {'strength': self.strength, 'rules': self.rule_count}

class ProbabilisticPath:
    """Вероятностный путь — выбор из нескольких вариантов."""
    def __init__(self, num_actions=4):
        self.action_weights = np.ones(num_actions) / num_actions
        self.num_actions = num_actions
        self.entropy_history = deque(maxlen=20)
        self.temperature = 1.0

    def decide(self, state):
        """Принимает вероятностное решение."""
        # Вычисляем вероятности
        if len(state) > 0:
            # Модифицируем веса в зависимости от состояния
            modified_weights = self.action_weights + state[:self.num_actions] * 0.1
            modified_weights = np.maximum(modified_weights, 0)
            probs = modified_weights / np.sum(modified_weights)
        else:
            probs = self.action_weights

        # Вычисляем энтропию (меру неопределённости)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        self.entropy_history.append(entropy)

        # Выбираем действие
        action = np.random.choice(self.num_actions, p=probs)
        return action, probs, entropy

    def update_weights(self, action, reward):
        """Обновляет веса на основе награды."""
        # Простое обновление
        self.action_weights[action] += reward * 0.1
        self.action_weights = np.maximum(self.action_weights, 0)
        self.action_weights = self.action_weights / np.sum(self.action_weights)

    def get_state(self):
        avg_entropy = np.mean(list(self.entropy_history)) if self.entropy_history else 0.5
        return {
            'weights': self.action_weights.tolist(),
            'entropy': avg_entropy,
            'temperature': self.temperature
        }

class FreeWillSystem:
    """Система, балансирующая между детерминизмом и свободой."""
    def __init__(self, num_actions=4):
        self.deterministic = DeterministicPath()
        self.probabilistic = ProbabilisticPath(num_actions)
        self.free_will_level = 0.5
        self.decision_history = deque(maxlen=30)
        self.uncertainty_factor = 0.5
        self.free_will_trace = deque(maxlen=50)

    def add_rule(self, condition, action):
        """Добавляет детерминированное правило."""
        self.deterministic.add_rule(condition, action)

    def decide(self, state):
        """Принимает решение с учётом свободы воли."""
        # 1. Проверяем детерминированные правила
        deterministic_action = self.deterministic.decide(state)
        if deterministic_action is not None:
            # Если правило сработало — оно имеет приоритет
            decision = {
                'action': deterministic_action,
                'path': 'deterministic',
                'free_will': 0.0,
                'uncertainty': 0.0
            }
            self.decision_history.append(decision)
            return decision

        # 2. Иначе — вероятностный выбор
        action, probs, entropy = self.probabilistic.decide(state)

        # 3. Вычисляем уровень свободы воли
        # Свобода = 1 / (детерминизм + epsilon)
        # Детерминизм = 1 - энтропия / max_entropy
        max_entropy = math.log(self.probabilistic.num_actions)
        determinism = 1.0 - (entropy / max_entropy)
        free_will = 1.0 / (determinism + 0.01)
        free_will = min(1.0, free_will / 10.0)  # нормализация

        # 4. Обновляем уровень свободы
        self.free_will_level = free_will * 0.7 + self.free_will_level * 0.3
        self.uncertainty_factor = 1.0 - free_will
        self.free_will_trace.append(free_will)

        decision = {
            'action': action,
            'path': 'probabilistic',
            'free_will': free_will,
            'uncertainty': self.uncertainty_factor,
            'entropy': entropy,
            'probs': probs.tolist()
        }
        self.decision_history.append(decision)
        return decision

    def update_weights(self, action, reward):
        """Обновляет вероятностные веса."""
        self.probabilistic.update_weights(action, reward)

    def get_state(self):
        return {
            'free_will_level': self.free_will_level,
            'uncertainty': self.uncertainty_factor,
            'deterministic': self.deterministic.get_state(),
            'probabilistic': self.probabilistic.get_state(),
            'history_size': len(self.decision_history),
            'avg_free_will': np.mean(list(self.free_will_trace)) if self.free_will_trace else 0.5
        }

if __name__ == "__main__":
    print("="*60)
    print("🌀 СВОБОДА ВОЛИ КАК НЕОПРЕДЕЛЁННОСТЬ")
    print("="*60)
    fw = FreeWillSystem(num_actions=3)
    # Добавляем детерминированные правила
    fw.add_rule(lambda s: s[0] > 0.5, 0)
    fw.add_rule(lambda s: s[0] < -0.5, 1)
    for i in range(12):
        state = np.random.randn(3) * 0.8
        decision = fw.decide(state)
        # Симулируем награду
        reward = np.random.randn() * 0.3
        fw.update_weights(decision['action'], reward)
        print(f"Шаг {i}: действие={decision['action']}, путь={decision['path']}, свобода={decision.get('free_will', 0):.2f}")
    print("\n💡 Гениальность: Свобода воли — это не иллюзия, а фундаментальная неопределённость.")
    print("   Свобода = 1 / (детерминизм + epsilon).")
