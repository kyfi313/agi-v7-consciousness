# -*- coding: utf-8 -*-
"""
Система оценки ценности предметов и действий
Основана на: полезность, редкость, полезность для целей
"""

import math
import random
from typing import Dict, Any, List, Optional


class ValueSystem:
    """
    Система оценки ценности предметов и действий
    
    Ценность определяется комбинацией факторов:
    1. Полезность (utility) — что предмет даёт системе
    2. Редкость (rarity) — как часто встречается
    3. Контекстуальная ценность — зависит от текущего состояния
    4. Эмоциональная ценность — связана с эмоциями
    5. Социальная ценность — что даёт для взаимодействия
    """
    
    def __init__(self):
        # Базовая ценность предметов (0-1)
        self.base_values = {
            # Ресурсы
            'diamond': 0.95,
            'emerald': 0.90,
            'gold': 0.85,
            'iron': 0.75,
            'coal': 0.60,
            'stone': 0.40,
            'dirt': 0.20,
            'grass': 0.10,
            
            # Инструменты
            'pickaxe': 0.80,
            'axe': 0.75,
            'shovel': 0.70,
            'sword': 0.85,
            
            # Еда
            'apple': 0.60,
            'bread': 0.65,
            'steak': 0.75,
            'cooked_porkchop': 0.70,
            
            # Особые
            'obsidian': 0.70,
            'ender_pearl': 0.80,
            'book': 0.50,
            'chest': 0.55,
            
            # Блоки
            'wood': 0.50,
            'planks': 0.40,
            'glass': 0.35,
            'torch': 0.45,
            
            # Неизвестное
            'unknown': 0.30,
        }
        
        # Контекстуальные модификаторы
        self.context_modifiers = {
            'low_energy': {'food': 0.3, 'rest': 0.2},  # голод → еда ценнее
            'high_danger': {'sword': 0.3, 'armor': 0.3},  # опасность → оружие ценнее
            'low_resources': {'pickaxe': 0.2, 'diamond': 0.2},  # нет ресурсов → инструменты ценнее
            'exploring': {'food': 0.1, 'torch': 0.2},  # исследование → еда и свет
        }
        
        # История оценок (для обучения)
        self.value_history = []
        self.learning_rate = 0.1
        self.discount_factor = 0.9  # Для Q-learning
        
        # Q-таблица: (state, action) -> value
        self.q_table = {}
        
        # Накопленный опыт для обучения
        self.experience_buffer = []
        self.max_buffer_size = 100
        self.batch_size = 10
        
        # Связи предмет → действие
        self.item_actions = {
            'diamond': ['mine', 'craft_tool'],
            'wood': ['mine', 'craft'],
            'apple': ['eat', 'collect'],
            'sword': ['attack', 'defend'],
            'torch': ['place', 'light'],
        }
    
    def evaluate_item(self, item: str, context: Dict[str, Any] = None) -> float:
        """
        Оценивает ценность предмета в заданном контексте
        
        Args:
            item: название предмета
            context: текущий контекст {'energy': 0.7, 'danger': 0.3, ...}
        
        Returns:
            float: ценность 0-1
        """
        context = context or {}
        
        # 1. Базовая ценность
        base_value = self.base_values.get(item, 0.3)
        
        # 2. Контекстуальная модификация
        context_mod = 0.0
        for context_type, modifiers in self.context_modifiers.items():
            if context_type in context:
                for key, mod in modifiers.items():
                    if key in item or item in key:
                        context_mod += mod
        
        # 3. Редкость (если известно)
        rarity_mod = self._get_rarity_modifier(item, context)
        
        # 4. Эмоциональная ценность
        emotional_mod = self._get_emotional_value(item, context)
        
        # 5. Ценность для текущей цели
        goal_mod = self._get_goal_value(item, context)
        
        # Итоговая ценность
        value = base_value + context_mod + rarity_mod + emotional_mod + goal_mod
        value = max(0.0, min(1.0, value))
        
        # Запоминаем оценку
        self.value_history.append({
            'item': item,
            'value': value,
            'context': context,
            'timestamp': len(self.value_history)
        })
        if len(self.value_history) > 1000:
            self.value_history.pop(0)
        
        return value
    
    def _get_rarity_modifier(self, item: str, context: Dict) -> float:
        """Модификатор редкости"""
        rarity = self.base_values.get(item, 0.3)
        # Редкие предметы ценнее в контексте исследования
        if context.get('exploring', False) and rarity > 0.7:
            return 0.2
        return 0.0
    
    def _get_emotional_value(self, item: str, context: Dict) -> float:
        """Эмоциональная ценность"""
        # Если система голодна, еда ценнее
        if 'food' in item and context.get('energy', 0.5) < 0.3:
            return 0.3
        # Если опасность, защита ценнее
        if 'sword' in item and context.get('danger', 0.0) > 0.5:
            return 0.3
        return 0.0
    
    def _get_goal_value(self, item: str, context: Dict) -> float:
        """Ценность для текущей цели"""
        goal = context.get('goal', 'explore')
        
        goal_relevance = {
            'explore': ['torch', 'food', 'pickaxe'],
            'mine': ['pickaxe', 'diamond', 'iron'],
            'build': ['wood', 'stone', 'glass'],
            'survive': ['food', 'sword', 'armor'],
        }
        
        if item in goal_relevance.get(goal, []):
            return 0.2
        return 0.0
    
    def evaluate_action(self, action: str, item: str, context: Dict) -> float:
        """
        Оценивает ценность действия с предметом
        
        Args:
            action: действие ('mine', 'eat', 'craft', ...)
            item: предмет
            context: контекст
        
        Returns:
            float: ценность 0-1
        """
        item_value = self.evaluate_item(item, context)
        
        # Базовая ценность действия
        action_base = {
            'mine': 0.5,
            'eat': 0.6,
            'craft': 0.4,
            'place': 0.3,
            'attack': 0.4,
            'defend': 0.5,
        }.get(action, 0.3)
        
        # Комбинация предмет + действие
        if item in self.item_actions:
            if action in self.item_actions[item]:
                action_base += 0.2
        
        # Контекстуальные модификаторы
        if action == 'eat' and context.get('energy', 0.5) < 0.3:
            action_base += 0.3
        if action == 'attack' and context.get('danger', 0.0) > 0.5:
            action_base += 0.3
        
        return min(1.0, item_value * 0.6 + action_base * 0.4)
    
    def get_best_item_for_context(self, items: List[str], context: Dict) -> str:
        """Выбирает лучший предмет для контекста"""
        if not items:
            return None
        
        best_item = None
        best_value = -1.0
        
        for item in items:
            value = self.evaluate_item(item, context)
            if value > best_value:
                best_value = value
                best_item = item
        
        return best_item
    
    def get_best_action_for_item(self, item: str, context: Dict) -> str:
        """Выбирает лучшее действие с предметом"""
        actions = self.item_actions.get(item, ['collect'])
        
        best_action = None
        best_value = -1.0
        
        for action in actions:
            value = self.evaluate_action(action, item, context)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action or 'collect'
    
    def learn_from_feedback(self, item: str, action: str, outcome: float, context: Dict = None):
        """
        Обучение на основе обратной связи (с Q-learning)
        
        Args:
            item: предмет
            action: действие
            outcome: результат (-1 до 1, где >0 награда, <0 наказание)
            context: контекст для Q-таблицы
        """
        context = context or {}
        
        # 1. Обновляем базовую ценность предмета
        current_value = self.base_values.get(item, 0.3)
        # Обучение с учётом знака: награда увеличивает, наказание уменьшает
        delta = self.learning_rate * outcome
        new_value = current_value + delta
        self.base_values[item] = max(0.0, min(1.0, new_value))
        
        # 2. Q-learning: обновляем Q-значение для пары (состояние, действие)
        state_key = self._get_state_key(context)
        action_key = f"{state_key}:{action}:{item}"
        
        old_q = self.q_table.get(action_key, 0.5)
        # Обновляем по формуле Q-learning
        new_q = old_q + self.learning_rate * (outcome + self.discount_factor * 0.5 - old_q)
        self.q_table[action_key] = max(0.0, min(1.0, new_q))
        
        # 3. Накопление опыта для пакетного обучения
        self.experience_buffer.append({
            'item': item,
            'action': action,
            'outcome': outcome,
            'state': state_key,
            'context': context
        })
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # 4. Пакетное обучение, если накоплено достаточно
        if len(self.experience_buffer) >= self.batch_size:
            self._batch_learn()
    
    def _get_state_key(self, context: Dict) -> str:
        """Создаёт ключ состояния для Q-таблицы"""
        # Упрощённо: используем основные параметры
        energy = 'low' if context.get('energy', 0.5) < 0.3 else 'high'
        danger = 'high' if context.get('danger', 0) > 0.5 else 'low'
        goal = context.get('goal', 'explore')
        return f"{energy}_{danger}_{goal}"
    
    def _batch_learn(self):
        """Пакетное обучение на накопленном опыте"""
        # Берём последние batch_size примеров
        batch = self.experience_buffer[-self.batch_size:]
        for exp in batch:
            # Повторно применяем обучение с меньшим шагом
            item = exp['item']
            action = exp['action']
            outcome = exp['outcome']
            state_key = exp['state']
            
            # Обновляем Q-таблицу
            action_key = f"{state_key}:{action}:{item}"
            old_q = self.q_table.get(action_key, 0.5)
            new_q = old_q + 0.05 * (outcome - old_q)
            self.q_table[action_key] = max(0.0, min(1.0, new_q))
    
    def get_summary(self) -> Dict:
        """Возвращает сводку системы оценки"""
        return {
            'items_count': len(self.base_values),
            'history_count': len(self.value_history),
            'top_items': sorted(self.base_values.items(), key=lambda x: x[1], reverse=True)[:5],
            'learning_rate': self.learning_rate,
        }


class ValueSystemWithMemory:
    """
    Расширенная система оценки с семантической памятью
    """
    
    def __init__(self, semantic_memory=None):
        self.value_system = ValueSystem()
        self.semantic_memory = semantic_memory
        self.memory = {}
    
    def evaluate_with_memory(self, item: str, context: Dict) -> float:
        """Оценивает предмет, используя семантическую память"""
        # 1. Базовая оценка
        value = self.value_system.evaluate_item(item, context)
        
        # 2. Проверяем память
        if self.semantic_memory:
            # Ищем концепт предмета
            concept_name = f"item_{item}"
            if concept_name in self.semantic_memory.concepts:
                concept = self.semantic_memory.concepts[concept_name]
                # Используем частоту как модификатор
                if concept.frequency > 0.5:
                    value = min(1.0, value + 0.1)
        
        # 3. Запоминаем оценку
        self.memory[item] = {
            'value': value,
            'context': context,
            'time': len(self.memory)
        }
        
        return value
    
    def get_action_value(self, action: str, target: str, context: Dict) -> float:
        """Оценивает ценность действия с целью"""
        # Базовая ценность
        base_value = 0.3
        
        if action == 'mine':
            base_value = 0.5
            # Ценность добычи зависит от предмета
            item_value = self.evaluate_with_memory(target, context)
            base_value = 0.3 + item_value * 0.5
            
        elif action == 'eat':
            base_value = 0.6
            if context.get('energy', 0.5) < 0.3:
                base_value = 0.9
                
        elif action == 'craft':
            base_value = 0.4
            # Сложные предметы ценнее
            if target in ['diamond_sword', 'enchanted_item']:
                base_value = 0.7
        
        return min(1.0, base_value)
