# -*- coding: utf-8 -*-
"""
УРОВЕНЬ 9: МЕТА-ОБУЧЕНИЕ

Агент учится учиться — анализирует свои стратегии обучения и адаптирует параметры.

Принцип работы:
1. Отслеживает эффективность различных стратегий
2. Анализирует, какие параметры дают лучшие результаты
3. Адаптирует скорость обучения, исследование, горизонт планирования
4. Учится на своих ошибках быстрее
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
import random
import time


class MetaLearning:
    """
    Мета-обучение — учится учиться.
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        
        # История действий и наград
        self.action_history: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []
        
        # Эффективность стратегий
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Параметры, которые могут меняться
        self.parameters: Dict[str, Any] = {
            'learning_rate': 0.1,
            'exploration_rate': 0.2,
            'planning_horizon': 10,
            'risk_tolerance': 0.5,
            'social_sensitivity': 0.5,
            'memory_decay': 0.9
        }
        
        # История параметров для отслеживания трендов
        self.param_history: List[Dict[str, Any]] = []
        
        # Шаг
        self.step = 0
        
        # Эффективность обучения
        self.learning_efficiency: float = 0.5
        self.efficiency_history: List[float] = []
        
    def record_experience(self, action: str, context: str, reward: float, 
                          success: bool, confidence: float):
        """Записывает опыт."""
        self.step += 1
        
        record = {
            'step': self.step,
            'action': action,
            'context': context,
            'reward': reward,
            'success': success,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        self.action_history.append(record)
        self.reward_history.append(reward)
        
        if len(self.action_history) > self.history_size:
            self.action_history = self.action_history[-self.history_size:]
            self.reward_history = self.reward_history[-self.history_size:]
        
        # Обновляем эффективность стратегий
        if context:
            self.strategy_performance[context].append(reward)
            if len(self.strategy_performance[context]) > 50:
                self.strategy_performance[context] = self.strategy_performance[context][-50:]
        
        # Мета-анализ каждые 20 шагов
        if self.step % 20 == 0:
            self._meta_analysis()
    
    def _meta_analysis(self):
        """Анализирует эффективность обучения и корректирует параметры."""
        if len(self.reward_history) < 10:
            return
        
        # 1. Анализ тренда наград
        recent = np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        older = np.mean(self.reward_history[-20:-10]) if len(self.reward_history) >= 20 else recent
        trend = recent - older
        
        # 2. Эффективность контекстов
        best_context = None
        best_reward = -float('inf')
        for ctx, rewards in self.strategy_performance.items():
            if len(rewards) > 3:
                avg = np.mean(rewards[-10:])
                if avg > best_reward:
                    best_reward = avg
                    best_context = ctx
        
        # 3. Корректировка параметров
        # Если награда падает — увеличиваем исследование
        if trend < -0.05:
            self.parameters['exploration_rate'] = min(0.5, 
                self.parameters['exploration_rate'] + 0.03)
        elif trend > 0.1:
            self.parameters['exploration_rate'] = max(0.05,
                self.parameters['exploration_rate'] - 0.02)
        
        # Если успешность высокая — увеличиваем горизонт планирования
        if recent > 0.5 and self.parameters['planning_horizon'] < 30:
            self.parameters['planning_horizon'] += 1
        elif recent < 0.1 and self.parameters['planning_horizon'] > 5:
            self.parameters['planning_horizon'] -= 1
        
        # Адаптация learning_rate
        if len(self.reward_history) > 20:
            variance = np.var(self.reward_history[-20:])
            if variance > 0.3:
                self.parameters['learning_rate'] = min(0.3, 
                    self.parameters['learning_rate'] + 0.01)
            elif variance < 0.05:
                self.parameters['learning_rate'] = max(0.01,
                    self.parameters['learning_rate'] - 0.01)
        
        # Обновляем эффективность обучения
        self.learning_efficiency = 0.9 * self.learning_efficiency + 0.1 * recent
        self.efficiency_history.append(self.learning_efficiency)
        
        # Запоминаем параметры
        self.param_history.append(self.parameters.copy())
        if len(self.param_history) > 50:
            self.param_history = self.param_history[-50:]
    
    def get_recommendation(self, context: str) -> Dict[str, Any]:
        """
        Возвращает рекомендацию по параметрам для данного контекста.
        """
        # Базовая рекомендация
        rec = self.parameters.copy()
        
        # Адаптация под контекст
        if context in self.strategy_performance:
            rewards = self.strategy_performance[context]
            if len(rewards) > 5:
                avg = np.mean(rewards[-10:])
                if avg < 0.1:
                    # Плохо в этом контексте — больше исследовать
                    rec['exploration_rate'] = min(0.6, rec['exploration_rate'] + 0.1)
                    rec['risk_tolerance'] = min(0.8, rec['risk_tolerance'] + 0.05)
                elif avg > 0.5:
                    # Хорошо в этом контексте — углубить планирование
                    rec['planning_horizon'] = min(30, rec['planning_horizon'] + 2)
        
        return rec
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        return {
            'step': self.step,
            'learning_efficiency': self.learning_efficiency,
            'parameters': self.parameters,
            'contexts': len(self.strategy_performance),
            'total_experiences': len(self.action_history)
        }
    
    def __repr__(self) -> str:
        return f"MetaLearning(efficiency={self.learning_efficiency:.2f}, contexts={len(self.strategy_performance)})"


class MetaLearningAgent:
    """
    Агент с мета-обучением.
    
    Оборачивает существующего агента и добавляет мета-обучение.
    """
    
    def __init__(self, base_agent, meta_learning: MetaLearning):
        self.base_agent = base_agent
        self.meta_learning = meta_learning
        
        # Применяем рекомендации
        self._apply_recommendations()
        
    def decide(self, grid: List[List[str]], position: Tuple[int, int], 
               other_agents: List[Tuple[int, Tuple[int, int]]]) -> Tuple[str, str]:
        """
        Принимает решение с учётом мета-обучения.
        """
        # Получаем рекомендации для текущего контекста
        x, y = position
        food_nearby = any(grid[ny][nx] == '🍎' for dx in [-1,0,1] for dy in [-1,0,1] 
                         if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
        danger_nearby = any(grid[ny][nx] == '⚠️' for dx in [-1,0,1] for dy in [-1,0,1] 
                           if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
        context = 'food_nearby_safe' if food_nearby and not danger_nearby else \
                  'food_with_danger' if food_nearby and danger_nearby else \
                  'danger_nearby' if danger_nearby else 'exploring'
        
        # Применяем рекомендации
        self._apply_recommendations(context)
        
        # Принимаем решение через базового агента
        action, thought = self.base_agent.decide(grid, position, other_agents)
        
        return action, thought
    
    def _apply_recommendations(self, context: str = None):
        """Применяет рекомендации мета-обучения."""
        if context:
            rec = self.meta_learning.get_recommendation(context)
        else:
            rec = self.meta_learning.parameters
        
        # Применяем параметры к базовому агенту, если есть соответствующие атрибуты
        if hasattr(self.base_agent, 'exploration_rate'):
            self.base_agent.exploration_rate = rec.get('exploration_rate', 0.2)
        if hasattr(self.base_agent, 'planning_horizon'):
            self.base_agent.planning_horizon = rec.get('planning_horizon', 10)
        if hasattr(self.base_agent, 'risk_tolerance'):
            self.base_agent.risk_tolerance = rec.get('risk_tolerance', 0.5)
    
    def learn(self, action: str, context: str, reward: float):
        """Учится на опыте."""
        success = reward > 0.1
        confidence = min(1.0, abs(reward) / 2.0)
        self.meta_learning.record_experience(action, context, reward, success, confidence)
        
        # Передаём обучение дальше
        if hasattr(self.base_agent, 'learn'):
            self.base_agent.learn(action, context, reward)
    
    def __getattr__(self, name):
        """Прокси для всех остальных методов."""
        if hasattr(self.base_agent, name):
            return getattr(self.base_agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __repr__(self) -> str:
        return f"MetaLearningAgent(base={self.base_agent})"
