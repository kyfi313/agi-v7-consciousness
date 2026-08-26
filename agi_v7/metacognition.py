# -*- coding: utf-8 -*-
"""
МЕТА-ПОЗНАНИЕ (Metacognition)

Уровень 7 адаптации: агент думает о собственном мышлении.

Принцип работы:
1. Агент анализирует свои прошлые решения
2. Оценивает эффективность своих стратегий
3. Корректирует параметры мышления в реальном времени
4. Ведёт внутренний диалог (self-talk)
5. Учится на ошибках и успехах

Это добавляет самосознание и рефлексию.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random
import time


@dataclass
class DecisionRecord:
    """Запись о принятом решении."""
    step: int
    action: str
    context: str
    predicted_reward: float
    actual_reward: float
    confidence: float
    success: bool
    thought: str
    
    def __repr__(self) -> str:
        return f"Decision({self.action}, {self.context}, success={self.success}, reward={self.actual_reward:.2f})"


class Metacognition:
    """
    Мета-познание.
    
    Анализирует собственные решения и корректирует стратегии.
    """
    
    def __init__(self, memory_size: int = 100, reflection_interval: int = 10):
        self.memory_size = memory_size
        self.reflection_interval = reflection_interval
        
        # История решений
        self.decision_history: List[DecisionRecord] = []
        
        # Статистика по контекстам
        self.context_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'best_action': None,
            'best_action_reward': -float('inf')
        })
        
        # Статистика по действиям
        self.action_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
        })
        
        # Уверенность в своих решениях
        self.confidence_history: List[float] = []
        self.confidence_trend: float = 0.0
        
        # Внутренний диалог
        self.inner_monologue: List[str] = []
        
        # Параметры, которые могут меняться
        self.parameters: Dict[str, Any] = {
            'exploration_rate': 0.2,
            'planning_horizon': 10,
            'risk_tolerance': 0.5,
            'social_sensitivity': 0.5,
        }
        
        # Самооценка
        self.self_esteem: float = 0.5
        self.self_esteem_history: List[float] = []
        
        # Шаг
        self.step = 0
        
    def record_decision(self, action: str, context: str, predicted_reward: float,
                        actual_reward: float, confidence: float, thought: str):
        """Записывает принятое решение."""
        self.step += 1
        
        success = actual_reward > 0.1
        record = DecisionRecord(
            step=self.step,
            action=action,
            context=context,
            predicted_reward=predicted_reward,
            actual_reward=actual_reward,
            confidence=confidence,
            success=success,
            thought=thought
        )
        
        self.decision_history.append(record)
        if len(self.decision_history) > self.memory_size:
            self.decision_history = self.decision_history[-self.memory_size:]
        
        # Обновляем статистику
        self._update_stats(record)
        
        # Записываем уверенность
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]
        
        # Обновляем тренд уверенности
        if len(self.confidence_history) > 10:
            recent = np.mean(self.confidence_history[-10:])
            old = np.mean(self.confidence_history[:-10]) if len(self.confidence_history) > 10 else recent
            self.confidence_trend = recent - old
        
        # Рефлексия
        if self.step % self.reflection_interval == 0:
            self._reflect()
        
    def _update_stats(self, record: DecisionRecord):
        """Обновляет статистику."""
        # Статистика по контексту
        ctx = self.context_stats[record.context]
        ctx['attempts'] += 1
        if record.success:
            ctx['successes'] += 1
        ctx['total_reward'] += record.actual_reward
        ctx['avg_reward'] = ctx['total_reward'] / ctx['attempts']
        
        # Лучшее действие для контекста
        if record.actual_reward > ctx['best_action_reward']:
            ctx['best_action_reward'] = record.actual_reward
            ctx['best_action'] = record.action
        
        # Статистика по действию
        act = self.action_stats[record.action]
        act['attempts'] += 1
        if record.success:
            act['successes'] += 1
        act['total_reward'] += record.actual_reward
        act['avg_reward'] = act['total_reward'] / act['attempts']
        
    def _reflect(self):
        """Рефлексия — анализ своих решений."""
        if len(self.decision_history) < 10:
            return
        
        recent = self.decision_history[-10:]
        
        # Вычисляем успешность
        success_rate = sum(1 for r in recent if r.success) / len(recent)
        avg_reward = np.mean([r.actual_reward for r in recent])
        
        # Обновляем самооценку
        self.self_esteem = 0.8 * self.self_esteem + 0.2 * success_rate
        self.self_esteem_history.append(self.self_esteem)
        
        # Внутренний диалог
        if success_rate > 0.7:
            thought = f"✅ Я хорошо действую! Успешность {success_rate:.0%}"
        elif success_rate > 0.4:
            thought = f"📊 Неплохо, но можно лучше. Успешность {success_rate:.0%}"
        else:
            thought = f"🤔 Нужно менять стратегию. Успешность {success_rate:.0%}"
        
        self.inner_monologue.append(f"[Шаг {self.step}] {thought}")
        if len(self.inner_monologue) > 20:
            self.inner_monologue = self.inner_monologue[-20:]
        
        # Корректируем параметры
        self._adjust_parameters(success_rate, avg_reward)
        
    def _adjust_parameters(self, success_rate: float, avg_reward: float):
        """Корректирует параметры на основе рефлексии."""
        # Если успешность низкая — увеличиваем исследование
        if success_rate < 0.3:
            self.parameters['exploration_rate'] = min(0.5, 
                self.parameters['exploration_rate'] + 0.05)
        elif success_rate > 0.7:
            self.parameters['exploration_rate'] = max(0.05,
                self.parameters['exploration_rate'] - 0.02)
        
        # Если награда растёт — увеличиваем горизонт планирования
        if avg_reward > 0.5 and len(self.confidence_history) > 20:
            if self.confidence_trend > 0.05:
                self.parameters['planning_horizon'] = min(30,
                    self.parameters['planning_horizon'] + 1)
        
        # Риск-толерантность зависит от самооценки
        self.parameters['risk_tolerance'] = 0.3 + 0.4 * self.self_esteem
        
        # Социальная чувствительность
        # Если много успехов в социальных контекстах — повышаем
        social_contexts = [r for r in self.decision_history[-20:] 
                          if 'food' in r.context or 'danger' in r.context]
        if social_contexts:
            social_success = sum(1 for r in social_contexts if r.success) / len(social_contexts)
            self.parameters['social_sensitivity'] = 0.3 + 0.5 * social_success
    
    def get_recommendation(self, context: str, available_actions: List[str]) -> Tuple[Optional[str], float]:
        """
        Возвращает рекомендованное действие для контекста.
        
        Returns:
            (лучшее действие, уверенность)
        """
        ctx_stats = self.context_stats.get(context)
        
        if ctx_stats and ctx_stats['attempts'] > 0:
            # Есть опыт в этом контексте
            best_action = ctx_stats['best_action']
            confidence = min(0.9, ctx_stats['successes'] / max(1, ctx_stats['attempts']))
            
            # Смешиваем с исследованием
            if random.random() < self.parameters['exploration_rate']:
                # Выбираем случайное действие, но не самое плохое
                other_actions = [a for a in available_actions if a != best_action]
                if other_actions:
                    return random.choice(other_actions), 0.3
            
            return best_action, confidence
        
        # Нет опыта — рекомендуем случайное
        return random.choice(available_actions) if available_actions else None, 0.1
    
    def get_self_assessment(self) -> Dict[str, Any]:
        """Возвращает самооценку агента."""
        if not self.decision_history:
            return {'self_esteem': 0.5, 'experience': 0}
        
        total = len(self.decision_history)
        successes = sum(1 for r in self.decision_history if r.success)
        total_reward = sum(r.actual_reward for r in self.decision_history)
        
        # Коэффициент полезности
        utility = total_reward / max(1, total)
        
        # Сильные и слабые стороны
        strengths = []
        weaknesses = []
        
        for context, stats in self.context_stats.items():
            if stats['attempts'] >= 3:
                rate = stats['successes'] / stats['attempts']
                if rate > 0.7:
                    strengths.append(context)
                elif rate < 0.3:
                    weaknesses.append(context)
        
        return {
            'self_esteem': self.self_esteem,
            'total_decisions': total,
            'success_rate': successes / max(1, total),
            'total_reward': total_reward,
            'utility': utility,
            'strengths': strengths[:3],
            'weaknesses': weaknesses[:3],
            'confidence_trend': self.confidence_trend,
            'parameters': self.parameters.copy(),
            'inner_monologue': self.inner_monologue[-5:],
        }
    
    def get_latest_thought(self) -> Optional[str]:
        """Возвращает последнюю мысль из внутреннего диалога."""
        return self.inner_monologue[-1] if self.inner_monologue else None
    
    def __repr__(self) -> str:
        assessment = self.get_self_assessment()
        return (f"Metacognition(esteem={assessment['self_esteem']:.2f}, "
                f"success_rate={assessment['success_rate']:.2f}, "
                f"decisions={assessment['total_decisions']})")


class MetaCognitiveAgent:
    """
    Агент с мета-познанием.
    
    Расширяет SocialAwareAgent, добавляя:
    - Самоанализ
    - Внутренний диалог
    - Динамическая корректировка стратегий
    - Обучение на ошибках
    """
    
    def __init__(self, base_agent, world_model, planner, theory_of_mind,
                 metacognition: Metacognition = None):
        self.base_agent = base_agent
        self.world_model = world_model
        self.planner = planner
        self.theory_of_mind = theory_of_mind
        self.metacognition = metacognition or Metacognition()
        
        # Текущее состояние
        self.last_action = None
        self.last_context = None
        self.last_prediction = 0.0
        self.last_confidence = 0.0
        
    def decide(self, grid: List[List[str]], position: Tuple[int, int],
               other_agents: List[Tuple[int, Tuple[int, int]]]) -> Tuple[str, str]:
        """
        Принимает решение с использованием мета-познания.
        """
        # Обновляем теорию разума
        self.theory_of_mind.observe(grid, position, other_agents)
        
        # Обновляем модель мира
        self.world_model.observe(grid, position, other_agents)
        
        # Определяем контекст
        x, y = position
        food_nearby = any(grid[ny][nx] == '🍎' for dx in [-1,0,1] for dy in [-1,0,1] 
                         if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
        danger_nearby = any(grid[ny][nx] == '⚠️' for dx in [-1,0,1] for dy in [-1,0,1] 
                           if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
        
        if food_nearby and not danger_nearby:
            context = 'food_nearby_safe'
        elif food_nearby and danger_nearby:
            context = 'food_with_danger'
        elif danger_nearby:
            context = 'danger_nearby'
        elif self.world_model.get_food_hotspots():
            context = 'predicting_food'
        else:
            context = 'exploring'
        
        # Используем мета-познание для выбора действия
        actions = ['up', 'down', 'left', 'right', 'wait']
        action, confidence = self.metacognition.get_recommendation(context, actions)
        
        # Если мета-познание не дало рекомендации — используем планировщик
        if action is None:
            action, plan = self.planner.get_next_action(grid, position, self.world_model)
            confidence = 0.5
            thought = f"План: {plan.actions[:3] if plan else 'нет'}"
        else:
            # Формируем мысль
            stats = self.metacognition.context_stats.get(context, {})
            success_rate = stats.get('successes', 0) / max(1, stats.get('attempts', 1))
            thought = f"Выбираю {action} в контексте '{context}' (успешность {success_rate:.0%})"
        
        # Проверяем социальный контекст
        competition = self.theory_of_mind.get_competitive_advantage(position)
        if competition['has_competition']:
            # Добавляем социальное измерение в мысль
            thought += f" | рядом агент {competition['nearest_agent']}"
        
        # Запоминаем решение для последующей оценки
        self.last_action = action
        self.last_context = context
        self.last_prediction = 0.1  # Будет обновлено после получения награды
        self.last_confidence = confidence
        
        return action, thought
    
    def learn(self, action: str, context: str, reward: float):
        """
        Обучает модель на основе полученного опыта.
        """
        # Записываем решение в мета-познание
        self.metacognition.record_decision(
            action=action,
            context=context,
            predicted_reward=self.last_prediction,
            actual_reward=reward,
            confidence=self.last_confidence,
            thought=f"Награда: {reward:.2f}"
        )
        
        # Обучаем модель мира
        self.world_model.learn_causal_relation(action, context, reward)
        
    def get_self_assessment(self) -> Dict[str, Any]:
        """Возвращает самооценку агента."""
        return self.metacognition.get_self_assessment()
    
    def get_inner_monologue(self) -> List[str]:
        """Возвращает внутренний диалог агента."""
        return self.metacognition.inner_monologue
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку состояния."""
        return {
            'world_model': self.world_model.get_memory_summary(),
            'planner': self.planner.get_plan_stats(),
            'theory_of_mind': self.theory_of_mind.get_summary(),
            'metacognition': self.metacognition.get_self_assessment(),
            'inner_monologue': self.metacognition.inner_monologue[-3:],
        }


# Тест
if __name__ == "__main__":
    print("🧠 Тест мета-познания")
    
    from agi_v7.world_model import WorldModel
    from agi_v7.planner import Planner
    from agi_v7.theory_of_mind import TheoryOfMind
    
    # Создаём компоненты
    world_model = WorldModel(grid_size=8)
    planner = Planner(horizon=8, num_samples=15)
    theory_of_mind = TheoryOfMind(grid_size=8)
    metacognition = Metacognition()
    
    # Создаём агента
    agent = MetaCognitiveAgent(None, world_model, planner, theory_of_mind, metacognition)
    
    # Симулируем принятие решений
    grid = [['.' for _ in range(8)] for _ in range(8)]
    grid[2][3] = '🍎'
    
    other_agents = [(1, (1, 1))]
    
    for i in range(30):
        action, thought = agent.decide(grid, (0, 0), other_agents)
        
        # Симулируем награду
        reward = random.random() * 2 - 0.5
        if action in ['up', 'down', 'left', 'right']:
            reward += 0.2
        
        agent.learn(action, 'exploring', reward)
    
    print(f"📊 Самооценка: {agent.get_self_assessment()}")
    print(f"💭 Внутренний диалог:")
    for thought in agent.get_inner_monologue()[-3:]:
        print(f"   {thought}")
    
    print("✅ Мета-познание работает!")
