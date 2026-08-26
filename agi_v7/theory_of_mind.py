# -*- coding: utf-8 -*-
"""
ТЕОРИЯ РАЗУМА (Theory of Mind)

Уровень 6 адаптации: агент моделирует других агентов.

Принцип работы:
1. Агент наблюдает за поведением других агентов
2. Строит модели их целей и стратегий
3. Предсказывает их будущие действия
4. Использует это для принятия решений (конкуренция или сотрудничество)

Это добавляет социальный интеллект.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random


@dataclass
class AgentModel:
    """Модель другого агента."""
    agent_id: int
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    food_collected_history: List[int] = field(default_factory=list)
    
    # Предполагаемая цель
    inferred_goal: str = 'unknown'  # 'food', 'explore', 'avoid_danger'
    goal_confidence: float = 0.0
    
    # Предполагаемая стратегия
    strategy: str = 'random'
    strategy_confidence: float = 0.0
    
    # Предсказание следующей позиции
    predicted_next_position: Optional[Tuple[int, int]] = None
    prediction_confidence: float = 0.0


class TheoryOfMind:
    """
    Модуль теории разума.
    
    Строит модели других агентов и предсказывает их поведение.
    """
    
    def __init__(self, grid_size: int = 8, memory_steps: int = 30):
        self.grid_size = grid_size
        self.memory_steps = memory_steps
        
        # Модели агентов
        self.agent_models: Dict[int, AgentModel] = {}
        
        # История взаимодействий
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Своя позиция (для сравнения)
        self.my_position: Optional[Tuple[int, int]] = None
        
        # Статистика
        self.step = 0
        self.predictions_correct = 0
        self.predictions_total = 0
        
    def observe(self, grid: List[List[str]], my_position: Tuple[int, int],
                other_agents: List[Tuple[int, Tuple[int, int]]]):
        """
        Наблюдает за другими агентами.
        
        Args:
            grid: Текущая сетка
            my_position: Позиция нашего агента
            other_agents: Список (id, позиция) других агентов
        """
        self.step += 1
        self.my_position = my_position
        
        # Обновляем модели
        for agent_id, pos in other_agents:
            if agent_id not in self.agent_models:
                self.agent_models[agent_id] = AgentModel(agent_id=agent_id)
            
            model = self.agent_models[agent_id]
            model.position_history.append(pos)
            
            # Ограничиваем историю
            if len(model.position_history) > self.memory_steps:
                model.position_history = model.position_history[-self.memory_steps:]
            
            # Определяем цель агента
            self._infer_goal(model, grid)
            
            # Предсказываем следующую позицию
            self._predict_next_position(model)
    
    def _infer_goal(self, model: AgentModel, grid: List[List[str]]):
        """Определяет предполагаемую цель агента на основе его траектории."""
        if len(model.position_history) < 3:
            return
        
        # Проверяем, направляется ли к еде
        food_positions = self._find_food_positions(grid)
        if food_positions:
            last_pos = model.position_history[-1]
            min_dist = min(abs(p[0] - last_pos[0]) + abs(p[1] - last_pos[1]) 
                          for p in food_positions)
            if min_dist < 3:
                model.inferred_goal = 'food'
                model.goal_confidence = min(1.0, (3 - min_dist) / 3)
                return
        
        # Проверяем, убегает ли от опасности
        danger_positions = self._find_danger_positions(grid)
        if danger_positions:
            last_pos = model.position_history[-1]
            min_dist = min(abs(p[0] - last_pos[0]) + abs(p[1] - last_pos[1]) 
                          for p in danger_positions)
            if min_dist < 2:
                model.inferred_goal = 'avoid_danger'
                model.goal_confidence = min(1.0, (2 - min_dist) / 2)
                return
        
        # Иначе — исследование
        model.inferred_goal = 'explore'
        model.goal_confidence = 0.5
    
    def _predict_next_position(self, model: AgentModel):
        """Предсказывает следующую позицию агента."""
        if len(model.position_history) < 2:
            model.predicted_next_position = None
            model.prediction_confidence = 0.0
            return
        
        # Простая экстраполяция: вычисляем среднее смещение
        positions = model.position_history
        dx_sum, dy_sum = 0, 0
        for i in range(1, min(len(positions), 5)):
            prev = positions[-i-1]
            curr = positions[-i]
            dx_sum += curr[0] - prev[0]
            dy_sum += curr[1] - prev[1]
        
        count = min(len(positions) - 1, 4)
        if count == 0:
            model.predicted_next_position = None
            model.prediction_confidence = 0.0
            return
        
        dx = int(round(dx_sum / count))
        dy = int(round(dy_sum / count))
        
        last_pos = positions[-1]
        pred_x = max(0, min(self.grid_size - 1, last_pos[0] + dx))
        pred_y = max(0, min(self.grid_size - 1, last_pos[1] + dy))
        
        model.predicted_next_position = (pred_x, pred_y)
        
        # Уверенность зависит от стабильности движения
        if dx != 0 or dy != 0:
            # Проверяем, насколько стабильно направление
            consistent = True
            for i in range(1, min(len(positions), 4)):
                p_prev = positions[-i-1]
                p_curr = positions[-i]
                dx_curr = p_curr[0] - p_prev[0]
                dy_curr = p_curr[1] - p_prev[1]
                if (dx_curr != 0) != (dx != 0) or (dy_curr != 0) != (dy != 0):
                    consistent = False
                    break
            model.prediction_confidence = 0.8 if consistent else 0.4
        else:
            model.prediction_confidence = 0.1
    
    def _find_food_positions(self, grid: List[List[str]]) -> List[Tuple[int, int]]:
        """Находит все позиции еды на сетке."""
        positions = []
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == '🍎':
                    positions.append((x, y))
        return positions
    
    def _find_danger_positions(self, grid: List[List[str]]) -> List[Tuple[int, int]]:
        """Находит все позиции опасности на сетке."""
        positions = []
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == '⚠️':
                    positions.append((x, y))
        return positions
    
    def predict_agent_movement(self, agent_id: int, steps_ahead: int = 3) -> List[Tuple[int, int]]:
        """
        Предсказывает траекторию агента на несколько шагов вперёд.
        """
        model = self.agent_models.get(agent_id)
        if not model or not model.predicted_next_position:
            return []
        
        trajectory = []
        pos = model.predicted_next_position
        for _ in range(steps_ahead):
            trajectory.append(pos)
            # Простая экстраполяция
            if len(model.position_history) >= 2:
                dx = pos[0] - model.position_history[-1][0]
                dy = pos[1] - model.position_history[-1][1]
                pos = (pos[0] + dx, pos[1] + dy)
                pos = (max(0, min(self.grid_size - 1, pos[0])),
                       max(0, min(self.grid_size - 1, pos[1])))
        
        return trajectory
    
    def get_competitive_advantage(self, my_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Оценивает конкурентное преимущество.
        
        Возвращает:
        - Ближайший агент
        - Направление к нему
        - Ожидаемый конфликт (если оба идут к одной еде)
        """
        if not self.agent_models:
            return {'has_competition': False}
        
        my_x, my_y = my_position
        nearest_agent = None
        min_dist = float('inf')
        
        for agent_id, model in self.agent_models.items():
            if not model.position_history:
                continue
            pos = model.position_history[-1]
            dist = abs(pos[0] - my_x) + abs(pos[1] - my_y)
            if dist < min_dist:
                min_dist = dist
                nearest_agent = agent_id
        
        result = {
            'has_competition': min_dist < 5,
            'nearest_agent': nearest_agent,
            'distance': min_dist,
            'direction': self._get_direction(my_position, 
                self.agent_models[nearest_agent].position_history[-1] if nearest_agent else (0, 0))
        }
        
        return result
    
    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Возвращает направление от одной позиции к другой."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def should_compete_or_cooperate(self, grid: List[List[str]], 
                                     my_position: Tuple[int, int]) -> str:
        """
        Определяет, стоит ли конкурировать или сотрудничать.
        
        Returns:
            'compete' или 'cooperate'
        """
        # Проверяем, есть ли рядом агент
        competition = self.get_competitive_advantage(my_position)
        
        if not competition['has_competition']:
            return 'compete'  # Нет конкуренции — просто действуем
        
        # Проверяем, есть ли еда в зоне конфликта
        food_positions = self._find_food_positions(grid)
        if not food_positions:
            return 'cooperate'  # Нет еды — смысла конкурировать нет
        
        # Проверяем, идёт ли другой агент к той же еде
        other_pos = self.agent_models[competition['nearest_agent']].position_history[-1]
        my_food_target = min(food_positions, 
                            key=lambda p: abs(p[0] - my_position[0]) + abs(p[1] - my_position[1]))
        other_food_target = min(food_positions,
                               key=lambda p: abs(p[0] - other_pos[0]) + abs(p[1] - other_pos[1]))
        
        if my_food_target == other_food_target:
            # Конкуренция за ту же еду
            dist_to_food = abs(my_food_target[0] - my_position[0]) + abs(my_food_target[1] - my_position[1])
            other_dist_to_food = abs(my_food_target[0] - other_pos[0]) + abs(my_food_target[1] - other_pos[1])
            
            if dist_to_food < other_dist_to_food:
                return 'compete'  # Я ближе — конкурирую
            else:
                return 'cooperate'  # Другой ближе — лучше уступить
        
        return 'compete'
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку теории разума."""
        return {
            'agents_tracked': len(self.agent_models),
            'predictions_correct': self.predictions_correct,
            'predictions_total': self.predictions_total,
            'accuracy': self.predictions_correct / max(1, self.predictions_total),
            'models': {
                agent_id: {
                    'goal': model.inferred_goal,
                    'goal_confidence': model.goal_confidence,
                    'predicted_position': model.predicted_next_position,
                    'prediction_confidence': model.prediction_confidence,
                }
                for agent_id, model in self.agent_models.items()
            }
        }
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        return (f"TheoryOfMind(agents={summary['agents_tracked']}, "
                f"accuracy={summary['accuracy']:.2f})")


class SocialAwareAgent:
    """
    Агент с теорией разума.
    
    Расширяет PlanningAwareAgent, добавляя:
    - Социальное восприятие
    - Конкуренция и сотрудничество
    - Предсказание поведения других агентов
    """
    
    def __init__(self, base_agent, world_model, planner, theory_of_mind: TheoryOfMind = None):
        self.base_agent = base_agent
        self.world_model = world_model
        self.planner = planner
        self.theory_of_mind = theory_of_mind or TheoryOfMind()
        
        # Социальная стратегия
        self.social_strategy = 'adaptive'  # 'aggressive', 'cooperative', 'adaptive'
        self.last_interaction = None
        
    def decide(self, grid: List[List[str]], position: Tuple[int, int],
               other_agents: List[Tuple[int, Tuple[int, int]]]) -> Tuple[str, str]:
        """
        Принимает решение с использованием теории разума.
        """
        # Обновляем теорию разума
        self.theory_of_mind.observe(grid, position, other_agents)
        
        # Обновляем модель мира
        self.world_model.observe(grid, position, other_agents)
        
        # Определяем социальный контекст
        competition = self.theory_of_mind.get_competitive_advantage(position)
        
        if competition['has_competition']:
            # Есть конкуренция — проверяем, стоит ли конкурировать
            action_choice = self.theory_of_mind.should_compete_or_cooperate(grid, position)
            
            if action_choice == 'cooperate':
                # Уступаем — идём в другую сторону
                # Используем модель мира для поиска альтернативной цели
                hotspots = self.world_model.get_food_hotspots()
                if hotspots:
                    # Выбираем горячую точку, которая дальше от другого агента
                    other_pos = self.theory_of_mind.agent_models[
                        competition['nearest_agent']
                    ].position_history[-1]
                    
                    best_hotspot = None
                    max_dist = -1
                    for hx, hy in hotspots:
                        dist_to_me = abs(hx - position[0]) + abs(hy - position[1])
                        dist_to_other = abs(hx - other_pos[0]) + abs(hy - other_pos[1])
                        if dist_to_me < dist_to_other and dist_to_me > max_dist:
                            max_dist = dist_to_me
                            best_hotspot = (hx, hy)
                    
                    if best_hotspot:
                        # Идём к альтернативной цели
                        tx, ty = best_hotspot
                        dx = tx - position[0]
                        dy = ty - position[1]
                        
                        if abs(dx) >= abs(dy):
                            return ('right' if dx > 0 else 'left'), f"Сотрудничаю: иду к другой еде ({tx}, {ty})"
                        else:
                            return ('down' if dy > 0 else 'up'), f"Сотрудничаю: иду к другой еде ({tx}, {ty})"
                
                # Если нет альтернативы — просто исследуем
                action = random.choice(['up', 'down', 'left', 'right'])
                return action, "Сотрудничаю: исследую"
            else:
                # Конкуренция — ускоряемся к цели
                # Используем планировщик для более агрессивной стратегии
                action, plan = self.planner.get_next_action(grid, position, self.world_model)
                if plan:
                    return action, f"Конкурирую! План: {plan.actions[:3]}..."
                
                # Запасной вариант — случайное действие
                action = random.choice(['up', 'down', 'left', 'right'])
                return action, "Конкурирую: действую"
        
        # Нет конкуренции — используем обычное планирование
        action, plan = self.planner.get_next_action(grid, position, self.world_model)
        if plan:
            return action, f"План: {plan.actions[:3]}..."
        
        # Запасной вариант
        action = random.choice(['up', 'down', 'left', 'right'])
        return action, "Исследую"
    
    def learn(self, action: str, context: str, reward: float):
        """Обучает модель на основе полученного опыта."""
        self.world_model.learn_causal_relation(action, context, reward)
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку состояния."""
        return {
            'world_model': self.world_model.get_memory_summary(),
            'planner': self.planner.get_plan_stats(),
            'theory_of_mind': self.theory_of_mind.get_summary(),
        }


# Тест
if __name__ == "__main__":
    print("🧠 Тест теории разума")
    
    from agi_v7.world_model import WorldModel
    from agi_v7.planner import Planner
    
    # Создаём компоненты
    world_model = WorldModel(grid_size=8)
    planner = Planner(horizon=8, num_samples=15)
    theory_of_mind = TheoryOfMind(grid_size=8)
    
    # Создаём агента
    agent = SocialAwareAgent(None, world_model, planner, theory_of_mind)
    
    # Создаём сетку
    grid = [['.' for _ in range(8)] for _ in range(8)]
    grid[2][3] = '🍎'
    grid[5][5] = '🍎'
    grid[6][2] = '⚠️'
    
    # Симулируем других агентов
    other_agents = [
        (1, (1, 1)),
        (2, (7, 7)),
    ]
    
    # Принимаем решение
    action, thought = agent.decide(grid, (0, 0), other_agents)
    print(f"📌 Действие: {action}")
    print(f"   Мысль: {thought}")
    
    print(f"\n📊 Сводка:")
    print(f"   {agent.get_summary()['theory_of_mind']}")
    
    print("✅ Теория разума работает!")
