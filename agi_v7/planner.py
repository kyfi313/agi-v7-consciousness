# -*- coding: utf-8 -*-
"""
ПЛАНИРОВЩИК С ПРЕДСКАЗАНИЕМ

Уровень 5 адаптации: агент строит многошаговые планы и выбирает оптимальный.

Принцип работы:
1. Моделирует возможные действия на N шагов вперёд (горизонт 10-30)
2. Оценивает каждый маршрут по ожидаемой награде
3. Выбирает план с максимальной ценностью
4. Исполняет первый шаг плана, затем перепланирует

Это добавляет способность к долгосрочному мышлению.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Plan:
    """План — последовательность действий."""
    actions: List[str]
    predicted_reward: float
    predicted_positions: List[Tuple[int, int]]
    confidence: float = 0.5
    
    def __repr__(self) -> str:
        return f"Plan({self.actions[:3]}..., reward={self.predicted_reward:.2f}, conf={self.confidence:.2f})"


class Planner:
    """
    Планировщик с древовидным поиском.
    
    Использует модель мира для предсказания результатов действий.
    """
    
    def __init__(self, horizon: int = 10, num_samples: int = 20, discount: float = 0.9):
        """
        Args:
            horizon: Горизонт планирования (сколько шагов вперёд)
            num_samples: Количество маршрутов для оценки
            discount: Коэффициент дисконтирования будущих наград
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.discount = discount
        
        # Кэш для ускорения
        self._plan_cache: Dict[str, Plan] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def plan(self, 
             grid: List[List[str]], 
             position: Tuple[int, int],
             world_model,
             goal: str = 'collect_food') -> Optional[Plan]:
        """
        Строит план действий на основе модели мира.
        
        Args:
            grid: Текущая сетка
            position: Текущая позиция агента
            world_model: Модель мира (для предсказаний)
            goal: Цель ('collect_food', 'avoid_danger', 'explore')
        
        Returns:
            Лучший план или None, если планирование невозможно
        """
        # Генерируем кандидатов
        candidates = self._generate_candidates(grid, position, world_model, goal)
        
        if not candidates:
            return None
        
        # Оцениваем каждого кандидата
        scored_candidates = []
        for actions in candidates:
            reward, positions = self._simulate_path(grid, position, actions, world_model, goal)
            scored_candidates.append((actions, reward, positions))
        
        # Выбираем лучший
        if not scored_candidates:
            return None
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_actions, best_reward, best_positions = scored_candidates[0]
        
        # Вычисляем уверенность (разница между лучшим и средним)
        rewards = [r for _, r, _ in scored_candidates]
        confidence = 0.5 + 0.5 * (best_reward - np.mean(rewards)) / (max(1, np.std(rewards) + 1e-6))
        confidence = max(0.0, min(1.0, confidence))
        
        plan = Plan(
            actions=best_actions,
            predicted_reward=best_reward,
            predicted_positions=best_positions,
            confidence=confidence
        )
        
        # Кэшируем
        cache_key = f"{position}_{goal}_{self.horizon}"
        self._plan_cache[cache_key] = plan
        
        return plan
    
    def _generate_candidates(self, grid: List[List[str]], position: Tuple[int, int],
                             world_model, goal: str) -> List[List[str]]:
        """Генерирует кандидатов — последовательности действий."""
        candidates = []
        actions = ['up', 'down', 'left', 'right']
        
        # Базовый случай: если нет модели мира, просто случайные маршруты
        has_world_model = world_model is not None and hasattr(world_model, 'get_food_hotspots')
        
        # 1. Случайные маршруты
        for _ in range(max(1, self.num_samples // 2)):
            path = []
            pos = position
            for _ in range(self.horizon):
                if has_world_model and random.random() < 0.7:
                    action = self._get_goal_directed_action(grid, pos, world_model, goal)
                else:
                    action = random.choice(actions)
                path.append(action)
                pos = self._apply_action(pos, action, grid)
            candidates.append(path)
        
        # 2. Варианты с повторением одного действия
        for action in actions:
            path = [action] * self.horizon
            candidates.append(path)
        
        # 3. Если нет модели мира, добавляем простые маршруты: все в одном направлении
        if not has_world_model:
            for action in actions:
                path = [action] * (self.horizon // 2) + [random.choice(actions)] * (self.horizon // 2)
                candidates.append(path)
        
        # Дедупликация
        unique_candidates = []
        seen = set()
        for path in candidates:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(path)
        
        # Гарантируем, что есть хотя бы один кандидат
        if not unique_candidates:
            unique_candidates = [[random.choice(actions)] * self.horizon]
        
        return unique_candidates[:self.num_samples]
    
    def _get_goal_directed_action(self, grid: List[List[str]], position: Tuple[int, int],
                                   world_model, goal: str) -> str:
        """Возвращает действие, направленное к цели."""
        x, y = position
        height, width = len(grid), len(grid[0]) if grid else 0
        
        if goal == 'collect_food':
            # Используем модель мира для поиска горячих точек
            hotspots = world_model.get_food_hotspots() if world_model else []
            
            # Если есть горячие точки, идём к ближайшей
            if hotspots:
                target = min(hotspots, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                dx = target[0] - x
                dy = target[1] - y
                return self._direction_to_action(dx, dy)
            
            # Ищем еду на сетке
            min_dist = float('inf')
            best_dir = random.choice(['up', 'down', 'left', 'right'])
            for ny in range(height):
                for nx in range(width):
                    if grid[ny][nx] == '🍎':
                        dist = abs(nx - x) + abs(ny - y)
                        if dist < min_dist:
                            min_dist = dist
                            dx = nx - x
                            dy = ny - y
                            best_dir = self._direction_to_action(dx, dy)
            return best_dir
            
        elif goal == 'avoid_danger':
            # Идём от опасности
            max_dist = 0
            best_dir = random.choice(['up', 'down', 'left', 'right'])
            for ny in range(height):
                for nx in range(width):
                    if grid[ny][nx] == '⚠️':
                        dist = abs(nx - x) + abs(ny - y)
                        if dist > max_dist:
                            max_dist = dist
                            dx = x - nx  # Противоположное направление
                            dy = y - ny
                            best_dir = self._direction_to_action(dx, dy)
            return best_dir
        
        # По умолчанию — исследование
        return random.choice(['up', 'down', 'left', 'right'])
    
    def _direction_to_action(self, dx: int, dy: int) -> str:
        """Преобразует дельту координат в действие."""
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def _apply_action(self, position: Tuple[int, int], action: str, grid: List[List[int]]) -> Tuple[int, int]:
        """Применяет действие и возвращает новую позицию."""
        x, y = position
        height, width = len(grid), len(grid[0]) if grid else 0
        
        dx, dy = 0, 0
        if action == 'up':
            dy = -1
        elif action == 'down':
            dy = 1
        elif action == 'left':
            dx = -1
        elif action == 'right':
            dx = 1
        
        nx = max(0, min(width - 1, x + dx))
        ny = max(0, min(height - 1, y + dy))
        return (nx, ny)
    
    def _simulate_path(self, grid: List[List[int]], start_pos: Tuple[int, int],
                       actions: List[str], world_model, goal: str) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Симулирует выполнение плана и возвращает ожидаемую награду.
        """
        total_reward = 0.0
        positions = [start_pos]
        pos = start_pos
        
        for step, action in enumerate(actions):
            # Применяем действие
            pos = self._apply_action(pos, action, grid)
            positions.append(pos)
            
            # Вычисляем награду за этот шаг
            reward = self._get_reward_for_position(grid, pos, step, goal)
            total_reward += reward * (self.discount ** step)
            
            # Если достигли цели, можно остановиться
            if reward > 1.0:
                break
        
        # Штраф за стояние на месте или петли
        if len(set(positions)) < len(positions) // 2:
            total_reward -= 0.5
        
        # Бонус за исследование новых клеток
        unique_positions = len(set(positions))
        total_reward += 0.01 * unique_positions
        
        return total_reward, positions
    
    def _get_reward_for_position(self, grid: List[List[int]], position: Tuple[int, int],
                                  step: int, goal: str) -> float:
        """Вычисляет награду за нахождение в позиции."""
        x, y = position
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            cell = grid[y][x]
            
            if goal == 'collect_food':
                if cell == 1:  # еда
                    return 2.0
                elif cell == 2:  # опасность
                    return -0.5
            elif goal == 'avoid_danger':
                if cell == 2:
                    return -1.0
                elif cell == 1:
                    return 0.5
        
        # Маленькое положительное вознаграждение за движение
        return 0.01
    
    def get_next_action(self, grid: List[List[str]], position: Tuple[int, int],
                        world_model, goal: str = 'collect_food') -> Tuple[Optional[str], Optional[Plan]]:
        """
        Возвращает следующее действие и план.
        
        Использует планирование для выбора оптимального действия.
        """
        # Проверяем кэш
        cache_key = f"{position}_{goal}_{self.horizon}"
        if cache_key in self._plan_cache:
            plan = self._plan_cache[cache_key]
            self._cache_hits += 1
            if plan.actions:
                return plan.actions[0], plan
        
        self._cache_misses += 1
        
        # Строим новый план
        plan = self.plan(grid, position, world_model, goal)
        if plan and plan.actions:
            return plan.actions[0], plan
        
        # Запасной вариант — случайное действие
        return random.choice(['up', 'down', 'left', 'right']), None
    
    def get_plan_stats(self) -> Dict[str, Any]:
        """Возвращает статистику планировщика."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'horizon': self.horizon,
            'num_samples': self.num_samples,
        }


class PlanningAwareAgent:
    """
    Агент, использующий планирование для принятия решений.
    
    Расширяет WorldModelAwareAgent, добавляя:
    - Многошаговое планирование
    - Динамическое перепланирование
    - Учёт горизонта планирования
    """
    
    def __init__(self, base_agent, world_model, planner: Planner = None):
        self.base_agent = base_agent
        self.world_model = world_model
        self.planner = planner or Planner(horizon=10, num_samples=20)
        
        # Текущий план
        self.current_plan: Optional[Plan] = None
        self.plan_step = 0
        self.replan_interval = 5  # Перепланируем каждые 5 шагов
        self.steps_since_replan = 0
        
    def decide(self, grid: List[List[str]], position: Tuple[int, int],
               other_agents: List[Tuple[int, Tuple[int, int]]]) -> Tuple[str, str]:
        """
        Принимает решение с использованием планирования.
        """
        # Обновляем модель мира
        self.world_model.observe(grid, position, other_agents)
        
        # Определяем контекст
        x, y = position
        food_nearby = any(grid[ny][nx] == '🍎' for dx in [-1,0,1] for dy in [-1,0,1] 
                         if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
        danger_nearby = any(grid[ny][nx] == '⚠️' for dx in [-1,0,1] for dy in [-1,0,1] 
                           if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
        
        # Определяем цель
        if danger_nearby:
            goal = 'avoid_danger'
        elif food_nearby:
            goal = 'collect_food'
        else:
            # Проверяем, есть ли предсказания еды
            hotspots = self.world_model.get_food_hotspots()
            if hotspots:
                goal = 'collect_food'
            else:
                goal = 'explore'
        
        # Перепланируем при необходимости
        self.steps_since_replan += 1
        should_replan = (
            self.current_plan is None or
            self.steps_since_replan >= self.replan_interval or
            self.plan_step >= len(self.current_plan.actions) - 1
        )
        
        if should_replan:
            action, self.current_plan = self.planner.get_next_action(
                grid, position, self.world_model, goal
            )
            self.plan_step = 0
            self.steps_since_replan = 0
            
            if self.current_plan:
                thought = f"План: {self.current_plan.actions[:3]}... (награда={self.current_plan.predicted_reward:.2f})"
            else:
                thought = f"Исследую: {action}"
        else:
            # Исполняем следующий шаг плана
            self.plan_step += 1
            if self.current_plan and self.plan_step < len(self.current_plan.actions):
                action = self.current_plan.actions[self.plan_step]
                thought = f"Исполняю план: шаг {self.plan_step+1}/{len(self.current_plan.actions)}"
            else:
                # Запасной вариант
                action = random.choice(['up', 'down', 'left', 'right'])
                thought = "Перепланирую..."
        
        return action, thought
    
    def learn(self, action: str, context: str, reward: float):
        """Обучает модель на основе полученного опыта."""
        self.world_model.learn_causal_relation(action, context, reward)
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку состояния."""
        return {
            'world_model': self.world_model.get_memory_summary(),
            'planner': self.planner.get_plan_stats(),
            'current_plan': str(self.current_plan) if self.current_plan else None,
            'plan_step': self.plan_step,
        }


# Тест
if __name__ == "__main__":
    print("🧠 Тест планировщика")
    
    from agi_v7.world_model import WorldModel
    
    # Создаём модель мира
    model = WorldModel(grid_size=8)
    
    # Создаём планировщик
    planner = Planner(horizon=8, num_samples=15)
    
    # Создаём сетку
    grid = [['.' for _ in range(8)] for _ in range(8)]
    grid[2][3] = '🍎'
    grid[5][5] = '🍎'
    grid[6][2] = '⚠️'
    
    # Планируем
    plan = planner.plan(grid, (0, 0), model, goal='collect_food')
    
    if plan:
        print(f"✅ План: {plan}")
        print(f"   Действия: {plan.actions[:5]}...")
        print(f"   Позиции: {plan.predicted_positions[:5]}...")
        print(f"   Уверенность: {plan.confidence:.2f}")
    else:
        print("❌ Не удалось построить план")
    
    # Тестируем получение следующего действия
    action, new_plan = planner.get_next_action(grid, (0, 0), model)
    print(f"\n📌 Следующее действие: {action}")
    if new_plan:
        print(f"   План: {new_plan}")
    
    print("\n📊 Статистика:", planner.get_plan_stats())
    print("✅ Планировщик работает!")
