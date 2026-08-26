# -*- coding: utf-8 -*-
"""
ВНУТРЕННЯЯ МОДЕЛЬ МИРА

Агент строит карту окружения на основе наблюдений и учится предсказывать:
- Где появляется еда (паттерны)
- Где опасность (статистика)
- Как перемещаются другие агенты
- Причинно-следственные связи

Уровень 4 адаптации: от рефлексов к моделированию мира.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
import random


class WorldModel:
    """
    Внутренняя модель мира агента.
    
    Хранит:
    - Карту местности (память о том, что видел)
    - Паттерны появления еды (временные и пространственные)
    - Статистику опасностей
    - Модели движения других агентов
    - Причинно-следственные связи (действие → результат)
    """
    
    def __init__(self, grid_size: int = 8, memory_steps: int = 50):
        self.grid_size = grid_size
        self.memory_steps = memory_steps
        
        # Карта памяти: что видел на каждой клетке
        # 0 - неизвестно, 1 - пусто, 2 - еда была, 3 - опасность была
        self.memory_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Уверенность в каждой клетке (0-1)
        self.confidence_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # История наблюдений (для поиска паттернов)
        self.observation_history = deque(maxlen=memory_steps)
        
        # Статистика появления еды: (x, y) -> [времена появления]
        self.food_appearance: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # Статистика опасностей
        self.danger_appearance: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # Модели других агентов: id -> {позиции во времени}
        self.agent_trajectories: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # Причинно-следственные связи: (действие, контекст) -> средний результат
        self.causal_memory: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        # Текущий шаг
        self.step = 0
        
        # Предсказания (кэш)
        self.food_prediction = None
        self.danger_prediction = None
        
        # Параметры обучения
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
    
    def get_food_memory(self) -> List[Tuple[float, float]]:
        """Возвращает список запомненных позиций еды."""
        return [(float(x), float(y)) for (x, y) in self.food_appearance.keys()]
        
    def observe(self, grid: List[List[str]], position: Tuple[int, int], 
                other_agents: List[Tuple[int, Tuple[int, int]]]):
        """
        Наблюдает за миром и обновляет модель.
        
        Args:
            grid: Текущая сетка
            position: Позиция агента
            other_agents: Список (id, позиция) других агентов
        """
        self.step += 1
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # Обновляем карту памяти
        for y in range(min(height, self.grid_size)):
            for x in range(min(width, self.grid_size)):
                cell = grid[y][x]
                if cell == '🍎':
                    self.memory_map[y][x] = 2  # еда
                    self.confidence_map[y][x] = min(1.0, self.confidence_map[y][x] + 0.2)
                    self.food_appearance[(x, y)].append(self.step)
                elif cell == '⚠️':
                    self.memory_map[y][x] = 3  # опасность
                    self.confidence_map[y][x] = min(1.0, self.confidence_map[y][x] + 0.2)
                    self.danger_appearance[(x, y)].append(self.step)
                elif cell == '·' or cell == '.':
                    if self.confidence_map[y][x] > 0:
                        # Если видели пустую клетку, уменьшаем уверенность, что там еда/опасность
                        self.confidence_map[y][x] = max(0, self.confidence_map[y][x] - 0.05)
                        if self.memory_map[y][x] in [2, 3]:
                            self.memory_map[y][x] = 1  # пусто
                elif cell == '🤖':
                    # Агент на клетке — обновляем, если это не наш агент
                    pass
        
        # Запоминаем позицию
        self.observation_history.append((position, self.step))
        
        # Обновляем траектории других агентов
        for agent_id, pos in other_agents:
            self.agent_trajectories[agent_id].append(pos)
        
        # Делаем предсказания
        self._predict_food()
        self._predict_danger()
        
    def _predict_food(self):
        """Предсказывает, где появится еда на основе паттернов."""
        if not self.food_appearance:
            self.food_prediction = None
            return
        
        # Ищем клетки с частым появлением еды
        food_counts = {pos: len(times) for pos, times in self.food_appearance.items()}
        if not food_counts:
            self.food_prediction = None
            return
        
        # Учитываем недавность появления (экспоненциальное затухание)
        recent_weight = 0.7
        max_count = max(food_counts.values()) if food_counts else 1
        
        predictions = []
        for (x, y), count in food_counts.items():
            # Чем больше раз появлялась, тем выше вероятность
            base_score = count / max_count
            
            # Чем недавнее, тем выше
            last_time = self.food_appearance[(x, y)][-1]
            recency = 1.0 / (1.0 + (self.step - last_time) / 10.0)
            
            score = base_score * 0.5 + recency * 0.5
            predictions.append((x, y, score))
        
        # Сортируем по убыванию
        predictions.sort(key=lambda p: p[2], reverse=True)
        
        # Берём топ-3
        self.food_prediction = [(x, y, score) for x, y, score in predictions[:3] if score > 0.3]
        
    def _predict_danger(self):
        """Предсказывает, где появится опасность."""
        if not self.danger_appearance:
            self.danger_prediction = None
            return
        
        danger_counts = {pos: len(times) for pos, times in self.danger_appearance.items()}
        if not danger_counts:
            self.danger_prediction = None
            return
        
        max_count = max(danger_counts.values()) if danger_counts else 1
        
        predictions = []
        for (x, y), count in danger_counts.items():
            base_score = count / max_count
            last_time = self.danger_appearance[(x, y)][-1]
            recency = 1.0 / (1.0 + (self.step - last_time) / 10.0)
            score = base_score * 0.5 + recency * 0.5
            predictions.append((x, y, score))
        
        predictions.sort(key=lambda p: p[2], reverse=True)
        self.danger_prediction = [(x, y, score) for x, y, score in predictions[:3] if score > 0.3]
        
    def get_food_hotspots(self) -> List[Tuple[int, int]]:
        """Возвращает список клеток, где вероятно появится еда."""
        if self.food_prediction:
            return [(x, y) for x, y, _ in self.food_prediction]
        return []
        
    def get_danger_zones(self) -> List[Tuple[int, int]]:
        """Возвращает список клеток, где вероятно опасность."""
        if self.danger_prediction:
            return [(x, y) for x, y, _ in self.danger_prediction]
        return []
    
    def get_agent_trajectory(self, agent_id: int) -> List[Tuple[int, int]]:
        """Возвращает траекторию другого агента."""
        return self.agent_trajectories.get(agent_id, [])
    
    def predict_agent_position(self, agent_id: int, steps_ahead: int = 3) -> Optional[Tuple[int, int]]:
        """Предсказывает позицию другого агента через steps_ahead шагов."""
        traj = self.agent_trajectories.get(agent_id, [])
        if len(traj) < 2:
            return None
        
        # Простая экстраполяция: среднее смещение
        dx = 0
        dy = 0
        for i in range(1, min(len(traj), 5)):
            prev = traj[-i-1] if len(traj) > i else traj[0]
            curr = traj[-i]
            dx += curr[0] - prev[0]
            dy += curr[1] - prev[1]
        
        count = min(len(traj) - 1, 4)
        if count == 0:
            return None
        
        dx = int(round(dx / count * steps_ahead))
        dy = int(round(dy / count * steps_ahead))
        
        last_pos = traj[-1]
        pred_x = max(0, min(self.grid_size - 1, last_pos[0] + dx))
        pred_y = max(0, min(self.grid_size - 1, last_pos[1] + dy))
        
        return (pred_x, pred_y)
    
    def learn_causal_relation(self, action: str, context: str, reward: float):
        """
        Запоминает, какое действие в каком контексте привело к какому результату.
        
        Args:
            action: Действие (например, 'move_up', 'eat')
            context: Контекст (например, 'food_nearby', 'danger_nearby')
            reward: Полученная награда
        """
        key = (action, context)
        self.causal_memory[key].append(reward)
        
        # Ограничиваем размер
        if len(self.causal_memory[key]) > 50:
            self.causal_memory[key] = self.causal_memory[key][-50:]
    
    def get_best_action(self, context: str, actions: List[str]) -> Optional[str]:
        """
        Возвращает лучшее действие для данного контекста на основе прошлого опыта.
        """
        best_action = None
        best_reward = -float('inf')
        
        for action in actions:
            key = (action, context)
            if key in self.causal_memory:
                avg_reward = np.mean(self.causal_memory[key])
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_action = action
        
        # С вероятностью exploration_rate выбираем случайное действие
        if random.random() < self.exploration_rate:
            return random.choice(actions) if actions else None
        
        return best_action
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Возвращает краткую сводку модели мира."""
        known_cells = int(np.sum(self.confidence_map > 0.1))
        total_cells = self.grid_size * self.grid_size
        
        return {
            'known_cells': known_cells,
            'known_ratio': known_cells / total_cells,
            'food_hotspots': len(self.food_appearance),
            'danger_zones': len(self.danger_appearance),
            'tracked_agents': len(self.agent_trajectories),
            'causal_rules': len(self.causal_memory),
            'food_prediction': self.food_prediction,
            'danger_prediction': self.danger_prediction,
        }
    
    def __repr__(self) -> str:
        summary = self.get_memory_summary()
        return (f"WorldModel(known={summary['known_cells']}/{summary['known_ratio']:.0%}, "
                f"food_hotspots={summary['food_hotspots']}, "
                f"danger_zones={summary['danger_zones']}, "
                f"rules={summary['causal_rules']})")


class WorldModelAwareAgent:
    """
    Агент, который использует внутреннюю модель мира для принятия решений.
    
    Расширяет TerminalAgent, добавляя:
    - Планирование на основе модели мира
    - Предсказание поведения других агентов
    - Использование причинно-следственных связей
    """
    
    def __init__(self, base_agent, world_model: WorldModel = None):
        self.base_agent = base_agent
        self.world_model = world_model or WorldModel()
        self.step = 0
        
    def observe(self, grid: List[List[str]], position: Tuple[int, int], 
                other_agents: List[Tuple[int, Tuple[int, int]]]):
        """Наблюдает за миром и обновляет модель."""
        self.step += 1
        self.world_model.observe(grid, position, other_agents)
        
    def decide(self, grid: List[List[str]], position: Tuple[int, int],
              other_agents: List[Tuple[int, Tuple[int, int]]]) -> Tuple[str, str]:
        """
        Принимает решение с использованием модели мира.
        """
        # Сначала обновляем наблюдения
        self.observe(grid, position, other_agents)
        
        # Получаем предсказания модели
        food_hotspots = self.world_model.get_food_hotspots()
        danger_zones = self.world_model.get_danger_zones()
        
        # Определяем контекст
        x, y = position
        
        # Проверяем, есть ли еда рядом
        food_nearby = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                    if grid[ny][nx] == '🍎':
                        food_nearby = True
        
        # Проверяем, есть ли опасность рядом
        danger_nearby = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                    if grid[ny][nx] == '⚠️':
                        danger_nearby = True
        
        # Формируем контекст
        if food_nearby and not danger_nearby:
            context = 'food_nearby_safe'
        elif food_nearby and danger_nearby:
            context = 'food_with_danger'
        elif danger_nearby:
            context = 'danger_nearby'
        else:
            # Если есть предсказание еды, направляемся к ней
            if food_hotspots:
                context = 'exploring_for_food'
            else:
                context = 'exploring'
        
        # Используем причинно-следственную память для выбора действия
        actions = ['up', 'down', 'left', 'right', 'wait']
        
        # Если еда рядом, пытаемся съесть
        if food_nearby:
            # Находим направление к ближайшей еде
            min_dist = 10
            best_dir = None
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx] == '🍎':
                            dist = abs(dx) + abs(dy)
                            if dist < min_dist:
                                min_dist = dist
                                if dx == 0 and dy == -1:
                                    best_dir = 'up'
                                elif dx == 0 and dy == 1:
                                    best_dir = 'down'
                                elif dx == -1:
                                    best_dir = 'left'
                                elif dx == 1:
                                    best_dir = 'right'
            if best_dir:
                return best_dir, f"Иду к еде! (модель: {food_hotspots})"
        
        # Если опасность рядом, уходим
        if danger_nearby:
            # Находим направление от опасности
            best_dir = None
            max_dist = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx] == '⚠️':
                            # Противоположное направление
                            opp_dx = -dx
                            opp_dy = -dy
                            if opp_dx == 0 and opp_dy == -1:
                                dir_name = 'up'
                            elif opp_dx == 0 and opp_dy == 1:
                                dir_name = 'down'
                            elif opp_dx == -1:
                                dir_name = 'left'
                            elif opp_dx == 1:
                                dir_name = 'right'
                            else:
                                dir_name = 'wait'
                            
                            # Проверяем, не ведёт ли это к другой опасности
                            nx2, ny2 = x + opp_dx, y + opp_dy
                            if 0 <= nx2 < len(grid[0]) and 0 <= ny2 < len(grid):
                                if grid[ny2][nx2] != '⚠️':
                                    return dir_name, f"Ухожу от опасности! (модель: {danger_zones})"
        
        # Используем предсказания модели для навигации к еде
        if food_hotspots:
            target_x, target_y = food_hotspots[0]
            dx = target_x - x
            dy = target_y - y
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    return 'right', f"Иду к предсказанной еде ({target_x}, {target_y})"
                else:
                    return 'left', f"Иду к предсказанной еде ({target_x}, {target_y})"
            else:
                if dy > 0:
                    return 'down', f"Иду к предсказанной еде ({target_x}, {target_y})"
                else:
                    return 'up', f"Иду к предсказанной еде ({target_x}, {target_y})"
        
        # Если есть причинно-следственная память, используем её
        best_action = self.world_model.get_best_action(context, actions)
        if best_action:
            return best_action, f"Рефлекс: {context} -> {best_action}"
        
        # Случайное исследование
        action = random.choice(actions)
        return action, f"Исследую: {action}"
    
    def learn(self, action: str, context: str, reward: float):
        """Обучает модель на основе полученного опыта."""
        self.world_model.learn_causal_relation(action, context, reward)
    
    def get_world_model_summary(self) -> Dict[str, Any]:
        """Возвращает сводку модели мира."""
        return self.world_model.get_memory_summary()


# Тест
if __name__ == "__main__":
    print("🧠 Тест модели мира")
    
    # Создаём модель
    model = WorldModel(grid_size=8)
    
    # Симулируем наблюдения
    for step in range(20):
        # Создаём случайную сетку
        grid = [['.' for _ in range(8)] for _ in range(8)]
        
        # Добавляем еду в случайные места
        for _ in range(3):
            x = random.randint(0, 7)
            y = random.randint(0, 7)
            if grid[y][x] == '.':
                grid[y][x] = '🍎'
        
        # Добавляем опасности
        for _ in range(2):
            x = random.randint(0, 7)
            y = random.randint(0, 7)
            if grid[y][x] == '.':
                grid[y][x] = '⚠️'
        
        # Наблюдаем
        model.observe(grid, (random.randint(0, 7), random.randint(0, 7)), [])
    
    print(f"📊 Модель: {model}")
    print(f"   Горячие точки еды: {model.get_food_hotspots()}")
    print(f"   Опасные зоны: {model.get_danger_zones()}")
    print("✅ Модель мира работает!")
