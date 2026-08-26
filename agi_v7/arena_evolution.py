# -*- coding: utf-8 -*-
"""
АРЕНА ЭВОЛЮЦИИ
Конкуренция агентов с естественным отбором.

Принцип работы:
1. Несколько агентов живут на одной арене
2. Каждый агент имеет свою эстафетную эволюцию
3. Лучшие агенты передают свои конфигурации следующему поколению
4. Естественный отбор ускоряет эволюцию

Это добавляет третий уровень адаптации:
- Уровень 1: Эстафетная эволюция (модули)
- Уровень 2: Глобальный Глаз (усиление модулей)
- Уровень 3: Арена (отбор агентов)
"""

import random
import copy
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import time

from agi_v7.terminal_agent import TerminalAgent
from agi_v7.global_overseer import AdaptiveRelayEvolution, patch_evolvable_module
from agi_v7.relay_evolution import RelayEvolution
from agi_v7.world_model import WorldModel, WorldModelAwareAgent
from agi_v7.planner import Planner, PlanningAwareAgent
from agi_v7.theory_of_mind import TheoryOfMind, SocialAwareAgent
from agi_v7.metacognition import Metacognition, MetaCognitiveAgent
from agi_v7.communication import CommunicationModule, CommunicatingAgent
from agi_v7.meta_learning import MetaLearning, MetaLearningAgent
from agi_v7.consciousness import ConsciousnessModule
from agi_v7.consciousness_dispatcher import ConsciousAgent

# Патчим модули для поддержки Глобального Глаза
patch_evolvable_module()


class ArenaAgent:
    """
    Агент на арене. Имеет свою эстафетную эволюцию и Глобальный Глаз.
    """
    
    def __init__(self, agent_id: int, config: Dict[str, Any] = None, arena=None):
        self.agent_id = agent_id
        self.config = config or {}
        self.arena = arena
        self.comm_module = None
        self.meta_learning = None
        self.consciousness = None
        
        # Создаём агента
        base_agent = TerminalAgent(
            num_neurons=config.get('num_neurons', 80),
            connectivity=config.get('connectivity', 0.06),
            input_dim=7
        )
        
        # Создаём модель мира
        world_model = WorldModel(grid_size=8)
        
        # Уровень 4: Модель мира
        world_agent = WorldModelAwareAgent(base_agent, world_model)
        
        # Уровень 5: Планировщик
        planner = Planner(horizon=config.get('horizon', 10), num_samples=20)
        planning_agent = PlanningAwareAgent(world_agent, world_model, planner)
        
        # Уровень 6: Теория разума
        theory_of_mind = TheoryOfMind(grid_size=8)
        social_agent = SocialAwareAgent(planning_agent, world_model, planner, theory_of_mind)
        
        # Уровень 7: Мета-познание
        metacognition = Metacognition()
        meta_agent = MetaCognitiveAgent(social_agent, world_model, planner, theory_of_mind, metacognition)
        
        # Уровень 8: Коммуникация
        self.comm_module = CommunicationModule(broadcast_range=5)
        comm_agent = CommunicatingAgent(meta_agent, self.comm_module, agent_id)
        
        # Уровень 9: Мета-обучение
        self.meta_learning = MetaLearning()
        meta_agent_full = MetaLearningAgent(comm_agent, self.meta_learning)
        
        # Уровень 10: Сознание (с эволюцией)
        self.consciousness = ConsciousnessModule()
        
        # Финальный агент с сознанием - используем эволюционирующий диспетчер
        # Передаём флаг, чтобы создать эволюционирующий диспетчер
        self.agent = ConsciousAgent(use_evolution=True)
        # Сохраняем мета-агента для использования в step
        self.meta_agent = meta_agent_full
        # Сохраняем ссылку на диспетчер для оценки fitness (если он эволюционирующий)
        if hasattr(self.agent, 'dispatcher') and hasattr(self.agent.dispatcher, 'evaluate_fitness'):
            self.dispatcher = self.agent.dispatcher
        else:
            self.dispatcher = None
        
        # Адаптивная эстафетная эволюция (передаём в базового агента)
        self.relay = AdaptiveRelayEvolution()
        base_agent.relay = self.relay
        
        # Состояние на арене
        self.position = (0, 0)
        self.energy = 100.0
        self.food_collected = 0
        self.alive = True
        self.age = 0
        
        # Статистика
        self.food_history = []
        self.energy_history = []
        self.action_history = []
        self.reward_history = []
        
        # Параметры мутации при размножении
        self.mutation_rate = 0.1
        
    def get_state(self) -> Dict[str, Any]:
        """Возвращает состояние агента."""
        return {
            'grid': [],  # Заполняется извне
            'position': self.position,
            'energy': self.energy,
            'hunger': self.agent.hunger if hasattr(self.agent, 'hunger') else 0.0,
            'food_collected': self.food_collected,
            'agent_id': self.agent_id,
        }
    
    def decide(self, grid: List[List[str]]) -> Tuple[str, str]:
        """Принимает решение на основе восприятия."""
        # Получаем позиции других агентов
        other_agents = []
        for a in self.arena.agents if hasattr(self, 'arena') else []:
            if a.agent_id != self.agent_id and a.alive:
                other_agents.append((a.agent_id, a.position))
        
        # Используем модель мира для принятия решения
        action, thought = self.agent.decide(grid, self.position, other_agents)
        return action, thought
    
    def _build_perception(self, grid: List[List[str]]) -> Dict[str, Any]:
        """Строит восприятие из сетки."""
        x, y = self.position
        height, width = len(grid), len(grid[0]) if grid else 0
        
        # Проверяем, что рядом
        food_nearby = False
        danger_nearby = False
        min_food_dist = 10
        min_danger_dist = 10
        food_count = 0
        danger_count = 0
        
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    cell = grid[ny][nx]
                    dist = abs(dx) + abs(dy)
                    if cell == '🍎':
                        food_count += 1
                        if dist < min_food_dist:
                            min_food_dist = dist
                        if dist <= 1:
                            food_nearby = True
                    elif cell == '⚠️':
                        danger_count += 1
                        if dist < min_danger_dist:
                            min_danger_dist = dist
                        if dist <= 1:
                            danger_nearby = True
        
        return {
            'energy': self.energy,
            'food_nearby': food_nearby,
            'danger_nearby': danger_nearby,
            'min_food_dist': min_food_dist,
            'min_danger_dist': min_danger_dist,
            'food_count': food_count,
            'danger_count': danger_count,
        }
    
    def step(self, grid: List[List[str]]) -> Tuple[str, float, Tuple[int, int]]:
        """
        Выполняет шаг на арене.
        Возвращает (действие, награда, новая позиция).
        """
        if not self.alive:
            return 'dead', -10.0, self.position
        
        self.age += 1
        
        # Решение
        action, thought = self.decide(grid)
        self.action_history.append(action)
        
        # Движение
        dx, dy = 0, 0
        if action == 'up':
            dy = -1
        elif action == 'down':
            dy = 1
        elif action == 'left':
            dx = -1
        elif action == 'right':
            dx = 1
        elif action == 'explore':
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = random.choice(directions)
        elif action == 'rest':
            self.energy = min(100.0, self.energy + 2.0)
        
        # Перемещение
        if dx != 0 or dy != 0:
            x, y = self.position
            nx = max(0, min(len(grid[0]) - 1, x + dx))
            ny = max(0, min(len(grid) - 1, y + dy))
            self.position = (nx, ny)
        
        # Награда
        reward = self._calculate_reward(grid)
        self.reward_history.append(reward)
        
        # Обучение (через модель мира)
        if hasattr(self.agent, 'learn'):
            # Определяем контекст
            x, y = self.position
            food_nearby = any(grid[ny][nx] == '🍎' for dx in [-1,0,1] for dy in [-1,0,1] 
                             if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
            danger_nearby = any(grid[ny][nx] == '⚠️' for dx in [-1,0,1] for dy in [-1,0,1] 
                               if 0 <= (nx:=x+dx) < len(grid[0]) and 0 <= (ny:=y+dy) < len(grid))
            context = 'food_nearby_safe' if food_nearby and not danger_nearby else \
                      'food_with_danger' if food_nearby and danger_nearby else \
                      'danger_nearby' if danger_nearby else 'exploring'
            self.agent.learn(action, context, reward)
        
        # Эстафетная эволюция
        relay_data = {
            'action': action,
            'reward': reward,
            'energy': self.energy,
            'food_collected': self.food_collected,
            'step': self.age,
            'position': self.position,
            'grid': grid,
        }
        self.relay.process(relay_data)
        
        # Базовое потребление энергии
        self.energy = max(0.0, self.energy - 0.3)
        
        # Голод
        if hasattr(self.agent, 'hunger'):
            self.agent.hunger = min(1.0, self.agent.hunger + 0.01)
        
        # Проверка на смерть
        if self.energy <= 0:
            self.alive = False
            # Оцениваем fitness диспетчера перед смертью
            if hasattr(self, 'dispatcher') and hasattr(self.dispatcher, 'evaluate_fitness'):
                survival = 1.0 if self.age > 30 else 0.0
                self.dispatcher.evaluate_fitness(
                    food_collected=self.food_collected,
                    energy=self.energy,
                    survival=survival
                )
        
        # ОНЛАЙН-ОЦЕНКА: оцениваем диспетчера каждые 10 шагов
        if self.age % 10 == 0 and hasattr(self, 'dispatcher') and hasattr(self.dispatcher, 'evaluate_fitness'):
            survival = 1.0 if self.energy > 30 else 0.0
            self.dispatcher.evaluate_fitness(
                food_collected=self.food_collected,
                energy=self.energy,
                survival=survival
            )
        
        # Статистика
        self.energy_history.append(self.energy)
        self.food_history.append(self.food_collected)
        
        return action, reward, self.position
    
    def _calculate_reward(self, grid: List[List[str]]) -> float:
        """Вычисляет награду за текущее состояние."""
        reward = -0.01
        x, y = self.position
        
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            if grid[y][x] == '🍎':
                reward += 1.0
                self.food_collected += 1
                self.energy = min(100.0, self.energy + 10.0)
                # Удаляем еду с карты (будет обработано ареной)
            elif grid[y][x] == '⚠️':
                reward -= 0.5
                self.energy = max(0.0, self.energy - 15.0)
        
        if self.energy > 50:
            reward += 0.02
        
        return reward
    
    def get_fitness(self) -> float:
        """
        Вычисляет приспособленность агента.
        Основные критерии: еда, энергия, выживание.
        """
        food_score = self.food_collected * 10.0
        energy_score = self.energy / 100.0 * 5.0
        survival_score = self.age / 10.0
        
        # Бонус за эволюционные улучшения
        relay = self.relay.relay if hasattr(self.relay, 'relay') else self.relay
        improvements = len(relay.get_improvements()) if hasattr(relay, 'get_improvements') else 0
        evolution_score = improvements * 0.1
        
        return food_score + energy_score + survival_score + evolution_score
    
    def get_summary(self) -> str:
        """Возвращает краткую сводку."""
        return (f"🤖 Агент {self.agent_id}: "
                f"🍎{self.food_collected} "
                f"⚡{self.energy:.1f} "
                f"📊{self.get_fitness():.1f} "
                f"{'✅' if self.alive else '💀'}")


class EvolutionArena:
    """
    Арена эволюции — несколько агентов соревнуются за выживание.
    """
    
    def __init__(
        self,
        grid_size: int = 16,
        num_agents: int = 6,
        num_steps: int = 100,
        generations: int = 5,
        food_count: int = 15,
        danger_count: int = 8
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.generations = generations
        self.food_count = food_count
        self.danger_count = danger_count
        
        self.grid = []
        self.food_positions = []
        self.danger_positions = []
        self.agents: List[ArenaAgent] = []
        self.generation = 0
        
        # Статистика по поколениям
        self.generation_stats = []
        
    def _init_grid(self):
        """Инициализирует сетку арены."""
        self.grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Еда
        self.food_positions = []
        attempts = 0
        while len(self.food_positions) < self.food_count and attempts < 200:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in self.food_positions:
                # Проверяем, что там нет агента
                occupied = any(a.position == (x, y) for a in self.agents)
                if not occupied:
                    self.food_positions.append((x, y))
                    self.grid[y][x] = '🍎'
            attempts += 1
        
        # Опасности
        self.danger_positions = []
        attempts = 0
        while len(self.danger_positions) < self.danger_count and attempts < 200:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in self.food_positions and (x, y) not in self.danger_positions:
                occupied = any(a.position == (x, y) for a in self.agents)
                if not occupied:
                    self.danger_positions.append((x, y))
                    self.grid[y][x] = '⚠️'
            attempts += 1
    
    def _init_agents(self, parent_configs: List[Dict[str, Any]] = None):
        """Инициализирует или воссоздаёт агентов."""
        self.agents = []
        
        if parent_configs:
            # Размножение от лучших родителей
            for i in range(self.num_agents):
                # Выбираем родителя
                parent = random.choice(parent_configs)
                # Мутируем конфигурацию
                child_config = self._mutate_config(parent)
                agent = ArenaAgent(i, child_config, arena=self)
                # Передаём частичную эволюционную историю
                self.agents.append(agent)
            
            # Наследуем лучшие веса диспетчера от лучшего родителя
            best_parent = max(parent_configs, key=lambda p: p.get('fitness', 0))
            if 'dispatcher_weights' in best_parent:
                best_weights = best_parent['dispatcher_weights']
                for agent in self.agents:
                    if hasattr(agent, 'dispatcher') and hasattr(agent.dispatcher, 'restore_best'):
                        agent.dispatcher.best_weights = best_weights.copy()
                        agent.dispatcher.restore_best()
                        agent.dispatcher.best_fitness = best_parent.get('fitness', 0)
        else:
            # Первое поколение — случайные агенты
            for i in range(self.num_agents):
                config = {
                    'num_neurons': random.randint(60, 100),
                    'connectivity': random.uniform(0.04, 0.08),
                }
                agent = ArenaAgent(i, config, arena=self)
                self.agents.append(agent)
        
        # Размещаем агентов на арене
        self._place_agents()
    
    def _place_agents(self):
        """Размещает агентов на свободных клетках."""
        positions = []
        for agent in self.agents:
            attempts = 0
            while attempts < 100:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                # Жёсткая проверка координат
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if (x, y) not in positions and (x, y) not in self.food_positions and (x, y) not in self.danger_positions:
                        agent.position = (x, y)
                        positions.append((x, y))
                        # Проверяем, что grid имеет нужную размерность
                        if len(self.grid) > y and len(self.grid[y]) > x:
                            self.grid[y][x] = '🤖'
                            break
                attempts += 1
            else:
                # Если не нашли место, ставим в (0,0) — но это аварийно
                agent.position = (0, 0)
                if len(self.grid) > 0 and len(self.grid[0]) > 0:
                    self.grid[0][0] = '🤖'
    
    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Мутирует конфигурацию агента."""
        new_config = config.copy()
        
        if random.random() < 0.3:
            new_config['num_neurons'] = int(max(40, min(120, 
                config.get('num_neurons', 80) + random.randint(-10, 10))))
        
        if random.random() < 0.3:
            new_config['connectivity'] = max(0.02, min(0.12, 
                config.get('connectivity', 0.06) + random.uniform(-0.01, 0.01)))
        
        if random.random() < 0.3:
            new_config['horizon'] = int(max(5, min(30, 
                config.get('horizon', 10) + random.randint(-3, 3))))
        
        return new_config
    
    def _collect_food(self):
        """Обновляет сетку — удаляет съеденную еду."""
        for x, y in self.food_positions:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                if self.grid[y][x] == '🍎':
                    # Проверяем, есть ли агент на этой клетке
                    agent_here = any(a.position == (x, y) and a.alive for a in self.agents)
                    if not agent_here:
                        self.grid[y][x] = '🍎'
                    else:
                        self.grid[y][x] = '🤖'
    
    def _update_grid(self):
        """Обновляет сетку для отображения."""
        # Очищаем сетку
        self.grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Восстанавливаем еду и опасности
        for x, y in self.food_positions:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                self.grid[y][x] = '🍎'
        for x, y in self.danger_positions:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                self.grid[y][x] = '⚠️'
        
        # Размещаем агентов
        for agent in self.agents:
            if agent.alive:
                x, y = agent.position
                if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                    self.grid[y][x] = '🤖'
    
    def _print_grid(self):
        """Печатает сетку с агентами."""
        print("  " + " ".join(str(i) for i in range(self.grid_size)))
        for y, row in enumerate(self.grid):
            # Добавляем номера агентов на карту
            row_str = []
            for x, cell in enumerate(row):
                if cell == '🤖':
                    # Находим ID агента
                    agent = next((a for a in self.agents if a.position == (x, y) and a.alive), None)
                    if agent:
                        row_str.append(str(agent.agent_id))
                    else:
                        row_str.append('🤖')
                else:
                    row_str.append(cell)
            print(f"{y} " + " ".join(row_str))
    
    def run(self):
        """Запускает эволюцию на арене."""
        print("=" * 70)
        print("🏟️ АРЕНА ЭВОЛЮЦИИ")
        print("   Естественный отбор агентов")
        print("=" * 70)
        print(f"📋 Сетка: {self.grid_size}x{self.grid_size}")
        print(f"🤖 Агентов: {self.num_agents}")
        print(f"🍎 Еды: {self.food_count}")
        print(f"⚠️ Опасностей: {self.danger_count}")
        print(f"🔄 Шагов на поколение: {self.num_steps}")
        print(f"🧬 Поколений: {self.generations}")
        print("-" * 70)
        
        best_agent = None
        best_fitness = -float('inf')
        
        for gen in range(self.generations):
            self.generation = gen + 1
            print(f"\n🧬 ПОКОЛЕНИЕ {self.generation}/{self.generations}")
            print("-" * 40)
            
            # Инициализируем агентов
            if gen == 0:
                self._init_agents()
            else:
                # Отбираем лучших для размножения
                sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness(), reverse=True)
                top_agents = sorted_agents[:max(1, self.num_agents // 2)]
                parent_configs = [a.config for a in top_agents]
                self._init_agents(parent_configs)
            
            self._init_grid()
            self._update_grid()
            
            # Симуляция
            for step in range(1, self.num_steps + 1):
                # Каждый агент делает шаг
                for agent in self.agents:
                    if agent.alive:
                        action, reward, pos = agent.step(self.grid)
                        # Если агент съел еду, удаляем её
                        if action == 'collect' or (action == 'explore' and reward > 0.5):
                            x, y = pos
                            if (x, y) in self.food_positions:
                                self.food_positions.remove((x, y))
                
                # Обновляем сетку
                self._update_grid()
                
                # Выводим статус каждые 10 шагов
                if step % 10 == 0:
                    print(f"\n📊 [Шаг {step}]")
                    alive = sum(1 for a in self.agents if a.alive)
                    total_food = sum(a.food_collected for a in self.agents)
                    print(f"  Живых: {alive}/{self.num_agents}")
                    print(f"  Всего еды собрано: {total_food}")
                    print(f"  Позиции: {[(a.agent_id, a.position) for a in self.agents if a.alive]}")
                    self._print_grid()
            
            # Оценка поколения
            print(f"\n📊 ИТОГИ ПОКОЛЕНИЯ {self.generation}")
            print("-" * 40)
            
            sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness(), reverse=True)
            for i, agent in enumerate(sorted_agents[:3]):
                print(f"  #{i+1} {agent.get_summary()}")
            
            # Сохраняем статистику
            gen_stats = {
                'generation': self.generation,
                'best_fitness': sorted_agents[0].get_fitness() if sorted_agents else 0,
                'avg_fitness': sum(a.get_fitness() for a in self.agents) / self.num_agents,
                'total_food': sum(a.food_collected for a in self.agents),
                'survivors': sum(1 for a in self.agents if a.alive),
            }
            self.generation_stats.append(gen_stats)
            
            # Обновляем лучшего агента
            if sorted_agents and sorted_agents[0].get_fitness() > best_fitness:
                best_fitness = sorted_agents[0].get_fitness()
                best_agent = sorted_agents[0]
        
        # Итоговый отчёт
        self._print_final_report(best_agent)
    
    def _print_final_report(self, best_agent: Optional[ArenaAgent]):
        """Печатает итоговый отчёт."""
        print("\n" + "=" * 70)
        print("🏆 ИТОГОВЫЙ ОТЧЁТ")
        print("=" * 70)
        
        print("\n📊 Эволюция поколений:")
        for stats in self.generation_stats:
            print(f"  Поколение {stats['generation']}: "
                  f"лучший = {stats['best_fitness']:.1f}, "
                  f"средний = {stats['avg_fitness']:.1f}, "
                  f"еды = {stats['total_food']}, "
                  f"выжило = {stats['survivors']}")
        
        if best_agent:
            print(f"\n🏆 ЛУЧШИЙ АГЕНТ:")
            print(f"  {best_agent.get_summary()}")
            print(f"  Параметры: {best_agent.config}")
            
            # Показываем эволюцию модулей
            if hasattr(best_agent.relay, 'relay'):
                print(f"  Эволюция: {best_agent.relay.relay}")
        
        print("\n" + "=" * 70)


def main():
    """Точка входа."""
    arena = EvolutionArena(
        grid_size=16,
        num_agents=6,
        num_steps=100,
        generations=5,
        food_count=15,
        danger_count=8
    )
    arena.run()


if __name__ == "__main__":
    main()
