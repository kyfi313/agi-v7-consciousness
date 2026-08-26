# -*- coding: utf-8 -*-
"""
ТЕСТ ПОВЕДЕНЧЕСКОЙ ЭВОЛЮЦИИ
Проверяет, как эстафетная эволюция влияет на поведение агента.
Агент получает награду за сбор еды и учится адаптироваться.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
from typing import Dict, Any, Tuple, List
from agi_v7.terminal_agent import TerminalAgent
from agi_v7.relay_evolution import RelayEvolution


class BehaviorTest:
    """Тест поведения агента с подкреплением."""
    
    def __init__(self, grid_size: int = 8, num_steps: int = 100):
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.grid = []
        self.agent_pos = (0, 0)
        self.food_positions = []
        self.danger_positions = []
        self.food_collected = 0
        self.step_rewards = []
        
        # Параметры агента
        self.agent = TerminalAgent(num_neurons=80, connectivity=0.06, input_dim=7)
        
        # Инициализируем эстафетную эволюцию
        self.agent.relay = RelayEvolution()
        
        # Статистика
        self.improvements = []
        self.energy_history = []
        self.food_history = []
        self.action_history = []
        
        self._init_grid()
        
    def _init_grid(self):
        """Инициализирует сетку с едой и опасностями."""
        self.grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Агент в центре
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.agent_pos = (cx, cy)
        self.grid[cy][cx] = '🤖'
        
        # Еда (5-7 штук)
        num_food = random.randint(5, 7)
        self.food_positions = []
        attempts = 0
        while len(self.food_positions) < num_food and attempts < 100:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) != self.agent_pos and (x, y) not in self.food_positions:
                self.food_positions.append((x, y))
                self.grid[y][x] = '🍎'
            attempts += 1
        
        # Опасности (2-3 штуки)
        num_danger = random.randint(2, 3)
        self.danger_positions = []
        attempts = 0
        while len(self.danger_positions) < num_danger and attempts < 100:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) != self.agent_pos and (x, y) not in self.food_positions and (x, y) not in self.danger_positions:
                self.danger_positions.append((x, y))
                self.grid[y][x] = '⚠️'
            attempts += 1
            
    def get_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние для агента."""
        return {
            'grid': self.grid,
            'position': self.agent_pos,
            'energy': self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0,
            'hunger': self.agent.hunger if hasattr(self.agent, 'hunger') else 0.0,
            'food_collected': self.food_collected,
        }
    
    def get_reward(self, action: str) -> float:
        """
        Вычисляет награду за действие.
        
        Награда:
        - +1.0 за сбор еды
        - -0.5 за попадание в опасность
        - -0.01 за каждый шаг (стимул к эффективности)
        - +0.1 за поддержание энергии > 50% (бонус за выживание)
        """
        reward = -0.01  # Базовый штраф за время
        
        # Проверяем, есть ли еда на текущей позиции
        x, y = self.agent_pos
        if (x, y) in self.food_positions:
            reward += 1.0
            self.food_positions.remove((x, y))
            self.grid[y][x] = '.'
            self.food_collected += 1
            if hasattr(self.agent, 'brain'):
                self.agent.brain.network_energy = min(100.0, self.agent.brain.network_energy + 10.0)
        
        # Проверяем опасность
        if (x, y) in self.danger_positions:
            reward -= 0.5
            if hasattr(self.agent, 'brain'):
                self.agent.brain.network_energy = max(0.0, self.agent.brain.network_energy - 15.0)
        
        # Бонус за высокую энергию
        energy = self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0
        if energy > 50:
            reward += 0.05
            
        return reward
    
    def move_agent(self, dx: int, dy: int):
        """Перемещает агента в новую позицию."""
        x, y = self.agent_pos
        nx = max(0, min(self.grid_size - 1, x + dx))
        ny = max(0, min(self.grid_size - 1, y + dy))
        
        # Удаляем старого агента
        self.grid[y][x] = '.'
        
        # Перемещаем агента
        self.agent_pos = (nx, ny)
        self.grid[ny][nx] = '🤖'
        
    def step(self, step_num: int) -> Tuple[str, float]:
        """
        Выполняет один шаг симуляции.
        Возвращает (действие, награда).
        """
        # Получаем состояние
        state = self.get_state()
        
        # Выбираем действие через агента
        action, thought = self.agent.decide(state)
        self.action_history.append(action)
        
        # Выполняем действие
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
            # Исследование — случайное движение
            import random
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = random.choice(directions)
        elif action == 'collect':
            # Сбор еды — пробуем собрать с текущей позиции
            pass
        elif action == 'rest':
            # Отдых — восстанавливаем немного энергии
            if hasattr(self.agent, 'brain'):
                self.agent.brain.network_energy = min(100.0, self.agent.brain.network_energy + 2.0)
        
        # Перемещаем агента
        if dx != 0 or dy != 0:
            self.move_agent(dx, dy)
        
        # Вычисляем награду
        reward = self.get_reward(action)
        self.step_rewards.append(reward)
        
        # Обучаем агента на основе награды
        self.agent.learn_from_outcome(reward, action)
        
        # Получаем текущую энергию
        energy = self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0
        
        # Запускаем эстафетную эволюцию
        relay_data = {
            'action': action,
            'reward': reward,
            'energy': energy,
            'food_collected': self.food_collected,
            'step': step_num,
            'position': self.agent_pos,
            'grid': self.grid,
        }
        improved = self.agent.relay.process(relay_data)
        
        # Запоминаем улучшения
        if improved:
            self.improvements.append((step_num, improved))
        
        # Обновляем энергию агента (базовое потребление)
        if hasattr(self.agent, 'brain'):
            self.agent.brain.network_energy = max(0.0, self.agent.brain.network_energy - 0.5)
        
        # Обновляем статистику
        energy = self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0
        self.energy_history.append(energy)
        self.food_history.append(self.food_collected)
        
        return action, reward
    
    def run(self):
        """Запускает тест."""
        print("=" * 70)
        print("🧪 ТЕСТ ПОВЕДЕНЧЕСКОЙ ЭВОЛЮЦИИ")
        print("=" * 70)
        print(f"📋 Сетка: {self.grid_size}x{self.grid_size}")
        print(f"🍎 Еды: {len(self.food_positions)}")
        print(f"⚠️ Опасностей: {len(self.danger_positions)}")
        print(f"🔄 Шагов: {self.num_steps}")
        print("-" * 70)
        
        for step in range(1, self.num_steps + 1):
            action, reward = self.step(step)
            
            # Выводим прогресс каждые 10 шагов
            if step % 10 == 0:
                energy = self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0
                print(f"\n📊 [Шаг {step}] Статус:")
                print(f"  Энергия: {energy:.1f}")
                print(f"  Еды собрано: {self.food_collected}")
                print(f"  Последнее действие: {action}")
                print(f"  Награда: {reward:.2f}")
                print(f"  Fitness: {self.agent.relay}")
                
                # Показываем карту
                self._print_grid()
            
            # Если энергия упала до нуля, агент умирает
            energy = self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0
            if energy <= 0:
                print(f"\n💀 Агент умер на шаге {step}!")
                break
        
        self._print_summary()
        
    def _print_grid(self):
        """Печатает карту."""
        print("  " + " ".join(str(i) for i in range(self.grid_size)))
        for y, row in enumerate(self.grid):
            print(f"{y} " + " ".join(row))
            
    def _print_summary(self):
        """Печатает итоговый отчёт."""
        print("\n" + "=" * 70)
        print("📊 ИТОГОВЫЙ ОТЧЁТ")
        print("=" * 70)
        
        energy = self.agent.brain.network_energy if hasattr(self.agent, 'brain') else 100.0
        print(f"\n📈 Всего улучшений: {len(self.improvements)}")
        print(f"🍎 Еды собрано: {self.food_collected}")
        print(f"⚡ Финальная энергия: {energy:.1f}")
        print(f"📊 Средняя награда: {sum(self.step_rewards) / max(1, len(self.step_rewards)):.3f}")
        
        print("\n📌 Эволюция модулей:")
        summary = str(self.agent.relay)
        print(f"  {summary}")
        
        print("\n🔍 АНАЛИЗ:")
        if self.food_collected > 0:
            print("  ✅ Агент научился собирать еду!")
        else:
            print("  ⚠️ Агент не собрал ни одной еды.")
            
        if len(self.improvements) > 0:
            print(f"  ✅ Зафиксировано {len(self.improvements)} улучшений.")
        else:
            print("  ⚠️ Улучшений не зафиксировано.")
            
        print("=" * 70)


def main():
    """Точка входа."""
    test = BehaviorTest(grid_size=8, num_steps=100)
    test.run()


if __name__ == "__main__":
    main()
