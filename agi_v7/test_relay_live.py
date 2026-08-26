# -*- coding: utf-8 -*-
"""
Тест интеграции эстафетной эволюции в агента.
Запускается прямо из папки agi_v7, чтобы избежать проблем с импортами.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from terminal_agent import TerminalAgent
from relay_evolution import RelayEvolution


def create_environment():
    grid = [
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '🍎', '.'],
        ['.', '.', '🤖', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '🍎', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '⚠️', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '🍎', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '🍎', '.', '.', '⚠️'],
    ]
    return grid


def run_test():
    print("=" * 70)
    print("🧪 ТЕСТ ЭСТАФЕТНОЙ ЭВОЛЮЦИИ В АГЕНТЕ (LIVE)")
    print("=" * 70)
    print()
    
    agent = TerminalAgent(num_neurons=80, connectivity=0.06, input_dim=7)
    agent.relay = RelayEvolution()
    
    print("📋 Агент инициализирован с эстафетной эволюцией")
    print()
    
    grid = create_environment()
    agent_position = (2, 2)
    energy = 100
    food_collected = 0
    
    print("🔄 Запуск симуляции...")
    print("-" * 70)
    
    improvements_count = 0
    seen_modules = set()
    
    for step in range(1, 31):
        perception = {
            'grid': grid,
            'position': agent_position,
            'energy': energy,
            'food_nearby': False,
            'danger_nearby': False,
            'food_count': 0,
            'danger_count': 0,
        }
        
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == '🍎' and abs(x - agent_position[0]) + abs(y - agent_position[1]) <= 2:
                    perception['food_nearby'] = True
                    perception['food_count'] += 1
                if cell == '⚠️' and abs(x - agent_position[0]) + abs(y - agent_position[1]) <= 2:
                    perception['danger_nearby'] = True
                    perception['danger_count'] += 1
        
        action, thought = agent.decide(perception)
        
        state = {
            'energy': energy,
            'hunger': agent.hunger,
            'fatigue': agent.fatigue,
            'food_nearby': perception['food_nearby'],
            'danger_nearby': perception['danger_nearby'],
            'food_collected': food_collected,
        }
        
        relay_data = {
            'grid': grid,
            'position': agent_position,
            'energy': energy,
            'hunger': agent.hunger,
            'state': state,
            'action': action,
        }
        
        relay_result = agent.relay.process(relay_data)
        
        improvements = agent.relay.get_improvements()
        if improvements:
            for imp in improvements:
                if imp['module'] not in seen_modules:
                    seen_modules.add(imp['module'])
                    print(f"  ✅ [Шаг {step}] {imp['module']} улучшился! Fitness: {imp['fitness']:.3f}")
                    improvements_count += 1
        
        if action == 'collect':
            food_found = False
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell == '🍎' and abs(x - agent_position[0]) + abs(y - agent_position[1]) <= 1:
                        grid[y][x] = '.'
                        food_collected += 1
                        energy = min(100, energy + 30)
                        food_found = True
                        print(f"  🍎 [Шаг {step}] Собрал еду! Всего: {food_collected}")
                        break
                if food_found:
                    break
        
        elif action == 'flee':
            danger_positions = []
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell == '⚠️':
                        danger_positions.append((x, y))
            
            if danger_positions:
                nearest = min(danger_positions, key=lambda p: abs(p[0] - agent_position[0]) + abs(p[1] - agent_position[1]))
                dx = 1 if agent_position[0] < nearest[0] else (-1 if agent_position[0] > nearest[0] else 0)
                dy = 1 if agent_position[1] < nearest[1] else (-1 if agent_position[1] > nearest[1] else 0)
                dx = -dx if dx != 0 else (1 if agent_position[0] < 7 else -1)
                dy = -dy if dy != 0 else (1 if agent_position[1] < 7 else -1)
                agent_position = (max(0, min(7, agent_position[0] + dx)), max(0, min(7, agent_position[1] + dy)))
                print(f"  🏃 [Шаг {step}] Убегаю от опасности!")
        
        elif action == 'rest':
            energy = min(100, energy + 10)
            print(f"  💤 [Шаг {step}] Отдыхаю...")
        
        elif action == 'explore':
            food_positions = []
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell == '🍎':
                        food_positions.append((x, y))
            
            if food_positions:
                nearest = min(food_positions, key=lambda p: abs(p[0] - agent_position[0]) + abs(p[1] - agent_position[1]))
                dx = 1 if nearest[0] > agent_position[0] else (-1 if nearest[0] < agent_position[0] else 0)
                dy = 1 if nearest[1] > agent_position[1] else (-1 if nearest[1] < agent_position[1] else 0)
                agent_position = (max(0, min(7, agent_position[0] + dx)), max(0, min(7, agent_position[1] + dy)))
                print(f"  🔍 [Шаг {step}] Исследую...")
        
        energy = max(0, energy - 2)
        
        if action == 'collect' and food_found:
            agent.learn_from_outcome(1.0, action)
        elif action == 'flee' and perception['danger_nearby']:
            agent.learn_from_outcome(0.5, action)
        else:
            agent.learn_from_outcome(0.0, action)
        
        if step % 10 == 0:
            print(f"\n📊 [Шаг {step}] Статус:")
            print(f"  Энергия: {energy:.1f}, Голод: {agent.hunger:.2f}")
            print(f"  Еды собрано: {food_collected}")
            print(f"  {agent.relay}")
            print()
        
        if energy <= 0:
            print(f"\n💀 Агент умер на шаге {step}!")
            break
    
    print()
    print("=" * 70)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)
    
    print(f"\n📈 Всего улучшений: {improvements_count}")
    print(f"🍎 Еды собрано: {food_collected}")
    print(f"⚡ Финальная энергия: {energy:.1f}")
    
    print("\n📌 Финальные конфигурации:")
    final_configs = agent.relay.get_best_configs()
    for name, config in final_configs.items():
        print(f"  {name}: {config}")
    
    print()
    print("=" * 70)
    
    print("\n🔍 АНАЛИЗ:")
    if improvements_count > 0:
        print(f"  ✅ Эстафетная эволюция работает! Зафиксировано {improvements_count} улучшений.")
    else:
        print("  ❌ Эстафетная эволюция не сработала!")
    
    if food_collected > 0:
        print(f"  ✅ Агент собрал {food_collected} еды!")
    else:
        print("  ⚠️ Агент не собрал ни одной еды.")
    
    print()
    return improvements_count > 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
