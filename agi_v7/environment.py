# -*- coding: utf-8 -*-
"""
Тестовая среда для AGI v7 — текстовый мир

Ресурсы:
- 🍎 Еда (восстанавливает энергию)
- 💧 Вода (восстанавливает энергию)
- 🪨 Камень (для строительства)
- 🌳 Дерево (для строительства)

Угрозы:
- 🐺 Хищник (снижает энергию)
- 🔥 Огонь (снижает энергию)
- 🌊 Потоп (снижает энергию)

Состояния:
- Энергия (0-100)
- Голод (0-100)
- Усталость (0-100)
- Опасность (0-1)
"""

import random
from typing import Dict, Any, List, Optional


class Environment:
    """
    Текстовая среда для тестирования AGI v7
    """
    
    def __init__(self):
        # Состояние среды
        self.energy = 100.0
        self.hunger = 0.0
        self.fatigue = 0.0
        self.danger = 0.0
        
        # Ресурсы (еда теперь доступна)
        self.resources = {
            'food': 10,
            'water': 10,
            'stone': 5,
            'wood': 5,
        }
        self.food_available = True  # Еда есть в среде
        
        # Угрозы
        self.threats = {
            'predator': 0.0,
            'fire': 0.0,
            'flood': 0.0,
        }
        
        # История
        self.history = []
        self.step_count = 0
        
        # Параметры
        self.energy_decay = 0.5  # Потеря энергии за шаг
        self.hunger_growth = 0.3  # Рост голода за шаг
        self.fatigue_growth = 0.2  # Рост усталости за шаг
        
        # Случайные события
        self.event_probability = 0.1  # Вероятность события за шаг
    
    def step(self, action: str) -> Dict[str, Any]:
        """
        Выполняет шаг среды на основе действия AGI
        
        Args:
            action: действие AGI ('eat', 'drink', 'explore', 'rest', 'craft', 'defend')
        
        Returns:
            perception: словарь с восприятием среды
        """
        self.step_count += 1
        
        # 1. Применяем действие
        self._apply_action(action)
        
        # 2. Обновляем состояние
        self._update_state()
        
        # 3. Генерируем случайные события
        self._generate_events()
        
        # 4. Создаём восприятие
        perception = self._get_perception()
        
        # 5. Сохраняем историю
        self.history.append({
            'step': self.step_count,
            'action': action,
            'energy': self.energy,
            'hunger': self.hunger,
            'danger': self.danger,
            'resources': self.resources.copy(),
        })
        
        return perception
    
    def _apply_action(self, action: str):
        """Применяет действие AGI к среде"""
        if action == 'eat':
            if self.resources['food'] > 0 and self.hunger > 0:
                self.resources['food'] -= 1
                self.energy = min(100.0, self.energy + 10.0)
                self.hunger = max(0.0, self.hunger - 15.0)
        
        elif action == 'drink':
            if self.resources['water'] > 0 and self.hunger > 0:
                self.resources['water'] -= 1
                self.energy = min(100.0, self.energy + 8.0)
                self.hunger = max(0.0, self.hunger - 10.0)
        
        elif action == 'explore':
            # Исследование: находим ресурсы (больше шанс найти еду)
            found = random.choices(['food', 'water', 'stone', 'wood', 'nothing'], 
                                  weights=[4, 2, 1, 1, 2])[0]
            if found != 'nothing' and self.resources.get(found, 0) < 20:
                self.resources[found] = self.resources.get(found, 0) + 1
            self.fatigue = min(100.0, self.fatigue + 5.0)
        
        elif action == 'rest':
            self.fatigue = max(0.0, self.fatigue - 15.0)
            self.energy = min(100.0, self.energy + 5.0)
        
        elif action == 'craft':
            # Крафт: тратим ресурсы на улучшение
            if self.resources['stone'] > 0 and self.resources['wood'] > 0:
                self.resources['stone'] -= 1
                self.resources['wood'] -= 1
                self.energy = min(100.0, self.energy + 3.0)
                self.fatigue = min(100.0, self.fatigue + 10.0)
        
        elif action == 'defend':
            # Защита: снижаем опасность
            self.danger = max(0.0, self.danger - 0.3)
            self.fatigue = min(100.0, self.fatigue + 8.0)
        
        elif action == 'idle':
            # Ничего не делаем
            pass
    
    def _update_state(self):
        """Обновляет внутреннее состояние среды"""
        # Энергия расходуется
        self.energy = max(0.0, self.energy - self.energy_decay)
        
        # Голод растёт
        self.hunger = min(100.0, self.hunger + self.hunger_growth)
        
        # Усталость растёт
        self.fatigue = min(100.0, self.fatigue + self.fatigue_growth)
        
        # Если голод или усталость высоки, энергия падает быстрее
        if self.hunger > 70:
            self.energy = max(0.0, self.energy - 1.0)
        if self.fatigue > 70:
            self.energy = max(0.0, self.energy - 0.5)
        
        # Если энергия низкая, усталость растёт быстрее
        if self.energy < 30:
            self.fatigue = min(100.0, self.fatigue + 0.5)
    
    def _generate_events(self):
        """Генерирует случайные события"""
        if random.random() < self.event_probability:
            events = [
                ('predator', 0.3),
                ('fire', 0.2),
                ('flood', 0.1),
                ('resource', 0.2),
            ]
            event, intensity = random.choice(events)
            
            if event == 'predator':
                self.danger = min(1.0, self.danger + intensity)
                self.energy = max(0.0, self.energy - intensity * 10)
            elif event == 'fire':
                self.danger = min(1.0, self.danger + intensity * 0.5)
                self.energy = max(0.0, self.energy - intensity * 5)
            elif event == 'flood':
                self.danger = min(1.0, self.danger + intensity * 0.3)
                self.energy = max(0.0, self.energy - intensity * 3)
            elif event == 'resource':
                # Находим ресурсы
                resource = random.choice(['food', 'water', 'stone', 'wood'])
                if self.resources.get(resource, 0) < 20:
                    self.resources[resource] = self.resources.get(resource, 0) + 2
    
    def _get_perception(self) -> Dict[str, Any]:
        """Формирует восприятие для AGI"""
        perception = {
            'objects': [],
            'danger': self.danger > 0.5,
            'energy': self.energy,
            'hunger': self.hunger,
            'fatigue': self.fatigue,
            'resources': self.resources.copy(),
        }
        
        # Добавляем объекты в зависимости от состояния
        if self.resources['food'] > 0:
            perception['objects'].append('food')
        if self.resources['water'] > 0:
            perception['objects'].append('water')
        if self.resources['stone'] > 0:
            perception['objects'].append('stone')
        if self.resources['wood'] > 0:
            perception['objects'].append('wood')
        
        if self.danger > 0.5:
            perception['objects'].append('danger')
        if self.energy < 30:
            perception['objects'].append('low_energy')
        if self.hunger > 70:
            perception['objects'].append('hungry')
        if self.fatigue > 70:
            perception['objects'].append('tired')
        
        return perception
    
    def get_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние среды"""
        return {
            'energy': self.energy,
            'hunger': self.hunger,
            'fatigue': self.fatigue,
            'danger': self.danger,
            'resources': self.resources.copy(),
            'step': self.step_count,
        }
    
    def reset(self):
        """Сбрасывает среду в начальное состояние"""
        self.energy = 100.0
        self.hunger = 0.0
        self.fatigue = 0.0
        self.danger = 0.0
        self.resources = {
            'food': 10,
            'water': 10,
            'stone': 5,
            'wood': 5,
        }
        self.threats = {
            'predator': 0.0,
            'fire': 0.0,
            'flood': 0.0,
        }
        self.history = []
        self.step_count = 0


if __name__ == "__main__":
    # Тест среды
    env = Environment()
    print("🌍 ТЕСТОВАЯ СРЕДА СОЗДАНА")
    print(f"Ресурсы: {env.resources}")
    print(f"Энергия: {env.energy}")
    
    # Тестовые шаги
    actions = ['explore', 'eat', 'drink', 'rest', 'craft', 'explore', 'defend']
    for action in actions:
        perception = env.step(action)
        print(f"\nШаг {env.step_count}: {action}")
        print(f"  Энергия: {env.energy:.1f}")
        print(f"  Голод: {env.hunger:.1f}")
        print(f"  Опасность: {env.danger:.1f}")
        print(f"  Восприятие: {perception['objects']}")
