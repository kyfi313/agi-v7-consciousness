# -*- coding: utf-8 -*-
"""
Моторная кора — планирование и выполнение движений
"""

import numpy as np
from collections import deque
from ...core.base import BaseModule
from ...core.state import GlobalState


class MotorCortexModule(BaseModule):
    name = "motor_cortex"
    
    def __init__(self):
        # Запланированные движения
        self.planned_actions = deque(maxlen=10)
        # Выполненные движения
        self.executed_actions = deque(maxlen=20)
        # Текущее движение
        self.current_action = None
        # Прогресс выполнения (0-1)
        self.progress = 1.0
        # Типы движений
        self.action_types = [
            'explore', 'attack', 'defend', 'eat', 'rest',
            'follow', 'flee', 'investigate', 'communicate', 'build'
        ]
        # Время выполнения
        self.execution_time = 1.0
        self.time_elapsed = 0.0
        
    def update(self, state: GlobalState) -> GlobalState:
        # Получаем решение из базальных ганглий
        action = state.final_action if state.final_action else 'explore'
        
        # Если действие изменилось, планируем новое движение
        if action != self.current_action:
            self._plan_action(action, state)
            self.current_action = action
            self.progress = 0.0
            self.time_elapsed = 0.0
        
        # Выполняем движение
        if self.current_action:
            self._execute_action(state)
            self.executed_actions.append({
                'action': self.current_action,
                'progress': self.progress,
                'step': state.step,
            })
        
        # Сохраняем моторные данные в состояние
        state.motor = {
            'current_action': self.current_action,
            'progress': self.progress,
            'executed_actions': list(self.executed_actions)[-10:],
            'planned_actions': list(self.planned_actions),
        }
        
        # Влияние на тело
        if self.progress >= 1.0:
            # Движение завершено
            state.energy = max(0.0, state.energy - 0.5)  # трата энергии
        
        # Обновляем проприоцепцию
        state.perception['motor_active'] = self.progress < 1.0
        state.perception['motor_action'] = self.current_action
        
        return state
    
    def _plan_action(self, action: str, state: GlobalState):
        """Планирует движение"""
        # Время выполнения зависит от сложности
        complexity = {
            'explore': 0.5,
            'attack': 0.8,
            'defend': 0.6,
            'eat': 0.4,
            'rest': 0.3,
            'follow': 0.7,
            'flee': 0.6,
            'investigate': 0.5,
            'communicate': 0.4,
            'build': 1.2,
        }
        self.execution_time = complexity.get(action, 0.5)
        
        # Добавляем в план
        plan = {
            'action': action,
            'execution_time': self.execution_time,
            'step': state.step,
            'energy_cost': self.execution_time * 0.5,
        }
        self.planned_actions.append(plan)
        
        # Сохраняем план в состоянии
        state.motor_plan = plan
    
    def _execute_action(self, state: GlobalState):
        """Выполняет движение"""
        # Обновляем прогресс
        self.time_elapsed += 0.1  # шаг времени
        self.progress = min(1.0, self.time_elapsed / self.execution_time)
        
        # Эмуляция движения
        if self.progress < 1.0:
            # Движение в процессе
            state.motor_progress = self.progress
            state.motor_state = 'executing'
        else:
            # Движение завершено
            state.motor_progress = 1.0
            state.motor_state = 'completed'
            # Генерируем результат
            result = self._generate_result(state)
            state.motor_result = result
            
            # Обратная связь
            if result.get('success', False):
                state.learning['reward'] += result.get('reward', 0.0)
            else:
                state.learning['reward'] -= 0.2
    
    def _generate_result(self, state: GlobalState) -> dict:
        """Генерирует результат выполненного движения"""
        action = self.current_action
        
        # Эмуляция результатов
        results = {
            'explore': {
                'success': np.random.random() > 0.2,
                'reward': np.random.uniform(0.1, 0.4),
                'message': 'Found something interesting',
            },
            'attack': {
                'success': np.random.random() > 0.4,
                'reward': np.random.uniform(-0.2, 0.5),
                'message': 'Attack performed',
            },
            'defend': {
                'success': np.random.random() > 0.3,
                'reward': np.random.uniform(-0.1, 0.3),
                'message': 'Defense successful',
            },
            'eat': {
                'success': np.random.random() > 0.1,
                'reward': np.random.uniform(0.2, 0.6),
                'message': 'Ate food',
            },
            'rest': {
                'success': True,
                'reward': np.random.uniform(0.1, 0.3),
                'message': 'Rested and recovered',
            },
            'follow': {
                'success': np.random.random() > 0.3,
                'reward': np.random.uniform(-0.1, 0.4),
                'message': 'Following target',
            },
            'flee': {
                'success': np.random.random() > 0.2,
                'reward': np.random.uniform(0.0, 0.3),
                'message': 'Fled from danger',
            },
            'investigate': {
                'success': np.random.random() > 0.3,
                'reward': np.random.uniform(0.1, 0.5),
                'message': 'Investigated area',
            },
            'communicate': {
                'success': np.random.random() > 0.4,
                'reward': np.random.uniform(-0.1, 0.4),
                'message': 'Communication attempted',
            },
            'build': {
                'success': np.random.random() > 0.5,
                'reward': np.random.uniform(0.0, 0.5),
                'message': 'Building in progress',
            },
        }
        
        return results.get(action, {'success': False, 'reward': -0.1, 'message': 'Unknown action'})
    
    def reset(self):
        self.planned_actions.clear()
        self.executed_actions.clear()
        self.current_action = None
        self.progress = 1.0
        self.time_elapsed = 0.0
