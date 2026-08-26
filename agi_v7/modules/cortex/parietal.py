# -*- coding: utf-8 -*-
"""
Париетальная доля — пространственное восприятие, навигация и интеграция сенсорной информации
"""

import numpy as np
from collections import deque
from ...core.base import BaseModule
from ...core.state import GlobalState


class ParietalLobeModule(BaseModule):
    name = "parietal_lobe"

    def __init__(self):
        # Пространственная память (последние 20 позиций)
        self.spatial_memory = deque(maxlen=20)
        # Текущая позиция
        self.position = np.array([0.0, 0.0, 0.0])
        # Направление взгляда
        self.gaze_direction = np.array([0.0, 0.0, 1.0])
        # Карта объектов (позиции объектов в пространстве)
        self.object_map = {}
        # История движений
        self.movement_history = deque(maxlen=10)

    def update(self, state: GlobalState) -> GlobalState:
        # Получаем проприоцептивные и визуальные данные
        thalamus_filtered = state.perception.get('thalamus_filtered', {})
        proprio = thalamus_filtered.get('proprioceptive', {})
        visual = thalamus_filtered.get('visual', {})

        # Обновляем пространственную информацию
        if proprio:
            self._update_position(proprio)

        if visual:
            self._update_object_map(visual)

        # Сохраняем пространственную информацию
        spatial_info = self._get_spatial_info(state)
        self.spatial_memory.append(spatial_info)

        # Сохраняем в состояние
        state.perception['spatial'] = spatial_info
        state.perception['position'] = self.position.tolist()
        state.perception['gaze_direction'] = self.gaze_direction.tolist()
        state.perception['object_map'] = self.object_map

        # Влияние на планирование (пространственные цели)
        if self.spatial_memory:
            state.perception['spatial_context'] = self._get_spatial_context()

        # Обновляем навигационные цели
        if 'action' in state.final_action:
            self._update_navigation(state)

        return state

    def _update_position(self, proprio: dict):
        """Обновляет позицию на основе проприоцептивных данных"""
        pos = proprio.get('position', [0, 0, 0])
        velocity = proprio.get('velocity', [0, 0, 0])

        # Обновляем позицию
        self.position = np.array(pos)
        # Обновляем направление взгляда (поворот)
        rotation = proprio.get('rotation', [0, 0, 0])
        self.gaze_direction = self._rotate_vector(np.array([0.0, 0.0, 1.0]), rotation)

        # Сохраняем в историю
        self.movement_history.append({
            'position': pos.copy(),
            'velocity': velocity.copy(),
            'step': self._get_current_step(),
        })

    def _update_object_map(self, visual: dict):
        """Обновляет карту объектов на основе визуальных данных"""
        objects = visual.get('objects', [])
        for obj in objects:
            if isinstance(obj, dict):
                obj_type = obj.get('type', 'unknown')
                # Вычисляем относительную позицию
                rel_pos = self._compute_relative_position(obj)
                # Абсолютная позиция (относительно агента)
                abs_pos = self.position + rel_pos
                # Сохраняем в карту
                key = f"{obj_type}_{len(self.object_map)}"
                self.object_map[key] = {
                    'type': obj_type,
                    'position': abs_pos.tolist(),
                    'relative': rel_pos.tolist(),
                    'time': self._get_current_step(),
                    'active': obj.get('active', True),
                }

        # Удаляем неактивные объекты
        self.object_map = {k: v for k, v in self.object_map.items() if v.get('active', True)}

    def _compute_relative_position(self, obj: dict) -> np.ndarray:
        """Вычисляет относительную позицию объекта"""
        pos = obj.get('position', [0, 0, 0])
        # Добавляем случайное смещение для реалистичности
        noise = np.random.uniform(-0.1, 0.1, 3)
        return np.array(pos) + noise

    def _rotate_vector(self, vector: np.ndarray, rotation: list) -> np.ndarray:
        """Поворачивает вектор на заданные углы"""
        pitch, yaw, roll = rotation
        # Простое вращение (эмуляция)
        # В реальной системе здесь были бы матрицы вращения
        return vector * np.array([np.cos(yaw), np.sin(pitch), np.cos(roll)])

    def _get_spatial_info(self, state: GlobalState) -> dict:
        """Получает пространственную информацию"""
        return {
            'position': self.position.tolist(),
            'gaze': self.gaze_direction.tolist(),
            'num_objects': len(self.object_map),
            'nearest_object': self._get_nearest_object(),
            'step': state.step,
        }

    def _get_nearest_object(self) -> dict:
        """Находит ближайший объект"""
        if not self.object_map:
            return None

        nearest = None
        min_dist = float('inf')

        for key, obj in self.object_map.items():
            pos = np.array(obj.get('position', [0, 0, 0]))
            dist = np.linalg.norm(pos - self.position)
            if dist < min_dist:
                min_dist = dist
                nearest = {'key': key, 'object': obj, 'distance': dist}

        return nearest

    def _get_spatial_context(self) -> dict:
        """Получает пространственный контекст"""
        if not self.spatial_memory:
            return {'context': 'unknown', 'movement': 'static'}

        last_positions = [self.spatial_memory[i].get('position', [0, 0, 0]) for i in range(-3, 0)]
        if len(last_positions) < 3:
            return {'context': 'unknown', 'movement': 'static'}

        # Вычисляем движение
        start = np.array(last_positions[0])
        end = np.array(last_positions[-1])
        movement = end - start
        distance = np.linalg.norm(movement)

        if distance < 0.1:
            movement_type = 'static'
        elif distance < 0.5:
            movement_type = 'slow'
        elif distance < 1.0:
            movement_type = 'moderate'
        else:
            movement_type = 'fast'

        # Определяем контекст
        context = 'unknown'
        if distance > 0.5:
            context = 'moving'
        elif len(self.object_map) > 5:
            context = 'crowded'
        elif self.object_map:
            context = 'oriented'
        else:
            context = 'empty'

        return {
            'context': context,
            'movement': movement_type,
            'distance_travelled': distance,
            'direction': movement.tolist(),
        }

    def _update_navigation(self, state: GlobalState):
        """Обновляет навигационные цели"""
        action = state.final_action
        # Эмулируем навигацию
        if action in ['explore', 'investigate']:
            # Исследуем случайное направление
            new_pos = self.position + np.random.uniform(-0.2, 0.2, 3)
            self.position = new_pos
        elif action in ['follow', 'flee']:
            # Двигаемся к цели или от неё
            nearest = self._get_nearest_object()
            if nearest:
                target_pos = np.array(nearest['object'].get('position', [0, 0, 0]))
                if action == 'follow':
                    direction = target_pos - self.position
                    self.position += direction * 0.1
                elif action == 'flee':
                    direction = self.position - target_pos
                    self.position += direction * 0.1

    def _get_current_step(self) -> int:
        """Получает текущий шаг из состояния"""
        # В реальной системе это будет из state.step
        return len(self.spatial_memory)

    def reset(self):
        self.spatial_memory.clear()
        self.position = np.array([0.0, 0.0, 0.0])
        self.gaze_direction = np.array([0.0, 0.0, 1.0])
        self.object_map = {}
        self.movement_history.clear()
