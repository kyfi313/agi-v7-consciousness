# -*- coding: utf-8 -*-
"""
Модуль воображения — сервис для работы с нейронными паттернами.

Принцип работы:
1. Хранит паттерны активации нейронов (векторы)
2. Ищет по косинусному сходству, а не по ключевым словам
3. Рекомбинирует паттерны в новые (композиция образов)
4. Поддерживает сборку сложных образов из простых

Мозг универсален — он использует нейроны как кубики для сборки любых образов.
Никаких текстовых меток, только числовые паттерны.
"""

import random
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional


class ImaginationModule:
    """
    Сервис воображения — работает с нейронными паттернами.
    """

    def __init__(self, pattern_dim: int = 10, max_fragments: int = 200):
        self.pattern_dim = pattern_dim
        self.max_fragments = max_fragments
        
        # Хранилище паттернов (векторы активации нейронов)
        self.patterns = []  # Каждый: {'vector': [...], 'metadata': {...}}
        
        # Ассоциативные связи между паттернами (по сходству)
        self.association_graph = defaultdict(list)  # индекс паттерна → список связанных индексов
        
        # Композиция образов: сложные паттерны собираются из простых
        self.composition_tree = defaultdict(list)  # индекс сложного → список индексов простых
        
        # Параметры
        self.similarity_threshold = 0.4
        self.recombination_rate = 0.6
        self.novelty_bonus = 0.2
        
        self.step_count = 0
        self.last_imagined = []
        self.recent_activations = deque(maxlen=10)

    def store_pattern(self, vector: List[float], metadata: Dict[str, Any] = None):
        """
        Сохраняет паттерн активации нейронов.
        
        Args:
            vector: Вектор активации нейронов (длина = pattern_dim)
            metadata: Дополнительные данные (эмоции, действие, награда)
        """
        # Преобразуем в список если нужно
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        # Ограничиваем размер
        if len(vector) > self.pattern_dim:
            vector = vector[:self.pattern_dim]
        elif len(vector) < self.pattern_dim:
            vector = vector + [0.0] * (self.pattern_dim - len(vector))
        
        # Нормализация
        norm = np.linalg.norm(vector) or 1.0
        vector = [v / norm for v in vector]
        
        pattern_entry = {
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': self.step_count,
            'index': len(self.patterns)
        }
        
        self.patterns.append(pattern_entry)
        if len(self.patterns) > self.max_fragments:
            self.patterns.pop(0)
        
        # Строим ассоциации с последними активированными паттернами
        for recent_idx in self.recent_activations:
            if recent_idx != len(self.patterns) - 1:
                similarity = self._cosine_similarity(
                    vector, 
                    self.patterns[recent_idx]['vector']
                )
                if similarity > self.similarity_threshold:
                    self.association_graph[len(self.patterns) - 1].append(recent_idx)
                    self.association_graph[recent_idx].append(len(self.patterns) - 1)
        
        self.recent_activations.append(len(self.patterns) - 1)

    def imagine(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует воображаемые паттерны на основе контекста.
        
        Args:
            context: Словарь с текущим состоянием
                {
                    'brain': {...},       # спайки, эмоции
                    'perception': {...},  # среда
                    'energy': float,
                    'memory': [...]       # недавние эпизоды
                }
        
        Returns:
            Список воображаемых паттернов с метаданными
        """
        self.step_count += 1
        
        # 1. Извлекаем текущий вектор из мозга (если есть)
        brain = context.get('brain', {})
        spikes = brain.get('spikes', [])
        if spikes:
            current_vector = self._spikes_to_vector(spikes)
        else:
            # Если спайков нет — используем восприятие
            perception = context.get('perception', {})
            current_vector = self._perception_to_vector(perception)
        
        # 2. Ищем похожие паттерны в памяти
        similar_patterns = self._find_similar(current_vector, top_k=10)
        
        # 3. Если нет похожих — создаём случайные
        if not similar_patterns:
            similar_patterns = self._generate_random_patterns(5)
        
        # 4. Композиция: собираем сложные образы из простых
        composed = []
        for _ in range(3):
            # Берём 2-3 случайных паттерна из похожих
            if len(similar_patterns) >= 2:
                selected = random.sample(similar_patterns, min(3, len(similar_patterns)))
                composed_pattern = self._compose(selected)
                composed.append(composed_pattern)
            else:
                # Если мало похожих — просто копируем с шумом
                base = similar_patterns[0] if similar_patterns else None
                if base:
                    composed_pattern = self._mutate(base)
                    composed.append(composed_pattern)
        
        # 5. Добавляем новизну
        for img in composed:
            img['novelty'] = self._calculate_novelty(img['vector'])
        
        # 6. Сортируем по новизне и ожидаемой награде
        composed.sort(key=lambda x: x['novelty'] + x.get('expected_reward', 0) * 0.3, reverse=True)
        
        self.last_imagined = composed
        return composed

    def _spikes_to_vector(self, spikes: List[float]) -> List[float]:
        """Преобразует спайки в вектор фиксированной длины."""
        if len(spikes) >= self.pattern_dim:
            return spikes[:self.pattern_dim]
        else:
            return spikes + [0.0] * (self.pattern_dim - len(spikes))

    def _perception_to_vector(self, perception: Dict[str, Any]) -> List[float]:
        """Преобразует восприятие в вектор."""
        vec = [
            perception.get('energy', 100) / 100.0,
            1.0 if perception.get('food_nearby', False) else 0.0,
            1.0 if perception.get('danger_nearby', False) else 0.0,
            perception.get('min_food_dist', 10) / 10.0,
            perception.get('min_danger_dist', 10) / 10.0,
        ]
        # Дополняем до pattern_dim
        while len(vec) < self.pattern_dim:
            vec.append(0.0)
        return vec[:self.pattern_dim]

    def _find_similar(self, vector: List[float], top_k: int = 10) -> List[Dict]:
        """Находит похожие паттерны по косинусному сходству."""
        if not self.patterns:
            return []
        
        similarities = []
        for i, pattern in enumerate(self.patterns):
            sim = self._cosine_similarity(vector, pattern['vector'])
            if sim > self.similarity_threshold:
                similarities.append((i, sim, pattern))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [s[2] for s in similarities[:top_k]]

    def _compose(self, patterns: List[Dict]) -> Dict[str, Any]:
        """
        Композиция образов: собирает сложный паттерн из простых.
        Использует взвешенное смешивание с учётом значимости каждого паттерна.
        """
        # Веса на основе новизны и награды
        weights = []
        for p in patterns:
            reward = p.get('metadata', {}).get('reward', 0)
            novelty = self._calculate_novelty(p['vector'])
            weight = 0.5 + 0.3 * reward + 0.2 * novelty
            weights.append(max(0.1, weight))
        
        # Нормализация весов
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Смешивание
        combined = [0.0] * self.pattern_dim
        for i, p in enumerate(patterns):
            for j, val in enumerate(p['vector']):
                combined[j] += val * weights[i]
        
        # Нормализация
        norm = np.linalg.norm(combined) or 1.0
        combined = [v / norm for v in combined]
        
        # Средняя награда
        avg_reward = sum(p.get('metadata', {}).get('reward', 0) for p in patterns) / len(patterns)
        
        # Сохраняем композицию в дерево
        indices = [p.get('index', -1) for p in patterns if 'index' in p]
        new_index = len(self.patterns)
        self.composition_tree[new_index] = indices
        
        return {
            'vector': combined,
            'expected_reward': avg_reward,
            'source': 'composition',
            'components': len(patterns),
            'novelty': self._calculate_novelty(combined)
        }

    def _mutate(self, pattern: Dict) -> Dict[str, Any]:
        """Мутирует паттерн с добавлением шума."""
        vector = pattern['vector'].copy()
        noise_strength = random.uniform(0.05, 0.2)
        noise = [random.uniform(-noise_strength, noise_strength) for _ in range(self.pattern_dim)]
        vector = [v + n for v, n in zip(vector, noise)]
        
        # Нормализация
        norm = np.linalg.norm(vector) or 1.0
        vector = [v / norm for v in vector]
        
        return {
            'vector': vector,
            'expected_reward': pattern.get('metadata', {}).get('reward', 0) * 0.8,
            'source': 'mutation',
            'novelty': self._calculate_novelty(vector)
        }

    def _generate_random_patterns(self, count: int) -> List[Dict]:
        """Генерирует случайные паттерны."""
        patterns = []
        for _ in range(count):
            vector = [random.uniform(-1, 1) for _ in range(self.pattern_dim)]
            norm = np.linalg.norm(vector) or 1.0
            vector = [v / norm for v in vector]
            patterns.append({
                'vector': vector,
                'expected_reward': random.uniform(-0.3, 0.3),
                'source': 'random',
                'novelty': 1.0
            })
        return patterns

    def _calculate_novelty(self, vector: List[float]) -> float:
        """Вычисляет новизну паттерна относительно известных."""
        if not self.patterns:
            return 1.0
        
        max_similarity = 0.0
        for pattern in self.patterns:
            sim = self._cosine_similarity(vector, pattern['vector'])
            if sim > max_similarity:
                max_similarity = sim
        
        return 1.0 - max_similarity

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Вычисляет косинусное сходство между двумя векторами."""
        if not a or not b:
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_current_imaginations(self) -> List[Dict[str, Any]]:
        """Возвращает текущие воображаемые паттерны."""
        return self.last_imagined
