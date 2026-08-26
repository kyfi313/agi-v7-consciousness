# -*- coding: utf-8 -*-
"""
Семантическая память — хранение фактов, концептов и обобщённых знаний
Аналог коры головного мозга, где хранятся обобщённые представления о мире
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Concept:
    """Концепт — единица знания в семантической памяти"""
    name: str
    embedding: np.ndarray  # векторное представление
    connections: Dict[str, float] = field(default_factory=dict)  # связанные концепты + сила связи
    frequency: int = 1  # как часто встречался
    last_accessed: int = 0
    is_abstract: bool = False  # абстрактное понятие vs конкретный объект
    
    def add_connection(self, other_name: str, strength: float):
        """Добавляет или усиливает связь с другим концептом"""
        if other_name in self.connections:
            self.connections[other_name] = min(1.0, self.connections[other_name] + strength * 0.1)
        else:
            self.connections[other_name] = min(1.0, strength)
    
    def get_similarity(self, other_embedding: np.ndarray) -> float:
        """Косинусное сходство с другим вектором"""
        if self.embedding is None or other_embedding is None:
            return 0.0
        norm = np.linalg.norm(self.embedding) * np.linalg.norm(other_embedding)
        if norm == 0:
            return 0.0
        return float(np.dot(self.embedding, other_embedding) / norm)


class SemanticMemory:
    """
    Семантическая память — хранилище обобщённых знаний.
    
    Особенности:
    - Концепты хранятся с векторными представлениями (эмбеддингами)
    - Между концептами есть взвешенные связи (ассоциации)
    - Поддерживается обобщение: из нескольких конкретных событий → абстрактный концепт
    - Консолидация: периодическое обновление и обобщение знаний
    """
    
    def __init__(self, embedding_dim: int = 64, consolidation_interval: int = 10):
        self.concepts: Dict[str, Concept] = {}
        self.embedding_dim = embedding_dim
        self.consolidation_interval = consolidation_interval
        self.step_counter = 0
        
        # Буфер для временных наблюдений (перед консолидацией)
        self.observation_buffer: List[Dict] = []
        self.max_buffer_size = 100
        
        # Статистика
        self.total_consolidations = 0
        self.total_concepts_created = 0
        
        # Создаём базовые концепты (врождённые знания)
        self._initialize_base_concepts()
    
    def _initialize_base_concepts(self):
        """Создаёт базовые концепты, с которыми система рождается"""
        base_concepts = [
            ('self', True),
            ('food', False),
            ('danger', False),
            ('social', False),
            ('explore', False),
            ('reward', False),
            ('rest', False),
        ]
        
        for name, is_abstract in base_concepts:
            embedding = np.random.randn(self.embedding_dim) * 0.3
            concept = Concept(
                name=name,
                embedding=embedding,
                is_abstract=is_abstract,
                frequency=1
            )
            self.concepts[name] = concept
            self.total_concepts_created += 1
        
        # Базовые связи (врождённые ассоциации)
        self._add_connection('food', 'reward', 0.8)
        self._add_connection('danger', 'self', 0.7)
        self._add_connection('social', 'reward', 0.6)
        self._add_connection('explore', 'reward', 0.5)
    
    def _add_connection(self, name1: str, name2: str, strength: float):
        """Добавляет связь между двумя концептами"""
        if name1 in self.concepts and name2 in self.concepts:
            self.concepts[name1].add_connection(name2, strength)
            self.concepts[name2].add_connection(name1, strength)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Ищет концепты, похожие на запрос.
        Возвращает список словарей с информацией о концептах.
        """
        results = []
        query_lower = query.lower()
        
        # Точное совпадение
        if query_lower in self.concepts:
            concept = self.concepts[query_lower]
            concept.last_accessed += 1
            results.append({
                'name': concept.name,
                'frequency': concept.frequency,
                'is_abstract': concept.is_abstract,
                'connections': dict(concept.connections),
                'valence': concept.connections.get('valence', 0.0)
            })
            return results[:top_k]
        
        # Поиск по частичному совпадению
        for name, concept in self.concepts.items():
            if query_lower in name or name in query_lower:
                concept.last_accessed += 1
                results.append({
                    'name': concept.name,
                    'frequency': concept.frequency,
                    'is_abstract': concept.is_abstract,
                    'connections': dict(concept.connections),
                    'valence': concept.connections.get('valence', 0.0)
                })
        
        # Сортировка по частоте
        results.sort(key=lambda x: x['frequency'], reverse=True)
        return results[:top_k]
    
    def observe(self, observation: Dict, state) -> List[str]:
        """
        Наблюдает событие и сохраняет его в буфер.
        Возвращает список активированных концептов.
        """
        self.step_counter += 1
        
        # Извлекаем ключевые элементы из наблюдения
        activated_concepts = self._extract_concepts(observation, state)
        
        # Сохраняем в буфер
        self.observation_buffer.append({
            'concepts': activated_concepts,
            'state': state.__dict__.copy() if hasattr(state, '__dict__') else {},
            'step': self.step_counter,
            'reward': state.reward if hasattr(state, 'reward') else 0.0,
        })
        
        # Ограничиваем размер буфера
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer = self.observation_buffer[-self.max_buffer_size:]
        
        # Периодическая консолидация
        if self.step_counter % self.consolidation_interval == 0:
            self.consolidate()
        
        return activated_concepts
    
    def _extract_concepts(self, observation: Dict, state) -> List[str]:
        """Извлекает концепты из наблюдения"""
        activated = []
        
        # Из perception
        for key in ['action', 'emotion', 'perception']:
            value = observation.get(key, None)
            if value is not None:
                concepts_from_value = self._value_to_concepts(value)
                activated.extend(concepts_from_value)
        
        # Из состояния
        if hasattr(state, 'emotions'):
            for emotion, intensity in state.emotions.items():
                if intensity > 0.3:
                    activated.append(emotion)
        
        # Из объектов
        if hasattr(state, 'objects') and state.objects:
            for obj in state.objects[:3]:  # не более 3 объектов за раз
                if isinstance(obj, dict):
                    obj_type = obj.get('type', 'unknown')
                else:
                    obj_type = str(obj)
                activated.append(obj_type)
        
        # Уникализируем
        activated = list(set(activated))
        
        # Обновляем или создаём концепты
        for name in activated:
            self._update_or_create_concept(name, observation)
        
        # Обновляем связи между концептами, которые встретились вместе
        for i, name1 in enumerate(activated):
            for name2 in activated[i+1:]:
                self._add_connection(name1, name2, 0.3)
        
        return activated
    
    def _value_to_concepts(self, value) -> List[str]:
        """Преобразует значение в список концептов"""
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return [str(v) for v in value[:3]]
        elif isinstance(value, dict):
            return list(value.keys())[:3]
        elif isinstance(value, (int, float)):
            return [f'value_{int(value * 10)}']
        return []
    
    def _update_or_create_concept(self, name: str, observation: Dict):
        """Обновляет существующий или создаёт новый концепт"""
        if name in self.concepts:
            concept = self.concepts[name]
            concept.frequency += 1
            concept.last_accessed = self.step_counter
            # Обновляем эмбеддинг с учётом нового контекста
            context_vector = self._observation_to_vector(observation)
            if context_vector is not None:
                concept.embedding = 0.9 * concept.embedding + 0.1 * context_vector
                concept.embedding = concept.embedding / (np.linalg.norm(concept.embedding) + 1e-8)
        else:
            # Создаём новый концепт
            embedding = self._observation_to_vector(observation)
            if embedding is None:
                embedding = np.random.randn(self.embedding_dim) * 0.3
            
            is_abstract = len(name) > 5 or name in ['strategy', 'plan', 'goal']
            concept = Concept(
                name=name,
                embedding=embedding,
                is_abstract=is_abstract,
                frequency=1,
                last_accessed=self.step_counter
            )
            self.concepts[name] = concept
            self.total_concepts_created += 1
    
    def _observation_to_vector(self, observation: Dict) -> Optional[np.ndarray]:
        """Преобразует наблюдение в вектор"""
        try:
            # Собираем числовые значения
            values = []
            for key, val in observation.items():
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, str):
                    # Хеш строки в число
                    values.append(hash(val) % 100 / 100.0)
                elif isinstance(val, list) and len(val) > 0:
                    values.append(len(val) / 10.0)
            
            if not values:
                return None
            
            # Расширяем до embedding_dim
            vector = np.zeros(self.embedding_dim)
            for i, v in enumerate(values[:self.embedding_dim]):
                vector[i] = v
            
            # Нормализация
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
        except Exception:
            return None
    
    def consolidate(self) -> Dict:
        """
        Консолидация памяти — обобщение наблюдений в устойчивые знания.
        
        Процесс:
        1. Анализ буфера наблюдений
        2. Поиск повторяющихся паттернов
        3. Создание абстрактных концептов из повторяющихся событий
        4. Усиление часто используемых связей
        5. Ослабление/удаление редко используемых связей
        """
        if len(self.observation_buffer) < 2:
            return {'status': 'not_enough_data'}
        
        self.total_consolidations += 1
        
        # 1. Собираем статистику по концептам
        concept_freq = defaultdict(int)
        concept_pairs = defaultdict(int)
        
        for obs in self.observation_buffer:
            concepts = obs.get('concepts', [])
            for c in concepts:
                concept_freq[c] += 1
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    if c1 < c2:
                        concept_pairs[(c1, c2)] += 1
                    else:
                        concept_pairs[(c2, c1)] += 1
        
        # 2. Создаём абстрактные концепты для частых паттернов
        new_concepts = []
        for (c1, c2), freq in concept_pairs.items():
            if freq >= 3 and c1 in self.concepts and c2 in self.concepts:
                # Создаём абстрактный концепт, объединяющий c1 и c2
                abstract_name = f"{c1}_{c2}_pattern"
                if abstract_name not in self.concepts:
                    # Вектор = среднее векторов c1 и c2
                    v1 = self.concepts[c1].embedding
                    v2 = self.concepts[c2].embedding
                    avg_embedding = (v1 + v2) / 2
                    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                    
                    concept = Concept(
                        name=abstract_name,
                        embedding=avg_embedding,
                        is_abstract=True,
                        frequency=freq,
                        last_accessed=self.step_counter
                    )
                    self.concepts[abstract_name] = concept
                    self.total_concepts_created += 1
                    new_concepts.append(abstract_name)
                    
                    # Связываем с исходными концептами
                    self._add_connection(abstract_name, c1, 0.7)
                    self._add_connection(abstract_name, c2, 0.7)
        
        # 3. Усиливаем частые связи
        for (c1, c2), freq in concept_pairs.items():
            if freq >= 2 and c1 in self.concepts and c2 in self.concepts:
                strength = min(1.0, 0.3 + freq * 0.1)
                self._add_connection(c1, c2, strength)
        
        # 4. Ослабляем старые связи (если не использовались давно)
        for concept in self.concepts.values():
            if self.step_counter - concept.last_accessed > self.consolidation_interval * 3:
                # Удаляем или ослабляем связи этого концепта
                for other in list(concept.connections.keys()):
                    if other in self.concepts:
                        old_strength = concept.connections[other]
                        new_strength = old_strength * 0.9
                        if new_strength < 0.1:
                            del concept.connections[other]
                            if other in self.concepts:
                                del self.concepts[other].connections[concept.name]
                        else:
                            concept.connections[other] = new_strength
        
        # 5. Очищаем буфер (оставляем только последние наблюдения)
        self.observation_buffer = self.observation_buffer[-5:]
        
        return {
            'status': 'consolidated',
            'new_concepts': new_concepts,
            'total_concepts': len(self.concepts),
            'concept_freq': dict(concept_freq),
        }
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Поиск похожих концептов по строке запроса.
        Возвращает список (имя_концепта, сходство).
        """
        # Создаём вектор запроса
        query_vector = self._text_to_vector(query)
        if query_vector is None:
            return []
        
        scores = []
        for name, concept in self.concepts.items():
            similarity = concept.get_similarity(query_vector)
            # Учитываем частоту: чем чаще концепт, тем выше приоритет
            freq_boost = min(1.5, 1.0 + concept.frequency / 50.0)
            score = similarity * freq_boost
            scores.append((name, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """Преобразует текст в вектор"""
        # Простая эмбеддинг-функция
        vector = np.zeros(self.embedding_dim)
        for i, char in enumerate(text[:self.embedding_dim]):
            vector[i] = ord(char) % 256 / 256.0
        
        # Нормализация
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    
    def get_connections(self, concept_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Возвращает связанные концепты для данного"""
        if concept_name not in self.concepts:
            return []
        
        concept = self.concepts[concept_name]
        connections = sorted(concept.connections.items(), key=lambda x: x[1], reverse=True)
        return connections[:top_k]
    
    def get_summary(self) -> Dict:
        """Возвращает сводку по семантической памяти"""
        total_connections = sum(len(c.connections) for c in self.concepts.values())
        avg_freq = sum(c.frequency for c in self.concepts.values()) / max(1, len(self.concepts))
        
        # Самые частые концепты
        top_concepts = sorted(
            [(name, c.frequency) for name, c in self.concepts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_concepts': len(self.concepts),
            'total_connections': total_connections,
            'avg_frequency': avg_freq,
            'total_consolidations': self.total_consolidations,
            'total_concepts_created': self.total_concepts_created,
            'buffer_size': len(self.observation_buffer),
            'top_concepts': top_concepts,
        }
    
    def reset(self):
        """Сброс семантической памяти"""
        self.concepts = {}
        self.observation_buffer = []
        self.step_counter = 0
        self.total_consolidations = 0
        self.total_concepts_created = 0
        self._initialize_base_concepts()
