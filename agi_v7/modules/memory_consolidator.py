# -*- coding: utf-8 -*-
"""
Консолидатор памяти — управление переносом знаний из эпизодической в семантическую память
Аналог гиппокампальной консолидации во сне
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set

from .semantic_memory import SemanticMemory


class MemoryConsolidator:
    """
    Оркестратор памяти: связывает гиппокамп (эпизоды) и семантическую память (факты).
    
    Процесс консолидации:
    1. Извлекает значимые паттерны из эпизодической памяти
    2. Обобщает их в семантические концепты
    3. Связывает новые концепты с уже существующими
    4. Усиливает или ослабляет связи на основе частоты
    """
    
    def __init__(self, semantic_memory: Optional[SemanticMemory] = None):
        self.semantic = semantic_memory or SemanticMemory()
        
        # Буфер для эпизодов, ожидающих консолидации
        self.consolidation_buffer: List[Dict] = []
        self.max_buffer_size = 200
        
        # Статистика консолидации
        self.consolidation_count = 0
        self.total_patterns_extracted = 0
        
        # Параметры
        self.min_episodes_for_consolidation = 5
        self.similarity_threshold = 0.4
    
    def add_episode(self, episode: Dict, state) -> None:
        """
        Добавляет эпизод в буфер для будущей консолидации.
        Эпизод должен содержать: {'concepts': [...], 'reward': float, 'action': str, 'emotions': {...}}
        """
        # Извлекаем концепты из эпизода
        concepts = self._extract_concepts_from_episode(episode, state)
        
        self.consolidation_buffer.append({
            'concepts': concepts,
            'reward': episode.get('reward', 0.0),
            'action': episode.get('action', 'unknown'),
            'emotions': episode.get('emotions', {}),
            'step': state.step if hasattr(state, 'step') else 0,
            'success': episode.get('success', False),
        })
        
        # Ограничиваем размер буфера
        if len(self.consolidation_buffer) > self.max_buffer_size:
            self.consolidation_buffer = self.consolidation_buffer[-self.max_buffer_size:]
    
    def _extract_concepts_from_episode(self, episode: Dict, state) -> List[str]:
        """Извлекает концепты из эпизода"""
        concepts = []
        
        # Из действия
        action = episode.get('action', '')
        if action and action not in ['unknown', 'none']:
            concepts.append(f'action_{action}')
        
        # Из эмоций
        emotions = episode.get('emotions', {})
        for emotion, intensity in emotions.items():
            if intensity > 0.4:
                concepts.append(f'emotion_{emotion}')
        
        # Из награды
        reward = episode.get('reward', 0.0)
        if reward > 0.3:
            concepts.append('reward_high')
        elif reward < -0.3:
            concepts.append('reward_low')
        
        # Из объектов (если есть в state)
        if hasattr(state, 'objects') and state.objects:
            for obj in state.objects[:3]:
                if isinstance(obj, dict):
                    obj_type = obj.get('type', 'unknown')
                else:
                    obj_type = str(obj)
                concepts.append(f'object_{obj_type}')
        
        # Из состояния body
        if hasattr(state, 'body'):
            for key, val in state.body.items():
                if isinstance(val, (int, float)) and (val > 0.7 or val < 0.3):
                    concepts.append(f'body_{key}_{int(val*10)}')
        
        # Из perception
        perception = episode.get('perception', {})
        for key, val in perception.items():
            if isinstance(val, (int, float)) and val > 0.5:
                concepts.append(f'perception_{key}')
        
        # Возвращаем уникальные концепты
        return list(set(concepts))[:20]  # ограничиваем
    
    def consolidate(self, state) -> Dict:
        """
        Консолидирует буфер эпизодов в семантическую память.
        
        Процесс:
        1. Группирует похожие эпизоды
        2. Извлекает общие паттерны
        3. Обновляет семантическую память
        """
        if len(self.consolidation_buffer) < self.min_episodes_for_consolidation:
            return {'status': 'not_enough_episodes', 'buffer_size': len(self.consolidation_buffer)}
        
        self.consolidation_count += 1
        
        # 1. Группируем эпизоды по ключевым концептам
        concept_groups = defaultdict(list)
        for episode in self.consolidation_buffer:
            for concept in episode['concepts']:
                concept_groups[concept].append(episode)
        
        # 2. Для каждой группы извлекаем паттерны
        patterns_extracted = []
        for concept, episodes in concept_groups.items():
            if len(episodes) >= 2:
                # Ищем общие паттерны в этой группе
                pattern = self._extract_pattern(episodes)
                if pattern:
                    patterns_extracted.append(pattern)
        
        # 3. Обновляем семантическую память
        for pattern in patterns_extracted:
            self._update_semantic_memory(pattern, state)
        
        # 4. Очищаем буфер (оставляем только последние эпизоды)
        self.consolidation_buffer = self.consolidation_buffer[-5:]
        
        self.total_patterns_extracted += len(patterns_extracted)
        
        return {
            'status': 'consolidated',
            'patterns_extracted': len(patterns_extracted),
            'buffer_size': len(self.consolidation_buffer),
            'total_patterns': self.total_patterns_extracted,
            'concepts_updated': len(patterns_extracted),
        }
    
    def _extract_pattern(self, episodes: List[Dict]) -> Optional[Dict]:
        """Извлекает паттерн из группы эпизодов"""
        # Считаем частоту действий
        action_freq = defaultdict(int)
        reward_sum = 0
        
        for ep in episodes:
            action_freq[ep['action']] += 1
            reward_sum += ep['reward']
        
        # Находим самое частое действие
        if not action_freq:
            return None
        
        most_common_action = max(action_freq.items(), key=lambda x: x[1])
        avg_reward = reward_sum / len(episodes)
        
        # Определяем успешность паттерна
        success_rate = sum(1 for ep in episodes if ep.get('success', False)) / len(episodes)
        
        # Собираем эмоциональный профиль
        emotion_profile = defaultdict(float)
        for ep in episodes:
            for emotion, intensity in ep.get('emotions', {}).items():
                emotion_profile[emotion] += intensity
        
        # Усредняем эмоции
        for emotion in emotion_profile:
            emotion_profile[emotion] /= len(episodes)
        
        # Собираем все концепты из эпизодов
        all_concepts = set()
        for ep in episodes:
            all_concepts.update(ep['concepts'])
        
        if len(all_concepts) < 2:
            return None
        
        return {
            'primary_concept': most_common_action[0],
            'action_freq': dict(action_freq),
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'emotion_profile': dict(emotion_profile),
            'all_concepts': list(all_concepts),
            'episode_count': len(episodes),
        }
    
    def _update_semantic_memory(self, pattern: Dict, state):
        """Обновляет семантическую память на основе паттерна"""
        primary = pattern['primary_concept']
        
        # Создаём или обновляем концепт для основного действия
        self.semantic._update_or_create_concept(primary, {
            'action': primary,
            'reward': pattern['avg_reward'],
            'emotions': pattern['emotion_profile'],
            'success': pattern['success_rate'] > 0.5,
        })
        
        # Связываем основной концепт с другими
        for concept in pattern['all_concepts']:
            if concept != primary:
                # Сила связи зависит от успешности и частоты
                strength = pattern['success_rate'] * 0.6 + pattern['episode_count'] / 20.0
                self.semantic._add_connection(primary, concept, min(1.0, strength))
        
        # Создаём абстрактный концепт для паттерна
        abstract_name = f"{primary}_pattern"
        if abstract_name not in self.semantic.concepts:
            # Вектор = среднее всех связанных концептов
            embeddings = []
            for concept in pattern['all_concepts'][:5]:
                if concept in self.semantic.concepts:
                    embeddings.append(self.semantic.concepts[concept].embedding)
            
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            else:
                avg_embedding = np.random.randn(self.semantic.embedding_dim) * 0.3
            
            from .semantic_memory import Concept
            concept = Concept(
                name=abstract_name,
                embedding=avg_embedding,
                is_abstract=True,
                frequency=pattern['episode_count']
            )
            self.semantic.concepts[abstract_name] = concept
            self.semantic.total_concepts_created += 1
            
            # Связываем с исходными концептами
            for concept_name in pattern['all_concepts'][:3]:
                self.semantic._add_connection(abstract_name, concept_name, 0.6)
    
    def retrieve_knowledge(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Поиск знаний в семантической памяти по запросу.
        Использует семантический поиск.
        """
        return self.semantic.retrieve(query, top_k)
    
    def get_connections(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Возвращает связанные концепты для данного"""
        return self.semantic.get_connections(concept, top_k)
    
    def get_summary(self) -> Dict:
        """Возвращает сводку по состоянию памяти"""
        semantic_summary = self.semantic.get_summary()
        
        return {
            'consolidation_count': self.consolidation_count,
            'total_patterns': self.total_patterns_extracted,
            'buffer_size': len(self.consolidation_buffer),
            'semantic': semantic_summary,
        }
    
    def reset(self):
        """Сброс консолидатора"""
        self.consolidation_buffer = []
        self.consolidation_count = 0
        self.total_patterns_extracted = 0
        self.semantic.reset()
