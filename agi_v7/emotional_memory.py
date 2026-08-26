# -*- coding: utf-8 -*-
"""
ЭМОЦИОНАЛЬНАЯ ПАМЯТЬ (нейронно-подобная динамика)
Воспоминания имеют эмоциональную окраску и влияют на будущие решения.
Память — это не просто склад, а активное нейронное поле.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import time
import hashlib


class MemoryTrace:
    """
    След памяти с эмоциональной окраской.
    """
    def __init__(self, content: Any, emotion: str, valence: float, 
                 intensity: float, timestamp: float):
        self.content = content
        self.emotion = emotion
        self.valence = valence  # -1...1 (негативно...позитивно)
        self.intensity = intensity  # 0...1
        self.timestamp = timestamp
        self.recall_count = 0
        self.last_recall = timestamp
        self.strength = 0.5  # начальная сила
        
    def recall(self):
        """Восстановление памяти увеличивает её силу."""
        self.recall_count += 1
        self.last_recall = time.time()
        self.strength = min(1.0, self.strength + 0.05)
        
    def decay(self, decay_rate: float = 0.01):
        """Угасание памяти со временем."""
        self.strength *= (1.0 - decay_rate)
        if self.strength < 0.01:
            self.strength = 0.01


class EmotionalMemory:
    """
    Эмоциональная память.
    Работает как нейронное поле, где воспоминания активируются
    по сходству и эмоциональному соответствию.
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.traces: List[MemoryTrace] = []
        self.current_emotion = 'neutral'
        self.current_valence = 0.0
        self.current_intensity = 0.0
        
        # Эмоциональные состояния (нейронные поля)
        self.emotion_states = {
            'fear': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'surprise': 0.0,
            'neutral': 1.0
        }
        
        # Память активаций
        self.activation_history = deque(maxlen=100)
        
    def add_memory(self, content: Any, emotion: str = 'neutral', 
                   valence: float = 0.0, intensity: float = 0.5) -> None:
        """
        Добавляет воспоминание с эмоциональной окраской.
        """
        trace = MemoryTrace(content, emotion, valence, intensity, time.time())
        self.traces.append(trace)
        
        # Ограничиваем размер
        if len(self.traces) > self.capacity:
            # Удаляем самые слабые воспоминания
            self.traces.sort(key=lambda t: t.strength)
            self.traces = self.traces[-self.capacity:]
        
        # Обновляем состояние эмоций
        self.emotion_states[emotion] = min(1.0, self.emotion_states.get(emotion, 0) + intensity * 0.3)
        self._normalize_emotions()
    
    def recall(self, query: Any = None, emotion_cue: str = None) -> List[MemoryTrace]:
        """
        Восстанавливает воспоминания по запросу или эмоциональному ключу.
        """
        results = []
        
        # Если есть эмоциональный ключ — ищем похожие эмоции
        if emotion_cue and emotion_cue in self.emotion_states:
            cue_intensity = self.emotion_states[emotion_cue]
            for trace in self.traces:
                if trace.emotion == emotion_cue:
                    score = trace.strength * (1 + cue_intensity)
                    results.append((trace, score))
        
        # Если есть конкретный запрос — ищем по содержанию (упрощённо)
        elif query is not None:
            query_str = str(query).lower()
            for trace in self.traces:
                content_str = str(trace.content).lower()
                if query_str in content_str:
                    score = trace.strength * 1.5
                    results.append((trace, score))
        
        # Если ничего не указано — возвращаем самые сильные воспоминания
        else:
            for trace in self.traces:
                results.append((trace, trace.strength))
        
        # Сортируем по счёту
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Восстанавливаем топ-5
        top_results = [trace for trace, _ in results[:5]]
        for trace in top_results:
            trace.recall()
        
        return top_results
    
    def get_emotional_context(self) -> Dict[str, float]:
        """
        Возвращает текущий эмоциональный контекст.
        """
        return {
            'emotions': self.emotion_states.copy(),
            'current_emotion': self.current_emotion,
            'valence': self.current_valence,
            'intensity': self.current_intensity
        }
    
    def update_emotion(self, emotion: str, intensity: float) -> None:
        """
        Обновляет текущее эмоциональное состояние.
        """
        self.current_emotion = emotion
        self.current_intensity = intensity
        
        if emotion in self.emotion_states:
            self.emotion_states[emotion] = min(1.0, self.emotion_states[emotion] + intensity * 0.2)
            self._normalize_emotions()
    
    def _normalize_emotions(self) -> None:
        """Нормализует эмоциональные состояния."""
        total = sum(self.emotion_states.values()) + 0.001
        for key in self.emotion_states:
            self.emotion_states[key] /= total
            self.emotion_states[key] = max(0.0, min(1.0, self.emotion_states[key]))
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по памяти.
        """
        emotions_count = {}
        for trace in self.traces:
            emotions_count[trace.emotion] = emotions_count.get(trace.emotion, 0) + 1
        
        return {
            'total_traces': len(self.traces),
            'emotions_distribution': emotions_count,
            'current_emotion': self.current_emotion,
            'current_valence': self.current_valence,
            'emotion_states': self.emotion_states,
            'avg_strength': np.mean([t.strength for t in self.traces]) if self.traces else 0.0
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает состояние системы.
        """
        return self.get_memory_summary()
