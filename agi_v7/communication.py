# -*- coding: utf-8 -*-
"""
УРОВЕНЬ 8: КОММУНИКАЦИЯ И ЯЗЫК

Агенты обмениваются сообщениями о:
- Местоположении еды
- Опасностях
- Своих намерениях

Принцип работы:
1. Каждый агент может отправлять сообщения другим
2. Сообщения имеют структуру (тип, координаты, важность)
3. Агенты накапливают информацию от других
4. Интеграция с планировщиком и моделью мира
"""

import random
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time


@dataclass
class Message:
    """Сообщение между агентами."""
    sender_id: int
    msg_type: str  # 'food', 'danger', 'intent', 'info'
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 5  # 1-10, где 10 — наивысший
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sender_id': self.sender_id,
            'msg_type': self.msg_type,
            'content': self.content,
            'timestamp': self.timestamp,
            'priority': self.priority
        }


class CommunicationModule:
    """
    Модуль коммуникации.
    
    Управляет обменом сообщениями между агентами.
    """
    
    def __init__(self, broadcast_range: int = 5, max_messages_per_step: int = 3):
        self.broadcast_range = broadcast_range
        self.max_messages_per_step = max_messages_per_step
        
        # Все сообщения в системе
        self.messages: List[Message] = []
        
        # Сообщения для каждого агента (по id)
        self.inbox: Dict[int, List[Message]] = defaultdict(list)
        
        # Память о полученных сообщениях
        self.message_memory: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        # Статистика
        self.total_messages_sent = 0
        self.total_messages_received = 0
        
        # Параметры
        self.noise_level = 0.1  # Вероятность потери сообщения
        self.delay = 1  # Шагов задержки
        
    def send_message(
        self,
        sender_id: int,
        receiver_id: Optional[int],
        msg_type: str,
        content: Dict[str, Any],
        priority: int = 5
    ) -> bool:
        """
        Отправляет сообщение от агента.
        
        Args:
            sender_id: Отправитель
            receiver_id: Получатель (None — всем)
            msg_type: Тип сообщения
            content: Содержание
            priority: Приоритет (1-10)
        
        Returns:
            Успех отправки
        """
        # Проверяем лимит
        sent_count = sum(1 for m in self.messages if m.sender_id == sender_id 
                        and time.time() - m.timestamp < 10)
        if sent_count >= self.max_messages_per_step:
            return False
        
        # Создаём сообщение
        msg = Message(
            sender_id=sender_id,
            msg_type=msg_type,
            content=content,
            priority=priority
        )
        
        # Добавляем шум
        if random.random() < self.noise_level:
            # Сообщение потеряно
            return False
        
        # Добавляем задержку
        if self.delay > 0:
            # Сохраняем для последующей доставки
            msg.timestamp += self.delay
            self.messages.append(msg)
            # Пока просто добавляем в общий список
            # Доставка будет в процессе доставки
            if receiver_id is None:
                # Всем
                for agent_id in range(10):  # Максимум 10 агентов
                    if agent_id != sender_id:
                        self.inbox[agent_id].append(msg)
            else:
                self.inbox[receiver_id].append(msg)
        
        self.total_messages_sent += 1
        return True
    
    def deliver_messages(self, agent_id: int) -> List[Message]:
        """
        Доставляет сообщения агенту.
        
        Args:
            agent_id: ID агента
        
        Returns:
            Список сообщений для агента
        """
        # Получаем сообщения из инбокса
        messages = self.inbox.get(agent_id, [])
        
        # Очищаем инбокс
        self.inbox[agent_id] = []
        
        # Обновляем статистику
        self.total_messages_received += len(messages)
        
        # Сохраняем в память
        for msg in messages:
            self.message_memory[agent_id].append(msg.to_dict())
            if len(self.message_memory[agent_id]) > 100:
                self.message_memory[agent_id] = self.message_memory[agent_id][-100:]
        
        return messages
    
    def get_recent_messages(
        self,
        agent_id: int,
        msg_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Возвращает недавние сообщения агента.
        
        Args:
            agent_id: ID агента
            msg_type: Фильтр по типу
            limit: Максимальное количество
        
        Returns:
            Список сообщений
        """
        messages = self.message_memory.get(agent_id, [])
        
        if msg_type:
            messages = [m for m in messages if m['msg_type'] == msg_type]
        
        # Сортируем по времени (новые сначала)
        messages = sorted(messages, key=lambda m: m['timestamp'], reverse=True)
        
        return messages[:limit]
    
    def get_social_knowledge(self, agent_id: int) -> Dict[str, Any]:
        """
        Возвращает социальное знание агента.
        
        Извлекает информацию из полученных сообщений.
        
        Returns:
            Словарь с известной информацией
        """
        knowledge = {
            'known_food': [],
            'known_dangers': [],
            'known_agents': set(),
            'last_updates': {}
        }
        
        messages = self.message_memory.get(agent_id, [])
        
        for msg in messages:
            if msg['msg_type'] == 'food':
                pos = msg['content'].get('position')
                if pos and pos not in knowledge['known_food']:
                    knowledge['known_food'].append(pos)
                    knowledge['last_updates'][str(pos)] = msg['timestamp']
            elif msg['msg_type'] == 'danger':
                pos = msg['content'].get('position')
                if pos and pos not in knowledge['known_dangers']:
                    knowledge['known_dangers'].append(pos)
                    knowledge['last_updates'][str(pos)] = msg['timestamp']
            elif msg['msg_type'] == 'info':
                agent_info = msg['content'].get('agent')
                if agent_info:
                    knowledge['known_agents'].add(agent_info.get('id'))
        
        return knowledge
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        return {
            'total_sent': self.total_messages_sent,
            'total_received': self.total_messages_received,
            'inbox_size': sum(len(v) for v in self.inbox.values()),
            'memory_size': sum(len(v) for v in self.message_memory.values())
        }


class CommunicatingAgent:
    """
    Агент с коммуникацией.
    
    Оборачивает существующего агента и добавляет коммуникацию.
    """
    
    def __init__(self, base_agent, comm_module: CommunicationModule, agent_id: int):
        self.base_agent = base_agent
        self.comm = comm_module
        self.agent_id = agent_id
        
        # Социальное знание
        self.social_knowledge = {}
        
        # История коммуникации
        self.communication_history = []
        
        # Параметры
        self.share_food_info = True
        self.share_danger_info = True
        self.share_intent = False
        
    def decide(self, grid: List[List[str]], position: Tuple[int, int], 
               other_agents: List[Tuple[int, Tuple[int, int]]]) -> Tuple[str, str]:
        """
        Принимает решение с учётом коммуникации.
        
        Сначала получает сообщения, затем интегрирует их в решение.
        """
        # Получаем новые сообщения
        messages = self.comm.deliver_messages(self.agent_id)
        
        # Обновляем социальное знание
        self.social_knowledge = self.comm.get_social_knowledge(self.agent_id)
        
        # Если есть важные сообщения о еде — обновляем модель мира
        for msg in messages:
            if msg.msg_type == 'food' and self.share_food_info:
                pos = msg.content.get('position')
                if pos:
                    # Добавляем в знание агента (через модель мира, если есть)
                    if hasattr(self.base_agent, 'world_model'):
                        # Помечаем как потенциальную еду
                        pass
        
        # Принимаем решение через базового агента
        action, thought = self.base_agent.decide(grid, position, other_agents)
        
        # Отправляем сообщение о своих намерениях
        if self.share_intent and action != 'rest':
            self.comm.send_message(
                sender_id=self.agent_id,
                receiver_id=None,  # Всем
                msg_type='intent',
                content={
                    'action': action,
                    'position': position,
                    'timestamp': time.time()
                },
                priority=3
            )
        
        return action, thought
    
    def share_food_location(self, position: Tuple[int, int], confidence: float = 1.0):
        """Делится информацией о еде."""
        if self.share_food_info:
            self.comm.send_message(
                sender_id=self.agent_id,
                receiver_id=None,
                msg_type='food',
                content={
                    'position': position,
                    'confidence': confidence,
                    'timestamp': time.time()
                },
                priority=8  # Высокий приоритет
            )
    
    def share_danger_location(self, position: Tuple[int, int], confidence: float = 1.0):
        """Делится информацией об опасности."""
        if self.share_danger_info:
            self.comm.send_message(
                sender_id=self.agent_id,
                receiver_id=None,
                msg_type='danger',
                content={
                    'position': position,
                    'confidence': confidence,
                    'timestamp': time.time()
                },
                priority=9  # Очень высокий приоритет
            )
    
    def __getattr__(self, name):
        """Прокси для всех остальных методов."""
        if hasattr(self.base_agent, name):
            return getattr(self.base_agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __repr__(self) -> str:
        return f"CommunicatingAgent(id={self.agent_id}, base={self.base_agent})"
