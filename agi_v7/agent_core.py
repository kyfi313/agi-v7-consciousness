# -*- coding: utf-8 -*-
"""
Единое ядро агента.
Объединяет все модули и предоставляет единый интерфейс.
"""

import sys
import os

# Добавляем путь к нейробиологическим модулям (используем короткое имя)
base_dir = r"C:\Users\CA52~1\Desktop\мои проекты"
modules_path = os.path.join(base_dir, "Новая папка (2)")
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

# Импорт всех модулей
from agi_v7.consciousness_dispatcher import ConsciousAgent, ConsciousnessDispatcher, EvolvableDispatcher
from agi_v7.world_model import WorldModel
from agi_v7.planner import Planner
from agi_v7.theory_of_mind import TheoryOfMind
from agi_v7.metacognition import Metacognition
from agi_v7.communication import CommunicationModule
from agi_v7.meta_learning import MetaLearning
from agi_v7.brain_interface import BrainInterface
from agi_v7.neuro_protocol import NeuroSignal, SignalType, NeuroProtocol
from genius_modules.system1_system2_switch import System1System2Switch
from agi_v7.consciousness_stream import ConsciousnessStream, MindIntegration
from agi_v7.desire_system import DesireSystem
from agi_v7.emotional_memory import EmotionalMemory

# Нейробиологические модули (базальные ганглии, таламус, мозжечок, гиппокамп, кора, лимбика, мост, средний мозг, чёрная субстанция)
from agi_v7.modules.basal_ganglia import BasalGangliaModule
from agi_v7.modules.thalamus import ThalamusModule
from agi_v7.modules.cerebellum import CerebellumModule
from agi_v7.modules.hippocampus import HippocampusModule
from agi_v7.modules.cortex import PrefrontalCortexModule
from agi_v7.modules.limbic import LimbicModule
from agi_v7.modules.pons import PonsModule
from agi_v7.modules.midbrain import MidbrainModule
from agi_v7.modules.substantia_nigra import SubstantiaNigraModule


@dataclass
class AgentState:
    """Состояние агента."""
    position: Tuple[int, int] = (0, 0)
    energy: float = 1.0
    health: float = 1.0
    food: int = 0
    age: int = 0
    alive: bool = True
    emotions: Dict[str, float] = field(default_factory=lambda: {
        'dopamine': 0.0,
        'noradrenaline': 0.0,
        'serotonin': 0.0,
        'fear': 0.0,
        'curiosity': 0.5
    })
    memory: List[Dict[str, Any]] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)


class AgentCore:
    """
    Единое ядро агента с сознанием.
    Интегрирует все модули: WorldModel, Planner, TheoryOfMind, Metacognition,
    Communication, MetaLearning, Consciousness.
    """
    
    def __init__(self, 
                 use_evolution: bool = False,
                 grid_size: int = 16,
                 max_agents: int = 10,
                 planning_horizon: int = 10,
                 num_neurons: int = 200):
        
        # Сознание (диспетчер)
        self.conscious_agent = ConsciousAgent(use_evolution=use_evolution)
        self.dispatcher = self.conscious_agent.dispatcher
        self.use_evolution = use_evolution
        
        # Состояние агента
        self.state = AgentState()
        
        # Мозг (нейронная сеть)
        self.brain = BrainInterface(num_neurons=num_neurons)
        
        # Система 1 / Система 2 (подсознание и сознание)
        self.thinking_switch = System1System2Switch(input_dim=10, output_dim=4)
        self.current_thinking_mode = 'system1'  # 'system1' или 'system2'
        
        # --- НЕЙРОБИОЛОГИЧЕСКИЕ МОДУЛИ ---
        # Базальные ганглии — выбор действий (Q-learning)
        self.basal_ganglia = BasalGangliaModule(state_size=10, action_size=10)
        # Таламус — сенсорное реле
        self.thalamus = ThalamusModule()
        # Мозжечок — двигательное обучение и координация
        self.cerebellum = CerebellumModule()
        # Гиппокамп — эпизодическая память
        self.hippocampus = HippocampusModule()
        # Префронтальная кора — исполнительные функции
        self.prefrontal_cortex = PrefrontalCortexModule()
        # Лимбическая система (миндалина) — эмоции
        self.limbic = LimbicModule()
        # Варолиев мост — связь между мозжечком и корой
        self.pons = PonsModule()
        # Средний мозг — рефлексы и ориентация
        self.midbrain = MidbrainModule()
        # Чёрная субстанция — дофаминовая регуляция
        self.substantia_nigra = SubstantiaNigraModule()
        
        # --- ПОТОК СОЗНАНИЯ И ИНТЕГРАЦИЯ МОДУЛЕЙ ---
        # Поток сознания — непрерывный внутренний монолог
        self.consciousness_stream = ConsciousnessStream(buffer_size=100)
        # Интегратор модулей — позволяет мыслить с помощью модулей
        self.mind_integration = MindIntegration()
        
        # --- АВТОНОМНЫЕ ЦЕЛИ И ЭМОЦИОНАЛЬНАЯ ПАМЯТЬ ---
        # Система желаний (нейронная динамика)
        self.desire_system = DesireSystem(num_desires=8)
        # Эмоциональная память
        self.emotional_memory = EmotionalMemory(capacity=1000)
        
        # Регистрируем все модули в интеграторе
        self._register_modules_in_integrator()
        
        # Модули (уровни 1-9)
        self.world_model = WorldModel(grid_size=grid_size)
        self.planner = Planner(horizon=planning_horizon)
        self.theory_of_mind = TheoryOfMind()
        self.metacognition = Metacognition()
        self.communication = CommunicationModule()
        self.meta_learning = MetaLearning()
        
        # Регистрируем модули в диспетчере
        self._register_modules()
        
        # Счётчик шагов для сна
        self.steps_since_sleep = 0
        self.sleep_interval = 20  # спать каждые 20 шагов
        
        # Внутренний монолог и саморефлексия
        self.self_awareness = {
            'name': 'Агент',
            'purpose': 'Выжить и понять мир',
            'questions': [
                'Кто я?',
                'Зачем я здесь?',
                'Что я чувствую?',
                'Что я хочу?',
                'Что я боюсь?',
                'Что я помню?',
                'Что я узнал сегодня?',
                'Кто я для других?',
                'Что будет, если я умру?',
                'Я один?'
            ],
            'answers': {},
            'monologue': [],
            'self_reflection_step': 0,
            'reflection_interval': 5  # рефлексировать каждые 5 шагов
        }
        
        # Первая мысль в потоке сознания
        self.consciousness_stream.add_thought(
            content="Я просыпаюсь... Кто я? Где я?",
            source='self_awareness',
            emotional_valence=0.3,
            intensity=1.0
        )
    
    def _register_modules_in_integrator(self):
        """Регистрирует все модули в интеграторе для мышления через них."""
        # Список всех модулей, которые будут использоваться как инструменты мышления
        modules_to_register = [
            ('brain', self.brain),
            ('thalamus', self.thalamus),
            ('hippocampus', self.hippocampus),
            ('prefrontal_cortex', self.prefrontal_cortex),
            ('limbic', self.limbic),
            ('basal_ganglia', self.basal_ganglia),
            ('cerebellum', self.cerebellum),
            ('pons', self.pons),
            ('midbrain', self.midbrain),
            ('substantia_nigra', self.substantia_nigra),
            ('world_model', None),  # будет добавлен позже
            ('planner', None),
            ('metacognition', None),
        ]
        for name, module in modules_to_register:
            if module is not None:
                self.mind_integration.register_module(name, module)
    
    def _register_modules(self):
        """Регистрирует все модули в диспетчере сознания."""
        modules = [
            'world_model', 'planner', 'theory_of_mind', 
            'metacognition', 'communication', 'meta_learning'
        ]
        for mod in modules:
            self.dispatcher.register_module(mod)
        
        # Маршруты между модулями
        routes = [
            ('world_model', 'planner'),
            ('world_model', 'theory_of_mind'),
            ('planner', 'metacognition'),
            ('planner', 'world_model'),
            ('theory_of_mind', 'communication'),
            ('theory_of_mind', 'metacognition'),
            ('metacognition', 'planner'),
            ('metacognition', 'world_model'),
            ('communication', 'theory_of_mind'),
            ('communication', 'meta_learning'),
            ('meta_learning', 'planner'),
            ('meta_learning', 'world_model'),
        ]
        for source, target in routes:
            self.dispatcher.add_route(source, target, weight=0.5)
    
    def _think(self, grid: np.ndarray, position: Tuple[int, int], other_agents: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Процесс мышления агента.
        Вместо простого step() — это непрерывный поток сознания.
        """
        other_agents = other_agents or []
        
        # --- 1. ПОДСОЗНАНИЕ ОБРАБАТЫВАЕТ СЫРЫЕ СИГНАЛЫ ---
        # Таламус фильтрует сенсорные данные
        # Мозг (нейроны) обрабатывает и передаёт эмоции
        
        brain_signal = self.brain.step_with_signal()
        
        # Страх, любопытство, удивление
        fear = self.brain.fear_level
        curiosity = self.brain.curiosity_level
        surprise = self.brain.surprise_level
        
        # --- 2. ПЕРЕКЛЮЧЕНИЕ МЕЖДУ SYSTEM 1 И SYSTEM 2 ---
        novelty = surprise * 0.7 + (1.0 - getattr(self.world_model, 'confidence', 0.5)) * 0.3
        error = 1.0 - getattr(self.brain, '_get_survival_rate', lambda: 0.7)()
        
        thinking_output = self.thinking_switch.process(
            input_vector=np.random.randn(10),
            fear=fear,
            novelty=novelty,
            error=error
        )
        self.current_thinking_mode = self.thinking_switch.mode
        
        # Применяем режим к мозгу
        if self.current_thinking_mode == 'system1':
            self.brain.chain_length_multiplier = 0.3
            self.brain.recurrent_gain = 0.2
            self.brain.depth_penetration = 0.3
            mode_label = '⚡ быстрое мышление (подсознание)'
        else:
            self.brain.chain_length_multiplier = 1.0
            self.brain.recurrent_gain = 0.8
            self.brain.depth_penetration = 0.9
            mode_label = '🧠 медленное мышление (сознание)'
        
        # --- 3. ПОТОК СОЗНАНИЯ — ВНУТРЕННИЙ МОНОЛОГ ---
        # Агент осознаёт, что он чувствует и что происходит
        
        # Эмоциональное состояние
        if fear > 0.6:
            self.consciousness_stream.add_thought(
                content=f"Я чувствую страх! Что-то угрожает мне.",
                source='emotion',
                emotional_valence=-0.8,
                intensity=fear
            )
        elif curiosity > 0.6:
            self.consciousness_stream.add_thought(
                content=f"Мне интересно! Что это? Я хочу узнать.",
                source='emotion',
                emotional_valence=0.7,
                intensity=curiosity
            )
        
        # Режим мышления
        self.consciousness_stream.add_thought(
            content=f"Я думаю в режиме: {mode_label}",
            source='metacognition',
            emotional_valence=0.2,
            intensity=0.5
        )
        
        # --- 4. МЫШЛЕНИЕ ЧЕРЕЗ МОДУЛИ (как продолжение мышления) ---
        # Агент использует модули как инструменты мышления
        
        # Обращение к памяти (мгновенно, как продолжение мышления)
        memory_query = {'position': position, 'time': self.state.age}
        recalled = self.mind_integration.recall_from_memory(memory_query)
        if recalled:
            self.consciousness_stream.add_thought(
                content=f"Я вспоминаю что-то похожее...",
                source='memory',
                emotional_valence=0.1,
                intensity=0.4
            )
        
        # Обновление модели мира (через интегратор)
        grid_str = [[str(cell) for cell in row] for row in grid]
        self.world_model.observe(grid_str, position, other_agents)
        
        # Планирование (через интегратор)
        plan_obj = self.planner.plan(
            grid=grid_str,
            position=position,
            world_model=self.world_model,
            goal='collect_food'
        )
        
        if plan_obj and hasattr(plan_obj, 'actions') and plan_obj.actions:
            action = plan_obj.actions[0]
            self.consciousness_stream.add_thought(
                content=f"Я планирую: {action}. Уверенность: {plan_obj.confidence:.2f}",
                source='planning',
                emotional_valence=0.3,
                intensity=plan_obj.confidence
            )
        else:
            action = 'explore'
            self.consciousness_stream.add_thought(
                content="Я исследую мир, я не знаю, что делать.",
                source='planning',
                emotional_valence=-0.2,
                intensity=0.3
            )
        
        # --- 5. ОСОЗНАНИЕ ДЕЙСТВИЙ (обратная связь) ---
        # Агент анализирует свои действия и осознаёт себя через этот анализ
        
        self.consciousness_stream.add_thought(
            content=f"Я собираюсь сделать: {action}",
            source='action_awareness',
            emotional_valence=0.1,
            intensity=0.6
        )
        
        # Обновляем самосознание
        self.consciousness_stream.update_self_awareness(
            feeling='страх' if fear > 0.6 else 'интерес' if curiosity > 0.6 else 'спокойствие',
            wanting='выжить' if fear > 0.6 else 'узнать новое' if curiosity > 0.6 else 'исследовать',
            remembering=str(recalled)[:50] if recalled else 'ничего конкретного',
            planning=action
        )
        
        # --- 6. ВОЗВРАЩАЕМ РЕЗУЛЬТАТ ---
        return {
            'action': action,
            'plan': plan_obj,
            'thoughts': list(self.consciousness_stream.thoughts)[-5:],
            'consciousness_summary': self.consciousness_stream.get_conscious_summary(),
            'mode': self.current_thinking_mode,
            'emotions': {'fear': fear, 'curiosity': curiosity, 'surprise': surprise}
        }
    
    def step(self, grid: np.ndarray, position: Tuple[int, int], 
             other_agents: Optional[List[Dict]] = None) -> Tuple[str, str]:
        """
        Один шаг агента.
        
        Args:
            grid: 2D массив с типами клеток (0=пусто, 1=еда, 2=опасность)
            position: позиция агента (x, y)
            other_agents: список других агентов с их позициями и состояниями
        
        Returns:
            (action, thought) — действие и мысль
        """
        other_agents = other_agents or []
        
        # Обновляем позицию
        self.state.position = position
        self.state.age += 1
        self.steps_since_sleep += 1
        
        # Преобразуем grid в строковый формат для совместимости с модулями
        grid_str = [[str(cell) for cell in row] for row in grid]
        for y in range(len(grid_str)):
            for x in range(len(grid_str[y])):
                if grid_str[y][x] == '0':
                    grid_str[y][x] = '·'
                elif grid_str[y][x] == '1':
                    grid_str[y][x] = '🍎'
                elif grid_str[y][x] == '2':
                    grid_str[y][x] = '⚠️'
        
        # 1. Обновляем модель мира
        self.world_model.observe(grid_str, position, other_agents)
        
        # 1.5. Мозг (нейронная сеть) обрабатывает текущее состояние
        brain_signal = self.brain.step_with_signal()
        
        # 1.5.5. Система 1/2: переключаем режим мышления на основе эмоций
        fear = self.brain.fear_level
        curiosity = self.brain.curiosity_level
        surprise = self.brain.surprise_level
        
        # Вычисляем новизну и ошибку на основе текущего состояния
        novelty = surprise * 0.7 + (1.0 - self.world_model.confidence) * 0.3 if hasattr(self.world_model, 'confidence') else surprise * 0.5
        error = 1.0 - self.brain._get_survival_rate() if hasattr(self.brain, '_get_survival_rate') else 0.3
        
        # Переключатель выбирает режим
        thinking_output = self.thinking_switch.process(
            input_vector=np.random.randn(10),  # контекстный вектор (в будущем можно передать реальные данные)
            fear=fear,
            novelty=novelty,
            error=error
        )
        self.current_thinking_mode = self.thinking_switch.mode
        
        # Применяем режим к мозгу: меняем длину цепочек
        if self.current_thinking_mode == 'system1':
            # Быстрое мышление — короткие цепочки, экономия энергии
            self.brain.chain_length_multiplier = 0.3
            self.brain.recurrent_gain = 0.2
            self.brain.depth_penetration = 0.3
        else:  # system2
            # Медленное мышление — длинные цепочки, глубокий анализ
            self.brain.chain_length_multiplier = 1.0
            self.brain.recurrent_gain = 0.8
            self.brain.depth_penetration = 0.9
        
        # 1.6. Обновляем нейромедиаторы на основе эмоций мозга
        self.dispatcher.update_neurotransmitters({
            'fear': self.brain.fear_level,
            'curiosity': self.brain.curiosity_level,
            'surprise': self.brain.surprise_level,
            'pleasure': 0.3 + 0.5 * (1.0 - self.brain.fear_level),
            'anger': 0.0,
            'sadness': 0.0
        })
        
        # Отправляем сигнал от мозга в диспетчер сознания (в конкуренцию за внимание)
        # Если активна система 2 — усиливаем сигнал (сознание активно)
        signal_strength = brain_signal.strength
        if self.current_thinking_mode == 'system2':
            signal_strength *= 1.5  # сознание усиливает сигнал
        
        self.dispatcher.route_signal(
            source='brain',
            data=brain_signal.to_dict(),
            strength=signal_strength,
            context=self.state.__dict__
        )
        # Добавляем сигнал в пул конкуренции GWT
        self.dispatcher.competition_pool.append(
            Signal('brain', 'consciousness', brain_signal.to_dict(), signal_strength)
        )
        
        # 2. Планируем действие
        plan_obj = self.planner.plan(
            grid=grid_str,
            position=position,
            world_model=self.world_model,
            goal='collect_food'
        )
        
        # 2. Планируем действие (используем планировщик для выбора действия)
        plan_obj = self.planner.plan(
            grid=grid_str,
            position=position,
            world_model=self.world_model,
            goal='collect_food'
        )
        
        # Извлекаем действие из плана
        if plan_obj and hasattr(plan_obj, 'actions') and plan_obj.actions:
            action = plan_obj.actions[0]
            thought = f"План на {len(plan_obj.actions)} шагов, уверенность {plan_obj.confidence:.2f}"
        else:
            # Если план не построен, используем модель мира для выбора направления
            food_hotspots = self.world_model.get_food_hotspots()
            if food_hotspots:
                # Идём к ближайшему предсказанному источнику еды
                x, y = position
                target = min(food_hotspots, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                dx = target[0] - x
                dy = target[1] - y
                if abs(dx) >= abs(dy):
                    action = 'right' if dx > 0 else 'left'
                else:
                    action = 'down' if dy > 0 else 'up'
                thought = f"Иду к предсказанной еде {target}"
            else:
                action = 'explore'
                thought = 'Исследую мир'
        
        # 3. Теория разума: оцениваем других агентов
        if other_agents and hasattr(self.theory_of_mind, 'update'):
            self.theory_of_mind.update(other_agents, self.state.emotions)
        
        # 4. Метапознание: оцениваем собственное мышление
        if hasattr(self.metacognition, 'reflect'):
            metacog_feedback = self.metacognition.reflect(
                action=action,
                state=self.state.__dict__,
                success=True
            )
        else:
            metacog_feedback = {'confidence': 0.5, 'improvement': 0.0}
        
        # 5. Коммуникация: обмениваемся сообщениями
        if hasattr(self.communication, 'process'):
            messages = self.communication.process(
                sender='agent',
                messages=[{'from': 'self', 'content': 'Я действую'}],
                context={'other_agents': other_agents}
            )
        else:
            messages = []
        
        # 6. Мета-обучение: адаптируем параметры
        if hasattr(self.meta_learning, 'update'):
            self.meta_learning.update(
                action=action,
                reward=0.1,
                state=self.state.__dict__
            )
        
        # 7. Сознание: диспетчер обрабатывает все сигналы (всегда в формате словаря)
        world_summary = self.world_model.get_memory_summary() if hasattr(self.world_model, 'get_memory_summary') else {}
        plan_data = {
            'actions': plan_obj.actions if plan_obj and hasattr(plan_obj, 'actions') else ['explore'],
            'confidence': plan_obj.confidence if plan_obj and hasattr(plan_obj, 'confidence') else 0.5,
            'reward': plan_obj.predicted_reward if plan_obj and hasattr(plan_obj, 'predicted_reward') else 0.0
        }
        signals = [
            ('world_model', world_summary, 0.6),
            ('planner', plan_data, 0.7),
            ('theory_of_mind', {'agents': len(other_agents)}, 0.5),
            ('metacognition', metacog_feedback, 0.4),
            ('communication', messages, 0.3),
            ('meta_learning', {'learning_rate': getattr(self.meta_learning, 'learning_rate', 0.1)}, 0.2),
        ]
        
        # Пропускаем сигналы через диспетчер
        for source, data, strength in signals:
            self.dispatcher.route_signal(source, data, strength, context=self.state.__dict__)
            # Добавляем сигналы в пул конкуренции GWT
            self.dispatcher.competition_pool.append(
                Signal(source, 'consciousness', data, strength)
            )
        
        # 7.5. Запускаем глобальное рабочее пространство (конкуренция за внимание)
        gwt_result = self.dispatcher.process_consciousness(self.state.__dict__)
        if gwt_result.get('broadcasted'):
            # Сигнал попал в сознание — он доступен всем модулям
            winner = gwt_result['winner']
            # Если сознание выбрало сигнал от мозга, усиливаем его влияние
            if winner and winner.source == 'brain':
                # Эмоциональное состояние модулирует действие
                thought += f" (сознание усилило сигнал мозга, сила={gwt_result['winner_strength']:.2f})"
                # Можем скорректировать действие на основе сигнала мозга
                if hasattr(self.brain, 'fear_level') and self.brain.fear_level > 0.6:
                    # Страх → избегание опасности
                    if grid[position[0], position[1]] == 2:
                        action = 'flee'
                        thought += " 🏃 Избегаю опасности!"
        
        # Добавляем информацию о режиме мышления в состояние
        self.state.thinking_mode = self.current_thinking_mode
        if self.current_thinking_mode == 'system1':
            thought += " (⚡ быстрое мышление — подсознание)"
        else:
            thought += " (🧠 медленное мышление — сознание)"
        
        # 8. Саморефлексия и внутренний монолог
        self._self_reflect(grid, position, action, thought)
        
        # 9. Онлайн-обучение сознания на основе награды (до движения)
        reward = self._calculate_reward(grid, position)
        success = reward > 0.1
        
        self.dispatcher.learn_from_step(
            action=action,
            reward=reward,
            success=success,
            context=self.state.emotions
        )
        
        # 10. Применяем действие (движение) ПОСЛЕ обучения
        # Используем действие от планировщика
        x, y = position
        new_pos = position
        
        # Проверяем, не пытается ли агент съесть еду
        if action == 'eat':
            # Ищем еду рядом
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if grid[nx, ny] == 1:  # еда
                            new_pos = (nx, ny)
                            break
                if new_pos != position:
                    break
            if new_pos == position:
                # Нет еды рядом, двигаемся случайно
                import random
                actions = ['up', 'down', 'left', 'right']
                for _ in range(3):
                    rand_action = random.choice(actions)
                    if rand_action == 'up' and y > 0:
                        new_pos = (x, y - 1)
                        break
                    elif rand_action == 'down' and y < grid.shape[1] - 1:
                        new_pos = (x, y + 1)
                        break
                    elif rand_action == 'left' and x > 0:
                        new_pos = (x - 1, y)
                        break
                    elif rand_action == 'right' and x < grid.shape[0] - 1:
                        new_pos = (x + 1, y)
                        break
        else:
            # Двигаемся по плану
            if action == 'up' and y > 0:
                new_pos = (x, y - 1)
            elif action == 'down' and y < grid.shape[1] - 1:
                new_pos = (x, y + 1)
            elif action == 'left' and x > 0:
                new_pos = (x - 1, y)
            elif action == 'right' and x < grid.shape[0] - 1:
                new_pos = (x + 1, y)
            else:
                # Случайное движение
                import random
                actions = ['up', 'down', 'left', 'right']
                for _ in range(3):
                    rand_action = random.choice(actions)
                    if rand_action == 'up' and y > 0:
                        new_pos = (x, y - 1)
                        break
                    elif rand_action == 'down' and y < grid.shape[1] - 1:
                        new_pos = (x, y + 1)
                        break
                    elif rand_action == 'left' and x > 0:
                        new_pos = (x - 1, y)
                        break
                    elif rand_action == 'right' and x < grid.shape[0] - 1:
                        new_pos = (x + 1, y)
                        break
        
        # Проверяем, что на позиции
        nx, ny = new_pos
        if grid[nx, ny] == 1:  # еда
            self.state.food += 1
            grid[nx, ny] = 0  # съедаем
            thought += " 🍎 Съел еду!"
        elif grid[nx, ny] == 2:  # опасность
            self.state.health -= 0.2
            thought += " ⚠️ Попал в опасность!"
        
        # Обновляем позицию агента
        self.state.position = new_pos
        
        # 10. Сон (консолидация памяти + восстановление энергии) каждые N шагов
        if self.steps_since_sleep >= self.sleep_interval:
            self.dispatcher.sleep(steps=10)
            # Сон восстанавливает энергию
            self.state.energy = min(1.0, self.state.energy + 0.1)
            self.steps_since_sleep = 0
        
        # Сохраняем мысль
        self.state.thoughts.append(thought)
        if len(self.state.thoughts) > 50:
            self.state.thoughts.pop(0)
        
        return action, thought
    
    def _calculate_reward(self, grid: np.ndarray, position: Tuple[int, int]) -> float:
        """
        Рассчитывает награду за текущий шаг.
        """
        x, y = position
        reward = -0.05  # штраф за каждый шаг (чтобы не стоять на месте)
        
        # Еда (1) даёт большую положительную награду
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
            cell = grid[x, y]
            if cell == 1:  # еда
                reward += 2.0  # увеличенная награда
                self.state.food += 1
                self.state.energy = min(1.0, self.state.energy + 0.25)  # восстановление энергии
            elif cell == 2:  # опасность
                reward -= 1.0
                self.state.health -= 0.2
        
        # Энергия: базальный метаболизм + стоимость мышления + сознание (смягчённая модель)
        basal_cost = 0.002  # поддержание нейронов (было 0.005)
        thinking_cost = 0.005 * len([m for m in ['world_model', 'planner', 'theory_of_mind', 'metacognition', 'communication', 'meta_learning'] if hasattr(self, m)])  # было 0.01
        consciousness_cost = 0.008 * self.dispatcher.consciousness_level  # было 0.015
        total_cost = basal_cost + thinking_cost + consciousness_cost
        self.state.energy -= total_cost
        
        # Восстановление энергии от еды (уже добавлено выше)
        # Если энергия падает ниже 0.15 — кризис (было 0.2)
        if self.state.energy < 0.15:
            reward -= 0.1
        if self.state.energy < 0:
            self.state.energy = 0
            self.state.alive = False
            reward -= 1.0
        
        # Если еды много, дополнительная награда (подкрепление)
        if self.state.food > 3:
            reward += 0.2 * min(1.0, self.state.food / 10.0)
        
        # Если здоровье низкое, штраф
        if self.state.health < 0.3:
            reward -= 0.3
        
        # Бонус за выживание (каждый шаг)
        if self.state.alive:
            reward += 0.02
        
        return reward
    
    def _self_reflect(self, grid: np.ndarray, position: Tuple[int, int], 
                      action: str, thought: str):
        """
        Внутренний монолог и саморефлексия агента.
        Агент задаёт себе вопросы и размышляет о себе.
        """
        self.self_awareness['self_reflection_step'] += 1
        
        # Добавляем мысль в монолог
        monologue_entry = {
            'step': self.state.age,
            'position': position,
            'action': action,
            'thought': thought,
            'energy': self.state.energy,
            'food': self.state.food,
            'emotions': self.state.emotions.copy()
        }
        self.self_awareness['monologue'].append(monologue_entry)
        if len(self.self_awareness['monologue']) > 20:
            self.self_awareness['monologue'].pop(0)
        
        # Размышления каждые N шагов
        if self.self_awareness['self_reflection_step'] % self.self_awareness['reflection_interval'] == 0:
            # Выбираем случайный вопрос из списка
            import random
            question = random.choice(self.self_awareness['questions'])
            
            # Формируем ответ на основе состояния
            answer = self._answer_question(question)
            self.self_awareness['answers'][question] = answer
            
            # Добавляем размышление в мысли
            reflection = f"🤔 {question} -> {answer}"
            self.state.thoughts.append(reflection)
            if len(self.state.thoughts) > 50:
                self.state.thoughts.pop(0)
    
    def _answer_question(self, question: str) -> str:
        """Агент отвечает на вопрос о себе."""
        if question == 'Кто я?':
            return f"Я агент, я ищу еду и избегаю опасностей. Мне {self.state.age} шагов."
        elif question == 'Зачем я здесь?':
            if self.state.food > 0:
                return f"Я здесь, чтобы выжить. Я уже нашёл {self.state.food} еды."
            else:
                return "Я здесь, чтобы найти еду и понять этот мир."
        elif question == 'Что я чувствую?':
            emo = self.state.emotions
            feeling = []
            if emo.get('dopamine', 0) > 0.3:
                feeling.append("радость")
            if emo.get('noradrenaline', 0) > 0.3:
                feeling.append("возбуждение")
            if emo.get('serotonin', 0) > 0.3:
                feeling.append("спокойствие")
            if emo.get('fear', 0) > 0.3:
                feeling.append("страх")
            if emo.get('curiosity', 0) > 0.5:
                feeling.append("любопытство")
            if not feeling:
                feeling.append("спокойствие")
            return f"Я чувствую {', '.join(feeling)}. Энергия: {self.state.energy:.2f}"
        elif question == 'Что я хочу?':
            if self.state.energy < 0.3:
                return "Я хочу найти еду, я голоден!"
            elif self.state.food < 3:
                return "Я хочу собрать больше еды, чтобы выжить."
            else:
                return "Я хочу исследовать мир и понять, что здесь происходит."
        elif question == 'Что я боюсь?':
            if self.state.health < 0.5:
                return "Я боюсь умереть от ран."
            elif self.state.energy < 0.2:
                return "Я боюсь умереть от голода."
            else:
                return "Я боюсь опасностей, которые могут мне навредить."
        elif question == 'Что я помню?':
            memory_summary = self.world_model.get_memory_summary() if hasattr(self.world_model, 'get_memory_summary') else {}
            food_seen = memory_summary.get('food_hotspots', 0)
            if isinstance(food_seen, int):
                return f"Я помню, что видел еду в {food_seen} местах."
            elif food_seen:
                return f"Я помню, что видел еду в {len(food_seen)} местах."
            else:
                return "Я пока мало что помню, я только учусь."
        elif question == 'Что я узнал сегодня?':
            return f"Я узнал, что еда даёт мне энергию, а опасности отнимают здоровье. Я собрал {self.state.food} еды."
        elif question == 'Кто я для других?':
            return "Я — часть этого мира. Другие агенты видят меня как существо, которое ищет еду."
        elif question == 'Что будет, если я умру?':
            return "Если я умру, я перестану существовать. Поэтому я буду бороться за жизнь."
        elif question == 'Я один?':
            return "Я не знаю, есть ли другие такие же, как я. Но я продолжаю искать."
        else:
            return "Я размышляю об этом..."
    
    def get_state(self) -> AgentState:
        """Возвращает текущее состояние агента."""
        return self.state
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Возвращает отчёт о сознании."""
        return self.dispatcher.get_consciousness_report()
    
    def get_summary(self) -> str:
        """Краткое описание состояния агента."""
        return (f"AgentCore: pos={self.state.position}, "
                f"energy={self.state.energy:.2f}, "
                f"food={self.state.food}, "
                f"consciousness={self.dispatcher.consciousness_level:.2f}, "
                f"alive={self.state.alive}")
