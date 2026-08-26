# -*- coding: utf-8 -*-
"""
HTN-планировщик — иерархическое планирование как у человека

Человеческое планирование:
1. Есть цель (например, "добыть алмаз")
2. Разбивается на подцели ("найти", "добыть", "вернуться")
3. Каждая подцель — это план действий
4. План выполняется по шагам
5. Если что-то идёт не так — перепланирование

"""

import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """Статус выполнения задачи"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Action:
    """Примитивное действие (может быть выполнено напрямую)"""
    name: str
    prerequisites: List[str] = field(default_factory=list)  # Что нужно иметь
    effects: Dict[str, Any] = field(default_factory=dict)  # Что меняется
    cost: float = 1.0  # Стоимость выполнения (энергия, время)
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        """Выполняет действие и возвращает (успех, новое_состояние)"""
        # Проверяем предусловия
        for prereq in self.prerequisites:
            if not state.get(prereq, False):
                return False, state
        
        # Применяем эффекты
        new_state = state.copy()
        for key, value in self.effects.items():
            new_state[key] = value
        
        return True, new_state


@dataclass
class Task:
    """Составная задача (разбивается на подзадачи)"""
    name: str
    subtasks: List[Any]  # Список Task или Action
    description: str = ""
    
    def decompose(self) -> List[Any]:
        """Разбивает задачу на подзадачи"""
        return self.subtasks


class HTNPlanner:
    """
    HTN-планировщик — иерархическое планирование
    
    Принцип работы:
    1. Получает цель (задачу верхнего уровня)
    2. Рекурсивно разбивает на подзадачи
    3. Доходит до примитивных действий
    4. Выполняет их по порядку
    5. Отслеживает статус и перепланирует при неудаче
    """
    
    def __init__(self):
        # База знаний: как достичь целей
        self.methods: Dict[str, List[Task]] = {}
        
        # Доступные примитивные действия
        self.actions: Dict[str, Action] = {}
        
        # Текущий план
        self.current_plan: List[Any] = []
        self.plan_index: int = 0
        self.current_goal: str = ""
        self.status: TaskStatus = TaskStatus.PENDING
        
        # Память о прошлых планах (для обучения)
        self.plan_history: List[Dict] = []
        self.max_history = 100
        
        # Контекст (текущее состояние мира)
        self.world_state: Dict = {}
        
        # Инициализируем базовые методы
        self._init_default_methods()
        
    def _init_default_methods(self):
        """Инициализирует базовые методы планирования (как человеческие инстинкты)"""
        
        # --- ПРИМИТИВНЫЕ ДЕЙСТВИЯ ---
        self.actions["move_to"] = Action(
            name="move_to",
            prerequisites=["has_location"],
            effects={"is_at_target": True},
            cost=0.5
        )
        self.actions["mine"] = Action(
            name="mine",
            prerequisites=["has_pickaxe"],
            effects={"has_ore": True, "energy": -10},
            cost=2.0
        )
        self.actions["craft_pickaxe"] = Action(
            name="craft_pickaxe",
            prerequisites=["has_wood", "has_stone"],
            effects={"has_pickaxe": True, "has_wood": False, "has_stone": False},
            cost=1.5
        )
        self.actions["eat"] = Action(
            name="eat",
            prerequisites=["has_food"],
            effects={"energy": 20, "has_food": False},
            cost=0.3
        )
        self.actions["explore"] = Action(
            name="explore",
            prerequisites=[],
            effects={"has_location": True, "discovered": True},
            cost=1.0
        )
        self.actions["collect"] = Action(
            name="collect",
            prerequisites=[],
            effects={"has_item": True},
            cost=0.5
        )
        self.actions["attack"] = Action(
            name="attack",
            prerequisites=["has_weapon"],
            effects={"enemy_defeated": True, "energy": -15},
            cost=2.5
        )
        self.actions["defend"] = Action(
            name="defend",
            prerequisites=["has_armor"],
            effects={"damage_reduced": True},
            cost=1.0
        )
        self.actions["rest"] = Action(
            name="rest",
            prerequisites=[],
            effects={"energy": 5},
            cost=0.2
        )
        
        # --- МЕТОДЫ ДЛЯ ДОСТИЖЕНИЯ ЦЕЛЕЙ ---
        
        # Цель: добыть ресурс
        self.methods["get_resource"] = [
            Task(
                name="get_resource_plan",
                subtasks=[
                    Task("prepare_tools", [
                        Task("craft_tools", [
                            Action("collect", [], {"has_wood": True}, 0.5),
                            Action("collect", [], {"has_stone": True}, 0.5),
                            Action("craft_pickaxe", ["has_wood", "has_stone"], {"has_pickaxe": True}, 1.5)
                        ])
                    ]),
                    Task("find_resource", [
                        Action("explore", [], {"has_location": True}, 1.0),
                        Action("move_to", ["has_location"], {"is_at_target": True}, 0.5)
                    ]),
                    Task("mine_resource", [
                        Action("mine", ["has_pickaxe"], {"has_ore": True}, 2.0)
                    ]),
                    Task("return", [
                        Action("move_to", ["has_location"], {"is_at_target": True}, 0.5)
                    ])
                ],
                description="Добыча ресурса: подготовка → поиск → добыча → возврат"
            )
        ]
        
        # Цель: выжить
        self.methods["survive"] = [
            Task(
                name="survive_plan",
                subtasks=[
                    Task("check_needs", [
                        Task("if_hungry", [
                            Task("get_food", [
                                Action("collect", [], {"has_food": True}, 0.5),
                                Action("eat", ["has_food"], {"energy": 20}, 0.3)
                            ])
                        ]),
                        Task("if_danger", [
                            Task("defend_self", [
                                Action("defend", ["has_armor"], {"damage_reduced": True}, 1.0),
                                Action("attack", ["has_weapon"], {"enemy_defeated": True}, 2.5)
                            ])
                        ])
                    ])
                ],
                description="Выживание: проверка потребностей → устранение угроз"
            )
        ]
        
        # Цель: исследовать
        self.methods["explore"] = [
            Task(
                name="explore_plan",
                subtasks=[
                    Action("explore", [], {"has_location": True}, 1.0),
                    Action("move_to", ["has_location"], {"is_at_target": True}, 0.5),
                    Task("collect_data", [
                        Action("collect", [], {"has_item": True}, 0.5)
                    ])
                ],
                description="Исследование: движение → сбор данных"
            )
        ]
        
        # Цель: построить
        self.methods["build"] = [
            Task(
                name="build_plan",
                subtasks=[
                    Task("collect_materials", [
                        Action("collect", [], {"has_wood": True}, 0.5),
                        Action("collect", [], {"has_stone": True}, 0.5)
                    ]),
                    Task("construct", [
                        Action("craft_pickaxe", ["has_wood", "has_stone"], {"has_pickaxe": True}, 1.5)
                    ])
                ],
                description="Строительство: сбор материалов → конструкция"
            )
        ]
        
        # Цель: обучить
        self.methods["learn"] = [
            Task(
                name="learn_plan",
                subtasks=[
                    Action("collect", [], {"has_item": True}, 0.5),
                    Task("explore", [
                        Action("explore", [], {"has_location": True}, 1.0)
                    ])
                ],
                description="Обучение: получение опыта через исследование"
            )
        ]
        
        # Цель: взаимодействовать (социальная)
        self.methods["socialize"] = [
            Task(
                name="social_plan",
                subtasks=[
                    Task("find_others", [
                        Action("explore", [], {"has_location": True}, 1.0)
                    ]),
                    Task("communicate", [
                        Action("collect", [], {"has_item": True}, 0.5)
                    ])
                ],
                description="Социальное взаимодействие: найти → обменяться"
            )
        ]
    
    def plan(self, goal: str, world_state: Dict) -> List[Any]:
        """
        Создаёт план для достижения цели
        
        Args:
            goal: цель (например, 'get_resource', 'survive', 'explore')
            world_state: текущее состояние мира
        
        Returns:
            Список действий (план)
        """
        self.current_goal = goal
        self.world_state = world_state
        self.status = TaskStatus.RUNNING
        self.plan_index = 0
        
        # Находим метод для достижения цели
        if goal not in self.methods:
            self.status = TaskStatus.FAILED
            return []
        
        # Берём первый метод для цели
        task = self.methods[goal][0]
        
        # Разворачиваем задачу в план
        plan = self._decompose(task, world_state)
        
        # Сохраняем план в историю
        self.plan_history.append({
            'goal': goal,
            'plan': plan,
            'success': False
        })
        if len(self.plan_history) > self.max_history:
            self.plan_history.pop(0)
        
        self.current_plan = plan
        
        # Если план пустой — провал
        if not plan:
            self.status = TaskStatus.FAILED
        
        return plan
    
    def _decompose(self, task: Any, state: Dict) -> List[Any]:
        """
        Рекурсивно разбивает задачу на примитивные действия
        
        Args:
            task: задача или действие
            state: текущее состояние
        
        Returns:
            Список примитивных действий
        """
        # Если это примитивное действие — возвращаем его
        if isinstance(task, Action):
            # Проверяем, можно ли выполнить действие
            if self._check_prerequisites(task, state):
                return [task]
            else:
                # Если нельзя — пытаемся найти альтернативу
                return self._find_alternative(task, state)
        
        # Если это составная задача — разбиваем
        if isinstance(task, Task):
            plan = []
            for subtask in task.subtasks:
                # Рекурсивно разбиваем подзадачу
                subplan = self._decompose(subtask, state)
                plan.extend(subplan)
                
                # Обновляем состояние после каждой подзадачи
                for action in subplan:
                    if isinstance(action, Action):
                        success, new_state = action.execute(state)
                        if success:
                            state = new_state
                        else:
                            # Если действие провалилось — возвращаем что есть
                            return plan
            return plan
        
        # Неизвестный тип
        return []
    
    def _check_prerequisites(self, action: Action, state: Dict) -> bool:
        """Проверяет, можно ли выполнить действие"""
        for prereq in action.prerequisites:
            if not state.get(prereq, False):
                return False
        return True
    
    def _find_alternative(self, action: Action, state: Dict) -> List[Any]:
        """
        Находит альтернативный способ выполнить действие
        (как человек ищет обходной путь)
        """
        # Если нет кирки — попробовать создать
        if 'has_pickaxe' in action.prerequisites and not state.get('has_pickaxe', False):
            # Проверяем, есть ли материалы для крафта
            if state.get('has_wood', False) and state.get('has_stone', False):
                return [Action('craft_pickaxe', ['has_wood', 'has_stone'], {'has_pickaxe': True}, 1.5)]
            else:
                # Нет материалов — сначала собрать
                alt_plan = []
                if not state.get('has_wood', False):
                    alt_plan.append(Action('collect', [], {'has_wood': True}, 0.5))
                if not state.get('has_stone', False):
                    alt_plan.append(Action('collect', [], {'has_stone': True}, 0.5))
                alt_plan.append(Action('craft_pickaxe', ['has_wood', 'has_stone'], {'has_pickaxe': True}, 1.5))
                return alt_plan
        
        # Если нет еды — собрать
        if 'has_food' in action.prerequisites and not state.get('has_food', False):
            return [Action('collect', [], {'has_food': True}, 0.5)]
        
        # Если нет оружия — сделать или убежать
        if 'has_weapon' in action.prerequisites and not state.get('has_weapon', False):
            return [Action('collect', [], {'has_wood': True}, 0.5), Action('craft_pickaxe', ['has_wood'], {'has_weapon': True}, 1.5)]
        
        # Если нет локации — исследовать
        if 'has_location' in action.prerequisites and not state.get('has_location', False):
            return [Action('explore', [], {'has_location': True}, 1.0)]
        
        # Не нашли альтернативу
        return [action]  # Пробуем выполнить как есть
    
    def execute_next(self, state: Dict) -> Tuple[bool, Dict, str]:
        """
        Выполняет следующий шаг плана
        
        Returns:
            (успех, новое_состояние, сообщение)
        """
        if self.plan_index >= len(self.current_plan):
            self.status = TaskStatus.SUCCESS
            # Отмечаем план как успешный
            if self.plan_history:
                self.plan_history[-1]['success'] = True
            return True, state, "План выполнен успешно"
        
        action = self.current_plan[self.plan_index]
        
        if not isinstance(action, Action):
            # Это не действие — пропускаем
            self.plan_index += 1
            return True, state, f"Пропускаем {action}"
        
        # Выполняем действие
        success, new_state = action.execute(state)
        
        if success:
            self.plan_index += 1
            return True, new_state, f"Выполнено: {action.name}"
        else:
            # Действие провалилось — пытаемся перепланировать
            self.status = TaskStatus.RUNNING
            # Запоминаем неудачу для обучения
            return False, state, f"Ошибка при выполнении: {action.name}"
    
    def replan(self, goal: str, state: Dict) -> List[Any]:
        """
        Перепланирование (адаптация) — как человек корректирует план
        """
        # Учимся на предыдущих ошибках
        failed_plans = [p for p in self.plan_history if not p.get('success', False)]
        
        if len(failed_plans) > 3:
            # Если много неудач — выбираем другую стратегию
            print(f"⚠️ Перепланирование: слишком много неудач ({len(failed_plans)})")
            # Пробуем альтернативный метод
            if goal in self.methods and len(self.methods[goal]) > 1:
                # Используем другой метод
                alt_task = self.methods[goal][1]
                new_plan = self._decompose(alt_task, state)
                if new_plan:
                    self.current_plan = new_plan
                    self.plan_index = 0
                    return new_plan
        
        # Стандартный план
        return self.plan(goal, state)
    
    def get_plan_summary(self) -> str:
        """Возвращает текстовое описание текущего плана"""
        if not self.current_plan:
            return "Нет активного плана"
        
        summary = f"Цель: {self.current_goal}\n"
        summary += f"Шагов: {len(self.current_plan)}\n"
        summary += f"Выполнено: {self.plan_index}/{len(self.current_plan)}\n"
        summary += f"Статус: {self.status.value}\n"
        
        # Показываем оставшиеся шаги
        remaining = self.current_plan[self.plan_index:]
        if remaining:
            summary += "Оставшиеся шаги:\n"
            for i, action in enumerate(remaining[:5]):
                if isinstance(action, Action):
                    summary += f"  {i+1}. {action.name}\n"
        
        return summary
    
    def learn_from_plan(self, goal: str, success: bool, plan: List[Any]) -> None:
        """
        Обучение на основе выполненного плана
        (как человек запоминает, что сработало, а что нет)
        """
        # Сохраняем в историю
        self.plan_history.append({
            'goal': goal,
            'plan': plan,
            'success': success,
            'length': len(plan)
        })
        if len(self.plan_history) > self.max_history:
            self.plan_history.pop(0)
        
        # Если план успешен — запоминаем как хороший метод
        if success:
            if goal in self.methods:
                # Добавляем успешный план как метод
                success_task = Task(
                    name=f"{goal}_learned",
                    subtasks=plan,
                    description="Изученный план"
                )
                self.methods[goal].append(success_task)
    
    def get_suggestions(self, state: Dict) -> List[str]:
        """
        Предлагает возможные цели на основе состояния
        (как человек выбирает, что делать дальше)
        """
        suggestions = []
        
        # Проверяем потребности
        if state.get('energy', 50) < 20:
            suggestions.append("Низкая энергия → цель: выжить (найти еду)")
        
        if state.get('danger', 0) > 0.5:
            suggestions.append("Опасность → цель: выжить (защититься)")
        
        if not state.get('has_location', False):
            suggestions.append("Нет информации → цель: исследовать")
        
        if state.get('has_wood', False) and state.get('has_stone', False):
            suggestions.append("Есть материалы → цель: построить (инструменты)")
        
        if not suggestions:
            suggestions.append("Нет очевидных целей → цель: исследовать")
            suggestions.append("Цель: обучить (получить опыт)")
            suggestions.append("Цель: социальное взаимодействие")
        
        return suggestions
