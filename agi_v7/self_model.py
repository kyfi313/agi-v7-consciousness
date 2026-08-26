# -*- coding: utf-8 -*-
"""
САМОМОДЕЛЬ (Self-Model)
Динамическая модель самого себя: сильные/слабые стороны, привычки, ценности.
"""

import numpy as np
from collections import defaultdict, deque


class SelfModel:
    """
    Динамическая модель самого себя.
    
    Хранит:
    - Черты личности (открытость, добросовестность, экстраверсия, доброжелательность, нейротизм)
    - Оценку навыков (какие навыки есть, насколько сильны)
    - История успехов/неудач для динамической самооценки
    - Ценности и предпочтения
    """
    
    def __init__(self):
        # Большая пятёрка (OCEAN)
        self.traits = {
            'openness': 0.5,      # Открытость опыту
            'conscientiousness': 0.5,  # Добросовестность
            'extraversion': 0.5,  # Экстраверсия
            'agreeableness': 0.5, # Доброжелательность
            'neuroticism': 0.3,   # Нейротизм (эмоциональная стабильность)
        }
        
        # Навыки: {название: { 'level': 0.0-1.0, 'successes': 0, 'failures': 0 }}
        self.skills = {}
        
        # История самооценки (для динамического изменения)
        self.self_esteem_history = deque(maxlen=100)
        self.self_esteem = 0.5  # 0.0-1.0
        
        # Важные события, изменившие самооценку
        self.defining_moments = []
        
        # Предпочтения (что нравится/не нравится)
        self.preferences = defaultdict(float)  # действие -> предпочтение (-1.0 до 1.0)
        
        # Уверенность в разных типах задач
        self.task_confidence = defaultdict(lambda: 0.5)
        
        print("🧠 Самомодель инициализирована")
    
    def update_skill(self, skill_name: str, success: bool, reward: float = 0.0):
        """
        Обновляет оценку навыка на основе результата.
        
        Args:
            skill_name: Название навыка
            success: Успешно ли выполнено действие
            reward: Дополнительная награда (0.0-1.0)
        """
        if skill_name not in self.skills:
            self.skills[skill_name] = {'level': 0.3, 'successes': 0, 'failures': 0}
        
        skill = self.skills[skill_name]
        if success:
            skill['successes'] += 1
            # Уровень повышается с успехом, но с насыщением
            skill['level'] = min(1.0, skill['level'] + 0.05 + reward * 0.1)
        else:
            skill['failures'] += 1
            # Неудача снижает уровень, но не так сильно
            skill['level'] = max(0.0, skill['level'] - 0.03 - reward * 0.05)
        
        # Обновляем самооценку
        self._update_self_esteem(skill_name, success, reward)
    
    def _update_self_esteem(self, skill_name: str, success: bool, reward: float):
        """Обновляет глобальную самооценку на основе опыта"""
        # Если успех в важном навыке → самооценка растёт
        importance = self._get_skill_importance(skill_name)
        
        if success:
            change = 0.02 * importance + reward * 0.03
            self.self_esteem = min(1.0, self.self_esteem + change)
        else:
            change = 0.03 * importance
            self.self_esteem = max(0.0, self.self_esteem - change)
        
        # Запоминаем важные изменения
        if abs(change) > 0.05:
            self.defining_moments.append({
                'skill': skill_name,
                'success': success,
                'change': change,
                'new_esteem': self.self_esteem
            })
            if len(self.defining_moments) > 20:
                self.defining_moments = self.defining_moments[-20:]
    
    def _get_skill_importance(self, skill_name: str) -> float:
        """Оценивает важность навыка для выживания"""
        important_skills = ['flee', 'collect', 'eat', 'rest', 'explore']
        if skill_name in important_skills:
            return 0.8
        return 0.4
    
    def get_skill_level(self, skill_name: str) -> float:
        """Возвращает уровень навыка"""
        if skill_name in self.skills:
            return self.skills[skill_name]['level']
        return 0.3
    
    def update_preference(self, action: str, valence: float):
        """
        Обновляет предпочтение действия на основе валентности.
        
        Args:
            action: Название действия
            valence: Эмоциональная валентность (-1.0 до 1.0)
        """
        old = self.preferences[action]
        self.preferences[action] = old + 0.1 * (valence - old)
        self.preferences[action] = max(-1.0, min(1.0, self.preferences[action]))
    
    def get_preference(self, action: str) -> float:
        """Возвращает предпочтение действия"""
        return self.preferences.get(action, 0.0)
    
    def get_risk_tolerance(self) -> float:
        """
        Возвращает склонность к риску на основе самооценки и нейротизма.
        
        Returns:
            0.0-1.0: чем выше, тем больше склонность к риску
        """
        # Уверенные люди рискуют больше
        esteem_factor = 0.3 + 0.7 * self.self_esteem
        # Высокий нейротизм → меньше риска
        neuroticism_factor = 1.0 - self.traits['neuroticism'] * 0.5
        # Открытость → больше риска
        openness_factor = 0.5 + 0.5 * self.traits['openness']
        
        return min(1.0, esteem_factor * neuroticism_factor * openness_factor)
    
    def get_exploration_bias(self) -> float:
        """
        Возвращает склонность к исследованию.
        
        Returns:
            0.0-1.0: чем выше, тем больше исследование
        """
        return 0.3 + 0.4 * self.traits['openness'] + 0.3 * self.self_esteem
    
    def get_social_tendency(self) -> float:
        """
        Возвращает склонность к социальному взаимодействию.
        
        Returns:
            0.0-1.0: чем выше, тем более социальный
        """
        return 0.3 + 0.4 * self.traits['extraversion'] + 0.3 * self.traits['agreeableness']
    
    def update_trait(self, trait: str, change: float):
        """Обновляет черту личности"""
        if trait in self.traits:
            self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + change))
    
    def get_confidence_for_task(self, task_type: str) -> float:
        """Возвращает уверенность в выполнении задачи"""
        return self.task_confidence[task_type]
    
    def update_confidence(self, task_type: str, success: bool):
        """Обновляет уверенность в задаче"""
        old = self.task_confidence[task_type]
        if success:
            self.task_confidence[task_type] = old + 0.1 * (1.0 - old)
        else:
            self.task_confidence[task_type] = old - 0.1 * old
    
    def get_state(self) -> dict:
        """Возвращает текущее состояние самомодели"""
        return {
            'self_esteem': round(self.self_esteem, 2),
            'risk_tolerance': round(self.get_risk_tolerance(), 2),
            'exploration_bias': round(self.get_exploration_bias(), 2),
            'social_tendency': round(self.get_social_tendency(), 2),
            'traits': {k: round(v, 2) for k, v in self.traits.items()},
            'skills': {k: round(v['level'], 2) for k, v in self.skills.items()},
            'defining_moments': len(self.defining_moments)
        }
    
    def get_summary(self) -> str:
        """Возвращает краткое описание личности"""
        traits = self.traits
        if traits['openness'] > 0.6:
            openness_desc = "любознательный"
        else:
            openness_desc = "осторожный"
        
        if traits['conscientiousness'] > 0.6:
            consc_desc = "организованный"
        else:
            consc_desc = "спонтанный"
        
        if traits['extraversion'] > 0.6:
            extra_desc = "социальный"
        else:
            extra_desc = "замкнутый"
        
        if self.self_esteem > 0.6:
            esteem_desc = "уверенный"
        elif self.self_esteem > 0.4:
            esteem_desc = "умеренный"
        else:
            esteem_desc = "неуверенный"
        
        return f"{openness_desc}, {consc_desc}, {extra_desc}, {esteem_desc}"
