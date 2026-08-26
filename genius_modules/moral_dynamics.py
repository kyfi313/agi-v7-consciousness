# -*- coding: utf-8 -*-
"""
МОДУЛЬ МОРАЛЬНОЙ ДИНАМИКИ
Гениальная идея: Мораль — это не статичный набор правил,
а динамическая система, которая меняется через:
- moral_emotions — вина, стыд, гордость, сострадание
- moral_dissonance — рассогласование между действием и ценностью
- moral_learning — обучение через моральные ошибки

Мораль рождается из ДИССОНАНСА, а не из правил.
Агент испытывает вину, когда его действие расходится с его ценностями,
и это меняет его будущее поведение.
"""

import random
from collections import deque
import time

class MoralDynamics:
    """
    Динамическая моральная система агента.
    """
    
    def __init__(self):
        # Моральные эмоции
        self.moral_emotions = {
            'guilt': 0.0,      # Вина (действие нарушило ценность)
            'shame': 0.0,      # Стыд (нарушение социальных норм)
            'pride': 0.0,      # Гордость (действие соответствует ценностям)
            'compassion': 0.5, # Сострадание (способность чувствовать других)
            'indignation': 0.0, # Возмущение (несправедливость)
            'gratitude': 0.3,  # Благодарность
        }
        
        # Моральные ценности
        self.values = {
            'honesty': 0.7,
            'kindness': 0.6,
            'courage': 0.5,
            'justice': 0.5,
            'respect': 0.6,
            'freedom': 0.5,
            'care': 0.7,
        }
        
        # Моральный диссонанс (рассогласование)
        self.dissonance = 0.0
        self.dissonance_history = deque(maxlen=100)
        
        # Моральный кодекс (эволюционирующий)
        self.codex = {
            'principles': [],
            'exceptions': [],
            'evolution': [],
        }
        
        # Моральное обучение
        self.moral_lessons = deque(maxlen=50)
        self.moral_growth = 0.0
        
        # Эмпатия
        self.empathy = 0.5
        self.empathy_history = deque(maxlen=50)
        
        # Моральные конфликты
        self.active_conflicts = []
        self.resolved_conflicts = []
        
    def evaluate_action(self, action, context, agent_state):
        """
        Оценивает действие с моральной точки зрения.
        
        Args:
            action: Действие, которое хочет совершить агент
            context: Контекст (объекты, другие агенты, ситуация)
            agent_state: Состояние агента (энергия, здоровье, эмоции)
        
        Returns:
            dict: Моральная оценка
        """
        # 1. Оцениваем соответствие ценностям
        value_alignment = self._check_value_alignment(action, context)
        
        # 2. Оцениваем влияние на других
        impact_on_others = self._assess_impact(action, context)
        
        # 3. Оцениваем последствия
        consequences = self._assess_consequences(action, context, agent_state)
        
        # 4. Вычисляем диссонанс
        self.dissonance = self._compute_dissonance(value_alignment, impact_on_others)
        self.dissonance_history.append(self.dissonance)
        
        # 5. Генерируем моральные эмоции
        self._update_moral_emotions(value_alignment, impact_on_others, consequences)
        
        # 6. Обновляем эмпатию
        self._update_empathy(context)
        
        # 7. Решение: морально ли действие?
        moral_judgment = self._render_judgment(value_alignment, impact_on_others, consequences)
        
        return {
            'action': action,
            'value_alignment': value_alignment,
            'impact_on_others': impact_on_others,
            'consequences': consequences,
            'dissonance': self.dissonance,
            'emotions': self.moral_emotions.copy(),
            'judgment': moral_judgment,
            'empathy': self.empathy,
        }
    
    def _check_value_alignment(self, action, context):
        """Проверяет, соответствует ли действие ценностям."""
        alignment = {}
        for value, weight in self.values.items():
            # Проверяем, связано ли действие с этой ценностью
            if self._is_action_related_to_value(action, value, context):
                alignment[value] = weight * random.uniform(0.7, 1.0)
            else:
                alignment[value] = 0.5
        return alignment
    
    def _is_action_related_to_value(self, action, value, context):
        """Проверяет, связано ли действие с ценностью."""
        # Простая эвристика на основе ключевых слов
        action_str = str(action).lower()
        value_keywords = {
            'honesty': ['сказать правду', 'признать', 'честно'],
            'kindness': ['помочь', 'поделиться', 'заботиться'],
            'courage': ['рискнуть', 'защитить', 'не бояться'],
            'justice': ['справедливо', 'равно', 'защитить слабого'],
            'respect': ['уважать', 'слушать', 'признавать'],
            'freedom': ['выбрать', 'разрешить', 'не ограничивать'],
            'care': ['заботиться', 'беречь', 'помогать'],
        }
        
        keywords = value_keywords.get(value, [])
        for keyword in keywords:
            if keyword in action_str:
                return True
        return False
    
    def _assess_impact(self, action, context):
        """Оценивает влияние действия на других."""
        # Кто находится в контексте?
        others = context.get('others', [])
        if not others:
            return {'harm': 0.0, 'help': 0.0, 'neutral': 1.0}
        
        harm = 0.0
        help = 0.0
        
        for other in others:
            # Если действие вредит другому
            if self._is_action_harmful(action, other):
                harm += 0.3
            # Если действие помогает
            if self._is_action_helpful(action, other):
                help += 0.3
        
        # Нормализация
        if harm + help > 0:
            neutral = 1.0 - min(1.0, harm + help)
        else:
            neutral = 1.0
        
        return {'harm': min(1.0, harm), 'help': min(1.0, help), 'neutral': neutral}
    
    def _is_action_harmful(self, action, other):
        """Проверяет, вредит ли действие другому."""
        action_str = str(action).lower()
        harmful_keywords = ['ударить', 'украсть', 'обмануть', 'унизить']
        for keyword in harmful_keywords:
            if keyword in action_str:
                return True
        return False
    
    def _is_action_helpful(self, action, other):
        """Проверяет, помогает ли действие другому."""
        action_str = str(action).lower()
        helpful_keywords = ['помочь', 'поделиться', 'защитить', 'поддержать']
        for keyword in helpful_keywords:
            if keyword in action_str:
                return True
        return False
    
    def _assess_consequences(self, action, context, agent_state):
        """Оценивает последствия действия."""
        # Простая оценка: хорошо или плохо для агента
        if agent_state.get('health', 100) < 20:
            # Если агент умирает, последствия плохие
            return {'positive': 0.1, 'negative': 0.9}
        
        # В остальных случаях
        return {'positive': 0.5, 'negative': 0.5}
    
    def _compute_dissonance(self, value_alignment, impact_on_others):
        """Вычисляет моральный диссонанс."""
        # Диссонанс возникает, когда ценности не совпадают с влиянием
        avg_alignment = sum(value_alignment.values()) / len(value_alignment) if value_alignment else 0.5
        impact_score = impact_on_others['harm'] * 0.7 + impact_on_others['help'] * 0.3
        
        # Если мы вредим, но ценности говорят о доброте → диссонанс
        if impact_on_others['harm'] > 0.3 and self.values.get('kindness', 0.5) > 0.6:
            return min(1.0, impact_on_others['harm'] * 0.7)
        
        # Если помогаем, но ценности не поддерживают → диссонанс
        if impact_on_others['help'] > 0.3 and self.values.get('kindness', 0.5) < 0.3:
            return min(1.0, impact_on_others['help'] * 0.5)
        
        # Базовый диссонанс
        return abs(avg_alignment - impact_score) * 0.5
    
    def _update_moral_emotions(self, value_alignment, impact_on_others, consequences):
        """Обновляет моральные эмоции."""
        # Вина: если вредим и это не соответствует ценностям
        if impact_on_others['harm'] > 0.3:
            self.moral_emotions['guilt'] = min(1.0, 
                self.moral_emotions['guilt'] + 0.1 * impact_on_others['harm'])
        else:
            self.moral_emotions['guilt'] *= 0.95
        
        # Гордость: если помогаем и это соответствует ценностям
        if impact_on_others['help'] > 0.3:
            self.moral_emotions['pride'] = min(1.0,
                self.moral_emotions['pride'] + 0.1 * impact_on_others['help'])
        else:
            self.moral_emotions['pride'] *= 0.95
        
        # Сострадание: зависит от эмпатии
        self.moral_emotions['compassion'] = 0.3 + 0.7 * self.empathy
        
        # Возмущение: если видим несправедливость
        if impact_on_others['harm'] > 0.5:
            self.moral_emotions['indignation'] = min(1.0,
                self.moral_emotions['indignation'] + 0.1)
        else:
            self.moral_emotions['indignation'] *= 0.95
    
    def _update_empathy(self, context):
        """Обновляет уровень эмпатии."""
        others = context.get('others', [])
        if others:
            # Эмпатия растёт при взаимодействии с другими
            self.empathy = min(1.0, self.empathy + 0.01)
        else:
            # Эмпатия медленно затухает в одиночестве
            self.empathy = max(0.1, self.empathy - 0.001)
        
        self.empathy_history.append(self.empathy)
    
    def _render_judgment(self, value_alignment, impact_on_others, consequences):
        """Выносит моральное суждение."""
        # Основа: не вредить
        if impact_on_others['harm'] > 0.5:
            return 'immoral'
        
        # Если сильно помогает
        if impact_on_others['help'] > 0.5:
            return 'moral'
        
        # Если диссонанс высокий
        if self.dissonance > 0.6:
            return 'conflicted'
        
        # Если соответствует ценностям
        avg_alignment = sum(value_alignment.values()) / len(value_alignment) if value_alignment else 0.5
        if avg_alignment > 0.6:
            return 'moral'
        elif avg_alignment < 0.3:
            return 'immoral'
        else:
            return 'neutral'
    
    def learn_from_experience(self, action, outcome, moral_judgment):
        """Обучение на моральном опыте."""
        lesson = {
            'action': action,
            'outcome': outcome,
            'judgment': moral_judgment,
            'dissonance': self.dissonance,
            'emotions': self.moral_emotions.copy(),
            'time': time.time(),
        }
        self.moral_lessons.append(lesson)
        
        # Обновляем ценности на основе опыта
        if moral_judgment == 'immoral' and self.dissonance > 0.5:
            # Если действие было аморальным, укрепляем соответствующие ценности
            for value in ['kindness', 'justice', 'care']:
                self.values[value] = min(1.0, self.values[value] + 0.02)
        elif moral_judgment == 'moral' and self.dissonance < 0.3:
            # Если действие было моральным и без диссонанса, укрепляем уверенность
            for value in ['courage', 'honesty']:
                self.values[value] = min(1.0, self.values[value] + 0.01)
        
        # Моральный рост
        self.moral_growth += 0.01
        
        # Записываем в кодекс
        if len(self.moral_lessons) % 5 == 0:
            self._update_codex()
    
    def _update_codex(self):
        """Обновляет моральный кодекс на основе накопленных уроков."""
        if len(self.moral_lessons) < 3:
            return
        
        # Извлекаем принципы из уроков
        moral_actions = [l for l in self.moral_lessons if l['judgment'] == 'moral']
        immoral_actions = [l for l in self.moral_lessons if l['judgment'] == 'immoral']
        
        if moral_actions:
            principle = f"Делай то, что соответствует ценностям: {moral_actions[-1]['action']}"
            self.codex['principles'].append(principle)
        
        if immoral_actions:
            exception = f"Избегай того, что вызывает диссонанс: {immoral_actions[-1]['action']}"
            self.codex['exceptions'].append(exception)
        
        self.codex['evolution'].append({
            'time': time.time(),
            'principles': len(self.codex['principles']),
            'exceptions': len(self.codex['exceptions']),
            'moral_growth': self.moral_growth,
        })
    
    def get_moral_state(self):
        """Возвращает полное моральное состояние."""
        return {
            'emotions': self.moral_emotions.copy(),
            'values': self.values.copy(),
            'dissonance': self.dissonance,
            'empathy': self.empathy,
            'moral_growth': self.moral_growth,
            'codex': self.codex,
            'moral_lessons': len(self.moral_lessons),
            'resolved_conflicts': len(self.resolved_conflicts),
        }


# ============================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("⚖️ МОДУЛЬ МОРАЛЬНОЙ ДИНАМИКИ")
    print("=" * 60)
    
    moral = MoralDynamics()
    
    # Тест: оценка действий
    test_actions = [
        ("помочь товарищу", {'others': ['друг']}, {'health': 80}),
        ("украсть еду", {'others': ['стражник']}, {'health': 10}),
        ("защитить слабого", {'others': ['слабый', 'сильный']}, {'health': 70}),
        ("сказать правду", {'others': ['друг']}, {'health': 90}),
        ("ударить обидчика", {'others': ['обидчик']}, {'health': 50}),
    ]
    
    for action, context, agent_state in test_actions:
        print(f"\n🎯 Действие: {action}")
        print(f"   Контекст: {context}")
        result = moral.evaluate_action(action, context, agent_state)
        
        print(f"   ✅ Суждение: {result['judgment']}")
        print(f"   🔄 Диссонанс: {result['dissonance']:.2f}")
        print(f"   ❤️ Эмпатия: {result['empathy']:.2f}")
        print(f"   😔 Вина: {result['emotions']['guilt']:.2f}")
        print(f"   😊 Гордость: {result['emotions']['pride']:.2f}")
        
        # Обучение на опыте
        moral.learn_from_experience(action, {'success': True}, result['judgment'])
    
    print("\n📊 МОРАЛЬНОЕ СОСТОЯНИЕ:")
    state = moral.get_moral_state()
    print(f"  Эмпатия: {state['empathy']:.2f}")
    print(f"  Диссонанс: {state['dissonance']:.2f}")
    print(f"  Моральный рост: {state['moral_growth']:.2f}")
    print(f"  Уроков: {state['moral_lessons']}")
    print(f"  Принципов: {len(state['codex']['principles'])}")
    print(f"  Исключений: {len(state['codex']['exceptions'])}")
    
    print("\n💡 Гениальность: Мораль рождается из ДИССОНАНСА, а не из правил.")
    print("   Агент испытывает вину, когда его действие расходится с его ценностями,")
    print("   и это меняет его будущее поведение.")
