# -*- coding: utf-8 -*-
"""
Модуль сознания для агента.
Реализует глобальное рабочее пространство (Global Workspace Theory):
- Собирает сигналы от всех модулей
- Выбирает доминирующую эмоцию
- Активирует соответствующий алгоритм размышления
- Принимает решение и транслирует его обратно

Каждая эмоция активирует свой алгоритм:
- Страх → алгоритм спасения
- Печаль → алгоритм анализа и смены стратегии
- Любопытство → алгоритм исследования
- Удовольствие → алгоритм повторения успеха
- Гнев → алгоритм решительного действия
- Скука → алгоритм поиска новизны
"""

from typing import Dict, Any, Tuple, Optional
import random


class ConsciousnessModule:
    """
    Модуль сознания, реализующий глобальное рабочее пространство.
    """

    def __init__(self):
        self.last_thought = ""
        self.last_action = None
        self.decision_history = []
        self.thought_history = []
        self.emotion_history = []
        self.step_count = 0
        
        # Настройки эмоциональных порогов
        self.fear_threshold = 0.4
        self.curiosity_threshold = 0.5
        self.surprise_threshold = 0.5
        
        # История для анализа
        self.action_results = []  # (action, reward, success)
        self.strategy_history = []
        
    def think(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Основной метод сознания.
        Мысли — это результаты предсказаний, высвеченные сознанием.
        
        Args:
            signals: словарь с сигналами от всех модулей
                {
                    'brain': {...},       # спайки, эмоции
                    'perception': {...},  # среда, еда, опасность
                    'energy': float,      # уровень энергии
                    'memory': [...],      # история действий
                    'action_history': [...] # предыдущие действия
                    'predictions': {...}, # предсказания будущего
                    'consciousness': {...} # отчёт о сознании
                }
        
        Returns:
            (action, thought) — выбранное действие и мысль
        """
        self.step_count += 1
        
        # 1. Извлекаем эмоции
        emotions = self._extract_emotions(signals)
        self.emotion_history.append(emotions)
        if len(self.emotion_history) > 20:
            self.emotion_history.pop(0)
        
        # 2. Генерируем мысли-кандидаты из предсказаний
        thought_candidates = self._generate_thoughts_from_predictions(signals)
        
        # 3. Сортируем мысли по важности
        thought_candidates.sort(key=lambda x: x['importance'], reverse=True)
        
        # 4. "Высвечиваем" топ-3 мысли в сознании
        self.visible_thoughts = thought_candidates[:3]
        self.thought_candidates = thought_candidates
        
        # 5. Определяем доминирующую эмоцию
        dominant_emotion = self._get_dominant_emotion(emotions)
        
        # 6. Выбираем алгоритм размышления (с учётом высвеченных мыслей)
        thought, action = self._think_with_emotion_and_thoughts(
            dominant_emotion, 
            signals, 
            self.visible_thoughts
        )
        
        # 7. Сохраняем историю
        self.last_thought = thought
        self.last_action = action
        self.thought_history.append((self.step_count, thought, action))
        if len(self.thought_history) > 50:
            self.thought_history.pop(0)
        
        return action, thought
    
    def _generate_thoughts_from_predictions(self, signals: Dict[str, Any]) -> list:
        """Генерирует мысли-кандидаты на основе предсказаний."""
        thoughts = []
        predictions = signals.get('predictions', {})
        perception = signals.get('perception', {})
        brain = signals.get('brain', {})
        energy = signals.get('energy', 100)
        
        # Мысли из предсказаний
        if predictions:
            for horizon, pred in predictions.items():
                if isinstance(pred, (int, float)):
                    thoughts.append({
                        'text': f"Через {horizon} шагов я буду в состоянии {pred:.2f}",
                        'importance': 0.5 + 0.1 * horizon,
                        'source': 'prediction',
                        'horizon': horizon,
                        'value': pred
                    })
                elif isinstance(pred, dict):
                    for key, value in pred.items():
                        if isinstance(value, (int, float)):
                            thoughts.append({
                                'text': f"Через {horizon} шагов {key} будет {value:.2f}",
                                'importance': 0.4 + 0.05 * horizon,
                                'source': 'prediction',
                                'horizon': horizon,
                                'key': key,
                                'value': value
                            })
        
        # Мысли о текущем состоянии (эмоции)
        if brain.get('fear', 0) > 0.5:
            thoughts.append({
                'text': 'Я чувствую страх!',
                'importance': 0.8,
                'source': 'emotion',
                'emotion': 'fear'
            })
        if brain.get('curiosity', 0) > 0.5:
            thoughts.append({
                'text': 'Мне интересно, что там дальше!',
                'importance': 0.5,
                'source': 'emotion',
                'emotion': 'curiosity'
            })
        
        # Мысли о восприятии
        if perception.get('food_nearby', False):
            thoughts.append({
                'text': 'Я вижу еду!',
                'importance': 0.7,
                'source': 'perception',
                'perception': 'food'
            })
        if perception.get('danger_nearby', False):
            thoughts.append({
                'text': 'Опасно!',
                'importance': 0.9,
                'source': 'perception',
                'perception': 'danger'
            })
        
        # Мысли об энергии
        if energy < 30:
            thoughts.append({
                'text': 'У меня мало энергии...',
                'importance': 0.7,
                'source': 'energy',
                'energy': energy
            })
        elif energy > 80:
            thoughts.append({
                'text': 'У меня много энергии!',
                'importance': 0.3,
                'source': 'energy',
                'energy': energy
            })
        
        return thoughts
    
    def _think_with_emotion_and_thoughts(self, emotion: str, signals: Dict[str, Any], visible_thoughts: list) -> Tuple[str, str]:
        """Принимает решение на основе эмоции и высвеченных мыслей."""
        brain = signals.get('brain', {})
        perception = signals.get('perception', {})
        energy = signals.get('energy', 100)
        
        # Выбираем лучшую мысль (самую важную)
        best_thought = visible_thoughts[0]['text'] if visible_thoughts else ""
        
        # Приоритет действий
        if perception.get('danger_nearby', False) and brain.get('fear', 0) > 0.4:
            action = 'flee'
            thought = f"{best_thought} Надо бежать!" if best_thought else "Надо бежать!"
        elif perception.get('food_nearby', False) and energy < 60:
            action = 'collect'
            thought = f"{best_thought} Надо собрать еду!" if best_thought else "Надо собрать еду!"
        elif energy < 20:
            action = 'rest'
            thought = f"{best_thought} Нужно отдохнуть." if best_thought else "Нужно отдохнуть."
        else:
            # Выбор на основе эмоции
            if emotion == 'fear':
                action = 'flee'
                thought = f"{best_thought} Я боюсь!" if best_thought else "Я боюсь!"
            elif emotion == 'curiosity':
                action = 'explore'
                thought = f"{best_thought} Мне интересно!" if best_thought else "Мне интересно!"
            elif emotion == 'pleasure':
                action = 'interact'
                thought = f"{best_thought} Мне нравится!" if best_thought else "Мне нравится!"
            elif emotion == 'sadness':
                action = 'rest'
                thought = f"{best_thought} Я расстроен..." if best_thought else "Я расстроен..."
            elif emotion == 'boredom':
                action = 'explore'
                thought = f"{best_thought} Мне скучно, хочу исследовать!" if best_thought else "Мне скучно, хочу исследовать!"
            else:
                action = 'explore'
                thought = f"{best_thought} Исследую мир." if best_thought else "Исследую мир."
        
        return thought, action
    
    def learn_from_outcome(self, action: str, reward: float, success: bool):
        """
        Обучение на основе результата действия.
        """
        self.action_results.append((action, reward, success))
        if len(self.action_results) > 100:
            self.action_results.pop(0)
    
    def _extract_emotions(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """Извлекает эмоции из сигналов мозга."""
        brain = signals.get('brain', {})
        
        fear = brain.get('fear', 0.0)
        curiosity = brain.get('curiosity', 0.0)
        surprise = brain.get('surprise', 0.0)
        
        # Модифицируем эмоции на основе энергии
        energy = signals.get('energy', 100) / 100.0
        if energy < 0.2:
            fear = min(1.0, fear + 0.3)  # Низкая энергия → страх
        elif energy > 0.7:
            curiosity = min(1.0, curiosity + 0.2)  # Высокая энергия → любопытство
        
        # Эмоции на основе недавних результатов
        if len(self.action_results) >= 3:
            recent_rewards = [r[1] for r in self.action_results[-3:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            if avg_reward < 0:
                # Неудачи → печаль (добавляем как отдельную эмоцию)
                sadness = min(1.0, -avg_reward * 0.5)
                fear = min(1.0, fear + sadness * 0.3)
            elif avg_reward > 1.0:
                # Успехи → удовольствие
                pleasure = min(1.0, avg_reward * 0.3)
                curiosity = min(1.0, curiosity + pleasure * 0.2)
        
        return {
            'fear': fear,
            'curiosity': curiosity,
            'surprise': surprise,
            'sadness': signals.get('sadness', 0.0),
            'pleasure': signals.get('pleasure', 0.0),
            'anger': signals.get('anger', 0.0),
            'boredom': signals.get('boredom', 0.0),
        }
    
    def _get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """Определяет доминирующую эмоцию."""
        # Список эмоций с приоритетом (страх — самый важный для выживания)
        priority_order = ['fear', 'anger', 'sadness', 'pleasure', 'curiosity', 'boredom']
        
        # Находим максимальное значение
        max_emotion = 'curiosity'
        max_value = 0.0
        
        for emotion in priority_order:
            value = emotions.get(emotion, 0.0)
            if value > max_value:
                max_value = value
                max_emotion = emotion
        
        # Если все эмоции низкие → скука
        if max_value < 0.1:
            return 'boredom'
        
        return max_emotion
    
    def _think_with_emotion(self, emotion: str, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Выбирает алгоритм размышления на основе эмоции.
        """
        if emotion == 'fear':
            return self._think_fear(signals)
        elif emotion == 'curiosity':
            return self._think_curiosity(signals)
        elif emotion == 'sadness':
            return self._think_sadness(signals)
        elif emotion == 'pleasure':
            return self._think_pleasure(signals)
        elif emotion == 'anger':
            return self._think_anger(signals)
        else:  # boredom или default
            return self._think_boredom(signals)
    
    # --- АЛГОРИТМЫ РАЗМЫШЛЕНИЯ ---
    
    def _think_fear(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Алгоритм страха: "Как спастись?"
        """
        perception = signals.get('perception', {})
        danger_nearby = perception.get('danger_nearby', False)
        energy = signals.get('energy', 100)
        
        thought = "😨 СТРАХ! Нужно спасаться!"
        
        if danger_nearby:
            thought = "😨 ОПАСНОСТЬ! Бежать! Спасаться!"
            action = 'flee'
        elif energy < 30:
            thought = "😨 Истощение! Нужно отдохнуть и восстановиться!"
            action = 'rest'
        else:
            # Ищем путь к безопасной зоне
            if self._has_safe_zone(signals):
                thought = "😨 Ищу безопасное место..."
                action = 'explore'
            else:
                thought = "😨 Страх, но нужно действовать осторожно..."
                action = 'explore'
        
        return thought, action
    
    def _think_curiosity(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Алгоритм любопытства: "Что новое?"
        """
        perception = signals.get('perception', {})
        food_nearby = perception.get('food_nearby', False)
        energy = signals.get('energy', 100)
        
        thought = "🤔 Любопытно! Что там новенького?"
        
        if food_nearby and energy < 70:
            thought = "🤔 О! Еда! Интересно, как она повлияет на меня?"
            action = 'collect'
        elif energy > 40:
            # Исследуем новое
            thought = "🤔 Исследую мир! Что я ещё не видел?"
            action = 'explore'
        else:
            thought = "🤔 Мало энергии... Но хочется узнать, что вокруг!"
            action = 'explore'
        
        return thought, action
    
    def _think_sadness(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Алгоритм печали: "Что я делаю не так?"
        """
        thought = "😞 Печально... Может, я что-то делаю не так?"
        
        # Анализируем историю действий
        if len(self.action_results) >= 5:
            recent_actions = self.action_results[-5:]
            failures = [r for r in recent_actions if not r[2]]  # success=False
            
            if len(failures) >= 3:
                thought = "😞 Много ошибок... Нужно изменить стратегию!"
                action = self._try_new_strategy(signals)
            else:
                thought = "😞 Нужно проанализировать свои действия..."
                action = 'explore'
        else:
            thought = "😞 Что-то не получается... Попробую по-другому."
            action = 'explore'
        
        return thought, action
    
    def _think_pleasure(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Алгоритм удовольствия: "Как повторить успех?"
        """
        thought = "😊 Приятно! Хочу ещё раз!"
        
        # Находим успешное действие
        success_actions = [r[0] for r in self.action_results if r[2]]
        
        if success_actions:
            # Повторяем последнее успешное действие
            last_success = success_actions[-1]
            thought = f"😊 Отлично! Повторю {last_success}!"
            action = last_success
        else:
            thought = "😊 Попробую что-то новое для удовольствия!"
            action = 'explore'
        
        return thought, action
    
    def _think_anger(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Алгоритм гнева: "Изменить!"
        """
        thought = "😤 НЕПРИЕМЛЕМО! Нужно изменить ситуацию!"
        
        # Решительное действие
        perception = signals.get('perception', {})
        danger_nearby = perception.get('danger_nearby', False)
        
        if danger_nearby:
            thought = "😤 Я не убегу! Я пойду на опасность!"
            action = 'interact'  # решительное взаимодействие
        else:
            thought = "😤 Хватит! Меняю стратегию кардинально!"
            action = self._try_new_strategy(signals)
        
        return thought, action
    
    def _think_boredom(self, signals: Dict[str, Any]) -> Tuple[str, str]:
        """
        Алгоритм скуки: "Что-то новое!"
        """
        thought = "🥱 Скучно... Хочу чего-то нового!"
        
        # Ищем что-то новое
        perception = signals.get('perception', {})
        food_nearby = perception.get('food_nearby', False)
        
        if food_nearby:
            thought = "🥱 О, еда! Хотя бы что-то новое..."
            action = 'collect'
        else:
            thought = "🥱 Исследую новое место!"
            action = 'explore'
        
        return thought, action
    
    # --- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ---
    
    def _has_safe_zone(self, signals: Dict[str, Any]) -> bool:
        """Проверяет, есть ли безопасная зона."""
        perception = signals.get('perception', {})
        danger_nearby = perception.get('danger_nearby', False)
        return not danger_nearby
    
    def _try_new_strategy(self, signals: Dict[str, Any]) -> str:
        """Пробует новую стратегию."""
        actions = ['explore', 'collect', 'interact', 'rest']
        
        # Исключаем действия, которые недавно провалились
        if len(self.action_results) >= 5:
            recent_actions = self.action_results[-5:]
            failed_actions = set(r[0] for r in recent_actions if not r[2])
            available = [a for a in actions if a not in failed_actions]
            if available:
                return random.choice(available)
        
        return random.choice(actions)
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку о состоянии сознания."""
        return {
            'last_thought': self.last_thought,
            'last_action': self.last_action,
            'step_count': self.step_count,
            'decision_count': len(self.decision_history),
            'emotion_count': len(self.emotion_history),
            'action_results_count': len(self.action_results),
        }