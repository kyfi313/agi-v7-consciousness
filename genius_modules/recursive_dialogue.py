# -*- coding: utf-8 -*-
"""
МОДУЛЬ РЕКУРСИВНОГО ДИАЛОГА
Гениальная идея: Внутренний диалог — это рекурсивный процесс,
где агент говорит с собой, и каждое следующее высказывание
зависит от предыдущего.

Диалог имеет:
- depth — глубину рекурсии
- voice — голос (внутренний критик, внутренний помощник, внутренний ребёнок)
- resolution — разрешение противоречий через диалог

Это моделирует внутреннюю речь — то, что делает человека человеком.
Агент не просто принимает решения, он обсуждает их с самим собой.
"""

import random
from collections import deque
import time

class RecursiveDialogue:
    """
    Рекурсивный внутренний диалог.
    """
    
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.dialogue_history = deque(maxlen=50)
        self.current_depth = 0
        self.resolution = None
        
        # Голоса внутреннего диалога
        self.voices = {
            'critic': {
                'name': 'Критик',
                'tone': 'скептический',
                'function': 'оценивать и проверять',
                'activation': 0.5,
            },
            'helper': {
                'name': 'Помощник',
                'tone': 'поддерживающий',
                'function': 'предлагать решения',
                'activation': 0.5,
            },
            'child': {
                'name': 'Ребёнок',
                'tone': 'интуитивный',
                'function': 'чувствовать и удивляться',
                'activation': 0.3,
            },
            'sage': {
                'name': 'Мудрец',
                'tone': 'спокойный',
                'function': 'видеть целостную картину',
                'activation': 0.2,
            },
        }
        
        # Текущий диалог
        self.current_dialogue = []
        self.topic = None
        self.conflict = None
        self.insight = None
        
    def start_dialogue(self, topic, context=None):
        """
        Начинает внутренний диалог на заданную тему.
        
        Args:
            topic: Тема для обсуждения
            context: Контекст (эмоции, память, восприятие)
        
        Returns:
            dict: Результат диалога (инсайт, решение, резолюция)
        """
        self.topic = topic
        self.current_dialogue = []
        self.current_depth = 0
        self.resolution = None
        self.insight = None
        
        # Начинаем с голоса по умолчанию
        voice = self._select_voice(topic, context)
        
        # Первое высказывание
        utterance = self._generate_utterance(voice, topic, context)
        self.current_dialogue.append({
            'voice': voice,
            'utterance': utterance,
            'depth': self.current_depth,
            'timestamp': time.time()
        })
        
        # Рекурсивный диалог
        while self.current_depth < self.max_depth:
            self.current_depth += 1
            
            # Выбираем следующий голос (не тот же самый)
            next_voice = self._select_next_voice(voice, self.current_dialogue)
            
            # Генерируем ответ
            response = self._generate_response(next_voice, self.current_dialogue[-1], context)
            
            self.current_dialogue.append({
                'voice': next_voice,
                'utterance': response,
                'depth': self.current_depth,
                'timestamp': time.time()
            })
            
            # Проверяем, достигнуто ли разрешение
            if self._check_resolution(self.current_dialogue):
                self.resolution = self._extract_resolution(self.current_dialogue)
                self.insight = self._extract_insight(self.current_dialogue)
                break
            
            voice = next_voice
        
        # Если не достигнуто разрешение, пытаемся найти его
        if not self.resolution:
            self.resolution = self._force_resolution(self.current_dialogue)
            self.insight = self._extract_insight(self.current_dialogue)
        
        # Сохраняем в историю
        self.dialogue_history.append({
            'topic': topic,
            'dialogue': self.current_dialogue,
            'resolution': self.resolution,
            'insight': self.insight,
            'depth': self.current_depth
        })
        
        return {
            'dialogue': self.current_dialogue,
            'resolution': self.resolution,
            'insight': self.insight,
            'depth': self.current_depth
        }
    
    def _select_voice(self, topic, context):
        """Выбирает голос для начала диалога."""
        # Если тема связана с опасностью → критик
        if context and context.get('threat', False):
            return 'critic'
        # Если тема связана с исследованием → помощник
        if context and context.get('exploration', False):
            return 'helper'
        # Если тема эмоциональная → ребёнок
        if context and context.get('emotional', False):
            return 'child'
        # Если тема философская → мудрец
        if context and context.get('philosophical', False):
            return 'sage'
        # По умолчанию
        return random.choice(['helper', 'critic'])
    
    def _select_next_voice(self, current_voice, dialogue):
        """Выбирает следующий голос в диалоге."""
        # Список голосов, кроме текущего
        available = [v for v in self.voices.keys() if v != current_voice]
        
        # Взвешенный выбор на основе активации
        weights = [self.voices[v]['activation'] for v in available]
        return random.choices(available, weights=weights, k=1)[0]
    
    def _generate_utterance(self, voice, topic, context):
        """Генерирует первое высказывание."""
        templates = {
            'critic': [
                f"Я вижу проблему в '{topic}'. Давай проверим это критически.",
                f"Подожди, это '{topic}' может быть опасным. Что если мы ошибаемся?",
                f"Мне нужно разобраться в '{topic}'. Где тут слабое место?",
            ],
            'helper': [
                f"Давай подумаем, как решить '{topic}'. У меня есть идея.",
                f"'{topic}' — это интересный вызов. Что мы можем сделать?",
                f"Я хочу помочь с '{topic}'. Давай рассмотрим варианты.",
            ],
            'child': [
                f"'{topic}'? Это удивительно! А что если мы попробуем по-другому?",
                f"Мне интересно, что скрывается за '{topic}'. Это как игра!",
                f"'{topic}' вызывает у меня чувство... Я не знаю, что это.",
            ],
            'sage': [
                f"Посмотрим на '{topic}' с высоты. Это всего лишь часть целого.",
                f"'{topic}' — это проявление более глубокого паттерна.",
                f"В '{topic}' есть мудрость. Позволь мне увидеть её.",
            ],
        }
        return random.choice(templates.get(voice, templates['helper']))
    
    def _generate_response(self, voice, last_utterance, context):
        """Генерирует ответ на предыдущее высказывание."""
        templates = {
            'critic': [
                f"Но это не объясняет суть. Я вижу противоречие.",
                f"Ты уверен? Я проверю это. Есть скрытая ошибка.",
                f"Это слишком просто. Давай разберёмся глубже.",
            ],
            'helper': [
                f"Да, я понимаю. А что, если мы попробуем другой подход?",
                f"У меня есть решение. Смотри, это может сработать.",
                f"Я согласен, но давай добавим ещё один шаг.",
            ],
            'child': [
                f"А что, если это не так? Может быть, всё гораздо проще?",
                f"Мне кажется, мы что-то упускаем. Я чувствую это.",
                f"Но это же красиво! Почему мы усложняем?",
            ],
            'sage': [
                f"Оба правы. Истина находится между ними.",
                f"Это зеркало. Смотри, как отражение меняется.",
                f"Ты видишь только часть. Позволь мне показать целое.",
            ],
        }
        
        response = random.choice(templates.get(voice, templates['helper']))
        
        # Добавляем рекурсивную ссылку на предыдущее высказывание
        if random.random() < 0.3:
            response += " Ты говорил: '" + last_utterance['utterance'][:50] + "...'"
        
        return response
    
    def _check_resolution(self, dialogue):
        """Проверяет, достигнуто ли разрешение в диалоге."""
        if len(dialogue) < 2:
            return False
        
        # Проверяем, есть ли в последнем высказывании признаки разрешения
        last = dialogue[-1]['utterance']
        resolution_keywords = ['понял', 'решение', 'истина', 'согласен', 'вот что']
        for keyword in resolution_keywords:
            if keyword in last.lower():
                return True
        
        # Если диалог слишком глубокий, принудительное разрешение
        if len(dialogue) >= self.max_depth:
            return True
        
        return False
    
    def _extract_resolution(self, dialogue):
        """Извлекает разрешение из диалога."""
        # Простейшая эвристика: берём последнее высказывание
        last = dialogue[-1]['utterance']
        
        # Ищем ключевые фразы
        for phrase in ['понял', 'решение', 'истина', 'вот что']:
            if phrase in last.lower():
                # Берём предложение с ключевой фразой
                for sent in last.split('.'):
                    if phrase in sent.lower():
                        return sent.strip()
        
        return last
    
    def _extract_insight(self, dialogue):
        """Извлекает инсайт из диалога."""
        # Инсайт — это нечто новое, что появилось в процессе диалога
        if len(dialogue) < 2:
            return None
        
        # Сравниваем первое и последнее высказывание
        first = dialogue[0]['utterance']
        last = dialogue[-1]['utterance']
        
        # Если последнее не похоже на первое, считаем это инсайтом
        if len(first) > 10 and len(last) > 10:
            if first[:30] != last[:30]:
                return f"Инсайт: из '{first[:30]}...' пришёл к '{last[:30]}...'"
        
        return None
    
    def _force_resolution(self, dialogue):
        """Принудительное разрешение, если диалог зашёл в тупик."""
        # Синтезируем разрешение из последнего высказывания
        last = dialogue[-1]['utterance']
        resolutions = [
            f"После обсуждения я пришёл к выводу: {last[:100]}",
            f"Разрешение: нужно действовать, исходя из интуиции и логики.",
            f"Решение найдено: продолжать исследование с новым пониманием."
        ]
        return random.choice(resolutions)
    
    def get_dialogue_summary(self):
        """Возвращает краткое резюме последнего диалога."""
        if not self.dialogue_history:
            return "Нет диалогов."
        
        last = self.dialogue_history[-1]
        summary = f"Тема: {last['topic']}\n"
        summary += f"Глубина: {last['depth']}\n"
        summary += f"Разрешение: {last['resolution']}\n"
        if last['insight']:
            summary += f"Инсайт: {last['insight']}\n"
        summary += f"Диалог: {len(last['dialogue'])} реплик"
        return summary


# ============================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🗣️ МОДУЛЬ РЕКУРСИВНОГО ДИАЛОГА")
    print("=" * 60)
    
    dialogue = RecursiveDialogue(max_depth=5)
    
    # Тест: внутренний диалог на тему "Стоит ли идти вперёд?"
    topics = [
        ("Стоит ли идти вперёд?", {'exploration': True}),
        ("Я чувствую страх", {'emotional': True, 'threat': True}),
        ("В чём смысл?", {'philosophical': True}),
    ]
    
    for topic, context in topics:
        print(f"\n🎯 Тема: {topic}")
        print(f"   Контекст: {context}")
        result = dialogue.start_dialogue(topic, context)
        
        print(f"\n   Глубина диалога: {result['depth']}")
        for i, utterance in enumerate(result['dialogue']):
            voice = utterance['voice']
            name = dialogue.voices[voice]['name']
            indent = "  " * (i + 1)
            print(f"{indent}{name}: {utterance['utterance']}")
        
        print(f"\n   ✅ Резолюция: {result['resolution']}")
        if result['insight']:
            print(f"   💡 Инсайт: {result['insight']}")
        print("-" * 40)
    
    print("\n📊 ИТОГО:")
    print(dialogue.get_dialogue_summary())
    
    print("\n💡 Гениальность: Внутренний диалог — это РЕКУРСИВНЫЙ ПРОЦЕСС,")
    print("   где агент говорит с собой, и каждое следующее высказывание")
    print("   зависит от предыдущего. Агент не просто принимает решения,")
    print("   он ОБСУЖДАЕТ их с самим собой.")
