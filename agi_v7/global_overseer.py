# -*- coding: utf-8 -*-
"""
ГЛОБАЛЬНЫЙ ГЛАЗ (Global Overseer)
Наблюдает за эволюцией и усиливает полезные модули.

Принцип работы:
1. Собирает метрики поведения агента (еда, энергия, опасность)
2. Оценивает вклад каждого модуля в успех
3. Усиливает эволюцию модулей, которые приносят пользу
4. Ослабляет бесполезные модули

Это превращает эстафетную эволюцию из параметрической настройки
в адаптивную систему, где эволюция меняет стратегию поведения.
"""

import random
from collections import deque
from typing import Dict, Any, List, Tuple


class GlobalOverseer:
    """
    Глобальный наблюдатель за эволюцией.
    Управляет динамическим усилением модулей.
    """

    def __init__(self, history_length: int = 20):
        """
        Args:
            history_length: Длина истории для анализа трендов
        """
        self.history_length = history_length
        
        # История поведения
        self.food_history = deque(maxlen=history_length)
        self.energy_history = deque(maxlen=history_length)
        self.danger_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        
        # Оценка модулей
        self.module_scores = {
            'Vision': 0.5,
            'Consciousness': 0.5,
            'Memory': 0.5,
            'Predictor': 0.5,
            'Action': 0.5,
        }
        
        # Коэффициенты усиления
        self.boost_factors = {
            'Vision': 1.0,
            'Consciousness': 1.0,
            'Memory': 1.0,
            'Predictor': 1.0,
            'Action': 1.0,
        }
        
        # Счётчики для отслеживания
        self.step_count = 0
        self.last_food_count = 0
        self.last_energy = 100.0
        
        # Данные для анализа вклада
        self.module_contributions = {name: deque(maxlen=10) for name in self.module_scores.keys()}
        
    def observe(self, state: Dict[str, Any], relay_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Наблюдает за состоянием и обновляет оценки модулей.
        
        Args:
            state: Состояние агента (энергия, еда, позиция)
            relay_data: Данные от эстафетной эволюции
            
        Returns:
            Словарь с коэффициентами усиления для каждого модуля
        """
        self.step_count += 1
        
        # Извлекаем метрики
        food_count = state.get('food_collected', 0)
        energy = state.get('energy', 100.0)
        position = state.get('position', (0, 0))
        grid = state.get('grid', [])
        
        # Сохраняем историю
        self.food_history.append(food_count)
        self.energy_history.append(energy)
        
        # Определяем, есть ли опасность рядом
        danger_nearby = self._detect_danger_nearby(position, grid)
        self.danger_history.append(1 if danger_nearby else 0)
        
        # Вычисляем изменения
        food_gain = food_count - self.last_food_count
        energy_change = energy - self.last_energy
        
        # Обновляем состояние
        self.last_food_count = food_count
        self.last_energy = energy
        
        # --- АНАЛИЗ ВКЛАДА МОДУЛЕЙ ---
        # Используем данные от эстафетной эволюции
        module_improvements = relay_data.get('improvements', [])
        
        # Оцениваем каждый модуль
        self._evaluate_modules(food_gain, energy_change, danger_nearby, module_improvements)
        
        # Обновляем коэффициенты усиления
        self._update_boost_factors()
        
        return self.boost_factors.copy()
    
    def _detect_danger_nearby(self, position: Tuple[int, int], grid: List[List[str]]) -> bool:
        """Проверяет, есть ли опасность рядом с агентом."""
        if not grid or not position:
            return False
        
        x, y = position
        height, width = len(grid), len(grid[0]) if grid else 0
        
        # Проверяем клетки вокруг
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny][nx] == '⚠️':
                        return True
        return False
    
    def _evaluate_modules(self, food_gain: int, energy_change: float, danger_nearby: bool, improvements: List[Dict]):
        """Оценивает вклад каждого модуля в успех."""
        
        # --- ВКЛАД МОДУЛЕЙ ---
        # Vision: помогает видеть еду и опасность
        if food_gain > 0:
            self.module_scores['Vision'] = min(1.0, self.module_scores['Vision'] + 0.1)
            self.module_scores['Action'] = min(1.0, self.module_scores['Action'] + 0.08)
        elif food_gain == 0 and self.step_count > 10:
            # Если нет прогресса, снижаем оценку
            self.module_scores['Vision'] = max(0.1, self.module_scores['Vision'] - 0.01)
        
        # Consciousness: помогает оценивать важность сигналов
        if danger_nearby and energy_change < -5:
            # Опасность нанесла урон → сознание плохо сработало
            self.module_scores['Consciousness'] = max(0.1, self.module_scores['Consciousness'] - 0.05)
        elif danger_nearby and energy_change > -2:
            # Опасность удалось избежать → сознание сработало хорошо
            self.module_scores['Consciousness'] = min(1.0, self.module_scores['Consciousness'] + 0.05)
        
        # Memory: помогает запоминать позиции еды
        if food_gain > 0:
            self.module_scores['Memory'] = min(1.0, self.module_scores['Memory'] + 0.05)
        
        # Predictor: помогает предсказывать
        if energy_change > 0:
            self.module_scores['Predictor'] = min(1.0, self.module_scores['Predictor'] + 0.02)
        elif energy_change < -10:
            self.module_scores['Predictor'] = max(0.1, self.module_scores['Predictor'] - 0.03)
        
        # Action: выбирает действия
        if food_gain > 0:
            self.module_scores['Action'] = min(1.0, self.module_scores['Action'] + 0.1)
        if danger_nearby and energy_change < 0:
            self.module_scores['Action'] = max(0.1, self.module_scores['Action'] - 0.05)
        
        # --- УЧИТЫВАЕМ УЛУЧШЕНИЯ ОТ ЭВОЛЮЦИИ ---
        for imp in improvements:
            module_name = imp.get('module', '')
            fitness = imp.get('fitness', 0.0)
            if module_name in self.module_scores:
                # Улучшение fitness повышает оценку модуля
                self.module_scores[module_name] = min(1.0, self.module_scores[module_name] + fitness * 0.05)
        
        # Нормализуем оценки
        for name in self.module_scores:
            self.module_scores[name] = max(0.1, min(1.0, self.module_scores[name]))
    
    def _update_boost_factors(self):
        """Обновляет коэффициенты усиления на основе оценок."""
        for name, score in self.module_scores.items():
            # Базовое усиление 0.5-2.0
            # Чем выше оценка, тем сильнее усиление
            self.boost_factors[name] = 0.5 + score * 1.5
            
            # Сглаживаем изменения
            self.boost_factors[name] = max(0.3, min(2.5, self.boost_factors[name]))
    
    def get_boost(self, module_name: str) -> float:
        """Возвращает коэффициент усиления для модуля."""
        return self.boost_factors.get(module_name, 1.0)
    
    def get_scores(self) -> Dict[str, float]:
        """Возвращает текущие оценки модулей."""
        return self.module_scores.copy()
    
    def get_summary(self) -> str:
        """Возвращает краткую сводку."""
        lines = []
        lines.append(f"📊 ГЛОБАЛЬНЫЙ ГЛАЗ (шаг {self.step_count})")
        for name in ['Vision', 'Consciousness', 'Memory', 'Predictor', 'Action']:
            score = self.module_scores.get(name, 0.5)
            boost = self.boost_factors.get(name, 1.0)
            lines.append(f"  {name}: оценка {score:.2f}, усиление x{boost:.2f}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        scores = [f"{k}: {v:.2f}" for k, v in self.module_scores.items()]
        return f"GlobalOverseer({', '.join(scores)})"


# ============================================================
# ИНТЕГРАЦИЯ С ЭСТАФЕТНОЙ ЭВОЛЮЦИЕЙ
# ============================================================

class AdaptiveRelayEvolution:
    """
    Расширенная версия эстафетной эволюции с Глобальным Глазом.
    """
    
    def __init__(self):
        from agi_v7.relay_evolution import RelayEvolution
        
        self.relay = RelayEvolution()
        self.overseer = GlobalOverseer()
        self.improvements_history = []
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные через эстафетную эволюцию с адаптивным усилением."""
        
        # Запускаем обычную эволюцию
        result = self.relay.process(data)
        
        # Получаем улучшения
        improvements = self.relay.get_improvements()
        if improvements:
            self.improvements_history.extend(improvements)
        
        # Наблюдаем за поведением
        state = {
            'food_collected': data.get('food_collected', 0),
            'energy': data.get('energy', 100.0),
            'position': data.get('position', (0, 0)),
            'grid': data.get('grid', []),
        }
        relay_data = {
            'improvements': improvements,
        }
        
        boost_factors = self.overseer.observe(state, relay_data)
        
        # Применяем усиление к модулям
        self._apply_boost(boost_factors)
        
        # Добавляем информацию об усилении в результат
        result['boost_factors'] = boost_factors
        result['module_scores'] = self.overseer.get_scores()
        
        return result
    
    def _apply_boost(self, boost_factors: Dict[str, float]):
        """Применяет коэффициенты усиления к модулям."""
        for i, (module, name) in enumerate(zip(self.relay.modules, self.relay.module_names)):
            boost = boost_factors.get(name, 1.0)
            
            # Усиливаем скорость мутации
            base_mutation = 0.1
            module.mutation_rate = base_mutation * boost
            
            # Усиливаем частоту эволюции (чем выше буст, тем чаще эволюция)
            # Это реализовано через более частые тесты в evolve()
            # Буст влияет на то, как часто модуль проверяет новые конфигурации
            module._boost = boost  # Сохраняем для использования в evolve
    
    def get_summary(self) -> str:
        """Возвращает сводку состояния."""
        lines = []
        lines.append("=" * 50)
        lines.append("🧠 АДАПТИВНАЯ ЭСТАФЕТНАЯ ЭВОЛЮЦИЯ")
        lines.append("=" * 50)
        lines.append(str(self.relay))
        lines.append("")
        lines.append(self.overseer.get_summary())
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"AdaptiveRelayEvolution({self.relay}, overseer={self.overseer})"


# ============================================================
# ИЗМЕНЕНИЕ EvolvableModule ДЛЯ ПОДДЕРЖКИ БУСТА
# ============================================================

def patch_evolvable_module():
    """
    Патчит EvolvableModule для поддержки динамического буста.
    Вызывать после импорта relay_evolution.
    """
    from agi_v7.relay_evolution import EvolvableModule
    
    original_evolve = EvolvableModule.evolve
    
    def patched_evolve(self, data, output):
        """Расширенная версия evolve с учётом буста."""
        self.evolution_step += 1
        
        # Проверяем текущую fitness
        current_fitness = self.test(self.config)
        self.fitness_history.append(current_fitness)
        
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_config = self.config.copy()
            return True
        
        # Получаем буст
        boost = getattr(self, '_boost', 1.0)
        
        # Частота мутации зависит от буста
        mutation_frequency = max(1, int(5 / boost))  # Чем выше буст, тем чаще мутация
        
        if self.evolution_step % mutation_frequency == 0:
            new_config = self._mutate(self.config)
            new_fitness = self.test(new_config)
            
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_config = new_config.copy()
                self.config = new_config.copy()
                return True
            else:
                self.config = self.best_config.copy()
        
        return False
    
    EvolvableModule.evolve = patched_evolve
    print("✅ EvolvableModule пропатчен для поддержки Глобального Глаза")
