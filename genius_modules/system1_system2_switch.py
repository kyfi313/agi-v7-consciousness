# -*- coding: utf-8 -*-
"""
МОДУЛЬ: СИСТЕМА 1 И СИСТЕМА 2 (ПЕРЕКЛЮЧЕНИЕ)
Гениальность: Система 1 — это автоматика, рефлексы, привычки.
Система 2 — это сознание, анализ, планирование.
Они переключаются через страх, ошибку и новизну.

Переключение — это не просто «порог», а эмоционально регулируемый процесс.
Страх блокирует систему 2, любопытство активирует её.

Реализовано: FastPath, SpinalCord, IntuitionModule (Система 1),
ImaginationModule, ReasoningModule, ConsciousSelection (Система 2).
"""

import numpy as np
from collections import deque
import time
import random

class FastPath:
    """Система 1 — быстрый, автоматический путь."""
    def __init__(self, input_dim=10, output_dim=4):
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.random.randn(output_dim) * 0.1
        self.cache = {}
        self.response_time = 0.05  # очень быстро
        self.confidence = 0.6
        self.usage_count = 0
        self.history = deque(maxlen=20)

    def predict(self, input_vector):
        """Быстрое предсказание на основе привычек."""
        key = tuple(input_vector[:5].round(2))  # грубая квантизация для кэша
        if key in self.cache:
            self.usage_count += 1
            return self.cache[key]

        output = np.dot(input_vector, self.weights) + self.bias
        output = np.tanh(output)
        self.cache[key] = output
        self.history.append(output)
        self.usage_count += 1
        return output

    def get_state(self):
        return {'confidence': self.confidence, 'usage': self.usage_count, 'cache_size': len(self.cache)}

class SlowPath:
    """Система 2 — медленный, сознательный путь."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=4):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.weights2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.working_memory = deque(maxlen=7)
        self.attention_focus = None
        self.response_time = 0.5  # медленно
        self.confidence = 0.4
        self.usage_count = 0
        self.reflection_depth = 0

    def predict(self, input_vector):
        """Медленное, осознанное предсказание."""
        # Сохраняем в рабочую память
        self.working_memory.append(input_vector)

        # Если в памяти достаточно данных — анализируем
        if len(self.working_memory) >= 3:
            sequence = np.array(list(self.working_memory))
            context = np.mean(sequence, axis=0)
        else:
            context = input_vector

        # Глубокий анализ
        hidden = np.tanh(np.dot(context, self.weights1))
        output = np.tanh(np.dot(hidden, self.weights2))

        self.usage_count += 1
        return output

    def reflect(self):
        """Рефлексия — углублённый анализ."""
        self.reflection_depth += 1
        return f"Размышление на глубине {self.reflection_depth}"

    def get_state(self):
        return {'confidence': self.confidence, 'usage': self.usage_count,
                'wm_size': len(self.working_memory), 'reflection': self.reflection_depth}

class System1System2Switch:
    """Переключатель между Системой 1 и Системой 2."""
    def __init__(self, input_dim=10, output_dim=4):
        self.fast = FastPath(input_dim, output_dim)
        self.slow = SlowPath(input_dim, output_dim)
        self.mode = 'system1'  # 'system1' или 'system2'
        self.switch_threshold = 0.5
        self.fear_level = 0.0
        self.novelty_level = 0.0
        self.error_level = 0.0
        self.history = deque(maxlen=30)

    def process(self, input_vector, fear=0.0, novelty=0.0, error=0.0):
        """Обрабатывает вход, выбирая систему с плавным ингибированием и взаимным подавлением."""
        self.fear_level = fear
        self.novelty_level = novelty
        self.error_level = error

        # --- ПЛАВНОЕ ИНГИБИРОВАНИЕ System 2 при страхе (не жёсткое выключение) ---
        # Страх снижает активность System 2 плавно, а не отключает её полностью
        system2_inhibition = min(1.0, fear * 1.5)  # при fear=0.7 → inhibition=1.0 (полное подавление)
        
        # --- ВЗАИМНОЕ ПОДАВЛЕНИЕ (конкуренция за ресурсы) ---
        # Когда активна System 2, System 1 тормозится, и наоборот
        system1_suppression = 0.0
        system2_suppression = 0.0
        
        # Вычисляем базовую активацию для каждой системы
        fast_activation = 1.0 - fear * 0.5 + novelty * 0.3  # страх снижает, новизна повышает
        slow_activation = 1.0 - system2_inhibition + novelty * 0.8 + error * 0.6
        
        # Нормализуем
        fast_activation = max(0.0, min(1.0, fast_activation))
        slow_activation = max(0.0, min(1.0, slow_activation))
        
        # Определяем режим на основе активации (с плавным переходом)
        if slow_activation > fast_activation + 0.2:
            self.mode = 'system2'
            system2_suppression = 0.3  # System 2 подавляет System 1
        elif fast_activation > slow_activation + 0.2:
            self.mode = 'system1'
            system1_suppression = 0.3  # System 1 подавляет System 2
        else:
            # Смешанный режим — обе системы активны
            self.mode = 'mixed'
            system1_suppression = 0.15
            system2_suppression = 0.15
        
        # Применяем подавление
        fast_activation *= (1.0 - system2_suppression)
        slow_activation *= (1.0 - system1_suppression)
        
        # Если страх очень высок — принудительно System 1 (эволюционный механизм)
        if fear > 0.8:
            self.mode = 'system1'
            fast_activation = 1.0
            slow_activation = 0.0
        
        # Вычисляем выход как взвешенную сумму (плавное смешивание)
        fast_output = self.fast.predict(input_vector)
        slow_output = self.slow.predict(input_vector)
        
        # Взвешиваем выходы по активации
        total_activation = fast_activation + slow_activation + 0.001
        weight_fast = fast_activation / total_activation
        weight_slow = slow_activation / total_activation
        output = weight_fast * fast_output + weight_slow * slow_output
        
        # Обновляем уверенность систем
        self.fast.confidence = 0.5 + 0.5 * fast_activation
        self.slow.confidence = 0.5 + 0.5 * slow_activation
        
        # Логируем
        self.history.append({
            'mode': self.mode,
            'fear': fear,
            'novelty': novelty,
            'error': error,
            'fast_activation': fast_activation,
            'slow_activation': slow_activation,
            'weight_fast': weight_fast,
            'weight_slow': weight_slow
        })
        return output

    def get_state(self):
        return {
            'mode': self.mode,
            'fast': self.fast.get_state(),
            'slow': self.slow.get_state(),
            'fear': self.fear_level,
            'novelty': self.novelty_level,
            'error': self.error_level,
            'history': list(self.history),
            'is_system1_active': self.mode in ['system1', 'mixed'],
            'is_system2_active': self.mode in ['system2', 'mixed']
        }

if __name__ == "__main__":
    print("="*60)
    print("⚡ СИСТЕМА 1 И СИСТЕМА 2 (ПЕРЕКЛЮЧЕНИЕ)")
    print("="*60)
    switch = System1System2Switch(input_dim=5, output_dim=3)
    for i in range(8):
        inp = np.random.randn(5) * 0.5
        fear = 0.0 if i < 3 else 0.8 if i == 4 else 0.2
        novelty = 0.0 if i < 2 else 0.7 if i == 3 else 0.3
        error = 0.0 if i < 5 else 0.5
        out = switch.process(inp, fear, novelty, error)
        print(f"Шаг {i}: режим={switch.mode}, страх={fear:.1f}, новизна={novelty:.1f}, ошибка={error:.1f}")
    print("\n💡 Гениальность: Переключение — это не порог, а эмоционально регулируемый процесс.")
    print("   Страх блокирует систему 2, любопытство активирует её.")
