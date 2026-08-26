#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест 1000 нейронов с рекуррентными связями, WTA и сном.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Добавляем текущую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

def test_1000_neurons():
    """Тестирование сети с 1000 нейронами"""
    
    print("=" * 60)
    print("ЗАПУСК ТЕСТА: 1000 НЕЙРОНОВ С БИОЛОГИЧЕСКИМИ МЕХАНИЗМАМИ")
    print("=" * 60)
    
    # Импортируем модули
    try:
        from config import CONFIG
        print(f"✅ Конфигурация загружена: {CONFIG.get('DEFAULT_NUM_NEURONS', 'не найдено')} нейронов")
    except ImportError as e:
        print(f"❌ Ошибка импорта config: {e}")
        return
    
    try:
        from brain_module import NeuronNetwork
        print("✅ Модуль brain_module импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта brain_module: {e}")
        return
    
    try:
        import compute
        print(f"✅ Вычислительный бэкенд: {compute.detect_backend()}")
    except ImportError as e:
        print(f"⚠️ compute.py не найден, используем NumPy: {e}")
    
    # Параметры сети
    num_neurons = 1000
    connectivity = 0.05
    
    print(f"\n📊 Создание сети с {num_neurons} нейронами, связность {connectivity}...")
    start_time = time.time()
    
    try:
        network = NeuronNetwork(
            num_neurons=num_neurons,
            connectivity=connectivity,
            recurrent_strength=0.1,
            wta_radius=0.2
        )
        print(f"✅ Сеть создана за {time.time() - start_time:.2f}с")
    except Exception as e:
        print(f"❌ Ошибка создания сети: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Проверяем атрибуты
    print("\n🔍 Проверка атрибутов сети:")
    attrs = [
        ('recurrent_weights', 'рекуррентные веса'),
        ('recurrent_buffer', 'буфер задержки'),
        ('wta_clusters', 'кластеры WTA'),
        ('lateral_inhibition_strength', 'сила латерального торможения'),
        ('sleep_counter', 'счётчик шагов до сна'),
        ('consolidation_buffer', 'буфер консолидации')
    ]
    for attr, name in attrs:
        if hasattr(network, attr):
            val = getattr(network, attr)
            if isinstance(val, np.ndarray):
                print(f"  ✅ {name}: {val.shape}")
            else:
                print(f"  ✅ {name}: {val}")
        else:
            print(f"  ❌ {name}: отсутствует")
    
    # Симуляция
    print("\n⚡ Запуск симуляции на 200 шагов...")
    steps = 200
    firing_rates = []
    
    for i in range(steps):
        # Генерируем случайный вход
        input_data = np.random.randn(num_neurons) * 0.5
        
        try:
            result = network.step(input_data)
            # Определяем тип результата
            if isinstance(result, dict):
                # Если результат словарь, извлекаем spikes или firing_rates
                if 'firing_rates' in result:
                    output = np.array(result['firing_rates'])
                elif 'spikes' in result:
                    # Если только спайки, строим массив активности
                    output = np.zeros(num_neurons)
                    for idx in result['spikes']:
                        if idx < num_neurons:
                            output[idx] = 1.0
                else:
                    # Пытаемся найти числовой массив в словаре
                    for key, val in result.items():
                        if isinstance(val, (list, np.ndarray)) and len(val) == num_neurons:
                            output = np.array(val)
                            break
                    else:
                        raise ValueError(f"Не удалось найти массив активности в результате: {result.keys()}")
            else:
                output = np.array(result)
            
            firing_rate = np.mean(output)
            firing_rates.append(firing_rate)
            
            if i % 20 == 0:
                print(f"  Шаг {i:3d}: средняя активность = {firing_rate:.4f}, "
                      f"спящих нейронов = {np.sum(output < 0.01)}")
        except Exception as e:
            print(f"  ❌ Ошибка на шаге {i}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Статистика
    if firing_rates:
        print("\n📈 Статистика симуляции:")
        print(f"  Средняя частота спайков: {np.mean(firing_rates):.4f}")
        print(f"  Макс. частота: {np.max(firing_rates):.4f}")
        print(f"  Мин. частота: {np.min(firing_rates):.4f}")
        print(f"  Стандартное отклонение: {np.std(firing_rates):.4f}")
    
    # Проверка состояния сна
    if hasattr(network, 'sleep_counter'):
        print(f"\n💤 Состояние сна:")
        print(f"  Счётчик до сна: {network.sleep_counter}")
        if hasattr(network, '_process_sleep'):
            print("  ✅ Метод _process_sleep присутствует")
    
    print("\n" + "=" * 60)
    print("✅ ТЕСТ ЗАВЕРШЁН")
    print("=" * 60)

if __name__ == "__main__":
    test_1000_neurons()
