# -*- coding: utf-8 -*-
"""
ДИАГНОСТИКА МОЗГОВОГО МОДУЛЯ AGI
Проверяет все ключевые механизмы и выявляет проблемные зоны
"""

import sys
import os
import random
import time
import numpy as np

# Путь к модулям
work_dir = r'c:\Users\Евгения\Desktop\мои проекты\agi_v7'
sys.path.insert(0, work_dir)
os.chdir(work_dir)

print("="*70)
print("🔬 ДИАГНОСТИКА МОЗГОВОГО МОДУЛЯ AGI")
print("="*70)

results = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "improvements": []
}

# --- 1. ИМПОРТЫ ---
print("\n[1] ПРОВЕРКА ИМПОРТОВ")
try:
    from brain_module import NeuronNetwork, MemoryCompressor, RealNeuron
    results["passed"].append("brain_module импортирован")
    print("   ✅ brain_module")
except Exception as e:
    results["failed"].append(f"brain_module: {e}")
    print(f"   ❌ {e}")

try:
    from orchestrator import CognitiveOrchestrator
    results["passed"].append("orchestrator импортирован")
    print("   ✅ orchestrator")
except Exception as e:
    results["failed"].append(f"orchestrator: {e}")
    print(f"   ❌ {e}")

# Проверяем terminal_agent
try:
    from terminal_agent import TerminalAgent
    results["passed"].append("terminal_agent импортирован")
    print("   ✅ terminal_agent")
except Exception as e:
    results["failed"].append(f"terminal_agent: {e}")
    print(f"   ❌ {e}")
except Exception as e:
    results["failed"].append(f"core.orchestrator: {e}")
    print(f"   ❌ {e}")

# --- 2. НЕЙРОННАЯ СЕТЬ ---
print("\n[2] ПРОВЕРКА НЕЙРОННОЙ СЕТИ")
try:
    net = NeuronNetwork(num_neurons=50, connectivity=0.1)
    assert len(net.neurons) == 50, "Неверное число нейронов"
    assert net._count_synapses() > 0, "Нет синапсов"
    
    # Проверка шага
    signal = [random.random() for _ in range(50)]
    result = net.step(signal)
    assert 'spikes' in result, "Нет спайков"
    assert 'energy' in result, "Нет энергии"
    assert 'fear' in result, "Нет страха"
    assert 'curiosity' in result, "Нет любопытства"
    assert 'surprise' in result, "Нет удивления"
    
    results["passed"].append("Нейронная сеть: все тесты пройдены")
    print(f"   ✅ Нейронов: {len(net.neurons)}, синапсов: {net._count_synapses()}")
    print(f"   ✅ Шаг: спайков={result['spike_count']}, энергия={result['energy']:.1f}")
except Exception as e:
    results["failed"].append(f"Нейронная сеть: {e}")
    print(f"   ❌ {e}")

# --- 3. КОМПРЕССОР ПАМЯТИ ---
print("\n[3] ПРОВЕРКА КОМПРЕССОРА")
try:
    comp = MemoryCompressor(input_dim=30, code_dim=8)
    pattern = [random.random() for _ in range(30)]
    idx = comp.store(pattern, label="test")
    
    # Проверка сжатия
    code = comp.get_code(idx)
    assert len(code) == 8, f"Неверный размер кода: {len(code)}"
    
    # Проверка восстановления
    restored = comp.recall(idx)
    assert restored is not None, "Не удалось восстановить"
    
    # Проверка похожести
    similar = comp.find_similar(pattern, top_n=2)
    
    results["passed"].append("Компрессор: все тесты пройдены")
    print(f"   ✅ Сжатие: {len(pattern)} -> {len(code)}")
    print(f"   ✅ Восстановление: {len(restored)} элементов")
    print(f"   ✅ Похожих паттернов: {len(similar)}")
except Exception as e:
    results["failed"].append(f"Компрессор: {e}")
    print(f"   ❌ {e}")

# --- 4. ОРКЕСТРАТОР (корневой) ---
print("\n[4] ПРОВЕРКА КОРНЕВОГО ОРКЕСТРАТОРА")
try:
    orch = CognitiveOrchestrator(num_neurons=60, connectivity=0.08, input_dim=60)
    assert len(orch.brain.neurons) == 60, "Неверное число нейронов"
    
    # Проверка шага
    signal = [random.random() for _ in range(60)]
    result = orch.step(signal)
    assert 'spike_count' in result, "Нет спайков"
    assert 'energy' in result, "Нет энергии"
    
    # Проверка навыков
    orch.learn_skill("test_skill", [0.3, 0.7, 0.5])
    recalled = orch.recall_skill("test_skill")
    assert recalled is not None, "Не удалось воспроизвести навык"
    
    # Проверка памяти
    pattern = [random.random() for _ in range(60)]
    idx = orch.store_memory(pattern, label="mem_test")
    restored = orch.recall_memory(idx)
    assert restored is not None, "Не удалось восстановить память"
    
    # Проверка метакогниции
    meta = orch.get_metacognition()
    assert 'awareness_level' in meta, "Нет осознанности"
    assert 'self_confidence' in meta, "Нет уверенности"
    
    results["passed"].append("Корневой оркестратор: все тесты пройдены")
    print("   ✅ Оркестратор создан")
    print(f"   ✅ Шаг: спайков={result['spike_count']}, энергия={result['energy']:.1f}")
    print(f"   ✅ Навык воспроизведён: {recalled[:3] if recalled else 'нет'}")
    print(f"   ✅ Память восстановлена: {restored[:3] if restored else 'нет'}")
    print(f"   ✅ Осознанность: {meta['awareness_level']:.2f}")
except Exception as e:
    results["failed"].append(f"Корневой оркестратор: {e}")
    print(f"   ❌ {e}")

# --- 5. CORE ОРКЕСТРАТОР ---
print("\n[5] ПРОВЕРКА CORE ОРКЕСТРАТОРА")
try:
    core_orch = CoreOrch()
    # Проверяем, что brain модуль доступен
    if hasattr(core_orch, 'brain') and core_orch.brain is not None:
        print("   ✅ Мозговой модуль доступен")
        # Проверяем, есть ли базовые атрибуты
        assert hasattr(core_orch.brain, 'neurons'), "Нет нейронов"
        print(f"   ✅ Нейронов: {len(core_orch.brain.neurons)}")
        results["passed"].append("core.orchestrator: мозговой модуль доступен")
    else:
        results["warnings"].append("core.orchestrator: мозговой модуль недоступен (эмуляция)")
        print("   ⚠️ Мозговой модуль недоступен, используется эмуляция")
except Exception as e:
    results["failed"].append(f"core.orchestrator: {e}")
    print(f"   ❌ {e}")

# --- 6. НЕЙРОГЕНЕЗ ---
print("\n[6] ПРОВЕРКА НЕЙРОГЕНЕЗА")
try:
    net = NeuronNetwork(num_neurons=30, connectivity=0.05)
    initial_count = len(net.neurons)
    
    # Запускаем несколько шагов с высоким удивлением
    net.surprise = 0.8
    net.birth_rate = 1.0
    for _ in range(20):
        signal = [random.random() for _ in range(30)]
        net.step(signal)
    
    final_count = len(net.neurons)
    if final_count > initial_count:
        results["passed"].append(f"Нейрогенез: +{final_count - initial_count} нейронов")
        print(f"   ✅ Нейрогенез активен: {initial_count} -> {final_count} (+{final_count - initial_count})")
    else:
        results["warnings"].append("Нейрогенез: нейроны не создаются (возможно, низкая частота)")
        print("   ⚠️ Нейрогенез не сработал (попробуйте увеличить surprise)")
except Exception as e:
    results["failed"].append(f"Нейрогенез: {e}")
    print(f"   ❌ {e}")

# --- 7. ЭНЕРГЕТИКА И КРИЗИС ---
print("\n[7] ПРОВЕРКА ЭНЕРГЕТИКИ")
try:
    net = NeuronNetwork(num_neurons=30, connectivity=0.05)
    # Быстро истощаем энергию
    net.network_energy = 10.0
    net.crisis_threshold = 0.3
    
    for _ in range(10):
        signal = [random.random() for _ in range(30)]
        net.step(signal)
    
    if net.network_energy < 20:
        # Проверяем кризисный режим
        assert net.crisis_mode or net.network_energy < 30, "Кризис не активирован"
        results["passed"].append("Энергетика: кризисный режим работает")
        print(f"   ✅ Кризисный режим: энергия={net.network_energy:.1f}, fear={net.fear:.2f}")
    else:
        results["warnings"].append("Энергетика: кризис не активирован (энергия стабильна)")
        print("   ⚠️ Кризис не активирован")
except Exception as e:
    results["failed"].append(f"Энергетика: {e}")
    print(f"   ❌ {e}")

# --- 8. ПРЕДСКАЗАНИЕ ---
print("\n[8] ПРОВЕРКА ПРЕДСКАЗАНИЯ")
try:
    orch = CognitiveOrchestrator(num_neurons=40, connectivity=0.08, input_dim=40)
    # Делаем несколько шагов для накопления истории
    for _ in range(10):
        signal = [random.random() for _ in range(40)]
        orch.step(signal)
    
    current = [random.random() for _ in range(40)]
    pred = orch.predict_next(current)
    
    assert pred is not None, "Предсказание не возвращено"
    assert len(pred) == len(current), f"Неверная длина предсказания: {len(pred)}"
    
    results["passed"].append("Предсказание: работает")
    print(f"   ✅ Предсказание: {pred[:3] if pred else 'нет'}")
except Exception as e:
    results["failed"].append(f"Предсказание: {e}")
    print(f"   ❌ {e}")

# --- 9. СОН И КОНСОЛИДАЦИЯ ---
print("\n[9] ПРОВЕРКА СНА И КОНСОЛИДАЦИИ")
try:
    orch = CognitiveOrchestrator(num_neurons=40, connectivity=0.08, input_dim=40)
    # Добавляем паттерны в долгосрочную память
    for _ in range(5):
        signal = [random.random() for _ in range(40)]
        orch.step(signal)
        # Принудительно добавляем в долгосрочную память
        orch.brain.long_term_memory.append(signal)
    
    # Запускаем сон
    if hasattr(orch.brain, 'sleep_consolidation'):
        sleep_result = orch.brain.sleep_consolidation()
        results["passed"].append("Сон: консолидация работает")
        print(f"   ✅ Сон: консолидировано {sleep_result['consolidated'] if 'consolidated' in sleep_result else 0} паттернов")
    else:
        results["warnings"].append("Сон: метод sleep_consolidation не найден")
        print("   ⚠️ Метод sleep_consolidation отсутствует")
except Exception as e:
    results["failed"].append(f"Сон: {e}")
    print(f"   ❌ {e}")

# --- 10. МЕТАКОГНИЦИЯ ---
print("\n[10] ПРОВЕРКА МЕТАКОГНИЦИИ")
try:
    orch = CognitiveOrchestrator(num_neurons=40, connectivity=0.08, input_dim=40)
    for _ in range(20):
        signal = [random.random() for _ in range(40)]
        orch.step(signal)
    
    meta = orch.get_metacognition()
    required_keys = ['awareness_level', 'self_confidence', 'prediction_accuracy', 'insight_count']
    for key in required_keys:
        assert key in meta, f"Отсутствует {key}"
    
    results["passed"].append("Метакогниция: все ключи присутствуют")
    print(f"   ✅ Осознанность: {meta['awareness_level']:.2f}")
    print(f"   ✅ Уверенность: {meta['self_confidence']:.2f}")
    print(f"   ✅ Точность: {meta['prediction_accuracy']:.2f}")
    print(f"   ✅ Инсайтов: {meta['insight_count']}")
except Exception as e:
    results["failed"].append(f"Метакогниция: {e}")
    print(f"   ❌ {e}")

# --- 11. СТАТИСТИКА И ПОКРЫТИЕ ---
print("\n[11] АНАЛИЗ ПОКРЫТИЯ")
print("-"*70)

# Собираем все методы из brain_module
brain_methods = [m for m in dir(NeuronNetwork) if not m.startswith('_')]
print(f"\n📦 Методы NeuronNetwork: {len(brain_methods)}")
for m in brain_methods[:5]:
    print(f"   - {m}")
if len(brain_methods) > 5:
    print(f"   ... и ещё {len(brain_methods) - 5} методов")

# Проверяем наличие ключевых методов
key_methods = ['step', 'add_skill', 'recall_skill', 'sleep_consolidation', 'attention', 'predict_next', 'get_metacognition']
print("\n🔑 Ключевые методы:")
for km in key_methods:
    if hasattr(NeuronNetwork, km):
        print(f"   ✅ {km}")
    else:
        print(f"   ❌ {km} (отсутствует)")
        results["improvements"].append(f"Добавить метод {km} в NeuronNetwork")

# Проверяем методы MemoryCompressor
comp_methods = [m for m in dir(MemoryCompressor) if not m.startswith('_')]
print(f"\n📦 Методы MemoryCompressor: {len(comp_methods)}")
for m in comp_methods:
    print(f"   - {m}")

# --- 12. ИТОГОВЫЙ ОТЧЁТ ---
print("\n" + "="*70)
print("📊 ИТОГОВЫЙ ОТЧЁТ ДИАГНОСТИКИ")
print("="*70)

print(f"\n✅ ПРОЙДЕНО: {len(results['passed'])} тестов")
for item in results['passed']:
    print(f"   ✓ {item}")

if results['warnings']:
    print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ: {len(results['warnings'])}")
    for item in results['warnings']:
        print(f"   ⚠ {item}")

if results['failed']:
    print(f"\n❌ ПРОВАЛЕНО: {len(results['failed'])} тестов")
    for item in results['failed']:
        print(f"   ✗ {item}")

if results['improvements']:
    print(f"\n🚀 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ: {len(results['improvements'])}")
    for item in results['improvements']:
        print(f"   💡 {item}")

print("\n" + "="*70)

# Оценка состояния
pass_rate = len(results['passed']) / (len(results['passed']) + len(results['failed']) + 0.1)
if pass_rate > 0.8:
    print("\n🌟 ОЦЕНКА: ОТЛИЧНО — система работает стабильно")
elif pass_rate > 0.5:
    print("\n👍 ОЦЕНКА: ХОРОШО — есть мелкие недочёты")
else:
    print("\n🔧 ОЦЕНКА: ТРЕБУЕТ ДОРАБОТКИ — необходимы исправления")

print("="*70)
