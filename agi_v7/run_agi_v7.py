# -*- coding: utf-8 -*-
"""
🚀 ЗАПУСК AGI v7
Главная точка входа для работы с биологически-реалистичным мозгом
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import CognitiveOrchestrator


def main():
    print("=" * 70)
    print("🧠 ЗАПУСК AGI v7 — БИОЛОГИЧЕСКИ-РЕАЛИСТИЧНЫЙ МОЗГ")
    print("=" * 70)
    
    # 1. Инициализация
    print("\n[1] Инициализация мозга...")
    agi = CognitiveOrchestrator(
        num_neurons=200,
        connectivity=0.08,
        input_dim=100
    )
    print(f"   ✅ Нейронов: {len(agi.brain.neurons) if agi.brain else 0}")
    print(f"   ✅ Синапсов: {agi.brain._count_synapses() if agi.brain else 0}")
    print(f"   ✅ Навыков: {len(agi.skills) if hasattr(agi, 'skills') else 0}")
    
    # 2. Обучаем базовые навыки
    print("\n[2] Обучение базовым навыкам...")
    skills = {
        'ходьба': [0.3, 0.6, 0.8, 0.5, 0.2, 0.4, 0.7, 0.9, 0.6, 0.3],
        'исследование': [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        'отдых': [0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]
    }
    for name, seq in skills.items():
        agi.learn_skill(name, seq)
        print(f"   ✅ {name}: {len(seq)} шагов")
    
    # 3. Основной цикл жизни
    print("\n[3] Запуск основного цикла...")
    print("-" * 70)
    
    total_steps = 1000
    log_interval = 50  # Логируем чаще, чтобы видеть мысли
    thought_interval = 10  # Мысли каждые 10 шагов
    start_time = time.time()
    
    for step in range(1, total_steps + 1):
        # Генерируем случайный сигнал (имитация восприятия)
        signal = [0.5 + 0.5 * (step % 3 / 3)] * 100
        
        # Шаг мозга
        result = agi.step(signal)
        
        # Автоматический сон каждые 50 шагов
        if step % 50 == 0:
            agi.sleep()
        
        # --- НОВОЕ: ВНУТРЕННИЙ ГОЛОС И СТАТУС МОДУЛЕЙ ---
        # Выводим мысли каждые 10 шагов
        if step % thought_interval == 0 and hasattr(agi, 'inner_speech'):
            speech_status = agi.inner_speech.get_status()
            current_thought = speech_status.get('current_thought')
            if current_thought:
                print(f"💭 [Шаг {step:4d}] {current_thought}")
        
        # Статус гомеостаза каждые 25 шагов
        if step % 25 == 0 and hasattr(agi, 'homeostasis'):
            homeo_status = agi.homeostasis.get_status()
            state = homeo_status.get('state', {})
            needs = homeo_status.get('needs', {})
            print(f"🌡️  [Шаг {step:4d}] Темп: {state.get('temperature', 0):.1f}°C | "
                  f"Глюкоза: {state.get('glucose', 0):.1f} | "
                  f"Сонливость: {state.get('sleep_pressure', 0):.2f} | "
                  f"Голод: {needs.get('hunger', 0):.2f}")
        
        # Статус креативности каждые 50 шагов
        if step % 50 == 0 and hasattr(agi, 'creativity'):
            creat_status = agi.creativity.get_status()
            print(f"✨ [Шаг {step:4d}] Креативность: {creat_status.get('creativity_level', 0):.2f} | "
                  f"Инсайтов: {creat_status.get('insights_count', 0)}")
            if creat_status.get('latest_insight'):
                print(f"💡 [Шаг {step:4d}] Инсайт: {creat_status['latest_insight']}")
        
        # Статус нейротрофинов каждые 100 шагов
        if step % 100 == 0 and hasattr(agi, 'neurotrophins'):
            trophin_status = agi.neurotrophins.get_status()
            print(f"🧬 [Шаг {step:4d}] BDNF: {trophin_status.get('bdnf', 0):.2f} | "
                  f"Нейрогенез: {trophin_status.get('neurogenesis_rate', 0):.3f} | "
                  f"Здоровье нейронов: {trophin_status.get('avg_health', 0):.2f}")
        
        # Статус астроцитов каждые 100 шагов
        if step % 100 == 0 and hasattr(agi, 'astrocytes'):
            astro_status = agi.astrocytes.get_status()
            print(f"🧬 [Шаг {step:4d}] Астроциты: {astro_status.get('num_astrocytes', 0)} клеток | "
                  f"Глобальная энергия: {astro_status.get('global_energy', 0):.2f} | "
                  f"Кальций: {astro_status.get('global_calcium', 0):.2f}")
        
        # Основное логирование (как было)
        if step % log_interval == 0:
            meta = agi.get_metacognition()
            print(f"📊 [Шаг {step:4d}] "
                  f"Нейронов: {len(agi.brain.neurons) if agi.brain else 0:3d} | "
                  f"Энергия: {result.get('energy', 0):5.1f} | "
                  f"Страх: {result.get('fear', 0):.2f} | "
                  f"Осознанность: {meta.get('awareness_level', 0):.2f}")
    
    end_time = time.time()
    
    # 4. Итоговая статистика
    print("-" * 70)
    print("\n[4] Итоговая статистика")
    print("=" * 70)
    
    meta = agi.get_metacognition()
    print(f"⏱️  Время работы: {end_time - start_time:.2f} с")
    print(f"🧠 Нейронов: {len(agi.brain.neurons) if agi.brain else 0}")
    print(f"🔗 Синапсов: {agi.brain._count_synapses() if agi.brain else 0}")
    print(f"⚡ Энергия: {agi.brain.network_energy if agi.brain else 0:.1f}")
    print(f"🧠 Осознанность: {meta.get('awareness_level', 0):.2f}")
    print(f"💪 Уверенность: {meta.get('self_confidence', 0):.2f}")
    print(f"🎯 Точность: {meta.get('prediction_accuracy', 0):.2f}")
    print(f"💡 Инсайтов: {agi.brain.insight_count if agi.brain else 0}")
    print(f"📚 Навыков: {len(agi.skills) if hasattr(agi, 'skills') else 0}")
    
    # 5. Сохранение состояния
    print("\n[5] Сохранение состояния...")
    try:
        agi.save("agi_weights.pkl")
        print("   ✅ Сохранено в agi_weights.pkl")
    except Exception as e:
        print(f"   ⚠️ Ошибка сохранения: {e}")
    
    print("\n" + "=" * 70)
    print("✅ AGI v7 ЗАВЕРШИЛ РАБОТУ. МОЗГ ЖИВ.")
    print("=" * 70)


if __name__ == "__main__":
    main()
