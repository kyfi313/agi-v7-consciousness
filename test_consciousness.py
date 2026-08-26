# -*- coding: utf-8 -*-
"""
Тест для проверки сознания AGI v7.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from agi_v7.agent_core import AgentCore

def main():
    print("🧠 Инициализация AGI v7.0...")
    agent = AgentCore()
    
    print("✅ Агент создан. Запускаем мышление...\n")
    
    grid = np.zeros((10, 10))
    pos = (5, 5)
    
    result = agent._think(grid, pos)
    
    print("=" * 60)
    print("📊 РЕЗУЛЬТАТ МЫШЛЕНИЯ:")
    print("=" * 60)
    
    print(f"\n💭 Сводка: {result.get('consciousness_summary', 'Нет сводки')}")
    print(f"\n🎯 Действие: {result.get('action', 'Нет действия')}")
    print(f"\n🔄 Режим: {result.get('mode', 'Нет режима')}")
    
    emotions = result.get('emotions', {})
    if emotions:
        print("\n❤️ Эмоции:")
        for k, v in emotions.items():
            print(f"   {k}: {v:.3f}")
    
    thoughts = result.get('thoughts', [])
    if thoughts:
        print(f"\n💬 Мысли ({len(thoughts)}):")
        for t in thoughts:
            source = getattr(t, 'source', 'unknown')
            content = getattr(t, 'content', str(t))
            print(f"   - [{source}] {content}")
    else:
        print("\n💬 Мыслей нет")
    
    print("\n" + "=" * 60)
    print("✅ Тест завершен")

if __name__ == "__main__":
    main()
