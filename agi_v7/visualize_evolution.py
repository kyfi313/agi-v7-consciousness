# -*- coding: utf-8 -*-
"""
ВИЗУАЛИЗАЦИЯ ЭВОЛЮЦИИ
Графики и диаграммы для анализа адаптивной эстафетной эволюции.
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("⚠️ matplotlib не установлен. Установите: pip install matplotlib")
    import sys
    sys.exit(1)
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import os


def plot_evolution_metrics(
    food_history: List[int],
    energy_history: List[float],
    reward_history: List[float],
    boost_history: List[Tuple[int, Dict[str, float]]],
    fitness_history: List[Dict[str, float]],
    title: str = "Эволюция агента",
    save_path: str = None
):
    """
    Строит комплексный график эволюции.
    
    Args:
        food_history: История сбора еды
        energy_history: История энергии
        reward_history: История наград
        boost_history: История буст-факторов
        fitness_history: История fitness модулей
        title: Заголовок графика
        save_path: Путь для сохранения (опционально)
    """
    steps = list(range(1, len(food_history) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Еда и энергия
    ax1 = axes[0, 0]
    ax1.plot(steps, food_history, 'g-', linewidth=2, label='Еда собрана')
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Количество еды', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, alpha=0.3)
    
    ax1_2 = ax1.twinx()
    ax1_2.plot(steps, energy_history, 'b-', linewidth=2, label='Энергия')
    ax1_2.set_ylabel('Энергия', color='b')
    ax1_2.tick_params(axis='y', labelcolor='b')
    ax1_2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Критический уровень')
    
    ax1.set_title('🍎 Еда и ⚡ Энергия')
    ax1.legend(loc='upper left')
    ax1_2.legend(loc='upper right')
    
    # 2. Награда
    ax2 = axes[0, 1]
    ax2.plot(steps, reward_history, 'purple', linewidth=1.5, alpha=0.7, label='Награда')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Скользящее среднее
    if len(reward_history) > 5:
        window = min(10, len(reward_history))
        smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        smooth_steps = steps[:len(smoothed)]
        ax2.plot(smooth_steps, smoothed, 'r-', linewidth=2, label='Средняя (10 шагов)')
    
    ax2.set_xlabel('Шаг')
    ax2.set_ylabel('Награда')
    ax2.set_title('🎯 Награда')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Буст-факторы
    ax3 = axes[1, 0]
    if boost_history:
        boost_steps = [b[0] for b in boost_history]
        
        # Собираем данные по модулям
        modules = ['Vision', 'Consciousness', 'Memory', 'Predictor', 'Action']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for module, color in zip(modules, colors):
            values = []
            for step, boosts in boost_history:
                values.append(boosts.get(module, 1.0))
            ax3.plot(boost_steps, values, '-', linewidth=2, color=color, label=module)
        
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Базовый уровень')
        ax3.set_xlabel('Шаг')
        ax3.set_ylabel('Буст-фактор')
        ax3.set_title('📈 Буст-факторы модулей')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3.set_ylim(0.5, 2.5)
    
    # 4. Fitness модулей (последнее состояние)
    ax4 = axes[1, 1]
    if fitness_history:
        # Берём последние значения
        last_fitness = fitness_history[-1] if isinstance(fitness_history[-1], dict) else {}
        
        modules = list(last_fitness.keys())
        values = list(last_fitness.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax4.bar(modules, values, color=colors[:len(modules)], edgecolor='black', linewidth=1)
        ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Хорошо')
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Средне')
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Fitness')
        ax4.set_title('🧠 Fitness модулей (финал)')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()
        
        # Добавляем значения на бары
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ График сохранён: {save_path}")
    
    plt.show()


def plot_evolution_heatmap(
    boost_history: List[Tuple[int, Dict[str, float]]],
    title: str = "Тепловая карта усиления модулей",
    save_path: str = None
):
    """
    Строит тепловую карту буст-факторов.
    
    Args:
        boost_history: История буст-факторов
        title: Заголовок
        save_path: Путь для сохранения
    """
    if not boost_history:
        print("⚠️ Нет данных для тепловой карты")
        return
    
    modules = ['Vision', 'Consciousness', 'Memory', 'Predictor', 'Action']
    steps = [b[0] for b in boost_history]
    
    # Строим матрицу
    data = []
    for step, boosts in boost_history:
        row = [boosts.get(m, 1.0) for m in modules]
        data.append(row)
    
    data = np.array(data).T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
    
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f'{s}' for s in steps], rotation=90, fontsize=8)
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules)
    
    ax.set_xlabel('Шаг')
    ax.set_title(title)
    
    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Буст-фактор')
    
    # Добавляем значения в ячейки
    for i in range(len(modules)):
        for j in range(len(steps)):
            value = data[i, j]
            color = 'white' if value < 1.2 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', color=color, fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Тепловая карта сохранена: {save_path}")
    
    plt.show()


def plot_action_distribution(
    action_history: List[str],
    title: str = "Распределение действий",
    save_path: str = None
):
    """
    Строит диаграмму распределения действий.
    
    Args:
        action_history: История действий
        title: Заголовок
        save_path: Путь для сохранения
    """
    if not action_history:
        print("⚠️ Нет данных о действиях")
        return
    
    # Подсчёт действий
    action_counts = defaultdict(int)
    for action in action_history:
        action_counts[action] += 1
    
    # Сортировка
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    actions = [a[0] for a in sorted_actions]
    counts = [a[1] for a in sorted_actions]
    
    # Цвета для действий
    colors = {
        'explore': '#4ECDC4',
        'collect': '#2ECC71',
        'flee': '#FF6B6B',
        'rest': '#F39C12',
        'up': '#3498DB',
        'down': '#3498DB',
        'left': '#3498DB',
        'right': '#3498DB',
    }
    bar_colors = [colors.get(a, '#95A5A6') for a in actions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(actions, counts, color=bar_colors, edgecolor='black', linewidth=1)
    
    # Добавляем значения
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Действие')
    ax.set_ylabel('Количество')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Легенда
    legend_elements = [
        mpatches.Patch(color='#4ECDC4', label='Исследование'),
        mpatches.Patch(color='#2ECC71', label='Сбор'),
        mpatches.Patch(color='#FF6B6B', label='Бегство'),
        mpatches.Patch(color='#F39C12', label='Отдых'),
        mpatches.Patch(color='#3498DB', label='Движение'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Диаграмма действий сохранена: {save_path}")
    
    plt.show()


def generate_evolution_report(
    food_history: List[int],
    energy_history: List[float],
    reward_history: List[float],
    boost_history: List[Tuple[int, Dict[str, float]]],
    fitness_history: List[Dict[str, float]],
    action_history: List[str],
    title: str = "Отчёт об эволюции",
    save_dir: str = "./evolution_reports"
):
    """
    Генерирует полный отчёт с визуализациями.
    
    Args:
        food_history: История сбора еды
        energy_history: История энергии
        reward_history: История наград
        boost_history: История буст-факторов
        fitness_history: История fitness модулей
        action_history: История действий
        title: Заголовок
        save_dir: Директория для сохранения
    """
    # Создаём директорию
    os.makedirs(save_dir, exist_ok=True)
    
    # Генерируем имя файла
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"evolution_report_{timestamp}"
    
    # 1. Основной график
    plot_evolution_metrics(
        food_history, energy_history, reward_history, boost_history, fitness_history,
        title=title,
        save_path=f"{save_dir}/{base_name}_metrics.png"
    )
    
    # 2. Тепловая карта
    if boost_history:
        plot_evolution_heatmap(
            boost_history,
            title="Тепловая карта усиления модулей",
            save_path=f"{save_dir}/{base_name}_heatmap.png"
        )
    
    # 3. Распределение действий
    if action_history:
        plot_action_distribution(
            action_history,
            title="Распределение действий",
            save_path=f"{save_dir}/{base_name}_actions.png"
        )
    
    # 4. Генерируем текстовый отчёт
    report_path = f"{save_dir}/{base_name}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"📊 ОТЧЁТ ОБ ЭВОЛЮЦИИ\n")
        f.write(f"   {title}\n")
        f.write("=" * 70 + "\n\n")
        
        # Основная статистика
        f.write("📈 ОСНОВНАЯ СТАТИСТИКА\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Всего шагов: {len(food_history)}\n")
        f.write(f"  Еды собрано: {food_history[-1] if food_history else 0}\n")
        f.write(f"  Финальная энергия: {energy_history[-1] if energy_history else 0:.1f}\n")
        f.write(f"  Средняя награда: {sum(reward_history) / max(1, len(reward_history)):.3f}\n")
        f.write("\n")
        
        # Fitness модулей
        if fitness_history:
            f.write("🧠 FITNESS МОДУЛЕЙ (финал)\n")
            f.write("-" * 40 + "\n")
            last_fitness = fitness_history[-1] if isinstance(fitness_history[-1], dict) else {}
            for module, value in last_fitness.items():
                f.write(f"  {module}: {value:.3f}\n")
            f.write("\n")
        
        # Буст-факторы
        if boost_history:
            f.write("📈 БУСТ-ФАКТОРЫ (финал)\n")
            f.write("-" * 40 + "\n")
            last_boosts = boost_history[-1][1] if boost_history else {}
            for module, value in last_boosts.items():
                f.write(f"  {module}: x{value:.2f}\n")
            f.write("\n")
        
        # Распределение действий
        if action_history:
            f.write("🎯 РАСПРЕДЕЛЕНИЕ ДЕЙСТВИЙ\n")
            f.write("-" * 40 + "\n")
            action_counts = defaultdict(int)
            for action in action_history:
                action_counts[action] += 1
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / len(action_history) * 100
                f.write(f"  {action}: {count} ({pct:.1f}%)\n")
            f.write("\n")
        
        # Выводы
        f.write("🔍 ВЫВОДЫ\n")
        f.write("-" * 40 + "\n")
        
        if food_history and food_history[-1] > 0:
            f.write("  ✅ Агент научился собирать еду!\n")
        else:
            f.write("  ⚠️ Агент не собрал ни одной еды.\n")
        
        if energy_history and energy_history[-1] > 20:
            f.write("  ✅ Энергия поддерживается на стабильном уровне.\n")
        else:
            f.write("  ⚠️ Энергия критически низкая.\n")
        
        if boost_history:
            last_boosts = boost_history[-1][1]
            max_module = max(last_boosts, key=last_boosts.get)
            f.write(f"  🔥 Самый усиленный модуль: {max_module} (x{last_boosts[max_module]:.2f})\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"✅ Отчёт сохранён: {report_path}")
    print(f"   📁 {save_dir}/{base_name}_*")


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

def demo_visualization():
    """Демонстрация визуализации на случайных данных."""
    import random
    
    # Генерируем случайные данные
    steps = 50
    food_history = [random.randint(0, i // 10 + 1) for i in range(steps)]
    energy_history = [100 - i * 0.5 + random.uniform(-5, 5) for i in range(steps)]
    reward_history = [random.uniform(-0.5, 1.0) for _ in range(steps)]
    
    boost_history = []
    modules = ['Vision', 'Consciousness', 'Memory', 'Predictor', 'Action']
    for i in range(steps):
        boosts = {}
        for m in modules:
            base = 1.0 + 0.5 * (i / steps)
            boosts[m] = base + random.uniform(-0.2, 0.2)
        boost_history.append((i, boosts))
    
    fitness_history = []
    for i in range(steps):
        fitness = {}
        for m in modules:
            fitness[m] = 0.5 + 0.5 * (i / steps) + random.uniform(-0.05, 0.05)
        fitness_history.append(fitness)
    
    action_history = ['explore'] * 20 + ['collect'] * 10 + ['flee'] * 8 + ['rest'] * 7 + ['up'] * 5
    random.shuffle(action_history)
    
    # Генерируем отчёт
    generate_evolution_report(
        food_history, energy_history, reward_history, boost_history, fitness_history, action_history,
        title="Демонстрация эволюции",
        save_dir="./demo_reports"
    )
    
    print("\n🎉 Демонстрация завершена!")


if __name__ == "__main__":
    demo_visualization()
