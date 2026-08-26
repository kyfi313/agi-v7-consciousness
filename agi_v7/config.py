# -*- coding: utf-8 -*-
"""
Конфигурация AGI
"""

CONFIG = {
    'INITIAL_ENERGY': 100.0,
    'ENERGY_DRAIN_DEFAULT': 0.1,
    'ENERGY_DRAIN_FOCUSED': 0.3,
    'ENERGY_DRAIN_CREATIVE': 0.4,
    'ENERGY_REGEN_RATE': 0.05,
    'DOPAMINE_BASE': 0.5,
    'SEROTONIN_BASE': 0.5,
    'NOREPINEPHRINE_BASE': 0.3,
    'ACETYLCHOLINE_BASE': 0.3,
    'CASCADE_STAGES': ['ambition', 'focus', 'frustration', 'analysis', 'reappraisal', 'resolution'],
    'CASCADE_DURATION': 5,
    'HABIT_REPETITIONS': 5,
    'HABIT_STRENGTH_THRESHOLD': 0.7,
    'PRUNE_INTERVAL': 100,
    'PRUNE_THRESHOLD': 0.01,
    'ATTENTION_CAPACITY': 7,
    'LEARNING_RATE': 0.1,
    'GAMMA': 0.95,
    'EXPLORATION_RATE': 0.1,
    'PLANNING_HORIZON': 5,
    'MEMORY_CAPACITY': 10000,
    
    # --- МАСШТАБИРОВАНИЕ ---
    'DEFAULT_NUM_NEURONS': 1000,
    'DEFAULT_CONNECTIVITY': 0.05,
    'MAX_NEURONS': 5000,
    
    # --- ПРЕДИКТИВНОЕ КОДИРОВАНИЕ ---
    'PREDICTION_LEARNING_RATE': 0.05,
    'PREDICTION_HORIZON': 10,
    'PREDICTION_ERROR_THRESHOLD': 0.3,
    
    # --- СОН И КОНСОЛИДАЦИЯ ---
    'SLEEP_INTERVAL': 100,
    'SLEEP_DURATION': 20,
    'CONSOLIDATION_STRENGTH': 0.02,
}
