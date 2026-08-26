# -*- coding: utf-8 -*-
"""
Тесты для новых модулей AGI v7 (нейробиологические улучшения)
Запуск: python tests/test_new_modules.py
"""

import sys
import os

# Добавляем путь к корню проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Проверяем, что путь корректный
print(f"Project root: {project_root}")

import unittest
import numpy as np

# Импортируем модули
from core.state import GlobalState
from modules.dlPFC import DL_PFC
from modules.vmPFC import VM_PFC
from modules.hippocampal_replay import HippocampalReplay
from modules.insular_cortex import InsularCortex


class TestDLPFC(unittest.TestCase):
    """Тесты для дорсолатеральной префронтальной коры"""
    
    def test_creation(self):
        dlpfc = DL_PFC()
        self.assertIsNotNone(dlpfc)
        self.assertEqual(dlpfc.max_wm_size, 7)
        self.assertEqual(dlpfc.attention_focus, 1.0)
    
    def test_update(self):
        dlpfc = DL_PFC()
        state = GlobalState()
        state.concepts = ['food', 'danger', 'explore']
        state.planning['current_goal'] = 'survive'
        state.emotions['threat'] = 0.3
        state.body['energy'] = 0.6
        
        new_state = dlpfc.update(state)
        self.assertIsNotNone(new_state)
        self.assertTrue(hasattr(new_state, 'working_memory'))
        self.assertGreater(len(new_state.working_memory), 0)
        self.assertGreaterEqual(new_state.attention_focus, 0.0)
        self.assertLessEqual(new_state.attention_focus, 1.0)
    
    def test_inhibition(self):
        dlpfc = DL_PFC()
        state = GlobalState()
        state.impulsive_action = {'action': 'attack', 'strength': 0.8}
        state.time = 10
        
        new_state = dlpfc.update(state)
        self.assertTrue(new_state.impulsive_action['inhibited'])
        self.assertLess(new_state.impulsive_action['strength'], 0.8)


class TestVMPFC(unittest.TestCase):
    """Тесты для вентромедиальной префронтальной коры"""
    
    def test_creation(self):
        vmpfc = VM_PFC()
        self.assertIsNotNone(vmpfc)
        self.assertEqual(vmpfc.emotion_regulation, 0.5)
        self.assertEqual(vmpfc.risk_tolerance, 0.5)
    
    def test_update(self):
        vmpfc = VM_PFC()
        state = GlobalState()
        state.emotions['cortisol'] = 0.7
        state.emotions['threat'] = 0.6
        state.neuromodulators['dopamine'] = 0.8
        state.actions = ['explore', 'flee', 'eat']
        state.time = 5
        
        new_state = vmpfc.update(state)
        self.assertIsNotNone(new_state)
        self.assertTrue(hasattr(new_state, 'emotion_regulation'))
        self.assertGreater(new_state.emotion_regulation, 0.5)
        self.assertIn('final_action', new_state.__dict__)
    
    def test_value_estimation(self):
        vmpfc = VM_PFC()
        state = GlobalState()
        state.neuromodulators['dopamine'] = 0.9
        state.emotions['cortisol'] = 0.2
        state.actions = ['explore', 'rest']
        
        vmpfc._evaluate_value(state)
        self.assertIn('explore', vmpfc.value_estimates)
        self.assertIn('rest', vmpfc.value_estimates)


class TestHippocampalReplay(unittest.TestCase):
    """Тесты для гиппокампального реплея"""
    
    def test_creation(self):
        replay = HippocampalReplay()
        self.assertIsNotNone(replay)
        self.assertEqual(replay.max_buffer_size, 50)
        self.assertEqual(replay.sleep_interval, 20)
    
    def test_store_episode(self):
        replay = HippocampalReplay()
        state = GlobalState()
        state.final_action = 'explore'
        state.perception['objects'] = ['tree', 'river']
        state.concepts = ['forest', 'water']
        state.emotions = {'joy': 0.8}
        state.learning['reward'] = 0.6
        state.planning['current_goal'] = 'find_food'
        state.time = 1
        
        replay._store_episode(state)
        self.assertEqual(len(replay.episode_buffer), 1)
        self.assertEqual(replay.episode_buffer[0]['action'], 'explore')
    
    def test_update(self):
        replay = HippocampalReplay()
        state = GlobalState()
        state.final_action = 'explore'
        state.perception['objects'] = ['tree']
        state.concepts = ['forest']
        state.emotions = {'joy': 0.5}
        state.learning['reward'] = 0.5
        state.time = 1
        
        # Добавляем несколько эпизодов
        for i in range(25):
            state.time = i
            replay._store_episode(state)
            if i % 5 == 0:
                state.concepts = [f'concept_{i}']
        
        # Обновляем (должен запустить реплей)
        new_state = replay.update(state)
        self.assertIsNotNone(new_state)
        self.assertGreater(len(replay.synaptic_weights), 0)
    
    def test_consolidation(self):
        replay = HippocampalReplay()
        state = GlobalState()
        state.concepts = ['food', 'apple', 'sweet']
        state.final_action = 'eat'
        state.learning['reward'] = 0.9
        
        for _ in range(3):
            replay._store_episode(state)
            replay._replay_episode(replay.episode_buffer[-1], state)
        
        self.assertGreater(len(replay.synaptic_weights), 0)
        self.assertIn(('food', 'apple'), replay.synaptic_weights)
        self.assertGreater(replay.synaptic_weights.get(('food', 'apple'), 0), 0.1)


class TestInsularCortex(unittest.TestCase):
    """Тесты для островковой коры"""
    
    def test_creation(self):
        insula = InsularCortex()
        self.assertIsNotNone(insula)
        self.assertEqual(insula.filter_window, 10)
        self.assertEqual(insula.homeostatic_drive, 0.5)
    
    def test_update(self):
        insula = InsularCortex()
        state = GlobalState()
        state.body['heart_rate'] = 80.0
        state.body['breath_rate'] = 16.0
        state.body['temperature'] = 37.0
        state.body['blood_pressure'] = 120.0
        state.body['sleep_debt'] = 0.3
        state.emotions['threat'] = 0.2
        state.time = 1
        
        new_state = insula.update(state)
        self.assertIsNotNone(new_state)
        self.assertTrue(hasattr(new_state, 'interoceptive_state'))
        self.assertIn('body_signals', new_state.interoceptive_state)
        self.assertIn('somatic_marker', new_state.interoceptive_state)
        self.assertIn('homeostatic_drive', new_state.interoceptive_state)
    
    def test_homeostatic_computation(self):
        insula = InsularCortex()
        state = GlobalState()
        state.body['heart_rate'] = 100.0
        state.body['temperature'] = 38.5
        state.body['sleep_debt'] = 0.8
        
        drive = insula._compute_homeostatic_drive(state.body)
        self.assertGreaterEqual(drive, 0.0)
        self.assertLessEqual(drive, 1.0)
        self.assertGreater(drive, 0.5)


class TestIntegration(unittest.TestCase):
    """Тесты интеграции всех новых модулей"""
    
    def test_all_modules_update(self):
        state = GlobalState()
        state.perception['objects'] = ['food']
        state.concepts = ['food', 'eat']
        state.emotions = {'joy': 0.7, 'threat': 0.1}
        state.body['energy'] = 0.5
        state.body['heart_rate'] = 75.0
        state.body['temperature'] = 37.0
        state.neuromodulators['dopamine'] = 0.6
        state.planning['current_goal'] = 'eat'
        state.actions = ['eat', 'explore', 'rest']
        state.time = 1
        state.learning['reward'] = 0.5
        
        # Инициализация модулей
        dlpfc = DL_PFC()
        vmpfc = VM_PFC()
        replay = HippocampalReplay()
        insula = InsularCortex()
        
        # Последовательная обработка
        state = insula.update(state)
        state = replay.update(state)
        state = dlpfc.update(state)
        state = vmpfc.update(state)
        
        # Проверка результатов
        self.assertIn('interoceptive_state', state.__dict__)
        self.assertIn('working_memory', state.__dict__)
        self.assertIn('emotion_regulation', state.__dict__)
        
        # Проверка целостности
        self.assertIsInstance(state.working_memory, list)
        self.assertIsInstance(state.interoceptive_state, dict)
        self.assertGreaterEqual(state.emotion_regulation, 0.0)
        self.assertLessEqual(state.emotion_regulation, 1.0)


def run_tests():
    """Запуск всех тестов"""
    print("="*60)
    print("ЗАПУСК ТЕСТОВ НОВЫХ НЕЙРОБИОЛОГИЧЕСКИХ МОДУЛЕЙ")
    print("="*60)
    
    # Создаём тестовый набор
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print(f"ИТОГИ: {result.testsRun} тестов выполнено")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Провалено: {len(result.failures)}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    run_tests()
