# -*- coding: utf-8 -*-
"""
Тесты для проверки всех модулей AGI v7
Запуск: python -m pytest agi_v7/tests/test_modules.py
Или: python agi_v7/tests/test_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
from agi_v7.core.state import GlobalState
from agi_v7.modules import (
    VisualCortexModule,
    LimbicModule,
    BasalGangliaModule,
    HippocampusModule,
    SemanticMemory,
    MemoryConsolidator,
)
from agi_v7.modules.neural_chains import Neuron, NeuralChain, ChainPool


class TestState(unittest.TestCase):
    """Тесты глобального состояния"""
    
    def test_state_creation(self):
        state = GlobalState()
        self.assertIsNotNone(state)
        self.assertEqual(state.step, 0)
        self.assertEqual(state.energy, 100.0)
        self.assertIn('valence', state.emotions)
    
    def test_state_update(self):
        state = GlobalState()
        state.step = 10
        state.energy = 80.0
        self.assertEqual(state.step, 10)
        self.assertEqual(state.energy, 80.0)


class TestNeuralChains(unittest.TestCase):
    """Тесты нейронных цепочек"""
    
    def test_neuron_creation(self):
        neuron = Neuron()
        self.assertEqual(neuron.membrane_potential, 0.0)
        self.assertFalse(neuron.spiked)
    
    def test_neuron_update(self):
        neuron = Neuron()
        neuron.membrane_potential = 0.5
        neuron.update(0.3, 0.1)
        self.assertGreater(neuron.membrane_potential, 0.5)
    
    def test_chain_creation(self):
        chain = NeuralChain(length=5)
        self.assertEqual(len(chain.neurons), 5)
        self.assertEqual(chain.energy_cost, 0.5)
    
    def test_chain_process(self):
        chain = NeuralChain(length=5)
        output = chain.process(0.5, 0.1)
        self.assertIsInstance(output, float)
        self.assertGreaterEqual(output, 0.0)
        self.assertLessEqual(output, 1.0)
    
    def test_chain_pool(self):
        pool = ChainPool(num_short=10, num_medium=5, num_long=2)
        self.assertEqual(pool.total_chains, 17)
        self.assertEqual(pool.short_chains, 10)
        self.assertEqual(pool.medium_chains, 5)
        self.assertEqual(pool.long_chains, 2)
    
    def test_chain_pool_process(self):
        pool = ChainPool(num_short=5, num_medium=3, num_long=1)
        output, stats = pool.process(0.5, 0.7, 0.3, 0.1)
        self.assertIsInstance(output, float)
        self.assertIn('num_active', stats)
        self.assertIn('energy_used', stats)
        self.assertGreater(stats['num_active'], 0)


class TestVisualCortex(unittest.TestCase):
    """Тесты зрительной коры"""
    
    def test_visual_cortex_creation(self):
        cortex = VisualCortexModule()
        self.assertIsNotNone(cortex.chain_pool)
        self.assertEqual(cortex.chain_pool.total_chains, 170)
    
    def test_visual_cortex_update(self):
        cortex = VisualCortexModule()
        state = GlobalState()
        state.perception['visual'] = {'brightness': 0.7, 'motion': 0.3}
        state.energy = 90.0
        state.attention_salience = 0.5
        state.perception['novelty'] = 0.2
        
        new_state = cortex.update(state)
        self.assertIsNotNone(new_state)
        self.assertTrue(new_state.perception.get('visual_processed', False))
        self.assertIn('chain_stats', new_state.perception)


class TestLimbicModule(unittest.TestCase):
    """Тесты лимбической системы"""
    
    def test_limbic_creation(self):
        limbic = LimbicModule()
        self.assertIsNotNone(limbic)
        self.assertEqual(limbic.emotional_stage, 'ambition')
    
    def test_limbic_update(self):
        limbic = LimbicModule()
        state = GlobalState()
        state.emotions['ambition'] = 0.7
        state.emotions['valence'] = 0.5
        state.emotions['arousal'] = 0.4
        state.energy = 80.0
        state.step = 5
        
        new_state = limbic.update(state)
        self.assertIsNotNone(new_state)
        self.assertIn('limbic_stage', new_state.emotions)


class TestBasalGanglia(unittest.TestCase):
    """Тесты базальных ганглий"""
    
    def test_basal_ganglia_creation(self):
        bg = BasalGangliaModule()
        self.assertIsNotNone(bg)
        self.assertIsNotNone(bg.q_table)
    
    def test_basal_ganglia_update(self):
        bg = BasalGangliaModule()
        state = GlobalState()
        state.step = 1
        state.perception['danger'] = False
        state.energy = 70.0
        state.emotions['valence'] = 0.4
        
        new_state = bg.update(state)
        self.assertIsNotNone(new_state)
        self.assertIn('basal_ganglia', new_state.candidates)


class TestHippocampus(unittest.TestCase):
    """Тесты гиппокампа"""
    
    def test_hippocampus_creation(self):
        hc = HippocampusModule()
        self.assertIsNotNone(hc)
        self.assertIsNotNone(hc.episodic_buffer)
    
    def test_hippocampus_update(self):
        hc = HippocampusModule()
        state = GlobalState()
        state.step = 1
        state.perception = {'danger': False, 'objects': ['food']}
        state.emotions = {'valence': 0.6, 'arousal': 0.4}
        state.action = 'eat'
        state.energy = 80.0
        
        new_state = hc.update(state)
        self.assertIsNotNone(new_state)
        self.assertGreater(len(hc.episodic_buffer), 0)


class TestSemanticMemory(unittest.TestCase):
    """Тесты семантической памяти"""
    
    def test_semantic_memory_creation(self):
        sm = SemanticMemory()
        self.assertIsNotNone(sm)
        self.assertEqual(len(sm.concepts), 0)
    
    def test_semantic_memory_observe(self):
        sm = SemanticMemory()
        state = GlobalState()
        state.perception = {'danger': True, 'objects': ['predator']}
        state.emotions = {'fear': 0.7}
        state.learning = {'reward': -0.5}
        
        observation = {
            'perception': state.perception,
            'emotions': state.emotions,
            'reward': state.learning.get('reward', 0.0)
        }
        
        concepts = sm.observe(observation, state)
        self.assertIsInstance(concepts, list)
        self.assertGreater(len(sm.concepts), 0)
    
    def test_semantic_memory_retrieve(self):
        sm = SemanticMemory()
        state = GlobalState()
        state.perception = {'danger': True}
        state.emotions = {'fear': 0.7}
        
        observation = {
            'perception': state.perception,
            'emotions': state.emotions,
            'reward': 0.0
        }
        
        sm.observe(observation, state)
        results = sm.retrieve('danger', top_k=3)
        self.assertIsInstance(results, list)


class TestMemoryConsolidator(unittest.TestCase):
    """Тесты консолидатора памяти"""
    
    def test_consolidator_creation(self):
        sm = SemanticMemory()
        consolidator = MemoryConsolidator(sm)
        self.assertIsNotNone(consolidator)
        self.assertEqual(len(consolidator.episodes), 0)
    
    def test_consolidator_add_episode(self):
        sm = SemanticMemory()
        consolidator = MemoryConsolidator(sm)
        state = GlobalState()
        state.perception = {'danger': False, 'objects': ['food']}
        state.emotions = {'joy': 0.8}
        state.learning = {'reward': 0.7}
        
        episode = {
            'perception': state.perception,
            'emotions': state.emotions,
            'reward': 0.7,
            'action': 'eat'
        }
        
        consolidator.add_episode(episode, state)
        self.assertEqual(len(consolidator.episodes), 1)
    
    def test_consolidator_consolidate(self):
        sm = SemanticMemory()
        consolidator = MemoryConsolidator(sm)
        
        # Добавляем несколько эпизодов
        for i in range(3):
            state = GlobalState()
            state.perception = {'objects': [f'obj_{i}']}
            state.emotions = {'joy': 0.5 + i * 0.1}
            state.learning = {'reward': 0.5 + i * 0.1}
            
            episode = {
                'perception': state.perception,
                'emotions': state.emotions,
                'reward': 0.5 + i * 0.1,
                'action': 'explore'
            }
            consolidator.add_episode(episode, state)
        
        patterns = consolidator.consolidate()
        self.assertIsNotNone(patterns)


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_full_pipeline(self):
        """Тест полного пайплайна"""
        from agi_v7 import CognitiveOrchestrator
        
        orchestrator = CognitiveOrchestrator()
        
        # Регистрируем модули
        orchestrator.register('visual', VisualCortexModule())
        orchestrator.register('limbic', LimbicModule())
        orchestrator.register('basal_ganglia', BasalGangliaModule())
        orchestrator.register('hippocampus', HippocampusModule())
        
        # Добавляем семантическую память
        sm = SemanticMemory()
        orchestrator.semantic_memory = sm
        orchestrator.consolidator = MemoryConsolidator(sm)
        
        # Запускаем несколько шагов
        for i in range(5):
            perception = {
                'danger': i % 2 == 0,
                'objects': [f'obj_{i}'],
                'novelty': i * 0.1
            }
            state = orchestrator.step(perception=perception)
            self.assertIsNotNone(state)
            self.assertGreaterEqual(state.energy, 0.0)
            self.assertLessEqual(state.energy, 100.0)
        
        self.assertGreater(len(sm.concepts), 0)


def run_tests():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов AGI v7...")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем все тесты
    suite.addTests(loader.loadTestsFromTestCase(TestState))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralChains))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualCortex))
    suite.addTests(loader.loadTestsFromTestCase(TestLimbicModule))
    suite.addTests(loader.loadTestsFromTestCase(TestBasalGanglia))
    suite.addTests(loader.loadTestsFromTestCase(TestHippocampus))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryConsolidator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    print(f"✅ Тестов пройдено: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Ошибок: {len(result.errors)}")
    print(f"⚠️  Падений: {len(result.failures)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
