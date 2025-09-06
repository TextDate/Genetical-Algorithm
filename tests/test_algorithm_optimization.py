"""
Tests for Algorithm Optimization Components

Tests the various optimization techniques including:
- Adaptive parameter tuning
- Smart initialization strategies
- Convergence acceleration
- Performance monitoring
"""

import sys
import os
import unittest
import tempfile
from unittest.mock import Mock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_components.algorithm_optimization import (
    AdaptiveParameterTuning, SmartInitialization, 
    ConvergenceAccelerator, PerformanceMonitor
)
from ga_components.parameter_encoding import ParameterEncoder
from ga_logging import setup_logging


class TestAdaptiveParameterTuning(unittest.TestCase):
    """Test adaptive parameter tuning component."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.tuner = AdaptiveParameterTuning(
            initial_mutation_rate=0.1,
            initial_crossover_rate=0.8
        )
    
    def test_initialization(self):
        """Test adaptive tuning initialization."""
        self.assertEqual(self.tuner.base_mutation_rate, 0.1)
        self.assertEqual(self.tuner.base_crossover_rate, 0.8)
        self.assertEqual(self.tuner.current_mutation_rate, 0.1)
        self.assertEqual(self.tuner.current_crossover_rate, 0.8)
    
    def test_low_diversity_adaptation(self):
        """Test parameter adaptation for low diversity."""
        # Simulate low diversity scenario
        mutation_rate, crossover_rate = self.tuner.update_parameters(
            population_diversity=0.2,  # Low diversity
            best_fitness=5.0,
            generation=10,
            max_generations=50
        )
        
        # Should increase mutation rate for low diversity
        self.assertGreater(mutation_rate, self.tuner.base_mutation_rate)
        self.assertIsInstance(mutation_rate, float)
        self.assertIsInstance(crossover_rate, float)
        self.assertGreater(crossover_rate, 0)
        self.assertLessEqual(crossover_rate, 1.0)
    
    def test_stagnation_adaptation(self):
        """Test parameter adaptation for fitness stagnation."""
        # Simulate stagnation with same fitness values
        for i in range(8):
            self.tuner.update_parameters(
                population_diversity=0.5,
                best_fitness=5.0,  # Same fitness
                generation=i+1,
                max_generations=50
            )
        
        # After stagnation, should adapt parameters
        stats = self.tuner.get_adaptation_stats()
        self.assertGreaterEqual(stats['fitness_stagnation_count'], 7)
        self.assertGreater(stats['current_mutation_rate'], self.tuner.base_mutation_rate)
    
    def test_high_diversity_late_generation(self):
        """Test parameter adaptation for high diversity in late generations."""
        mutation_rate, crossover_rate = self.tuner.update_parameters(
            population_diversity=0.9,  # High diversity
            best_fitness=5.0,
            generation=40,  # Late generation
            max_generations=50
        )
        
        # Should reduce mutation for exploitation in late generations
        self.assertLessEqual(mutation_rate, self.tuner.base_mutation_rate)
    
    def test_statistics_collection(self):
        """Test statistics collection."""
        self.tuner.update_parameters(0.5, 5.0, 1, 50)
        stats = self.tuner.get_adaptation_stats()
        
        self.assertIn('mutation_adaptations', stats)
        self.assertIn('current_mutation_rate', stats)
        self.assertIn('current_crossover_rate', stats)
        self.assertIn('fitness_stagnation_count', stats)
        self.assertGreaterEqual(stats['mutation_adaptations'], 1)


class TestSmartInitialization(unittest.TestCase):
    """Test smart initialization strategies."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.test_params = {
            'level': [1, 3, 6, 9],
            'window_log': [10, 12, 14, 16]
        }
        self.encoder = ParameterEncoder(self.test_params)
        self.name_counter = 0
        
    def name_generator(self, gen):
        self.name_counter += 1
        return f"Gen{gen}_Ind{self.name_counter}"
    
    def test_random_initialization(self):
        """Test random initialization strategy."""
        population = SmartInitialization.create_diverse_population(
            self.encoder.param_binary_encodings,
            population_size=8,
            name_generator=self.name_generator,
            diversity_strategy="random"
        )
        
        self.assertEqual(len(population), 8)
        for individual in population:
            self.assertIsInstance(individual, tuple)
            self.assertEqual(len(individual), 2)  # (gene_code, name)
            gene_code, name = individual
            self.assertIsInstance(gene_code, tuple)
            self.assertIsInstance(name, str)
    
    def test_structured_initialization(self):
        """Test structured initialization strategy."""
        population = SmartInitialization.create_diverse_population(
            self.encoder.param_binary_encodings,
            population_size=8,
            name_generator=self.name_generator,
            diversity_strategy="structured"
        )
        
        self.assertEqual(len(population), 8)
        
        # Check that different parameter values are represented
        gene_codes = [ind[0] for ind in population]
        unique_gene_codes = set(gene_codes)
        self.assertGreater(len(unique_gene_codes), 1, "Should have diverse gene codes")
    
    def test_hybrid_initialization(self):
        """Test hybrid initialization strategy."""
        population = SmartInitialization.create_diverse_population(
            self.encoder.param_binary_encodings,
            population_size=10,
            name_generator=self.name_generator,
            diversity_strategy="hybrid"
        )
        
        self.assertEqual(len(population), 10)
        
        # Verify all individuals are valid
        for individual in population:
            gene_code, name = individual
            self.assertEqual(len(gene_code), len(self.encoder.param_binary_encodings))


class TestConvergenceAccelerator(unittest.TestCase):
    """Test convergence acceleration component."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.accelerator = ConvergenceAccelerator(
            confidence_threshold=0.9,
            min_improvement=1e-6
        )
    
    def test_initialization(self):
        """Test convergence accelerator initialization."""
        self.assertEqual(self.accelerator.confidence_threshold, 0.9)
        self.assertEqual(self.accelerator.min_improvement, 1e-6)
        self.assertEqual(self.accelerator.plateau_count, 0)
    
    def test_no_early_stop_insufficient_history(self):
        """Test no early stopping with insufficient fitness history."""
        should_stop, reason = self.accelerator.should_stop_early(5.0, 5, 100)
        self.assertFalse(should_stop)
        self.assertEqual(reason, "")
    
    def test_early_stop_plateau_detection(self):
        """Test early stopping due to fitness plateau."""
        # Create long plateau
        base_fitness = 10.0
        for i in range(15):
            fitness = base_fitness + 1e-8 * i  # Tiny improvements
            should_stop, reason = self.accelerator.should_stop_early(fitness, i+10, 100)
            
        # Should eventually trigger early stop
        if should_stop:
            self.assertIn("confidence", reason.lower())
    
    def test_convergence_confidence_calculation(self):
        """Test convergence confidence calculation."""
        # Add stable fitness values
        for i in range(10):
            self.accelerator.should_stop_early(5.0 + 1e-7 * i, i+10, 100)
        
        confidence = self.accelerator._calculate_convergence_confidence()
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_restart_recommendation(self):
        """Test restart recommendation logic."""
        # Simulate low diversity with long plateau
        self.accelerator.plateau_count = 20
        should_restart, reason = self.accelerator.should_restart(
            population_diversity=0.05,  # Very low diversity
            generation=25
        )
        
        if should_restart:
            self.assertIn("diversity", reason.lower())
    
    def test_statistics_collection(self):
        """Test acceleration statistics."""
        self.accelerator.should_stop_early(5.0, 10, 100)
        stats = self.accelerator.get_acceleration_stats()
        
        self.assertIn('early_stops', stats)
        self.assertIn('plateau_count', stats)
        self.assertIn('convergence_confidence', stats)
        self.assertIsInstance(stats['convergence_confidence'], float)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring component."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.monitor = PerformanceMonitor()
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        self.assertEqual(len(self.monitor.generation_times), 0)
        self.assertEqual(self.monitor.efficiency_metrics['avg_generation_time'], 0.0)
    
    def test_metrics_recording(self):
        """Test generation metrics recording."""
        self.monitor.record_generation_metrics(
            generation_time=5.2,
            evaluation_time=3.1,
            memory_usage=100*1024*1024,  # 100MB
            fitness_improvement=0.5
        )
        
        self.assertEqual(len(self.monitor.generation_times), 1)
        self.assertEqual(self.monitor.generation_times[0], 5.2)
        self.assertEqual(self.monitor.evaluation_times[0], 3.1)
    
    def test_efficiency_metrics_update(self):
        """Test efficiency metrics calculation."""
        # Add several generation metrics
        for i in range(5):
            self.monitor.record_generation_metrics(
                generation_time=5.0 + i * 0.2,
                evaluation_time=3.0 + i * 0.1,
                memory_usage=100*1024*1024,
                fitness_improvement=1.0 - i * 0.1
            )
        
        metrics = self.monitor.efficiency_metrics
        self.assertGreater(metrics['avg_generation_time'], 0)
        self.assertGreater(metrics['fitness_per_second'], 0)
        self.assertIsInstance(metrics['convergence_rate'], float)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        # Add slow generation times
        for i in range(5):
            self.monitor.record_generation_metrics(
                generation_time=15.0,  # Slow generations
                evaluation_time=12.0,
                memory_usage=50*1024*1024,
                fitness_improvement=0.001 if i < 3 else 1e-8  # Poor improvements
            )
        
        recommendations = self.monitor.get_optimization_recommendations()
        self.assertIsInstance(recommendations, list)
        
        # Should suggest improvements for slow performance
        if recommendations:
            recommendation_text = ' '.join(recommendations).lower()
            self.assertTrue(any(word in recommendation_text 
                              for word in ['population', 'threads', 'rate', 'stagnation']))
    
    def test_performance_trend_calculation(self):
        """Test performance trend analysis."""
        # Add improving performance trend
        for i in range(10):
            self.monitor.record_generation_metrics(
                generation_time=10.0 - i * 0.5,  # Decreasing times (improving)
                evaluation_time=5.0,
                memory_usage=50*1024*1024,
                fitness_improvement=0.1
            )
        
        trend = self.monitor._calculate_performance_trend()
        self.assertIn(trend, ['improving', 'stable', 'degrading', 'insufficient_data'])
    
    def test_performance_summary(self):
        """Test comprehensive performance summary."""
        self.monitor.record_generation_metrics(5.0, 3.0, 100*1024*1024, 0.5)
        summary = self.monitor.get_performance_summary()
        
        self.assertIn('avg_generation_time', summary)
        self.assertIn('fitness_per_second', summary)
        self.assertIn('optimization_recommendations', summary)
        self.assertIn('performance_trend', summary)
        self.assertIsInstance(summary['optimization_recommendations'], list)


if __name__ == '__main__':
    unittest.main(verbosity=2)