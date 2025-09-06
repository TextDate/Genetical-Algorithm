"""
Integration Tests for Complete Genetic Algorithm Pipeline

Tests the full end-to-end GA workflow including:
- Complete GA runs with all components
- Component integration and data flow
- Error handling and recovery
- Configuration management
- Performance and resource usage
"""

import os
import sys
import unittest
import tempfile
import time
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithm import GeneticAlgorithm
from ga_config import GAConfig
from ga_logging import setup_logging
from ga_exceptions import GAException, CompressionError, InvalidFitnessError
from tests.test_fixtures import TestFixtures, MockCompressorFactory, TestDataBuilder


class TestGAIntegration(unittest.TestCase):
    """Integration tests for complete GA pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Suppress logging during tests
        setup_logging(level="ERROR", log_to_file=False)
        
        # Test data
        self.test_params = TestFixtures.get_minimal_param_values()
        self.test_config = TestFixtures.get_test_config(
            population_size=6, generations=2, max_threads=2
        )
        self.temp_files = []
        self.temp_dirs = []
    
    def tearDown(self):
        """Clean up test environment."""
        for temp_file in self.temp_files:
            TestFixtures.cleanup_path(temp_file)
        for temp_dir in self.temp_dirs:
            TestFixtures.cleanup_path(temp_dir)
    
    def test_complete_ga_run_success(self):
        """Test successful complete GA run."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_reliable_compressor(2.5))
                .with_temp_output_dir()
                .build())
            
            # Create and run GA
            ga = GeneticAlgorithm(params, compressor, config)
            
            # Test that GA runs successfully
            best_solution, best_fitness = ga.run()
            
            # Assertions
            self.assertIsNotNone(best_solution, "GA should return best solution")
            self.assertIsNotNone(best_fitness, "GA should return best fitness")
            self.assertIsInstance(best_solution, list, "Best solution should be list")
            self.assertIsInstance(best_fitness, (int, float), "Best fitness should be numeric")
            self.assertGreater(best_fitness, 0, "Best fitness should be positive")
            
            # Check solution structure
            self.assertEqual(len(best_solution), compressor.nr_models)
            for model_params in best_solution:
                self.assertIsInstance(model_params, dict)
                for param_name in params.keys():
                    self.assertIn(param_name, model_params)
    
    def test_ga_with_configuration_validation(self):
        """Test GA creation with various configurations."""
        # Valid configuration
        valid_config = GAConfig(
            population_size=8, generations=3, mutation_rate=0.01,
            crossover_rate=0.8, max_threads=2, output_dir="test_valid"
        )
        
        compressor = MockCompressorFactory.create_reliable_compressor()
        ga = GeneticAlgorithm(self.test_params, compressor, valid_config)
        self.assertIsNotNone(ga, "GA should be created with valid config")
        
        # Invalid configuration should raise error during GAConfig creation
        with self.assertRaises(ValueError):
            invalid_config = GAConfig(
                population_size=3,  # Too small
                generations=1,
                mutation_rate=0.01,
                crossover_rate=0.8,
                max_threads=2,
                output_dir="test_invalid"
            )
    
    def test_ga_component_integration(self):
        """Test that all GA components work together correctly."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=8, generations=2)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_varying_fitness_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            
            # Test component initialization
            self.assertIsNotNone(ga.parameter_encoder, "Parameter encoder should be initialized")
            self.assertIsNotNone(ga.population_manager, "Population manager should be initialized")
            self.assertIsNotNone(ga.selection_methods, "Selection methods should be initialized")
            self.assertIsNotNone(ga.genetic_operations, "Genetic operations should be initialized")
            self.assertIsNotNone(ga.evaluation_engine, "Evaluation engine should be initialized")
            self.assertIsNotNone(ga.duplicate_prevention, "Duplicate prevention should be initialized")
            self.assertIsNotNone(ga.convergence_detector, "Convergence detector should be initialized")
            self.assertIsNotNone(ga.reporter, "Reporter should be initialized")
            
            # Test component configuration
            self.assertEqual(ga.population_manager.population_size, config.population_size)
            self.assertEqual(ga.evaluation_engine.max_threads, config.max_threads)
            self.assertEqual(ga.genetic_operations.mutation_rate, config.mutation_rate)
            self.assertEqual(ga.genetic_operations.crossover_rate, config.crossover_rate)
            
            # Test that components can work together
            best_solution, best_fitness = ga.run()
            self.assertIsNotNone(best_solution)
            self.assertGreater(best_fitness, 0)
    
    def test_ga_error_handling_and_recovery(self):
        """Test GA behavior with various error conditions."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_unreliable_compressor(0.2))
                .with_temp_output_dir()
                .build())
            
            # GA should handle compression errors gracefully
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            # Should still complete despite some failures
            self.assertIsNotNone(best_solution)
            self.assertGreaterEqual(best_fitness, config.min_fitness)
            
            # Check that error statistics were recorded
            self.assertGreaterEqual(ga.evaluation_engine.stats['evaluation_errors'], 0)
    
    def test_ga_parallel_processing(self):
        """Test GA with parallel processing."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=8, generations=1, max_threads=4)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            
            start_time = time.time()
            best_solution, best_fitness = ga.run()
            end_time = time.time()
            
            # Should complete successfully
            self.assertIsNotNone(best_solution)
            self.assertGreater(best_fitness, 0)
            
            # Check parallel processing was used
            self.assertEqual(ga.evaluation_engine.max_threads, 4)
            self.assertGreater(ga.evaluation_engine.stats['parallel_batches'], 0)
    
    def test_ga_convergence_detection(self):
        """Test GA convergence detection."""
        with TestDataBuilder() as builder:
            # Use a compressor that returns consistent fitness for convergence
            consistent_compressor = MockCompressorFactory.create_reliable_compressor(3.0)
            
            config, params, compressor = (builder
                .with_config(
                    population_size=6, 
                    generations=10,  # High generations to test early convergence
                    convergence_generations=3,
                    convergence_threshold=0.001
                )
                .with_parameters(self.test_params)
                .with_compressor(consistent_compressor)
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            # Should converge early
            self.assertIsNotNone(best_solution)
            self.assertAlmostEqual(best_fitness, 3.0, places=2)
            
            # Check convergence was detected
            convergence_stats = ga.convergence_detector.get_statistics()
            self.assertGreater(convergence_stats['fitness_history_length'], 0)
    
    def test_ga_output_generation(self):
        """Test that GA generates expected output files."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=4, generations=2)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            # Check output directory was created
            self.assertTrue(os.path.exists(config.output_dir), "Output directory should exist")
            
            # Check that some files were generated
            file_count = TestFixtures.count_test_files_created(config.output_dir)
            self.assertGreater(file_count, 0, "Should generate output files")
            
            # Check for expected file types
            csv_files = TestFixtures.count_test_files_created(config.output_dir, ".csv")
            json_files = TestFixtures.count_test_files_created(config.output_dir, ".json")
            log_files = TestFixtures.count_test_files_created(config.output_dir, ".log")
            
            self.assertGreater(csv_files, 0, "Should generate CSV files")
            self.assertGreater(json_files, 0, "Should generate JSON files")
    
    def test_ga_memory_usage(self):
        """Test GA memory usage and cleanup."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            # Track memory before and after
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # Should complete successfully
            self.assertIsNotNone(best_solution)
            
            # Memory increase should be reasonable (less than 100MB)
            self.assertLess(memory_increase, 100 * 1024 * 1024, 
                          "Memory increase should be reasonable")
    
    def test_ga_with_different_parameter_sets(self):
        """Test GA with different parameter configurations."""
        test_cases = [
            # Minimal parameters
            TestFixtures.get_minimal_param_values(),
            # Single parameter
            {'level': [1, 2, 3, 4, 5]},
            # Two parameters
            {'level': [1, 3], 'window_log': [10, 12]},
        ]
        
        for i, params in enumerate(test_cases):
            with self.subTest(f"Parameter set {i+1}"):
                with TestDataBuilder() as builder:
                    config, _, compressor = (builder
                        .with_config(population_size=4, generations=1)
                        .with_parameters(params)
                        .with_compressor(MockCompressorFactory.create_reliable_compressor())
                        .with_temp_output_dir()
                        .build())
                    
                    ga = GeneticAlgorithm(params, compressor, config)
                    best_solution, best_fitness = ga.run()
                    
                    self.assertIsNotNone(best_solution)
                    self.assertGreater(best_fitness, 0)
                    
                    # Check solution has all parameters
                    for param_name in params.keys():
                        self.assertIn(param_name, best_solution[0])
    
    def test_ga_statistics_collection(self):
        """Test that GA collects comprehensive statistics."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters(self.test_params)
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            # Check component statistics
            eval_stats = ga.evaluation_engine.get_statistics()
            self.assertGreater(eval_stats['evaluations_performed'], 0)
            self.assertGreaterEqual(eval_stats['parallel_batches'], 0)
            
            pop_stats = ga.population_manager.get_statistics()
            self.assertIn('individuals_created', pop_stats)
            
            selection_stats = ga.selection_methods.get_statistics()
            self.assertIn('selection_operations', selection_stats)
            
            genetic_stats = ga.genetic_operations.get_statistics()
            self.assertIn('crossover_operations', genetic_stats)
            self.assertIn('mutation_operations', genetic_stats)
            
            dup_stats = ga.duplicate_prevention.get_statistics()
            self.assertIn('duplicates_detected', dup_stats)
            
            conv_stats = ga.convergence_detector.get_statistics()
            self.assertIn('fitness_history_length', conv_stats)


class TestGAEdgeCases(unittest.TestCase):
    """Test GA behavior in edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test environment."""
        setup_logging(level="ERROR", log_to_file=False)
    
    def test_minimal_population_and_generations(self):
        """Test GA with minimal viable configuration."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=4, generations=1)  # Minimum viable
                .with_parameters({'level': [1, 2]})  # Minimal params
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            self.assertGreater(best_fitness, 0)
    
    def test_single_thread_execution(self):
        """Test GA with single-threaded execution."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=4, generations=1, max_threads=1)
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            self.assertEqual(ga.evaluation_engine.max_threads, 1)
    
    def test_high_failure_rate_resilience(self):
        """Test GA resilience to high compression failure rates."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(MockCompressorFactory.create_unreliable_compressor(0.5))
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            # Should still complete despite 50% failure rate
            self.assertIsNotNone(best_solution)
            self.assertGreaterEqual(best_fitness, config.min_fitness)
            self.assertGreater(ga.evaluation_engine.stats['evaluation_errors'], 0)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestGAIntegration))
    suite.addTest(unittest.makeSuite(TestGAEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)