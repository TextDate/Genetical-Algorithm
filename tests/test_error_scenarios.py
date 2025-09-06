"""
Error Scenarios and Edge Cases Tests

Tests GA behavior under various error conditions, edge cases,
and boundary conditions to ensure robustness and reliability.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithm import GeneticAlgorithm
from ga_config import GAConfig
from ga_logging import setup_logging
from ga_exceptions import (
    CompressionError, CompressionTimeoutError, InvalidFitnessError,
    validate_fitness, handle_compression_error
)
from tests.test_fixtures import TestFixtures, MockCompressorFactory, TestDataBuilder


class TestErrorHandling(unittest.TestCase):
    """Test GA error handling capabilities."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
    
    def test_compression_failures(self):
        """Test GA behavior with compression failures."""
        with TestDataBuilder() as builder:
            # Create compressor that fails 30% of the time
            failing_compressor = MockCompressorFactory.create_unreliable_compressor(0.3)
            
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(failing_compressor)
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            
            # Should complete despite failures
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            self.assertGreaterEqual(best_fitness, config.min_fitness)
            
            # Should have recorded errors but continued
            self.assertGreaterEqual(ga.evaluation_engine.stats['evaluation_errors'], 0)
            self.assertGreater(ga.evaluation_engine.stats['evaluations_performed'], 0)
    
    def test_compression_timeouts(self):
        """Test GA behavior with compression timeouts."""
        with TestDataBuilder() as builder:
            # Create compressor that times out 20% of the time
            timeout_compressor = MockCompressorFactory.create_timeout_compressor(0.2)
            
            config, params, compressor = (builder
                .with_config(population_size=6, generations=1)
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(timeout_compressor)
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            # Should complete despite timeouts
            self.assertIsNotNone(best_solution)
            self.assertGreaterEqual(best_fitness, config.min_fitness)
    
    def test_invalid_fitness_handling(self):
        """Test handling of invalid fitness values."""
        # Test fitness validation function
        with self.assertRaises(InvalidFitnessError):
            validate_fitness(None, "test_individual")
        
        with self.assertRaises(InvalidFitnessError):
            validate_fitness(-1.0, "test_individual", min_expected=0.0)
        
        with self.assertRaises(InvalidFitnessError):
            validate_fitness("invalid", "test_individual")
        
        # Valid fitness should pass
        result = validate_fitness(2.5, "test_individual")
        self.assertEqual(result, 2.5)
    
    def test_parallel_processing_fallback(self):
        """Test fallback to sequential processing when parallel fails."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=4, generations=1, max_threads=2)
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            
            # Mock ProcessPoolExecutor to fail
            with patch('ga_components.evaluation.concurrent.futures.ProcessPoolExecutor') as mock_pool:
                mock_pool.side_effect = Exception("ProcessPoolExecutor failed")
                
                # Should fallback to sequential and still work
                best_solution, best_fitness = ga.run()
                
                self.assertIsNotNone(best_solution)
                self.assertGreater(best_fitness, 0)
    
    def test_memory_constraints(self):
        """Test GA behavior under memory constraints."""
        with TestDataBuilder() as builder:
            # Large population to test memory usage
            config, params, compressor = (builder
                .with_config(population_size=20, generations=1, max_threads=4)
                .with_parameters(TestFixtures.get_comprehensive_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            
            # Monitor memory usage
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            best_solution, best_fitness = ga.run()
            
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            self.assertIsNotNone(best_solution)
            # Memory increase should be reasonable (less than 200MB for this test)
            self.assertLess(memory_increase, 200 * 1024 * 1024)
    
    def test_file_system_errors(self):
        """Test GA behavior with file system errors."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=4, generations=1)
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .build())
            
            # Use invalid output directory (should create it)
            config = config.update(output_dir="/invalid/path/that/should/fail")
            
            with self.assertRaises((OSError, PermissionError)):
                ga = GeneticAlgorithm(params, compressor, config)
                best_solution, best_fitness = ga.run()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
    
    def test_minimal_viable_configuration(self):
        """Test with minimal viable GA configuration."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(
                    population_size=4,  # Minimum even number
                    generations=1,      # Minimum generations
                    max_threads=1       # Single thread
                )
                .with_parameters({'level': [1, 2]})  # Minimal parameter space
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            self.assertGreater(best_fitness, 0)
    
    def test_single_parameter_optimization(self):
        """Test optimization with single parameter."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=6, generations=2)
                .with_parameters({'level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
                .with_compressor(MockCompressorFactory.create_varying_fitness_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            self.assertEqual(len(best_solution), 1)  # One model
            self.assertIn('level', best_solution[0])
            self.assertIn(best_solution[0]['level'], params['level'])
    
    def test_large_parameter_space(self):
        """Test with large parameter space."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(population_size=8, generations=1, max_threads=4)
                .with_parameters(TestFixtures.get_comprehensive_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            # Should have all parameters
            for param_name in params.keys():
                self.assertIn(param_name, best_solution[0])
    
    def test_high_mutation_rate(self):
        """Test GA with high mutation rate."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(
                    population_size=6,
                    generations=2,
                    mutation_rate=0.5,  # High mutation rate
                    crossover_rate=0.9
                )
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            # Should have high mutation operations
            self.assertGreater(ga.genetic_operations.get_statistics()['mutation_operations'], 0)
    
    def test_zero_crossover_rate(self):
        """Test GA with no crossover (mutation only)."""
        with TestDataBuilder() as builder:
            config, params, compressor = (builder
                .with_config(
                    population_size=6,
                    generations=2,
                    mutation_rate=0.2,
                    crossover_rate=0.0  # No crossover
                )
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(MockCompressorFactory.create_reliable_compressor())
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            # Should have no crossover operations but have mutations
            stats = ga.genetic_operations.get_statistics()
            self.assertEqual(stats['crossover_operations'], 0)
            self.assertGreater(stats['mutation_operations'], 0)
    
    def test_immediate_convergence(self):
        """Test GA that converges immediately."""
        with TestDataBuilder() as builder:
            # Use compressor that returns identical fitness for all individuals
            identical_compressor = MockCompressorFactory.create_reliable_compressor(3.0)
            
            config, params, compressor = (builder
                .with_config(
                    population_size=6,
                    generations=10,  # High generations
                    convergence_generations=2,  # Quick convergence detection
                    convergence_threshold=0.001
                )
                .with_parameters(TestFixtures.get_minimal_param_values())
                .with_compressor(identical_compressor)
                .with_temp_output_dir()
                .build())
            
            ga = GeneticAlgorithm(params, compressor, config)
            best_solution, best_fitness = ga.run()
            
            self.assertIsNotNone(best_solution)
            self.assertAlmostEqual(best_fitness, 3.0, places=2)
            
            # Should detect convergence early
            conv_stats = ga.convergence_detector.get_statistics()
            self.assertLess(conv_stats['generations_tracked'], 10)


class TestResourceLimits(unittest.TestCase):
    """Test GA behavior under resource constraints."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
    
    def test_thread_limit_validation(self):
        """Test thread limit validation and adjustment."""
        import os
        
        # Test with reasonable thread count
        config = GAConfig(
            population_size=8,
            generations=1,
            mutation_rate=0.01,
            crossover_rate=0.8,
            max_threads=min(4, os.cpu_count() or 2),
            output_dir="test_threads"
        )
        
        self.assertGreater(config.max_threads, 0)
        self.assertLessEqual(config.max_threads, os.cpu_count() * 4)  # Within reasonable bounds
    
    def test_population_size_constraints(self):
        """Test population size constraints and validation."""
        # Valid even population size
        config = GAConfig(
            population_size=8,
            generations=1,
            mutation_rate=0.01,
            crossover_rate=0.8,
            max_threads=2,
            output_dir="test_pop"
        )
        self.assertEqual(config.population_size, 8)
        
        # Invalid odd population size should be caught by GAConfig validation
        with self.assertRaises(ValueError):
            GAConfig(
                population_size=7,  # Odd number
                generations=1,
                mutation_rate=0.01,
                crossover_rate=0.8,
                max_threads=2,
                output_dir="test_pop_invalid"
            )
    
    def test_generation_constraints(self):
        """Test generation count constraints."""
        # Valid generation count
        config = GAConfig(
            population_size=4,
            generations=100,
            mutation_rate=0.01,
            crossover_rate=0.8,
            max_threads=2,
            output_dir="test_gen"
        )
        self.assertEqual(config.generations, 100)
        
        # Invalid generation count
        with self.assertRaises(ValueError):
            GAConfig(
                population_size=4,
                generations=0,  # Invalid
                mutation_rate=0.01,
                crossover_rate=0.8,
                max_threads=2,
                output_dir="test_gen_invalid"
            )


class TestExceptionHandling(unittest.TestCase):
    """Test specific exception handling scenarios."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
    
    def test_compression_error_handling(self):
        """Test specific compression error handling."""
        # Test handle_compression_error function
        test_error = Exception("Test compression failure")
        
        # Should return fallback fitness
        fallback_fitness = handle_compression_error(
            test_error, "zstd", "test_individual", use_fallback=True
        )
        self.assertEqual(fallback_fitness, 0.1)
        
        # Should raise exception when fallback disabled
        with self.assertRaises(CompressionError):
            handle_compression_error(
                test_error, "zstd", "test_individual", use_fallback=False
            )
    
    def test_timeout_error_handling(self):
        """Test timeout-specific error handling."""
        timeout_error = Exception("Compression timed out after 30 seconds")
        
        fallback_fitness = handle_compression_error(
            timeout_error, "lzma", "timeout_individual", use_fallback=True
        )
        self.assertEqual(fallback_fitness, 0.1)
    
    def test_custom_exception_creation(self):
        """Test custom exception classes."""
        # Test CompressionError
        comp_error = CompressionError(
            "Test error", 
            compressor_type="zstd",
            individual_name="test_ind",
            parameters={"level": 5}
        )
        self.assertEqual(comp_error.compressor_type, "zstd")
        self.assertEqual(comp_error.individual_name, "test_ind")
        self.assertIn("level", comp_error.parameters)
        
        # Test CompressionTimeoutError
        timeout_error = CompressionTimeoutError(
            30, compressor_type="lzma", individual_name="slow_ind"
        )
        self.assertEqual(timeout_error.timeout_seconds, 30)
        self.assertEqual(timeout_error.compressor_type, "lzma")
        
        # Test InvalidFitnessError
        fitness_error = InvalidFitnessError(
            -1.0, individual_name="bad_fitness", expected_range=(0.0, 10.0)
        )
        self.assertEqual(fitness_error.fitness_value, -1.0)
        self.assertEqual(fitness_error.expected_range, (0.0, 10.0))


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add error handling tests
    suite.addTest(unittest.makeSuite(TestErrorHandling))
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    suite.addTest(unittest.makeSuite(TestResourceLimits))
    suite.addTest(unittest.makeSuite(TestExceptionHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)