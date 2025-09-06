"""
Component Integration Tests

Tests integration between individual GA components to ensure
they work together correctly and data flows properly between them.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_components.parameter_encoding import ParameterEncoder
from ga_components.population_management import PopulationManager
from ga_components.selection import SelectionMethods
from ga_components.genetic_operations import GeneticOperations
from ga_components.evaluation import EvaluationEngine
from ga_components.convergence_detection import ConvergenceDetector
from ga_components.reporting import GAReporter

from tests.test_fixtures import TestFixtures, MockCompressorFactory, TestDataBuilder
from ga_logging import setup_logging


class TestParameterEncodingIntegration(unittest.TestCase):
    """Test parameter encoding integration with other components."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.test_params = TestFixtures.get_minimal_param_values()
        self.encoder = ParameterEncoder(self.test_params)
    
    def test_encoding_population_manager_integration(self):
        """Test parameter encoder integration with population manager."""
        population_manager = PopulationManager(
            self.encoder.param_binary_encodings,
            population_size=8,
            nr_models=1
        )
        
        # Test population creation
        population = population_manager.initialize_population()
        self.assertEqual(len(population), 8)
        
        # Test encoding/decoding consistency
        for individual in population:
            TestFixtures.assert_valid_individual(individual, self.encoder)
            
            # Test decoding
            decoded = population_manager.decode_individual(individual)
            self.assertIsInstance(decoded, list)
            self.assertEqual(len(decoded), 1)  # nr_models = 1
            
            decoded_params = decoded[0]
            self.assertIsInstance(decoded_params, dict)
            
            # Check all parameters are present
            for param_name in self.test_params.keys():
                self.assertIn(param_name, decoded_params)
                self.assertIn(decoded_params[param_name], self.test_params[param_name])
    
    def test_encoding_genetic_operations_integration(self):
        """Test parameter encoding with genetic operations."""
        population_manager = PopulationManager(
            self.encoder.param_binary_encodings,
            population_size=6,
            nr_models=1
        )
        
        genetic_ops = GeneticOperations(crossover_rate=0.8, mutation_rate=0.1)
        
        # Create test individuals
        parent1 = population_manager.create_random_individual("Parent1")
        parent2 = population_manager.create_random_individual("Parent2")
        
        # Test crossover
        child1, child2 = genetic_ops.crossover(
            parent1[0], parent2[0], 2,
            self.encoder.param_binary_encodings,
            population_manager.create_individual_name
        )
        
        # Children should be valid individuals
        TestFixtures.assert_valid_individual(child1, self.encoder)
        TestFixtures.assert_valid_individual(child2, self.encoder)
        
        # Test mutation
        mutated = genetic_ops.mutate(
            child1[0], 2,
            population_manager.create_individual_name
        )
        
        TestFixtures.assert_valid_individual(mutated, self.encoder)
        
        # Test that mutated individual can be decoded
        decoded = population_manager.decode_individual(mutated)
        self.assertIsInstance(decoded, list)
        self.assertEqual(len(decoded), 1)


class TestEvaluationIntegration(unittest.TestCase):
    """Test evaluation engine integration with other components."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.test_params = TestFixtures.get_minimal_param_values()
        self.encoder = ParameterEncoder(self.test_params)
        self.population_manager = PopulationManager(
            self.encoder.param_binary_encodings,
            population_size=6,
            nr_models=1
        )
        self.compressor = MockCompressorFactory.create_reliable_compressor(2.0)
    
    def test_evaluation_population_manager_integration(self):
        """Test evaluation engine with population manager."""
        evaluation_engine = EvaluationEngine(
            self.compressor,
            self.population_manager,
            max_threads=2,
            min_fitness=0.1
        )
        
        # Create test population
        population = self.population_manager.initialize_population()
        
        # Test parallel evaluation
        fitness_results, peak_memory = evaluation_engine.evaluate_population_parallel(population)
        
        # Validate results
        TestFixtures.assert_valid_fitness_results(fitness_results, len(population))
        self.assertGreater(peak_memory, 0)
        
        # Check that all fitness values are the expected value (2.0 from mock)
        for fitness in fitness_results:
            self.assertAlmostEqual(fitness, 2.0, places=1)
    
    def test_evaluation_error_handling_integration(self):
        """Test evaluation engine error handling integration."""
        # Use unreliable compressor
        unreliable_compressor = MockCompressorFactory.create_unreliable_compressor(0.3)
        
        evaluation_engine = EvaluationEngine(
            unreliable_compressor,
            self.population_manager,
            max_threads=2,
            min_fitness=0.1
        )
        
        population = self.population_manager.initialize_population()
        fitness_results, _ = evaluation_engine.evaluate_population_parallel(population)
        
        # Should still return results (with fallback values for failed evaluations)
        TestFixtures.assert_valid_fitness_results(fitness_results, len(population), min_fitness=0.1)
        
        # Should have recorded some errors
        self.assertGreaterEqual(evaluation_engine.stats['evaluation_errors'], 0)
    
    def test_evaluation_single_vs_parallel_consistency(self):
        """Test that single and parallel evaluation give consistent results."""
        evaluation_engine = EvaluationEngine(
            self.compressor,
            self.population_manager,
            max_threads=2,
            min_fitness=0.1
        )
        
        # Create small test population
        population = self.population_manager.initialize_population()[:4]
        
        # Single evaluation
        single_results = []
        for individual in population:
            fitness = evaluation_engine.evaluate_fitness(individual)
            single_results.append(fitness)
        
        # Reset stats
        evaluation_engine.stats['evaluations_performed'] = 0
        evaluation_engine.stats['evaluation_errors'] = 0
        
        # Parallel evaluation
        parallel_results, _ = evaluation_engine.evaluate_population_parallel(population)
        
        # Results should be similar (allowing for small variations)
        self.assertEqual(len(single_results), len(parallel_results))
        for single, parallel in zip(single_results, parallel_results):
            self.assertAlmostEqual(single, parallel, places=1)


class TestSelectionGeneticOpsIntegration(unittest.TestCase):
    """Test selection and genetic operations integration."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.test_params = TestFixtures.get_minimal_param_values()
        self.encoder = ParameterEncoder(self.test_params)
        self.population_manager = PopulationManager(
            self.encoder.param_binary_encodings,
            population_size=8,
            nr_models=1
        )
        self.selection = SelectionMethods(tournament_size=3, elite_ratio=0.2)
        self.genetic_ops = GeneticOperations(crossover_rate=0.8, mutation_rate=0.1)
    
    def test_selection_genetic_operations_workflow(self):
        """Test complete selection -> crossover -> mutation workflow."""
        # Create population with fitness values
        population = self.population_manager.initialize_population()
        fitness_values = [2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3]
        population_with_fitness = list(zip(population, fitness_values))
        
        # Sort by fitness (best first)
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Test parent selection
        parent_pairs = self.selection.select_parents(population_with_fitness, 2)
        self.assertEqual(len(parent_pairs), 2)
        
        # Test crossover and mutation on selected parents
        offspring = []
        for parent1, parent2 in parent_pairs:
            # Crossover
            child1, child2 = self.genetic_ops.crossover(
                parent1[0], parent2[0], 2,
                self.encoder.param_binary_encodings,
                self.population_manager.create_individual_name
            )
            
            # Mutation
            child1 = self.genetic_ops.mutate(
                child1[0], 2,
                self.population_manager.create_individual_name
            )
            child2 = self.genetic_ops.mutate(
                child2[0], 2,
                self.population_manager.create_individual_name
            )
            
            offspring.extend([child1, child2])
        
        # Validate offspring
        self.assertEqual(len(offspring), 4)
        for child in offspring:
            TestFixtures.assert_valid_individual(child, self.encoder)
            
            # Check child can be decoded
            decoded = self.population_manager.decode_individual(child)
            self.assertIsInstance(decoded, list)
            self.assertEqual(len(decoded), 1)
    
    def test_elitist_selection_integration(self):
        """Test elitist selection with population and offspring."""
        # Create initial population
        population = self.population_manager.initialize_population()
        population_fitness = [(ind, 2.0 - i * 0.1) for i, ind in enumerate(population)]
        
        # Create offspring
        offspring = []
        for i in range(4):
            child = self.population_manager.create_random_individual(f"Offspring_{i+1}")
            offspring.append(child)
        offspring_fitness = [(ind, 1.5 + i * 0.1) for i, ind in enumerate(offspring)]
        
        # Test elitist selection
        next_generation = self.selection.elitist_selection(
            population_fitness, offspring_fitness, 8
        )
        
        # Should return correct number of individuals
        self.assertEqual(len(next_generation), 8)
        
        # Should be sorted by fitness (best first)
        fitness_values = [fitness for _, fitness in next_generation]
        self.assertEqual(fitness_values, sorted(fitness_values, reverse=True))
        
        # All individuals should be valid
        for individual, _ in next_generation:
            TestFixtures.assert_valid_individual(individual, self.encoder)


class TestReportingIntegration(unittest.TestCase):
    """Test reporting integration with other components."""
    
    def setUp(self):
        setup_logging(level="ERROR", log_to_file=False)
        self.temp_dirs = []
    
    def tearDown(self):
        for temp_dir in self.temp_dirs:
            TestFixtures.cleanup_path(temp_dir)
    
    def test_reporter_population_manager_integration(self):
        """Test reporter integration with population manager."""
        temp_dir = TestFixtures.create_temp_test_dir()
        self.temp_dirs.append(temp_dir)
        
        # Setup components
        test_params = TestFixtures.get_minimal_param_values()
        encoder = ParameterEncoder(test_params)
        population_manager = PopulationManager(
            encoder.param_binary_encodings,
            population_size=6,
            nr_models=1
        )
        
        reporter = GAReporter(output_dir=temp_dir, experiment_name="test_integration")
        
        # Create test population with fitness
        population = population_manager.initialize_population()
        population_with_fitness = [(ind, 2.0 - i * 0.1) for i, ind in enumerate(population)]
        
        # Test generation data saving
        reporter.save_generation_data(1, population_with_fitness, population_manager)
        
        # Check files were created
        csv_files = TestFixtures.count_test_files_created(temp_dir, ".csv")
        self.assertGreater(csv_files, 0, "Should create CSV files")
        
        # Test best parameters saving
        reporter.save_best_parameters_csv(population_manager)
        
        # Should create additional files
        csv_files_after = TestFixtures.count_test_files_created(temp_dir, ".csv")
        self.assertGreater(csv_files_after, csv_files, "Should create more CSV files")


class TestConvergenceDetectionIntegration(unittest.TestCase):
    """Test convergence detection integration."""
    
    def test_convergence_with_fitness_progression(self):
        """Test convergence detection with realistic fitness progression."""
        convergence_detector = ConvergenceDetector(
            total_generations=20,
            convergence_generations=5,
            convergence_threshold=0.01
        )
        
        # Simulate fitness progression that should converge
        fitness_sequence = [1.0, 1.5, 1.8, 1.9, 1.95, 1.96, 1.965, 1.968, 1.969, 1.970]
        
        converged = False
        for gen, fitness in enumerate(fitness_sequence, 1):
            convergence_detector.add_fitness(fitness)
            is_converged, reason = convergence_detector.check_convergence(gen)
            if is_converged:
                converged = True
                break
        
        self.assertTrue(converged, "Should detect convergence in fitness sequence")
        
        # Test statistics
        stats = convergence_detector.get_statistics()
        self.assertGreater(stats['fitness_history_length'], 0)
        self.assertGreater(stats['best_fitness'], 1.0)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add component integration tests
    suite.addTest(unittest.makeSuite(TestParameterEncodingIntegration))
    suite.addTest(unittest.makeSuite(TestEvaluationIntegration))
    suite.addTest(unittest.makeSuite(TestSelectionGeneticOpsIntegration))
    suite.addTest(unittest.makeSuite(TestReportingIntegration))
    suite.addTest(unittest.makeSuite(TestConvergenceDetectionIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)