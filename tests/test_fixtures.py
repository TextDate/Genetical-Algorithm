"""
Test Fixtures and Utilities for GA Integration Tests

Provides reusable test data, mock objects, and helper functions
for comprehensive GA component and integration testing.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_config import GAConfig
from ga_components.parameter_encoding import ParameterEncoder


class TestFixtures:
    """Centralized test fixtures and utilities."""
    
    @staticmethod
    def get_minimal_param_values() -> Dict[str, List[Any]]:
        """Get minimal parameter set for fast testing."""
        return {
            'level': [1, 3, 6],
            'window_log': [10, 12, 14],
            'strategy': [0, 1, 2]
        }
    
    @staticmethod
    def get_comprehensive_param_values() -> Dict[str, List[Any]]:
        """Get comprehensive parameter set for thorough testing."""
        return {
            'level': [-5, -1, 1, 3, 6, 9, 12],
            'window_log': [10, 12, 14, 16, 18, 20],
            'chain_log': [6, 8, 10, 12, 14],
            'hash_log': [6, 8, 10, 12, 14, 16],
            'search_log': [1, 2, 3, 4, 5],
            'min_match': [3, 4, 5, 6],
            'target_length': [4096, 8192, 16384, 32768],
            'strategy': [0, 1, 2, 3, 4, 5]
        }
    
    @staticmethod 
    def get_test_config(population_size: int = 10, generations: int = 2,
                       max_threads: int = 2) -> GAConfig:
        """Get test configuration with sensible defaults."""
        return GAConfig(
            population_size=population_size,
            generations=generations,
            mutation_rate=0.05,
            crossover_rate=0.8,
            max_threads=max_threads,
            output_dir="test_output"
        )
    
    @staticmethod
    def create_mock_compressor(fitness_values: List[float] = None,
                              compressor_type: str = "ZstdCompressor",
                              nr_models: int = 1) -> Mock:
        """Create mock compressor for testing."""
        mock = Mock()
        mock.nr_models = nr_models
        mock.__class__.__name__ = compressor_type
        
        # Set up evaluate method - return tuple (fitness, compression_time)
        if fitness_values:
            # Convert fitness values to tuples if needed
            tuple_values = []
            for value in fitness_values:
                if isinstance(value, tuple):
                    tuple_values.append(value)
                else:
                    tuple_values.append((value, 0.1))  # Add mock timing
            mock.evaluate.side_effect = tuple_values
        else:
            mock.evaluate.return_value = (2.5, 0.1)  # Default fitness with timing
            
        # Set up other methods
        mock.erase_temp_files.return_value = None
        
        # Create a temporary test input file
        temp_file = TestFixtures.create_temp_test_file()
        mock.input_file_path = temp_file
        mock.temp = "test_temp"
        
        return mock
    
    @staticmethod
    def create_test_population(param_encoder: ParameterEncoder,
                              size: int = 10) -> List[Tuple[Tuple[str, ...], str]]:
        """Create test population with encoded individuals."""
        population = []
        for i in range(size):
            # Create random individual
            individual_data = param_encoder.create_random_individual()
            individual_name = f"Test_Ind_{i+1}"
            population.append((individual_data, individual_name))
        return population
    
    @staticmethod
    def create_temp_test_file(content: str = "Test file content\n" * 100) -> str:
        """Create temporary test file and return path."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def create_temp_test_dir() -> str:
        """Create temporary test directory and return path."""
        return tempfile.mkdtemp(prefix='ga_test_')
    
    @staticmethod
    def cleanup_path(path: str):
        """Clean up test files or directories."""
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except (OSError, PermissionError):
            pass  # Ignore cleanup errors
    
    @staticmethod
    def assert_valid_individual(individual: Tuple[Tuple[str, ...], str],
                               param_encoder: ParameterEncoder):
        """Assert that individual is properly formatted."""
        assert isinstance(individual, tuple), "Individual must be tuple"
        assert len(individual) == 2, "Individual must have (gene_code, name)"
        
        gene_code, name = individual
        assert isinstance(gene_code, tuple), "Gene code must be tuple"
        assert isinstance(name, str), "Individual name must be string"
        assert len(gene_code) == len(param_encoder.param_binary_encodings), \
            "Gene code length must match parameter count"
    
    @staticmethod
    def assert_valid_fitness_results(fitness_results: List[float],
                                   expected_count: int,
                                   min_fitness: float = 0.0):
        """Assert fitness results are valid."""
        assert isinstance(fitness_results, list), "Fitness results must be list"
        assert len(fitness_results) == expected_count, \
            f"Expected {expected_count} fitness values, got {len(fitness_results)}"
        
        for i, fitness in enumerate(fitness_results):
            assert isinstance(fitness, (int, float)), \
                f"Fitness {i} must be numeric, got {type(fitness)}"
            assert fitness >= min_fitness, \
                f"Fitness {i} ({fitness}) below minimum ({min_fitness})"
    
    @staticmethod
    def count_test_files_created(base_dir: str, pattern: str = "") -> int:
        """Count test files created in directory."""
        if not os.path.exists(base_dir):
            return 0
        
        count = 0
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if pattern in file:
                    count += 1
        return count


class MockCompressorFactory:
    """Factory for creating different types of mock compressors."""
    
    @staticmethod
    def create_reliable_compressor(base_fitness: float = 2.0) -> Mock:
        """Create compressor that always returns valid fitness."""
        mock = TestFixtures.create_mock_compressor()
        mock.evaluate.return_value = (base_fitness, 0.1)  # Return tuple (fitness, time)
        return mock
    
    @staticmethod 
    def create_unreliable_compressor(failure_rate: float = 0.3) -> Mock:
        """Create compressor that fails some percentage of evaluations."""
        mock = TestFixtures.create_mock_compressor()
        
        def evaluate_with_failures(*args, **kwargs):
            import random
            if random.random() < failure_rate:
                raise Exception("Simulated compression failure")
            return 2.0
        
        mock.evaluate.side_effect = evaluate_with_failures
        return mock
    
    @staticmethod
    def create_timeout_compressor(timeout_rate: float = 0.2) -> Mock:
        """Create compressor that times out some percentage of time."""
        mock = TestFixtures.create_mock_compressor()
        
        def evaluate_with_timeouts(*args, **kwargs):
            import random, time
            if random.random() < timeout_rate:
                time.sleep(0.1)  # Simulate timeout
                raise Exception("Compression timed out")
            return 1.5
        
        mock.evaluate.side_effect = evaluate_with_timeouts
        return mock
    
    @staticmethod
    def create_varying_fitness_compressor(fitness_range: Tuple[float, float] = (1.0, 3.0)) -> Mock:
        """Create compressor with realistic fitness variation."""
        mock = TestFixtures.create_mock_compressor()
        
        def evaluate_with_variation(*args, **kwargs):
            import random
            return random.uniform(fitness_range[0], fitness_range[1])
        
        mock.evaluate.side_effect = evaluate_with_variation
        return mock


class TestDataBuilder:
    """Builder pattern for creating complex test data."""
    
    def __init__(self):
        self.config = TestFixtures.get_test_config()
        self.param_values = TestFixtures.get_minimal_param_values()
        self.compressor = TestFixtures.create_mock_compressor()
        self.temp_files = []
        self.temp_dirs = []
    
    def with_config(self, **config_updates) -> 'TestDataBuilder':
        """Update configuration parameters."""
        current_config = self.config.to_dict()
        current_config.update(config_updates)
        self.config = GAConfig.from_dict(current_config)
        return self
    
    def with_parameters(self, param_values: Dict[str, List[Any]]) -> 'TestDataBuilder':
        """Set parameter values."""
        self.param_values = param_values
        return self
    
    def with_compressor(self, compressor: Mock) -> 'TestDataBuilder':
        """Set mock compressor."""
        self.compressor = compressor
        return self
    
    def with_temp_input_file(self, content: str = None) -> 'TestDataBuilder':
        """Create temporary input file."""
        temp_file = TestFixtures.create_temp_test_file(content)
        self.temp_files.append(temp_file)
        self.compressor.input_file_path = temp_file
        return self
    
    def with_temp_output_dir(self) -> 'TestDataBuilder':
        """Create temporary output directory."""
        temp_dir = TestFixtures.create_temp_test_dir()
        self.temp_dirs.append(temp_dir)
        self.config = self.config.update(output_dir=temp_dir)
        return self
    
    def build(self) -> Tuple[GAConfig, Dict[str, List[Any]], Mock]:
        """Build the test data."""
        return self.config, self.param_values, self.compressor
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        for temp_file in self.temp_files:
            TestFixtures.cleanup_path(temp_file)
        for temp_dir in self.temp_dirs:
            TestFixtures.cleanup_path(temp_dir)
        self.temp_files.clear()
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Global test constants
TEST_TIMEOUT = 30  # seconds
MIN_TEST_FITNESS = 0.1
MAX_TEST_FITNESS = 10.0
DEFAULT_TEST_POPULATION_SIZE = 8
DEFAULT_TEST_GENERATIONS = 3