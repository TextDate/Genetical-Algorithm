"""
Tests for the Adaptive Hybrid Duplicate Prevention System

Comprehensive tests covering all phases, edge cases, and performance characteristics
of the duplicate prevention system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock
from ga_components.duplicate_prevention import DuplicatePreventionSystem


class TestDuplicatePreventionSystem(unittest.TestCase):
    """Test suite for the DuplicatePreventionSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = DuplicatePreventionSystem(
            total_parameter_combinations=1000,
            population_size=50,
            total_generations=100,
            enabled=True
        )
        
        # Mock parameter encodings
        self.param_encodings = {
            'param1': {
                'values': [0, 1, 2, 3],
                'binary_map': {0: '00', 1: '01', 2: '10', 3: '11'},
                'bit_length': 2
            },
            'param2': {
                'values': ['a', 'b', 'c', 'd'],
                'binary_map': {0: '00', 1: '01', 2: '10', 3: '11'},
                'bit_length': 2
            }
        }
        
        # Mock individual name generator
        self.name_counter = 0
        def name_generator(gen):
            self.name_counter += 1
            return f"Gen{gen}_Ind{self.name_counter}"
        
        self.name_generator = name_generator
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.system.total_parameter_combinations, 1000)
        self.assertEqual(self.system.population_size, 50)
        self.assertEqual(self.system.total_generations, 100)
        self.assertTrue(self.system.enabled)
        
    def test_parameter_space_density_calculation(self):
        """Test density calculation across generations."""
        # Early generation - full density adjustment
        density_early = self.system.calculate_parameter_space_density(1)
        
        # Late generation - reduced density adjustment  
        density_late = self.system.calculate_parameter_space_density(100)
        
        # Early should be higher than late (more strict enforcement needed)
        self.assertGreater(density_early, density_late)
        
    def test_adaptive_thresholds_phases(self):
        """Test phase transitions across generations."""
        # Early phase (strict)
        phase, prob = self.system.get_adaptive_thresholds(5)
        self.assertEqual(phase, "strict")
        self.assertAlmostEqual(prob, 0.9, places=1)
        
        # Middle phase (gradual)  
        phase, prob = self.system.get_adaptive_thresholds(50)
        self.assertEqual(phase, "gradual")
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 0.9)
        
        # Late phase (none)
        phase, prob = self.system.get_adaptive_thresholds(85)
        self.assertEqual(phase, "none")
        self.assertEqual(prob, 0.0)
    
    def test_individual_signature_generation(self):
        """Test individual signature creation."""
        individual1 = (("0011", "1100"), "Gen1_Ind1")
        individual2 = (("0011", "1100"), "Gen1_Ind2")  # Same genes, different name
        individual3 = (("1100", "0011"), "Gen1_Ind3")  # Different genes
        
        sig1 = self.system.individual_to_signature(individual1)
        sig2 = self.system.individual_to_signature(individual2)
        sig3 = self.system.individual_to_signature(individual3)
        
        # Same genes should produce same signature regardless of name
        self.assertEqual(sig1, sig2)
        # Different genes should produce different signatures
        self.assertNotEqual(sig1, sig3)
    
    def test_enforce_population_diversity_strict_phase(self):
        """Test diversity enforcement in strict phase."""
        # Create population with duplicates
        population = [
            (("0011",), "Gen1_Ind1"),
            (("1100",), "Gen1_Ind2"), 
            (("0011",), "Gen1_Ind3"),  # Duplicate
            (("1010",), "Gen1_Ind4")
        ]
        
        result = self.system.enforce_population_diversity(
            population, 5, self.param_encodings, 1, self.name_generator, verbose=False)
        
        # Should have same length
        self.assertEqual(len(result), len(population))
        
        # Should have unique signatures
        signatures = [self.system.individual_to_signature(ind) for ind in result]
        self.assertEqual(len(signatures), len(set(signatures)))
    
    def test_enforce_population_diversity_none_phase(self):
        """Test diversity enforcement in none phase (late generation)."""
        # Create population with duplicates
        population = [
            (("0011",), "Gen85_Ind1"),
            (("1100",), "Gen85_Ind2"), 
            (("0011",), "Gen85_Ind3"),  # Duplicate
            (("1010",), "Gen85_Ind4")
        ]
        
        result = self.system.enforce_population_diversity(
            population, 85, self.param_encodings, 1, self.name_generator, verbose=False)
        
        # Should have same length
        self.assertEqual(len(result), len(population))
        
        # May have duplicates (allowed in none phase)
        signatures = [self.system.individual_to_signature(ind) for ind in result]
        # Don't enforce uniqueness in none phase
    
    def test_statistics_tracking(self):
        """Test statistics collection."""
        initial_stats = self.system.get_statistics()
        self.assertEqual(initial_stats['total_duplicates'], 0)
        
        # Process population with duplicates
        population = [
            (("0011",), "Gen1_Ind1"),
            (("0011",), "Gen1_Ind2"),  # Duplicate
            (("1100",), "Gen1_Ind3")
        ]
        
        self.system.enforce_population_diversity(
            population, 5, self.param_encodings, 1, self.name_generator, verbose=False)
        
        stats = self.system.get_statistics()
        self.assertGreater(stats['total_duplicates'], 0)
    
    def test_disabled_system(self):
        """Test system behavior when disabled."""
        disabled_system = DuplicatePreventionSystem(1000, 50, 100, enabled=False)
        
        population = [
            (("0011",), "Gen1_Ind1"),
            (("0011",), "Gen1_Ind2"),  # Duplicate
        ]
        
        result = disabled_system.enforce_population_diversity(
            population, 5, self.param_encodings, 1, self.name_generator, verbose=False)
        
        # Should return unchanged population
        self.assertEqual(result, population)


if __name__ == '__main__':
    unittest.main()