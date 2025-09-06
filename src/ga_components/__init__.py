"""
GA Components Module

Modular components for the Genetic Algorithm implementation.
Each component handles a specific aspect of the GA process:

- ParameterEncoder: Handles binary encoding/decoding of parameters
- PopulationManager: Manages population initialization and operations
- SelectionMethods: Implements selection strategies (tournament, elitist)
- GeneticOperations: Crossover and mutation operations
- EvaluationEngine: Fitness evaluation with parallel processing
- DuplicatePreventionSystem: Maintains population diversity
- ConvergenceDetector: Monitors convergence and stopping criteria
- GAReporter: Generates reports and saves results
- AlgorithmOptimization: Advanced optimization techniques

Usage:
    from ga_components import ParameterEncoder, PopulationManager
    from ga_components.algorithm_optimization import AdaptiveParameterTuning
"""

# Core GA components
from .parameter_encoding import ParameterEncoder
from .population_management import PopulationManager
from .selection import SelectionMethods
from .genetic_operations import GeneticOperations
from .evaluation import EvaluationEngine
from .duplicate_prevention import DuplicatePreventionSystem
from .convergence_detection import ConvergenceDetector
from .reporting import GAReporter

# Optimization components
from .algorithm_optimization import (
    AdaptiveParameterTuning,
    SmartInitialization,
    ConvergenceAccelerator,
    PerformanceMonitor
)

__all__ = [
    # Core components
    'ParameterEncoder',
    'PopulationManager', 
    'SelectionMethods',
    'GeneticOperations',
    'EvaluationEngine',
    'DuplicatePreventionSystem',
    'ConvergenceDetector',
    'GAReporter',
    
    # Optimization components
    'AdaptiveParameterTuning',
    'SmartInitialization', 
    'ConvergenceAccelerator',
    'PerformanceMonitor'
]

# Version information
__version__ = '2.0.0'
__author__ = 'Genetic Algorithm Team'
__description__ = 'Modular Genetic Algorithm for Parameter Optimization'