"""
Algorithm Optimization Module

Provides advanced optimization techniques for genetic algorithms including:
- Adaptive parameter tuning based on population diversity and convergence
- Smart population initialization with diversity seeding
- Convergence acceleration through fitness landscape analysis
- Dynamic selection pressure adjustment
- Early stopping with confidence intervals
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import statistics
import math
from ga_constants import (
    GAConstants, 
    AlgorithmConstants, 
    MemoryConstants,
    bytes_to_mb
)


class AdaptiveParameterTuning:
    """
    Dynamically adjusts GA parameters based on population state and convergence metrics.
    
    Adapts mutation rates, crossover rates, and selection pressure based on:
    - Population diversity levels
    - Convergence stagnation periods
    - Fitness landscape exploration vs exploitation balance
    """
    
    def __init__(self, initial_mutation_rate: float = AlgorithmConstants.INITIAL_MUTATION_RATE, 
                 initial_crossover_rate: float = AlgorithmConstants.INITIAL_CROSSOVER_RATE):
        self.base_mutation_rate = initial_mutation_rate
        self.base_crossover_rate = initial_crossover_rate
        self.current_mutation_rate = initial_mutation_rate
        self.current_crossover_rate = initial_crossover_rate
        
        # Adaptation history
        self.diversity_history = deque(maxlen=10)
        self.fitness_stagnation_count = 0
        self.last_best_fitness = None
        self.adaptation_stats = {
            'mutation_adaptations': 0,
            'crossover_adaptations': 0,
            'diversity_boosts': 0,
            'exploration_phases': 0
        }
    
    def update_parameters(self, population_diversity: float, best_fitness: float, 
                         generation: int, max_generations: int) -> Tuple[float, float]:
        """
        Adapt mutation and crossover rates based on current state.
        
        Args:
            population_diversity: Current population diversity measure (0-1)
            best_fitness: Current best fitness value
            generation: Current generation number
            max_generations: Total generations planned
            
        Returns:
            Tuple of (adapted_mutation_rate, adapted_crossover_rate)
        """
        self.diversity_history.append(population_diversity)
        
        # Track fitness stagnation
        if self.last_best_fitness is not None:
            if abs(best_fitness - self.last_best_fitness) < 1e-6:
                self.fitness_stagnation_count += 1
            else:
                self.fitness_stagnation_count = 0
        self.last_best_fitness = best_fitness
        
        # Calculate adaptation factors
        diversity_trend = self._calculate_diversity_trend()
        progress_ratio = generation / max_generations
        stagnation_factor = min(self.fitness_stagnation_count / 5, 1.0)
        
        # Adaptive mutation rate
        if population_diversity < 0.3 or stagnation_factor > 0.6:
            # Low diversity or stagnation -> increase mutation
            diversity_boost = min(0.15, 0.05 * (1 - population_diversity))
            stagnation_boost = 0.05 * stagnation_factor
            self.current_mutation_rate = min(0.3, self.base_mutation_rate + diversity_boost + stagnation_boost)
            self.adaptation_stats['diversity_boosts'] += 1
        elif population_diversity > 0.8 and progress_ratio > 0.7:
            # High diversity in late generations -> reduce mutation for exploitation
            self.current_mutation_rate = max(0.01, self.base_mutation_rate * 0.5)
        else:
            # Gradual adaptation based on progress
            progress_factor = 1.0 + 0.5 * progress_ratio
            self.current_mutation_rate = self.base_mutation_rate * progress_factor
        
        # Adaptive crossover rate
        if diversity_trend < -0.05:  # Diversity decreasing
            # Increase crossover to promote diversity
            self.current_crossover_rate = min(GAConstants.MAX_CROSSOVER_RATE, 
                                              self.base_crossover_rate + GAConstants.CROSSOVER_RATE_BOOST)
            self.adaptation_stats['crossover_adaptations'] += 1
        elif stagnation_factor > 0.4:
            # Stagnation -> increase crossover and mutation
            self.current_crossover_rate = min(0.9, self.base_crossover_rate + 0.05)
        else:
            # Standard crossover with slight progress adjustment
            self.current_crossover_rate = self.base_crossover_rate + 0.05 * progress_ratio
        
        self.adaptation_stats['mutation_adaptations'] += 1
        
        return self.current_mutation_rate, self.current_crossover_rate
    
    def _calculate_diversity_trend(self) -> float:
        """Calculate trend in population diversity over recent generations."""
        if len(self.diversity_history) < 3:
            return 0.0
        
        recent_values = list(self.diversity_history)[-5:]  # Last 5 generations
        x_values = list(range(len(recent_values)))
        
        # Simple linear regression slope
        n = len(recent_values)
        x_mean = sum(x_values) / n
        y_mean = sum(recent_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about parameter adaptations."""
        return {
            **self.adaptation_stats,
            'current_mutation_rate': self.current_mutation_rate,
            'current_crossover_rate': self.current_crossover_rate,
            'fitness_stagnation_count': self.fitness_stagnation_count,
            'diversity_trend': self._calculate_diversity_trend(),
            'recent_diversity': list(self.diversity_history)[-3:] if self.diversity_history else []
        }


class SmartInitialization:
    """
    Advanced population initialization techniques for better genetic diversity and coverage.
    
    Uses techniques like:
    - Latin Hypercube Sampling for parameter space coverage
    - Diversity seeding with strategic parameter combinations
    - Hybrid random + structured initialization
    """
    
    @staticmethod
    def create_diverse_population(param_encodings: Dict[str, Dict], population_size: int,
                                 name_generator, diversity_strategy: str = "hybrid") -> List[Tuple]:
        """
        Create initial population with enhanced diversity characteristics.
        
        Args:
            param_encodings: Parameter binary encodings
            population_size: Target population size
            name_generator: Function to generate individual names
            diversity_strategy: Strategy ("random", "structured", "hybrid")
            
        Returns:
            List of diverse individuals
        """
        if diversity_strategy == "random":
            return SmartInitialization._random_initialization(param_encodings, population_size, name_generator)
        elif diversity_strategy == "structured":
            return SmartInitialization._structured_initialization(param_encodings, population_size, name_generator)
        else:  # hybrid
            return SmartInitialization._hybrid_initialization(param_encodings, population_size, name_generator)
    
    @staticmethod
    def _random_initialization(param_encodings: Dict[str, Dict], population_size: int, 
                             name_generator) -> List[Tuple]:
        """Standard random initialization."""
        population = []
        for i in range(population_size):
            individual_genes = []
            for param_name, encoding in param_encodings.items():
                # Random selection from available values
                random_value_idx = random.randint(0, len(encoding['values']) - 1)
                binary_repr = encoding['binary_map'][random_value_idx]
                individual_genes.append(binary_repr)
            
            individual = (tuple(individual_genes), name_generator(1))
            population.append(individual)
        
        return population
    
    @staticmethod
    def _structured_initialization(param_encodings: Dict[str, Dict], population_size: int,
                                 name_generator) -> List[Tuple]:
        """Structured initialization using Latin Hypercube-like sampling."""
        population = []
        param_names = list(param_encodings.keys())
        
        # Create stratified samples for each parameter
        stratified_samples = {}
        for param_name, encoding in param_encodings.items():
            n_values = len(encoding['values'])
            samples_per_stratum = max(1, population_size // n_values)
            
            stratified_indices = []
            for value_idx in range(n_values):
                # Add multiple samples from each stratum
                for _ in range(samples_per_stratum):
                    stratified_indices.append(value_idx)
            
            # Fill remaining slots randomly
            while len(stratified_indices) < population_size:
                stratified_indices.append(random.randint(0, n_values - 1))
            
            # Shuffle to avoid ordering bias
            random.shuffle(stratified_indices)
            stratified_samples[param_name] = stratified_indices[:population_size]
        
        # Create individuals using stratified samples
        for i in range(population_size):
            individual_genes = []
            for param_name in param_names:
                value_idx = stratified_samples[param_name][i]
                encoding = param_encodings[param_name]
                binary_repr = encoding['binary_map'][value_idx]
                individual_genes.append(binary_repr)
            
            individual = (tuple(individual_genes), name_generator(1))
            population.append(individual)
        
        return population
    
    @staticmethod
    def _hybrid_initialization(param_encodings: Dict[str, Dict], population_size: int,
                             name_generator) -> List[Tuple]:
        """Hybrid initialization combining structured and random approaches."""
        # Use structured for 70% of population
        structured_size = int(population_size * 0.7)
        structured_pop = SmartInitialization._structured_initialization(
            param_encodings, structured_size, name_generator)
        
        # Use random for remaining 30%
        random_size = population_size - structured_size
        random_pop = SmartInitialization._random_initialization(
            param_encodings, random_size, name_generator)
        
        combined_population = structured_pop + random_pop
        random.shuffle(combined_population)
        
        return combined_population


class ConvergenceAccelerator:
    """
    Techniques to accelerate convergence while maintaining solution quality.
    
    Includes:
    - Fitness landscape analysis for exploration/exploitation balance
    - Early stopping with confidence intervals
    - Dynamic population sizing based on convergence state
    - Smart restart mechanisms
    """
    
    def __init__(self, confidence_threshold: float = 0.95, min_improvement: float = 1e-6):
        self.confidence_threshold = confidence_threshold
        self.min_improvement = min_improvement
        
        self.fitness_history = deque(maxlen=20)
        self.improvement_history = deque(maxlen=10)
        self.plateau_count = 0
        self.acceleration_stats = {
            'early_stops': 0,
            'restarts_triggered': 0,
            'convergence_accelerations': 0
        }
    
    def should_stop_early(self, current_fitness: float, generation: int, 
                         max_generations: int) -> Tuple[bool, str]:
        """
        Determine if algorithm should stop early based on convergence analysis.
        
        Args:
            current_fitness: Current best fitness
            generation: Current generation
            max_generations: Maximum generations planned
            
        Returns:
            Tuple of (should_stop, reason)
        """
        self.fitness_history.append(current_fitness)
        
        # Need sufficient history
        if len(self.fitness_history) < 10:
            return False, ""
        
        # Calculate recent improvement
        recent_fitness = list(self.fitness_history)
        improvement = recent_fitness[-1] - recent_fitness[-10]
        self.improvement_history.append(improvement)
        
        # Check for plateau
        if abs(improvement) < self.min_improvement:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
        
        # Early stop conditions
        if self.plateau_count >= 8:  # Long plateau
            confidence = self._calculate_convergence_confidence()
            if confidence > self.confidence_threshold:
                self.acceleration_stats['early_stops'] += 1
                return True, f"High confidence convergence (confidence: {confidence:.3f})"
        
        # Statistical convergence test
        if len(self.improvement_history) >= 5:
            recent_improvements = list(self.improvement_history)[-5:]
            if all(abs(imp) < self.min_improvement for imp in recent_improvements):
                progress_ratio = generation / max_generations
                if progress_ratio > 0.5:  # Only in later stages
                    self.acceleration_stats['early_stops'] += 1
                    return True, "Statistical convergence detected"
        
        return False, ""
    
    def _calculate_convergence_confidence(self) -> float:
        """Calculate confidence level that algorithm has converged."""
        if len(self.fitness_history) < 5:
            return 0.0
        
        recent_fitness = list(self.fitness_history)[-10:]
        
        # Calculate variance in recent fitness values
        variance = statistics.variance(recent_fitness) if len(recent_fitness) > 1 else float('inf')
        
        # Calculate improvement trend
        improvements = [recent_fitness[i] - recent_fitness[i-1] 
                       for i in range(1, len(recent_fitness))]
        avg_improvement = statistics.mean(improvements) if improvements else 0
        
        # Confidence based on low variance and minimal improvement
        variance_factor = max(0, 1.0 - variance * 1000)  # Lower variance = higher confidence
        improvement_factor = max(0, 1.0 - abs(avg_improvement) * 100)  # Lower improvement = higher confidence
        plateau_factor = min(1.0, self.plateau_count / 10)  # Longer plateau = higher confidence
        
        confidence = (variance_factor + improvement_factor + plateau_factor) / 3
        return min(1.0, confidence)
    
    def should_restart(self, population_diversity: float, generation: int) -> Tuple[bool, str]:
        """
        Determine if algorithm should restart with new population.
        
        Args:
            population_diversity: Current population diversity (0-1)
            generation: Current generation
            
        Returns:
            Tuple of (should_restart, reason)
        """
        # Restart conditions
        if population_diversity < GAConstants.MIN_DIVERSITY_THRESHOLD and generation > GAConstants.LATE_PHASE_GENERATION_THRESHOLD:
            if self.plateau_count > 15:
                self.acceleration_stats['restarts_triggered'] += 1
                return True, "Low diversity with long plateau"
        
        return False, ""
    
    def get_acceleration_stats(self) -> Dict[str, Any]:
        """Get convergence acceleration statistics."""
        return {
            **self.acceleration_stats,
            'plateau_count': self.plateau_count,
            'convergence_confidence': self._calculate_convergence_confidence(),
            'recent_improvements': list(self.improvement_history)[-3:] if self.improvement_history else []
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring and optimization recommendations.
    
    Tracks:
    - Algorithm efficiency metrics
    - Resource utilization
    - Convergence rate analysis
    - Optimization suggestions
    """
    
    def __init__(self):
        self.generation_times = []
        self.evaluation_times = []
        self.memory_usage = []
        self.fitness_improvements = []
        self.efficiency_metrics = {
            'avg_generation_time': 0.0,
            'fitness_per_second': 0.0,
            'convergence_rate': 0.0,
            'memory_efficiency': 0.0
        }
    
    def record_generation_metrics(self, generation_time: float, evaluation_time: float,
                                memory_usage: float, fitness_improvement: float):
        """Record metrics for a generation."""
        self.generation_times.append(generation_time)
        self.evaluation_times.append(evaluation_time)
        self.memory_usage.append(memory_usage)
        self.fitness_improvements.append(fitness_improvement)
        
        self._update_efficiency_metrics()
    
    def _update_efficiency_metrics(self):
        """Update efficiency metrics based on recorded data."""
        if not self.generation_times:
            return
        
        # Average generation time
        self.efficiency_metrics['avg_generation_time'] = statistics.mean(self.generation_times[-10:])
        
        # Fitness improvement per second
        recent_improvements = sum(self.fitness_improvements[-10:])
        recent_time = sum(self.generation_times[-10:])
        if recent_time > 0:
            self.efficiency_metrics['fitness_per_second'] = recent_improvements / recent_time
        
        # Convergence rate (improvement per generation)
        if len(self.fitness_improvements) >= 5:
            recent_improvements = self.fitness_improvements[-5:]
            self.efficiency_metrics['convergence_rate'] = statistics.mean(recent_improvements)
        
        # Memory efficiency (fitness per MB)
        if self.memory_usage and self.fitness_improvements:
            avg_memory = bytes_to_mb(statistics.mean(self.memory_usage[-MemoryConstants.MEMORY_HISTORY_SIZE:]))  # Convert to MB
            total_improvement = sum(self.fitness_improvements[-10:])
            if avg_memory > 0:
                self.efficiency_metrics['memory_efficiency'] = total_improvement / avg_memory
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if not self.generation_times:
            return recommendations
        
        # Analyze generation time trends
        if len(self.generation_times) >= 5:
            recent_times = self.generation_times[-5:]
            if statistics.mean(recent_times) > 10.0:  # Slow generations
                recommendations.append("Consider reducing population size or using more threads")
        
        # Analyze convergence efficiency
        if self.efficiency_metrics['convergence_rate'] < 0.01:
            recommendations.append("Low convergence rate - consider adaptive parameter tuning")
        
        # Memory usage analysis
        if self.memory_usage and statistics.mean(self.memory_usage[-MemoryConstants.MEMORY_WARNING_SIZE:]) > GAConstants.MEMORY_WARNING_THRESHOLD_MB * MemoryConstants.BYTES_PER_MB:
            recommendations.append("High memory usage - consider reducing population size")
        
        # Fitness improvement analysis
        if len(self.fitness_improvements) >= 10:
            recent_improvements = self.fitness_improvements[-10:]
            if all(imp < 1e-6 for imp in recent_improvements[-5:]):
                recommendations.append("Fitness stagnation detected - consider restart or parameter adaptation")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            **self.efficiency_metrics,
            'total_generations_tracked': len(self.generation_times),
            'optimization_recommendations': self.get_optimization_recommendations(),
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate overall performance trend."""
        if len(self.generation_times) < 5:
            return "insufficient_data"
        
        recent_times = self.generation_times[-5:]
        early_times = self.generation_times[:5]
        
        recent_avg = statistics.mean(recent_times)
        early_avg = statistics.mean(early_times)
        
        if recent_avg < early_avg * 0.9:
            return "improving"
        elif recent_avg > early_avg * 1.1:
            return "degrading"
        else:
            return "stable"