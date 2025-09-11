"""
Multi-Objective Evaluation Module

Combines compression fitness, timing, and RAM usage into a weighted score with extensibility
for additional objectives. Normalizes metrics and provides configurable weighting.

Features:
- Weighted combination of fitness, compression time, and RAM usage
- Time and RAM normalization to [0,1] range for fair comparison  
- Extensible framework for adding new objectives
- Configurable penalties and thresholds
- Statistical tracking of objective components
"""

import statistics
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from ga_logging import get_logger


@dataclass
class ObjectiveWeights:
    """Configuration for multi-objective evaluation weights."""
    fitness_weight: float = 0.5
    time_weight: float = 0.3
    ram_weight: float = 0.2  # Direct RAM weight for 3-tuple system
    additional_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_weights is None:
            self.additional_weights = {}
        
        # Validate weights sum to 1.0 (includes RAM weight now)
        total_weight = self.fitness_weight + self.time_weight + self.ram_weight + sum(self.additional_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"All weights must sum to 1.0, got {total_weight:.3f}")


@dataclass  
class EvaluationMetrics:
    """Container for multi-objective evaluation results."""
    combined_score: float
    fitness_component: float
    time_component: float
    ram_component: float  # Add RAM component
    normalized_time: float
    normalized_ram: float  # Add normalized RAM
    raw_fitness: float
    raw_time: float
    raw_ram: float  # Add raw RAM usage
    additional_components: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_components is None:
            self.additional_components = {}


class MultiObjectiveEvaluator:
    """
    Multi-objective evaluator that combines fitness, time, and additional objectives
    into a weighted combined score for genetic algorithm optimization.
    """
    
    def __init__(self, weights: ObjectiveWeights, normalize_time: bool = True, 
                 enable_time_penalty: bool = False, time_penalty_threshold: float = 10.0):
        """
        Initialize multi-objective evaluator.
        
        Args:
            weights: ObjectiveWeights configuration
            normalize_time: Whether to normalize compression times to [0,1]
            enable_time_penalty: Whether to apply penalties for excessive times
            time_penalty_threshold: Time threshold for applying penalties (seconds)
        """
        self.weights = weights
        self.normalize_time = normalize_time
        self.enable_time_penalty = enable_time_penalty
        self.time_penalty_threshold = time_penalty_threshold
        self.logger = get_logger("MultiObjectiveEvaluator")
        
        # Time normalization statistics (updated per generation)
        self.time_stats = {
            'min_time': float('inf'),
            'max_time': 0.0,
            'mean_time': 0.0,
            'std_time': 0.0
        }
        
        # RAM normalization statistics (updated per generation)
        self.ram_stats = {
            'min_ram': float('inf'),
            'max_ram': 0.0,
            'mean_ram': 0.0,
            'std_ram': 0.0
        }
        
        # Evaluation statistics
        self.stats = {
            'evaluations_performed': 0,
            'time_penalties_applied': 0,
            'avg_fitness_component': 0.0,
            'avg_time_component': 0.0,
            'avg_ram_component': 0.0
        }
    
    def update_time_statistics(self, compression_times: List[float]) -> None:
        """
        Update time normalization statistics based on current generation times.
        
        Args:
            compression_times: List of compression times from current generation
        """
        if not compression_times or all(t == 0.0 for t in compression_times):
            self.logger.warning("No valid compression times for normalization")
            return
        
        valid_times = [t for t in compression_times if t > 0.0]
        if not valid_times:
            return
            
        self.time_stats['min_time'] = min(valid_times)
        self.time_stats['max_time'] = max(valid_times)
        self.time_stats['mean_time'] = statistics.mean(valid_times)
        
        if len(valid_times) > 1:
            self.time_stats['std_time'] = statistics.stdev(valid_times)
        else:
            self.time_stats['std_time'] = 0.0
        
        self.logger.debug("Time statistics updated",
                         min_time=self.time_stats['min_time'],
                         max_time=self.time_stats['max_time'],
                         mean_time=self.time_stats['mean_time'])
    
    def normalize_time_value(self, time_value: float) -> float:
        """
        Normalize a time value to [0,1] range based on current statistics.
        
        Args:
            time_value: Raw compression time in seconds
            
        Returns:
            Normalized time value in [0,1] range (lower is better)
        """
        if not self.normalize_time or time_value <= 0.0:
            return 0.0
        
        min_time = self.time_stats['min_time']
        max_time = self.time_stats['max_time']
        
        if min_time == max_time:
            return 0.0  # All times are equal
        
        # Normalize to [0,1] where 0 is fastest, 1 is slowest
        normalized = (time_value - min_time) / (max_time - min_time)
        return max(0.0, min(1.0, normalized))
    
    def update_ram_statistics(self, ram_usage_values: List[float]) -> None:
        """
        Update RAM normalization statistics based on current generation RAM usage.
        
        Args:
            ram_usage_values: List of RAM usage values from current generation (in MB)
        """
        if not ram_usage_values or all(r == 0.0 for r in ram_usage_values):
            self.logger.warning("No valid RAM usage values for normalization")
            return
        
        valid_rams = [r for r in ram_usage_values if r > 0.0]
        if not valid_rams:
            return
            
        self.ram_stats['min_ram'] = min(valid_rams)
        self.ram_stats['max_ram'] = max(valid_rams)
        self.ram_stats['mean_ram'] = statistics.mean(valid_rams)
        
        if len(valid_rams) > 1:
            self.ram_stats['std_ram'] = statistics.stdev(valid_rams)
        else:
            self.ram_stats['std_ram'] = 0.0
        
        self.logger.debug("RAM statistics updated",
                         min_ram=self.ram_stats['min_ram'],
                         max_ram=self.ram_stats['max_ram'],
                         mean_ram=self.ram_stats['mean_ram'])
    
    def normalize_ram_value(self, ram_value: float) -> float:
        """
        Normalize a RAM value to [0,1] range based on current statistics.
        
        Args:
            ram_value: Raw RAM usage in MB
            
        Returns:
            Normalized RAM value in [0,1] range (lower is better)
        """
        if ram_value <= 0.0:
            return 0.0
        
        min_ram = self.ram_stats['min_ram']
        max_ram = self.ram_stats['max_ram']
        
        if min_ram == max_ram:
            return 0.0  # All RAM usage is equal
        
        # Normalize to [0,1] where 0 is lowest RAM, 1 is highest RAM
        normalized = (ram_value - min_ram) / (max_ram - min_ram)
        return max(0.0, min(1.0, normalized))
    
    def apply_time_penalty(self, time_value: float, normalized_time: float) -> float:
        """
        Apply penalty for excessive compression times (if enabled).
        
        Args:
            time_value: Raw compression time in seconds
            normalized_time: Normalized time component
            
        Returns:
            Time component with penalty applied if necessary and enabled
        """
        if not self.enable_time_penalty:
            return normalized_time
            
        if time_value > self.time_penalty_threshold:
            # Apply exponential penalty for times above threshold
            penalty_factor = 1.0 + (time_value - self.time_penalty_threshold) / self.time_penalty_threshold
            penalty_component = min(1.0, normalized_time * penalty_factor)
            self.stats['time_penalties_applied'] += 1
            return penalty_component
        return normalized_time
    
    def evaluate_individual(self, fitness: float, compression_time: float, ram_usage: float = 0.0,
                          additional_objectives: Optional[Dict[str, float]] = None) -> EvaluationMetrics:
        """
        Evaluate a single individual using multi-objective scoring.
        
        Args:
            fitness: Compression ratio (higher is better)
            compression_time: Compression time in seconds (lower is better)
            ram_usage: RAM usage in MB (lower is better)
            additional_objectives: Optional dictionary of additional objective values
            
        Returns:
            EvaluationMetrics containing combined score and components
        """
        if additional_objectives is None:
            additional_objectives = {}
        
        # Normalize time component (0 = fastest, 1 = slowest)
        normalized_time = self.normalize_time_value(compression_time)
        
        # Apply time penalty if necessary
        time_component = self.apply_time_penalty(compression_time, normalized_time)
        
        # Normalize RAM component (0 = lowest RAM, 1 = highest RAM)
        normalized_ram = self.normalize_ram_value(ram_usage)
        ram_component = normalized_ram
        
        # Fitness component (higher is better, so use directly)
        fitness_component = max(0.0, fitness)
        
        # Calculate additional objective components (excluding RAM which is now primary)
        additional_components = {}
        additional_score = 0.0
        
        for obj_name, obj_value in additional_objectives.items():
            if obj_name in self.weights.additional_weights and obj_name != 'ram':  # Skip RAM, it's handled as primary
                weight = self.weights.additional_weights[obj_name]
                obj_component = max(0.0, obj_value)
                additional_components[obj_name] = obj_component
                # Additional objectives are maximized
                additional_score += weight * obj_component
        
        # Combine objectives using weighted sum
        # Note: time_component and ram_component are inverted (1 - component) because lower values are better
        combined_score = (
            self.weights.fitness_weight * fitness_component +
            self.weights.time_weight * (1.0 - time_component) +
            self.weights.ram_weight * (1.0 - ram_component) +
            additional_score
        )
        
        # Update statistics
        self.stats['evaluations_performed'] += 1
        self.stats['avg_fitness_component'] = (
            (self.stats['avg_fitness_component'] * (self.stats['evaluations_performed'] - 1) + fitness_component) /
            self.stats['evaluations_performed']
        )
        self.stats['avg_time_component'] = (
            (self.stats['avg_time_component'] * (self.stats['evaluations_performed'] - 1) + time_component) /
            self.stats['evaluations_performed']
        )
        self.stats['avg_ram_component'] = (
            (self.stats['avg_ram_component'] * (self.stats['evaluations_performed'] - 1) + ram_component) /
            self.stats['evaluations_performed']
        )
        
        return EvaluationMetrics(
            combined_score=combined_score,
            fitness_component=fitness_component,
            time_component=time_component,
            ram_component=ram_component,
            normalized_time=normalized_time,
            normalized_ram=normalized_ram,
            raw_fitness=fitness,
            raw_time=compression_time,
            raw_ram=ram_usage,
            additional_components=additional_components
        )
    
    def evaluate_population(self, fitness_values: List[float], 
                          compression_times: List[float],
                          ram_usage_values: List[float],
                          additional_objectives: Optional[List[Dict[str, float]]] = None) -> List[EvaluationMetrics]:
        """
        Evaluate entire population using multi-objective scoring.
        
        Args:
            fitness_values: List of compression ratios
            compression_times: List of compression times
            ram_usage_values: List of RAM usage values (in MB)
            additional_objectives: Optional list of additional objective dictionaries
            
        Returns:
            List of EvaluationMetrics for each individual
        """
        if len(fitness_values) != len(compression_times) or len(fitness_values) != len(ram_usage_values):
            raise ValueError("Fitness values, compression times, and RAM usage values must have same length")
        
        # Update time and RAM normalization statistics
        self.update_time_statistics(compression_times)
        self.update_ram_statistics(ram_usage_values)
        
        # Evaluate each individual
        results = []
        for i, (fitness, time_val, ram_val) in enumerate(zip(fitness_values, compression_times, ram_usage_values)):
            additional_objs = additional_objectives[i] if additional_objectives else None
            metrics = self.evaluate_individual(fitness, time_val, ram_val, additional_objs)
            results.append(metrics)
        
        self.logger.debug(f"Evaluated population of {len(results)} individuals",
                         avg_combined_score=statistics.mean(r.combined_score for r in results),
                         avg_fitness_component=self.stats['avg_fitness_component'],
                         avg_time_component=self.stats['avg_time_component'],
                         avg_ram_component=self.stats['avg_ram_component'])
        
        return results
    
    def get_combined_scores(self, evaluation_results: List[EvaluationMetrics]) -> List[float]:
        """Extract combined scores from evaluation results."""
        return [result.combined_score for result in evaluation_results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        stats = self.stats.copy()
        stats.update({
            'time_normalization': self.time_stats.copy(),
            'ram_normalization': self.ram_stats.copy(),
            'weights': {
                'fitness_weight': self.weights.fitness_weight,
                'time_weight': self.weights.time_weight,
                'ram_weight': self.weights.ram_weight,
                'additional_weights': self.weights.additional_weights.copy()
            },
            'configuration': {
                'normalize_time': self.normalize_time,
                'enable_time_penalty': self.enable_time_penalty,
                'time_penalty_threshold': self.time_penalty_threshold
            }
        })
        return stats
    
    def reset_statistics(self) -> None:
        """Reset evaluation statistics."""
        self.stats = {
            'evaluations_performed': 0,
            'time_penalties_applied': 0,
            'avg_fitness_component': 0.0,
            'avg_time_component': 0.0,
            'avg_ram_component': 0.0
        }
        self.time_stats = {
            'min_time': float('inf'),
            'max_time': 0.0,
            'mean_time': 0.0,
            'std_time': 0.0
        }
        self.ram_stats = {
            'min_ram': float('inf'),
            'max_ram': 0.0,
            'mean_ram': 0.0,
            'std_ram': 0.0
        }