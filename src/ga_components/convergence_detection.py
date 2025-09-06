"""
Intelligent Convergence Detection System

This module implements a sophisticated convergence detection system for genetic algorithms
that prevents premature convergence while allowing natural termination when optimization stagnates.

Features:
- Minimum generation requirements to ensure adequate exploration
- Fitness improvement tracking over configurable windows
- Adaptive thresholds based on algorithm parameters
"""

from typing import List, Tuple


class ConvergenceDetector:
    """
    Intelligent convergence detection for genetic algorithms.
    
    This system prevents premature convergence by requiring minimum generations
    for exploration while detecting genuine stagnation in fitness improvement.
    """
    
    def __init__(self, total_generations: int, convergence_generations: int = 20, 
                 convergence_threshold: float = 0.001):
        """
        Initialize the convergence detection system.
        
        Args:
            total_generations: Total number of planned generations
            convergence_generations: Number of recent generations to analyze for convergence
            convergence_threshold: Minimum relative fitness improvement required
        """
        self.total_generations = total_generations
        self.convergence_generations = convergence_generations
        self.convergence_threshold = convergence_threshold
        
        # Fitness tracking
        self.fitness_history: List[float] = []
        
        # Calculate minimum generations before convergence detection activates
        self.min_generations_before_convergence = max(20, total_generations // 4)  # At least 20 or 25% of total
        
    def add_fitness(self, fitness: float) -> None:
        """
        Add a fitness value to the history.
        
        Args:
            fitness: Best fitness value from current generation
        """
        self.fitness_history.append(fitness)
    
    def check_convergence(self, current_generation: int) -> Tuple[bool, str]:
        """
        Check if the algorithm has converged based on fitness history.
        
        Args:
            current_generation: Current generation number (1-indexed)
            
        Returns:
            Tuple of (converged: bool, reason: str)
        """
        # Prevent early convergence - require minimum generations for exploration
        if current_generation < self.min_generations_before_convergence:
            return False, f"Early exploration phase (min {self.min_generations_before_convergence} generations required)"
            
        # Need sufficient fitness history
        if len(self.fitness_history) < self.convergence_generations:
            return False, f"Insufficient fitness history ({len(self.fitness_history)}/{self.convergence_generations} generations)"
        
        # Check if fitness has improved significantly in the last N generations
        recent_fitness = self.fitness_history[-self.convergence_generations:]
        best_recent = max(recent_fitness)
        worst_recent = min(recent_fitness)
        
        # Calculate relative improvement
        if best_recent > 0:
            improvement = (best_recent - worst_recent) / best_recent
        else:
            improvement = 0
        
        # Check convergence condition
        is_converged = improvement < self.convergence_threshold
        
        if is_converged:
            reason = f"Convergence detected: improvement of {improvement:.6f} below threshold {self.convergence_threshold}"
            return True, reason
        else:
            reason = f"Still improving: {improvement:.6f} > {self.convergence_threshold}"
            return False, reason
    
    def get_fitness_statistics(self) -> dict:
        """
        Get statistics about fitness progression.
        
        Returns:
            Dictionary with fitness statistics
        """
        if not self.fitness_history:
            return {
                'generations': 0,
                'best_overall': None,
                'current_fitness': None,
                'improvement_trend': None
            }
        
        return {
            'generations': len(self.fitness_history),
            'best_overall': max(self.fitness_history),
            'current_fitness': self.fitness_history[-1],
            'improvement_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> float:
        """
        Calculate the recent fitness improvement trend.
        
        Returns:
            Positive value for improving trend, negative for declining
        """
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Calculate trend over last 10 generations or available history
        window_size = min(10, len(self.fitness_history))
        recent_window = self.fitness_history[-window_size:]
        
        if len(recent_window) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x_values = list(range(len(recent_window)))
        y_values = recent_window
        
        n = len(recent_window)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Linear regression slope
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        else:
            return 0.0
    
    def reset(self):
        """Reset the convergence detector for a new run."""
        self.fitness_history = []
    
    def get_configuration(self) -> dict:
        """Get current configuration parameters."""
        return {
            'total_generations': self.total_generations,
            'convergence_generations': self.convergence_generations, 
            'convergence_threshold': self.convergence_threshold,
            'min_generations_before_convergence': self.min_generations_before_convergence
        }
    
    def _calculate_fitness_trend(self) -> float:
        """Calculate fitness trend over recent generations."""
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Calculate slope of recent fitness values
        recent_fitness = self.fitness_history[-min(5, len(self.fitness_history)):]
        n = len(recent_fitness)
        
        if n < 2:
            return 0.0
        
        x_values = list(range(n))
        y_values = recent_fitness
        
        # Linear regression slope calculation
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_statistics(self) -> dict:
        """Get convergence detection statistics."""
        return {
            'fitness_history_length': len(self.fitness_history),
            'best_fitness': max(self.fitness_history) if self.fitness_history else 0.0,
            'fitness_trend': self._calculate_fitness_trend() if len(self.fitness_history) >= 2 else 0.0,
            'generations_tracked': len(self.fitness_history),
            'convergence_window': self.convergence_generations
        }