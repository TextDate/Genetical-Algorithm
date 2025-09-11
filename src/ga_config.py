"""
Configuration Management for Genetic Algorithm

Validates and organizes user-provided CLI parameters into a clean structure.
Replaces the 12+ parameter constructor with a single config object.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict
from ga_constants import GAConstants


@dataclass
class GAConfig:
    """
    Configuration container that validates and organizes user-provided parameters.
    
    Takes CLI arguments and organizes them into a validated structure,
    keeping all user control while improving code organization.
    """
    
    # Core GA Parameters (user-provided via CLI)
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    max_threads: int
    output_dir: str
    
    # Advanced GA Parameters (with sensible internal defaults)
    offspring_ratio: float = 0.9
    elite_ratio: float = 0.1
    tournament_size: int = 3
    min_fitness: float = GAConstants.MIN_FITNESS
    convergence_generations: int = 20
    convergence_threshold: float = 0.001
    
    # Algorithm Optimization Features
    enable_adaptive_tuning: bool = True
    enable_smart_initialization: bool = False
    enable_convergence_acceleration: bool = True
    enable_performance_monitoring: bool = True
    enable_dynamic_thread_scaling: bool = False
    diversity_strategy: str = "hybrid"  # "random", "structured", "hybrid"
    
    # Multi-Objective Evaluation Configuration
    enable_multi_objective: bool = True
    fitness_weight: float = 0.5          # Weight for compression ratio (0-1)
    time_weight: float = 0.3             # Weight for compression time (0-1)
    ram_weight: float = 0.2              # Weight for RAM usage (0-1)
    normalize_time: bool = True          # Normalize time values to [0,1] range
    enable_time_penalty: bool = True     # Enable/disable time penalty system
    time_penalty_threshold: float = 10.0 # Time threshold in seconds for penalty
    additional_objectives: Dict[str, float] = None  # For future extensibility
    
    def __post_init__(self):
        """Validate user parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate critical user parameters to catch errors early."""
        errors = []
        
        # Core parameter validation
        if self.population_size < 4:
            errors.append(f"Population size ({self.population_size}) must be at least 4")
        if self.population_size % 2 != 0:
            errors.append(f"Population size ({self.population_size}) must be even for crossover")
        if self.generations < 1:
            errors.append(f"Generations ({self.generations}) must be positive")
            
        # Rate validation
        if not 0.0 <= self.mutation_rate <= 1.0:
            errors.append(f"Mutation rate ({self.mutation_rate}) must be between 0.0 and 1.0")
        if not 0.0 <= self.crossover_rate <= 1.0:
            errors.append(f"Crossover rate ({self.crossover_rate}) must be between 0.0 and 1.0")
        
        # Performance validation
        if self.max_threads < 1:
            errors.append(f"Max threads ({self.max_threads}) must be positive")
        if self.max_threads > os.cpu_count() * 4:  # Allow some flexibility
            print(f"Warning: max_threads ({self.max_threads}) is high for {os.cpu_count()} CPU cores")
            
        # Tournament validation
        if self.tournament_size > self.population_size:
            errors.append(f"Tournament size ({self.tournament_size}) cannot exceed population size ({self.population_size})")
        if self.tournament_size < 2:
            errors.append(f"Tournament size ({self.tournament_size}) must be at least 2")
            
        # Directory validation
        if not self.output_dir or not self.output_dir.strip():
            errors.append("Output directory cannot be empty")
        
        # Multi-objective evaluation validation
        if self.enable_multi_objective:
            if not 0.0 <= self.fitness_weight <= 1.0:
                errors.append(f"Fitness weight ({self.fitness_weight}) must be between 0.0 and 1.0")
            if not 0.0 <= self.time_weight <= 1.0:
                errors.append(f"Time weight ({self.time_weight}) must be between 0.0 and 1.0")
            if not 0.0 <= self.ram_weight <= 1.0:
                errors.append(f"RAM weight ({self.ram_weight}) must be between 0.0 and 1.0")
            
            total_weight = self.fitness_weight + self.time_weight + self.ram_weight
            if abs(total_weight - 1.0) > 0.001:
                # Check if user is using old default weights (0.7 + 0.3) and auto-adjust
                if abs(self.fitness_weight - 0.7) < 0.001 and abs(self.time_weight - 0.3) < 0.001:
                    errors.append(f"Legacy weight configuration detected. Please update to: --fitness_weight 0.5 --time_weight 0.3 --ram_weight 0.2")
                else:
                    errors.append(f"All weights must sum to 1.0: fitness_weight ({self.fitness_weight}) + time_weight ({self.time_weight}) + ram_weight ({self.ram_weight}) = {total_weight:.3f}")
                errors.append(f"Suggested weights: --fitness_weight 0.5 --time_weight 0.3 --ram_weight 0.2")
            if self.enable_time_penalty and self.time_penalty_threshold <= 0:
                errors.append(f"Time penalty threshold ({self.time_penalty_threshold}) must be positive when penalties are enabled")
            
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  • {error}" for error in errors))
    
    @classmethod
    def from_args(cls, args) -> 'GAConfig':
        """
        Create configuration from parsed CLI arguments.
        
        Args:
            args: argparse.Namespace from CLI parsing
            
        Returns:
            Validated GAConfig instance
        """
        return cls(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            max_threads=args.max_threads,
            output_dir=args.output_dir
        )
    
    @property
    def num_offspring(self) -> int:
        """Calculate number of offspring needed to fill population after elites."""
        # We need enough offspring to fill the remaining slots after elites are preserved
        return self.population_size - self.num_elites
    
    @property
    def num_elites(self) -> int:
        """Calculate number of elites based on population size and ratio.""" 
        return int(self.population_size * self.elite_ratio)
    
    def summary(self) -> str:
        """Generate human-readable configuration summary."""
        summary = f"""GA Configuration:
  Population: {self.population_size} (offspring: {self.num_offspring}, elites: {self.num_elites})
  Generations: {self.generations}
  Rates: mutation={self.mutation_rate:.3f}, crossover={self.crossover_rate:.3f}
  Threads: {self.max_threads}
  Output: {self.output_dir}"""
        
        if self.enable_multi_objective:
            summary += f"""
  Multi-Objective Evaluation:
    Fitness weight: {self.fitness_weight:.3f}
    Time weight: {self.time_weight:.3f}
    RAM weight: {self.ram_weight:.3f}
    Time penalties: {'enabled' if self.enable_time_penalty else 'disabled'}"""
            if self.enable_time_penalty:
                summary += f"""
    Time penalty threshold: {self.time_penalty_threshold:.1f}s"""
        else:
            summary += "\n  Evaluation: Single-objective (fitness only)"
        
        return summary
    
    def __str__(self) -> str:
        return f"GAConfig(pop={self.population_size}, gen={self.generations}, threads={self.max_threads})"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'max_threads': self.max_threads,
            'output_dir': self.output_dir,
            'offspring_ratio': self.offspring_ratio,
            'elite_ratio': self.elite_ratio,
            'tournament_size': self.tournament_size,
            'min_fitness': self.min_fitness,
            'convergence_generations': self.convergence_generations,
            'convergence_threshold': self.convergence_threshold,
            'enable_adaptive_tuning': self.enable_adaptive_tuning,
            'enable_smart_initialization': self.enable_smart_initialization,
            'enable_convergence_acceleration': self.enable_convergence_acceleration,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_dynamic_thread_scaling': self.enable_dynamic_thread_scaling,
            'diversity_strategy': self.diversity_strategy
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GAConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'GAConfig':
        """Create a new config with updated values."""
        current_config = self.to_dict()
        current_config.update(kwargs)
        return self.from_dict(current_config)